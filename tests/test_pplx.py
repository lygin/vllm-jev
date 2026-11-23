"""CPU tests for the pplx-decider adapter. No GPU, no weights download.

Tests marked ``reference`` compare with the release's own ``protocol.py``, its
tokenizer and its processor; they skip unless ``PPLX_DECIDER_RELEASE`` names a
local release directory. Only ``protocol.py`` and ``types.py`` are imported;
the release's model code and weights are not used.
"""

import asyncio
import base64
import importlib.util
import io
import json
import os
import shutil
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from vllm_jev import D3_QWEN35_ARCHITECTURE, d3, pplx
from vllm_jev import pplx_export as px
from vllm_jev.decision_template import DecisionTemplate

HIDDEN = 8
CODES = d3.candidate_codes()[: d3.MAX_OPTIONS]
SPECIALS = {
    "<|endoftext|>": 900,
    "<|im_start|>": 901,
    "<|im_end|>": 902,
    "<|vision_start|>": 903,
    "<|vision_end|>": 904,
    "<|image_pad|>": 905,
    "<|video_pad|>": 906,
    "<|fim_middle|>": 907,
    "<|fim_suffix|>": 908,
}
# As in the release's tokenizer, <|im_start|> and the FIM tokens are not "special".
ALL_SPECIAL = ["<|im_end|>", "<|endoftext|>", "<|image_pad|>", "<|video_pad|>"]
ESCAPED = sorted(
    {*ALL_SPECIAL, pplx.MARKER, pplx.SEPARATOR}, key=lambda t: (-len(t), t)
)
IMAGE_PAD = SPECIALS["<|image_pad|>"]
MARKER_ID = SPECIALS[pplx.MARKER]
SEPARATOR_ID = SPECIALS[pplx.SEPARATOR]
SYSTEM = f"<|im_start|>system\n{pplx.SYSTEM_PROMPT}<|im_end|>\n"


def data_url(width=64, height=48, color=(200, 30, 30), kind="PNG", mode="RGB"):
    buffer = io.BytesIO()
    Image.new(mode, (width, height), color).save(buffer, format=kind)
    subtype = {"PNG": "png", "JPEG": "jpeg", "WEBP": "webp"}[kind]
    return (
        f"data:image/{subtype};base64," + base64.b64encode(buffer.getvalue()).decode()
    )


class FakeTokenizer:
    """Greedy tokens: ChatML/FIM specials and answer codes are single tokens, else characters."""

    all_special_tokens = ALL_SPECIAL

    def __init__(self):
        self.vocab = {**SPECIALS, **{code: 1000 + i for i, code in enumerate(CODES)}}
        self.names = {token: name for name, token in self.vocab.items()}

    def encode(self, text, add_special_tokens=False):
        ids = []
        while text:
            token = max(
                (item for item in self.vocab if text.startswith(item)),
                key=len,
                default=None,
            )
            if token is None:
                ids.append(ord(text[0]))
                text = text[1:]
            else:
                ids.append(self.vocab[token])
                text = text[len(token) :]
        return ids

    def decode(self, ids):
        return "".join(self.names.get(token, chr(token)) for token in ids)

    def __call__(self, text, add_special_tokens=False):
        if isinstance(text, list):
            return {"input_ids": [self.encode(item) for item in text]}
        return {"input_ids": self.encode(text)}

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token)

    def convert_ids_to_tokens(self, token):
        return self.names.get(token)

    def apply_chat_template(
        self, conversation, tokenize, add_generation_prompt, enable_thinking
    ):
        assert not tokenize and not add_generation_prompt and enable_thinking is False
        parts = []
        for message in conversation:
            content = message["content"]
            if not isinstance(content, str):
                content = "".join(
                    "<|vision_start|><|image_pad|><|vision_end|>"
                    if item["type"] == "image"
                    else item["text"]
                    for item in content
                )
            parts.append(
                f"<|im_start|>{message['role']}\n{content.strip()}<|im_end|>\n"
            )
        return "".join(parts)


class FakeImageProcessor:
    merge_size = 2

    def __init__(self):
        self.size = {}
        self.requested = []

    def get_number_of_image_patches(self, height, width, kwargs):
        self.requested.append(kwargs)
        return 4 * max(1, height // 32) * max(1, width // 32)


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.image_processor = FakeImageProcessor()

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)


class FakeEngine:
    """vLLM stand-in: expands image placeholders and returns one last state."""

    def __init__(self, processor, size):
        self.processor = processor
        self.size = size
        self.requests = []
        self.states = []
        self.aborted = []

    async def encode(self, prompt, pooling_params, request_id):
        self.requests.append((prompt, pooling_params))
        images = prompt.get("multi_modal_data", {}).get("image", [])
        images = images if isinstance(images, list) else [images]
        counts = iter(
            d3.image_tokens(self.processor.image_processor, i, self.size)
            for i in images
        )
        ids = []
        for token in prompt["prompt_token_ids"]:
            ids += [token] * (next(counts) if token == IMAGE_PAD else 1)
        generator = torch.Generator().manual_seed(len(self.states))
        state = torch.randn(HIDDEN, generator=generator)
        self.states.append(state)
        yield SimpleNamespace(
            finished=True, prompt_token_ids=ids, outputs=SimpleNamespace(data=state)
        )

    async def abort(self, request_id):
        self.aborted.append(request_id)


def readout_weight(seed=0):
    return torch.randn(
        d3.MAX_OPTIONS, HIDDEN, generator=torch.Generator().manual_seed(seed)
    ).bfloat16()


def make_service(tmp_path, monkeypatch, **manifest):
    (tmp_path / d3.MANIFEST_FILE).write_text(
        json.dumps(
            {
                "prompt_protocol": pplx.PROTOCOL,
                "codes": CODES,
                "hidden_size": HIDDEN,
                "temperature": 1.25,
                "readout_dtype": "bfloat16",
                "vision": True,
                "max_images": 8,
                "image_token_id": IMAGE_PAD,
                "max_length": None,
                "max_pixels": 1_638_400,
                "marker_id": MARKER_ID,
                "separator_id": SEPARATOR_ID,
                "escaped_tokens": ESCAPED,
                **manifest,
            }
        )
    )
    save_file({"weight": readout_weight()}, str(tmp_path / d3.READOUT_FILE))
    processor = FakeProcessor()
    monkeypatch.setattr(
        "transformers.AutoProcessor.from_pretrained", lambda *a, **k: processor
    )
    service = pplx.PplxService(None, tmp_path, "perplexity-ai/pplx-decider-v1.3", 4096)
    service.engine_client = FakeEngine(processor, service.image_size)
    return service, service.engine_client


def request(questions, state="The light is red.", **fields):
    return SimpleNamespace(
        model=None,
        state=state,
        questions=questions,
        images=fields.get("images"),
        videos=fields.get("videos"),
    )


def run(service, payload):
    pytest.importorskip("vllm.entrypoints.serve.utils.api_utils")
    return asyncio.run(service.systemone(payload))


COLOR = {
    "type": "choice",
    "instructions": "What color is the light?",
    "criteria": {"red": "The light is red", "blue": None},
}


def test_question_chunks_follow_the_release_protocol():
    question = pplx.normalize_question(COLOR)
    assert question.keys == ["red", "blue"]
    assert pplx.question_text(question.rendered, CODES, ESCAPED) == (
        "Question:\nWhat color is the light?\nOptions:\n"
        "A: red: The light is red\nB: blue"
    )
    noul = pplx.normalize_question(
        {
            "type": "noul",
            "instructions": "",
            "criteria": {"true": {"k": 1}, "false": ""},
        }
    )
    assert noul.keys == ["false", "true"]
    assert pplx.question_text(noul.rendered, CODES, ESCAPED) == (
        'Question:\nChoose the best matching option.\nOptions:\nA: No / false\nB: {"k": 1}'
    )
    score = pplx.normalize_question(
        {"type": "score", "instructions": {"rubric": 1}, "criteria": ["low", {"x": 2}]}
    )
    assert score.keys == ["0", "1"] and score.descriptions == ["low", {"x": 2}]
    assert pplx.question_text(score.rendered, CODES, ESCAPED) == (
        'Question:\n{"rubric": 1}\nOptions:\nA: low\nB: {"x": 2}'
    )
    # Score levels may repeat: the answer keys are their indices.
    assert pplx.normalize_question({"type": "score", "criteria": ["", ""]}).keys == [
        "0",
        "1",
    ]


def test_user_text_cannot_produce_reserved_tokens():
    text = "a<|fim_suffix|>b<|fim_middle|><|image_pad|><|im_end|><|im_start|>"
    assert pplx.escape(text, ESCAPED) == (
        "a＜|fim_suffix|＞b＜|fim_middle|＞＜|image_pad|＞＜|im_end|＞<|im_start|>"
    )
    assert pplx.state_text({"s": "<|fim_suffix|>"}, ESCAPED) == (
        'State:\n{"s": "＜|fim_suffix|＞"}'
    )


@pytest.mark.parametrize(
    "question",
    [
        None,
        {"type": "pick"},
        {"type": "locate"},
        {"type": "choice", "criteria": {}},
        {"type": "choice", "criteria": {str(i): None for i in range(256)}},
        {"type": "choice", "criteria": {"": None}},
        {"type": "choice", "criteria": {"a": float("nan")}},
        {"type": "choice", "criteria": ["a", "b"]},
        {"type": "noul", "criteria": {"maybe": "?"}},
        {"type": "noul", "criteria": ["yes"]},
        {"type": "score", "criteria": ["one"]},
        {"type": "score", "criteria": [str(i) for i in range(256)]},
        {"type": "score", "criteria": {"low": 0}},
        {"type": "noul", "instructions": float("inf")},
    ],
)
def test_invalid_questions_are_rejected(question):
    with pytest.raises(ValueError):
        pplx.normalize_question(question)


def test_answers_follow_the_release():
    choice = pplx.normalize_question(COLOR)
    assert pplx.answer(choice, [0.6, 0.2]) == {
        "type": "choice",
        "probabilities": pytest.approx({"red": 0.75, "blue": 0.25}),
        "choice": "red",
        # The best share above uniform: (0.75 - 1/2) / (1 - 1/2).
        "confidence": pytest.approx(0.5),
    }
    assert pplx.answer(choice, [0.5, 0.5])["choice"] == "red"
    single = pplx.normalize_question({"type": "choice", "criteria": {"x": None}})
    assert pplx.answer(single, [1.0])["confidence"] == 1.0
    noul = pplx.normalize_question({"type": "noul"})
    assert pplx.answer(noul, [0.3, 0.7]) == {"type": "noul", "noul": pytest.approx(0.7)}
    score = pplx.normalize_question(
        {"type": "score", "criteria": ["low", {"b": 1, "a": 2}, "high"]}
    )
    result = pplx.answer(score, [0.25, 0.5, 0.25])
    assert result["score"] == pytest.approx(1.0)
    assert result["legend"] == {"0": "low", "1": {"b": 1, "a": 2}, "2": "high"}
    # 1 - E|level - best| / mean distance from the middle = 1 - 0.5 / (2/3).
    assert result["confidence"] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        pplx.answer(choice, [float("nan"), 0.1])


def test_service_scores_each_question_after_the_shared_context(tmp_path, monkeypatch):
    service, engine = make_service(tmp_path, monkeypatch)
    questions = {
        "color": COLOR,
        "urgent": {"type": "noul", "instructions": "Is it urgent?"},
        "level": {"type": "score", "criteria": ["low", "mid", "high"]},
    }
    response = run(service, request(questions))
    assert list(response["answers"]) == ["color", "urgent", "level"]
    tokenizer = service.tokenizer
    context = tokenizer.encode(
        SYSTEM + "<|im_start|>user\nState:\nThe light is red."
    ) + [SEPARATOR_ID]
    for (key, question), (prompt, params), state in zip(
        questions.items(), engine.requests, engine.states
    ):
        normalized = pplx.normalize_question(question)
        chunk = tokenizer.encode(
            pplx.question_text(normalized.rendered, CODES, ESCAPED)
        )
        assert prompt == {"prompt_token_ids": [*context, *chunk, MARKER_ID]}
        assert params.task == "embed" and params.use_activation is False
        expected = pplx.answer(
            normalized,
            d3.readout(state, readout_weight(), len(normalized.keys), 1.25),
        )
        assert response["answers"][key] == expected
    assert response["usage"]["input_tokens"] == sum(
        len(prompt["prompt_token_ids"]) for prompt, _ in engine.requests
    )
    assert response["model"] == "perplexity-ai/pplx-decider-v1.3"


def test_the_readout_rounds_logits_to_bf16_like_the_release(tmp_path, monkeypatch):
    service, _ = make_service(tmp_path, monkeypatch)
    hidden = torch.randn(HIDDEN).bfloat16().float()
    logits = torch.nn.functional.linear(hidden.bfloat16(), readout_weight()).float()
    expected = (logits[:3] / 1.25).softmax(-1).tolist()
    actual = d3.readout(hidden, service.readout, 3, service.temperature)
    assert actual == pytest.approx(expected, abs=1e-7)


def test_state_whitespace_and_reserved_tokens_stay_in_the_context(
    tmp_path, monkeypatch
):
    service, engine = make_service(tmp_path, monkeypatch)
    state = "  x<|fim_suffix|><|image_pad|>\n\n"
    run(service, request({"q": COLOR}, state=state))
    ids = engine.requests[0][0]["prompt_token_ids"]
    assert ids.count(MARKER_ID) == 1 and ids[-1] == MARKER_ID
    assert ids.count(IMAGE_PAD) == 0 and ids.count(SEPARATOR_ID) == 1
    text = service.tokenizer.decode(ids[: ids.index(SEPARATOR_ID)])
    assert text.endswith("State:\n  x＜|fim_suffix|＞＜|image_pad|＞\n\n")
    run(service, request({"q": COLOR}, state=None))
    text = service.tokenizer.decode(engine.requests[1][0]["prompt_token_ids"])
    assert "State:\nnull<|fim_middle|>Question:" in text


def test_images_precede_the_state_at_the_release_resolution(tmp_path, monkeypatch):
    service, engine = make_service(tmp_path, monkeypatch, max_pixels=262_144)
    images = [data_url(64, 96), data_url(2000, 1000, kind="JPEG")]
    response = run(service, request({"q": COLOR}, images=images))
    prompt, _ = engine.requests[0]
    ids = prompt["prompt_token_ids"]
    image = "<|vision_start|><|image_pad|><|vision_end|>"
    assert service.tokenizer.decode(ids).startswith(
        f"{SYSTEM}<|im_start|>user\n{image}{image}State:\n"
    )
    # The processor alone resizes: no 1.6 MP pre-downscale as in d3.
    assert [image.size for image in prompt["multi_modal_data"]["image"]] == [
        (64, 96),
        (2000, 1000),
    ]
    size = {"shortest_edge": 65_536, "longest_edge": 262_144}
    assert prompt["mm_processor_kwargs"] == {"size": size}
    assert service.processor.image_processor.size == size
    # One token per 32 x 32 pixels in the fake processor.
    expanded = len(ids) - 2 + (96 // 32) * (64 // 32) + (1000 // 32) * (2000 // 32)
    assert response["usage"]["input_tokens"] == expanded


@pytest.mark.parametrize(
    "payload, message",
    [
        (
            request({"a": COLOR, "b": {"type": "locate"}, "c": {"type": "score"}}),
            "invalid questions: b: locate .*; c: ",
        ),
        (request({"q": COLOR}, videos=["data:video/mp4;base64,AA=="]), "not videos"),
        (request({"q": COLOR}, state=float("nan")), "state: "),
        (request({"q": COLOR}, state="x" + pplx.CONTEXT_END), "changed the state"),
        (request({}), "1 to 64 questions"),
        (request({"q": COLOR}, images=[data_url()] * 9), "at most 8 images"),
        (request({"q": COLOR}, images=["data:image/png;base64,AA=="]), r"images\[0\]"),
        (
            SimpleNamespace(**{**vars(request({"q": COLOR})), "model": "other"}),
            "not loaded",
        ),
    ],
)
def test_invalid_requests_run_nothing(tmp_path, monkeypatch, payload, message):
    service, engine = make_service(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match=message):
        service._compile(payload)
    assert not engine.requests


def test_overlong_questions_are_not_truncated(tmp_path, monkeypatch):
    service, _ = make_service(tmp_path, monkeypatch)
    service.max_length = 150
    with pytest.raises(
        ValueError, match="q: the question prompt has .* nothing was truncated"
    ):
        service._compile(request({"q": COLOR}, state="x" * 200))


@pytest.mark.parametrize(
    "config",
    [
        {"layout": "instructions-first"},
        {"context_order": ["instructions", "state", "criteria"]},
    ],
)
def test_reordered_templates_are_rejected(tmp_path, monkeypatch, config):
    monkeypatch.setattr(pplx, "current_template", lambda: DecisionTemplate(config))
    with pytest.raises(ValueError, match="native order"):
        make_service(tmp_path, monkeypatch)
    pplx.check_template(DecisionTemplate({"state_template": "Ticket: {state}"}))
    pplx.check_template(
        DecisionTemplate({"context_order": ["state", "instructions", "criteria"]})
    )


def make_release(path: Path, **decision):
    path.mkdir(parents=True, exist_ok=True)
    tensors = {
        "language_model.embed_tokens.weight": torch.ones(4, HIDDEN).bfloat16(),
        "language_model.layers.0.mlp.up_proj.weight": torch.ones(2, HIDDEN).bfloat16(),
        "visual.patch_embed.proj.weight": torch.ones(2, 2).bfloat16(),
    }
    save_file(
        dict(list(tensors.items())[:2]), str(path / "model-00001-of-00002.safetensors")
    )
    save_file(
        dict(list(tensors.items())[2:]), str(path / "model-00002-of-00002.safetensors")
    )
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    name: f"model-0000{1 if i < 2 else 2}-of-00002.safetensors"
                    for i, name in enumerate(tensors)
                }
            }
        )
    )
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5Model"],
                "model_type": "qwen3_5",
                "image_token_id": IMAGE_PAD,
                "text_config": {"model_type": "qwen3_5_text", "hidden_size": HIDDEN},
            }
        )
    )
    tokenizer = FakeTokenizer()
    (path / "decision_config.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "base_model": "Qwen/Qwen3.8-27B",
                "revision": "0" * 40,
                "codes": CODES,
                "token_ids": [tokenizer.vocab[code] for code in CODES],
                "temperature": 1.25,
                "attention_mode": "noncausal_full_attention",
                "pooling": "last",
                "model_type": "qwen3_5",
                "max_pixels": 1_638_400,
                **decision,
            }
        )
    )
    (path / "joint_config.json").write_text(json.dumps(pplx.JOINT_PROTOCOL, indent=2))
    save_file({"weight": readout_weight()}, str(path / "readout.safetensors"))
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "processor_config.json",
    ):
        (path / name).write_text("{}")
    # Release code and the Locate adapter beside the weights are not exported.
    (path / "source" / "src").mkdir(parents=True, exist_ok=True)
    (path / "source" / "src" / "model.py").write_text("raise SystemExit('executed')\n")
    (path / "locate-span").mkdir(exist_ok=True)
    (path / "locate-span" / "adapter.safetensors").write_text("")
    return path


@pytest.fixture
def fake_tokenizer(monkeypatch):
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *a, **k: FakeTokenizer()
    )


def test_export_links_weights_and_declares_bidirectional_attention(
    tmp_path, fake_tokenizer
):
    source = make_release(tmp_path / "release")
    output = tmp_path / "out"
    manifest = px.export_pplx(source, output, "local-pplx")
    assert manifest["architecture"] == D3_QWEN35_ARCHITECTURE
    assert manifest["prompt_protocol"] == pplx.PROTOCOL
    assert manifest["codes"] == CODES and manifest["vision"] is True
    assert (manifest["marker_id"], manifest["separator_id"]) == (
        MARKER_ID,
        SEPARATOR_ID,
    )
    assert manifest["escaped_tokens"] == ESCAPED
    assert manifest["readout_dtype"] == "bfloat16" and manifest["max_length"] is None
    assert manifest["default_max_length"] == pplx.SERVE_MAX_LENGTH
    config = json.loads((output / "config.json").read_text())
    assert config["architectures"] == [D3_QWEN35_ARCHITECTURE]
    assert config["is_causal"] is False and config["text_config"]["is_causal"] is False
    for name in ("model-00001-of-00002.safetensors", "readout.safetensors"):
        assert (output / name).stat().st_ino == (source / name).stat().st_ino
    assert (output / "joint_config.json").is_file()
    assert not (output / "source").exists() and not (output / "locate-span").exists()
    from vllm_jev.checkpoint import verify

    assert verify(output, full=True)["format"] == "vllm-jev-d3-v1"


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda p: (p / "joint_config.json").unlink(), "chunked joint protocol"),
        (
            lambda p: _edit(p / "joint_config.json", version=3),
            "chunked joint protocol",
        ),
        (
            lambda p: _edit(p / "decision_config.json", format_version=1),
            "format_version 2",
        ),
        (
            lambda p: _edit(p / "decision_config.json", attention_mode="causal"),
            "noncausal",
        ),
        (
            lambda p: _edit(p / "decision_config.json", model_type="qwen3_5_moe"),
            "dense Qwen3.5",
        ),
        (
            lambda p: _edit(p / "decision_config.json", codes=CODES[::-1]),
            "answer codes",
        ),
        (
            lambda p: _edit(p / "decision_config.json", codes=["a", *CODES[1:]]),
            "letter codes",
        ),
        (lambda p: _edit(p / "decision_config.json", temperature=0), "temperature"),
        (lambda p: _edit(p / "config.json", model_type="qwen3_vl"), "Qwen3.5"),
        (
            lambda p: save_file(
                {"weight": torch.ones(3, HIDDEN)}, str(p / "readout.safetensors")
            ),
            "readout",
        ),
        (
            lambda p: _edit(
                p / "model.safetensors.index.json",
                weight_map={
                    "model.language_model.norm.weight": "model-00001-of-00002.safetensors"
                },
            ),
            "bare Qwen3_5Model",
        ),
    ],
)
def test_export_rejects_other_layouts(tmp_path, fake_tokenizer, change, message):
    source = make_release(tmp_path / "release")
    change(source)
    with pytest.raises(ValueError, match=message):
        px.export_pplx(source, tmp_path / "out")


def _edit(path, **fields):
    path.write_text(json.dumps({**json.loads(path.read_text()), **fields}))


def test_pinned_releases_are_checked_like_a_download(tmp_path, fake_tokenizer):
    source = make_release(tmp_path / "release")
    assert set(px.MODELS) == set(px.RELEASE_HASHES)
    for model_id in px.MODELS:
        with pytest.raises(ValueError, match="checksum mismatch"):
            px.export_pplx(source, tmp_path / "out", model_id)


def test_verify_rejects_a_manifest_without_its_escapes(tmp_path, fake_tokenizer):
    from vllm_jev.d3_export import verify_d3

    px.export_pplx(make_release(tmp_path / "release"), tmp_path / "out")
    _edit(tmp_path / "out" / d3.MANIFEST_FILE, escaped_tokens=["<|im_end|>"])
    with pytest.raises(ValueError, match="escapes"):
        verify_d3(tmp_path / "out", full=False)


def test_local_release_is_prepared_as_pplx(tmp_path, fake_tokenizer):
    from vllm_jev import native_models

    source = make_release(tmp_path / "release")
    output = native_models.prepare_local(source, tmp_path / "workspace")
    manifest = json.loads((output / d3.MANIFEST_FILE).read_text())
    assert manifest["prompt_protocol"] == pplx.PROTOCOL
    assert manifest["source_repository"] == str(source.resolve())
    assert native_models.prepare_local(source, tmp_path / "workspace") == output
    with pytest.raises(ValueError, match="uses pplx_joint_readout_v4"):
        native_models.prepare_local(
            source, tmp_path / "workspace", "d3_code_readout_v1"
        )


def test_hub_release_downloads_only_weights_and_protocol(
    tmp_path, monkeypatch, fake_tokenizer
):
    from vllm_jev import native_models

    source = make_release(tmp_path / "release")
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(source)

    siblings = [
        SimpleNamespace(rfilename=name)
        for name in ("decision_config.json", "readout.safetensors", "joint_config.json")
    ]
    monkeypatch.setattr(native_models, "snapshot_download", download)
    monkeypatch.setattr(
        native_models,
        "HfApi",
        lambda: SimpleNamespace(
            model_info=lambda *a, **k: SimpleNamespace(sha="f" * 40, siblings=siblings)
        ),
    )
    output = native_models.prepare("someone/pplx-like", tmp_path / "workspace")
    manifest = json.loads((output / d3.MANIFEST_FILE).read_text())
    assert manifest["prompt_protocol"] == pplx.PROTOCOL
    assert manifest["source_repository"] == "someone/pplx-like"
    assert calls[0]["allow_patterns"] == px.SOURCE_FILES
    assert calls[0]["revision"] == "f" * 40
    # A cached checkpoint keeps its protocol.
    assert native_models.prepare("someone/pplx-like", tmp_path / "workspace") == output
    with pytest.raises(ValueError, match="differs from cached"):
        native_models.prepare(
            "someone/pplx-like", tmp_path / "workspace", "d3_code_readout_v1"
        )


def serve(monkeypatch, *arguments):
    from vllm_jev import cli

    calls = {}
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli.os,
        "execvpe",
        lambda executable, argv, env: calls.update(argv=argv, env=env),
    )
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setattr(cli.sys, "argv", ["vllm-jev", "serve", *arguments])
    cli.main()
    return calls


def test_cli_serves_pplx_with_whole_prompt_last_state_pooling(
    tmp_path, monkeypatch, fake_tokenizer
):
    from vllm_jev import native_models

    source = make_release(tmp_path / "release", max_pixels=262_144)
    output = native_models.prepare_local(source, tmp_path / "workspace")
    calls = serve(monkeypatch, str(output))
    assert calls["env"]["VLLM_USE_V2_MODEL_RUNNER"] == "0"
    argv = calls["argv"]
    assert argv[argv.index("--max-model-len") + 1] == str(pplx.SERVE_MAX_LENGTH + 1)
    assert (
        "--no-enable-prefix-caching" in argv and "--enable-prefix-caching" not in argv
    )
    assert "--mamba-cache-mode" not in argv
    assert json.loads(argv[argv.index("--pooler-config") + 1]) == {
        "task": "embed",
        "seq_pooling_type": "LAST",
    }
    assert json.loads(argv[argv.index("--mm-processor-kwargs") + 1]) == {
        "size": {"shortest_edge": 65_536, "longest_edge": 262_144}
    }
    with pytest.raises(SystemExit):
        serve(monkeypatch, str(output), "--decision-template", "instructions-first")


def test_endpoint_serves_pplx_checkpoints_with_their_service(tmp_path, monkeypatch):
    pytest.importorskip("vllm")
    from vllm_jev.endpoint import JevEndpointPlugin

    (tmp_path / d3.MANIFEST_FILE).write_text(
        json.dumps({"prompt_protocol": pplx.PROTOCOL})
    )
    built = []

    class Recording:
        label = "pplx-decider"

        def __init__(self, engine, path, model_id, max_length, **kwargs):
            built.append((model_id, max_length))

    monkeypatch.setattr(pplx, "PplxService", Recording)
    engine = SimpleNamespace(model_config=SimpleNamespace(max_model_len=32769))
    state = SimpleNamespace()
    args = SimpleNamespace(model=str(tmp_path), served_model_name=None)
    asyncio.run(JevEndpointPlugin().init_state(engine, state, args))
    assert isinstance(state.vllm_d3_service, Recording)
    assert built == [("pplx-decider", 32768)]


def release():
    path = os.environ.get("PPLX_DECIDER_RELEASE")
    source = Path(path or "") / "source" / "src" / "pplx_decider"
    if not path or not (source / "protocol.py").is_file():
        pytest.skip("PPLX_DECIDER_RELEASE is not set")
    return Path(path)


@pytest.fixture
def reference(monkeypatch):
    """The release's ``protocol`` module, without importing its model code."""
    path = release()
    source = path / "source" / "src" / "pplx_decider"
    package = types.ModuleType("pplx_reference")
    package.__path__ = [str(source)]
    monkeypatch.setitem(sys.modules, "pplx_reference", package)
    for name in ("types", "protocol"):
        spec = importlib.util.spec_from_file_location(
            f"pplx_reference.{name}", source / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
    return path, sys.modules["pplx_reference.protocol"]


def release_service(path: Path, tmp_path: Path):
    """A service over the release's tokenizer and processor, with a zero readout."""
    from transformers import AutoTokenizer

    for name in ("config.json", *px.COPIED_FILES):
        if (path / name).is_file():
            shutil.copy2(path / name, tmp_path / name)
    decision = json.loads((path / "decision_config.json").read_text())
    config = json.loads((path / "config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(tmp_path, local_files_only=True)
    hidden = config["text_config"]["hidden_size"]
    (tmp_path / d3.MANIFEST_FILE).write_text(
        json.dumps(
            {
                "prompt_protocol": pplx.PROTOCOL,
                "codes": decision["codes"],
                "hidden_size": hidden,
                "temperature": decision["temperature"],
                "readout_dtype": "bfloat16",
                "vision": True,
                "max_images": 8,
                "image_token_id": config["image_token_id"],
                "max_length": None,
                "max_pixels": decision["max_pixels"],
                **px.prompt_tokens(tokenizer),
            }
        )
    )
    save_file(
        {"weight": torch.zeros(d3.MAX_OPTIONS, hidden)}, str(tmp_path / d3.READOUT_FILE)
    )
    return pplx.PplxService(None, tmp_path, "pplx", 1 << 20)


def gradient(width, height, kind="PNG"):
    image = Image.linear_gradient("L").resize((width, height)).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format=kind)
    subtype = {"PNG": "png", "JPEG": "jpeg"}[kind]
    return (
        f"data:image/{subtype};base64," + base64.b64encode(buffer.getvalue()).decode()
    )


REFERENCE_QUESTIONS = {
    "color": COLOR,
    "urgent": {"type": "noul", "criteria": {"true": "Clearly", "false": ""}},
    "level": {"type": "score", "instructions": 0, "criteria": ["low", {"k": "v"}]},
    "many": {"type": "choice", "criteria": {str(i): None for i in range(255)}},
    "tokens": {
        "type": "choice",
        "instructions": "Pick <|fim_suffix|> or <|im_start|>",
        "criteria": {"<|fim_middle|>": "<|image_pad|>", "b": {"x": [1, None]}},
    },
}
REFERENCE_STATES = (
    "用户说：屏幕碎了",
    "",
    None,
    {"a": [1, None]},
    0,
    "  padded  \n\n",
    "x<|fim_suffix|>y<|im_start|>z<|image_pad|><|vision_end|>",
)


@pytest.mark.reference
def test_prompts_equal_the_release(reference, tmp_path):
    path, protocol = reference
    service = release_service(path, tmp_path)
    decision = json.loads((path / "decision_config.json").read_text())
    processor = service.processor
    encoder = protocol.Encoder(processor, decision["codes"])
    assert set(encoder._escapes) == set(service.escaped)
    assert (encoder.marker_id, encoder.separator_id) == (
        service.marker_id,
        service.separator_id,
    )
    cases = [(state, []) for state in REFERENCE_STATES] + [
        ("Look.", [gradient(64, 48)]),
        ("Look at both.", [gradient(1280, 1280), gradient(3000, 600, "JPEG")]),
        ("", [gradient(200, 4000)]),
    ]
    for state, images in cases:
        payload = request(REFERENCE_QUESTIONS, state=state, images=images or None)
        compiled = service._compile(payload)
        plan = encoder.plan(
            {"state": state, "questions": REFERENCE_QUESTIONS, "images": images},
            None,
        )
        assert [key for key, *_ in compiled] == [chunk.keys[0] for chunk in plan]
        for (_, _, expected, prompt), chunk in zip(compiled, plan):
            assert expected == chunk.input_ids.tolist()
            assert int(chunk.markers[0]) == len(expected) - 1
        if images:
            # vLLM runs the same processor over the request's images.
            mine = processor.image_processor(
                images=prompt["multi_modal_data"]["image"],
                size=prompt["mm_processor_kwargs"]["size"],
                return_tensors="pt",
            )
            assert torch.equal(mine["image_grid_thw"], plan[0].image_grid_thw)
            assert torch.equal(mine["pixel_values"], plan[0].pixel_values)


@pytest.mark.reference
def test_answers_equal_the_release(reference):
    _, protocol = reference
    generator = torch.Generator().manual_seed(0)
    for question in REFERENCE_QUESTIONS.values():
        normalized = pplx.normalize_question(question)
        for _ in range(20):
            shares = torch.rand(len(normalized.keys), generator=generator).tolist()
            ours = pplx.answer(normalized, shares)
            theirs = protocol.answer(question, shares)
            assert ours.keys() == theirs.keys()
            for field, value in theirs.items():
                if field == "probabilities" or isinstance(value, float):
                    assert ours[field] == pytest.approx(value, abs=1e-12)
                else:
                    assert ours[field] == value
