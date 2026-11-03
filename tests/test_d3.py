"""CPU tests for the d3 (WeMM-JEV) adapter. No GPU, no weights download.

Tests marked ``reference`` compare with the release's own ``d3_format.py`` and
``inference/_images.py``, its tokenizer and its processor; they skip unless
``WEMM_JEV_RELEASE`` names a local release directory. Only those two code files
are imported; the release's runtime and weights are not used.
"""

import asyncio
import base64
import importlib.util
import inspect
import io
import json
import os
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from vllm_jev import D3_QWEN35_ARCHITECTURE, d3
from vllm_jev import d3_export as dx
from vllm_jev.decision_template import DecisionTemplate

HIDDEN = 8
CODES = d3.candidate_codes()[: d3.MAX_OPTIONS]
SPECIALS = {
    "<|im_start|>": 901,
    "<|im_end|>": 902,
    "<|vision_start|>": 903,
    "<|vision_end|>": 904,
    "<|image_pad|>": 905,
    "<|video_pad|>": 906,
}
IMAGE_PAD = SPECIALS["<|image_pad|>"]


def data_url(width=64, height=48, color=(200, 30, 30), kind="PNG", mode="RGB"):
    buffer = io.BytesIO()
    Image.new(mode, (width, height), color).save(buffer, format=kind)
    subtype = {"PNG": "png", "JPEG": "jpeg", "WEBP": "webp", "GIF": "gif"}[kind]
    return (
        f"data:image/{subtype};base64," + base64.b64encode(buffer.getvalue()).decode()
    )


class FakeTokenizer:
    """Greedy tokens: ChatML specials and answer codes are single tokens, else characters."""

    def __init__(self):
        self.vocab = {
            **SPECIALS,
            **{code: 1000 + i for i, code in enumerate(d3.candidate_codes())},
        }

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

    def __call__(self, text, add_special_tokens=False):
        if isinstance(text, list):
            return {"input_ids": [self.encode(item) for item in text]}
        return {"input_ids": self.encode(text)}

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token)

    def apply_chat_template(
        self, conversation, tokenize, add_generation_prompt, enable_thinking
    ):
        assert not tokenize and add_generation_prompt and enable_thinking is False
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
        return "".join(parts) + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


class FakeImageProcessor:
    merge_size = 2

    def __init__(self):
        self.size = {}

    def get_number_of_image_patches(self, height, width, kwargs):
        assert kwargs == {"size": d3.IMAGE_SIZE}
        return 4 * max(1, height // 32) * max(1, width // 32)


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.image_processor = FakeImageProcessor()

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)


class FakeEngine:
    """vLLM stand-in: expands image placeholders and returns one last state."""

    def __init__(self, processor, mismatch=False):
        self.processor = processor
        self.mismatch = mismatch
        self.requests = []
        self.states = []
        self.aborted = []

    async def encode(self, prompt, pooling_params, request_id):
        self.requests.append((prompt, pooling_params))
        images = prompt.get("multi_modal_data", {}).get("image", [])
        images = images if isinstance(images, list) else [images]
        counts = iter(
            d3.image_tokens(self.processor.image_processor, i) for i in images
        )
        ids = []
        for token in prompt["prompt_token_ids"]:
            ids += [token] * (next(counts) if token == IMAGE_PAD else 1)
        if self.mismatch:
            ids = ids[1:]
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
    )


def make_service(tmp_path, monkeypatch, **manifest):
    (tmp_path / d3.MANIFEST_FILE).write_text(
        json.dumps(
            {
                "codes": CODES,
                "hidden_size": HIDDEN,
                "temperature": 0.7,
                "readout_dtype": "float32",
                "vision": True,
                "max_images": 8,
                "image_token_id": IMAGE_PAD,
                "max_length": None,
                **manifest,
            }
        )
    )
    save_file({"weight": readout_weight()}, str(tmp_path / d3.READOUT_FILE))
    processor = FakeProcessor()
    monkeypatch.setattr(
        "transformers.AutoProcessor.from_pretrained", lambda *a, **k: processor
    )
    engine = FakeEngine(processor)
    return d3.D3Service(engine, tmp_path, "Rover00/WeMM-JEV", 4096), engine


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
    "criteria": {"A": "Red", "B": "Blue"},
}


def test_text_prompt_follows_the_release_contract():
    question = d3.normalize_question(COLOR)
    assert d3.user_content("The light is red.", question.rendered, CODES) == (
        "State:\nThe light is red.\n\n"
        "Question:\nWhat color is the light?\n\n"
        "Options:\nA: A: Red\nB: B: Blue\n\n"
        "Reply with only the code of the best option."
    )
    noul = d3.normalize_question({"type": "noul", "criteria": {"true": "Urgent"}})
    assert d3.user_content(None, noul.rendered, CODES) == (
        "State:\n(empty)\n\n"
        "Question:\nChoose the best matching option.\n\n"
        "Options:\nA: No / false\nB: Urgent\n\n"
        "Reply with only the code of the best option."
    )
    score = d3.normalize_question(
        {"type": "score", "instructions": {"rubric": 1}, "criteria": ["low", {"x": 2}]}
    )
    assert d3.user_content({"a": [1, None]}, score.rendered, CODES) == (
        'State:\n{"a": [1, null]}\n\n'
        'Question:\n{"rubric": 1}\n\n'
        'Options:\nA: low\nB: {"x": 2}\n\n'
        "Reply with only the code of the best option."
    )
    assert score.keys == ["0", "1"]
    conversation = d3.messages("s", question.rendered, CODES)
    assert conversation[0] == {"role": "system", "content": d3.SYSTEM_PROMPT}


def test_images_precede_the_state_in_every_context_order():
    question = d3.normalize_question(COLOR)
    native = d3.user_content("s", question.rendered, CODES, 2)
    assert [item["type"] for item in native] == ["image", "image", "text"]
    assert native[-1]["text"] == d3.user_content("s", question.rendered, CODES)
    for order in permutations(d3.NATIVE_ORDER):
        text = d3.user_content("s", question.rendered, CODES, 0, order)
        content = d3.user_content("s", question.rendered, CODES, 2, order)
        flat = "".join(
            "<img>" if item["type"] == "image" else item["text"] for item in content
        )
        assert flat.replace("<img><img>", "") == text
        assert flat.index("<img><img>State:") >= 0
        assert text.endswith("\n\n" + d3.CLOSING_LINE)
        headings = [line for line in text.splitlines() if line.endswith(":")]
        assert headings == [
            {"state": "State:", "instructions": "Question:", "criteria": "Options:"}[
                name
            ]
            for name in order
        ]


@pytest.mark.parametrize(
    "question",
    [
        None,
        {"type": "pick"},
        {"type": "choice", "criteria": {}},
        {"type": "choice", "criteria": {str(i): None for i in range(256)}},
        {"type": "choice", "criteria": {"": None}},
        {"type": "choice", "criteria": {"a": float("nan")}},
        {"type": "choice", "criteria": ["a", "b"]},
        {"type": "noul", "criteria": {"maybe": "?"}},
        {"type": "score", "criteria": ["one"]},
        {"type": "score", "criteria": [str(i) for i in range(11)]},
        {"type": "score", "criteria": ["same", "same"]},
        {"type": "score", "criteria": ["", "x"]},
        {"type": "noul", "instructions": float("inf")},
    ],
)
def test_invalid_questions_are_rejected(question):
    with pytest.raises(ValueError):
        d3.normalize_question(question)


def test_answers_follow_the_release():
    choice = d3.normalize_question(COLOR)
    result = d3.answer(choice, [0.6, 0.2])
    assert result["type"] == "choice" and result["choice"] == "A"
    assert result["probabilities"] == pytest.approx({"A": 0.75, "B": 0.25})
    entropy = -(
        0.75 * torch.log(torch.tensor(0.75)) + 0.25 * torch.log(torch.tensor(0.25))
    )
    assert result["confidence"] == pytest.approx(
        1 - float(entropy) / float(torch.log(torch.tensor(2.0)))
    )
    noul = d3.normalize_question({"type": "noul"})
    assert d3.answer(noul, [0.3, 0.7]) == {"type": "noul", "noul": pytest.approx(0.7)}
    score = d3.normalize_question(
        {"type": "score", "criteria": ["low", {"b": 1, "a": 2}]}
    )
    result = d3.answer(score, [0.25, 0.75])
    assert result["score"] == pytest.approx(0.75)
    assert result["legend"] == {"0": "low", "1": '{"a":2,"b":1}'}
    single = d3.answer(
        d3.normalize_question({"type": "choice", "criteria": {"x": None}}), [1.0]
    )
    assert single["confidence"] == 1.0
    with pytest.raises(ValueError):
        d3.answer(choice, [float("nan"), 0.1])


def test_readout_masks_codes_and_runs_in_fp32():
    weight = readout_weight(3)
    hidden = torch.randn(HIDDEN).bfloat16()
    actual = d3.readout(hidden, weight, 3, 0.5)
    expected = torch.softmax((hidden.float() @ weight.T)[:3] / 0.5, -1).tolist()
    assert actual == pytest.approx(expected, abs=1e-6)
    half = d3.readout(hidden, weight.bfloat16(), 2, 1.0)
    reference = torch.nn.functional.linear(hidden, weight.bfloat16()).float()[:2]
    assert half == pytest.approx(torch.softmax(reference, -1).tolist(), abs=1e-6)


def test_images_are_capped_like_the_release_engine():
    small = d3.load_image(data_url(64, 48))
    assert small.size == (64, 48) and small.mode == "RGB"
    large = d3.load_image(data_url(2000, 1000, kind="JPEG"))
    assert large.size == d3.fit(2000, 1000, d3.IMAGE_CAP_PIXELS)
    assert large.width * large.height <= d3.IMAGE_CAP_PIXELS
    assert d3.load_image(data_url(1500, 1500, (1, 2, 3, 4), mode="RGBA")).mode == "RGB"
    assert d3.load_image(data_url(40, 40, kind="WEBP")).size == (40, 40)
    for value in (
        data_url(8, 8, kind="GIF"),
        "data:image/png;base64,not base64!",
        "https://example.com/a.png",
        data_url(4001, 4000),
        3,
    ):
        with pytest.raises(ValueError):
            d3.load_image(value)


def test_service_scores_each_question_from_its_last_state(tmp_path, monkeypatch):
    service, engine = make_service(tmp_path, monkeypatch)
    questions = {
        "color": COLOR,
        "urgent": {"type": "noul", "instructions": "Is it urgent?"},
        "level": {"type": "score", "criteria": ["low", "mid", "high"]},
    }
    response = run(service, request(questions))
    assert list(response["answers"]) == ["color", "urgent", "level"]
    tokenizer = service.tokenizer
    weight = readout_weight()
    for (key, question), (prompt, params), state in zip(
        questions.items(), engine.requests, engine.states
    ):
        normalized = d3.normalize_question(question)
        text = tokenizer.apply_chat_template(
            d3.messages("The light is red.", normalized.rendered, CODES),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        assert prompt == {"prompt_token_ids": tokenizer.encode(text)}
        assert params.task == "embed" and params.use_activation is False
        expected = d3.answer(
            normalized, d3.readout(state, weight, len(normalized.keys), 0.7)
        )
        assert response["answers"][key] == expected
    assert response["usage"]["input_tokens"] == sum(
        len(prompt["prompt_token_ids"]) for prompt, _ in engine.requests
    )
    assert response["model"] == "Rover00/WeMM-JEV"


def test_service_sends_collapsed_image_placeholders(tmp_path, monkeypatch):
    service, engine = make_service(tmp_path, monkeypatch)
    images = [data_url(64, 96), data_url(320, 64)]
    response = run(service, request({"q": COLOR}, images=images))
    prompt, _ = engine.requests[0]
    assert prompt["prompt_token_ids"].count(IMAGE_PAD) == 2
    assert [image.size for image in prompt["multi_modal_data"]["image"]] == [
        (64, 96),
        (320, 64),
    ]
    assert prompt["mm_processor_kwargs"] == {"size": d3.IMAGE_SIZE}
    expanded = len(prompt["prompt_token_ids"]) - 2 + (2 * 3) + (10 * 2)
    assert response["usage"]["input_tokens"] == expanded
    assert service.processor.image_processor.size == d3.IMAGE_SIZE


def test_service_moves_blocks_for_context_order(tmp_path, monkeypatch):
    template = DecisionTemplate(
        {"context_order": ["criteria", "instructions", "state"]}
    )
    monkeypatch.setattr(d3, "current_template", lambda: template)
    service, engine = make_service(tmp_path, monkeypatch)
    run(service, request({"q": COLOR}, images=[data_url()]))
    names = {token: name for name, token in service.tokenizer.vocab.items()}
    text = "".join(
        names.get(token, chr(token))
        for token in engine.requests[0][0]["prompt_token_ids"]
    )
    image = "<|vision_start|><|image_pad|><|vision_end|>"
    assert (
        text.index("Options:") < text.index("Question:") < text.index(image + "State:")
    )


@pytest.mark.parametrize(
    "payload, message",
    [
        (
            request({"a": COLOR, "b": {"type": "pick"}, "c": {"type": "score"}}),
            "invalid questions: b: .*; c: ",
        ),
        (request({"q": COLOR}, videos=["data:video/mp4;base64,AA=="]), "not videos"),
        (request({"q": COLOR}, state="x<|image_pad|>"), "reserved image"),
        (request({"q": COLOR}, state=float("nan")), "state: "),
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
    service, engine = make_service(tmp_path, monkeypatch, max_length=120)
    with pytest.raises(
        ValueError, match="q: the question prompt has .* nothing was truncated"
    ):
        service._compile(request({"q": COLOR}, state="x" * 200))
    text_only = make_service(tmp_path, monkeypatch, vision=False)[0]
    with pytest.raises(ValueError, match="no vision tower"):
        text_only._compile(request({"q": COLOR}, images=[data_url()]))


def test_service_rejects_a_different_vllm_tokenization(tmp_path, monkeypatch):
    service, engine = make_service(tmp_path, monkeypatch)
    engine.mismatch = True
    with pytest.raises(RuntimeError, match="tokenization differs"):
        run(service, request({"q": COLOR}))
    assert len(engine.aborted) == 1


def make_release(path: Path, *, attention="noncausal_full_attention", **decision):
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
                "format_version": 1,
                "format_id": "d3-code-readout-v1",
                "prompt": "d3",
                "pooling": "last",
                "max_length": None,
                "readout_dtype": "float32",
                "base_model": "Qwen/Qwen3.8-27B",
                "revision": "0" * 40,
                "codes": CODES,
                "token_ids": [tokenizer.vocab[code] for code in CODES],
                "temperature": 1.0,
                "attention_mode": attention,
                **decision,
            }
        )
    )
    save_file({"weight": readout_weight()}, str(path / "readout.safetensors"))
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "preprocessor_config.json",
    ):
        (path / name).write_text("{}")
    # Release code beside the weights is neither copied nor executed.
    (path / "d3_runtime.py").write_text("raise SystemExit('executed')\n")
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
    manifest = dx.export_d3(source, output, "local-d3")
    assert manifest["architecture"] == D3_QWEN35_ARCHITECTURE
    assert manifest["codes"] == CODES and manifest["vision"] is True
    config = json.loads((output / "config.json").read_text())
    assert config["architectures"] == [D3_QWEN35_ARCHITECTURE]
    assert config["is_causal"] is False and config["text_config"]["is_causal"] is False
    for name in ("model-00001-of-00002.safetensors", "readout.safetensors"):
        assert (output / name).stat().st_ino == (source / name).stat().st_ino
    assert not (output / "d3_runtime.py").exists()
    from vllm_jev.checkpoint import verify

    assert verify(output, full=True)["format"] == dx.FORMAT


def test_export_keeps_causal_releases_causal(tmp_path, fake_tokenizer):
    source = make_release(tmp_path / "release", attention="causal")
    dx.export_d3(source, tmp_path / "out")
    config = json.loads((tmp_path / "out" / "config.json").read_text())
    assert config["is_causal"] is True and config["text_config"]["is_causal"] is True


@pytest.mark.parametrize(
    "change, message",
    [
        (
            lambda p: _edit(p / "decision_config.json", format_id="other"),
            "not a d3-code-readout-v1",
        ),
        (
            lambda p: _edit(p / "decision_config.json", attention_mode="sliding"),
            "attention_mode",
        ),
        (
            lambda p: _edit(p / "decision_config.json", codes=CODES[::-1]),
            "answer codes",
        ),
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
        dx.export_d3(source, tmp_path / "out")


def _edit(path, **fields):
    path.write_text(json.dumps({**json.loads(path.read_text()), **fields}))


def test_pinned_release_is_checked_like_a_download(tmp_path, fake_tokenizer):
    source = make_release(tmp_path / "release")
    with pytest.raises(ValueError, match="checksum mismatch"):
        dx.export_d3(source, tmp_path / "out", "Rover00/WeMM-JEV")


def test_verify_rejects_a_causal_config_for_a_bidirectional_release(
    tmp_path, fake_tokenizer
):
    source = make_release(tmp_path / "release")
    dx.export_d3(source, tmp_path / "out")
    _edit(tmp_path / "out" / "config.json", is_causal=True)
    with pytest.raises(ValueError, match="attention mode mismatch"):
        dx.verify_d3(tmp_path / "out", full=False)


def test_local_release_is_prepared_once(tmp_path, fake_tokenizer):
    from vllm_jev import native_models

    source = make_release(tmp_path / "release")
    output = native_models.prepare_local(source, tmp_path / "workspace")
    manifest = json.loads((output / d3.MANIFEST_FILE).read_text())
    assert manifest["prompt_protocol"] == d3.PROTOCOL
    assert manifest["source_repository"] == str(source.resolve())
    assert native_models.prepare_local(source, tmp_path / "workspace") == output
    with pytest.raises(ValueError, match="uses d3_code_readout_v1"):
        native_models.prepare_local(source, tmp_path / "workspace", "mica_labels_v1")


def test_cli_serves_d3_with_whole_prompt_last_state_pooling(
    tmp_path, monkeypatch, fake_tokenizer
):
    from vllm_jev import cli, native_models

    source = make_release(tmp_path / "release")
    output = native_models.prepare_local(source, tmp_path / "workspace")
    calls = {}
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli.os,
        "execvpe",
        lambda executable, argv, env: calls.update(argv=argv, env=env),
    )
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setattr(cli.sys, "argv", ["vllm-jev", "serve", str(output)])
    cli.main()
    assert calls["env"]["VLLM_USE_V2_MODEL_RUNNER"] == "0"
    argv = calls["argv"]
    assert argv[argv.index("--max-model-len") + 1] == str(d3.SERVE_MAX_LENGTH + 1)
    assert (
        "--no-enable-prefix-caching" in argv and "--enable-prefix-caching" not in argv
    )
    assert "--mamba-cache-mode" not in argv
    assert json.loads(argv[argv.index("--pooler-config") + 1]) == {
        "task": "embed",
        "seq_pooling_type": "LAST",
    }
    assert json.loads(argv[argv.index("--limit-mm-per-prompt") + 1]) == {
        "image": 8,
        "video": 0,
    }
    assert json.loads(argv[argv.index("--mm-processor-kwargs") + 1]) == {
        "size": d3.IMAGE_SIZE
    }
    assert argv[argv.index("--served-model-name") + 1] == str(source.resolve())


def test_vllm_backbone_maps_weights_and_requires_whole_prompts():
    pytest.importorskip("triton")
    from vllm_jev.model import VllmD3Qwen35ForEmbedding

    mapper = VllmD3Qwen35ForEmbedding.hf_to_vllm_mapper
    assert (
        mapper._map_name("language_model.layers.3.self_attn.q_proj.weight")
        == "language_model.model.layers.3.self_attn.q_proj.weight"
    )
    assert (
        mapper._map_name("visual.blocks.0.attn.qkv.weight")
        == "visual.blocks.0.attn.qkv.weight"
    )
    assert mapper._map_name("mtp.layers.0.weight") is None
    # vLLM passes vllm_config only when the signature names it and prefix.
    parameters = inspect.signature(VllmD3Qwen35ForEmbedding.__init__).parameters
    assert {"vllm_config", "prefix"} <= set(parameters)
    for chunked, cached, top, v2, message in (
        (True, False, False, False, "whole-prompt prefill"),
        (False, True, False, False, "whole-prompt prefill"),
        (False, False, True, False, "whole-prompt prefill"),
        (False, False, False, True, "V1 model runner"),
    ):
        config = SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(is_causal=top),
                hf_text_config=SimpleNamespace(is_causal=False),
            ),
            scheduler_config=SimpleNamespace(enable_chunked_prefill=chunked),
            cache_config=SimpleNamespace(enable_prefix_caching=cached),
            use_v2_model_runner=v2,
        )
        with pytest.raises(ValueError, match=message):
            VllmD3Qwen35ForEmbedding(vllm_config=config, prefix="")


def release():
    path = os.environ.get("WEMM_JEV_RELEASE")
    if not path or not (Path(path) / "d3_format.py").is_file():
        pytest.skip("WEMM_JEV_RELEASE is not set")
    return Path(path)


def release_module(path, name):
    spec = importlib.util.spec_from_file_location(f"wemm_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.reference
def test_prompts_and_codes_equal_the_release():
    from transformers import AutoProcessor, AutoTokenizer

    path = release()
    reference = release_module(path / "d3_format.py", "format")
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    codes, token_ids = d3.answer_codes(tokenizer)
    config = json.loads((path / "decision_config.json").read_text())
    assert (codes, token_ids) == (config["codes"], config["token_ids"])
    for question in (
        COLOR,
        {"type": "noul", "criteria": {"true": "Clearly", "false": ""}},
        {"type": "score", "instructions": 0, "criteria": ["low", {"k": "v"}]},
        {"type": "choice", "criteria": {str(i): None for i in range(255)}},
    ):
        rendered = d3.normalize_question(question).rendered
        for state in ("用户说：屏幕碎了", "", None, {"a": [1, None]}, 0):
            ours = d3.chat_text(tokenizer, d3.messages(state, rendered, codes))
            assert ours == reference.render(tokenizer, state, rendered, codes)
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    processor.image_processor.size = dict(d3.IMAGE_SIZE)
    rendered = d3.normalize_question(COLOR).rendered
    text = d3.chat_text(processor, d3.messages("Look.", rendered, codes, 1))
    for size in ((64, 48), (1280, 1280), (200, 9000)):
        image = Image.new("RGB", size)
        ids = processor(text=[text], images=[image], return_tensors="pt")["input_ids"][
            0
        ]
        pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        assert int((ids == pad).sum()) == d3.image_tokens(
            processor.image_processor, image
        )


@pytest.mark.reference
def test_image_cap_equals_the_release_engine():
    images = release_module(release() / "inference" / "_images.py", "images")
    for value in (
        data_url(2000, 1000, kind="JPEG"),
        data_url(1500, 1500, (1, 2, 3, 4), mode="RGBA"),
        data_url(3000, 600),
        data_url(640, 480),
    ):
        expected = images.pil(value, max_pixels=images.MAX_PIXELS).convert("RGB")
        actual = d3.load_image(value)
        assert actual.size == expected.size
        assert actual.tobytes() == expected.tobytes()
