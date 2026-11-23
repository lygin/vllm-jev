"""Export a pplx-decider release (chunked joint protocol v4) for native vLLM pooling.

The release stores a bf16 ``Qwen3_5Model`` backbone with a separate 255-row
readout, like a d3 release. The export links its weight shards unchanged,
names the d3 pooling architecture in ``config.json`` together with the
noncausal attention mode, and keeps the readout, protocol, tokenizer and
processor beside it. The release's ``source/`` package and Locate adapter are
neither downloaded nor executed.
"""

import argparse
import json
import math
import shutil
import string
from pathlib import Path

import torch
from safetensors import safe_open

from . import D3_QWEN35_ARCHITECTURE
from .checkpoint import sha256
from .d3 import IMAGE_SIZE, MANIFEST_FILE, MAX_OPTIONS, READOUT_FILE
from .d3_export import FORMAT, MAX_IMAGES, _index, _link, verify_d3
from .pplx import (
    JOINT_PROTOCOL,
    LEGACY_MAX_PIXELS,
    MARKER,
    PROTOCOL,
    SEPARATOR,
    SERVE_MAX_LENGTH,
)

MODELS = {
    "perplexity-ai/pplx-decider-v1.3": "9c2ebb27470d3dec2ad7ab727727e40d142c7213",
    "perplexity-ai/pplx-decider-flash-v1.3": "d06fb3f2d97e0a52325a0d463c450cf0338f44fe",
}
# sha256 of the released files (as in each release-manifest.json), so a local
# copy is checked like a download.
RELEASE_HASHES = {
    "perplexity-ai/pplx-decider-v1.3": {
        "model-00001-of-00011.safetensors": "683762f9ba120ed6980f6888d808c55812d1f3c54d79517aab7b0f5705ec7b20",
        "model-00002-of-00011.safetensors": "b8a1694cc4ef7ae24e411adab0092e87a78abe547c97c47a8e19c57e345f1738",
        "model-00003-of-00011.safetensors": "2ee7d8d814af369542d01b590f701c5f42e22ee756235a52125232f3d6e1a9b0",
        "model-00004-of-00011.safetensors": "e853d09c8b2dedcb3f1a28d10f406df732d54d7603f55b1e6133e56deef8d3e5",
        "model-00005-of-00011.safetensors": "87e712601578fccf91b0874c6972dd24ff1368e80d3075c0577eae956a07d637",
        "model-00006-of-00011.safetensors": "a19864e0e8e77147ba8f0d0005dc70fca46f1521e24bcab2334a53492d2c4062",
        "model-00007-of-00011.safetensors": "7a84ac88396740b1d02e88be50c4926bea99a3ec430dcac8b26738c8f1e4727a",
        "model-00008-of-00011.safetensors": "413bd8c46e78ae179382511fdc0c0134a4db7d532f1981594b0fb6a55c0df36e",
        "model-00009-of-00011.safetensors": "afb31b650ae67b9b271ee02a251e6ac5b775e811db0abe6084789d8920c83e6a",
        "model-00010-of-00011.safetensors": "733da075428d6eb9ff0d6fc6e07caf482d70c1c86d77e4f37238fe4721be04f4",
        "model-00011-of-00011.safetensors": "e68776c2fd0f3a503049fce03f4612850045a618819992af564967ed089878c3",
        "chat_template.jinja": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
        "config.json": "687f20bf493b8ee787dadaf338cf6df644dad44e32bcd2c3677a47c6c213d8b3",
        "decision_config.json": "4ee1c1e80044f6249c82e644b5334678670701b87f3972989c542ca86483e200",
        "joint_config.json": "f7836e2e3fdaebaee8705e2ca25f5bb75bb146043ad530c84a6779a4af90801b",
        "model.safetensors.index.json": "0da01ffdc85b6f1bc596d3c38dd0d17663e18b6c81140269cbef28ea123f4b12",
        "processor_config.json": "b4b905eee2d1cf96dcbdac1c68b35113c8151958a4c1f76667242b01168572cd",
        "readout.safetensors": "46928c7cd0084289f6237a833720683c9d8ff91965706e746cc34b3ef0f12039",
        "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
        "tokenizer_config.json": "7488550a894dcf741616126e322ea31d66cf9457b59ec5825f4c7efdced421f8",
    },
    "perplexity-ai/pplx-decider-flash-v1.3": {
        "model-00001-of-00002.safetensors": "0048d1ddf2c5754fc05540fccdbc4ac50407e5cf14a35326be01cac0756b4de6",
        "model-00002-of-00002.safetensors": "40e0c0ccb3248a59db0724ffd1ad1c81fc935d9f474e6fd52e9df32af4690608",
        "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
        "config.json": "1bdb718faa99e32673f1622ea2f536affe20867bcb4703888aac9a030ec59e22",
        "decision_config.json": "f2068e686bf676fa14374d91c5979ca56b93f2c9bdaa990d6090681edfc6eb69",
        "joint_config.json": "f7836e2e3fdaebaee8705e2ca25f5bb75bb146043ad530c84a6779a4af90801b",
        "model.safetensors.index.json": "0e6c2d31d343632e4a5443d20f01b83e7906e1b67fae818545a33e2a15815ea2",
        "processor_config.json": "b4b905eee2d1cf96dcbdac1c68b35113c8151958a4c1f76667242b01168572cd",
        "readout.safetensors": "46358a2e2639f1956561687ba261dd9b37506f878e90eb5eabc15f114532bec3",
        "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
        "tokenizer_config.json": "7488550a894dcf741616126e322ea31d66cf9457b59ec5825f4c7efdced421f8",
    },
}
COPIED_FILES = (
    "decision_config.json",
    "joint_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "processor_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)
# Weights, readout and the files the export reads; release code is not downloaded.
SOURCE_FILES = [
    "config.json",
    "model.safetensors.index.json",
    "model*.safetensors",
    READOUT_FILE,
    *COPIED_FILES,
]


def _decision_config(source: Path) -> dict:
    """The release's readout contract, checked as its runtime checks it."""
    path = source / "joint_config.json"
    if not path.is_file() or json.loads(path.read_text()) != JOINT_PROTOCOL:
        raise ValueError(
            "joint_config.json does not declare the chunked joint protocol (version 4)"
        )
    config = json.loads((source / "decision_config.json").read_text())
    if not isinstance(config, dict) or config.get("format_version") != 2:
        raise ValueError("decision_config.json is not format_version 2")
    if config.get("attention_mode") != "noncausal_full_attention":
        raise ValueError("pplx-decider releases use noncausal full attention")
    if config.get("pooling", "last") != "last":
        raise ValueError("pplx-decider releases read out at the marker position")
    if config.get("model_type", "qwen3_5") != "qwen3_5":
        raise ValueError("pplx-decider serving supports dense Qwen3.5 backbones")
    temperature = config.get("temperature")
    if (
        not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("pplx-decider temperature must be finite and positive")
    codes, token_ids = config.get("codes"), config.get("token_ids")
    if (
        not isinstance(codes, list)
        or not isinstance(token_ids, list)
        or len(codes) != MAX_OPTIONS
        or len(token_ids) != MAX_OPTIONS
        or len(set(codes)) != MAX_OPTIONS
        or any(
            not isinstance(code, str)
            or not code
            or set(code) - set(string.ascii_uppercase)
            for code in codes
        )
        or any(type(token) is not int for token in token_ids)
    ):
        raise ValueError(f"decision_config.json must list {MAX_OPTIONS} letter codes")
    pixels = config.setdefault("max_pixels", LEGACY_MAX_PIXELS)
    if type(pixels) is not int or pixels < IMAGE_SIZE["shortest_edge"]:
        raise ValueError("invalid pplx-decider max_pixels")
    return config


def _single_token(tokenizer, token: str) -> int:
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1 or tokenizer.convert_ids_to_tokens(ids[0]) != token:
        raise ValueError(f"{token} must be a single existing token")
    return ids[0]


def prompt_tokens(tokenizer) -> dict:
    """The manifest's marker, separator and the reserved tokens escaped in user text."""
    return {
        "marker_id": _single_token(tokenizer, MARKER),
        "separator_id": _single_token(tokenizer, SEPARATOR),
        # As the release's encoder; longest first, then by name.
        "escaped_tokens": sorted(
            {*tokenizer.all_special_tokens, MARKER, SEPARATOR},
            key=lambda token: (-len(token), token),
        ),
    }


def export_pplx(source: Path, output: Path, model_id: str = "pplx-decider") -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    known = {}
    for name, digest in RELEASE_HASHES.get(model_id, {}).items():
        if not (source / name).is_file() or sha256(source / name) != digest:
            raise ValueError(f"pplx-decider release checksum mismatch: {name}")
        known[name] = digest
    # The export rewrites its config; every other checked file is linked or copied.
    known.pop("config.json", None)
    decision = _decision_config(source)
    config = json.loads((source / "config.json").read_text())
    text = config.get("text_config", {})
    if (
        config.get("model_type") != "qwen3_5"
        or text.get("model_type") != "qwen3_5_text"
    ):
        raise ValueError("pplx-decider serving supports dense Qwen3.5 backbones")
    hidden_size = text.get("hidden_size")
    if type(hidden_size) is not int or hidden_size <= 0:
        raise ValueError("invalid pplx-decider hidden size")
    index = _index(source)
    weights = index.get("weight_map")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("pplx-decider release has no weight index")
    if any(
        not isinstance(shard, str) or Path(shard).name != shard
        for shard in weights.values()
    ):
        raise ValueError("pplx-decider shard index must use local filenames")
    # A bare Qwen3_5Model: the language model and vision tower, no output head.
    if any(not name.startswith(("language_model.", "visual.")) for name in weights):
        raise ValueError("pplx-decider release is not a bare Qwen3_5Model checkpoint")
    with safe_open(source / READOUT_FILE, framework="pt", device="cpu") as file:
        if set(file.keys()) != {"weight"}:
            raise ValueError("pplx-decider readout must hold one weight tensor")
        readout = file.get_tensor("weight")
    if (
        tuple(readout.shape) != (MAX_OPTIONS, hidden_size)
        or readout.dtype not in (torch.float32, torch.bfloat16)
        or not torch.isfinite(readout).all()
    ):
        raise ValueError("invalid pplx-decider readout weight")

    output.mkdir(parents=True, exist_ok=True)
    for shard in sorted(set(weights.values())):
        _link(source / shard, output / shard)
    _link(source / READOUT_FILE, output / READOUT_FILE)
    if (source / "model.safetensors.index.json").is_file():
        shutil.copy2(
            source / "model.safetensors.index.json",
            output / "model.safetensors.index.json",
        )
    else:
        (output / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2) + "\n"
        )
    for name in COPIED_FILES:
        if (source / name).is_file():
            shutil.copy2(source / name, output / name)
    # vLLM reads the top-level flag for scheduling, the text one for its layers.
    config["architectures"] = [D3_QWEN35_ARCHITECTURE]
    config["is_causal"] = False
    config["text_config"] = {**text, "is_causal": False}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(output, local_files_only=True)
    if any(
        tokenizer.encode(code, add_special_tokens=False) != [token_id]
        for code, token_id in zip(decision["codes"], decision["token_ids"])
    ):
        raise ValueError(
            "the answer codes in decision_config.json differ from the tokenizer's"
        )
    probe = "State:\nA"
    if (
        tokenizer(probe)["input_ids"]
        != tokenizer(probe, add_special_tokens=False)["input_ids"]
    ):
        raise ValueError(
            "pplx-decider prompts are tokenized without added special tokens"
        )
    image_token_id = config.get("image_token_id")
    if tokenizer.convert_tokens_to_ids("<|image_pad|>") != image_token_id:
        raise ValueError(
            "pplx-decider tokenizer and config disagree on the image token"
        )

    manifest = {
        "format": FORMAT,
        "architecture": D3_QWEN35_ARCHITECTURE,
        "prompt_protocol": PROTOCOL,
        "source_repository": model_id,
        "source_checkpoint": str(source),
        "foundation_repository": decision.get("base_model"),
        "foundation_revision": decision.get("revision"),
        "hidden_size": hidden_size,
        "attention_mode": decision["attention_mode"],
        "temperature": float(decision["temperature"]),
        # The release reads out in bf16 on CUDA, whatever the stored dtype.
        "readout_dtype": "bfloat16",
        "max_length": None,
        "default_max_length": SERVE_MAX_LENGTH,
        "codes": decision["codes"],
        "image_token_id": image_token_id,
        "vision": any(name.startswith("visual.") for name in weights),
        "max_images": MAX_IMAGES,
        "max_pixels": decision["max_pixels"],
        **prompt_tokens(tokenizer),
        "files": {
            item.name: known.get(item.name) or sha256(item)
            for item in output.iterdir()
            if item.is_file()
        },
    }
    (output / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n")
    # Publishing verifies every file; here only the layout is checked.
    verify_d3(output, full=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="pplx-decider")
    args = parser.parse_args()
    manifest = export_pplx(args.checkpoint, args.output, args.model_id)
    verify_d3(args.output, full=True)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
