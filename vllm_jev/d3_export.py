"""Export a d3 code-readout release (WeMM-JEV) for native vLLM pooling.

The release stores a bf16 ``Qwen3_5Model`` backbone with a separate FP32
readout. The export links its weight shards unchanged, names the pooling
architecture in ``config.json`` together with the release's attention mode,
and keeps the readout, tokenizer and processor beside it.
"""

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open

from . import D3_QWEN35_ARCHITECTURE
from .checkpoint import sha256, verify_files
from .d3 import (
    FORMAT_ID,
    MANIFEST_FILE,
    MAX_OPTIONS,
    PROTOCOL,
    READOUT_FILE,
    answer_codes,
)

FORMAT = "vllm-jev-d3-v1"
MODELS = {"Rover00/WeMM-JEV": "a463f840c9259ff9139ab02181e01f20aca91b4b"}
# sha256 of the released files, so a local copy is checked like a download.
RELEASE_HASHES = {
    "Rover00/WeMM-JEV": {
        "model-00001-of-00011.safetensors": "6830f6d5b5146b6814d53d894c7cd148d9e2a134167d195b46f99abccaf6bde9",
        "model-00002-of-00011.safetensors": "e97b1c34dbb1ac8f4905d772486f91d87971f3b92f2b4871e2ef38f485eb9692",
        "model-00003-of-00011.safetensors": "6ed1737ed2a3e9a3b279df2aa639625855120efff904e49138fe31b011011dfc",
        "model-00004-of-00011.safetensors": "79d722aefaa1396f9af7d8e93ce7af402ed240935c5d175fcc4ce69889ca6bdd",
        "model-00005-of-00011.safetensors": "a19bfd74aafcbf8d7cd48c2416164ba0e8dcdee96b59a0ffa867e934edad345d",
        "model-00006-of-00011.safetensors": "a3f44c388ca7ab9550b576dba57582c434192f8ed0f719bb8b56bb05c64df744",
        "model-00007-of-00011.safetensors": "8e479719ce913f929340871ca32622d6383251095d47ab2b4032a040ba80479a",
        "model-00008-of-00011.safetensors": "57350248c36e5d0bf929cfd14b090baf908b62e079e877701f54f0c4187db91d",
        "model-00009-of-00011.safetensors": "c8eaeeae921151c88ac938e747c9e3597fae6376a9afb2b1640cde774ffc364a",
        "model-00010-of-00011.safetensors": "d0c4398fd0671736dc2baa39f7a251a4e9a79b053f5460f5c7dff9da52823781",
        "model-00011-of-00011.safetensors": "788b52f92575537456718b9085271c9416f583328a5e0b5418538460323fe562",
        "model.safetensors.index.json": "d1613f01583cd96fcc5930f13052d0ddc111bf42614101275529e6501dfc290f",
        "config.json": "0327faa7eaca18ee38721bcd3a12a1f58fc7ab960f068ecfcae698480cfb8ebc",
        "decision_config.json": "fe87936d3fb4fe012b790098ea8b06e5fb74aca8cd985695706e8d5897129f03",
        "readout.safetensors": "cda91c34d2564ddc922415adcf0c83ade766618ccceb3dfac2050d1877dac298",
        "tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
        "tokenizer_config.json": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
        "chat_template.jinja": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
        "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
        "video_preprocessor_config.json": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
        "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
        "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
    },
}
COPIED_FILES = (
    "decision_config.json",
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
ATTENTION_MODES = ("causal", "noncausal_full_attention")
MAX_IMAGES = 8


def _link(source: Path, target: Path) -> None:
    """Hard-link a large file when both sides share a file system."""
    try:
        os.link(source.resolve(), target)
    except OSError:
        shutil.copy2(source, target)


def _decision_config(source: Path) -> dict:
    """The release's readout contract, with the runtime's defaults filled in."""
    config = json.loads((source / "decision_config.json").read_text())
    if not isinstance(config, dict) or config.get("format_version") != 1:
        raise ValueError("decision_config.json is not format_version 1")
    if config.get("format_id") != FORMAT_ID:
        raise ValueError(f"decision_config.json is not a {FORMAT_ID} release")
    for key, default, allowed in (
        ("prompt", "d3", ("d3",)),
        ("pooling", "last", ("last",)),
        ("attention_mode", "causal", ATTENTION_MODES),
        ("readout_dtype", "float32", ("float32", "bfloat16")),
    ):
        config.setdefault(key, default)
        if config[key] not in allowed:
            raise ValueError(f"unsupported d3 {key}: {config[key]!r}")
    temperature = config.setdefault("temperature", 1.0)
    if (
        not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("d3 temperature must be finite and positive")
    limit = config.get("max_length")
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("d3 max_length must be a positive integer or null")
    return config


def _index(source: Path) -> dict:
    """The release's shard index; a single-file release gets one written for it."""
    if (source / "model.safetensors.index.json").is_file():
        return json.loads((source / "model.safetensors.index.json").read_text())
    with safe_open(source / "model.safetensors", framework="pt", device="cpu") as file:
        return {"weight_map": {name: "model.safetensors" for name in file.keys()}}


def export_d3(source: Path, output: Path, model_id: str = "d3") -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    known = {}
    for name, digest in RELEASE_HASHES.get(model_id, {}).items():
        if not (source / name).is_file() or sha256(source / name) != digest:
            raise ValueError(f"d3 release checksum mismatch: {name}")
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
        raise ValueError("d3 serving supports Qwen3.5 backbones")
    hidden_size = text.get("hidden_size")
    if type(hidden_size) is not int or hidden_size <= 0:
        raise ValueError("invalid d3 hidden size")
    index = _index(source)
    weights = index.get("weight_map")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("d3 release has no weight index")
    if any(
        not isinstance(shard, str) or Path(shard).name != shard
        for shard in weights.values()
    ):
        raise ValueError("d3 shard index must use local filenames")
    # A bare Qwen3_5Model: the language model and vision tower, no output head.
    if any(not name.startswith(("language_model.", "visual.")) for name in weights):
        raise ValueError("d3 release is not a bare Qwen3_5Model checkpoint")
    with safe_open(source / READOUT_FILE, framework="pt", device="cpu") as file:
        if set(file.keys()) != {"weight"}:
            raise ValueError("d3 readout must hold one weight tensor")
        readout = file.get_tensor("weight")
    if (
        tuple(readout.shape) != (MAX_OPTIONS, hidden_size)
        or readout.dtype not in (torch.float32, torch.bfloat16)
        or not torch.isfinite(readout).all()
    ):
        raise ValueError("invalid d3 readout weight")

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
    causal = decision["attention_mode"] == "causal"
    config["architectures"] = [D3_QWEN35_ARCHITECTURE]
    config["is_causal"] = causal
    config["text_config"] = {**text, "is_causal": causal}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(output, local_files_only=True)
    codes, token_ids = answer_codes(tokenizer)
    if decision.get("codes") != codes or decision.get("token_ids") != token_ids:
        raise ValueError(
            "the answer codes in decision_config.json differ from the tokenizer's"
        )
    probe = "State:\nA"
    if (
        tokenizer(probe)["input_ids"]
        != tokenizer(probe, add_special_tokens=False)["input_ids"]
    ):
        raise ValueError("d3 prompts are tokenized without added special tokens")
    image_token_id = config.get("image_token_id")
    if tokenizer.convert_tokens_to_ids("<|image_pad|>") != image_token_id:
        raise ValueError("d3 tokenizer and config disagree on the image token")

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
        "readout_dtype": decision["readout_dtype"],
        "max_length": decision.get("max_length"),
        "codes": codes,
        "image_token_id": image_token_id,
        "vision": any(name.startswith("visual.") for name in weights),
        "max_images": MAX_IMAGES,
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


def verify_d3(path: Path, *, full: bool = True) -> dict:
    from .pplx import MARKER, SEPARATOR
    from .pplx import PROTOCOL as PPLX_PROTOCOL

    path = path.resolve()
    manifest = json.loads((path / MANIFEST_FILE).read_text())
    if (
        manifest.get("format") != FORMAT
        or manifest.get("architecture") != D3_QWEN35_ARCHITECTURE
        or manifest.get("prompt_protocol") not in (PROTOCOL, PPLX_PROTOCOL)
        or manifest.get("attention_mode") not in ATTENTION_MODES
        or manifest.get("readout_dtype") not in ("float32", "bfloat16")
        or not isinstance(manifest.get("vision"), bool)
    ):
        raise ValueError("unrecognized d3 checkpoint")
    pplx = manifest["prompt_protocol"] == PPLX_PROTOCOL
    hidden = manifest.get("hidden_size")
    codes = manifest.get("codes")
    temperature = manifest.get("temperature")
    limit = manifest.get("max_length")
    if (
        type(hidden) is not int
        or hidden <= 0
        or not isinstance(codes, list)
        or len(codes) != MAX_OPTIONS
        or len(set(codes)) != MAX_OPTIONS
        or any(not isinstance(code, str) or not code for code in codes)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
        or (limit is not None and (type(limit) is not int or limit <= 0))
        or type(manifest.get("image_token_id")) is not int
        or type(manifest.get("max_images")) is not int
        or manifest["max_images"] < 0
    ):
        raise ValueError("invalid d3 dimensions, codes or limits")
    if pplx and (
        manifest["attention_mode"] != "noncausal_full_attention"
        or type(manifest.get("marker_id")) is not int
        or type(manifest.get("separator_id")) is not int
        or type(manifest.get("max_pixels")) is not int
        or type(manifest.get("default_max_length")) is not int
        or not isinstance(manifest.get("escaped_tokens"), list)
        or any(not isinstance(token, str) for token in manifest["escaped_tokens"])
        or not {MARKER, SEPARATOR} <= set(manifest["escaped_tokens"])
    ):
        raise ValueError("invalid pplx-decider markers, image budget or escapes")
    config = json.loads((path / "config.json").read_text())
    causal = manifest["attention_mode"] == "causal"
    if (
        config.get("architectures") != [manifest["architecture"]]
        or config.get("is_causal") is not causal
        or config.get("text_config", {}).get("is_causal") is not causal
    ):
        raise ValueError("d3 architecture or attention mode mismatch")
    index = json.loads((path / "model.safetensors.index.json").read_text())
    checked = verify_files(
        path,
        manifest,
        index,
        (
            "config.json",
            "model.safetensors.index.json",
            READOUT_FILE,
            "decision_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            *(("joint_config.json",) if pplx else ()),
        ),
        full=full,
    )
    with safe_open(path / READOUT_FILE, framework="pt", device="cpu") as file:
        if set(file.keys()) != {"weight"}:
            raise ValueError("invalid d3 readout tensors")
        weight = file.get_tensor("weight")
        if (
            tuple(weight.shape) != (MAX_OPTIONS, hidden)
            or not torch.isfinite(weight).all()
        ):
            raise ValueError("invalid d3 readout weight")
    return {
        "format": manifest["format"],
        "architecture": manifest["architecture"],
        "checked_files": checked,
        "full": full,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="d3")
    args = parser.parse_args()
    manifest = export_d3(args.checkpoint, args.output, args.model_id)
    verify_d3(args.output, full=True)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
