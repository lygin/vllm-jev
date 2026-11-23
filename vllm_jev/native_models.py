"""Select a verified native vLLM scoring protocol from a Hugging Face model ID."""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from filelock import FileLock
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import validate_repo_id

from .checkpoint import sha256, verify
from .public_models import PROFILES
from .public_models import prepare as prepare_open_jev

BRANCH_PROTOCOL = "openjev_branch_v03"
CHOICE_PROTOCOL = "open_jev_choice"
TINY_PROTOCOL = "tiny_jev_marker"
D3_PROTOCOL = "d3_code_readout_v1"
D3_FILES = {"decision_config.json", "readout.safetensors"}
# pplx-decider releases share the d3 layout and add their request protocol.
PPLX_PROTOCOL = "pplx_joint_readout_v4"
PPLX_FILE = "joint_config.json"
D3_PROTOCOLS = (D3_PROTOCOL, PPLX_PROTOCOL)
DECISION_PROTOCOLS = (
    "kev_pointer_v1",
    "decider_slot_v1",
    "task_json_v1",
    "thisthat_slot_v1",
    "mica_labels_v1",
    "jevk5_letters_v1",
)
BRANCH_FILES = {
    "MANIFEST.json",
    "adapter_config.json",
    "adapter_model.safetensors",
    "head.safetensors",
    "openjev_config.json",
    "calibration-v03.json",
}
SCALAR_FILES = {
    "package/checkpoint/model.json",
    "package/checkpoint/head.pt",
    "package/checkpoint/temperature.json",
    "package/checkpoint/adapter/adapter_config.json",
    "package/checkpoint/adapter/adapter_model.safetensors",
}


def _publish(
    output: Path,
    model_id: str,
    revision: str,
    protocol: str,
    build: Callable[[Path], dict],
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".jev-", dir=output.parent))
    try:
        manifest = build(temporary)
        if manifest["prompt_protocol"] != protocol:
            raise ValueError("exported checkpoint protocol does not match the source")
        manifest.update(source_repository=model_id, source_revision=revision)
        name = (
            "decision_manifest.json"
            if manifest.get("format") == "vllm-jev-decision-v1"
            else "jevany_pointer_manifest.json"
            if protocol == "jevany_pointer_v1"
            else "d3_manifest.json"
            if protocol in D3_PROTOCOLS
            else "jev_manifest.json"
        )
        (temporary / name).write_text(json.dumps(manifest, indent=2) + "\n")
        verify(temporary, full=True)
        temporary.replace(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output


def prepare(model_id: str, workspace: Path, protocol: str = "auto") -> Path:
    if model_id in PROFILES:
        model_id = PROFILES[model_id][0]
    validate_repo_id(model_id)
    if protocol not in (
        "auto",
        CHOICE_PROTOCOL,
        BRANCH_PROTOCOL,
        TINY_PROTOCOL,
        "laya_markers_v1",
        "valen_qwen_v1",
        "vjev_vision_v1",
        "rsijev_xattn_v1",
        "clef_joint_v1",
        "jevany_pointer_v1",
        *D3_PROTOCOLS,
        *DECISION_PROTOCOLS,
    ):
        raise ValueError(f"unknown protocol: {protocol}")
    known = {spec[0]: name for name, spec in PROFILES.items()}
    if model_id in known:
        if protocol not in ("auto", CHOICE_PROTOCOL):
            raise ValueError(f"{model_id} uses {CHOICE_PROTOCOL}, not {protocol}")
        return prepare_open_jev(known[model_id], workspace)

    workspace = workspace.resolve()
    output = workspace / "checkpoint" / model_id
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock"):
        if sys.platform == "darwin":
            from .laya_export import MODELS as LAYA_MODELS

            if model_id in LAYA_MODELS:
                if protocol not in ("auto", "laya_markers_v1"):
                    raise ValueError(f"{model_id} uses laya_markers_v1, not {protocol}")
                from .mac_laya import prepare as prepare_laya_mps

                return prepare_laya_mps(model_id, workspace)
        return _prepare(model_id, workspace, protocol)


LOCAL_PROTOCOLS = (
    "decider_slot_v1",
    "jevk5_letters_v1",
    "mica_labels_v1",
    "task_json_v1",
    "thisthat_slot_v1",
    *D3_PROTOCOLS,
)


def local_checkpoint(source: Path, workspace: Path) -> Path:
    """Key a local release's prepared checkpoint by a fingerprint of its files."""
    digest = hashlib.sha256()
    for file in sorted(source.resolve().iterdir()):
        if not file.is_file():
            continue
        digest.update(f"{file.name}\0{file.stat().st_size}\0".encode())
        if file.suffix != ".safetensors":
            digest.update(sha256(file).encode())
    return workspace.resolve() / "checkpoint" / "local" / digest.hexdigest()[:16]


def prepare_local(source: Path, workspace: Path, protocol: str = "auto") -> Path:
    """Prepare a raw decision release that is already on disk, without downloads."""
    from .decision_export import (
        export_decider,
        export_jevk5,
        export_mica,
        export_task_json,
        export_thisthat,
    )

    source = source.resolve()
    files = {file.name for file in source.iterdir() if file.is_file()}
    if D3_FILES <= files:
        selected = PPLX_PROTOCOL if PPLX_FILE in files else D3_PROTOCOL
    elif "jevk5_config.json" in files:
        selected = "jevk5_letters_v1"
    elif protocol in ("mica_labels_v1", "thisthat_slot_v1", "task_json_v1"):
        selected = protocol
    elif "decider_config.json" in files:
        selected = "decider_slot_v1"
    else:
        raise ValueError(
            f"{source} is neither a prepared checkpoint nor a supported local release; "
            f"local releases support: {', '.join(LOCAL_PROTOCOLS)}"
        )
    if protocol not in ("auto", selected):
        raise ValueError(f"{source} uses {selected}, not {protocol}")
    if selected == "task_json_v1" and "adapter_config.json" in files:
        raise ValueError("local task_json_v1 adapters must be merged before serving")
    if selected in D3_PROTOCOLS and sys.platform == "darwin":
        raise ValueError("d3 serving requires Linux and native vLLM")
    from .d3_export import export_d3
    from .pplx_export import export_pplx

    export = {
        "decider_slot_v1": export_decider,
        "jevk5_letters_v1": export_jevk5,
        "mica_labels_v1": export_mica,
        "task_json_v1": export_task_json,
        "thisthat_slot_v1": export_thisthat,
        D3_PROTOCOL: lambda source, output, model_id, _: export_d3(
            source, output, model_id
        ),
        PPLX_PROTOCOL: lambda source, output, model_id, _: export_pplx(
            source, output, model_id
        ),
    }[selected]
    name = "d3_manifest.json" if selected in D3_PROTOCOLS else "decision_manifest.json"
    output = local_checkpoint(source, workspace)
    output.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output) + ".lock"):
        if output.exists():
            manifest = json.loads((output / name).read_text())
            if manifest.get("prompt_protocol") != selected:
                raise ValueError("requested protocol differs from cached checkpoint")
            verify(output, full=True)
            return output
        revision = output.name
        return _publish(
            output,
            str(source),
            revision,
            selected,
            lambda temporary: export(source, temporary, str(source), revision),
        )


def _prepare_d3(
    model_id: str,
    workspace: Path,
    protocol: str,
    revision: str | None,
    selected: str = D3_PROTOCOL,
) -> Path:
    from .d3_export import SOURCE_FILES, export_d3, verify_d3
    from .pplx_export import SOURCE_FILES as PPLX_SOURCE_FILES
    from .pplx_export import export_pplx

    if sys.platform == "darwin":
        raise ValueError("d3 serving requires Linux and native vLLM")
    if protocol not in ("auto", selected):
        raise ValueError(f"{model_id} uses {selected}, not {protocol}")
    output = workspace / "checkpoint" / model_id
    if output.exists():
        manifest = json.loads((output / "d3_manifest.json").read_text())
        if manifest.get("source_repository") != model_id:
            raise ValueError("cached checkpoint belongs to another repository")
        if manifest.get("prompt_protocol") != selected:
            raise ValueError("requested protocol differs from cached checkpoint")
        verify_d3(output, full=True)
        return output
    if revision is None:
        revision = HfApi().model_info(model_id, files_metadata=False, token=False).sha
    pplx = selected == PPLX_PROTOCOL
    source = Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            token=False,
            local_dir=workspace / "public" / model_id,
            allow_patterns=PPLX_SOURCE_FILES if pplx else SOURCE_FILES,
        )
    )
    export = export_pplx if pplx else export_d3
    return _publish(
        output,
        model_id,
        revision,
        selected,
        lambda temporary: export(source, temporary, model_id),
    )


def _prepare(model_id: str, workspace: Path, protocol: str) -> Path:
    from .d3_export import MODELS as D3_MODELS
    from .jevany_export import MODELS as JEVANY_MODELS
    from .pplx_export import MODELS as PPLX_MODELS

    cached = workspace / "checkpoint" / model_id / "d3_manifest.json"
    cached_protocol = (
        json.loads(cached.read_text()).get("prompt_protocol")
        if cached.is_file()
        else None
    )
    if (
        model_id in PPLX_MODELS
        or protocol == PPLX_PROTOCOL
        or (protocol == "auto" and cached_protocol == PPLX_PROTOCOL)
    ):
        return _prepare_d3(
            model_id, workspace, protocol, PPLX_MODELS.get(model_id), PPLX_PROTOCOL
        )
    if model_id in D3_MODELS or protocol == D3_PROTOCOL or cached.is_file():
        return _prepare_d3(model_id, workspace, protocol, D3_MODELS.get(model_id))

    if model_id in JEVANY_MODELS:
        from .jevany_export import export_jevany_pointer, verify_jevany_pointer

        if sys.platform == "darwin":
            raise ValueError("JevAny Pointer serving requires Linux and native vLLM")
        if protocol not in ("auto", "jevany_pointer_v1"):
            raise ValueError(f"{model_id} uses jevany_pointer_v1, not {protocol}")
        output = workspace / "checkpoint" / model_id
        if output.exists():
            manifest = json.loads((output / "jevany_pointer_manifest.json").read_text())
            if manifest.get("source_repository") != model_id:
                raise ValueError("cached checkpoint belongs to another repository")
            verify_jevany_pointer(output, full=True)
            return output
        source = Path(
            snapshot_download(
                repo_id=model_id,
                revision=JEVANY_MODELS[model_id],
                token=False,
                local_dir=workspace / "public" / model_id,
            )
        )
        import torch

        meta = torch.load(source / "head.pt", map_location="cpu", weights_only=True)
        if not isinstance(meta.get("base"), str) or not meta.get("base_revision"):
            raise ValueError("JevAny release must identify a pinned base checkpoint")
        base = Path(
            snapshot_download(
                repo_id=meta["base"],
                revision=meta["base_revision"],
                token=False,
                cache_dir=os.environ.get("HF_HUB_CACHE"),
            )
        )
        return _publish(
            output,
            model_id,
            JEVANY_MODELS[model_id],
            "jevany_pointer_v1",
            lambda temporary: export_jevany_pointer(base, source, temporary, model_id),
        )

    from .laya_export import MODELS as LAYA_MODELS

    if model_id in LAYA_MODELS:
        if protocol not in ("auto", "laya_markers_v1"):
            raise ValueError(f"{model_id} uses laya_markers_v1, not {protocol}")
        from .laya_export import (
            SOURCE_FILES,
            export_laya,
            verify_laya,
        )

        output = workspace / "checkpoint" / model_id
        if output.exists():
            verify_laya(output, full=True)
            return output
        source = Path(
            snapshot_download(
                repo_id=model_id,
                revision=LAYA_MODELS[model_id][0],
                token=False,
                local_dir=workspace / "public" / model_id,
                allow_patterns=SOURCE_FILES,
            )
        )
        temporary = Path(tempfile.mkdtemp(prefix=".laya-", dir=output.parent))
        try:
            export_laya(source, temporary, model_id)
            verify_laya(temporary, full=True)
            temporary.replace(output)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return output

    from .clef_export import MODELS as CLEF_MODELS

    if model_id in CLEF_MODELS:
        if protocol not in ("auto", "clef_joint_v1"):
            raise ValueError(f"{model_id} uses clef_joint_v1, not {protocol}")
        from .clef_export import SOURCE_FILES, export_clef, verify_clef

        output = workspace / "checkpoint" / model_id
        if output.exists():
            manifest = json.loads((output / "clef_manifest.json").read_text())
            if manifest.get("source_repository") != model_id:
                raise ValueError("cached checkpoint belongs to another repository")
            verify_clef(output, full=True)
            return output
        source = Path(
            snapshot_download(
                repo_id=model_id,
                revision=CLEF_MODELS[model_id],
                token=False,
                local_dir=workspace / "public" / model_id,
                allow_patterns=SOURCE_FILES,
            )
        )
        temporary = Path(tempfile.mkdtemp(prefix=".clef-", dir=output.parent))
        try:
            export_clef(source, temporary, model_id)
            verify_clef(temporary, full=True)
            temporary.replace(output)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return output

    from .rsijev_export import MODELS as RSIJEV_MODELS

    if model_id in RSIJEV_MODELS:
        if protocol not in ("auto", "rsijev_xattn_v1"):
            raise ValueError(f"{model_id} uses rsijev_xattn_v1, not {protocol}")
        from .rsijev_export import (
            BASE_FILES,
            BASE_ID,
            BASE_REVISION,
            SOURCE_FILES,
            export_rsijev,
            verify_rsijev,
        )

        output = workspace / "checkpoint" / model_id
        if output.exists():
            manifest = json.loads((output / "rsijev_manifest.json").read_text())
            if manifest.get("source_repository") != model_id:
                raise ValueError("cached checkpoint belongs to another repository")
            verify_rsijev(output, full=True)
            return output
        source = Path(
            snapshot_download(
                repo_id=model_id,
                revision=RSIJEV_MODELS[model_id],
                token=False,
                local_dir=workspace / "public" / model_id,
                allow_patterns=SOURCE_FILES,
            )
        )
        base = Path(
            snapshot_download(
                repo_id=BASE_ID,
                revision=BASE_REVISION,
                token=False,
                cache_dir=os.environ.get("HF_HUB_CACHE"),
                allow_patterns=BASE_FILES,
            )
        )
        temporary = Path(tempfile.mkdtemp(prefix=".rsijev-", dir=output.parent))
        try:
            export_rsijev(source, base, temporary, model_id)
            verify_rsijev(temporary, full=True)
            temporary.replace(output)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return output

    from .vjev_export import REVISIONS as VJEV_REVISIONS

    if model_id in VJEV_REVISIONS:
        if protocol not in ("auto", "vjev_vision_v1"):
            raise ValueError(f"{model_id} uses vjev_vision_v1, not {protocol}")
        from .vjev_export import export_vjev, verify_vjev

        output = workspace / "checkpoint" / model_id
        if output.exists():
            manifest = json.loads((output / "vjev_manifest.json").read_text())
            if manifest.get("source_repository") != model_id:
                raise ValueError("cached checkpoint belongs to another repository")
            verify_vjev(output, full=True)
            return output
        source = Path(
            snapshot_download(
                repo_id=model_id,
                revision=VJEV_REVISIONS[model_id],
                token=False,
                local_dir=workspace / "public" / model_id,
                allow_patterns=[
                    "model-*.safetensors",
                    "model.safetensors.index.json",
                    "config.json",
                    "tokenizer*.json",
                    "processor_config.json",
                    "chat_template.jinja",
                    "generation_config.json",
                    "head.pt",
                    "vjev.json",
                ],
            )
        )
        temporary = Path(tempfile.mkdtemp(prefix=".vjev-", dir=output.parent))
        try:
            export_vjev(source, temporary, model_id)
            verify_vjev(temporary, full=True)
            temporary.replace(output)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return output

    from .valen_export import MODEL_ID as VALEN_ID

    if model_id == VALEN_ID:
        if protocol not in ("auto", "valen_qwen_v1"):
            raise ValueError(f"{model_id} uses valen_qwen_v1, not {protocol}")
        from .valen_export import (
            BASE_ID,
            BASE_REVISION,
            MODEL_REVISION,
            export_valen,
            verify_valen,
        )

        workspace = workspace.resolve()
        output = workspace / "checkpoint" / model_id
        if output.exists():
            verify_valen(output, full=True)
            return output
        source = Path(
            snapshot_download(
                repo_id=model_id,
                revision=MODEL_REVISION,
                token=False,
                local_dir=workspace / "public" / model_id,
                allow_patterns=[
                    "checkpoint.pt",
                    "config.json",
                    "training_metadata.json",
                ],
            )
        )
        base = Path(
            snapshot_download(
                repo_id=BASE_ID,
                revision=BASE_REVISION,
                token=False,
                cache_dir=os.environ.get("HF_HUB_CACHE"),
            )
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".valen-", dir=output.parent))
        try:
            export_valen(base, source, temporary)
            verify_valen(temporary, full=True)
            temporary.replace(output)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return output

    workspace = workspace.resolve()
    output = workspace / "checkpoint" / model_id
    if output.exists():
        name = (
            "decision_manifest.json"
            if (output / "decision_manifest.json").is_file()
            else "jev_manifest.json"
        )
        manifest = json.loads((output / name).read_text())
        if manifest.get("source_repository") != model_id:
            raise ValueError("cached checkpoint belongs to another repository")
        if protocol not in ("auto", manifest.get("prompt_protocol")):
            raise ValueError("requested protocol differs from cached checkpoint")
        verify(output, full=True)
        return output

    info = HfApi().model_info(model_id, files_metadata=False, token=False)
    files = {item.rfilename for item in info.siblings}
    if D3_FILES <= files:
        selected = PPLX_PROTOCOL if PPLX_FILE in files else D3_PROTOCOL
        return _prepare_d3(model_id, workspace, protocol, info.sha, selected)
    from .decision_protocols import TASK_JSON_MODELS, THIS_THAT_MODELS

    if "jevk5_config.json" in files or protocol == "jevk5_letters_v1":
        from .decision_export import export_jevk5

        if protocol not in ("auto", "jevk5_letters_v1"):
            raise ValueError("this checkpoint uses jevk5_letters_v1")
        source = Path(
            snapshot_download(
                repo_id=model_id,
                revision=info.sha,
                token=False,
                local_dir=workspace / "public" / model_id,
                allow_patterns=[
                    "model*.safetensors",
                    "*.json",
                    "*.model",
                    "merges.txt",
                    "*.jinja",
                ],
            )
        )
        return _publish(
            output,
            model_id,
            info.sha,
            "jevk5_letters_v1",
            lambda temporary: export_jevk5(source, temporary, model_id, info.sha),
        )
    if model_id == "sky7350/Mica-v0.1-4B" or protocol == "mica_labels_v1":
        from .decision_export import export_mica

        if protocol not in ("auto", "mica_labels_v1"):
            raise ValueError("this checkpoint uses mica_labels_v1")
        source = Path(
            snapshot_download(
                repo_id=model_id,
                revision=info.sha,
                token=False,
                local_dir=workspace / "public" / model_id,
                allow_patterns=[
                    "model*.safetensors",
                    "*.json",
                    "*.model",
                    "merges.txt",
                    "*.jinja",
                ],
            )
        )
        return _publish(
            output,
            model_id,
            info.sha,
            "mica_labels_v1",
            lambda temporary: export_mica(source, temporary, model_id, info.sha),
        )
    if model_id in THIS_THAT_MODELS or protocol == "thisthat_slot_v1":
        from .decision_export import export_thisthat

        if protocol not in ("auto", "thisthat_slot_v1"):
            raise ValueError("this checkpoint uses thisthat_slot_v1")
        source = Path(
            snapshot_download(
                repo_id=model_id,
                revision=info.sha,
                token=False,
                local_dir=workspace / "public" / model_id,
                allow_patterns=[
                    "model*.safetensors",
                    "*.json",
                    "*.model",
                    "merges.txt",
                    "*.jinja",
                ],
            )
        )
        return _publish(
            output,
            model_id,
            info.sha,
            "thisthat_slot_v1",
            lambda temporary: export_thisthat(source, temporary, model_id, info.sha),
        )
    if model_id in TASK_JSON_MODELS or protocol == "task_json_v1":
        from .decision_export import export_task_adapter, export_task_json

        if protocol not in ("auto", "task_json_v1"):
            raise ValueError("this checkpoint uses task_json_v1")
        source = Path(
            snapshot_download(
                repo_id=model_id,
                revision=info.sha,
                token=False,
                local_dir=workspace / "public" / model_id,
                allow_patterns=[
                    "model*.safetensors",
                    "*.json",
                    "adapter_model.safetensors",
                    "*.model",
                    "merges.txt",
                    "*.jinja",
                ],
            )
        )
        if (source / "adapter_config.json").is_file():
            spec = json.loads((source / "adapter_config.json").read_text())
            base_id = spec.get("base_model_name_or_path")
            if not isinstance(base_id, str):
                raise ValueError("decision adapter has no foundation model")
            base_info = HfApi().model_info(
                base_id, revision=spec.get("revision"), token=False
            )
            base = Path(
                snapshot_download(
                    repo_id=base_id,
                    revision=base_info.sha,
                    token=False,
                    cache_dir=os.environ.get("HF_HUB_CACHE"),
                )
            )

            def build(temporary):
                return export_task_adapter(base, source, temporary, model_id, info.sha)
        else:

            def build(temporary):
                return export_task_json(source, temporary, model_id, info.sha)

        return _publish(output, model_id, info.sha, "task_json_v1", build)
    if "decider_config.json" in files or {"head.pt", "adapter_config.json"} <= files:
        from .decision_export import export_decider, export_kev

        selected = (
            "decider_slot_v1" if "decider_config.json" in files else "kev_pointer_v1"
        )
        if protocol not in ("auto", selected):
            raise ValueError(f"{model_id} uses {selected}, not {protocol}")
        source = Path(
            snapshot_download(
                repo_id=model_id,
                revision=info.sha,
                token=False,
                local_dir=workspace / "public" / model_id,
                allow_patterns=[
                    "model*.safetensors",
                    "*.json",
                    "head.pt",
                    "adapter_model.safetensors",
                    "*.model",
                    "merges.txt",
                    "*.jinja",
                ],
            )
        )
        if selected == "decider_slot_v1":

            def build(temporary):
                return export_decider(source, temporary, model_id, info.sha)
        else:
            import torch

            meta = torch.load(source / "head.pt", map_location="cpu", weights_only=True)
            if not isinstance(meta.get("base"), str) or "head" not in meta:
                raise ValueError(
                    "repository does not contain a Kev pointer-head release"
                )
            base_info = HfApi().model_info(
                meta["base"], revision=meta.get("base_revision"), token=False
            )
            base = Path(
                snapshot_download(
                    repo_id=meta["base"],
                    revision=base_info.sha,
                    token=False,
                    cache_dir=os.environ.get("HF_HUB_CACHE"),
                )
            )

            def build(temporary):
                return export_kev(base, source, temporary, model_id, info.sha)

        return _publish(output, model_id, info.sha, selected, build)
    if SCALAR_FILES <= files:
        if protocol not in ("auto", CHOICE_PROTOCOL):
            raise ValueError(f"{model_id} uses {CHOICE_PROTOCOL}, not {protocol}")
        spec = json.loads(
            Path(
                hf_hub_download(
                    repo_id=model_id,
                    filename="package/checkpoint/model.json",
                    revision=info.sha,
                    token=False,
                )
            ).read_text()
        )
        base_id, base_revision = spec.get("model_id"), spec.get("revision")
        if (
            not isinstance(base_id, str)
            or not isinstance(base_revision, str)
            or len(base_revision) != 40
        ):
            raise ValueError("native scalar-head export requires a pinned base")
        base_config = json.loads(
            Path(
                hf_hub_download(
                    repo_id=base_id,
                    filename="config.json",
                    revision=base_revision,
                    token=False,
                )
            ).read_text()
        )
        if (
            base_config.get("model_type") != "qwen3_5"
            or base_config.get("text_config", {}).get("model_type") != "qwen3_5_text"
        ):
            raise ValueError(
                "base architecture is not supported by native Qwen3.5 pooling"
            )
        source = (
            Path(
                snapshot_download(
                    repo_id=model_id,
                    revision=info.sha,
                    token=False,
                    local_dir=workspace / "public" / model_id,
                    allow_patterns=sorted(SCALAR_FILES),
                )
            )
            / "package"
            / "checkpoint"
        )
        base = Path(
            snapshot_download(repo_id=base_id, revision=base_revision, token=False)
        )
        from .export import export

        return _publish(
            output,
            model_id,
            info.sha,
            CHOICE_PROTOCOL,
            lambda temporary: export(
                base, source / "adapter", source / "head.pt", temporary
            ),
        )
    if {"modeling_tiny_jev.py", "model.safetensors", "config.json"} <= files:
        if protocol not in ("auto", TINY_PROTOCOL):
            raise ValueError(f"{model_id} uses {TINY_PROTOCOL}, not {protocol}")
        from .tiny import export_tiny

        source = Path(
            snapshot_download(
                repo_id=model_id,
                revision=info.sha,
                token=False,
                local_dir=workspace / "public" / model_id,
                allow_patterns=[
                    "config.json",
                    "model.safetensors",
                    "tokenizer*",
                    "chat_template.jinja",
                ],
            )
        )
        return _publish(
            output,
            model_id,
            info.sha,
            TINY_PROTOCOL,
            lambda temporary: export_tiny(source, temporary, model_id, info.sha),
        )
    if not BRANCH_FILES <= files:
        if "openjet_runtime/candidate_projection.py" in files:
            raise ValueError(
                "this release uses a vocabulary-projection decision protocol"
            )
        if "open_jev_config.json" in files and "model.safetensors" in files:
            raise ValueError("this release uses an encoder/span decision head")
        raise ValueError(
            f"{model_id} has no supported native vLLM Jev checkpoint layout; "
            f"supported protocols: {CHOICE_PROTOCOL}, {BRANCH_PROTOCOL}, {TINY_PROTOCOL}"
        )
    if protocol not in ("auto", BRANCH_PROTOCOL):
        raise ValueError(f"{model_id} uses {BRANCH_PROTOCOL}, not {protocol}")

    package = Path(
        snapshot_download(
            repo_id=model_id,
            revision=info.sha,
            token=False,
            local_dir=workspace / "public" / model_id,
            allow_patterns=sorted(BRANCH_FILES),
        )
    )
    release = json.loads((package / "MANIFEST.json").read_text())
    config = json.loads((package / "openjev_config.json").read_text())
    adapter = json.loads((package / "adapter_config.json").read_text())
    if (
        config.get("architecture") != "SharedStateCandidateBranches"
        or config.get("model_name") != "Qwen/Qwen3-0.6B"
        or adapter.get("base_model_name_or_path") != config["model_name"]
        or adapter.get("revision") != config.get("model_revision")
        or release.get("base_model") != config["model_name"]
        or release.get("base_revision") != config.get("model_revision")
    ):
        raise ValueError(
            "repository is not a matching Qwen3 branch/scalar-head release"
        )
    for name in BRANCH_FILES - {"MANIFEST.json"}:
        if sha256(package / name) != release.get("files", {}).get(name):
            raise ValueError(f"release checksum mismatch: {name}")

    base = Path(
        snapshot_download(
            repo_id=config["model_name"],
            revision=config["model_revision"],
            token=False,
        )
    )
    from .export import export

    return _publish(
        output,
        model_id,
        info.sha,
        BRANCH_PROTOCOL,
        lambda temporary: export(
            base, package, package / "head.safetensors", temporary
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        choices=(
            "auto",
            CHOICE_PROTOCOL,
            BRANCH_PROTOCOL,
            TINY_PROTOCOL,
            "laya_markers_v1",
            "valen_qwen_v1",
            "vjev_vision_v1",
            "rsijev_xattn_v1",
            "clef_joint_v1",
            "jevany_pointer_v1",
            *D3_PROTOCOLS,
            *DECISION_PROTOCOLS,
        ),
        default="auto",
    )
    args = parser.parse_args()
    source = Path(args.model_id)
    print(
        "JEV_CHECKPOINT_READY",
        prepare_local(source, args.workspace, args.protocol)
        if source.is_dir()
        else prepare(args.model_id, args.workspace, args.protocol),
        flush=True,
    )


if __name__ == "__main__":
    main()
