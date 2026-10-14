"""CPU regressions for selecting a previously prepared checkpoint."""

import json

import pytest

from vllm_jev import clef_export, native_models


@pytest.mark.parametrize("model_id", sorted(clef_export.MODELS))
def test_clef_cache_belongs_to_requested_model(tmp_path, monkeypatch, model_id):
    output = tmp_path / "checkpoint" / model_id
    output.mkdir(parents=True)
    manifest = output / "clef_manifest.json"
    verified = []
    monkeypatch.setattr(
        clef_export,
        "verify_clef",
        lambda path, *, full: verified.append((path, full)),
    )
    monkeypatch.setattr(
        native_models,
        "snapshot_download",
        lambda **kwargs: pytest.fail("an existing checkpoint must not download"),
    )

    other_model = next(name for name in clef_export.MODELS if name != model_id)
    manifest.write_text(json.dumps({"source_repository": other_model}))
    with pytest.raises(ValueError, match="another repository"):
        native_models._prepare(model_id, tmp_path, "auto")
    assert not verified

    manifest.write_text(json.dumps({"source_repository": model_id}))
    assert native_models._prepare(model_id, tmp_path, "auto") == output
    assert verified == [(output, True)]


def _decider_release(path):
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"model_type": "qwen3_5_text"}))
    (path / "decider_config.json").write_text(json.dumps({"temperature": 1.0}))
    (path / "model.safetensors").write_bytes(b"weights")
    return path


def test_local_release_is_keyed_by_its_contents(tmp_path):
    source = _decider_release(tmp_path / "saved_model")
    first = native_models.local_checkpoint(source, tmp_path / "ws")
    assert first.parent == (tmp_path / "ws" / "checkpoint" / "local").resolve()
    assert native_models.local_checkpoint(source, tmp_path / "ws") == first
    (source / "decider_config.json").write_text(json.dumps({"temperature": 2.0}))
    assert native_models.local_checkpoint(source, tmp_path / "ws") != first


def test_local_decider_release_exports_without_downloading(tmp_path, monkeypatch):
    source = _decider_release(tmp_path / "saved_model")
    published = {}

    def publish(output, model_id, revision, protocol, build):
        published.update(output=output, model_id=model_id, protocol=protocol)
        return output

    monkeypatch.setattr(native_models, "_publish", publish)
    monkeypatch.setattr(
        native_models,
        "snapshot_download",
        lambda **kwargs: pytest.fail("a local release must not download"),
    )
    output = native_models.prepare_local(source, tmp_path / "ws")
    assert output == native_models.local_checkpoint(source, tmp_path / "ws")
    assert published["protocol"] == "decider_slot_v1"
    assert published["model_id"] == str(source.resolve())
    with pytest.raises(ValueError, match="uses decider_slot_v1"):
        native_models.prepare_local(source, tmp_path / "ws", "kev_pointer_v1")


def test_local_directory_without_a_release_is_rejected(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="supported local release"):
        native_models.prepare_local(tmp_path / "empty", tmp_path / "ws")


def test_cli_prepares_a_raw_local_release(tmp_path, monkeypatch):
    from vllm_jev import cli

    source = _decider_release(tmp_path / "saved_model")
    workspace = tmp_path / "ws"
    expected = native_models.local_checkpoint(source, workspace)
    calls = {}

    def run(command, **kwargs):
        calls["prepare"] = command
        expected.mkdir(parents=True)
        (expected / "decision_manifest.json").write_text(
            json.dumps({"prompt_protocol": "decider_slot_v1"})
        )

    monkeypatch.setattr(cli.subprocess, "run", run)
    monkeypatch.setattr(
        cli.os, "execvpe", lambda exe, argv, env: calls.update(argv=argv)
    )
    monkeypatch.setattr(cli.sys, "platform", "linux")
    monkeypatch.setattr(
        cli.sys,
        "argv",
        ["vllm-jev", "serve", str(source), "--workspace", str(workspace)],
    )
    cli.main()
    assert calls["prepare"][1:4] == [
        "-m",
        "vllm_jev.native_models",
        str(source.resolve()),
    ]
    argv = calls["argv"]
    assert argv[argv.index("serve") + 1] == str(expected)
    assert argv[argv.index("--served-model-name") + 1] == str(source)
    assert argv[argv.index("--convert") + 1] == "embed"
