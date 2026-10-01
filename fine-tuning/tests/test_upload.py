from __future__ import annotations

from pathlib import Path

import pytest

import train.cli as cli
import train.validate_export as validate_export


@pytest.mark.parametrize("private", [False, True])
def test_upload_validates_offline_before_creating_or_uploading_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, private: bool
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    calls: list[tuple[str, object]] = []
    weights_sha = "a" * 64
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(cli, "_read_json", lambda _path: {"files": {"model.safetensors": weights_sha}})
    monkeypatch.setattr(cli, "sha256_file", lambda _path: weights_sha)

    def validate(path: Path) -> dict:
        calls.append(("validate", path))
        return {"valid": True, "offline_load_and_smoke_inference": True}

    monkeypatch.setattr(validate_export, "validate", validate)

    class FakeHfApi:
        def __init__(self, token: str) -> None:
            assert token == "test-token"

        def create_repo(self, **kwargs: object) -> None:
            calls.append(("create_repo", kwargs))

        def upload_folder(self, **kwargs: object) -> None:
            calls.append(("upload_folder", kwargs))

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)
    cli._upload(checkpoint, "owner/legal-ner", private)

    assert [call[0] for call in calls] == ["validate", "create_repo", "upload_folder"]
    create_repo = calls[1][1]
    upload_folder = calls[2][1]
    assert create_repo["private"] is private
    assert upload_folder["repo_id"] == "owner/legal-ner"
    assert "model.safetensors" in upload_folder["allow_patterns"]
    assert "dataset_manifest.json" in upload_folder["allow_patterns"]
    assert not any(".sqlite" in pattern or "goldenset" in pattern for pattern in upload_folder["allow_patterns"])


def test_upload_does_not_contact_hub_when_offline_smoke_validation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    weights_sha = "a" * 64
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(cli, "_read_json", lambda _path: {"files": {"model.safetensors": weights_sha}})
    monkeypatch.setattr(cli, "sha256_file", lambda _path: weights_sha)
    monkeypatch.setattr(
        validate_export,
        "validate",
        lambda _path: {"valid": True, "offline_load_and_smoke_inference": False},
    )

    with pytest.raises(ValueError, match="validação offline"):
        cli._upload(checkpoint, "owner/legal-ner", private=True)
