"""Frozen-run checks reject legacy coordinate spaces and changed LF artifacts."""
import hashlib
import json
from pathlib import Path

import pytest

from scripts.evaluate import verify_frozen_run
from src.config import NER_MODEL_DIR, ROOT_DIR
from src.resolution.header_index import HeaderIndex


def test_frozen_run_rejects_manifest_without_lf_policy(tmp_path):
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(ValueError, match="não declara offsets LF"):
        verify_frozen_run(tmp_path / "submission.csv", manifest)


def test_frozen_run_rejects_modified_submission_and_lf_artifacts(tmp_path, monkeypatch):
    output = tmp_path / "output"
    submission = output / "submission.csv"
    submission.parent.mkdir(parents=True)
    submission.write_bytes(b"documento_id,citacoes\nfixture,-\n")
    (output / "text_lf").mkdir()
    (output / "text_lf" / "fixture.txt").write_text("texto\n", encoding="utf-8")
    (output / "json").mkdir()
    (output / "json" / "fixture.json").write_text("{}", encoding="utf-8")
    (output / "json_original").mkdir()
    (output / "json_original" / "fixture.json").write_text("{}", encoding="utf-8")

    original_fingerprint = HeaderIndex.fingerprint
    fixed_paths = {
        (ROOT_DIR / "src").resolve(),
        (ROOT_DIR / "scripts").resolve(),
        (ROOT_DIR / "requirements.txt").resolve(),
        NER_MODEL_DIR.resolve(),
        (NER_MODEL_DIR / "model.safetensors").resolve(),
        (ROOT_DIR / "data" / "desafio-jusbrasil-bracis-2026" / "kaggle_metric.py").resolve(),
    }

    def fingerprint(path):
        resolved = Path(path).resolve()
        if resolved in fixed_paths:
            return "frozen-code"
        return original_fingerprint(path)

    monkeypatch.setattr(HeaderIndex, "fingerprint", staticmethod(fingerprint))
    manifest = {
        "offset_policy": "LF",
        "offset_policy_version": 1,
        "submission_offset_space": "LF",
        "source_sha256": "frozen-code",
        "scripts_sha256": "frozen-code",
        "dependencies_sha256": "frozen-code",
        "checkpoint_sha256": "frozen-code",
        "model_weights_sha256": "frozen-code",
        "metric_sha256": "frozen-code",
        "submission_sha256": hashlib.sha256(submission.read_bytes()).hexdigest(),
        "normalized_texts_sha256": HeaderIndex.fingerprint(output / "text_lf"),
        "lf_json_sha256": HeaderIndex.fingerprint(output / "json"),
        "original_json_sha256": HeaderIndex.fingerprint(output / "json_original"),
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    verify_frozen_run(submission, manifest_path)
    (output / "text_lf" / "fixture.txt").write_text("alterado\n", encoding="utf-8")
    with pytest.raises(ValueError, match="normalized_texts_sha256"):
        verify_frozen_run(submission, manifest_path)

    (output / "text_lf" / "fixture.txt").write_text("texto\n", encoding="utf-8")
    submission.write_bytes(b"arquivo alterado")
    with pytest.raises(ValueError, match="submission_sha256"):
        verify_frozen_run(submission, manifest_path)
