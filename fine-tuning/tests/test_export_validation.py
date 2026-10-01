from __future__ import annotations

import os
from pathlib import Path

from train.config import NER_LABELS, SOURCE_LICENSE, SOURCE_MODEL, SOURCE_REVISION
from train.cli import _record_offline_validation
from train.provenance import sha256_file, write_json
from train.validate_export import validate


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get("FT_TEST_SOURCE_MODEL", ROOT / "models" / "ner_legal_large"))


def test_derived_checkpoint_loads_offline_with_thirteen_labels(tmp_path: Path) -> None:
    from transformers import AutoTokenizer, BertConfig, BertForTokenClassification

    tokenizer = AutoTokenizer.from_pretrained(str(SOURCE), local_files_only=True)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        num_labels=len(NER_LABELS),
        id2label={index: label for index, label in enumerate(NER_LABELS)},
        label2id={label: index for index, label in enumerate(NER_LABELS)},
    )
    config.base_model_revision = SOURCE_REVISION
    model = BertForTokenClassification(config)
    model.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    files = {
        path.name: sha256_file(path)
        for path in sorted(checkpoint.iterdir())
        if path.is_file()
    }
    write_json(checkpoint / "provenance.json", {
        "source_model": SOURCE_MODEL,
        "revision": SOURCE_REVISION,
        "license": SOURCE_LICENSE,
        "files": files,
    })
    result = validate(checkpoint)
    assert result["valid"] is True
    assert result["num_labels"] == 13
    assert result["offline_load_and_smoke_inference"] is True


def test_record_validation_requires_matching_exported_weights(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model_hash = "a" * 64
    write_json(run_dir / "final_export.json", {"model_weights_sha256": model_hash, "validated_offline": False})
    report = tmp_path / "validation.json"
    write_json(report, {
        "valid": True,
        "offline_load_and_smoke_inference": True,
        "model_sha256": model_hash,
        "num_labels": 13,
    })
    updated = _record_offline_validation(run_dir, report)
    assert updated["validated_offline"] is True
    assert updated["offline_validation_report_sha256"] == sha256_file(report)
