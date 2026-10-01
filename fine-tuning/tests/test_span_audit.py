from __future__ import annotations

import csv
import json
from pathlib import Path

from train.span_audit import span_audit


def test_span_and_type_scores_are_reported_by_level_and_type(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "prediction"
    json_dir = prediction_dir / "json"
    json_dir.mkdir(parents=True)
    payload = {
        "documento_id": "case_n1_test",
        "citacoes": [
            {"inicio": 0, "fim": 10, "tipo": "lei"},
            {"inicio": 20, "fim": 30, "tipo": "lei"},
            {"inicio": 40, "fim": 50, "tipo": "jurisprudencia"},
        ],
    }
    (json_dir / "case_n1_test.json").write_text(json.dumps(payload), encoding="utf-8")
    golden = tmp_path / "gold.csv"
    with golden.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["documento_id", "inicio", "fim", "tipo"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow({"documento_id": "case_n1_test", "inicio": 0, "fim": 10, "tipo": "legislacao"})
        writer.writerow({"documento_id": "case_n1_test", "inicio": 20, "fim": 30, "tipo": "jurisprudencia"})
    report = span_audit(prediction_dir, golden, tmp_path / "report.json")
    assert report["precision"] == 2 / 3
    assert report["recall"] == 1.0
    assert report["typed_precision"] == 1 / 3
    assert report["typed_recall"] == 0.5
    assert report["per_level"]["1"]["typed_recall"] == 0.5
    assert report["per_type"]["legislacao"]["precision"] == 0.5
    assert report["per_type"]["jurisprudencia"]["recall"] == 0.0
