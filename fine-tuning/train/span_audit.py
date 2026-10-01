"""Compute the challenge IoU span precision used only to break local-dev ties."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .provenance import sha256_file, write_json


def _iou(gold: dict[str, int], prediction: dict[str, int]) -> float:
    intersection = max(0, min(gold["end"], prediction["end"]) - max(gold["start"], prediction["start"]))
    if not intersection:
        return 0.0
    union = (gold["end"] - gold["start"]) + (prediction["end"] - prediction["start"]) - intersection
    return intersection / union if union else 0.0


def _normalize_type(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"lei", "legislacao", "legislação"}:
        return "legislacao"
    if normalized in {"jurisprudencia", "jurisprudência"}:
        return "jurisprudencia"
    return normalized


def span_audit(prediction_dir: Path, golden_csv: Path, report_path: Path) -> dict[str, Any]:
    gold_by_doc: dict[str, list[dict[str, int | str]]] = defaultdict(list)
    with golden_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            gold_by_doc[row["documento_id"]].append({
                "start": int(row["inicio"]), "end": int(row["fim"]),
                "type": _normalize_type(row["tipo"]),
            })

    predicted_by_doc: dict[str, list[dict[str, int | str]]] = {}
    for path in sorted((prediction_dir / "json").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        predicted_by_doc[path.stem] = [
            {
                "start": int(citation["inicio"]), "end": int(citation["fim"]),
                "type": _normalize_type(str(citation.get("tipo", "unknown"))),
            }
            for citation in payload.get("citacoes", [])
        ]

    counts: dict[str, Counter] = defaultdict(Counter)
    type_counts: dict[str, Counter] = defaultdict(Counter)
    for doc_id in sorted(set(gold_by_doc) | set(predicted_by_doc)):
        golds = gold_by_doc.get(doc_id, [])
        predictions = predicted_by_doc.get(doc_id, [])
        pairs = []
        for gold_index, gold in enumerate(golds):
            for prediction_index, prediction in enumerate(predictions):
                overlap = _iou(gold, prediction)
                if overlap >= 0.5:
                    pairs.append((-overlap, gold_index, prediction_index))
        pairs.sort()
        used_golds: set[int] = set()
        used_predictions: set[int] = set()
        matched_pairs = []
        for _negative_iou, gold_index, prediction_index in pairs:
            if gold_index in used_golds or prediction_index in used_predictions:
                continue
            used_golds.add(gold_index)
            used_predictions.add(prediction_index)
            matched_pairs.append((golds[gold_index], predictions[prediction_index]))
        match = re.search(r"_n([12])_", doc_id)
        level = match.group(1) if match else "unknown"
        bucket = counts[level]
        bucket["matched"] += len(used_predictions)
        bucket["gold"] += len(golds)
        bucket["predictions"] += len(predictions)
        for gold in golds:
            type_counts[str(gold["type"])]["gold"] += 1
        for prediction in predictions:
            type_counts[str(prediction["type"])]["predictions"] += 1
        typed_matches = 0
        for gold, prediction in matched_pairs:
            gold_type = str(gold["type"])
            prediction_type = str(prediction["type"])
            if gold_type == prediction_type:
                typed_matches += 1
                type_counts[gold_type]["matched"] += 1
        bucket["typed_matched"] += typed_matches

    per_level = {
        level: {
            **dict(bucket),
            "precision": bucket["matched"] / bucket["predictions"] if bucket["predictions"] else 0.0,
            "recall": bucket["matched"] / bucket["gold"] if bucket["gold"] else 0.0,
            "typed_precision": bucket["typed_matched"] / bucket["predictions"] if bucket["predictions"] else 0.0,
            "typed_recall": bucket["typed_matched"] / bucket["gold"] if bucket["gold"] else 0.0,
        }
        for level, bucket in sorted(counts.items())
    }
    total = sum(counts.values(), Counter())
    result = {
        "iou_threshold": 0.5,
        "precision": total["matched"] / total["predictions"] if total["predictions"] else 0.0,
        "recall": total["matched"] / total["gold"] if total["gold"] else 0.0,
        "typed_precision": total["typed_matched"] / total["predictions"] if total["predictions"] else 0.0,
        "typed_recall": total["typed_matched"] / total["gold"] if total["gold"] else 0.0,
        "per_type": {
            kind: {
                **dict(bucket),
                "precision": bucket["matched"] / bucket["predictions"] if bucket["predictions"] else 0.0,
                "recall": bucket["matched"] / bucket["gold"] if bucket["gold"] else 0.0,
            }
            for kind, bucket in sorted(type_counts.items())
        },
        "matched_spans": total["matched"],
        "gold_spans": total["gold"],
        "predicted_spans": total["predictions"],
        "per_level": per_level,
        "gold_sha256": sha256_file(golden_csv),
        "prediction_directory": str(prediction_dir),
    }
    write_json(report_path, result)
    return result
