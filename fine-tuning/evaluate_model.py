"""Run frozen pipeline diagnostics and score local or held-out official data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import re
import random
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("/app")
MODEL_DIR = ROOT / "models" / "ner_legal_large"
OFFICIAL_ROOT = ROOT / "data" / "desafio-jusbrasil-bracis-2026"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OFFICIAL_ROOT))
import kaggle_metric


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _level(document_id: str) -> int:
    match = re.search(r"_n([12])_", document_id)
    if not match:
        raise ValueError(f"ID sem nível N1/N2: {document_id}")
    return int(match.group(1))


def _expected_texts(input_dir: Path, expected_ids: set[str] | None = None) -> dict[str, str]:
    texts: dict[str, str] = {}
    for path in sorted(input_dir.glob("*.txt")):
        text = path.read_bytes().decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
        texts[path.stem] = text
    if expected_ids is not None and set(texts) != expected_ids:
        raise ValueError(
            f"O subconjunto oficial diverge dos IDs reservados: "
            f"ausentes={sorted(expected_ids-set(texts))}; extras={sorted(set(texts)-expected_ids)}."
        )
    if not texts:
        raise ValueError(f"Nenhum .txt encontrado em {input_dir}.")
    return texts


def _solution_from_csv(golden_path: Path, document_ids: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(
        golden_path,
        dtype={"documento_id": str, "id_canonico": str, "inicio": int, "fim": int, "nivel": int},
    )
    observed = set(frame["documento_id"].astype(str))
    extras = observed - document_ids
    if extras:
        raise ValueError(f"O gabarito contém documentos fora do subconjunto: {sorted(extras)}.")
    grouped: dict[str, list[str]] = {doc_id: [] for doc_id in document_ids}
    level_by_doc = {doc_id: _level(doc_id) for doc_id in document_ids}
    for row in frame.to_dict(orient="records"):
        doc_id = str(row["documento_id"])
        level_by_doc[doc_id] = int(row["nivel"])
        citation_type = str(row["classificacao"]).strip().lower()
        raw_ids = row.get("id_canonico")
        canonical_ids = str(raw_ids).strip() if pd.notna(raw_ids) else "-"
        if citation_type == "real" and canonical_ids not in ("", "-", "nan"):
            canonical_ids = canonical_ids.lstrip("0") or "0"
        else:
            canonical_ids = "-"
        grouped[doc_id].append(
            f"{int(row['inicio'])},{int(row['fim'])},{citation_type},{canonical_ids}"
        )
    return pd.DataFrame([
        {"documento_id": doc_id, "nivel": level_by_doc[doc_id], "citacoes": "|".join(grouped[doc_id]) or "-"}
        for doc_id in sorted(document_ids)
    ])


def _context_group(document_id: str) -> str:
    synthetic = re.fullmatch(r"syn_n[12]_(\d{6})", document_id)
    if synthetic:
        return f"synthetic_{(int(synthetic.group(1)) - 1) // 3 + 1:03d}"
    lener = re.fullmatch(r"lener_(?:train|dev|test)_n[12]_(.+)", document_id)
    if lener:
        return f"lener_{lener.group(1)}"
    return document_id


def _context_bootstrap(
    solution: pd.DataFrame, submission: pd.DataFrame, document_ids: set[str],
    bootstrap_seed: int = 42, replicates: int = 1000,
) -> dict[str, Any]:
    """Cluster bootstrap of the mean per-context score; diagnostic, not challenge score."""
    grouped: dict[str, set[str]] = {}
    for document_id in document_ids:
        grouped.setdefault(_context_group(document_id), set()).add(document_id)
    context_scores = []
    negative_contexts = []
    for group, ids in sorted(grouped.items()):
        gold = solution[solution["documento_id"].astype(str).isin(ids)]
        predicted = submission[submission["documento_id"].astype(str).isin(ids)][["documento_id", "citacoes"]]
        has_gold_citations = gold["citacoes"].fillna("").astype(str).str.strip().map(
            lambda cell: cell not in {"", "-"}
        ).any()
        if not has_gold_citations:
            negative_contexts.append({
                "context_group": group,
                "document_count": len(ids),
                "reason": "no_gold_citations; excluded because the official score is undefined for an all-negative subset",
            })
            continue
        group_score = kaggle_metric.avaliar(gold, predicted)
        context_scores.append({"context_group": group, "document_count": len(ids),
                               "score_final": float(group_score["score_final"])})
    if not context_scores:
        raise ValueError("Não há grupos para o bootstrap local por contexto.")
    values = [row["score_final"] for row in context_scores]
    rng = random.Random(bootstrap_seed)
    bootstrap_means = sorted(
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(replicates)
    )
    lower = bootstrap_means[int(0.025 * (replicates - 1))]
    upper = bootstrap_means[int(0.975 * (replicates - 1))]
    return {
        "method": "context-cluster bootstrap of unweighted per-context score_final means; groups with no gold citations excluded",
        "interpretation": "descriptive diagnostic; the challenge score remains computed once on the complete document set, and negative documents remain covered by false-positive audits",
        "seed": bootstrap_seed, "replicates": replicates,
        "total_context_group_count": len(grouped),
        "context_count": len(context_scores),
        "negative_context_count_excluded_from_score_bootstrap": len(negative_contexts),
        "negative_context_groups": negative_contexts,
        "mean_context_score": sum(values) / len(values),
        "bootstrap_95_percent_interval": [lower, upper],
        "context_scores": context_scores,
    }


def _span_metrics(golden_path: Path, stage_rows: dict[str, list[dict[str, Any]]], document_ids: set[str]) -> dict[str, Any]:
    frame = pd.read_csv(golden_path, dtype={"documento_id": str, "inicio": int, "fim": int})
    gold_by_doc: dict[str, list[dict[str, Any]]] = {doc_id: [] for doc_id in document_ids}
    for row in frame.to_dict(orient="records"):
        gold_by_doc[row["documento_id"]].append({
            "start": int(row["inicio"]), "end": int(row["fim"]),
            "type": str(row["tipo"]).strip().lower(),
        })
    result: dict[str, Any] = {}
    negative_documents = {doc_id for doc_id, spans in gold_by_doc.items() if not spans}

    def score_subset(by_doc: dict[str, list[dict[str, Any]]], docs: set[str], kind: str | None = None) -> dict[str, Any]:
        matched = gold_count = predicted = typed_matched = exact_matched = 0
        for doc_id in docs:
            gold = [span for span in gold_by_doc.get(doc_id, []) if kind is None or span["type"] == kind]
            predictions = [span for span in by_doc.get(doc_id, []) if kind is None or span["type"] == kind]
            gold_count += len(gold)
            predicted += len(predictions)
            candidates = []
            for gi, g in enumerate(gold):
                for pi, p in enumerate(predictions):
                    intersection = max(0, min(g["end"], p["end"]) - max(g["start"], p["start"]))
                    union = (g["end"] - g["start"]) + (p["end"] - p["start"]) - intersection
                    iou = intersection / union if intersection and union else 0.0
                    if iou >= 0.5:
                        candidates.append((-iou, gi, pi))
            used_g: set[int] = set()
            used_p: set[int] = set()
            for _neg_iou, gi, pi in sorted(candidates):
                if gi in used_g or pi in used_p:
                    continue
                used_g.add(gi)
                used_p.add(pi)
                matched += 1
                typed_matched += gold[gi]["type"] == predictions[pi]["type"]
                exact_matched += gold[gi]["start"] == predictions[pi]["start"] and gold[gi]["end"] == predictions[pi]["end"]
        return {
            "iou_threshold": 0.5,
            "matched": matched,
            "typed_matched": typed_matched,
            "exact_span_matched": exact_matched,
            "gold": gold_count,
            "predictions": predicted,
            "precision": matched / predicted if predicted else 0.0,
            "recall": matched / gold_count if gold_count else 0.0,
            "typed_precision": typed_matched / predicted if predicted else 0.0,
            "typed_recall": typed_matched / gold_count if gold_count else 0.0,
        }

    for stage, by_doc in stage_rows.items():
        overall = score_subset(by_doc, document_ids)
        result[stage] = {
            **overall,
            "gold_negative_documents": len(negative_documents),
            "false_positives_in_gold_negative_documents": sum(
                len(by_doc.get(doc_id, [])) for doc_id in negative_documents
            ),
            "per_level": {
                str(level): score_subset(by_doc, {
                    doc_id for doc_id in document_ids if _level(doc_id) == level
                })
                for level in (1, 2)
            },
            "per_type": {
                kind: score_subset(by_doc, document_ids, kind)
                for kind in ("legislacao", "jurisprudencia")
            },
        }
    return result


def _diagnostic_pipeline(input_dir: Path, database: Path, output_dir: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    pipeline = importlib.import_module("src.pipeline")
    active: dict[str, Any] = {"document": None}
    captured: dict[str, dict[str, list[dict[str, Any]]]] = {
        "regex": {}, "ner": {}, "fused": {}, "resolved": {},
    }
    originals = {
        name: getattr(pipeline, name)
        for name in ("process_document", "extract_regex_citations", "extract_ner_citations", "fuse_extractions", "classify_citation")
    }

    def record(kind: str, rows: list[dict[str, Any]]) -> None:
        document_id = active["document"]
        if document_id is not None:
            captured[kind][document_id] = [dict(row) for row in rows]

    def regex(text: str) -> list[dict[str, Any]]:
        rows = originals["extract_regex_citations"](text)
        record("regex", rows)
        return rows

    def ner(text: str, model: Any, tokenizer: Any) -> list[dict[str, Any]]:
        rows = originals["extract_ner_citations"](text, model, tokenizer)
        record("ner", rows)
        return rows

    def fuse(text: str, regex_rows: list[dict[str, Any]], ner_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = originals["fuse_extractions"](text, regex_rows, ner_rows)
        record("fused", rows)
        return rows

    def classify(citation: dict[str, Any], resolution: dict[str, Any]) -> dict[str, Any]:
        decision = originals["classify_citation"](citation, resolution)
        document_id = active["document"]
        if document_id is not None:
            captured["resolved"].setdefault(document_id, []).append({
                "inicio": citation["inicio"], "fim": citation["fim"],
                "tipo": citation.get("tipo"), "classificacao": decision.get("classificacao"),
                "id_canonico": decision.get("id_canonico"),
            })
        return decision

    def process_document(doc_id: str, text: str, resolver: Any, model: Any, tokenizer: Any) -> dict[str, Any]:
        active["document"] = doc_id
        captured["resolved"][doc_id] = []
        try:
            return originals["process_document"](doc_id, text, resolver, model, tokenizer)
        finally:
            active["document"] = None

    pipeline.extract_regex_citations = regex
    pipeline.extract_ner_citations = ner
    pipeline.fuse_extractions = fuse
    pipeline.classify_citation = classify
    pipeline.process_document = process_document
    try:
        pipeline.run_pipeline(input_dir, database, output_dir, "submission.csv")
    finally:
        for name, value in originals.items():
            setattr(pipeline, name, value)
    return captured


def _diagnostic_spans(rows: dict[str, list[dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    converted = {}
    for doc_id, items in rows.items():
        converted[doc_id] = []
        for row in items:
            citation_type = str(row.get("tipo", "unknown")).lower()
            if citation_type in {"lei", "legislacao", "legislação"}:
                citation_type = "legislacao"
            elif citation_type in {"jurisprudencia", "jurisprudência"}:
                citation_type = "jurisprudencia"
            converted[doc_id].append({
                "start": int(row["inicio"]), "end": int(row["fim"]), "type": citation_type,
            })
    return converted


def _official_train_ids(run_dir: Path) -> set[str]:
    manifest = json.loads((run_dir / "prepared" / "dataset_manifest.json").read_text(encoding="utf-8"))
    training_ids = set(manifest.get("official_documents_authorized_for_training", []))
    expected_training = {
        *(f"gen_n1_{number:03d}" for number in (1, 5, 6, 8, 13)),
        *(f"gen_n2_{number:03d}" for number in (1, 2, 6, 11, 13)),
    }
    if training_ids != expected_training:
        raise ValueError(f"Manifesto de treino oficial inesperado: {sorted(training_ids)}.")
    return training_ids


def _audit_official_overlap(run_dir: Path, heldout_texts: dict[str, str]) -> list[dict[str, Any]]:
    manifest = json.loads((run_dir / "prepared" / "dataset_manifest.json").read_text(encoding="utf-8"))
    source_rows = manifest["documents"]
    records = []
    test_norms = {
        test_id: " ".join(unicodedata.normalize("NFC", test_text).split())
        for test_id, test_text in heldout_texts.items()
    }
    test_shingles = {
        test_id: {norm[index:index + 5] for index in range(max(1, len(norm) - 4))}
        for test_id, norm in test_norms.items()
    }
    for row in source_rows:
        source = row.get("source")
        split = row.get("split")
        doc_id = row["document_id"]
        if source == "official":
            train_path = run_dir / "sources" / "official" / "txt" / f"{doc_id}.txt"
        else:
            train_path = run_dir / "prepared" / "local" / split / "txt" / f"{doc_id}.txt"
        if not train_path.is_file():
            raise FileNotFoundError(f"Texto do manifesto ausente para a auditoria de sobreposição: {doc_id}.")
        train_text = train_path.read_bytes().decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
        train_norm = " ".join(unicodedata.normalize("NFC", train_text).split())
        train_grams = {train_norm[index:index + 5] for index in range(max(1, len(train_norm) - 4))}
        for test_id, test_norm in test_norms.items():
            exact = train_norm == test_norm
            query = test_shingles[test_id]
            union_count = len(train_grams | query)
            similarity = len(train_grams & query) / union_count if union_count else 1.0
            if exact or similarity >= 0.90:
                records.append({
                    "comparison_id": doc_id,
                    "comparison_source": source,
                    "comparison_split": split,
                    "heldout_id": test_id,
                    "exact_normalized_match": exact,
                    "character_5gram_jaccard": similarity,
                })
    return records


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    official_gold_metadata = None
    official_text_metadata = None
    official_offset_validation = None
    if args.mode == "official":
        from official_gold import filter_official_gold, stage_official_text_lf, validate_official_offsets

        source_text_dir = Path("/input/official/txt")
        source_golden = Path("/input/official/goldenset_offsets.csv")
        database = Path("/input/official/desafio1_bracis.db")
        training_ids = _official_train_ids(run_dir)
        all_ids = {path.stem for path in source_text_dir.glob("gen_n[12]_*.txt")}
        if len(all_ids) != 26:
            raise ValueError(f"Esperados os 26 documentos oficiais; encontrados {len(all_ids)}.")
        expected_ids = all_ids - training_ids
        if len(expected_ids) != 16 or expected_ids & training_ids:
            raise ValueError("A separação dos 16 documentos oficiais reservados falhou.")
        official_dir = run_dir / "official_evaluation" / args.name
        input_dir = official_dir / "txt"
        input_dir.mkdir(parents=True, exist_ok=True)
        official_text_metadata = {}
        for doc_id in sorted(expected_ids):
            source_text = source_text_dir / f"{doc_id}.txt"
            staged_text = input_dir / f"{doc_id}.txt"
            official_text_metadata[doc_id] = stage_official_text_lf(source_text, staged_text)
        heldout_texts = _expected_texts(input_dir, expected_ids)
        overlaps = _audit_official_overlap(run_dir, heldout_texts)
        scoring_golden = official_dir / "goldenset_offsets.heldout.csv"
        official_gold_metadata = filter_official_gold(
            source_golden, scoring_golden, all_ids, expected_ids
        )
        official_offset_validation = validate_official_offsets(scoring_golden, input_dir, expected_ids)
    else:
        input_dir = args.input_dir.resolve()
        source_golden = args.golden.resolve()
        scoring_golden = source_golden
        database = Path("/input/reference.sqlite")
        expected_ids = {path.stem for path in input_dir.glob("*.txt")}
        heldout_texts = _expected_texts(input_dir, expected_ids)
        overlaps = []

    if not database.is_file():
        raise FileNotFoundError(f"Base canônica de avaliação ausente: {database}.")
    output_dir = run_dir / "evaluations" / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = _diagnostic_pipeline(input_dir, database, output_dir)
    submission_path = output_dir / "submission.csv"
    from scripts.evaluate import verify_frozen_run
    run_manifest_path = output_dir / "run_manifest.json"
    verify_frozen_run(submission_path, run_manifest_path)
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    submission = pd.read_csv(submission_path, dtype={"documento_id": str, "citacoes": str}, keep_default_na=False)
    submitted_ids = submission["documento_id"].astype(str).tolist()
    if len(submitted_ids) != len(set(submitted_ids)) or set(submitted_ids) != expected_ids:
        raise ValueError("A predição não contém exatamente uma linha por documento esperado.")
    solution = _solution_from_csv(scoring_golden, expected_ids)
    score = kaggle_metric.avaliar(solution, submission[["documento_id", "citacoes"]])
    context_bootstrap = (
        _context_bootstrap(solution, submission, expected_ids)
        if args.mode == "local" else None
    )
    stage_spans = {
        "regex": _diagnostic_spans(diagnostics["regex"]),
        "ner": _diagnostic_spans(diagnostics["ner"]),
        "fused": _diagnostic_spans(diagnostics["fused"]),
        "resolved": _diagnostic_spans(diagnostics["resolved"]),
    }
    report = {
        "schema_version": 1,
        "mode": args.mode,
        "run_id": run_dir.name,
        "evaluation_name": args.name,
        "expected_document_ids": sorted(expected_ids),
        "expected_documents": len(expected_ids),
        "documents_without_gold_citations": int((solution["citacoes"] == "-").sum()),
        "official_training_document_ids": sorted(training_ids) if args.mode == "official" else None,
        "official_heldout_overlap_audit": overlaps,
        "official_golden_sha256": _sha256(scoring_golden),
        "official_golden_source_sha256": _sha256(source_golden),
        "official_golden_filter_code_sha256": (
            _sha256(Path(__file__).resolve().with_name("official_gold.py"))
            if args.mode == "official" else None
        ),
        "official_golden_filter": official_gold_metadata,
        "official_offset_validation": official_offset_validation,
        "official_text_normalization": (
            "UTF-8-sig decode; CRLF and CR converted to LF; UTF-8 staged copy"
            if args.mode == "official" else None
        ),
        "official_text_source_sha256": (
            {doc_id: item["source_sha256"] for doc_id, item in official_text_metadata.items()}
            if official_text_metadata is not None else None
        ),
        "official_text_staging": official_text_metadata,
        "canonical_database_sha256": _sha256(database),
        "checkpoint_sha256": run_manifest.get("checkpoint_sha256"),
        "model_weights_sha256": _sha256(MODEL_DIR / "model.safetensors"),
        "evaluation_code_sha256": _sha256(Path(__file__).resolve()),
        "metric_code_sha256": _sha256(Path(kaggle_metric.__file__).resolve()),
        "text_sha256": {
            doc_id: _sha256(input_dir / f"{doc_id}.txt") for doc_id in sorted(expected_ids)
        },
        "submission_sha256": _sha256(submission_path),
        "challenge_metrics": score,
        "context_bootstrap": context_bootstrap,
        "stage_span_metrics": _span_metrics(scoring_golden, stage_spans, expected_ids),
        "stage_predictions": {stage: stage_rows for stage, stage_rows in diagnostics.items()},
        "run_manifest_sha256": _sha256(run_manifest_path),
    }
    report_path = Path(args.report) if args.report else run_dir / "metrics" / f"{args.name}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    spans_path = report_path.with_name(f"{report_path.stem}_spans.json")
    spans_path.write_text(
        json.dumps(report["stage_span_metrics"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("local", "official"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--golden", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.mode == "local" and (args.input_dir is None or args.golden is None):
        parser.error("A avaliação local exige --input-dir e --golden.")
    report = evaluate(args)
    levels = report["challenge_metrics"].get("niveis", {})
    print(json.dumps({
        "mode": report["mode"],
        "run_id": report["run_id"],
        "evaluation_name": report["evaluation_name"],
        "documents": report["expected_documents"],
        "documents_without_gold_citations": report["documents_without_gold_citations"],
        "score_final": report["challenge_metrics"]["score_final"],
        "score_n1": levels.get("1", levels.get(1, {})).get("score"),
        "score_n2": levels.get("2", levels.get(2, {})).get("score"),
        "report": str(Path(args.report) if args.report else args.run_dir / "metrics" / f"{args.name}.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
