# -*- coding: utf-8 -*-
"""Execução reproduzível de extração, resolução e entrega."""
import argparse
import hashlib
import json
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from src.classification.classifier import classify_citation
from src.classification.confidence import CONFIDENCE_POLICY_VERSION, calculate_confidence
from src.config import NER_MODEL_DIR, ROOT_DIR, assert_output_outside_official_data
from src.extraction.fusion import fuse_extractions
from src.extraction.ner_inference import extract_ner_citations
from src.extraction.ner_model import load_ner_model_and_tokenizer
from src.extraction.regex_extractor import extract_regex_citations
from src.output.json_builder import build_document_json, save_document_json, validate_document_json
from src.output.submission_builder import generate_submission_csv
from src.resolution.canonical_db import get_connection
from src.resolution.header_index import HeaderIndex, PARSER_VERSION
from src.resolution.resolver import UnifiedResolver
from src.utils_data import (
    LineEndingOffsetMap,
    load_all_documents,
    map_span_to_original,
    normalize_line_endings,
)


def process_document(
    doc_id: str,
    text: str,
    resolver: UnifiedResolver,
    ner_model: Any,
    tokenizer: Any,
) -> dict[str, Any]:
    """Normaliza para LF e processa usando offsets Unicode da representação LF."""
    text, _, _ = normalize_line_endings(text)
    regex_citations = extract_regex_citations(text)
    ner_citations = extract_ner_citations(text, ner_model, tokenizer)
    citations = fuse_extractions(text, regex_citations, ner_citations)
    final = []
    for citation in citations:
        resolution = resolver.resolve(citation)
        decision = classify_citation(citation, resolution)
        result = dict(citation)
        result["classificacao"] = decision["classificacao"]
        result["id_canonico"] = decision["id_canonico"]
        result["confianca"] = calculate_confidence(
            citation, resolution, decision["classificacao"]
        )
        final.append(result)

    payload = build_document_json(doc_id, final)
    validate_document_json(payload, text)
    return payload


def _documents_fingerprint(documents: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for doc_id, text in sorted(documents.items()):
        digest.update(doc_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _remap_document_to_original(
    document: dict[str, Any], original_text: str, boundaries: LineEndingOffsetMap
) -> dict[str, Any]:
    """Cria o mesmo documento com offsets que apontam para o arquivo original."""
    remapped = {"documento_id": document["documento_id"], "citacoes": []}
    for citation in document["citacoes"]:
        start, end = map_span_to_original(citation["inicio"], citation["fim"], boundaries)
        item = dict(citation)
        item["inicio"], item["fim"] = start, end
        item["trecho"] = original_text[start:end]
        remapped["citacoes"].append(item)
    validate_document_json(remapped, original_text)
    return remapped


def _versions() -> dict[str, str | None]:
    versions = {}
    for package in ("torch", "transformers", "numpy", "pandas"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def run_pipeline(
    txt_dir: Path | str,
    db_path: Path | str,
    output_dir: Path | str,
    output_filename: str = "submission.csv",
) -> Path:
    """Gera JSONs, submissão CSV e manifesto em uma única execução."""
    started = time.perf_counter()
    txt_path, database_path, output_path = Path(txt_dir), Path(db_path), Path(output_dir)
    if (
        not output_filename
        or output_filename in {".", ".."}
        or "/" in output_filename
        or "\\" in output_filename
    ):
        raise ValueError("--output-file precisa ser apenas um nome de arquivo.")
    assert_output_outside_official_data(output_path)
    if not txt_path.is_dir() or not database_path.is_file():
        raise FileNotFoundError("Diretório TXT ou banco SQLite não encontrado.")

    documents = load_all_documents(txt_path)
    if not documents:
        raise ValueError(f"Nenhum arquivo .txt encontrado em {txt_path}.")
    output_path.mkdir(parents=True, exist_ok=True)
    json_dir = output_path / "json"
    json_dir.mkdir(exist_ok=True)
    original_json_dir = output_path / "json_original"
    original_json_dir.mkdir(exist_ok=True)
    normalized_text_dir = output_path / "text_lf"
    normalized_text_dir.mkdir(exist_ok=True)
    for old_json in json_dir.glob("*.json"):
        if old_json.stem not in documents:
            old_json.unlink()
    for old_json in original_json_dir.glob("*.json"):
        if old_json.stem not in documents:
            old_json.unlink()
    for old_text in normalized_text_dir.glob("*.txt"):
        if old_text.stem not in documents:
            old_text.unlink()

    normalized_documents: dict[str, str] = {}
    line_endings: dict[str, dict[str, int]] = {}
    original_boundaries: dict[str, LineEndingOffsetMap] = {}
    for doc_id, original_text in documents.items():
        normalized_text, boundaries, counts = normalize_line_endings(original_text)
        normalized_documents[doc_id] = normalized_text
        original_boundaries[doc_id] = boundaries
        line_endings[doc_id] = counts
        (normalized_text_dir / f"{doc_id}.txt").write_text(
            normalized_text, encoding="utf-8", newline="\n"
        )

    print(f"[Pipeline] Lendo SQLite somente para leitura: {database_path}")
    database_sha = HeaderIndex.fingerprint(database_path)
    frozen = {
        "source_sha256": HeaderIndex.fingerprint(ROOT_DIR / "src"),
        "scripts_sha256": HeaderIndex.fingerprint(ROOT_DIR / "scripts"),
        "dependencies_sha256": HeaderIndex.fingerprint(ROOT_DIR / "requirements.txt"),
        "checkpoint_sha256": HeaderIndex.fingerprint(NER_MODEL_DIR),
        "metric_sha256": HeaderIndex.fingerprint(
            ROOT_DIR / "data" / "desafio-jusbrasil-bracis-2026" / "kaggle_metric.py"
        ),
    }
    index_path = output_path / ".cache" / "header_index.json"
    connection = get_connection(database_path)
    try:
        header_index = HeaderIndex()
        try:
            header_index.load(index_path, expected_database_sha256=database_sha)
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
            print("[Pipeline] Preparando índice de cabeçalhos...")
            header_index.build_from_db(connection, database_sha)
            header_index.save(index_path)

        resolver = UnifiedResolver(connection, header_index)
        print("[Pipeline] Carregando checkpoint NER local...")
        model, tokenizer = load_ner_model_and_tokenizer()
        device = str(next(model.parameters()).device)

        print(f"[Pipeline] Dispositivo: {device}; documentos: {len(documents)}")
        payloads = {}
        for doc_id, original_text in documents.items():
            text = normalized_documents[doc_id]
            payload = process_document(doc_id, text, resolver, model, tokenizer)
            payloads[doc_id] = payload
            save_document_json(payload, json_dir / f"{doc_id}.json")
            original_payload = _remap_document_to_original(
                payload, original_text, original_boundaries[doc_id]
            )
            save_document_json(original_payload, original_json_dir / f"{doc_id}.json")
    finally:
        connection.close()

    submission_path = output_path / output_filename
    generate_submission_csv(payloads, submission_path)
    current_source = HeaderIndex.fingerprint(ROOT_DIR / "src")
    current_scripts = HeaderIndex.fingerprint(ROOT_DIR / "scripts")
    current_checkpoint = HeaderIndex.fingerprint(NER_MODEL_DIR)
    if (current_source, current_scripts, current_checkpoint) != (
        frozen["source_sha256"], frozen["scripts_sha256"], frozen["checkpoint_sha256"]
    ):
        raise RuntimeError("Código ou checkpoint mudou durante a execução; submissão não foi congelada.")
    manifest = {
        "database_sha256": database_sha,
        **frozen,
        "offset_policy": "LF",
        "offset_policy_version": 1,
        "submission_offset_space": "LF",
        "input_documents_original_sha256": _documents_fingerprint(documents),
        "input_documents_lf_sha256": _documents_fingerprint(normalized_documents),
        "input_line_endings": line_endings,
        "normalized_texts_sha256": HeaderIndex.fingerprint(normalized_text_dir),
        "lf_json_sha256": HeaderIndex.fingerprint(json_dir),
        "original_json_sha256": HeaderIndex.fingerprint(original_json_dir),
        "input_documents_sha256": _documents_fingerprint(documents),
        "submission_sha256": HeaderIndex.fingerprint(submission_path),
        "parser_version": PARSER_VERSION,
        "device": device,
        "confidence_policy_version": CONFIDENCE_POLICY_VERSION,
        "confidence_policy": "patamares heurísticos por evidência; não calibrada",
        "documents_processed": len(documents),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "python_version": __import__("platform").python_version(),
        "package_versions": _versions(),
    }
    (output_path / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[Pipeline] Submissão salva: {submission_path}")
    return submission_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline Jusbrasil x BRACIS")
    parser.add_argument("--input", required=True, help="Diretório com os documentos .txt")
    parser.add_argument("--db", required=True, help="Banco SQLite de consulta")
    parser.add_argument("--output", required=True, help="Diretório para as saídas")
    parser.add_argument(
        "--output-file",
        default="submission.csv",
        help="Nome do CSV de submissão dentro do diretório --output",
    )
    args = parser.parse_args()
    run_pipeline(args.input, args.db, args.output, args.output_file)


if __name__ == "__main__":
    main()
