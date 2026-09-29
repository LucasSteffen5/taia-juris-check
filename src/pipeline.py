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
from src.config import DEFAULT_CONFIDENCE, NER_MODEL_DIR, ROOT_DIR, assert_output_outside_official_data
from src.extraction.fusion import fuse_extractions
from src.extraction.ner_inference import extract_ner_citations
from src.extraction.ner_model import load_ner_model_and_tokenizer
from src.extraction.regex_extractor import extract_regex_citations
from src.output.json_builder import build_document_json, save_document_json, validate_document_json
from src.output.submission_builder import generate_submission_csv
from src.resolution.canonical_db import get_connection
from src.resolution.header_index import HeaderIndex, PARSER_VERSION
from src.resolution.resolver import UnifiedResolver
from src.utils_data import load_all_documents


def process_document(
    doc_id: str,
    text: str,
    resolver: UnifiedResolver,
    ner_model: Any,
    tokenizer: Any,
) -> dict[str, Any]:
    """Processa um documento sem alterar o texto usado nos offsets."""
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
        result["confianca"] = DEFAULT_CONFIDENCE
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
) -> Path:
    """Gera JSONs, submissão CSV e manifesto em uma única execução."""
    started = time.perf_counter()
    txt_path, database_path, output_path = Path(txt_dir), Path(db_path), Path(output_dir)
    assert_output_outside_official_data(output_path)
    if not txt_path.is_dir() or not database_path.is_file():
        raise FileNotFoundError("Diretório TXT ou banco SQLite não encontrado.")

    documents = load_all_documents(txt_path)
    if not documents:
        raise ValueError(f"Nenhum arquivo .txt encontrado em {txt_path}.")
    output_path.mkdir(parents=True, exist_ok=True)
    json_dir = output_path / "json"
    json_dir.mkdir(exist_ok=True)
    for old_json in json_dir.glob("*.json"):
        if old_json.stem not in documents:
            old_json.unlink()

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
        for doc_id, text in documents.items():
            payload = process_document(doc_id, text, resolver, model, tokenizer)
            payloads[doc_id] = payload
            save_document_json(payload, json_dir / f"{doc_id}.json")
    finally:
        connection.close()

    submission_path = output_path / "submission.csv"
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
        "input_documents_sha256": _documents_fingerprint(documents),
        "submission_sha256": HeaderIndex.fingerprint(submission_path),
        "parser_version": PARSER_VERSION,
        "device": device,
        "confidence_policy": "constante neutra 0.5; não calibrada",
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
    args = parser.parse_args()
    run_pipeline(args.input, args.db, args.output)


if __name__ == "__main__":
    main()
