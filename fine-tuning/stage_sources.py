"""Stage only the ten explicitly authorized official training examples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from train.config import OFFICIAL_TRAIN_DOCUMENTS, SEED
from train.provenance import sha256_file, write_json


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normalize_annotation_surface(value: str) -> str:
    return value.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")


def stage(run_id: str, raw_text_dir: Path, raw_annotations: Path, runs_root: Path) -> dict:
    if not run_id or run_id in {".", ".."} or any(char in run_id for char in "/\\"):
        raise ValueError("run-id precisa ser um nome simples de diretório.")
    selected = set(OFFICIAL_TRAIN_DOCUMENTS)
    rows: list[dict[str, str]] = []
    with raw_annotations.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or [])
        required = {
            "nivel", "documento_id", "citacao_id", "inicio", "fim", "trecho",
            "tipo", "classificacao", "id_canonico",
        }
        if not required.issubset(fields):
            raise ValueError(f"Colunas ausentes no CSV oficial: {sorted(required-set(fields))}.")
        rows = [row for row in reader if row["documento_id"] in selected]
    if {row["documento_id"] for row in rows} != selected:
        raise ValueError("CSV oficial não contém anotações para todos os dez documentos autorizados.")

    run_dir = runs_root / run_id
    target_root = run_dir / "sources" / "official"
    target_text = target_root / "txt"
    target_text.mkdir(parents=True, exist_ok=True)
    text_hashes: dict[str, str] = {}
    for document_id in sorted(selected):
        source = raw_text_dir / f"{document_id}.txt"
        if not source.is_file():
            raise FileNotFoundError(f"Texto oficial autorizado ausente: {source}.")
        raw = source.read_bytes()
        text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
        text_hashes[document_id] = _sha256_bytes(raw)
        staged = target_text / source.name
        if staged.exists() and staged.read_bytes() != raw:
            raise ValueError(f"O arquivo staged já existe com conteúdo diferente: {staged}.")
        if not staged.exists():
            staged.write_bytes(raw)

    unexpected = sorted(path.name for path in target_text.glob("*.txt") if path.stem not in selected)
    if unexpected:
        raise ValueError(f"A pasta staged contém documentos oficiais não autorizados: {unexpected}.")

    escaped_newline_surfaces = 0
    for row in rows:
        source = raw_text_dir / f"{row['documento_id']}.txt"
        text = source.read_bytes().decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
        start, end = int(row["inicio"]), int(row["fim"])
        normalized_surface = _normalize_annotation_surface(row["trecho"])
        escaped_newline_surfaces += normalized_surface != row["trecho"]
        if start < 0 or end <= start or end > len(text) or text[start:end] != normalized_surface:
            raise ValueError(f"Offset oficial não corresponde ao texto LF: {row['documento_id']}/{row['citacao_id']}.")

    filtered = target_root / "goldenset_offsets.csv"
    temporary = filtered.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    filtered_bytes = temporary.read_bytes()
    if filtered.exists() and filtered.read_bytes() != filtered_bytes:
        temporary.unlink()
        raise ValueError(f"O CSV staged já existe com conteúdo diferente: {filtered}.")
    if filtered.exists():
        temporary.unlink()
    else:
        os.replace(temporary, filtered)

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "seed": SEED,
        "selection_rule": "primeiros cinco documentos por nível ordenados por SHA-256 de 42:<documento_id>",
        "selected_documents": sorted(selected),
        "selected_document_count": len(selected),
        "selected_annotation_rows": len(rows),
        "annotation_surfaces_with_escaped_newlines": escaped_newline_surfaces,
        "selected_text_raw_sha256": text_hashes,
        "source_annotation_file_sha256": sha256_file(raw_annotations),
        "filtered_annotation_file_sha256": _sha256_bytes(filtered_bytes),
        "line_endings_for_offset_validation": "LF-normalized Unicode code points",
        "staged_only_selected_documents": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = target_root / "sources_manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable = {key: value for key, value in manifest.items() if key != "created_at_utc"}
        previous_comparable = {key: value for key, value in previous.items() if key != "created_at_utc"}
        if previous_comparable != comparable:
            raise ValueError("O manifesto staged mudou para a mesma execução; inicie um novo run-id.")
    else:
        write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--raw-text-dir", type=Path, default=Path("/source/raw/official/txt"))
    parser.add_argument("--raw-annotations", type=Path, default=Path("/source/raw/official/goldenset_offsets.csv"))
    parser.add_argument("--runs-root", type=Path, default=Path("/runs"))
    args = parser.parse_args()
    print(json.dumps(stage(args.run_id, args.raw_text_dir, args.raw_annotations, args.runs_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
