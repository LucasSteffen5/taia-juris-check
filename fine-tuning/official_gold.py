"""Stage official texts and gold for the authorized held-out document set."""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_official_text_lf(source_text: Path, staged_text: Path) -> dict[str, Any]:
    """Stage UTF-8 LF text to match the official character offsets.

    An earlier evaluator copied CRLF bytes verbatim. Replacing that exact old
    copy is allowed during resume; any other pre-existing content is rejected.
    """
    raw = source_text.read_bytes()
    normalized = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    existing = staged_text.read_bytes() if staged_text.exists() else None
    if existing is not None and existing not in (raw, normalized):
        raise ValueError(f"Texto oficial preparado diverge da fonte: {source_text.stem}.")
    if existing != normalized:
        staged_text.parent.mkdir(parents=True, exist_ok=True)
        temporary = staged_text.with_name(staged_text.name + ".tmp")
        try:
            temporary.write_bytes(normalized)
            os.replace(temporary, staged_text)
        finally:
            if temporary.exists():
                temporary.unlink()
    return {
        "source_sha256": _sha256(source_text),
        "staged_sha256": _sha256(staged_text),
        "source_crlf_count": raw.count(b"\r\n"),
        "source_has_bom": raw.startswith(b"\xef\xbb\xbf"),
        "changed_from_source": normalized != raw,
    }


def filter_official_gold(
    source_csv: Path,
    output_csv: Path,
    all_official_ids: set[str],
    expected_ids: set[str],
) -> dict[str, Any]:
    """Write a stable CSV containing only annotations for ``expected_ids``.

    The source is read-only and remains untouched. IDs absent from the CSV are
    allowed because official documents with no citations have no annotation rows;
    callers must still pass all expected IDs to the metric evaluator.
    """
    if not expected_ids or not expected_ids <= all_official_ids:
        raise ValueError("O subconjunto esperado precisa ser não vazio e pertencer aos IDs oficiais.")

    with source_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        if not fieldnames or "documento_id" not in fieldnames:
            raise ValueError("O gabarito oficial não contém a coluna documento_id.")
        rows = []
        observed_ids: set[str] = set()
        source_row_count = 0
        for row in reader:
            source_row_count += 1
            document_id = str(row.get("documento_id") or "").strip()
            if not document_id:
                raise ValueError(f"Linha {source_row_count + 1} do gabarito sem documento_id.")
            if document_id not in all_official_ids:
                raise ValueError(f"O gabarito contém ID ausente do conjunto oficial: {document_id}.")
            if document_id in expected_ids:
                rows.append(row)
                observed_ids.add(document_id)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_name(output_csv.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, output_csv)
    finally:
        if temporary.exists():
            temporary.unlink()

    return {
        "source_sha256": _sha256(source_csv),
        "filtered_sha256": _sha256(output_csv),
        "source_row_count": source_row_count,
        "included_row_count": len(rows),
        "included_annotation_document_ids": sorted(observed_ids),
        "expected_document_ids": sorted(expected_ids),
        "expected_documents_without_annotation_rows": sorted(expected_ids - observed_ids),
    }


def validate_official_offsets(
    filtered_csv: Path, staged_text_dir: Path, expected_ids: set[str]
) -> dict[str, Any]:
    """Reject an official score when any prepared text disagrees with its gold offsets."""
    texts: dict[str, str] = {}
    for document_id in expected_ids:
        text = (staged_text_dir / f"{document_id}.txt").read_bytes().decode("utf-8")
        if "\r" in text:
            raise ValueError(f"Texto oficial preparado ainda contém CR: {document_id}.")
        texts[document_id] = text

    counts = {document_id: 0 for document_id in expected_ids}
    with filtered_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"documento_id", "inicio", "fim", "trecho"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("O gabarito filtrado não contém as colunas de offsets e trecho.")
        for line_number, row in enumerate(reader, start=2):
            document_id = str(row["documento_id"])
            if document_id not in expected_ids:
                raise ValueError(f"Anotação fora do subconjunto oficial: {document_id}.")
            start, end = int(row["inicio"]), int(row["fim"])
            surface = str(row["trecho"]).replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
            text = texts[document_id]
            if not (0 <= start < end <= len(text)) or text[start:end] != surface:
                raise ValueError(
                    f"Offset oficial divergente em {document_id}, linha {line_number}: "
                    f"[{start}:{end}] não corresponde ao trecho anotado."
                )
            counts[document_id] += 1
    return {
        "validated_annotation_rows": sum(counts.values()),
        "expected_document_count": len(expected_ids),
        "annotation_rows_per_document": dict(sorted(counts.items())),
        "documents_without_annotations": sorted(doc_id for doc_id, count in counts.items() if count == 0),
    }
