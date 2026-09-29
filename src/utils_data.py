# -*- coding: utf-8 -*-
"""Utilitários de carregamento, parsing de documentos e gabaritos."""
import csv
from pathlib import Path
from typing import Any


def load_document(txt_path: Path | str) -> str:
    """Lê documento em codepoints Unicode UTF-8 exatos."""
    with Path(txt_path).open("r", encoding="utf-8", newline="") as stream:
        return stream.read()


def load_all_documents(txt_dir: Path | str) -> dict[str, str]:
    """Carrega todos os arquivos .txt do diretório mapeados por documento_id."""
    docs = {}
    p = Path(txt_dir)
    for f in sorted(p.glob("*.txt")):
        docs[f.stem] = load_document(f)
    return docs


def load_golden_set(csv_path: Path | str) -> dict[str, list[dict[str, Any]]]:
    """
    Carrega gabarito (goldenset_offsets.csv) indexado por documento_id.
    Colunas: nivel, documento_id, citacao_id, inicio, fim, trecho, tipo, classificacao, id_canonico
    """
    golds: dict[str, list[dict[str, Any]]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = row["documento_id"].strip()
            item = {
                "nivel": int(row.get("nivel", 1)),
                "citacao_id": row.get("citacao_id", ""),
                "inicio": int(row["inicio"]),
                "fim": int(row["fim"]),
                "trecho": row.get("trecho", ""),
                "tipo": row.get("tipo", "").strip().lower(),
                "classificacao": row.get("classificacao", "").strip().lower(),
                "id_canonico": str(row.get("id_canonico", "")).strip() or None,
            }
            golds.setdefault(doc_id, []).append(item)
    return golds


def validate_offsets(text: str, inicio: int, fim: int, expected_trecho: str | None = None) -> bool:
    """Valida se o slice [inicio:fim] em codepoints corresponde ao trecho literal esperado."""
    extracted = text[inicio:fim]
    if expected_trecho is not None:
        # Tratar quebras de linha escapadas no CSV
        clean_exp = expected_trecho.replace("\\n", "\n")
        return extracted == clean_exp
    return len(extracted) > 0
