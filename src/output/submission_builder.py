# -*- coding: utf-8 -*-
"""Conversor de predições JSON para o formato tabular submission.csv."""
import csv
import math
from pathlib import Path
from typing import Any
from src.config import assert_output_outside_official_data

VALID_CLASSES = {"real", "inventada", "incompleta"}


def _validate_citation(citation: dict[str, Any]) -> None:
    ini, fim = citation.get("inicio"), citation.get("fim")
    if type(ini) is not int or type(fim) is not int or ini < 0 or fim <= ini:
        raise ValueError("Span da submissão inválido.")
    if citation.get("classificacao") not in VALID_CLASSES:
        raise ValueError("Classe inválida na submissão.")
    if citation.get("tipo") not in ("lei", "jurisprudencia"):
        raise ValueError("Tipo de citação inválido na submissão.")
    resolution = citation.get("resolucao")
    if citation["classificacao"] == "real":
        if not isinstance(resolution, dict) or type(resolution.get("id_canonico")) is not int or resolution["id_canonico"] <= 0:
            raise ValueError("Citação real sem id_canonico positivo.")
    elif resolution is not None:
        raise ValueError("Citação não real não pode conter id_canonico.")
    confidence = citation.get("confianca")
    if confidence is not None and (
        isinstance(confidence, bool) or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0
    ):
        raise ValueError("Confiança precisa estar entre 0 e 1.")


def encode_citation_entry(c: dict[str, Any]) -> str:
    """Codifica citação no padrão oficial: inicio,fim,classe,id_canonico,confianca."""
    _validate_citation(c)
    ini = c["inicio"]
    fim = c["fim"]
    classe = c["classificacao"]
    resolucao = c.get("resolucao")
    id_can = str(resolucao["id_canonico"]) if (resolucao and resolucao.get("id_canonico")) else "-"
    conf = f"{float(c['confianca']):.4f}" if c.get("confianca") is not None else "-"
    return f"{ini},{fim},{classe},{id_can},{conf}"


def generate_submission_csv(
    documents: dict[str, dict[str, Any]],
    output_csv_path: Path | str,
) -> None:
    """
    Gera o CSV oficial com colunas 'documento_id' e 'citacoes'.
    Separador de citações: '|'. Documentos sem citações recebem '-'.
    """
    p = Path(output_csv_path)
    assert_output_outside_official_data(p)
    p.parent.mkdir(exist_ok=True, parents=True)
    
    with open(p, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["documento_id", "citacoes"])
        
        for doc_id in sorted(documents.keys()):
            doc = documents[doc_id]
            if doc.get("documento_id") != doc_id:
                raise ValueError(f"Chave e documento_id divergem para {doc_id}.")
            cits = doc.get("citacoes", [])
            if not isinstance(cits, list):
                raise ValueError(f"citacoes precisa ser lista no documento {doc_id}.")
            if not cits:
                writer.writerow([doc_id, "-"])
            else:
                encoded_cits = [encode_citation_entry(c) for c in cits]
                writer.writerow([doc_id, "|".join(encoded_cits)])
