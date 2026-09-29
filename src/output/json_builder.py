# -*- coding: utf-8 -*-
"""Construtor de JSON por documento conforme contrato Schema 1.2."""
import json
import math
from pathlib import Path
from typing import Any
from src.config import assert_output_outside_official_data

VALID_CLASSES = {"real", "inventada", "incompleta"}
VALID_TYPES = {"lei", "jurisprudencia"}


def build_document_json(doc_id: str, citations: list[dict[str, Any]]) -> dict[str, Any]:
    """Monta a estrutura JSON conforme contrato oficial da competição."""
    cits_payload = []
    
    for c in citations:
        classe = c["classificacao"]
        resolucao = None
        if classe == "real" and c.get("id_canonico"):
            resolucao = {"id_canonico": int(c["id_canonico"])}
            
        cits_payload.append({
            "inicio": int(c["inicio"]),
            "fim": int(c["fim"]),
            "trecho": c["trecho"],
            "tipo": c["tipo"],
            "classificacao": classe,
            "resolucao": resolucao,
            "confianca": float(c["confianca"]) if c.get("confianca") is not None else None,
        })
        
    return {
        "documento_id": doc_id,
        "citacoes": cits_payload,
    }


def validate_document_json(document: dict[str, Any], source_text: str) -> None:
    """Validate contract fields and exact codepoint offsets before emitting output."""
    if not isinstance(document.get("documento_id"), str) or not document["documento_id"]:
        raise ValueError("documento_id ausente ou inválido.")
    citations = document.get("citacoes")
    if not isinstance(citations, list):
        raise ValueError("citacoes precisa ser uma lista.")
    for index, citation in enumerate(citations):
        start, end = citation.get("inicio"), citation.get("fim")
        if type(start) is not int or type(end) is not int or start < 0 or end <= start or end > len(source_text):
            raise ValueError(f"Offsets inválidos na citação {index}.")
        if citation.get("trecho") != source_text[start:end]:
            raise ValueError(f"Trecho diverge dos offsets na citação {index}.")
        if citation.get("classificacao") not in VALID_CLASSES or citation.get("tipo") not in VALID_TYPES:
            raise ValueError(f"Classe ou tipo inválido na citação {index}.")
        confidence = citation.get("confianca")
        if confidence is not None and (
            isinstance(confidence, bool) or not isinstance(confidence, (float, int))
            or not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0
        ):
            raise ValueError(f"Confiança fora do intervalo na citação {index}.")
        resolution = citation.get("resolucao")
        if citation["classificacao"] == "real":
            if not isinstance(resolution, dict) or type(resolution.get("id_canonico")) is not int or resolution["id_canonico"] <= 0:
                raise ValueError(f"Citação real sem id_canonico válido na posição {index}.")
        elif resolution is not None:
            raise ValueError(f"Citação não real com resolução canônica na posição {index}.")


def save_document_json(doc_dict: dict[str, Any], output_path: Path | str) -> None:
    """Salva o JSON codificado em UTF-8 com indentação padrão."""
    p = Path(output_path)
    assert_output_outside_official_data(p)
    p.parent.mkdir(exist_ok=True, parents=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(doc_dict, f, ensure_ascii=False, indent=2)
