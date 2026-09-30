# -*- coding: utf-8 -*-
"""Estimativa heurística e auditável da confiança da citação resolvida."""
from typing import Any

from src.resolution.result import ResolutionResult

CONFIDENCE_POLICY_VERSION = "evidence-rules-v2"

_STRENGTH_RANK = {"fraca": 0, "forte": 1, "muito_forte": 2}


def calculate_confidence(
    citation: dict[str, Any],
    result: ResolutionResult,
    classification: str,
) -> float:
    """Calcula confiança heurística sem interpretar a saída do NER como calibração.

    Os patamares representam evidência estrutural da decisão, não probabilidades
    calibradas. Extração exclusiva pelo NER recebe uma redução conservadora.
    """
    strength = result.identifier_strength
    rank = _STRENGTH_RANK.get(strength, 0)

    if classification == "real" and result.n_candidates == 1:
        confidence = (0.65, 0.75, 0.80)[rank]
        sources = set(citation.get("_confidence_sources", ()))
        iou = citation.get("_fusion_iou")
        agreement = (
            sources == {"regex", "ner"}
            and isinstance(iou, (int, float))
            and iou >= 0.90
        )
        if agreement:
            confidence += 0.05
    elif classification == "inventada":
        confidence = (0.55, 0.65, 0.75)[rank]
    elif classification == "incompleta":
        if result.n_candidates > 1 and rank >= _STRENGTH_RANK["forte"]:
            confidence = 0.70
        elif result.n_candidates > 1 or (
            citation.get("is_vague")
            and "regex" in set(citation.get("_confidence_sources", ()))
        ):
            confidence = 0.65
        else:
            confidence = 0.55
    else:
        confidence = 0.55

    sources = set(citation.get("_confidence_sources", ()))
    if sources == {"ner"}:
        confidence -= 0.05

    return round(min(0.85, max(0.50, confidence)), 2)
