# -*- coding: utf-8 -*-
"""Camada de fusão inteligente entre extração Regex e NER Transformer."""
from typing import Any
from ..config import IOU_MIN, NER_THRESHOLD
from .distractor_filter import is_distractor_span, find_header_boundary


def compute_iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    """Calcula Intersecção sobre União (IoU) entre dois spans."""
    inter = max(0, min(a["fim"], b["fim"]) - max(a["inicio"], b["inicio"]))
    if inter == 0:
        return 0.0
    uniao = (a["fim"] - a["inicio"]) + (b["fim"] - b["inicio"]) - inter
    return inter / uniao if uniao > 0 else 0.0


def fuse_extractions(
    text: str,
    regex_cits: list[dict[str, Any]],
    ner_cits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Combina spans do Regex e do NER com matching 1-para-1 guloso por maior IoU.
    """
    header_end = find_header_boundary(text)
    
    # Calcular matriz de candidatos IoU >= 0.5
    pairs = []
    for ri, r in enumerate(regex_cits):
        for ni, n in enumerate(ner_cits):
            if r.get("tipo") != n.get("tipo"):
                continue
            v = compute_iou(r, n)
            if v >= IOU_MIN:
                pairs.append((-v, ri, ni))
                
    pairs.sort()
    used_r, used_n = set(), set()
    fused: list[dict[str, Any]] = []

    # 1. Spans em que Regex e NER concordam
    for _, ri, ni in pairs:
        if ri in used_r or ni in used_n:
            continue
        used_r.add(ri)
        used_n.add(ni)
        r = regex_cits[ri]
        n = ner_cits[ni]
        
        # O span do regex é preferido para bordas precisas
        combined = dict(r)
        # Uma entidade de relator/data do NER não pode apagar um identificador
        # explícito já extraído pelo Regex.
        combined["is_vague"] = bool(r.get("is_vague"))
        fused.append(combined)

    # 2. Spans exclusivamente Regex
    for ri, r in enumerate(regex_cits):
        if ri not in used_r:
            item = dict(r)
            fused.append(item)

    # 3. Spans exclusivamente NER (Safety Net & Autonomous Discovery)
    for ni, n in enumerate(ner_cits):
        if ni not in used_n:
            prob = n.get("prob", 0.0)
            overlaps_regex = any(
                max(r["inicio"], n["inicio"]) < min(r["fim"], n["fim"])
                for r in regex_cits
            )
            if overlaps_regex:
                continue
            is_v = n.get("is_vague", False)
            if prob >= NER_THRESHOLD and not is_distractor_span(text, n["inicio"], n["fim"], header_end, is_vague=is_v):
                item = dict(n)
                item["is_vague"] = is_v
                fused.append(item)

    # Ordenar por início no documento
    fused.sort(key=lambda x: x["inicio"])
    return fused
