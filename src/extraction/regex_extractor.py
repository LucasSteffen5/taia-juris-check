# -*- coding: utf-8 -*-
"""Motor de extração determinístico por expressões regulares."""
import re
from typing import Any
from .patterns import RE_JURIS_CLASSICA, RE_JURIS_CNJ, RE_SUMULA, RE_LEI, RE_VAGA, RE_VAGA_GENERIC
from .distractor_filter import find_header_boundary, is_distractor_span
from ..normalization.matching_text import normalized_view, map_normalized_span


def extract_regex_citations(text: str) -> list[dict[str, Any]]:
    """
    Varre o documento e extrai todas as citações candidatas com offsets precisos em codepoints.
    """
    header_end = find_header_boundary(text)
    match_text, starts, ends = normalized_view(text)
    raw_candidates = []

    def add_matches(pattern, pattern_name, citation_type, is_vague):
        for match in pattern.finditer(match_text):
            begin, finish = map_normalized_span(match.start(), match.end(), starts, ends)
            raw_candidates.append({
                "inicio": begin,
                "fim": finish,
                "trecho": text[begin:finish],
                "tipo": citation_type,
                "is_vague": is_vague,
                "pattern": pattern_name,
            })

    # 1. Padrão Vago
    add_matches(RE_VAGA, "vaga", "jurisprudencia", True)
    add_matches(RE_VAGA_GENERIC, "vaga_generica", "jurisprudencia", True)

    # 2. Padrão Súmula
    add_matches(RE_SUMULA, "sumula", "jurisprudencia", False)

    # 3. Padrão Lei
    add_matches(RE_LEI, "lei", "lei", False)

    # 4. Padrão CNJ
    add_matches(RE_JURIS_CNJ, "cnj", "jurisprudencia", False)

    # 5. Padrão Clássico
    add_matches(RE_JURIS_CLASSICA, "classica", "jurisprudencia", False)

    for c in raw_candidates:
        if c["trecho"].startswith("Rp nº") and text[max(0, c["inicio"]-2):c["inicio"]] == "R-":
            c["inicio"] -= 2
            c["trecho"] = text[c["inicio"]:c["fim"]]

    for m in re.finditer(r'(?:processo\s+n[ºo°\.]\s+)?TST-[^\n,;]{10,50}', text, re.I):
        if not any(max(c["inicio"], m.start()) < min(c["fim"], m.end()) for c in raw_candidates):
            raw_candidates.append({
                "inicio": m.start(),
                "fim": m.end(),
                "trecho": text[m.start():m.end()],
                "pattern": "cnj",
                "tipo": "jurisprudencia",
                "is_vague": False,
            })

    # Filtrar distratores
    filtered = []
    for c in raw_candidates:
        if not is_distractor_span(text, c["inicio"], c["fim"], header_end, is_vague=c.get("is_vague", False)):
            filtered.append(c)

    # Form connected overlap groups, so a spanning candidate cannot leave a duplicate
    # behind by only comparing with the immediately previous span.
    filtered.sort(key=lambda x: (x["inicio"], -(x["fim"] - x["inicio"])))
    groups = []
    for candidate in filtered:
        if not groups or candidate["inicio"] >= groups[-1]["end"]:
            groups.append({"end": candidate["fim"], "items": [candidate]})
        else:
            groups[-1]["end"] = max(groups[-1]["end"], candidate["fim"])
            groups[-1]["items"].append(candidate)

    pattern_rank = {"sumula": 5, "lei": 5, "cnj": 4, "classica": 3, "vaga": 2, "vaga_generica": 1}
    deduped = [
        max(group["items"], key=lambda item: (
            pattern_rank.get(item["pattern"], 0),
            (item["fim"] - item["inicio"]),
        ))
        for group in groups
    ]
    return deduped
