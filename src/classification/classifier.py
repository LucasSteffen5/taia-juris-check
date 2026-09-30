# -*- coding: utf-8 -*-
"""Classificação e atribuição de classes: real, inventada, incompleta."""
from src.resolution.result import ResolutionResult


def classify_citation(citation: dict, result: ResolutionResult) -> dict[str, str | None]:
    """
    Classifica a citação segundo a cardinalidade de candidatos da base congelada.
    
    Regras da competição:
    - real: exatamente 1 registro correspondente no acervo; exige id_canonico.
    - inventada: identificadores suficientes para busca, mas 0 correspondências na base.
    - incompleta: insuficiente para formular consulta ou 2+ candidatos sem desempate.
    """
    if citation.get("is_vague", False) or not result.sufficient:
        classification, canonical_id = "incompleta", None
    elif result.n_candidates == 1 and result.canonical_id is not None:
        classification, canonical_id = "real", str(result.canonical_id)
    elif result.n_candidates == 0:
        classification, canonical_id = "inventada", None
    else:
        classification, canonical_id = "incompleta", None
    return {"classificacao": classification, "id_canonico": canonical_id}
