# -*- coding: utf-8 -*-
"""Classificação e atribuição de classes: real, inventada, incompleta."""
from typing import Any
from src.resolution.result import ResolutionResult, coerce_resolution


def classify_citation(citation: dict[str, Any], resolve_tuple: ResolutionResult | tuple) -> dict[str, Any]:
    """
    Classifica a citação segundo a cardinalidade de candidatos da base congelada.
    
    Regras da competição:
    - real: exatamente 1 registro correspondente no acervo; exige id_canonico.
    - inventada: identificadores suficientes para busca, mas 0 correspondências na base.
    - incompleta: insuficiente para formular consulta ou 2+ candidatos sem desempate.
    """
    result = coerce_resolution(resolve_tuple)
    n_cands, canon_id, target_type = result.n_candidates, result.canonical_id, result.target_type
    
    if citation.get("is_vague", False) or not result.sufficient:
        return {
            "classificacao": "incompleta",
            "id_canonico": None,
            "target_type": target_type,
            "n_candidates": n_cands,
            "resolution_method": result.method,
            "resolution_evidence": list(result.evidence),
            "resolution_reason": result.reason,
        }
        
    if n_cands == 1 and canon_id is not None:
        return {
            "classificacao": "real",
            "id_canonico": str(canon_id),
            "target_type": target_type,
            "n_candidates": 1,
            "resolution_method": result.method,
            "resolution_evidence": list(result.evidence),
            "resolution_reason": result.reason,
        }
    elif n_cands == 0 and result.sufficient:
        return {
            "classificacao": "inventada",
            "id_canonico": None,
            "target_type": target_type,
            "n_candidates": 0,
            "resolution_method": result.method,
            "resolution_evidence": list(result.evidence),
            "resolution_reason": result.reason,
        }
    else:  # n_cands >= 2
        return {
            "classificacao": "incompleta",
            "id_canonico": None,
            "target_type": target_type,
            "n_candidates": n_cands,
            "resolution_method": result.method,
            "resolution_evidence": list(result.evidence),
            "resolution_reason": result.reason,
        }
