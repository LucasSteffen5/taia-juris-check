# -*- coding: utf-8 -*-
"""Extração e normalização de Unidade Federativa (UF) em citações."""
import re

VALID_UFS = {
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA",
    "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN",
    "RS", "RO", "RR", "SC", "SP", "SE", "TO",
}

STATE_NAMES = {
    "ACRE": "AC", "ALAGOAS": "AL", "AMAPÁ": "AP", "AMAPA": "AP",
    "AMAZONAS": "AM", "BAHIA": "BA", "CEARÁ": "CE", "CEARA": "CE",
    "DISTRITO FEDERAL": "DF", "ESPÍRITO SANTO": "ES", "ESPIRITO SANTO": "ES",
    "GOIÁS": "GO", "GOIAS": "GO", "MARANHÃO": "MA", "MARANHAO": "MA",
    "MATO GROSSO": "MT", "MATO GROSSO DO SUL": "MS", "MINAS GERAIS": "MG",
    "PARÁ": "PA", "PARA": "PA", "PARAÍBA": "PB", "PARAIBA": "PB",
    "PARANÁ": "PR", "PARANA": "PR", "PERNAMBUCO": "PE", "PIAUÍ": "PI",
    "PIAUI": "PI", "RIO DE JANEIRO": "RJ", "RIO GRANDE DO NORTE": "RN",
    "RIO GRANDE DO SUL": "RS", "RONDÔNIA": "RO", "RONDONIA": "RO",
    "RORAIMA": "RR", "SANTA CATARINA": "SC", "SÃO PAULO": "SP",
    "SAO PAULO": "SP", "SERGIPE": "SE", "TOCANTINS": "TO",
}

UF_PATTERN = re.compile(
    r"(?:[/–\-]\s*|\(\s*)([A-Za-z]{2})(?:\s*\)|$|\s+)",
    flags=re.IGNORECASE,
)


def extract_uf(text: str) -> str | None:
    """Extrai UF válida ao final de uma citação ou após separadores típicos."""
    matches = list(UF_PATTERN.finditer(text))
    if matches:
        last_match = matches[-1]
        candidate = last_match.group(1).upper()
        if candidate in VALID_UFS:
            return candidate
    for state, uf in STATE_NAMES.items():
        if re.search(rf"\b{state}\b", text, re.IGNORECASE):
            return uf
    return None
