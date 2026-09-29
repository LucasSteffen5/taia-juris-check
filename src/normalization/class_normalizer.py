# -*- coding: utf-8 -*-
"""Normalização e mapeamento canônico de classes processuais brasileiras."""
import re
from .ocr_denoiser import clean_spaces_and_newlines

CLASS_ALIASES = {
    # Recurso Especial
    "resp": "REsp",
    "rec. esp.": "REsp",
    "rec esp": "REsp",
    "r.esp.": "REsp",
    "r. esp.": "REsp",
    "recurso especial": "REsp",
    # Agravo em Recurso Especial
    "aresp": "AREsp",
    "a.resp": "AREsp",
    "a. resp": "AREsp",
    "agravo em recurso especial": "AREsp",
    "agresp": "AREsp",
    "ag.resp": "AREsp",
    # Agravo Interno
    "agint": "AgInt",
    "ag. int.": "AgInt",
    "ag. int": "AgInt",
    "ag int": "AgInt",
    "agravo interno": "AgInt",
    # Agravo Regimental
    "agrg": "AgRg",
    "agr": "AgRg",
    "ag. rg.": "AgRg",
    "agravo regimental": "AgRg",
    # Reclamação
    "rcl": "Rcl",
    "recl.": "Rcl",
    "recl": "Rcl",
    "reclamacao": "Rcl",
    "reclamação": "Rcl",
    # Habeas Corpus e Recursos em HC
    "rhc": "RHC",
    "r.hc": "RHC",
    "recurso em habeas corpus": "RHC",
    "hc": "HC",
    "h.c.": "HC",
    "habeas corpus": "HC",
    # Recurso Extraordinário
    "re": "RE",
    "r.e.": "RE",
    "recurso extraordinario": "RE",
    "recurso extraordinário": "RE",
    "are": "ARE",
    # Recurso Ordinário / Mandado de Segurança
    "rms": "RMS",
    "recurso em mandado de seguranca": "RMS",
    "recurso em mandado de segurança": "RMS",
    # Eleitoral
    "respe": "REspe",
    "respe.": "REspe",
    "recurso especial eleitoral": "REspe",
    "arespel": "AREspEl",
    "ai": "AI",
    "agravo de instrumento": "AI",
    "r-rp": "R-Rp",
    "rrp": "R-Rp",
    # Militar
    "apl": "APL",
    "apelacao": "APL",
    "apelação": "APL",
    "rse": "RSE",
    "recurso em sentido estrito": "RSE",
    # Trabalhista (TST)
    "rr": "RR",
    "arr": "ARR",
    "airr": "AIRR",
    "e-rr": "E-RR",
    # Embargos
    "edcl": "EDcl",
    "ed": "EDcl",
    "eds": "EDcl",
    "embargos de declaracao": "EDcl",
    "embargos de declaração": "EDcl",
    # Outros
    "ar": "AR",
    "sl": "SL",
    "suspensão de liminar": "SL",
    "suspensao de liminar": "SL",
    "suspensao de liminar e de sentenca": "SLS",
    "suspensão de liminar e de sentença": "SLS",
}


def normalize_class_name(raw_class: str) -> str:
    """
    Normaliza sequências compostas de classes (ex: 'EDcl nos EDcl no AgInt no AREsp').
    """
    text = clean_spaces_and_newlines(raw_class)
    # Substituir expressões multi-palavras primeiro
    for phrase, canonical in sorted(CLASS_ALIASES.items(), key=lambda x: -len(x[0])):
        if " " in phrase or "." in phrase:
            pattern = r"\b" + re.escape(phrase) + r"\b"
            text = re.sub(pattern, canonical, text, flags=re.IGNORECASE)

    # Substituir conectivos (incluindo 'e' como conjunção)
    text = re.sub(r"\b(no|na|nos|nas|em|de|do|da|dos|das|e)\b", " ", text, flags=re.IGNORECASE)
    parts = [p.strip() for p in text.split() if p.strip()]
    
    canonical_parts = []
    for part in parts:
        part_clean = part.lower().rstrip(".:,;")
        if part_clean in CLASS_ALIASES:
            canonical_parts.append(CLASS_ALIASES[part_clean])
        else:
            # Manter capitalizado se não tiver alias exato
            canonical_parts.append(part.capitalize())
            
    return " ".join(canonical_parts) if canonical_parts else raw_class.strip()
