# -*- coding: utf-8 -*-
"""Filtro de falsos positivos e distratores em documentos judiciais."""
import re


def find_header_boundary(text: str) -> int:
    """Encontra o offset onde termina o cabeçalho formal do documento."""
    m_section = re.search(r"\n\s*(?:I\s*[-—–]|EMENTA\b|RELAT[ÓO]RIO\b|MEMORIAL\b|CONTRARRAZ[ÕO]ES\b|PARECER\b)", text)
    if m_section:
        return m_section.start()
        
    lines = text.splitlines(keepends=True)
    limit_lines = min(len(lines), 16)
    return sum(len(l) for l in lines[:limit_lines])


def is_distractor_span(text: str, inicio: int, fim: int, header_end: int, is_vague: bool = False) -> bool:
    """Avalia se um span extraído é um distrator (falso positivo)."""
    if inicio < header_end and is_vague:
        context_before = text[max(0, inicio - 60):inicio].lower()
        if re.search(r"\b(?:assunto|interessado|refer[êe]ncia)\s*:", context_before):
            return True

    trecho = text[inicio:fim].strip().lower()
    
    # 1. Distratores de cabeçalho explícitos (apenas no preâmbulo do documento)
    if inicio < header_end:
        context_before = text[max(0, inicio - 60):inicio].lower()
        is_cited = any(w in context_before for w in ("como", "conforme", "consoante", "destacar", "vide", "ver", "julgado", "acórdão", "acordao", "precedente", "apreciação", "apreciar"))
        if not is_cited:
            if any(trecho.startswith(p) for p in ("autos nº", "autos n.", "protocolo nº", "protocolo n.", "memorial nº", "feito principal", "processo nº", "processo n.", "parecer nº", "parecer n.", "parecer jurídico")):
                return True
            if inicio < 300 and re.search(r"^(?:autos|processo|protocolo|parecer)\s*(?:n[ºo°.]|:)", text[max(0, inicio - 20):fim].strip(), re.IGNORECASE):
                return True
        
    # 2. Folhas de processo
    if re.search(r"\bfls?\.?\s*\d+", trecho):
        return True
    if re.search(r"\bfolhas?\s+\d+", trecho):
        return True
        
    # 3. Identificador interno de sistema
    if "identificador interno" in trecho:
        return True
        
    # 4. Inscrição na OAB
    context = text[max(0, inicio - 35):min(len(text), fim + 12)]
    if re.search(r"\boab\s*/\s*[a-z]{2}\b", context, re.IGNORECASE):
        return True

    # 5. Acrônimos ou palavras isoladas sem número/identificador
    if trecho in ("re", "rs", "stf", "stj", "tst", "tse", "trf", "tj", "art", "art.", "lei", "sum", "súmula"):
        return True

    return False
