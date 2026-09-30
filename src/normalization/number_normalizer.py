# -*- coding: utf-8 -*-
"""Normalização e formatação de números de processo (padrão antigo e padrão CNJ)."""
import re
from .ocr_denoiser import denoise_number_str, clean_spaces_and_newlines
from .class_normalizer import CLASS_ALIASES


def extract_number_token(citation_text: str) -> str:
    """
    Localiza e extrai o trecho estritamente numérico do processo em uma citação.
    Ignora classes, conectivos ('no', 'de') e UF.
    """
    cleaned = clean_spaces_and_newlines(citation_text)
    
    # 1. Padrão CNJ flexível (1 a 7 dígitos iniciais)
    m_cnj = re.search(r"(\d{1,7}[\s\-\.]*\d{2}[\s\-\.]*\d{4}[\s\-\.]*\d[\s\-\.]*\d{2}[\s\-\.]*\d{4})", cleaned)
    if m_cnj:
        return m_cnj.group(1)

    # 2. Número após indicador de número (nº, n., no, etc.)
    m_num_prefix = re.search(
        r"(?:n[ºo°\.]|n\.\s*|n\s+|no\s+|n\xba\s*)([\d\.\s\-_lIOoGgSs]{3,22})",
        cleaned,
        re.IGNORECASE,
    )
    if m_num_prefix:
        tok = m_num_prefix.group(1).strip()
        tok = re.sub(r"[\s/\-–]+[A-Za-z]{2}\b", "", tok)
        if any(c.isdigit() for c in tok):
            return tok

    # 3. Remover classes conhecidas para não confundir letras de classe (ex: 'Rcl') com dígitos
    text_no_class = cleaned
    for alias in sorted(CLASS_ALIASES.keys(), key=len, reverse=True):
        text_no_class = re.sub(rf"\b{re.escape(alias)}\b", " ", text_no_class, flags=re.IGNORECASE)

    # Remover conectivos
    text_no_class = re.sub(r"\b(no|na|nos|nas|em|de|do|da|dos|das|processo)\b", " ", text_no_class, flags=re.IGNORECASE)
    # Remover UF
    text_no_class = re.sub(r"[\s/\-–]+\([A-Za-z]{2}\)|[\s/\-–]+[A-Za-z]{2}\b", " ", text_no_class)

    # Procurar bloco numérico
    matches = re.findall(r"([\d\.\s\-_lIOoGgSs]{3,22})", text_no_class)
    valid_candidates = []
    for m in matches:
        m_s = m.strip()
        dig_count = sum(1 for c in m_s if c.isdigit())
        if dig_count >= 2:
            valid_candidates.append((dig_count, m_s))
            
    if valid_candidates:
        valid_candidates.sort(key=lambda x: x[0], reverse=True)
        return valid_candidates[0][1]

    return cleaned


def extract_digits(s: str) -> str:
    """Aplica denoiser OCR no token numérico isolado e extrai apenas dígitos."""
    num_tok = extract_number_token(s)
    denoised = denoise_number_str(num_tok)
    return re.sub(r"\D", "", denoised)


def format_triplet_number(digits: str) -> str:
    """
    Formata sequência de dígitos no padrão clássico brasileiro (agrupado de 3 em 3 a partir da direita).
    Exemplo: '1741784' -> '1.741.784'
    """
    if not digits:
        return ""
    groups = []
    for i in range(len(digits), 0, -3):
        start = max(0, i - 3)
        groups.append(digits[start:i])
    return ".".join(reversed(groups))


def format_cnj_number(digits: str) -> str:
    """Formata sequência de dígitos no padrão CNJ."""
    if len(digits) == 20:
        return f"{digits[:7]}-{digits[7:9]}.{digits[9:13]}.{digits[13]}.{digits[14:16]}.{digits[16:]}"
    elif 14 <= len(digits) <= 19:
        # Padrão CNJ com sequencial curto (ex: TST antigo, 4 ou 5 dígitos no início)
        tail = digits[-13:]
        head = digits[:-13]
        return f"{head}-{tail[:2]}.{tail[2:6]}.{tail[6]}.{tail[7:9]}.{tail[9:]}"
    return digits


def normalize_process_number(citation_text: str) -> dict[str, str]:
    """Analisa citação e extrai os componentes canônicos do número de processo."""
    digits = extract_digits(citation_text)
    
    if len(digits) >= 14:
        fmt = format_cnj_number(digits)
        return {
            "digits": digits,
            "formatted": fmt,
        }
    else:
        fmt = format_triplet_number(digits)
        return {
            "digits": digits,
            "formatted": fmt,
        }
