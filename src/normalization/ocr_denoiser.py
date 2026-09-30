# -*- coding: utf-8 -*-
"""Denoiser para ruído de OCR e variações de superfície típicas do Nível 2."""
import re


def clean_spaces_and_newlines(text: str) -> str:
    """Substitui non-breaking spaces, remove quebras de linha em números e normaliza espaços."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"(\d)\s*\n\s*(\d)", r"\1\2", text)
    text = re.sub(r"(\d)\s*-\s*\n\s*\.?(\d)", r"\1\2", text)
    text = re.sub(r"\n\s*-\s*([A-Z]{2})", r" - \1", text)
    text = re.sub(r"\n\s*/\s*([A-Z]{2})", r"/\1", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def denoise_word(word: str) -> str:
    """Corrige palavras comuns com erro de OCR."""
    w_lower = word.lower()
    if w_lower.startswith("5úm") or w_lower.startswith("5um"):
        return "Súmula" if word[0] == "5" else "súmula"
    if w_lower == "dc":
        return "de"
    if w_lower == "ja":
        return "já"
    if "rãcio" in w_lower or "premissa equivocãda" in w_lower:
        word = word.replace("ã", "a")
    return word


def denoise_number_str(num_str: str) -> str:
    """
    Substitui letras que são claramente dígitos confundidos pelo OCR em contexto numérico.
    Garantia do desafio: dígito NUNCA troca por dígito.
    """
    res = []
    for i, ch in enumerate(num_str):
        prev_is_digit = (i > 0 and (num_str[i - 1].isdigit() or num_str[i - 1] in (".", "-")))
        next_is_digit = (i < len(num_str) - 1 and (num_str[i + 1].isdigit() or num_str[i + 1] in (".", "-")))
        in_digit_context = prev_is_digit or next_is_digit

        if ch in ("l", "I", "|") and in_digit_context:
            res.append("1")
        elif ch in ("O", "o") and in_digit_context:
            res.append("0")
        elif ch == "G" and in_digit_context:
            res.append("6")
        elif ch in ("S", "s") and (prev_is_digit and next_is_digit):
            res.append("5")
        elif ch == "g" and (prev_is_digit and next_is_digit):
            res.append("9")
        else:
            res.append(ch)
    return "".join(res)
