# -*- coding: utf-8 -*-
"""Whitespace-normalized matching view mapped back to source codepoint offsets."""
import re


def normalized_view(text: str) -> tuple[str, list[int], list[int]]:
    """Collapse whitespace for matching while preserving exact source span boundaries."""
    chars: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    index = 0
    while index < len(text):
        if text[index].isspace() or text[index] == "\xa0":
            finish = index + 1
            while finish < len(text) and (text[finish].isspace() or text[finish] == "\xa0"):
                finish += 1
            chars.append(" ")
            starts.append(index)
            ends.append(finish)
            index = finish
        else:
            chars.append(text[index])
            starts.append(index)
            ends.append(index + 1)
            index += 1
    return "".join(chars), starts, ends


def map_normalized_span(start: int, end: int, starts: list[int], ends: list[int]) -> tuple[int, int]:
    if start < 0 or end <= start or end > len(starts):
        raise ValueError("Span normalizado vazio ou fora da entrada.")
    return starts[start], ends[end - 1]
