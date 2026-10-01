# -*- coding: utf-8 -*-
"""Utilitários para carregar os documentos de entrada."""
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LineEndingOffsetMap:
    """Compact map from LF boundaries to the original Unicode codepoints."""

    normalized_length: int
    crlf_positions: tuple[int, ...]

    def to_original(self, boundary: int) -> int:
        if boundary < 0 or boundary > self.normalized_length:
            raise ValueError("Fronteira LF fora da entrada.")
        return boundary + bisect_left(self.crlf_positions, boundary)


def load_document(txt_path: Path | str) -> str:
    """Lê documento em codepoints Unicode UTF-8 exatos."""
    with Path(txt_path).open("r", encoding="utf-8", newline="") as stream:
        return stream.read()


def normalize_line_endings(text: str) -> tuple[str, LineEndingOffsetMap, dict[str, int]]:
    """Converte CRLF e CR isolado para LF com mapa compacto ao texto original."""
    if "\r" not in text:
        return text, LineEndingOffsetMap(len(text), ()), {
            "crlf": 0,
            "lf": text.count("\n"),
            "cr": 0,
        }

    normalized: list[str] = []
    crlf_positions: list[int] = []
    crlf_count = 0
    lone_cr_count = 0
    index = 0
    normalized_length = 0

    while index < len(text):
        if text[index] == "\r":
            if index + 1 < len(text) and text[index + 1] == "\n":
                crlf_positions.append(normalized_length)
                index += 2
                crlf_count += 1
            else:
                index += 1
                lone_cr_count += 1
            normalized.append("\n")
            normalized_length += 1
        else:
            normalized.append(text[index])
            index += 1
            normalized_length += 1

    return "".join(normalized), LineEndingOffsetMap(
        normalized_length, tuple(crlf_positions)
    ), {
        "crlf": crlf_count,
        "lf": text.count("\n") - crlf_count,
        "cr": lone_cr_count,
    }


def map_span_to_original(
    start: int, end: int, offset_map: LineEndingOffsetMap
) -> tuple[int, int]:
    """Map an LF span to its exact half-open span in the original text."""
    if start < 0 or end <= start or end > offset_map.normalized_length:
        raise ValueError("Span LF vazio ou fora do mapa da entrada original.")
    return offset_map.to_original(start), offset_map.to_original(end)


def load_all_documents(txt_dir: Path | str) -> dict[str, str]:
    """Carrega todos os arquivos .txt do diretório mapeados por documento_id."""
    docs = {}
    p = Path(txt_dir)
    for f in sorted(p.glob("*.txt")):
        docs[f.stem] = load_document(f)
    return docs
