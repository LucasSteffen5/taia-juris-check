# -*- coding: utf-8 -*-
"""Index only the principal case identifier stated in each canonical record."""
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from ..normalization.class_normalizer import CLASS_ALIASES
from ..normalization.ocr_denoiser import clean_spaces_and_newlines
from ..normalization.number_normalizer import extract_digits, extract_number_token
from ..normalization.uf_normalizer import extract_uf
from ..config import assert_output_outside_official_data
from .result import ResolutionResult

PARSER_VERSION = 3
INDEX_VERSION = 3


def _families(class_text: str) -> list[str]:
    """Find canonical class families in a class label, longest aliases first."""
    found: list[tuple[int, int, str]] = []
    for alias, family in sorted(CLASS_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        escaped = re.escape(alias).replace(r"\ ", r"\s+")
        for match in re.finditer(rf"(?<!\w){escaped}(?!\w)", class_text, re.IGNORECASE):
            found.append((match.start(), match.end(), family))
    # Keep overlapping aliases at the longest match, but retain compound classes.
    selected: list[tuple[int, int, str]] = []
    for candidate in sorted(found, key=lambda item: (item[0], -(item[1] - item[0]))):
        if any(candidate[0] < old[1] and old[0] < candidate[1] for old in selected):
            continue
        selected.append(candidate)
    return list(dict.fromkeys(item[2] for item in selected))


def extract_class_families(class_text: str) -> list[str]:
    """Public class-label parser used by indexing and lookup."""
    return _families(class_text)


def parse_header_tokens(header_text: str) -> list[dict[str, Any]]:
    """Parse principal identifiers from a case title, autuação, or front matter."""
    m_sec = re.search(r"\n\s*(?:EMENTA|RELAT[ÓO]RIO|ACÓRDÃO\s+VISTOS|ACORDAM)\b", header_text, re.IGNORECASE)
    clean_header = (
        header_text[:m_sec.start()]
        if m_sec
        else "\n".join(header_text.splitlines()[:30])[:2000]
    )
    text = clean_spaces_and_newlines(clean_header)
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    cnj_spans: list[tuple[int, int]] = []

    # 1. TST autuação
    m_tst = re.search(
        r"n[ºo°.]\s*(?:TST[-\s]*)?([A-Za-z0-9\-\.\s]{8,50}?)\s*(?:,?\s*em\s+que\s+[ée]|\b)",
        text[:5000],
        re.IGNORECASE,
    )
    if m_tst:
        raw = m_tst.group(1).strip(" .-/")
        digs = extract_digits(raw)
        if len(digs) >= 10:
            seen.add(digs)
            results.append({
                "digits": digs,
                "uf": extract_uf(raw),
                "class_families": _families(raw),
            })

    # 2. CNJ (14 to 20 digits) in header title
    for m in re.finditer(
        r"\b(\d{1,7}[\s\-.]*\d{2}[\s\-.]*\d{4}[\s\-.]*\d[\s\-.]*\d{2}[\s\-.]*\d{4})\b|\b(\d{14,20})\b",
        text[:5000],
    ):
        digs = extract_digits(m.group(0))
        if len(digs) >= 14 and digs not in seen:
            seen.add(digs)
            cnj_spans.append(m.span())
            results.append({
                "digits": digs,
                "uf": extract_uf(text[:m.end() + 20]),
                "class_families": _families(text[:m.start()]),
            })

    # 3. Classical process numbers strictly outside CNJ spans (and excluding dates)
    for m in re.finditer(
        r"(?:n[ºo°.]\s*|n\.\s*|n\s+|no\s+|n\xba\s*)?(\d{1,3}(?:\.\d{3})+|\b\d{4,8}\b)",
        text[:5000],
        re.IGNORECASE,
    ):
        if any(s <= m.start() < e or s < m.end() <= e for s, e in cnj_spans):
            continue
        if m.start() >= 3 and re.search(r"\d{1,2}/\d{1,2}/$", text[:m.start()]):
            continue
        if m.group(1).isdigit() and 1990 <= int(m.group(1)) <= 2030 and "." not in m.group(0):
            continue
        digs = extract_digits(m.group(1))
        if 4 <= len(digs) <= 8 and digs not in seen:
            seen.add(digs)
            results.append({
                "digits": digs,
                "uf": extract_uf(text[:m.end() + 30]),
                "class_families": _families(text[max(0, m.start() - 40):m.end()]),
            })
            break

    return results


class HeaderIndex:
    """Database-fingerprinted index of principal identifiers for canonical rulings."""

    def __init__(self):
        self.exact_map: dict[str, list[dict[str, Any]]] = {}
        self.database_sha256: str | None = None

    def build_from_db(self, conn: sqlite3.Connection, database_sha256: str | None = None) -> None:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT id, tribunal, substr(texto, 1, 5000) AS texto
                 FROM documentos WHERE natureza = 'acordao' ORDER BY id"""
        )
        self.exact_map.clear()
        self.database_sha256 = database_sha256
        for row in cursor.fetchall():
            cab = row["texto"] or ""
            canon_id = int(row["id"])
            parsed = parse_header_tokens(cab)
            if not parsed:
                continue
            for principal in parsed:
                entry = {
                    "id": canon_id,
                    "tribunal": (row["tribunal"] or "").upper() or None,
                    "uf": principal["uf"],
                    "class_families": principal["class_families"],
                    "cabecalho": cab[:600],
                }
                bucket = self.exact_map.setdefault(principal["digits"], [])
                if not any(item["id"] == canon_id for item in bucket):
                    bucket.append(entry)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        assert_output_outside_official_data(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "index_version": INDEX_VERSION,
            "parser_version": PARSER_VERSION,
            "database_sha256": self.database_sha256,
            "exact_map": self.exact_map,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    def load(self, path: Path | str, expected_database_sha256: str | None = None) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("index_version") != INDEX_VERSION or data.get("parser_version") != PARSER_VERSION:
            raise ValueError("Índice de cabeçalhos incompatível com esta versão do parser.")
        if expected_database_sha256 and data.get("database_sha256") != expected_database_sha256:
            raise ValueError("Índice de cabeçalhos pertence a outra versão do banco canônico.")
        self.database_sha256 = data.get("database_sha256")
        self.exact_map = data.get("exact_map", {})

    def lookup(
        self,
        digits: str,
        uf: str | None = None,
        tribunal: str | None = None,
        class_hint: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return exact principal-identifier matches satisfying every explicit qualifier."""
        matches = list(self.exact_map.get(digits, []))
        if not matches:
            return []

        if tribunal:
            matches = [item for item in matches if item.get("tribunal") == tribunal.upper()]
        if uf:
            matches = [item for item in matches if item.get("uf") == uf.upper()]
        if class_hint:
            c_low = class_hint.lower()
            if "embargos de diverg" in c_low:
                matches = [m for m in matches if "embargos de diverg" in m.get("cabecalho", "").lower()]
            elif "recurso especial" in c_low or "resp" in c_low:
                matches = [m for m in matches if "embargos de diverg" not in m.get("cabecalho", "").lower()]
            requested_families = _families(class_hint)
            if requested_families:
                main_classes = {"HC", "REsp", "RE", "ADI", "MS", "Rcl", "RO", "APL", "RSE", "RMS"}
                requested = set(requested_families) & main_classes
                if requested:
                    matches = [
                        item for item in matches
                        if requested & (set(item.get("class_families", [])) & main_classes)
                    ]
        return matches

    @staticmethod
    def fingerprint(path: Path | str) -> str:
        digest = hashlib.sha256()
        path = Path(path)
        if path.is_dir():
            for file_path in sorted(
                item for item in path.rglob("*")
                if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc"
            ):
                digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
                with file_path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
        else:
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        return digest.hexdigest()
