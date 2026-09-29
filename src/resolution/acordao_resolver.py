# -*- coding: utf-8 -*-
"""Resolve a case number only when it identifies the record's principal case."""
import re
import sqlite3
from typing import Any

from .canonical_db import query_fts
from .header_index import HeaderIndex, extract_class_families, parse_header_tokens
from .result import ResolutionResult
from ..normalization.number_normalizer import normalize_process_number
from ..normalization.uf_normalizer import extract_uf


class AcordaoResolver:
    """Resolve case citations against the frozen canonical ruling collection."""

    def __init__(self, conn: sqlite3.Connection, header_index: HeaderIndex):
        self.conn = conn
        self.header_index = header_index

    def resolve(self, text: str) -> ResolutionResult:
        norm_info = normalize_process_number(text)
        digits = norm_info["digits"]
        class_families = extract_class_families(text)
        uf = extract_uf(text)
        m_trib = re.search(r"\b(STF|STJ|TST|TSE|STM)\b", text, re.IGNORECASE)
        tribunal = m_trib.group(1).upper() if m_trib else None
        is_cnj = len(digits) >= 14
        is_tema = "tema" in text.lower() or "temã" in text.lower()
        sufficient = (len(digits) >= 5 or is_tema or bool(class_families) or bool(uf) or bool(tribunal)) and len(digits) >= 3
        if not sufficient:
            return ResolutionResult.make(
                sufficient=False,
                method="case_parse",
                evidence=[digits] if digits else [],
                reason="número e qualificadores insuficientes para uma busca identificável",
                target_type="acordao",
            )

        indexed = self.header_index.lookup(
            digits,
            uf=uf,
            tribunal=tribunal,
            class_hint=text,
        )
        if indexed:
            return self._result_from_entries(indexed, "header_exact", digits, uf, tribunal, class_families)

        # FTS retrieves records that mention the number; the same principal-title
        # parser used to build the index decides whether a result is that case.
        rows = query_fts(self.conn, norm_info["formatted"], limit=None)
        confirmed: list[dict[str, Any]] = []
        for row in rows:
            if row.get("natureza") != "acordao":
                continue
            principal = parse_header_tokens(row.get("cabecalho", ""))
            if not any(item["digits"] == digits for item in principal):
                continue
            parsed = next(item for item in principal if item["digits"] == digits)
            entry = {
                "id": int(row["id"]),
                "tribunal": (row.get("tribunal") or "").upper() or None,
                "uf": parsed.get("uf"),
                "class_families": parsed.get("class_families", []),
                "principal_text": parsed.get("principal_text", ""),
            }
            if tribunal and entry["tribunal"] != tribunal:
                continue
            if uf and entry["uf"] != uf:
                continue
            if class_families and not set(class_families).issubset(set(entry["class_families"])):
                continue
            confirmed.append(entry)

        return self._result_from_entries(confirmed, "fts_principal_exact", digits, uf, tribunal, class_families)

    @staticmethod
    def _result_from_entries(
        entries: list[dict[str, Any]],
        method: str,
        digits: str,
        uf: str | None,
        tribunal: str | None,
        class_families: list[str],
    ) -> ResolutionResult:
        qualifier_evidence = [f"número:{digits}"]
        qualifier_evidence.extend(f"classe:{family}" for family in class_families)
        if uf:
            qualifier_evidence.append(f"UF:{uf}")
        if tribunal:
            qualifier_evidence.append(f"tribunal:{tribunal}")
        ids = list(dict.fromkeys(int(entry["id"]) for entry in entries))
        if len(ids) == 1:
            reason = "identificador principal confirmado no registro canônico"
        elif ids:
            reason = "mais de um registro canônico satisfaz os identificadores"
        else:
            reason = "consulta suficiente sem registro correspondente na cobertura congelada"
        return ResolutionResult.make(
            ids,
            sufficient=True,
            method=method,
            evidence=qualifier_evidence,
            reason=reason,
            target_type="acordao",
        )
