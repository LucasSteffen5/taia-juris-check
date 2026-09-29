# -*- coding: utf-8 -*-
"""Resolução canônica de dispositivos legais (artigos de lei e códigos)."""
import re
import sqlite3
from typing import Any
from ..normalization.ocr_denoiser import clean_spaces_and_newlines
from .result import ResolutionResult

DIPLOMA_MAP = {
    # Lei Complementar 64/1990
    "lei complementar nº 64": "LC64",
    "lei complementar n. 64": "LC64",
    "lei complementar 64": "LC64",
    "lc 64": "LC64",
    "lc nº 64": "LC64",
    "lc n. 64": "LC64",
    # CLT
    "clt": "CLT",
    "consolidação das leis do trabalho": "CLT",
    "consolidacao das leis do trabalho": "CLT",
    "decreto-lei nº 5.452": "CLT",
    "decreto-lei n. 5.452": "CLT",
    "5.452": "CLT",
    "5452": "CLT",
    # CPC
    "cpc": "CPC",
    "código de processo civil": "CPC",
    "codigo de processo civil": "CPC",
    "lei nº 13.105": "CPC",
    "lei n. 13.105": "CPC",
    "lei 13.105": "CPC",
    "13.105": "CPC",
    "13105": "CPC",
    # Código Civil
    "cc": "CC",
    "código civil": "CC",
    "codigo civil": "CC",
    "lei nº 10.406": "CC",
    "lei 10.406": "CC",
    "10.406": "CC",
    "10406": "CC",
    # Constituição Federal
    "cf": "CF",
    "constituição federal": "CF",
    "constituicao federal": "CF",
    "constituição fedcral": "CF",
    "constituicao fedcral": "CF",
    "constituição da república": "CF",
    "constituicao da republica": "CF",
    "constituição": "CF",
    "constituicao": "CF",
    "carta magna": "CF",
    "carta da república": "CF",
    "carta da republica": "CF",
    # CPP
    "cpp": "CPP",
    "código de processo penal": "CPP",
    "codigo de processo penal": "CPP",
    "decreto-lei nº 3.689": "CPP",
    "decreto-lei n. 3.689": "CPP",
    "3.689": "CPP",
    "3689": "CPP",
    # CPM
    "cpm": "CPM",
    "código penal militar": "CPM",
    "codigo penal militar": "CPM",
    "decreto-lei nº 1.001": "CPM",
    "decreto-lei n. 1.001": "CPM",
    "1.001": "CPM",
    "1001": "CPM",
    # CDC
    "cdc": "CDC",
    "código de defesa do consumidor": "CDC",
    "codigo de defesa do consumidor": "CDC",
    "lei nº 8.078": "CDC",
    "lei 8.078": "CDC",
    "8.078": "CDC",
    "8078": "CDC",
    # Código Eleitoral
    "código eleitoral": "CE",
    "codigo eleitoral": "CE",
    "lei nº 4.737": "CE",
    "lei 4.737": "CE",
    "4.737": "CE",
    "4737": "CE",
}


def normalize_diploma(text: str) -> str | None:
    t_clean = clean_spaces_and_newlines(text).lower()
    # Se 'lei complementar nº 64' estiver presente, deve ter prioridade máxima
    lc_match = re.search(
        r"(?<!\w)(?:lei\s+complementar\s+n[ºo.]?\s*64|lc\s+n?[ºo.]?\s*64)(?!\d)",
        t_clean,
    )
    if lc_match:
        return "LC64"
    for k in sorted(DIPLOMA_MAP.keys(), key=len, reverse=True):
        escaped = re.escape(k).replace(r"\ ", r"\s+")
        if re.search(rf"(?<!\w){escaped}(?!\w)", t_clean):
            return DIPLOMA_MAP[k]
    return None


def extract_article_number(text: str) -> str | None:
    """Extrai o número do artigo normalizado (removendo pontos e símbolos de ordinal)."""
    m = re.search(r"art(?:igo)?\.?\s*([\d\.]+[ºo°]?)", text, re.IGNORECASE)
    if not m:
        return None
    raw = m.group(1)
    cleaned = raw.replace(".", "").replace("º", "").replace("o", "").replace("°", "").strip()
    return cleaned if cleaned else None


class DispositivoResolver:
    """Carrega os 13 dispositivos legais da base congelada e resolve por artigo e diploma."""

    def __init__(self, conn: sqlite3.Connection):
        self.dispositivos: list[dict[str, Any]] = []
        self._load_index(conn)

    def _load_index(self, conn: sqlite3.Connection) -> None:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT documento_id, id, tribunal, texto
              FROM documentos
             WHERE natureza = 'dispositivo'
            """
        )
        for row in cursor.fetchall():
            doc_id, canon_id, texto = row["documento_id"], row["id"], row["texto"]
            # Extrair diploma APENAS das primeiras 2 linhas do texto para evitar notas de rodapé
            first_lines = "\n".join(texto.splitlines()[:2])
            diploma = normalize_diploma(first_lines) or normalize_diploma(doc_id)
            art = extract_article_number(first_lines)
            
            self.dispositivos.append({
                "id": canon_id,
                "documento_id": doc_id,
                "texto": first_lines,
                "diploma": diploma,
                "art": art,
            })

    def resolve(self, text: str) -> ResolutionResult:
        """
        Resolve citação de lei.
        Regras:
          - (1, id_canonico): único registro correspondente (REAL)
          - (0, None): identificadores suficientes (artigo + diploma) mas ausente na cobertura (INVENTADA)
          - (-1, None): informação insuficiente para buscar (INCOMPLETA)
        """
        text_clean = clean_spaces_and_newlines(text)
        art = extract_article_number(text_clean)
        diploma = normalize_diploma(text_clean)
        
        # Se não tem artigo, é vaga/incompleta (ex: 'Constituição Federal')
        if not art:
            return ResolutionResult.make(sufficient=False, method="law_parse", evidence=[text_clean], reason="número do artigo ausente", target_type="dispositivo")
            
        # Checar se tem menção a alguma lei ou código (mesmo que fora dos 13 diplomas)
        has_law_identifier = (
            diploma is not None
            or bool(re.search(r"\b(?:lei|c[óo]digo|decreto|constitui[çc][ãa]o|estatuto)\b", text_clean, re.IGNORECASE))
        )
        
        if not has_law_identifier:
            # Tem artigo mas sem lei nenhuma -> Incompleta
            return ResolutionResult.make(sufficient=False, method="law_parse", evidence=[f"art.{art}"], reason="diploma legal ausente", target_type="dispositivo")

        has_specific_unknown_law = bool(re.search(
            r"\b(?:lei(?:\s+complementar)?|decreto(?:-lei)?)\s+(?:n[ºo°.]\s*)?\d[\d./-]*",
            text_clean,
            re.IGNORECASE,
        ))
        if diploma is None and not has_specific_unknown_law:
            return ResolutionResult.make(
                sufficient=False,
                method="law_parse",
                evidence=[f"art.{art}"],
                reason="nome do diploma não identifica uma lei específica",
                target_type="dispositivo",
            )
            
        # Comparar contra os 13 dispositivos
        if diploma:
            candidates = [d["id"] for d in self.dispositivos if d["diploma"] == diploma and d["art"] == art]
            if len(candidates) == 1:
                return ResolutionResult.make(candidates, sufficient=True, method="law_exact", evidence=[f"art.{art}", diploma], reason="artigo e diploma canônicos coincidem", target_type="dispositivo")
            elif len(candidates) > 1:
                return ResolutionResult.make(candidates, sufficient=True, method="law_exact", evidence=[f"art.{art}", diploma], reason="mais de um registro de lei compatível", target_type="dispositivo")
                
        # Tem identificador suficiente (Artigo + Lei/Código), mas não existe nos 13 registros congelados:
        # Pelo contrato da competição, isso é INVENTADA!
        return ResolutionResult.make(
            sufficient=True,
            method="law_exact",
            evidence=[f"art.{art}", diploma or "lei identificada por número"],
            reason="nenhum dispositivo correspondente na cobertura congelada",
            target_type="dispositivo",
        )
