# -*- coding: utf-8 -*-
"""Resolução canônica de Súmulas e Súmulas Vinculantes."""
import re
import sqlite3
from typing import Any
from ..normalization.ocr_denoiser import clean_spaces_and_newlines, denoise_word
from .result import ResolutionResult


class SumulaResolver:
    """Carrega em memória os registros de natureza='sumula' e resolve por chave canônica."""

    def __init__(self, conn: sqlite3.Connection):
        self.sumulas: dict[tuple[str, int, str | None], set[int]] = {}
        self.by_num: dict[int, list[dict[str, Any]]] = {}
        self._load_index(conn)

    def _load_index(self, conn: sqlite3.Connection) -> None:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT documento_id, id, tribunal, texto
              FROM documentos
             WHERE natureza = 'sumula'
            """
        )
        for row in cursor.fetchall():
            doc_id, canon_id, trib, texto = row["documento_id"], row["id"], row["tribunal"], row["texto"]
            trib_norm = trib.upper().strip() if trib else None
            
            # Identificar número da súmula a partir do texto
            m_vinc = re.search(r"S[uú]mula\s+Vinculante\s+(?:n[ºo.]\s*|n\.\s*)?(\d+)", texto, re.IGNORECASE)
            m_std = re.search(r"S[uú]mula\s+(?:n[ºo.]\s*|n\.\s*)?(\d+)", texto, re.IGNORECASE)
            
            is_vinculante = bool(m_vinc) or "vinculante" in texto.lower() or "vinculante" in doc_id.lower()
            num = int(m_vinc.group(1)) if m_vinc else (int(m_std.group(1)) if m_std else None)
            
            if num is None:
                # Tentar extrair dígitos isolados por delimitadores do documento_id
                m_doc = re.search(r"(?:^|_)(\d{1,4})(?:_|$)", doc_id)
                num = int(m_doc.group(1)) if m_doc else None

            if num is not None:
                tipo = "vinculante" if is_vinculante else "sumula"
                key = (tipo, num, trib_norm)
                self.sumulas.setdefault(key, set()).add(int(canon_id))
                
                # Chave sem tribunal se vinculante (vinculante é sempre STF)
                if is_vinculante:
                    self.sumulas.setdefault((tipo, num, None), set()).add(int(canon_id))
                    self.sumulas.setdefault((tipo, num, "STF"), set()).add(int(canon_id))
                
                entry = {"id": canon_id, "tipo": tipo, "num": num, "tribunal": trib_norm}
                self.by_num.setdefault(num, []).append(entry)

    def resolve(self, text: str) -> ResolutionResult:
        """
        Resolve citação de súmula contra o acervo.
        Retorna:
          (1, id_canonico) se resolveu a registro único existente (REAL)
          (0, None) se tem número suficiente mas não existe no acervo (INVENTADA)
          (-1, None) se insuficiente ou ambígua (INCOMPLETA)
        """
        text_clean = clean_spaces_and_newlines(text)
        
        # Denoiser OCR nas palavras
        tokens = [denoise_word(t) for t in text_clean.split()]
        normalized_str = " ".join(tokens)
        
        is_vinculante = bool(re.search(r"\bvinculante\b", normalized_str, re.IGNORECASE))
        
        # Extrair número da súmula imediatamente após palavra-chave
        m_num = re.search(r"(?:s[uú]mula|5[uú]mula|s[uú]m\.?)\s*(?:vinculante\s*)?(?:n[ºo°\.]\s*|n\.\s*|[^\w\s]*\s*)*(\d+)", normalized_str, re.IGNORECASE)
        if not m_num:
            return ResolutionResult.make(sufficient=False, method="sumula_parse", evidence=[text_clean], reason="número da súmula ausente", target_type="sumula")
        
        num = int(m_num.group(1))
        
        # Extrair tribunal
        m_trib = re.search(r"\b(STF|STJ|TST|TSE|STM)\b", normalized_str, re.IGNORECASE)
        trib = m_trib.group(1).upper() if m_trib else None
        
        tipo = "vinculante" if is_vinculante else "sumula"
        
        # Tentativa 1: chave exata
        if (tipo, num, trib) in self.sumulas:
            ids = sorted(self.sumulas[(tipo, num, trib)])
            reason = "chave canônica única" if len(ids) == 1 else "chave canônica ambígua"
            return ResolutionResult.make(ids, sufficient=True, method="sumula_exact", evidence=[f"{tipo}:{num}", trib or "tribunal omitido"], reason=reason, target_type="sumula")
        
        # Tentativa 2: vinculante sem tribunal
        if is_vinculante and (tipo, num, None) in self.sumulas:
            ids = sorted(self.sumulas[(tipo, num, None)])
            reason = "chave canônica única" if len(ids) == 1 else "chave canônica ambígua"
            return ResolutionResult.make(ids, sufficient=True, method="sumula_exact", evidence=[f"{tipo}:{num}"], reason=reason, target_type="sumula")
            
        # Tentativa 3: buscar por número no índice
        matches = self.by_num.get(num, [])
        if matches:
            if trib is not None:
                filtered = [m for m in matches if m["tribunal"] == trib and m["tipo"] == tipo]
                if len(filtered) == 1:
                    return ResolutionResult.make([filtered[0]["id"]], sufficient=True, method="sumula_exact", evidence=[f"{tipo}:{num}", trib], reason="número e tribunal confirmados", target_type="sumula")
                elif len(filtered) > 1:
                    return ResolutionResult.make([item["id"] for item in filtered], sufficient=True, method="sumula_exact", evidence=[f"{tipo}:{num}", trib], reason="mais de uma súmula compatível", target_type="sumula")
            else:
                filtered = [m for m in matches if m["tipo"] == tipo]
                if len(filtered) == 1:
                    return ResolutionResult.make([filtered[0]["id"]], sufficient=True, method="sumula_exact", evidence=[f"{tipo}:{num}"], reason="número e tipo confirmados", target_type="sumula")
                elif len(filtered) > 1:
                    return ResolutionResult.make([item["id"] for item in filtered], sufficient=True, method="sumula_exact", evidence=[f"{tipo}:{num}"], reason="número da súmula ambíguo", target_type="sumula")
                    
        # Se tem número e tipo/tribunal bem formulado, mas não está no acervo: INVENTADA!
        if trib is None and not is_vinculante:
            return ResolutionResult.make(sufficient=False, method="sumula_parse", evidence=[f"súmula:{num}"], reason="tribunal ausente e número não é único na cobertura", target_type="sumula")
        return ResolutionResult.make(sufficient=True, method="sumula_exact", evidence=[f"{tipo}:{num}", trib or ""], reason="nenhuma súmula correspondente na cobertura congelada", target_type="sumula")
