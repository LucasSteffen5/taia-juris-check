# -*- coding: utf-8 -*-
"""Resolvedor unificado integrando acórdãos, súmulas e dispositivos legais."""
import sqlite3
from typing import Any
from .header_index import HeaderIndex
from .sumula_resolver import SumulaResolver
from .dispositivo_resolver import DispositivoResolver
from .acordao_resolver import AcordaoResolver
from .result import ResolutionResult


class UnifiedResolver:
    """Resolvedor de citações jurídicas contra a base canônica congelada."""

    def __init__(self, conn: sqlite3.Connection, header_index: HeaderIndex):
        self.sumula_res = SumulaResolver(conn)
        self.disp_res = DispositivoResolver(conn)
        self.acordao_res = AcordaoResolver(conn, header_index)

    def resolve(self, citation: dict[str, Any]) -> ResolutionResult:
        """
        Resolve citação contra o acervo canônico congelado.
        Retorna ResolutionResult contendo:
          - candidate_ids: tupla com os IDs numéricos encontrados
          - canonical_id: ID canônico se resultado for unívoco (len == 1)
          - sufficient: booleano indicando suficiência dos dados de entrada
          - method: método de resolução utilizado
          - evidence: tupla de evidências textuais/chaves
          - reason: justificativa detalhada da resolução
          - target_type: 'sumula', 'dispositivo', 'acordao', 'vague'
        """
        trecho = citation.get("trecho", "")
        tipo = citation.get("tipo", "jurisprudencia").lower()
        is_vague = citation.get("is_vague", False)

        if is_vague:
            return ResolutionResult.make(sufficient=False, method="vague_rule", evidence=[trecho], reason="referência vaga sem identificador pesquisável", target_type="vague")

        if tipo == "lei":
            return self.disp_res.resolve(trecho)
        else:
            # Jurisprudência: verificar se é súmula
            t_low = trecho.lower()
            if "súmula" in t_low or "sumula" in t_low or "5úmula" in t_low or "súm." in t_low:
                return self.sumula_res.resolve(trecho)
            else:
                return self.acordao_res.resolve(trecho)
