# -*- coding: utf-8 -*-
"""Interface de acesso e consulta ao banco SQLite congelado."""
import sqlite3
from pathlib import Path
from typing import Any


def get_connection(db_path: Path | str) -> sqlite3.Connection:
    """Abre conexão somente leitura com o banco SQLite."""
    conn = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def get_all_by_natureza(conn: sqlite3.Connection, natureza: str) -> list[dict[str, Any]]:
    """Recupera todos os registros de uma determinada natureza (ex: sumula, dispositivo)."""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT documento_id, id, tribunal, ano, relator, natureza, tipo, texto, texto_len
          FROM documentos
         WHERE natureza = ?
         ORDER BY id
        """,
        (natureza,),
    )
    return [dict(row) for row in cursor.fetchall()]


def get_all_acordaos(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Recupera metadados e os primeiros 600 caracteres de todos os acórdãos para indexação."""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT documento_id, id, tribunal, ano, relator, natureza, tipo, substr(texto, 1, 600) AS cabecalho
          FROM documentos
         WHERE natureza = 'acordao'
         ORDER BY id
        """
    )
    return [dict(row) for row in cursor.fetchall()]


def query_fts(conn: sqlite3.Connection, phrase: str, limit: int | None = None) -> list[dict[str, Any]]:
    """
    Executa busca por frase exata na tabela virtual FTS5.
    O FTS5 usa unicode61 tokenizer; frases como '"1.741.784"' reproduzem a sequência de tokens.
    """
    cursor = conn.cursor()
    clean_phrase = phrase.replace('"', '""')
    try:
        sql = """SELECT d.documento_id, d.id, d.tribunal, d.ano, d.relator, d.natureza, d.tipo,
                        substr(d.texto, 1, 5000) AS cabecalho, d.texto_len
                   FROM documentos_fts f
                   JOIN documentos d ON d.rowid = f.rowid
                  WHERE f.documentos_fts MATCH ?"""
        params: tuple[Any, ...] = (f'"{clean_phrase}"',)
        if limit is not None:
            sql += " LIMIT ?"
            params += (limit,)
        cursor.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.OperationalError as exc:
        raise RuntimeError(f"Falha na consulta FTS5 para {phrase!r}: {exc}") from exc


def get_doc_by_id(conn: sqlite3.Connection, id_canonico: int) -> dict[str, Any] | None:
    """Busca registro específico pelo id numérico Jusbrasil."""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT documento_id, id, tribunal, ano, relator, natureza, tipo, texto
          FROM documentos
         WHERE id = ?
        """,
        (id_canonico,),
    )
    row = cursor.fetchone()
    return dict(row) if row else None
