# -*- coding: utf-8 -*-
"""Interface de acesso e consulta ao banco SQLite congelado."""
import sqlite3
from pathlib import Path


def get_connection(db_path: Path | str) -> sqlite3.Connection:
    """Abre conexão somente leitura com o banco SQLite."""
    conn = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def query_fts(conn: sqlite3.Connection, phrase: str) -> list[dict[str, object]]:
    """
    Executa busca por frase exata na tabela virtual FTS5.
    O FTS5 usa unicode61 tokenizer; frases como '"1.741.784"' reproduzem a sequência de tokens.
    """
    cursor = conn.cursor()
    clean_phrase = phrase.replace('"', '""')
    try:
        sql = """SELECT d.id, d.tribunal, d.natureza, substr(d.texto, 1, 5000) AS cabecalho
                   FROM documentos_fts f
                   JOIN documentos d ON d.rowid = f.rowid
                  WHERE f.documentos_fts MATCH ?"""
        cursor.execute(sql, (f'"{clean_phrase}"',))
        return [dict(row) for row in cursor.fetchall()]
    except sqlite3.OperationalError as exc:
        raise RuntimeError(f"Falha na consulta FTS5 para {phrase!r}: {exc}") from exc
