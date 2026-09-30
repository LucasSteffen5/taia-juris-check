# -*- coding: utf-8 -*-
"""Utilitários para carregar os documentos de entrada."""
from pathlib import Path


def load_document(txt_path: Path | str) -> str:
    """Lê documento em codepoints Unicode UTF-8 exatos."""
    with Path(txt_path).open("r", encoding="utf-8", newline="") as stream:
        return stream.read()


def load_all_documents(txt_dir: Path | str) -> dict[str, str]:
    """Carrega todos os arquivos .txt do diretório mapeados por documento_id."""
    docs = {}
    p = Path(txt_dir)
    for f in sorted(p.glob("*.txt")):
        docs[f.stem] = load_document(f)
    return docs
