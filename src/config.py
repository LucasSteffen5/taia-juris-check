# -*- coding: utf-8 -*-
"""Parâmetros do pipeline e proteção de dados no contêiner."""
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OFFICIAL_DATA_DIR = Path(
    os.environ.get("PROTECTED_DATA_DIR", ROOT_DIR / "data" / "desafio-jusbrasil-bracis-2026")
)
NER_MODEL_DIR = ROOT_DIR / "models" / "ner_legal_large"

IOU_MIN = 0.5
DEFAULT_CONFIDENCE = 0.5
WINDOW_SIZE = 510
WINDOW_OVERLAP = 128
NER_LABELS = ["O", "B-JUR", "I-JUR", "B-LEI", "I-LEI"]
NER_THRESHOLD = 0.65


def assert_output_outside_official_data(path: Path | str) -> None:
    candidate = Path(path).resolve()
    protected = OFFICIAL_DATA_DIR.resolve()
    if candidate == protected or protected in candidate.parents:
        raise ValueError(
            f"Dados oficiais só podem ser usados para teste final; caminho de saída bloqueado ({candidate})."
        )
