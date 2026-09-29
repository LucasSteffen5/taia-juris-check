# -*- coding: utf-8 -*-
"""Caminhos e parâmetros únicos da execução."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OFFICIAL_DATA_DIR = ROOT_DIR / "data" / "desafio-jusbrasil-bracis-2026"
OFFICIAL_DB_PATH = OFFICIAL_DATA_DIR / "desafio1_bracis.db"
OFFICIAL_GOLDEN_PATH = OFFICIAL_DATA_DIR / "goldenset_offsets.csv"
OFFICIAL_TXT_DIR = OFFICIAL_DATA_DIR / "txt"
MODELS_DIR = ROOT_DIR / "models"
NER_MODEL_DIR = MODELS_DIR / "ner_legal_large"
HEADER_INDEX_PATH = ROOT_DIR / "output" / ".cache" / "header_index.json"

IOU_MIN = 0.5
CLASSES = ("real", "inventada", "incompleta")
WINDOW_SIZE = 510
WINDOW_OVERLAP = 128
NER_LABELS = ["O", "B-JUR", "I-JUR", "B-LEI", "I-LEI"]
NER_THRESHOLD = 0.65


def assert_not_training_on_test(path: Path | str) -> None:
    assert_path_outside_official_data(path, operation="treinamento")


def assert_output_outside_official_data(path: Path | str) -> None:
    assert_path_outside_official_data(path, operation="saída")


def assert_path_outside_official_data(path: Path | str, *, operation: str) -> None:
    candidate = Path(path).resolve()
    protected = OFFICIAL_DATA_DIR.resolve()
    if candidate == protected or protected in candidate.parents:
        raise ValueError(
            f"Dados oficiais só podem ser usados para teste final; caminho de {operation} bloqueado ({candidate})."
        )
