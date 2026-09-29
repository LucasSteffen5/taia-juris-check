# -*- coding: utf-8 -*-
"""Load an explicitly trained, locally available NER checkpoint."""
from pathlib import Path
import json
import re
from typing import Any

from src.config import NER_MODEL_DIR, NER_LABELS


def load_ner_model_and_tokenizer(model_path: Path | str | None = None) -> tuple[Any, Any]:
    path = Path(model_path) if model_path else NER_MODEL_DIR
    if not path.is_dir() or not any(path.iterdir()):
        raise FileNotFoundError(
            f"Checkpoint NER local não encontrado em {path}."
        )

    provenance = json.loads((path / "provenance.json").read_text(encoding="utf-8-sig"))
    if (
        provenance.get("source_model") != "dominguesm/legal-bert-ner-base-cased-ptbr"
        or provenance.get("license") != "CC-BY-4.0"
    ):
        raise ValueError("Procedência ou licença do checkpoint NER incompatível.")

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(str(path), local_files_only=True)
    expected = {label: index for index, label in enumerate(NER_LABELS)}
    is_5_label = (model.config.label2id == expected)
    is_legal_schema = bool(model.config.label2id and any("JUR" in str(k) for k in model.config.label2id.keys()))
    if not (is_5_label or is_legal_schema):
        raise ValueError(
            f"Mapa de rótulos do checkpoint incompatível: esperado {expected} ou compatível com entidades jurídicas, "
            f"recebido {model.config.label2id}."
        )
    if is_5_label and model.config.num_labels != len(NER_LABELS):
        raise ValueError("Checkpoint NER tem quantidade incorreta de rótulos.")
    if not re.fullmatch(r"[0-9a-f]{40}", getattr(model.config, "base_model_revision", "")):
        raise ValueError("Checkpoint não registra SHA fixo do modelo-base.")
    if getattr(model.config, "base_model_revision", None) != provenance.get("revision"):
        raise ValueError("Checkpoint não corresponde à revisão registrada.")
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer do checkpoint não define pad_token_id.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    if device == "cuda":
        model.half()
    return model, tokenizer
