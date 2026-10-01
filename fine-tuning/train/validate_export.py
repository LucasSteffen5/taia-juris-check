"""Offline integrity/load check for an exported 13-label checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCE_MODEL = "dominguesm/legal-bert-ner-base-cased-ptbr"
SOURCE_REVISION = "44210927c925448df025985e0ed48081bb5ac57c"
SOURCE_LICENSE = "CC-BY-4.0"
LABELS = (
    "O", "B-ORGANIZACAO", "I-ORGANIZACAO", "B-PESSOA", "I-PESSOA",
    "B-TEMPO", "I-TEMPO", "B-LOCAL", "I-LOCAL", "B-LEGISLACAO",
    "I-LEGISLACAO", "B-JURISPRUDENCIA", "I-JURISPRUDENCIA",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(checkpoint_dir: Path, smoke_inference: bool = True) -> dict:
    manifest_path = checkpoint_dir / "provenance.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"provenance.json ausente em {checkpoint_dir}.")
    provenance = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        provenance.get("source_model"),
        provenance.get("revision"),
        provenance.get("license"),
    ) != (SOURCE_MODEL, SOURCE_REVISION, SOURCE_LICENSE):
        raise ValueError("Procedência, revisão ou licença do modelo derivado é incompatível.")
    hashes = provenance.get("files")
    if not isinstance(hashes, dict) or "model.safetensors" not in hashes:
        raise ValueError("Manifesto sem hash dos pesos safetensors.")
    for relative, expected in hashes.items():
        path = checkpoint_dir / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Hash incorreto no artefato derivado: {relative}.")

    import torch
    from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(
        str(checkpoint_dir), local_files_only=True, use_safetensors=True
    )
    config = AutoConfig.from_pretrained(str(checkpoint_dir), local_files_only=True)
    expected_labels = {label: index for index, label in enumerate(LABELS)}
    actual_labels = {str(label): int(index) for label, index in config.label2id.items()}
    if actual_labels != expected_labels or config.num_labels != len(LABELS):
        raise ValueError(f"Mapa de rótulos derivado divergente: {actual_labels}.")
    if config.base_model_revision != provenance.get("revision"):
        raise ValueError("Revisão-base no config.json difere da procedência exportada.")
    if tokenizer.pad_token_id is None or not tokenizer.is_fast:
        raise ValueError("O tokenizer exportado precisa ser rápido e definir pad_token_id.")

    if smoke_inference:
        model.eval()
        sample = tokenizer(
            "Súmula 123 do STJ e artigo 5º da Constituição Federal.",
            return_tensors="pt",
            truncation=False,
        )
        with torch.inference_mode():
            logits = model(**sample).logits
        if tuple(logits.shape) != (1, sample["input_ids"].shape[1], len(LABELS)):
            raise ValueError(f"Shape de inferência incompatível: {tuple(logits.shape)}.")
    return {
        "valid": True,
        "checkpoint_dir": str(checkpoint_dir),
        "model_sha256": _sha256(checkpoint_dir / "model.safetensors"),
        "num_labels": config.num_labels,
        "tokenizer_vocab_size": len(tokenizer),
        "offline_load_and_smoke_inference": smoke_inference,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument("--no-smoke-inference", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = validate(args.checkpoint_dir, smoke_inference=not args.no_smoke_inference)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
