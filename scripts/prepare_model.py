# -*- coding: utf-8 -*-
"""Garante que os pesos NER estejam disponíveis antes de validar a imagem."""
import hashlib
import os
import shutil
import tempfile
from pathlib import Path

from verify_model import SOURCE_MODEL, SOURCE_REVISION, SOURCE_WEIGHTS_SHA256


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "ner_legal_large"
MODEL_FILE = MODEL_DIR / "model.safetensors"
PROVENANCE = MODEL_DIR / "provenance.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    import json

    metadata = json.loads(PROVENANCE.read_text(encoding="utf-8-sig"))
    expected_identity = (SOURCE_MODEL, SOURCE_REVISION, "CC-BY-4.0")
    actual_identity = (
        metadata.get("source_model"),
        metadata.get("revision"),
        metadata.get("license"),
    )
    if actual_identity != expected_identity:
        raise SystemExit("Origem, revisão ou licença do checkpoint não confere.")
    if metadata.get("files", {}).get("model.safetensors") != SOURCE_WEIGHTS_SHA256:
        raise SystemExit("O hash esperado dos pesos não confere com a revisão fixada.")

    if MODEL_FILE.is_file() and sha256(MODEL_FILE) == SOURCE_WEIGHTS_SHA256:
        print("Pesos NER já disponíveis localmente.")
        return

    print("Baixando os pesos NER da revisão fixada do Hugging Face...")
    # O download ocorre apenas durante o build. O contêiner final continua offline.
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    from huggingface_hub import hf_hub_download

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="juris-check-model-") as cache_dir:
        downloaded = Path(
            hf_hub_download(
                repo_id=SOURCE_MODEL,
                filename="model.safetensors",
                revision=SOURCE_REVISION,
                cache_dir=cache_dir,
            )
        )
        if sha256(downloaded) != SOURCE_WEIGHTS_SHA256:
            raise SystemExit("Os pesos baixados não correspondem ao hash fixado.")

        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix="model.safetensors.", suffix=".tmp", dir=MODEL_DIR
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        try:
            shutil.copyfile(downloaded, temporary_path)
            os.replace(temporary_path, MODEL_FILE)
        finally:
            temporary_path.unlink(missing_ok=True)
    print("Pesos NER baixados e hash conferido.")


if __name__ == "__main__":
    main()
