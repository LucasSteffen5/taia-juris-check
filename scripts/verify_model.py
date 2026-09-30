# -*- coding: utf-8 -*-
"""Confere os pesos e o tokenizer contra a procedência incluída no repositório."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models" / "ner_legal_large"
MANIFEST = MODEL_DIR / "provenance.json"
SOURCE_MODEL = "dominguesm/legal-bert-ner-base-cased-ptbr"
SOURCE_REVISION = "44210927c925448df025985e0ed48081bb5ac57c"
SOURCE_WEIGHTS_SHA256 = "dea15cbbe0935ec20da635f6861ffdcad0d737a648ad48dff1affd686cf776a6"


def main() -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
    if (data.get("source_model"), data.get("revision"), data.get("license")) != (
        SOURCE_MODEL, SOURCE_REVISION, "CC-BY-4.0"
    ):
        raise SystemExit("Origem, revisão ou licença do checkpoint não confere.")
    if data["files"].get("model.safetensors") != SOURCE_WEIGHTS_SHA256:
        raise SystemExit("O hash dos pesos diverge da revisão publicada do modelo.")
    for relative, expected in data["files"].items():
        path = MODEL_DIR / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise SystemExit(f"Hash do checkpoint não confere: {relative}")
    print(f"Checkpoint íntegro: {len(data['files'])} arquivos; licença {data['license']}.")


if __name__ == "__main__":
    main()
