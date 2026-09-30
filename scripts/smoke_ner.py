# -*- coding: utf-8 -*-
"""Confirma checkpoint e tempo do NER sem abrir os dados do desafio."""
import argparse
import time

from src.extraction.ner_inference import extract_ner_citations
from src.extraction.ner_model import load_ner_model_and_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    import torch

    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("CUDA solicitada, mas indisponível neste contêiner.")
    started = time.perf_counter()
    model, tokenizer = load_ner_model_and_tokenizer()
    device = str(next(model.parameters()).device)
    extract_ner_citations(
        "A Súmula 211 do STJ e o art. 5º da Constituição Federal são citados.",
        model,
        tokenizer,
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        memory_mb = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    else:
        memory_mb = None
    print(f"device={device} elapsed_seconds={time.perf_counter() - started:.3f} peak_gpu_mb={memory_mb}")


if __name__ == "__main__":
    main()
