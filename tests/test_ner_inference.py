# -*- coding: utf-8 -*-
"""NER window boundary tests use a tiny deterministic fake tokenizer/model."""
from types import SimpleNamespace

import torch

from src.config import NER_LABELS
from src.extraction.ner_inference import extract_ner_citations
from src.extraction.ner_model import load_ner_model_and_tokenizer


class FakeTokenizer:
    cls_token_id = 101
    sep_token_id = 102

    def __call__(self, text, **_kwargs):
        return {"input_ids": [ord(char) for char in text],
                "offset_mapping": [(index, index + 1) for index in range(len(text))]}

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def build_inputs_with_special_tokens(self, body):
        return [self.cls_token_id, *body, self.sep_token_id]

    def get_special_tokens_mask(self, body, already_has_special_tokens=False):
        assert not already_has_special_tokens
        return [1, *([0] * len(body)), 1]


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(id2label={index: label for index, label in enumerate(NER_LABELS)})

    def parameters(self):
        return iter([torch.nn.Parameter(torch.zeros(1))])

    def __call__(self, input_ids, attention_mask):
        assert attention_mask.shape == input_ids.shape
        logits = torch.zeros((1, input_ids.shape[1], len(NER_LABELS)))
        for position, token_id in enumerate(input_ids[0].tolist()):
            tag = 1 if token_id == ord("a") else 2 if token_id == ord("b") else 0
            logits[0, position, tag] = 8
        return SimpleNamespace(logits=logits)


def test_ner_inference_keeps_special_tokens_out_and_joins_overlap():
    spans = extract_ner_citations("abc", FakeModel(), FakeTokenizer(), window_size=4, overlap=1)
    assert len(spans) == 1
    assert (spans[0]["inicio"], spans[0]["fim"], spans[0]["trecho"]) == (0, 2, "ab")


def test_ner_requires_an_existing_local_finetuned_checkpoint():
    try:
        load_ner_model_and_tokenizer("definitely-missing-checkpoint-for-test")
    except FileNotFoundError as exc:
        assert "Checkpoint NER local" in str(exc)
    else:
        raise AssertionError("Checkpoint ausente deve ser erro explícito")
