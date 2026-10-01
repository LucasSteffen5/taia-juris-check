from __future__ import annotations

import csv
import hashlib
import os
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from stage_sources import stage
from train.config import AUXILIARY_LOSS_WEIGHT, NER_LABELS, OFFICIAL_TRAIN_DOCUMENTS, WINDOW_SIZE
from train.data import (
    PreparedDocument,
    _make_ocr_variant,
    _collect_documents,
    _context_split_map,
    _confirmed_header_distractor,
    _feature_windows,
    _make_document,
    _make_windows,
    _official_selection,
    _read_text,
)


ROOT = Path(__file__).resolve().parents[2]
LOCAL = Path(os.environ.get("FT_TEST_LOCAL_DIR", ROOT / "data" / "dataset"))
MODEL = Path(os.environ.get("FT_TEST_SOURCE_MODEL", ROOT / "models" / "ner_legal_large"))


@pytest.fixture(scope="session")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)


def test_local_splits_are_disjoint_and_grouped(tmp_path: Path) -> None:
    annotation_path = LOCAL / "goldenset_offsets.csv"
    with annotation_path.open(encoding="utf-8-sig", newline="") as stream:
        annotated_ids = {row["documento_id"] for row in csv.DictReader(stream)}
    assignments, groups = _context_split_map(LOCAL / "txt", annotation_path)
    assert Counter(assignments.values()) == {"train": 412, "validation": 62, "test": 56}
    group_splits: dict[str, set[str]] = defaultdict(set)
    for document_id, group in groups.items():
        group_splits[group].add(assignments[document_id])
    assert all(len(splits) == 1 for splits in group_splits.values())
    assert len([group for group in group_splits if group.startswith("synthetic_")]) == 65
    negative_counts = Counter(
        assignments[document_id]
        for document_id in assignments
        if document_id.startswith("syn_") and document_id not in annotated_ids
    )
    assert negative_counts == {"train": 6, "validation": 6, "test": 6}


def test_only_authorized_official_examples_are_staged_and_read(tmp_path: Path) -> None:
    assert _official_selection() == tuple(sorted(OFFICIAL_TRAIN_DOCUMENTS))
    raw_text = tmp_path / "official_input" / "txt"
    raw_text.mkdir(parents=True)
    annotation_path = tmp_path / "official_input" / "goldenset_offsets.csv"
    fields = ["nivel", "documento_id", "citacao_id", "inicio", "fim", "trecho", "tipo", "classificacao", "id_canonico"]
    rows = []
    for index, document_id in enumerate(OFFICIAL_TRAIN_DOCUMENTS):
        text = f"Documento de teste {index}: Súmula {index + 1}.\n"
        raw_text.joinpath(f"{document_id}.txt").write_text(text, encoding="utf-8", newline="\n")
        surface = f"Súmula {index + 1}"
        start = text.index(surface)
        rows.append({
            "nivel": "N1" if "_n1_" in document_id else "N2",
            "documento_id": document_id,
            "citacao_id": f"annotation-{index}",
            "inicio": str(start),
            "fim": str(start + len(surface)),
            "trecho": surface,
            "tipo": "jurisprudencia",
            "classificacao": "real",
            "id_canonico": "test-only",
        })
    # Ensure the staging filter rejects an unauthorized official record too.
    raw_text.joinpath("gen_n1_003.txt").write_text("Não autorizado.\n", encoding="utf-8", newline="\n")
    rows.append({**rows[0], "documento_id": "gen_n1_003", "citacao_id": "unauthorized"})
    with annotation_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manifest = stage(
        "selection-test",
        raw_text,
        annotation_path,
        tmp_path / "runs",
    )
    stage_root = tmp_path / "runs" / "selection-test" / "sources" / "official"
    assert manifest["selected_documents"] == sorted(OFFICIAL_TRAIN_DOCUMENTS)
    assert len(list((stage_root / "txt").glob("*.txt"))) == 10
    with (stage_root / "goldenset_offsets.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["documento_id"] for row in rows} == set(OFFICIAL_TRAIN_DOCUMENTS)
    _, audit = _collect_documents(
        LOCAL / "txt",
        LOCAL / "goldenset_offsets.csv",
        stage_root / "txt",
        stage_root / "goldenset_offsets.csv",
    )
    assert audit["local_documents"] == {"train": 412, "validation": 62, "test": 56}
    assert audit["official_documents_authorized_for_training"] == sorted(OFFICIAL_TRAIN_DOCUMENTS)
    assert audit["official_documents_reserved_for_requested_final_score"] == 16
    assert audit["official_staging_manifest_sha256"]
    assert audit["local_documents_without_source_annotations_by_split"] == {
        "train": 6, "validation": 6, "test": 6,
    }


def test_lf_normalization_preserves_unicode_offsets(tmp_path: Path) -> None:
    path = tmp_path / "unicode.txt"
    path.write_bytes("AÇÃO\r\nSúmula 123\r\n".encode("utf-8"))
    normalized, raw_digest = _read_text(path)
    start = normalized.index("Súmula")
    end = start + len("Súmula 123")
    assert normalized == "AÇÃO\nSúmula 123\n"
    assert normalized[start:end] == "Súmula 123"
    assert raw_digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_ocr_augmentation_is_seeded_length_preserving_and_auditable() -> None:
    text = ("Súmula 123 do STF; processo 0001. O relatório cita legislação. " * 30)
    start = text.index("Súmula 123")
    document = PreparedDocument(
        document_id="lener_train_n1_case001", source="local", split="train", level=1,
        text=text, annotations=[{"start": start, "end": start + len("Súmula 123"), "kind": "JURISPRUDENCIA", "annotation_id": "a1"}],
        ignored_regions=[], raw_file_sha256="source-hash", negative_regions=[],
    )
    first, repairs = _make_ocr_variant(document, 0.03, 42)
    second, repairs_again = _make_ocr_variant(document, 0.03, 42)
    assert first.text == second.text
    assert repairs == repairs_again and repairs
    assert len(first.text) == len(text)
    assert first.text[start:start + len("Súmula 123")] != ""
    assert first.base_document_id == document.document_id
    assert first.augmentation_rate == 0.03


def test_header_distractor_filter_is_conservative() -> None:
    text = "PROCESSO: 0001234-55.2024.8.26.0001\nEMENTA\nO STF decidiu no processo citado."
    boundary = text.index("\nEMENTA")
    own_case = text.index("0001234")
    body_stf = text.index("STF")
    assert _confirmed_header_distractor(text, own_case, own_case + len("0001234"), boundary)
    assert _confirmed_header_distractor(text, body_stf, body_stf + 3, boundary) is None
    no_marker = "Súmula 123 do STF deve permanecer positiva."
    assert _confirmed_header_distractor(no_marker, 0, 6, 0) is None


def test_long_unicode_document_uses_overlapping_complete_windows(tokenizer) -> None:
    phrase = "Súmula 123 do STJ; Constituição Federal. "
    text = phrase * 180
    # Put an annotated span in the overlap between the first two windows.
    encoding = tokenizer(text, add_special_tokens=False, truncation=False, return_offsets_mapping=True)
    special_count = tokenizer.num_special_tokens_to_add(pair=False)
    payload = WINDOW_SIZE - special_count
    stride = payload - 128
    token_start = stride + 20
    token_end = token_start + 5
    start = encoding["offset_mapping"][token_start][0]
    end = encoding["offset_mapping"][token_end - 1][1]
    annotation_text = text[start:end]
    document = PreparedDocument(
        "local_n1_test", "local", "train", 1, text,
        [{"start": start, "end": end, "kind": "LEGISLACAO", "annotation_id": "u1"}],
        [], "raw",
    )
    windows, labels, offsets = _make_windows(document, tokenizer)
    assert len(windows) >= 3
    assert max(len(window["input_ids"]) for window in windows) <= WINDOW_SIZE
    assert len(offsets) == len(encoding["input_ids"])
    assert text[start:end] == annotation_text
    assert labels[token_start] == NER_LABELS.index("B-LEGISLACAO")
    assert all(label != -100 for label in labels[token_start:token_end])
    appearances = 0
    for window in windows:
        body_start, body_end = window["token_start"], window["token_end"]
        if body_start <= token_start < body_end:
            positions = window["body_positions"]
            local_index = token_start - body_start
            assert window["labels"][positions[local_index]] == NER_LABELS.index("B-LEGISLACAO")
            appearances += 1
    assert appearances == 2


def test_ambiguous_spans_are_ignored_and_no_citation_docs_are_negative(tokenizer) -> None:
    text = "A Súmula 123 do STJ foi citada."
    start = text.index("Súmula")
    rows = [
        {"inicio": str(start), "fim": str(start + 10), "trecho": text[start:start + 10], "citacao_id": "a", "tipo": "jurisprudencia"},
        {"inicio": str(start + 4), "fim": str(start + 12), "trecho": text[start + 4:start + 12], "citacao_id": "b", "tipo": "legislacao"},
    ]
    doc = _make_document("x", "local", "train", text, "hash", rows, Counter())
    assert not doc.annotations
    assert doc.ignored_regions

    no_citation = PreparedDocument("empty_n1_doc", "local", "train", 1, "Sem citação neste documento.", [], [], "hash")
    windows, labels, _ = _make_windows(no_citation, tokenizer)
    assert windows and set(labels) == {0}


def test_teacher_pseudolabels_are_confident_weighted_and_auditable(tokenizer) -> None:
    import torch
    from types import SimpleNamespace

    class Teacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, attention_mask, token_type_ids):
            logits = torch.zeros((*input_ids.shape, len(NER_LABELS)), device=input_ids.device)
            logits[..., NER_LABELS.index("B-PESSOA")] = 8.0
            return SimpleNamespace(logits=logits)

    text = "A Súmula 123 do STJ foi aplicada."
    start = text.index("Súmula")
    end = start + len("Súmula 123 do STJ")
    document = PreparedDocument(
        "synthetic_n1_train", "local", "train", 1, text,
        [{"start": start, "end": end, "kind": "JURISPRUDENCIA", "annotation_id": "gold"}],
        [], "source-hash",
    )
    windows = _feature_windows(document, tokenizer, Teacher(), torch)
    pseudo_tokens = [
        (label, weight, confidence)
        for window in windows
        for label, weight, confidence in zip(
            window["auxiliary_labels"], window["auxiliary_weights"], window["pseudo_confidences"]
        )
        if label != -100
    ]
    assert pseudo_tokens
    assert all(label == NER_LABELS.index("B-PESSOA") for label, _weight, _confidence in pseudo_tokens)
    assert all(confidence >= 0.90 for _label, _weight, confidence in pseudo_tokens)
    assert all(weight == 1.0 for _label, weight, _confidence in pseudo_tokens)
    manual = [
        (label, weight)
        for window in windows
        for label, weight in zip(window["labels"], window["loss_weights"])
        if label in {NER_LABELS.index("B-JURISPRUDENCIA"), NER_LABELS.index("I-JURISPRUDENCIA")}
    ]
    assert manual and all(weight == 1.0 for _label, weight in manual)
