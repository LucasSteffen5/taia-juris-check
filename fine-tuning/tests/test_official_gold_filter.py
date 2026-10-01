from __future__ import annotations

import csv

import pytest

from official_gold import filter_official_gold, stage_official_text_lf, validate_official_offsets


def test_official_text_staging_converts_crlf_and_preserves_unicode_offsets(tmp_path):
    source = tmp_path / "official.txt"
    raw = "\ufeffIntrodução\r\nCitação: Súmula 7\r\n".encode("utf-8")
    source.write_bytes(raw)
    staged = tmp_path / "prepared" / "official.txt"
    staged.parent.mkdir()
    staged.write_bytes(raw)  # A copy from the earlier CRLF evaluator can be repaired.

    metadata = stage_official_text_lf(source, staged)
    text = staged.read_bytes().decode("utf-8")
    start = text.index("Súmula 7")
    assert text[start:start + len("Súmula 7")] == "Súmula 7"
    assert staged.read_bytes() == "Introdução\nCitação: Súmula 7\n".encode("utf-8")
    assert source.read_bytes() == raw
    assert metadata["source_crlf_count"] == 2
    assert metadata["source_has_bom"] is True
    assert metadata["source_sha256"] != metadata["staged_sha256"]
    assert stage_official_text_lf(source, staged) == metadata

    (staged.parent / "negative.txt").write_text("Sem citações.\n", encoding="utf-8")
    gold = tmp_path / "heldout.csv"
    gold.write_text(
        f"documento_id,inicio,fim,trecho\n"
        f"official,{start},{start + len('Súmula 7')},Súmula 7\n",
        encoding="utf-8",
    )
    audit = validate_official_offsets(gold, staged.parent, {"official", "negative"})
    assert audit["validated_annotation_rows"] == 1
    assert audit["documents_without_annotations"] == ["negative"]


def test_official_text_staging_rejects_unrelated_existing_copy(tmp_path):
    source = tmp_path / "official.txt"
    source.write_bytes(b"source\r\n")
    staged = tmp_path / "staged.txt"
    staged.write_bytes(b"different\n")

    with pytest.raises(ValueError, match="diverge da fonte"):
        stage_official_text_lf(source, staged)


def test_official_offset_gate_rejects_crlf_staged_text(tmp_path):
    (tmp_path / "official.txt").write_bytes(b"prefix\r\nCitation")
    gold = tmp_path / "gold.csv"
    gold.write_text("documento_id,inicio,fim,trecho\nofficial,8,16,Citation\n", encoding="utf-8")

    with pytest.raises(ValueError, match="ainda contém CR"):
        validate_official_offsets(gold, tmp_path, {"official"})


def test_filter_keeps_only_heldout_annotations_and_allows_negative_documents(tmp_path):
    source = tmp_path / "official.csv"
    source.write_text(
        "nivel,documento_id,inicio,fim,tipo\n"
        "1,train_a,1,4,jurisprudencia\n"
        "1,test_a,5,9,legislacao\n"
        "2,test_a,10,12,jurisprudencia\n",
        encoding="utf-8",
        newline="",
    )
    output = tmp_path / "filtered" / "gold.csv"
    all_ids = {"train_a", "test_a", "test_negative"}
    expected = {"test_a", "test_negative"}

    metadata = filter_official_gold(source, output, all_ids, expected)
    with output.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert [row["documento_id"] for row in rows] == ["test_a", "test_a"]
    assert metadata["included_row_count"] == 2
    assert metadata["included_annotation_document_ids"] == ["test_a"]
    assert metadata["expected_documents_without_annotation_rows"] == ["test_negative"]
    assert metadata["filtered_sha256"]

    first_bytes = output.read_bytes()
    repeated = filter_official_gold(source, output, all_ids, expected)
    assert output.read_bytes() == first_bytes
    assert repeated == metadata


def test_filter_rejects_gold_ids_missing_from_official_document_inventory(tmp_path):
    source = tmp_path / "official.csv"
    source.write_text(
        "documento_id,inicio,fim\nunknown,1,2\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="ausente do conjunto oficial"):
        filter_official_gold(source, tmp_path / "filtered.csv", {"test_a"}, {"test_a"})


def test_filter_rejects_expected_ids_outside_official_inventory(tmp_path):
    source = tmp_path / "official.csv"
    source.write_text("documento_id\n", encoding="utf-8")

    with pytest.raises(ValueError, match="subconjunto esperado"):
        filter_official_gold(source, tmp_path / "filtered.csv", {"known"}, {"unknown"})
