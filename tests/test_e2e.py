# -*- coding: utf-8 -*-
"""Small end-to-end contract test with synthetic text and a fake resolver."""
import csv
import io
import json
import sqlite3
from types import SimpleNamespace

import src.pipeline as pipeline
from src.pipeline import process_document
from src.resolution.result import ResolutionResult
from src.output.submission_builder import generate_submission_csv
from src.output.json_builder import validate_document_json


class FixtureResolver:
    def resolve(self, citation):
        if citation["tipo"] == "lei":
            return ResolutionResult.make(
            [101], sufficient=True, method="law_exact", evidence=["art.5", "CF"],
                reason="fixture match", target_type="dispositivo",
                identifier_strength="muito_forte",
            )
        return ResolutionResult.make(
            sufficient=True, method="sumula_exact", evidence=["súmula:211", "STJ"],
            reason="fixture absent", target_type="sumula",
            identifier_strength="muito_forte",
        )


class CapturedStringIO(io.StringIO):
    def close(self):
        self.flush()


def test_synthetic_pipeline_emits_valid_submission(monkeypatch):
    text = "Aplica-se o art. 5º da CF/88. Súmula 999 do STJ."
    ner_calls = []
    monkeypatch.setattr(
        pipeline,
        "extract_ner_citations",
        lambda value, model, tokenizer: (ner_calls.append((value, model, tokenizer)) or []),
    )
    fake_model, fake_tokenizer = object(), object()
    result = process_document("fixture_1", text, FixtureResolver(), fake_model, fake_tokenizer)
    assert result["documento_id"] == "fixture_1"
    assert len(result["citacoes"]) == 2
    assert result["citacoes"][0]["classificacao"] == "real"
    assert result["citacoes"][0]["resolucao"]["id_canonico"] == 101
    assert [item["confianca"] for item in result["citacoes"]] == [0.80, 0.75]
    assert ner_calls == [(text, fake_model, fake_tokenizer)]
    assert all(text[item["inicio"]:item["fim"]] == item["trecho"] for item in result["citacoes"])

    empty = {"documento_id": "fixture_empty", "citacoes": []}
    buffer = CapturedStringIO(newline="")
    import src.output.submission_builder as builder
    monkeypatch.setattr(builder.Path, "mkdir", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(builder, "open", lambda *_args, **_kwargs: buffer, raising=False)
    generate_submission_csv({"fixture_1": result, "fixture_empty": empty}, "virtual/submission.csv")
    buffer.seek(0)
    rows = list(csv.DictReader(buffer))
    assert len(rows) == 2
    assert rows[0]["citacoes"].count("|") == 1
    assert rows[1]["citacoes"] == "-"
    assert rows[0]["citacoes"].split("|")[0].endswith("101,0.8000")
    assert rows[0]["citacoes"].split("|")[1].endswith("-,0.7500")


def test_json_validator_rejects_offsets_that_do_not_slice_the_source():
    payload = {
        "documento_id": "fixture",
        "citacoes": [{
            "inicio": 1, "fim": 3, "trecho": "xy", "tipo": "lei",
            "classificacao": "incompleta", "resolucao": None, "confianca": None,
        }],
    }
    try:
        validate_document_json(payload, "abc")
    except ValueError as exc:
        assert "Trecho diverge" in str(exc)
    else:
        raise AssertionError("Validador deve rejeitar span que não corresponde ao texto")


def test_generic_vague_citation_is_kept_and_classified_incomplete(monkeypatch):
    text = "Conforme a jurisprudência pacífica do tribunal, aplica-se a regra."
    monkeypatch.setattr(pipeline, "extract_ner_citations", lambda *_args: [])
    result = process_document("fixture_vague", text, FixtureResolver(), object(), object())
    assert len(result["citacoes"]) == 1
    assert result["citacoes"][0]["classificacao"] == "incompleta"
    assert text[result["citacoes"][0]["inicio"]:result["citacoes"][0]["fim"]] == result["citacoes"][0]["trecho"]


def test_pipeline_persists_outputs_and_rebuilds_old_index(tmp_path, monkeypatch):
    input_dir = tmp_path / "txt"
    input_dir.mkdir()
    (input_dir / "fixture.txt").write_text("Súmula 999 do STJ.", encoding="utf-8")

    database = tmp_path / "reference.sqlite"
    with sqlite3.connect(database) as conn:
        conn.execute(
            "CREATE TABLE documentos (documento_id TEXT, id INTEGER, tribunal TEXT, "
            "texto TEXT, natureza TEXT, tipo TEXT)"
        )

    output = tmp_path / "persisted"
    cache_dir = output / ".cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "header_index.json").write_text(json.dumps({
        "index_version": 2,
        "parser_version": pipeline.PARSER_VERSION,
        "database_sha256": "old-database",
        "exact_map": {},
        "entries_by_id": {},
    }), encoding="utf-8")

    fake_model = SimpleNamespace(parameters=lambda: iter([SimpleNamespace(device="cpu")]))
    monkeypatch.setattr(pipeline, "load_ner_model_and_tokenizer", lambda: (fake_model, object()))
    monkeypatch.setattr(pipeline, "extract_ner_citations", lambda *_args: [])

    submission = pipeline.run_pipeline(input_dir, database, output, "submission custom.csv")

    assert submission.is_file()
    assert submission.name == "submission custom.csv"
    assert not (output / "submission.csv").exists()
    assert (output / "json" / "fixture.json").is_file()
    assert (output / "json_original" / "fixture.json").is_file()
    assert (output / "text_lf" / "fixture.txt").read_text(encoding="utf-8") == "Súmula 999 do STJ."
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["documents_processed"] == 1
    assert manifest["offset_policy"] == "LF"
    assert manifest["submission_offset_space"] == "LF"
    assert manifest["input_documents_lf_sha256"] == manifest["input_documents_original_sha256"]
    assert manifest["device"] == "cpu"
    assert manifest["confidence_policy_version"] == "evidence-rules-v2"
    assert "heurísticos" in manifest["confidence_policy"]
    assert manifest["submission_sha256"] == pipeline.HeaderIndex.fingerprint(submission)
    rebuilt = json.loads((cache_dir / "header_index.json").read_text(encoding="utf-8"))
    assert rebuilt["index_version"] == 3
    assert "entries_by_id" not in rebuilt


def test_pipeline_rejects_output_filename_paths_before_processing(tmp_path):
    try:
        pipeline.run_pipeline(tmp_path, tmp_path / "missing.db", tmp_path / "output", "../escape.csv")
    except ValueError as exc:
        assert "apenas um nome de arquivo" in str(exc)
    else:
        raise AssertionError("--output-file não pode escapar do diretório de saída")
