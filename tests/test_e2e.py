# -*- coding: utf-8 -*-
"""Small end-to-end contract test with synthetic text and a fake resolver."""
import csv
import io

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
            )
        return ResolutionResult.make(
            sufficient=True, method="sumula_exact", evidence=["súmula:211", "STJ"],
            reason="fixture absent", target_type="sumula",
        )


class CapturedStringIO(io.StringIO):
    def close(self):
        self.flush()


def test_synthetic_pipeline_emits_valid_submission(monkeypatch):
    text = "Aplica-se o art. 5º da CF/88. Súmula 999 do STJ."
    result = process_document("fixture_1", text, FixtureResolver())
    assert result["documento_id"] == "fixture_1"
    assert len(result["citacoes"]) == 2
    assert result["citacoes"][0]["classificacao"] == "real"
    assert result["citacoes"][0]["resolucao"]["id_canonico"] == 101
    assert all(item["confianca"] == 0.5 for item in result["citacoes"])
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
    assert rows[0]["citacoes"].endswith("-,0.5000")


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


def test_generic_vague_citation_is_kept_and_classified_incomplete():
    text = "Conforme a jurisprudência pacífica do tribunal, aplica-se a regra."
    result = process_document("fixture_vague", text, FixtureResolver())
    assert len(result["citacoes"]) == 1
    assert result["citacoes"][0]["classificacao"] == "incompleta"
    assert text[result["citacoes"][0]["inicio"]:result["citacoes"][0]["fim"]] == result["citacoes"][0]["trecho"]
