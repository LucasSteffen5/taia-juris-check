# -*- coding: utf-8 -*-
"""Synthetic resolver tests; no challenge documents or labels are loaded."""
import sqlite3

from src.resolution.acordao_resolver import AcordaoResolver
from src.resolution.dispositivo_resolver import DispositivoResolver
from src.resolution.header_index import HeaderIndex, parse_header_tokens
from src.resolution.result import ResolutionResult
from src.resolution.sumula_resolver import SumulaResolver
from src.config import OFFICIAL_DATA_DIR, assert_output_outside_official_data


def test_output_path_guard_uses_path_containment_not_substring():
    try:
        assert_output_outside_official_data(OFFICIAL_DATA_DIR / "output.csv")
    except ValueError:
        pass
    else:
        raise AssertionError("Saídas não podem ser gravadas dentro da pasta oficial")


def _connection():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE documentos (documento_id TEXT, id INTEGER, tribunal TEXT, texto TEXT, natureza TEXT)")
    return conn


def test_sumula_needs_unique_tribunal_and_number():
    conn = _connection()
    conn.executemany(
        "INSERT INTO documentos VALUES (?, ?, ?, ?, 'sumula')",
        [("sumula_stj_1", 10, "STJ", "Súmula 1"), ("sumula_stf_1", 11, "STF", "Súmula 1")],
    )
    resolver = SumulaResolver(conn)
    ambiguous = resolver.resolve("Súmula 1")
    assert ambiguous.n_candidates == 2 and ambiguous.sufficient
    assert resolver.resolve("Súmula 1 do TSE").n_candidates == 0
    assert not resolver.resolve("Súmula sem número").sufficient
    conn.close()


def test_law_resolution_distinguishes_unknown_diploma_from_specific_absence():
    conn = _connection()
    conn.execute("INSERT INTO documentos VALUES (?, ?, NULL, ?, 'dispositivo')", ("cpc_373", 22, "Código de Processo Civil\nArt. 373"))
    resolver = DispositivoResolver(conn)
    assert resolver.resolve("art. 373, I, do CPC").canonical_id == 22
    assert not resolver.resolve("art. 10 do Código").sufficient
    absent = resolver.resolve("art. 10 da Lei nº 99.999/2020")
    assert absent.sufficient and absent.n_candidates == 0
    conn.close()


def test_header_parser_indexes_only_principal_case_and_preserves_full_cnj():
    parsed = parse_header_tokens(
        "RECURSO ESPECIAL nº 1.234.567/RS\nEMENTA: citado o RE 9.876.543/SP"
    )
    assert len(parsed) == 1
    assert parsed[0]["digits"] == "1234567"
    assert parsed[0]["uf"] == "RS"
    cnj = parse_header_tokens(
        "REspe 0600216-46.2020.6.14.0022\nEMENTA: processo 0600999-00.2022.6.14.0001"
    )
    assert len(cnj) == 1 and cnj[0]["digits"] == "06002164620206140022"

    index = HeaderIndex()
    entry = {"id": 1, "tribunal": "STJ", "uf": "RS", "class_families": ["REsp"]}
    index.exact_map["1234567"] = [entry]
    assert index.lookup("1234567", tribunal="STF") == []
    assert index.lookup("1234567", uf="SP") == []
    assert index.lookup("1234567", class_hint="Habeas Corpus") == []
    index.exact_map["11111112220205010001"] = [{"id": 2}]
    assert index.lookup("99999992220205010001") == []


def test_acordao_fts_failure_is_not_misreported_as_no_record(monkeypatch):
    import src.resolution.acordao_resolver as module

    monkeypatch.setattr(module, "query_fts", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("fts offline")))
    resolver = AcordaoResolver(sqlite3.connect(":memory:"), HeaderIndex())
    try:
        resolver.resolve("Reclamação nº 66.516/RO")
    except RuntimeError as exc:
        assert "fts offline" in str(exc)
    else:
        raise AssertionError("Falha do FTS deve ser propagada")


def test_result_classification_retains_evidence():
    from src.classification.classifier import classify_citation

    result = ResolutionResult.make(
        [99], sufficient=True, method="header_exact", evidence=["número:12345"],
        reason="identificador confirmado", target_type="acordao",
    )
    classified = classify_citation({"trecho": "REsp 12345"}, result)
    assert classified == {"classificacao": "real", "id_canonico": "99"}
    assert result.evidence == ("número:12345",)


def test_default_confidence_is_neutral_and_uncalibrated():
    from src.config import DEFAULT_CONFIDENCE

    assert DEFAULT_CONFIDENCE == 0.5


def test_header_index_rejects_qualifier_conflicts_instead_of_dropping_them():
    index = HeaderIndex()
    index.exact_map["1234567"] = [{
        "id": 1, "tribunal": "STJ", "uf": "RS", "class_families": ["REsp"],
    }]
    assert index.lookup("1234567", tribunal="STF") == []
    assert index.lookup("1234567", uf="SP") == []
    assert index.lookup("1234567", class_hint="Habeas Corpus") == []


def test_sumula_duplicate_canonical_key_remains_ambiguous():
    conn = _connection()
    conn.executemany(
        "INSERT INTO documentos VALUES (?, ?, ?, ?, 'sumula')",
        [("sumula_stj_3a", 31, "STJ", "Súmula 3"),
         ("sumula_stj_3b", 32, "STJ", "Súmula 3")],
    )
    result = SumulaResolver(conn).resolve("Súmula 3 do STJ")
    assert result.n_candidates == 2
    conn.close()


def test_duplicate_law_records_remain_ambiguous():
    conn = _connection()
    conn.executemany(
        "INSERT INTO documentos VALUES (?, ?, NULL, ?, 'dispositivo')",
        [("cpc_5a", 51, "Código de Processo Civil\nArt. 5"),
         ("cpc_5b", 52, "Código de Processo Civil\nArt. 5")],
    )
    result = DispositivoResolver(conn).resolve("art. 5 do CPC")
    assert result.n_candidates == 2
    conn.close()
