"""LF normalization and source-offset mapping tests use only synthetic text."""
import src.pipeline as pipeline
from src.resolution.result import ResolutionResult
from src.utils_data import map_span_to_original, normalize_line_endings


class FixtureResolver:
    def resolve(self, citation):
        if citation["tipo"] == "lei":
            return ResolutionResult.make(
                [101], sufficient=True, method="law_exact", evidence=["art.5", "CF"],
                reason="fixture match", target_type="dispositivo",
                identifier_strength="muito_forte",
            )
        return ResolutionResult.make(
            sufficient=True, method="sumula_exact", evidence=["sumula:211", "STJ"],
            reason="fixture absent", target_type="sumula",
            identifier_strength="muito_forte",
        )


def test_normalize_line_endings_preserves_lf_and_maps_all_newline_forms():
    lf = "início\nmeio\nfim"
    variants = [
        lf,
        lf.replace("\n", "\r\n"),
        lf.replace("\n", "\r"),
        "início\r\nmeio\nfim",
    ]

    for original in variants:
        normalized, boundaries, counts = normalize_line_endings(original)
        assert normalized == lf
        assert boundaries.normalized_length == len(lf)
        assert boundaries.to_original(0) == 0
        assert boundaries.to_original(len(lf)) == len(original)
        assert counts == {
            "crlf": original.count("\r\n"),
            "lf": original.count("\n") - original.count("\r\n"),
            "cr": original.count("\r") - original.count("\r\n"),
        }

        start, end = 6, 10
        original_start, original_end = map_span_to_original(start, end, boundaries)
        assert normalize_line_endings(original[original_start:original_end])[0] == lf[start:end]


def test_process_document_is_identical_for_lf_crlf_cr_and_mixed_inputs(monkeypatch):
    lf = "😀 Súmula 211 do STJ.\nart. 5º da CF/88.\nFim"
    variants = [
        lf,
        lf.replace("\n", "\r\n"),
        lf.replace("\n", "\r"),
        "😀 Súmula 211 do STJ.\r\nart. 5º da CF/88.\nFim",
    ]
    monkeypatch.setattr(pipeline, "extract_ner_citations", lambda *_args: [])
    resolver = FixtureResolver()

    results = [
        pipeline.process_document("fixture", original, resolver, object(), object())
        for original in variants
    ]
    assert all(result == results[0] for result in results)
    for citation in results[0]["citacoes"]:
        assert lf[citation["inicio"]:citation["fim"]] == citation["trecho"]

    for original in variants:
        normalized, boundaries, _ = normalize_line_endings(original)
        mapped = pipeline._remap_document_to_original(results[0], original, boundaries)
        for canonical, citation in zip(results[0]["citacoes"], mapped["citacoes"]):
            expected_start, expected_end = map_span_to_original(
                canonical["inicio"], canonical["fim"], boundaries
            )
            assert (citation["inicio"], citation["fim"]) == (expected_start, expected_end)
            assert original[citation["inicio"]:citation["fim"]] == citation["trecho"]
            assert normalized[canonical["inicio"]:canonical["fim"]] == canonical["trecho"]
