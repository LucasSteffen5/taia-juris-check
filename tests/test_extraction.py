# -*- coding: utf-8 -*-
"""Extraction tests use hand-written fixtures and never open the blind set."""
from src.extraction.regex_extractor import extract_regex_citations
from src.extraction.fusion import fuse_extractions
from src.normalization.matching_text import normalized_view, map_normalized_span


def test_law_article_with_constitution_year_suffix():
    text = "A regra decorre do art. 5º da CF/88."
    citations = extract_regex_citations(text)
    citation = next(item for item in citations if item["tipo"] == "lei")
    assert text[citation["inicio"]:citation["fim"]] == citation["trecho"]
    assert citation["trecho"].endswith("CF/88")


def test_generic_jurisprudence_is_incomplete_candidate():
    text = "Conforme jurisprudência pacífica do tribunal, o pedido deve ser acolhido."
    citations = extract_regex_citations(text)
    assert any(item["is_vague"] and "jurisprudência pacífica" in item["trecho"] for item in citations)


def test_whitespace_mapping_preserves_crlf_and_unicode_offsets():
    text = "😀 Súmula\r\n 211 do STJ"
    view, starts, ends = normalized_view(text)
    begin = view.index("211")
    mapped = map_normalized_span(begin, begin + 3, starts, ends)
    assert mapped == (text.index("211"), text.index("211") + 3)
    citation = next(item for item in extract_regex_citations(text) if item["pattern"] == "sumula")
    assert text[citation["inicio"]:citation["fim"]] == citation["trecho"]


def test_document_loader_preserves_crlf_offsets(tmp_path):
    from src.utils_data import load_document

    path = tmp_path / "crlf.txt"
    path.write_bytes("😀 Súmula\r\n211 do STJ".encode("utf-8"))
    text = load_document(path)
    assert "\r\n" in text
    citation = next(item for item in extract_regex_citations(text) if item["pattern"] == "sumula")
    assert text[citation["inicio"]:citation["fim"]] == citation["trecho"]


def test_ocr_surface_and_distractors():
    text = (
        "Autos nº 0000000-00.2024.8.26.0000\n"
        "EMENTA\n"
        "5úmula 211 do STJ; RHC nº 93 967 - SC.\n"
        "Advogado OAB/SP 123456."
    )
    citations = extract_regex_citations(text)
    spans = [item["trecho"] for item in citations]
    assert any("211" in span and "5úmula" in span for span in spans)
    assert any("93 967" in span for span in spans)
    assert not any("0000000" in span for span in spans)
    assert not any("OAB" in span for span in spans)


def test_adjacent_citations_are_not_duplicated():
    text = "Súmula 1 do STJ; Súmula 2 do STF."
    citations = extract_regex_citations(text)
    assert len(citations) == 2
    assert citations[0]["fim"] <= citations[1]["inicio"]


def test_fusion_keeps_regex_when_ner_is_empty_or_conflicts():
    text = "art. 5 da CF/88"
    regex = [{"inicio": 0, "fim": 14, "trecho": text[:14], "tipo": "lei"}]
    ner = [{"inicio": 0, "fim": len(text), "trecho": text, "tipo": "jurisprudencia", "prob": 0.99}]
    for fused in (fuse_extractions(text, regex, []), fuse_extractions(text, regex, ner)):
        assert len(fused) == 1
        assert fused[0]["tipo"] == "lei"
        assert "confianca" not in fused[0]


def test_fusion_preserves_extraction_provenance_for_confidence():
    text = "Súmula 211 do STJ"
    regex = [{"inicio": 0, "fim": len(text), "trecho": text, "tipo": "jurisprudencia"}]
    ner = [{
        "inicio": 0, "fim": len(text), "trecho": text,
        "tipo": "jurisprudencia", "prob": 0.98,
    }]

    both = fuse_extractions(text, regex, ner)
    regex_only = fuse_extractions(text, regex, [])
    ner_only = fuse_extractions(text, [], ner)

    assert both[0]["_confidence_sources"] == ("regex", "ner")
    assert both[0]["_confidence_agreement"] is True
    assert both[0]["_fusion_iou"] == 1.0
    assert both[0]["_ner_prob"] == 0.98
    assert regex_only[0]["_confidence_sources"] == ("regex",)
    assert regex_only[0]["_confidence_agreement"] is False
    assert regex_only[0]["_fusion_iou"] is None
    assert ner_only[0]["_confidence_sources"] == ("ner",)
    assert ner_only[0]["_ner_prob"] == 0.98


def test_fusion_requires_high_iou_for_confidence_agreement():
    text = "Súmula 211 do STJ"
    regex = [{"inicio": 0, "fim": 10, "trecho": text[:10], "tipo": "jurisprudencia"}]
    ner = [{
        "inicio": 0, "fim": len(text), "trecho": text,
        "tipo": "jurisprudencia", "prob": 0.98,
    }]

    fused = fuse_extractions(text, regex, ner)

    assert fused[0]["_fusion_iou"] == 10 / len(text)
    assert fused[0]["_confidence_agreement"] is False
