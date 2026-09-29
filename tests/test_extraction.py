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


def test_fusion_does_not_duplicate_low_iou_or_conflicting_ner_overlap():
    text = "art. 5 da CF/88"
    regex = [{"inicio": 0, "fim": 14, "trecho": text[:14], "tipo": "lei", "confianca_regex": 0.9}]
    ner = [{"inicio": 0, "fim": len(text), "trecho": text, "tipo": "jurisprudencia", "prob": 0.99}]
    fused = fuse_extractions(text, regex, ner)
    assert len(fused) == 1 and fused[0]["source"] == "regex_only"
