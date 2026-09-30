"""Synthetic confidence policy tests; no challenge documents or labels are loaded."""
import pytest

from src.classification.confidence import (
    CONFIDENCE_POLICY_VERSION,
    calculate_confidence,
)
from src.resolution.result import ResolutionResult


def _result(candidate_ids=(), *, sufficient=True, identifier_strength="fraca"):
    return ResolutionResult.make(
        candidate_ids,
        sufficient=sufficient,
        method="fixture",
        reason="synthetic confidence fixture",
        target_type="acordao",
        identifier_strength=identifier_strength,
    )


def test_confidence_policy_has_a_version():
    assert CONFIDENCE_POLICY_VERSION == "evidence-rules-v2"


@pytest.mark.parametrize(
    ("citation", "result", "classification", "expected"),
    [
        ({"_confidence_sources": ("regex",)}, _result([10]), "real", 0.65),
        (
            {"_confidence_sources": ("regex",)},
            _result([10], identifier_strength="forte"),
            "real",
            0.75,
        ),
        (
            {"_confidence_sources": ("regex",)},
            _result([10], identifier_strength="muito_forte"),
            "real",
            0.80,
        ),
        (
            {
                "_confidence_sources": ("regex", "ner"),
                "_fusion_iou": 0.90,
            },
            _result([10], identifier_strength="muito_forte"),
            "real",
            0.85,
        ),
        ({}, _result([10, 11], identifier_strength="forte"), "incompleta", 0.70),
        ({}, _result([10, 11]), "incompleta", 0.65),
        ({}, _result([], identifier_strength="forte"), "inventada", 0.65),
        ({}, _result([], identifier_strength="muito_forte"), "inventada", 0.75),
        ({}, _result(sufficient=False), "incompleta", 0.55),
        (
            {"is_vague": True, "_confidence_sources": ("regex",)},
            _result(sufficient=False),
            "incompleta",
            0.65,
        ),
        (
            {"_confidence_sources": ("ner",)},
            _result([10], identifier_strength="muito_forte"),
            "real",
            0.75,
        ),
    ],
)
def test_confidence_uses_evidence_tiers(citation, result, classification, expected):
    assert calculate_confidence(citation, result, classification) == expected


def test_ner_only_penalty_is_clamped_to_policy_minimum():
    confidence = calculate_confidence(
        {"_confidence_sources": ("ner",)},
        _result(sufficient=False),
        "incompleta",
    )
    assert confidence == 0.50
    assert 0.0 <= confidence <= 1.0


def test_iou_below_agreement_threshold_does_not_add_bonus():
    confidence = calculate_confidence(
        {"_confidence_sources": ("regex", "ner"), "_fusion_iou": 0.89},
        _result([10], identifier_strength="muito_forte"),
        "real",
    )
    assert confidence == 0.80
