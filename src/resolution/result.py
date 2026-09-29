# -*- coding: utf-8 -*-
"""Structured outcome shared by the canonical resolvers."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ResolutionResult:
    """Resolution evidence without collapsing ambiguity and failed queries."""

    candidate_ids: tuple[int, ...]
    sufficient: bool
    method: str
    evidence: tuple[str, ...]
    reason: str
    target_type: str

    @property
    def n_candidates(self) -> int:
        return len(self.candidate_ids)

    @property
    def canonical_id(self) -> int | None:
        return self.candidate_ids[0] if len(self.candidate_ids) == 1 else None

    @classmethod
    def make(
        cls,
        candidate_ids: list[int] | tuple[int, ...] = (),
        *,
        sufficient: bool,
        method: str,
        evidence: list[str] | tuple[str, ...] = (),
        reason: str,
        target_type: str,
    ) -> "ResolutionResult":
        return cls(
            tuple(dict.fromkeys(int(value) for value in candidate_ids)),
            sufficient,
            method,
            tuple(evidence),
            reason,
            target_type,
        )
