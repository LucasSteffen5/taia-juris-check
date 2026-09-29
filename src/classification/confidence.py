# -*- coding: utf-8 -*-
"""Confiança neutra, sem alegar calibração estatística."""


def compute_confidence(*_args, **_kwargs) -> float:
    """Usa a probabilidade neutra até existir calibração com procedência auditada."""
    return 0.5
