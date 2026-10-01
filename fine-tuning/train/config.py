"""Versioned defaults for a complete fine-tuning campaign."""

from __future__ import annotations

SEED = 42
SOURCE_MODEL = "dominguesm/legal-bert-ner-base-cased-ptbr"
SOURCE_REVISION = "44210927c925448df025985e0ed48081bb5ac57c"
SOURCE_WEIGHTS_SHA256 = "dea15cbbe0935ec20da635f6861ffdcad0d737a648ad48dff1affd686cf776a6"
SOURCE_LICENSE = "CC-BY-4.0"

WINDOW_SIZE = 510
WINDOW_OVERLAP = 128
TRAINING_BATCH_SIZE_OPTIONS = (16, 8, 4, 2, 1)
GRADIENT_ACCUMULATION_OPTIONS = {16: 1, 8: 2, 4: 4, 2: 8, 1: 16}
OFFICIAL_REPEAT = 5
AUXILIARY_CONFIDENCE_THRESHOLD = 0.90
AUXILIARY_LOSS_WEIGHT = 0.25
MAX_EPOCHS = 12
EARLY_STOPPING_PATIENCE = 8
EVALUATIONS_PER_EPOCH = 4
MAX_TRAINING_WALL_SECONDS = None
GPU_MEMORY_SAFETY_FRACTION = 0.20

LEARNING_RATES = (5e-6, 1e-5, 2e-5, 3e-5)

# The base campaign evaluates both legal starting points. Later ablation stages
# append runs to the same manifest rather than mutating these definitions.
EXPERIMENTS = tuple(
    {
        "name": f"{start}_lr_{learning_rate:.0e}",
        "stage": "base",
        "initial_checkpoint": start,
        "learning_rate": learning_rate,
        "seed": 42,
        "auxiliary_loss_weight": AUXILIARY_LOSS_WEIGHT,
        "official_repeat": OFFICIAL_REPEAT,
        "ocr_noise_rate": 0.0,
        "sampler": "window",
    }
    for start in ("original", "v1")
    for learning_rate in LEARNING_RATES
)

OFFICIAL_TRAIN_DOCUMENTS = (
    "gen_n1_001",
    "gen_n1_005",
    "gen_n1_006",
    "gen_n1_008",
    "gen_n1_013",
    "gen_n2_001",
    "gen_n2_002",
    "gen_n2_006",
    "gen_n2_011",
    "gen_n2_013",
)

NER_LABELS = (
    "O",
    "B-ORGANIZACAO",
    "I-ORGANIZACAO",
    "B-PESSOA",
    "I-PESSOA",
    "B-TEMPO",
    "I-TEMPO",
    "B-LOCAL",
    "I-LOCAL",
    "B-LEGISLACAO",
    "I-LEGISLACAO",
    "B-JURISPRUDENCIA",
    "I-JURISPRUDENCIA",
)

AUXILIARY_LABELS = frozenset(
    {
        "B-ORGANIZACAO",
        "I-ORGANIZACAO",
        "B-PESSOA",
        "I-PESSOA",
        "B-TEMPO",
        "I-TEMPO",
        "B-LOCAL",
        "I-LOCAL",
    }
)
