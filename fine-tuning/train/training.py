"""CUDA preflight, repeatable training runs, and resumable Trainer state."""

from __future__ import annotations

import json
import hashlib
import inspect
import math
import os
import random
import re
import signal
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

from .config import (
    AUXILIARY_CONFIDENCE_THRESHOLD,
    AUXILIARY_LOSS_WEIGHT,
    EVALUATIONS_PER_EPOCH,
    EARLY_STOPPING_PATIENCE,
    EXPERIMENTS,
    GRADIENT_ACCUMULATION_OPTIONS,
    GPU_MEMORY_SAFETY_FRACTION,
    MAX_EPOCHS,
    MAX_TRAINING_WALL_SECONDS,
    NER_LABELS,
    OFFICIAL_REPEAT,
    SEED,
    SOURCE_LICENSE,
    SOURCE_MODEL,
    SOURCE_REVISION,
    TRAINING_BATCH_SIZE_OPTIONS,
    WINDOW_OVERLAP,
    WINDOW_SIZE,
)
from .data import (
    _collect_documents, _feature_windows, _make_ocr_variant, _read_annotations,
    _write_gold, _write_jsonl, _write_text, prepare_campaign,
)
from .provenance import (
    append_jsonl,
    current_git_state,
    environment_manifest,
    sha256_file,
    sha256_tree,
    verify_source_model,
    write_json,
)


@dataclass(frozen=True)
class Paths:
    local_text: Path
    local_annotations: Path
    official_text: Path
    official_annotations: Path
    source_model: Path
    runs_root: Path

    @classmethod
    def from_environment(cls) -> "Paths":
        return cls(
            Path(os.environ.get("FT_LOCAL_TEXT", "/source/local/txt")),
            Path(os.environ.get("FT_LOCAL_GOLD", "/source/local/goldenset_offsets.csv")),
            Path(os.environ.get("FT_OFFICIAL_TEXT", "/source/official/txt")),
            Path(os.environ.get("FT_OFFICIAL_GOLD", "/source/official/goldenset_offsets.csv")),
            Path(os.environ.get("FT_SOURCE_MODEL", "/source/model")),
            Path(os.environ.get("FT_RUNS_ROOT", "/runs")),
        )

    def run_dir(self, run_id: str) -> Path:
        if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id:
            raise ValueError("run-id deve ser um nome de diretório simples.")
        return self.runs_root / run_id


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _write_pretty(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _budget_snapshot(run_dir: Path, stage: str, stage_seconds: float) -> dict[str, Any]:
    budget_path = run_dir / "budget.json"
    previous = json.loads(budget_path.read_text(encoding="utf-8")) if budget_path.is_file() else {}
    stages = dict(previous.get("stages_seconds", {}))
    stages[stage] = round(stage_seconds, 3)
    elapsed = sum(stages.values())
    remaining = None if MAX_TRAINING_WALL_SECONDS is None else max(0, round(MAX_TRAINING_WALL_SECONDS - elapsed, 3))
    result = {
        "budget_limit_seconds": MAX_TRAINING_WALL_SECONDS,
        "elapsed_seconds": round(elapsed, 3),
        "remaining_seconds": remaining,
        "stages_seconds": stages,
        "last_updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_pretty(budget_path, result)
    return result


def _record_budget(
    run_dir: Path, started: float, stage: str, stage_seconds_before: float | None = None
) -> dict[str, Any]:
    """Count wall time conservatively for preparation and model-fitting stages."""
    seconds = time.monotonic() - started
    if stage_seconds_before is None:
        budget_path = run_dir / "budget.json"
        previous = json.loads(budget_path.read_text(encoding="utf-8")) if budget_path.is_file() else {}
        stage_seconds_before = float(previous.get("stages_seconds", {}).get(stage, 0.0))
    return _budget_snapshot(run_dir, stage, stage_seconds_before + seconds)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def _model_label_map(model: Any) -> dict[str, int]:
    result = {str(name): int(index) for name, index in model.config.label2id.items()}
    expected = {name: index for index, name in enumerate(NER_LABELS)}
    if result != expected or model.config.num_labels != len(NER_LABELS):
        raise ValueError(f"Mapa de rótulos do checkpoint inicial diverge: {result}.")
    return result


def _capture_campaign_config(paths: Paths, run_dir: Path) -> dict[str, Any]:
    verify_source_model(paths.source_model)
    config = AutoConfig.from_pretrained(str(paths.source_model), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(paths.source_model), local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(
        str(paths.source_model), local_files_only=True, use_safetensors=True
    )
    label_map = _model_label_map(model)
    payload = {
        "schema_version": 1,
        "objective": "token_classification_for_citation_span_extraction",
        "source_model": SOURCE_MODEL,
        "source_revision": SOURCE_REVISION,
        "source_license": SOURCE_LICENSE,
        "source_checkpoint_sha256": sha256_tree(paths.source_model),
        "source_model_weights_sha256": sha256_file(paths.source_model / "model.safetensors"),
        "architecture": config.to_dict(),
        "tokenizer": {
            "class": tokenizer.__class__.__name__,
            "is_fast": tokenizer.is_fast,
            "model_max_length": tokenizer.model_max_length,
            "padding_side": tokenizer.padding_side,
            "truncation_side": tokenizer.truncation_side,
            "vocab_size": len(tokenizer),
            "pad_token_id": tokenizer.pad_token_id,
            "special_tokens_map": tokenizer.special_tokens_map,
            "init_kwargs": _json_safe(tokenizer.init_kwargs),
        },
        "labels": {"names_by_id": list(NER_LABELS), "ids_by_name": label_map},
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "all_parameters_trainable": all(parameter.requires_grad for parameter in model.parameters()),
        "loss": {
            "kind": "projected_five_class_citation_cross_entropy_plus_auxiliary_teacher_cross_entropy",
            "citation_annotation_weight": 1.0,
            "auxiliary_pseudolabel_weight": AUXILIARY_LOSS_WEIGHT,
            "auxiliary_pseudolabel_confidence_threshold": AUXILIARY_CONFIDENCE_THRESHOLD,
            "citation_nonentity_projection": "logsumexp(O,ORGANIZACAO,PESSOA,TEMPO,LOCAL)",
            "ambiguous_span_label": -100,
            "confirmed_distractor_target": "O",
            "ignored_special_padding_label": -100,
        },
        "windowing": {
            "total_tokens": WINDOW_SIZE,
            "overlap_tokens": WINDOW_OVERLAP,
            "truncation": False,
            "batch_padding_to_multiple_of": 8,
            "maximum_padded_batch_length": math.ceil(WINDOW_SIZE / 8) * 8,
        },
        "official_data": {
            "training_document_count": 10,
            "documents_used_only_for_training": True,
            "sampling_repeats": OFFICIAL_REPEAT,
            "validation_or_model_selection_use": False,
        },
        "experiments": list(EXPERIMENTS),
        "planned_ablation_stages": {
            "ablation": {"parents": 2, "auxiliary_loss_weights": [0.0, 0.10, 0.25], "official_repeats": [1, 3, 5]},
            "robustness": {"parents": 2, "ocr_noise_rates": [0.0, 0.01, 0.03], "samplers": ["window", "context"]},
            "stability": {"parents": 2, "additional_seeds": [17, 2026], "seed_42_reference": True},
            "cross_validation": {"planned_folds": 3, "grouping": "local training context", "status": "not yet executed"},
        },
        "checkpoint_selection": {
            "metric": "complete local challenge score from isolated evaluation pipeline",
            "tie_breakers": ["N2 score", "raw NER span precision"],
            "evaluation_snapshots": "lightweight model plus tokenizer at every Trainer save/evaluation checkpoint",
            "resume_checkpoints": "latest and best eval-loss full states, including optimizer, scheduler, RNG and GradScaler",
        },
        "checkpoint_retention_policy": {
            "during_active_experiment": "retain Trainer's latest complete states for interruption and resume",
            "after_local_score_selection": "write per-file hashes and sizes to checkpoint_archive.json, then remove bulky optimizer states and temporary snapshots; preserve selected model weights, resolved config, metrics and archive manifest",
            "resume_after_trial_completion": "start a new experiment from the preserved selected candidate; completed Trainer states are not resumable after disk cleanup",
        },
        "training_defaults": {
            "seed": SEED,
            "data_seed": SEED,
            "max_epochs": MAX_EPOCHS,
            "evaluations_per_epoch": EVALUATIONS_PER_EPOCH,
            "evaluation_strategy": "approximately_four_times_per_epoch",
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "early_stopping": {
                "monitor": "eval_loss_diagnostic_only",
                "patience_evaluations": EARLY_STOPPING_PATIENCE,
                "threshold": 0.0,
                "greater_is_better": False,
            },
            "weight_decay": 0.01,
            "warmup_ratio": 0.10,
            "gradient_clip_norm": 1.0,
            "precision": "fp16",
            "gradient_checkpointing": True,
            "dataloader_workers": 0,
            "logging_steps": 1,
            "evaluation_strategy": "steps",
            "evaluation_events_per_epoch": EVALUATIONS_PER_EPOCH,
            "save_strategy": "same_steps_as_evaluation",
            "save_total_limit": 2,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "optimizer": "adamw_torch",
            "optimizer_constructor_defaults": {
                name: _json_safe(parameter.default)
                for name, parameter in inspect.signature(torch.optim.AdamW).parameters.items()
                if name != "params" and parameter.default is not inspect.Parameter.empty
            },
            "optimizer_effective_parameters": {
                "betas": [0.9, 0.999], "epsilon": 1e-8,
                "weight_decay_by_parameter_group": [0.01, 0.0],
            },
            "lr_scheduler": "linear",
            "lr_scheduler_kwargs": {},
            "deterministic_torch_algorithms": False,
            "max_training_wall_seconds": MAX_TRAINING_WALL_SECONDS,
            "gpu_memory_safety_fraction": GPU_MEMORY_SAFETY_FRACTION,
        },
        "selection_policy": {
            "validation_only": True,
            "primary_metric": "score_final",
            "eligibility_constraint": "tau by level does not exceed original or v1 and score exceeds v1",
            "tie_breakers": ["score_N2", "span_precision"],
            "fallback_without_eligible_candidates": "export_as_experimental_only",
            "selected_local_test_evaluations": 1,
        },
        "environment": environment_manifest(),
        "git_state": current_git_state(),
        "prepared_dataset_manifest_sha256": sha256_file(run_dir / "prepared" / "dataset_manifest.json"),
    }
    _write_pretty(run_dir / "campaign_config.resolved.json", payload)
    del model
    return payload


def _experiment_queue(run_dir: Path) -> tuple[dict[str, Any], ...]:
    path = run_dir / "experiment_queue.json"
    if not path.is_file():
        return tuple(EXPERIMENTS)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("experiments")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Fila de experimentos inválida: {path}.")
    names = [row.get("name") for row in rows]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("A fila de experimentos contém nomes ausentes ou repetidos.")
    return tuple(rows)


def initialize_experiment_queue(run_id: str) -> dict[str, Any]:
    run_dir = Paths.from_environment().run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "experiment_queue.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1, "run_id": run_id, "stage_added": None,
        "experiments": list(EXPERIMENTS), "selection_source": "local_validation_only",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_pretty(path, payload)
    return payload


def _load_local_dataset(
    path: Path, experiment: dict[str, Any] | None = None, run_dir: Path | None = None,
    fold_validation: bool = False,
) -> list[dict[str, Any]]:
    experiment = experiment or {}
    official_repeat = int(experiment.get("official_repeat", OFFICIAL_REPEAT))
    if not 1 <= official_repeat <= OFFICIAL_REPEAT:
        raise ValueError(f"Repetição oficial fora do intervalo preparado (1–{OFFICIAL_REPEAT}).")
    with path.open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    seen_official_windows: dict[tuple[str, Any], int] = defaultdict(int)
    result = []
    fold_number = experiment.get("fold")
    fold_count = int(experiment.get("fold_count", 3))

    def keep_for_fold(row: dict[str, Any]) -> bool:
        if fold_number is None:
            return True
        if row.get("source") != "local":
            return False
        group = str(row.get("sampling_group") or _sampling_group(str(row.get("document_id", "")), "local"))
        bucket = int(hashlib.sha256(f"{SEED}:{group}".encode("utf-8")).hexdigest()[:8], 16) % fold_count
        return (bucket == int(fold_number)) == fold_validation
    model_fields = (
        "input_ids", "attention_mask", "token_type_ids", "labels", "loss_weights",
        "auxiliary_labels", "auxiliary_weights",
    )
    metadata_fields = (
        "document_id", "source", "level", "sampling_group", "augmentation_rate",
        "base_document_id", "official_repeat_index", "window_index",
        "window_token_start", "window_token_end",
    )
    for row in rows:
        if not keep_for_fold(row):
            continue
        if row.get("source") == "official":
            identity = row.get("window_index", tuple(row["input_ids"]))
            key = (str(row.get("document_id", "")), identity)
            seen_official_windows[key] += 1
            if seen_official_windows[key] > official_repeat:
                continue
        item = {key: row[key] for key in model_fields}
        item.update({key: row[key] for key in metadata_fields if key in row})
        result.append(item)
    noise_rate = float(experiment.get("ocr_noise_rate", 0.0))
    if noise_rate:
        if run_dir is None:
            raise ValueError("Experimento com ruído OCR precisa do diretório da execução.")
        suffix = int(round(noise_rate * 100))
        augmentation_path = run_dir / "prepared" / f"augmentation_ocr_{suffix:02d}.jsonl"
        if not augmentation_path.is_file():
            raise FileNotFoundError(f"Features de aumento OCR não preparadas: {augmentation_path}.")
        with augmentation_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    if not keep_for_fold(row):
                        continue
                    item = {key: row[key] for key in model_fields}
                    item.update({key: row[key] for key in metadata_fields if key in row})
                    result.append(item)
    return result


def _sampling_group(document_id: str, source: str) -> str:
    if source == "official":
        return document_id
    synthetic = re.fullmatch(r"syn_n[12]_(\d{6})(?:__ocr_\d+)?", document_id)
    if synthetic:
        context_number = (int(synthetic.group(1)) - 1) // 3 + 1
        return f"synthetic_{context_number:03d}"
    lener = re.fullmatch(r"lener_(?:train|dev|test)_n[12]_(.+?)(?:__ocr_\d+)?", document_id)
    if lener:
        return f"lener_{lener.group(1)}"
    return document_id


def _context_sampler_weights(rows: list[dict[str, Any]]) -> list[float]:
    by_source_group: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    source_counts: Counter[str] = Counter()
    for row in rows:
        source = str(row.get("source", "local"))
        document_id = str(row.get("document_id", "unknown"))
        group = str(row.get("sampling_group") or _sampling_group(document_id, source))
        by_source_group[source][group][document_id] += 1
        if not float(row.get("augmentation_rate", 0.0)):
            source_counts[source] += 1
    total = sum(source_counts.values())
    weights = []
    for row in rows:
        source = str(row.get("source", "local"))
        document_id = str(row.get("document_id", "unknown"))
        group = str(row.get("sampling_group") or _sampling_group(document_id, source))
        group_map = by_source_group[source]
        document_map = group_map[group]
        group_mass = (source_counts[source] / total) / len(group_map)
        document_mass = group_mass / len(document_map)
        weights.append(document_mass / document_map[document_id])
    return weights


def _window_sampler_weights(rows: list[dict[str, Any]]) -> list[float]:
    """Keep the prepared local/official exposure ratio when OCR variants are appended."""
    source_targets: Counter[str] = Counter(
        str(row.get("source", "local")) for row in rows
        if not float(row.get("augmentation_rate", 0.0))
    )
    total_target = sum(source_targets.values())
    source_row_counts = Counter(str(row.get("source", "local")) for row in rows)
    return [
        (source_targets[str(row.get("source", "local"))] / total_target)
        / source_row_counts[str(row.get("source", "local"))]
        for row in rows
    ]


def _pipeline_model_fingerprint(path: Path) -> str:
    """Match HeaderIndex.fingerprint used in the frozen prediction manifest."""
    digest = hashlib.sha256()
    for file_path in sorted(
        item for item in path.rglob("*")
        if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc"
    ):
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        with file_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _archive_completed_checkpoint_states(run_dir: Path, experiment_dir: Path) -> dict[str, Any]:
    """Record hashes and remove bulky resumable states after a round is selected."""
    checkpoint_rows = []
    for checkpoint in sorted(experiment_dir.glob("checkpoint-*")):
        if not checkpoint.is_dir():
            continue
        files = {
            path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in sorted(checkpoint.iterdir()) if path.is_file()
        }
        checkpoint_rows.append({
            "directory": checkpoint.name,
            "resume_compatible_before_archive": (checkpoint / "checkpoint.complete.json").is_file()
                and all((checkpoint / name).is_file() for name in (
                    "model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth", "scaler.pt"
                )),
            "files": files,
        })
    payload = {
        "schema_version": 1, "run_id": run_dir.name,
        "experiment": experiment_dir.name,
        "states_archived_after_full_local_score_selection": True,
        "retention_reason": "keep the campaign within available disk while preserving selected model weights and all state hashes",
        "full_checkpoint_states_retained": False,
        "free_disk_bytes_before_archive": shutil.disk_usage(experiment_dir).free,
        "checkpoints": checkpoint_rows,
        "archived_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    archive_path = experiment_dir / "checkpoint_archive.json"
    _write_pretty(archive_path, payload)
    for checkpoint in experiment_dir.glob("checkpoint-*"):
        if checkpoint.is_dir():
            shutil.rmtree(checkpoint)
    experiment_manifest_path = experiment_dir / "experiment_manifest.json"
    if experiment_manifest_path.is_file():
        experiment_manifest = json.loads(experiment_manifest_path.read_text(encoding="utf-8"))
        experiment_manifest.update({
            "full_checkpoint_states_retained": False,
            "checkpoint_archive": archive_path.name,
            "checkpoint_archive_sha256": sha256_file(archive_path),
        })
        _write_pretty(experiment_manifest_path, experiment_manifest)
    append_jsonl(run_dir / "events.jsonl", {
        "event": "completed_checkpoint_states_archived",
        "experiment": experiment_dir.name,
        "checkpoint_archive": str(archive_path),
        "checkpoint_archive_sha256": sha256_file(archive_path),
        "checkpoints_archived": len(checkpoint_rows),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    })
    return {"archive_path": str(archive_path), "archive_sha256": sha256_file(archive_path),
            "checkpoint_count": len(checkpoint_rows)}


def extend_experiment_queue(run_id: str, stage: str) -> dict[str, Any]:
    """Append the next ablation stage using only measured local validation scores."""
    if stage not in {"ablation", "robustness", "stability", "cross_validation"}:
        raise ValueError(f"Estágio experimental desconhecido: {stage}.")
    run_dir = Paths.from_environment().run_dir(run_id)
    queue = list(_experiment_queue(run_dir))
    previous_stage = {"ablation": "base", "robustness": "ablation", "stability": "robustness", "cross_validation": "stability"}[stage]
    candidates = []
    for experiment in queue:
        if experiment.get("stage") != previous_stage:
            continue
        score_path = run_dir / "metrics" / f"{experiment['name']}_validation.json"
        model_path = run_dir / "candidates" / experiment["name"] / "model.safetensors"
        if score_path.is_file() and model_path.is_file():
            candidates.append((_parse_best_candidate(score_path)["score_final"], experiment, model_path.parent))
    if not candidates:
        raise FileNotFoundError(f"Não há candidatos {previous_stage} avaliados localmente.")
    parents = sorted(candidates, key=lambda row: (-row[0], row[1]["name"]))[:2]
    if stage == "cross_validation":
        # Collapse seed replicates, then start each fold from the seed-42 model
        # of that finalist configuration, never from its held-out fold weights.
        by_configuration: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
        for score, experiment, _path in candidates:
            by_configuration[str(experiment.get("configuration_id", experiment["name"]))].append((score, experiment))
        ranked_configs = sorted(
            ((sum(score for score, _ in values) / len(values), key) for key, values in by_configuration.items()),
            reverse=True,
        )[:2]
        robust_experiments = {str(row.get("configuration_id")): row for row in queue if row.get("stage") == "robustness"}
        fold_manifest_path = run_dir / "prepared" / "fold_manifest.json"
        if not fold_manifest_path.is_file():
            raise FileNotFoundError("Prepare os folds internos antes de acrescentá-los à fila.")
        fold_manifest = json.loads(fold_manifest_path.read_text(encoding="utf-8"))
        parents = []
        for mean_score, configuration_id in ranked_configs:
            robust = robust_experiments.get(configuration_id)
            if not robust:
                raise ValueError(f"Checkpoint seed 42 ausente para configuração finalista {configuration_id}.")
            robust_path = run_dir / "candidates" / robust["name"]
            parents.append((mean_score, robust, robust_path))
    known = {row["name"] for row in queue}
    generated: list[dict[str, Any]] = []
    for rank, (score, parent, parent_path) in enumerate(parents, start=1):
        parent_digest = sha256_file(parent_path / "model.safetensors")
        parent_hash = parent_digest[:8]
        common = {
            "initial_checkpoint": "candidate", "initial_candidate": parent["name"],
            "parent_checkpoint_sha256": parent_digest, "parent_validation_score": score,
            "seed": int(parent.get("seed", SEED)), "learning_rate": float(parent["learning_rate"]),
            "auxiliary_loss_weight": float(parent.get("auxiliary_loss_weight", AUXILIARY_LOSS_WEIGHT)),
            "official_repeat": int(parent.get("official_repeat", OFFICIAL_REPEAT)),
            "ocr_noise_rate": float(parent.get("ocr_noise_rate", 0.0)),
            "sampler": str(parent.get("sampler", "window")),
        }
        variants: list[dict[str, Any]] = []
        if stage == "ablation":
            for aux_weight in (0.0, 0.10, 0.25):
                for repeat in (1, 3, 5):
                    variants.append({**common, "auxiliary_loss_weight": aux_weight, "official_repeat": repeat,
                                     "name": f"s2_{rank}_{parent_hash}_a{int(aux_weight*100):02d}_r{repeat}",
                                     "stage": stage, "configuration_id": f"{parent_hash}_a{int(aux_weight*100):02d}_r{repeat}"})
        elif stage == "robustness":
            for noise in (0.0, 0.01, 0.03):
                for sampler in ("window", "context"):
                    variants.append({**common, "ocr_noise_rate": noise, "sampler": sampler,
                                     "name": f"s3_{rank}_{parent_hash}_o{int(noise*100):02d}_{sampler[0]}",
                                     "stage": stage,
                                     "configuration_id": f"{parent.get('configuration_id', parent['name'])}_o{int(noise*100):02d}_{sampler}"})
        elif stage == "stability":
            for seed in (17, 2026):
                variants.append({**common, "seed": seed, "name": f"s4_{rank}_{parent_hash}_seed{seed}",
                                 "stage": stage, "configuration_id": parent.get("configuration_id", parent["name"])})
        else:
            fold_manifest = json.loads((run_dir / "prepared" / "fold_manifest.json").read_text(encoding="utf-8"))
            for fold in range(3):
                variants.append({**common, "name": f"cv_{rank}_{parent_hash}_fold{fold}",
                                 "stage": stage, "fold": fold, "fold_count": 3,
                                 "fold_document_count": fold_manifest["folds"][str(fold)]["document_count"],
                                 "configuration_id": parent.get("configuration_id", parent["name"])})
        generated.extend(item for item in variants if item["name"] not in known)
    queue.extend(generated)
    payload = {
        "schema_version": 1, "run_id": run_id, "stage_added": stage, "experiments": queue,
        "parents": [{"name": parent[1]["name"], "validation_score": parent[0],
                     "checkpoint_sha256": sha256_file(parent[2] / "model.safetensors")} for parent in parents],
        "selection_source": "local_validation_only", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_pretty(run_dir / "experiment_queue.json", payload)
    append_jsonl(run_dir / "events.jsonl", {"event": "experiment_stage_added", **payload})
    return {"stage": stage, "added": [item["name"] for item in generated], "parents": payload["parents"]}


def prepare_ocr_augmentations(run_id: str) -> dict[str, Any]:
    """Build seeded character-noise variants from local training N1 documents only."""
    paths = Paths.from_environment()
    run_dir = paths.run_dir(run_id)
    manifest_path = run_dir / "prepared" / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Execute prepare antes de gerar aumento OCR.")
    _verify_prepared_artifacts(run_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(paths.source_model), local_files_only=True)
    teacher = AutoModelForTokenClassification.from_pretrained(
        str(paths.source_model), local_files_only=True, use_safetensors=True
    ).to("cuda").eval()
    documents, _audit = _collect_documents(
        paths.local_text, paths.local_annotations, paths.official_text, paths.official_annotations
    )
    preflight = json.loads((run_dir / "preflight.json").read_text(encoding="utf-8"))
    teacher_batch = int(preflight.get("teacher_batch_size", 4))
    records: dict[str, Any] = {}
    for rate in (0.01, 0.03):
        rows: list[dict[str, Any]] = []
        transforms = []
        for document in documents:
            if document.source != "local" or document.split != "train" or document.level != 1:
                continue
            variant, repairs = _make_ocr_variant(document, rate, SEED + int(rate * 10000))
            rows.extend(_feature_windows(variant, tokenizer, teacher, torch, teacher_batch_size=teacher_batch))
            transforms.append({"document_id": document.document_id, "variant_id": variant.document_id,
                               "changed_characters": len(repairs), "repairs": repairs})
        target = run_dir / "prepared" / f"augmentation_ocr_{int(rate * 100):02d}.jsonl"
        count = _write_jsonl(target, rows)
        records[str(rate)] = {"path": target.name, "sha256": sha256_file(target), "feature_rows": count,
                              "source_documents": len(transforms), "source_level": 1,
                              "source_split": "train", "transformations": transforms}
    del teacher, tokenizer
    torch.cuda.empty_cache()
    audit_path = run_dir / "prepared" / "augmentation_manifest.json"
    _write_pretty(audit_path, records)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["ocr_augmentations"] = records
    manifest["prepared_artifacts_sha256"] = {
        path.relative_to(run_dir / "prepared").as_posix(): sha256_file(path)
        for path in sorted((run_dir / "prepared").rglob("*"))
        if path.is_file() and path.name != "dataset_manifest.json"
    }
    _write_pretty(manifest_path, manifest)
    _verify_prepared_artifacts(run_dir)
    return {"rates": list(records), "feature_rows": {rate: row["feature_rows"] for rate, row in records.items()}}


def prepare_internal_folds(run_id: str) -> dict[str, Any]:
    """Create three group-disjoint holdouts inside local training contexts."""
    paths = Paths.from_environment()
    run_dir = paths.run_dir(run_id)
    _verify_prepared_artifacts(run_dir)
    documents, _audit = _collect_documents(
        paths.local_text, paths.local_annotations, paths.official_text, paths.official_annotations
    )
    local_train = [doc for doc in documents if doc.source == "local" and doc.split == "train"]
    groups: dict[int, list[Any]] = {fold: [] for fold in range(3)}
    group_ids: dict[int, set[str]] = {fold: set() for fold in range(3)}
    all_groups: set[str] = set()
    for document in local_train:
        group = _sampling_group(document.document_id, "local")
        fold = int(hashlib.sha256(f"{SEED}:{group}".encode("utf-8")).hexdigest()[:8], 16) % 3
        groups[fold].append(document)
        group_ids[fold].add(group)
        all_groups.add(group)
    if any(not groups[fold] for fold in range(3)):
        raise ValueError("Um fold interno ficou sem documentos.")
    from . import data as data_module
    data_module._ORIGINAL_ROWS = defaultdict(list)
    for row in _read_annotations(paths.local_annotations):
        data_module._ORIGINAL_ROWS[row["documento_id"]].append(row)
    fold_records = {}
    fold_root = run_dir / "prepared" / "folds"
    for fold, fold_docs in groups.items():
        target = fold_root / f"fold_{fold:02d}"
        text_dir = target / "txt"
        for document in fold_docs:
            _write_text(text_dir / f"{document.document_id}.txt", document.text)
        _write_gold(target / "gold.csv", fold_docs)
        fold_records[str(fold)] = {
            "document_count": len(fold_docs),
            "context_group_count": len(group_ids[fold]),
            "document_ids": sorted(doc.document_id for doc in fold_docs),
            "context_groups": sorted(group_ids[fold]),
            "gold_sha256": sha256_file(target / "gold.csv"),
        }
    if set.union(*(group_ids[fold] for fold in range(3))) != all_groups:
        raise AssertionError("A atribuição de grupos aos folds perdeu contextos.")
    manifest = {
        "schema_version": 1, "seed": SEED, "fold_count": 3,
        "grouping": "local synthetic context or LeNER case; N1/N2 variants stay together",
        "folds": fold_records,
        "group_sets_are_pairwise_disjoint": all(not (group_ids[left] & group_ids[right])
                                                  for left in range(3) for right in range(left + 1, 3)),
    }
    if not manifest["group_sets_are_pairwise_disjoint"]:
        raise ValueError("Um grupo de contexto local atravessou folds internos.")
    manifest_path = run_dir / "prepared" / "fold_manifest.json"
    _write_pretty(manifest_path, manifest)
    dataset_manifest_path = run_dir / "prepared" / "dataset_manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    dataset_manifest["internal_cross_validation"] = manifest
    dataset_manifest["prepared_artifacts_sha256"] = {
        path.relative_to(run_dir / "prepared").as_posix(): sha256_file(path)
        for path in sorted((run_dir / "prepared").rglob("*"))
        if path.is_file() and path.name != "dataset_manifest.json"
    }
    _write_pretty(dataset_manifest_path, dataset_manifest)
    _verify_prepared_artifacts(run_dir)
    return manifest


def import_existing_checkpoints_as_snapshots(run_id: str, experiment_name: str) -> dict[str, Any]:
    """Create lightweight evaluation copies for complete checkpoints from older runs."""
    run_dir = Paths.from_environment().run_dir(run_id)
    experiment = next((row for row in _experiment_queue(run_dir) if row["name"] == experiment_name), None)
    if experiment is None:
        raise ValueError(f"Experimento desconhecido: {experiment_name}.")
    experiment_dir = run_dir / "experiments" / experiment_name
    candidate_dir = run_dir / "candidates" / experiment_name
    if not (candidate_dir / "tokenizer_config.json").is_file():
        raise FileNotFoundError("Tokenizador ausente no candidato legado; não é possível avaliar checkpoints.")
    tokenizer = AutoTokenizer.from_pretrained(str(candidate_dir), local_files_only=True)
    destination = experiment_dir / "evaluation_checkpoints"
    imported = []
    for checkpoint in sorted(experiment_dir.glob("checkpoint-*")):
        if not checkpoint.is_dir() or not (checkpoint / "model.safetensors").is_file():
            continue
        try:
            step = int(checkpoint.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        target = destination / checkpoint.name
        if target.exists() and (target / "snapshot_manifest.json").is_file():
            continue
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint / "model.safetensors", target / "model.safetensors")
        shutil.copy2(checkpoint / "config.json", target / "config.json")
        tokenizer.save_pretrained(target)
        _write_snapshot_provenance(target, run_id, experiment_name, step)
        files = {
            path.name: sha256_file(path)
            for path in sorted(target.iterdir())
            if path.is_file() and path.name != "snapshot_manifest.json"
        }
        _write_pretty(target / "snapshot_manifest.json", {
            "schema_version": 1, "global_step": step,
            "epoch": json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8")).get("epoch"),
            "files": files, "imported_from_legacy_checkpoint": str(checkpoint),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        imported.append(step)
    if not imported and not any(destination.glob("checkpoint-*/snapshot_manifest.json")):
        raise FileNotFoundError(f"Nenhum checkpoint com pesos completos em {experiment_dir}.")
    return {"experiment": experiment_name, "imported_steps": imported}


def _write_snapshot_provenance(
    snapshot_dir: Path, run_id: str, experiment_name: str, global_step: int
) -> dict[str, str]:
    """Make an intermediate snapshot loadable by the frozen pipeline with derived metadata."""
    config = AutoConfig.from_pretrained(str(snapshot_dir), local_files_only=True)
    config.base_model_name = SOURCE_MODEL
    config.base_model_revision = SOURCE_REVISION
    config.base_model_license = SOURCE_LICENSE
    config.save_pretrained(snapshot_dir)
    files = {
        path.name: sha256_file(path)
        for path in sorted(snapshot_dir.iterdir())
        if path.is_file() and path.name not in {"provenance.json", "snapshot_manifest.json"}
    }
    _write_pretty(snapshot_dir / "provenance.json", {
        "source_model": SOURCE_MODEL,
        "revision": SOURCE_REVISION,
        "license": SOURCE_LICENSE,
        "derived_from_finetuned_checkpoint": True,
        "training_run_id": run_id,
        "training_experiment": experiment_name,
        "training_checkpoint_step": int(global_step),
        "files": files,
    })
    return files


def repair_snapshot_metadata(run_id: str) -> dict[str, Any]:
    """Repair and rehash snapshots created before the pipeline provenance fix."""
    run_dir = Paths.from_environment().run_dir(run_id)
    repaired = []
    experiments_dir = run_dir / "experiments"
    if not experiments_dir.is_dir():
        return {"run_id": run_id, "repaired_snapshots": repaired, "count": 0}
    for experiment_dir in sorted(experiments_dir.iterdir()):
        if not experiment_dir.is_dir():
            continue
        snapshot_root = experiment_dir / "evaluation_checkpoints"
        for snapshot_dir in sorted(snapshot_root.glob("checkpoint-*")):
            manifest_path = snapshot_dir / "snapshot_manifest.json"
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            try:
                step = int(manifest.get("global_step", snapshot_dir.name.rsplit("-", 1)[1]))
            except (TypeError, ValueError):
                raise ValueError(f"Global step inválido no snapshot: {snapshot_dir}.")
            _write_snapshot_provenance(snapshot_dir, run_id, experiment_dir.name, step)
            manifest["files"] = {
                path.name: sha256_file(path)
                for path in sorted(snapshot_dir.iterdir())
                if path.is_file() and path.name != "snapshot_manifest.json"
            }
            manifest["pipeline_compatible_provenance"] = True
            _write_pretty(manifest_path, manifest)
            repaired.append({"experiment": experiment_dir.name, "snapshot": snapshot_dir.name})
    return {"run_id": run_id, "repaired_snapshots": repaired, "count": len(repaired)}


def _verify_prepared_artifacts(run_dir: Path) -> dict[str, str]:
    prepared_dir = run_dir / "prepared"
    manifest_path = prepared_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("prepared_artifacts_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("O manifesto não contém os hashes dos artefatos preparados; faça uma nova preparação.")
    actual = {
        path.relative_to(prepared_dir).as_posix(): sha256_file(path)
        for path in sorted(prepared_dir.rglob("*"))
        if path.is_file() and path.name != "dataset_manifest.json"
    }
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        changed = sorted(key for key in set(actual) & set(expected) if actual[key] != expected[key])
        raise ValueError(
            f"Artefatos preparados mudaram; ausentes={missing}, extras={unexpected}, hashes divergentes={changed}."
        )
    return actual


def _preflight_batch(source_dir: Path, run_dir: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponível; ajuste de pesos só pode ocorrer na RTX reservada.")
    device_index = torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    if free_bytes < 8 * 1024**3:
        raise RuntimeError(
            f"GPU com apenas {free_bytes / 1024**3:.2f} GiB livres; pare as outras execuções e libere a RTX 3060."
        )
    attempts = []
    max_length = math.ceil(WINDOW_SIZE / 8) * 8
    chosen: tuple[int, float, float] | None = None
    for batch_size in TRAINING_BATCH_SIZE_OPTIONS:
        started = time.perf_counter()
        model = optimizer = input_ids = attention = labels = output = None
        try:
            model = AutoModelForTokenClassification.from_pretrained(
                str(source_dir), local_files_only=True, use_safetensors=True
            )
            _model_label_map(model)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.config.use_cache = False
            model.train().to("cuda")
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
            torch.cuda.reset_peak_memory_stats(device_index)
            token_id = min(100, model.config.vocab_size - 1)
            cls_token_id = getattr(model.config, "cls_token_id", None)
            sep_token_id = getattr(model.config, "sep_token_id", None)
            input_ids = torch.full((batch_size, max_length), token_id, dtype=torch.long, device="cuda")
            input_ids[:, 0] = 101 if cls_token_id is None else cls_token_id
            input_ids[:, -1] = 102 if sep_token_id is None else sep_token_id
            attention = torch.ones_like(input_ids)
            labels = torch.zeros_like(input_ids)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            input_started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(input_ids=input_ids, attention_mask=attention, labels=labels)
            output.loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            step_seconds = time.perf_counter() - input_started
            peak_bytes = torch.cuda.max_memory_reserved(device_index)
            attempts.append(
                {
                    "batch_size": batch_size,
                    "result": "fits",
                    "device_index": device_index,
                    "step_seconds": round(step_seconds, 4),
                    "peak_reserved_bytes": peak_bytes,
                    "attempt_seconds": round(time.perf_counter() - started, 3),
                    "device_index": device_index,
                }
            )
            safe_reserve = max(2 * 1024**3, int(total_bytes * GPU_MEMORY_SAFETY_FRACTION))
            if total_bytes - peak_bytes >= safe_reserve:
                chosen = (batch_size, step_seconds, peak_bytes)
                del output
                break
            attempts[-1]["result"] = "fits_without_requested_memory_margin"
        except torch.cuda.OutOfMemoryError as exc:
            attempts.append(
                {
                    "batch_size": batch_size,
                    "result": "cuda_out_of_memory",
                    "message": str(exc)[:400],
                    "attempt_seconds": round(time.perf_counter() - started, 3),
                }
            )
            chosen = None
        except BaseException as exc:
            attempts.append({
                "batch_size": batch_size,
                "result": "error",
                "error_type": type(exc).__name__,
                "message": str(exc)[:800],
                "attempt_seconds": round(time.perf_counter() - started, 3),
            })
            _write_pretty(run_dir / "preflight_failure.json", {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "attempts": attempts,
                "cuda_free_memory_before_bytes": free_bytes,
                "cuda_total_memory_bytes": total_bytes,
                "cuda_device_index": device_index,
            })
            raise
        finally:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            optimizer = model = input_ids = attention = labels = output = None
            torch.cuda.empty_cache()
    if chosen is None:
        _write_pretty(run_dir / "preflight_failure.json", {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "attempts": attempts,
            "cuda_free_memory_before_bytes": free_bytes,
            "cuda_total_memory_bytes": total_bytes,
            "cuda_device_index": device_index,
        })
        raise RuntimeError("Nenhum batch seguro coube na GPU; confira a VRAM e o checkpoint.")
    batch_size, step_seconds, peak_bytes = chosen
    environment = environment_manifest()
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "batch_size_per_device": batch_size,
        "teacher_batch_size": max(1, min(4, batch_size)),
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_OPTIONS[batch_size],
        "effective_batch_size": batch_size * GRADIENT_ACCUMULATION_OPTIONS[batch_size],
        "requested_memory_safety_fraction": GPU_MEMORY_SAFETY_FRACTION,
        "sequence_length": max_length,
        "gradient_checkpointing": True,
        "mixed_precision": "fp16",
        "cuda_free_memory_before_bytes": free_bytes,
        "cuda_total_memory_bytes": total_bytes,
        "cuda_device_index": device_index,
        "peak_reserved_bytes": peak_bytes,
        "one_step_seconds": round(step_seconds, 4),
        "gpu": environment["gpu"],
        "torch_cuda_version": environment["torch_cuda_version"],
        "cudnn_version": environment["cudnn_version"],
        "attempts": attempts,
    }
    _write_pretty(run_dir / "preflight.json", result)
    return result


class FeatureDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class WeightedNERTrainer(Trainer):
    """Apply annotation and distillation weights without changing the 13-label head."""

    def __init__(
        self, *args: Any,
        auxiliary_loss_weight: float = AUXILIARY_LOSS_WEIGHT,
        sampler_mode: str = "window",
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.auxiliary_loss_weight = float(auxiliary_loss_weight)
        if sampler_mode not in {"window", "context"}:
            raise ValueError(f"Sampler desconhecido: {sampler_mode}.")
        self.sampler_mode = sampler_mode

    def _get_train_sampler(self) -> Any:
        dataset = self.train_dataset
        has_augmentation = any(float(row.get("augmentation_rate", 0.0)) for row in dataset.rows)
        if self.sampler_mode == "context" or has_augmentation:
            weights = _context_sampler_weights(dataset.rows) if self.sampler_mode == "context" else _window_sampler_weights(dataset.rows)
            generator = torch.Generator()
            generator.manual_seed(int(self.args.data_seed))
            return WeightedRandomSampler(
                torch.tensor(weights, dtype=torch.double),
                num_samples=len(dataset), replacement=True, generator=generator,
            )
        return super()._get_train_sampler()

    def _save_optimizer_and_scheduler(self, output_dir: str) -> None:
        super()._save_optimizer_and_scheduler(output_dir)
        _save_grad_scaler_state(
            getattr(self.accelerator, "scaler", None), Path(output_dir), required=bool(self.args.fp16)
        )

    def _load_optimizer_and_scheduler(self, checkpoint: str | None) -> None:
        super()._load_optimizer_and_scheduler(checkpoint)
        if checkpoint is not None:
            _load_grad_scaler_state(
                getattr(self.accelerator, "scaler", None), Path(checkpoint), required=bool(self.args.fp16)
            )

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Any | None = None,
    ) -> Any:
        weights = inputs.pop("loss_weights").float()
        auxiliary_labels = inputs.pop("auxiliary_labels")
        auxiliary_weights = inputs.pop("auxiliary_weights").float()
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.float()
        names = list(NER_LABELS)
        auxiliary_ids = [
            index for index, name in enumerate(names)
            if name not in {"O", "B-LEGISLACAO", "I-LEGISLACAO", "B-JURISPRUDENCIA", "I-JURISPRUDENCIA"}
        ]
        citation_groups = [
            [0, *auxiliary_ids],
            [names.index("B-LEGISLACAO")],
            [names.index("I-LEGISLACAO")],
            [names.index("B-JURISPRUDENCIA")],
            [names.index("I-JURISPRUDENCIA")],
        ]
        citation_logits = torch.stack(
            [torch.logsumexp(logits[..., group], dim=-1) for group in citation_groups],
            dim=-1,
        )
        citation_labels = labels.clone()
        citation_map = {
            "O": 0,
            "B-ORGANIZACAO": 0, "I-ORGANIZACAO": 0,
            "B-PESSOA": 0, "I-PESSOA": 0,
            "B-TEMPO": 0, "I-TEMPO": 0,
            "B-LOCAL": 0, "I-LOCAL": 0,
            "B-LEGISLACAO": 1, "I-LEGISLACAO": 2,
            "B-JURISPRUDENCIA": 3, "I-JURISPRUDENCIA": 4,
        }
        for label_id, name in enumerate(names):
            citation_labels[labels == label_id] = citation_map[name]
        citation_per_token = F.cross_entropy(
            citation_logits.reshape(-1, citation_logits.shape[-1]),
            citation_labels.reshape(-1), ignore_index=-100, reduction="none",
        ).reshape_as(weights)
        active_weights = weights * labels.ne(-100)
        citation_denominator = active_weights.sum(dim=-1).clamp(min=1.0)
        citation_loss_by_window = (citation_per_token * active_weights).sum(dim=-1) / citation_denominator
        citation_loss = citation_loss_by_window.mean()
        aux_active = auxiliary_weights * auxiliary_labels.ne(-100)
        if bool(aux_active.any()):
            auxiliary_per_token = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                auxiliary_labels.reshape(-1), ignore_index=-100, reduction="none",
            ).reshape_as(auxiliary_weights)
            auxiliary_denominator = aux_active.sum(dim=-1).clamp(min=1.0)
            auxiliary_loss_by_window = (auxiliary_per_token * aux_active).sum(dim=-1) / auxiliary_denominator
            auxiliary_loss = auxiliary_loss_by_window.mean()
        else:
            auxiliary_loss = citation_loss.new_zeros(())
        loss = citation_loss + getattr(self, "auxiliary_loss_weight", AUXILIARY_LOSS_WEIGHT) * auxiliary_loss
        return (loss, outputs) if return_outputs else loss


class DelayedEarlyStoppingCallback(EarlyStoppingCallback):
    """Wait until one complete epoch before counting non-improving evaluations."""

    def on_evaluate(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if state.epoch is None or state.epoch < 1.0:
            return control
        return super().on_evaluate(args, state, control, **kwargs)


def _save_grad_scaler_state(scaler: Any | None, checkpoint_dir: Path, required: bool) -> str | None:
    if scaler is None:
        if required:
            raise RuntimeError("FP16 está ativo, mas o Accelerator não expôs um GradScaler para salvar.")
        return None
    state = scaler.state_dict()
    if not state:
        raise RuntimeError("O GradScaler FP16 retornou um estado vazio; o checkpoint não é retomável.")
    target = checkpoint_dir / "scaler.pt"
    temporary = checkpoint_dir / "scaler.pt.tmp"
    torch.save(state, temporary)
    os.replace(temporary, target)
    return sha256_file(target)


def _load_grad_scaler_state(scaler: Any | None, checkpoint_dir: Path, required: bool) -> str | None:
    path = checkpoint_dir / "scaler.pt"
    if scaler is None:
        if required:
            raise RuntimeError("FP16 está ativo, mas o Accelerator não expôs um GradScaler para restaurar.")
        return None
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"Checkpoint FP16 sem estado do GradScaler: {path}.")
        return None
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Checkpoint contém um estado GradScaler vazio ou inválido: {path}.")
    scaler.load_state_dict(state)
    return sha256_file(path)


def _verify_complete_checkpoint(checkpoint_dir: Path) -> dict[str, Any]:
    marker_path = checkpoint_dir / "checkpoint.complete.json"
    if not marker_path.is_file():
        raise ValueError(f"Checkpoint sem marcador de gravação completa: {checkpoint_dir}.")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("schema_version") != 1:
        raise ValueError("Versão do marcador de checkpoint desconhecida.")
    for relative, expected in marker.get("files", {}).items():
        target = checkpoint_dir / relative
        if not target.is_file() or sha256_file(target) != expected:
            raise ValueError(f"Arquivo de checkpoint ausente ou alterado: {relative}.")
    required = {"model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth", "scaler.pt"}
    missing = sorted(required - set(marker.get("files", {})))
    if missing:
        raise ValueError(f"Checkpoint sem estados necessários: {missing}.")
    return marker


class WeightedTokenCollator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.base_collator = DataCollatorForTokenClassification(
            tokenizer=tokenizer, padding=True, pad_to_multiple_of=8
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        feature_copies = []
        all_weights = []
        all_auxiliary_labels = []
        all_auxiliary_weights = []
        for feature in features:
            copied = dict(feature)
            all_weights.append(list(copied.pop("loss_weights")))
            all_auxiliary_labels.append(list(copied.pop("auxiliary_labels")))
            all_auxiliary_weights.append(list(copied.pop("auxiliary_weights")))
            for metadata_key in (
                "document_id", "source", "level", "sampling_group", "augmentation_rate",
                "base_document_id", "official_repeat_index", "window_index",
                "window_token_start", "window_token_end",
            ):
                copied.pop(metadata_key, None)
            feature_copies.append(copied)
        batch = self.base_collator(feature_copies)
        target_length = batch["input_ids"].shape[1]
        padded_weights = []
        for values in all_weights:
            amount = target_length - len(values)
            padding = [0.0] * amount
            padded_weights.append(values + padding if self.tokenizer.padding_side == "right" else padding + values)
        batch["loss_weights"] = torch.tensor(padded_weights, dtype=torch.float32)
        padded_auxiliary_labels = []
        padded_auxiliary_weights = []
        for labels, weights in zip(all_auxiliary_labels, all_auxiliary_weights):
            amount = target_length - len(labels)
            label_padding = [-100] * amount
            weight_padding = [0.0] * amount
            if self.tokenizer.padding_side == "right":
                padded_auxiliary_labels.append(labels + label_padding)
                padded_auxiliary_weights.append(weights + weight_padding)
            else:
                padded_auxiliary_labels.append(label_padding + labels)
                padded_auxiliary_weights.append(weight_padding + weights)
        batch["auxiliary_labels"] = torch.tensor(padded_auxiliary_labels, dtype=torch.long)
        batch["auxiliary_weights"] = torch.tensor(padded_auxiliary_weights, dtype=torch.float32)
        return batch


class StructuredLogCallback(TrainerCallback):
    def __init__(self, run_dir: Path, experiment: str):
        self.log_path = run_dir / "history.jsonl"
        self.experiment = experiment

    def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        if logs:
            append_jsonl(
                self.log_path,
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "experiment": self.experiment,
                    "step": state.global_step,
                    "epoch": state.epoch,
                    "metrics": _json_safe(logs),
                    "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(torch.cuda.current_device()) if torch.cuda.is_available() else None,
                    "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(torch.cuda.current_device()) if torch.cuda.is_available() else None,
                },
            )
        return control


class ModelSnapshotCallback(TrainerCallback):
    """Keep a lightweight, inference-loadable model snapshot at every eval/save point."""

    def __init__(self, run_dir: Path, experiment: str, tokenizer: Any):
        self.run_dir = run_dir
        self.experiment = experiment
        self.destination = run_dir / "experiments" / experiment / "evaluation_checkpoints"
        self.tokenizer = tokenizer

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        source = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        target = self.destination / source.name
        target.mkdir(parents=True, exist_ok=True)
        for filename in ("model.safetensors", "config.json"):
            source_file = source / filename
            if not source_file.is_file():
                raise FileNotFoundError(f"Snapshot sem arquivo obrigatório: {source_file}.")
            shutil.copy2(source_file, target / filename)
        self.tokenizer.save_pretrained(target)
        _write_snapshot_provenance(target, self.run_dir.name, self.experiment, int(state.global_step))
        files = {
            path.name: sha256_file(path)
            for path in sorted(target.iterdir())
            if path.is_file() and path.name != "snapshot_manifest.json"
        }
        _write_pretty(target / "snapshot_manifest.json", {
            "schema_version": 1, "global_step": int(state.global_step), "epoch": state.epoch,
            "files": files, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        return control


class CampaignBudgetCallback(TrainerCallback):
    def __init__(
        self, run_dir: Path, started: float, elapsed_before_run: float,
        stage: str, stage_seconds_before: float,
    ):
        self.run_dir = run_dir
        self.started = started
        self.elapsed_before_run = elapsed_before_run
        self.stage = stage
        self.stage_seconds_before = stage_seconds_before
        self.stopped_for_budget = False

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        elapsed = self.elapsed_before_run + time.monotonic() - self.started
        if MAX_TRAINING_WALL_SECONDS is not None and elapsed >= MAX_TRAINING_WALL_SECONDS:
            self.stopped_for_budget = True
            control.should_training_stop = True
        return control

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        required = ("model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth", "scaler.pt")
        missing = [name for name in required if not (checkpoint / name).is_file()]
        if missing:
            raise RuntimeError(f"Trainer deixou o checkpoint sem estado retomável: {missing}.")
        _write_pretty(checkpoint / "checkpoint.complete.json", {
            "schema_version": 1,
            "global_step": int(state.global_step),
            "files": {name: sha256_file(checkpoint / name) for name in required},
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })
        budget = _budget_snapshot(
            self.run_dir, self.stage,
            self.stage_seconds_before + time.monotonic() - self.started,
        )
        append_jsonl(self.run_dir / "events.jsonl", {
            "event": "checkpoint_saved",
            "stage": self.stage,
            "global_step": state.global_step,
            "epoch": state.epoch,
            "budget_elapsed_seconds": budget["elapsed_seconds"],
            "remaining_seconds": budget["remaining_seconds"],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })
        return control


def prepare_run(run_id: str) -> dict[str, Any]:
    paths = Paths.from_environment()
    run_dir = paths.run_dir(run_id)
    if run_dir.exists() and (run_dir / "prepared" / "dataset_manifest.json").is_file():
        if (run_dir / "campaign_config.resolved.json").is_file():
            raise FileExistsError(f"Preparação já existe em {run_dir}; use a mesma run-id só para retomar.")
        _verify_prepared_artifacts(run_dir)
        verify_source_model(paths.source_model)
        recovered_config = _capture_campaign_config(paths, run_dir)
        preflight_path = run_dir / "preflight.json"
        preflight = json.loads(preflight_path.read_text(encoding="utf-8")) if preflight_path.is_file() else {}
        preflight_time = datetime.fromisoformat(preflight["timestamp_utc"])
        recovered_seconds = max(0.0, (datetime.now(preflight_time.tzinfo) - preflight_time).total_seconds())
        budget = _budget_snapshot(run_dir, "prepare_and_teacher_inference", recovered_seconds)
        audit = json.loads((run_dir / "prepared" / "dataset_manifest.json").read_text(encoding="utf-8"))
        _write_pretty(run_dir / "preparation_summary.json", {
            "recovered_after_campaign_config_write_failure": True,
            "preparation_seconds_lower_bound": round(recovered_seconds, 3),
            "preflight": preflight,
            "dataset_manifest": audit,
            "remaining_training_budget_seconds": budget["remaining_seconds"],
            "campaign_config_sha256": sha256_file(run_dir / "campaign_config.resolved.json"),
            "campaign_config_schema_version": recovered_config["schema_version"],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })
        print(json.dumps({"run_id": run_id, "recovered_preparation": True, "remaining_budget_seconds": budget["remaining_seconds"]}))
        return audit
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    verify_source_model(paths.source_model)
    if not torch.cuda.is_available():
        raise RuntimeError("A criação das pseudoanotações exige a GPU CUDA reservada.")
    preflight = _preflight_batch(paths.source_model, run_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(paths.source_model), local_files_only=True)
    teacher = AutoModelForTokenClassification.from_pretrained(
        str(paths.source_model), local_files_only=True, use_safetensors=True
    )
    _model_label_map(teacher)
    teacher.eval().to("cuda").half()
    audit = prepare_campaign(
        paths.local_text,
        paths.local_annotations,
        paths.official_text,
        paths.official_annotations,
        paths.source_model,
        run_dir,
        tokenizer,
        teacher,
        torch,
        teacher_batch_size=int(preflight["teacher_batch_size"]),
    )
    del teacher
    torch.cuda.empty_cache()
    config = _capture_campaign_config(paths, run_dir)
    preparation_seconds = time.monotonic() - started
    budget = _record_budget(run_dir, started, "prepare_and_teacher_inference")
    write_json = _write_pretty
    write_json(run_dir / "preparation_summary.json", {
        "preparation_wall_seconds": round(preparation_seconds, 3),
        "preflight": preflight,
        "dataset_manifest": audit,
        "remaining_training_budget_seconds": budget["remaining_seconds"],
        "campaign_config_sha256": sha256_file(run_dir / "campaign_config.resolved.json"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(json.dumps({"run_id": run_id, "preflight": preflight, "remaining_budget_seconds": budget["remaining_seconds"]}))
    return audit


def _latest_checkpoint(experiment_dir: Path) -> Path | None:
    checkpoints = []
    for path in experiment_dir.glob("checkpoint-*"):
        try:
            checkpoints.append((int(path.name.rsplit("-", 1)[1]), path))
        except ValueError:
            continue
    return max(checkpoints)[1] if checkpoints else None


def _training_arguments(
    output_dir: Path, experiment: dict[str, Any], batch: int, resumed: bool
) -> TrainingArguments:
    run_dir = output_dir.parent.parent
    train_path = run_dir / "prepared" / "train.jsonl"
    training_examples = len(_load_local_dataset(train_path, experiment, run_dir))
    steps_per_epoch = math.ceil(
        math.ceil(training_examples / batch) / GRADIENT_ACCUMULATION_OPTIONS[batch]
    )
    evaluation_interval = max(1, math.ceil(steps_per_epoch / EVALUATIONS_PER_EPOCH))
    return TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=False,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_OPTIONS[batch],
        num_train_epochs=MAX_EPOCHS,
        learning_rate=experiment["learning_rate"],
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        lr_scheduler_type="linear",
        warmup_ratio=0.10,
        max_grad_norm=1.0,
        optim="adamw_torch",
        evaluation_strategy="steps",
        eval_steps=evaluation_interval,
        save_strategy="steps",
        save_steps=evaluation_interval,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        group_by_length=False,
        remove_unused_columns=False,
        prediction_loss_only=True,
        seed=int(experiment.get("seed", SEED)),
        data_seed=int(experiment.get("seed", SEED)),
        full_determinism=False,
        report_to="none",
        save_safetensors=True,
        save_only_model=False,
        disable_tqdm=False,
        include_num_input_tokens_seen=True,
        restore_callback_states_from_checkpoint=True,
    )


def _actual_training_config(
    model: Any, args: TrainingArguments, batch: int, experiment: dict[str, Any],
    run_dir: Path, experiment_dir: Path,
) -> dict[str, Any]:
    decay_names = {
        name for name in get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        if "bias" not in name
    }
    named_parameters = list(model.named_parameters())
    decay_count = sum(parameter.numel() for name, parameter in named_parameters if name in decay_names)
    no_decay_count = sum(parameter.numel() for name, parameter in named_parameters if name not in decay_names)
    no_decay_names = sorted(name for name, _parameter in named_parameters if name not in decay_names)
    train_examples = len(_load_local_dataset(run_dir / "prepared" / "train.jsonl", experiment, run_dir))
    steps_per_epoch = math.ceil(
        math.ceil(train_examples / args.per_device_train_batch_size) / args.gradient_accumulation_steps
    )
    estimated_total_steps = math.ceil(steps_per_epoch * args.num_train_epochs)
    effective_warmup_steps = args.get_warmup_steps(estimated_total_steps)
    tokenizer = AutoTokenizer.from_pretrained(
        os.environ.get("FT_SOURCE_MODEL", "/source/model"), local_files_only=True
    )
    trainable = [
        {"name": name, "shape": list(parameter.shape), "parameters": parameter.numel(), "requires_grad": parameter.requires_grad}
        for name, parameter in model.named_parameters()
    ]
    payload = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "experiment": experiment,
        "training_arguments": _json_safe(vars(args)),
        "model_config": model.config.to_dict(),
        "tokenizer": {
            "class": tokenizer.__class__.__name__,
            "is_fast": tokenizer.is_fast,
            "model_max_length": tokenizer.model_max_length,
            "padding_side": tokenizer.padding_side,
            "truncation_side": tokenizer.truncation_side,
            "vocab_size": len(tokenizer),
            "pad_token_id": tokenizer.pad_token_id,
            "special_tokens_map": tokenizer.special_tokens_map,
            "init_kwargs": _json_safe(tokenizer.init_kwargs),
        },
        "trainable_parameter_inventory": trainable,
        "optimizer_groups": [
            {"group_index": 0, "parameter_element_count": decay_count,
             "tensor_count": sum(1 for name, _parameter in named_parameters if name in decay_names),
             "weight_decay_parameter_names": sorted(decay_names),
             "learning_rate": experiment["learning_rate"], "weight_decay": 0.01,
             "betas": [args.adam_beta1, args.adam_beta2], "epsilon": args.adam_epsilon},
            {"group_index": 1, "parameter_element_count": no_decay_count,
             "tensor_count": len(no_decay_names), "weight_decay_parameter_names": no_decay_names,
             "learning_rate": experiment["learning_rate"], "weight_decay": 0.0,
             "betas": [args.adam_beta1, args.adam_beta2], "epsilon": args.adam_epsilon},
        ],
        "optimizer_constructor_defaults": {
            name: _json_safe(parameter.default)
            for name, parameter in inspect.signature(torch.optim.AdamW).parameters.items()
            if name != "params" and parameter.default is not inspect.Parameter.empty
        },
        "training_examples": train_examples,
        "campaign_config_sha256": sha256_file(run_dir / "campaign_config.resolved.json"),
        "prepared_dataset_manifest_sha256": sha256_file(run_dir / "prepared" / "dataset_manifest.json"),
        "steps_per_epoch": steps_per_epoch,
        "effective_batch_size": batch * args.gradient_accumulation_steps,
        "estimated_optimizer_steps": estimated_total_steps,
        "effective_warmup_steps": effective_warmup_steps,
        "estimated_final_learning_rate": 0.0,
        "official_training_repeat": int(experiment.get("official_repeat", OFFICIAL_REPEAT)),
        "sampler": {
            "mode": experiment.get("sampler", "window"),
            "weights": "uniform over source, then context, document, and windows within selected group",
            "official_repeat": int(experiment.get("official_repeat", OFFICIAL_REPEAT)),
            "ocr_noise_rate": float(experiment.get("ocr_noise_rate", 0.0)),
        },
        "loss": {
            "kind": "projected_five_class_citation_cross_entropy_plus_auxiliary_teacher_cross_entropy",
            "citation_annotation_weight": 1.0,
            "auxiliary_pseudolabel_weight": float(experiment.get("auxiliary_loss_weight", AUXILIARY_LOSS_WEIGHT)),
            "auxiliary_pseudolabel_confidence_threshold": AUXILIARY_CONFIDENCE_THRESHOLD,
            "citation_nonentity_projection": "logsumexp(O,ORGANIZACAO,PESSOA,TEMPO,LOCAL)",
            "ambiguous_span_label": -100,
            "confirmed_distractor_target": "O",
            "special_and_padding_label": -100,
        },
        "windowing": {
            "total_tokens": WINDOW_SIZE,
            "overlap_tokens": WINDOW_OVERLAP,
            "overlap_loss_weight": "per-token inverse number of containing windows",
            "truncation": False,
            "batch_padding_to_multiple_of": 8,
            "maximum_padded_batch_length": math.ceil(WINDOW_SIZE / 8) * 8,
        },
        "experiment_output_dir": str(experiment_dir),
        "seed": int(experiment.get("seed", SEED)),
        "environment": environment_manifest(),
        "git_state": current_git_state(),
        "initial_checkpoint": experiment.get("initial_checkpoint", "original"),
        "initial_candidate": experiment.get("initial_candidate"),
        "source_checkpoint_sha256": sha256_tree(
            run_dir / "candidates" / experiment["initial_candidate"]
            if experiment.get("initial_checkpoint") == "candidate"
            else Path(os.environ.get("FT_V1_MODEL", "/source/v1"))
            if experiment.get("initial_checkpoint") == "v1"
            else Path(os.environ.get("FT_SOURCE_MODEL", "/source/model"))
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    return payload


def _export_model(model: Any, tokenizer: Any, target_dir: Path) -> dict[str, Any]:
    target_dir.mkdir(parents=True, exist_ok=True)
    model.config.base_model_name = SOURCE_MODEL
    model.config.base_model_revision = SOURCE_REVISION
    model.config.base_model_license = SOURCE_LICENSE
    model.config.id2label = {index: label for index, label in enumerate(NER_LABELS)}
    model.config.label2id = {label: index for index, label in enumerate(NER_LABELS)}
    model.save_pretrained(target_dir, safe_serialization=True)
    tokenizer.save_pretrained(target_dir)
    return {path.name: sha256_file(path) for path in sorted(target_dir.iterdir()) if path.is_file()}


def _trainer_for(
    run_dir: Path, experiment: dict[str, Any], batch: int,
    resume: Path | None = None,
) -> tuple[WeightedNERTrainer, Any, Any]:
    paths = Paths.from_environment()
    _verify_prepared_artifacts(run_dir)
    experiment_dir = run_dir / "experiments" / experiment["name"]
    initial_kind = experiment.get("initial_checkpoint", "original")
    if initial_kind == "candidate":
        candidate_name = str(experiment.get("initial_candidate", ""))
        initial_dir = run_dir / "candidates" / candidate_name
        if not candidate_name or not (initial_dir / "model.safetensors").is_file():
            raise FileNotFoundError(f"Checkpoint-pai candidato ausente: {initial_dir}.")
        expected_hash = experiment.get("parent_checkpoint_sha256")
        if expected_hash and not sha256_file(initial_dir / "model.safetensors").startswith(expected_hash):
            raise ValueError("O checkpoint-pai mudou após a criação da fila experimental.")
        initial_config = AutoConfig.from_pretrained(str(initial_dir), local_files_only=True)
        if (getattr(initial_config, "base_model_name", None), getattr(initial_config, "base_model_revision", None)) != (SOURCE_MODEL, SOURCE_REVISION):
            raise ValueError("O checkpoint-pai não deriva da revisão legal NER fixada.")
    elif initial_kind == "v1":
        initial_dir = Path(os.environ.get(
            "FT_V1_MODEL", "/source/v1"
        ))
        if not (initial_dir / "model.safetensors").is_file():
            raise FileNotFoundError(f"Checkpoint v1 de inicialização não encontrado: {initial_dir}.")
        initial_config = AutoConfig.from_pretrained(str(initial_dir), local_files_only=True)
        if (getattr(initial_config, "base_model_name", None), getattr(initial_config, "base_model_revision", None)) != (SOURCE_MODEL, SOURCE_REVISION):
            raise ValueError("O checkpoint v1 não deriva da revisão legal NER fixada.")
    else:
        initial_dir = paths.source_model
    model = AutoModelForTokenClassification.from_pretrained(
        str(initial_dir), local_files_only=True, use_safetensors=True
    )
    _model_label_map(model)
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(str(initial_dir), local_files_only=True)
    train = FeatureDataset(_load_local_dataset(run_dir / "prepared" / "train.jsonl", experiment, run_dir))
    if experiment.get("fold") is None:
        validation_rows = _load_local_dataset(run_dir / "prepared" / "local_validation.jsonl")
    else:
        validation_rows = _load_local_dataset(
            run_dir / "prepared" / "train.jsonl", experiment, run_dir, fold_validation=True
        )
    validation = FeatureDataset(validation_rows)
    if not len(train) or not len(validation):
        raise ValueError("Treino ou validação sem janelas utilizáveis.")
    preflight = json.loads((run_dir / "preflight.json").read_text(encoding="utf-8"))
    args = _training_arguments(experiment_dir, experiment, int(preflight["batch_size_per_device"]), resume is not None)
    _write_pretty(experiment_dir / "training_config.resolved.json", _actual_training_config(
        model, args, int(preflight["batch_size_per_device"]), experiment, run_dir, experiment_dir
    ))
    trainer = WeightedNERTrainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=validation,
        data_collator=WeightedTokenCollator(tokenizer),
            auxiliary_loss_weight=float(experiment.get("auxiliary_loss_weight", AUXILIARY_LOSS_WEIGHT)),
            sampler_mode=str(experiment.get("sampler", "window")),
            callbacks=[
                DelayedEarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE),
                StructuredLogCallback(run_dir, experiment["name"]),
                ModelSnapshotCallback(run_dir, experiment["name"], tokenizer),
            ],
        )
    return trainer, tokenizer, model


def _train_campaign(
    run_id: str, resume_experiment: str | None = None, checkpoint: str | None = None,
    only_experiment: str | None = None,
) -> dict[str, Any]:
    paths = Paths.from_environment()
    run_dir = paths.run_dir(run_id)
    if not (run_dir / "prepared" / "dataset_manifest.json").is_file():
        raise FileNotFoundError("Dados preparados não encontrados; execute prepare antes de treinar.")
    _verify_prepared_artifacts(run_dir)
    verify_source_model(paths.source_model)
    budget_file = run_dir / "budget.json"
    budget = json.loads(budget_file.read_text(encoding="utf-8"))
    elapsed_before = float(budget.get("elapsed_seconds", 0.0))
    if MAX_TRAINING_WALL_SECONDS is not None and elapsed_before >= MAX_TRAINING_WALL_SECONDS:
        raise TimeoutError("O orçamento de GPU configurado já foi consumido.")
    campaign_experiments = list(_experiment_queue(run_dir))
    if resume_experiment and resume_experiment not in {item["name"] for item in campaign_experiments}:
        raise ValueError(f"Experimento desconhecido: {resume_experiment}.")
    if resume_experiment and not checkpoint:
        raise ValueError("Retomar exige --checkpoint com o estado completo a recuperar.")
    checkpoint_path = Path(checkpoint).resolve() if checkpoint else None
    if resume_experiment and only_experiment:
        raise ValueError("Não combine --experiment com retomada de checkpoint.")
    if only_experiment:
        experiments = [item for item in campaign_experiments if item["name"] == only_experiment]
        if not experiments:
            raise ValueError(f"Experimento desconhecido: {only_experiment}.")
    elif resume_experiment:
        experiment_start = next(index for index, item in enumerate(campaign_experiments) if item["name"] == resume_experiment)
        experiments = [campaign_experiments[experiment_start]]
    else:
        experiments = campaign_experiments
    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponível para treinamento.")
    preflight = json.loads((run_dir / "preflight.json").read_text(encoding="utf-8"))
    batch = int(preflight["batch_size_per_device"])
    results = []

    for experiment in experiments:
        experiment_dir = run_dir / "experiments" / experiment["name"]
        candidate_dir = run_dir / "candidates" / experiment["name"]
        if resume_experiment != experiment["name"] and (candidate_dir / "model.safetensors").is_file():
            continue
        resume_path = checkpoint_path if resume_experiment == experiment["name"] else None
        train_examples = len(_load_local_dataset(run_dir / "prepared" / "train.jsonl", experiment, run_dir))
        steps_per_epoch = math.ceil(
            math.ceil(train_examples / batch) / GRADIENT_ACCUMULATION_OPTIONS[batch]
        )
        if resume_path:
            trainer_state_path = resume_path / "trainer_state.json"
            trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
            starting_step = int(trainer_state.get("global_step", 0))
            total_steps = int(trainer_state.get("max_steps", steps_per_epoch * MAX_EPOCHS))
            estimated_steps = max(0, total_steps - starting_step)
        else:
            estimated_steps = steps_per_epoch * MAX_EPOCHS
        accumulation = GRADIENT_ACCUMULATION_OPTIONS[batch]
        estimated_seconds = estimated_steps * float(preflight["one_step_seconds"]) * accumulation * 1.25
        remaining_before_trial = (
            None if MAX_TRAINING_WALL_SECONDS is None else max(
                0.0, MAX_TRAINING_WALL_SECONDS - elapsed_before - (time.monotonic() - started)
            )
        )
        estimate_record = {
            "experiment": experiment,
            "resumed_from_checkpoint": str(resume_path) if resume_path else None,
            "estimated_optimizer_steps_remaining": estimated_steps,
            "preflight_microbatch_step_seconds": preflight["one_step_seconds"],
            "gradient_accumulation_steps": accumulation,
            "estimated_seconds_with_25_percent_margin": round(estimated_seconds, 3),
            "remaining_training_budget_seconds": None if remaining_before_trial is None else round(remaining_before_trial, 3),
            "will_start": remaining_before_trial is None or estimated_seconds <= remaining_before_trial,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_pretty(experiment_dir / "budget_estimate.json", estimate_record)
        if remaining_before_trial is not None and estimated_seconds > remaining_before_trial:
            append_jsonl(run_dir / "events.jsonl", {
                "event": "round_skipped_budget_estimate",
                **estimate_record,
            })
            break
        _seed_everything(int(experiment.get("seed", SEED)))
        device_index = torch.cuda.current_device()
        trial_started = time.monotonic()
        stage_name = f"fine_tuning_{experiment['name']}"
        stage_seconds_before = float(budget.get("stages_seconds", {}).get(stage_name, 0.0))
        torch.cuda.reset_peak_memory_stats(device_index)
        trainer = tokenizer = _model = callback = None
        current_budget = None
        experiment_manifest = {
            "run_id": run_id,
            "experiment": experiment,
            "status": "running",
            "resumed_from_checkpoint": str(resume_path) if resume_path else None,
            "resolved_config_sha256": None,
            "training_wall_seconds_before_run": round(elapsed_before + time.monotonic() - started, 3),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        try:
            trainer, tokenizer, _model = _trainer_for(run_dir, experiment, batch, resume_path)
            config_path = experiment_dir / "training_config.resolved.json"
            arguments_sha256 = sha256_file(config_path)
            experiment_manifest["resolved_config_sha256"] = arguments_sha256
            append_jsonl(experiment_dir / "configuration_history.jsonl", {
                "event": "resolved_config_before_resume_check",
                "resumed": resume_path is not None,
                "checkpoint": str(resume_path) if resume_path else None,
                "resolved_config_sha256": arguments_sha256,
                "config": json.loads(config_path.read_text(encoding="utf-8")),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            })
            previous_config = experiment_dir / "training_config.resolved.previous.json"
            if resume_path and previous_config.is_file():
                previous = json.loads(previous_config.read_text(encoding="utf-8"))
                current = json.loads(config_path.read_text(encoding="utf-8"))
                compatibility_fields = (
                    "training_arguments", "experiment", "model_config", "tokenizer",
                    "trainable_parameter_inventory", "optimizer_groups", "training_examples",
                    "campaign_config_sha256", "prepared_dataset_manifest_sha256", "effective_batch_size",
                    "loss", "windowing", "official_training_repeat", "source_checkpoint_sha256",
                )
                for key in compatibility_fields:
                    previous_value, current_value = previous.get(key), current.get(key)
                    if key == "training_arguments" and isinstance(previous_value, dict) and isinstance(current_value, dict):
                        transient = {"output_dir", "logging_dir", "run_name"}
                        previous_value = {name: value for name, value in previous_value.items() if name not in transient}
                        current_value = {name: value for name, value in current_value.items() if name not in transient}
                    if previous_value != current_value:
                        raise ValueError(f"Configuração incompatível para retomar; campo alterado: {key}.")
                previous_environment = previous.get("environment", {})
                current_environment = current.get("environment", {})
                for key in ("torch", "transformers", "accelerate", "numpy", "torch_cuda_version", "trainer_source_sha256", "docker_image_id"):
                    if previous_environment.get(key) != current_environment.get(key):
                        raise ValueError(f"Ambiente incompatível para retomar; campo alterado: environment.{key}.")
            elif not previous_config.is_file():
                _write_pretty(previous_config, json.loads(config_path.read_text(encoding="utf-8")))
            experiment_manifest["resolved_config_sha256"] = arguments_sha256
            _write_pretty(experiment_dir / "experiment_manifest.json", experiment_manifest)
            append_jsonl(experiment_dir / "configuration_history.jsonl", {
                "event": "training_started",
                "resumed": resume_path is not None,
                "checkpoint": str(resume_path) if resume_path else None,
                "resolved_config_sha256": arguments_sha256,
                "config": json.loads(config_path.read_text(encoding="utf-8")),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            })
            callback = CampaignBudgetCallback(run_dir, trial_started, 0.0, stage_name, stage_seconds_before)
            trainer.add_callback(callback)
            result = trainer.train(resume_from_checkpoint=str(resume_path) if resume_path else None)
            evaluation = trainer.evaluate()
            best_checkpoint = Path(trainer.state.best_model_checkpoint or "")
            if not best_checkpoint.is_dir():
                raise RuntimeError(f"Checkpoint de melhor eval_loss não encontrado: {best_checkpoint}.")
            exported_files = _export_model(trainer.model, tokenizer, candidate_dir)
            _write_pretty(candidate_dir / "provenance.json", {
                "source_model": SOURCE_MODEL,
                "revision": SOURCE_REVISION,
                "license": SOURCE_LICENSE,
                "derived_from_finetuned_checkpoint": True,
                "training_run_id": run_id,
                "selected_experiment": experiment["name"],
                "files": exported_files,
            })
            actual_files = {
                path.name: sha256_file(path)
                for path in sorted(candidate_dir.iterdir())
                if path.is_file()
            }
            experiment_manifest.update(
                {
                    "status": "stopped_at_budget" if callback.stopped_for_budget else "completed",
                    "best_checkpoint": str(best_checkpoint),
                    "best_global_step": getattr(trainer.state, "best_global_step", None),
                    "best_metric": trainer.state.best_metric,
                    "last_global_step": trainer.state.global_step,
                    "last_epoch": trainer.state.epoch,
                    "effective_max_steps": trainer.state.max_steps,
                    "effective_warmup_steps": trainer.args.get_warmup_steps(trainer.state.max_steps),
                    "final_learning_rate_per_optimizer_group": trainer.lr_scheduler.get_last_lr(),
                    "train_runtime_seconds": round(time.monotonic() - trial_started, 3),
                    "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device_index),
                    "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device_index),
                    "train_metrics": _json_safe(result.metrics),
                    "evaluation_metrics": _json_safe(evaluation),
                    "candidate_files_sha256": actual_files,
                    "budget_callback_stopped_training": callback.stopped_for_budget,
                    "timestamp_finished_utc": datetime.now(timezone.utc).isoformat(),
                    "training_log_sha256": sha256_file(run_dir / "history.jsonl"),
                }
            )
            _write_pretty(experiment_dir / "experiment_manifest.json", experiment_manifest)
            results.append(experiment_manifest)
        except BaseException:
            experiment_manifest["status"] = "interrupted_or_failed"
            experiment_manifest["timestamp_interrupted_utc"] = datetime.now(timezone.utc).isoformat()
            _write_pretty(experiment_dir / "experiment_manifest.json", experiment_manifest)
            append_jsonl(run_dir / "events.jsonl", {
                "event": "training_interrupted_or_failed",
                "experiment": experiment["name"],
                "resumed_from_checkpoint": str(resume_path) if resume_path else None,
                "resolved_config_sha256": experiment_manifest.get("resolved_config_sha256"),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            })
            raise
        finally:
            if trainer is not None:
                del trainer
            if tokenizer is not None:
                del tokenizer
            if _model is not None:
                del _model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            current_budget = _record_budget(run_dir, trial_started, stage_name, stage_seconds_before)
        _write_pretty(run_dir / "campaign_progress.json", {
            "experiments": [
                json.loads(path.read_text(encoding="utf-8"))
                for path in sorted((run_dir / "experiments").glob("*/experiment_manifest.json"))
                if json.loads(path.read_text(encoding="utf-8")).get("status") in {"completed", "stopped_at_budget"}
            ],
            "budget": current_budget,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device_index),
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device_index),
        })
        if callback is not None and callback.stopped_for_budget:
            break
        if current_budget["remaining_seconds"] is not None and current_budget["remaining_seconds"] <= 0:
            break
    print(json.dumps({"run_id": run_id, "completed_experiments": [row["experiment"]["name"] for row in results]}))
    return {"experiments": results}


def _sigterm_as_keyboard_interrupt(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt(f"received SIGTERM ({signum}); latest saved epoch checkpoint can be resumed")


def train_campaign(
    run_id: str, resume_experiment: str | None = None, checkpoint: str | None = None,
    only_experiment: str | None = None,
) -> dict[str, Any]:
    previous_handler = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _sigterm_as_keyboard_interrupt)
    try:
        return _train_campaign(run_id, resume_experiment, checkpoint, only_experiment)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def resume_experiment(run_id: str, experiment_name: str, checkpoint: str) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint).resolve()
    experiment_dir = Paths.from_environment().run_dir(run_id) / "experiments" / experiment_name
    try:
        checkpoint_path.relative_to(experiment_dir.resolve())
    except ValueError as exc:
        raise ValueError("O checkpoint de retomada deve estar dentro do experimento declarado.") from exc
    _verify_complete_checkpoint(checkpoint_path)
    return train_campaign(run_id, experiment_name, str(checkpoint_path))


def _parse_best_candidate(metrics_path: Path) -> dict[str, Any]:
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    # The independent evaluator stores challenge scores under this key; accept
    # the earlier flattened shape as well for compatibility with existing runs.
    metrics = report.get("challenge_metrics", report)
    if not isinstance(metrics, dict) or "score_final" not in metrics:
        raise ValueError(f"A métrica não contém score_final: {metrics_path}.")
    levels = metrics.get("niveis", {})
    level_one = levels.get("1", levels.get(1, {}))
    level_two = levels.get("2", levels.get(2, {}))
    return {
        "score_final": float(metrics["score_final"]),
        "score_n2": float(level_two.get("score", 0.0)),
        "tau_n1": float(level_one.get("tau", 0.0)),
        "tau_n2": float(level_two.get("tau", 0.0)),
        "raw_metrics": report,
    }


def select_candidate(run_id: str) -> dict[str, Any]:
    run_dir = Paths.from_environment().run_dir(run_id)
    baseline_path = run_dir / "metrics" / "baseline_validation.json"
    if not baseline_path.is_file():
        raise FileNotFoundError("Score de baseline local ausente.")
    baseline = _parse_best_candidate(baseline_path)
    v1_path = run_dir / "metrics" / "baseline_v1_validation.json"
    if not v1_path.is_file():
        raise FileNotFoundError("Reavaliação local corrigida do modelo v1 ausente.")
    baseline_v1 = _parse_best_candidate(v1_path)
    candidates = []
    for experiment in _experiment_queue(run_dir):
        path = run_dir / "metrics" / f"{experiment['name']}_validation.json"
        model = run_dir / "candidates" / experiment["name"] / "model.safetensors"
        if not path.is_file() or not model.is_file():
            continue
        result = _parse_best_candidate(path)
        delta_tau = max(
            result["tau_n1"] - baseline["tau_n1"],
            result["tau_n2"] - baseline["tau_n2"],
            result["tau_n1"] - baseline_v1["tau_n1"],
            result["tau_n2"] - baseline_v1["tau_n2"],
        )
        improves_v1 = result["score_final"] > baseline_v1["score_final"]
        result.update({
            "experiment": experiment["name"],
            "stage": experiment.get("stage", "base"),
            "seed": int(experiment.get("seed", SEED)),
            "configuration_id": experiment.get("configuration_id") or (
                f"{experiment.get('initial_checkpoint')}_{experiment.get('learning_rate')}_"
                f"a{experiment.get('auxiliary_loss_weight')}_r{experiment.get('official_repeat')}_"
                f"o{experiment.get('ocr_noise_rate')}_{experiment.get('sampler')}"
            ),
            "checkpoint": str(model.parent),
            "eligible": delta_tau <= 1e-12 and improves_v1,
            "maximum_tau_increase": delta_tau,
            "improves_v1_validation_score": improves_v1,
            "validation_metrics_sha256": sha256_file(path),
        })
        spans_path = run_dir / "metrics" / f"{experiment['name']}_validation_spans.json"
        span_report = json.loads(spans_path.read_text(encoding="utf-8")) if spans_path.is_file() else {}
        ner_span_report = span_report.get("ner", span_report)
        result["span_precision"] = float(ner_span_report.get("precision", 0.0))
        result["span_recall"] = float(ner_span_report.get("recall", 0.0))
        result["span_metrics_sha256"] = sha256_file(spans_path) if spans_path.is_file() else None
        candidates.append(result)
    if not candidates:
        raise FileNotFoundError("Nenhum candidato com peso e score de validação está completo.")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["configuration_id"]].append(candidate)
    configuration_results = []
    for configuration_id, members in grouped.items():
        selected_seed = next((item for item in members if item["seed"] == SEED), members[0])
        mean = lambda key: sum(float(item[key]) for item in members) / len(members)
        score_mean = mean("score_final")
        score_sd = math.sqrt(sum((float(item["score_final"]) - score_mean) ** 2 for item in members) / len(members))
        configuration_results.append({
            "configuration_id": configuration_id,
            "experiments": [item["experiment"] for item in members],
            "seeds": [item["seed"] for item in members],
            "mean_score_final": score_mean,
            "population_std_score_final": score_sd,
            "mean_score_n2": mean("score_n2"),
            "mean_span_precision": mean("span_precision"),
            "all_seeds_eligible": all(item["eligible"] for item in members),
            "checkpoint_experiment_seed_42": selected_seed["experiment"],
            "checkpoint": selected_seed["checkpoint"],
            "members": members,
        })
    eligible = [row for row in configuration_results if row["all_seeds_eligible"]]
    selected_pool = eligible or configuration_results
    selected = sorted(selected_pool, key=lambda row: (
        row["mean_score_final"], row["mean_score_n2"], row["mean_span_precision"]
    ), reverse=True)[0]
    result = {
        "schema_version": 1,
        "run_id": run_id,
            "selection_metric": "weighted_local_dev_challenge_score",
        "candidate_selection_uses_official_training_docs_for_evaluation": False,
        "baseline": {key: value for key, value in baseline.items() if key != "raw_metrics"},
        "baseline_v1": {key: value for key, value in baseline_v1.items() if key != "raw_metrics"},
        "candidates": [{key: value for key, value in row.items() if key != "raw_metrics"} for row in candidates],
        "configuration_results": [{key: value for key, value in row.items() if key != "members"} for row in configuration_results],
        "selected_experiment": selected["checkpoint_experiment_seed_42"],
        "selected_checkpoint": selected["checkpoint"],
        "eligible_for_recommendation": bool(eligible),
        "experimental_only": not bool(eligible),
        "selected_validation_score": selected["mean_score_final"],
        "selected_validation_score_n2": selected["mean_score_n2"],
        "selected_checkpoint_sha256": sha256_file(Path(selected["checkpoint"]) / "model.safetensors"),
        "tie_breaking": ["score_final", "score_n2", "span_precision"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_pretty(run_dir / "selection.json", result)
    _write_pretty(run_dir / "candidates" / "selected" / "selection.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def select_experiment_checkpoint(run_id: str, experiment_name: str) -> dict[str, Any]:
    """Select an experiment's weights by the complete pipeline challenge score."""
    run_dir = Paths.from_environment().run_dir(run_id)
    experiment = next((row for row in _experiment_queue(run_dir) if row["name"] == experiment_name), None)
    if experiment is None:
        raise ValueError(f"Experimento desconhecido: {experiment_name}.")
    experiment_dir = run_dir / "experiments" / experiment_name
    snapshot_root = experiment_dir / "evaluation_checkpoints"
    fold_number = experiment.get("fold")
    split_label = f"fold_{int(fold_number):02d}" if fold_number is not None else "validation"
    reports = sorted((run_dir / "metrics").glob(f"snapshot_{experiment_name}_step_*_{split_label}.json"))
    scored = []
    for report_path in reports:
        match = re.fullmatch(rf"snapshot_{re.escape(experiment_name)}_step_(\d+)_{split_label}\.json", report_path.name)
        if not match:
            continue
        step = int(match.group(1))
        snapshot = snapshot_root / f"checkpoint-{step}"
        manifest_path = snapshot / "snapshot_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for filename, expected_hash in manifest.get("files", {}).items():
            target = snapshot / filename
            if not target.is_file() or sha256_file(target) != expected_hash:
                raise ValueError(f"Snapshot de avaliação alterado: {target}.")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected_documents = int(experiment.get("fold_document_count", 62))
        if report.get("mode") != "local" or report.get("expected_documents") != expected_documents:
            raise ValueError(f"Relatório não é validação local completa: {report_path}.")
        model_path = snapshot / "model.safetensors"
        if report.get("checkpoint_sha256") != _pipeline_model_fingerprint(snapshot):
            raise ValueError(f"Relatório e pesos do snapshot não correspondem: {report_path}.")
        levels = report["challenge_metrics"].get("niveis", {})
        spans_path = report_path.with_name(f"{report_path.stem}_spans.json")
        spans = json.loads(spans_path.read_text(encoding="utf-8")) if spans_path.is_file() else {}
        ner_spans = spans.get("ner", spans)
        scored.append({
            "step": step,
            "epoch": manifest.get("epoch"),
            "snapshot": str(snapshot),
            "snapshot_sha256": sha256_file(model_path),
            "report": str(report_path),
            "report_sha256": sha256_file(report_path),
            "span_report": str(spans_path) if spans_path.is_file() else None,
            "score_final": float(report["challenge_metrics"]["score_final"]),
            "score_n2": float(levels.get("2", levels.get(2, {})).get("score", 0.0)),
            "span_precision": float(ner_spans.get("precision", 0.0)),
        })
    if not scored:
        raise FileNotFoundError(f"Nenhum snapshot teve avaliação completa: {experiment_name}.")
    selected = sorted(scored, key=lambda row: (row["score_final"], row["score_n2"], row["span_precision"]), reverse=True)[0]
    candidate_dir = run_dir / "candidates" / experiment_name
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)
    candidate_dir.mkdir(parents=True)
    snapshot_dir = Path(selected["snapshot"])
    for source in snapshot_dir.iterdir():
        if source.is_file():
            shutil.copy2(source, candidate_dir / source.name)
    archive = _archive_completed_checkpoint_states(run_dir, experiment_dir)
    candidate_manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "experiment": experiment,
        "selected_checkpoint_step": selected["step"],
        "selected_checkpoint_epoch": selected["epoch"],
        "selection_metric": "complete_local_challenge_score",
        "selected_validation_report_sha256": selected["report_sha256"],
        "selected_validation_score": selected["score_final"],
        "full_checkpoint_archive": archive,
        "selected_snapshot_retention": "copied into the candidate package; temporary evaluation snapshots removed",
        "evaluated_snapshots": scored,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_pretty(candidate_dir / "checkpoint_selection.json", candidate_manifest)
    source_report = Path(selected["report"])
    standard_metric_name = f"{experiment_name}_{split_label}.json"
    shutil.copyfile(source_report, run_dir / "metrics" / standard_metric_name)
    source_spans = Path(selected["span_report"]) if selected["span_report"] else None
    if source_spans:
        shutil.copyfile(source_spans, run_dir / "metrics" / f"{experiment_name}_{split_label}_spans.json")
    provenance = {
        "source_model": SOURCE_MODEL, "revision": SOURCE_REVISION, "license": SOURCE_LICENSE,
        "derived_from_finetuned_checkpoint": True, "training_run_id": run_id,
        "selected_experiment": experiment_name, "selected_snapshot_sha256": selected["snapshot_sha256"],
        "validation_report_sha256": selected["report_sha256"],
        "files": {path.name: sha256_file(path) for path in sorted(candidate_dir.iterdir()) if path.is_file()},
    }
    _write_pretty(candidate_dir / "provenance.json", provenance)
    if snapshot_root.exists():
        shutil.rmtree(snapshot_root)
    return candidate_manifest
