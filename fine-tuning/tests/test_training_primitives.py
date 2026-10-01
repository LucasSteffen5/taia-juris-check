from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import os

import torch

from train.config import NER_LABELS
from train.provenance import sha256_file
from train.training import (
    WeightedNERTrainer, WeightedTokenCollator, _context_sampler_weights,
    _load_grad_scaler_state, _load_local_dataset, _save_grad_scaler_state,
    _verify_prepared_artifacts, _window_sampler_weights, ModelSnapshotCallback, extend_experiment_queue,
    initialize_experiment_queue, repair_snapshot_metadata, select_experiment_checkpoint, _pipeline_model_fingerprint,
    _parse_best_candidate,
)


ROOT = Path(__file__).resolve().parents[2]
MODEL = Path(os.environ.get("FT_TEST_SOURCE_MODEL", ROOT / "models" / "ner_legal_large"))


def test_collator_ignores_special_tokens_and_padding_and_preserves_weights() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    collator = WeightedTokenCollator(tokenizer)
    batch = collator([
        {
            "input_ids": [tokenizer.cls_token_id, 20, tokenizer.sep_token_id],
            "token_type_ids": [0, 0, 0],
            "attention_mask": [1, 1, 1],
            "labels": [-100, NER_LABELS.index("B-JURISPRUDENCIA"), -100],
            "loss_weights": [0.0, 1.0, 0.0],
            "auxiliary_labels": [-100, -100, -100],
            "auxiliary_weights": [0.0, 0.0, 0.0],
        },
        {
            "input_ids": [tokenizer.cls_token_id, 21, 22, 23, tokenizer.sep_token_id],
            "token_type_ids": [0, 0, 0, 0, 0],
            "attention_mask": [1, 1, 1, 1, 1],
            "labels": [-100, 0, 0, 0, -100],
            "loss_weights": [0.0, 1.0, 1.0, 1.0, 0.0],
            "auxiliary_labels": [-100, -100, -100, -100, -100],
            "auxiliary_weights": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
    ])
    assert batch["input_ids"].shape[1] % 8 == 0
    assert torch.all(batch["labels"][:, 0] == -100)
    assert torch.all(batch["loss_weights"][:, 0] == 0)
    assert torch.all(batch["labels"][0, 3:] == -100)
    assert torch.all(batch["loss_weights"][0, 3:] == 0)
    assert batch["labels"][0, 1].item() == NER_LABELS.index("B-JURISPRUDENCIA")
    assert batch["loss_weights"][0, 1].item() == 1.0


def test_ocr_variants_keep_prepared_source_exposure_ratio() -> None:
    rows = ([{"source": "local", "augmentation_rate": 0.0}] * 2
            + [{"source": "official", "augmentation_rate": 0.0}] * 5
            + [{"source": "local", "augmentation_rate": 0.01}] * 4)
    weights = _window_sampler_weights(rows)
    local_mass = sum(weight for row, weight in zip(rows, weights) if row["source"] == "local")
    official_mass = sum(weight for row, weight in zip(rows, weights) if row["source"] == "official")
    assert abs(local_mass - 2 / 7) < 1e-12
    assert abs(official_mass - 5 / 7) < 1e-12


def test_context_sampler_balances_contexts_and_keeps_official_mass() -> None:
    rows = [
        {"source": "local", "document_id": "syn_n1_000001", "sampling_group": "syn_ctx_1"},
        {"source": "local", "document_id": "syn_n2_000001", "sampling_group": "syn_ctx_1"},
        {"source": "local", "document_id": "syn_n1_000004", "sampling_group": "syn_ctx_2"},
        {"source": "official", "document_id": "gen_n1_001", "sampling_group": "gen_n1_001"},
        {"source": "official", "document_id": "gen_n1_001", "sampling_group": "gen_n1_001"},
    ]
    weights = _context_sampler_weights(rows)
    assert abs(sum(weights) - 1.0) < 1e-12
    assert abs(sum(weights[:3]) - 3 / 5) < 1e-12
    assert abs(sum(weights[3:]) - 2 / 5) < 1e-12
    assert abs(weights[0] + weights[1] - weights[2]) < 1e-12


def test_loader_applies_official_repeat_ocr_and_group_disjoint_folds(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    prepared = run_dir / "prepared"
    prepared.mkdir(parents=True)
    model_row = {
        "input_ids": [101, 5, 102], "attention_mask": [1, 1, 1], "token_type_ids": [0, 0, 0],
        "labels": [-100, 0, -100], "loss_weights": [0.0, 1.0, 0.0],
        "auxiliary_labels": [-100, -100, -100], "auxiliary_weights": [0.0, 0.0, 0.0],
    }
    rows = []
    for doc_id in ("syn_n1_000001", "syn_n2_000001", "syn_n1_000004"):
        rows.append({**model_row, "document_id": doc_id, "source": "local", "sampling_group": doc_id[:-1]})
    for _ in range(5):
        rows.append({**model_row, "document_id": "gen_n1_001", "source": "official", "sampling_group": "gen_n1_001"})
    train_path = prepared / "train.jsonl"
    train_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (prepared / "augmentation_ocr_01.jsonl").write_text(
        json.dumps({**model_row, "document_id": "syn_n1_000001__ocr_01", "source": "local",
                    "sampling_group": "syn_n1_00000", "augmentation_rate": 0.01,
                    "base_document_id": "syn_n1_000001"}) + "\n", encoding="utf-8")
    without_repeat = _load_local_dataset(train_path, {"official_repeat": 1})
    assert sum(item.get("source") == "official" for item in without_repeat) == 1
    with_ocr = _load_local_dataset(train_path, {"official_repeat": 3, "ocr_noise_rate": 0.01}, run_dir)
    assert sum(float(item.get("augmentation_rate", 0.0)) == 0.01 for item in with_ocr) == 1
    fold_train = _load_local_dataset(train_path, {"fold": 0, "fold_count": 3}, run_dir)
    fold_valid = _load_local_dataset(train_path, {"fold": 0, "fold_count": 3}, run_dir, fold_validation=True)
    assert all(item["source"] == "local" for item in fold_train + fold_valid)
    train_groups = {item["sampling_group"] for item in fold_train}
    valid_groups = {item["sampling_group"] for item in fold_valid}
    assert train_groups.isdisjoint(valid_groups)


def test_campaign_queue_expands_preregistered_ablation_and_robustness_grids(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FT_RUNS_ROOT", str(tmp_path))
    run_dir = tmp_path / "campaign"
    (run_dir / "metrics").mkdir(parents=True)
    (run_dir / "candidates").mkdir()
    initial = initialize_experiment_queue("campaign")
    for experiment in initial["experiments"]:
        if experiment["name"] not in {"original_lr_5e-06", "v1_lr_5e-06"}:
            continue
        candidate = run_dir / "candidates" / experiment["name"]
        candidate.mkdir(parents=True)
        (candidate / "model.safetensors").write_bytes(experiment["name"].encode())
        (run_dir / "metrics" / f"{experiment['name']}_validation.json").write_text(
            json.dumps({"mode": "local", "challenge_metrics": {
                "score_final": 0.70, "niveis": {"1": {"score": 0.7}, "2": {"score": 0.7}}
            }}),
            encoding="utf-8",
        )
    ablation = extend_experiment_queue("campaign", "ablation")
    assert len(ablation["added"]) == 18
    queue = json.loads((run_dir / "experiment_queue.json").read_text(encoding="utf-8"))
    stage_two = [row for row in queue["experiments"] if row.get("stage") == "ablation"]
    assert {row["auxiliary_loss_weight"] for row in stage_two} == {0.0, 0.10, 0.25}
    assert {row["official_repeat"] for row in stage_two} == {1, 3, 5}
    for experiment in stage_two:
        candidate = run_dir / "candidates" / experiment["name"]
        candidate.mkdir(parents=True)
        (candidate / "model.safetensors").write_bytes(experiment["name"].encode())
        score = 0.71 if experiment["auxiliary_loss_weight"] == 0.10 else 0.69
        (run_dir / "metrics" / f"{experiment['name']}_validation.json").write_text(
            json.dumps({"score_final": score, "niveis": {"1": {"score": score}, "2": {"score": score}}}),
            encoding="utf-8",
        )
    robustness = extend_experiment_queue("campaign", "robustness")
    assert len(robustness["added"]) == 12
    queue = json.loads((run_dir / "experiment_queue.json").read_text(encoding="utf-8"))
    stage_three = [row for row in queue["experiments"] if row.get("stage") == "robustness"]
    assert {row["ocr_noise_rate"] for row in stage_three} == {0.0, 0.01, 0.03}
    assert {row["sampler"] for row in stage_three} == {"window", "context"}


def test_candidate_score_parser_accepts_nested_and_flat_evaluator_reports(tmp_path: Path) -> None:
    nested = tmp_path / "nested.json"
    nested.write_text(json.dumps({"challenge_metrics": {
        "score_final": 0.71,
        "niveis": {"1": {"tau": 0.1}, "2": {"score": 0.72, "tau": 0.2}},
    }}), encoding="utf-8")
    flat = tmp_path / "flat.json"
    flat.write_text(json.dumps({
        "score_final": 0.69,
        "niveis": {"1": {"tau": 0.0}, "2": {"score": 0.68, "tau": 0.1}},
    }), encoding="utf-8")

    assert _parse_best_candidate(nested)["score_final"] == 0.71
    assert _parse_best_candidate(nested)["score_n2"] == 0.72
    assert _parse_best_candidate(nested)["tau_n1"] == 0.1
    assert _parse_best_candidate(nested)["tau_n2"] == 0.2
    assert _parse_best_candidate(flat)["score_final"] == 0.69


def test_checkpoint_choice_uses_full_challenge_report_and_preserves_selected_tokenizer(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FT_RUNS_ROOT", str(tmp_path))
    run_dir = tmp_path / "snapshot-run"
    exp_name = "original_lr_5e-06"
    experiment = {"name": exp_name, "stage": "base", "initial_checkpoint": "original", "learning_rate": 5e-6}
    (run_dir / "experiment_queue.json").parent.mkdir(parents=True)
    (run_dir / "experiment_queue.json").write_text(json.dumps({"experiments": [experiment]}), encoding="utf-8")
    experiment_dir = run_dir / "experiments" / exp_name
    snap_root = experiment_dir / "evaluation_checkpoints"
    snapshot = snap_root / "checkpoint-100"
    snapshot.mkdir(parents=True)
    files = {"model.safetensors": b"weights", "config.json": b"{}", "tokenizer_config.json": b"{}", "vocab.txt": b"vocab"}
    for filename, content in files.items():
        (snapshot / filename).write_bytes(content)
    hashes = {filename: sha256_file(snapshot / filename) for filename in files}
    (snapshot / "snapshot_manifest.json").write_text(json.dumps({"global_step": 100, "epoch": 1.5, "files": hashes}), encoding="utf-8")
    (run_dir / "metrics").mkdir(parents=True)
    report = {
        "mode": "local", "expected_documents": 62,
        "challenge_metrics": {"score_final": 0.7, "niveis": {"2": {"score": 0.75}}},
        "checkpoint_sha256": _pipeline_model_fingerprint(snapshot),
    }
    report_path = run_dir / "metrics" / f"snapshot_{exp_name}_step_100_validation.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    (run_dir / "metrics" / f"snapshot_{exp_name}_step_100_validation_spans.json").write_text(
        json.dumps({"ner": {"precision": 0.8}}), encoding="utf-8"
    )
    selection = select_experiment_checkpoint("snapshot-run", exp_name)
    candidate = run_dir / "candidates" / exp_name
    assert selection["selected_checkpoint_step"] == 100
    assert (candidate / "model.safetensors").read_bytes() == b"weights"
    assert (candidate / "vocab.txt").read_bytes() == b"vocab"
    assert (run_dir / "metrics" / f"{exp_name}_validation.json").is_file()
    assert not snapshot.exists()
    assert (experiment_dir / "checkpoint_archive.json").is_file()


def test_evaluation_snapshot_has_derived_provenance_and_correct_license(tmp_path: Path, monkeypatch) -> None:
    from transformers import BertConfig

    labels = {index: label for index, label in enumerate(NER_LABELS)}
    source = tmp_path / "trainer" / "checkpoint-42"
    source.mkdir(parents=True)
    config = BertConfig(
        num_labels=len(NER_LABELS), id2label=labels,
        label2id={label: index for index, label in labels.items()},
    )
    config.base_model_name = "dominguesm/legal-bert-ner-base-cased-ptbr"
    config.base_model_revision = "44210927c925448df025985e0ed48081bb5ac57c"
    config.base_model_license = "MIT"
    config.save_pretrained(source)
    (source / "model.safetensors").write_bytes(b"test weights")

    class Tokenizer:
        def save_pretrained(self, target: Path) -> None:
            (target / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    run_dir = tmp_path / "runs" / "snapshot-test"
    callback = ModelSnapshotCallback(run_dir, "original_lr_5e-06", Tokenizer())
    callback.on_save(
        SimpleNamespace(output_dir=str(tmp_path / "trainer")),
        SimpleNamespace(global_step=42, epoch=1.5),
        None,
    )
    snapshot = run_dir / "experiments" / "original_lr_5e-06" / "evaluation_checkpoints" / "checkpoint-42"
    provenance = json.loads((snapshot / "provenance.json").read_text(encoding="utf-8"))
    repaired_config = BertConfig.from_pretrained(snapshot)
    snapshot_manifest = json.loads((snapshot / "snapshot_manifest.json").read_text(encoding="utf-8"))
    assert provenance["license"] == "CC-BY-4.0"
    assert provenance["derived_from_finetuned_checkpoint"] is True
    assert provenance["files"]["model.safetensors"] == sha256_file(snapshot / "model.safetensors")
    assert repaired_config.base_model_license == "CC-BY-4.0"
    assert snapshot_manifest["files"]["provenance.json"] == sha256_file(snapshot / "provenance.json")

    (snapshot / "provenance.json").unlink()
    repaired_config.base_model_license = "MIT"
    repaired_config.save_pretrained(snapshot)
    snapshot_manifest["files"] = {
        path.name: sha256_file(path)
        for path in snapshot.iterdir()
        if path.is_file() and path.name != "snapshot_manifest.json"
    }
    (snapshot / "snapshot_manifest.json").write_text(json.dumps(snapshot_manifest), encoding="utf-8")
    monkeypatch.setenv("FT_RUNS_ROOT", str(tmp_path / "runs"))
    repaired = repair_snapshot_metadata("snapshot-test")
    migrated_provenance = json.loads((snapshot / "provenance.json").read_text(encoding="utf-8"))
    assert repaired["count"] == 1
    assert migrated_provenance["license"] == "CC-BY-4.0"
    assert (snapshot / "snapshot_manifest.json").is_file()


def test_snapshot_repair_accepts_a_fresh_run_without_experiments(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("FT_RUNS_ROOT", str(tmp_path / "runs"))
    (tmp_path / "runs" / "empty-run").mkdir(parents=True)
    assert repair_snapshot_metadata("empty-run") == {
        "run_id": "empty-run", "repaired_snapshots": [], "count": 0,
    }


def test_weighted_loss_ignores_ambiguous_positions() -> None:
    class TinyModel:
        def __call__(self, input_ids, attention_mask=None, token_type_ids=None):
            logits = torch.zeros((1, 3, len(NER_LABELS)), requires_grad=True)
            logits.data[0, 0, NER_LABELS.index("B-LEGISLACAO")] = 2.0
            logits.data[0, 1, NER_LABELS.index("B-JURISPRUDENCIA")] = 100.0
            return SimpleNamespace(logits=logits)

    trainer = WeightedNERTrainer.__new__(WeightedNERTrainer)
    inputs = {
        "input_ids": torch.tensor([[101, 102, 103]]),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
        "labels": torch.tensor([[
            NER_LABELS.index("B-LEGISLACAO"), -100, 0
        ]]),
        "loss_weights": torch.tensor([[1.0, 0.0, 0.0]]),
        "auxiliary_labels": torch.tensor([[-100, -100, -100]]),
        "auxiliary_weights": torch.tensor([[0.0, 0.0, 0.0]]),
    }
    loss = trainer.compute_loss(TinyModel(), inputs)
    expected_logits = torch.tensor([[
        torch.logsumexp(torch.zeros(9), dim=0).item(),
        2.0,
        0.0,
        0.0,
        0.0,
    ]])
    expected = torch.nn.functional.cross_entropy(
        expected_logits,
        torch.tensor([1]),
    )
    assert torch.isfinite(loss)
    assert torch.allclose(loss, expected)


def test_weighted_loss_is_invariant_to_microbatch_partition() -> None:
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.linspace(-0.2, 0.2, len(NER_LABELS)))

        def forward(self, input_ids, attention_mask=None, token_type_ids=None):
            return SimpleNamespace(logits=self.logits.view(1, 1, -1).expand(input_ids.shape[0], input_ids.shape[1], -1))

    def batch(indices):
        labels = torch.tensor([
            [-100, NER_LABELS.index("B-LEGISLACAO"), 0, -100],
            [-100, 0, NER_LABELS.index("B-JURISPRUDENCIA"), -100],
        ])[indices]
        return {
            "input_ids": torch.tensor([[101, 12, 13, 102], [101, 14, 15, 102]])[indices],
            "attention_mask": torch.ones((len(indices), 4), dtype=torch.long),
            "token_type_ids": torch.zeros((len(indices), 4), dtype=torch.long),
            "labels": labels,
            "loss_weights": torch.tensor([[0.0, 0.25, 1.0, 0.0], [0.0, 1.0, 0.5, 0.0]])[indices],
            "auxiliary_labels": torch.full((len(indices), 4), -100, dtype=torch.long),
            "auxiliary_weights": torch.zeros((len(indices), 4)),
        }

    trainer = WeightedNERTrainer.__new__(WeightedNERTrainer)
    trainer.auxiliary_loss_weight = 0.25
    full_model = TinyModel()
    full_loss = trainer.compute_loss(full_model, batch([0, 1]))
    full_loss.backward()
    full_gradient = full_model.logits.grad.detach().clone()
    split_model = TinyModel()
    split_model.load_state_dict(full_model.state_dict())
    split_model.zero_grad(set_to_none=True)
    split_losses = [trainer.compute_loss(split_model, batch([index])) for index in range(2)]
    (sum(split_losses) / 2).backward()
    assert torch.allclose(full_loss.detach(), sum(loss.detach() for loss in split_losses) / 2, atol=1e-7)
    assert torch.allclose(full_gradient, split_model.logits.grad, atol=1e-7)


def test_auxiliary_loss_is_separate_and_configurable() -> None:
    class TinyModel:
        def __call__(self, input_ids, attention_mask=None, token_type_ids=None):
            logits = torch.zeros((1, 2, len(NER_LABELS)), requires_grad=True)
            logits.data[0, 0, NER_LABELS.index("B-PESSOA")] = 3.0
            return SimpleNamespace(logits=logits)

    inputs = {
        "input_ids": torch.tensor([[101, 102]]),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
        "labels": torch.tensor([[0, -100]]),
        "loss_weights": torch.tensor([[1.0, 0.0]]),
        "auxiliary_labels": torch.tensor([[NER_LABELS.index("B-PESSOA"), -100]]),
        "auxiliary_weights": torch.tensor([[1.0, 0.0]]),
    }
    trainer = WeightedNERTrainer.__new__(WeightedNERTrainer)
    trainer.auxiliary_loss_weight = 0.0
    no_auxiliary = trainer.compute_loss(TinyModel(), dict(inputs))
    trainer.auxiliary_loss_weight = 0.25
    with_auxiliary = trainer.compute_loss(TinyModel(), dict(inputs))
    assert with_auxiliary > no_auxiliary


def test_prepared_features_are_hash_verified_before_resume(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    prepared = run_dir / "prepared"
    prepared.mkdir(parents=True)
    features = prepared / "train.jsonl"
    features.write_text('{"input_ids":[1]}\n', encoding="utf-8")
    (prepared / "dataset_manifest.json").write_text(json.dumps({
        "prepared_artifacts_sha256": {"train.jsonl": sha256_file(features)},
    }), encoding="utf-8")
    assert _verify_prepared_artifacts(run_dir) == {"train.jsonl": sha256_file(features)}
    features.write_text('{"input_ids":[2]}\n', encoding="utf-8")
    try:
        _verify_prepared_artifacts(run_dir)
    except ValueError as error:
        assert "hashes divergentes" in str(error)
    else:
        raise AssertionError("alteração de pseudoanotações/features deveria bloquear retomada")


def test_fp16_scaler_state_is_saved_and_restored_for_resume(tmp_path: Path) -> None:
    class Scaler:
        def __init__(self):
            self.values = {"scale": 32768.0, "growth_tracker": 7}

        def state_dict(self):
            return dict(self.values)

        def load_state_dict(self, values):
            self.values = dict(values)

    original = Scaler()
    saved_hash = _save_grad_scaler_state(original, tmp_path, required=True)
    assert saved_hash and (tmp_path / "scaler.pt").is_file()
    restored = Scaler()
    restored.values = {"scale": 1.0, "growth_tracker": 0}
    loaded_hash = _load_grad_scaler_state(restored, tmp_path, required=True)
    assert loaded_hash == saved_hash
    assert restored.values == original.values
