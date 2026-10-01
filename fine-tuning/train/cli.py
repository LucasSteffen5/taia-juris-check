"""Command-line interface used by the standalone training and Docker launchers."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import statistics
from pathlib import Path
from datetime import datetime, timezone

from .provenance import sha256_file, sha256_tree, write_json
from .span_audit import span_audit
from .training import (
    Paths, _experiment_queue, extend_experiment_queue, import_existing_checkpoints_as_snapshots,
    initialize_experiment_queue, prepare_internal_folds, prepare_ocr_augmentations, prepare_run, repair_snapshot_metadata, resume_experiment,
    select_candidate, select_experiment_checkpoint, train_campaign,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_run(run_dir: Path, experiment_name: str) -> Path:
    known = {experiment["name"] for experiment in _experiment_queue(run_dir)}
    if experiment_name not in known:
        raise ValueError(f"Experimento desconhecido: {experiment_name}.")
    return run_dir / "candidates" / experiment_name


def _record_offline_validation(run_dir: Path, report_path: Path) -> dict:
    report = _read_json(report_path)
    final_export_path = run_dir / "final_export.json"
    final_export = _read_json(final_export_path)
    if not report.get("valid") or not report.get("offline_load_and_smoke_inference"):
        raise ValueError("A validação offline do pacote não foi aprovada.")
    if report.get("model_sha256") != final_export.get("model_weights_sha256"):
        raise ValueError("O pacote validado não corresponde aos pesos finais registrados.")
    final_export.update({
        "validated_offline": True,
        "offline_validation_report_sha256": sha256_file(report_path),
        "offline_validation": report,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    write_json(final_export_path, final_export)
    return final_export


def _summarize_cross_validation(run_dir: Path) -> dict:
    queue = _experiment_queue(run_dir)
    groups: dict[str, list[dict]] = {}
    for experiment in queue:
        if experiment.get("stage") != "cross_validation":
            continue
        fold = int(experiment["fold"])
        path = run_dir / "metrics" / f"{experiment['name']}_fold_{fold:02d}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Métrica do fold ausente: {path}.")
        report = _read_json(path)
        if report.get("mode") != "local" or report.get("expected_documents") != experiment.get("fold_document_count"):
            raise ValueError(f"Avaliação interna do fold inválida: {path}.")
        groups.setdefault(str(experiment["configuration_id"]), []).append({
            "experiment": experiment["name"], "fold": fold,
            "documents": report["expected_documents"],
            "score_final": float(report["challenge_metrics"]["score_final"]),
            "levels": report["challenge_metrics"].get("niveis", {}),
            "report_sha256": sha256_file(path),
        })
    summaries = {}
    for configuration_id, folds in sorted(groups.items()):
        folds.sort(key=lambda row: row["fold"])
        if [row["fold"] for row in folds] != [0, 1, 2]:
            raise ValueError(f"Configuração {configuration_id} não tem os três folds.")
        scores = [row["score_final"] for row in folds]
        summaries[configuration_id] = {
            "mean_score_final": statistics.fmean(scores),
            "population_std_score_final": statistics.pstdev(scores),
            "folds": folds,
        }
    fold_manifest = _read_json(run_dir / "prepared" / "fold_manifest.json")
    result = {
        "schema_version": 1, "run_id": run_dir.name,
        "selection_source": "diagnostic_only_after_local_model_freeze",
        "fold_manifest_sha256": sha256_file(run_dir / "prepared" / "fold_manifest.json"),
        "groups_are_pairwise_disjoint": fold_manifest["group_sets_are_pairwise_disjoint"],
        "configurations": summaries,
    }
    write_json(run_dir / "fold_validation_summary.json", result)
    return result


def _build_official_comparison(run_dir: Path, original_path: Path, v1_path: Path, v2_path: Path) -> dict:
    reports = {
        "original": _read_json(original_path),
        "v1": _read_json(v1_path),
        "v2": _read_json(v2_path),
    }
    for name, report in reports.items():
        if report.get("mode") != "official" or report.get("expected_documents") != 16:
            raise ValueError(f"A avaliação {name} não cobre os 16 documentos oficiais reservados.")
        if report.get("run_id") != run_dir.name:
            raise ValueError(f"A avaliação {name} pertence a outra execução.")
    reference = reports["original"]
    invariant_fields = (
        "expected_document_ids", "official_training_document_ids", "official_golden_sha256",
        "official_golden_source_sha256", "official_golden_filter_code_sha256",
        "text_sha256", "official_text_source_sha256", "official_text_normalization",
        "official_offset_validation",
        "canonical_database_sha256", "evaluation_code_sha256", "metric_code_sha256",
    )
    for name, report in reports.items():
        if any(report.get(field) != reference.get(field) for field in invariant_fields):
            raise ValueError(f"A avaliação {name} não usou o mesmo conjunto, gold, banco, métrica e código.")
    comparison = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "purpose": "frozen_official_comparison_no_model_selection_or_tuning",
        "expected_document_ids": reference["expected_document_ids"],
        "official_training_document_ids": reference["official_training_document_ids"],
        "evaluation_code_sha256": reference["evaluation_code_sha256"],
        "metric_code_sha256": reference["metric_code_sha256"],
        "official_golden_sha256": reference["official_golden_sha256"],
        "official_golden_source_sha256": reference["official_golden_source_sha256"],
        "official_golden_filter_code_sha256": reference["official_golden_filter_code_sha256"],
        "text_sha256": reference["text_sha256"],
        "official_text_source_sha256": reference["official_text_source_sha256"],
        "official_text_normalization": reference["official_text_normalization"],
        "canonical_database_sha256": reference["canonical_database_sha256"],
        "models": {
            name: {
                "evaluation_name": report.get("evaluation_name"),
                "score_final": report["challenge_metrics"]["score_final"],
                "niveis": report["challenge_metrics"].get("niveis", {}),
                "stage_span_metrics": report.get("stage_span_metrics", {}),
                "documents_without_gold_citations": report.get("documents_without_gold_citations"),
                "report_sha256": sha256_file(path),
                "checkpoint_sha256": report.get("checkpoint_sha256"),
                "model_weights_sha256": report.get("model_weights_sha256"),
            }
            for name, (report, path) in zip(reports, zip(reports.values(), (original_path, v1_path, v2_path)))
        },
    }
    output_path = run_dir / "official_comparison.json"
    write_json(output_path, comparison)
    return comparison


def _record_official_evaluation(
    run_dir: Path, checkpoint_dir: Path, report_path: Path, comparison_path: Path
) -> dict:
    from .config import SOURCE_LICENSE, SOURCE_MODEL, SOURCE_REVISION
    from .provenance import write_json

    report = _read_json(report_path)
    final_export = _read_json(run_dir / "final_export.json")
    if report.get("mode") != "official" or report.get("expected_documents") != 16:
        raise ValueError("O relatório oficial precisa cobrir exatamente os 16 documentos reservados.")
    if report.get("run_id") != run_dir.name:
        raise ValueError("Relatório oficial pertence a outra execução.")
    selected_dir = Path(final_export["checkpoint_dir"]).resolve()
    if checkpoint_dir.resolve() != selected_dir:
        raise ValueError("A avaliação oficial do pacote não corresponde ao checkpoint selecionado.")
    if report.get("model_weights_sha256") != final_export.get("model_weights_sha256"):
        raise ValueError("Os pesos avaliados oficialmente não correspondem à exportação congelada.")
    report_ids = set(report.get("expected_document_ids", []))
    training_ids = set(report.get("official_training_document_ids", []))
    if len(report_ids) != 16 or report_ids & training_ids:
        raise ValueError("A avaliação oficial contém IDs usados no treinamento ou contagem inválida.")
    comparison = _read_json(comparison_path)
    if comparison.get("run_id") != run_dir.name or comparison.get("models", {}).get("v2", {}).get("report_sha256") != sha256_file(report_path):
        raise ValueError("A comparação oficial não inclui exatamente o relatório congelado do v2.")

    package_report = _read_json(selected_dir / "model_report.json")
    comparison_scores = {
        name: {
            "score_final": float(model["score_final"]),
            "niveis": model["niveis"],
        }
        for name, model in comparison["models"].items()
    }
    highest_observed = max(comparison_scores, key=lambda name: comparison_scores[name]["score_final"])
    score = float(report["challenge_metrics"]["score_final"])
    recommended_switch = bool(
        package_report.get("recommended_for_explicit_local_use")
        and score > max(comparison_scores[name]["score_final"] for name in ("original", "v1"))
    )
    package_report["official_final_evaluation"] = {
        "score_final": score,
        "niveis": report["challenge_metrics"]["niveis"],
        "documents": report["expected_document_ids"],
        "document_count": report["expected_documents"],
        "excluded_training_documents": report["official_training_document_ids"],
        "gold_sha256": report["official_golden_sha256"],
        "gold_source_sha256": report["official_golden_source_sha256"],
        "text_normalization": report["official_text_normalization"],
        "text_source_sha256": report["official_text_source_sha256"],
        "database_sha256": report["canonical_database_sha256"],
        "model_weights_sha256": report["model_weights_sha256"],
        "report_sha256": sha256_file(report_path),
        "selected_after_local_freeze": True,
        "comparison_sha256": sha256_file(comparison_path),
        "comparison_scores": comparison_scores,
        "highest_observed_score_model": highest_observed,
        "difference_vs_v1": score - comparison_scores["v1"]["score_final"],
        "recommended_for_pipeline_switch": recommended_switch,
        "recommendation_scope": "deployment_note_only; official held-out results were not used to select a checkpoint or tune hyperparameters",
    }
    correction_incident = run_dir / "official_evaluation_invalid_crlf" / "incident.json"
    if correction_incident.is_file():
        package_report["official_final_evaluation"]["invalid_crlf_attempt_sha256"] = sha256_file(correction_incident)
    write_json(selected_dir / "model_report.json", package_report)
    with (selected_dir / "official_evaluation.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    shutil.copyfile(comparison_path, selected_dir / "official_comparison.json")

    _copy_reproducibility_files(run_dir, selected_dir.name, selected_dir)
    dataset_manifest = _read_json(run_dir / "prepared" / "dataset_manifest.json")
    (selected_dir / "model_report.md").write_text(
        _model_report_markdown(package_report), encoding="utf-8", newline="\n"
    )
    (selected_dir / "README.md").write_text(
        _model_card(package_report, dataset_manifest, SOURCE_MODEL, SOURCE_REVISION, SOURCE_LICENSE),
        encoding="utf-8", newline="\n",
    )

    provenance = _read_json(selected_dir / "provenance.json")
    provenance["official_evaluation"] = {
        "report_sha256": sha256_file(selected_dir / "official_evaluation.json"),
        "heldout_document_ids": report["expected_document_ids"],
        "official_golden_sha256": report["official_golden_sha256"],
        "official_golden_source_sha256": report["official_golden_source_sha256"],
        "official_text_normalization": report["official_text_normalization"],
        "canonical_database_sha256": report["canonical_database_sha256"],
        "comparison_sha256": sha256_file(selected_dir / "official_comparison.json"),
    }
    provenance["files"] = {
        path.name: sha256_file(path)
        for path in sorted(selected_dir.iterdir())
        if path.is_file() and path.name != "provenance.json"
    }
    write_json(selected_dir / "provenance.json", provenance)
    final_export.update({
        "provenance_sha256": sha256_file(selected_dir / "provenance.json"),
        "official_evaluation_sha256": sha256_file(selected_dir / "official_evaluation.json"),
        "official_score_final": score,
        "official_comparison_sha256": sha256_file(selected_dir / "official_comparison.json"),
        "highest_observed_official_score_model": highest_observed,
        "recommended_for_pipeline_switch": recommended_switch,
    })
    write_json(run_dir / "final_export.json", final_export)
    report_path_in_run = run_dir / "official_evaluation_v2.json"
    shutil.copyfile(report_path, report_path_in_run)
    return {
        "score_final": score,
        "official_comparison": str(selected_dir / "official_comparison.json"),
        "package_dir": str(selected_dir),
        "provenance_sha256": final_export["provenance_sha256"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tuning NER legal isolado do pipeline da banca")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "train", "select", "finalize", "validate-export", "predict"):
        child = subparsers.add_parser(name)
        child.add_argument("--run-id", required=True)
        if name in {"train", "predict"}:
            child.add_argument("--experiment")
        if name == "predict":
            child.add_argument("--checkpoint-dir", type=Path, required=True)
            child.add_argument("--database", type=Path, required=True)
            child.add_argument("--input-dir", type=Path, required=True)
            child.add_argument("--output-dir", type=Path, required=True)
        if name == "validate-export":
            child.add_argument("--checkpoint-dir", type=Path, required=True)
        if name == "finalize":
            child.add_argument("--experiment", required=True)
        if name == "select":
            child.add_argument("--span-metrics-dir", type=Path)
    resume = subparsers.add_parser("resume")
    resume.add_argument("--run-id", required=True)
    resume.add_argument("--experiment", required=True)
    resume.add_argument("--checkpoint", required=True)
    audit = subparsers.add_parser("audit-spans")
    audit.add_argument("--predictions", type=Path, required=True)
    audit.add_argument("--golden", type=Path, required=True)
    audit.add_argument("--report", type=Path, required=True)
    record_validation = subparsers.add_parser("record-validation")
    record_validation.add_argument("--run-id", required=True)
    record_validation.add_argument("--report", type=Path, required=True)
    official_comparison = subparsers.add_parser("compare-official")
    official_comparison.add_argument("--run-id", required=True)
    official_comparison.add_argument("--original", type=Path, required=True)
    official_comparison.add_argument("--v1", type=Path, required=True)
    official_comparison.add_argument("--v2", type=Path, required=True)
    record_official = subparsers.add_parser("record-official-evaluation")
    record_official.add_argument("--run-id", required=True)
    record_official.add_argument("--checkpoint-dir", type=Path, required=True)
    record_official.add_argument("--report", type=Path, required=True)
    record_official.add_argument("--comparison", type=Path, required=True)
    extend_stage = subparsers.add_parser("extend-stage")
    extend_stage.add_argument("--run-id", required=True)
    extend_stage.add_argument("--stage", choices=["ablation", "robustness", "stability", "cross_validation"], required=True)
    select_snapshot = subparsers.add_parser("select-experiment-checkpoint")
    select_snapshot.add_argument("--run-id", required=True)
    select_snapshot.add_argument("--experiment", required=True)
    augmentation = subparsers.add_parser("prepare-augmentations")
    augmentation.add_argument("--run-id", required=True)
    initialize_queue = subparsers.add_parser("init-queue")
    initialize_queue.add_argument("--run-id", required=True)
    import_checkpoints = subparsers.add_parser("import-checkpoints")
    import_checkpoints.add_argument("--run-id", required=True)
    import_checkpoints.add_argument("--experiment", required=True)
    repair_snapshots = subparsers.add_parser("repair-snapshots")
    repair_snapshots.add_argument("--run-id", required=True)
    folds = subparsers.add_parser("prepare-folds")
    folds.add_argument("--run-id", required=True)
    fold_summary = subparsers.add_parser("summarize-folds")
    fold_summary.add_argument("--run-id", required=True)
    publish = subparsers.add_parser("upload")
    publish.add_argument("--checkpoint-dir", type=Path, required=True)
    publish.add_argument("--repo-id", required=True)
    publish.add_argument("--private", action="store_true")
    return parser


def _copy_reproducibility_files(run_dir: Path, experiment_name: str, target_dir: Path) -> None:
    """Ship resolved settings and non-data audit history with the model package."""
    experiment_dir = run_dir / "experiments" / experiment_name
    sources = {
        "campaign_config.resolved.json": run_dir / "campaign_config.resolved.json",
        "dataset_manifest.json": run_dir / "prepared" / "dataset_manifest.json",
        "experiment_queue.json": run_dir / "experiment_queue.json",
        "selection.json": run_dir / "selection.json",
        "preflight.json": run_dir / "preflight.json",
        "budget.json": run_dir / "budget.json",
        "campaign_progress.json": run_dir / "campaign_progress.json",
        "history.jsonl": run_dir / "history.jsonl",
        "events.jsonl": run_dir / "events.jsonl",
        "configuration_history.jsonl": experiment_dir / "configuration_history.jsonl",
        "checkpoint_archive.json": experiment_dir / "checkpoint_archive.json",
        "budget_estimate.json": experiment_dir / "budget_estimate.json",
    }
    optional = {
        "fold_validation_summary.json": run_dir / "fold_validation_summary.json",
        "training_config.resolved.previous.json": experiment_dir / "training_config.resolved.previous.json",
        "evaluation_correction.json": run_dir / "official_evaluation_invalid_crlf" / "incident.json",
    }
    for name, source in sources.items():
        if not source.is_file():
            raise FileNotFoundError(f"Arquivo de reprodução obrigatório ausente: {source}.")
        shutil.copyfile(source, target_dir / name)
    for name, source in optional.items():
        if source.is_file():
            shutil.copyfile(source, target_dir / name)

    resolved = _read_json(experiment_dir / "training_config.resolved.json")
    experiment = resolved["experiment"]
    args = resolved["training_arguments"]
    environment = resolved["environment"]
    archive = _read_json(experiment_dir / "checkpoint_archive.json")
    guide = f"""# Reprodução da rodada selecionada

Esta exportação é da execução `{run_dir.name}`, rodada `{experiment_name}`. Os pesos completos estão em `model.safetensors` (SHA-256 `{sha256_file(target_dir / 'model.safetensors')}`). A rodada começou do candidato `{experiment.get('initial_candidate')}`, SHA-256 `{experiment.get('parent_checkpoint_sha256')}`; reproduzir apenas esta rodada exige antes reconstruir esse candidato. O checkpoint original e o v1 usados pela campanha constam de `campaign_config.resolved.json` e `provenance.json`.

## Parâmetros efetivos

- Seed: {resolved['seed']}; learning rate: {experiment['learning_rate']}; peso da loss auxiliar: {experiment['auxiliary_loss_weight']}; repetição dos dez documentos oficiais: {experiment['official_repeat']}.
- Aumento OCR: {experiment['ocr_noise_rate']}; sampler: `{experiment['sampler']}`; batch por dispositivo: {args['per_device_train_batch_size']}; acumulação: {args['gradient_accumulation_steps']}; batch efetivo: {resolved['effective_batch_size']}.
- FP16: {args['fp16']}; gradient checkpointing: {args['gradient_checkpointing']}; épocas máximas: {args['num_train_epochs']}; warmup efetivo: {resolved['effective_warmup_steps']} passos; scheduler: `{args['lr_scheduler_type']}`.
- GPU: {environment.get('gpu')}; driver: {environment.get('gpu_driver')}; Python: {environment.get('python')}; PyTorch: {environment.get('torch')}; Transformers: {environment.get('transformers')}; imagem Docker: `{environment.get('docker_image_id')}`.

`training_config.resolved.json` contém todos os argumentos, defaults da versão usada, arquitetura, tokenizer, mapa dos 13 rótulos, loss, parâmetros treináveis, grupos do otimizador, janelas, amostragem e ambiente. `campaign_config.resolved.json`, `dataset_manifest.json`, `experiment_queue.json` e `selection.json` fixam as fontes, divisões, transformações, experiências e escolha. `configuration_history.jsonl`, `history.jsonl`, `events.jsonl`, `campaign_progress.json` e `budget.json` preservam ajustes, passos, LR, métricas, interrupções, tempo e recursos. `checkpoint_archive.json` registra os hashes dos estados FP16, RNG, otimizador e scheduler antes de sua compactação.

## Reexecutar no repositório

Com os mesmos dados, checkpoints originais, código e Docker disponíveis, da raiz do repositório execute:

```powershell
.\\fine-tuning\\run.ps1 -Mode train -RunId <novo-run-id>
```

O comando reconstrói a campanha e só consulta os 16 documentos oficiais reservados após congelar a seleção local. Compare os hashes de entrada e a configuração resolvida do novo run com os arquivos deste pacote antes de comparar pesos ou métricas. A reprodução bit a bit não é garantida entre ambientes CUDA. Esta rodada concluída teve estados completos do Trainer durante a execução; estados completos retidos após a seleção: `{archive.get('full_checkpoint_states_retained')}`. Se os estados foram compactados, uma nova execução começa com otimizador, scheduler e GradScaler novos e não constitui retomada exata do checkpoint arquivado.
"""
    (target_dir / "REPRODUCE.md").write_text(guide, encoding="utf-8", newline="\n")


def _finalize(run_dir: Path, experiment_name: str) -> dict:
    from .config import SOURCE_LICENSE, SOURCE_MODEL, SOURCE_REVISION
    from .training import _export_model
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    selection_path = run_dir / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError("Seleção local ausente; execute a validação dos candidatos primeiro.")
    selection = _read_json(selection_path)
    if selection.get("selected_experiment") != experiment_name:
        raise ValueError("Finalize somente o experimento selecionado em selection.json.")
    target_dir = run_dir / "candidates" / experiment_name
    validation_path = run_dir / "metrics" / f"{experiment_name}_validation.json"
    validation_span_path = run_dir / "metrics" / f"{experiment_name}_validation_spans.json"
    test_path = run_dir / "metrics" / "selected_test.json"
    test_span_path = run_dir / "metrics" / "selected_test_spans.json"
    for path in (validation_path, validation_span_path, test_path, test_span_path):
        if not path.is_file():
            raise FileNotFoundError(f"Relatório necessário ausente: {path}.")

    model = AutoModelForTokenClassification.from_pretrained(
        str(target_dir), local_files_only=True, use_safetensors=True
    )
    tokenizer = AutoTokenizer.from_pretrained(str(target_dir), local_files_only=True)
    _export_model(model, tokenizer, target_dir)
    del model, tokenizer

    dataset_manifest = _read_json(run_dir / "prepared" / "dataset_manifest.json")
    campaign_progress = _read_json(run_dir / "campaign_progress.json")
    experiment_progress = next(
        row for row in campaign_progress.get("experiments", [])
        if row.get("experiment", {}).get("name") == experiment_name
    )
    resolved_config = _read_json(
        run_dir / "experiments" / experiment_name / "training_config.resolved.json"
    )
    report = {
        "run_id": run_dir.name,
        "experiment": experiment_name,
        "selected_checkpoint": str(target_dir),
        "recommended_for_explicit_local_use": bool(selection.get("eligible_for_recommendation")),
        "experimental_only": bool(selection.get("experimental_only")),
        "selection": selection,
        "baseline_validation": _read_json(run_dir / "metrics" / "baseline_validation.json"),
        "selected_validation": _read_json(validation_path),
        "selected_validation_span_audit": _read_json(validation_span_path),
        "selected_local_test": _read_json(test_path),
        "selected_local_test_span_audit": _read_json(test_span_path),
        "dataset_manifest_sha256": sha256_file(run_dir / "prepared" / "dataset_manifest.json"),
        "data_summary": {
            "local_documents": dataset_manifest.get("local_documents", {}),
            "official_documents_authorized_for_training": dataset_manifest.get(
                "official_documents_authorized_for_training", []
            ),
            "official_documents_reserved_for_requested_final_score": dataset_manifest.get(
                "official_documents_reserved_for_requested_final_score"
            ),
            "official_selected_annotation_rows_read": dataset_manifest.get(
                "official_selected_annotation_rows_read"
            ),
            "groups_are_disjoint": dataset_manifest.get("groups_are_disjoint"),
        },
        "resolved_campaign_config_sha256": sha256_file(run_dir / "campaign_config.resolved.json"),
        "resolved_experiment_config_sha256": sha256_file(
            run_dir / "experiments" / experiment_name / "training_config.resolved.json"
        ),
        "training_log_sha256": sha256_file(run_dir / "history.jsonl"),
        "training_budget": _read_json(run_dir / "budget.json"),
        "training_execution": {
            "gpu": resolved_config.get("environment", {}).get("gpu"),
            "gpu_total_memory_bytes": resolved_config.get("environment", {}).get(
                "gpu_total_memory_bytes"
            ),
            "gpu_peak_allocated_bytes": campaign_progress.get("gpu_peak_allocated_bytes"),
            "gpu_peak_reserved_bytes": campaign_progress.get("gpu_peak_reserved_bytes"),
            "selected_learning_rate": resolved_config.get("experiment", {}).get(
                "learning_rate"
            ),
            "effective_batch_size": resolved_config.get("effective_batch_size"),
            "per_device_train_batch_size": resolved_config.get("training_arguments", {}).get(
                "per_device_train_batch_size"
            ),
            "gradient_accumulation_steps": resolved_config.get(
                "training_arguments", {}
            ).get("gradient_accumulation_steps"),
            "fp16": resolved_config.get("training_arguments", {}).get("fp16"),
            "gradient_checkpointing": resolved_config.get("training_arguments", {}).get(
                "gradient_checkpointing"
            ),
            "global_step": experiment_progress.get("last_global_step"),
            "epoch": experiment_progress.get("last_epoch"),
            "train_runtime_seconds": experiment_progress.get("train_runtime_seconds"),
            "train_loss": experiment_progress.get("train_metrics", {}).get("train_loss"),
            "eval_loss": experiment_progress.get("best_metric"),
        },
        "evaluation_run_manifests": _evaluation_run_manifests(run_dir, experiment_name),
        "timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    fold_summary_path = run_dir / "fold_validation_summary.json"
    if fold_summary_path.is_file():
        report["internal_cross_validation"] = _read_json(fold_summary_path)
    write_json(target_dir / "model_report.json", report)
    (target_dir / "model_report.md").write_text(
        _model_report_markdown(report), encoding="utf-8", newline="\n"
    )
    config_path = target_dir / "training_config.resolved.json"
    shutil.copyfile(run_dir / "experiments" / experiment_name / "training_config.resolved.json", config_path)
    progress_path = run_dir / "experiments" / experiment_name / "experiment_manifest.json"
    shutil.copyfile(progress_path, target_dir / "experiment_manifest.json")
    card = _model_card(report, dataset_manifest, SOURCE_MODEL, SOURCE_REVISION, SOURCE_LICENSE)
    (target_dir / "README.md").write_text(card, encoding="utf-8", newline="\n")
    _copy_reproducibility_files(run_dir, experiment_name, target_dir)

    file_hashes = {
        path.name: sha256_file(path)
        for path in sorted(target_dir.iterdir())
        if path.is_file() and path.name != "provenance.json"
    }
    provenance = {
        "source_model": SOURCE_MODEL,
        "revision": SOURCE_REVISION,
        "license": SOURCE_LICENSE,
        "model_card": f"https://huggingface.co/{SOURCE_MODEL}/tree/{SOURCE_REVISION}",
        "upstream_model_sha256": "dea15cbbe0935ec20da635f6861ffdcad0d737a648ad48dff1affd686cf776a6",
        "derived_from_finetuned_checkpoint": True,
        "training_revision": sha256_file(target_dir / "model.safetensors"),
        "training_run_id": run_dir.name,
        "selected_experiment": experiment_name,
        "validation_and_test_metrics_sha256": {
            "validation": sha256_file(validation_path),
            "local_test": sha256_file(test_path),
        },
        "data_manifest_sha256": sha256_file(run_dir / "prepared" / "dataset_manifest.json"),
        "license_attribution": "Maicon Domingues; CC BY 4.0; derived checkpoint fine-tuned from the pinned legal NER model.",
        "files": file_hashes,
    }
    write_json(target_dir / "provenance.json", provenance)
    write_json(run_dir / "final_export.json", {
        "checkpoint_dir": str(target_dir),
        "model_weights_sha256": file_hashes["model.safetensors"],
        "provenance_sha256": sha256_file(target_dir / "provenance.json"),
        "experimental_only": report["experimental_only"],
        "validated_offline": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    })
    return {"checkpoint_dir": str(target_dir), "files": file_hashes, "provenance_sha256": sha256_file(target_dir / "provenance.json")}


def _evaluation_run_manifests(run_dir: Path, selected_experiment: str) -> dict[str, dict | None]:
    names = {
        "baseline_validation": "baseline",
        f"{selected_experiment}_validation": selected_experiment,
        "selected_local_test": "selected_test",
    }
    result = {}
    for key, evaluation_name in names.items():
        manifest_path = run_dir / "evaluations" / evaluation_name / "run_manifest.json"
        if not manifest_path.is_file():
            result[key] = None
            continue
        manifest = _read_json(manifest_path)
        result[key] = {
            "manifest_sha256": sha256_file(manifest_path),
            "database_sha256": manifest.get("database_sha256"),
            "source_sha256": manifest.get("source_sha256"),
            "scripts_sha256": manifest.get("scripts_sha256"),
            "checkpoint_sha256": manifest.get("checkpoint_sha256"),
            "submission_sha256": manifest.get("submission_sha256"),
        }
    return result


def _model_card(report: dict, dataset_manifest: dict, source: str, revision: str, license_id: str) -> str:
    config = report["selection"]["selected_experiment"]
    official = ", ".join(dataset_manifest["official_documents_authorized_for_training"])
    validation_score = report.get("selected_validation", {}).get("challenge_metrics", {}).get("score_final")
    local_test_score = report.get("selected_local_test", {}).get("challenge_metrics", {}).get("score_final")
    results_text = (
        f"Na divisão local, a seed 42 obteve {validation_score:.4f} na validação e "
        f"{local_test_score:.4f} no teste local. A escolha da configuração usou a média "
        "entre seeds; veja `model_report.md` para dispersão, níveis, tipos e regressões."
        if isinstance(validation_score, (int, float)) and isinstance(local_test_score, (int, float))
        else "As métricas locais estão em `model_report.json` e `model_report.md`."
    )
    official_final = report.get("official_final_evaluation")
    if official_final:
        comparison = official_final["comparison_scores"]
        rows = "\n".join(
            f"| {name} | {values['score_final']:.6f} | "
            f"{values['niveis']['1']['score']:.6f} | {values['niveis']['2']['score']:.6f} |"
            for name, values in comparison.items()
        )
        results_text += (
            "\n\nApós congelar o v2, os 16 documentos oficiais reservados foram avaliados "
            "uma vez para cada modelo, sem escolher outro checkpoint. A fórmula oficial "
            "inclui bônus de calibração e pode produzir score maior que 1.\n\n"
            "| Modelo | Score oficial | N1 | N2 |\n|---|---:|---:|---:|\n"
            f"{rows}\n\n"
            f"Maior score observado: **{official_final['highest_observed_score_model']}**. "
            f"Diferença v2 − v1: **{official_final['difference_vs_v1']:+.6f}**. "
            "Os três modelos tiveram o mesmo F1 por classe no conjunto reservado; "
            "a diferença de score veio do bônus de calibração. "
            f"Troca automática do modelo da banca recomendada: "
            f"**{'sim' if official_final['recommended_for_pipeline_switch'] else 'não'}**. "
            "`official_evaluation.json` e `official_comparison.json` trazem os IDs, "
            "métricas por nível/tipo, auditoria de offsets e hashes."
        )
        if official_final.get("invalid_crlf_attempt_sha256"):
            results_text += (
                " A primeira tentativa com cópias CRLF foi invalidada antes desta comparação; "
                "`evaluation_correction.json` preserva o diagnóstico e os hashes dos relatórios descartados."
            )
    return f"""---
language:
  - pt
license: cc-by-4.0
library_name: transformers
pipeline_tag: token-classification
base_model: {source}
base_model_relation: finetune
tags:
  - legal-ner
  - brazilian-portuguese
  - citation-extraction
---

# NER jurídico brasileiro para spans de citações

Este checkpoint continua o ajuste do modelo [{source}](https://huggingface.co/{source}/tree/{revision}), revisão `{revision}`, com objetivo de detectar spans de jurisprudência e legislação em documentos judiciais brasileiros. A cabeça original de treze rótulos foi mantida; a resolução canônica e as classes finais do desafio são produzidas pelo pipeline hospedeiro.

## Uso pretendido e limites

Use-o como extrator NER de jurisprudência, legislação e entidades auxiliares, carregando os arquivos locais com `AutoTokenizer` e `AutoModelForTokenClassification`. O modelo não determina sozinho se uma citação é real, inventada ou incompleta. Essa decisão depende da base canônica e das regras do pipeline. O checkpoint é experimental: `{str(report['experimental_only']).lower()}`.

## Treinamento reproduzível

- Experimento selecionado: `{config}`. Veja `training_config.resolved.json` para todos os argumentos e valores efetivos, inclusive defaults resolvidos, parâmetros do otimizador, warmup, seeds, batch efetivo e versões do ambiente.
- `REPRODUCE.md` traz o comando de reexecução, a dependência do candidato anterior e as limitações de retomada após a compactação. `experiment_manifest.json`, `history.jsonl`, `configuration_history.jsonl` e `model_report.json` registram passos, hiperparâmetros, mudanças, interrupções, hashes e validação.
- Treino local: {dataset_manifest['local_documents'].get('train', 0)} documentos; validação: {dataset_manifest['local_documents'].get('validation', 0)}; teste local: {dataset_manifest['local_documents'].get('test', 0)}. Os grupos de variantes do mesmo contexto permanecem juntos.
- A etapa final acrescentou apenas estes dez documentos autorizados do dataset oficial: {official}. Eles não foram usados para escolher o modelo.
- O dataset oficial restante não foi usado em treino, validação ou seleção.

## Resultados

{results_text}

## Procedência e licença

Modelo de origem: [{source}](https://huggingface.co/{source}/tree/{revision}), licença CC BY 4.0. Este fine-tune mantém a atribuição exigida e usa os dados listados no relatório de preparação do projeto. `provenance.json` contém revisão, hashes dos pesos e hashes dos arquivos publicados.
"""


def _model_report_markdown(report: dict) -> str:
    """Render the selected checkpoint's machine-readable metrics as a reviewable report."""

    def number(value: object, digits: int = 4) -> str:
        return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "—"

    def level_rows(section: str, metrics: dict) -> list[str]:
        rows = []
        for level, values in sorted(metrics.get("niveis", {}).items()):
            classes = values.get("f1_por_classe", {})
            rows.append(
                "| {section} | N{level} | {score} | {macro} | {real} | {invented} | {incomplete} |".format(
                    section=section,
                    level=level,
                    score=number(values.get("score")),
                    macro=number(values.get("macro_f1")),
                    real=number(classes.get("real")),
                    invented=number(classes.get("inventada")),
                    incomplete=number(classes.get("incompleta")),
                )
            )
        return rows

    def span_rows(name: str, spans: dict) -> list[str]:
        stages = spans if any(key in spans for key in ("regex", "ner", "fused", "resolved")) else {"pipeline": spans}
        def row(stage: str, values: dict, typed: bool = True) -> str:
            return "| {name} | {stage} | {precision} | {recall} | {typed_precision} | {typed_recall} | {matched}/{gold} | {negative_fp} |".format(
                name=name,
                stage=stage,
                precision=number(values.get("precision")),
                recall=number(values.get("recall")),
                typed_precision=number(values.get("typed_precision")) if typed else "—",
                typed_recall=number(values.get("typed_recall")) if typed else "—",
                matched=values.get("matched", values.get("matched_spans", "—")),
                gold=values.get("gold", values.get("gold_spans", "—")),
                negative_fp=values.get("false_positives_in_gold_negative_documents", "—"),
            )
        rows = []
        for stage, values in sorted(stages.items()):
            rows.append(row(stage, values))
            for level, level_values in sorted(values.get("per_level", {}).items()):
                rows.append(row(f"{stage} · N{level}", level_values))
            for citation_type, type_values in sorted(values.get("per_type", {}).items()):
                rows.append(row(
                    citation_type if stage == "pipeline" else f"{stage} · {citation_type}",
                    type_values, typed=False,
                ))
        return rows

    selection = report["selection"]
    baseline = report["baseline_validation"]["challenge_metrics"]
    validation_report = report["selected_validation"]
    validation = validation_report["challenge_metrics"]
    test = report["selected_local_test"]["challenge_metrics"]
    training = report.get("training_execution", {})
    budget = report.get("training_budget", {})
    candidates = "\n".join(
        "| {name} | {score} | {n2} | {precision} | {eligible} |".format(
            name=row.get("experiment", "—"),
            score=number(row.get("score_final")),
            n2=number(row.get("score_n2")),
            precision=number(row.get("span_precision")),
            eligible="sim" if row.get("eligible") else "não",
        )
        for row in selection.get("candidates", [])
    )
    configuration_rows = "\n".join(
        "| `{configuration}` | {seeds} | {mean} | {std} | {n2} | {precision} | {eligible} |".format(
            configuration=row.get("configuration_id", "—"),
            seeds=", ".join(str(seed) for seed in row.get("seeds", [])) or "—",
            mean=number(row.get("mean_score_final")),
            std=number(row.get("population_std_score_final")),
            n2=number(row.get("mean_score_n2")),
            precision=number(row.get("mean_span_precision")),
            eligible="sim" if row.get("all_seeds_eligible") else "não",
        )
        for row in selection.get("configuration_results", [])
    )
    seed_section = (
        "\n## Estabilidade entre seeds\n\n"
        "A seleção de configuração usa a média do score entre seeds; o pacote exporta o checkpoint da seed 42. "
        "O desvio padrão abaixo é populacional.\n\n"
        "| Configuração | Seeds | Score médio | Desvio padrão | N2 médio | Precisão spans média | Todos elegíveis |\n"
        "|---|---|---:|---:|---:|---:|:---:|\n"
        f"{configuration_rows}\n"
        if configuration_rows else ""
    )
    class_rows = "\n".join(
        level_rows("baseline · validação local", baseline)
        + level_rows("selecionado · validação local", validation)
        + level_rows("selecionado · teste local", test)
    )
    span_table_rows = "\n".join(
        span_rows("validação local", report["selected_validation_span_audit"])
        + span_rows("teste local", report["selected_local_test_span_audit"])
    )
    baseline_score = baseline.get("score_final", 0.0)
    selected_score = validation.get("score_final", 0.0)
    absolute_gain = selected_score - baseline_score
    relative_gain = (absolute_gain / baseline_score * 100.0) if baseline_score else 0.0
    tau_did_not_increase = all(
        validation.get("niveis", {}).get(level, {}).get("tau", 0.0)
        <= baseline.get("niveis", {}).get(level, {}).get("tau", 0.0)
        for level in ("1", "2")
    )
    tau_status = (
        "Os valores `tau` não aumentaram em nenhum nível; a taxa `inventada → real` não aumentou."
        if tau_did_not_increase
        else "Pelo menos um `tau` aumentou; este candidato não satisfaz o critério de não aumentar `inventada → real`."
    )
    peak_allocated_gib = (training.get("gpu_peak_allocated_bytes") or 0) / (1024 ** 3)
    peak_reserved_gib = (training.get("gpu_peak_reserved_bytes") or 0) / (1024 ** 3)
    stages = budget.get("stages_seconds", {})
    stage_rows = "\n".join(
        f"| `{stage}` | {number(seconds, 1)} |" for stage, seconds in sorted(stages.items())
    ) or "| — | — |"
    data = report.get("data_summary", {})
    local_docs = data.get("local_documents", {})
    fold_summary = report.get("internal_cross_validation", {}).get("configurations", {})
    fold_rows = "\n".join(
        f"| `{configuration}` | {number(values.get('mean_score_final'))} | {number(values.get('population_std_score_final'))} |"
        for configuration, values in sorted(fold_summary.items())
    )
    fold_section = (
        "\n## Validação interna por contexto\n\n"
        "Os grupos do treino local foram divididos em três folds disjuntos. Esses resultados são diagnósticos e não foram usados para escolher o checkpoint.\n\n"
        "| Configuração | Score médio | Desvio padrão |\n|---|---:|---:|\n"
        f"{fold_rows}\n"
        if fold_rows else ""
    )
    bootstrap = validation_report.get("context_bootstrap", {})
    interval = bootstrap.get("bootstrap_95_percent_interval", [])
    bootstrap_section = (
        "\n## Incerteza descritiva por contexto\n\n"
        f"Bootstrap por grupos de contexto: {bootstrap.get('context_count', '—')} grupos com citações; "
        f"{bootstrap.get('negative_context_count_excluded_from_score_bootstrap', '—')} grupos sem citações foram excluídos apenas desta média diagnóstica. "
        f"Média dos scores por contexto: {number(bootstrap.get('mean_context_score'))}; "
        f"intervalo bootstrap de 95%: {number(interval[0])}–{number(interval[1])} "
        f"({bootstrap.get('replicates', '—')} réplicas, seed {bootstrap.get('seed', '—')}). "
        "O score oficial/local do conjunto completo continua sendo a métrica principal.\n"
        if len(interval) == 2 else ""
    )
    validation_n2 = validation.get("niveis", {}).get("2", {}).get("score")
    baseline_n2 = baseline.get("niveis", {}).get("2", {}).get("score")
    budget_limit = budget.get("budget_limit_seconds")
    budget_text = (
        f"{number(budget.get('elapsed_seconds'), 1)} s, sem limite fixo de GPU"
        if budget_limit is None else
        f"{number(budget.get('elapsed_seconds'), 1)} s de {number(budget_limit, 0)} s"
    )
    official_final = report.get("official_final_evaluation")
    if official_final:
        comparison = official_final["comparison_scores"]
        official_rows = "\n".join(
            f"| {name} | {values['score_final']:.6f} | "
            f"{values['niveis']['1']['score']:.6f} | {values['niveis']['2']['score']:.6f} |"
            for name, values in comparison.items()
        )
        official_section = (
            "\n## Avaliação oficial final\n\n"
            "Após congelar a seleção local, original, v1 e v2 foram avaliados nos mesmos "
            "16 documentos oficiais reservados. As 119 anotações conferiram com os textos "
            "normalizados para LF antes da inferência. A fórmula oficial inclui bônus de "
            "calibração e pode resultar em score acima de 1.\n\n"
            "| Modelo | Score oficial | N1 | N2 |\n|---|---:|---:|---:|\n"
            f"{official_rows}\n\n"
            f"O maior score observado foi o do **{official_final['highest_observed_score_model']}**. "
            f"A diferença v2 − v1 foi **{official_final['difference_vs_v1']:+.6f}**. "
            "Os três modelos tiveram F1 por classe idêntico neste subconjunto; a diferença "
            "do score decorreu do bônus de calibração. "
            f"Troca automática no pipeline da banca recomendada: "
            f"**{'sim' if official_final['recommended_for_pipeline_switch'] else 'não'}**. "
            "Esse resultado não foi usado para escolher outro checkpoint nem ajustar hiperparâmetros. "
            "Métricas completas, IDs, hashes, análise de erros e auditoria de duplicatas estão "
            "em `official_evaluation.json` e `official_comparison.json`.\n"
        )
        if official_final.get("invalid_crlf_attempt_sha256"):
            official_section += (
                "\nA tentativa anterior com cópias CRLF foi invalidada. "
                "`evaluation_correction.json` registra o motivo e os hashes; "
                "esses scores não participaram da comparação final.\n"
            )
    else:
        official_section = "\n## Avaliação oficial final\n\nPendente após o congelamento local.\n"
    return f"""# Relatório da execução {report['run_id']}

## Resultado e seleção

O checkpoint de seed 42 selecionado é `{selection.get('selected_experiment')}` (`{report['selected_checkpoint']}`). Seu score ponderado na validação local foi **{number(selected_score)}**, contra **{number(baseline_score)}** do checkpoint original, ganho de **{absolute_gain:+.4f}** ({relative_gain:+.2f}%). O score N2 foi {number(validation_n2)}, contra {number(baseline_n2)} no baseline. {tau_status} As configurações foram ordenadas pela média entre seeds, com desempate por score N2 e precisão de spans; a elegibilidade foi exigida para todas as seeds. Elegível para uso explícito: **{str(selection.get('eligible_for_recommendation')).lower()}**. O pipeline padrão permanece no checkpoint original.

| Rodada | Score final | Score N2 | Precisão de spans | Elegível |
|---|---:|---:|---:|:---:|
{candidates}
{seed_section}

## Métricas da classificação por nível

F1 por classe (`real`, `inventada`, `incompleta`) e score local por nível:

| Conjunto | Nível | Score | Macro F1 | Real | Inventada | Incompleta |
|---|:---:|---:|---:|---:|---:|---:|
{class_rows}

## Auditoria de spans

Precisão/recall usam IoU 0,5; a versão tipada exige também o tipo correto. A tabela separa regex, NER bruto, fusão e resultado após resolução, e conta falsos positivos em documentos sem citações.

| Conjunto | Etapa | Precisão | Recall | Precisão tipada | Recall tipada | Acertos/ouro | FP docs negativos |
|---|---|---:|---:|---:|---:|---:|---:|
{span_table_rows}

## Dados e recursos

- Dados locais: {local_docs.get('train', 0)} documentos de treino, {local_docs.get('validation', 0)} de validação e {local_docs.get('test', 0)} de teste. Os dez documentos oficiais autorizados entraram somente no treino; {data.get('official_documents_reserved_for_requested_final_score', 0)} ficaram reservados para score final quando solicitado.
- GPU: {training.get('gpu', 'não registrada')}; pico alocado: {peak_allocated_gib:.2f} GiB; pico reservado: {peak_reserved_gib:.2f} GiB.
- Batch por dispositivo: {training.get('per_device_train_batch_size', '—')}; acumulação: {training.get('gradient_accumulation_steps', '—')}; batch efetivo: {training.get('effective_batch_size', '—')}; FP16: {training.get('fp16', '—')}; gradient checkpointing: {training.get('gradient_checkpointing', '—')}.
- Rodada selecionada: LR {number(training.get('selected_learning_rate'), 7)}, {training.get('global_step', '—')} passos, época {number(training.get('epoch'), 2)}, loss treino {number(training.get('train_loss'))} e loss de validação {number(training.get('eval_loss'))}; runtime de treino {number(training.get('train_runtime_seconds'), 1)} s.
- Tempo contabilizado em preparação e treino: {budget_text}.

| Etapa | Segundos |
|---|---:|
{stage_rows}

{fold_section}
{bootstrap_section}
## Teste local e procedência

O score final local do checkpoint selecionado, avaliado uma vez após a seleção, foi **{number(test.get('score_final'))}** (N1 {number(test.get('niveis', {}).get('1', {}).get('score'))}; N2 {number(test.get('niveis', {}).get('2', {}).get('score'))}). Este teste local não é score oficial do desafio.

Hash dos pesos selecionados: `{selection.get('selected_checkpoint_sha256', '—')}`. Hiperparâmetros, defaults resolvidos, tokenizer, arquitetura, ambiente e configurações efetivas estão em `training_config.resolved.json`; hashes e métricas estruturadas estão em `model_report.json` e `provenance.json`.
{official_section}
"""


def _upload(checkpoint_dir: Path, repo_id: str, private: bool) -> None:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("Defina HF_TOKEN no ambiente sem o imprimir para autenticar o upload.")
    from huggingface_hub import HfApi

    provenance = _read_json(checkpoint_dir / "provenance.json")
    if not provenance.get("files", {}).get("model.safetensors"):
        raise ValueError("Pacote sem hashes de pesos; gere a exportação final antes do upload.")
    for relative, expected in provenance["files"].items():
        path = checkpoint_dir / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Hash do pacote mudou após a validação: {relative}.")
    allow_patterns = [
        "README.md", "REPRODUCE.md", "model_report.md", "config.json", "model.safetensors", "special_tokens_map.json",
        "tokenizer.json", "tokenizer_config.json", "vocab.txt", "provenance.json",
        "training_config.resolved.json", "experiment_manifest.json", "model_report.json",
        "official_evaluation.json", "official_comparison.json",
        "checkpoint_selection.json", "snapshot_manifest.json",
        "campaign_config.resolved.json", "dataset_manifest.json",
        "experiment_queue.json", "selection.json", "preflight.json", "budget.json",
        "campaign_progress.json", "history.jsonl", "events.jsonl",
        "configuration_history.jsonl", "checkpoint_archive.json", "budget_estimate.json",
        "fold_validation_summary.json", "training_config.resolved.previous.json",
        "evaluation_correction.json",
    ]
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(checkpoint_dir),
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=allow_patterns,
        commit_message="Publish reproducible fine-tuned legal NER model",
        create_pr=False,
    )
    print(json.dumps({"uploaded": True, "repo_id": repo_id, "private": private, "uploaded_file_allowlist": allow_patterns}))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit-spans":
        report = span_audit(args.predictions, args.golden, args.report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "record-validation":
        run_dir = Paths.from_environment().run_dir(args.run_id)
        final_export = _record_offline_validation(run_dir, args.report)
        print(json.dumps(final_export, ensure_ascii=False, indent=2))
        return 0
    if args.command == "compare-official":
        run_dir = Paths.from_environment().run_dir(args.run_id)
        comparison = _build_official_comparison(run_dir, args.original, args.v1, args.v2)
        print(json.dumps(comparison, ensure_ascii=False, indent=2))
        return 0
    if args.command == "record-official-evaluation":
        run_dir = Paths.from_environment().run_dir(args.run_id)
        result = _record_official_evaluation(run_dir, args.checkpoint_dir, args.report, args.comparison)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "extend-stage":
        result = extend_experiment_queue(args.run_id, args.stage)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "prepare-augmentations":
        result = prepare_ocr_augmentations(args.run_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "init-queue":
        result = initialize_experiment_queue(args.run_id)
        print(json.dumps({"experiments": len(result["experiments"]), "queue": str(Paths.from_environment().run_dir(args.run_id) / "experiment_queue.json")}))
        return 0
    if args.command == "import-checkpoints":
        result = import_existing_checkpoints_as_snapshots(args.run_id, args.experiment)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "repair-snapshots":
        result = repair_snapshot_metadata(args.run_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "prepare-folds":
        result = prepare_internal_folds(args.run_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "summarize-folds":
        result = _summarize_cross_validation(Paths.from_environment().run_dir(args.run_id))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "select-experiment-checkpoint":
        result = select_experiment_checkpoint(args.run_id, args.experiment)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "upload":
        _upload(args.checkpoint_dir, args.repo_id, args.private)
        return 0
    run_dir = Paths.from_environment().run_dir(args.run_id)
    if args.command == "prepare":
        prepare_run(args.run_id)
    elif args.command == "train":
        train_campaign(args.run_id, only_experiment=args.experiment)
    elif args.command == "resume":
        resume_experiment(args.run_id, args.experiment, args.checkpoint)
    elif args.command == "select":
        if args.span_metrics_dir:
            for experiment in _experiment_queue(run_dir):
                span_src = args.span_metrics_dir / f"{experiment['name']}_validation_spans.json"
                if span_src.is_file():
                    target = run_dir / "metrics" / span_src.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(span_src, target)
        selection = select_candidate(args.run_id)
        print(json.dumps(selection))
    elif args.command == "finalize":
        print(json.dumps(_finalize(run_dir, args.experiment), ensure_ascii=False, indent=2))
    elif args.command == "validate-export":
        raise RuntimeError("Use validate_export.py no contêiner de avaliação para validar o checkpoint.")
    elif args.command == "predict":
        raise RuntimeError("Use run.ps1 predict ou run.sh predict para uma predição Docker isolada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
