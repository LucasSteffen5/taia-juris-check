from __future__ import annotations

from train.cli import _model_card, _model_report_markdown


def test_model_report_markdown_includes_selection_metrics_and_resources() -> None:
    report = {
        "run_id": "report-test",
        "experimental_only": False,
        "selected_checkpoint": "/runs/report-test/candidates/lr_3e-5",
        "selection": {
            "selected_experiment": "lr_3e-5",
            "selected_checkpoint_sha256": "abc123",
            "eligible_for_recommendation": True,
            "candidates": [{
                "experiment": "lr_3e-5",
                "score_final": 0.67,
                "score_n2": 0.64,
                "span_precision": 0.85,
                "eligible": True,
            }],
            "configuration_results": [{
                "configuration_id": "stable-config",
                "seeds": [17, 42, 2026],
                "mean_score_final": 0.68,
                "population_std_score_final": 0.01,
                "mean_score_n2": 0.65,
                "mean_span_precision": 0.86,
                "all_seeds_eligible": True,
            }],
        },
        "baseline_validation": {
            "challenge_metrics": {
                "score_final": 0.65,
                "niveis": {"2": {"score": 0.61, "macro_f1": 0.6, "f1_por_classe": {}}},
            },
        },
        "selected_validation": {
            "challenge_metrics": {
                "score_final": 0.67,
                "niveis": {"2": {"score": 0.64, "macro_f1": 0.62, "f1_por_classe": {
                    "real": 0.8, "inventada": 0.5, "incompleta": 0.56,
                }}},
            },
            "context_bootstrap": {
                "context_count": 31,
                "negative_context_count_excluded_from_score_bootstrap": 6,
                "mean_context_score": 0.66,
                "bootstrap_95_percent_interval": [0.62, 0.70],
                "replicates": 1000,
                "seed": 42,
            },
        },
        "selected_local_test": {"challenge_metrics": {
            "score_final": 0.68, "niveis": {"1": {"score": 0.7}, "2": {"score": 0.66}}
        }},
        "selected_validation_span_audit": {
            "precision": 0.85, "recall": 0.75, "typed_precision": 0.84,
            "typed_recall": 0.74, "matched_spans": 85, "gold_spans": 100,
            "per_level": {"2": {"precision": 0.86, "recall": 0.76, "typed_precision": 0.85, "typed_recall": 0.75, "matched": 43, "gold": 56}},
            "per_type": {"legislacao": {"precision": 0.97, "recall": 0.83, "matched": 50, "gold": 60}},
        },
        "selected_local_test_span_audit": {
            "precision": 0.88, "recall": 0.80, "typed_precision": 0.87,
            "typed_recall": 0.79, "matched_spans": 88, "gold_spans": 110,
        },
        "data_summary": {
            "local_documents": {"train": 412, "validation": 62, "test": 56},
            "official_documents_reserved_for_requested_final_score": 16,
        },
        "training_execution": {
            "gpu": "NVIDIA GeForce RTX 3060", "gpu_peak_allocated_bytes": 2**30,
            "gpu_peak_reserved_bytes": 2**31, "selected_learning_rate": 3e-5,
            "effective_batch_size": 16, "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4, "fp16": True,
            "gradient_checkpointing": True, "global_step": 486, "epoch": 3.0,
            "train_runtime_seconds": 342.0, "train_loss": 0.03, "eval_loss": 0.39,
        },
        "training_budget": {
            "elapsed_seconds": 1100.0, "budget_limit_seconds": 36000,
            "stages_seconds": {"prepare": 60.0, "fine_tuning": 1000.0},
        },
    }

    markdown = _model_report_markdown(report)

    assert "ganho de **+0.0200**" in markdown
    assert "| baseline · validação local | N2 | 0.6100" in markdown
    assert "| validação local | legislacao | 0.9700 | 0.8300" in markdown
    assert "Os valores `tau` não aumentaram" in markdown
    assert "NVIDIA GeForce RTX 3060" in markdown
    assert "16 ficaram reservados" in markdown
    assert "Hash dos pesos selecionados: `abc123`" in markdown
    assert "foi **0.6800** (N1 0.7000; N2 0.6600)" in markdown
    assert "Pendente após o congelamento local" in markdown
    assert "Estabilidade entre seeds" in markdown
    assert "stable-config" in markdown and "0.0100" in markdown
    assert "Incerteza descritiva por contexto" in markdown
    assert "0.6200–0.7000" in markdown

    report["official_final_evaluation"] = {
        "comparison_scores": {
            name: {"score_final": score, "niveis": {
                "1": {"score": 1.09}, "2": {"score": 1.07},
            }}
            for name, score in (("original", 1.0825), ("v1", 1.0827), ("v2", 1.0826))
        },
        "highest_observed_score_model": "v1",
        "difference_vs_v1": -0.0001,
        "recommended_for_pipeline_switch": False,
    }
    with_official = _model_report_markdown(report)
    assert "O maior score observado foi o do **v1**" in with_official
    assert "Troca automática no pipeline da banca recomendada: **não**" in with_official
    assert "Pendente após o congelamento local" not in with_official

    card = _model_card(
        report,
        {"official_documents_authorized_for_training": ["gen_n1_001"],
         "local_documents": {"train": 412, "validation": 62, "test": 56}},
        "source/model", "revision", "CC-BY-4.0",
    )
    assert "0.6700 na validação" in card
    assert "0.6800 no teste local" in card
    assert "Maior score observado: **v1**" in card
    assert "Troca automática do modelo da banca recomendada: **não**" in card
