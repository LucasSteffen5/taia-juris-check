# -*- coding: utf-8 -*-
"""Avaliador oficial local executando a métrica exata da competição (kaggle_metric.py)."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import pandas as pd

# Importar kaggle_metric oficial
OFFICIAL_METRIC_DIR = Path(__file__).resolve().parent.parent / "data" / "desafio-jusbrasil-bracis-2026"
sys.path.insert(0, str(OFFICIAL_METRIC_DIR))
import kaggle_metric
from src.config import NER_MODEL_DIR, ROOT_DIR
from src.resolution.header_index import HeaderIndex


def goldenset_to_solution_df(goldenset_path: Path | str) -> pd.DataFrame:
    """Converte a planilha goldenset_offsets.csv para o DataFrame de solution esperado pelo kaggle_metric."""
    # Importante: carregar id_canonico como string para evitar conversão float64/notação científica
    df = pd.read_csv(goldenset_path, dtype={"id_canonico": str, "inicio": int, "fim": int, "nivel": int})
    rows = []
    
    # Agrupar citações por documento_id e nivel
    for (doc_id, nivel), group in df.groupby(["documento_id", "nivel"]):
        cits = []
        for _, r in group.iterrows():
            ini = int(r["inicio"])
            fim = int(r["fim"])
            cls = str(r["classificacao"]).strip().lower()
            val_id = r["id_canonico"]
            id_can = str(val_id).strip() if pd.notna(val_id) and str(val_id).strip() not in ("", "nan", "None") else "-"
            cits.append(f"{ini},{fim},{cls},{id_can}")
        rows.append({
            "documento_id": doc_id,
            "nivel": int(nivel),
            "citacoes": "|".join(cits) if cits else "-",
        })
        
    return pd.DataFrame(rows)


def evaluate(submission_path: Path | str, golden_path: Path | str) -> dict:
    """Executa a avaliação completa e detalhada da submissão."""
    sub_df = pd.read_csv(submission_path, dtype={"documento_id": str, "citacoes": str})
    sol_df = goldenset_to_solution_df(golden_path)

    resultados = kaggle_metric.avaliar(sol_df, sub_df)
    return resultados


def verify_frozen_run(submission_path: Path | str, manifest_path: Path | str) -> None:
    """Recusa pontuar saídas se o CSV, código, dependências ou checkpoint mudaram."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("offset_policy") != "LF" or manifest.get("submission_offset_space") != "LF":
        raise ValueError(
            "A execução não declara offsets LF. Gere novamente a submissão com o pipeline atualizado."
        )
    if manifest.get("offset_policy_version") != 1:
        raise ValueError("Versão de política de offsets incompatível; gere novamente a submissão.")

    output_dir = Path(submission_path).resolve().parent
    current = {
        "source_sha256": HeaderIndex.fingerprint(ROOT_DIR / "src"),
        "scripts_sha256": HeaderIndex.fingerprint(ROOT_DIR / "scripts"),
        "dependencies_sha256": HeaderIndex.fingerprint(ROOT_DIR / "requirements.txt"),
        "checkpoint_sha256": HeaderIndex.fingerprint(NER_MODEL_DIR),
        "model_weights_sha256": HeaderIndex.fingerprint(NER_MODEL_DIR / "model.safetensors"),
        "metric_sha256": HeaderIndex.fingerprint(OFFICIAL_METRIC_DIR / "kaggle_metric.py"),
        "submission_sha256": hashlib.sha256(Path(submission_path).read_bytes()).hexdigest(),
        "normalized_texts_sha256": HeaderIndex.fingerprint(output_dir / "text_lf"),
        "lf_json_sha256": HeaderIndex.fingerprint(output_dir / "json"),
        "original_json_sha256": HeaderIndex.fingerprint(output_dir / "json_original"),
    }
    for key, digest in current.items():
        if manifest.get(key) != digest:
            raise ValueError(f"A execução não está congelada; hash divergente: {key}.")


def print_report(res: dict) -> None:
    """Imprime relatório formatado dos resultados por nível e final."""
    print("=" * 65)
    print("  RESULTADO OFICIAL DE AVALIAÇÃO — CAÇA-ALUCINAÇÕES BRACIS 2026")
    print("=" * 65)
    
    niveis = res.get("niveis", {})
    n1 = niveis.get(1, {})
    n2 = niveis.get(2, {})
    
    n1_f1 = n1.get("macro_f1", 0.0)
    n1_tau = n1.get("tau", 0.0)
    n1_bonus = n1.get("b", 0.0)
    n1_brier = n1.get("brier", max(0.0, 1.0 - (n1_bonus / 0.10)) if n1_bonus > 0 else None)
    n1_score = n1.get("score", 0.0)
    
    n2_f1 = n2.get("macro_f1", 0.0)
    n2_tau = n2.get("tau", 0.0)
    n2_bonus = n2.get("b", 0.0)
    n2_brier = n2.get("brier", max(0.0, 1.0 - (n2_bonus / 0.10)) if n2_bonus > 0 else None)
    n2_score = n2.get("score", 0.0)
    
    final_score = res.get("score_final", 0.0)

    print(f"\n--- NÍVEL 1 (Texto Limpo - Peso 1x) ---")
    print(f"  Macro-F1 (3 classes):  {n1_f1:.4f}")
    print(f"  F1 por classe:         {n1.get('f1_por_classe', {})}")
    print(f"  Taxa Tau (Alucinação): {n1_tau:.4f}  (Penalidade: {0.5 * n1_tau:.4f})")
    print(f"  Brier Score:           {n1_brier:.4f}" if n1_brier is not None else "  Brier Score:           N/A")
    print(f"  Bônus de Calibração:   +{n1_bonus * 100:.2f}%")
    print(f"  SCORE NÍVEL 1:         {n1_score:.4f}")

    print(f"\n--- NÍVEL 2 (Texto com Ruído OCR - Peso 2x) ---")
    print(f"  Macro-F1 (3 classes):  {n2_f1:.4f}")
    print(f"  F1 por classe:         {n2.get('f1_por_classe', {})}")
    print(f"  Taxa Tau (Alucinação): {n2_tau:.4f}  (Penalidade: {0.5 * n2_tau:.4f})")
    print(f"  Brier Score:           {n2_brier:.4f}" if n2_brier is not None else "  Brier Score:           N/A")
    print(f"  Bônus de Calibração:   +{n2_bonus * 100:.2f}%")
    print(f"  SCORE NÍVEL 2:         {n2_score:.4f}")

    print("\n" + "=" * 65)
    print(f"  >>> PONTUAÇÃO FINAL PONDERADA: {final_score:.4f} <<<")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Avaliador Oficial Local")
    parser.add_argument("--submission", required=True, help="Caminho do submission.csv")
    parser.add_argument("--golden", required=True, help="Caminho do goldenset_offsets.csv")
    parser.add_argument("--report", type=Path, help="Grava o resultado completo em JSON")
    parser.add_argument("--manifest", required=True, help="Manifesto da inferência congelada")
    args = parser.parse_args()

    verify_frozen_run(args.submission, args.manifest)
    res = evaluate(args.submission, args.golden)
    print_report(res)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
