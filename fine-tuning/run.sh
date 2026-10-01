#!/usr/bin/env bash
set -euo pipefail

FT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$FT_DIR/.." && pwd)"
COMPOSE_FILE="$FT_DIR/compose.yaml"
RUNS_ROOT="$FT_DIR/runs"
V1_MODEL_DIR="$FT_DIR/artifacts/v1-reference"
PROJECT="juris-check-finetuning"

usage() {
  cat <<'EOF'
Uso:
  bash fine-tuning/run.sh train [run-id]
  bash fine-tuning/run.sh campaign <run-id>
  bash fine-tuning/run.sh evaluate-official <run-id>
  bash fine-tuning/run.sh resume <run-id> <experimento> <caminho/checkpoint-N>
  bash fine-tuning/run.sh predict <checkpoint-exportado> <diretorio-txt> <banco.sqlite> <saida.csv>
  bash fine-tuning/run.sh validate-export <checkpoint-exportado>
  bash fine-tuning/run.sh upload <checkpoint-exportado> <usuario/repositorio> [--private]
EOF
}

compose() {
  docker compose --project-name "$PROJECT" --file "$COMPOSE_FILE" "$@"
}

init_compose_environment() {
  export REPO_ROOT="$ROOT"
  export RUNS_HOST_DIR="$RUNS_ROOT"
  export MODEL_HOST_DIR="$ROOT/models/ner_legal_large"
  export FT_V1_MODEL="/source/v1"
  export FT_V1_HOST_DIR="$V1_MODEL_DIR"
  export FT_GIT_COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || printf unknown)"
  export FT_DOCKER_IMAGE_ID="${FT_DOCKER_IMAGE_ID:-unknown}"
  unset FT_GIT_WORKTREE_SHA256 || true
  export OFFICIAL_TRAIN_HOST_DIR="$RUNS_ROOT/_unused_official_subset"
}

assert_gpu() {
  command -v docker >/dev/null || { echo 'Docker CLI não encontrado.' >&2; exit 2; }
  [[ "$(docker info --format '{{.OSType}}')" == linux ]] || {
    echo 'Inicie o Docker Desktop em modo Linux.' >&2; exit 2;
  }
  command -v nvidia-smi >/dev/null || { echo 'nvidia-smi não encontrado.' >&2; exit 2; }
  local free_mib
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d '[:space:]')"
  [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= 8192 )) || {
    echo "A GPU tem ${free_mib:-?} MiB livres; libere pelo menos 8192 MiB." >&2; exit 2;
  }
}

stage_official_subset() {
  local run_id="$1"
  export OFFICIAL_TRAIN_HOST_DIR="$RUNS_ROOT/$run_id/sources/official"
  compose run --rm stager --run-id "$run_id"
}

score_local() {
  local run_id="$1" name="$2" split="$3" model_dir="$4"
  export MODEL_HOST_DIR="$model_dir"
  local golden text_dir
  if [[ "$split" =~ ^fold_([0-9]{2})$ ]]; then
    golden="/runs/$run_id/prepared/folds/fold_${BASH_REMATCH[1]}/gold.csv"
    text_dir="/runs/$run_id/prepared/folds/fold_${BASH_REMATCH[1]}/txt"
  else
    golden="/runs/$run_id/prepared/local_$split.csv"
    text_dir="/runs/$run_id/prepared/local/$split/txt"
  fi
  compose run --rm --entrypoint python evaluator /opt/fine-tuning/evaluate_model.py \
    --mode local --run-dir "/runs/$run_id" --name "$name" --input-dir "$text_dir" \
    --golden "$golden" --report "/runs/$run_id/metrics/${name}_$split.json"
}

trainer_cli() {
  local cpu_only="${1:-false}"
  if [[ "$cpu_only" == true ]]; then shift; compose run --rm -e CUDA_VISIBLE_DEVICES= --entrypoint python trainer -m train.cli "$@"
  else shift; compose run --rm --entrypoint python trainer -m train.cli "$@"; fi
}

queue_lines() {
  local run_id="$1"
  compose run --rm -e CUDA_VISIBLE_DEVICES= --entrypoint python trainer -c \
    'import json,sys; q=json.load(open(sys.argv[1],encoding="utf-8")); [print("\t".join((str(x["name"]),str(x.get("stage","base")),str(x.get("fold",""))))) for x in q["experiments"]]' \
    "/runs/$run_id/experiment_queue.json"
}

run_experiment_stage() {
  local run_id="$1" stage="$2" run_dir="$RUNS_ROOT/$run_id"
  local name actual_stage fold experiment_dir candidate_dir snapshot_root manifest_path
  while IFS=$'\t' read -r name actual_stage fold; do
    [[ "$actual_stage" == "$stage" ]] || continue
    experiment_dir="$run_dir/experiments/$name"
    candidate_dir="$run_dir/candidates/$name"
    snapshot_root="$experiment_dir/evaluation_checkpoints"
    local split=validation
    if [[ -n "$fold" ]]; then printf -v split 'fold_%02d' "$fold"; fi
    if [[ -f "$candidate_dir/checkpoint_selection.json" && -f "$run_dir/metrics/${name}_${split}.json" ]]; then continue; fi
    if [[ ! -f "$candidate_dir/model.safetensors" ]]; then
      trainer_cli false train --run-id "$run_id" --experiment "$name"
    fi
    if ! find "$snapshot_root" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -print -quit 2>/dev/null | grep -q .; then
      trainer_cli true import-checkpoints --run-id "$run_id" --experiment "$name"
    fi
    while IFS= read -r snapshot; do
      local step snapshot_name report
      step="${snapshot##*checkpoint-}"
      snapshot_name="snapshot_${name}_step_${step}"
      report="$run_dir/metrics/${snapshot_name}_${split}.json"
      if [[ ! -f "$report" ]]; then score_local "$run_id" "$snapshot_name" "$split" "$snapshot"; fi
    done < <(find "$snapshot_root" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' -print | sort -V)
    trainer_cli true select-experiment-checkpoint --run-id "$run_id" --experiment "$name"
  done < <(queue_lines "$run_id")
}

evaluate_official_frozen() {
  local run_id="$1" build_images="${2:-true}"
  [[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ ]] || { echo 'run-id inválido.' >&2; exit 2; }
  assert_gpu
  init_compose_environment
  export OFFICIAL_TRAIN_HOST_DIR="$RUNS_ROOT/$run_id/sources/official"
  if [[ "$build_images" == true ]]; then compose build trainer evaluator; fi
  local run_dir="$RUNS_ROOT/$run_id" selected selected_model model_name model_dir report
  [[ -f "$run_dir/selection.json" && -f "$run_dir/final_export.json" ]] || {
    echo 'A escolha local e a exportação validada são obrigatórias antes do score oficial.' >&2; exit 2;
  }
  selected="$(compose run --rm -e CUDA_VISIBLE_DEVICES= --entrypoint python trainer -c '
import hashlib,json,sys
from pathlib import Path
run=Path(sys.argv[1]); selection=json.loads((run/"selection.json").read_text()); export=json.loads((run/"final_export.json").read_text())
name=selection["selected_experiment"]; model=run/"candidates"/name/"model.safetensors"
digest=hashlib.sha256(model.read_bytes()).hexdigest()
assert export.get("validated_offline") and digest==selection["selected_checkpoint_sha256"]==export["model_weights_sha256"]
assert export["checkpoint_dir"]==str(run/"candidates"/name)
print(name)
' "/runs/$run_id")"
  selected_model="$run_dir/candidates/$selected"
  for model_name in original v1 v2; do
    case "$model_name" in
      original) model_dir="$ROOT/models/ner_legal_large" ;;
      v1) model_dir="$V1_MODEL_DIR" ;;
      v2) model_dir="$selected_model" ;;
    esac
    export MODEL_HOST_DIR="$model_dir"
    report="/runs/$run_id/metrics/official_$model_name.json"
    compose --profile final-evaluation run --rm --entrypoint python official-evaluator \
      /opt/fine-tuning/evaluate_model.py --mode official --run-dir "/runs/$run_id" \
      --name "official_$model_name" --report "$report"
  done
  trainer_cli true compare-official --run-id "$run_id" --original "/runs/$run_id/metrics/official_original.json" \
    --v1 "/runs/$run_id/metrics/official_v1.json" --v2 "/runs/$run_id/metrics/official_v2.json"
  trainer_cli true record-official-evaluation --run-id "$run_id" --checkpoint-dir "/runs/$run_id/candidates/$selected" \
    --report "/runs/$run_id/metrics/official_v2.json" --comparison "/runs/$run_id/official_comparison.json"
  export MODEL_HOST_DIR="$selected_model"
  compose run --rm --entrypoint python evaluator /opt/fine-tuning/validate_export.py \
    /app/models/ner_legal_large --report "/runs/$run_id/offline_export_validation_final.json"
  trainer_cli true record-validation --run-id "$run_id" --report "/runs/$run_id/offline_export_validation_final.json"
  printf 'Comparação oficial finalizada e pacote validado: %s\n' "$selected_model"
}

run_campaign_workflow() {
  local run_id="$1"
  [[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ ]] || { echo 'run-id inválido.' >&2; exit 2; }
  assert_gpu
  mkdir -p "$RUNS_ROOT/$run_id"
  init_compose_environment
  compose build trainer evaluator
  export FT_DOCKER_IMAGE_ID="$(docker image inspect juris-check-finetuning-trainer:local --format '{{.Id}}' 2>/dev/null || printf unknown)"
  stage_official_subset "$run_id"
  local run_dir="$RUNS_ROOT/$run_id"
  if [[ ! -f "$run_dir/prepared/dataset_manifest.json" ]]; then trainer_cli false prepare --run-id "$run_id"; fi
  trainer_cli true init-queue --run-id "$run_id"
  trainer_cli true repair-snapshots --run-id "$run_id"
  if [[ ! -f "$run_dir/metrics/baseline_validation.json" ]]; then
    score_local "$run_id" baseline validation "$ROOT/models/ner_legal_large"
  fi
  if [[ ! -f "$run_dir/metrics/baseline_v1_validation.json" ]]; then
    score_local "$run_id" baseline_v1 validation "$V1_MODEL_DIR"
  fi

  run_experiment_stage "$run_id" base
  trainer_cli true extend-stage --run-id "$run_id" --stage ablation
  run_experiment_stage "$run_id" ablation
  trainer_cli false prepare-augmentations --run-id "$run_id"
  trainer_cli true extend-stage --run-id "$run_id" --stage robustness
  run_experiment_stage "$run_id" robustness
  trainer_cli true extend-stage --run-id "$run_id" --stage stability
  run_experiment_stage "$run_id" stability
  trainer_cli true prepare-folds --run-id "$run_id"
  trainer_cli true extend-stage --run-id "$run_id" --stage cross_validation
  run_experiment_stage "$run_id" cross_validation
  trainer_cli true summarize-folds --run-id "$run_id"
  trainer_cli true select --run-id "$run_id"

  local selected selected_model
  selected="$(compose run --rm -e CUDA_VISIBLE_DEVICES= --entrypoint python trainer -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["selected_experiment"])' "/runs/$run_id/selection.json")"
  selected_model="$run_dir/candidates/$selected"
  if [[ ! -f "$run_dir/metrics/selected_test.json" ]]; then
    score_local "$run_id" selected_test test "$selected_model"
    mv -f "$run_dir/metrics/selected_test_test.json" "$run_dir/metrics/selected_test.json"
    [[ ! -f "$run_dir/metrics/selected_test_test_spans.json" ]] || mv -f "$run_dir/metrics/selected_test_test_spans.json" "$run_dir/metrics/selected_test_spans.json"
  fi
  trainer_cli true finalize --run-id "$run_id" --experiment "$selected"
  export MODEL_HOST_DIR="$selected_model"
  compose run --rm --entrypoint python evaluator /opt/fine-tuning/validate_export.py /app/models/ner_legal_large --report "/runs/$run_id/offline_export_validation.json"
  trainer_cli true record-validation --run-id "$run_id" --report "/runs/$run_id/offline_export_validation.json"

  evaluate_official_frozen "$run_id" false
  printf 'Campanha independente finalizada: %s\n' "$selected_model"
}

run_training_workflow() {
  local run_id="$1" mode="$2" experiment="${3:-}" checkpoint="${4:-}"
  [[ "$run_id" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ ]] || {
    echo 'run-id deve conter somente letras, números, ponto, sublinhado ou hífen.' >&2; exit 2;
  }
  assert_gpu
  mkdir -p "$RUNS_ROOT"
  init_compose_environment
  export OFFICIAL_TRAIN_HOST_DIR="$RUNS_ROOT/$run_id/sources/official"
  compose build trainer evaluator
  export FT_DOCKER_IMAGE_ID="$(docker image inspect juris-check-finetuning-trainer:local --format '{{.Id}}' 2>/dev/null || printf unknown)"

  local run_dir="$RUNS_ROOT/$run_id"
  if [[ "$mode" == train ]]; then
    if [[ -e "$run_dir" ]]; then
      [[ -f "$run_dir/prepared/dataset_manifest.json" ]] || {
        echo "A execução já existe sem preparação completa: $run_dir. Use outro ID." >&2; exit 2;
      }
      if [[ -d "$run_dir/experiments" ]] && find "$run_dir/experiments" -type d -name 'checkpoint-*' -print -quit | grep -q .; then
        echo "A execução já tem checkpoints retomáveis: $run_dir. Use resume com um checkpoint completo." >&2; exit 2;
      fi
    else
      mkdir -p "$run_dir"
    fi
  else
    [[ -f "$run_dir/prepared/dataset_manifest.json" ]] || {
      echo "A execução $run_id não tem dados preparados para retomada." >&2; exit 2;
    }
  fi
  stage_official_subset "$run_id"

  if [[ "$mode" == train && ! -f "$run_dir/campaign_config.resolved.json" ]]; then
    compose run --rm trainer prepare --run-id "$run_id"
  elif [[ "$mode" == resume ]]; then
    [[ "$experiment" =~ ^(original|v1)_lr_(5e-06|1e-05|2e-05|3e-05)$ ]] || { echo 'Experimento inválido para resume.' >&2; exit 2; }
    local experiment_root="$run_dir/experiments/$experiment"
    checkpoint="$(cd -- "$checkpoint" && pwd)"
    case "$checkpoint/" in "$experiment_root"/*) ;; *) echo 'Checkpoint fora do experimento desta execução.' >&2; exit 2;; esac
    for state in optimizer.pt scheduler.pt trainer_state.json rng_state.pth scaler.pt; do
      [[ -f "$checkpoint/$state" ]] || { echo "Checkpoint incompleto: falta $state." >&2; exit 2; }
    done
    compose run --rm trainer resume --run-id "$run_id" --experiment "$experiment" \
      --checkpoint "/runs/$run_id/experiments/$experiment/$(basename -- "$checkpoint")"
  fi

  score_local "$run_id" baseline validation "$ROOT/models/ner_legal_large"
  score_local "$run_id" baseline_v1 validation "$V1_MODEL_DIR"
  if [[ "$mode" == train ]]; then
    compose run --rm trainer train --run-id "$run_id"
  fi

  local count=0 candidate
  for candidate in original_lr_5e-06 original_lr_1e-05 original_lr_2e-05 original_lr_3e-05 \
    v1_lr_5e-06 v1_lr_1e-05 v1_lr_2e-05 v1_lr_3e-05; do
    if [[ -f "$run_dir/candidates/$candidate/model.safetensors" ]]; then
      score_local "$run_id" "$candidate" validation "$run_dir/candidates/$candidate"
      count=$((count + 1))
    fi
  done
  (( count > 0 )) || { echo 'Nenhum checkpoint completo foi produzido.' >&2; exit 2; }
  compose run --rm -e CUDA_VISIBLE_DEVICES= trainer select --run-id "$run_id"

  local selected
  selected="$(compose run --rm -e CUDA_VISIBLE_DEVICES= --entrypoint python trainer \
    -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected_experiment"])' \
    "/runs/$run_id/selection.json")"
  local selected_model="$run_dir/candidates/$selected"
  if [[ ! -f "$run_dir/metrics/selected_test.json" ]]; then
    score_local "$run_id" selected_test test "$selected_model"
    mv -f "$run_dir/metrics/selected_test_test.json" "$run_dir/metrics/selected_test.json"
    if [[ -f "$run_dir/metrics/selected_test_test_spans.json" ]]; then
      mv -f "$run_dir/metrics/selected_test_test_spans.json" "$run_dir/metrics/selected_test_spans.json"
    fi
  fi
  compose run --rm -e CUDA_VISIBLE_DEVICES= trainer finalize --run-id "$run_id" --experiment "$selected"
  export MODEL_HOST_DIR="$selected_model"
  compose run --rm --entrypoint python evaluator /opt/fine-tuning/validate_export.py \
    /app/models/ner_legal_large --report "/runs/$run_id/offline_export_validation.json"
  compose run --rm -e CUDA_VISIBLE_DEVICES= trainer record-validation --run-id "$run_id" \
    --report "/runs/$run_id/offline_export_validation.json"

  local v1_model="$V1_MODEL_DIR" model_name model_dir report
  for model_name in original v1 v2; do
    case "$model_name" in
      original) model_dir="$ROOT/models/ner_legal_large" ;;
      v1) model_dir="$v1_model" ;;
      v2) model_dir="$selected_model" ;;
    esac
    export MODEL_HOST_DIR="$model_dir"
    report="/runs/$run_id/metrics/official_$model_name.json"
    compose --profile final-evaluation run --rm --entrypoint python official-evaluator \
      /opt/fine-tuning/evaluate_model.py --mode official --run-dir "/runs/$run_id" \
      --name "official_$model_name" --report "$report"
  done
  compose run --rm -e CUDA_VISIBLE_DEVICES= --entrypoint python trainer -m train.cli compare-official \
    --run-id "$run_id" --original "/runs/$run_id/metrics/official_original.json" \
    --v1 "/runs/$run_id/metrics/official_v1.json" --v2 "/runs/$run_id/metrics/official_v2.json"
  compose run --rm -e CUDA_VISIBLE_DEVICES= --entrypoint python trainer -m train.cli record-official-evaluation \
    --run-id "$run_id" --checkpoint-dir "/runs/$run_id/candidates/$selected" \
    --report "/runs/$run_id/metrics/official_v2.json" --comparison "/runs/$run_id/official_comparison.json"
  printf 'Pacote exportado: %s\nRelatório: %s\n' "$selected_model" "$selected_model/model_report.json"
  printf 'Comparação oficial: %s\n' "$selected_model/official_comparison.json"
}

predict_with_checkpoint() {
  local checkpoint="$1" input_dir="$2" database="$3" output_file="$4"
  assert_gpu
  checkpoint="$(cd -- "$checkpoint" && pwd)"
  input_dir="$(cd -- "$input_dir" && pwd)"
  database="$(cd -- "$(dirname -- "$database")" && pwd)/$(basename -- "$database")"
  [[ -f "$database" ]] || { echo "Banco SQLite não encontrado: $database" >&2; exit 2; }
  [[ -d "$input_dir" ]] || { echo "Diretório de entrada não encontrado: $input_dir" >&2; exit 2; }
  local output_dir output_name
  output_dir="$(mkdir -p -- "$(dirname -- "$output_file")" && cd -- "$(dirname -- "$output_file")" && pwd)"
  output_name="$(basename -- "$output_file")"
  [[ -n "$output_name" && "$output_name" != . && "$output_name" != .. ]] || { echo 'Saída precisa ser um arquivo.' >&2; exit 2; }
  case "$output_dir/" in "$input_dir/"*) echo 'A saída não pode ficar dentro da pasta de entrada.' >&2; exit 2;; esac
  [[ "$output_dir/$output_name" != "$database" ]] || { echo 'A saída não pode sobrescrever o banco SQLite.' >&2; exit 2; }
  init_compose_environment
  compose build evaluator
  docker run --rm --gpus all --network none \
    --mount "type=bind,source=$input_dir,target=/input/txt,readonly" \
    --mount "type=bind,source=$database,target=/input/reference.sqlite,readonly" \
    --mount "type=bind,source=$output_dir,target=/output" \
    --mount "type=bind,source=$checkpoint,target=/app/models/ner_legal_large,readonly" \
    --env PROTECTED_DATA_DIR=/input --env "OUTPUT_FILE=$output_name" \
    --entrypoint sh juris-check-finetuning-evaluator:local -ec \
    'python /opt/fine-tuning/validate_export.py /app/models/ner_legal_large && exec python -m src.pipeline --input /input/txt --db /input/reference.sqlite --output /output --output-file "$OUTPUT_FILE"'
  printf 'Submissão gerada: %s/%s\n' "$output_dir" "$output_name"
}

upload_checkpoint() {
  local checkpoint="$1" repo_id="$2" private="${3:-}"
  [[ -n "${HF_TOKEN:-}" ]] || { echo 'Defina HF_TOKEN no ambiente antes do upload.' >&2; exit 2; }
  export CHECKPOINT_HOST_DIR="$(cd -- "$checkpoint" && pwd)"
  init_compose_environment
  compose build trainer
  export FT_DOCKER_IMAGE_ID="$(docker image inspect juris-check-finetuning-trainer:local --format '{{.Id}}' 2>/dev/null || printf unknown)"
  if [[ "$private" == --private ]]; then
    compose --profile publish run --rm publisher --repo-id "$repo_id" --private
  else
    compose --profile publish run --rm publisher --repo-id "$repo_id"
  fi
}

validate_checkpoint() {
  local checkpoint="$1"
  checkpoint="$(cd -- "$checkpoint" && pwd)"
  init_compose_environment
  export CHECKPOINT_HOST_DIR="$checkpoint"
  compose build validator
  compose --profile validation run --rm validator
}

main() {
  local mode="${1:-}"
  [[ -n "$mode" ]] || { usage; exit 2; }
  shift
  case "$mode" in
    train)
      local run_id="${1:-ft-$(date +%Y%m%d-%H%M%S)}"
      run_campaign_workflow "$run_id"
      ;;
    campaign)
      [[ $# -eq 1 ]] || { usage; exit 2; }
      run_campaign_workflow "$1"
      ;;
    evaluate-official)
      [[ $# -eq 1 ]] || { usage; exit 2; }
      evaluate_official_frozen "$1"
      ;;
    resume)
      [[ $# -eq 3 ]] || { usage; exit 2; }
      local run_id="$1" experiment="$2" checkpoint="$3"
      [[ "$experiment" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$ ]] || { echo 'Experimento inválido.' >&2; exit 2; }
      assert_gpu
      init_compose_environment
      export OFFICIAL_TRAIN_HOST_DIR="$RUNS_ROOT/$run_id/sources/official"
      compose build trainer evaluator
      local experiment_root="$RUNS_ROOT/$run_id/experiments/$experiment"
      experiment_root="$(cd -- "$experiment_root" && pwd)"
      checkpoint="$(cd -- "$checkpoint" && pwd)"
      case "$checkpoint/" in "$experiment_root"/*) ;; *) echo 'Checkpoint fora do experimento.' >&2; exit 2;; esac
      for state in optimizer.pt scheduler.pt trainer_state.json rng_state.pth scaler.pt; do
        [[ -f "$checkpoint/$state" ]] || { echo "Checkpoint incompleto: $state ausente." >&2; exit 2; }
      done
      local relative_checkpoint="${checkpoint#"$RUNS_ROOT/"}"
      trainer_cli false resume --run-id "$run_id" --experiment "$experiment" --checkpoint "/runs/$relative_checkpoint"
      run_campaign_workflow "$run_id"
      ;;
    predict)
      [[ $# -eq 4 ]] || { usage; exit 2; }
      predict_with_checkpoint "$1" "$2" "$3" "$4"
      ;;
    validate-export)
      [[ $# -eq 1 ]] || { usage; exit 2; }
      validate_checkpoint "$1"
      ;;
    upload)
      [[ $# -ge 2 && $# -le 3 ]] || { usage; exit 2; }
      upload_checkpoint "$1" "$2" "${3:-}"
      ;;
    *) usage; exit 2;;
  esac
}

main "$@"
