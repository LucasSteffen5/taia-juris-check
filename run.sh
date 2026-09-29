#!/usr/bin/env bash
set -euo pipefail

action="${1:-}"
if [[ $# -ne 1 || ! "$action" =~ ^(build|test|predict|score)$ ]]; then
  echo "Uso: ./run.sh {build|test|predict|score}" >&2
  exit 2
fi

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"
image="juris-check:local"
official="$root/data/desafio-jusbrasil-bracis-2026"
output="$root/output/simulation"

case "$action" in
  build)
    docker build --tag "$image" .
    ;;
  test)
    docker run --rm --network none --env CUDA_VISIBLE_DEVICES= --entrypoint python "$image" -m pytest -q
    docker run --rm --network none --env CUDA_VISIBLE_DEVICES= --entrypoint python "$image" -m scripts.smoke_ner
    if docker run --rm --network none --gpus all --entrypoint python "$image" -m scripts.smoke_ner --require-cuda; then
      :
    else
      echo "GPU indisponível para o teste; a execução em CPU foi validada."
    fi
    ;;
  predict)
    mkdir -p "$output"
    gpu_args=()
    if docker run --rm --network none --gpus all --entrypoint python "$image" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'; then
      echo "GPU NVIDIA disponível; iniciando inferência em CUDA."
      gpu_args=(--gpus all)
    else
      echo "GPU NVIDIA indisponível no Docker; iniciando inferência em CPU."
    fi
    docker run --rm --network none "${gpu_args[@]}" \
      --mount "type=bind,source=$official/txt,target=/input/txt,readonly" \
      --mount "type=bind,source=$official/desafio1_bracis.db,target=/input/desafio1_bracis.db,readonly" \
      --mount "type=bind,source=$output,target=/output" \
      "$image" --input /input/txt --db /input/desafio1_bracis.db --output /output
    ;;
  score)
    docker run --rm --network none \
      --mount "type=bind,source=$official/goldenset_offsets.csv,target=/input/goldenset_offsets.csv,readonly" \
      --mount "type=bind,source=$output,target=/output" \
      --entrypoint python "$image" -m scripts.evaluate \
      --submission /output/submission.csv \
      --golden /input/goldenset_offsets.csv \
      --manifest /output/run_manifest.json \
      --report /output/score.json
    ;;
esac
