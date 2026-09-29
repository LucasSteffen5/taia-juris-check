#!/usr/bin/env bash
set -euo pipefail

action="${1:-}"
if [[ $# -ne 1 || ! "$action" =~ ^(build|test|predict|score)$ ]]; then
  echo "Uso: ./run.sh {build|test|predict|score}" >&2
  exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")"

case "$action" in
  build)
    docker compose build predict
    ;;
  test)
    docker compose run --rm --build test
    if ! docker run --rm --network none --gpus all --entrypoint python juris-check:local -m scripts.smoke_ner --require-cuda; then
      echo "GPU indisponível; teste em CPU concluído."
    fi
    ;;
  predict)
    docker compose build predict
    if docker run --rm --network none --gpus all --entrypoint python juris-check:local -c 'import torch; assert torch.cuda.is_available()'; then
      docker compose up --no-build --exit-code-from predict
    else
      echo "GPU indisponível no Docker; executando em CPU."
      docker compose -f compose.yaml up --no-build --exit-code-from predict
    fi
    ;;
  score)
    docker compose run --rm --no-deps score
    ;;
esac
