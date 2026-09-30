#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  echo "Uso: bash run.sh <caminho_db> <pasta_txt> <arquivo_saida>" >&2
  exit 2
}

[[ $# -eq 3 ]] || usage
[[ -n "$3" ]] || usage

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

if ! command -v docker >/dev/null 2>&1; then
  echo "Erro: Docker CLI não encontrado. Instale e inicie o Docker Desktop ou Docker Engine." >&2
  exit 1
fi

if [[ ! -f "$1" ]]; then
  echo "Erro: banco SQLite não encontrado: $1" >&2
  exit 1
fi
if [[ ! -d "$2" ]]; then
  echo "Erro: pasta de documentos não encontrada: $2" >&2
  exit 1
fi
shopt -s nullglob
txt_files=("$2"/*.txt)
shopt -u nullglob
if [[ ${#txt_files[@]} -eq 0 ]]; then
  echo "Erro: a pasta não contém arquivos .txt: $2" >&2
  exit 1
fi

database_path="$(realpath -- "$1")"
txt_dir="$(realpath -- "$2")"
output_path="$(realpath -m -- "$3")"
output_dir="$(dirname -- "$output_path")"
output_name="$(basename -- "$output_path")"
if [[ "$output_name" == "." || "$output_name" == ".." || -z "$output_name" ]]; then
  echo "Erro: informe um caminho de arquivo CSV válido." >&2
  exit 1
fi
if [[ -d "$output_path" ]]; then
  echo "Erro: o caminho de saída aponta para um diretório, não para um arquivo." >&2
  exit 1
fi
if [[ "$output_path" == "$database_path" ]]; then
  echo "Erro: o arquivo de saída não pode substituir o banco de entrada." >&2
  exit 1
fi
case "$output_path" in
  "$txt_dir"/*)
    echo "Erro: o arquivo de saída não pode ficar dentro da pasta de documentos." >&2
    exit 1
    ;;
esac
if [[ "$output_path" == "$txt_dir" ]]; then
  echo "Erro: o caminho de saída não pode substituir a pasta de documentos." >&2
  exit 1
fi
mkdir -p -- "$output_dir"
if [[ ! -w "$output_dir" ]]; then
  echo "Erro: sem permissão para gravar no diretório de saída: $output_dir" >&2
  exit 1
fi

docker_os="$(docker info --format '{{.OSType}}' 2>/dev/null)" || {
  echo "Erro: Docker não está ativo ou não pode ser acessado." >&2
  exit 1
}
if [[ "$docker_os" != "linux" ]]; then
  echo "Erro: este pipeline requer o Docker configurado para contêineres Linux." >&2
  exit 1
fi

echo "Construindo a imagem do pipeline (o primeiro build pode baixar os pesos do modelo)..."
image_tag="juris-check:local"
docker build --progress plain --tag "$image_tag" --file "$repo_dir/Dockerfile" "$repo_dir" || {
  echo "Erro: não foi possível construir a imagem Docker do pipeline." >&2
  exit 1
}
image_id="$(docker image inspect --format '{{.Id}}' "$image_tag")" || {
  echo "Erro: não foi possível identificar a imagem recém-construída." >&2
  exit 1
}
if [[ ! "$image_id" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "Erro: o build não retornou um ID de imagem Docker válido." >&2
  exit 1
fi

echo "Validando GPU NVIDIA no contêiner e executando o pipeline..."
docker run --rm --gpus all --network none \
  --mount "type=bind,source=$txt_dir,target=/input/txt,readonly" \
  --mount "type=bind,source=$database_path,target=/input/reference.sqlite,readonly" \
  --mount "type=bind,source=$output_dir,target=/output" \
  --env PROTECTED_DATA_DIR=/input \
  --env "OUTPUT_FILE=$output_name" \
  --entrypoint sh "$image_id" -ec \
  'python -m scripts.smoke_ner --require-cuda && exec python -m src.pipeline --input /input/txt --db /input/reference.sqlite --output /output --output-file "$OUTPUT_FILE"'
