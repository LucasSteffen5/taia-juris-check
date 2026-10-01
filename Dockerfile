FROM python:3.11.11-slim-bookworm@sha256:081075da77b2b55c23c088251026fb69a7b2bf92471e491ff5fd75c192fd38e5

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --disable-pip-version-check -r requirements.txt

COPY src ./src
COPY scripts/evaluate.py scripts/smoke_ner.py ./scripts/
COPY tests ./tests
COPY fine-tuning/artifacts/v2-selected ./models/ner_legal_large
COPY fine-tuning/train/validate_export.py /opt/fine-tuning/validate_export.py
COPY data/desafio-jusbrasil-bracis-2026/kaggle_metric.py ./data/desafio-jusbrasil-bracis-2026/kaggle_metric.py
RUN python /opt/fine-tuning/validate_export.py ./models/ner_legal_large

ENTRYPOINT ["python", "-m", "src.pipeline"]
