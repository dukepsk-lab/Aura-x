# Aura-X service image (API + data/feature/labeling stack).
# MetaTrader5 is intentionally NOT installed here — broker connectivity runs on a
# dedicated Windows host; this image covers ingestion-consumer, features, labels,
# validation and the FastAPI surface.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt README.md ./
COPY src ./src

RUN pip install --upgrade pip && pip install -e ".[db,api,models]"

EXPOSE 8000

CMD ["uvicorn", "aurax.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
