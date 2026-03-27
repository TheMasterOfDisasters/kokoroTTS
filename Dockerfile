FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential espeak-ng ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE /app/
COPY kokoro /app/kokoro
COPY kokorotts /app/kokorotts

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -e . \
    && python -m pip install -r /app/kokorotts/requirements.txt \
    && python -m pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PORT=7860 \
    HOST=0.0.0.0 \
    UVICORN_RELOAD=0

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends espeak-ng ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local /usr/local
COPY --from=builder /app /app

EXPOSE 7860

CMD ["python", "-u", "kokorotts/app.py"]

