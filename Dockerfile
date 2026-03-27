FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    HOST=0.0.0.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends espeak-ng ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE /app/
COPY kokoro /app/kokoro
COPY kokorotts /app/kokorotts

RUN python -m pip install --upgrade pip \
    && python -m pip install -e . \
    && python -m pip install -r /app/kokorotts/requirements.txt

EXPOSE 7860

CMD ["python", "-u", "kokorotts/app.py"]

