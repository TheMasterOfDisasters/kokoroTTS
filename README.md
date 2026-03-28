# kokoroTTS

`kokoroTTS` packages Kokoro TTS as a practical self-hosted app with:
- Gradio UI on `/`
- HTTP API on `/tts/*`
- One container, one port (`7860`)

This repository is a focused fork of the upstream project:
- This repo: https://github.com/TheMasterOfDisasters/kokoroTTS
- Upstream model/library: https://github.com/hexgrad/kokoro

## Release

Current release version: `0.0.1`

## Quick Start (Docker)

Build:

```bash
docker build -t kokorotts:local .
```

Run (CPU):

```bash
docker run -p 7860:7860 kokorotts:local
```

Run (NVIDIA GPU):

```bash
docker run -p 7860:7860 --gpus all kokorotts:local
```

Open UI at `http://localhost:7860`.

## API

Ping:

```bash
curl -sS http://localhost:7860/tts/ping
```

Synthesize:

```bash
curl -X POST "http://localhost:7860/tts/convert" \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello from kokoroTTS.","voice":"af_heart","speed":1.0,"device":"auto"}' \
  --output hello.wav
```

## Taskfile Workflow (Windows-friendly)

From repo root:

```bash
task image
task imagerun
task imageweb
task imageapi
```

Hot-swap local app code into container (no rebuild loop):

```bash
task localrun
task logs
```

## Runtime Environment Variables

- `KOKORO_REPO_ID` (default: `hexgrad/Kokoro-82M`)
- `KOKOROTTS_DEVICE` (default: `auto`; supports `cpu`, `cuda:0`, `cuda:1`, ...)
- `CUDA_VISIBLE_DEVICES` (GPU visibility control)
- `HF_TOKEN` (optional, for higher HF Hub limits)
- `PORT` (default: `7860`)
- `UVICORN_RELOAD` (`1` for auto-reload in hot-swap mode)

## Offline Behavior

- Docker build prefetches model/config + voice assets into Hugging Face cache.
- Runtime sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
- A built/pulled image is intended to run without internet access.

## Project Layout

- `kokorotts/app.py`: UI + FastAPI app mounted on one service.
- `kokorotts/model.py` and `kokorotts/pipeline.py`: core inference pipeline.
- `kokorotts/prefetch_assets.py`: build-time asset prefetch.
- `Dockerfile`: production image build/run definition.
- `Taskfile.yml`: local build/run/debug automation.
- `docs/dockerhub.md`: Docker Hub description copy.
