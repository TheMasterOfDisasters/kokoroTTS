# KokoroTTS (Docker) - UI + API Ready

This image is built for users who want a practical, self-hosted Kokoro deployment with both a Web UI and an HTTP API available immediately after `docker run`.

- Docker Hub image: https://hub.docker.com/r/sensejworld/kokorotts
- Fork project: https://github.com/TheMasterOfDisasters/kokoroTTS
- Original upstream project: https://github.com/hexgrad/kokoro

## Why choose this image variant

- UI and API are served from one container, one port, out of the box.
- API-first usage is ready immediately (`/tts/ping`, `/tts/convert`) while keeping an interactive UI at `/`.
- GPU-aware runtime options are included (`auto`, `cpu`, `cuda:N`).
- Image is prepared for offline serving after build by prefetching model + voice assets.

Container serves both:
- Gradio UI on `/`
- HTTP API on `/tts/*`

Default port: `7860`

## Quick Start

CPU:
```bash
docker run -p 7860:7860 sensejworld/kokorotts:latest
```

NVIDIA GPU (all visible GPUs):
```bash
docker run -p 7860:7860 --gpus all sensejworld/kokorotts:latest
```

Single GPU by index (example host GPU 1):
```bash
docker run -p 7860:7860 --gpus "device=1" -e CUDA_VISIBLE_DEVICES=1 sensejworld/kokorotts:latest
```

Open UI:
- `http://localhost:7860`

## API Examples

Ping:
```bash
curl -sS http://localhost:7860/tts/ping
```

Generate WAV (auto device):
```bash
curl -X POST "http://localhost:7860/tts/convert" \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world from Kokoro.","voice":"af_heart","speed":1.0,"device":"auto"}' \
  --output hello.wav
```

Force CPU:
```bash
curl -X POST "http://localhost:7860/tts/convert" \
  -H "Content-Type: application/json" \
  -d '{"text":"CPU synthesis","voice":"af_heart","speed":1.0,"device":"cpu"}' \
  --output cpu.wav
```

## Runtime Environment Variables

- `KOKORO_REPO_ID` (default: `hexgrad/Kokoro-82M`)
- `KOKOROTTS_DEVICE` (default: `auto`; options: `auto`, `cpu`, `cuda:0`, `cuda:1`, ...)
- `CUDA_VISIBLE_DEVICES` (Docker/NVIDIA visibility control)
- `HF_TOKEN` (optional, for higher Hugging Face rate limits)
- `PORT` (default: `7860`)

## Offline Behavior

The Docker build prefetches model/config + UI voice packs into the image cache.
Runtime sets Hugging Face offline env flags so the app can serve without internet access after image build.

## Notes

- First request can still take longer due to initial model/device warm-up.
- API endpoint currently returns WAV stream from `/tts/convert`.
- Recommended image tag: `sensejworld/kokorotts:latest`.

