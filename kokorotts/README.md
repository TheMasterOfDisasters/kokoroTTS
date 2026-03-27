# kokorotts

Python UI + API application for Kokoro TTS.

## Run without Docker

```bash
python -m pip install -e .
python -m pip install -r kokorotts/requirements.txt
python kokorotts/app.py
```

- UI: `http://localhost:7860/`
- API ping: `GET /tts/ping`
- API synthesis: `POST /tts/convert`

## Docker and Task workflow

From repository root:

```bash
task image
task imagerun
task imageweb
task imageapi
```

For hot-swapping local app files into the running container:

```bash
task localrun
task logs
```

`localrun` mounts the full local `kokorotts/` directory into `/app/kokorotts` and enables auto-reload via `UVICORN_RELOAD=1`.

Optional runtime env vars:

- `HF_TOKEN`: Hugging Face access token for higher hub rate limits.
- `KOKORO_REPO_ID`: override model repo (default `hexgrad/Kokoro-82M`).

