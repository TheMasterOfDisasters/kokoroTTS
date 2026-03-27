# kokorotts

Python UI + API application for Kokoro TTS.

## Run locally

```bash
python -m pip install -e .
python -m pip install -r kokorotts/requirements.txt
python kokorotts/app.py
```

- UI: `http://localhost:7860/`
- API ping: `GET /tts/ping`
- API synthesis: `POST /tts/convert`

