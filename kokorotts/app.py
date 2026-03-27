import io
import os
import random
import wave
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from kokoro import KModel, KPipeline

SAMPLE_RATE = 24000
DEFAULT_REPO_ID = os.getenv("KOKORO_REPO_ID", "hexgrad/Kokoro-82M")
DATA_DIR = Path(__file__).resolve().parent
CUDA_AVAILABLE = torch.cuda.is_available()

models = {
    gpu: KModel(repo_id=DEFAULT_REPO_ID).to("cuda" if gpu else "cpu").eval()
    for gpu in [False] + ([True] if CUDA_AVAILABLE else [])
}
pipelines = {
    lang_code: KPipeline(lang_code=lang_code, repo_id=DEFAULT_REPO_ID, model=False)
    for lang_code in "ab"
}
pipelines["a"].g2p.lexicon.golds["kokoro"] = "kˈOkəɹO"
pipelines["b"].g2p.lexicon.golds["kokoro"] = "kˈQkəɹQ"


def forward_gpu(ps, ref_s, speed):
    return models[True](ps, ref_s, speed)


def synthesize_full(text, voice="af_heart", speed=1, use_gpu=CUDA_AVAILABLE):
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE
    audio_chunks = []
    phoneme_chunks = []

    for _, ps, _ in pipeline(text, voice, speed):
        ref_s = pack[len(ps) - 1]
        try:
            if use_gpu:
                audio = forward_gpu(ps, ref_s, speed)
            else:
                audio = models[False](ps, ref_s, speed)
        except gr.exceptions.Error as exc:
            if use_gpu:
                gr.Warning(str(exc))
                gr.Info("Retrying with CPU. To avoid this error, change Hardware to CPU.")
                audio = models[False](ps, ref_s, speed)
            else:
                raise gr.Error(exc)
        audio_chunks.append(audio.numpy())
        phoneme_chunks.append(ps)

    if not audio_chunks:
        return None, ""

    merged_audio = np.concatenate(audio_chunks)
    merged_ps = "\n".join(phoneme_chunks)
    return (SAMPLE_RATE, merged_audio), merged_ps


def generate_first(text, voice="af_heart", speed=1, use_gpu=CUDA_AVAILABLE):
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE
    for _, ps, _ in pipeline(text, voice, speed):
        ref_s = pack[len(ps) - 1]
        try:
            if use_gpu:
                audio = forward_gpu(ps, ref_s, speed)
            else:
                audio = models[False](ps, ref_s, speed)
        except gr.exceptions.Error as exc:
            if use_gpu:
                gr.Warning(str(exc))
                gr.Info("Retrying with CPU. To avoid this error, set Hardware to CPU.")
                audio = models[False](ps, ref_s, speed)
            else:
                raise gr.Error(exc)
        return (SAMPLE_RATE, audio.numpy()), ps
    return None, ""


def tokenize_first(text, voice="af_heart"):
    pipeline = pipelines[voice[0]]
    for _, ps, _ in pipeline(text, voice):
        return ps
    return ""


def predict(text, voice="af_heart", speed=1):
    return generate_first(text, voice, speed, use_gpu=False)[0]


def generate_all(text, voice="af_heart", speed=1, use_gpu=CUDA_AVAILABLE):
    pipeline = pipelines[voice[0]]
    pack = pipeline.load_voice(voice)
    use_gpu = use_gpu and CUDA_AVAILABLE
    first = True
    for _, ps, _ in pipeline(text, voice, speed):
        ref_s = pack[len(ps) - 1]
        try:
            if use_gpu:
                audio = forward_gpu(ps, ref_s, speed)
            else:
                audio = models[False](ps, ref_s, speed)
        except gr.exceptions.Error as exc:
            if use_gpu:
                gr.Warning(str(exc))
                gr.Info("Switching to CPU")
                audio = models[False](ps, ref_s, speed)
            else:
                raise gr.Error(exc)
        yield SAMPLE_RATE, audio.numpy()
        if first:
            first = False
            yield SAMPLE_RATE, torch.zeros(1).numpy()


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    return buffer.getvalue()


def get_random_quote():
    with (DATA_DIR / "en.txt").open("r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]
    return random.choice(lines)


def get_gatsby():
    return (DATA_DIR / "gatsby5k.md").read_text(encoding="utf-8").strip()


def get_frankenstein():
    return (DATA_DIR / "frankenstein5k.md").read_text(encoding="utf-8").strip()


CHOICES = {
    "🇺🇸 🚺 Heart ❤️": "af_heart",
    "🇺🇸 🚺 Bella 🔥": "af_bella",
    "🇺🇸 🚺 Nicole 🎧": "af_nicole",
    "🇺🇸 🚺 Aoede": "af_aoede",
    "🇺🇸 🚺 Kore": "af_kore",
    "🇺🇸 🚺 Sarah": "af_sarah",
    "🇺🇸 🚺 Nova": "af_nova",
    "🇺🇸 🚺 Sky": "af_sky",
    "🇺🇸 🚺 Alloy": "af_alloy",
    "🇺🇸 🚺 Jessica": "af_jessica",
    "🇺🇸 🚺 River": "af_river",
    "🇺🇸 🚹 Michael": "am_michael",
    "🇺🇸 🚹 Fenrir": "am_fenrir",
    "🇺🇸 🚹 Puck": "am_puck",
    "🇺🇸 🚹 Echo": "am_echo",
    "🇺🇸 🚹 Eric": "am_eric",
    "🇺🇸 🚹 Liam": "am_liam",
    "🇺🇸 🚹 Onyx": "am_onyx",
    "🇺🇸 🚹 Santa": "am_santa",
    "🇺🇸 🚹 Adam": "am_adam",
    "🇬🇧 🚺 Emma": "bf_emma",
    "🇬🇧 🚺 Isabella": "bf_isabella",
    "🇬🇧 🚺 Alice": "bf_alice",
    "🇬🇧 🚺 Lily": "bf_lily",
    "🇬🇧 🚹 George": "bm_george",
    "🇬🇧 🚹 Fable": "bm_fable",
    "🇬🇧 🚹 Lewis": "bm_lewis",
    "🇬🇧 🚹 Daniel": "bm_daniel",
}
for voice_id in CHOICES.values():
    pipelines[voice_id[0]].load_voice(voice_id)

TOKEN_NOTE = """
💡 Customize pronunciation with Markdown link syntax and /slashes/ like `[Kokoro](/kˈOkəɹO/)`

💬 To adjust intonation, try punctuation `;:,.!?—…"()“”` or stress `ˈ` and `ˌ`

⬇️ Lower stress `[1 level](-1)` or `[2 levels](-2)`

⬆️ Raise stress 1 level `[or](+2)` 2 levels (only works on less stressed, usually short words)
"""

with gr.Blocks() as generate_tab:
    out_audio = gr.Audio(label="Output Audio", interactive=False, streaming=False, autoplay=True)
    generate_btn = gr.Button("Generate", variant="primary")
    with gr.Accordion("Output Tokens", open=True):
        out_ps = gr.Textbox(
            interactive=False,
            show_label=False,
            info="Tokens used to generate the audio, up to 510 context length.",
        )
        tokenize_btn = gr.Button("Tokenize", variant="secondary")
        gr.Markdown(TOKEN_NOTE)
        predict_btn = gr.Button("Predict", variant="secondary", visible=False)

STREAM_NOTE = ["⚠️ There is an unknown Gradio bug that might yield no audio the first time you click `Stream`."]
STREAM_NOTE = "\n\n".join(STREAM_NOTE)

with gr.Blocks() as stream_tab:
    out_stream = gr.Audio(label="Output Audio Stream", interactive=False, streaming=True, autoplay=True)
    with gr.Row():
        stream_btn = gr.Button("Stream", variant="primary")
        stop_btn = gr.Button("Stop", variant="stop")
    with gr.Accordion("Note", open=True):
        gr.Markdown(STREAM_NOTE)
        gr.DuplicateButton()

with gr.Blocks(title="KokoroTTS") as ui:
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label="Input Text", info="Arbitrarily many characters supported")
            with gr.Row():
                # Non-filterable dropdown avoids browser password manager misclassification.
                voice = gr.Dropdown(
                    choices=list(CHOICES.items()),
                    value="af_heart",
                    label="Voice",
                    info="Quality and availability vary by language",
                    filterable=False,
                    allow_custom_value=False,
                )
                use_gpu = gr.Dropdown(
                    [("ZeroGPU 🚀", True), ("CPU 🐌", False)],
                    value=CUDA_AVAILABLE,
                    label="Hardware",
                    info="GPU is usually faster, but has a usage quota",
                    interactive=CUDA_AVAILABLE,
                )
            speed = gr.Slider(minimum=0.5, maximum=2, value=1, step=0.1, label="Speed")
            random_btn = gr.Button("🎲 Random Quote 💬", variant="secondary")
            with gr.Row():
                gatsby_btn = gr.Button("🥂 Gatsby 📕", variant="secondary")
                frankenstein_btn = gr.Button("💀 Frankenstein 📗", variant="secondary")
        with gr.Column():
            gr.TabbedInterface([generate_tab, stream_tab], ["Generate", "Stream"])

    random_btn.click(fn=get_random_quote, inputs=[], outputs=[text])
    gatsby_btn.click(fn=get_gatsby, inputs=[], outputs=[text])
    frankenstein_btn.click(fn=get_frankenstein, inputs=[], outputs=[text])
    generate_btn.click(fn=synthesize_full, inputs=[text, voice, speed, use_gpu], outputs=[out_audio, out_ps])
    tokenize_btn.click(fn=tokenize_first, inputs=[text, voice], outputs=[out_ps])
    stream_event = stream_btn.click(fn=generate_all, inputs=[text, voice, speed, use_gpu], outputs=[out_stream])
    stop_btn.click(fn=None, cancels=stream_event)
    predict_btn.click(fn=predict, inputs=[text, voice, speed], outputs=[out_audio])

api = FastAPI(title="KokoroTTS API", version="1.0")


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: str = "af_heart"
    speed: float = 1.0
    use_gpu: bool = True


@api.get("/tts/ping")
def ping() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@api.post("/tts/convert")
def convert(payload: TTSRequest) -> StreamingResponse:
    audio_tuple, _ = synthesize_full(
        text=payload.text,
        voice=payload.voice,
        speed=payload.speed,
        use_gpu=payload.use_gpu,
    )
    if audio_tuple is None:
        return StreamingResponse(io.BytesIO(b""), media_type="audio/wav")
    _, waveform = audio_tuple
    wav_bytes = audio_to_wav_bytes(waveform)
    return StreamingResponse(
        io.BytesIO(wav_bytes),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=kokoro.wav"},
    )



app = gr.mount_gradio_app(api, ui, path="/")


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    reload_enabled = os.getenv("UVICORN_RELOAD", "0").lower() in {"1", "true", "yes"}
    uvicorn.run("kokorotts.app:app", host=host, port=port, reload=reload_enabled)

