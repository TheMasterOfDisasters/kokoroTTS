import io
import os
import wave
from typing import Dict, Tuple

import gradio as gr
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from kokoro import KModel, KPipeline

SAMPLE_RATE = 24000

VOICE_CHOICES: Dict[str, str] = {
    "US Female Heart": "af_heart",
    "US Female Bella": "af_bella",
    "US Female Nicole": "af_nicole",
    "US Female Sarah": "af_sarah",
    "US Male Michael": "am_michael",
    "US Male Fenrir": "am_fenrir",
    "US Male Puck": "am_puck",
    "UK Female Emma": "bf_emma",
    "UK Female Isabella": "bf_isabella",
    "UK Male George": "bm_george",
    "UK Male Lewis": "bm_lewis",
}


class KokoroService:
    def __init__(self) -> None:
        self.cuda_available = torch.cuda.is_available()
        self.models: Dict[bool, KModel] = {False: KModel().to("cpu").eval()}
        if self.cuda_available:
            self.models[True] = KModel().to("cuda").eval()

        self.pipelines = {lang_code: KPipeline(lang_code=lang_code, model=False) for lang_code in "ab"}

        # Preserve Kokoro custom pronunciation in both English pipelines.
        self.pipelines["a"].g2p.lexicon.golds["kokoro"] = "kˈOkəɹO"
        self.pipelines["b"].g2p.lexicon.golds["kokoro"] = "kˈQkəɹQ"

    def synthesize(self, text: str, voice: str = "af_heart", speed: float = 1.0, use_gpu: bool = True) -> Tuple[np.ndarray, str]:
        text = text.strip()
        if not text:
            raise ValueError("Input text is empty")

        lang_code = voice[0]
        if lang_code not in self.pipelines:
            raise ValueError(f"Unsupported voice language prefix '{lang_code}'")

        pipeline = self.pipelines[lang_code]
        pack = pipeline.load_voice(voice)
        selected_gpu = use_gpu and self.cuda_available
        model = self.models[selected_gpu]

        audio_chunks = []
        phoneme_chunks = []

        for result in pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+"):
            ps = result.phonemes
            if not ps:
                continue
            ref_s = pack[len(ps) - 1]
            audio = model(ps, ref_s, speed)
            audio_chunks.append(audio.cpu())
            phoneme_chunks.append(ps)

        if not audio_chunks:
            raise RuntimeError("No audio generated from provided input")

        merged = torch.cat(audio_chunks).numpy()
        return merged, "\n".join(phoneme_chunks)


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


service = KokoroService()
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
    audio, _ = service.synthesize(
        text=payload.text,
        voice=payload.voice,
        speed=payload.speed,
        use_gpu=payload.use_gpu,
    )
    wav_bytes = audio_to_wav_bytes(audio)
    return StreamingResponse(
        io.BytesIO(wav_bytes),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=kokoro.wav"},
    )


def ui_generate(text: str, voice_label: str, speed: float, use_gpu: bool):
    try:
        voice = VOICE_CHOICES[voice_label]
        audio, phonemes = service.synthesize(text=text, voice=voice, speed=speed, use_gpu=use_gpu)
        return (SAMPLE_RATE, audio), phonemes
    except Exception as exc:
        raise gr.Error(str(exc))


with gr.Blocks(title="KokoroTTS") as ui:
    gr.Markdown("# KokoroTTS")
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label="Input Text", lines=6, placeholder="Type text to synthesize")
            voice = gr.Dropdown(choices=list(VOICE_CHOICES.keys()), value="US Female Heart", label="Voice")
            speed = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.1, label="Speed")
            use_gpu = gr.Checkbox(value=True, label="Use GPU if available")
            generate = gr.Button("Generate", variant="primary")
        with gr.Column():
            audio = gr.Audio(label="Output Audio", interactive=False)
            phonemes = gr.Textbox(label="Phonemes", lines=8, interactive=False)

    generate.click(
        fn=ui_generate,
        inputs=[text, voice, speed, use_gpu],
        outputs=[audio, phonemes],
    )


app = gr.mount_gradio_app(api, ui, path="/")


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("kokorotts.app:app", host=host, port=port, reload=False)

