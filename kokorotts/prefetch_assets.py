import os

from huggingface_hub import hf_hub_download

REPO_ID = os.getenv("KOKORO_REPO_ID", "hexgrad/Kokoro-82M")

MODEL_FILE_BY_REPO = {
    "hexgrad/Kokoro-82M": "kokoro-v1_0.pth",
    "hexgrad/Kokoro-82M-v1.1-zh": "kokoro-v1_1-zh.pth",
}

VOICE_IDS = [
    "af_heart",
    "af_bella",
    "af_nicole",
    "af_aoede",
    "af_kore",
    "af_sarah",
    "af_nova",
    "af_sky",
    "af_alloy",
    "af_jessica",
    "af_river",
    "am_michael",
    "am_fenrir",
    "am_puck",
    "am_echo",
    "am_eric",
    "am_liam",
    "am_onyx",
    "am_santa",
    "am_adam",
    "bf_emma",
    "bf_isabella",
    "bf_alice",
    "bf_lily",
    "bm_george",
    "bm_fable",
    "bm_lewis",
    "bm_daniel",
]


def main() -> None:
    if REPO_ID not in MODEL_FILE_BY_REPO:
        supported = ", ".join(sorted(MODEL_FILE_BY_REPO.keys()))
        raise ValueError(f"Unsupported KOKORO_REPO_ID '{REPO_ID}'. Supported: {supported}")

    files = [
        "config.json",
        MODEL_FILE_BY_REPO[REPO_ID],
        *[f"voices/{voice_id}.pt" for voice_id in VOICE_IDS],
    ]

    for filename in files:
        print(f"Prefetching {REPO_ID}:{filename}")
        hf_hub_download(repo_id=REPO_ID, filename=filename)


if __name__ == "__main__":
    main()

