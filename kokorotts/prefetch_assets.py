import os

from huggingface_hub import hf_hub_download

from kokorotts.voices import voice_ids

REPO_ID = os.getenv("KOKORO_REPO_ID", "hexgrad/Kokoro-82M")

MODEL_FILE_BY_REPO = {
    "hexgrad/Kokoro-82M": "kokoro-v1_0.pth",
    "hexgrad/Kokoro-82M-v1.1-zh": "kokoro-v1_1-zh.pth",
}

VOICE_IDS = voice_ids()


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

