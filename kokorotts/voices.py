LANGUAGE_CHOICES = {
    "a": "American English",
    "b": "British English",
    "d": "German",
    "j": "Japanese",
    "ko": "Korean",
    "z": "Mandarin Chinese",
    "e": "Spanish",
    "f": "French",
    "h": "Hindi",
    "i": "Italian",
    "p": "Brazilian Portuguese",
}

EXPERIMENTAL_VOICE_ASSETS = {
    "df_eva": {
        "repo_id": "dida-80b/kokoro-deutsch-eva-k",
        "model_repo_id": "dida-80b/kokoro-deutsch-eva-k",
        "model_file": "kokoro_german_converted.pth",
        "config_repo_id": "hexgrad/Kokoro-82M",
        "config_file": "config.json",
        "voice_file": "eva_k.pt",
        "note": "Experimental German Eva-K voicepack from dida-80b/kokoro-deutsch-eva-k.",
    },
}

VOICE_CHOICES = {
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
    "🇯🇵 🚺 Alpha": "jf_alpha",
    "🇯🇵 🚺 Gongitsune": "jf_gongitsune",
    "🇯🇵 🚺 Nezumi": "jf_nezumi",
    "🇯🇵 🚺 Tebukuro": "jf_tebukuro",
    "🇯🇵 🚹 Kumo": "jm_kumo",
    "🇨🇳 🚺 Xiaobei": "zf_xiaobei",
    "🇨🇳 🚺 Xiaoni": "zf_xiaoni",
    "🇨🇳 🚺 Xiaoxiao": "zf_xiaoxiao",
    "🇨🇳 🚺 Xiaoyi": "zf_xiaoyi",
    "🇨🇳 🚹 Yunjian": "zm_yunjian",
    "🇨🇳 🚹 Yunxi": "zm_yunxi",
    "🇨🇳 🚹 Yunxia": "zm_yunxia",
    "🇨🇳 🚹 Yunyang": "zm_yunyang",
    "🇪🇸 🚺 Dora": "ef_dora",
    "🇪🇸 🚹 Alex": "em_alex",
    "🇪🇸 🚹 Santa": "em_santa",
    "🇫🇷 🚺 Siwis": "ff_siwis",
    "🇮🇳 🚺 Alpha": "hf_alpha",
    "🇮🇳 🚺 Beta": "hf_beta",
    "🇮🇳 🚹 Omega": "hm_omega",
    "🇮🇳 🚹 Psi": "hm_psi",
    "🇮🇹 🚺 Sara": "if_sara",
    "🇮🇹 🚹 Nicola": "im_nicola",
    "🇧🇷 🚺 Dora": "pf_dora",
    "🇧🇷 🚹 Alex": "pm_alex",
    "🇧🇷 🚹 Santa": "pm_santa",
    "🇩🇪 🚺 Eva-K (experimental)": "df_eva",
}


def is_experimental_voice(voice_id: str) -> bool:
    return voice_id in EXPERIMENTAL_VOICE_ASSETS


def get_experimental_voice_asset(voice_id: str) -> dict[str, str] | None:
    return EXPERIMENTAL_VOICE_ASSETS.get(voice_id)


def voice_ids(include_experimental: bool = False) -> list[str]:
    if include_experimental:
        return list(VOICE_CHOICES.values())
    return [voice_id for voice_id in VOICE_CHOICES.values() if not is_experimental_voice(voice_id)]
