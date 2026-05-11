"""German G2P support for KokoroTTS."""

from __future__ import annotations

from typing import Tuple

from loguru import logger

from .de_normalizer import normalize_text_de


class DEG2P:
    """Normalize German text, then phonemize with misaki's espeak-ng wrapper."""

    def __init__(self) -> None:
        try:
            from misaki import espeak as _espeak

            self._g2p = _espeak.EspeakG2P(language="de")
        except ImportError as exc:
            raise ImportError("misaki is required for German G2P support.") from exc
        except Exception as exc:
            raise RuntimeError(
                "espeak-ng is required for German G2P support. Install the system package before using lang_code='d'. "
                f"Original error: {exc}"
            ) from exc

    def __call__(self, text: str) -> Tuple[str, list]:
        normalized = normalize_text_de(text)
        logger.debug(f"DEG2P normalized text: {normalized[:80]}")
        result = self._g2p(normalized)
        if isinstance(result, tuple):
            return result
        return result, []
