"""KokoroTTS application package (UI + API)."""

from pathlib import Path


def _read_version() -> str:
    version_file = Path(__file__).resolve().parent.parent / "VERSION"
    try:
        return version_file.read_text(encoding="utf-8").strip()
    except OSError:
        return "0.2-snapshot"


__version__ = _read_version()

from loguru import logger
import sys

# Replace the default logger format with concise module:line context.
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <cyan>{module:>16}:{line}</cyan> | <level>{level: >8}</level> | <level>{message}</level>",
    colorize=True,
    level="INFO",
)

logger.disable("kokorotts")

from .model import KModel
from .pipeline import KPipeline
