"""KokoroTTS application package (UI + API)."""

__version__ = "0.0.1"

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
