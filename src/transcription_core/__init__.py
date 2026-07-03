"""transcription-core: provider-agnostic, stateless audio transcription.

Shared by Brain Co services. Never resolves credentials, never touches a
database, never persists audio, never sends messages, never reads env vars,
and never logs — configuration comes only from ``TranscriptionConfig``.
"""

from .config import TranscriptionConfig
from .exceptions import (
    AllProvidersFailed,
    MediaFetchError,
    MediaTooLarge,
    NotAudio,
    TranscriptionError,
)
from .result import TranscriptionResult
from .transcribe import transcribe_bytes
from .whatsapp_media import fetch_whatsapp_media, transcribe_whatsapp_media

__version__ = "0.1.0"

__all__ = [
    "TranscriptionConfig",
    "TranscriptionResult",
    "transcribe_bytes",
    "fetch_whatsapp_media",
    "transcribe_whatsapp_media",
    "TranscriptionError",
    "MediaTooLarge",
    "NotAudio",
    "MediaFetchError",
    "AllProvidersFailed",
    "__version__",
]
