"""Exception hierarchy for transcription-core.

All exceptions raised by this library derive from :class:`TranscriptionError`.
"""

from __future__ import annotations


class TranscriptionError(Exception):
    """Base class for all transcription-core errors."""


class MediaTooLarge(TranscriptionError):
    """Raised when audio bytes exceed the configured size limit.

    Triggered both for the local ``max_bytes`` guard in :func:`transcribe_bytes`
    and for WhatsApp media that is too large (per metadata or after download).
    """


class NotAudio(TranscriptionError):
    """Raised when WhatsApp media metadata reports a non-audio mime type."""


class MediaFetchError(TranscriptionError):
    """Raised when fetching WhatsApp media (metadata or content) fails."""


class AllProvidersFailed(TranscriptionError):
    """Raised when every configured transcription provider has failed."""
