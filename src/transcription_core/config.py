"""Configuration for transcription-core.

``TranscriptionConfig`` is the only way callers pass settings into this library.
It never reads environment variables and never resolves credentials itself —
callers must supply already-resolved API keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

_VALID_PRIMARIES = ("openai", "groq")


@dataclass(frozen=True)
class TranscriptionConfig:
    openai_api_key: str
    openai_model: str = "gpt-4o-mini-transcribe"
    groq_api_key: str | None = None
    groq_model: str = "whisper-large-v3-turbo"
    primary: Literal["openai", "groq"] = "openai"
    language: str = "pt"
    domain_prompt: str | None = None
    max_bytes: int = 16 * 1024 * 1024
    min_transcript_chars: int = 2
    request_timeout_s: float = 60.0

    def __post_init__(self) -> None:
        if not self.openai_api_key:
            raise ValueError("openai_api_key must be a non-empty string")

        if self.openai_model != "whisper-1" and "transcribe" not in self.openai_model:
            raise ValueError(
                f"openai_model={self.openai_model!r} is not an OpenAI transcription "
                "model; use whisper-1 / gpt-4o-mini-transcribe / gpt-4o-transcribe"
            )

        if self.primary not in _VALID_PRIMARIES:
            raise ValueError(
                f"primary must be one of {_VALID_PRIMARIES!r}, got {self.primary!r}"
            )

        if self.primary == "groq" and not self.groq_api_key:
            raise ValueError("primary='groq' requires groq_api_key to be set")

        if self.max_bytes <= 0:
            raise ValueError(f"max_bytes must be > 0, got {self.max_bytes!r}")

        if self.min_transcript_chars < 0:
            raise ValueError(
                f"min_transcript_chars must be >= 0, got {self.min_transcript_chars!r}"
            )

        if self.request_timeout_s <= 0:
            raise ValueError(
                f"request_timeout_s must be > 0, got {self.request_timeout_s!r}"
            )
