"""Result type returned by transcription calls."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    provider_used: str
    is_low_confidence: bool
    char_count: int
    duration_seconds: float | None = None
