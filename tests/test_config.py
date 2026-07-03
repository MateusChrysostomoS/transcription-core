"""Tests for TranscriptionConfig validation (fail fast, in __post_init__)."""

from __future__ import annotations

import dataclasses

import pytest

from transcription_core import TranscriptionConfig


def make_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {"openai_api_key": "sk-test-openai"}
    kwargs.update(overrides)
    return kwargs


def test_default_config_is_valid() -> None:
    config = TranscriptionConfig(**make_kwargs())
    assert config.openai_model == "gpt-4o-mini-transcribe"
    assert config.primary == "openai"
    assert config.groq_api_key is None
    assert config.max_bytes == 16 * 1024 * 1024
    assert config.min_transcript_chars == 2
    assert config.request_timeout_s == 60.0


@pytest.mark.parametrize(
    "model", ["whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"]
)
def test_valid_openai_transcription_models_are_accepted(model: str) -> None:
    config = TranscriptionConfig(**make_kwargs(openai_model=model))
    assert config.openai_model == model


@pytest.mark.parametrize("model", ["gpt-4.1-nano", "gpt-4o-mini", "gpt-5-mini"])
def test_non_transcription_openai_models_are_rejected(model: str) -> None:
    with pytest.raises(ValueError, match="not an OpenAI transcription model"):
        TranscriptionConfig(**make_kwargs(openai_model=model))


def test_empty_openai_api_key_is_rejected() -> None:
    with pytest.raises(ValueError):
        TranscriptionConfig(openai_api_key="")


def test_primary_groq_without_groq_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="groq_api_key"):
        TranscriptionConfig(**make_kwargs(primary="groq"))


def test_primary_groq_with_groq_key_is_accepted() -> None:
    config = TranscriptionConfig(
        **make_kwargs(primary="groq", groq_api_key="gsk-test-groq")
    )
    assert config.primary == "groq"
    assert config.groq_api_key == "gsk-test-groq"


def test_invalid_primary_value_is_rejected() -> None:
    with pytest.raises(ValueError):
        TranscriptionConfig(**make_kwargs(primary="anthropic"))  # type: ignore[arg-type]


@pytest.mark.parametrize("max_bytes", [0, -1])
def test_non_positive_max_bytes_is_rejected(max_bytes: int) -> None:
    with pytest.raises(ValueError):
        TranscriptionConfig(**make_kwargs(max_bytes=max_bytes))


def test_negative_min_transcript_chars_is_rejected() -> None:
    with pytest.raises(ValueError):
        TranscriptionConfig(**make_kwargs(min_transcript_chars=-1))


def test_zero_min_transcript_chars_is_accepted() -> None:
    config = TranscriptionConfig(**make_kwargs(min_transcript_chars=0))
    assert config.min_transcript_chars == 0


@pytest.mark.parametrize("timeout", [0, -5.0])
def test_non_positive_request_timeout_is_rejected(timeout: float) -> None:
    with pytest.raises(ValueError):
        TranscriptionConfig(**make_kwargs(request_timeout_s=timeout))


def test_config_is_frozen() -> None:
    config = TranscriptionConfig(**make_kwargs())
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.openai_api_key = "changed"  # type: ignore[misc]
