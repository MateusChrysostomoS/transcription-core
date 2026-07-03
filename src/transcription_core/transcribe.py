"""Core transcription logic: provider-agnostic, stateless, fully async.

Talks to OpenAI and Groq through the same OpenAI-compatible
``POST /audio/transcriptions`` REST endpoint (one code path for both), with a
retry-then-fallback policy across configured providers.
"""

from __future__ import annotations

import httpx

from .config import TranscriptionConfig
from .exceptions import AllProvidersFailed, MediaTooLarge
from .result import TranscriptionResult

_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
}

_RETRYABLE_TRANSPORT_ERRORS = (httpx.TimeoutException, httpx.TransportError)

# A single provider attempt either yields an httpx.Response or a caught
# transport-level exception (timeout / connection failure).
_Outcome = httpx.Response | BaseException


def _provider_order(config: TranscriptionConfig) -> list[str]:
    """Primary provider first, then the other provider only if configured.

    openai is always configured (its key is required by TranscriptionConfig);
    groq is configured iff groq_api_key is set.
    """
    configured = {"openai"}
    if config.groq_api_key:
        configured.add("groq")

    others = [name for name in ("openai", "groq") if name != config.primary]
    ordered = [config.primary, *others]
    return [name for name in ordered if name in configured]


def _provider_credentials(provider: str, config: TranscriptionConfig) -> tuple[str, str]:
    """Return (api_key, model) for the given provider."""
    if provider == "openai":
        return config.openai_api_key, config.openai_model
    # groq_api_key is guaranteed non-None here: _provider_order() only ever
    # includes "groq" when config.groq_api_key is set.
    assert config.groq_api_key is not None
    return config.groq_api_key, config.groq_model


async def _post_transcription(
    client: httpx.AsyncClient,
    provider: str,
    audio: bytes,
    config: TranscriptionConfig,
) -> httpx.Response:
    api_key, model = _provider_credentials(provider, config)
    url = f"{_BASE_URLS[provider]}/audio/transcriptions"

    data: dict[str, str] = {
        "model": model,
        "language": config.language,
        "response_format": "text",
    }
    if config.domain_prompt:
        data["prompt"] = config.domain_prompt

    files = {"file": ("audio.ogg", audio, "audio/ogg")}
    headers = {"Authorization": f"Bearer {api_key}"}

    return await client.post(
        url,
        data=data,
        files=files,
        headers=headers,
        timeout=config.request_timeout_s,
    )


def _is_retryable(outcome: _Outcome) -> bool:
    if isinstance(outcome, httpx.Response):
        return outcome.status_code == 429 or outcome.status_code >= 500
    return isinstance(outcome, _RETRYABLE_TRANSPORT_ERRORS)


def _describe_attempt(provider: str, outcome: _Outcome) -> str:
    if isinstance(outcome, httpx.Response):
        body = outcome.text[:200]
        return f"provider={provider} status={outcome.status_code} body={body!r}"
    return f"provider={provider} status={type(outcome).__name__}"


async def _attempt(
    client: httpx.AsyncClient,
    provider: str,
    audio: bytes,
    config: TranscriptionConfig,
) -> _Outcome:
    try:
        return await _post_transcription(client, provider, audio, config)
    except _RETRYABLE_TRANSPORT_ERRORS as exc:
        return exc


async def _transcribe_via_client(
    client: httpx.AsyncClient,
    audio: bytes,
    config: TranscriptionConfig,
) -> TranscriptionResult:
    attempt_log: list[str] = []

    for provider in _provider_order(config):
        # First attempt plus exactly one retry on a retryable failure.
        for _attempt_index in range(2):
            outcome = await _attempt(client, provider, audio, config)

            if isinstance(outcome, httpx.Response) and outcome.status_code == 200:
                text = outcome.text.strip()
                return TranscriptionResult(
                    text=text,
                    provider_used=provider,
                    is_low_confidence=len(text) < config.min_transcript_chars,
                    char_count=len(text),
                )

            attempt_log.append(_describe_attempt(provider, outcome))

            if not _is_retryable(outcome):
                break  # non-retryable: move on to the next provider immediately

    raise AllProvidersFailed(
        "all configured transcription providers failed: " + "; ".join(attempt_log)
    )


async def transcribe_bytes(
    audio: bytes,
    *,
    config: TranscriptionConfig,
    http_client: httpx.AsyncClient | None = None,
) -> TranscriptionResult:
    """Transcribe raw audio bytes, trying providers in order with fallback.

    Raises MediaTooLarge before any HTTP call if ``audio`` exceeds
    ``config.max_bytes``. Raises AllProvidersFailed if every configured
    provider is exhausted.

    If ``http_client`` is omitted, an ephemeral ``httpx.AsyncClient`` is
    created and closed for the duration of this call. If provided, it is used
    as-is and never closed by this function.
    """
    if len(audio) > config.max_bytes:
        raise MediaTooLarge(
            f"audio is {len(audio)} bytes, exceeds max_bytes={config.max_bytes}"
        )

    if http_client is not None:
        return await _transcribe_via_client(http_client, audio, config)

    async with httpx.AsyncClient() as client:
        return await _transcribe_via_client(client, audio, config)
