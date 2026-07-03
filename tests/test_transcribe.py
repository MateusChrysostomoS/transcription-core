"""Tests for transcribe_bytes: happy path, retry-then-fallback, low confidence.

All HTTP is mocked with httpx.MockTransport — no network access, no real keys.
"""

from __future__ import annotations

import httpx
import pytest

from transcription_core import (
    AllProvidersFailed,
    MediaTooLarge,
    TranscriptionConfig,
    TranscriptionResult,
    transcribe_bytes,
)

OPENAI_HOST = "api.openai.com"
GROQ_HOST = "api.groq.com"
AUDIO_BYTES = b"fake-audio-bytes-not-real-media"


def make_config(**overrides: object) -> TranscriptionConfig:
    kwargs: dict[str, object] = {"openai_api_key": "sk-test-openai"}
    kwargs.update(overrides)
    return TranscriptionConfig(**kwargs)


def make_client(
    responses: dict[str, list[httpx.Response | type[BaseException]]],
    requests: list[httpx.Request],
) -> httpx.AsyncClient:
    """Build an AsyncClient backed by MockTransport.

    ``responses`` maps a request host to a queue of outcomes: either an
    httpx.Response to return, or an exception *class* to instantiate (bound to
    the live request) and raise. Every request the handler sees is appended to
    ``requests`` so tests can assert on URL/headers/body.
    """
    queues = {host: list(items) for host, items in responses.items()}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        outcome = queues[request.url.host].pop(0)
        if isinstance(outcome, type) and issubclass(outcome, BaseException):
            raise outcome("simulated transport failure", request=request)
        return outcome

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def decode(request: httpx.Request) -> str:
    return request.content.decode("utf-8")


def field_value(body: str, field: str) -> str | None:
    """Extract a multipart/form-data field's textual value from a decoded body."""
    marker = f'name="{field}"'
    idx = body.find(marker)
    if idx == -1:
        return None
    start = body.index("\r\n\r\n", idx) + 4
    end = body.index("\r\n--", start)
    return body[start:end]


# 1. Happy path (openai).
async def test_happy_path_openai() -> None:
    requests: list[httpx.Request] = []
    client = make_client(
        {OPENAI_HOST: [httpx.Response(200, text="Olá, doutor.\n")]}, requests
    )
    config = make_config()

    async with client:
        result = await transcribe_bytes(AUDIO_BYTES, config=config, http_client=client)

    assert result == TranscriptionResult(
        text="Olá, doutor.",
        provider_used="openai",
        is_low_confidence=False,
        char_count=12,
        duration_seconds=None,
    )

    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://api.openai.com/v1/audio/transcriptions"
    assert request.headers["authorization"] == "Bearer sk-test-openai"

    body = decode(request)
    assert field_value(body, "model") == config.openai_model
    assert field_value(body, "language") == "pt"
    assert field_value(body, "response_format") == "text"


# 2. domain_prompt presence/absence in the multipart body.
async def test_domain_prompt_included_when_set() -> None:
    requests: list[httpx.Request] = []
    client = make_client({OPENAI_HOST: [httpx.Response(200, text="ok")]}, requests)
    config = make_config(domain_prompt="Contexto médico: consulta de rotina.")

    async with client:
        await transcribe_bytes(AUDIO_BYTES, config=config, http_client=client)

    body = decode(requests[0])
    assert field_value(body, "prompt") == "Contexto médico: consulta de rotina."


async def test_domain_prompt_absent_when_not_set() -> None:
    requests: list[httpx.Request] = []
    client = make_client({OPENAI_HOST: [httpx.Response(200, text="ok")]}, requests)
    config = make_config()

    async with client:
        await transcribe_bytes(AUDIO_BYTES, config=config, http_client=client)

    body = decode(requests[0])
    assert 'name="prompt"' not in body


# 3. max_bytes guard fires before any HTTP call.
async def test_media_too_large_raises_before_any_request() -> None:
    requests: list[httpx.Request] = []
    client = make_client({}, requests)
    config = make_config(max_bytes=10)

    async with client:
        with pytest.raises(MediaTooLarge):
            await transcribe_bytes(b"x" * 11, config=config, http_client=client)

    assert requests == []


# 4. Low-confidence threshold behavior.
@pytest.mark.parametrize(
    "text,expected_low_confidence,expected_char_count",
    [
        ("", True, 0),
        ("x", True, 1),
        ("Ok", False, 2),
    ],
)
async def test_low_confidence_threshold(
    text: str, expected_low_confidence: bool, expected_char_count: int
) -> None:
    requests: list[httpx.Request] = []
    client = make_client({OPENAI_HOST: [httpx.Response(200, text=text)]}, requests)
    config = make_config()  # default min_transcript_chars == 2

    async with client:
        result = await transcribe_bytes(AUDIO_BYTES, config=config, http_client=client)

    assert result.is_low_confidence is expected_low_confidence
    assert result.char_count == expected_char_count


# 5. Retryable 429 twice on primary -> fallback to groq.
async def test_openai_429_twice_falls_back_to_groq() -> None:
    requests: list[httpx.Request] = []
    client = make_client(
        {
            OPENAI_HOST: [
                httpx.Response(429, text="rate limited"),
                httpx.Response(429, text="rate limited"),
            ],
            GROQ_HOST: [httpx.Response(200, text="from groq")],
        },
        requests,
    )
    config = make_config(groq_api_key="gsk-test-groq")

    async with client:
        result = await transcribe_bytes(AUDIO_BYTES, config=config, http_client=client)

    assert result.provider_used == "groq"
    assert result.text == "from groq"
    assert len(requests) == 3
    assert [r.url.host for r in requests] == [OPENAI_HOST, OPENAI_HOST, GROQ_HOST]

    groq_request = requests[2]
    assert str(groq_request.url) == "https://api.groq.com/openai/v1/audio/transcriptions"
    body = decode(groq_request)
    assert field_value(body, "model") == config.groq_model


# 6. Retryable 500 then 200 on retry -> success on same provider.
async def test_openai_500_then_200_retry_succeeds() -> None:
    requests: list[httpx.Request] = []
    client = make_client(
        {
            OPENAI_HOST: [
                httpx.Response(500, text="server error"),
                httpx.Response(200, text="recovered"),
            ]
        },
        requests,
    )
    config = make_config()

    async with client:
        result = await transcribe_bytes(AUDIO_BYTES, config=config, http_client=client)

    assert result.provider_used == "openai"
    assert result.text == "recovered"
    assert len(requests) == 2


# 7. Non-retryable 400 -> immediate fallback, no retry of the same provider.
async def test_openai_400_falls_back_without_retry() -> None:
    requests: list[httpx.Request] = []
    client = make_client(
        {
            OPENAI_HOST: [httpx.Response(400, text="bad request")],
            GROQ_HOST: [httpx.Response(200, text="from groq")],
        },
        requests,
    )
    config = make_config(groq_api_key="gsk-test-groq")

    async with client:
        result = await transcribe_bytes(AUDIO_BYTES, config=config, http_client=client)

    assert result.provider_used == "groq"
    openai_requests = [r for r in requests if r.url.host == OPENAI_HOST]
    assert len(openai_requests) == 1
    assert len(requests) == 2


# 8. groq not configured, openai 429 twice -> AllProvidersFailed.
async def test_groq_not_configured_openai_429_twice_raises() -> None:
    requests: list[httpx.Request] = []
    client = make_client(
        {
            OPENAI_HOST: [
                httpx.Response(429, text="slow down"),
                httpx.Response(429, text="slow down"),
            ]
        },
        requests,
    )
    config = make_config()  # groq_api_key is None

    async with client:
        with pytest.raises(AllProvidersFailed) as exc_info:
            await transcribe_bytes(AUDIO_BYTES, config=config, http_client=client)

    assert len(requests) == 2
    message = str(exc_info.value)
    assert "openai" in message
    assert "429" in message


# 9. Both providers exhausted -> AllProvidersFailed, 4 requests total.
async def test_both_providers_fail_raises_all_providers_failed() -> None:
    requests: list[httpx.Request] = []
    client = make_client(
        {
            OPENAI_HOST: [httpx.Response(500, text="e1"), httpx.Response(500, text="e2")],
            GROQ_HOST: [httpx.Response(503, text="e3"), httpx.Response(503, text="e4")],
        },
        requests,
    )
    config = make_config(groq_api_key="gsk-test-groq")

    async with client:
        with pytest.raises(AllProvidersFailed):
            await transcribe_bytes(AUDIO_BYTES, config=config, http_client=client)

    assert len(requests) == 4


# 10. Transport-level timeout on primary (both attempts) -> fallback works on exceptions too.
async def test_transport_timeout_on_openai_falls_back_to_groq() -> None:
    requests: list[httpx.Request] = []
    client = make_client(
        {
            OPENAI_HOST: [httpx.ConnectTimeout, httpx.ConnectTimeout],
            GROQ_HOST: [httpx.Response(200, text="from groq")],
        },
        requests,
    )
    config = make_config(groq_api_key="gsk-test-groq")

    async with client:
        result = await transcribe_bytes(AUDIO_BYTES, config=config, http_client=client)

    assert result.provider_used == "groq"
    assert len(requests) == 3


# 11. primary="groq" -> groq is tried first.
async def test_primary_groq_is_tried_first() -> None:
    requests: list[httpx.Request] = []
    client = make_client({GROQ_HOST: [httpx.Response(200, text="from groq")]}, requests)
    config = make_config(primary="groq", groq_api_key="gsk-test-groq")

    async with client:
        result = await transcribe_bytes(AUDIO_BYTES, config=config, http_client=client)

    assert result.provider_used == "groq"
    assert len(requests) == 1
    assert requests[0].url.host == GROQ_HOST
