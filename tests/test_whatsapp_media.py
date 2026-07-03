"""Tests for fetch_whatsapp_media and transcribe_whatsapp_media.

All HTTP is mocked with httpx.MockTransport — no network access, no real keys.
"""

from __future__ import annotations

import httpx
import pytest

from transcription_core import (
    MediaFetchError,
    MediaTooLarge,
    NotAudio,
    TranscriptionConfig,
    fetch_whatsapp_media,
    transcribe_whatsapp_media,
)

GRAPH_HOST = "graph.facebook.com"
MEDIA_HOST = "lookaside.fbsbx.com"
OPENAI_HOST = "api.openai.com"

API_VERSION = "v23.0"
MEDIA_ID = "MEDIA_ID"
ACCESS_TOKEN = "waba-test-token"
MEDIA_URL = "https://lookaside.fbsbx.com/whatsapp_business/attachments/?mid=X"


def make_config(**overrides: object) -> TranscriptionConfig:
    kwargs: dict[str, object] = {"openai_api_key": "sk-test-openai"}
    kwargs.update(overrides)
    return TranscriptionConfig(**kwargs)


def make_client(
    responses: dict[str, list[httpx.Response | type[BaseException]]],
    requests: list[httpx.Request],
) -> httpx.AsyncClient:
    """Same MockTransport pattern as test_transcribe.py, dispatched by host."""
    queues = {host: list(items) for host, items in responses.items()}

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        outcome = queues[request.url.host].pop(0)
        if isinstance(outcome, type) and issubclass(outcome, BaseException):
            raise outcome("simulated transport failure", request=request)
        return outcome

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# 13. Happy path: metadata + download, both carry the Bearer token.
async def test_fetch_whatsapp_media_happy_path() -> None:
    requests: list[httpx.Request] = []
    metadata = {
        "url": MEDIA_URL,
        "mime_type": "audio/ogg; codecs=opus",
        "file_size": 12345,
        "id": MEDIA_ID,
    }
    audio_bytes = b"\x00\x01binary-audio-data"
    client = make_client(
        {
            GRAPH_HOST: [httpx.Response(200, json=metadata)],
            MEDIA_HOST: [httpx.Response(200, content=audio_bytes)],
        },
        requests,
    )

    async with client:
        result = await fetch_whatsapp_media(
            MEDIA_ID,
            ACCESS_TOKEN,
            api_version=API_VERSION,
            http_client=client,
            max_bytes=16 * 1024 * 1024,
        )

    assert result == audio_bytes
    assert len(requests) == 2
    assert str(requests[0].url) == f"https://graph.facebook.com/{API_VERSION}/{MEDIA_ID}"
    assert requests[0].headers["authorization"] == f"Bearer {ACCESS_TOKEN}"
    assert requests[1].headers["authorization"] == f"Bearer {ACCESS_TOKEN}"


# 14. Non-audio mime type -> NotAudio, no download attempted.
async def test_fetch_whatsapp_media_not_audio_skips_download() -> None:
    requests: list[httpx.Request] = []
    metadata = {"url": MEDIA_URL, "mime_type": "image/jpeg", "file_size": 100, "id": MEDIA_ID}
    client = make_client({GRAPH_HOST: [httpx.Response(200, json=metadata)]}, requests)

    async with client:
        with pytest.raises(NotAudio):
            await fetch_whatsapp_media(
                MEDIA_ID,
                ACCESS_TOKEN,
                api_version=API_VERSION,
                http_client=client,
                max_bytes=1000,
            )

    assert len(requests) == 1


# 15. MediaTooLarge: reported file_size, and actual downloaded size.
async def test_fetch_whatsapp_media_metadata_file_size_too_large() -> None:
    requests: list[httpx.Request] = []
    metadata = {"url": MEDIA_URL, "mime_type": "audio/ogg", "file_size": 999_999, "id": MEDIA_ID}
    client = make_client({GRAPH_HOST: [httpx.Response(200, json=metadata)]}, requests)

    async with client:
        with pytest.raises(MediaTooLarge):
            await fetch_whatsapp_media(
                MEDIA_ID,
                ACCESS_TOKEN,
                api_version=API_VERSION,
                http_client=client,
                max_bytes=1000,
            )

    assert len(requests) == 1  # no download attempted


async def test_fetch_whatsapp_media_downloaded_content_too_large() -> None:
    requests: list[httpx.Request] = []
    metadata = {"url": MEDIA_URL, "mime_type": "audio/ogg", "file_size": 10, "id": MEDIA_ID}
    big_content = b"x" * 2000
    client = make_client(
        {
            GRAPH_HOST: [httpx.Response(200, json=metadata)],
            MEDIA_HOST: [httpx.Response(200, content=big_content)],
        },
        requests,
    )

    async with client:
        with pytest.raises(MediaTooLarge):
            await fetch_whatsapp_media(
                MEDIA_ID,
                ACCESS_TOKEN,
                api_version=API_VERSION,
                http_client=client,
                max_bytes=1000,
            )

    assert len(requests) == 2


# 16. Metadata fetch 401 -> MediaFetchError, no token leaked in the message.
async def test_fetch_whatsapp_media_metadata_401() -> None:
    requests: list[httpx.Request] = []
    client = make_client({GRAPH_HOST: [httpx.Response(401, text="Unauthorized")]}, requests)

    async with client:
        with pytest.raises(MediaFetchError) as exc_info:
            await fetch_whatsapp_media(
                MEDIA_ID,
                ACCESS_TOKEN,
                api_version=API_VERSION,
                http_client=client,
                max_bytes=1000,
            )

    message = str(exc_info.value)
    assert "(status 401)" in message
    assert ACCESS_TOKEN not in message


# 17. transcribe_whatsapp_media end-to-end: metadata + download + openai transcription.
async def test_transcribe_whatsapp_media_end_to_end() -> None:
    requests: list[httpx.Request] = []
    metadata = {
        "url": MEDIA_URL,
        "mime_type": "audio/ogg; codecs=opus",
        "file_size": 100,
        "id": MEDIA_ID,
    }
    audio_bytes = b"fake-ogg-opus-bytes"
    client = make_client(
        {
            GRAPH_HOST: [httpx.Response(200, json=metadata)],
            MEDIA_HOST: [httpx.Response(200, content=audio_bytes)],
            OPENAI_HOST: [httpx.Response(200, text="Paciente relata dor de cabeça.")],
        },
        requests,
    )
    config = make_config()

    async with client:
        result = await transcribe_whatsapp_media(
            MEDIA_ID,
            ACCESS_TOKEN,
            api_version=API_VERSION,
            config=config,
            http_client=client,
        )

    assert result.text == "Paciente relata dor de cabeça."
    assert result.provider_used == "openai"
    assert len(requests) == 3
