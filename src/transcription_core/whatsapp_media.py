"""WhatsApp Cloud API media fetch, and a fetch+transcribe convenience wrapper.

Everything happens in memory: media bytes are never written to disk, never
logged, and never persisted anywhere.
"""

from __future__ import annotations

import httpx

from .config import TranscriptionConfig
from .exceptions import MediaFetchError, MediaTooLarge, NotAudio
from .result import TranscriptionResult
from .transcribe import transcribe_bytes

_GRAPH_BASE = "https://graph.facebook.com"


async def fetch_whatsapp_media(
    media_id: str,
    access_token: str,
    *,
    api_version: str,
    http_client: httpx.AsyncClient,
    max_bytes: int,
) -> bytes:
    """Fetch WhatsApp media bytes via the Graph API's two-step lookup.

    1. GET the media metadata (which yields a short-lived download `url`).
    2. GET that url to download the actual bytes.

    Raises NotAudio if the mime type is not audio/*, MediaTooLarge if metadata
    or the downloaded content exceeds max_bytes, and MediaFetchError for any
    non-200 response or transport failure. Exception messages never include
    the access token, the media URL, or response bodies.
    """
    headers = {"Authorization": f"Bearer {access_token}"}

    try:
        metadata_response = await http_client.get(
            f"{_GRAPH_BASE}/{api_version}/{media_id}",
            headers=headers,
        )
    except httpx.HTTPError as exc:
        raise MediaFetchError(type(exc).__name__) from exc

    if metadata_response.status_code != 200:
        raise MediaFetchError(
            f"WhatsApp media metadata fetch failed (status {metadata_response.status_code})"
        )

    metadata = metadata_response.json()

    url = metadata.get("url")
    if not url:
        raise MediaFetchError("WhatsApp media metadata missing 'url'")

    raw_mime = metadata.get("mime_type") or ""
    mime_type = raw_mime.split(";")[0].strip().lower()
    if not mime_type.startswith("audio/"):
        raise NotAudio(mime_type)

    file_size = metadata.get("file_size")
    if file_size is not None and int(file_size) > max_bytes:
        raise MediaTooLarge(
            f"WhatsApp media file_size={file_size} exceeds max_bytes={max_bytes}"
        )

    try:
        media_response = await http_client.get(
            url,
            headers=headers,
            follow_redirects=True,
        )
    except httpx.HTTPError as exc:
        raise MediaFetchError(type(exc).__name__) from exc

    if media_response.status_code != 200:
        raise MediaFetchError(
            f"WhatsApp media download failed (status {media_response.status_code})"
        )

    content = media_response.content
    if len(content) > max_bytes:
        raise MediaTooLarge(
            f"WhatsApp media download is {len(content)} bytes, exceeds max_bytes={max_bytes}"
        )
    if not content:
        raise MediaFetchError("empty media download")

    return content


async def transcribe_whatsapp_media(
    media_id: str,
    access_token: str,
    *,
    api_version: str,
    config: TranscriptionConfig,
    http_client: httpx.AsyncClient,
) -> TranscriptionResult:
    """Fetch WhatsApp media then transcribe it. Convenience wrapper only."""
    audio = await fetch_whatsapp_media(
        media_id,
        access_token,
        api_version=api_version,
        http_client=http_client,
        max_bytes=config.max_bytes,
    )
    return await transcribe_bytes(audio, config=config, http_client=http_client)
