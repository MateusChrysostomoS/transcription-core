# transcription-core

Small, provider-agnostic, stateless audio-transcription library shared by Brain Co
services. It wraps the OpenAI-compatible `audio/transcriptions` REST endpoint for
both OpenAI and Groq, with automatic retry-then-fallback between providers, plus a
WhatsApp Cloud API media-fetch helper.

## Design constraints

- **Stateless.** No database access, no disk writes, no credential resolution, no
  environment variable reads, no logging. All configuration is passed in explicitly
  via `TranscriptionConfig`.
- **Async only**, Python 3.12+.
- **`httpx` is the only runtime dependency.** No `openai` SDK.

Callers own credential resolution, persistence, and logging; this library only
turns bytes into text (and optionally fetches those bytes from WhatsApp).

## Install

```bash
uv add transcription-core
```

(or add it as a path/git dependency from your service's `pyproject.toml`.)

## Usage

```python
import httpx
from transcription_core import TranscriptionConfig, transcribe_bytes

config = TranscriptionConfig(
    openai_api_key="sk-...",
    groq_api_key="gsk-...",   # optional fallback provider
    primary="openai",
    language="pt",
)

async with httpx.AsyncClient() as client:
    result = await transcribe_bytes(audio_bytes, config=config, http_client=client)

print(result.text, result.provider_used, result.is_low_confidence)
```

### WhatsApp Cloud API media

```python
from transcription_core import transcribe_whatsapp_media

async with httpx.AsyncClient() as client:
    result = await transcribe_whatsapp_media(
        media_id,
        access_token,
        api_version="v23.0",
        config=config,
        http_client=client,
    )
```

`http_client` is always optional on `transcribe_bytes` (an ephemeral client is
created and closed when omitted) but required on the WhatsApp helpers, and it is
never closed by this library when you provide one — you own its lifecycle.

## Behavior notes

- Provider order: `config.primary` first, then the other provider only if it is
  configured (`groq_api_key` set).
- Retry policy: HTTP 429/5xx and network-level timeouts/transport errors retry the
  same provider once before falling back; any other 4xx falls back immediately
  without retrying. If every configured provider fails, the library raises
  `AllProvidersFailed` summarizing each attempt.
- `TranscriptionResult.is_low_confidence` is `True` when the stripped transcript is
  shorter than `config.min_transcript_chars` (default `2`).

## Development

```bash
uv sync
uv run python -m pytest -q
```

Tests mock all HTTP calls with `httpx.MockTransport` — no network access or real API
keys are required.
