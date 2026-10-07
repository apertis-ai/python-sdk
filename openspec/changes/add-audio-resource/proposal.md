## Why

The gateway serves `POST /v1/audio/speech`, `/v1/audio/transcriptions` and
`/v1/audio/translations`, but the SDK has no audio resource (issue #5), so callers use
the OpenAI SDK instead.

## What Changes

- `client.audio.speech.create()` returns the binary audio with a `write_to_file()` helper.
- `client.audio.transcriptions.create()` and `client.audio.translations.create()` upload
  the file as multipart form data and return a typed result, or plain text for the
  text formats.
- The HTTP client can send multipart requests: it omits the JSON content type and
  reads the file once, so a retry resends the same bytes.

## Impact

- New `apertis/resources/audio.py` and `apertis/types/audio.py`; `apertis/_base_client.py`,
  `apertis/_client.py`, exports, tests and README. Existing JSON requests are unchanged.
