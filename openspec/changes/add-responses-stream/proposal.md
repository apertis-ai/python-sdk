## Why

`responses.create` cannot stream (issue #15), although the gateway streams `/v1/responses`.
It passes OpenAI channels' events through, and builds the same event types itself for other
providers. Callers fall back to the OpenAI SDK.

## What Changes

- `responses.create` (sync and async) gains `stream`.
- `stream=True` returns an iterator of `ResponseStreamEvent`. Each event keeps its `type` and
  every field the API sent.
- An `error` event raises `APIError`.

## Impact

- `apertis/resources/responses.py`, `apertis/types/responses.py`, exports, tests and README.
- Non-streaming calls send the same body as before.
