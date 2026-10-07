## Why

`messages.create` cannot stream or request extended thinking, although the gateway's
`/v1/messages` supports both (issue #3). Callers fall back to the Anthropic SDK.

## What Changes

- `messages.create` (sync and async) gains `stream`, `thinking` and `extra_body`.
- `stream=True` returns an iterator of Anthropic message stream events.
- `Message.content` accepts `thinking` and `redacted_thinking` blocks.
- The SSE stream classes take a parse function, so chat and messages share one reader.

## Impact

- `apertis/_streaming.py`, `apertis/resources/messages.py`, `apertis/types/messages.py`,
  tests and README. Non-streaming calls without the new arguments send the same body as
  before.
