## ADDED Requirements

### Requirement: Messages streaming
`messages.create` SHALL accept `stream=True` on sync and async clients, send
`"stream": true`, and return an iterator of typed Anthropic stream events
(`message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`,
`message_delta`, `message_stop`) in arrival order. The iterator SHALL end at the end of
the SSE body without a `[DONE]` sentinel.

#### Scenario: Text stream
- **WHEN** the API streams a text answer
- **THEN** the caller receives every event in order, `text_delta` deltas carry the text,
  and the `message_delta` event exposes `stop_reason` and output `usage`

#### Scenario: Keep-alive and error events
- **WHEN** the stream contains a `ping` event
- **THEN** the SDK skips it
- **WHEN** the stream contains an `error` event
- **THEN** the SDK raises `APIError` carrying the error body

#### Scenario: HTTP error on a streamed request
- **WHEN** a streamed chat or messages request returns an HTTP error status
- **THEN** the SDK raises the matching `APIError` subclass with the error message

### Requirement: Extended thinking
`messages.create` SHALL accept a `thinking` object and send it unchanged. Responses and
stream events SHALL expose `thinking` and `redacted_thinking` content blocks and
`thinking_delta` / `signature_delta` deltas.

#### Scenario: Thinking response
- **WHEN** a non-streaming response contains a `thinking` block before a `text` block
- **THEN** `Message.content` holds a typed thinking block with its text and signature,
  followed by the text block

### Requirement: Extra body fields
`messages.create` SHALL accept `extra_body` and merge its keys into the top level of the
request body, so fields the SDK does not name yet reach the API.

#### Scenario: Unnamed field
- **WHEN** a caller passes `extra_body={"service_tier": "auto"}`
- **THEN** the request body contains `"service_tier": "auto"` at the top level
