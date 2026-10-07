## ADDED Requirements

### Requirement: Responses streaming
`responses.create` SHALL accept `stream=True` on sync and async clients, send `"stream": true`,
and return an iterator of `ResponseStreamEvent` in arrival order. Each event SHALL expose `type`
and keep every other field of the event payload. The iterator SHALL end at a `[DONE]` sentinel,
or at the end of the SSE body when there is none. Without `stream`, the request body SHALL be
unchanged.

#### Scenario: Gateway-built stream
- **WHEN** the API streams `response.created`, `response.output_text.delta` events and
  `response.completed`, followed by `data: [DONE]`
- **THEN** the caller receives those events in order, the deltas join to the answer text, and
  iteration stops at `[DONE]`

#### Scenario: Passed-through stream
- **WHEN** the stream ends without `[DONE]` and its events carry `sequence_number`, `item_id`
  and fields the SDK does not name
- **THEN** the caller receives every event, with those fields readable

#### Scenario: Error event
- **WHEN** the stream contains an `error` event
- **THEN** the SDK raises `APIError` carrying the event body and closes the response
