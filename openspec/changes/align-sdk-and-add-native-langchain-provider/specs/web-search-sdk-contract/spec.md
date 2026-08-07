## ADDED Requirements

### Requirement: Documented Web Search request options
The Apertis Chat Completions SDK SHALL accept `web_results_count` and
`web_content_length` on synchronous and asynchronous `create` calls, include only
specified values in the JSON request body, and preserve the caller-supplied model ID.

#### Scenario: Current Web Search call
- **WHEN** a caller invokes Chat Completions with a `:web` model, `web_results_count`,
  and `web_content_length`
- **THEN** the SDK sends those exact fields with the messages and does not rewrite the
  model ID

#### Scenario: Legacy compatibility
- **WHEN** a caller uses the existing `web_search_options` parameter without the new
  Web Search fields
- **THEN** the SDK preserves its existing request behavior

#### Scenario: Conflicting Web Search styles
- **WHEN** a caller supplies legacy and current Web Search request styles that cannot be
  represented in one documented request
- **THEN** the SDK raises a deterministic validation error before making an HTTP request

### Requirement: Web Search source response fidelity
The SDK SHALL deserialize the documented top-level `web_sources` array into typed source
records containing the documented title, URL, and snippet fields while preserving existing
message annotations and response fields.

#### Scenario: Non-stream Web Search result
- **WHEN** an API response includes a top-level `web_sources` array
- **THEN** the returned `ChatCompletion` exposes every documented source record without
  dropping message content, usage, or annotations

### Requirement: Web Search stream compatibility
The SDK SHALL parse the documented Web Search preliminary SSE chunk and subsequent model
chunks without terminating or silently discarding valid content, for both sync and async
streams.

#### Scenario: Preliminary Web Search chunk
- **WHEN** a streamed response begins with the documented Web Search status chunk and is
  followed by regular completion chunks and `[DONE]`
- **THEN** iteration yields the status and regular chunks in order and terminates normally

### Requirement: Hermetic SDK verification
SDK unit tests for the new Web Search contract SHALL use mocked HTTP and SHALL execute
without Apertis credentials or live network access.

#### Scenario: Docker SDK test gate
- **WHEN** the SDK test gate runs in its task-owned Docker container with no API key
- **THEN** the Web Search unit suite and existing SDK regression suite complete without
  contacting `api.apertis.ai`
