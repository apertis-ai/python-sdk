## ADDED Requirements

### Requirement: Native independently distributed chat model
The Apertis LangChain integration SHALL be distributed as public package
`langchain-apertis` from an `apertis-ai` repository and export `ChatApertis` from
`langchain_apertis`.

#### Scenario: Standard installation and import
- **WHEN** a user installs the package with its declared dependencies
- **THEN** `from langchain_apertis import ChatApertis` succeeds without requiring
  `langchain-openai`

### Requirement: Apertis client initialization
`ChatApertis` SHALL accept an explicit API key or read `APERTIS_API_KEY`, default to the
official Apertis API base URL, keep secrets out of serializable metadata, and report a
clear local configuration error before a request when no key is available.

#### Scenario: Environment credential
- **WHEN** `APERTIS_API_KEY` is set and no explicit key is passed
- **THEN** the model constructs the official Apertis client with that credential

#### Scenario: Missing credential
- **WHEN** no explicit API key or `APERTIS_API_KEY` is present
- **THEN** model initialization fails without emitting a network request or revealing a
  secret

### Requirement: Standard LangChain chat behavior
`ChatApertis` SHALL implement synchronous and asynchronous invocation, streaming, tool
calling, structured output conversion, model identification, and token usage according to
the `BaseChatModel` and `langchain-tests` contracts for the capabilities it claims.

#### Scenario: Tool call completion
- **WHEN** Apertis returns a valid OpenAI-compatible tool call
- **THEN** `ChatApertis` returns an `AIMessage` with the standard LangChain tool-call
  representation and a stable call ID

#### Scenario: Usage metadata
- **WHEN** Apertis returns prompt, completion, and total token usage
- **THEN** `ChatApertis` exposes matching `input_tokens`, `output_tokens`, and
  `total_tokens` usage metadata

#### Scenario: Async stream
- **WHEN** a caller iterates `astream` over an Apertis streamed completion
- **THEN** it receives ordered `AIMessageChunk` objects and an eventual terminal state
  without a synchronous client call

### Requirement: Provider-specific metadata preservation
`ChatApertis` SHALL preserve supported Apertis reasoning and Web Search source metadata in
documented LangChain response structures, while excluding credentials, provider routing
details, and raw transport headers.

#### Scenario: Web Search source result
- **WHEN** an Apertis result includes Web Search sources
- **THEN** the returned LangChain message exposes the sources in its documented provider
  metadata field without changing normal text content or usage metadata

#### Scenario: Web Search stream status
- **WHEN** Apertis emits its preliminary Web Search stream status chunk
- **THEN** `ChatApertis` emits a corresponding visible content chunk rather than dropping
  the provider event

### Requirement: Standard and hermetic package verification
The package SHALL subclass the current LangChain unit and integration standard test suites
and maintain deterministic mock-based tests for all claimed features. Its Docker gate
SHALL install the package from source and run the hermetic tests without API credentials.

#### Scenario: Clean Docker install
- **WHEN** the provider package Docker test gate installs the source distribution and
  dependencies in a disposable container
- **THEN** imports, package unit tests, and LangChain standard unit tests pass without
  using a live Apertis account
