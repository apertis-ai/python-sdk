## Why

Apertis is listed in the LangChain provider directory, but users currently reach it
through `ChatOpenAI` and a custom base URL. That path drops Apertis-specific behavior,
and the released Python SDK does not yet model the documented Web Search response and
request contract needed by a native integration.

## What Changes

- Align the official Python SDK's Chat Completions Web Search request and response types
  with the current public API while preserving existing callers.
- Add deterministic SDK unit coverage for synchronous, asynchronous, and streaming Web
  Search responses, including the documented failure-safe stream shape.
- Create a standalone `langchain-apertis` package under `apertis-ai` that exposes a
  native `ChatApertis` implementation backed by the official SDK rather than a
  `ChatOpenAI` base-URL override.
- Validate the SDK and provider package in disposable Docker environments, then prepare
  the published package and the existing LangChain docs listing for the package name and
  verified capabilities.

## Capabilities

### New Capabilities

- `web-search-sdk-contract`: The official SDK accepts and preserves the currently
  documented Web Search request options and source metadata for sync, async, and stream
  responses without breaking its existing public request parameters.
- `native-langchain-chat-model`: A separately distributed `ChatApertis` implements the
  LangChain chat-model contract with Apertis authentication, standard messages/tool
  calls/usage, and verified Apertis-specific metadata.
- `provider-package-discovery`: The released standalone package is represented in the
  LangChain external integration listing with only verified capability claims.

### Modified Capabilities

- None.

## Impact

- `apertis.resources.chat.completions`, the chat response types, streaming parser,
  SDK tests, package versioning, and provider documentation.
- New public Python distribution `langchain-apertis` in a dedicated
  `apertis-ai/langchain-apertis` repository, depending on compatible `apertis`,
  `langchain-core`, and `langchain-tests` versions.
- The existing `theQuert/docs` fork will later update its already-merged Apertis listing;
  publication and the docs PR remain distinct external delivery gates.
