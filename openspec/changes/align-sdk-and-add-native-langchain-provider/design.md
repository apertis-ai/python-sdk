## Context

The Apertis SDK is the first-party, typed transport for `https://api.apertis.ai/v1`.
Its released Chat Completions API has legacy `web_search_options` and message-level
`annotations`, while the current public Web Search contract uses a `:web` model suffix,
`web_results_count`, `web_content_length`, and a top-level `web_sources` array. The
existing LangChain guide uses `ChatOpenAI` with a custom base URL, so it does not own
provider-specific conversion or metadata preservation.

LangChain's `ChatOpenRouter` demonstrates the required boundary: a dedicated
`BaseChatModel` implementation backed by the provider's official SDK, standard-test
coverage, and explicitly converted provider metadata. Current LangChain policy requires
new integrations to live outside the `langchain-ai/langchain` monorepo.

## Goals / Non-Goals

**Goals:**

- Make the SDK faithfully represent the documented Web Search request/response contract
  for sync, async, and streamed Chat Completions calls.
- Preserve existing SDK arguments and response fields so the alignment is additive.
- Ship a separately versioned, public `apertis-ai/langchain-apertis` package with a
  native `ChatApertis` chat model and LangChain standard tests.
- Keep test traffic hermetic: unit and Docker tests use mocked HTTP, while credentialed
  integration tests are opt-in and never require a production key for the default gate.

**Non-Goals:**

- Reimplement the Apertis transport or copy `ChatOpenAI`/`ChatOpenRouter` source.
- Claim support for the SDK's separate Images, Audio, Video, Responses, Messages, or
  Rerank resources in the first LangChain release without their own LangChain component
  contracts and standard-test coverage.
- Publish to PyPI or merge a LangChain docs PR before build, credential, and maintainer
  gates are independently satisfied.

## Decisions

### SDK owns provider protocol; LangChain adapts it

`ChatApertis` will construct the official sync and async `Apertis` clients and translate
between LangChain messages/results and the SDK types. This follows OpenRouter's boundary
and ensures new Apertis protocol fields are fixed once in the SDK.

Alternative: subclass `ChatOpenAI` with a fixed `base_url`. Rejected because LangChain
explicitly limits that class to OpenAI's public API surface, which would lose Apertis
Web Search sources and provider-specific reasoning metadata.

### Add the documented Web Search fields without removing legacy fields

The SDK will add typed `web_results_count` and `web_content_length` request parameters,
plus typed top-level `web_sources` response metadata. Existing `web_search_options` and
`annotations` remain supported to avoid breaking current clients. Supplying incompatible
old and new Web Search request styles in one request must fail before an HTTP call.

The low-level `create` method will send the caller's exact model ID; callers opt into
current Web Search by passing a model ending in `:web`. The convenience helper will
normalize a supplied non-free model to the suffix form and expose only documented
options, while leaving the legacy helper's compatible options intact.

### Use `apertis-ai` for the provider package

The official SDK and public API are owned by `apertis-ai`, so the integration repository
will be `apertis-ai/langchain-apertis`, not a personal `theQuert` repository. It will
publish `langchain-apertis`, export `ChatApertis`, and declare bounded compatible ranges
for `apertis`, `langchain-core`, and `langchain-tests`.

### Treat source metadata as content, not hidden transport data

`ChatApertis` will expose standard token usage as `usage_metadata`, model/id/finish state
as `response_metadata`, and turn reasoning/source information into documented LangChain
message content blocks or response metadata without exposing credentials, internal route
details, or upstream headers. Web Search's preliminary stream status remains a valid
visible content chunk rather than being silently discarded.

### Docker proof is task-owned and disposable

The gate runs the source tree inside a named, `--rm` Python container with no API key,
using the project test dependencies and mocked HTTP. This proves clean-install behavior
without touching shared databases, volumes, Docker networks, or production services.

## Risks / Trade-offs

- [Public docs and released SDK have drifted] → Add contract tests from the current docs,
  preserve legacy calls, and release the SDK before the integration declares the feature.
- [LangChain interface changes faster than the SDK] → Pin compatible dependency ranges,
  consume `langchain-tests`, and test against the declared lower bounds in Docker.
- [Web Search source schema changes] → Accept only the documented stable fields and retain
  raw provider metadata in an explicitly namespaced response field; no speculative model
  profiles are shipped.
- [Real model availability is key/plan dependent] → Standard integration tests are
  explicitly opt-in, choose their model via environment configuration, and default gates
  do not make network calls.

## Migration Plan

1. Release an additive SDK version with Web Search source/request alignment.
2. Build and test `langchain-apertis` against that released SDK version.
3. Publish the new package only with an authorized PyPI credential or trusted publisher.
4. Update the existing Apertis LangChain docs listing to its package row and verified
   capability flags; request maintainer approval as required by the docs contribution
   policy.
5. Roll back consumers by pinning the prior SDK/package versions; no server migration or
   data migration is involved.

## Open Questions

- Which exact stable non-free model and test account should be used for the opt-in live
  LangChain integration test? This does not block hermetic implementation or Docker proof.
- Whether LangChain maintainers will mark Apertis as featured is external to the package;
  default discovery remains the external YAML listing.
