## 1. Apertis SDK Web Search contract

- [x] 1.1 Add additive typed request fields and top-level source response models while preserving legacy Web Search inputs and annotations.
- [x] 1.2 Implement synchronous and asynchronous request validation/body construction for current and legacy Web Search forms.
- [x] 1.3 Extend streaming and response parsing tests for source metadata, preliminary Web Search events, conflict failure, and async parity.
- [x] 1.4 Run SDK formatting, type checks, hermetic unit tests, and a task-owned Docker test container; build a release artifact.

## 2. Standalone LangChain provider package

- [x] 2.1 Create public `apertis-ai/langchain-apertis` from a clean base and lock its OpenSpec contract, package metadata, and dependency bounds.
- [x] 2.2 Implement `ChatApertis` using the official Apertis sync/async SDK clients, standard message conversion, streaming, tool calls, structured output, configuration validation, and usage metadata.
- [x] 2.3 Map verified Apertis reasoning and Web Search metadata without exposing secrets, routing details, or raw transport headers.
- [x] 2.4 Add deterministic mocks, `langchain-tests` unit/integration standard-test subclasses, project quality commands, and package documentation.
- [x] 2.5 Run the provider package's clean-install Docker gate and build its source/wheel artifacts.

## 3. Delivery and discovery gates

- [x] 3.1 Review exact diffs, run finish sweeps, commit focused SDK and provider changes, and push their branches to `apertis-ai`.
- [ ] 3.2 Publish compatible SDK and provider releases only through an authenticated PyPI trusted-publishing or token gate, then verify the registries.
- [ ] 3.3 Update the existing Apertis LangChain external-listing metadata with `ChatApertis`, its PyPI distribution, docs URL, and verified capability flags; push a docs branch/PR after prior-approval requirements are satisfied.
- [ ] 3.4 Verify remote CI, publication, docs-PR merge, and live directory visibility as separate final states.
