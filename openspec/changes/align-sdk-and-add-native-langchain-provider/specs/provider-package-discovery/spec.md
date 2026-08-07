## ADDED Requirements

### Requirement: Accurate LangChain package discovery
After `langchain-apertis` is publicly published, the Apertis LangChain docs listing SHALL
identify `ChatApertis`, its PyPI package, a maintained Apertis documentation URL, and only
the capabilities verified by the provider package tests.

#### Scenario: Default external listing
- **WHEN** the package does not meet LangChain's hosted-guide eligibility criteria and is
  not maintainer-featured
- **THEN** its docs contribution updates the external integration YAML listing rather than
  adding a hosted provider page

#### Scenario: Existing generic Apertis row
- **WHEN** the package listing replaces the current generic Apertis chat row
- **THEN** the generated table contains one non-duplicated package row with the verified
  package metadata

### Requirement: Delivery gate separation
The delivery workflow SHALL report and verify package build, test, Git delivery, PyPI
publication, docs PR merge, and live documentation visibility as distinct states.

#### Scenario: Pre-publication state
- **WHEN** local code and Docker tests pass but PyPI publication has not completed
- **THEN** the integration is reported as implemented and verified locally, not published
  or live in the LangChain package directory
