# GenOps Governance

This document defines the governance model for the GenOps Governance Specification. This governance model applies to the normative specification repository only. It does not govern implementations, SDKs, or runtimes that adopt the specification.

---

## Project Independence

GenOps is an independent open specification project. It is not a CNCF project, CNCF sandbox project, or part of the OpenTelemetry project. The governance model is designed for compatibility with CNCF-style governance practices but operates independently.

The GenOps specification extends OpenTelemetry semantic conventions. It does not modify or replace existing OpenTelemetry semantics.

---

## Change Control Boundary

This repository governs the normative GenOps specification only. The following are explicitly out of scope for this governance model:

- Reference implementations, including the Koshi Runtime.
- Vendor implementations, SDKs, or client libraries.
- Runtimes or platforms that adopt or implement the specification.

Compliance claims made by implementations are the responsibility of those implementations. This governance model does not certify, endorse, or audit implementation-level compliance.

---

## Roles

### Maintainers

Maintainers have merge authority on the specification repository. Maintainers are responsible for:

- Semantic stability of the specification.
- Versioning decisions (major, minor, patch classification).
- Merge authority on pull requests.
- Backward compatibility enforcement within major versions.
- Adjudicating disputes when consensus cannot be reached.

Addition and removal of maintainers requires a supermajority (2/3) vote of existing active maintainers.

### Specification Editors

Specification editors assist maintainers with:

- Documentation clarity and structural integrity of the specification text.
- RFC 2119 and RFC 8174 keyword consistency across the specification.
- Cross-reference accuracy between specification sections.
- Formatting and editorial standards.

Specification editors do not have merge authority. They provide advisory reviews. Editors are appointed by maintainers.

### Contributors

Contributors are individuals who propose specification changes via pull requests. Any person may become a contributor by submitting a pull request. Contributors are expected to follow the process defined in [CONTRIBUTING.md](CONTRIBUTING.md).

### Community Participants

Community participants engage with the specification through issue discussions, feedback, and review comments. Participation does not require a pull request.

---

## Vendor Neutrality

No single vendor or organization may hold more than 50% of active maintainer seats.

Specification decisions must not require the use of a particular vendor's products, services, or implementations. The specification defines interoperable governance semantics; it does not prescribe implementation choices.

Maintainers must recuse themselves from votes where they have a direct commercial conflict of interest.

---

## Decision Model

### Lazy Consensus

Lazy consensus is the default decision model for all specification changes. A proposal is considered accepted if:

- The required number of maintainer approvals is met (see [CONTRIBUTING.md](CONTRIBUTING.md)).
- The review window has elapsed (7 days for patch and minor changes, 14 days for major changes).
- No maintainer has raised a blocking objection.

A blocking objection must include a technical rationale. Objections without technical rationale may be overridden by maintainer vote.

### Explicit Voting

When lazy consensus cannot be reached, explicit voting is used. Voting is conducted via GitHub pull request comments or GitHub Discussions.

- **Simple majority** of active maintainers for routine decisions.
- **Supermajority (2/3)** of active maintainers for:
  - Major version increments.
  - Changes to this governance document.
  - Addition or removal of maintainers.
  - Changes to compliance requirements (Section 9 of the specification).

**Quorum:** A majority of active maintainers must participate for a vote to be valid.

### Active Maintainer Definition

An active maintainer is one who has participated in specification review activity within the preceding 12 months. Maintainers who have not participated in 12 months are considered emeritus and do not count toward quorum or voting thresholds. Emeritus status may be reversed by resuming participation.

---

## Specification Lifecycle

The specification progresses through defined lifecycle states. Each state has entry criteria, exit criteria, and a defined promotion authority.

| State | Entry Criteria | Exit Criteria | Promotion Authority |
|-------|---------------|---------------|---------------------|
| **Draft** | Initial proposal submitted as PR | Maintainer review confirms scope alignment | Any contributor may submit |
| **Working Draft** | At least 2 maintainer approvals; scope confirmed | Community review complete; no unresolved blocking objections | 2 maintainers |
| **Candidate** | Working Draft with resolved review feedback; compliance impact assessed | At least one independent implementation demonstrates feasibility | 2/3 maintainer supermajority |
| **Stable** | Candidate with demonstrated interoperability across independent implementations | Superseded by new major version or deprecated | 2/3 maintainer supermajority |
| **Deprecated** | Superseded by newer version or found to contain irreconcilable defects | Archival after 12-month deprecation notice | 2/3 maintainer supermajority |
| **Archived** | Deprecation period elapsed | N/A (terminal state) | 2 maintainers |

Demotion (moving backward in the lifecycle) requires the same authority as promotion to the target state.

The current specification status is **Working Draft** (v0.1.0).

---

## Version Governance

The specification follows [Semantic Versioning 2.0.0](https://semver.org/).

### Patch Versions

Clarifications to existing language that do not alter semantics. No normative meaning is changed.

- **Authority:** 2 maintainer approvals.
- **Process:** Lazy consensus with 7-day review window.

### Minor Versions

Additive changes that extend the specification without modifying or removing existing definitions. This includes new optional attributes, new events, new reason_codes, and new appendix entries.

- **Authority:** 2 maintainer approvals.
- **Process:** Lazy consensus with 7-day review window. Specification editor review recommended.

### Major Versions

Breaking changes that modify or remove existing normative definitions, alter compliance requirements, change the semantics of existing attributes or events, or remove reason_codes.

- **Authority:** All active maintainers.
- **Process:** 14-day review window. Supermajority (2/3) vote if consensus is not reached.

### Release Tagging Policy

- Releases MUST be tagged following the pattern `vX.Y.Z` (e.g., `v0.1.0`, `v0.2.0`, `v1.0.0`).
- Release notes MUST summarize semantic changes, categorized by type (clarification, additive, breaking).
- Breaking changes MUST include migration guidance describing the impact on existing compliant implementations.

**Release process:**

1. Maintainer proposes version bump with changelog summary.
2. Review window per version type (7 days for patch/minor, 14 days for major).
3. Upon approval, maintainer updates the version number in the specification header.
4. Tag the release in git.
5. Publish a GitHub Release with release notes.

---

## Conflict Resolution

### Step 1: Pull Request Discussion

All technical disagreements should first be resolved in the pull request or issue where they arise. Participants should provide technical rationale for their positions.

### Step 2: Maintainer Review

If discussion does not resolve the disagreement, any participant may request explicit maintainer review. Maintainers will evaluate the technical merits and specification impact.

### Step 3: Community Discussion Window

For disputes affecting normative definitions, compliance requirements, or the enforcement decision model, a 14-day community discussion window is opened. The discussion is conducted in a dedicated GitHub Discussion thread.

### Step 4: Maintainer Decision

Final decision authority rests with the maintainers. Decisions are made by vote per the thresholds defined in the Decision Model section. The decision and its rationale must be documented in the relevant issue or pull request.

---

## Transparency and Decision Logging

All governance decisions are made in public GitHub issues, pull requests, or Discussions. Private channels are not used for specification decisions. Voting results are recorded in the relevant GitHub thread.

Major decisions -- including version bumps, lifecycle state changes, and governance amendments -- MUST be summarized in a pinned GitHub Discussion or maintained decision log accessible from the repository.

---

## Amendments

This governance document may be amended by supermajority (2/3) vote of active maintainers. Proposed amendments follow the same pull request process as specification changes, with a 14-day review window.

---

## Maintainers

Initial maintainers are the repository owners of the `koshihq/genops-spec` organization. Additional maintainers are added by supermajority (2/3) vote of active maintainers per the process defined in this document.

| Name | GitHub Handle | Role | Status |
|------|---------------|------|--------|
| Guy Derry | [@nordiclion](https://github.com/nordiclion) | Maintainer | Active |
