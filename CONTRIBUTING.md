# Contributing to the GenOps Governance Specification

## Scope

This repository contains the GenOps Governance Specification. Contributions are changes to the normative specification text. This repository does not contain code, SDKs, or runtime implementations.

This document defines the process for proposing, reviewing, and accepting specification changes.

---

## Types of Specification Changes

### Clarification (Patch Version)

Changes that clarify existing language without altering semantics. Examples include fixing ambiguous wording, correcting typographical errors, and improving examples.

Clarification changes MUST NOT alter the meaning of any normative statement.

**Version impact:** Patch increment (e.g., 0.1.0 to 0.1.1).

### Additive (Minor Version)

Changes that extend the specification without modifying or removing existing definitions. Examples include adding new optional attributes, adding new reason_codes to the taxonomy, adding new optional events, and adding new appendix entries.

Additive changes MUST NOT alter or remove existing normative definitions.

**Version impact:** Minor increment (e.g., 0.1.0 to 0.2.0).

### Breaking (Major Version)

Changes that modify or remove existing normative definitions, alter compliance requirements, change the semantics of existing attributes or events, or remove reason_codes from the taxonomy.

**Version impact:** Major increment (e.g., 0.1.0 to 1.0.0).

---

## Backward Compatibility Policy

Within a major version, all existing normative definitions MUST be preserved. New attributes, events, and reason codes MAY be added. Existing attributes, events, and reason codes MUST NOT be removed or redefined.

This policy is consistent with the specification's own backward compatibility guarantees:

- Section 5.6: New reason codes MAY be added in future minor versions; existing reason codes MUST NOT be removed or redefined within the same major version.
- Section 8.5: Stable event schemas MUST NOT change within the same major version. Experimental event schemas MAY change in minor versions.

The specification follows [Semantic Versioning 2.0.0](https://semver.org/).

---

## Normative Language Discipline

All normative requirements in the specification MUST use RFC 2119 and RFC 8174 keywords: MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT, RECOMMENDED, MAY, and OPTIONAL.

Lowercase "must", "should", and "may" MUST NOT appear in normative sections where they carry requirement-level meaning. These words are reserved for their RFC 2119 interpretation when capitalized.

Specification editors are responsible for enforcing keyword consistency during review. Contributors should familiarize themselves with [RFC 2119](https://datatracker.ietf.org/doc/html/rfc2119) and [RFC 8174](https://datatracker.ietf.org/doc/html/rfc8174) before proposing normative text.

---

## Closed Taxonomy Enforcement

The reason_code taxonomy (Section 5) and the decision state enumeration (Section 4.1) are closed by design. The closed taxonomy property is a design invariant of the specification.

Proposals to add new reason_codes or decision states MUST include explicit justification for taxonomy inclusion:

- Why the existing taxonomy is insufficient for the identified governance scenario.
- Why the proposed addition is necessary for interoperable governance.
- Whether the use case can be addressed through the extension code mechanism (x\_ prefix, Section 5.5) without modifying the normative taxonomy.

Taxonomy additions require Minor or Major review thresholds as defined in the Review Process section of this document. Extension codes are the designated mechanism for implementation-specific needs and do not require specification changes.

---

## Proposal Requirements

All specification changes MUST be submitted as GitHub pull requests. Each proposal MUST include the following sections in the pull request description:

### Problem Statement

What deficiency, ambiguity, or gap does this change address?

### Rationale

Why is this change necessary? What alternatives were considered?

### Compatibility Analysis

- **For clarifications:** Confirm that no semantic change results from the proposed rewording.
- **For additive changes:** Confirm that no existing definitions are altered or removed.
- **For breaking changes:** Enumerate all affected definitions and describe the migration path for existing compliant implementations.

### Compliance Impact

How does this change affect Section 9 compliance requirements? Specifically:

- Does it add, modify, or remove compliance criteria?
- Does it affect the boundary between full compliance (Section 9.1) and partial compliance (Section 9.2)?
- Does it change what an implementation must emit to be considered compliant?

---

## Review Process

### Clarification Changes (Patch)

- **Required approvals:** 2 maintainers.
- **Review window:** 7 days under lazy consensus.
- If no blocking objections are raised and approvals are met, the change is merged.

### Additive Changes (Minor)

- **Required approvals:** 2 maintainers.
- **Review window:** 7 days under lazy consensus.
- Specification editor review for RFC 2119/8174 keyword consistency.
- Compliance impact section must be reviewed explicitly.

### Breaking Changes (Major)

- **Required approvals:** All active maintainers.
- **Review window:** 14 days.
- Supermajority (2/3) vote if consensus is not reached.
- See [GOVERNANCE.md](GOVERNANCE.md) for the full decision model.

---

## Specification Change Criteria

All specification changes MUST satisfy the following criteria:

- Preserve deterministic behavior of the accounting invariant (Section 3).
- Remain within the scope of runtime governance semantics. The specification's non-goals (Section 10) define the boundary.
- Maintain the closed taxonomy property of reason_codes (Section 5) and decision states (Section 4.1).
- Use RFC 2119/8174 keywords consistently and correctly.
- Maintain backward compatibility within a major version.
- Not introduce vendor-specific requirements.

---

## Submitting a Change

1. **Open a GitHub issue** describing the proposed change. This is recommended for additive and breaking changes, and optional for clarifications.
2. **Fork the repository** and create a branch.
3. **Make the proposed change** to the specification text.
4. **Include the required proposal sections** (problem statement, rationale, compatibility analysis, compliance impact) in the pull request description.
5. **Submit the pull request** against the `main` branch.
6. **Address review feedback.**

---

## Licensing

All contributions to this repository are made under the [Apache License 2.0](LICENSE). By submitting a pull request, you agree that your contributions will be licensed under the same terms.

---

## Code of Conduct

All participants in the GenOps specification project are expected to engage respectfully and constructively. Technical disagreements should focus on the merits of proposals, not on individuals. Maintainers may moderate or close discussions that do not meet this standard.
