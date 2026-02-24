<p align="center">
  <img width="500" src="./assets/brand/genops-logo-optimized.jpg" alt="GenOps: Governance Specification for AI Workloads" style="max-width: 100%;">
</p>

# GenOps Governance Specification

**[Specification: GenOps Governance Specification v0.1](spec/genops-spec-v0.1.md)**

GenOps is an open, runtime-agnostic governance specification for AI workloads.

GenOps defines the minimal interoperable governance surface for AI workloads. It specifies the semantic contract required for consistent runtime governance across independently developed systems.

This repository contains the GenOps specification document. Governance model: [GOVERNANCE.md](GOVERNANCE.md). Contribution process: [CONTRIBUTING.md](CONTRIBUTING.md).

---

## What GenOps Is

GenOps standardizes governance semantics for AI workloads. It defines:

- The **AI Workload Unit (AWU)**, the atomic boundary of governed work.
- A **Deterministic Accounting Invariant**, ensuring reservation-before-execution resource tracking.
- A canonical **Enforcement Decision Model**.
- A fixed **reason_code taxonomy** for governance outcomes.
- A minimal set of **required telemetry attributes and events**.
- Clear **compliance criteria**.

GenOps does not define APIs, SDKs, user interfaces, policy languages, or control planes. It defines the semantic surface that a compliant runtime MUST emit.

---

## Why This Exists

AI workloads consume measurable resources and operate under governance constraints:

- Budget limits
- Rate restrictions
- SLA boundaries
- Attribution requirements

These controls are implemented inconsistently across vendors and runtimes. GenOps establishes a shared vocabulary and invariant model so that:

- Governance decisions are machine-readable.
- Accounting is deterministic.
- Telemetry is interoperable.
- Independent runtimes can be evaluated consistently.

---

## Design Principles

GenOps is intentionally narrow. It follows these principles:

- **Runtime-agnostic.** No dependency on a specific architecture.
- **Telemetry-first.** Governance must be externally verifiable.
- **Deterministic.** Identical inputs produce identical reservation outcomes.
- **Closed taxonomy.** No freeform reason strings.
- **Minimal surface area.** Only what is required for interoperable governance is defined.

---

## Relationship to OpenTelemetry

GenOps extends OpenTelemetry semantic conventions. All GenOps telemetry:

- Uses the `genops.*` namespace.
- MUST be representable as standard OpenTelemetry signals (traces, logs, metrics).

GenOps does not modify or replace existing OpenTelemetry semantics.

GenOps defines governance semantics. OpenTelemetry defines observability transport and structure. They are complementary.

---

## Compliance

A runtime is **GenOps v0.1 compliant** if it:

1. Emits all required attributes on every AI Workload Unit.
2. Implements the Deterministic Accounting Invariant.
3. Emits required governance events at defined lifecycle points.
4. Uses decision states and reason codes exclusively from the specification.
5. Maintains attribution immutability and identity uniqueness.

Compliance is verifiable by inspecting telemetry output. Partial compliance is permitted but must be declared explicitly.

See Section 9 of the [specification](spec/genops-spec-v0.1.md) for full details.

---

## What GenOps Does Not Define

GenOps does not define:

- Policy authoring languages
- Control planes
- Budget coordination mechanisms
- Multi-cluster aggregation
- Storage formats
- SDK implementations
- UI requirements
- Model routing strategies
- Authentication or authorization

These concerns may be built on top of GenOps, but are outside the scope of the specification.

---

## Versioning

The specification follows [Semantic Versioning](https://semver.org/).

- **Major** versions introduce breaking changes.
- **Minor** versions may add attributes, events, or reason codes.
- **Patch** versions clarify language without altering semantics.

The current version is: **v0.1.0**

---

## Repository Structure

```
spec/
  genops-spec-v0.1.md
GOVERNANCE.md
CONTRIBUTING.md
```

Future specification versions will be added under the `spec/` directory.

---

## Governance and Contributing

GenOps is governed by a transparent model based on lazy consensus, with explicit voting and supermajority requirements for breaking changes. The governance model defines roles, decision thresholds, version governance, and conflict resolution.

- **[GOVERNANCE.md](GOVERNANCE.md)** -- Project roles, decision model, specification lifecycle, version governance.
- **[CONTRIBUTING.md](CONTRIBUTING.md)** -- Change types, proposal requirements, review process, backward compatibility policy.

---

## Status

GenOps v0.1.0 is a **Working Draft**. Feedback from platform engineers, runtime implementers, and standards communities is welcome.

---

## License

[Apache License 2.0](LICENSE)
