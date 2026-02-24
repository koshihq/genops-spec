# GenOps Governance Specification

**Version:** 0.1.0

**Status:** Working Draft

**Date:** 2026-02-23

**Editors:** GenOps Project Authors

**License:** Apache License 2.0

---

## Abstract

This document defines the GenOps Governance Specification, a vendor-neutral semantic contract for runtime governance of AI workloads. It specifies the minimum vocabulary, invariants, and telemetry conventions required for interoperable governance across independently developed runtimes.

GenOps extends OpenTelemetry semantic conventions into the governance domain. It does not replace OpenTelemetry. It does not define APIs, SDKs, or runtime behavior. It defines the semantic surface that any compliant runtime MUST emit.

---

## Notational Conventions

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [AI Workload Unit Definition](#2-ai-workload-unit-definition)
3. [Deterministic Accounting Invariant](#3-deterministic-accounting-invariant)
4. [Enforcement Decision Model](#4-enforcement-decision-model)
5. [reason_code Taxonomy](#5-reason_code-taxonomy)
6. [SLA Boundary Model](#6-sla-boundary-model)
7. [Required Telemetry Semantic Conventions](#7-required-telemetry-semantic-conventions)
8. [Event Schema Definitions](#8-event-schema-definitions)
9. [Compliance Requirements](#9-compliance-requirements)
10. [Non-Goals](#10-non-goals)
11. [Versioning and Compatibility Policy](#11-versioning-and-compatibility-policy)
- [Appendix A: Attribute Quick Reference](#appendix-a-attribute-quick-reference)
- [Appendix B: Event Quick Reference](#appendix-b-event-quick-reference)
- [References](#references)

---

## 1. Introduction

### 1.1 What GenOps Is

GenOps is a governance specification for AI workloads. It defines the semantic contract — the vocabulary, invariants, and telemetry conventions — that enables runtime governance of AI systems at the infrastructure level.

GenOps standardizes governance semantics for AI workloads. It defines the minimum interoperable surface for runtime governance of AI systems.

### 1.2 What This Specification Defines

This specification defines:

- The **AI Workload Unit**, the atomic boundary of governed work.
- The **Deterministic Accounting Invariant**, a reservation-first model for resource consumption tracking.
- The **Enforcement Decision Model**, the canonical states and fields for governance decisions.
- The **reason_code Taxonomy**, a fixed enumeration of governance decision reasons.
- The **SLA Boundary Model**, the semantic schema for service level enforcement.
- The **Telemetry Semantic Conventions**, the required attributes and events for governance telemetry.
- The **Compliance Requirements**, the criteria for a conformant implementation.

### 1.3 Relationship to OpenTelemetry

GenOps extends OpenTelemetry. It defines additional semantic conventions in the `genops.*` attribute namespace. All GenOps telemetry MUST be representable using standard OpenTelemetry signals (traces, metrics, logs). GenOps does not modify, replace, or conflict with existing OpenTelemetry semantic conventions.

### 1.4 Versioning

This is version 0.1.0 of the GenOps Governance Specification. The specification follows [Semantic Versioning 2.0.0](https://semver.org/). For version 1.0.0 and later, backwards-incompatible changes increment the major version. New attributes, events, or reason codes that do not remove or change existing definitions increment the minor version. For pre-1.0 stability rules, see Section 11.1. See Section 11 for the full versioning and compatibility policy.

---

## 2. AI Workload Unit Definition

### 2.1 Definition

An **AI Workload Unit (AWU)** is a semantic execution boundary. It is the atomic unit of governed AI work. Every governance decision, accounting record, and telemetry event defined in this specification is scoped to exactly one AWU.

An AWU is not defined as a particular runtime construct. It is a governance abstraction. An AWU MUST be representable as an OpenTelemetry span, but MAY be mapped to other execution primitives in runtimes that do not use OpenTelemetry natively.

### 2.2 Identity

Each AWU MUST have a unique, opaque identifier assigned at creation time. This identifier MUST remain stable for the lifetime of the AWU.

When an AWU is represented as an OpenTelemetry span, its identity maps to the tuple `(trace_id, span_id)` as defined by the [W3C Trace Context](https://www.w3.org/TR/trace-context/) specification.

Implementations MUST NOT reuse AWU identifiers.

### 2.3 Attribution Boundary

Every AWU MUST carry the following attribution attributes:

| Attribute | Type | Description |
|---|---|---|
| `genops.team` | string | The team responsible for this workload. |
| `genops.project` | string | The project or product this workload belongs to. |
| `genops.environment` | string | The deployment environment (e.g., `production`, `staging`, `development`). |

These attributes define the governance attribution boundary. They determine how governance decisions, accounting records, and telemetry are aggregated and attributed.

Attribution attributes MUST be immutable for the lifetime of the AWU. Once set, they MUST NOT change.

### 2.4 Lifecycle

An AWU has three ordered phases:

1. **Pre-execution.** Resource reservation and policy evaluation occur. The accounting invariant (Section 3) and enforcement decision (Section 4) are recorded. The pre-execution phase MUST complete before governed resource consumption begins.

2. **Execution.** The AI workload performs its work. The specification does not constrain what occurs during execution.

3. **Post-execution.** Actual resource consumption is recorded and reconciliation occurs. This phase completes the accounting invariant.

A compliant runtime MUST emit telemetry from all three phases for every AWU that reaches execution. If an AWU does not reach execution due to a pre-execution decision, only pre-execution telemetry is required.

### 2.5 Composition

Compound workloads — chains, pipelines, agent loops, multi-step workflows — are composed of multiple AWUs. Each step that constitutes a governed unit of work is a separate AWU.

The composition model for AWUs is out of scope for this specification. Runtimes MAY use OpenTelemetry parent-child span relationships, MAY define their own composition semantics, or MAY treat each AWU independently.

---

## 3. Deterministic Accounting Invariant

### 3.1 Purpose

The Deterministic Accounting Invariant ensures that resource consumption for every AI Workload Unit is tracked with a reservation-before-execution guarantee. This prevents unaccounted resource consumption and enables pre-execution governance decisions based on projected resource availability.

### 3.2 Accounting Unit

The accounting unit is the measurable resource being governed. It MAY be currency (e.g., USD), tokens, compute-seconds, or any other quantifiable unit. The specification does not prescribe which unit to use. The unit MUST be declared on AWUs that reach resource reservation, subject to the conditional exceptions in Section 7.1.

### 3.3 The Invariant

For every AWU that proceeds to execution, the following three conditions MUST hold:

**Condition 1: Reservation.** A resource reservation MUST be recorded before execution begins. The reservation represents the estimated resource consumption for the AWU. The `genops.budget.reservation` event (Section 8.2) MUST be emitted at this point.

**Condition 2: Actual.** The actual resource consumption MUST be recorded after execution completes. This value represents the measured consumption, which MAY differ from the reservation.

**Condition 3: Reconciliation.** A reconciliation MUST occur that compares the reserved amount against the actual amount. The `genops.budget.reconciliation` event (Section 8.3) MUST be emitted at this point.

### 3.4 Required Attributes

The following attributes enforce the accounting invariant:

| Attribute | Type | Phase | Description |
|---|---|---|---|
| `genops.accounting.reserved` | double | Pre-execution | The estimated resource consumption reserved before execution. |
| `genops.accounting.actual` | double | Post-execution | The measured resource consumption after execution. |
| `genops.accounting.unit` | string | Reservation and reconciliation | The unit of measure (e.g., `USD`, `tokens`, `compute_seconds`). |

### 3.5 Failure Semantics

**Reservation failure.** If a reservation cannot be fulfilled — for example, because the available resource capacity is insufficient — the AWU MUST NOT proceed to execution. The enforcement decision (Section 4) MUST record a `BLOCKED` result with an appropriate `reason_code` (Section 5).

**Reconciliation failure.** If reconciliation cannot be completed — for example, because the actual consumption value is unavailable — the runtime MUST NOT silently discard the accounting record. Implementations SHOULD emit a reconciliation event with the best available data and SHOULD flag the record as incomplete.

**Determinism.** Given identical reservation inputs and identical governance state, the reservation outcome MUST be identical. Governance state encompasses all inputs that influence reservation decisions, including but not limited to resource availability, active policies, and budget positions.

---

## 4. Enforcement Decision Model

### 4.1 Decision States

Every governance evaluation produces exactly one of the following decision states:

| State | Semantics |
|---|---|
| `ALLOWED` | The AWU is authorized to proceed. No governance constraint was violated. |
| `BLOCKED` | The AWU MUST NOT proceed. A governance constraint was violated. |
| `WARNING` | The AWU is authorized to proceed. A governance condition was noted but did not block execution. |
| `RATE_LIMITED` | The AWU was rejected or deferred due to rate constraints. The runtime MAY retry after a delay. |

These four states are the complete set for version 0.1. Implementations MUST NOT emit decision states outside this set.

### 4.2 Required Fields

Every enforcement decision MUST include:

| Field | Type | Requirement | Description |
|---|---|---|---|
| `genops.policy.result` | string | REQUIRED | One of: `ALLOWED`, `BLOCKED`, `WARNING`, `RATE_LIMITED`. |
| `genops.policy.reason_code` | string | REQUIRED when result is not `ALLOWED` | A value from the reason_code taxonomy (Section 5). |

### 4.3 Decision Immutability

An enforcement decision, once recorded, MUST NOT be modified. If a subsequent evaluation produces a different result (for example, a retry after a rate limit), it MUST be recorded as a separate decision event. Retries SHOULD be represented as distinct AWUs.

### 4.4 Attribution

Every enforcement decision is scoped to exactly one AWU. The decision inherits the attribution boundary of its AWU (Section 2.3). Implementations MUST NOT emit enforcement decisions without a parent AWU.

---

## 5. reason_code Taxonomy

### 5.1 Version

This is the reason_code taxonomy for GenOps Specification version 0.1. The taxonomy is a fixed enumeration. All values are defined in this section.

### 5.2 Enumeration

| reason_code | Category | Description |
|---|---|---|
| `BUDGET_EXCEEDED` | Budget | The resource budget has been exhausted. No remaining capacity. |
| `BUDGET_RESERVATION_FAILED` | Budget | The requested reservation could not be fulfilled against the available budget. |
| `POLICY_DENY_MODEL` | Policy | The requested model is not permitted by the governing policy. |
| `POLICY_DENY_REGION` | Policy | The requested provider region is not permitted by the governing policy. |
| `POLICY_DENY_CONTENT` | Policy | The request was denied due to an organization-defined policy rule evaluating request content. |
| `RATE_LIMITED` | Rate | The request exceeded the permitted request rate. |
| `SLA_LATENCY_VIOLATION` | SLA | The predicted or observed latency exceeds the SLA boundary. |
| `PROVIDER_ERROR_UPSTREAM` | Provider | The upstream provider returned an error preventing fulfillment. |
| `ACCOUNTING_INVARIANT_VIOLATION` | Accounting | The deterministic accounting invariant (Section 3) was violated. |

### 5.3 Usage Requirements

When `genops.policy.result` is not `ALLOWED`, the `genops.policy.reason_code` attribute MUST be set to exactly one value from the enumeration above, or to an extension code (Section 5.5).

When `genops.policy.result` is `ALLOWED`, the `genops.policy.reason_code` attribute MUST NOT be set.

### 5.4 Non-Freeform Requirement

The `reason_code` field MUST contain a value from this taxonomy or a valid extension code. Freeform strings are not permitted. This requirement enables deterministic processing, aggregation, and alerting on governance decisions without natural language parsing.

### 5.5 Extension Codes

Implementations MAY define additional reason codes using the `x_` prefix (e.g., `x_custom_compliance_check`, `x_internal_quota`). Extension codes MUST begin with `x_` followed by one or more lowercase alphanumeric characters or underscores.

Non-prefixed reason codes are reserved for the specification. Implementations MUST NOT define non-prefixed reason codes outside this specification.

### 5.6 Taxonomy Evolution Rules

For specification versions 1.0.0 and later, new reason codes MAY be added in future minor versions. Existing reason codes MUST NOT be removed or redefined within the same major version. Consumers of reason_code values SHOULD handle unrecognized codes gracefully.

For specification versions prior to 1.0.0, reason code stability follows the pre-1.0 stability policy in Section 11.1.

---

## 6. SLA Boundary Model

### 6.1 Purpose

The SLA Boundary Model defines the semantic schema for service level enforcement on AI Workload Units. It specifies how SLA targets are declared and how compliance is measured. It does not define SLA tier structures, escalation procedures, or remediation workflows.

### 6.2 Boundary Definition

An SLA boundary is a declared performance target against which an AWU is evaluated. In version 0.1, the specification defines a single boundary dimension: latency.

### 6.3 Attributes

The following attributes define the SLA boundary. These attributes are OPTIONAL (see Section 7.3). When SLA enforcement is active for an AWU, all attributes in this section that are marked REQUIRED below MUST be present.

| Attribute | Type | Requirement | Description |
|---|---|---|---|
| `genops.sla.tier` | string | REQUIRED | An opaque, organization-defined tier identifier (e.g., `basic`, `premium`). |
| `genops.sla.latency_target_ms` | int64 | REQUIRED | The maximum acceptable end-to-end latency in milliseconds. |
| `genops.sla.actual_latency_ms` | int64 | REQUIRED | The measured end-to-end latency in milliseconds. |
| `genops.sla.met` | boolean | REQUIRED | `true` if `actual_latency_ms <= latency_target_ms`, `false` otherwise. |

### 6.4 Enforcement

When `genops.sla.met` is `false`, the runtime MAY emit an enforcement decision with `reason_code` of `SLA_LATENCY_VIOLATION`. The specification does not mandate a specific enforcement response to SLA violations. Runtimes MAY log, alert, reroute, or take no action.

### 6.5 Scope

The SLA boundary is evaluated per-AWU. The specification does not define aggregation of SLA compliance across multiple AWUs, time windows, or organizational units. Aggregation semantics are an implementation concern.

---

## 7. Required Telemetry Semantic Conventions

All attributes defined in this specification use the `genops.*` namespace. Implementations MUST NOT use alternative namespaces for the concepts defined here.

### 7.1 Required Attributes

The following attributes are the normative minimum. A compliant runtime MUST emit these attributes on every AWU, subject to the conditional exceptions below.

| Attribute | Type | Description |
|---|---|---|
| `genops.team` | string | Team responsible for this workload. |
| `genops.project` | string | Project this workload belongs to. |
| `genops.environment` | string | Deployment environment. |
| `genops.operation.name` | string | Human-readable name of the operation. |
| `genops.operation.type` | string | Category of operation (e.g., `inference`, `embedding`, `fine_tune`). |
| `genops.accounting.reserved` | double | Resource reservation recorded before execution. |
| `genops.accounting.actual` | double | Actual resource consumption recorded after execution. |
| `genops.accounting.unit` | string | Unit of measure for accounting values. |
| `genops.policy.result` | string | Enforcement decision: `ALLOWED`, `BLOCKED`, `WARNING`, or `RATE_LIMITED`. |
| `genops.policy.reason_code` | string | Reason code from the taxonomy in Section 5. REQUIRED when `genops.policy.result` is not `ALLOWED`. |
| `genops.spec.version` | string | GenOps specification version implemented, SemVer `MAJOR.MINOR.PATCH`. |

`genops.spec.version` MUST be emitted on every AWU, including AWUs that do not reach resource reservation due to a pre-execution decision.

For AWUs that do not reach resource reservation due to a pre-execution decision (for example, `BLOCKED` or `RATE_LIMITED`), `genops.accounting.reserved`, `genops.accounting.actual`, and `genops.accounting.unit` MAY be omitted. In such cases, the omission does not constitute non-compliance with Section 7.1.

For AWUs that reach resource reservation, `genops.accounting.reserved` and `genops.accounting.unit` MUST be present.

For AWUs that reach execution completion, `genops.accounting.actual` MUST be present.

### 7.2 Required Events

A compliant runtime MUST emit the following events at the specified lifecycle points. Event schemas are defined in Section 8.

| Event Name | Lifecycle Phase | Description |
|---|---|---|
| `genops.policy.evaluated` | Pre-execution | Emitted on every enforcement decision. |
| `genops.budget.reservation` | Pre-execution | Emitted when a resource reservation is recorded. |
| `genops.budget.reconciliation` | Post-execution | Emitted when actual consumption is reconciled against the reservation. |

#### 7.2.1 Non-Executing AWU Event Requirements

If an AWU does not reach resource reservation due to a pre-execution decision (for example, `BLOCKED` or `RATE_LIMITED`), only the `genops.policy.evaluated` event is REQUIRED.

If an AWU does not reach execution but is stopped after reservation and before execution begins, the `genops.budget.reservation` event MUST have been emitted. In this case, `genops.budget.reconciliation` is not required.

An AWU that does not reach execution MUST NOT emit `genops.budget.reconciliation`.

### 7.3 Optional Extensions

The following attributes and events are OPTIONAL. They are not required for compliance but are defined here to promote interoperability across implementations that choose to support them.

#### Accounting Detail

| Attribute | Type | Description |
|---|---|---|
| `genops.accounting.reconciliation_delta` | double | Difference between actual and reserved (`actual - reserved`). |

#### Budget Structure

| Attribute | Type | Description |
|---|---|---|
| `genops.budget.name` | string | Budget identifier. |
| `genops.budget.allocated` | double | Total budget allocated. |
| `genops.budget.consumed` | double | Budget consumed to date. |
| `genops.budget.remaining` | double | Budget remaining. |
| `genops.budget.period` | string | Budget period identifier (e.g., `2026-02`, `daily`). |

#### Policy Detail

| Attribute | Type | Description |
|---|---|---|
| `genops.policy.name` | string | Policy identifier that produced the decision. |
| `genops.policy.reason` | string | Human-readable explanation of the decision. |
| `genops.policy.timestamp` | int64 | Decision timestamp in Unix epoch milliseconds. |

#### Cost Breakdown

| Attribute | Type | Description |
|---|---|---|
| `genops.cost.total` | double | Total monetary cost. |
| `genops.cost.currency` | string | Currency code per [ISO 4217](https://www.iso.org/iso-4217-currency-codes.html). |
| `genops.cost.provider` | string | AI provider name. |
| `genops.cost.model` | string | Model identifier. |

#### Token Counts

| Attribute | Type | Description |
|---|---|---|
| `genops.tokens.input` | int64 | Input token count. |
| `genops.tokens.output` | int64 | Output token count. |
| `genops.tokens.total` | int64 | Total token count. |

#### Evaluation

| Attribute | Type | Description |
|---|---|---|
| `genops.eval.metric` | string | Evaluation metric name. |
| `genops.eval.score` | double | Evaluation score. |
| `genops.eval.threshold` | double | Pass/fail threshold. |
| `genops.eval.passed` | boolean | Whether the evaluation passed. |

#### SLA

See Section 6.3 for the full SLA attribute table. All SLA attributes are optional at the AWU level. When SLA enforcement is active, the requirements in Section 6.3 apply.

#### Customer Context

| Attribute | Type | Description |
|---|---|---|
| `genops.customer_id` | string | Customer or tenant identifier. |
| `genops.feature` | string | Feature or capability identifier. |
| `genops.session_id` | string | Session identifier. |
| `genops.cost_center` | string | Cost center for financial attribution. |

#### Optional Events

The following events MAY be emitted. They are not required for compliance.

| Event Name | Description |
|---|---|
| `genops.cost.calculated` | Emitted after monetary cost computation completes. |
| `genops.budget.alert` | Emitted when budget utilization exceeds a threshold. |
| `genops.evaluation.complete` | Emitted after a quality or safety evaluation completes. |

---

## 8. Event Schema Definitions

This section defines the normative schema for each required event and the schema for optional events.

### 8.1 genops.policy.evaluated

**When emitted:** Pre-execution, on every enforcement decision.

**Required attributes:**

| Attribute | Type | Description |
|---|---|---|
| `genops.policy.result` | string | `ALLOWED`, `BLOCKED`, `WARNING`, or `RATE_LIMITED`. |
| `genops.policy.reason_code` | string | From the taxonomy in Section 5. REQUIRED when result is not `ALLOWED`. |

When `genops.policy.result` is `ALLOWED`, `genops.policy.reason_code` MUST NOT be set.

**Optional attributes:**

| Attribute | Type | Description |
|---|---|---|
| `genops.policy.name` | string | Policy identifier. |
| `genops.policy.reason` | string | Human-readable explanation. |

### 8.2 genops.budget.reservation

**When emitted:** Pre-execution, when a resource reservation is recorded per the accounting invariant (Section 3).

**Required attributes:**

| Attribute | Type | Description |
|---|---|---|
| `genops.accounting.reserved` | double | The amount reserved. |
| `genops.accounting.unit` | string | The unit of measure. |

**Optional attributes:**

| Attribute | Type | Description |
|---|---|---|
| `genops.budget.name` | string | Budget identifier the reservation is against. |
| `genops.budget.remaining` | double | Remaining budget after this reservation. |

### 8.3 genops.budget.reconciliation

**When emitted:** Post-execution, when actual consumption is reconciled against the reservation per the accounting invariant (Section 3).

**Required attributes:**

| Attribute | Type | Description |
|---|---|---|
| `genops.accounting.actual` | double | The measured consumption. |
| `genops.accounting.reserved` | double | The amount that was reserved. |
| `genops.accounting.unit` | string | The unit of measure. |

**Optional attributes:**

| Attribute | Type | Description |
|---|---|---|
| `genops.accounting.reconciliation_delta` | double | `actual - reserved`. |
| `genops.budget.name` | string | Budget identifier. |

### 8.4 Optional Event Schemas

#### genops.cost.calculated

**When emitted:** After monetary cost computation completes.

| Attribute | Type | Requirement | Description |
|---|---|---|---|
| `genops.cost.total` | double | REQUIRED | Total monetary cost. |
| `genops.cost.currency` | string | REQUIRED | ISO 4217 currency code. |
| `genops.cost.provider` | string | OPTIONAL | AI provider name. |
| `genops.cost.model` | string | OPTIONAL | Model identifier. |

#### genops.budget.alert

**When emitted:** When budget utilization exceeds a configured threshold.

| Attribute | Type | Requirement | Description |
|---|---|---|---|
| `genops.budget.name` | string | REQUIRED | Budget identifier. |
| `genops.budget.consumed` | double | REQUIRED | Amount consumed. |
| `genops.budget.allocated` | double | REQUIRED | Total allocation. |

#### genops.evaluation.complete

**When emitted:** After a quality or safety evaluation completes.

| Attribute | Type | Requirement | Description |
|---|---|---|---|
| `genops.eval.metric` | string | REQUIRED | Evaluation metric name. |
| `genops.eval.score` | double | REQUIRED | Evaluation score. |
| `genops.eval.threshold` | double | OPTIONAL | Pass/fail threshold. |
| `genops.eval.passed` | boolean | OPTIONAL | Whether the evaluation passed. |

### 8.5 Event Stability

For specification versions 1.0.0 and later, the required events listed in Section 7.2 (schemas in Sections 8.1–8.3) have **stability: stable**. Their schemas MUST NOT change within the same major version.

For versions prior to 1.0.0, required event schemas follow the pre-1.0 stability policy in Section 11.1.

All events in Section 8.4 have **stability: experimental**. Their schemas MAY change in minor versions.

---

## 9. Compliance Requirements

### 9.1 Compliant Runtime

A **GenOps v0.1 compliant runtime** MUST satisfy all of the following:

1. Emit all required attributes defined in Section 7.1 on every AWU, subject to any conditional exceptions defined in Section 7.1.
2. Implement the Deterministic Accounting Invariant defined in Section 3 — specifically, the reservation-before-execution guarantee.
3. Emit all required events defined in Section 7.2 at the specified lifecycle points.
4. Use the `genops.policy.result` values exclusively from the set defined in Section 4.1.
5. Use `genops.policy.reason_code` values exclusively from the taxonomy defined in Section 5 or from valid extension codes (Section 5.5).
6. Maintain AWU identity uniqueness (Section 2.2).
7. Maintain attribution immutability (Section 2.3).
8. Emit required attributes and required events with semantically meaningful values. A compliant runtime MUST NOT emit required attributes or required events with null, empty, placeholder, or semantically meaningless values.

### 9.2 Partial Compliance

An implementation that satisfies requirements 1, 4, 5, 6, 7, and 8 above but does not implement the Deterministic Accounting Invariant (requirements 2 and 3) is considered **partially compliant**. Partial compliance indicates a telemetry-conformant implementation that emits well-formed governance attributes and events but does not enforce the reservation-before-execution guarantee. Partially compliant runtimes MUST NOT claim full GenOps compliance. Partially compliant runtimes MUST declare their compliance level as "GenOps v0.1 — partial".

### 9.3 Non-Compliance

An implementation that does not satisfy the requirements in Section 9.1 or 9.2 MUST NOT claim GenOps compliance.

### 9.4 Testability

Compliance MUST be verifiable by inspecting the telemetry output of a runtime. A conformance test suite MAY be provided separately. The specification itself does not define test procedures.

---

## 10. Non-Goals

The following are explicitly out of scope for this specification:

- **Policy authoring language.** The specification defines decision states and reason codes, not how policies are written, stored, or evaluated.
- **User interface requirements.** The specification defines telemetry semantics, not presentation.
- **Control plane coordination.** Multi-node, multi-cluster, or distributed governance coordination is not defined.
- **Multi-cluster aggregation.** Aggregation of governance telemetry across clusters or regions is not defined.
- **Cost provider adapters.** How cost data is obtained from AI providers is an implementation concern.
- **SDK implementation details.** The specification defines the semantic contract, not library design.
- **Persistence models.** How telemetry is stored, indexed, or queried is not defined.
- **Prompt or response content schemas.** The specification does not define AI input/output formats.
- **Model selection or routing logic.** How models are selected, load-balanced, or routed is an implementation concern.
- **Authentication and authorization.** Infrastructure-level access control is not defined.
- **Pricing tables or cost databases.** The specification defines how to record resource consumption, not how to calculate monetary cost.
- **Agent orchestration.** Multi-agent graph topology, coordination protocols, and agent lifecycle management are not defined.
- **Data retention or storage format.** The specification defines wire-level semantics, not storage.
- **Specific SLA tier definitions.** Tiers are organization-defined. The specification defines the attribute schema only.

---

## 11. Versioning and Compatibility Policy

GenOps follows [Semantic Versioning 2.0.0](https://semver.org/).

The GenOps specification is the authoritative definition of event names, attribute names, decision states, and reason codes. Implementations MUST NOT redefine or reinterpret these definitions.

### 11.1 Pre-1.0 Stability (0.x)

Versions prior to 1.0.0 are considered stabilization releases. Breaking changes MAY occur in minor version increments. Required attributes, required events, decision states, and reason codes MAY evolve. Implementations MUST declare the exact GenOps version they implement.

### 11.2 1.0.0 and Later

Starting at version 1.0.0:

- Required attributes (Section 7.1) and required events (Section 7.2) are stable. Their semantics, types, and cardinality MUST NOT change within the same major version.
- Optional attributes (Section 7.3) and events designated **stability: experimental** by this specification (Sections 8.4–8.5) MAY evolve in minor versions. They are not covered by the stability guarantee.
- `reason_code` values defined in the major version MUST NOT be removed or redefined within that major version.
- Backwards-incompatible changes — including removal, renaming, or redefinition of required attributes, required events, decision states, or reason codes — MUST increment the major version.

### 11.3 Version Declaration

A runtime MUST declare the specification version it implements using the format `MAJOR.MINOR.PATCH` as defined by [Semantic Versioning 2.0.0](https://semver.org/). At minimum, this declaration MUST be machine-verifiable via the `genops.spec.version` attribute (Section 7.1). The canonical location for this machine-verifiable declaration is the AWU span attributes (that is, the `genops.spec.version` attribute emitted on the AWU). Additional declaration mechanisms are implementation-defined.

A runtime MAY declare compatibility with a minor version range within the same major version, provided all versions in the declared range share identical required attributes, required events, decision states, and reason codes. The runtime MUST be able to demonstrate this identity via documentation, test artifact, or the published conformance suite.

### 11.4 Runtime Compatibility

A runtime claiming GenOps compliance MUST:

1. Declare the GenOps specification version it implements (Section 11.3).
2. Emit all REQUIRED attributes and REQUIRED events required by that version, including any conditional or phase-specific exceptions defined in Sections 7.1 and 7.2.1.
3. Pass the published GenOps conformance suite for the declared version, if available.

A runtime MAY emit OPTIONAL attributes and EXPERIMENTAL events defined by that version. A runtime MAY emit extension reason codes permitted under Section 5.5. A runtime MAY emit additional non-GenOps telemetry, but such telemetry MUST NOT conflict with or redefine GenOps required definitions. A runtime MUST NOT claim compliance if REQUIRED definitions are missing or reinterpreted.

In the absence of a published conformance suite, compliance MUST be verifiable through inspection of emitted telemetry as defined in Section 9.4.

A runtime MUST NOT claim compliance with a specification version it does not implement. Partial compliance (Section 9.2) MUST be declared explicitly, including the specification version. Compliance criteria are defined in Section 9; this section defines version declaration and compatibility expectations for compliance claims.

### 11.5 Change Control (Non-Normative)

Semantic changes to required definitions are spec-first: the specification SHOULD be updated before any reference implementation reflects the change. Reference implementations SHOULD update to match the specification within the same release cycle.

---

## Appendix A: Attribute Quick Reference

### Required Attributes

| Attribute | Type | Section |
|---|---|---|
| `genops.team` | string | 2.3 |
| `genops.project` | string | 2.3 |
| `genops.environment` | string | 2.3 |
| `genops.operation.name` | string | 7.1 |
| `genops.operation.type` | string | 7.1 |
| `genops.accounting.reserved` | double | 3.4 |
| `genops.accounting.actual` | double | 3.4 |
| `genops.accounting.unit` | string | 3.4 |
| `genops.policy.result` | string | 4.2 |
| `genops.policy.reason_code` | string | 4.2, 5.3 |
| `genops.spec.version` | string | 7.1, 11.3 |

### Optional Attributes

| Attribute | Type | Section |
|---|---|---|
| `genops.accounting.reconciliation_delta` | double | 7.3 |
| `genops.budget.name` | string | 7.3 |
| `genops.budget.allocated` | double | 7.3 |
| `genops.budget.consumed` | double | 7.3 |
| `genops.budget.remaining` | double | 7.3 |
| `genops.budget.period` | string | 7.3 |
| `genops.policy.name` | string | 7.3 |
| `genops.policy.reason` | string | 7.3 |
| `genops.policy.timestamp` | int64 | 7.3 |
| `genops.cost.total` | double | 7.3 |
| `genops.cost.currency` | string | 7.3 |
| `genops.cost.provider` | string | 7.3 |
| `genops.cost.model` | string | 7.3 |
| `genops.tokens.input` | int64 | 7.3 |
| `genops.tokens.output` | int64 | 7.3 |
| `genops.tokens.total` | int64 | 7.3 |
| `genops.eval.metric` | string | 7.3 |
| `genops.eval.score` | double | 7.3 |
| `genops.eval.threshold` | double | 7.3 |
| `genops.eval.passed` | boolean | 7.3 |
| `genops.sla.tier` | string | 6.3 |
| `genops.sla.latency_target_ms` | int64 | 6.3 |
| `genops.sla.actual_latency_ms` | int64 | 6.3 |
| `genops.sla.met` | boolean | 6.3 |
| `genops.customer_id` | string | 7.3 |
| `genops.feature` | string | 7.3 |
| `genops.session_id` | string | 7.3 |
| `genops.cost_center` | string | 7.3 |

---

## Appendix B: Event Quick Reference

### Required Events

| Event Name | Phase | Stability | Section |
|---|---|---|---|
| `genops.policy.evaluated` | Pre-execution | Stable (1.0+), see 8.5 (0.x) | 8.1 |
| `genops.budget.reservation` | Pre-execution | Stable (1.0+), see 8.5 (0.x) | 8.2 |
| `genops.budget.reconciliation` | Post-execution | Stable (1.0+), see 8.5 (0.x) | 8.3 |

### Optional Events

| Event Name | Stability | Section |
|---|---|---|
| `genops.cost.calculated` | Experimental | 8.4 |
| `genops.budget.alert` | Experimental | 8.4 |
| `genops.evaluation.complete` | Experimental | 8.4 |

---

## References

- [RFC 2119: Key words for use in RFCs to Indicate Requirement Levels](https://www.rfc-editor.org/rfc/rfc2119)
- [OpenTelemetry Specification](https://opentelemetry.io/docs/specs/otel/)
- [OpenTelemetry Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/)
- [W3C Trace Context](https://www.w3.org/TR/trace-context/)
- [Semantic Versioning 2.0.0](https://semver.org/)
- [ISO 4217 Currency Codes](https://www.iso.org/iso-4217-currency-codes.html)

---

*GenOps Project Authors. Licensed under Apache License 2.0.*
