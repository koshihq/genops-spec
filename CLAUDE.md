# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Strategic Context

### Mission: "Governance for AI, Built on OpenTelemetry"

GenOps AI builds on the **OpenTelemetry foundation** — interoperable by design, independent governance for AI systems.
It standardizes **cost, policy, compliance, and evaluation telemetry** for AI workloads across internal teams, departments, and per-customer usage.

**GenOps** extends standard LLM observability with *why and how* — the governance layer that turns telemetry into actionable accountability for AI systems.

---

## Project Overview

This Python SDK is the reference implementation of **GenOps-OTel** — the open-source governance telemetry layer for AI systems.

It provides a **vendor-neutral, OTel-native SDK** that enables teams to:

* Capture and export governance signals: cost, budget, policy, compliance, and evaluation metrics
* Integrate with existing observability stacks (Datadog, Honeycomb, Grafana Tempo, etc.)
* Feed data into governance dashboards and enterprise control planes

---

### Core Purpose

* Extend the OpenTelemetry signal model to cover **governance semantics** (`genops.cost.*`, `genops.policy.*`, `genops.eval.*`, `genops.budget.*`)
* Provide **interoperable adapters** for LLM providers and frameworks (OpenAI, Anthropic, Gemini, Bedrock, Mistral, LangChain, etc.)
* Enable **transparent per-team, per-feature, per-customer cost governance** across AI systems
* Serve as the **foundation for enterprise governance automation**

---

## Open Source Strategy

### Licensing & Governance

* **License:** Apache 2.0 (permissive and OSS-friendly)
* **Governance:** Open community development with maintainer stewardship
* **Community:** Public RFC process for spec evolution; open contribution model
* **Vendor Neutrality:** Aligns with OpenTelemetry standards; avoids lock-in with any observability or model vendor

### Philosophy: Developer-first, Governance-aware

* Minimal setup and frictionless integration
* Multipattern instrumentation (decorators, context managers, auto-detect)
* Zero-dashboards-by-default → reuse existing observability stack
* Extensible adapter architecture to empower community contributions

---

## Architectural Principles

### Design Layering

```
OpenTelemetry (foundation)
   └── GenOps-OTel (AI governance: cost, policy, compliance, evaluation)
```

GenOps extends OpenTelemetry with AI governance semantics — interoperable by design, independent by governance.
Standard OTLP signals enable any organization to maintain full autonomy and transparency.

---

### Core Components

1. **Governance Semantics Spec:** Defines official telemetry keys for cost, policy, and evaluation
2. **Provider Adapters:** Thin wrappers for major AI providers and frameworks
3. **Collector Processors:** OpenTelemetry Collector extensions (Go-based) for cost, redaction, policy, and budget processing
4. **CLI / Local Dashboard:** Minimal local tools for developers
5. **Exporter Configs:** OTLP examples for all major observability backends

---

### Integration Patterns

* `@track_usage`: Function-level instrumentation decorator
* `with genops.track():`: Context manager for block-level tracking
* `enforce_policy`: Declarative runtime guardrail enforcement
* Auto-instrumentation for OpenAI, Anthropic, Bedrock, Gemini, LangChain, LlamaIndex, Chroma, and others

---

## Technical Standards

### Package Structure

```
genops-ai/
├── src/genops/
│   ├── core/              # Core telemetry engine
│   ├── providers/         # Provider adapters (OpenAI, Anthropic, etc.)
│   ├── exporters/         # OTLP exporters
│   ├── processors/        # OTel Collector processors
│   ├── config/            # Config & environment handling
│   └── cli/               # Local CLI tools
├── tests/
├── examples/
├── docs/
├── pyproject.toml
└── CONTRIBUTING.md
```

### Development Tools

* **Build**: `python -m build`
* **Test**: `pytest`
* **Format**: `ruff format`
* **Lint**: `ruff check`
* **Type Check**: `mypy src/`

---

## Ecosystem Vision

### Cross-Stack Tracking Without Tool Replacement

GenOps connects your existing AI tools without replacing them — it adds the **cross-stack tracking layer your AI stack is missing**.

**Keep the tools you love, add the tracking you need:**

**✅ AI & LLM Ecosystem**
* **OpenAI** - Direct API tracking with cost attribution
* **Anthropic** - Complete request/response telemetry  
* **AWS Bedrock** - Multi-model cost aggregation
* **Google Gemini** - Usage and performance tracking
* **Hugging Face** - Inference endpoint monitoring
* **OpenRouter** - Multi-provider routing telemetry
* **LangChain** - Framework-level operation tracking

**✅ Observability Platforms**
* **Datadog** - Native OpenTelemetry integration
* **Grafana/Tempo** - Distributed tracing support
* **Honeycomb** - AI-specific dashboard templates
* **Prometheus** - Cost and usage metrics collection
* **Standard OTLP** - Works with 15+ observability platforms

**☐ Coming Soon**
* **LlamaIndex** - RAG pipeline tracking
* **Replicate** - Model hosting integration
* **Mistral** - European AI provider support
* **Ollama** - Local model tracking

### Community Growth & Contributions

**5-minute contributions welcome!** Every improvement helps the community grow.

**🚀 Quick Wins for New Contributors:**
* Add new AI provider adapters - follow established patterns
* Create dashboard templates for popular observability platforms  
* Improve documentation with real-world examples
* Fix CI tests and improve development workflows

**🏗️ Bigger Impact Opportunities:**
* Cross-stack tracking patterns for complex AI workflows
* Enterprise governance features and policy automation
* Performance optimization and scaling improvements
* Integration with emerging AI platforms and tools

**📋 Community Standards:**
* Public RFC process for major feature decisions (like OpenTelemetry SIGs)
* Recognition for high-quality integrations and community contributions
* Open development with transparent roadmaps and contributor guidelines
* Collaborative approach to expanding the AI observability ecosystem

---

## Success Metrics

### Community Goals

* 1K+ GitHub stars
* 10+ provider adapters  
* 5+ supported observability platforms
* Community RFC adoption and schema contributions
* Active FinOps Foundation and enterprise adoption

---

## Framework Adapter Development

### Framework Adapter Architecture Pattern

**Required 4-Module Structure:**
```
providers/{framework}/
├── adapter.py           # Main GenOps{Framework}Adapter
├── cost_aggregator.py   # Multi-provider cost tracking
├── specialized_monitor.py # Framework-specific instrumentation
└── registration.py      # Auto-instrumentation registry
```

### Key Design Patterns

**1. Base Provider Interface**
- All adapters inherit from `BaseFrameworkProvider`
- Standardized method signatures ensure consistency
- Framework detection and graceful degradation built-in

**2. Context Manager Pattern**
```python
# Cost tracking with automatic cleanup
with create_chain_cost_context(chain_id) as context:
    context.add_llm_call(provider, model, tokens_in, tokens_out)
    # Automatic finalization and telemetry export
```

**3. Multi-Provider Cost Aggregation**
```python
@dataclass
class FrameworkCostSummary:
    cost_by_provider: Dict[str, float]  # OpenAI, Anthropic, etc.
    cost_by_model: Dict[str, float]     # Model-specific costs
    unique_providers: Set[str]          # Automatic deduplication
```

### Implementation Best Practices

**Architecture:**
1. **Modular Design**: Separate concerns into focused modules
2. **Context Management**: Use context managers for operation lifecycle
3. **Fallback Strategies**: Graceful degradation when dependencies unavailable
4. **Provider Agnostic**: Design for multiple backend providers

**Cost Tracking:**
1. **Dataclass Structures**: Type-safe, serializable cost objects
2. **Generic Fallbacks**: Cost estimation when provider-specific calculators unavailable
3. **Nested Operations**: Support for complex operation hierarchies
4. **Real-time Aggregation**: Automatic cost summaries and breakdowns

**Telemetry Standards:**
1. **Consistent Naming**: `genops.{framework}.{operation}.{metric}` pattern
2. **Rich Attributes**: Governance attributes (team, project, customer_id) propagation
3. **Framework-Specific Metrics**: Specialized telemetry for each framework's features
4. **Performance Tracking**: Automatic timing and resource usage capture

### Anti-Patterns to Avoid

**Architecture:**
- ❌ Hardcoded provider detection logic
- ❌ Incomplete error handling in cost calculations
- ❌ Tight coupling to specific framework versions
- ❌ Missing configuration options for runtime customization

**Testing:**
- ❌ Over-mocking (50+ line mock setups)
- ❌ Testing implementation details vs behavior
- ❌ Insufficient boundary condition coverage

---

## Developer Experience Excellence Standards

### Universal Development Principles

These standards MUST be applied to all development work to ensure consistent developer-first excellence:

### **1. Progressive Complexity Architecture (The "Golden Path")**

**Mandatory Development Pattern:**
- **5-minute value demonstration** - Zero-code auto-instrumentation with immediate working results
- **30-minute guided exploration** - Manual instrumentation with clear governance examples  
- **2-hour mastery path** - Advanced features, production patterns, and enterprise deployment

**Implementation Requirements:**
- Every feature MUST demonstrate value within 5 minutes of installation
- Auto-instrumentation MUST work with zero code changes to existing applications
- Progressive learning paths MUST be clearly documented with working examples
- Each complexity level MUST build naturally on the previous level

### **2. Dual Documentation Strategy (Non-Negotiable)**

**Required Documentation Architecture:**
1. **Quickstart Guides** (`{feature}-quickstart.md`)
   - Maximum 5-minute time-to-value
   - Single working example that can be copied/pasted
   - Zero-code auto-instrumentation demonstration
   - Basic troubleshooting with actionable fixes

2. **Comprehensive Integration Guides** (`integrations/{feature}.md`)
   - Complete feature documentation with multiple examples
   - All integration patterns (auto, manual, context managers)
   - Advanced use cases and production deployment patterns
   - Performance considerations and scaling guidance
   - Complete API reference with all parameters

**Quality Gates:**
- New developers must achieve value within 5 minutes using quickstart
- All examples must be executable immediately without modification
- Documentation must answer the top 10 most common questions proactively

### **3. Universal Validation and Error Handling Framework**

**Mandatory Validation Standards:**
```python
# Every provider/feature MUST implement this pattern
def validate_setup() -> ValidationResult:
    """Comprehensive setup validation with structured results."""
    # Check environment variables and configuration
    # Verify dependencies and versions
    # Test live connectivity where applicable
    # Return actionable error messages with specific fixes

def print_validation_result(result: ValidationResult) -> None:
    """User-friendly display with fix suggestions."""
    # Structured output with clear success/error indicators
    # Specific fix suggestions for each error type
    # Links to documentation for complex issues
```

**Error Handling Excellence:**
- Graceful degradation when dependencies are unavailable
- Specific error messages with actionable solutions (not generic failures)
- Built-in retry logic with exponential backoff for network operations
- Context preservation during failures for debugging
- Comprehensive diagnostic information in debug mode

### **4. API Design Consistency and Naming Standards**

**Universal Naming Conventions (Enforced):**
- `instrument_{provider}()` for main adapter factory functions
- `auto_instrument()` for zero-code setup (must always be available)
- `validate_setup()` and `print_validation_result()` for all providers
- `{feature}_create()` methods following established provider conventions
- `multi_provider_cost_tracking()` for unified cost aggregation

**Governance Attribute Standards:**
```python
# These MUST be supported consistently across ALL features
standard_governance_attrs = {
    "team": str,              # Cost attribution and access control
    "project": str,           # Project-level cost tracking  
    "customer_id": str,       # Customer attribution for billing
    "environment": str,       # Environment segregation (dev/staging/prod)
    "cost_center": str,       # Financial reporting alignment
    "feature": str           # Feature-level cost attribution
}
```

### **5. Testing Excellence Framework (Mandatory Standards)**

**Required Test Coverage (75+ Tests per Major Feature):**
- **Unit Tests** (~35 tests): Individual component validation
- **Integration Tests** (~17 tests): End-to-end workflow verification
- **Cross-Provider Tests** (~24 tests): Multi-provider compatibility scenarios
- **Error Handling Tests**: Comprehensive failure mode coverage
- **Performance Tests**: Load and scalability validation

**Critical Testing Patterns:**
- Context manager lifecycle testing (`__enter__`/`__exit__` scenarios)
- Exception handling within instrumentation code
- Cost calculation accuracy across all supported providers
- Framework detection and graceful degradation behaviors
- Real-world scenario simulation (not just unit test mocking)

### **6. Production-Ready Architecture Patterns**

**Enterprise Workflow Templates (Required):**
```python
# Context manager pattern for complex operations
with production_workflow_context(workflow_name, customer_id, **kwargs) as (span, workflow_id):
    # Multi-step operations with unified governance
    # Automatic cost attribution and error handling
    # Performance monitoring and alerting integration
```

**Performance and Scaling Considerations:**
- Sampling configuration for high-volume applications
- Async telemetry export to minimize application overhead
- Configurable log levels and debug modes
- Circuit breaker patterns for external API dependencies
- Graceful degradation when observability systems are unavailable

### **7. Cost Optimization and Multi-Provider Excellence**

**Universal Cost Tracking Requirements:**
- Real-time cost calculation and attribution across all providers
- Multi-provider cost aggregation with unified governance
- Budget-constrained operation strategies
- Migration cost analysis utilities
- Provider-agnostic cost comparison tools

**Intelligence Features:**
- Task complexity-based model/provider selection
- Cost-aware completion strategies with budget enforcement  
- Cross-provider performance vs cost optimization
- Automatic cost optimization recommendations

### **8. Developer Onboarding Optimization (Measured and Validated)**

**Onboarding Success Metrics:**
- Time-to-first-value ≤ 5 minutes (measured and validated)
- Setup validation catches 95%+ of common configuration issues
- Progressive complexity path completion rates >80%
- Developer satisfaction scores >4.5/5.0 
- Documentation self-service success >90%

**User Experience Validation:**
- New developer testing with no prior framework knowledge
- Documentation walkthroughs with time measurement
- Error scenario testing with fix success rates
- Integration testing across different development environments

### **9. Quality Gates and Release Standards**

**Before Any Feature Release:**
✅ Zero-code auto-instrumentation works with no API changes
✅ 5-minute quickstart guide validates with new developers  
✅ Comprehensive integration guide covers all major use cases
✅ All required examples are implemented and tested
✅ Validation utilities provide actionable diagnostics
✅ Test coverage meets minimum standards (75+ tests)
✅ Performance benchmarks are documented
✅ Production deployment patterns are validated

**Continuous Quality Assurance:**
- Regular developer onboarding testing with external developers
- Documentation currency validation (examples work with latest versions)
- Performance regression testing on every release
- Cross-platform compatibility validation
- Community feedback integration and response

### **10. Community and Enterprise Alignment**

**Developer Community Standards:**
- Open source development with transparent roadmaps
- Community contribution guidelines with clear reviewer expectations
- Regular community demos and feedback sessions
- Public RFCs for major feature decisions
- Comprehensive contributor onboarding documentation

**Enterprise Integration Excellence:**
- Production-ready patterns for all major deployment scenarios
- Enterprise security and compliance documentation
- Support for existing observability and governance tools
- Migration guides from competitive solutions
- Professional services and support pathway documentation

---

### **Quality Commitment**

These standards represent our commitment to developer experience excellence. Every development effort MUST demonstrate adherence to these principles before being considered complete.

The goal is not just functional correctness, but developer delight—creating tools that developers actively want to use and recommend to their colleagues.

**Developer Experience Validation Question:**
*"Would a developer with no prior GenOps knowledge be productive and successful within 5 minutes of following our documentation?"*

If the answer is not an emphatic "yes," the implementation is not ready for release.

---

## Key References

* **OpenTelemetry:** [https://opentelemetry.io](https://opentelemetry.io)
* **Next.js → Vercel Playbook**
* **Sentry → Sentry Cloud Playbook**
* **Open-core Model:** "Open Source standard, Best-in-class developer experience"

---

## README Integration List Formatting Standards (MANDATORY)

### Critical Formatting Requirements

**This section addresses a recurring issue that MUST be prevented going forward.**

### Approved Format Patterns (ONLY these are allowed):

**✅ Completed Integrations:**
```
- ✅ [Name](internal-link) (<a href="external-link" target="_blank">↗</a>)
```

**☐ Planned Integrations:**
```
- ☐ Name (<a href="external-link" target="_blank">↗</a>)
```

### FORBIDDEN Patterns (These cause violations):

**❌ NEVER add descriptive text:**
```
- ✅ [Name](link) (<a href="external" target="_blank">↗</a>) - Any descriptive text here
```

**❌ NEVER add explanations:**
```
- ✅ [Name](link) (<a href="external" target="_blank">↗</a>) - Multi-agent conversation governance
```

### Enforcement Mechanisms

This formatting standard is enforced by:

1. **Pre-commit Hook**: `scripts/validate-readme-format.py` runs on every commit
2. **CI/CD Validation**: GitHub Actions workflow validates all README changes
3. **Automated Detection**: Pattern matching detects violations automatically
4. **Developer Guidelines**: This CLAUDE.md section provides clear rules

### Historical Context

The user has explicitly stated: *"remember that we are never supposed to add anything other than name and links for these, this should be in your CLAUDE.md memory"*

This recurring issue has been identified as a pattern that must be prevented through comprehensive automated validation and clear documentation.

### Implementation Details

- **Validation Script**: `scripts/validate-readme-format.py`
- **Pre-commit Integration**: `.pre-commit-config.yaml` includes README validation
- **GitHub Actions**: `.github/workflows/validate-readme-format.yml`
- **Error Messages**: Provide specific fix suggestions for each violation type

**When working with the README integration list, NEVER add descriptive text. Only include the integration name and required links.**

---

## Whitepaper Research and Strategic Insights

### Core Authority Control Plane Concepts

**The Missing Infrastructure Layer**
Modern AI stacks have sophisticated layers for execution, orchestration, and observability, but **lack a native authority control plane**. GenOps AI fills this gap by providing:
- **Runtime Authority Constraints**: Prevent unauthorized operations before they execute
- **Infrastructure-Native Governance**: Authority control built into the substrate, not bolted onto applications
- **OpenTelemetry Extension**: Standard OTLP telemetry enhanced with governance semantics

**Key Insight**: Observability shows "what happened" - GenOps AI controls "what is allowed to happen"

### Technical Architecture Standards

**CaMeL Framework Integration**
- **C**ontext: Operational environment and governance rules  
- **A**gent: Identity and capability scope with authority delegation chains
- **M**odel: Resource boundaries and permission constraints
- **E**valuation: Real-time policy assessment and compliance checking
- **L**imits: Enforceable runtime constraints and intervention points

**Governance Semantic Conventions (OpenTelemetry Extension)**
- `genops.cost.*`: Cost attribution, budget tracking, and optimization
- `genops.policy.*`: Policy evaluation, enforcement actions, and compliance status
- `genops.budget.*`: Budget management, allocation, and constraint enforcement
- `genops.eval.*`: Quality assessment, compliance validation, and governance metrics

**Multi-Provider Authority Management**
- **35+ Provider Integrations**: Unified governance across OpenAI, Anthropic, Bedrock, Gemini, etc.
- **Cross-Stack Tracking**: Authority constraints that span LLM providers, AI frameworks, and infrastructure
- **Zero-Code Auto-Instrumentation**: Immediate governance without application changes

### Market Positioning Strategy

**vs Observability Platforms (Complementary)**
- **Control vs Observe**: Prevention vs detection, runtime enforcement vs post-hoc analysis
- **Infrastructure Layer**: Authority control plane vs monitoring enhancement
- **Enhancement Strategy**: Extends existing Datadog/Honeycomb/Grafana investments

**vs Governance Platforms (Differentiated)**  
- **Runtime vs Policy**: Technical enforcement vs document-based compliance
- **Infrastructure-Native**: Built into AI stack vs procedural overlay
- **Developer-First**: Engineering adoption vs governance team ownership

**vs Traditional Approaches (Complementary)**
- **Authority vs Content**: Resource constraints vs output filtering (guardrails)
- **Continuous vs Point-in-Time**: Runtime enforcement vs periodic assessment (red teaming)
- **Substrate-Level**: Infrastructure control vs application-level policies

### The FinOps Parallel (Inevitability Argument)

**Historical Pattern Recognition**
- **Cloud Evolution**: Infrastructure → Cost Control (FinOps) → Business Enablement
- **AI Evolution**: Infrastructure → Authority Control (GenOps) → Governance Enablement
- **Adoption Timeline**: Regulatory pressure + cost pain → mainstream adoption (75% Forbes Global 2000)

**Market Inevitability Indicators**
- **EU AI Act Compliance**: Regulatory requirements driving governance adoption
- **Enterprise AI Budget Growth**: 300% annual growth creates cost control urgency
- **Shadow AI Costs**: $670K average additional costs when controls are circumvented
- **Governance Market Growth**: $4.83B by 2034 (35.7% CAGR)

### Organizational Implementation Insights

**Target Audience Hierarchy**
1. **Platform/Infrastructure Engineers**: Authority control architecture, OpenTelemetry integration
2. **AI/ML Engineering Teams**: Governance constraints that enable rather than hinder development
3. **FinOps Practitioners**: Natural extension of existing cost management discipline

**Adoption Pattern Progression**
- **Individual**: Budget and policy constraints for personal AI operations
- **Team**: Shared authority boundaries with unified constraint enforcement  
- **Organization**: Enterprise authority infrastructure with compliance integration

**Authority Evolution Framework**
- **Current State**: Informal delegation, document-based policies, after-the-fact cost attribution
- **GenOps AI State**: Infrastructure-enforced constraints, runtime policy evaluation, proactive intervention
- **Future State**: Governance that enables confident autonomous AI system deployment

### Key Competitive Differentiators

**Technical Differentiation**
- **Standards-Based Foundation**: OpenTelemetry ensures interoperability and prevents vendor lock-in
- **Infrastructure-First Architecture**: Control plane vs application-layer solutions
- **Developer Experience Excellence**: 5-minute time-to-value with zero-code setup

**Strategic Positioning**
- **Governance Enablement**: Infrastructure that enables rather than constrains AI adoption
- **Complementary Enhancement**: Works alongside existing tools without replacement
- **Open Source Strategy**: Community-driven development with transparent governance

**Business Model Insights**
- **ROI Projections**: 400-600% Year 1 ROI across organization sizes
- **Payback Periods**: 2.4-3.2 months typical investment recovery
- **Risk Mitigation**: Prevents $670K average shadow AI costs, reduces compliance burden

### Implementation Roadmap Implications

**Near-Term Adoption Drivers (2025-2026)**
- **Regulatory Compliance**: EU AI Act requirements driving governance adoption
- **Cost Control Urgency**: Enterprise AI budget pressures requiring constraint enforcement
- **Early Adopter Advantage**: First-mover benefits in AI governance infrastructure

**Market Maturation Trajectory (2026-2028)**
- **Standard Infrastructure**: Authority control becomes requirement like FinOps
- **Vendor Ecosystem**: Native platform integration by major AI providers
- **Community Development**: Open-source practitioner community growth

**Long-Term Vision (2028+)**
- **Governance as Enablement**: Infrastructure that increases rather than decreases AI adoption confidence
- **Competitive Differentiation**: Organizational capability advantage through governance maturity
- **Innovation Acceleration**: Confident delegation enables faster AI system deployment

---

### TL;DR (for Claude Code)

> GenOps AI extends OpenTelemetry with governance semantics for AI systems —
> **cross-stack tracking without vendor lock-in.**
> It defines open-source telemetry standards for AI cost, policy, and compliance.
> Built and maintained as an open-source project under the Apache 2.0 license.
> 
> **CRITICAL**: README integration entries must NEVER include descriptive text - only name and links are allowed.