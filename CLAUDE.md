# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Strategic Context

### Mission: "Governance for AI, Built on OpenTelemetry"

GenOps AI builds **alongside OpenLLMetry**, on the **OpenTelemetry foundation** — interoperable by design, independent by governance.
It standardizes **cost, policy, compliance, and evaluation telemetry** for AI workloads across internal teams, departments, and per-customer usage.

Where **OpenLLMetry** defines *what* to trace (LLM prompts, completions, tokens),
**GenOps** defines *why and how* — the governance layer that turns telemetry into actionable accountability.

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
* **Vendor Neutrality:** Aligns with OpenTelemetry and OpenLLMetry specs; avoids lock-in with any observability or model vendor

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
   ├── OpenLLMetry (LLM observability: prompts, completions, latency)
   └── GenOps-OTel (AI governance: cost, policy, compliance, evaluation)
```

GenOps operates *next to* OpenLLMetry, not on top of it — interoperable by design, but independent by governance.
Both layers export standard OTLP signals, enabling any organization to maintain full autonomy and transparency.

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

### Interoperability Focus

GenOps doesn't replace observability — it **enriches it with governance semantics**.

Compatible with:

* **OpenTelemetry** (base standard)
* **OpenLLMetry** (LLM observability layer)
* **All major observability platforms:** Datadog, Honeycomb, New Relic, Grafana Tempo, Dynatrace, Splunk, Instana, Highlight, Traceloop
* **All major model providers:** OpenAI, Anthropic, Bedrock, Gemini, Mistral, Together, Ollama, WatsonX

### Community Growth

* Encourage third-party adapters via `genops-adapters` template repo
* Recognize contributors who add new frameworks and exporters
* Co-marketing for high-quality integrations and case studies
* Public governance RFCs (like OTel SIGs)

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

### Testing Strategy (Required Standards)

**Layered Test Architecture:**
- **Unit Tests**: Individual component testing (~35 tests per adapter)  
- **Integration Tests**: End-to-end workflow testing (~17 tests per adapter)
- **Cost Aggregation Tests**: Multi-provider scenarios (~24 tests per adapter)

**Critical Testing Patterns:**
- Context manager lifecycle testing (`__enter__`/`__exit__`)
- Exception handling within instrumentation
- Cost calculation accuracy across providers
- Framework detection and graceful degradation

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

### Framework Adapter Roadmap

**Training Frameworks (High Priority):**
- **PyTorch**: Training cost tracking, GPU utilization, distributed training overhead
- **TensorFlow**: Model checkpointing costs, gradient computation, data loading performance
- Cost models: Compute hours, GPU memory usage, storage for model artifacts

**Framework Implementation Priority:**
1. PyTorch (Training/Inference)
2. TensorFlow (Training/Inference) 
3. LlamaIndex (RAG/Retrieval)
4. Haystack (NLP Pipelines)
5. Transformers (Model Loading/Inference)
6. AutoGen (Multi-agent workflows)

---

## Developer Experience Standards

### Universal Framework Adapter Principles

All framework adapters MUST follow these developer experience standards to ensure consistent, frictionless adoption:

### Documentation Architecture (Required for Every Adapter)

**Dual Documentation Strategy:**
1. **`{framework}-quickstart.md`** - 5-minute value demonstration
   - Zero-code auto-instrumentation setup
   - Single working example with immediate value
   - Basic cost tracking demonstration
   - Setup validation and troubleshooting

2. **`integrations/{framework}.md`** - Comprehensive integration guide
   - Complete feature documentation with code examples
   - All integration patterns (auto, manual, context managers)
   - Advanced use cases and production patterns
   - Performance considerations and best practices
   - Complete API reference with governance attributes

### Developer Onboarding Workflow (The "Golden Path")

**Phase 1: Immediate Value (≤ 5 minutes)**
```python
# Every framework must support this pattern
from genops import auto_instrument
auto_instrument()  # Zero-code setup

# Existing framework code works unchanged
result = framework_operation()  # Automatically tracked!
```

**Phase 2: Progressive Control (≤ 30 minutes)**
```python
# Manual instrumentation with consistent API
from genops.providers.{framework} import instrument_{framework}

adapter = instrument_{framework}()
result = adapter.instrument_{operation_type}(
    framework_object,
    # Governance attributes (consistent across ALL frameworks)
    team="team-name", 
    project="project-name",
    customer_id="customer-id"
)
```

**Phase 3: Advanced Features (≤ 2 hours)**
- Multi-provider cost aggregation
- Context manager patterns
- Custom policy enforcement
- Performance optimization

### API Design Standards (Enforced Consistency)

**Universal Method Naming:**
- `instrument_{framework}()` - Main adapter factory
- `instrument_{operation_type}()` - Operation-specific instrumentation
- `validate_setup()` - Built-in setup verification
- `create_{framework}_cost_context()` - Cost aggregation context manager

**Consistent Governance Attributes:**
```python
# These attributes MUST be supported by every adapter
governance_attrs = {
    "team": str,           # Cost attribution and access control
    "project": str,        # Project-level cost tracking  
    "customer_id": str,    # Customer attribution for billing
    "environment": str,    # Environment segregation
    "cost_center": str,    # Financial reporting
    "feature": str         # Feature-level cost attribution
}
```

**Error Handling Standards:**
- Graceful degradation when framework not available
- Specific error messages with actionable fix suggestions
- Built-in validation utilities with diagnostic information
- Fallback behavior for missing provider dependencies

### Required Examples Structure

**Every framework adapter MUST include these examples:**
```
examples/{framework}/
├── README.md                     # Framework-specific overview
├── setup_validation.py          # Verify setup works
├── basic_tracking.py            # Simple instrumentation  
├── auto_instrumentation.py      # Zero-code setup demo
├── cost_tracking.py            # Multi-provider cost aggregation
├── {framework}_specific_advanced.py  # Framework specialized features
└── production_patterns.py      # Performance & observability patterns
```

### Validation Utilities (Required for Every Adapter)

**Standard Validation Interface:**
```python
from genops.providers.{framework} import validate_setup, print_validation_result

def validate_setup() -> ValidationResult:
    """Comprehensive setup validation returning structured results."""
    pass

def print_validation_result(result: ValidationResult) -> None:
    """User-friendly validation result display with fix suggestions.""" 
    pass
```

**Validation Coverage Requirements:**
- Environment variables and API keys
- Framework and GenOps dependencies  
- OpenTelemetry configuration
- Provider-specific setup
- Live integration testing (when possible)
- Specific fix suggestions for each issue type

### Cost Tracking Architecture (Universal Patterns)

**Multi-Provider Support:**
```python
# Every adapter supports multiple backend providers
with create_{framework}_cost_context("operation_id") as context:
    # Multiple providers automatically aggregated
    result1 = provider1_operation()
    result2 = provider2_operation() 
    
    summary = context.get_final_summary()
    # Unified cost across all providers
```

**Framework-Agnostic Cost Structure:**
```python
@dataclass
class FrameworkCostSummary:
    total_cost: float
    currency: str = "USD"
    cost_by_provider: Dict[str, float]
    cost_by_model: Dict[str, float] 
    unique_providers: Set[str]
    total_time: float
    governance_attributes: Dict[str, str]
```

### Testing Standards

**Required Test Coverage:**
- Unit tests for adapter functionality (~35 tests)
- Integration tests for end-to-end workflows (~17 tests)  
- Cost aggregation tests for multi-provider scenarios (~24 tests)
- Framework detection and graceful degradation tests
- Validation utility comprehensive coverage

### Performance & Production Considerations

**Built-in Performance Features:**
- Sampling configuration for high-volume applications
- Async telemetry export to minimize overhead
- Configurable log levels to control verbosity  
- Framework-specific performance optimizations

**Production Integration Patterns:**
- Container/Docker configuration examples
- Kubernetes deployment configurations
- CI/CD integration examples  
- Observability platform integrations

### Quality Gates for Framework Adapters

**Before any framework adapter is considered complete:**
✅ Dual documentation (quickstart + comprehensive) exists
✅ Auto-instrumentation works with zero code changes  
✅ All required examples are implemented and tested
✅ Validation utilities provide comprehensive diagnostics
✅ Multi-provider cost aggregation is implemented
✅ Test coverage meets standards (75+ tests total)
✅ Performance considerations are documented
✅ Production integration patterns are provided

**Developer Experience Validation:**
- New developer can get value in ≤ 5 minutes
- Setup validation catches common issues with fixes
- Progressive complexity path is clear and documented
- All examples are executable and work immediately
- Documentation answers common questions proactively

This framework ensures every adapter provides consistent, high-quality developer experience, accelerating adoption and reducing support overhead.

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
* **OpenLLMetry:** [https://github.com/traceloop/openllmetry](https://github.com/traceloop/openllmetry)
* **Next.js → Vercel Playbook**
* **Sentry → Sentry Cloud Playbook**
* **Open-core Model:** "Open Source standard, Closed Source experience"

---

### TL;DR (for Claude Code)

> GenOps AI builds *alongside* OpenLLMetry on the OpenTelemetry foundation —
> **interoperable by design, independent by governance.**
> It defines open-source semantics for AI cost, policy, and compliance telemetry.
> Built and maintained as an open-source project under the Apache 2.0 license.