# Hugging Face GenOps Examples

This directory demonstrates comprehensive Hugging Face integration with GenOps AI governance and telemetry.

## 🧭 Quick Navigation

**New to GenOps?** → [5-Minute Value Demo](#5-minute-value-demonstration) → [Basic Examples](#try-basic-examples)  
**Ready for Advanced Features?** → [Advanced Context Managers](#cost-context-manager-pattern-new) → [Production Deployment](#enterprise-integration-examples)  
**Need Help?** → [Troubleshooting](#troubleshooting) → [Environment Setup](#environment-setup)  
**Production Ready?** → [Performance Config](#performance-features) → [Enterprise Examples](#enterprise-integration-examples)

## Quick Start Path

### 5-Minute Value Demonstration
**Zero-code setup with immediate results:**
```bash
# 1. Verify setup
python setup_validation.py

# 2. Hello World (30 seconds - simplest possible example)
python hello_genops.py

# 3. Try auto-instrumentation (zero code changes needed!)
python auto_instrumentation.py

# 4. See basic usage patterns
python basic_tracking.py
```

### 30-Minute Guided Exploration
**Manual instrumentation with governance attributes:**
```bash
# Multi-provider cost tracking
python cost_tracking.py

# Hugging Face specific advanced features
python huggingface_specific_advanced.py
```

### 2-Hour Mastery Path
**Advanced features and production deployment:**
```bash
# Production-ready patterns with enterprise workflows
python production_patterns.py
```

## Example Overview

### Core Examples

#### [`setup_validation.py`](setup_validation.py)
**Comprehensive setup verification with diagnostic feedback**
- Environment variable validation
- Dependency checking with fix suggestions
- Hugging Face connectivity testing
- GenOps integration validation
- Cost calculation verification

#### [`hello_genops.py`](hello_genops.py) 🆕
**Ultra-simple Hello World example (30 seconds)**
- Minimal example to verify GenOps is working
- Single AI operation with automatic governance
- Perfect for first-time users
- Immediate confirmation of successful setup
- Clear next steps guidance

#### [`auto_instrumentation.py`](auto_instrumentation.py) 
**Zero-code instrumentation demonstration**
- Automatic telemetry injection
- Works with existing Hugging Face code unchanged
- Multiple AI task support (text generation, chat, embeddings, image generation)
- Governance attribute propagation

#### [`basic_tracking.py`](basic_tracking.py)
**Essential patterns for GenOps Hugging Face integration**
- Manual adapter usage patterns
- Governance attribute examples
- Basic cost tracking demonstration
- Task-specific instrumentation

### Multi-Provider Examples

#### [`cost_tracking.py`](cost_tracking.py)
**Cost tracking across multiple providers accessed through Hugging Face**
- OpenAI models via Hugging Face
- Anthropic models via Hugging Face  
- Native Hugging Face Hub models
- Unified cost aggregation and reporting
- Provider comparison and optimization

### Advanced Features

#### [`huggingface_specific_advanced.py`](huggingface_specific_advanced.py)
**Advanced Hugging Face-specific features and patterns**
- Multi-task AI operation workflows with cost optimization
- Cross-provider model comparison and intelligent routing
- Advanced cost context management and workflow orchestration
- Hub integration patterns and community model usage
- Production workflow templates with enterprise governance
- Task-specific optimization strategies and provider detection

### Production Examples

#### [`production_patterns.py`](production_patterns.py)
**Production-ready deployment patterns**
- High-volume instrumentation strategies
- Async telemetry export patterns
- Error handling and circuit breakers
- Performance optimization techniques
- Monitoring and alerting integration

### Enterprise Integration Examples

#### [`docker_integration.py`](docker_integration.py)
**Container deployment patterns**
- Docker configuration management
- Container-optimized telemetry setup
- Health check patterns for containers
- Multi-stage Docker builds for production
- OTLP endpoint configuration for containerized environments

#### [`kubernetes_integration.py`](kubernetes_integration.py)
**Kubernetes deployment patterns**
- ConfigMap and Secret management
- Kubernetes-native health checks (readiness/liveness probes)
- Service mesh integration patterns
- HPA with custom metrics
- Pod lifecycle and resource monitoring

#### [`cicd_integration.py`](cicd_integration.py)
**CI/CD pipeline integration**
- Automated testing with telemetry validation
- Deployment readiness checks
- Performance regression testing
- Cost impact analysis for CI/CD
- GitHub Actions and GitLab CI examples


## Key Features Demonstrated

### 🤗 Comprehensive Hugging Face Support
- **Multi-task coverage**: Text generation, chat, embeddings, image generation, and specialized NLP tasks
- **Provider detection**: Automatic detection of underlying providers (OpenAI, Anthropic, etc. via HF)
- **Hub model support**: Native Hugging Face Hub models with cost optimization
- **Zero-code instrumentation**: Works with existing Hugging Face applications unchanged

### 💰 Advanced Cost Intelligence
- **Multi-provider cost tracking**: Unified costs across OpenAI, Anthropic, Cohere, Meta, Google via HF
- **Real-time cost calculation**: Accurate cost attribution with token-level precision
- **Cost optimization**: Automatic suggestions for cost-effective model selection
- **Budget awareness**: Operation strategies that respect cost constraints

### 🏛️ Enterprise Governance
- **Team attribution**: Comprehensive cost attribution by team, project, customer
- **Policy enforcement**: Governance rules with automatic enforcement
- **Audit trails**: Complete operation tracking for compliance
- **Multi-tenant support**: Customer-specific cost tracking and attribution

### 📊 OpenTelemetry Integration
- **OTel-native**: Full OpenTelemetry standard compliance
- **Rich telemetry**: Comprehensive operation and cost telemetry
- **Observability platform integration**: Works with Datadog, Honeycomb, Grafana, etc.
- **Custom exporters**: Support for any OTLP-compatible backend

### ⚡ Production-Ready Performance (NEW!)
- **Configurable Sampling**: Control telemetry overhead with sampling rates (0.0-1.0)
- **Async Export**: Non-blocking telemetry export to minimize application impact
- **Circuit Breaker**: Automatic failure protection for external API dependencies
- **Batch Processing**: Optimized telemetry batching for high-volume applications
- **Performance Monitoring**: Built-in metrics for operation latency and resource usage

## Usage Patterns

### Function Decorator Pattern
```python
from genops import track_usage

@track_usage(
    operation_name="document_analysis",
    team="research-team", 
    project="document-ai",
    customer_id="enterprise-123"
)
def analyze_document(text: str) -> dict:
    from genops.providers.huggingface import create_instrumented_client
    
    client = create_instrumented_client()
    
    # Multi-step analysis with automatic cost tracking
    summary = client.text_generation(
        f"Summarize: {text}",
        model="microsoft/DialoGPT-medium"
    )
    
    sentiment = client.text_generation(
        f"Analyze sentiment: {text}",
        model="cardiffnlp/twitter-roberta-base-sentiment"  
    )
    
    return {"summary": summary, "sentiment": sentiment}
    # All costs automatically tracked and attributed
```

### Context Manager Pattern  
```python
from genops.providers.huggingface import GenOpsHuggingFaceAdapter

with GenOpsHuggingFaceAdapter() as hf:
    # Multi-provider operations with unified tracking
    
    # OpenAI via Hugging Face
    openai_result = hf.text_generation(
        "Write a product description", 
        model="gpt-3.5-turbo",
        team="marketing",
        customer_id="client-456"
    )
    
    # Native Hub model
    hub_result = hf.text_generation(
        "Write a product description",
        model="microsoft/DialoGPT-medium", 
        team="marketing",
        customer_id="client-456"  
    )
    
    # Automatic cost aggregation across providers
```

### Cost Context Manager Pattern (NEW!)
```python
from genops.providers.huggingface import create_huggingface_cost_context

# Advanced cost tracking with automatic aggregation
with create_huggingface_cost_context("multi_provider_analysis") as context:
    # All operations within this context are automatically aggregated
    adapter = GenOpsHuggingFaceAdapter()
    
    # Multiple providers - costs automatically unified
    openai_result = adapter.text_generation(
        "Analyze this data", 
        model="gpt-3.5-turbo",
        team="data-team"
    )
    
    anthropic_result = adapter.text_generation(
        "Cross-validate the analysis",
        model="claude-3-haiku", 
        team="data-team"
    )
    
    hub_result = adapter.feature_extraction(
        ["embedding", "vector", "similarity"],
        model="sentence-transformers/all-MiniLM-L6-v2",
        team="data-team"
    )
    
    # Get comprehensive cost summary
    summary = context.get_current_summary()
    print(f"Total cost: ${summary.total_cost:.4f}")
    print(f"Providers used: {list(summary.unique_providers)}")
    print(f"Cost breakdown: {summary.get_provider_breakdown()}")
```

### Production Workflow Context (NEW!)
```python
from genops.providers.huggingface import production_workflow_context

# Enterprise-grade workflow with full governance
with production_workflow_context(
    workflow_name="customer_document_processing",
    customer_id="enterprise-corp",
    team="document-ai",
    project="intelligent-processing",
    environment="production",
    # Enterprise governance attributes
    cost_center="R&D",
    compliance_level="SOC2",
    data_classification="confidential"
) as (workflow, workflow_id):
    
    adapter = GenOpsHuggingFaceAdapter()
    
    # Step 1: Document classification
    workflow.record_step("document_classification")
    classification = adapter.text_generation(
        f"Classify document type: {document_text}",
        model="microsoft/DialoGPT-medium",
        max_new_tokens=50
    )
    
    # Step 2: Content extraction
    workflow.record_step("content_extraction") 
    extraction = adapter.text_generation(
        f"Extract key information: {document_text}",
        model="gpt-3.5-turbo",
        max_new_tokens=200
    )
    
    # Step 3: Compliance check
    workflow.record_step("compliance_validation")
    compliance = adapter.text_generation(
        f"Check compliance requirements: {extraction}",
        model="claude-3-haiku",
        max_new_tokens=100
    )
    
    # Automatic cost attribution, governance tracking, and alerting
    final_cost = workflow.get_current_cost_summary()
    workflow.record_performance_metric("total_workflow_cost", final_cost.total_cost, "USD")
    
    # Workflow automatically exports comprehensive governance telemetry
```

### Auto-Instrumentation Pattern
```python
from genops.providers.huggingface import instrument_huggingface

# Enable zero-code instrumentation
instrument_huggingface()

# Your existing code works unchanged with automatic telemetry
from huggingface_hub import InferenceClient

client = InferenceClient()

# This call is now automatically tracked with GenOps telemetry
response = client.text_generation(
    "Hello, world!",
    model="microsoft/DialoGPT-medium"
)
# Cost, performance, and governance data automatically captured
```

## Environment Setup

### Required Dependencies
```bash
# Core installation
pip install genops-ai[huggingface]

# Or install components separately
pip install genops-ai huggingface_hub
```

### Optional Dependencies for Advanced Features
```bash
# AI/ML enhancements
pip install torch transformers datasets accelerate

# Observability integrations  
pip install opentelemetry-exporter-datadog
pip install opentelemetry-exporter-jaeger
```

### Environment Variables
```bash
# Optional but recommended for enhanced functionality
export HF_TOKEN="your-hugging-face-token"

# OpenTelemetry configuration
export OTEL_SERVICE_NAME="my-ai-application"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"

# GenOps configuration
export GENOPS_ENVIRONMENT="production"
export GENOPS_PROJECT="my-ai-project"

# NEW! Performance and Production Configuration
export GENOPS_SAMPLING_RATE="1.0"              # Full sampling (0.0-1.0)
export GENOPS_ASYNC_EXPORT="true"              # Async telemetry export
export GENOPS_BATCH_SIZE="100"                 # Telemetry batch size
export GENOPS_EXPORT_TIMEOUT="5"               # Export timeout (seconds)

# NEW! Circuit Breaker Configuration
export GENOPS_CIRCUIT_BREAKER="true"           # Enable circuit breaker
export GENOPS_CB_THRESHOLD="5"                 # Failure threshold
export GENOPS_CB_WINDOW="60"                   # Reset window (seconds)
```

## Running Examples

### Validate Your Setup
```bash
# Check everything is working
python examples/huggingface/setup_validation.py

# Quick validation check
python -c "from genops.providers.huggingface import quick_validate; quick_validate()"
```

### Try Basic Examples
```bash
# Start with zero-code instrumentation
python examples/huggingface/auto_instrumentation.py

# Explore manual instrumentation
python examples/huggingface/basic_tracking.py

# Test multi-provider costs
python examples/huggingface/cost_tracking.py

# NEW! Try advanced context managers
python -c "
from genops.providers.huggingface import create_huggingface_cost_context
print('Testing cost context manager...')
with create_huggingface_cost_context('test') as ctx:
    print('✅ Cost context manager working!')
"
```

### Advanced Usage
```bash
# Advanced Hugging Face specific features
python examples/huggingface/huggingface_specific_advanced.py

# Production deployment patterns  
python examples/huggingface/production_patterns.py

# Enterprise deployment examples
python examples/huggingface/docker_integration.py
python examples/huggingface/kubernetes_integration.py
python examples/huggingface/cicd_integration.py

# NEW! Performance configuration testing
python -c "
from genops.providers.huggingface import GenOpsHuggingFaceAdapter
adapter = GenOpsHuggingFaceAdapter()
config = adapter.get_performance_config()
print('Performance Configuration:')
for key, value in config.items():
    print(f'  {key}: {value}')
"
```

## 🏭 Real-World Industry Examples

### Healthcare AI Compliance
```python
from genops.providers.huggingface import production_workflow_context

# HIPAA-compliant medical text analysis
with production_workflow_context(
    workflow_name="medical_document_analysis",
    customer_id="hospital_001",
    team="healthcare_ai",
    project="patient_document_processing",
    # Healthcare-specific governance
    compliance_level="HIPAA",
    data_classification="PHI",
    audit_required=True
) as (workflow, workflow_id):
    
    adapter = GenOpsHuggingFaceAdapter()
    
    # Medical entity extraction
    medical_entities = adapter.text_generation(
        "Extract medical entities: Patient has diabetes...",
        model="microsoft/DialoGPT-medium",
        temperature=0.1  # High precision for medical data
    )
    
    # Automatic compliance tracking and audit trail
    workflow.record_checkpoint("hipaa_compliance_verified", {
        "phi_detected": True,
        "audit_trail_complete": True
    })
```

### Financial Services Risk Analysis  
```python
# Financial compliance with cost controls
with production_workflow_context(
    workflow_name="loan_risk_assessment",
    customer_id="bank_alpha",
    team="risk_management",
    # Financial governance
    compliance_level="SOX",
    cost_center="risk_analytics",
    budget_limit=50.00  # Strict budget control
) as (workflow, workflow_id):
    
    # Multi-model risk assessment
    risk_analysis = adapter.text_generation(
        "Analyze credit risk for loan application...",
        model="claude-3-haiku",  # High-accuracy model for financial decisions
        team="risk_management"
    )
    
    # Cost-aware processing with automatic alerts
    if workflow.get_current_cost_summary().total_cost > 25.00:
        workflow.record_alert("high_cost_risk_analysis", 
                             "Risk analysis exceeding cost threshold", "warning")
```

### E-commerce Content Generation
```python
# High-volume e-commerce content with cost optimization
with create_huggingface_cost_context("product_content_batch") as context:
    adapter = GenOpsHuggingFaceAdapter()
    
    products = ["laptop", "smartphone", "tablet", "headphones"]
    
    for product in products:
        # Cost-optimized content generation
        description = adapter.text_generation(
            f"Write compelling product description for {product}",
            model="microsoft/DialoGPT-medium",  # Cost-efficient for content
            team="ecommerce_content",
            customer_id="marketplace_001"
        )
        
        # Generate SEO keywords
        keywords = adapter.feature_extraction(
            [description],
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    # Monitor costs across batch processing
    final_cost = context.get_current_summary()
    print(f"💰 Batch processing cost: ${final_cost.total_cost:.4f}")
    
    # Cost optimization insights
    if final_cost.total_cost > 0.50:
        print("💡 Consider using Hub models for better cost efficiency")
```

### Customer Support Automation
```python
# Multi-language customer support with performance optimization
import os
os.environ['GENOPS_SAMPLING_RATE'] = '0.3'  # Optimized for high volume

with production_workflow_context(
    workflow_name="customer_support_ai",
    customer_id="support_platform",
    team="customer_success",
    environment="production"
) as (workflow, workflow_id):
    
    adapter = GenOpsHuggingFaceAdapter()
    
    # Customer inquiry processing
    response = adapter.text_generation(
        "Customer asks: How do I return my order?",
        model="microsoft/DialoGPT-medium",
        max_new_tokens=150,
        temperature=0.5
    )
    
    # Sentiment analysis
    sentiment = adapter.text_generation(
        "Analyze customer sentiment: frustrated about delivery delay",
        model="cardiffnlp/twitter-roberta-base-sentiment"
    )
    
    # Performance tracking for support SLAs
    workflow.record_performance_metric("response_time", 2.3, "seconds")
    workflow.record_performance_metric("customer_satisfaction_score", 4.2, "rating")
```

## Integration with Observability Platforms

### Datadog Integration
```python
# Set up Datadog exporter for telemetry
from opentelemetry.exporter.datadog import DatadogExporter

# GenOps telemetry automatically flows to Datadog dashboards
```

### Grafana/Prometheus Integration
```python
# OTLP export to Grafana Tempo
export OTEL_EXPORTER_OTLP_ENDPOINT="http://tempo:4317"

# Cost and performance metrics automatically available
```

### Custom Observability Integration
```python
# Works with any OTLP-compatible backend
export OTEL_EXPORTER_OTLP_ENDPOINT="http://your-collector:4317"

# NEW! Advanced telemetry with context managers
from genops.providers.huggingface import production_workflow_context

with production_workflow_context(
    workflow_name="observability_demo",
    team="platform",
    project="telemetry-integration"
) as (workflow, workflow_id):
    # All operations within automatically export to your backend
    # with comprehensive governance attributes
    pass
```

## ⚡ Performance Tuning Quick Reference

### High-Volume Applications (1000+ requests/hour)
```bash
export GENOPS_SAMPLING_RATE="0.1"           # Sample 10% for reduced overhead
export GENOPS_ASYNC_EXPORT="true"           # Non-blocking telemetry
export GENOPS_BATCH_SIZE="50"               # Smaller batches
export GENOPS_CIRCUIT_BREAKER="true"        # Protect against failures
```

### Development/Testing (Full telemetry)
```bash
export GENOPS_SAMPLING_RATE="1.0"           # Full sampling
export GENOPS_ASYNC_EXPORT="false"          # Synchronous for debugging  
export GENOPS_CIRCUIT_BREAKER="false"       # No circuit breaker
```

### Production (Balanced performance + observability)
```bash
export GENOPS_SAMPLING_RATE="0.5"           # 50% sampling
export GENOPS_ASYNC_EXPORT="true"           # Non-blocking
export GENOPS_BATCH_SIZE="100"              # Standard batches
export GENOPS_CIRCUIT_BREAKER="true"        # Resilience protection
export GENOPS_CB_THRESHOLD="3"              # Quick failure detection
```

### Quick Performance Check
```python
from genops.providers.huggingface import GenOpsHuggingFaceAdapter
adapter = GenOpsHuggingFaceAdapter()
config = adapter.get_performance_config()
print(f"Sampling: {config['sampling_rate']}")
print(f"Circuit Breaker: {'ON' if config['circuit_breaker_enabled'] else 'OFF'}")
print(f"Async Export: {'ON' if config['async_export'] else 'OFF'}")
```

## Troubleshooting

### Comprehensive Error Resolution Matrix

| Error | Symptom | Quick Fix | Detailed Solution |
|-------|---------|-----------|-------------------|
| **Import Error** | `ModuleNotFoundError: No module named 'genops'` | `pip install genops-ai[huggingface]` | [Installation Guide](#required-dependencies) |
| **Hugging Face Not Found** | `ModuleNotFoundError: No module named 'huggingface_hub'` | `pip install huggingface_hub` | Install HF dependencies |
| **Network/API Errors** | Connection timeout, HTTP 503/429 | Check internet connection | Try different model or add HF_TOKEN |
| **Circuit Breaker** | "Circuit breaker is open" | Wait 60s or disable | `export GENOPS_CIRCUIT_BREAKER="false"` |
| **No Telemetry** | Missing cost/performance data | Check OTEL config | Verify OTEL_EXPORTER_OTLP_ENDPOINT |
| **High Memory Usage** | Memory issues with sampling | Reduce sampling rate | `export GENOPS_SAMPLING_RATE="0.1"` |
| **Slow Performance** | Telemetry causing delays | Enable async export | `export GENOPS_ASYNC_EXPORT="true"` |

### Quick Diagnostics
```bash
# Run this to get instant diagnosis
python -c "
from genops.providers.huggingface import GenOpsHuggingFaceAdapter
import os
print('Environment Check:')
print(f'GENOPS_SAMPLING_RATE: {os.getenv(\"GENOPS_SAMPLING_RATE\", \"default:1.0\")}')
print(f'HF_TOKEN: {\"SET\" if os.getenv(\"HF_TOKEN\") else \"NOT SET\"}')
try:
    adapter = GenOpsHuggingFaceAdapter()
    config = adapter.get_performance_config()
    print(f'Circuit Breaker: {\"OPEN\" if config[\"circuit_breaker_open\"] else \"CLOSED\"}')
    print('✅ GenOps adapter working')
except Exception as e:
    print(f'❌ Adapter error: {e}')
"
```

### Emergency Reset (If Nothing Works)
```bash
# Reset all performance settings to defaults
unset GENOPS_SAMPLING_RATE
unset GENOPS_CIRCUIT_BREAKER
unset GENOPS_ASYNC_EXPORT
export GENOPS_SAMPLING_RATE="1.0"
python hello_genops.py  # Test with simple example
```

### Getting Help
- Run setup validation for diagnostic information
- Check the comprehensive integration guide: `docs/integrations/huggingface.md`
- Review troubleshooting section in documentation
- Report issues: https://github.com/KoshiHQ/GenOps-AI/issues

## 🎯 What's Next? - Your GenOps Journey

### 📚 Learning Path Based on Your Goals

#### "I just want to see if this works" → **Beginner (5 minutes)**
```bash
python hello_genops.py              # Ultra-simple test
python setup_validation.py          # Verify everything works
```
**Next:** Try `basic_tracking.py` to see governance attributes in action

#### "I need cost tracking for my team" → **Team Lead (15 minutes)**
```bash
python cost_tracking.py             # Multi-provider cost comparison
python basic_tracking.py            # Add team attribution
```
**Next:** Set up `OTEL_EXPORTER_OTLP_ENDPOINT` to export to your dashboards

#### "I want advanced cost management" → **FinOps Pro (30 minutes)**
```python
# Try advanced context managers
from genops.providers.huggingface import create_huggingface_cost_context
with create_huggingface_cost_context("my_analysis") as ctx:
    # Your multi-model operations here
    summary = ctx.get_current_summary()
```
**Next:** Explore `huggingface_specific_advanced.py` for optimization strategies

#### "I'm deploying to production" → **Production Ready (1 hour)**
```bash
python production_patterns.py        # Performance optimization
python docker_integration.py         # Container deployment
python kubernetes_integration.py     # K8s patterns
```
**Next:** Set up monitoring dashboards and alerting

#### "I need enterprise governance" → **Enterprise (2 hours)**
```python
# Try enterprise workflows
from genops.providers.huggingface import production_workflow_context
with production_workflow_context(
    workflow_name="compliance_workflow",
    compliance_level="SOX",
    audit_required=True
) as (workflow, workflow_id):
    # Your operations with full governance
```
**Next:** Integrate with your compliance and audit systems

### 🚀 Quick Wins by Use Case

| **If you want to...** | **Start here** | **Time** | **Next step** |
|----------------------|----------------|----------|---------------|
| Just verify it works | `hello_genops.py` | 30s | `setup_validation.py` |
| Track team costs | `basic_tracking.py` | 2min | Set governance attributes |
| Compare model costs | `cost_tracking.py` | 5min | Try cost context managers |
| Optimize performance | Performance tuning section | 10min | `production_patterns.py` |
| Deploy in containers | `docker_integration.py` | 15min | `kubernetes_integration.py` |
| Enterprise compliance | Industry examples | 20min | `production_workflow_context` |

### 🎓 Graduation Checklist

**✅ Beginner → Intermediate**
- [ ] Successfully run `hello_genops.py`
- [ ] Add governance attributes (team, project, customer_id)
- [ ] View cost data in your observability platform

**✅ Intermediate → Advanced**  
- [ ] Use `create_huggingface_cost_context()` for multi-operation tracking
- [ ] Configure performance settings for your use case
- [ ] Set up cost optimization strategies

**✅ Advanced → Production**
- [ ] Implement `production_workflow_context()` 
- [ ] Deploy with container/Kubernetes patterns
- [ ] Set up monitoring, alerting, and compliance

**✅ Production → Enterprise**
- [ ] Integrate with enterprise systems (CI/CD, compliance)
- [ ] Implement industry-specific patterns
- [ ] Scale across multiple teams and projects

For more comprehensive documentation, see:
- **Quick Start**: `docs/huggingface-quickstart.md`
- **Integration Guide**: `docs/integrations/huggingface.md`  
- **API Reference**: `docs/api/providers/huggingface.md`