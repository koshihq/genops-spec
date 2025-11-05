# AWS Bedrock GenOps Examples

Get GenOps AI governance working with AWS Bedrock through practical examples.

## 🚀 New to GenOps? Start Here!

**First Time Setup (5 minutes):**
1. Run `python hello_genops_minimal.py` - simplest possible test
2. Try `python hello_genops.py` - detailed example with guidance
3. Explore `python basic_tracking.py` - team cost attribution

**Having Issues?** → [Troubleshooting](#troubleshooting) | **Ready for More?** → [Advanced Examples](#advanced-examples)

## Basic Examples

#### [`hello_genops_minimal.py`](hello_genops_minimal.py) ⭐ **START HERE**
**Ultra-simple test (30 seconds)**
- Absolute simplest way to verify GenOps works
- No setup complexity, just run it
- Perfect confidence builder for first-time users

#### [`hello_genops.py`](hello_genops.py) 
**Detailed example with guidance**
- More detailed example with explanations
- Better error messages and troubleshooting
- Shows what happens when GenOps is working

#### [`basic_tracking.py`](basic_tracking.py)
**Team cost attribution**
- Track costs by team, project, and customer
- Multiple models and cost comparison
- Essential patterns for real usage

#### [`auto_instrumentation.py`](auto_instrumentation.py)
**Zero-code instrumentation** 
- Works with existing boto3 code unchanged
- Multiple AI model demonstrations
- Shows streaming and batch operations

## Advanced Examples

*Ready for more? These examples show powerful GenOps features:*

### Cost Intelligence

#### [`cost_optimization.py`](cost_optimization.py)
**Advanced cost intelligence and optimization**
- Multi-model cost comparison and intelligent selection
- Budget-aware operation strategies with real-time alerts
- Regional cost optimization across AWS regions
- On-demand vs provisioned throughput analysis
- Real-time cost monitoring with optimization recommendations

#### [`production_patterns.py`](production_patterns.py)
**Production-ready deployment patterns**
- Enterprise workflow orchestration with SOC2 compliance
- High-volume processing strategies with cost optimization
- Circuit breaker patterns and error resilience
- Comprehensive monitoring and alerting integration
- Performance optimization for large-scale deployments

### Enterprise Integration Examples

#### [`lambda_integration.py`](lambda_integration.py)
**AWS Lambda serverless patterns**
- Serverless function deployment with GenOps governance
- Cold start optimization and cost management
- Event-driven AI processing with automatic scaling
- Lambda-specific performance tuning
- Cost allocation for serverless architectures

#### [`ecs_integration.py`](ecs_integration.py)
**Container deployment patterns**
- Docker configuration for Bedrock applications
- ECS task definitions with GenOps integration
- Container-optimized telemetry and logging
- Auto-scaling policies based on AI workload metrics
- Health check patterns for containerized AI services

#### [`sagemaker_integration.py`](sagemaker_integration.py)
**ML pipeline integration patterns**
- SageMaker pipeline integration with Bedrock
- Model training and inference cost attribution
- MLOps workflow with comprehensive governance
- Data science experiment tracking
- Model versioning and deployment patterns

## Key Features Demonstrated

### 🏗️ Comprehensive AWS Bedrock Support
- **Multi-model coverage**: Claude, Titan, Jurassic, Command, Llama, Cohere, and Mistral models
- **Provider detection**: Automatic detection of underlying providers with cost optimization
- **Regional support**: Multi-region deployment with intelligent cost optimization
- **Zero-code instrumentation**: Works with existing boto3 applications unchanged

### 💰 Advanced Cost Intelligence
- **Multi-model cost tracking**: Unified costs across all Bedrock foundation models
- **Real-time cost calculation**: Accurate cost attribution with token-level precision
- **Regional optimization**: Cross-region cost comparison with automatic recommendations
- **Budget management**: Operation strategies that respect cost constraints and alerts

### 🏛️ Enterprise Governance
- **Team attribution**: Comprehensive cost attribution by team, project, customer
- **Compliance integration**: SOC2, HIPAA, PCI compliance frameworks with audit trails
- **AWS CloudTrail integration**: Complete operation tracking for enterprise compliance
- **Cost center tracking**: Integration with AWS Cost Explorer and billing

### 📊 OpenTelemetry Integration
- **OTel-native**: Full OpenTelemetry standard compliance with Bedrock-specific metrics
- **Rich telemetry**: Comprehensive operation and cost telemetry with AWS context
- **Observability platform integration**: Works with Datadog, Honeycomb, Grafana, etc.
- **Custom exporters**: Support for any OTLP-compatible backend with AWS tagging

### ⚡ Production-Ready Performance
- **Intelligent model selection**: Automatic optimization based on task complexity and budget
- **Circuit breaker patterns**: Automatic failure protection for AWS API dependencies
- **Multi-region failover**: Cost-optimized failover strategies across AWS regions
- **High-volume optimization**: Batch processing patterns for enterprise-scale workloads

## Usage Patterns

### Function Decorator Pattern
```python
from genops import track_usage

@track_usage(
    operation_name="document_analysis",
    team="ai-platform-team", 
    project="document-intelligence",
    customer_id="enterprise-client-456"
)
def analyze_document(document_content: str) -> dict:
    from genops.providers.bedrock import GenOpsBedrockAdapter
    
    adapter = GenOpsBedrockAdapter(region_name="us-east-1")
    
    # Multi-step analysis with automatic cost tracking
    classification = adapter.text_generation(
        f"Classify document type: {document_content[:500]}",
        model_id="anthropic.claude-3-haiku-20240307-v1:0"
    )
    
    extraction = adapter.text_generation(
        f"Extract key information: {document_content[:1000]}", 
        model_id="anthropic.claude-3-sonnet-20240229-v1:0"  # More powerful for extraction
    )
    
    return {"classification": classification.content, "extraction": extraction.content}
    # All costs automatically tracked and attributed across models
```

### Context Manager Pattern  
```python
from genops.providers.bedrock import GenOpsBedrockAdapter

adapter = GenOpsBedrockAdapter(region_name="us-east-1")

# Multi-model operations with unified tracking
claude_result = adapter.text_generation(
    "Analyze market trends in renewable energy", 
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    team="research-team",
    customer_id="energy-client-789"
)

# Amazon Titan for follow-up processing
titan_result = adapter.text_generation(
    "Summarize the key points from the analysis",
    model_id="amazon.titan-text-express-v1", 
    team="research-team",
    customer_id="energy-client-789"
)

# Automatic cost aggregation across different model providers
```

### Advanced Cost Context Manager (NEW!)
```python
from genops.providers.bedrock_cost_aggregator import create_bedrock_cost_context

# Advanced cost tracking with automatic aggregation and optimization
with create_bedrock_cost_context("multi_model_analysis_workflow") as context:
    adapter = GenOpsBedrockAdapter()
    
    # Multiple models - costs automatically unified with optimization recommendations
    claude_analysis = adapter.text_generation(
        "Perform detailed technical analysis of the proposed architecture",
        model_id="anthropic.claude-3-opus-20240229-v1:0",  # Premium model for complex analysis
        team="architecture-team"
    )
    
    titan_summary = adapter.text_generation(
        "Create executive summary of the technical analysis", 
        model_id="amazon.titan-text-express-v1",  # Cost-effective for summarization
        team="architecture-team"
    )
    
    jurassic_validation = adapter.text_generation(
        "Validate the technical recommendations",
        model_id="ai21.j2-mid-v1",  # Alternative provider for validation
        team="architecture-team"
    )
    
    # Get comprehensive cost summary with optimization insights
    summary = context.get_current_summary()
    print(f"💰 Total workflow cost: ${summary.total_cost:.6f}")
    print(f"🏗️  Models used: {list(summary.unique_models)}")
    print(f"🔧 Providers: {list(summary.unique_providers)}")
    
    # Get intelligent cost optimization recommendations
    for recommendation in summary.optimization_recommendations:
        print(f"💡 Optimization: {recommendation}")
```

### Production Workflow Context (NEW!)
```python
from genops.providers.bedrock_workflow import production_workflow_context, ComplianceLevel

# Enterprise-grade workflow with comprehensive governance and compliance
with production_workflow_context(
    workflow_name="customer_document_processing_pipeline",
    customer_id="enterprise-fortune500",
    team="document-ai-platform",
    project="intelligent-document-processing",
    environment="production",
    compliance_level=ComplianceLevel.SOC2,
    cost_center="AI-Platform-Engineering",
    budget_limit=10.00  # $10 budget with automatic alerts
) as (workflow, workflow_id):
    
    adapter = GenOpsBedrockAdapter(region_name="us-east-1")
    
    # Step 1: Document classification with audit trail
    workflow.record_step("document_classification")
    classification = adapter.text_generation(
        f"Classify document type and sensitivity: {document_text[:500]}",
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        temperature=0.1  # High consistency for classification
    )
    
    # Step 2: Content extraction with performance monitoring
    workflow.record_step("content_extraction") 
    extraction = adapter.text_generation(
        f"Extract structured data: {document_text}",
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        max_tokens=500
    )
    
    # Step 3: SOC2 compliance validation
    workflow.record_step("compliance_validation")
    compliance_check = adapter.text_generation(
        f"Validate SOC2 compliance for extracted data: {extraction.content}",
        model_id="anthropic.claude-3-haiku-20240307-v1:0"
    )
    
    # Record compliance checkpoint for audit trail
    workflow.record_checkpoint("soc2_compliance_verified", {
        "compliance_validated": True,
        "sensitive_data_handled": True,
        "audit_trail_complete": True
    })
    
    # Record performance metrics for monitoring
    workflow.record_performance_metric("documents_processed", 1, "count")
    workflow.record_performance_metric("classification_accuracy", 0.95, "percentage")
    
    # Automatic cost attribution, governance tracking, and compliance reporting
    final_cost = workflow.get_current_cost_summary()
    workflow.record_performance_metric("total_workflow_cost", final_cost.total_cost, "USD")
    
    # Workflow automatically exports comprehensive governance telemetry to CloudTrail
```

## Environment Setup

### Required Dependencies
```bash
# Core installation
pip install genops-ai[bedrock]

# Or install components separately
pip install genops-ai boto3 botocore
```

### Optional Dependencies for Advanced Features
```bash
# AWS CLI for credential management
pip install awscli

# Enhanced observability integrations  
pip install opentelemetry-exporter-datadog
pip install opentelemetry-exporter-jaeger

# Development and testing tools
pip install pytest boto3-stubs
```

### Environment Variables
```bash
# AWS Configuration (required)
export AWS_REGION="us-east-1"
export AWS_DEFAULT_REGION="us-east-1"
# Note: AWS credentials via aws configure, environment variables, or IAM roles

# OpenTelemetry configuration
export OTEL_SERVICE_NAME="bedrock-ai-application"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"

# GenOps configuration
export GENOPS_ENVIRONMENT="production"
export GENOPS_PROJECT="bedrock-ai-project"

# Advanced Bedrock configuration
export GENOPS_DEFAULT_REGION="us-east-1"
export GENOPS_DEFAULT_BEDROCK_MODEL="anthropic.claude-3-haiku-20240307-v1:0"

# Performance and production configuration
export GENOPS_SAMPLING_RATE="1.0"              # Full sampling (0.0-1.0)
export GENOPS_ASYNC_EXPORT="true"              # Async telemetry export
export GENOPS_BATCH_SIZE="100"                 # Telemetry batch size
export GENOPS_EXPORT_TIMEOUT="5"               # Export timeout (seconds)

# Circuit breaker configuration for production resilience
export GENOPS_CIRCUIT_BREAKER="true"           # Enable circuit breaker
export GENOPS_CB_THRESHOLD="5"                 # Failure threshold
export GENOPS_CB_WINDOW="60"                   # Reset window (seconds)

# AWS-specific configuration
export GENOPS_ENABLE_CLOUDTRAIL="true"         # CloudTrail integration
export GENOPS_COST_ALLOCATION_TAGS="true"      # AWS cost allocation tags
```

## Running Examples

### Validate Your Setup
```bash
# Check everything is working
python examples/bedrock/bedrock_validation.py

# Quick validation check
python -c "from genops.providers.bedrock import quick_validate; quick_validate()"
```

### Try Basic Examples
```bash
# Start with Hello World
python examples/bedrock/hello_genops.py

# Zero-code instrumentation
python examples/bedrock/auto_instrumentation.py

# Manual adapter usage
python examples/bedrock/basic_tracking.py

# Advanced cost optimization
python examples/bedrock/cost_optimization.py
```

### Advanced Usage
```bash
# Production deployment patterns  
python examples/bedrock/production_patterns.py

# Enterprise integration examples
python examples/bedrock/lambda_integration.py
python examples/bedrock/ecs_integration.py
python examples/bedrock/sagemaker_integration.py

# Cost context manager testing
python -c "
from genops.providers.bedrock_cost_aggregator import create_bedrock_cost_context
print('Testing cost context manager...')
with create_bedrock_cost_context('test') as ctx:
    print('✅ Cost context manager working!')
"
```

## 🏭 Real-World Industry Examples

### Healthcare AI Compliance (HIPAA)
```python
from genops.providers.bedrock_workflow import production_workflow_context, ComplianceLevel

# HIPAA-compliant medical text analysis with comprehensive audit trails
with production_workflow_context(
    workflow_name="medical_document_analysis_pipeline",
    customer_id="healthcare_system_001",
    team="healthcare_ai_platform",
    project="patient_document_processing",
    compliance_level=ComplianceLevel.HIPAA,
    cost_center="Healthcare-AI-Operations"
) as (workflow, workflow_id):
    
    adapter = GenOpsBedrockAdapter(region_name="us-east-1")
    
    # Medical entity extraction with high precision
    medical_entities = adapter.text_generation(
        "Extract medical entities and conditions from patient record...",
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",  # High accuracy for medical data
        temperature=0.1,  # Minimal randomness for medical consistency
        max_tokens=300
    )
    
    # Record HIPAA compliance checkpoint
    workflow.record_checkpoint("hipaa_compliance_verified", {
        "phi_properly_handled": True,
        "audit_trail_complete": True,
        "encryption_verified": True,
        "access_controls_applied": True
    })
```

### Financial Services Risk Analysis (SOC2/SOX)
```python
# Financial compliance with comprehensive cost controls and audit trails
with production_workflow_context(
    workflow_name="financial_risk_assessment_pipeline",
    customer_id="investment_bank_alpha",
    team="risk_management_ai",
    project="automated_risk_analysis",
    compliance_level=ComplianceLevel.SOX,
    cost_center="Risk-Analytics-Platform",
    budget_limit=25.00  # Strict budget control for financial operations
) as (workflow, workflow_id):
    
    # Multi-model risk assessment with cost optimization
    risk_analysis = adapter.text_generation(
        "Perform comprehensive risk analysis for investment portfolio...",
        model_id="anthropic.claude-3-opus-20240229-v1:0",  # Premium model for financial decisions
        team="risk_management_ai"
    )
    
    # Validation with different model for consensus
    risk_validation = adapter.text_generation(
        "Validate and cross-check the risk assessment...",
        model_id="ai21.j2-ultra-v1",  # Alternative high-quality model
        team="risk_management_ai"
    )
    
    # Monitor costs and alert on budget thresholds
    if workflow.get_current_cost_summary().total_cost > 20.00:
        workflow.record_alert("high_cost_risk_analysis", 
                             "Risk analysis approaching budget limit", "warning")
```

### E-commerce Content Generation (High Volume)
```python
# High-volume content generation with intelligent cost optimization
with create_bedrock_cost_context("ecommerce_content_batch_processing") as context:
    adapter = GenOpsBedrockAdapter()
    
    products = ["smart_watch", "wireless_earbuds", "laptop_stand", "phone_case", "tablet"] * 20  # 100 products
    
    for i, product in enumerate(products, 1):
        # Dynamic model selection based on remaining budget
        current_summary = context.get_current_summary()
        avg_cost = current_summary.get_average_cost_per_operation()
        
        if avg_cost < 0.001:  # Very cost-effective operations
            model = "anthropic.claude-3-sonnet-20240229-v1:0"  # Higher quality
        else:
            model = "amazon.titan-text-express-v1"  # Cost optimization
        
        # Generate product descriptions with cost tracking
        context.add_operation(
            operation_id=f"product_description_{i}",
            model_id=model,
            provider="anthropic" if "claude" in model else "amazon",
            region="us-east-1",
            input_tokens=len(product) * 10,  # Product name + template
            output_tokens=150,  # Average description length
            latency_ms=800,
            governance_attributes={
                "team": "ecommerce_content",
                "customer_id": "marketplace_platform",
                "product_category": product.split("_")[0],
                "batch_id": f"content_batch_{(i-1)//10 + 1}"
            }
        )
    
    # Analyze batch processing efficiency
    final_summary = context.get_current_summary()
    print(f"💰 Total batch cost: ${final_summary.total_cost:.4f}")
    print(f"📊 Average cost per product: ${final_summary.get_average_cost_per_operation():.6f}")
    print(f"🏭 Models used: {list(final_summary.unique_models)}")
    
    # Cost optimization recommendations for future batches
    for rec in final_summary.optimization_recommendations:
        print(f"💡 Optimization: {rec}")
```

## Integration with Observability Platforms

### AWS CloudWatch Integration
```python
# Native AWS CloudWatch integration for Bedrock operations
import boto3

# GenOps automatically exports custom metrics to CloudWatch
cloudwatch = boto3.client('cloudwatch')

# Custom dashboards automatically populated with:
# - bedrock.operation.cost (by model, region, team)
# - bedrock.operation.latency (P50, P95, P99)
# - bedrock.budget.utilization (real-time budget tracking)
# - bedrock.model.performance (success rates, error rates)
```

### Datadog Integration
```python
# Set up Datadog exporter for comprehensive Bedrock telemetry
from opentelemetry.exporter.datadog import DatadogExporter

# GenOps Bedrock telemetry automatically flows to Datadog with:
# - Cost attribution by team, project, customer
# - AWS region and model performance metrics
# - Budget alerts and optimization recommendations
# - Compliance and governance tracking
```

### Custom OTLP Integration
```python
# Works with any OTLP-compatible backend (Grafana, Jaeger, etc.)
export OTEL_EXPORTER_OTLP_ENDPOINT="http://your-collector:4317"

# Bedrock-specific telemetry includes:
# - AWS region and availability zone context
# - Bedrock model provider and version
# - Cost allocation with AWS billing integration
# - Performance metrics with AWS service context
```

## ⚡ Performance Tuning Quick Reference

### High-Volume Applications (10,000+ operations/day)
```bash
export GENOPS_SAMPLING_RATE="0.1"           # Sample 10% for reduced overhead
export GENOPS_ASYNC_EXPORT="true"           # Non-blocking telemetry
export GENOPS_BATCH_SIZE="50"               # Smaller batches for faster processing
export GENOPS_CIRCUIT_BREAKER="true"        # Protect against AWS API failures
export GENOPS_CB_THRESHOLD="3"              # Quick failure detection
```

### Development/Testing (Full telemetry)
```bash
export GENOPS_SAMPLING_RATE="1.0"           # Full sampling
export GENOPS_ASYNC_EXPORT="false"          # Synchronous for debugging  
export GENOPS_CIRCUIT_BREAKER="false"       # No circuit breaker
export GENOPS_ENABLE_CLOUDTRAIL="true"      # Full audit trail
```

### Production (Balanced performance + observability)
```bash
export GENOPS_SAMPLING_RATE="0.5"           # 50% sampling
export GENOPS_ASYNC_EXPORT="true"           # Non-blocking
export GENOPS_BATCH_SIZE="100"              # Standard batches
export GENOPS_CIRCUIT_BREAKER="true"        # Resilience protection
export GENOPS_COST_ALLOCATION_TAGS="true"   # AWS cost integration
```

## Troubleshooting

### Comprehensive Error Resolution Matrix

| Error | Symptom | Quick Fix | Detailed Solution |
|-------|---------|-----------|-------------------|
| **AWS Credentials** | `NoCredentialsError` | `aws configure` | [AWS Credentials Guide](#aws-credentials-setup) |
| **Bedrock Access** | `AccessDeniedException` | Enable model access in AWS console | [Model Access Guide](#bedrock-model-access) |
| **Region Issues** | `EndpointConnectionError` | Try `us-east-1` region | [Region Support](#supported-regions) |
| **Model Not Found** | `ValidationException` | Check model availability | [Model Catalog](#available-models) |
| **Budget Exceeded** | Budget alert triggered | Adjust budget or optimize models | [Cost Optimization](#cost-optimization-strategies) |
| **Circuit Breaker** | "Circuit breaker is open" | Wait or disable circuit breaker | [Resilience Patterns](#error-handling-patterns) |

### Quick Diagnostics
```bash
# Run comprehensive setup validation
python -c "from genops.providers.bedrock import validate_setup, print_validation_result; result = validate_setup(); print_validation_result(result, detailed=True)"

# Quick environment check
python -c "
from genops.providers.bedrock import validate_setup
result = validate_setup()
if result.success:
    print('✅ Bedrock setup ready!')
else:
    print('❌ Issues found:')
    for error in result.errors:
        print(f'   - {error}')
"
```

### Emergency Reset (If Nothing Works)
```bash
# Reset all configuration to defaults
unset AWS_PROFILE
unset GENOPS_SAMPLING_RATE
unset GENOPS_CIRCUIT_BREAKER
export AWS_DEFAULT_REGION="us-east-1"
python hello_genops.py  # Test with simple example
```

### Getting Help
- Run comprehensive setup validation with diagnostic information
- Check the integration guide: `docs/integrations/bedrock.md`
- Review AWS Bedrock documentation for model access and permissions
- Report issues: https://github.com/KoshiHQ/GenOps-AI/issues

## 🎯 What's Next? - Your GenOps Bedrock Journey

### 📚 Learning Path Based on Your Goals

#### "I just want to see if this works" → **Beginner (5 minutes)**
```bash
python hello_genops.py              # Ultra-simple test
python bedrock_validation.py        # Verify everything works
```
**Next:** Try `auto_instrumentation.py` to see zero-code setup in action

#### "I need cost tracking for my team" → **Team Lead (15 minutes)**
```bash
python basic_tracking.py            # Add team attribution
python cost_optimization.py         # Multi-model cost comparison
```
**Next:** Set up `OTEL_EXPORTER_OTLP_ENDPOINT` to export to your AWS dashboards

#### "I want advanced cost management" → **FinOps Pro (30 minutes)**
```python
# Try advanced context managers
from genops.providers.bedrock_cost_aggregator import create_bedrock_cost_context
with create_bedrock_cost_context("my_analysis") as ctx:
    # Your multi-model operations here
    summary = ctx.get_current_summary()
```
**Next:** Explore regional cost optimization and provisioned throughput analysis

#### "I'm deploying to production" → **Production Ready (1 hour)**
```bash
python production_patterns.py       # Enterprise workflow patterns
python lambda_integration.py        # Serverless deployment
python ecs_integration.py          # Container patterns
```
**Next:** Set up CloudWatch dashboards and AWS cost allocation tags

#### "I need enterprise governance" → **Enterprise (2 hours)**
```python
# Try enterprise workflows with compliance
from genops.providers.bedrock_workflow import production_workflow_context, ComplianceLevel
with production_workflow_context(
    workflow_name="compliance_workflow",
    compliance_level=ComplianceLevel.SOC2,
    enable_cloudtrail=True
) as (workflow, workflow_id):
    # Your operations with full governance and audit trails
```
**Next:** Integrate with your AWS compliance and audit systems

### 🚀 Quick Wins by Use Case

| **If you want to...** | **Start here** | **Time** | **Next step** |
|----------------------|----------------|----------|---------------|
| Just verify it works | `hello_genops.py` | 30s | Run validation command |
| Track team costs | `basic_tracking.py` | 2min | Add governance attributes |
| Compare model costs | `cost_optimization.py` | 5min | Try cost context managers |
| Optimize performance | Performance tuning section | 10min | `production_patterns.py` |
| Deploy serverlessly | `lambda_integration.py` | 15min | Set up CloudWatch monitoring |
| Enterprise compliance | Industry examples | 20min | `production_workflow_context` |

### 🎓 Graduation Checklist

**✅ Beginner → Intermediate**
- [ ] Successfully run `hello_genops.py` and validation
- [ ] Add governance attributes (team, project, customer_id)
- [ ] View cost data in AWS CloudWatch or your observability platform

**✅ Intermediate → Advanced**  
- [ ] Use `create_bedrock_cost_context()` for multi-operation tracking
- [ ] Configure performance settings for your use case
- [ ] Set up regional cost optimization strategies

**✅ Advanced → Production**
- [ ] Implement `production_workflow_context()` with compliance
- [ ] Deploy with Lambda or ECS patterns
- [ ] Set up monitoring, alerting, and AWS cost allocation

**✅ Production → Enterprise**
- [ ] Integrate with AWS enterprise systems (CloudTrail, Cost Explorer)
- [ ] Implement industry-specific compliance patterns
- [ ] Scale across multiple teams and AWS accounts

For more comprehensive documentation, see:
- **Quick Start**: `docs/bedrock-quickstart.md`
- **Integration Guide**: `docs/integrations/bedrock.md`  
- **API Reference**: `docs/api/providers/bedrock.md`