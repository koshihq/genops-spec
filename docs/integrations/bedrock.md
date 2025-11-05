# AWS Bedrock Integration Guide

Comprehensive guide for integrating AWS Bedrock with GenOps AI governance and telemetry.

## Table of Contents

- [Overview](#overview)
- [Installation & Setup](#installation--setup)
- [Integration Patterns](#integration-patterns)
- [Multi-Model Support](#multi-model-support)
- [Cost Intelligence](#cost-intelligence)
- [Enterprise Governance](#enterprise-governance)
- [Production Deployment](#production-deployment)
- [Performance Optimization](#performance-optimization)
- [Observability Integration](#observability-integration)
- [Advanced Use Cases](#advanced-use-cases)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

## Overview

GenOps provides comprehensive AWS Bedrock integration with:

- **Multi-model support**: Claude, Titan, Jurassic, Command, Llama, Cohere, and Mistral
- **Real-time cost tracking**: Token-level precision across all models
- **Enterprise governance**: SOC2, HIPAA, PCI compliance with audit trails
- **Zero-code instrumentation**: Works with existing boto3 applications unchanged
- **OpenTelemetry native**: Exports to any OTLP-compatible observability platform
- **Regional optimization**: Cross-region cost comparison and optimization

### Architecture Overview

```
Application Code
      ↓
GenOps Bedrock Adapter
      ↓
AWS Bedrock Service ← Multi-region support
      ↓
OpenTelemetry Pipeline ← Rich governance telemetry
      ↓
Your Observability Platform ← Datadog, Grafana, etc.
```

## Installation & Setup

### Quick Installation

```bash
# Core installation
pip install genops-ai[bedrock]

# Or install all components
pip install genops-ai[all]
```

### AWS Configuration

GenOps requires standard AWS credentials and Bedrock model access:

```bash
# Configure AWS credentials
aws configure

# Verify access
aws sts get-caller-identity
aws bedrock list-foundation-models --region us-east-1
```

**Required IAM Permissions:**
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream",
                "bedrock:ListFoundationModels"
            ],
            "Resource": "*"
        }
    ]
}
```

### Environment Configuration

```bash
# Required
export AWS_REGION="us-east-1"
export AWS_DEFAULT_REGION="us-east-1"

# OpenTelemetry configuration
export OTEL_SERVICE_NAME="bedrock-ai-application"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"

# GenOps configuration
export GENOPS_ENVIRONMENT="production"
export GENOPS_PROJECT="bedrock-ai-project"

# Performance tuning
export GENOPS_SAMPLING_RATE="1.0"        # Full sampling (0.0-1.0)
export GENOPS_ASYNC_EXPORT="true"        # Non-blocking telemetry
export GENOPS_CIRCUIT_BREAKER="true"     # Resilience protection
```

### Setup Validation

```python
from genops.providers.bedrock import validate_bedrock_setup, print_validation_result

result = validate_bedrock_setup()
print_validation_result(result)

if result.success:
    print("✅ Ready to start using GenOps with Bedrock!")
else:
    print("❌ Please resolve the issues above before continuing")
```

## Integration Patterns

### 1. Zero-Code Auto-Instrumentation

**Automatically instrument existing Bedrock applications with zero code changes:**

```python
from genops.providers.bedrock import auto_instrument_bedrock

# Enable automatic instrumentation
auto_instrument_bedrock()

# Your existing boto3 code now automatically tracked!
import boto3
import json

bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

response = bedrock.invoke_model(
    modelId='anthropic.claude-3-haiku-20240307-v1:0',
    body=json.dumps({
        "prompt": "Analyze this financial report...",
        "max_tokens": 300
    })
)

# Cost and performance automatically tracked and exported
```

### 2. Manual Adapter Integration

**Full control over instrumentation with governance attributes:**

```python
from genops.providers.bedrock import GenOpsBedrockAdapter

adapter = GenOpsBedrockAdapter(
    region_name='us-east-1',
    default_model='anthropic.claude-3-haiku-20240307-v1:0'
)

result = adapter.text_generation(
    prompt="Analyze market trends in renewable energy",
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    max_tokens=500,
    temperature=0.3,
    
    # Governance attributes for cost attribution
    team="research-team",
    project="market-analysis",
    customer_id="energy-client-789",
    environment="production",
    cost_center="Research-Analytics"
)

print(f"💰 Operation cost: ${result.cost_usd:.6f}")
print(f"⚡ Latency: {result.latency_ms}ms")
print(f"🏷️ Attributed to: {result.governance_attributes}")
```

### 3. Context Manager Pattern

**Multi-operation cost tracking with automatic aggregation:**

```python
from genops.providers.bedrock_cost_aggregator import create_bedrock_cost_context

with create_bedrock_cost_context("financial_analysis_workflow") as cost_context:
    adapter = GenOpsBedrockAdapter()
    
    # Step 1: Document classification
    classification = adapter.text_generation(
        prompt="Classify this document type...",
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        team="finance-ai"
    )
    
    # Step 2: Detailed analysis with more powerful model
    analysis = adapter.text_generation(
        prompt="Perform detailed financial analysis...",
        model_id="anthropic.claude-3-opus-20240229-v1:0",  # Premium model
        team="finance-ai"
    )
    
    # Step 3: Executive summary
    summary = adapter.text_generation(
        prompt="Create executive summary...",
        model_id="amazon.titan-text-express-v1",  # Cost-effective
        team="finance-ai"
    )
    
    # Get unified cost summary across all operations
    final_summary = cost_context.get_current_summary()
    print(f"💰 Total workflow cost: ${final_summary.total_cost:.6f}")
    print(f"🔧 Models used: {list(final_summary.unique_models)}")
    print(f"🏭 Providers: {list(final_summary.unique_providers)}")
```

### 4. Function Decorator Pattern

**Automatic instrumentation for specific functions:**

```python
from genops import track_usage

@track_usage(
    operation_name="document_analysis",
    team="ai-platform", 
    project="document-intelligence",
    customer_id="enterprise-client"
)
def analyze_document(document_content: str) -> dict:
    from genops.providers.bedrock import GenOpsBedrockAdapter
    
    adapter = GenOpsBedrockAdapter()
    
    result = adapter.text_generation(
        prompt=f"Analyze this document: {document_content}",
        model_id="anthropic.claude-3-sonnet-20240229-v1:0"
    )
    
    return {"analysis": result.content, "cost": result.cost_usd}

# Function calls automatically tracked with governance
result = analyze_document("QUARTERLY FINANCIAL RESULTS...")
```

## Multi-Model Support

GenOps supports all major Bedrock foundation models with intelligent cost optimization:

### Supported Models

**Anthropic Claude Models:**
```python
models = {
    "anthropic.claude-3-opus-20240229-v1:0": "Premium - highest quality",
    "anthropic.claude-3-sonnet-20240229-v1:0": "Balanced - quality + performance", 
    "anthropic.claude-3-haiku-20240307-v1:0": "Fast - cost-effective",
    "anthropic.claude-instant-v1": "Fastest - real-time responses"
}
```

**Amazon Titan Models:**
```python
models = {
    "amazon.titan-text-express-v1": "Balanced text generation",
    "amazon.titan-text-lite-v1": "Cost-effective option",
    "amazon.titan-embed-text-v1": "Text embeddings"
}
```

**AI21 Labs Jurassic Models:**
```python
models = {
    "ai21.j2-ultra-v1": "Highest quality",
    "ai21.j2-mid-v1": "Balanced performance", 
    "ai21.j2-light-v1": "Fast and cost-effective"
}
```

**Cohere Command Models:**
```python
models = {
    "cohere.command-text-v14": "Latest command model",
    "cohere.command-light-text-v14": "Lighter variant"
}
```

### Intelligent Model Selection

**Cost-aware model selection based on complexity and budget:**

```python
from genops.providers.bedrock import GenOpsBedrockAdapter
from genops.providers.bedrock_pricing import get_cost_optimization_recommendations

adapter = GenOpsBedrockAdapter()

# Analyze task complexity and recommend optimal model
task_prompt = "Analyze this complex financial derivative contract..."

recommendations = get_cost_optimization_recommendations(
    prompt=task_prompt,
    budget_constraint=0.05,  # $0.05 maximum
    quality_requirement="high",  # Options: low, medium, high, premium
    region="us-east-1"
)

print(f"🎯 Recommended model: {recommendations.recommended_model}")
print(f"💰 Estimated cost: ${recommendations.estimated_cost:.6f}")
print(f"⚡ Expected latency: {recommendations.estimated_latency_ms}ms")

# Use the recommendation
result = adapter.text_generation(
    prompt=task_prompt,
    model_id=recommendations.recommended_model,
    team="financial-analysis"
)
```

### Multi-Model Comparison

**Compare performance and costs across different models:**

```python
from genops.providers.bedrock_pricing import compare_bedrock_models

models_to_compare = [
    "anthropic.claude-3-opus-20240229-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0", 
    "anthropic.claude-3-haiku-20240307-v1:0",
    "amazon.titan-text-express-v1"
]

comparison = compare_bedrock_models(
    prompt="Analyze quarterly financial performance",
    models=models_to_compare,
    region="us-east-1",
    expected_output_tokens=300
)

for model_result in comparison.model_comparisons:
    print(f"🤖 {model_result.model_id}")
    print(f"   💰 Cost: ${model_result.estimated_cost:.6f}")
    print(f"   ⚡ Speed: {model_result.estimated_latency_ms}ms")
    print(f"   🎯 Quality Score: {model_result.quality_score}/10")
    print()

print(f"💡 Best for cost: {comparison.best_for_cost}")
print(f"🚀 Best for speed: {comparison.best_for_speed}")
print(f"🏆 Best for quality: {comparison.best_for_quality}")
```

## Cost Intelligence

### Real-Time Cost Tracking

**Accurate cost attribution with token-level precision:**

```python
from genops.providers.bedrock import GenOpsBedrockAdapter

adapter = GenOpsBedrockAdapter()

result = adapter.text_generation(
    prompt="Long complex analysis prompt...",
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    max_tokens=1000,
    team="analytics-team",
    project="cost-optimization-study"
)

# Detailed cost breakdown
print(f"💰 Total cost: ${result.cost_usd:.6f}")
print(f"📥 Input cost: ${result.input_cost:.6f} ({result.input_tokens} tokens)")
print(f"📤 Output cost: ${result.output_cost:.6f} ({result.output_tokens} tokens)")
print(f"🏷️ Cost per 1K tokens: ${result.cost_per_1k_tokens:.6f}")
print(f"🌎 Region: {result.region}")
```

### Budget-Constrained Operations

**Operate within budget constraints with automatic optimization:**

```python
from genops.providers.bedrock_workflow import production_workflow_context, ComplianceLevel

with production_workflow_context(
    workflow_name="budget_conscious_analysis",
    customer_id="startup-client",
    budget_limit=2.00,  # $2.00 maximum budget
    team="cost-optimization",
    compliance_level=ComplianceLevel.SOC2
) as (workflow, workflow_id):
    
    adapter = GenOpsBedrockAdapter()
    
    # Step 1: Quick classification with budget tracking
    workflow.record_step("classification")
    classification = adapter.text_generation(
        prompt="Classify document type quickly...",
        model_id="anthropic.claude-3-haiku-20240307-v1:0",  # Cost-effective
        max_tokens=50
    )
    
    # Check budget before expensive operation
    current_cost = workflow.get_current_cost_summary()
    if current_cost.total_cost < 1.50:  # Leave buffer
        # Step 2: Detailed analysis only if budget allows
        workflow.record_step("detailed_analysis")
        analysis = adapter.text_generation(
            prompt="Perform detailed analysis...",
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            max_tokens=500
        )
    else:
        print("⚠️ Skipping detailed analysis - budget constraint")
    
    final_cost = workflow.get_current_cost_summary()
    print(f"💰 Final cost: ${final_cost.total_cost:.6f}")
    print(f"📊 Budget utilization: {(final_cost.total_cost/2.00)*100:.1f}%")
```

### Regional Cost Optimization

**Compare costs across AWS regions and optimize:**

```python
from genops.providers.bedrock_pricing import calculate_regional_costs

prompt = "Analyze market opportunities in renewable energy sector"
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

regional_costs = calculate_regional_costs(
    prompt=prompt,
    model_id=model_id,
    regions=["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
    expected_output_tokens=400
)

print("🌎 Regional Cost Comparison:")
for region_cost in regional_costs:
    print(f"   {region_cost.region}: ${region_cost.total_cost:.6f}")
    print(f"     Input: ${region_cost.input_cost:.6f}")
    print(f"     Output: ${region_cost.output_cost:.6f}")
    print(f"     Availability: {region_cost.model_available}")
    print()

print(f"💡 Cheapest region: {regional_costs[0].region}")
print(f"💰 Potential savings: ${regional_costs[-1].total_cost - regional_costs[0].total_cost:.6f}")
```

## Enterprise Governance

### SOC2 Compliance Workflows

**Enterprise-grade workflows with comprehensive audit trails:**

```python
from genops.providers.bedrock_workflow import production_workflow_context, ComplianceLevel

with production_workflow_context(
    workflow_name="financial_document_analysis",
    customer_id="financial_services_client",
    team="compliance_ai",
    project="regulatory_reporting",
    environment="production",
    compliance_level=ComplianceLevel.SOC2,
    cost_center="Compliance-Technology",
    enable_cloudtrail=True,
    alert_webhooks=["https://alerts.company.com/compliance"]
) as (workflow, workflow_id):
    
    adapter = GenOpsBedrockAdapter()
    
    # Step 1: Document classification with compliance tracking
    workflow.record_step("document_classification", {
        "classification_types": ["financial", "pii", "confidential"],
        "compliance_framework": "SOC2"
    })
    
    classification = adapter.text_generation(
        prompt="Classify this financial document for SOC2 compliance...",
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        temperature=0.1  # Low temperature for consistency
    )
    
    # Compliance checkpoint
    workflow.record_checkpoint("classification_complete", {
        "pii_detected": False,
        "financial_data_classified": True,
        "compliance_level_maintained": "SOC2"
    })
    
    # Step 2: Content analysis with audit trail
    workflow.record_step("content_analysis", {
        "analysis_type": "financial_risk_assessment",
        "data_handling": "encrypted_in_transit_and_rest"
    })
    
    analysis = adapter.text_generation(
        prompt="Perform SOC2-compliant analysis...",
        model_id="anthropic.claude-3-sonnet-20240229-v1:0"
    )
    
    # Final compliance validation
    workflow.record_checkpoint("analysis_complete", {
        "audit_trail_complete": True,
        "data_retention_compliant": True,
        "access_controls_verified": True,
        "encryption_maintained": True
    })
    
    # Performance and cost metrics
    final_cost = workflow.get_current_cost_summary()
    workflow.record_performance_metric("total_cost", final_cost.total_cost, "USD")
    workflow.record_performance_metric("compliance_score", 1.0, "percentage")
    
    print(f"✅ SOC2 compliant workflow completed")
    print(f"🆔 Workflow ID: {workflow_id}")
    print(f"💰 Total cost: ${final_cost.total_cost:.6f}")
    print(f"📋 Compliance checkpoints: Passed")
```

### Multi-Tenant Customer Attribution

**Comprehensive cost attribution and isolation for multi-tenant applications:**

```python
from genops.providers.bedrock_cost_aggregator import create_bedrock_cost_context

# Process multiple customers with unified cost tracking
customers = [
    {"id": "enterprise_client_1", "tier": "premium"},
    {"id": "startup_client_2", "tier": "standard"}, 
    {"id": "enterprise_client_3", "tier": "premium"}
]

customer_costs = {}

for customer in customers:
    customer_id = customer["id"]
    tier = customer["tier"]
    
    # Customer-specific cost context
    with create_bedrock_cost_context(f"customer_analysis_{customer_id}") as cost_context:
        adapter = GenOpsBedrockAdapter()
        
        # Tier-based model selection
        if tier == "premium":
            model = "anthropic.claude-3-opus-20240229-v1:0"  # Best quality
        else:
            model = "anthropic.claude-3-haiku-20240307-v1:0"  # Cost-effective
        
        # Customer analysis
        result = adapter.text_generation(
            prompt=f"Analyze requirements for {customer_id}...",
            model_id=model,
            customer_id=customer_id,
            team="customer_success",
            service_tier=tier
        )
        
        # Store customer-specific costs
        summary = cost_context.get_current_summary()
        customer_costs[customer_id] = {
            "total_cost": summary.total_cost,
            "model_used": model,
            "tier": tier,
            "operations": summary.total_operations
        }

# Generate customer billing report
print("📊 Customer Cost Attribution Report:")
total_cost = 0
for customer_id, cost_data in customer_costs.items():
    print(f"   👤 {customer_id}")
    print(f"      💰 Cost: ${cost_data['total_cost']:.6f}")
    print(f"      🤖 Model: {cost_data['model_used']}")
    print(f"      🏷️ Tier: {cost_data['tier']}")
    print()
    total_cost += cost_data['total_cost']

print(f"💰 Total revenue: ${total_cost:.6f}")
```

## Production Deployment

### Serverless Deployment (AWS Lambda)

**Optimized Lambda deployment with cold-start optimization:**

```python
import json
import os
from genops.providers.bedrock import GenOpsBedrockAdapter, instrument_bedrock

# Enable auto-instrumentation for optimal Lambda performance
instrument_bedrock()

# Initialize outside handler for connection reuse
adapter = GenOpsBedrockAdapter(
    region_name=os.environ.get('AWS_REGION', 'us-east-1'),
    default_model="anthropic.claude-3-haiku-20240307-v1:0"  # Fast model for Lambda
)

def lambda_handler(event, context):
    """Lambda handler optimized for serverless AI processing."""
    
    try:
        document_text = event.get('document_text', '')
        customer_id = event.get('customer_id', 'unknown')
        
        # Fast document analysis optimized for Lambda
        result = adapter.text_generation(
            prompt=f"Quick analysis: {document_text[:500]}",
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            max_tokens=200,
            temperature=0.2,
            team="serverless-ai",
            customer_id=customer_id,
            environment="lambda"
        )
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'analysis': result.content,
                'cost': result.cost_usd,
                'latency': result.latency_ms,
                'customer_id': customer_id
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

**SAM Template for Lambda deployment:**

```yaml
# template.yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Globals:
  Function:
    Runtime: python3.9
    Timeout: 300
    MemorySize: 1024
    Environment:
      Variables:
        GENOPS_ENVIRONMENT: production
        GENOPS_PROJECT: bedrock-lambda
        OTEL_SERVICE_NAME: bedrock-lambda-ai

Resources:
  BedrockAnalysisFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: src/
      Handler: lambda_handler.lambda_handler
      Policies:
        - AWSLambdaBasicExecutionRole
        - Version: '2012-10-17'
          Statement:
            - Effect: Allow
              Action:
                - bedrock:InvokeModel
                - bedrock:InvokeModelWithResponseStream
              Resource: '*'
      Events:
        ApiEvent:
          Type: Api
          Properties:
            Path: /analyze
            Method: post
```

### Container Deployment (ECS)

**Production-ready container configuration:**

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Set GenOps environment
ENV GENOPS_ENVIRONMENT=production
ENV GENOPS_PROJECT=bedrock-ecs
ENV OTEL_SERVICE_NAME=bedrock-ecs-service

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run application
CMD ["python", "app.py"]
```

**ECS Task Definition:**

```json
{
    "family": "genops-bedrock-service",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "1024",
    "memory": "2048",
    "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
    "taskRoleArn": "arn:aws:iam::ACCOUNT:role/genops-bedrock-task-role",
    "containerDefinitions": [
        {
            "name": "genops-bedrock-app",
            "image": "your-account.dkr.ecr.region.amazonaws.com/genops-bedrock:latest",
            "portMappings": [{"containerPort": 8080, "protocol": "tcp"}],
            "environment": [
                {"name": "AWS_REGION", "value": "us-east-1"},
                {"name": "GENOPS_ENVIRONMENT", "value": "production"},
                {"name": "OTEL_SERVICE_NAME", "value": "bedrock-ecs"}
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/genops-bedrock-service",
                    "awslogs-region": "us-east-1",
                    "awslogs-stream-prefix": "ecs"
                }
            },
            "healthCheck": {
                "command": ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"],
                "interval": 30,
                "timeout": 5,
                "retries": 3
            }
        }
    ]
}
```

## Performance Optimization

### High-Volume Applications

**Configuration for applications processing 10,000+ operations per day:**

```python
import os

# Performance configuration
os.environ.update({
    "GENOPS_SAMPLING_RATE": "0.1",          # Sample 10% for reduced overhead
    "GENOPS_ASYNC_EXPORT": "true",          # Non-blocking telemetry
    "GENOPS_BATCH_SIZE": "50",              # Smaller batches
    "GENOPS_CIRCUIT_BREAKER": "true",       # Protect against failures
    "GENOPS_CB_THRESHOLD": "3"              # Quick failure detection
})

from genops.providers.bedrock import GenOpsBedrockAdapter

# High-volume processing with optimized configuration
adapter = GenOpsBedrockAdapter(
    enable_sampling=True,
    async_export=True,
    circuit_breaker_enabled=True
)

# Batch processing with cost optimization
batch_size = 10
documents = ["doc1", "doc2", "doc3"] * 100  # 300 documents

for i in range(0, len(documents), batch_size):
    batch = documents[i:i + batch_size]
    
    # Process batch with cost-effective model
    for doc in batch:
        result = adapter.text_generation(
            prompt=f"Process: {doc}",
            model_id="amazon.titan-text-lite-v1",  # Most cost-effective
            max_tokens=100,
            team="batch-processing"
        )
        
    # Batch telemetry export reduces overhead
    if i % 100 == 0:  # Every 10 batches
        print(f"Processed {i + batch_size} documents")
```

### Connection Pooling and Caching

**Optimize for repeated operations:**

```python
from genops.providers.bedrock import GenOpsBedrockAdapter
import functools

# Connection pooling for high-frequency operations
adapter = GenOpsBedrockAdapter(
    region_name='us-east-1',
    connection_pool_size=20,  # Increased pool size
    enable_connection_reuse=True
)

# Caching for repeated prompts
@functools.lru_cache(maxsize=1000)
def cached_classification(prompt_hash: str, model_id: str):
    """Cache classification results for repeated prompts."""
    return adapter.text_generation(
        prompt=prompt_hash,  # Use hash for cache key
        model_id=model_id,
        max_tokens=50,
        temperature=0.0  # Deterministic for caching
    )

# High-frequency processing with caching
for document in documents:
    prompt = f"Classify: {document}"
    prompt_hash = hash(prompt)  # Simple hash for demo
    
    # Use cached result if available
    result = cached_classification(prompt_hash, "anthropic.claude-3-haiku-20240307-v1:0")
```

### Circuit Breaker Pattern

**Resilience for production workloads:**

```python
from genops.providers.bedrock import GenOpsBedrockAdapter

# Circuit breaker configuration
adapter = GenOpsBedrockAdapter(
    circuit_breaker_enabled=True,
    circuit_breaker_threshold=5,        # Open after 5 failures
    circuit_breaker_timeout=60,         # Reset after 60 seconds
    circuit_breaker_fallback="cache"    # Fallback strategy
)

def resilient_analysis(document: str) -> dict:
    """Analysis with circuit breaker protection."""
    
    try:
        result = adapter.text_generation(
            prompt=f"Analyze: {document}",
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            team="resilient-ai"
        )
        
        return {
            "analysis": result.content,
            "cost": result.cost_usd,
            "source": "live"
        }
        
    except Exception as e:
        if "circuit breaker" in str(e).lower():
            # Fallback to cached result or simplified analysis
            return {
                "analysis": f"Circuit breaker active - using fallback analysis",
                "cost": 0.0,
                "source": "fallback",
                "error": str(e)
            }
        else:
            raise  # Re-raise non-circuit-breaker errors
```

## Observability Integration

### AWS CloudWatch Integration

**Native integration with CloudWatch for comprehensive monitoring:**

```python
import boto3
from genops.providers.bedrock import GenOpsBedrockAdapter

# CloudWatch metrics automatically exported by GenOps
cloudwatch = boto3.client('cloudwatch')

# Custom dashboard configuration
dashboard_config = {
    "dashboard_name": "GenOps-Bedrock-Operations",
    "widgets": [
        {
            "type": "metric",
            "properties": {
                "metrics": [
                    ["GenOps/Bedrock", "OperationCount", "Team", "ai-platform"],
                    ["GenOps/Bedrock", "TotalCost", "Team", "ai-platform"],
                    ["GenOps/Bedrock", "AverageLatency", "Team", "ai-platform"],
                    ["GenOps/Bedrock", "ErrorRate", "Team", "ai-platform"]
                ],
                "period": 300,
                "stat": "Average",
                "region": "us-east-1",
                "title": "GenOps Bedrock Metrics"
            }
        }
    ]
}

# Alarms for cost and performance monitoring
cost_alarm = {
    "alarm_name": "GenOps-Bedrock-HighCost",
    "description": "Alert when Bedrock costs exceed threshold",
    "metric_name": "CostPerOperation",
    "namespace": "GenOps/Bedrock",
    "threshold": 0.01,  # $0.01 per operation
    "comparison_operator": "GreaterThanThreshold",
    "evaluation_periods": 2
}
```

### Datadog Integration

**Export rich telemetry to Datadog:**

```python
# Configure Datadog exporter
import os

os.environ.update({
    "OTEL_EXPORTER_OTLP_ENDPOINT": "https://otlp.datadoghq.com:4317",
    "OTEL_EXPORTER_OTLP_HEADERS": "dd-api-key=your-datadog-api-key",
    "OTEL_RESOURCE_ATTRIBUTES": "service.name=bedrock-ai,env=production"
})

from genops.providers.bedrock import GenOpsBedrockAdapter

adapter = GenOpsBedrockAdapter()

# Telemetry automatically flows to Datadog with rich tags
result = adapter.text_generation(
    prompt="Customer support inquiry analysis",
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    # Rich tagging for Datadog dashboards
    team="customer-support",
    project="ai-assistant",
    customer_id="enterprise-client",
    priority="high",
    department="support"
)

# Datadog dashboard will show:
# - Costs by team, project, customer
# - Latency percentiles by model
# - Error rates and success metrics
# - Custom business metrics
```

### Custom OTLP Integration

**Works with any OTLP-compatible backend:**

```python
import os

# Configure for your observability platform
os.environ.update({
    "OTEL_EXPORTER_OTLP_ENDPOINT": "http://your-collector:4317",
    "OTEL_SERVICE_NAME": "bedrock-ai-service",
    "OTEL_RESOURCE_ATTRIBUTES": "deployment.environment=production,team.name=ai-platform"
})

from genops.providers.bedrock import GenOpsBedrockAdapter

adapter = GenOpsBedrockAdapter()

# Rich telemetry exported includes:
# - Span data with AWS context
# - Custom metrics for cost and performance
# - Resource attributes with business context
# - Baggage for cross-service correlation

result = adapter.text_generation(
    prompt="Multi-service analysis request",
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    # Business context propagated in telemetry
    team="analytics",
    project="cross-team-analysis",
    trace_id="parent-trace-id",  # Correlation with other services
    span_context="inherited"
)
```

## Advanced Use Cases

### Multi-Region Failover

**Automatic failover across AWS regions:**

```python
from genops.providers.bedrock import GenOpsBedrockAdapter

# Multi-region configuration
regions = ["us-east-1", "us-west-2", "eu-west-1"]
adapters = {
    region: GenOpsBedrockAdapter(region_name=region)
    for region in regions
}

def resilient_analysis(prompt: str, primary_region: str = "us-east-1"):
    """Analysis with automatic regional failover."""
    
    for region in [primary_region] + [r for r in regions if r != primary_region]:
        try:
            adapter = adapters[region]
            
            result = adapter.text_generation(
                prompt=prompt,
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                team="resilient-ai",
                region=region
            )
            
            print(f"✅ Success in region: {region}")
            return result
            
        except Exception as e:
            print(f"❌ Failed in region {region}: {e}")
            continue
    
    raise Exception("All regions failed - check service health")

# Use with automatic failover
result = resilient_analysis("Analyze customer feedback trends")
```

### A/B Testing for Model Performance

**Compare model performance in production:**

```python
import random
from genops.providers.bedrock import GenOpsBedrockAdapter

adapter = GenOpsBedrockAdapter()

def ab_test_models(prompt: str, customer_id: str):
    """A/B test different models for the same task."""
    
    # Model variants for testing
    models = {
        "variant_a": "anthropic.claude-3-haiku-20240307-v1:0",      # Control
        "variant_b": "anthropic.claude-3-sonnet-20240229-v1:0",    # Test
    }
    
    # Random assignment (50/50 split)
    variant = "variant_a" if random.random() < 0.5 else "variant_b"
    model = models[variant]
    
    result = adapter.text_generation(
        prompt=prompt,
        model_id=model,
        team="ab-testing",
        customer_id=customer_id,
        # A/B testing metadata
        experiment_variant=variant,
        experiment_name="model_quality_test",
        experiment_id="exp_001"
    )
    
    # Log for analysis
    print(f"🧪 A/B Test - Variant: {variant}, Cost: ${result.cost_usd:.6f}")
    
    return result, variant

# Usage in production
for customer_request in customer_requests:
    result, variant = ab_test_models(
        prompt=customer_request["prompt"],
        customer_id=customer_request["customer_id"]
    )
    
    # Track conversion metrics by variant
    track_conversion(variant, customer_request["customer_id"], result)
```

### Dynamic Budget Management

**Real-time budget management with alerts:**

```python
from genops.providers.bedrock_workflow import production_workflow_context
from genops.providers.bedrock import GenOpsBedrockAdapter

class BudgetManager:
    def __init__(self, daily_budget: float = 100.0):
        self.daily_budget = daily_budget
        self.current_spend = 0.0
        self.alert_thresholds = [0.5, 0.8, 0.9]  # 50%, 80%, 90%
    
    def check_budget(self, operation_cost: float) -> bool:
        """Check if operation is within budget."""
        projected_spend = self.current_spend + operation_cost
        return projected_spend <= self.daily_budget
    
    def record_spend(self, amount: float):
        """Record spending and check for alerts."""
        self.current_spend += amount
        utilization = self.current_spend / self.daily_budget
        
        for threshold in self.alert_thresholds:
            if utilization >= threshold:
                self.send_budget_alert(threshold, utilization)
                self.alert_thresholds.remove(threshold)  # Prevent duplicate alerts
    
    def send_budget_alert(self, threshold: float, utilization: float):
        """Send budget alert."""
        print(f"🚨 Budget Alert: {utilization:.1%} of daily budget used (threshold: {threshold:.1%})")

# Usage with budget management
budget_manager = BudgetManager(daily_budget=50.0)
adapter = GenOpsBedrockAdapter()

def budget_aware_analysis(prompt: str, max_cost: float = 0.05):
    """Perform analysis within budget constraints."""
    
    if not budget_manager.check_budget(max_cost):
        return {"error": "Budget exceeded - operation denied"}
    
    # Choose model based on remaining budget
    remaining_budget = budget_manager.daily_budget - budget_manager.current_spend
    
    if remaining_budget > 10.0:
        model = "anthropic.claude-3-opus-20240229-v1:0"  # Premium
    elif remaining_budget > 1.0:
        model = "anthropic.claude-3-sonnet-20240229-v1:0"  # Balanced
    else:
        model = "anthropic.claude-3-haiku-20240307-v1:0"  # Cost-effective
    
    result = adapter.text_generation(
        prompt=prompt,
        model_id=model,
        team="budget-conscious-ai"
    )
    
    # Record actual spend
    budget_manager.record_spend(result.cost_usd)
    
    return {
        "analysis": result.content,
        "cost": result.cost_usd,
        "model_used": model,
        "budget_remaining": budget_manager.daily_budget - budget_manager.current_spend
    }

# Budget-aware processing
for request in daily_requests:
    response = budget_aware_analysis(request["prompt"])
    if "error" not in response:
        print(f"✅ Analysis complete - Remaining budget: ${response['budget_remaining']:.2f}")
    else:
        print(f"❌ {response['error']}")
```

## Troubleshooting

### Common Issues and Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **AWS Credentials** | `NoCredentialsError`, `CredentialsNotFound` | Run `aws configure` or set environment variables |
| **Bedrock Access** | `AccessDeniedException`, `UnauthorizedOperation` | Enable model access in AWS Console → Bedrock → Model access |
| **Region Issues** | `EndpointConnectionError`, `InvalidRegion` | Use supported region like `us-east-1` |
| **Model Not Available** | `ValidationException`, `ModelNotFound` | Check model availability in your region |
| **High Costs** | Budget alerts, unexpected bills | Use cost optimization tools and budget limits |
| **Circuit Breaker** | "Circuit breaker is open" | Wait for cooldown or disable circuit breaker |
| **No Telemetry** | Missing observability data | Set `OTEL_EXPORTER_OTLP_ENDPOINT` |

### Comprehensive Diagnostics

```python
from genops.providers.bedrock import validate_bedrock_setup, print_validation_result

# Run complete diagnostic
result = validate_bedrock_setup(verbose=True)
print_validation_result(result)

# Check specific issues
if not result.success:
    print("\n🔍 Detailed Diagnostics:")
    
    for check_name, check_result in result.detailed_checks.items():
        if not check_result.passed:
            print(f"❌ {check_name}: {check_result.error}")
            print(f"💡 Fix: {check_result.fix_suggestion}")
            if check_result.documentation_link:
                print(f"📚 Docs: {check_result.documentation_link}")
            print()
```

### Debug Mode

```python
import logging
import os

# Enable debug mode
os.environ["GENOPS_LOG_LEVEL"] = "DEBUG"
logging.getLogger("genops").setLevel(logging.DEBUG)

from genops.providers.bedrock import GenOpsBedrockAdapter

# Debug information will be logged
adapter = GenOpsBedrockAdapter(debug_mode=True)

result = adapter.text_generation(
    prompt="Debug test prompt",
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    team="debugging"
)

# Debug output includes:
# - Request/response details
# - Cost calculations step-by-step  
# - Telemetry export information
# - AWS SDK interactions
```

### Performance Profiling

```python
import time
from genops.providers.bedrock import GenOpsBedrockAdapter

adapter = GenOpsBedrockAdapter(enable_profiling=True)

# Performance profiling
start_time = time.time()

result = adapter.text_generation(
    prompt="Performance test prompt",
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    team="performance-testing"
)

end_time = time.time()

print(f"⏱️ Total time: {(end_time - start_time)*1000:.2f}ms")
print(f"🚀 GenOps overhead: {result.genops_overhead_ms:.2f}ms")
print(f"🤖 Model latency: {result.model_latency_ms:.2f}ms")
print(f"📊 Telemetry export: {result.telemetry_export_ms:.2f}ms")
```

## API Reference

### Core Classes

#### `GenOpsBedrockAdapter`

**Main adapter class for Bedrock integration:**

```python
class GenOpsBedrockAdapter:
    def __init__(
        self,
        region_name: str = "us-east-1",
        default_model: str = "anthropic.claude-3-haiku-20240307-v1:0",
        enable_sampling: bool = True,
        sampling_rate: float = 1.0,
        async_export: bool = True,
        circuit_breaker_enabled: bool = False,
        debug_mode: bool = False
    ):
        """Initialize GenOps Bedrock adapter."""
    
    def text_generation(
        self,
        prompt: str,
        model_id: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        team: str = None,
        project: str = None,
        customer_id: str = None,
        environment: str = None,
        cost_center: str = None,
        feature: str = None,
        **kwargs
    ) -> BedrockResult:
        """Generate text with comprehensive governance tracking."""
    
    def is_available(self) -> bool:
        """Check if Bedrock service is available."""
    
    def get_supported_models(self, region: str = None) -> List[str]:
        """Get list of supported models in region."""
```

#### `BedrockResult`

**Result object with cost and governance data:**

```python
@dataclass
class BedrockResult:
    content: str                          # Generated content
    cost_usd: float                       # Total cost in USD
    input_cost: float                     # Input token cost
    output_cost: float                    # Output token cost
    input_tokens: int                     # Number of input tokens
    output_tokens: int                    # Number of output tokens
    latency_ms: float                     # Total latency
    model_latency_ms: float               # Model-only latency
    genops_overhead_ms: float             # GenOps processing overhead
    region: str                           # AWS region used
    model_id: str                         # Model identifier
    governance_attributes: Dict[str, str]  # Governance metadata
    span_id: str                          # OpenTelemetry span ID
    trace_id: str                         # OpenTelemetry trace ID
```

### Utility Functions

#### `validate_bedrock_setup()`

```python
def validate_bedrock_setup(
    region: str = "us-east-1",
    verbose: bool = False
) -> ValidationResult:
    """Comprehensive setup validation."""
```

#### `auto_instrument_bedrock()`

```python
def auto_instrument_bedrock(
    sampling_rate: float = 1.0,
    enable_cost_tracking: bool = True,
    export_to_cloudwatch: bool = True
) -> None:
    """Enable zero-code auto-instrumentation."""
```

#### Cost Intelligence Functions

```python
def calculate_bedrock_cost(
    input_tokens: int,
    output_tokens: int,
    model_id: str,
    region: str = "us-east-1"
) -> CostBreakdown:
    """Calculate precise costs for Bedrock operation."""

def compare_bedrock_models(
    prompt: str,
    models: List[str],
    region: str = "us-east-1"
) -> ModelComparison:
    """Compare costs and performance across models."""

def get_cost_optimization_recommendations(
    prompt: str,
    budget_constraint: float = None,
    quality_requirement: str = "medium",
    region: str = "us-east-1"
) -> OptimizationRecommendations:
    """Get intelligent model recommendations."""
```

### Context Managers

#### `create_bedrock_cost_context()`

```python
def create_bedrock_cost_context(
    context_id: str,
    budget_limit: float = None,
    alert_threshold: float = 0.8,
    enable_optimization_recommendations: bool = True
) -> BedrockCostContext:
    """Create cost tracking context for multi-operation workflows."""
```

#### `production_workflow_context()`

```python
def production_workflow_context(
    workflow_name: str,
    customer_id: str,
    team: str,
    project: str,
    environment: str = "production",
    compliance_level: ComplianceLevel = ComplianceLevel.BASIC,
    cost_center: str = None,
    budget_limit: float = None,
    region: str = "us-east-1",
    enable_cloudtrail: bool = False,
    alert_webhooks: List[str] = None
) -> Tuple[WorkflowContext, str]:
    """Create enterprise workflow context with full governance."""
```

---

## Next Steps

**🎯 You're now ready to use GenOps with AWS Bedrock!**

- **Quick Start**: Try the [5-minute quickstart guide](../bedrock-quickstart.md)
- **Examples**: Explore comprehensive examples in [`examples/bedrock/`](../../examples/bedrock/)
- **Community**: Join discussions at [GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)
- **Support**: Report issues at [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)

**📚 Related Documentation:**
- [OpenTelemetry Integration](./opentelemetry.md)
- [Multi-Provider Comparison](./providers-comparison.md)
- [Enterprise Deployment Guide](./enterprise-deployment.md)