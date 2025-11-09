# Together AI Examples

This directory contains comprehensive examples demonstrating GenOps governance telemetry integration with Together AI's 200+ open-source models.

## 🚀 Quick Start

If you're new to GenOps + Together AI, start here:

```bash
# Install dependencies
pip install genops-ai[together] together

# Set up your API key
export TOGETHER_API_KEY="your_together_key_here"

# Run setup validation
python setup_validation.py
```

## 📚 Examples by Complexity

### Level 1: Getting Started (5 minutes)

**[setup_validation.py](setup_validation.py)**
- Verify your Together AI + GenOps setup is working correctly
- Validate API keys, dependencies, and model access
- Get immediate feedback on configuration issues
- Test 200+ model catalog accessibility

**[basic_tracking.py](basic_tracking.py)**
- Simple chat completions with automatic cost tracking
- Multi-model comparison across pricing tiers
- Session-based operation tracking
- Introduction to governance attributes

**[auto_instrumentation.py](auto_instrumentation.py)**
- Zero-code setup using GenOps auto-instrumentation
- Drop-in replacement for existing Together AI code
- Automatic telemetry for all operations
- Seamless OpenTelemetry integration

### Level 2: Cost Optimization (30 minutes)

**[cost_optimization.py](cost_optimization.py)**
- Multi-model cost comparison across 200+ models
- Task-complexity based model recommendations
- Budget-constrained operations with automatic fallbacks
- Cost projection and savings analysis

**[interactive_setup_wizard.py](interactive_setup_wizard.py)**
- Interactive configuration wizard for team onboarding
- Automated environment setup and validation
- Template generation for common use cases
- Cost-aware model selection guidance

### Level 3: Advanced Features (2 hours)

**[advanced_features.py](advanced_features.py)**
- Multimodal operations with vision-language models
- Streaming responses with real-time cost tracking
- Code generation and completion workflows
- Async batch processing and fine-tuning cost estimation

**[production_patterns.py](production_patterns.py)**
- Enterprise-ready integration patterns
- Circuit breaker and resilience patterns
- Multi-tenant governance with strict budget enforcement
- Production monitoring and observability

## 🎯 Use Case Examples

Each example includes:
- ✅ **Complete working code** you can run immediately
- ✅ **Governance attributes** for cost attribution
- ✅ **Error handling** and validation
- ✅ **Performance considerations** and best practices
- ✅ **Comments explaining** GenOps integration points

## 🔧 Running Examples

### Prerequisites

```bash
# Install GenOps with Together AI support
pip install genops-ai[together] together

# Set environment variables
export TOGETHER_API_KEY="your_together_api_key"
export OTEL_SERVICE_NAME="together-examples"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"  # Optional
```

### Run Individual Examples

```bash
# Basic examples
python setup_validation.py
python basic_tracking.py
python auto_instrumentation.py

# Cost optimization examples
python cost_optimization.py
python interactive_setup_wizard.py

# Advanced examples
python advanced_features.py
python production_patterns.py
```

### View Telemetry

Start local observability stack to see your telemetry:

```bash
# Download observability stack
curl -O https://raw.githubusercontent.com/genops-ai/genops-ai/main/docker-compose.observability.yml

# Start services
docker-compose -f docker-compose.observability.yml up -d

# View dashboards
open http://localhost:3000   # Grafana
open http://localhost:16686  # Jaeger
```

## 📊 What You'll Learn

After completing these examples, you'll understand:

- **Auto-instrumentation** for zero-code GenOps integration
- **Cost optimization** across 200+ Together AI models
- **Multi-modal capabilities** with vision and code models
- **Advanced model selection** based on task complexity and budget
- **Production deployment** patterns and enterprise governance
- **Circuit breaker patterns** for resilient AI operations
- **Observability integration** with your existing monitoring stack

## 💡 Common Use Cases

These examples demonstrate patterns for:

- **Open-source model optimization** across Llama, DeepSeek, Mixtral families
- **Cost-effective AI operations** with intelligent model selection
- **Multi-modal applications** with vision-language capabilities
- **Code generation workflows** with specialized programming models
- **Customer billing** with per-customer cost attribution
- **Team cost allocation** across projects and features
- **Enterprise governance** with strict budget controls and audit trails
- **High-throughput applications** with async batch processing

## 🤖 Together AI Model Showcase

### Available Model Categories

**💬 Chat & Reasoning Models**
- **Llama 3.1 Family**: 8B ($0.10/M), 70B ($0.88/M), 405B ($5.00/M)
- **DeepSeek R1**: Advanced reasoning with step-by-step analysis
- **Mixtral Models**: 8x7B and 8x22B variants for balanced performance

**👀 Multimodal Models**
- **Qwen2.5-VL-72B**: Vision-language understanding
- **Llama-Vision-Free**: Lightweight multimodal processing

**💻 Code Generation**
- **DeepSeek-Coder-V2**: Specialized for programming tasks
- **Qwen2.5-Coder-32B**: Advanced code completion and analysis

**⚡ Performance Tiers**
- **Lite Tier**: Ultra-fast, cost-effective (8B models from $0.10/M tokens)
- **Standard Tier**: Balanced performance and cost (70B models)
- **Large Tier**: Maximum capability (405B and specialized models)
- **Premium Tier**: State-of-the-art reasoning and multimodal

## 💰 Cost Intelligence Features

### Smart Model Selection
- **Task complexity analysis**: Automatic model recommendation based on requirements
- **Budget-aware selection**: Choose optimal models within cost constraints
- **Performance vs cost optimization**: Balance quality and expense

### Cost Tracking & Management
- **Real-time cost calculation** with accurate token-based pricing
- **Budget enforcement** with configurable governance policies
- **Multi-tenant cost attribution** for customer billing
- **Cost projection tools** for planning and budgeting

### Pricing Transparency
```python
# Compare costs across model tiers
pricing_calc = TogetherPricingCalculator()
comparisons = pricing_calc.compare_models([
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",    # $0.10/M
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",   # $0.88/M  
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"   # $5.00/M
], estimated_tokens=1000)

# Get model recommendations
rec = pricing_calc.recommend_model(
    task_complexity="moderate",
    budget_per_operation=0.01
)
```

## 🏗️ Architecture Integration

### Zero-Code Integration
```python
# Add ONE line to existing code for full governance
from genops.providers.together import auto_instrument
auto_instrument()

# Your existing Together AI code works unchanged with governance
from together import Together
client = Together()
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    messages=[{"role": "user", "content": "Hello!"}]
)
# ✅ Automatic cost tracking, governance, and observability
```

### Enterprise Governance
```python
# Full control with enterprise features
adapter = GenOpsTogetherAdapter(
    team="ai-research",
    project="model-analysis", 
    customer_id="enterprise-client",
    daily_budget_limit=100.0,
    governance_policy="strict",  # Enforce budget limits
    enable_cost_alerts=True
)

with adapter.track_session("analysis-workflow") as session:
    result = adapter.chat_with_governance(
        messages=messages,
        model=TogetherModel.LLAMA_3_1_70B_INSTRUCT,
        session_id=session.session_id,
        feature="competitive-analysis"
    )
```

## 🚨 Troubleshooting

If you encounter issues:

1. **Run validation first**: `python setup_validation.py`
2. **Check API key**: Ensure your Together API key is set and valid
3. **Verify dependencies**: Run `pip install together genops-ai[together]`
4. **Enable debug logging**: Set `export GENOPS_LOG_LEVEL=debug`
5. **Check OpenTelemetry**: Verify OTLP endpoint configuration
6. **Validate model access**: Test with basic models first

### Common Issues

**API Key Issues**
```bash
# Check API key format
echo $TOGETHER_API_KEY  # Should start with 'sk-' or 'pk-'

# Test API access
python -c "from together import Together; print(len(Together().models.list().data))"
```

**Model Access Issues**
```python
# Test specific model access
adapter = GenOpsTogetherAdapter()
result = adapter.chat_with_governance(
    messages=[{"role": "user", "content": "test"}],
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    max_tokens=10
)
```

**Budget Issues**
```python
# Check cost summary and utilization
cost_summary = adapter.get_cost_summary()
print(f"Budget utilization: {cost_summary['daily_budget_utilization']:.1f}%")
```

## 📚 Next Steps

- **[Together AI Quickstart Guide](../../docs/together-quickstart.md)** - 5-minute setup guide
- **[Together AI Integration Guide](../../docs/integrations/together.md)** - Comprehensive documentation
- **[Governance Scenarios](../governance_scenarios/)** - Policy enforcement examples
- **[Cost Optimization Guide](../cost_optimization.py)** - Advanced optimization patterns

## 🔄 Migration from Other Providers

### From OpenAI
```python
# Before (OpenAI)
from openai import OpenAI
client = OpenAI()

# After (Together AI with governance)
from genops.providers.together import GenOpsTogetherAdapter
adapter = GenOpsTogetherAdapter()
# Same interface, better models, lower costs, full governance
```

### From Anthropic
```python
# Migration helper for switching providers
result = adapter.chat_with_governance(
    messages=anthropic_messages,  # Same format
    model=TogetherModel.LLAMA_3_1_70B_INSTRUCT,  # Higher capability, lower cost
    provider_migration="from_anthropic"
)
```

## 🌟 Advanced Features Showcase

### Multimodal Operations
```python
# Vision-language analysis
result = adapter.chat_with_governance(
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
    }],
    model=TogetherModel.QWEN_VL_72B
)
```

### Code Generation
```python
# Specialized code generation
result = adapter.chat_with_governance(
    messages=[{"role": "user", "content": "Write a Python API endpoint"}],
    model=TogetherModel.DEEPSEEK_CODER_V2,
    task_type="code_generation"
)
```

### Reasoning Tasks
```python
# Advanced reasoning with R1 models
result = adapter.chat_with_governance(
    messages=[{"role": "user", "content": "Solve this step by step: ..."}],
    model=TogetherModel.DEEPSEEK_R1,
    temperature=0.1  # Lower temperature for consistent reasoning
)
```

## 💬 Support

- **Issues**: [GitHub Issues](https://github.com/genops-ai/genops-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/genops-ai/genops-ai/discussions)
- **Documentation**: [GenOps Documentation](https://docs.genops.ai)
- **Together AI Docs**: [Together AI API Documentation](https://docs.together.ai)

## 🏆 Success Metrics

After implementing Together AI + GenOps integration:

- **Cost Reduction**: Up to 10x lower costs vs proprietary models
- **Model Selection**: Access to 200+ state-of-the-art open-source models  
- **Governance**: 100% cost visibility and attribution
- **Performance**: Optimized model selection for each use case
- **Compliance**: Full audit trails and budget enforcement
- **Flexibility**: Easy switching between models and providers