# Anthropic Examples

This directory contains comprehensive examples demonstrating GenOps governance telemetry integration with Anthropic Claude applications.

## 🚀 Quick Start

If you're new to GenOps + Anthropic, start here:

```bash
# Install dependencies
pip install genops-ai[anthropic]

# Set up your API key
export ANTHROPIC_API_KEY="your_anthropic_key_here"

# Run setup validation
python setup_validation.py
```

## 📚 Examples by Complexity

### Level 1: Getting Started (5 minutes)

**[setup_validation.py](setup_validation.py)**
- Verify your Anthropic + GenOps setup is working correctly
- Validate API keys, dependencies, and basic functionality
- Get immediate feedback on configuration issues

**[basic_tracking.py](basic_tracking.py)**
- Simple Claude message creation with automatic cost and performance tracking
- Introduction to governance attributes for cost attribution
- Minimal code changes to existing Anthropic applications

**[auto_instrumentation.py](auto_instrumentation.py)**
- Zero-code setup using GenOps auto-instrumentation
- Drop-in replacement for existing Anthropic code
- Automatic telemetry for all Claude operations

### Level 2: Cost Optimization (30 minutes)

**[cost_optimization.py](cost_optimization.py)**
- Multi-model cost comparison across Claude variants (Haiku, Sonnet, Opus)
- Dynamic model selection based on complexity and cost constraints
- Cost tracking across different Claude operation types

**[multi_provider_costs.py](multi_provider_costs.py)**
- Cross-provider cost comparison (Anthropic vs. OpenAI vs. others)
- Unified cost tracking and aggregation
- Provider migration cost analysis

### Level 3: Advanced Features (2 hours)

**[advanced_features.py](advanced_features.py)**
- Streaming responses with telemetry tracking
- Multi-turn conversation management
- Document analysis and processing workflows
- System prompt optimization and testing

**[production_patterns.py](production_patterns.py)**
- Enterprise-ready integration patterns
- Context managers for complex workflows
- Policy enforcement and governance automation
- Performance optimization and scaling considerations

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
# Install GenOps with Anthropic support
pip install genops-ai[anthropic]

# Set environment variables
export ANTHROPIC_API_KEY="your_anthropic_api_key"
export OTEL_SERVICE_NAME="anthropic-examples"
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
python multi_provider_costs.py

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
- **Cost attribution** using governance attributes
- **Multi-model optimization** across Claude variants (Haiku, Sonnet, Opus)
- **Advanced Claude features** (streaming, conversations, document analysis)
- **Production deployment** patterns and best practices
- **Policy enforcement** and governance automation
- **Observability integration** with your existing monitoring stack

## 💡 Common Use Cases

These examples demonstrate patterns for:

- **Customer billing** with per-customer cost attribution
- **Team cost allocation** across projects and features
- **Cost optimization** through intelligent Claude model selection
- **Document analysis** and content generation workflows
- **Conversational AI** with multi-turn dialogue tracking
- **Legal and compliance** document review processes
- **Multi-provider strategies** for cost and reliability

## 🚨 Troubleshooting

If you encounter issues:

1. **Run validation first**: `python setup_validation.py`
2. **Check API key**: Ensure your Anthropic API key is set and valid
3. **Verify dependencies**: Run `pip install genops-ai[anthropic]`
4. **Enable debug logging**: Set `export GENOPS_LOG_LEVEL=debug`
5. **Check OpenTelemetry**: Verify OTLP endpoint configuration

## 📚 Next Steps

- **[Anthropic Quickstart Guide](../../docs/anthropic-quickstart.md)** - 5-minute setup guide
- **[Anthropic Integration Guide](../../docs/integrations/anthropic.md)** - Comprehensive documentation
- **[Governance Scenarios](../governance_scenarios/)** - Policy enforcement examples
- **[Multi-Provider Examples](../multi_provider_costs.py)** - Cross-provider comparisons

## 💬 Support

- **Issues**: [GitHub Issues](https://github.com/genops-ai/genops-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/genops-ai/genops-ai/discussions)
- **Documentation**: [GenOps Documentation](https://docs.genops.ai)
- **Anthropic Docs**: [Claude API Documentation](https://docs.anthropic.com/claude/reference/)