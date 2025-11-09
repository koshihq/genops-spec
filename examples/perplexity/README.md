# Perplexity Examples

This directory contains comprehensive examples demonstrating GenOps governance telemetry integration with Perplexity AI search and research applications.

## 🚀 Quick Start

If you're new to GenOps + Perplexity, start here:

```bash
# Install dependencies
pip install genops-ai[perplexity]

# Set up your API key
export PERPLEXITY_API_KEY="pplx-your_perplexity_key_here"

# Run setup validation
python setup_validation.py
```

## 📚 Examples by Complexity

### Level 1: Getting Started (5 minutes)

**[setup_validation.py](setup_validation.py)**
- Verify your Perplexity + GenOps setup is working correctly
- Validate API keys, dependencies, and basic functionality
- Get immediate feedback on configuration issues

**[basic_search.py](basic_search.py)**
- Simple Perplexity search with automatic cost and performance tracking
- Introduction to governance attributes for search cost attribution
- Minimal code changes to existing Perplexity applications

**[auto_instrumentation.py](auto_instrumentation.py)**
- Zero-code setup using GenOps auto-instrumentation
- Drop-in replacement for existing Perplexity code
- Automatic telemetry for all search operations

### Level 2: Search Optimization (30 minutes)

**[cost_optimization.py](cost_optimization.py)**
- Multi-model cost comparison across Perplexity variants (Sonar models)
- Dynamic model selection based on search complexity and cost constraints
- Search context optimization (current, academic, news, etc.)

**[advanced_search.py](advanced_search.py)**
- Advanced search patterns with context management
- Citation tracking and source attribution
- Multi-turn research workflows with session management

### Level 3: Production Features (2 hours)

**[production_patterns.py](production_patterns.py)**
- Enterprise-ready integration patterns
- Context managers for complex research workflows
- Policy enforcement and governance automation
- Performance optimization and scaling considerations

**[interactive_setup_wizard.py](interactive_setup_wizard.py)**
- Interactive configuration wizard for team onboarding
- Automated environment setup and validation
- Template generation for common use cases

## 🎯 Use Case Examples

Each example includes:
- ✅ **Complete working code** you can run immediately
- ✅ **Governance attributes** for search cost attribution
- ✅ **Error handling** and validation
- ✅ **Performance considerations** and best practices
- ✅ **Comments explaining** GenOps integration points

## 🔧 Running Examples

### Prerequisites

```bash
# Install GenOps with Perplexity support
pip install genops-ai[perplexity]

# Set environment variables
export PERPLEXITY_API_KEY="pplx-your_perplexity_api_key"
export OTEL_SERVICE_NAME="perplexity-examples"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"  # Optional
```

### Run Individual Examples

```bash
# Basic examples
python setup_validation.py
python basic_search.py
python auto_instrumentation.py

# Search optimization examples
python cost_optimization.py
python advanced_search.py

# Production examples
python production_patterns.py
python interactive_setup_wizard.py
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
- **Search cost attribution** using governance attributes
- **Multi-model optimization** across Perplexity Sonar variants
- **Advanced search features** (citations, contexts, session management)
- **Production deployment** patterns and best practices
- **Research workflow** optimization and automation
- **Observability integration** with your existing monitoring stack

## 💡 Common Use Cases

These examples demonstrate patterns for:

- **Research workflows** with citation tracking and source attribution
- **Customer billing** with per-customer search cost attribution
- **Team cost allocation** across research projects and features
- **Search optimization** through intelligent model and context selection
- **Academic research** with scholarly source prioritization
- **News monitoring** and current events tracking
- **Multi-provider strategies** for search and content generation
- **Compliance tracking** for research and fact-checking workflows

## 🔍 Perplexity-Specific Features

### Search Contexts
- **Current**: Real-time web search with latest information
- **Academic**: Scholarly articles and research papers
- **News**: Current news and breaking stories
- **Social**: Social media and community discussions

### Model Selection
- **Sonar Small**: Fast, cost-effective for simple queries
- **Sonar Large**: Comprehensive analysis for complex research
- **Sonar Huge**: Maximum capability for in-depth research

### Citation Management
- **Source tracking**: Automatic citation collection and attribution
- **Quality scoring**: Source reliability and credibility assessment
- **Link preservation**: Permanent links to referenced materials

## 🚨 Troubleshooting

If you encounter issues:

1. **Run validation first**: `python setup_validation.py`
2. **Check API key**: Ensure your Perplexity API key is set and valid
3. **Verify dependencies**: Run `pip install genops-ai[perplexity]`
4. **Enable debug logging**: Set `export GENOPS_LOG_LEVEL=debug`
5. **Check OpenTelemetry**: Verify OTLP endpoint configuration
6. **Validate search context**: Ensure proper context selection for your use case

## 📚 Next Steps

- **[Perplexity Quickstart Guide](../../docs/perplexity-quickstart.md)** - 5-minute setup guide
- **[Perplexity Integration Guide](../../docs/integrations/perplexity.md)** - Comprehensive documentation
- **[Governance Scenarios](../governance_scenarios/)** - Policy enforcement examples
- **[Search Optimization Guide](../search_optimization.py)** - Advanced search patterns

## 💬 Support

- **Issues**: [GitHub Issues](https://github.com/genops-ai/genops-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/genops-ai/genops-ai/discussions)
- **Documentation**: [GenOps Documentation](https://docs.genops.ai)
- **Perplexity Docs**: [Perplexity API Documentation](https://docs.perplexity.ai)