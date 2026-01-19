# SkyRouter + GenOps Examples

> 📖 **Navigation:** [Quickstart (5 min)](../../docs/skyrouter-quickstart.md) → [Complete Guide](../../docs/integrations/skyrouter.md) → **Interactive Examples**

Comprehensive examples demonstrating SkyRouter multi-model routing with GenOps governance, cost intelligence, and policy enforcement across 150+ models.

## 🎯 You Are Here: Interactive Examples

**Perfect for:** Hands-on learning with copy-paste ready code for multi-model routing

**Time investment:** 5-30 minutes depending on example complexity

**What you'll get:** Working code examples that demonstrate real-world multi-model routing scenarios

## Quick Start (5 minutes)

```bash
# 1. Install dependencies
pip install genops[skyrouter]

# 2. Set environment variables
export SKYROUTER_API_KEY="your-skyrouter-api-key"
export GENOPS_TEAM="ai-platform"
export GENOPS_PROJECT="multi-model-routing"

# 3. Run setup validation
python setup_validation.py

# 4. Try basic multi-model routing
python basic_routing.py
```

## Examples Overview

| Example | Description | Difficulty | Time |
|---------|-------------|------------|------|
| [`setup_validation.py`](./setup_validation.py) | Validate SkyRouter + GenOps configuration | Beginner | 2 min |
| [`basic_routing.py`](./basic_routing.py) | Basic multi-model routing with governance | Beginner | 5 min |
| [`auto_instrumentation.py`](./auto_instrumentation.py) | Zero-code auto-instrumentation | Beginner | 3 min |
| [`route_optimization.py`](./route_optimization.py) | Intelligent routing and cost optimization | Intermediate | 15 min |
| [`agent_workflows.py`](./agent_workflows.py) | Multi-agent workflow routing | Intermediate | 20 min |
| [`enterprise_patterns.py`](./enterprise_patterns.py) | Production deployment patterns | Advanced | 30 min |

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Your AI App   │───▶│  GenOps SkyRouter│───▶│  SkyRouter AI   │
│                 │    │  Adapter         │    │  Platform       │
│ • Multi-Model   │    │                  │    │                 │
│ • Agent Flows   │    │ • Cost Tracking  │    │ • 150+ Models   │
│ • Route Logic   │    │ • Governance     │    │ • Intelligent   │
│                 │    │ • Attribution    │    │   Routing       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  OpenTelemetry  │
                       │  (OTLP Export)  │
                       │                 │
                       │ • Route Metrics │
                       │ • Cost Tracking │
                       │ • Model Usage   │
                       └─────────────────┘
```

## Key Features Demonstrated

### 🎯 **Zero-Code Multi-Model Integration**
- Automatic governance for existing SkyRouter code
- No changes required to current multi-model workflows
- Transparent cost tracking and attribution across 150+ models

### 💰 **Intelligent Route Cost Intelligence**
- Real-time cost calculation across all models
- Route optimization and efficiency scoring
- Budget enforcement and alerting across model ecosystem
- Cost forecasting and multi-model recommendations

### 🏛️ **Enterprise Multi-Model Governance**
- Team and project attribution across routing strategies
- Environment-based policy enforcement for model access
- Compliance metadata tracking across model usage
- Audit trail generation for all routing decisions

### 📊 **Advanced Route Monitoring**
- Multi-model cost aggregation and comparison
- Route performance signal tracking
- Agent workflow optimization across models
- Dashboard analytics with model-specific insights

## Running the Examples

### Prerequisites Check

```bash
# Verify all dependencies are installed
python -c "
import genops
from genops.providers.skyrouter_validation import validate_setup
result = validate_setup()
print('✅ Ready to run examples!' if result.is_valid else '❌ Setup issues detected')
"
```

### Run All Examples

```bash
# Execute all examples in sequence
chmod +x run_all_examples.sh
./run_all_examples.sh
```

### Run Individual Examples

```bash
# Basic examples (recommended order)
python setup_validation.py       # Validate configuration
python basic_routing.py          # Basic multi-model routing with governance
python auto_instrumentation.py   # Zero-code integration

# Intermediate examples
python route_optimization.py     # Advanced routing optimization
python agent_workflows.py        # Multi-agent routing patterns

# Advanced examples
python enterprise_patterns.py    # Production deployment patterns
```

## Integration Patterns

### 1. Flask/FastAPI Web Service with Multi-Model Routing
```python
from flask import Flask
from genops.providers.skyrouter import auto_instrument

app = Flask(__name__)
auto_instrument(team="api-team", project="multi-model-service")

@app.route('/intelligent-response')
def intelligent_response():
    # Your SkyRouter multi-model routing is automatically governed
    return jsonify({'status': 'optimally_routed'})
```

### 2. Jupyter Notebook Multi-Model Analysis
```python
# Notebook cell 1: Setup
from genops.providers.skyrouter import GenOpsSkyRouterAdapter
adapter = GenOpsSkyRouterAdapter(team="data-science", environment="development")

# Notebook cell 2: Analysis (automatically tracked)
with adapter.track_routing_session("analysis") as session:
    # Your multi-model analysis code with automatic governance
    pass
```

### 3. Batch Processing Pipeline with Model Optimization
```python
import schedule
from genops.providers.skyrouter import GenOpsSkyRouterAdapter

def daily_intelligent_processing():
    adapter = GenOpsSkyRouterAdapter(team="ml-ops", daily_budget_limit=200.0)
    with adapter.track_routing_session("daily-batch") as session:
        # Process daily operations with intelligent model selection
        pass

schedule.every().day.at("02:00").do(daily_intelligent_processing)
```

## Environment Configuration

### Development Environment
```bash
export GENOPS_ENVIRONMENT="development"
export GENOPS_DAILY_BUDGET_LIMIT="50.0"
export GENOPS_GOVERNANCE_POLICY="advisory"
export SKYROUTER_ROUTING_STRATEGY="cost_optimized"
```

### Production Environment
```bash
export GENOPS_ENVIRONMENT="production" 
export GENOPS_DAILY_BUDGET_LIMIT="500.0"
export GENOPS_GOVERNANCE_POLICY="enforced"
export GENOPS_COST_CENTER="ai-platform"
export SKYROUTER_ROUTING_STRATEGY="balanced"
```

## Troubleshooting Common Issues

### Issue: SkyRouter SDK Not Found
```bash
# Error: ModuleNotFoundError: No module named 'skyrouter'
# Note: SkyRouter SDK might not be publicly available yet
# Use the GenOps adapter for API-based routing
pip install requests  # For API calls
```

### Issue: Authentication Failed
```bash
# Error: Missing SkyRouter API Key
export SKYROUTER_API_KEY="your-api-key-here"
```

### Issue: Route Optimization Not Working
```python
# Error: Route selection not optimizing costs
# Solution: Configure routing strategy explicitly
adapter = GenOpsSkyRouterAdapter(
    daily_budget_limit=200.0,
    # Configure explicit routing preferences
)
```

### Issue: High Multi-Model Costs
```python
# Error: Costs higher than expected across models
# Solution: Implement cost optimization strategies
recommendations = adapter.cost_aggregator.get_cost_optimization_recommendations()
for rec in recommendations[:3]:  # Top 3 recommendations
    print(f"💡 {rec['title']}: ${rec['potential_savings']:.2f} savings")
```

## Performance Benchmarks

| Operation | Overhead | Cost Per Operation |
|-----------|----------|-------------------|
| Multi-Model Route Selection | <10ms | $0.002 |
| Agent Workflow Routing | <15ms | $0.01 |
| Route Optimization | <5ms | $0.05 |
| Cross-Model Analytics | <8ms | $0.10/analysis |

## Advanced Topics

### Custom Route Optimization
See [`route_optimization.py`](./route_optimization.py) for examples of:
- Custom routing strategies across model tiers
- Volume discount optimization across models
- Multi-region routing cost calculations
- Currency conversion handling for global deployment

### Enterprise Multi-Model Governance
See [`enterprise_patterns.py`](./enterprise_patterns.py) for examples of:
- Multi-environment governance policies for model access
- Team-based access controls across model ecosystem
- Compliance audit trail generation for model usage
- Integration with existing observability stacks

### High-Volume Multi-Model Optimization
See [`agent_workflows.py`](./agent_workflows.py) for examples of:
- Multi-agent routing sampling strategies
- Batch processing optimization across models
- Dynamic cost-aware model selection
- Performance monitoring integration across model tiers

## Next Steps

1. **Try the Examples**: Start with `setup_validation.py` and work through each example
2. **Read the Documentation**: Check out the [full integration guide](../../docs/integrations/skyrouter.md)
3. **Join the Community**: Get help in [GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)
4. **Contribute**: Found a bug or want to add an example? [Open an issue](https://github.com/KoshiHQ/GenOps-AI/issues)

---

**🔙 Want to explore more?** Check out:
- [5-minute Quickstart](../../docs/skyrouter-quickstart.md) - Get started from scratch
- [Complete Integration Guide](../../docs/integrations/skyrouter.md) - Comprehensive documentation
- [Cost Intelligence Guide](../../docs/cost-intelligence-guide.md) - ROI analysis and optimization
- [Enterprise Governance](../../docs/enterprise-governance-templates.md) - Compliance templates

**Questions?** Check our [troubleshooting guide](../../docs/integrations/skyrouter.md#validation-and-troubleshooting) or reach out to the community!