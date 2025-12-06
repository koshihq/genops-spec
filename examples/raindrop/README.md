# Raindrop AI + GenOps Examples

> 📖 **Navigation:** [Quickstart (5 min)](../../docs/raindrop-quickstart.md) → [Complete Guide](../../docs/integrations/raindrop.md) → **Interactive Examples**

Comprehensive examples demonstrating Raindrop AI agent monitoring with GenOps governance, cost intelligence, and policy enforcement.

## 🎯 You Are Here: Interactive Examples

**Perfect for:** Hands-on learning with copy-paste ready code

**Time investment:** 5-30 minutes depending on example complexity

**What you'll get:** Working code examples that demonstrate real-world scenarios

## Quick Start (5 minutes)

```bash
# 1. Install dependencies
pip install genops[raindrop]

# 2. Set environment variables
export RAINDROP_API_KEY="your-raindrop-api-key"
export GENOPS_TEAM="ai-platform"
export GENOPS_PROJECT="agent-monitoring"

# 3. Run setup validation
python setup_validation.py

# 4. Try basic tracking
python basic_tracking.py
```

## Examples Overview

| Example | Description | Difficulty | Time |
|---------|-------------|------------|------|
| [`setup_validation.py`](./setup_validation.py) | Validate Raindrop + GenOps configuration | Beginner | 2 min |
| [`basic_tracking.py`](./basic_tracking.py) | Basic agent monitoring with governance | Beginner | 5 min |
| [`auto_instrumentation.py`](./auto_instrumentation.py) | Zero-code auto-instrumentation | Beginner | 3 min |
| [`advanced_features.py`](./advanced_features.py) | Advanced monitoring and governance | Intermediate | 15 min |
| [`cost_optimization.py`](./cost_optimization.py) | Cost intelligence and optimization | Intermediate | 10 min |
| [`production_patterns.py`](./production_patterns.py) | Production deployment patterns | Advanced | 20 min |

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Your AI App   │───▶│  GenOps Raindrop │───▶│  Raindrop AI    │
│                 │    │  Adapter         │    │  Platform       │
│ • Agents        │    │                  │    │                 │
│ • Interactions  │    │ • Cost Tracking  │    │ • Dashboards    │
│ • Performance   │    │ • Governance     │    │ • Monitoring    │
│                 │    │ • Attribution    │    │ • Alerts        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  OpenTelemetry  │
                       │  (OTLP Export)  │
                       │                 │
                       │ • Cost Metrics  │
                       │ • Governance    │
                       │ • Attribution   │
                       └─────────────────┘
```

## Key Features Demonstrated

### 🎯 **Zero-Code Integration**
- Automatic governance for existing Raindrop code
- No changes required to current workflows
- Transparent cost tracking and attribution

### 💰 **Cost Intelligence**
- Real-time cost calculation and tracking
- Agent interaction cost optimization
- Budget enforcement and alerting
- Cost forecasting and recommendations

### 🏛️ **Enterprise Governance**
- Team and project attribution
- Environment-based policy enforcement
- Compliance metadata tracking
- Audit trail generation

### 📊 **Advanced Monitoring**
- Multi-agent cost aggregation
- Performance signal cost tracking
- Alert management cost optimization
- Dashboard analytics cost attribution

## Running the Examples

### Prerequisites Check

```bash
# Verify all dependencies are installed
python -c "
import genops
from genops.providers.raindrop_validation import validate_setup
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
python basic_tracking.py         # Basic monitoring with governance
python auto_instrumentation.py   # Zero-code integration

# Intermediate examples
python advanced_features.py      # Advanced monitoring features
python cost_optimization.py      # Cost intelligence and optimization

# Advanced examples
python production_patterns.py    # Production deployment patterns
```

## Integration Patterns

### 1. Flask/FastAPI Web Service
```python
from flask import Flask
from genops.providers.raindrop import auto_instrument

app = Flask(__name__)
auto_instrument(team="api-team", project="agent-service")

@app.route('/agent')
def agent():
    # Your Raindrop monitoring is automatically governed
    return jsonify({'status': 'tracked'})
```

### 2. Jupyter Notebook Analysis
```python
# Notebook cell 1: Setup
from genops.providers.raindrop import GenOpsRaindropAdapter
adapter = GenOpsRaindropAdapter(team="data-science", environment="development")

# Notebook cell 2: Analysis (automatically tracked)
with adapter.track_agent_monitoring_session("analysis") as session:
    # Your analysis code with automatic governance
    pass
```

### 3. Batch Processing Pipeline
```python
import schedule
from genops.providers.raindrop import GenOpsRaindropAdapter

def daily_monitoring():
    adapter = GenOpsRaindropAdapter(team="ml-ops", daily_budget_limit=75.0)
    with adapter.track_agent_monitoring_session("daily-batch") as session:
        # Process daily agent interactions with cost controls
        pass

schedule.every().day.at("02:00").do(daily_monitoring)
```

## Environment Configuration

### Development Environment
```bash
export GENOPS_ENVIRONMENT="development"
export GENOPS_DAILY_BUDGET_LIMIT="20.0"
export GENOPS_GOVERNANCE_POLICY="advisory"
```

### Production Environment
```bash
export GENOPS_ENVIRONMENT="production" 
export GENOPS_DAILY_BUDGET_LIMIT="100.0"
export GENOPS_GOVERNANCE_POLICY="enforced"
export GENOPS_COST_CENTER="ai-platform"
```

## Troubleshooting Common Issues

### Issue: SDK Not Found
```bash
# Error: ModuleNotFoundError: No module named 'raindrop'
pip install raindrop>=1.0.0
```

### Issue: Authentication Failed
```bash
# Error: Missing Raindrop API Key
export RAINDROP_API_KEY="your-api-key-here"
```

### Issue: Budget Exceeded
```python
# Error: Monitoring session would exceed daily budget
# Solution: Increase budget or switch to advisory mode
adapter = GenOpsRaindropAdapter(
    daily_budget_limit=200.0,  # Increase budget
    governance_policy="advisory"  # Or switch to advisory
)
```

## Performance Benchmarks

| Operation | Overhead | Cost Per Operation |
|-----------|----------|-------------------|
| Agent Interaction Logging | <1ms | $0.001 |
| Performance Signal Check | <5ms | $0.01 |
| Alert Creation | <2ms | $0.05 |
| Dashboard Analytics | <1ms | $0.10/day |

## Advanced Topics

### Custom Cost Models
See [`cost_optimization.py`](./cost_optimization.py) for examples of:
- Custom pricing tiers
- Volume discount optimization
- Multi-region cost calculations
- Currency conversion handling

### Enterprise Governance
See [`production_patterns.py`](./production_patterns.py) for examples of:
- Multi-environment governance policies
- Team-based access controls
- Compliance audit trail generation
- Integration with existing observability stacks

### High-Volume Optimization
See [`advanced_features.py`](./advanced_features.py) for examples of:
- Agent interaction sampling strategies
- Batch processing optimization
- Dynamic cost-aware monitoring
- Performance monitoring integration

## Next Steps

1. **Try the Examples**: Start with `setup_validation.py` and work through each example
2. **Read the Documentation**: Check out the [full integration guide](../../../docs/integrations/raindrop.md)
3. **Join the Community**: Get help in [GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)
4. **Contribute**: Found a bug or want to add an example? [Open an issue](https://github.com/KoshiHQ/GenOps-AI/issues)

---

**🔙 Want to explore more?** Check out:
- [5-minute Quickstart](../../../docs/raindrop-quickstart.md) - Get started from scratch
- [Complete Integration Guide](../../../docs/integrations/raindrop.md) - Comprehensive documentation
- [Cost Intelligence Guide](../../../docs/cost-intelligence-guide.md) - ROI analysis and optimization
- [Enterprise Governance](../../../docs/enterprise-governance-templates.md) - Compliance templates

**Questions?** Check our [troubleshooting guide](../../../docs/integrations/raindrop.md#validation-and-troubleshooting) or reach out to the community!