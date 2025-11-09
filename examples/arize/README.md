# Arize AI + GenOps Examples

> 📖 **Navigation:** [Quickstart (5 min)](../../../docs/arize-quickstart.md) → [Complete Guide](../../../docs/integrations/arize.md) → **Interactive Examples**

Comprehensive examples demonstrating Arize AI model monitoring with GenOps governance, cost intelligence, and policy enforcement.

## 🎯 You Are Here: Interactive Examples

**Perfect for:** Hands-on learning with copy-paste ready code

**Time investment:** 5-30 minutes depending on example complexity

**What you'll get:** Working code examples that demonstrate real-world scenarios

## Quick Start (5 minutes)

```bash
# 1. Install dependencies
pip install genops[arize]

# 2. Set environment variables
export ARIZE_API_KEY="your-arize-api-key"
export ARIZE_SPACE_KEY="your-arize-space-key"
export GENOPS_TEAM="ml-platform"
export GENOPS_PROJECT="fraud-detection"

# 3. Run setup validation
python setup_validation.py

# 4. Try basic tracking
python basic_tracking.py
```

## Examples Overview

| Example | Description | Difficulty | Time |
|---------|-------------|------------|------|
| [`setup_validation.py`](./setup_validation.py) | Validate Arize + GenOps configuration | Beginner | 2 min |
| [`basic_tracking.py`](./basic_tracking.py) | Basic model monitoring with governance | Beginner | 5 min |
| [`auto_instrumentation.py`](./auto_instrumentation.py) | Zero-code auto-instrumentation | Beginner | 3 min |
| [`advanced_features.py`](./advanced_features.py) | Advanced monitoring and governance | Intermediate | 15 min |
| [`cost_optimization.py`](./cost_optimization.py) | Cost intelligence and optimization | Intermediate | 10 min |
| [`production_patterns.py`](./production_patterns.py) | Production deployment patterns | Advanced | 20 min |

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Your ML App   │───▶│  GenOps Arize    │───▶│  Arize AI       │
│                 │    │  Adapter         │    │  Platform       │
│ • Predictions   │    │                  │    │                 │
│ • Data Quality  │    │ • Cost Tracking  │    │ • Dashboards    │
│ • Alerts        │    │ • Governance     │    │ • Monitoring    │
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
- Automatic governance for existing Arize code
- No changes required to current workflows
- Transparent cost tracking and attribution

### 💰 **Cost Intelligence**
- Real-time cost calculation and tracking
- Volume discount optimization
- Budget enforcement and alerting
- Cost forecasting and recommendations

### 🏛️ **Enterprise Governance**
- Team and project attribution
- Environment-based policy enforcement
- Compliance metadata tracking
- Audit trail generation

### 📊 **Advanced Monitoring**
- Multi-model cost aggregation
- Data quality cost tracking
- Alert management cost optimization
- Dashboard analytics cost attribution

## Running the Examples

### Prerequisites Check

```bash
# Verify all dependencies are installed
python -c "
import genops
from genops.providers.arize_validation import validate_setup
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

## Example Outputs

### Setup Validation Success
```
🔍 Arize AI Integration Validation Report
============================================================

✅ Overall Status: SUCCESS

📊 Validation Summary:
  • SDK Installation: 0 issues
  • Authentication: 0 issues
  • Configuration: 0 issues
  • Governance: 1 issues

💡 Recommendations:
  1. All validation checks passed successfully!

🚀 Next Steps:
  1. You can now use GenOps Arize integration with confidence
```

### Expected Example Outputs

#### Setup Validation (`setup_validation.py`)
```bash
$ python setup_validation.py

🔍 Arize AI + GenOps Setup Validation
============================================================

📋 Environment Configuration Check:
  ✅ ARIZE_API_KEY configured
  ✅ ARIZE_SPACE_KEY configured
  ✅ GENOPS_TEAM configured
  ✅ GENOPS_PROJECT configured

🔍 Arize AI Integration Validation Report
============================================================

✅ Overall Status: SUCCESS

📊 Validation Summary:
  • SDK Installation: 0 issues
  • Authentication: 0 issues
  • Configuration: 0 issues

💡 Recommendations:
  1. All validation checks passed successfully!

🚀 Next Steps:
  1. You can now use GenOps Arize integration with confidence

✅ Setup validation completed successfully!
```

#### Basic Tracking (`basic_tracking.py`)
```bash
$ python basic_tracking.py

🚀 Arize AI + GenOps Basic Tracking Example
============================================================

📋 Prerequisites Check:
  ✅ GenOps installed
  ✅ Arize AI SDK available
  ✅ ARIZE_API_KEY configured
  ✅ ARIZE_SPACE_KEY configured

🎯 Starting basic model monitoring with governance...

✅ Model monitoring session started: fraud-detection-basic
📊 Logged prediction batch: 1000 predictions, cost: $1.00
🔍 Data quality metrics logged, cost: $0.05
🚨 Performance alert created for accuracy, cost: $0.10

💰 Session Cost Summary:
  Total: $1.15
  Prediction Logging: $1.00
  Data Quality: $0.05
  Alert Management: $0.10
  Dashboard: $0.10
  Efficiency: 869.57 predictions/hour

📊 Governance Metrics:
  Team: basic-tracking-team
  Project: fraud-detection-demo
  Daily Usage: $1.15
  Budget Remaining: $48.85

✅ Basic tracking example completed successfully!
```

#### Auto-Instrumentation (`auto_instrumentation.py`)
```bash
$ python auto_instrumentation.py

🚀 Arize AI + GenOps Zero-Code Auto-Instrumentation Example
============================================================

🔄 Enabling auto-instrumentation for existing Arize workflows...
✅ Auto-instrumentation activated

📋 Your existing Arize code now includes:
  🏷️ Team and project attribution
  💰 Automatic cost tracking
  📊 Governance telemetry export
  🔍 Budget monitoring and alerts

🎯 Simulating existing Arize client usage...

✅ Prediction logged: pred-001 (fraud) - $0.001
✅ Prediction logged: pred-002 (legitimate) - $0.001
✅ Prediction logged: pred-003 (fraud) - $0.001

📊 Auto-Instrumentation Summary:
  Operations Tracked: 3
  Total Cost: $0.003
  Governance Attributes Added: 6
  Telemetry Spans Created: 3

💡 Zero code changes required - existing workflows now governed!
✅ Auto-instrumentation example completed successfully!
```

#### Cost Optimization (`cost_optimization.py`)
```bash
$ python cost_optimization.py

💡 Arize AI + GenOps Cost Optimization Example
============================================================

📊 Analyzing current monitoring costs...

📈 Monthly Cost Summary:
  Total Cost: $145.50
  Budget Utilization: 58.2%
  Top Cost Driver: fraud-model-v3-3.1 ($89.25)
  Models Monitored: 5
  Average Cost per Model: $29.10

🔧 Cost Optimization Opportunities:

  1. Optimize High-Frequency Prediction Logging
     💰 Potential Savings: $43.65/month
     ⚡ Effort Level: Medium
     📊 Priority Score: 75.0/100
     🔧 Actions:
       • Implement intelligent sampling (reduce volume by 30%)
       • Use batch prediction logging
       • Optimize prediction data payload size

  2. Streamline Alert Configuration
     💰 Potential Savings: $18.50/month
     ⚡ Effort Level: Low
     📊 Priority Score: 60.0/100
     🔧 Actions:
       • Consolidate similar alert rules
       • Increase alert thresholds for non-critical models
       • Implement alert suppression during maintenance

📊 Volume Discount Analysis:
  Current Tier: Silver (15% discount)
  Next Tier: Gold (25% discount) at 2M predictions/month
  Potential Additional Savings: $14.55/month

💰 Total Optimization Potential: $76.70/month (52.7% savings)

✅ Cost optimization analysis completed!
```

#### Advanced Features (`advanced_features.py`)
```bash
$ python advanced_features.py

🚀 Arize AI + GenOps Advanced Features Demo
============================================================

📊 Multi-Model Production Monitoring Demo
--------------------------------------------------

🔄 Starting concurrent model monitoring...
  ✅ fraud-detection-v3: $4.250 cost, 3 alerts
  ✅ recommendation-engine-v2: $15.750 cost, 2 alerts
  ✅ sentiment-analysis-v1: $6.825 cost, 2 alerts
  ✅ churn-prediction-v2: $2.160 cost, 4 alerts

📊 Multi-Model Monitoring Summary:
  💰 Total monitoring cost: $28.99
  📈 Total predictions monitored: 228,000
  🚨 Total active alerts: 11
  🏭 Models monitored: 4

💡 Advanced Cost Intelligence Demo
----------------------------------------

🔍 Cost breakdown by model:
  • recommendation-engine-v2: $15.75 (54.3%)
  • sentiment-analysis-v1: $6.83 (23.5%)
  • fraud-detection-v3: $4.25 (14.7%)
  • churn-prediction-v2: $2.16 (7.4%)

🚀 Cost Optimization Recommendations:
  1. Model Right-Sizing
     💰 Potential savings: $12.50
     ⚡ Effort level: Medium
     📊 Priority score: 82.5/100

📈 Monitoring Efficiency Analysis:
  📊 Cost per prediction: $0.0001
  🔍 Cost per data quality check: $0.050
  🚨 Cost per alert: $2.63
  💵 Predictions per dollar: 7,863

✅ Advanced features demo completed successfully!
```

#### Production Patterns (`production_patterns.py`)
```bash
$ python production_patterns.py

🏭 Arize AI + GenOps Production Deployment Patterns
================================================================================

🏗️ Enterprise Architecture Patterns
--------------------------------------

🌐 Multi-Region Enterprise Deployment:

📍 PRODUCTION-PRIMARY Configuration:
  🌍 Region: us-east-1
  🏗️ Instances: 3
  💰 Daily budget: $500.0
  🔒 Governance: enforced
  📊 Monitoring: comprehensive
  📋 Compliance: SOX, GDPR, HIPAA
  ✅ Adapter configured and ready

📍 PRODUCTION-SECONDARY Configuration:
  🌍 Region: us-west-2
  🏗️ Instances: 2
  💰 Daily budget: $300.0
  🔒 Governance: enforced
  📊 Monitoring: essential
  📋 Compliance: SOX, GDPR
  ✅ Adapter configured and ready

🏭 Enterprise Architecture Summary:
  🌐 Total regions: 2
  🖥️ Total instances: 6
  💰 Total budget: $900.0
  🔒 Compliance coverage: SOX, GDPR, HIPAA, Internal

⚡ High-Availability & Disaster Recovery
------------------------------------------

🔄 Active-Passive HA Configuration:
  🟢 Primary: us-east-1 (active)
  🟡 Secondary: us-west-2 (standby)

🎭 Disaster Recovery Simulation:
  🎯 Attempting primary region monitoring...
  ✅ Primary monitoring successful: 500 predictions
  🎉 Monitoring maintained via primary region

✅ Production deployment patterns demonstrated successfully!
```

## Integration Patterns

### 1. Flask/FastAPI Web Service
```python
from flask import Flask
from genops.providers.arize import auto_instrument

app = Flask(__name__)
auto_instrument(team="api-team", project="prediction-service")

@app.route('/predict')
def predict():
    # Your Arize logging is automatically governed
    return jsonify({'status': 'tracked'})
```

### 2. Jupyter Notebook Analysis
```python
# Notebook cell 1: Setup
from genops.providers.arize import GenOpsArizeAdapter
adapter = GenOpsArizeAdapter(team="data-science", environment="development")

# Notebook cell 2: Analysis (automatically tracked)
with adapter.track_model_monitoring_session("analysis") as session:
    # Your analysis code with automatic governance
    pass
```

### 3. Batch Processing Pipeline
```python
import schedule
from genops.providers.arize import GenOpsArizeAdapter

def daily_monitoring():
    adapter = GenOpsArizeAdapter(team="ml-ops", daily_budget_limit=75.0)
    with adapter.track_model_monitoring_session("daily-batch") as session:
        # Process daily predictions with cost controls
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
export GENOPS_COST_CENTER="ml-platform"
```

## Troubleshooting Common Issues

### Issue: SDK Not Found
```bash
# Error: ModuleNotFoundError: No module named 'arize'
pip install arize>=6.0.0
```

### Issue: Authentication Failed
```bash
# Error: Missing Arize API Key
export ARIZE_API_KEY="your-api-key-here"
export ARIZE_SPACE_KEY="your-space-key-here"
```

### Issue: Budget Exceeded
```python
# Error: Monitoring session would exceed daily budget
# Solution: Increase budget or switch to advisory mode
adapter = GenOpsArizeAdapter(
    daily_budget_limit=200.0,  # Increase budget
    governance_policy="advisory"  # Or switch to advisory
)
```

## Performance Benchmarks

| Operation | Overhead | Cost Per Operation |
|-----------|----------|-------------------|
| Prediction Logging | <1ms | $0.001 |
| Data Quality Check | <5ms | $0.01 |
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
- Prediction sampling strategies
- Batch processing optimization
- Dynamic cost-aware monitoring
- Performance monitoring integration

## Next Steps

1. **Try the Examples**: Start with `setup_validation.py` and work through each example
2. **Read the Documentation**: Check out the [full integration guide](../../../docs/integrations/arize.md)
3. **Join the Community**: Get help in [GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)
4. **Contribute**: Found a bug or want to add an example? [Open an issue](https://github.com/KoshiHQ/GenOps-AI/issues)

---

**🔙 Want to explore more?** Check out:
- [5-minute Quickstart](../../../docs/arize-quickstart.md) - Get started from scratch
- [Complete Integration Guide](../../../docs/integrations/arize.md) - Comprehensive documentation
- [Cost Intelligence Guide](../../../docs/cost-intelligence-guide.md) - ROI analysis and optimization
- [Enterprise Governance](../../../docs/enterprise-governance-templates.md) - Compliance templates

**Questions?** Check our [troubleshooting guide](../../../docs/integrations/arize.md#validation-and-troubleshooting) or reach out to the community!