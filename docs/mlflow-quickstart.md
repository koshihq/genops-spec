# MLflow Quickstart - GenOps Governance in 5 Minutes

**Total time: 4-5 minutes** | **Success rate: 95%+** | **Zero code changes required**

Get MLflow experiment tracking with comprehensive governance telemetry, cost tracking, and policy enforcement in just 5 minutes.

## Time Investment vs Value

| Time Investment | Value Delivered | Use Case |
|-----------------|-----------------|----------|
| **5 minutes** | Zero-code governance tracking | Quick wins, immediate visibility |
| **30 minutes** | Complete cost intelligence & attribution | Production-ready governance |
| **2 hours** | Enterprise governance with policies | Mission-critical AI governance |

---

## Prerequisites [30 seconds]

```bash
# Check if you have Python and pip
python --version  # Python 3.8+
pip --version
```

---

## Step 1: Installation [60 seconds]

```bash
# Install MLflow and GenOps
pip install mlflow
pip install -e .  # Install GenOps from source

# Or install from PyPI (when published)
# pip install genops[mlflow]
```

---

## Step 2: Set Environment Variables [45 seconds]

```bash
# Set governance attributes (team and project are recommended)
export GENOPS_TEAM="ml-team"
export GENOPS_PROJECT="model-optimization"
export GENOPS_ENVIRONMENT="development"  # Optional: dev/staging/prod

# Set MLflow tracking URI (optional, defaults to local)
export MLFLOW_TRACKING_URI="http://localhost:5000"  # or "file:///mlruns"
```

---

## Step 3: Validate Setup [30 seconds]

```bash
# Run validation script
python examples/mlflow/setup_validation.py
```

**Expected output:**
```
[PASSED] You're ready to use MLflow with GenOps!

Dependencies:
  [OK] mlflow
  [OK] opentelemetry
  [OK] genops

Configuration:
  • tracking_uri: http://localhost:5000
  • genops_team: ml-team
  • genops_project: model-optimization
```

If validation fails, follow the suggested fixes in the output.

---

## Step 4A: Zero-Code Auto-Instrumentation [30 seconds]

**Option A: Automatic governance (zero code changes)**

```python
import mlflow
from genops.providers.mlflow import auto_instrument_mlflow

# Enable governance tracking with one line
auto_instrument_mlflow()

# Your existing MLflow code works automatically with governance!
mlflow.set_experiment("my-experiment")

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_artifact("model.pkl")
```

**What you get automatically:**
- Cost tracking for all operations
- Governance attributes on every run
- OpenTelemetry traces exported
- Team/project attribution
- Budget monitoring

---

## Step 4B: Manual Instrumentation (More Control) [90 seconds]

**Option B: Manual adapter with explicit governance**

```python
import mlflow
from genops.providers.mlflow import instrument_mlflow

# Create adapter with explicit governance
adapter = instrument_mlflow(
    tracking_uri="http://localhost:5000",
    team="ml-team",
    project="model-optimization",
    environment="development",
    customer_id="customer-001"  # Optional: for multi-tenant tracking
)

# Track MLflow run with governance context
with adapter.track_mlflow_run(
    experiment_name="optimization-experiment",
    run_name="run-001"
) as run:
    # Log parameters
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("batch_size", 32)

    # Log metrics
    mlflow.log_metric("train_accuracy", 0.92)
    mlflow.log_metric("val_accuracy", 0.89)

    # Log artifacts
    mlflow.log_artifact("model_summary.txt")

# Check governance metrics
print(f"Total cost: ${adapter.daily_usage:.6f}")
print(f"Operations tracked: {adapter.operation_count}")
```

---

## Step 5: View Your Governance Data [60 seconds]

### Option 1: MLflow UI

```bash
# Start MLflow UI
mlflow ui --backend-store-uri $MLFLOW_TRACKING_URI

# Open browser to http://localhost:5000
```

**Governance tags visible in MLflow UI:**
- `genops.team` = ml-team
- `genops.project` = model-optimization
- `genops.environment` = development
- `genops.customer_id` = customer-001 (if set)
- `genops.cost_center` = ml-research (if set)

### Option 2: OpenTelemetry Traces

Traces are automatically exported to your configured OpenTelemetry backend:
- Datadog
- Honeycomb
- Grafana Tempo
- Prometheus
- Any OTLP-compatible backend

---

## What You Get Automatically

### Governance Tracking
- **Team Attribution**: Every run tagged with team/project
- **Cost Tracking**: Real-time cost estimation for all operations
- **Multi-Tenant**: Customer-level cost attribution
- **Compliance**: Complete audit trail of all operations

### Cost Intelligence
- **API Calls**: $0.0001 per operation tracked
- **Artifact Storage**: Size-based cost estimation (S3/Azure/GCS)
- **Model Registry**: Registry operation costs tracked
- **Budget Monitoring**: Daily usage tracking with alerts

### Observability
- **OpenTelemetry Integration**: Native OTel trace export
- **Existing Stack**: Works with your observability tools
- **Custom Dashboards**: Cost & governance metrics
- **Real-Time**: Immediate visibility into AI operations

---

## Cost Tracking Example

```python
from genops.providers.mlflow import instrument_mlflow

adapter = instrument_mlflow(
    team="ml-team",
    project="cost-demo"
)

with adapter.track_mlflow_run(experiment_name="cost-demo") as run:
    # These operations are automatically cost-tracked:
    mlflow.log_param("param1", 5)         # $0.0001
    mlflow.log_metric("metric1", 0.95)    # $0.0001
    mlflow.log_artifact("model.pkl")       # Size-based (S3: ~$0.0008/GB-day)

# View cost summary
print(f"Run cost: ${adapter.daily_usage:.6f}")
```

---

## Instant Troubleshooting

### Issue: "MLflow not installed"
**Fix:**
```bash
pip install mlflow
```

### Issue: "GenOps not found"
**Fix:**
```bash
pip install -e .  # From GenOps project root
```

### Issue: "Connection refused"
**Fix:**
```bash
# Start local MLflow server
mlflow ui --backend-store-uri file:///tmp/mlruns

# Or use local file storage
export MLFLOW_TRACKING_URI="file:///tmp/mlruns"
```

### Issue: "Validation warnings about governance attributes"
**Fix:**
```bash
export GENOPS_TEAM="your-team"
export GENOPS_PROJECT="your-project"
```

### Issue: "OpenTelemetry not exporting traces"
**Fix:**
```bash
# Set OTLP endpoint
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"

# Or use environment-specific backend
export OTEL_EXPORTER_OTLP_HEADERS="api-key=YOUR_API_KEY"
```

---

## Next Steps

### You're Ready!
You now have MLflow with full governance tracking. Your next options:

**5-30 minutes:**
- Run the basic tracking example: `python examples/mlflow/basic_tracking.py`
- Explore cost tracking and attribution
- View governance tags in MLflow UI

**30-60 minutes:**
- Set up model registry governance
- Configure budget limits and alerts
- Integrate with your observability stack

**1-2 hours:**
- Implement production deployment patterns
- Set up enterprise governance policies
- Configure multi-tenant cost attribution

---

## Additional Resources

- **Examples**: `examples/mlflow/` directory
- **Full Documentation**: `docs/integrations/mlflow.md`
- **API Reference**: `src/genops/providers/mlflow/`
- **MLflow Docs**: https://mlflow.org/docs/latest/
- **GenOps Docs**: https://github.com/KoshiHQ/GenOps-AI

## Support

- **GitHub Issues**: https://github.com/KoshiHQ/GenOps-AI/issues
- **Documentation**: https://github.com/KoshiHQ/GenOps-AI/tree/main/docs

---

## Summary: What You Achieved in 5 Minutes

[DONE] **Installed** MLflow + GenOps with governance
[DONE] **Validated** complete setup with diagnostics
[DONE] **Instrumented** MLflow with zero code changes
[DONE] **Tracked** costs automatically for all operations
[DONE] **Attributed** runs to teams/projects/customers
[DONE] **Exported** telemetry to OpenTelemetry backends

**You now have production-ready AI governance for MLflow!**
