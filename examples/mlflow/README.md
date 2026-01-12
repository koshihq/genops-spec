# MLflow + GenOps Examples

This directory contains examples demonstrating MLflow experiment tracking with GenOps governance telemetry and cost tracking.

## Quick Start

### 1. Setup Validation

First, validate your setup:

```bash
python examples/mlflow/setup_validation.py
```

This will check:
- ✅ Required dependencies (mlflow, opentelemetry, genops)
- ✅ Configuration (tracking URI, governance attributes)
- ✅ Connectivity (MLflow tracking server)
- ✅ Governance features

### 2. Basic Tracking

Run the basic tracking example:

```bash
# Set governance environment variables (optional)
export GENOPS_TEAM="ml-team"
export GENOPS_PROJECT="model-optimization"
export GENOPS_ENVIRONMENT="development"

# Run example
python examples/mlflow/basic_tracking.py
```

This demonstrates:
- Experiment creation with governance
- Parameter and metric logging
- Artifact logging with cost tracking
- Governance attribute propagation

## Examples Overview

### Available Examples

1. **setup_validation.py** - Validate your MLflow + GenOps setup
   - Dependency checks
   - Configuration validation
   - Connectivity tests
   - Governance feature validation

2. **basic_tracking.py** - Basic experiment tracking with governance
   - Simple MLflow workflow
   - Parameter and metric logging
   - Artifact tracking
   - Cost summary

### Advanced Examples

These examples demonstrate production-ready patterns and advanced MLflow features:

3. **model_registry.py** - Model Registry Integration
   Train, register, and version models with governance tracking.
   ```bash
   python examples/mlflow/model_registry.py
   ```

4. **artifact_logging.py** - Artifact Tracking
   Log various artifact types (files, directories, plots) with cost tracking.
   ```bash
   python examples/mlflow/artifact_logging.py
   ```

5. **auto_logging.py** - Auto-Logging Integration
   Zero-code integration with scikit-learn auto-logging.
   ```bash
   python examples/mlflow/auto_logging.py
   ```

6. **hierarchical_runs.py** - Nested Run Hierarchies
   Hyperparameter search and cross-validation with parent-child runs.
   ```bash
   python examples/mlflow/hierarchical_runs.py
   ```

7. **production_workflow.py** - Production Deployment Patterns
   Multi-environment workflow (dev/staging/prod) with validation gates.
   ```bash
   python examples/mlflow/production_workflow.py
   ```

## Environment Variables

### Required (or set in code)

```bash
export MLFLOW_TRACKING_URI="http://localhost:5000"  # or "file:///mlruns"
export GENOPS_TEAM="your-team-name"
export GENOPS_PROJECT="your-project-name"
```

### Optional

```bash
export MLFLOW_REGISTRY_URI="http://localhost:5000"
export GENOPS_ENVIRONMENT="development"              # dev/staging/prod
export GENOPS_CUSTOMER_ID="customer-id"
export GENOPS_COST_CENTER="ml-research"
export GENOPS_ENABLE_AUTO_PATCHING="true"            # Enable auto-instrumentation
```

## MLflow UI

View your tracked experiments:

```bash
# Start MLflow UI
mlflow ui --backend-store-uri file:///tmp/mlruns

# Or for remote tracking server
mlflow ui --backend-store-uri http://localhost:5000
```

Then open: http://localhost:5000

## Governance Features

### Cost Tracking

All examples automatically track:
- API call costs ($0.0001 per operation)
- Artifact storage costs (based on size and backend)
- Model storage costs
- Registry operation costs

### Governance Attributes

All runs include governance tags:
- `genops.team` - Team attribution
- `genops.project` - Project tracking
- `genops.environment` - Environment segregation
- `genops.customer_id` - Customer attribution
- `genops.cost_center` - Cost center allocation

These tags are visible in the MLflow UI and can be used for:
- Cost attribution and chargeback
- Access control and permissions
- Compliance and audit trails
- Multi-tenant organization

## Troubleshooting

### MLflow not installed

```bash
pip install mlflow
```

### GenOps not found

```bash
# Install from source
cd /path/to/GenOps-AI-OTel
pip install -e .
```

### Connection errors

```bash
# Check tracking URI
echo $MLFLOW_TRACKING_URI

# Test connectivity
mlflow experiments list --tracking-uri $MLFLOW_TRACKING_URI
```

### Validation failures

Run the validation script for detailed diagnostics:

```bash
python examples/mlflow/setup_validation.py
```

## Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [GenOps Documentation](../../docs/)
- [MLflow Quickstart](../../docs/mlflow-quickstart.md)
- [MLflow Integration Guide](../../docs/integrations/mlflow.md)

## Support

For issues or questions:
- GitHub Issues: https://github.com/KoshiHQ/GenOps-AI/issues
- Documentation: https://github.com/KoshiHQ/GenOps-AI/tree/main/docs
