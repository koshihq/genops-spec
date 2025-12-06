# Raindrop AI Integration

> 📖 **Navigation:** [Quickstart (5 min)](../raindrop-quickstart.md) → **Complete Guide** → [Examples](../../examples/raindrop/)

Complete integration guide for Raindrop AI agent monitoring with GenOps governance, cost intelligence, and policy enforcement.

## 🗺️ Choose Your Learning Path

**👋 New to Raindrop + GenOps?** Start here:
1. **[5-minute Quickstart](../raindrop-quickstart.md)** - Get running with zero code changes
2. **[Interactive Examples](../../examples/raindrop/)** - Copy-paste working code
3. **Come back here** for deep-dive documentation

**📚 Looking for specific info?** Jump to:
- [Cost Intelligence & ROI](../cost-intelligence-guide.md) - Calculate ROI and optimize costs
- [Performance Optimization](../raindrop-performance-benchmarks.md) - Benchmarks, scaling, memory optimization
- [Enterprise Governance](../enterprise-governance-templates.md) - Compliance templates (SOX, GDPR, HIPAA)
- [Production Patterns](#enterprise-deployment-patterns) - HA, scaling, monitoring

## 🗺️ Visual Learning Path

```
🚀 START HERE: 5-minute Quickstart
│   ├── Zero-code setup
│   ├── Basic validation
│   └── Success confirmation
│
├─── 📋 HANDS-ON: Interactive Examples (5-30 min)
│    ├── basic_tracking.py      → See governance in action
│    ├── cost_optimization.py   → Learn cost intelligence  
│    ├── advanced_features.py   → Multi-agent patterns
│    └── production_patterns.py → Enterprise deployment
│
├─── 📖 DEEP-DIVE: Complete Guide (15-60 min)
│    ├── Manual Configuration   → Full control & customization
│    ├── Governance Policies    → Team attribution & budgets
│    ├── Production Monitoring  → Dashboards & alerting
│    └── Troubleshooting       → Problem solving
│
├─── 💰 BUSINESS: Cost Intelligence (15-45 min)
│    ├── ROI Calculator        → Business justification
│    ├── Cost Optimization     → Reduce monitoring costs
│    └── Budget Forecasting    → Plan future investments
│
├─── ⚡ PERFORMANCE: Optimization & Scaling (15-60 min)
│    ├── Performance Benchmarks → Measure overhead impact
│    ├── Memory Optimization    → Large-scale deployments
│    ├── Concurrent Monitoring  → Multi-agent patterns
│    └── Production Tuning      → High-frequency scenarios
│
└─── 🏢 ENTERPRISE: Governance Templates (30-120 min)
     ├── SOX Compliance        → Financial regulations
     ├── GDPR Compliance       → EU data protection
     ├── HIPAA Compliance      → Healthcare requirements
     └── Multi-Tenant Setup    → SaaS deployments
```

**🎯 Choose your path based on:**
- **Time available:** 5 min (Quickstart) → 30 min (Examples) → 60+ min (Enterprise)
- **Role:** Developer (Examples) → FinOps (Cost Intelligence) → Architect (Enterprise)
- **Goal:** Quick setup → Production deployment → Compliance requirements

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start) ⏱️ 5 minutes
- [Manual Adapter Usage](#manual-adapter-usage) ⏱️ 15 minutes
- [Cost Intelligence](#cost-intelligence) ⏱️ 10 minutes  
- [Governance Configuration](#governance-configuration) ⏱️ 20 minutes
- [Enterprise Deployment Patterns](#enterprise-deployment-patterns) ⏱️ 30 minutes
- [Production Monitoring](#production-monitoring) ⏱️ 20 minutes
- [Validation and Troubleshooting](#validation-and-troubleshooting) ⏱️ 10 minutes
- [API Reference](#api-reference)

**🚀 Advanced Guides:**
- **[Cost Intelligence & ROI Guide](../cost-intelligence-guide.md)** - ROI templates, cost optimization, and budget forecasting
- **[Production Deployment Patterns](../examples/raindrop/production_patterns.py)** - Enterprise architecture and scaling patterns

## Overview

The GenOps Raindrop AI integration provides comprehensive governance for AI agent monitoring operations. Raindrop AI is an AI monitoring platform that discovers silent agent failures and provides performance insights for AI systems. This integration adds cost tracking, team attribution, and policy enforcement to your Raindrop AI workflows.

### 🚀 Quick Value Proposition

| ⏱️ Time Investment | 💰 Value Delivered | 🎯 Use Case |
|-------------------|-------------------|-------------|
| **5 minutes** | Zero-code governance for existing Raindrop workflows | Quick wins |
| **30 minutes** | Complete cost intelligence and optimization | Production ready |
| **2 hours** | Enterprise governance with compliance | Mission critical |

### Key Features

- **Agent Monitoring Governance**: Enhanced interaction tracking and performance monitoring with cost attribution
- **Performance Signal Intelligence**: Cost tracking for agent performance signals and evaluation metrics  
- **Alert Management**: Governed alert creation with cost optimization and team attribution
- **Deep Search Operations**: Cost tracking for agent behavior analysis and debugging
- **Experiment Management**: A/B testing cost tracking with governance integration
- **Budget Enforcement**: Real-time cost tracking with configurable budget limits and alerts
- **Zero-Code Auto-Instrumentation**: Transparent governance for existing Raindrop AI code
- **Multi-Environment Support**: Environment-specific monitoring with governance policies

> 💡 **New to Raindrop AI?** Check our [5-minute quickstart guide](../raindrop-quickstart.md) for immediate setup.

## Quick Start

### Prerequisites

```bash
# Install GenOps with Raindrop AI support
pip install genops[raindrop]

# Verify installation
python -c "import genops; print('✅ GenOps installed successfully!')"
```

### Environment Setup

```bash
# Required: Raindrop AI credentials
export RAINDROP_API_KEY="your-raindrop-api-key"

# Recommended: Team attribution
export GENOPS_TEAM="ai-platform"
export GENOPS_PROJECT="agent-monitoring"

# Optional: Budget and governance
export GENOPS_DAILY_BUDGET_LIMIT="100.0"
export GENOPS_GOVERNANCE_POLICY="enforced"
```

### Zero-Code Auto-Instrumentation

```python
from genops.providers.raindrop import auto_instrument

# Enable governance for all Raindrop AI operations
auto_instrument(
    team="ai-platform",
    project="agent-monitoring",
    daily_budget_limit=100.0
)

# Your existing Raindrop code now includes governance
import raindrop

client = raindrop.Client(api_key="your-api-key")
response = client.track_interaction(
    agent_id="support-bot-1",
    interaction_data={
        "input": "Customer support query",
        "output": "Agent response",
        "performance_signals": {"latency": 250, "accuracy": 0.94}
    }
)
# ✅ Automatically tracked with cost attribution and governance
```

## Manual Adapter Usage

For advanced use cases requiring fine-grained control:

```python
from genops.providers.raindrop import GenOpsRaindropAdapter

# Initialize adapter with custom configuration
adapter = GenOpsRaindropAdapter(
    raindrop_api_key="your-api-key",
    team="ai-platform",
    project="agent-monitoring",
    environment="production",
    daily_budget_limit=100.0,
    enable_cost_alerts=True,
    governance_policy="enforced"
)

# Context manager for session tracking
with adapter.track_agent_monitoring_session("support-agents") as session:
    # Track agent interactions
    cost_result = session.track_agent_interaction(
        agent_id="support-bot-1",
        interaction_data={
            "input": "Customer inquiry",
            "output": "Resolution provided",
            "performance_metrics": {
                "response_time": 250,
                "confidence_score": 0.94,
                "customer_satisfaction": 4.5
            }
        },
        complexity="enterprise"
    )
    
    # Track performance signals
    signal_cost = session.track_performance_signal(
        signal_name="accuracy_monitoring",
        signal_data={
            "threshold": 0.85,
            "current_value": 0.94,
            "monitoring_frequency": "high"
        },
        complexity="complex"
    )
    
    # Create alerts for performance issues
    alert_cost = session.create_alert(
        alert_name="performance_degradation",
        alert_config={
            "conditions": [
                {"metric": "accuracy", "operator": "<", "threshold": 0.85}
            ],
            "notification_channels": ["slack", "pagerduty"],
            "severity": "critical"
        }
    )
    
    print(f"Session cost: ${session.total_cost:.3f}")
```

## Cost Intelligence

### Real-Time Cost Tracking

```python
# Get comprehensive cost breakdown
summary = adapter.cost_aggregator.get_summary()

print(f"Total cost: ${summary.total_cost:.2f}")
print(f"Operations: {summary.total_operations}")

# Cost by operation type
for op_type, cost in summary.cost_by_operation_type.items():
    percentage = (cost / summary.total_cost) * 100
    print(f"  {op_type}: ${cost:.2f} ({percentage:.1f}%)")

# Cost by team/project
for team, cost in summary.cost_by_team.items():
    print(f"Team {team}: ${cost:.2f}")
```

### Volume Discount Optimization

```python
# Configure pricing for enterprise volume
from genops.providers.raindrop_pricing import RaindropPricingConfig

custom_pricing = RaindropPricingConfig()
custom_pricing.volume_tiers = {
    1000: 0.05,    # 5% discount for 1K+ interactions
    10000: 0.15,   # 15% discount for 10K+ interactions  
    100000: 0.25   # 25% discount for 100K+ interactions
}

adapter.pricing_calculator.config = custom_pricing
adapter.pricing_calculator.update_monthly_volume(25000)

# Get volume discount information
volume_info = adapter.pricing_calculator.get_volume_discount_info()
print(f"Current discount: {volume_info['current_discount_percentage']:.1f}%")
```

### Cost Optimization Recommendations

```python
# Get automated optimization recommendations
recommendations = adapter.cost_aggregator.get_cost_optimization_recommendations()

for rec in recommendations:
    print(f"💡 {rec['title']}")
    print(f"   Savings: ${rec['potential_savings']:.2f}/month")
    print(f"   Effort: {rec['effort_level']}")
    print(f"   Priority: {rec['priority_score']:.1f}/100")
```

## Governance Configuration

### Team-Based Budget Management

```python
# Set team-specific budgets
adapter.cost_aggregator.set_team_budget("ai-platform", 200.0)  # $200/day
adapter.cost_aggregator.set_project_budget("agent-monitoring", 150.0)  # $150/day

# Check budget status
budget_status = adapter.cost_aggregator.check_budget_status()

if budget_status['budget_alerts']:
    for alert in budget_status['budget_alerts']:
        print(f"🚨 {alert['message']}")
```

### Multi-Environment Governance

```python
# Environment-specific configurations
environments = {
    "development": {
        "daily_budget": 25.0,
        "governance_policy": "advisory",
        "monitoring_level": "basic"
    },
    "staging": {
        "daily_budget": 75.0,
        "governance_policy": "advisory", 
        "monitoring_level": "standard"
    },
    "production": {
        "daily_budget": 250.0,
        "governance_policy": "enforced",
        "monitoring_level": "comprehensive"
    }
}

# Initialize environment-specific adapter
env = "production"
adapter = GenOpsRaindropAdapter(
    environment=env,
    daily_budget_limit=environments[env]["daily_budget"],
    governance_policy=environments[env]["governance_policy"]
)
```

### Compliance Integration

```python
# SOX compliance configuration
sox_adapter = GenOpsRaindropAdapter(
    team="finance-ai",
    project="risk-assessment",
    environment="production",
    governance_policy="enforced",
    export_telemetry=True  # Required for audit trails
)

# Add compliance metadata
sox_adapter.governance_attrs.cost_center = "finance-operations"
sox_adapter.governance_attrs.feature = "fraud-detection"

# Track compliance-sensitive operations
with sox_adapter.track_agent_monitoring_session("compliance-monitoring") as session:
    # All operations automatically include audit trail
    pass
```

## Enterprise Deployment Patterns

### High-Availability Configuration

```python
# Primary region adapter
primary_adapter = GenOpsRaindropAdapter(
    team="production-primary",
    environment="production",
    daily_budget_limit=500.0,
    governance_policy="enforced"
)

# Secondary region adapter  
secondary_adapter = GenOpsRaindropAdapter(
    team="production-secondary",
    environment="production", 
    daily_budget_limit=300.0,
    governance_policy="enforced"
)

def monitor_with_failover():
    """Monitoring with automatic failover."""
    try:
        # Try primary region
        with primary_adapter.track_agent_monitoring_session("ha-monitoring") as session:
            return session
    except Exception:
        # Failover to secondary
        with secondary_adapter.track_agent_monitoring_session("ha-failover") as session:
            return session
```

### Multi-Tenant SaaS Configuration

```python
def create_tenant_adapter(tenant_id: str, plan: str) -> GenOpsRaindropAdapter:
    """Create tenant-specific adapter with plan-based limits."""
    
    plan_configs = {
        "starter": {"daily_budget": 10.0, "complexity": "simple"},
        "professional": {"daily_budget": 50.0, "complexity": "moderate"},
        "enterprise": {"daily_budget": 200.0, "complexity": "enterprise"}
    }
    
    config = plan_configs[plan]
    
    return GenOpsRaindropAdapter(
        team=f"tenant-{tenant_id}",
        project=f"saas-{plan}",
        customer_id=tenant_id,
        daily_budget_limit=config["daily_budget"],
        governance_policy="enforced"
    )

# Usage example
tenant_adapter = create_tenant_adapter("customer-123", "professional")
```

## Production Monitoring

### Dashboard Integration

```python
# OpenTelemetry dashboard configuration
import os
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://your-collector:4317"

# Grafana dashboard queries
grafana_queries = {
    "total_cost": 'sum(genops_cost_total{provider="raindrop"})',
    "cost_by_team": 'sum by (genops_team) (genops_cost_total{provider="raindrop"})',
    "operations_rate": 'rate(genops_operations_total{provider="raindrop"}[5m])',
    "error_rate": 'rate(genops_errors_total{provider="raindrop"}[5m])'
}

# Datadog dashboard configuration
datadog_metrics = [
    "genops.raindrop.cost.total",
    "genops.raindrop.operations.count",
    "genops.raindrop.session.duration", 
    "genops.raindrop.budget.utilization"
]
```

### Alerting Configuration

```python
# Custom alerting rules
alerting_config = {
    "budget_threshold": {
        "condition": "daily_cost > daily_budget * 0.8",
        "channels": ["slack", "email"],
        "severity": "warning"
    },
    "cost_spike": {
        "condition": "hourly_cost > avg_hourly_cost * 2",
        "channels": ["pagerduty", "slack"],
        "severity": "critical"
    },
    "failure_rate": {
        "condition": "error_rate > 0.05",
        "channels": ["pagerduty"],
        "severity": "critical"
    }
}

# Implement custom alerting
def setup_custom_alerts(adapter: GenOpsRaindropAdapter):
    """Setup custom alerting based on cost and performance thresholds."""
    
    @adapter.on_cost_threshold(threshold=0.8)
    def budget_warning(cost_info):
        print(f"⚠️ Budget warning: {cost_info['utilization']:.1f}% used")
    
    @adapter.on_error_rate_threshold(threshold=0.05)
    def error_alert(error_info):
        print(f"🚨 High error rate: {error_info['rate']:.2f}")
```

## Validation and Troubleshooting

### Setup Validation

```python
from genops.providers.raindrop_validation import validate_setup, print_validation_result

# Comprehensive validation
result = validate_setup()
print_validation_result(result, verbose=True)

# Interactive validation for missing config
if not result.is_valid:
    from genops.providers.raindrop_validation import validate_setup_interactive
    interactive_result = validate_setup_interactive()
```

### Common Issues and Solutions

#### Issue: API Authentication Failed
```python
# Diagnosis
import os
api_key = os.getenv("RAINDROP_API_KEY")
if not api_key:
    print("❌ RAINDROP_API_KEY not set")
elif len(api_key) < 20:
    print("⚠️ API key appears too short")

# Solution
export RAINDROP_API_KEY="your-complete-api-key-here"
```

#### Issue: High Costs
```python
# Diagnosis: Check cost breakdown
summary = adapter.cost_aggregator.get_summary()
print("Top cost drivers:")
for agent, cost in sorted(summary.cost_by_agent.items(), 
                         key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {agent}: ${cost:.2f}")

# Solution: Implement cost optimization
recommendations = adapter.cost_aggregator.get_cost_optimization_recommendations()
for rec in recommendations[:3]:  # Top 3 recommendations
    print(f"💡 {rec['title']}: ${rec['potential_savings']:.2f} savings")
```

#### Issue: Missing Telemetry Data
```python
# Diagnosis: Check OpenTelemetry configuration
import os
print(f"OTLP Endpoint: {os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT')}")

# Solution: Configure OTLP export
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://your-collector:4317"
os.environ["OTEL_SERVICE_NAME"] = "raindrop-monitoring"
```

### Performance Optimization

```python
# Batch processing for high-volume scenarios
class BatchedRaindropAdapter(GenOpsRaindropAdapter):
    def __init__(self, batch_size=100, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.batch_operations = []
    
    def batch_track_interactions(self, interactions):
        """Process interactions in batches for better performance."""
        for i in range(0, len(interactions), self.batch_size):
            batch = interactions[i:i + self.batch_size]
            with self.track_agent_monitoring_session(f"batch-{i}") as session:
                for interaction in batch:
                    session.track_agent_interaction(**interaction)

# Usage for high-volume monitoring
adapter = BatchedRaindropAdapter(
    batch_size=50,
    team="high-volume-team",
    daily_budget_limit=500.0
)
```

## API Reference

### GenOpsRaindropAdapter

#### Constructor Parameters

```python
GenOpsRaindropAdapter(
    raindrop_api_key: str = None,          # Raindrop AI API key
    team: str = "default",                 # Team for cost attribution
    project: str = "default",              # Project for cost attribution  
    environment: str = "production",       # Environment (dev/staging/prod)
    customer_id: str = None,               # Customer ID for multi-tenant
    cost_center: str = None,               # Cost center for financial reporting
    feature: str = None,                   # Feature for granular attribution
    daily_budget_limit: float = None,      # Daily spending limit in USD
    enable_cost_alerts: bool = True,       # Enable budget and cost alerting
    governance_policy: str = "enforced",   # Policy level (advisory/enforced)
    export_telemetry: bool = True          # Enable OpenTelemetry export
)
```

#### Methods

```python
# Context manager for session tracking
@contextmanager
def track_agent_monitoring_session(self, session_name: str, **kwargs) -> RaindropMonitoringSession

# Individual cost calculation methods  
def calculate_interaction_cost(self, agent_id: str, interaction_data: dict, complexity: str = "simple") -> RaindropCostResult
def calculate_signal_cost(self, signal_name: str, signal_data: dict, complexity: str = "simple") -> RaindropCostResult
def calculate_alert_cost(self, alert_name: str, alert_config: dict, complexity: str = "simple") -> RaindropCostResult
```

### RaindropMonitoringSession

#### Methods

```python
# Track individual operations
def track_agent_interaction(self, agent_id: str, interaction_data: dict, cost: float = None) -> RaindropCostResult
def track_performance_signal(self, signal_name: str, signal_data: dict, cost: float = None) -> RaindropCostResult
def create_alert(self, alert_name: str, alert_config: dict, cost: float = None) -> RaindropCostResult

# Session properties
@property
def total_cost(self) -> Decimal          # Total session cost
@property
def operation_count(self) -> int         # Number of operations
@property 
def duration_seconds(self) -> float      # Session duration
```

### Auto-Instrumentation

```python
# Enable zero-code governance
def auto_instrument(
    raindrop_api_key: str = None,
    team: str = "default", 
    project: str = "default",
    environment: str = "production",
    **kwargs
) -> GenOpsRaindropAdapter

# Disable auto-instrumentation
def restore_raindrop() -> None
```

### Validation

```python
# Validation functions
def validate_setup(raindrop_api_key: str = None) -> ValidationResult
def print_validation_result(result: ValidationResult, verbose: bool = True) -> None
def validate_setup_interactive() -> ValidationResult
```

---

## 🚀 Next Steps

1. **Try the Examples**: Start with our [interactive examples](../../examples/raindrop/) to see real-world patterns
2. **Production Deployment**: Follow our [enterprise deployment guide](../../examples/raindrop/production_patterns.py)
3. **Cost Optimization**: Run the [cost optimization example](../../examples/raindrop/cost_optimization.py) for immediate savings
4. **Join the Community**: Get help in [GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)

**📖 Additional Resources:**
- [Performance Optimization Guide](../raindrop-performance-benchmarks.md) - Benchmarks, scaling, and optimization
- [Cost Intelligence Guide](../cost-intelligence-guide.md) - ROI calculation and optimization
- [Enterprise Governance Templates](../enterprise-governance-templates.md) - Compliance patterns
- [Production Monitoring Guide](../production-monitoring-guide.md) - Dashboard and alerting setup