# SkyRouter Integration

> 📖 **Navigation:** [Quickstart (5 min)](../skyrouter-quickstart.md) → **Complete Guide** → [Examples](../../examples/skyrouter/)

Complete integration guide for SkyRouter multi-model routing platform with GenOps governance, cost intelligence, and policy enforcement across 150+ models.

## 🗺️ Choose Your Learning Path

**👋 New to SkyRouter + GenOps?** Start here:
1. **[5-minute Quickstart](../skyrouter-quickstart.md)** - Get running with zero code changes
2. **[Interactive Examples](../../examples/skyrouter/)** - Copy-paste working code
3. **Come back here** for deep-dive documentation

**📚 Looking for specific info?** Jump to:
- [Cost Intelligence & ROI](../cost-intelligence-guide.md) - Calculate ROI and optimize multi-model costs
- [Performance Optimization](../skyrouter-performance-benchmarks.md) - Benchmarks, scaling, memory optimization
- [Enterprise Governance](../enterprise-governance-templates.md) - Compliance templates (SOX, GDPR, HIPAA)
- [Production Patterns](#enterprise-deployment-patterns) - HA, scaling, monitoring

## 🗺️ Visual Learning Path

```
🚀 START HERE: 5-minute Quickstart
│   ├── Zero-code setup for multi-model routing
│   ├── Basic validation across model ecosystem
│   └── Success confirmation with route tracking
│
├─── 📋 HANDS-ON: Interactive Examples (5-30 min)
│    ├── route_optimization.py      → See intelligent routing in action
│    ├── multi_model_routing.py     → Learn cost-aware model selection  
│    ├── agent_workflows.py         → Multi-agent routing patterns
│    └── enterprise_patterns.py     → Production multi-model deployment
│
├─── 📖 DEEP-DIVE: Complete Guide (15-60 min)
│    ├── Manual Configuration       → Full control & customization
│    ├── Route Governance Policies  → Team attribution & budgets
│    ├── Production Monitoring      → Dashboards & alerting
│    └── Troubleshooting           → Problem solving
│
├─── 💰 BUSINESS: Multi-Model Cost Intelligence (15-45 min)
│    ├── Route ROI Calculator       → Business justification for routing
│    ├── Model Cost Optimization    → Reduce costs across 150+ models
│    └── Budget Forecasting         → Plan future multi-model investments
│
├─── ⚡ PERFORMANCE: Optimization & Scaling (15-60 min)
│    ├── Route Performance Benchmarks → Measure routing overhead impact
│    ├── Memory Optimization          → Large-scale multi-model deployments
│    ├── Concurrent Routing           → High-throughput routing patterns
│    └── Production Tuning            → High-frequency routing scenarios
│
└─── 🏢 ENTERPRISE: Multi-Model Governance Templates (30-120 min)
     ├── SOX Compliance              → Financial regulations for AI routing
     ├── GDPR Compliance             → EU data protection across models
     ├── HIPAA Compliance            → Healthcare requirements for routing
     └── Multi-Tenant Setup          → SaaS deployments with model isolation
```

**🎯 Choose your path based on:**
- **Time available:** 5 min (Quickstart) → 30 min (Examples) → 60+ min (Enterprise)
- **Role:** Developer (Examples) → FinOps (Cost Intelligence) → Architect (Enterprise)
- **Goal:** Quick setup → Production deployment → Compliance requirements

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start) ⏱️ 5 minutes
- [Manual Adapter Usage](#manual-adapter-usage) ⏱️ 15 minutes
- [Multi-Model Cost Intelligence](#multi-model-cost-intelligence) ⏱️ 10 minutes  
- [Route Governance Configuration](#route-governance-configuration) ⏱️ 20 minutes
- [Enterprise Deployment Patterns](#enterprise-deployment-patterns) ⏱️ 30 minutes
- [Production Monitoring](#production-monitoring) ⏱️ 20 minutes
- [Validation and Troubleshooting](#validation-and-troubleshooting) ⏱️ 10 minutes
- [API Reference](#api-reference)

**🚀 Advanced Guides:**
- **[Cost Intelligence & ROI Guide](../cost-intelligence-guide.md)** - ROI templates, cost optimization, and budget forecasting
- **[Production Deployment Patterns](../examples/skyrouter/enterprise_patterns.py)** - Enterprise architecture and scaling patterns

## Overview

The GenOps SkyRouter integration provides comprehensive governance for multi-model AI routing operations across 150+ models. SkyRouter is an AI routing platform that provides unified access to multiple LLM providers with intelligent routing, cost optimization, and agent-specific features for AI applications.

### 🚀 Quick Value Proposition

| ⏱️ Time Investment | 💰 Value Delivered | 🎯 Use Case |
|-------------------|-------------------|-------------|
| **5 minutes** | Zero-code governance for existing SkyRouter routing | Quick wins |
| **30 minutes** | Complete multi-model cost intelligence and optimization | Production ready |
| **2 hours** | Enterprise governance with compliance across model ecosystem | Mission critical |

### Key Features

- **Multi-Model Routing Governance**: Enhanced cost tracking across 150+ models with intelligent route selection
- **Agent Workflow Intelligence**: Cost tracking for complex multi-agent workflows with model optimization  
- **Route Efficiency Analysis**: Cost optimization recommendations and performance vs cost analysis
- **Global Load Balancing**: Cost tracking across regional deployments with intelligent failover
- **Experiment Management**: A/B testing cost tracking with multi-model comparison
- **Budget Enforcement**: Real-time cost tracking with configurable budget limits across all models
- **Zero-Code Auto-Instrumentation**: Transparent governance for existing SkyRouter code
- **Multi-Environment Support**: Environment-specific routing with governance policies

> 💡 **New to SkyRouter?** Check our [5-minute quickstart guide](../skyrouter-quickstart.md) for immediate setup.

## Quick Start

### Prerequisites

```bash
# Install GenOps with SkyRouter support
pip install genops[skyrouter]

# Verify installation
python -c "import genops; print('✅ GenOps installed successfully!')"
```

### Environment Setup

```bash
# Required: SkyRouter credentials
export SKYROUTER_API_KEY="your-skyrouter-api-key"

# Recommended: Team attribution
export GENOPS_TEAM="ai-platform"
export GENOPS_PROJECT="multi-model-routing"

# Optional: Budget and governance
export GENOPS_DAILY_BUDGET_LIMIT="200.0"
export GENOPS_GOVERNANCE_POLICY="enforced"
```

### Zero-Code Auto-Instrumentation

```python
from genops.providers.skyrouter import auto_instrument

# Enable governance for all SkyRouter routing operations
auto_instrument(
    team="ai-platform",
    project="multi-model-routing",
    daily_budget_limit=200.0
)

# Your existing SkyRouter code now includes governance
import skyrouter

client = skyrouter.Client(api_key="your-api-key")

# Multi-model routing with automatic governance
response = client.route_to_best_model(
    candidates=["gpt-4", "claude-3-sonnet", "gemini-pro"],
    prompt="Explain quantum computing to a 10-year-old",
    routing_strategy="cost_optimized",
    fallback_models=["gpt-3.5-turbo", "llama-2"]
)

# Agent workflow with automatic governance
workflow_result = client.run_agent_workflow(
    workflow_name="customer_support",
    steps=[
        {"model": "gpt-3.5-turbo", "task": "classify_intent"},
        {"model": "claude-3-sonnet", "task": "draft_response"},
        {"model": "gpt-4", "task": "quality_review"}
    ]
)
# ✅ Automatically tracked with cost attribution and governance
```

## Manual Adapter Usage

For advanced use cases requiring fine-grained control:

```python
from genops.providers.skyrouter import GenOpsSkyRouterAdapter

# Initialize adapter with custom configuration
adapter = GenOpsSkyRouterAdapter(
    skyrouter_api_key="your-api-key",
    team="ai-platform",
    project="multi-model-routing",
    environment="production",
    daily_budget_limit=200.0,
    enable_cost_alerts=True,
    governance_policy="enforced"
)

# Context manager for session tracking
with adapter.track_routing_session("intelligent-routing") as session:
    # Track multi-model routing operation
    cost_result = session.track_multi_model_routing(
        models=["gpt-4", "claude-3-sonnet", "gemini-pro"],
        input_data={
            "prompt": "Write a technical blog post about AI safety",
            "requirements": ["technical_depth", "accessibility", "1500_words"]
        },
        routing_strategy="balanced",
        complexity="enterprise"
    )
    
    # Track agent workflow with multiple steps
    workflow_cost = session.track_agent_workflow(
        workflow_name="content_creation",
        agent_steps=[
            {
                "model": "claude-3-sonnet",
                "input": {"task": "outline_creation", "topic": "AI safety"},
                "complexity": "moderate"
            },
            {
                "model": "gpt-4",
                "input": {"task": "content_writing", "outline": "..."},
                "complexity": "complex"
            },
            {
                "model": "gemini-pro",
                "input": {"task": "fact_checking", "content": "..."},
                "complexity": "simple"
            }
        ]
    )
    
    # Track single model call with route optimization
    single_model_cost = session.track_model_call(
        model="gpt-3.5-turbo",
        input_data={"prompt": "Summarize the blog post"},
        route_optimization="cost_optimized",
        complexity="simple"
    )
    
    print(f"Session cost: ${session.total_cost:.3f}")
    print(f"Operations: {session.operation_count}")
```

## Multi-Model Cost Intelligence

### Real-Time Cost Tracking

```python
# Get comprehensive cost breakdown across all models
summary = adapter.cost_aggregator.get_summary()

print(f"Total cost: ${summary.total_cost:.2f}")
print(f"Operations: {summary.total_operations}")

# Cost by model (across all 150+ supported models)
for model, cost in summary.cost_by_model.items():
    percentage = (cost / summary.total_cost) * 100
    print(f"  {model}: ${cost:.2f} ({percentage:.1f}%)")

# Cost by routing strategy
for route, cost in summary.cost_by_route.items():
    percentage = (cost / summary.total_cost) * 100
    print(f"  {route}: ${cost:.2f} ({percentage:.1f}%)")

# Cost by team/project
for team, cost in summary.cost_by_team.items():
    print(f"Team {team}: ${cost:.2f}")
```

### Route Optimization Analysis

```python
# Configure pricing for enterprise multi-model usage
from genops.providers.skyrouter_pricing import SkyRouterPricingConfig

custom_pricing = SkyRouterPricingConfig()
custom_pricing.volume_tiers = {
    1000: 0.05,    # 5% discount for 1K+ tokens
    10000: 0.15,   # 15% discount for 10K+ tokens  
    100000: 0.25,  # 25% discount for 100K+ tokens
    1000000: 0.35  # 35% discount for enterprise volume
}

adapter.pricing_calculator.config = custom_pricing
adapter.pricing_calculator.update_monthly_volume(50000)

# Get volume discount information
volume_info = adapter.pricing_calculator.get_volume_discount_info()
print(f"Current discount: {volume_info['current_discount_percentage']:.1f}%")
```

### Multi-Model Cost Optimization Recommendations

```python
# Get automated optimization recommendations across model ecosystem
recommendations = adapter.cost_aggregator.get_cost_optimization_recommendations()

for rec in recommendations:
    print(f"💡 {rec['title']}")
    print(f"   Savings: ${rec['potential_savings']:.2f}/month")
    print(f"   Effort: {rec['effort_level']}")
    print(f"   Priority: {rec['priority_score']:.1f}/100")
    print(f"   Strategy: {rec.get('optimization_type', 'general')}")
```

## Route Governance Configuration

### Team-Based Budget Management

```python
# Set team-specific budgets for multi-model usage
adapter.cost_aggregator.set_team_budget("ai-platform", 400.0)  # $400/day
adapter.cost_aggregator.set_project_budget("multi-model-routing", 300.0)  # $300/day

# Check budget status across all routing strategies
budget_status = adapter.cost_aggregator.check_budget_status()

if budget_status['budget_alerts']:
    for alert in budget_status['budget_alerts']:
        print(f"🚨 {alert['message']}")
```

### Multi-Environment Route Governance

```python
# Environment-specific configurations for different routing needs
environments = {
    "development": {
        "daily_budget": 50.0,
        "governance_policy": "advisory",
        "preferred_models": ["gpt-3.5-turbo", "claude-3-haiku"],
        "routing_strategy": "cost_optimized"
    },
    "staging": {
        "daily_budget": 150.0,
        "governance_policy": "advisory", 
        "preferred_models": ["gpt-4", "claude-3-sonnet", "gemini-pro"],
        "routing_strategy": "balanced"
    },
    "production": {
        "daily_budget": 500.0,
        "governance_policy": "enforced",
        "preferred_models": ["gpt-4", "claude-3-opus", "gemini-pro"],
        "routing_strategy": "reliability_first"
    }
}

# Initialize environment-specific adapter
env = "production"
adapter = GenOpsSkyRouterAdapter(
    environment=env,
    daily_budget_limit=environments[env]["daily_budget"],
    governance_policy=environments[env]["governance_policy"]
)
```

### Compliance Integration

```python
# SOX compliance configuration for financial AI routing
sox_adapter = GenOpsSkyRouterAdapter(
    team="finance-ai",
    project="risk-assessment",
    environment="production",
    governance_policy="enforced",
    export_telemetry=True  # Required for audit trails
)

# Add compliance metadata
sox_adapter.governance_attrs.cost_center = "finance-operations"
sox_adapter.governance_attrs.feature = "fraud-detection"

# Track compliance-sensitive multi-model operations
with sox_adapter.track_routing_session("compliance-routing") as session:
    # All routing operations automatically include audit trail
    compliance_result = session.track_multi_model_routing(
        models=["gpt-4", "claude-3-opus"],  # Only approved models
        input_data={"financial_analysis_request": "..."},
        routing_strategy="reliability_first"  # Compliance requires reliability
    )
```

## Enterprise Deployment Patterns

### High-Availability Multi-Model Configuration

```python
# Primary region adapter for high-availability routing
primary_adapter = GenOpsSkyRouterAdapter(
    team="production-primary",
    environment="production",
    daily_budget_limit=1000.0,
    governance_policy="enforced"
)

# Secondary region adapter with failover models  
secondary_adapter = GenOpsSkyRouterAdapter(
    team="production-secondary",
    environment="production", 
    daily_budget_limit=600.0,
    governance_policy="enforced"
)

def route_with_failover():
    """Multi-model routing with automatic failover."""
    try:
        # Try primary region with preferred models
        with primary_adapter.track_routing_session("ha-routing") as session:
            return session.track_multi_model_routing(
                models=["gpt-4", "claude-3-opus", "gemini-pro"],
                input_data={"critical_request": "..."},
                routing_strategy="reliability_first"
            )
    except Exception:
        # Failover to secondary with alternative models
        with secondary_adapter.track_routing_session("ha-failover") as session:
            return session.track_multi_model_routing(
                models=["gpt-3.5-turbo", "claude-3-sonnet", "llama-2"],
                input_data={"critical_request": "..."},
                routing_strategy="cost_optimized"
            )
```

### Multi-Tenant SaaS Configuration

```python
def create_tenant_router(tenant_id: str, plan: str) -> GenOpsSkyRouterAdapter:
    """Create tenant-specific router with plan-based model access."""
    
    plan_configs = {
        "starter": {
            "daily_budget": 25.0, 
            "models": ["gpt-3.5-turbo", "claude-3-haiku"],
            "routing_strategy": "cost_optimized"
        },
        "professional": {
            "daily_budget": 100.0, 
            "models": ["gpt-4", "claude-3-sonnet", "gemini-pro"],
            "routing_strategy": "balanced"
        },
        "enterprise": {
            "daily_budget": 500.0, 
            "models": ["gpt-4", "claude-3-opus", "gemini-pro", "gpt-4-turbo"],
            "routing_strategy": "reliability_first"
        }
    }
    
    config = plan_configs[plan]
    
    return GenOpsSkyRouterAdapter(
        team=f"tenant-{tenant_id}",
        project=f"saas-{plan}",
        customer_id=tenant_id,
        daily_budget_limit=config["daily_budget"],
        governance_policy="enforced"
    )

# Usage example
tenant_router = create_tenant_router("customer-123", "professional")
```

## Production Monitoring

### Dashboard Integration

```python
# OpenTelemetry dashboard configuration
import os
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://your-collector:4317"

# Grafana dashboard queries for multi-model monitoring
grafana_queries = {
    "total_cost": 'sum(genops_cost_total{provider="skyrouter"})',
    "cost_by_team": 'sum by (genops_team) (genops_cost_total{provider="skyrouter"})',
    "cost_by_model": 'sum by (skyrouter_model) (genops_cost_total{provider="skyrouter"})',
    "routing_efficiency": 'avg(skyrouter_route_efficiency_score)',
    "operations_rate": 'rate(genops_operations_total{provider="skyrouter"}[5m])',
    "error_rate": 'rate(genops_errors_total{provider="skyrouter"}[5m])'
}

# Datadog dashboard configuration for SkyRouter
datadog_metrics = [
    "genops.skyrouter.cost.total",
    "genops.skyrouter.operations.count",
    "genops.skyrouter.routing.efficiency", 
    "genops.skyrouter.budget.utilization",
    "genops.skyrouter.model.distribution"
]
```

### Alerting Configuration

```python
# Custom alerting rules for multi-model routing
alerting_config = {
    "budget_threshold": {
        "condition": "daily_cost > daily_budget * 0.8",
        "channels": ["slack", "email"],
        "severity": "warning"
    },
    "cost_spike": {
        "condition": "hourly_cost > avg_hourly_cost * 2.5",
        "channels": ["pagerduty", "slack"],
        "severity": "critical"
    },
    "route_efficiency_drop": {
        "condition": "route_efficiency_score < 0.7",
        "channels": ["slack"],
        "severity": "warning"
    },
    "model_failure_rate": {
        "condition": "model_error_rate > 0.05",
        "channels": ["pagerduty"],
        "severity": "critical"
    }
}

# Implement custom alerting for multi-model scenarios
def setup_custom_alerts(adapter: GenOpsSkyRouterAdapter):
    """Setup custom alerting based on cost and routing performance thresholds."""
    
    @adapter.on_cost_threshold(threshold=0.8)
    def budget_warning(cost_info):
        print(f"⚠️ Budget warning: {cost_info['utilization']:.1f}% used")
    
    @adapter.on_route_efficiency_threshold(threshold=0.7)
    def efficiency_alert(efficiency_info):
        print(f"🚨 Route efficiency low: {efficiency_info['score']:.2f}")
```

## Validation and Troubleshooting

### Setup Validation

```python
from genops.providers.skyrouter_validation import validate_setup, print_validation_result

# Comprehensive validation
result = validate_setup()
print_validation_result(result, verbose=True)

# Interactive validation for missing config
if not result.is_valid:
    from genops.providers.skyrouter_validation import validate_setup_interactive
    interactive_result = validate_setup_interactive()
```

### Common Issues and Solutions

#### Issue: API Authentication Failed
```python
# Diagnosis
import os
api_key = os.getenv("SKYROUTER_API_KEY")
if not api_key:
    print("❌ SKYROUTER_API_KEY not set")
elif len(api_key) < 20:
    print("⚠️ API key appears too short")

# Solution
export SKYROUTER_API_KEY="your-complete-api-key-here"
```

#### Issue: High Multi-Model Costs
```python
# Diagnosis: Check cost breakdown across models
summary = adapter.cost_aggregator.get_summary()
print("Top cost drivers:")
for model, cost in sorted(summary.cost_by_model.items(), 
                         key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {model}: ${cost:.2f}")

# Solution: Implement model optimization
recommendations = adapter.cost_aggregator.get_cost_optimization_recommendations()
for rec in recommendations[:3]:  # Top 3 recommendations
    print(f"💡 {rec['title']}: ${rec['potential_savings']:.2f} savings")
```

#### Issue: Route Optimization Not Working
```python
# Diagnosis: Check routing strategy effectiveness
route_summary = summary.cost_by_route
for route, cost in route_summary.items():
    efficiency = cost / summary.total_cost if summary.total_cost > 0 else 0
    print(f"Route {route}: ${cost:.2f} ({efficiency:.1%} of total)")

# Solution: Optimize routing strategy
optimal_routes = ["cost_optimized", "balanced"]
current_routes = list(route_summary.keys())
recommendations = []

for route in optimal_routes:
    if route not in current_routes:
        recommendations.append(f"Try '{route}' routing strategy for better cost efficiency")

for rec in recommendations:
    print(f"💡 {rec}")
```

#### Issue: Missing Telemetry Data
```python
# Diagnosis: Check OpenTelemetry configuration
import os
print(f"OTLP Endpoint: {os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT')}")

# Solution: Configure OTLP export
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://your-collector:4317"
os.environ["OTEL_SERVICE_NAME"] = "skyrouter-multi-model"
```

### Performance Optimization

```python
# Batch processing for high-volume multi-model routing
class BatchedSkyRouterAdapter(GenOpsSkyRouterAdapter):
    def __init__(self, batch_size=50, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.batch_operations = []
    
    def batch_route_models(self, routing_requests):
        """Process multi-model routing in batches for better performance."""
        for i in range(0, len(routing_requests), self.batch_size):
            batch = routing_requests[i:i + self.batch_size]
            with self.track_routing_session(f"batch-{i}") as session:
                for request in batch:
                    session.track_multi_model_routing(**request)

# Usage for high-volume multi-model scenarios
adapter = BatchedSkyRouterAdapter(
    batch_size=25,
    team="high-volume-team",
    daily_budget_limit=1000.0
)
```

## API Reference

### GenOpsSkyRouterAdapter

#### Constructor Parameters

```python
GenOpsSkyRouterAdapter(
    skyrouter_api_key: str = None,         # SkyRouter API key
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
def track_routing_session(self, session_name: str, **kwargs) -> SkyRouterSession

# Individual cost calculation methods  
def calculate_model_call_cost(self, model: str, input_data: dict, route_optimization: str = "balanced", complexity: str = "moderate") -> SkyRouterCostResult
def calculate_multi_model_cost(self, models: List[str], input_data: dict, routing_strategy: str = "cost_optimized") -> SkyRouterCostResult
```

### SkyRouterSession

#### Methods

```python
# Track individual operations
def track_model_call(self, model: str, input_data: dict, route_optimization: str = "balanced", cost: float = None) -> SkyRouterCostResult
def track_multi_model_routing(self, models: List[str], input_data: dict, routing_strategy: str = "cost_optimized", cost: float = None) -> SkyRouterCostResult
def track_agent_workflow(self, workflow_name: str, agent_steps: List[dict], cost: float = None) -> SkyRouterCostResult

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
    skyrouter_api_key: str = None,
    team: str = "default", 
    project: str = "default",
    environment: str = "production",
    **kwargs
) -> GenOpsSkyRouterAdapter

# Disable auto-instrumentation
def restore_skyrouter() -> None
```

### Validation

```python
# Validation functions
def validate_setup(skyrouter_api_key: str = None) -> ValidationResult
def print_validation_result(result: ValidationResult, verbose: bool = True) -> None
def validate_setup_interactive() -> ValidationResult
```

---

## 🚀 Next Steps

1. **Try the Examples**: Start with our [interactive examples](../../examples/skyrouter/) to see real-world multi-model patterns
2. **Production Deployment**: Follow our [enterprise deployment guide](../../examples/skyrouter/enterprise_patterns.py)
3. **Route Optimization**: Run the [route optimization example](../../examples/skyrouter/route_optimization.py) for immediate multi-model savings
4. **Join the Community**: Get help in [GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)

**📖 Additional Resources:**
- [Performance Optimization Guide](../skyrouter-performance-benchmarks.md) - Benchmarks, scaling, and optimization
- [Cost Intelligence Guide](../cost-intelligence-guide.md) - ROI calculation and optimization
- [Enterprise Governance Templates](../enterprise-governance-templates.md) - Compliance patterns
- [Production Monitoring Guide](../production-monitoring-guide.md) - Dashboard and alerting setup