# Ollama Integration Guide

**Complete reference for integrating GenOps AI governance with Ollama local model deployments**

This guide provides comprehensive documentation for all GenOps Ollama features, from basic infrastructure cost tracking to advanced production deployment patterns for local models.

## Overview

GenOps provides complete governance for local Ollama deployments including:

- **🖥️ Infrastructure Cost Tracking** - Monitor GPU time, CPU usage, and electricity costs
- **📊 Resource Utilization Monitoring** - Track hardware performance and optimization opportunities  
- **🤖 Model Performance Analytics** - Compare models, latencies, and efficiency metrics
- **🏷️ Team Attribution** - Attribute infrastructure costs to teams, projects, and customers
- **⚡ Hardware Optimization** - Get recommendations to reduce costs and improve performance
- **🛡️ Budget Controls** - Set limits, alerts, and automatic cost enforcement for local deployments
- **📊 OpenTelemetry Integration** - Export to your existing observability stack

## Quick Start

> **🚀 New to GenOps + Ollama?** Start with the [5-Minute Quickstart Guide](../ollama-quickstart.md) for an instant working example, then return here for comprehensive reference.

### Installation

```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama server
ollama serve

# Pull models for testing
ollama pull llama3.2:1b    # Fast, lightweight model
ollama pull llama3.2:3b    # Balanced performance

# Install GenOps with Ollama support
pip install genops-ai[ollama]
```

### Basic Setup

```python
from genops.providers.ollama import auto_instrument
import ollama

# Enable automatic instrumentation for local models
auto_instrument(
    team="ai-team",
    project="local-deployment"
)

# Your existing Ollama code now includes GenOps tracking
response = ollama.generate(
    model="llama3.2:1b",
    prompt="What is GenOps?"
)

# Infrastructure costs, resource usage, and performance automatically tracked
print(f"Response: {response['response']}")
```

## Core Components

### 1. GenOpsOllamaAdapter

The main adapter class for comprehensive Ollama instrumentation with infrastructure cost tracking.

```python
from genops.providers.ollama import instrument_ollama

# Create adapter with governance defaults for local models
adapter = instrument_ollama(
    ollama_base_url="http://localhost:11434",
    team="ai-research",
    project="local-models",
    customer_id="internal-demo",
    # Infrastructure cost rates (customize for your setup)
    gpu_hour_rate=0.50,     # $0.50/hour for GPU usage
    cpu_hour_rate=0.05,     # $0.05/hour for CPU usage  
    electricity_rate=0.12   # $0.12/kWh
)

# Generate text with comprehensive tracking
response = adapter.generate(
    model="llama3.2:3b",
    prompt="Explain machine learning",
    team="ai-research",
    priority="high"
)

# Chat with conversation tracking
response = adapter.chat(
    model="llama3.2:3b", 
    messages=[
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi! How can I help?"},
        {"role": "user", "content": "Explain local AI models"}
    ],
    customer_id="enterprise-123"
)

# List available models with governance
models = adapter.list_models(project="model-discovery")
```

#### Key Methods

- **`generate()`** - Track text generation with infrastructure cost attribution
- **`chat()`** - Track chat interactions with conversation context
- **`list_models()`** - List available models with governance tracking
- **`get_operation_summary()`** - Get comprehensive operation statistics
- **`get_model_metrics()`** - Get model-specific performance metrics

### 2. Resource Monitoring and Optimization

```python
from genops.providers.ollama import get_resource_monitor, create_resource_monitor

# Get global resource monitor
monitor = get_resource_monitor()

# Start monitoring system resources
monitor.start_monitoring()

# Monitor specific inference operations
with monitor.monitor_inference("llama3.2:3b") as inference_data:
    response = ollama.generate(
        model="llama3.2:3b",
        prompt="Complex analysis task"
    )
    inference_data["tokens"] = 150  # Track token count

# Get current system metrics
current_metrics = monitor.get_current_metrics()
print(f"CPU Usage: {current_metrics.cpu_usage_percent:.1f}%")
print(f"GPU Usage: {current_metrics.gpu_usage_percent:.1f}%") 
print(f"Memory Usage: {current_metrics.memory_usage_mb:.0f}MB")

# Get hardware utilization summary
hardware_summary = monitor.get_hardware_summary(duration_minutes=60)
print(f"Average CPU: {hardware_summary.avg_cpu_usage:.1f}%")
print(f"GPU Hours: {hardware_summary.gpu_hours:.2f}") 
print(f"Efficiency Score: {hardware_summary.energy_efficiency_score:.2f}")

# Get optimization recommendations
recommendations = monitor.get_optimization_recommendations()
for rec in recommendations:
    print(f"💡 {rec}")
```

#### ResourceMetrics Class

```python
@dataclass
class ResourceMetrics:
    timestamp: float
    cpu_usage_percent: float
    cpu_temperature: Optional[float]
    memory_usage_mb: float
    memory_available_mb: float
    memory_percent: float
    gpu_usage_percent: float
    gpu_memory_used_mb: float
    gpu_memory_total_mb: float
    gpu_temperature: Optional[float]
    gpu_power_draw_watts: Optional[float]
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
```

### 3. Model Management and Performance Tracking

```python
from genops.providers.ollama import get_model_manager, create_model_manager

# Get global model manager
manager = get_model_manager()

# Discover and catalog available models
models = manager.discover_models()
for model in models:
    print(f"📦 {model.name} ({model.size_gb:.1f}GB, {model.size_category.value})")

# Update model performance after operations
manager.update_model_performance(
    "llama3.2:3b",
    inference_time_ms=2500.0,
    tokens=75,
    memory_mb=4096.0,
    cost=0.002
)

# Get model performance summary
performance = manager.get_model_performance_summary("llama3.2:3b")
print(f"Avg Latency: {performance['avg_inference_latency_ms']:.0f}ms")
print(f"Tokens/Second: {performance['avg_tokens_per_second']:.1f}")
print(f"Cost/Inference: ${performance['cost_per_inference']:.6f}")

# Compare multiple models
comparison = manager.compare_models(
    ["llama3.2:1b", "llama3.2:3b", "mistral:7b"],
    metrics=["avg_inference_latency_ms", "cost_per_inference", "avg_tokens_per_second"]
)

print(f"Fastest Model: {comparison.best_for_speed}")
print(f"Most Cost-Effective: {comparison.best_for_cost}")

# Get optimization recommendations for specific models
recommendations = manager.get_optimization_recommendations("llama3.2:3b")
optimizer = recommendations["llama3.2:3b"]
for opportunity in optimizer.optimization_opportunities:
    print(f"🔧 {opportunity}")

# Get usage analytics
analytics = manager.get_model_usage_analytics(days=7)
print(f"Total Models: {analytics['total_models']}")
print(f"Active Models: {analytics['active_models']}")
print(f"Total Cost: ${analytics['total_cost']:.4f}")
```

#### ModelInfo Class

```python
@dataclass
class ModelInfo:
    name: str
    size_gb: float
    parameter_count: Optional[str]
    family: Optional[str]
    format: Optional[str]
    
    # Performance characteristics
    avg_tokens_per_second: float
    avg_memory_usage_mb: float
    avg_inference_latency_ms: float
    
    # Usage statistics
    total_inferences: int
    total_runtime_hours: float
    last_used: Optional[float]
    
    # Cost efficiency
    cost_per_inference: float
    tokens_per_dollar: float
    
    # Quality metrics
    success_rate: float
    error_count: int
    
    # Model categorization
    size_category: ModelSize  # TINY, SMALL, MEDIUM, LARGE, XLARGE
    model_type: ModelType     # CHAT, CODE, INSTRUCT, EMBEDDING, MULTIMODAL
    
    # Optimization recommendations
    recommended_for: List[str]
    optimization_notes: List[str]
```

### 4. Validation and Diagnostics

```python
from genops.providers.ollama.validation import (
    validate_setup, 
    print_validation_result,
    quick_validate,
    OllamaValidator
)

# Quick validation for CI/CD
if quick_validate():
    print("✅ Ollama setup ready for GenOps")
else:
    print("❌ Setup issues detected")

# Comprehensive validation with detailed output
result = validate_setup(
    ollama_base_url="http://localhost:11434",
    include_performance_tests=True
)

print_validation_result(result, detailed=True)

# Custom validation with specific configuration
validator = OllamaValidator(
    ollama_base_url="http://custom-host:11434",
    timeout=15.0,
    include_performance_tests=True
)

result = validator.validate_all()

# Check validation results
if result.success:
    print(f"✅ Validation passed ({result.score:.1f}%)")
else:
    print(f"❌ Validation failed - {len(result.issues)} issues")
    
    # Show critical issues
    for issue in result.issues:
        if issue.level == ValidationLevel.CRITICAL:
            print(f"🚨 {issue}")
```

#### ValidationResult Class

```python
@dataclass
class ValidationResult:
    success: bool
    total_checks: int
    passed_checks: int
    issues: List[ValidationIssue]
    performance_metrics: Dict[str, float]
    system_info: Dict[str, Any]
    recommendations: List[str]
    
    @property
    def has_critical_issues(self) -> bool
    
    @property
    def score(self) -> float  # 0-100 validation score
```

## Advanced Features

### Infrastructure Cost Attribution

Track the true cost of running local models with detailed attribution:

```python
# Configure cost rates for your infrastructure
adapter = instrument_ollama(
    # Hardware cost rates (customize for your setup)
    gpu_hour_rate=0.75,      # Higher-end GPU
    cpu_hour_rate=0.08,      # Server-grade CPU
    electricity_rate=0.15,   # Regional electricity rate
    
    # Governance attributes
    team="ml-engineering",
    project="production-inference",
    environment="production"
)

# Generate with cost tracking
response = adapter.generate(
    model="llama3.1:8b",  # Larger model = higher cost
    prompt="Comprehensive analysis of quarterly results",
    customer_id="enterprise-client-456",
    priority="high"
)

# Get detailed cost breakdown
summary = adapter.get_operation_summary()
print(f"Infrastructure Cost: ${summary['total_infrastructure_cost']:.6f}")
print(f"GPU Hours Consumed: {summary['total_gpu_hours']:.4f}")
print(f"Average Cost per Operation: ${summary['avg_cost_per_operation']:.6f}")

# Compare infrastructure vs cloud costs
for operation in summary['operations']:
    if operation['infrastructure_cost']:
        print(f"Local: ${operation['infrastructure_cost']:.6f} vs Cloud: ~${operation.get('estimated_cloud_cost', 0.02):.6f}")
```

### Production-Grade Resource Monitoring

```python
from genops.providers.ollama import create_resource_monitor

# Create production resource monitor
monitor = create_resource_monitor(
    monitoring_interval=10.0,    # 10-second intervals
    history_size=10000,          # Keep 10k historical points
    enable_gpu_monitoring=True,
    enable_detailed_metrics=True
)

monitor.start_monitoring()

# Set up continuous monitoring loop
async def production_monitoring_loop():
    while True:
        # Get current resource status
        current = monitor.get_current_metrics()
        
        # Check for resource alerts
        if current.gpu_usage_percent > 90:
            alert_team(f"High GPU usage: {current.gpu_usage_percent:.1f}%")
        
        if current.memory_usage_mb > 16000:  # > 16GB
            alert_team(f"High memory usage: {current.memory_usage_mb:.0f}MB")
        
        # Get optimization recommendations every hour
        recommendations = monitor.get_optimization_recommendations()
        if recommendations:
            log_recommendations(recommendations)
        
        await asyncio.sleep(60)  # Check every minute

# Production context manager for critical operations
with monitor.monitor_inference("production-model") as inference:
    try:
        response = await process_critical_request(user_query)
        inference["tokens"] = count_tokens(response)
        inference["success"] = True
    except Exception as e:
        inference["error"] = str(e)
        inference["success"] = False
        raise
```

### Advanced Model Performance Analysis

```python
# Detailed model performance tracking
manager = get_model_manager()

# Track performance across different scenarios
scenarios = [
    ("simple_qa", "What is the capital of France?"),
    ("complex_analysis", "Analyze the implications of quantum computing on cryptography"),
    ("code_generation", "Write a Python function to implement binary search"),
    ("creative_writing", "Write a short story about AI and humanity")
]

performance_data = {}

for scenario_name, prompt in scenarios:
    with monitor.monitor_inference("llama3.2:8b", scenario_name) as inference:
        start_time = time.time()
        response = ollama.generate(model="llama3.2:8b", prompt=prompt)
        
        inference["tokens"] = response.get('eval_count', 0)
        inference["scenario"] = scenario_name
        inference["complexity"] = len(prompt.split())

# Get detailed performance analysis
model_performance = manager.get_model_performance_summary("llama3.2:8b")

# Analyze performance by scenario
performance_by_scenario = {}
for entry in manager.performance_history["llama3.2:8b"]:
    scenario = entry.get("scenario", "unknown")
    if scenario not in performance_by_scenario:
        performance_by_scenario[scenario] = []
    performance_by_scenario[scenario].append(entry["inference_time_ms"])

# Calculate scenario-specific metrics
for scenario, times in performance_by_scenario.items():
    avg_time = sum(times) / len(times)
    print(f"{scenario}: {avg_time:.0f}ms avg ({len(times)} samples)")

# Get model comparison recommendations
comparison = manager.compare_models(
    ["llama3.2:1b", "llama3.2:3b", "llama3.2:8b"],
    metrics=["avg_inference_latency_ms", "avg_tokens_per_second", "cost_per_inference"]
)

print("\n📊 Model Comparison Results:")
for metric, values in comparison.comparison_metrics.items():
    print(f"\n{metric}:")
    for model, value in values.items():
        print(f"  {model}: {value}")

print(f"\n🏆 Best for speed: {comparison.best_for_speed}")
print(f"💰 Best for cost: {comparison.best_for_cost}")
```

### Auto-Instrumentation Patterns

```python
from genops.providers.ollama import auto_instrument, disable_auto_instrument
from genops.providers.ollama.registration import get_instrumentation_status

# Enable comprehensive auto-instrumentation
success = auto_instrument(
    ollama_base_url="http://localhost:11434",
    resource_monitoring=True,       # Enable resource monitoring
    model_management=True,          # Enable model performance tracking
    
    # Governance defaults applied to all operations
    team="ai-platform",
    project="auto-instrumented-app"
)

if success:
    print("✅ Auto-instrumentation enabled")
    
    # Your existing Ollama code now has comprehensive tracking
    import ollama
    
    # These calls are automatically instrumented
    models = ollama.list()
    response = ollama.generate(model="llama3.2:3b", prompt="Hello world")
    chat_response = ollama.chat(model="llama3.2:3b", messages=[
        {"role": "user", "content": "Hi there!"}
    ])
    
    # Get instrumentation status
    status = get_instrumentation_status()
    print(f"Monitoring active: {status['resource_monitoring_active']}")
    print(f"Models discovered: {status['models_discovered']}")
else:
    print("❌ Auto-instrumentation failed")

# Disable when needed (useful for testing)
disable_auto_instrument()
```

### Context Manager Patterns

```python
from genops.providers.ollama import instrument_ollama

adapter = instrument_ollama()

# Governance context for specific operations
with adapter.governance_context(
    customer_id="premium-customer",
    priority="high",
    cost_center="ml-research"
):
    # All operations in this context inherit these attributes
    response1 = adapter.generate(model="llama3.2:3b", prompt="Query 1")
    response2 = adapter.generate(model="llama3.2:8b", prompt="Query 2")
    
    # Attributes automatically applied:
    # - customer_id="premium-customer"
    # - priority="high" 
    # - cost_center="ml-research"

# Context for cost-controlled operations
from genops.providers.ollama import create_resource_monitor

monitor = create_resource_monitor()

with monitor.monitor_inference("critical-model", "critical-operation") as inference:
    try:
        # Resource monitoring active during this block
        response = ollama.generate(
            model="critical-model",
            prompt="Mission-critical query"
        )
        inference["tokens"] = response.get('eval_count', 0)
        inference["priority"] = "critical"
        
    except Exception as e:
        # Error automatically tracked
        inference["error"] = str(e)
        raise
    
    finally:
        # Metrics automatically recorded
        pass

# Check the results
performance = monitor.get_model_performance("critical-model")
print(f"Critical operations: {performance['critical-model'].total_inferences}")
```

## Configuration and Customization

### Environment Variables

```bash
# Ollama Configuration
export OLLAMA_HOST="http://localhost:11434"      # Ollama server URL
export OLLAMA_MODELS="/path/to/models"           # Model storage path

# GenOps Configuration
export GENOPS_TELEMETRY_ENABLED="true"          # Enable OpenTelemetry export
export GENOPS_COST_TRACKING_ENABLED="true"      # Enable cost calculation
export GENOPS_ENVIRONMENT="production"          # Environment designation
export GENOPS_DEBUG="false"                     # Debug logging

# OpenTelemetry Configuration
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
export OTEL_RESOURCE_ATTRIBUTES="service.name=genops-ollama"
export OTEL_SERVICE_NAME="genops-ollama-service"

# Cost Configuration (optional - can override in code)
export GENOPS_OLLAMA_GPU_HOUR_RATE="0.50"       # GPU cost per hour (USD)
export GENOPS_OLLAMA_CPU_HOUR_RATE="0.05"       # CPU cost per hour (USD)
export GENOPS_OLLAMA_ELECTRICITY_RATE="0.12"    # Electricity cost per kWh (USD)
```

### Custom Configuration

```python
from genops.providers.ollama import GenOpsOllamaAdapter

# Create adapter with comprehensive custom configuration
adapter = GenOpsOllamaAdapter(
    # Connection settings
    ollama_base_url="http://custom-host:11434",
    timeout=30.0,
    
    # Telemetry configuration
    telemetry_enabled=True,
    cost_tracking_enabled=True,
    debug=True,
    
    # Infrastructure cost rates (customize for your setup)
    gpu_hour_rate=0.60,        # $0.60/hour for high-end GPU
    cpu_hour_rate=0.08,        # $0.08/hour for server CPU
    electricity_rate=0.18,     # $0.18/kWh for regional rates
    
    # Governance defaults (applied to all operations)
    team="ml-platform",
    project="custom-deployment", 
    environment="production",
    cost_center="ai-infrastructure",
    
    # Advanced settings
    enable_retry=True,
    max_retries=3,
    retry_delay=1.0
)

# Custom resource monitor configuration
from genops.providers.ollama import create_resource_monitor

monitor = create_resource_monitor(
    monitoring_interval=5.0,      # 5-second monitoring
    history_size=5000,            # Keep 5k data points
    enable_gpu_monitoring=True,
    enable_detailed_metrics=True
)

# Custom model manager configuration
from genops.providers.ollama import create_model_manager

manager = create_model_manager(
    ollama_base_url="http://custom-host:11434",
    enable_auto_optimization=True,
    track_performance_history=True,
    history_size=2000
)
```

## Production Deployment Patterns

### Kubernetes Deployment

```python
from genops.providers.ollama import ProductionOllamaDeployment, ProductionConfig

# Production configuration
config = ProductionConfig(
    # Resource limits
    max_concurrent_requests=20,
    max_memory_usage_mb=32000,      # 32GB limit
    max_gpu_utilization=85.0,       # 85% max GPU usage
    max_cpu_utilization=80.0,       # 80% max CPU usage
    
    # Budget controls
    daily_budget_limit=50.0,        # $50/day infrastructure budget
    hourly_budget_limit=3.0,        # $3/hour infrastructure budget
    cost_alert_threshold=0.80,      # Alert at 80% of budget
    
    # Performance requirements  
    max_response_time_ms=8000.0,    # 8 second timeout
    min_success_rate=0.95,          # 95% success rate requirement
    target_availability=0.999,      # 99.9% uptime target
    
    # Operational settings
    health_check_interval=30,       # 30-second health checks
    metrics_collection_interval=10, # 10-second metrics
    log_level="INFO",
    
    # Auto-scaling
    enable_auto_scaling=True,
    scale_up_threshold=0.70,        # Scale up at 70% utilization
    scale_down_threshold=0.30,      # Scale down at 30% utilization
    
    # Compliance
    enable_audit_logging=True,
    data_retention_days=90,
    enable_request_tracing=True
)

# Initialize production deployment
deployment = ProductionOllamaDeployment(config)
await deployment.initialize()

# Production request handling
async def handle_production_request(customer_id: str, query: str, **metadata):
    async with deployment.track_request(customer_id, "inference", **metadata) as request:
        try:
            response = await deployment.process_request(
                prompt=query,
                customer_id=customer_id,
                timeout=config.max_response_time_ms / 1000
            )
            
            request["success"] = True
            request["tokens"] = response.get('eval_count', 0)
            request["model"] = response.get('model', 'unknown')
            
            return response
            
        except Exception as e:
            request["success"] = False
            request["error"] = str(e)
            raise

# Get production metrics
metrics = deployment.get_production_metrics()
print(f"Uptime: {metrics['deployment']['uptime_seconds']:.0f}s")
print(f"Total Requests: {metrics['deployment']['total_requests']}")
print(f"Total Cost: ${metrics['cost']['total_cost']:.4f}")
print(f"Cost per Request: ${metrics['cost']['cost_per_request']:.6f}")

# Generate Kubernetes manifests
k8s_manifests = deployment.generate_kubernetes_manifests()
with open('ollama-deployment.yaml', 'w') as f:
    f.write(k8s_manifests)
```

### Load Balancing and Health Monitoring

```python
from genops.providers.ollama import ProductionModelLoadBalancer, ModelEndpoint

# Set up load balancer for multiple models
load_balancer = ProductionModelLoadBalancer(config)

# Add model endpoints with priorities
load_balancer.add_endpoint("llama3.2:1b", priority=1, max_requests=10)  # Fastest
load_balancer.add_endpoint("llama3.2:3b", priority=2, max_requests=8)   # Balanced
load_balancer.add_endpoint("llama3.2:8b", priority=3, max_requests=5)   # Most capable

# Start health monitoring
await load_balancer.health_check_loop()

# Production request routing
async def route_request(query: str, complexity_hint: str = "medium"):
    # Select best endpoint based on current load and health
    endpoint = load_balancer.get_best_endpoint(complexity_hint)
    
    if not endpoint:
        raise Exception("No healthy endpoints available")
    
    print(f"Routing to {endpoint.model_name} (health: {endpoint.health_status})")
    
    try:
        response = await ollama.generate(
            model=endpoint.model_name,
            prompt=query
        )
        
        endpoint.success_count += 1
        return response
        
    except Exception as e:
        endpoint.error_count += 1
        if endpoint.error_count > 5:
            endpoint.health_status = "degraded"
        raise

# Monitor endpoint health
for endpoint in load_balancer.endpoints:
    success_rate = endpoint.success_count / max(endpoint.success_count + endpoint.error_count, 1)
    print(f"{endpoint.model_name}: {endpoint.health_status} ({success_rate:.1%} success)")
```

### Monitoring and Alerting Integration

```python
# Integration with monitoring systems
async def setup_monitoring_integrations():
    from genops.providers.ollama import get_resource_monitor, get_model_manager
    
    monitor = get_resource_monitor()
    manager = get_model_manager()
    
    # Start comprehensive monitoring
    monitor.start_monitoring()
    
    async def monitoring_loop():
        while True:
            # Collect current metrics
            current = monitor.get_current_metrics()
            hardware_summary = monitor.get_hardware_summary(duration_minutes=5)
            
            # Check for alerts
            alerts = []
            
            if current.gpu_usage_percent > 90:
                alerts.append({
                    "level": "warning",
                    "message": f"High GPU usage: {current.gpu_usage_percent:.1f}%",
                    "metric": "gpu_utilization",
                    "value": current.gpu_usage_percent,
                    "threshold": 90
                })
            
            if current.memory_usage_mb > 24000:  # > 24GB
                alerts.append({
                    "level": "warning", 
                    "message": f"High memory usage: {current.memory_usage_mb:.0f}MB",
                    "metric": "memory_usage",
                    "value": current.memory_usage_mb,
                    "threshold": 24000
                })
            
            # Check budget alerts
            total_cost = deployment.get_daily_cost()
            if total_cost > config.daily_budget_limit * 0.8:
                alerts.append({
                    "level": "critical",
                    "message": f"Approaching daily budget: ${total_cost:.2f}",
                    "metric": "daily_cost",
                    "value": total_cost,
                    "threshold": config.daily_budget_limit
                })
            
            # Send alerts to monitoring systems
            for alert in alerts:
                await send_alert_to_slack(alert)
                await send_alert_to_pagerduty(alert)
                await send_metric_to_datadog(alert)
            
            # Export metrics to OpenTelemetry
            await export_metrics_to_otel({
                "gpu_usage_percent": current.gpu_usage_percent,
                "memory_usage_mb": current.memory_usage_mb,
                "cpu_usage_percent": current.cpu_usage_percent,
                "daily_cost": total_cost,
                "active_models": len(manager.models),
                "avg_inference_time": hardware_summary.avg_cpu_usage
            })
            
            await asyncio.sleep(60)  # Check every minute
    
    # Start monitoring loop
    asyncio.create_task(monitoring_loop())

# Integration functions
async def send_alert_to_slack(alert):
    """Send alert to Slack channel."""
    pass  # Implement Slack webhook

async def send_alert_to_pagerduty(alert):
    """Send critical alerts to PagerDuty."""
    if alert["level"] == "critical":
        pass  # Implement PagerDuty API call

async def send_metric_to_datadog(alert):
    """Send metrics to Datadog."""
    pass  # Implement Datadog API call

async def export_metrics_to_otel(metrics):
    """Export metrics via OpenTelemetry."""
    from opentelemetry import metrics as otel_metrics
    
    meter = otel_metrics.get_meter(__name__)
    
    # Create and record metrics
    gpu_gauge = meter.create_gauge("ollama.gpu_usage_percent")
    gpu_gauge.set(metrics["gpu_usage_percent"])
    
    memory_gauge = meter.create_gauge("ollama.memory_usage_mb")
    memory_gauge.set(metrics["memory_usage_mb"])
    
    cost_gauge = meter.create_gauge("ollama.daily_cost")
    cost_gauge.set(metrics["daily_cost"])
```

## Testing and Validation

### Setup Validation

```python
from genops.providers.ollama.validation import validate_setup, print_validation_result, OllamaValidator

# Quick validation for development
result = validate_setup()

if result.success:
    print("✅ GenOps Ollama setup is ready!")
    print(f"Score: {result.score:.1f}%")
else:
    print("❌ Setup issues detected")
    print_validation_result(result, detailed=True)

# Custom validation with specific requirements
validator = OllamaValidator(
    ollama_base_url="http://localhost:11434",
    timeout=10.0,
    include_performance_tests=True
)

result = validator.validate_all()

# Check specific validation categories
dependency_issues = [issue for issue in result.issues 
                    if issue.category == ValidationCategory.DEPENDENCIES]

connectivity_issues = [issue for issue in result.issues
                      if issue.category == ValidationCategory.CONNECTIVITY]

model_issues = [issue for issue in result.issues
               if issue.category == ValidationCategory.MODELS]

print(f"Dependencies: {len(dependency_issues)} issues")
print(f"Connectivity: {len(connectivity_issues)} issues") 
print(f"Models: {len(model_issues)} issues")

# Get system information
print(f"System Memory: {result.system_info.get('system_memory_gb', 0):.1f}GB")
print(f"Available Models: {result.system_info.get('available_models_count', 0)}")

# Performance metrics
if result.performance_metrics:
    print(f"Server Response Time: {result.performance_metrics.get('server_response_time_ms', 0):.0f}ms")
    print(f"Test Generation Time: {result.performance_metrics.get('test_generation_time_ms', 0):.0f}ms")
```

### Infrastructure Cost Testing

```python
# Test cost calculations with different scenarios
from genops.providers.ollama import instrument_ollama

adapter = instrument_ollama(
    gpu_hour_rate=0.50,
    cpu_hour_rate=0.05,
    electricity_rate=0.12
)

# Test different model sizes and complexities
test_scenarios = [
    ("llama3.2:1b", "Hello world", "simple"),
    ("llama3.2:3b", "Explain quantum computing in detail", "complex"),
    ("llama3.2:8b", "Write comprehensive analysis with examples", "very_complex")
]

cost_analysis = {}

for model, prompt, complexity in test_scenarios:
    try:
        import time
        start_time = time.time()
        
        response = adapter.generate(
            model=model,
            prompt=prompt,
            test_scenario=complexity
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Get operation details
        operations = adapter.get_operation_summary()
        last_operation = operations['operations'][-1]
        
        cost_analysis[f"{model}_{complexity}"] = {
            "duration_seconds": duration,
            "infrastructure_cost": last_operation.get('infrastructure_cost', 0),
            "gpu_hours": last_operation.get('gpu_hours', 0),
            "cpu_hours": last_operation.get('cpu_hours', 0),
            "tokens": last_operation.get('output_tokens', 0),
            "cost_per_token": last_operation.get('infrastructure_cost', 0) / max(last_operation.get('output_tokens', 1), 1)
        }
        
        print(f"{model} ({complexity}): ${cost_analysis[f'{model}_{complexity}']['infrastructure_cost']:.6f}")
        
    except Exception as e:
        print(f"❌ Failed test for {model}: {e}")

# Analysis of cost efficiency
print("\n📊 Cost Efficiency Analysis:")
for scenario, data in cost_analysis.items():
    print(f"{scenario}:")
    print(f"  Duration: {data['duration_seconds']:.1f}s")
    print(f"  Infrastructure Cost: ${data['infrastructure_cost']:.6f}")
    print(f"  Cost per Token: ${data['cost_per_token']:.8f}")
    print(f"  GPU Hours: {data['gpu_hours']:.6f}")
```

### Performance Testing

```python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

async def performance_test_suite():
    """Comprehensive performance testing."""
    
    adapter = instrument_ollama()
    monitor = get_resource_monitor()
    manager = get_model_manager()
    
    monitor.start_monitoring()
    
    # Test 1: Sequential performance
    print("🧪 Test 1: Sequential Performance")
    sequential_start = time.time()
    
    for i in range(10):
        response = adapter.generate(
            model="llama3.2:3b",
            prompt=f"Test query {i}",
            test_id=f"sequential_{i}"
        )
    
    sequential_duration = time.time() - sequential_start
    print(f"Sequential: {sequential_duration:.2f}s for 10 requests")
    
    # Test 2: Concurrent performance  
    print("🧪 Test 2: Concurrent Performance")
    concurrent_start = time.time()
    
    async def concurrent_request(i):
        return adapter.generate(
            model="llama3.2:3b",
            prompt=f"Concurrent test query {i}",
            test_id=f"concurrent_{i}"
        )
    
    # Run 10 concurrent requests
    tasks = [concurrent_request(i) for i in range(10)]
    await asyncio.gather(*tasks)
    
    concurrent_duration = time.time() - concurrent_start
    print(f"Concurrent: {concurrent_duration:.2f}s for 10 requests")
    print(f"Speedup: {sequential_duration / concurrent_duration:.2f}x")
    
    # Test 3: Resource utilization under load
    print("🧪 Test 3: Resource Utilization Under Load")
    
    baseline_metrics = monitor.get_current_metrics()
    print(f"Baseline - CPU: {baseline_metrics.cpu_usage_percent:.1f}%, "
          f"Memory: {baseline_metrics.memory_usage_mb:.0f}MB")
    
    # High-load test
    load_tasks = [concurrent_request(f"load_{i}") for i in range(50)]
    await asyncio.gather(*load_tasks)
    
    load_metrics = monitor.get_current_metrics()
    print(f"Under Load - CPU: {load_metrics.cpu_usage_percent:.1f}%, "
          f"Memory: {load_metrics.memory_usage_mb:.0f}MB")
    
    # Test 4: Model performance comparison
    print("🧪 Test 4: Model Performance Comparison")
    
    models_to_test = ["llama3.2:1b", "llama3.2:3b"]
    test_prompt = "Explain machine learning briefly"
    
    model_performance = {}
    
    for model in models_to_test:
        start = time.time()
        
        response = adapter.generate(model=model, prompt=test_prompt)
        
        duration = time.time() - start
        model_performance[model] = {
            "duration": duration,
            "tokens": response.get('eval_count', 0),
            "tokens_per_second": response.get('eval_count', 0) / duration if duration > 0 else 0
        }
    
    for model, perf in model_performance.items():
        print(f"{model}: {perf['duration']:.2f}s, {perf['tokens_per_second']:.1f} tokens/sec")
    
    # Test 5: Memory usage patterns
    print("🧪 Test 5: Memory Usage Patterns")
    
    memory_start = monitor.get_current_metrics().memory_usage_mb
    
    # Process different sized prompts
    prompts = [
        "Short query",
        "Medium length query with more details and context",
        "Very long and comprehensive query with extensive context, multiple questions, detailed requirements, and complex reasoning that should stress the model's memory usage patterns significantly"
    ]
    
    for i, prompt in enumerate(prompts):
        response = adapter.generate(model="llama3.2:3b", prompt=prompt)
        current_memory = monitor.get_current_metrics().memory_usage_mb
        memory_delta = current_memory - memory_start
        
        print(f"Prompt {i+1} (len={len(prompt)}): +{memory_delta:.0f}MB memory")
    
    monitor.stop_monitoring()
    
    # Final summary
    summary = adapter.get_operation_summary()
    print(f"\n📊 Test Summary:")
    print(f"Total Operations: {summary['total_operations']}")
    print(f"Success Rate: {summary['success_rate_percent']:.1f}%")
    print(f"Total Cost: ${summary['total_infrastructure_cost']:.6f}")
    print(f"Avg Cost per Op: ${summary['avg_cost_per_operation']:.6f}")

# Run performance tests
await performance_test_suite()
```

## Troubleshooting

### Common Issues

| Issue | Symptoms | Why This Happens | Solution |
|-------|----------|------------------|----------|
| **Connection Refused** | `ConnectionError: Cannot connect to Ollama server` | Ollama server not running | Start Ollama: `ollama serve` |
| **No Models Found** | `[]` from `ollama.list()` | No models downloaded | Pull model: `ollama pull llama3.2:1b` |
| **Import Errors** | `ModuleNotFoundError: No module named 'ollama'` | Ollama client not installed | Install: `pip install ollama` |
| **High Memory Usage** | System running out of RAM | Large models loaded in memory | Use smaller models or increase RAM |
| **Slow Inference** | Very long response times | CPU-only inference or large model | Use GPU or smaller model |
| **GPU Not Detected** | GPU monitoring shows 0% | NVIDIA drivers or CUDA issues | Install NVIDIA drivers, check `nvidia-smi` |
| **Cost Calculation Issues** | Costs showing as 0 | Cost tracking disabled | Enable with `cost_tracking_enabled=True` |
| **Validation Failures** | Setup validation fails | Multiple potential issues | Run detailed validation for diagnosis |

### Detailed Troubleshooting

**Connection Issues**
```python
# Diagnose connection problems
from genops.providers.ollama.validation import validate_setup, print_validation_result

result = validate_setup()
print_validation_result(result, detailed=True)

# Manual connection test
import requests
try:
    response = requests.get("http://localhost:11434/api/version", timeout=5)
    print(f"✅ Server responding: {response.json()}")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("💡 Try: ollama serve")
```

**Model Issues**
```python
# Check available models
import ollama

try:
    models = ollama.list()
    print(f"📦 Available models: {len(models['models'])}")
    for model in models['models']:
        size_gb = model['size'] / (1024**3)
        print(f"  - {model['name']} ({size_gb:.1f}GB)")
except Exception as e:
    print(f"❌ Cannot list models: {e}")
    print("💡 Pull a model: ollama pull llama3.2:1b")
```

**Performance Issues**
```python
# System resource check
def check_system_requirements():
    try:
        import psutil
        
        # Check memory
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        print(f"💾 System Memory: {memory_gb:.1f}GB")
        
        if memory_gb < 8:
            print("⚠️  Low memory - recommend 8GB+ for local models")
        
        # Check CPU
        cpu_count = psutil.cpu_count()
        print(f"🖥️ CPU Cores: {cpu_count}")
        
        if cpu_count < 4:
            print("⚠️  Low CPU count - recommend 4+ cores")
            
        # Check GPU (if available)
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                for gpu in gpus:
                    print(f"🎮 GPU: {gpu.name} ({gpu.memoryTotal}MB)")
            else:
                print("ℹ️  No GPU detected - will use CPU (slower)")
        except ImportError:
            print("ℹ️  GPUtil not installed - cannot check GPU")
            
    except ImportError:
        print("❌ psutil not installed - cannot check system resources")
        print("💡 Install: pip install psutil")

check_system_requirements()
```

**Cost Tracking Issues**
```python
# Verify cost tracking setup
from genops.providers.ollama import instrument_ollama

adapter = instrument_ollama(cost_tracking_enabled=True)

# Test cost calculation
response = adapter.generate(
    model="llama3.2:1b",
    prompt="Test cost tracking"
)

summary = adapter.get_operation_summary()
print(f"Cost tracking enabled: {adapter.cost_tracking_enabled}")
print(f"Total cost: ${summary['total_infrastructure_cost']:.6f}")

if summary['total_infrastructure_cost'] == 0:
    print("❌ Cost tracking not working")
    print("💡 Check cost rates configuration")
    print(f"GPU rate: ${adapter.gpu_hour_rate}/hour")
    print(f"CPU rate: ${adapter.cpu_hour_rate}/hour")
else:
    print("✅ Cost tracking working")
```

**GPU Monitoring Issues**
```python
# Check GPU monitoring capabilities
from genops.providers.ollama.resource_monitor import HAS_GPUTIL, HAS_PYNVML

print(f"GPUtil available: {HAS_GPUTIL}")
print(f"PyNVML available: {HAS_PYNVML}")

if not HAS_GPUTIL and not HAS_PYNVML:
    print("❌ No GPU monitoring libraries available")
    print("💡 Install: pip install gputil pynvml")

# Test GPU monitoring
from genops.providers.ollama import get_resource_monitor

monitor = get_resource_monitor()
current_metrics = monitor.get_current_metrics()

print(f"GPU Usage: {current_metrics.gpu_usage_percent:.1f}%")
print(f"GPU Memory: {current_metrics.gpu_memory_used_mb:.0f}MB")

if current_metrics.gpu_usage_percent == 0:
    print("⚠️  GPU not detected or not in use")
    print("💡 Check: nvidia-smi")
```

### Performance Optimization

**Model Selection Optimization**
```python
# Optimize model selection based on use case
def recommend_model_for_task(task_type: str, performance_priority: str = "balanced"):
    """
    Recommend optimal model based on task and performance requirements.
    
    Args:
        task_type: "simple_qa", "complex_analysis", "code_generation", "creative_writing"
        performance_priority: "speed", "quality", "balanced", "cost"
    """
    
    recommendations = {
        "simple_qa": {
            "speed": "llama3.2:1b",
            "quality": "llama3.2:3b", 
            "balanced": "llama3.2:1b",
            "cost": "llama3.2:1b"
        },
        "complex_analysis": {
            "speed": "llama3.2:3b",
            "quality": "llama3.2:8b",
            "balanced": "llama3.2:3b", 
            "cost": "llama3.2:3b"
        },
        "code_generation": {
            "speed": "codellama:7b",
            "quality": "codellama:13b",
            "balanced": "codellama:7b",
            "cost": "codellama:7b"
        },
        "creative_writing": {
            "speed": "llama3.2:3b",
            "quality": "llama3.2:8b", 
            "balanced": "llama3.2:8b",
            "cost": "llama3.2:3b"
        }
    }
    
    return recommendations.get(task_type, {}).get(performance_priority, "llama3.2:3b")

# Example usage
model = recommend_model_for_task("code_generation", "speed")
print(f"Recommended model: {model}")
```

**Hardware Optimization**
```python
# Get hardware optimization recommendations
from genops.providers.ollama import get_resource_monitor, get_model_manager

monitor = get_resource_monitor()
manager = get_model_manager()

# Get optimization recommendations
recommendations = monitor.get_optimization_recommendations()
model_recommendations = manager.get_optimization_recommendations()

print("🔧 System Optimization Recommendations:")
for rec in recommendations:
    print(f"  • {rec}")

print("\n🤖 Model-Specific Recommendations:")
for model, optimizer in model_recommendations.items():
    if optimizer.optimization_opportunities:
        print(f"  {model}:")
        for opp in optimizer.optimization_opportunities:
            print(f"    • {opp}")

# Hardware utilization analysis
hardware_summary = monitor.get_hardware_summary(duration_minutes=60)

print(f"\n📊 Hardware Utilization (last hour):")
print(f"Average CPU: {hardware_summary.avg_cpu_usage:.1f}%")
print(f"Average GPU: {hardware_summary.avg_gpu_usage:.1f}%")
print(f"Max Memory: {hardware_summary.max_memory_usage_mb:.0f}MB")
print(f"GPU Hours: {hardware_summary.gpu_hours:.2f}")
print(f"Efficiency Score: {hardware_summary.energy_efficiency_score:.2f}")

# Provide actionable recommendations
if hardware_summary.avg_gpu_usage < 30:
    print("💡 GPU underutilized - consider larger models or batch processing")
elif hardware_summary.avg_gpu_usage > 90:
    print("⚠️ GPU overutilized - consider model optimization or scaling")

if hardware_summary.max_memory_usage_mb > 24000:  # >24GB
    print("⚠️ High memory usage - consider memory optimization")
```

## API Reference

### Main Classes

#### GenOpsOllamaAdapter

Main adapter for Ollama integration with comprehensive instrumentation.

```python
class GenOpsOllamaAdapter(BaseFrameworkProvider):
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        telemetry_enabled: bool = True,
        cost_tracking_enabled: bool = True,
        debug: bool = False,
        gpu_hour_rate: float = 0.50,
        cpu_hour_rate: float = 0.05,
        electricity_rate: float = 0.12,
        **governance_defaults
    )
    
    def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]
    
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]
    
    def list_models(self, **governance_attrs) -> List[Dict[str, Any]]
    
    def get_operation_summary(self) -> Dict[str, Any]
    def get_model_metrics(self, model: Optional[str] = None) -> Union[LocalModelMetrics, Dict[str, LocalModelMetrics]]
    
    @contextmanager
    def governance_context(self, **attributes)
```

#### OllamaResourceMonitor

Comprehensive resource monitoring for local Ollama deployments.

```python
class OllamaResourceMonitor:
    def __init__(
        self,
        monitoring_interval: float = 1.0,
        history_size: int = 1000,
        enable_gpu_monitoring: bool = True,
        enable_detailed_metrics: bool = True
    )
    
    def start_monitoring(self)
    def stop_monitoring(self)
    def get_current_metrics(self) -> Optional[ResourceMetrics]
    def get_hardware_summary(self, duration_minutes: int = 60) -> HardwareMetrics
    def get_optimization_recommendations(self) -> List[str]
    
    @contextmanager
    def monitor_inference(self, model_name: str, operation_id: str = None)
```

#### OllamaModelManager

Model lifecycle management and performance optimization.

```python
class OllamaModelManager:
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        enable_auto_optimization: bool = True,
        track_performance_history: bool = True,
        history_size: int = 1000
    )
    
    def discover_models(self) -> List[ModelInfo]
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]
    def update_model_performance(self, model_name: str, **performance_data)
    def compare_models(self, model_names: List[str], metrics: List[str] = None) -> ModelComparison
    def get_optimization_recommendations(self, model_name: str = None) -> Dict[str, ModelOptimizer]
    def get_model_usage_analytics(self, days: int = 30) -> Dict[str, Any]
    def export_model_data(self, format: str = "json") -> str
```

#### OllamaValidator

Comprehensive validation system for setup diagnostics.

```python
class OllamaValidator:
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        timeout: float = 10.0,
        include_performance_tests: bool = True
    )
    
    def validate_all(self) -> ValidationResult
```

### Factory Functions

```python
def instrument_ollama(
    ollama_base_url: str = "http://localhost:11434",
    telemetry_enabled: bool = True,
    cost_tracking_enabled: bool = True,
    **governance_defaults
) -> GenOpsOllamaAdapter

def auto_instrument(
    ollama_base_url: str = "http://localhost:11434",
    resource_monitoring: bool = True,
    model_management: bool = True,
    **governance_defaults
) -> bool

def validate_setup(ollama_base_url: str = "http://localhost:11434", **kwargs) -> ValidationResult
def quick_validate(ollama_base_url: str = "http://localhost:11434") -> bool
def print_validation_result(result: ValidationResult, detailed: bool = False)

def get_resource_monitor() -> OllamaResourceMonitor
def get_model_manager() -> OllamaModelManager
def create_resource_monitor(**kwargs) -> OllamaResourceMonitor
def create_model_manager(**kwargs) -> OllamaModelManager
```

### Data Classes

```python
@dataclass
class OllamaOperation:
    operation_id: str
    operation_type: str  # 'generate', 'chat', 'list_models'
    model: str
    start_time: float
    end_time: Optional[float] = None
    prompt: Optional[str] = None
    response: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    inference_time_ms: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    infrastructure_cost: Optional[float] = None
    gpu_hours: Optional[float] = None
    cpu_hours: Optional[float] = None
    governance_attributes: Optional[Dict[str, Any]] = None

@dataclass
class LocalModelMetrics:
    model_name: str
    total_operations: int
    total_inference_time_ms: float
    avg_gpu_memory_mb: float = 0.0
    avg_cpu_usage_percent: float = 0.0
    avg_inference_latency_ms: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    avg_tokens_per_second: float = 0.0
    total_infrastructure_cost: float = 0.0
    cost_per_operation: float = 0.0
    gpu_hours_consumed: float = 0.0
    success_rate: float = 100.0
    error_count: int = 0
    tokens_per_gpu_hour: float = 0.0
    operations_per_dollar: float = 0.0

@dataclass
class ResourceMetrics:
    timestamp: float
    cpu_usage_percent: float = 0.0
    cpu_temperature: Optional[float] = None
    memory_usage_mb: float = 0.0
    memory_available_mb: float = 0.0
    memory_percent: float = 0.0
    gpu_usage_percent: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_temperature: Optional[float] = None
    gpu_power_draw_watts: Optional[float] = None
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0
```

## Examples

Complete working examples are available in the [`examples/ollama/`](../../examples/ollama/) directory:

- **`hello_ollama_minimal.py`** - 30-second quickstart and confidence builder
- **`local_model_optimization.py`** - Cost optimization and performance analysis
- **`ollama_production_deployment.py`** - Enterprise deployment patterns

## Support and Community

- **Documentation**: [GenOps AI Docs](https://docs.genops.ai)
- **Examples**: [GitHub Examples](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/ollama)
- **Issues**: [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- **Discussions**: [Community Forum](https://github.com/KoshiHQ/GenOps-AI/discussions)

---

## 📚 Navigation & Next Steps

**🎯 Getting Started:**
- **[5-Minute Quickstart](../ollama-quickstart.md)** - Copy-paste examples to get running immediately
- **[Examples Directory](../../examples/ollama/)** - Step-by-step practical tutorials with clear progression

**🏗️ Production Deployment:**
- **[Security Best Practices](../security-best-practices.md)** - Enterprise security, compliance, and infrastructure management
- **[CI/CD Integration Guide](../ci-cd-integration.md)** - Automated testing, deployment pipelines, and infrastructure monitoring

**🤝 Community & Support:**
- **[GitHub Repository](https://github.com/KoshiHQ/GenOps-AI)** - Source code and latest updates
- **[Community Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)** - Questions, ideas, and community help
- **[Issue Tracker](https://github.com/KoshiHQ/GenOps-AI/issues)** - Bug reports and feature requests

**Ready to implement production-grade governance for your local Ollama models? Start with the [quickstart guide](../ollama-quickstart.md) or jump into the [examples](../../examples/ollama/)!**