# Griptape Integration Guide

**Complete integration guide for GenOps governance with Griptape AI framework across all structure types and deployment patterns.**

## Overview

Griptape is a modular Python framework for AI agents and workflows with chain-of-thought reasoning, tools, and memory. GenOps provides comprehensive governance integration supporting:

- **All Structure Types**: Agents, Pipelines, Workflows with unified tracking
- **Multiple LLM Providers**: OpenAI, Anthropic, Google, Cohere, Mistral, and more
- **Advanced Engines**: RAG, Extraction, Summary, Evaluation with cost attribution
- **Memory Systems**: Conversation, Task, and Meta Memory governance
- **Production Ready**: Enterprise deployment, scaling, monitoring

## Quick Links

- **[5-Minute Quickstart](../griptape-quickstart.md)** - Get started immediately
- **[Examples Suite](../../examples/griptape/)** - 8 progressive examples
- **[API Reference](#api-reference)** - Complete API documentation

## Installation & Setup

### Prerequisites

- **Python 3.9+**: Required for GenOps integration
- **Griptape Framework**: AI agent and workflow framework
- **LLM Provider Keys**: At least one supported provider API key

### Core Installation

```bash
# Python dependencies
pip install genops griptape

# Optional: Enhanced observability
pip install prometheus-client grafana-client

# Optional: Additional LLM providers
pip install openai anthropic google-generativeai cohere mistralai
```

### Environment Configuration

```bash
# GenOps governance (required)
export GENOPS_TEAM="your-team"
export GENOPS_PROJECT="griptape-demo"
export GENOPS_ENVIRONMENT="development"  # or staging/production

# LLM provider API keys (at least one required)
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"

# Optional: Advanced configuration
export GENOPS_COST_CENTER="ai-department"
export GENOPS_CUSTOMER_ID="customer-123"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://your-collector:4317"
```

### Validation

```bash
# Quick validation
python -c "from genops.providers.griptape.registration import validate_griptape_setup; print('✅ Ready!' if validate_griptape_setup()['griptape_available'] else '❌ Setup issues detected')"

# Comprehensive validation
python -c "from genops.providers.griptape.registration import validate_griptape_setup; import pprint; pprint.pprint(validate_griptape_setup())"
```

## Integration Patterns

### 1. Auto-Instrumentation Pattern (Recommended)

**Best for**: Existing applications, zero code changes required

```python
# Enable GenOps governance with one import
from genops.providers.griptape import auto_instrument

adapter = auto_instrument(
    team="your-team",
    project="griptape-demo",
    environment="development"
)

# Your existing Griptape code works unchanged!
from griptape.structures import Agent, Pipeline, Workflow
from griptape.tasks import PromptTask, TextSummaryTask

agent = Agent(tasks=[PromptTask("Analyze this data")])
result = agent.run("Input data")
# ✅ Automatic governance tracking added
```

### 2. Manual Instrumentation Pattern

**Best for**: Controlled governance, specific structure tracking

```python
from genops.providers.griptape import instrument_griptape

# Create instrumented wrapper
griptape = instrument_griptape(
    team="ai-research",
    project="analysis-pipeline",
    daily_budget_limit=100.0
)

# Use instrumented structures
agent = griptape.create_agent([PromptTask("Research task")])
pipeline = griptape.create_pipeline([task1, task2, task3])
workflow = griptape.create_workflow([[task1, task2], [task3]])

# Explicit execution tracking
result = agent.run("Research question")
```

### 3. Context Manager Pattern

**Best for**: Fine-grained control, custom governance

```python
from genops.providers.griptape import GenOpsGriptapeAdapter

adapter = GenOpsGriptapeAdapter(
    team="ai-team",
    project="custom-workflow"
)

# Track specific operations
with adapter.track_agent("research-agent") as request:
    agent = Agent(tasks=[PromptTask("Research AI governance")])
    result = agent.run("Input query")
    
    # Manual cost attribution (if needed)
    request.add_provider_cost("openai", "gpt-4", 0.002)
    print(f"Total cost: ${request.total_cost:.6f}")
```

## Structure Type Integration

### Agent Governance

```python
from genops.providers.griptape import auto_instrument
from griptape.structures import Agent
from griptape.tasks import PromptTask
from griptape.rules import Rule

# Enable governance
auto_instrument(team="ai-team", project="agents")

# Create Agent with governance
agent = Agent(
    tasks=[
        PromptTask(
            prompt="Analyze market trends and provide insights",
            rules=[
                Rule("Provide data-driven analysis"),
                Rule("Include confidence levels for predictions")
            ]
        )
    ]
)

# Execute with automatic tracking
result = agent.run("Quarterly market data: ...")
# ✅ Cost, performance, and governance automatically tracked
```

### Pipeline Governance

```python
from griptape.structures import Pipeline
from griptape.tasks import PromptTask, TextSummaryTask

# Sequential task pipeline
pipeline = Pipeline(
    tasks=[
        PromptTask(
            id="analyze",
            prompt="Analyze this data: {{ input }}"
        ),
        PromptTask(
            id="summarize", 
            prompt="Summarize the analysis: {{ parent_output }}"
        ),
        TextSummaryTask(
            id="final_summary"
        )
    ]
)

# Execute pipeline with governance
with adapter.track_pipeline("analysis-pipeline") as request:
    result = pipeline.run({"input": "Complex dataset..."})
    print(f"Pipeline cost: ${request.total_cost:.6f}")
    print(f"Tasks completed: {request.completed_tasks}")
```

### Workflow Governance

```python
from griptape.structures import Workflow
from griptape.tasks import PromptTask

# Parallel workflow with governance
workflow = Workflow(
    tasks=[
        # Parallel execution group 1
        [
            PromptTask(
                id="sentiment_analysis",
                prompt="Analyze sentiment: {{ input }}"
            ),
            PromptTask(
                id="topic_extraction", 
                prompt="Extract key topics: {{ input }}"
            )
        ],
        # Sequential task after parallel completion
        [
            PromptTask(
                id="synthesis",
                prompt="Synthesize findings: {{ sentiment_analysis.output }} and {{ topic_extraction.output }}"
            )
        ]
    ]
)

# Execute with governance tracking
with adapter.track_workflow("content-analysis") as request:
    result = workflow.run({"input": "Customer feedback data..."})
    print(f"Workflow cost: ${request.total_cost:.6f}")
    print(f"Parallel tasks: {request.parallel_tasks}")
```

## Engine Integration

### RAG Engine Governance

```python
from griptape.engines import RagEngine
from griptape.drivers.vector import ChromaVectorStoreDriver
from griptape.drivers.embedding import OpenAiEmbeddingDriver

# Create RAG engine with governance
with adapter.track_engine("document-rag", "rag") as request:
    rag_engine = RagEngine(
        vector_store_driver=ChromaVectorStoreDriver(),
        embedding_driver=OpenAiEmbeddingDriver()
    )
    
    # Process query with cost tracking
    response = rag_engine.process("What are the main findings in the research?")
    
    # Track RAG-specific metrics
    request.reasoning_steps += 3  # Query, retrieval, generation
    request.memory_operations += 1  # Vector search
```

### Extraction Engine Governance

```python
from griptape.engines import ExtractionEngine
from griptape.tasks import ExtractionTask

# Data extraction with governance
with adapter.track_engine("data-extraction", "extraction") as request:
    extraction_engine = ExtractionEngine()
    
    # Extract structured data
    result = extraction_engine.extract(
        "Extract company names, locations, and revenue from this text",
        "Apple Inc. in Cupertino reported $365B revenue..."
    )
    
    # Track extraction metrics
    request.add_task_completion(success=bool(result))
```

### Summary Engine Governance

```python
from griptape.engines import SummaryEngine

# Document summarization with governance
with adapter.track_engine("doc-summary", "summary") as request:
    summary_engine = SummaryEngine()
    
    # Generate summary with cost tracking
    summary = summary_engine.summarize("Long document content...")
    
    # Manual cost attribution if needed
    if hasattr(summary, 'usage'):
        request.add_provider_cost(
            "openai", "gpt-3.5-turbo",
            adapter.cost_aggregator.calculate_cost(
                "openai", "gpt-3.5-turbo",
                summary.usage.prompt_tokens,
                summary.usage.completion_tokens
            )["total_cost"]
        )
```

## Memory System Governance

### Conversation Memory

```python
from griptape.memory import ConversationMemory
from griptape.structures import Agent

# Agent with conversation memory governance
agent = Agent(
    memory=ConversationMemory(),
    tasks=[PromptTask("Continue our conversation about {{ input }}")]
)

# Track memory operations
with adapter.track_memory("conversation-1", "conversation") as request:
    # First interaction
    result1 = agent.run("AI ethics principles")
    request.memory_operations += 1  # Store conversation
    
    # Follow-up with memory context
    result2 = agent.run("How do these apply to healthcare?")
    request.memory_operations += 2  # Retrieve + store
```

### Task Memory

```python
from griptape.memory import TaskMemory

# Pipeline with task memory governance  
pipeline = Pipeline(
    memory=TaskMemory(),
    tasks=[
        PromptTask(
            id="analysis",
            prompt="Analyze data and store findings",
            memory=TaskMemory()
        ),
        PromptTask(
            id="report",
            prompt="Generate report using stored analysis"
        )
    ]
)

with adapter.track_pipeline("analysis-with-memory") as request:
    result = pipeline.run({"data": "Large dataset..."})
    # Memory operations tracked automatically
```

## Multi-Provider Configuration

### Provider Setup

```python
from griptape.drivers import OpenAiChatPromptDriver, AnthropicPromptDriver, GooglePromptDriver

# Configure multiple providers with governance
openai_driver = OpenAiChatPromptDriver(model="gpt-4")
anthropic_driver = AnthropicPromptDriver(model="claude-3-opus")
google_driver = GooglePromptDriver(model="gemini-pro")

# GenOps automatically tracks all providers
agents = {
    "fast": Agent(
        prompt_driver=openai_driver,
        tasks=[PromptTask("Quick analysis")]
    ),
    "thorough": Agent(
        prompt_driver=anthropic_driver, 
        tasks=[PromptTask("Detailed analysis")]
    ),
    "multimodal": Agent(
        prompt_driver=google_driver,
        tasks=[PromptTask("Image and text analysis")]
    )
}
```

### Cost Optimization Patterns

```python
from genops.providers.griptape.cost_aggregator import GriptapeCostAggregator

def select_optimal_agent(query_complexity: str, budget_limit: float):
    """Select most cost-effective agent for task complexity."""
    
    cost_aggregator = GriptapeCostAggregator()
    
    # Estimate costs for different agents
    providers = [
        ("openai", "gpt-3.5-turbo", "fast"),
        ("anthropic", "claude-3-haiku", "balanced"), 
        ("openai", "gpt-4", "thorough")
    ]
    
    for provider, model, agent_type in providers:
        # Estimate cost based on query complexity
        estimated_tokens = {"low": 500, "medium": 1500, "high": 3000}[query_complexity]
        
        cost_breakdown = cost_aggregator.calculate_cost(
            provider, model, estimated_tokens, estimated_tokens // 2
        )
        
        if cost_breakdown["total_cost"] <= budget_limit:
            return agents[agent_type], provider, model
    
    raise ValueError("No agent within budget limit")

# Use cost-optimized selection
agent, provider, model = select_optimal_agent("medium", 0.05)
with adapter.track_agent(f"optimized-{provider}") as request:
    result = agent.run("Analysis query")
```

## Production Deployment

### Docker Integration

```dockerfile
# Multi-stage build for Griptape + GenOps
FROM python:3.11-slim AS base
WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Environment variables for production
ENV GENOPS_TEAM=production
ENV GENOPS_ENVIRONMENT=production
ENV OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:14268/api/traces
ENV GRIPTAPE_PROMPT_DRIVER=openai

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "from genops.providers.griptape.registration import validate_griptape_setup; exit(0 if validate_griptape_setup()['griptape_available'] else 1)"

CMD ["python", "app.py"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: griptape-ai-app
  labels:
    app: griptape-ai-app
    genops.ai/instrumented: "true"
spec:
  replicas: 3
  selector:
    matchLabels:
      app: griptape-ai-app
  template:
    metadata:
      labels:
        app: griptape-ai-app
    spec:
      containers:
      - name: app
        image: your-registry/griptape-app:latest
        env:
        # GenOps Configuration
        - name: GENOPS_TEAM
          value: "production"
        - name: GENOPS_PROJECT
          value: "ai-agents"
        - name: GENOPS_ENVIRONMENT
          value: "production"
        
        # OpenTelemetry Configuration
        - name: OTEL_EXPORTER_OTLP_ENDPOINT
          value: "http://jaeger-collector:14268/api/traces"
        - name: OTEL_SERVICE_NAME
          value: "griptape-ai-service"
        
        # LLM Provider Keys (from secrets)
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-provider-keys
              key: openai-api-key
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-provider-keys
              key: anthropic-api-key
        
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        
        # Health checks
        livenessProbe:
          exec:
            command:
            - python
            - -c
            - "from genops.providers.griptape.registration import validate_griptape_setup; exit(0 if validate_griptape_setup()['griptape_available'] else 1)"
          initialDelaySeconds: 30
          periodSeconds: 30
        
        readinessProbe:
          exec:
            command:
            - python
            - -c
            - "from genops.providers.griptape.registration import is_instrumented; exit(0 if is_instrumented() else 1)"
          initialDelaySeconds: 5
          periodSeconds: 10

---
apiVersion: v1
kind: Secret
metadata:
  name: llm-provider-keys
type: Opaque
data:
  openai-api-key: <base64-encoded-key>
  anthropic-api-key: <base64-encoded-key>
  google-api-key: <base64-encoded-key>
```

### CI/CD Integration

```yaml
# GitHub Actions workflow
name: Deploy Griptape AI App

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install genops griptape pytest
    
    - name: Validate Griptape integration
      run: |
        python -c "
        from genops.providers.griptape.registration import validate_griptape_setup
        result = validate_griptape_setup()
        assert result['griptape_available'], 'Griptape not available'
        assert len(result['supported_structures']) > 0, 'No supported structures'
        print('✅ Griptape validation passed')
        "
      env:
        GENOPS_TEAM: ci-testing
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    
    - name: Run tests with governance
      run: |
        # Tests automatically include GenOps telemetry
        python -m pytest tests/ -v
        python examples/griptape/01_basic_agent.py
    
    - name: Build and deploy
      run: |
        docker build -t griptape-app:${{ github.sha }} .
        docker push your-registry/griptape-app:${{ github.sha }}
```

## Performance & Scaling

### Performance Characteristics

- **Telemetry Overhead**: <3ms per structure execution
- **Memory Usage**: ~15MB for adapter with monitoring  
- **Network Overhead**: Batched OTLP export (configurable)
- **CPU Impact**: Minimal (<1% additional CPU usage)

### High-Volume Configuration

```python
# Optimized for high-volume applications
adapter = GenOpsGriptapeAdapter(
    # Reduce monitoring overhead
    enable_performance_monitoring=False,
    
    # Sample for high-volume (20% sampling)
    sampling_rate=0.2,
    
    # Async telemetry export
    async_export=True,
    
    # Budget-based throttling
    daily_budget_limit=1000.0
)

# Configure OpenTelemetry sampling
import os
os.environ['OTEL_TRACES_SAMPLER'] = 'traceidratio'
os.environ['OTEL_TRACES_SAMPLER_ARG'] = '0.2'  # 20% sampling
```

### Scaling Patterns

```python
# Circuit breaker for external dependencies
from genops.providers.griptape import GenOpsGriptapeAdapter

class ResilientAdapter(GenOpsGriptapeAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.failure_count = 0
        self.circuit_open = False
        self.last_failure_time = 0
        
    def track_agent(self, *args, **kwargs):
        # Circuit breaker logic
        if self.circuit_open:
            import time
            if time.time() - self.last_failure_time > 60:  # 1 minute reset
                self.circuit_open = False
                self.failure_count = 0
            else:
                # Return minimal tracking context
                return self.minimal_tracking_context(*args, **kwargs)
        
        try:
            return super().track_agent(*args, **kwargs)
        except Exception as e:
            self.failure_count += 1
            if self.failure_count >= 5:
                self.circuit_open = True
                self.last_failure_time = time.time()
            raise
```

## Monitoring & Observability

### Dashboard Integration

#### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "GenOps Griptape AI Monitoring",
    "panels": [
      {
        "title": "Structure Execution Rate",
        "targets": [
          {
            "expr": "rate(genops_griptape_requests_total[5m])",
            "legendFormat": "{{structure_type}} - {{team}}"
          }
        ]
      },
      {
        "title": "Cost per Hour by Provider",
        "targets": [
          {
            "expr": "increase(genops_cost_total{provider=~\"openai|anthropic|google\"}[1h])",
            "legendFormat": "{{provider}} - {{project}}"
          }
        ]
      },
      {
        "title": "Structure Success Rate",
        "targets": [
          {
            "expr": "rate(genops_griptape_requests_total{status=\"completed\"}[5m]) / rate(genops_griptape_requests_total[5m]) * 100",
            "legendFormat": "{{structure_type}}"
          }
        ]
      },
      {
        "title": "Memory Operations",
        "targets": [
          {
            "expr": "genops_memory_operations_total",
            "legendFormat": "{{memory_type}} - {{operation}}"
          }
        ]
      }
    ]
  }
}
```

#### Datadog Integration

```python
# Custom Datadog metrics for Griptape
from datadog import initialize, statsd

def send_griptape_metrics(request_data):
    """Send custom metrics to Datadog."""
    tags = [
        f"team:{request_data.governance_attrs['team']}",
        f"structure_type:{request_data.structure_type}",
        f"environment:{request_data.governance_attrs.get('environment', 'unknown')}"
    ]
    
    # Structure execution metrics
    statsd.increment('griptape.executions.count', tags=tags)
    statsd.histogram('griptape.duration', request_data.duration or 0, tags=tags)
    
    # Cost metrics
    statsd.histogram('griptape.cost.total', float(request_data.total_cost), tags=tags)
    
    # Provider-specific costs
    for provider, cost in request_data.provider_costs.items():
        provider_tags = tags + [f"provider:{provider}"]
        statsd.histogram('griptape.cost.by_provider', float(cost), tags=provider_tags)
    
    # Task completion metrics
    if request_data.task_count > 0:
        success_rate = (request_data.completed_tasks / request_data.task_count) * 100
        statsd.histogram('griptape.success_rate', success_rate, tags=tags)
```

### Alerting Rules

```yaml
# Prometheus alerting rules
groups:
- name: genops_griptape
  rules:
  - alert: HighGriptapeCost
    expr: increase(genops_cost_total[1h]) > 50
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High Griptape AI costs detected"
      description: "Griptape costs exceeded $50/hour for team {{ $labels.team }}"
  
  - alert: GriptapeStructureFailures
    expr: rate(genops_griptape_requests_total{status="failed"}[5m]) > 0.1
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High Griptape structure failure rate"
      description: "Structure failure rate is {{ $value }} for {{ $labels.structure_type }}"
  
  - alert: GriptapeBudgetExceeded
    expr: genops_daily_spending > genops_budget_limit
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Daily budget limit exceeded"
      description: "Team {{ $labels.team }} exceeded daily budget of ${{ $labels.budget_limit }}"
```

## API Reference

### GenOpsGriptapeAdapter

```python
class GenOpsGriptapeAdapter:
    def __init__(
        self,
        team: Optional[str] = None,
        project: Optional[str] = None,
        environment: Optional[str] = None,
        cost_center: Optional[str] = None,
        customer_id: Optional[str] = None,
        feature: Optional[str] = None,
        daily_budget_limit: Optional[float] = None,
        enable_cost_tracking: bool = True,
        enable_performance_monitoring: bool = True,
        sampling_rate: float = 1.0,
        **kwargs
    ):
        """Initialize Griptape adapter with governance configuration."""
    
    def track_agent(self, agent_id: str, **kwargs) -> ContextManager[GriptapeRequest]:
        """Context manager for tracking Agent execution."""
    
    def track_pipeline(self, pipeline_id: str, **kwargs) -> ContextManager[GriptapeRequest]:
        """Context manager for tracking Pipeline execution."""
    
    def track_workflow(self, workflow_id: str, **kwargs) -> ContextManager[GriptapeRequest]:
        """Context manager for tracking Workflow execution."""
    
    def track_engine(self, engine_id: str, engine_type: str = "generic", **kwargs) -> ContextManager[GriptapeRequest]:
        """Context manager for tracking Engine operations."""
    
    def track_memory(self, memory_id: str, operation_type: str = "access", **kwargs) -> ContextManager[GriptapeRequest]:
        """Context manager for tracking Memory operations."""
```

### Auto-Instrumentation Functions

```python
def auto_instrument(
    team: Optional[str] = None,
    project: Optional[str] = None,
    environment: Optional[str] = None,
    daily_budget_limit: Optional[float] = None,
    **kwargs
) -> GenOpsGriptapeAdapter:
    """Enable automatic instrumentation for all Griptape structures."""

def instrument_griptape(
    team: Optional[str] = None,
    project: Optional[str] = None,
    **kwargs
) -> InstrumentedGriptape:
    """Create manually instrumented Griptape wrapper."""

def disable_auto_instrument() -> None:
    """Disable automatic instrumentation."""

def is_instrumented() -> bool:
    """Check if auto-instrumentation is enabled."""
```

### Cost Aggregation Functions

```python
class GriptapeCostAggregator:
    def add_structure_cost(
        self,
        structure_id: str,
        structure_type: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        **kwargs
    ) -> GriptapeCostBreakdown:
        """Add cost tracking for a structure operation."""
    
    def get_cost_summary(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        structure_type: Optional[str] = None,
        **kwargs
    ) -> GriptapeCostSummary:
        """Get aggregated cost summary with filtering."""
    
    def get_daily_costs(self, date: Optional[datetime] = None) -> Decimal:
        """Get total costs for a specific day."""
```

### Workflow Monitoring Functions

```python
class GriptapeWorkflowMonitor:
    def start_structure_monitoring(self, request_id: str, structure_type: str) -> None:
        """Start monitoring a structure execution."""
    
    def stop_structure_monitoring(self, request_id: str) -> Optional[GriptapeStructureMetrics]:
        """Stop monitoring and return metrics."""
    
    def get_performance_insights(
        self,
        structure_type: Optional[str] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """Get performance insights and optimization recommendations."""
```

## Advanced Use Cases

### Multi-Tenant SaaS

```python
# Customer-specific governance
def create_customer_adapter(customer_id: str, tier: str):
    budget_limits = {"basic": 10.0, "premium": 50.0, "enterprise": 200.0}
    
    return GenOpsGriptapeAdapter(
        team=f"customer-{customer_id}",
        project="saas-platform",
        customer_id=customer_id,
        cost_center=f"customer-revenue-{tier}",
        daily_budget_limit=budget_limits.get(tier, 10.0),
        
        # Tier-specific sampling
        sampling_rate={"basic": 0.1, "premium": 0.5, "enterprise": 1.0}[tier]
    )

# Usage in SaaS application
customer_adapter = create_customer_adapter("cust-123", "enterprise")
with customer_adapter.track_agent("customer-research-agent") as request:
    agent = Agent(tasks=[PromptTask("Customer-specific analysis")])
    result = agent.run(customer_data)
```

### Enterprise Budget Controls

```python
# Budget enforcement with escalation
class BudgetEnforcedAdapter(GenOpsGriptapeAdapter):
    def __init__(self, *args, monthly_budget: float, escalation_threshold: float = 0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.monthly_budget = monthly_budget
        self.escalation_threshold = escalation_threshold
    
    def track_agent(self, *args, **kwargs):
        # Check budget before execution
        current_spend = self.get_monthly_spending()
        utilization = float(current_spend / self.monthly_budget)
        
        if utilization >= 1.0:
            raise BudgetExceededException(f"Monthly budget ${self.monthly_budget} exceeded")
        elif utilization >= self.escalation_threshold:
            logger.warning(f"Budget utilization at {utilization:.1%}, approaching limit")
            # Could send alerts, request approvals, etc.
        
        return super().track_agent(*args, **kwargs)
```

### A/B Testing Integration

```python
# A/B testing with Griptape structures
def ab_test_agents(query: str, user_id: str):
    """A/B test different agent configurations."""
    
    test_group = hash(user_id) % 2
    
    if test_group == 0:
        # Control group - GPT-3.5 Turbo
        with track_agent("control-agent", feature="control-group") as request:
            agent = Agent(
                prompt_driver=OpenAiChatPromptDriver(model="gpt-3.5-turbo"),
                tasks=[PromptTask(query)]
            )
            return agent.run(), request.total_cost
    else:
        # Test group - GPT-4
        with track_agent("test-agent", feature="test-group") as request:
            agent = Agent(
                prompt_driver=OpenAiChatPromptDriver(model="gpt-4"),
                tasks=[PromptTask(query)]
            )
            return agent.run(), request.total_cost
```

## Migration Guide

### From Direct Griptape

**Before (Direct Griptape):**
```python
from griptape.structures import Agent
from griptape.tasks import PromptTask

agent = Agent(tasks=[PromptTask("Analyze data")])
result = agent.run("Input data")
```

**After (With GenOps):**
```python
# Option 1: Auto-instrumentation (zero code changes)
from genops.providers.griptape import auto_instrument
auto_instrument(team="your-team", project="your-project")

# Your existing code works unchanged
from griptape.structures import Agent
from griptape.tasks import PromptTask

agent = Agent(tasks=[PromptTask("Analyze data")])
result = agent.run("Input data")
# ✅ Now includes governance tracking

# Option 2: Manual instrumentation (more control)
from genops.providers.griptape import instrument_griptape
griptape = instrument_griptape(team="your-team", project="your-project")

agent = griptape.create_agent([PromptTask("Analyze data")])
result = agent.run("Input data")
```

### Migration Checklist

- [ ] Install GenOps: `pip install genops`
- [ ] Set governance environment variables
- [ ] Run validation: `validate_griptape_setup()`
- [ ] Choose instrumentation pattern (auto vs manual)
- [ ] Update imports if using manual instrumentation
- [ ] Verify telemetry export in observability dashboard
- [ ] Set up alerting and monitoring
- [ ] Document team-specific governance attributes

## Troubleshooting

### Common Issues

#### "Griptape not found"
```bash
# Install Griptape
pip install griptape

# Verify installation
python -c "import griptape; print(griptape.__version__)"
```

#### "Auto-instrumentation not working"
```bash
# Check instrumentation status
python -c "from genops.providers.griptape.registration import is_instrumented; print(f'Instrumented: {is_instrumented()}')"

# Validate setup
python -c "from genops.providers.griptape.registration import validate_griptape_setup; import pprint; pprint.pprint(validate_griptape_setup())"
```

#### "Cost calculation errors"
```bash
# Update provider pricing data
pip install --upgrade genops

# Check supported providers
python -c "from genops.providers.griptape.cost_aggregator import GriptapeCostAggregator; print(GriptapeCostAggregator().calculators.keys())"
```

#### "Telemetry not appearing in dashboard"
```bash
# Check OpenTelemetry configuration
echo $OTEL_EXPORTER_OTLP_ENDPOINT

# Verify collector connectivity
curl -v $OTEL_EXPORTER_OTLP_ENDPOINT/v1/traces

# Enable debug logging
export OTEL_LOG_LEVEL=debug
export GENOPS_LOG_LEVEL=debug
```

### Debug Mode

```python
# Enable comprehensive debugging
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable OpenTelemetry debug
import os
os.environ['OTEL_LOG_LEVEL'] = 'debug'

# Run with detailed validation
from genops.providers.griptape.registration import validate_griptape_setup
result = validate_griptape_setup()

if result['issues']:
    print("Issues found:")
    for issue in result['issues']:
        print(f"  - {issue}")

if result['recommendations']:
    print("Recommendations:")
    for rec in result['recommendations']:
        print(f"  - {rec}")
```

## Support & Community

### Getting Help

- **Documentation**: This guide and [quickstart](../griptape-quickstart.md)
- **Examples**: [Progressive examples suite](../../examples/griptape/)
- **Issues**: [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)

### Contributing

- **Code Contributions**: Follow [CONTRIBUTING.md](../../CONTRIBUTING.md)
- **Documentation**: Help improve guides and examples
- **Testing**: Add test cases and integration scenarios
- **Feedback**: Share usage patterns and improvement suggestions

### Roadmap

**Coming Soon:**
- [ ] Enhanced memory operation tracking
- [ ] Advanced agent behavior analytics
- [ ] Built-in cost optimization algorithms
- [ ] Visual workflow monitoring tools

**Long Term:**
- [ ] Multi-region deployment patterns
- [ ] Advanced governance policy engine
- [ ] Machine learning performance prediction
- [ ] Integration with Griptape Cloud

---

**Next Steps**: Try the [5-minute quickstart](../griptape-quickstart.md) or explore [progressive examples](../../examples/griptape/)