# LangChain Examples

This directory contains practical examples demonstrating how to integrate GenOps governance telemetry with LangChain applications.

## Examples Overview

### Basic Integration
- **[basic_chain_tracking.py](basic_chain_tracking.py)** - Simple chain execution with governance tracking
- **[auto_instrumentation.py](auto_instrumentation.py)** - Zero-code setup with automatic instrumentation
- **[manual_instrumentation.py](manual_instrumentation.py)** - Fine-grained control over telemetry

### Cost Management
- **[multi_provider_costs.py](multi_provider_costs.py)** - Track costs across multiple LLM providers
- **[cost_attribution.py](cost_attribution.py)** - Per-customer and per-team cost tracking
- **[budget_monitoring.py](budget_monitoring.py)** - Real-time cost monitoring and alerts

### RAG Applications
- **[rag_pipeline_monitoring.py](rag_pipeline_monitoring.py)** - Complete RAG workflow tracking
- **[vector_store_instrumentation.py](vector_store_instrumentation.py)** - Vector search performance monitoring
- **[embedding_cost_tracking.py](embedding_cost_tracking.py)** - Track embedding model usage and costs

### Agent Workflows
- **[agent_decision_tracking.py](agent_decision_tracking.py)** - Monitor agent tool usage and decisions
- **[multi_step_agent_costs.py](multi_step_agent_costs.py)** - Cost attribution for complex agent workflows
- **[agent_error_handling.py](agent_error_handling.py)** - Error tracking and recovery in agent systems

### Production Patterns
- **[middleware_integration.py](middleware_integration.py)** - Web framework integration patterns
- **[batch_processing.py](batch_processing.py)** - High-volume batch job monitoring
- **[async_chain_tracking.py](async_chain_tracking.py)** - Asynchronous LangChain operations

### Policy & Governance
- **[content_moderation.py](content_moderation.py)** - Policy enforcement in content pipelines
- **[compliance_audit.py](compliance_audit.py)** - Audit trail generation for compliance
- **[customer_data_governance.py](customer_data_governance.py)** - Customer data handling governance

## Quick Start

1. **Install dependencies:**
```bash
pip install genops-ai[langchain] langchain openai anthropic
```

2. **Set up environment:**
```bash
export OPENAI_API_KEY="your_openai_key"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
```

3. **Run a basic example:**
```bash
python basic_chain_tracking.py
```

## Example Structure

Each example follows a consistent structure:

```python
"""
Example: [Description]
Demonstrates: [Key features]
Use case: [Real-world scenario]
"""

# Setup and imports
from genops.providers.langchain import instrument_langchain
# ... other imports

# Configuration
adapter = instrument_langchain()

# Example implementation
def main():
    # Example code with explanatory comments
    pass

# Telemetry verification
def verify_telemetry():
    # Code to verify telemetry is working
    pass

if __name__ == "__main__":
    main()
    verify_telemetry()
```

## Running Examples

### Prerequisites

All examples require:
- Python 3.8+
- GenOps AI SDK with LangChain extras
- OpenTelemetry collector running (for full telemetry)

### Optional: Local Observability Stack

To see telemetry in action, run the local observability stack:

```bash
# From the root directory
docker-compose -f docker-compose.observability.yml up -d

# Examples will export telemetry to:
# - Grafana: http://localhost:3000
# - Jaeger: http://localhost:16686
```

### Environment Setup

Create a `.env` file with your API keys:

```bash
# LLM Provider API Keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
COHERE_API_KEY=your_cohere_key_here

# OpenTelemetry Configuration
OTEL_SERVICE_NAME=langchain-examples
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# GenOps Configuration
GENOPS_ENVIRONMENT=development
GENOPS_TEAM=examples-team
GENOPS_PROJECT=langchain-examples
```

## Best Practices Demonstrated

### 1. Governance Attribution
```python
# Always include governance attributes for cost attribution
result = adapter.instrument_chain_run(
    chain,
    input="user query",
    team="customer-support",
    project="chatbot-v2", 
    customer_id="customer_123",
    environment="production"
)
```

### 2. Cost Context Management
```python
# Use context managers for automatic cost aggregation
with create_chain_cost_context(operation_id) as context:
    # Multiple LLM operations automatically tracked
    result1 = chain1.run(query1)
    result2 = chain2.run(query2)
    # Costs automatically aggregated
```

### 3. Error Handling
```python
try:
    result = adapter.instrument_chain_run(chain, input=query)
except Exception as e:
    # Errors automatically captured in telemetry
    logger.error(f"Chain execution failed: {e}")
    raise
```

### 4. Performance Monitoring
```python
# Track performance metrics alongside costs
with adapter.performance_context("rag_query") as perf:
    documents = retriever.get_relevant_documents(query)
    # Performance metrics automatically captured
```

## Troubleshooting Examples

If examples aren't working:

1. **Check API keys:**
```bash
python -c "import os; print('OpenAI key configured:', bool(os.getenv('OPENAI_API_KEY')))"
```

2. **Verify GenOps installation:**
```bash
python -c "from genops.providers.langchain import instrument_langchain; print('LangChain adapter available')"
```

3. **Test OpenTelemetry:**
```bash
python -c "from opentelemetry import trace; tracer = trace.get_tracer(__name__); print('OpenTelemetry available')"
```

## Contributing

To add new examples:

1. Follow the example structure template
2. Include comprehensive comments explaining each step
3. Add governance attributes for cost attribution
4. Include error handling patterns
5. Add telemetry verification
6. Update this README with your example description

For questions or contributions, see our [Contributing Guide](../../CONTRIBUTING.md).