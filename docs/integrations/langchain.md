# LangChain Integration Guide

## Overview

The GenOps LangChain adapter provides comprehensive governance telemetry for LangChain applications, including:

- **Chain execution tracking** with detailed performance metrics
- **Multi-provider cost aggregation** across OpenAI, Anthropic, and other LLM providers
- **RAG operation monitoring** for retrieval, embedding, and vector search operations
- **Agent workflow telemetry** with decision tracking and tool usage
- **Policy enforcement** with governance attribute propagation

## Quick Start

### Installation

```bash
pip install genops-ai[langchain]
```

### Basic Setup

The simplest way to add GenOps tracking to your LangChain application:

```python
from genops.providers.langchain import instrument_langchain

# Initialize GenOps LangChain adapter
adapter = instrument_langchain()

# Your existing LangChain code works unchanged
from langchain.chains import LLMChain
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.7)
chain = LLMChain(llm=llm, prompt=your_prompt)

# Add governance tracking to chain execution
result = adapter.instrument_chain_run(
    chain,
    input="What is artificial intelligence?",
    team="ai-research",
    project="knowledge-base",
    customer_id="customer_123"
)
```

### Auto-Instrumentation (Recommended)

For zero-code setup, enable auto-instrumentation:

```python
from genops import auto_instrument

# Automatically instrument all supported frameworks
auto_instrument()

# Your LangChain code automatically gets governance telemetry
chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run("Your query here")  # Automatically tracked!
```

## Core Features

### 1. Chain Execution Tracking

Track any LangChain chain with detailed telemetry:

```python
from genops.providers.langchain import instrument_langchain

adapter = instrument_langchain()

# Track chain execution with governance attributes
result = adapter.instrument_chain_run(
    chain=my_chain,
    input="Analyze this document",
    
    # Governance attributes for cost attribution
    team="document-analysis",
    project="legal-review", 
    environment="production",
    customer_id="legal_corp_456",
    
    # Chain execution parameters
    temperature=0.3,
    max_tokens=1000
)
```

**Telemetry Captured:**
- Chain execution time and steps
- LLM provider costs (OpenAI, Anthropic, etc.)
- Token usage by provider and model
- Success/error rates
- Governance attribute propagation

### 2. Multi-Provider Cost Aggregation

Automatically track costs across multiple LLM providers in a single chain:

```python
from genops.providers.langchain import create_chain_cost_context

# Context manager automatically aggregates costs
with create_chain_cost_context("my_chain_id") as cost_context:
    
    # Multiple LLM calls are automatically tracked
    openai_result = openai_chain.run("First query")     # $0.015
    anthropic_result = claude_chain.run("Second query")  # $0.012
    cohere_result = cohere_chain.run("Third query")     # $0.008
    
    # Get comprehensive cost breakdown
    summary = cost_context.get_final_summary()
    
print(f"Total cost: ${summary.total_cost:.4f}")
print(f"Providers used: {list(summary.unique_providers)}")
print(f"Cost by provider: {summary.cost_by_provider}")
```

**Cost Summary Includes:**
- Total cost across all providers
- Cost breakdown by provider (OpenAI, Anthropic, etc.)
- Cost breakdown by model (gpt-4, claude-3, etc.)
- Token usage statistics
- Operation timing metrics

### 3. RAG Operation Monitoring

Comprehensive tracking for Retrieval-Augmented Generation workflows:

```python
from genops.providers.langchain import instrument_langchain

adapter = instrument_langchain()

# Track RAG query with detailed retrieval metrics
documents = adapter.instrument_rag_query(
    query="What are the latest AI safety guidelines?",
    retriever=vector_store_retriever,
    
    # Governance attributes
    team="safety-research",
    project="guideline-search",
    
    # RAG parameters
    k=5,  # Top-k documents
    score_threshold=0.7
)

# Instrument vector search operations
results = adapter.instrument_vector_search(
    vector_store=chroma_store,
    query="AI safety research",
    k=10,
    team="research-team"
)
```

**RAG Telemetry Captured:**
- Document retrieval performance and relevance scores
- Vector search latency and result quality
- Embedding model usage and costs
- RAG pipeline end-to-end performance

### 4. Agent Workflow Tracking

Monitor LangChain agents with decision tracking:

```python
from genops.providers.langchain import GenOpsLangChainCallbackHandler

# Create callback handler for agent monitoring
callback_handler = GenOpsLangChainCallbackHandler(
    adapter, 
    chain_id="agent_workflow_001"
)

# Agent execution with governance tracking
agent_result = agent.run(
    "Research the latest developments in quantum computing",
    callbacks=[callback_handler],
    
    # Governance context
    team="research-agents", 
    project="quantum-research"
)
```

**Agent Telemetry Captured:**
- Tool usage and decision paths
- Multi-step reasoning costs
- Agent performance metrics
- Error handling and recovery

## Integration Patterns

### Pattern 1: Decorator-Based Instrumentation

```python
from genops.decorators import track_langchain

@track_langchain(
    team="content-generation",
    project="blog-automation"
)
def generate_blog_post(topic: str) -> str:
    chain = create_blog_chain()
    return chain.run(topic=topic)

# Automatic telemetry on every call
post = generate_blog_post("AI in Healthcare")
```

### Pattern 2: Context Manager Pattern

```python
from genops.providers.langchain import create_chain_cost_context

def process_customer_queries(queries: list[str], customer_id: str):
    with create_chain_cost_context(f"batch_{customer_id}") as context:
        results = []
        
        for query in queries:
            result = qa_chain.run(query)
            results.append(result)
            
            # Costs automatically aggregated per customer
            
        # Get final cost summary for billing
        summary = context.get_final_summary()
        bill_customer(customer_id, summary.total_cost)
        
        return results
```

### Pattern 3: Policy Enforcement

```python
from genops.providers.langchain import instrument_langchain
from genops.core.policy import enforce_policy

adapter = instrument_langchain()

@enforce_policy("content_moderation")
def process_user_content(content: str, user_id: str):
    return adapter.instrument_chain_run(
        moderation_chain,
        input=content,
        user_id=user_id,
        team="content-safety"
    )
```

## Configuration

### Environment Variables

```bash
# OpenTelemetry configuration
export OTEL_SERVICE_NAME="my-langchain-app"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"

# GenOps LangChain configuration
export GENOPS_LANGCHAIN_AUTO_INSTRUMENT=true
export GENOPS_LANGCHAIN_COST_TRACKING=true
export GENOPS_LANGCHAIN_RAG_MONITORING=true

# Provider API keys (if using cost tracking)
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
```

### Programmatic Configuration

```python
from genops.providers.langchain import configure_langchain_adapter

configure_langchain_adapter({
    "auto_instrument": True,
    "cost_tracking": {
        "enabled": True,
        "providers": ["openai", "anthropic", "cohere"],
        "fallback_pricing": True
    },
    "rag_monitoring": {
        "enabled": True,
        "track_embeddings": True,
        "track_retrievals": True
    },
    "telemetry": {
        "service_name": "my-langchain-service",
        "attributes": {
            "deployment.environment": "production",
            "service.version": "1.0.0"
        }
    }
})
```

## Troubleshooting

### Common Issues

#### Issue: "LangChain package not found"
```python
# Solution: Install LangChain
pip install langchain

# Or install with GenOps extras
pip install genops-ai[langchain]
```

#### Issue: Cost tracking not working
```python
# Check if provider adapters are available
from genops.providers.langchain.cost_aggregator import get_cost_aggregator

aggregator = get_cost_aggregator()
print("Available cost calculators:", list(aggregator.provider_cost_calculators.keys()))

# Enable debug logging
import logging
logging.getLogger("genops.providers.langchain").setLevel(logging.DEBUG)
```

#### Issue: Telemetry not appearing in observability platform
```python
# Verify OpenTelemetry configuration
from opentelemetry import trace

tracer = trace.get_tracer(__name__)
with tracer.start_as_current_span("test-span") as span:
    span.set_attribute("test", "value")
    print("OpenTelemetry is working")

# Check OTLP exporter configuration
import os
print("OTLP Endpoint:", os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"))
```

### Debug Mode

Enable comprehensive debug logging:

```python
import logging

# Enable GenOps debug logging
logging.getLogger("genops").setLevel(logging.DEBUG)

# Enable LangChain adapter debug logging
logging.getLogger("genops.providers.langchain").setLevel(logging.DEBUG)

# Enable OpenTelemetry debug logging
logging.getLogger("opentelemetry").setLevel(logging.DEBUG)
```

### Validation Utilities

Verify your setup is working correctly:

```python
from genops.providers.langchain import validate_setup

# Run comprehensive setup validation
validation_result = validate_setup()

if validation_result.is_valid:
    print("✅ GenOps LangChain setup is valid!")
else:
    print("❌ Setup issues found:")
    for issue in validation_result.issues:
        print(f"  - {issue}")
```

## Performance Considerations

### Best Practices

1. **Use context managers** for cost tracking to ensure proper cleanup
2. **Enable sampling** for high-volume applications to reduce overhead
3. **Configure appropriate log levels** to avoid performance impact
4. **Use async patterns** when available for better concurrency

### Performance Tuning

```python
from genops.providers.langchain import configure_performance

configure_performance({
    "sampling_rate": 0.1,  # Sample 10% of operations
    "async_export": True,   # Export telemetry asynchronously
    "batch_size": 100,      # Batch telemetry exports
    "buffer_timeout": 5000  # Export buffer timeout (ms)
})
```

## Next Steps

- Explore the [complete examples](../examples/langchain/) for advanced patterns
- Check out [governance scenarios](../examples/governance_scenarios/) for policy enforcement
- Review [observability integration](../observability/) for dashboard setup
- See [API reference](../api/langchain.md) for detailed method documentation

## Support

- **Issues:** [GitHub Issues](https://github.com/genops-ai/genops-ai/issues)
- **Discussions:** [GitHub Discussions](https://github.com/genops-ai/genops-ai/discussions)  
- **Documentation:** [Full Documentation](https://docs.genops.ai)