# LangChain Quickstart

Get GenOps governance telemetry running with your LangChain application in under 5 minutes.

## 🚀 Quick Setup

### 1. Install GenOps with LangChain Support

```bash
pip install genops-ai[langchain]
```

### 2. Set Environment Variables

```bash
export OPENAI_API_KEY="your_openai_key_here"
export OTEL_SERVICE_NAME="my-langchain-app"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"  # Optional
```

### 3. Enable Auto-Instrumentation (Zero Code Changes)

```python
from genops import auto_instrument

# This one line enables telemetry for all LangChain operations
auto_instrument()

# Your existing LangChain code works unchanged!
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

chain = LLMChain(
    llm=OpenAI(temperature=0.7),
    prompt=PromptTemplate.from_template("Tell me about {topic}")
)

result = chain.run("artificial intelligence")  # Automatically tracked!
```

**That's it!** Your LangChain application now captures:
- ✅ Chain execution costs and performance
- ✅ Multi-provider cost aggregation 
- ✅ Token usage by provider and model
- ✅ Error tracking and success rates

## 💰 Add Cost Attribution

For cost attribution and billing, add governance attributes:

```python
from genops.core.context import set_governance_context

# Set once - applies to all operations
set_governance_context({
    "team": "customer-support",
    "project": "chatbot-v2", 
    "customer_id": "enterprise_customer_123",
    "environment": "production"
})

# All LangChain operations now include governance attributes
result = chain.run("How can I help you?")
```

## 🔍 Manual Instrumentation (Fine-Grained Control)

For more control, use manual instrumentation:

```python
from genops.providers.langchain import instrument_langchain

# Initialize adapter
adapter = instrument_langchain()

# Instrument specific chain runs
result = adapter.instrument_chain_run(
    chain,
    topic="machine learning",
    
    # Governance attributes for cost attribution
    team="ai-research",
    project="knowledge-base",
    customer_id="customer_456"
)

print(f"Result: {result}")
```

## 📊 Cost Tracking Context

Track costs across multiple LLM providers in a single operation:

```python
from genops.providers.langchain import create_chain_cost_context

with create_chain_cost_context("my_operation") as cost_context:
    # Multiple LLM calls automatically aggregated
    result1 = openai_chain.run("First query")
    result2 = anthropic_chain.run("Second query")
    result3 = cohere_chain.run("Third query")
    
    # Get comprehensive cost breakdown
    summary = cost_context.get_final_summary()
    print(f"Total cost: ${summary.total_cost:.4f}")
    print(f"Providers: {list(summary.unique_providers)}")
```

## 🔗 RAG Application Monitoring

For RAG applications, track retrieval and generation costs separately:

```python
from genops.providers.langchain import instrument_langchain

adapter = instrument_langchain()

# Track RAG query with detailed metrics
documents = adapter.instrument_rag_query(
    query="What is AI governance?",
    retriever=vector_store_retriever,
    team="knowledge-team",
    k=5
)

# Track vector search performance
results = adapter.instrument_vector_search(
    vector_store=chroma_store,
    query="AI safety guidelines",
    k=10,
    team="safety-team"
)
```

## 📈 View Your Telemetry

### Option 1: Local Observability Stack

```bash
# From your project root
curl -O https://raw.githubusercontent.com/genops-ai/genops-ai/main/docker-compose.observability.yml
docker-compose -f docker-compose.observability.yml up -d

# View dashboards
open http://localhost:3000  # Grafana
open http://localhost:16686 # Jaeger
```

### Option 2: Your Existing Platform

GenOps works with any OpenTelemetry-compatible platform:

**Datadog:** See the **[Datadog Quickstart Guide](datadog-quickstart.md)** for complete 5-minute setup.

```bash
# Quick Datadog setup
export DATADOG_API_KEY="your_datadog_api_key"
export DATADOG_SITE="datadoghq.com"
```

**Other Platforms:**

```bash
# Honeycomb
export OTEL_EXPORTER_OTLP_ENDPOINT="https://api.honeycomb.io"
export HONEYCOMB_API_KEY="your_honeycomb_key"

# New Relic
export OTEL_EXPORTER_OTLP_ENDPOINT="https://otlp.nr-data.net"
export NEW_RELIC_API_KEY="your_newrelic_key"
```

## ✅ Verify Setup

Run this verification script:

```python
from genops.providers.langchain import validate_setup

result = validate_setup()
if result.is_valid:
    print("✅ GenOps LangChain setup is working!")
else:
    print("❌ Setup issues:")
    for issue in result.issues:
        print(f"  - {issue}")
```

## 🎯 Common Use Cases

### Web Application Integration

```python
# FastAPI example
from fastapi import FastAPI
from genops import auto_instrument

app = FastAPI()
auto_instrument()  # Enable for all routes

@app.post("/chat")
async def chat_endpoint(message: str, user_id: str):
    # Automatically tracked with user attribution
    set_governance_context({"customer_id": user_id})
    return {"response": chain.run(message)}
```

### Batch Processing

```python
def process_customer_queries(queries: list, customer_id: str):
    with create_chain_cost_context(f"batch_{customer_id}") as context:
        results = []
        for query in queries:
            result = qa_chain.run(query)
            results.append(result)
        
        # Automatic cost aggregation for billing
        summary = context.get_final_summary()
        bill_customer(customer_id, summary.total_cost)
        
        return results
```

### Multi-Step Workflows

```python
def content_pipeline(topic: str):
    with create_chain_cost_context("content_generation") as context:
        # Step 1: Research
        research = research_chain.run(topic)
        
        # Step 2: Outline  
        outline = outline_chain.run(research)
        
        # Step 3: Draft
        draft = writing_chain.run(outline)
        
        # Step 4: Review
        final = review_chain.run(draft)
        
        # All costs automatically tracked and attributed
        return final
```

## 🔧 Troubleshooting

### Issue: No telemetry appearing

```bash
# Check OpenTelemetry configuration
python -c "import os; print('OTLP endpoint:', os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT'))"

# Enable debug logging
export OTEL_LOG_LEVEL=debug
export GENOPS_LOG_LEVEL=debug
```

### Issue: Cost tracking not working

```python
# Verify provider adapters
from genops.providers.langchain.cost_aggregator import get_cost_aggregator

aggregator = get_cost_aggregator()
print("Available calculators:", list(aggregator.provider_cost_calculators.keys()))
```

### Issue: LangChain not detected

```bash
# Ensure LangChain is installed
pip install langchain

# Verify GenOps can import LangChain
python -c "from genops.providers.langchain import instrument_langchain; print('LangChain available')"
```

## 📚 Next Steps

Once you have basic telemetry working:

1. **[Complete Integration Guide](integrations/langchain.md)** - Comprehensive documentation
2. **[Examples](examples/langchain/)** - Practical implementation patterns  
3. **[Cost Management](examples/langchain/multi_provider_costs.py)** - Advanced cost tracking
4. **[RAG Monitoring](examples/langchain/rag_pipeline_monitoring.py)** - RAG-specific patterns
5. **[Policy Enforcement](examples/governance_scenarios/)** - Governance and compliance

## 💬 Support

- **Issues**: [GitHub Issues](https://github.com/genops-ai/genops-ai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/genops-ai/genops-ai/discussions)
- **Documentation**: [Complete Docs](https://docs.genops.ai)

---

**🎉 You now have complete governance telemetry for your LangChain application!**

Your telemetry includes cost tracking, performance metrics, error monitoring, and governance attribution - all with minimal code changes.