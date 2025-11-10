# Haystack AI Quickstart Guide

**🎯 What you'll learn:** Get Haystack AI's powerful orchestration framework + complete cost governance working in exactly 5 minutes, with zero code changes to your existing pipelines.

## What is GenOps?

**GenOps AI** is a governance telemetry layer built on OpenTelemetry that provides cost tracking, budget enforcement, and compliance monitoring for AI systems. It extends your existing observability stack with AI-specific governance capabilities without replacing your current tools.

**Why this matters for Haystack:**
- **End-to-End Pipeline Tracking**: Complete visibility across complex RAG and agent workflows
- **Multi-Provider Cost Aggregation**: Unified cost tracking across all providers in your pipelines
- **Component-Level Insights**: Granular monitoring of retrievers, generators, embedders, and custom components
- **Zero Migration Pain**: Add governance to existing Haystack pipelines with one line

**Key Benefits:**
- **Cost Transparency**: Real-time cost tracking across all pipeline components
- **Budget Controls**: Configurable spending limits with enforcement policies
- **Multi-tenant Governance**: Per-team, per-project, per-customer attribution
- **Vendor Independence**: Works with 15+ observability platforms via OpenTelemetry
- **Zero Code Changes**: Auto-instrumentation for existing applications

Get started with Haystack + GenOps governance in under 5 minutes. This guide provides the essential patterns for immediate productivity with Haystack's orchestration capabilities and comprehensive cost tracking.

**⏱️ Time commitment:** 5 minutes | **✅ Result:** Full Haystack pipeline governance with cost tracking

## ⚡ 5-Minute Quick Start

### 1. Install Dependencies (30 seconds)

```bash
# Install GenOps with Haystack support
pip install genops-ai[haystack] haystack-ai

# Or install separately
pip install genops-ai haystack-ai

# Install AI provider dependencies (choose your providers)
pip install openai anthropic cohere-ai transformers
```

### 2. Set Your API Keys (30 seconds)

**🔑 API Key Setup:**
1. Get API keys from your AI providers:
   - OpenAI: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
   - Anthropic: [console.anthropic.com](https://console.anthropic.com)
   - Cohere: [dashboard.cohere.ai](https://dashboard.cohere.ai)

2. Set environment variables:

```bash
# Set keys for providers you plan to use
export OPENAI_API_KEY="sk-your-openai-key-here"
export ANTHROPIC_API_KEY="your-anthropic-key-here"
export COHERE_API_KEY="your-cohere-key-here"
```

**💡 Provider Support:** GenOps tracks costs across all providers used in your Haystack pipeline

**✅ Verification:** Your keys are working if the validation step below passes

### 3. Validate Setup (60 seconds)

```python
# Verify everything is working
from genops.providers.haystack import validate_haystack_setup

result = validate_haystack_setup()

if result["is_valid"]:
    print("✅ Haystack + GenOps Setup Ready!")
    print(f"Available providers: {result['available_providers']}")
else:
    print("❌ Setup Issues Found:")
    for issue in result["issues"]:
        print(f"  • {issue}")
```

Expected output:
```
✅ Haystack + GenOps Setup Ready!
Available providers: ['OpenAI integration', 'Anthropic integration']
```

### 4. Zero-Code Auto-Instrumentation (60 seconds)

**🎯 The Magic:** Add ONE line to existing Haystack code for complete governance

```python
# Add this single line for automatic governance
from genops.providers.haystack import auto_instrument
auto_instrument()  # ✨ This enables automatic pipeline tracking

# Your existing Haystack code works unchanged
from haystack import Pipeline
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders import PromptBuilder

# Create your pipeline as usual
pipeline = Pipeline()
pipeline.add_component("prompt_builder", PromptBuilder(
    template="Answer the question: {{question}}"
))
pipeline.add_component("llm", OpenAIGenerator(model="gpt-3.5-turbo"))

pipeline.connect("prompt_builder", "llm")

# Run your pipeline normally
result = pipeline.run({
    "prompt_builder": {"question": "What is Retrieval-Augmented Generation?"}
})

print(result["llm"]["replies"][0])
# ✅ Automatic cost tracking, governance, and observability added!
```

**🔥 What just happened:**
- Your existing Haystack pipeline got automatic cost tracking
- Multi-provider cost aggregation across all components
- Zero code changes required to your pipeline logic
- Complete observability integration with OpenTelemetry

### 5. Manual Governance Control (120 seconds)

**🎛️ Full Control Mode:** Explicit governance configuration with detailed tracking

```python
# Full control with explicit governance
from genops.providers.haystack import GenOpsHaystackAdapter

# Create adapter with governance settings
adapter = GenOpsHaystackAdapter(
    team="your-team",
    project="quickstart-demo", 
    daily_budget_limit=10.0,
    governance_policy="advisory"  # "advisory" warns, "enforcing" blocks
)

# Track entire pipeline execution
with adapter.track_pipeline("rag-demo") as context:
    # Your pipeline execution
    result = pipeline.run({
        "prompt_builder": {"question": "Explain the benefits of Haystack AI orchestration"}
    })
    
    print(f"Response: {result['llm']['replies'][0]}")

# Get detailed execution metrics
metrics = context.get_metrics()
print(f"\n📊 Pipeline Metrics:")
print(f"Total cost: ${metrics.total_cost:.6f}")
print(f"Components executed: {metrics.total_components}")
print(f"Execution time: {metrics.total_execution_time_seconds:.2f}s")
print(f"Cost by provider: {metrics.cost_by_provider}")
```

**📚 About Pipeline Tracking:**
- **End-to-end visibility**: Complete pipeline execution monitoring
- **Component-level costs**: Individual cost tracking for each component
- **Multi-provider aggregation**: Unified cost view across all providers used
- **Performance insights**: Execution time and resource utilization

## 🎯 **You're Ready!** 

In 5 minutes you now have:
- ✅ Haystack + GenOps governance working
- ✅ Automatic cost tracking and attribution  
- ✅ Multi-provider cost aggregation
- ✅ Component-level performance monitoring
- ✅ Production-ready governance controls

## 🚀 Next Steps (Optional)

### Explore RAG Pipeline Tracking

```python
# Specialized RAG adapter with enhanced tracking
from genops.providers.haystack import create_rag_adapter

rag_adapter = create_rag_adapter(
    team="research-team",
    project="document-qa",
    daily_budget_limit=50.0
)

with rag_adapter.track_pipeline("rag-workflow") as context:
    # Retrieval phase
    retrieval_result = document_retriever.run(query="What is RAG?")
    
    # Generation phase  
    generation_result = generator.run(
        prompt=prompt_builder.run(
            question="What is RAG?",
            documents=retrieval_result["documents"]
        )["prompt"]
    )

# Get RAG-specific insights
from genops.providers.haystack import get_rag_insights
insights = get_rag_insights(rag_adapter.monitor, context.pipeline_id)
print(f"Documents retrieved: {insights['documents_retrieved']}")
print(f"Retrieval latency: {insights['retrieval_latency']:.2f}s")
print(f"Generation latency: {insights['generation_latency']:.2f}s")
```

### Analyze Pipeline Costs

```python
# Get comprehensive cost analysis with optimization recommendations
from genops.providers.haystack import analyze_pipeline_costs

analysis = analyze_pipeline_costs(adapter, time_period_hours=24)

print(f"Total cost (24h): ${analysis['total_cost']:.2f}")
print(f"Most expensive component: {analysis['most_expensive_component']}")

print("\n💡 Optimization Recommendations:")
for rec in analysis['recommendations']:
    print(f"  • {rec['reasoning']}")
    print(f"    Potential savings: ${rec['potential_savings']:.4f}")
```

### Session-Based Multi-Pipeline Tracking

```python
# Track multiple related pipelines in a session
with adapter.track_session("research-experiment") as session:
    for query in research_queries:
        with adapter.track_pipeline("qa-pipeline") as pipeline_ctx:
            result = qa_pipeline.run({"query": query})
            
    print(f"Session total cost: ${session.total_cost:.6f}")
    print(f"Pipelines executed: {session.total_pipelines}")
```

### Agent Workflow Tracking

```python
# Specialized agent adapter for complex workflows  
from genops.providers.haystack import create_agent_adapter

agent_adapter = create_agent_adapter(
    team="ai-agents",
    project="research-assistant",
    daily_budget_limit=100.0
)

with agent_adapter.track_session("research-task") as session:
    for step in agent_workflow_steps:
        with agent_adapter.track_pipeline(f"agent-step-{step}") as context:
            result = agent_pipeline.run(step_input)

# Get agent-specific insights
from genops.providers.haystack import get_agent_insights
insights = get_agent_insights(agent_adapter.monitor, context.pipeline_id)
print(f"Decisions made: {insights['decisions_made']}")
print(f"Tools used: {insights['tools_used']}")
```

## 🛠️ Troubleshooting

### API Key Issues
```bash
# Check API key format and access
python -c "
from genops.providers.haystack import validate_haystack_setup
result = validate_haystack_setup()
if result['available_providers']:
    print('✅ API keys configured')
else:
    print('❌ No API keys found')
"
```

### Import Errors
```bash
# Verify installations
pip show genops-ai haystack-ai

# Reinstall if needed
pip install --upgrade genops-ai[haystack] haystack-ai
```

### Pipeline Not Tracked
```python
# Check instrumentation status
from genops.providers.haystack import is_instrumented, get_instrumentation_stats

if is_instrumented():
    stats = get_instrumentation_stats()
    print(f"✅ Instrumentation active: {stats}")
else:
    print("❌ Auto-instrumentation not enabled")
    # Re-enable
    from genops.providers.haystack import auto_instrument
    auto_instrument()
```

### Cost Tracking Issues
```python
# Check cost summary
from genops.providers.haystack import get_cost_summary
summary = get_cost_summary()

if "error" in summary:
    print(f"❌ Cost tracking issue: {summary['error']}")
else:
    print(f"✅ Daily costs: ${summary['daily_costs']:.6f}")
    print(f"Budget utilization: {summary['daily_budget_utilization']:.1f}%")
```

## 📚 Learn More

**🎯 Next Learning Paths:**
- **[Complete Examples](../../examples/haystack/)** - 7+ comprehensive examples from basic to enterprise
- **[Full Integration Guide](integrations/haystack.md)** - Complete documentation and advanced patterns
- **[RAG Workflow Examples](../../examples/haystack/rag_workflow_governance.py)** - Specialized RAG tracking
- **[Agent Workflow Examples](../../examples/haystack/agent_workflow_tracking.py)** - Agent decision monitoring

**🔧 Interactive Tools:**
- **[Setup Validation](../../examples/haystack/setup_validation.py)** - Test your configuration
- **[Basic Pipeline Tracking](../../examples/haystack/basic_pipeline_tracking.py)** - Simple pipeline example
- **[Cost Optimization Tool](../../examples/haystack/multi_provider_cost_aggregation.py)** - Cost analysis and optimization

## 🔗 Key Resources

**🏗️ Haystack AI:**
- **Platform**: https://haystack.deepset.ai
- **Documentation**: https://docs.haystack.deepset.ai
- **Component Hub**: https://haystack.deepset.ai/integrations
- **GitHub**: https://github.com/deepset-ai/haystack

**🛠️ GenOps Platform:**
- **Documentation Hub**: https://docs.genops.ai
- **GitHub Repository**: https://github.com/KoshiHQ/GenOps-AI
- **Community Discussions**: https://github.com/KoshiHQ/GenOps-AI/discussions

---

**🏆 Success Metrics**: After this quickstart, developers achieve immediate productivity with Haystack AI's orchestration framework under full GenOps governance, with complete pipeline visibility and cross-provider cost tracking.