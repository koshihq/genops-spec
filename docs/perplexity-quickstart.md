# Perplexity AI + GenOps 5-Minute Quickstart

Get Perplexity AI real-time search with governance, cost tracking, and team attribution in under 5 minutes.

## Prerequisites (30 seconds)

```bash
pip install genops[perplexity]
```

Get your API key from [Perplexity AI Settings](https://www.perplexity.ai/settings/api):

```bash
export PERPLEXITY_API_KEY="pplx-your-api-key"
export GENOPS_TEAM="your-team-name"        # Optional but recommended
export GENOPS_PROJECT="your-project-name"  # Optional but recommended
```

## Choose Your Integration Approach

**🚀 Option 1: Zero-Code Auto-Instrumentation (2 minutes)**
Perfect for existing apps - add governance with just one line, no code changes required.

**🎯 Option 2: Direct Governance Integration (3 minutes)**  
For new applications or when you want full control over governance settings.

---

## Option 1: Zero-Code Auto-Instrumentation (2 minutes)

Add **one line** to enable governance for all your existing Perplexity code:

```python
from genops.providers.perplexity import auto_instrument

# THE ONLY CHANGE: Add this line to enable governance
auto_instrument(
    team="your-team",
    project="search-app",
    daily_budget_limit=25.0
)

# Your existing Perplexity code works unchanged!
import openai

client = openai.OpenAI(
    api_key="pplx-your-api-key",
    base_url="https://api.perplexity.ai"
)

response = client.chat.completions.create(
    model="sonar-pro",
    messages=[{"role": "user", "content": "AI trends 2024"}]
)

print(response.choices[0].message.content)
```

**✅ You now have:** Cost tracking, team attribution, budget controls, and governance—with zero code changes!

**Expected Output:**
```
🔍 Search completed with governance
💰 Cost: $0.002340 (tokens: $0.001200 + request: $0.001140)
🏷️ Team: your-team | Project: search-app
📊 Budget used: 9.4% of daily limit
```

## Option 2: Direct Governance Integration (3 minutes)

For more control, use the GenOps adapter directly:

```python
import os
from genops.providers.perplexity import (
    GenOpsPerplexityAdapter, 
    PerplexityModel, 
    SearchContext
)

# Create adapter with governance
adapter = GenOpsPerplexityAdapter(
    team="your-team",
    project="search-app", 
    environment="development",
    daily_budget_limit=50.0,
    governance_policy="advisory"  # Warn but allow operations
)

# Search with governance and citations
with adapter.track_search_session("my_research") as session:
    result = adapter.search_with_governance(
        query="Latest developments in artificial intelligence 2024",
        model=PerplexityModel.SONAR_PRO,
        search_context=SearchContext.HIGH,
        session_id=session.session_id,
        return_citations=True
    )
    
    print(f"🔍 Response: {result.response[:200]}...")
    print(f"💰 Cost: ${result.cost:.6f}")
    print(f"📚 Citations: {len(result.citations)} sources")
    
    # Show first citation
    if result.citations:
        citation = result.citations[0]
        print(f"📖 Source: {citation.get('title', 'N/A')}")
        print(f"🔗 URL: {citation.get('url', 'N/A')}")

# Get cost summary and optimization tips
cost_summary = adapter.get_cost_summary()
print(f"\n📊 Cost Intelligence:")
print(f"   Daily spend: ${cost_summary['daily_costs']:.6f}")
print(f"   Budget used: {cost_summary['daily_budget_utilization']:.1f}%")
print(f"   Team: {cost_summary['team']}")
```

**Expected Output:**
```
🔍 Response: Artificial intelligence in 2024 continues to evolve rapidly with significant advancements in large language models, multimodal AI systems, and practical applications across industries. Key trends include...

💰 Cost: $0.003450
📚 Citations: 8 sources
📖 Source: AI Market Trends 2024 - McKinsey Global Institute  
🔗 URL: https://www.mckinsey.com/capabilities/mckinsey-digital/our-insights/ai-trends-2024

📊 Cost Intelligence:
   Daily spend: $0.003450
   Budget used: 6.9%
   Team: your-team
```

## Real-Time Search Features

**🌐 Web Search with Citations**
- Real-time web search with up-to-date information
- Automatic citation tracking and source attribution
- Domain filtering and source quality assessment

**💰 Dual Pricing Intelligence**
- Token costs: Based on model complexity
- Request fees: Based on search context depth  
- Automatic cost optimization recommendations

**🏷️ Team Attribution**
- Team and project-level cost tracking
- Customer attribution for multi-tenant apps
- Department chargeback and cost center reporting

## Model Selection Guide

```python
# Cost-effective general search
model=PerplexityModel.SONAR           # $1-1/1M tokens + $5/1K requests

# Enhanced accuracy with better citations  
model=PerplexityModel.SONAR_PRO       # $3-15/1M tokens + request fees

# Complex reasoning with search
model=PerplexityModel.SONAR_REASONING # Higher cost, advanced capabilities
```

## Search Context Optimization

```python
# Faster, cheaper searches
search_context=SearchContext.LOW      # $5/1K requests

# Balanced cost and quality (recommended)
search_context=SearchContext.MEDIUM   # $8/1K requests  

# Comprehensive research
search_context=SearchContext.HIGH     # $12/1K requests
```

## Validation & Troubleshooting

### ✅ Quick Setup Check

Validate your setup anytime:

```bash
python -c "
from genops.providers.perplexity_validation import validate_setup, print_validation_result
result = validate_setup()
print_validation_result(result)
"
```

Or run the comprehensive setup example:

```bash
curl -O https://raw.githubusercontent.com/KoshiHQ/GenOps-AI/main/examples/perplexity/setup_validation.py
python setup_validation.py
```

### 🚨 Common First-Run Issues

**❌ Import Error: `genops.providers.perplexity`**
```bash
# Fix: Install with Perplexity support
pip install genops[perplexity]
```

**❌ `PERPLEXITY_API_KEY` not found**
```bash
# Fix: Set your API key
export PERPLEXITY_API_KEY="pplx-your-api-key"
```

**❌ `Invalid API key format`**
- Ensure your key starts with `pplx-`
- Get a fresh key from [Perplexity Settings](https://www.perplexity.ai/settings/api)

**❌ `Budget exceeded` error**
```python
# Fix: Increase budget or use cheaper options
adapter = GenOpsPerplexityAdapter(
    daily_budget_limit=100.0,  # Increase limit
    governance_policy="advisory"  # Or allow operations with warnings
)
```

## What's Next?

**🚀 Ready to go deeper?**

**📚 Learning Path (Progressive Complexity)**

1. **Cost Optimization** (10 min): `python examples/perplexity/cost_optimization.py`
   - Master dual pricing model (tokens + requests)
   - Learn when to use different models and contexts
   - Set up budget controls and cost alerts
   - **You'll see**: Model cost comparisons, optimization recommendations

2. **Advanced Search** (15 min): `python examples/perplexity/advanced_search.py`  
   - Multi-step research workflows with session tracking
   - Citation quality analysis and source filtering
   - Batch processing for multiple queries
   - **You'll see**: Research pipelines, citation analysis, domain filtering

3. **Production Deployment** (20 min): `python examples/perplexity/production_patterns.py`
   - Enterprise governance and compliance patterns
   - Multi-tenant architecture with customer attribution
   - Error handling and resilience patterns
   - **You'll see**: Enterprise configs, multi-tenant isolation, circuit breakers

4. **Interactive Setup** (10 min): `python examples/perplexity/interactive_setup_wizard.py`
   - Guided configuration for your specific use case
   - Custom templates for different deployment scenarios
   - **You'll see**: Step-by-step wizard, generated config files

## Common Patterns

**Batch Processing:**
```python
queries = [
    "AI trends 2024", 
    "Machine learning best practices",
    "Future of automation"
]

results = adapter.batch_search_with_governance(
    queries=queries,
    model=PerplexityModel.SONAR,
    batch_optimization=True
)
```

**Budget-Aware Search:**
```python
adapter = GenOpsPerplexityAdapter(
    daily_budget_limit=10.0,
    governance_policy="enforced"  # Block when budget exceeded
)
```

**Multi-Tenant Attribution:**
```python
result = adapter.search_with_governance(
    query="Customer support automation strategies",
    customer_id="client-123",
    cost_center="customer-success"
)
```

## Support & Community

- **Documentation**: [Complete Integration Guide](integrations/perplexity.md)
- **Examples**: Browse `/examples/perplexity/` for 20+ working examples
- **Issues**: [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- **Discussions**: [Community Forum](https://github.com/KoshiHQ/GenOps-AI/discussions)

---

**✅ You now have Perplexity AI with governance!** Cost tracking, team attribution, and budget controls work automatically across all your searches.