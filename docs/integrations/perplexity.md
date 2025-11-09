# Perplexity AI Integration Guide

Complete integration guide for Perplexity AI real-time search with GenOps governance, cost intelligence, and team attribution.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Integration Patterns](#integration-patterns)
- [Cost Management](#cost-management)
- [Advanced Features](#advanced-features)
- [Production Deployment](#production-deployment)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)
- [Examples](#examples)

## Overview

Perplexity AI provides real-time web search with AI-powered analysis and citation tracking. This integration adds GenOps governance, cost intelligence, and team attribution to all Perplexity operations.

### Key Features

**🌐 Real-Time Web Search**
- Up-to-date information from live web sources
- Automatic citation tracking and source attribution
- Domain filtering and source quality assessment

**💰 Dual Pricing Intelligence**
- Token costs based on model complexity and usage
- Request fees based on search context depth
- Real-time cost tracking and optimization recommendations

**🏷️ Enterprise Governance**
- Team and project-level cost attribution
- Budget controls with configurable enforcement policies
- Multi-tenant customer attribution and chargeback

**⚡ Performance Optimization**
- Intelligent batch processing for multiple queries
- Query result caching to reduce costs
- Context-aware model selection for optimal cost/quality

## Quick Start

**Prerequisites:**
```bash
pip install genops[perplexity]
export PERPLEXITY_API_KEY="pplx-your-api-key"
```

**Zero-Code Integration:**
```python
from genops.providers.perplexity import auto_instrument

# Enable governance with one line
auto_instrument(team="your-team", daily_budget_limit=25.0)

# Your existing code works unchanged
import openai
client = openai.OpenAI(api_key="pplx-key", base_url="https://api.perplexity.ai")
response = client.chat.completions.create(
    model="sonar-pro",
    messages=[{"role": "user", "content": "AI trends 2024"}]
)
# ↑ This now has automatic cost tracking and governance!
```

**Expected Output:**
```
🔍 Perplexity search completed with governance
💰 Cost: $0.002340 | Token cost: $0.001200 | Request cost: $0.001140  
🏷️ Team: your-team | Project: default
📊 Budget used: 9.4% of daily limit
✅ Governance: advisory (warnings enabled)
```

## Installation

### Standard Installation

```bash
pip install genops[perplexity]
```

### Development Installation

```bash
git clone https://github.com/KoshiHQ/GenOps-AI.git
cd GenOps-AI
pip install -e ".[perplexity,dev]"
```

### Docker Installation

```dockerfile
FROM python:3.10-slim
RUN pip install genops[perplexity]
ENV PERPLEXITY_API_KEY="pplx-your-api-key"
ENV GENOPS_TEAM="your-team"
```

### Dependencies

- Python 3.8+
- OpenAI client library (for Perplexity API compatibility)
- OpenTelemetry SDK for telemetry export
- Optional: Pydantic for configuration validation

## Configuration

### Environment Variables

```bash
# Required
export PERPLEXITY_API_KEY="pplx-your-api-key"

# Recommended for governance
export GENOPS_TEAM="your-team-name"
export GENOPS_PROJECT="your-project-name" 
export GENOPS_ENVIRONMENT="development"

# Budget controls
export GENOPS_DAILY_BUDGET_LIMIT="50.0"
export GENOPS_MONTHLY_BUDGET_LIMIT="1500.0"

# Enterprise attribution
export GENOPS_CUSTOMER_ID="customer-123"
export GENOPS_COST_CENTER="ai-research-lab"

# Performance settings
export GENOPS_ENABLE_CACHING="true"
export GENOPS_RETRY_ATTEMPTS="3"
export GENOPS_TIMEOUT_SECONDS="30"
```

### Programmatic Configuration

```python
from genops.providers.perplexity import (
    GenOpsPerplexityAdapter,
    PerplexityModel,
    SearchContext
)

adapter = GenOpsPerplexityAdapter(
    # Basic identification
    team="ai-research-team",
    project="market-intelligence",
    environment="production",
    
    # Budget management
    daily_budget_limit=200.0,
    monthly_budget_limit=6000.0,
    governance_policy="enforced",  # advisory, enforced, strict
    enable_cost_alerts=True,
    
    # Enterprise attribution
    customer_id="enterprise-client-001",
    cost_center="research-division",
    
    # Search defaults
    default_search_context=SearchContext.HIGH,
    
    # Performance optimization
    enable_caching=True,
    cache_ttl_seconds=300,
    
    # Custom tags for attribution
    tags={
        "department": "research",
        "use_case": "market_analysis",
        "compliance_level": "high"
    }
)
```

### Configuration File

Create `genops_config.yaml`:

```yaml
perplexity:
  governance:
    team: "ai-research-team"
    project: "market-intelligence"
    environment: "production"
    customer_id: "enterprise-client-001"
    cost_center: "research-division"
    
  budget:
    daily_limit: 200.0
    monthly_limit: 6000.0
    policy: "enforced"
    enable_alerts: true
    alert_thresholds:
      warning: 0.8
      critical: 0.95
      
  search:
    default_model: "sonar-pro"
    default_context: "high"
    max_tokens: 500
    timeout_seconds: 30
    
  performance:
    enable_caching: true
    cache_ttl_seconds: 300
    retry_attempts: 3
    batch_optimization: true
    
  tags:
    department: "research"
    use_case: "market_analysis"
    compliance_level: "high"
```

Load configuration:

```python
from genops.config import load_config
from genops.providers.perplexity import GenOpsPerplexityAdapter

config = load_config("genops_config.yaml")
adapter = GenOpsPerplexityAdapter.from_config(config.perplexity)
```

## Integration Patterns

Choose the integration approach that best fits your use case:

| **Approach** | **Best For** | **Setup Time** | **Code Changes** |
|--------------|--------------|----------------|------------------|
| [Auto-Instrumentation](#1-zero-code-auto-instrumentation) | Existing Perplexity apps | 30 seconds | None required |
| [Direct Adapter](#2-direct-adapter-integration) | New apps, full control | 2 minutes | Minimal changes |  
| [Context Managers](#3-context-manager-pattern) | Complex workflows | 5 minutes | Structured code |
| [Batch Processing](#4-batch-processing-pattern) | Multiple queries | 3 minutes | Optimize for volume |
| [Async Pattern](#5-async-pattern-advanced) | High performance | 10 minutes | Advanced usage |

---

### 1. Zero-Code Auto-Instrumentation

Perfect for existing applications that already use Perplexity:

```python
from genops.providers.perplexity import auto_instrument

# Single line enables governance for all Perplexity operations
auto_instrument(
    team="your-team",
    project="existing-app",
    daily_budget_limit=100.0,
    governance_policy="advisory"
)

# Existing code works unchanged - now with governance!
import openai

client = openai.OpenAI(
    api_key=os.getenv("PERPLEXITY_API_KEY"),
    base_url="https://api.perplexity.ai"
)

# This now has automatic cost tracking and governance
response = client.chat.completions.create(
    model="sonar-pro",
    messages=[{"role": "user", "content": "Latest AI developments"}]
)
```

**Benefits:**
- Zero code changes to existing applications
- Automatic cost tracking and team attribution  
- Budget controls and governance policies
- Session management and performance monitoring

### 2. Direct Adapter Integration

For new applications or when you need more control:

```python
from genops.providers.perplexity import (
    GenOpsPerplexityAdapter, 
    PerplexityModel, 
    SearchContext
)

adapter = GenOpsPerplexityAdapter(
    team="search-team",
    project="content-research",
    daily_budget_limit=150.0
)

# Basic search with governance
result = adapter.search_with_governance(
    query="Sustainable energy innovations 2024",
    model=PerplexityModel.SONAR_PRO,
    search_context=SearchContext.HIGH,
    max_tokens=400,
    return_citations=True
)

print(f"Response: {result.response}")
print(f"Cost: ${result.cost:.6f}")
print(f"Citations: {len(result.citations)}")
```

### 3. Context Manager Pattern

For complex operations with session tracking:

```python
with adapter.track_search_session("market_research_2024") as session:
    # Multi-step research workflow
    background = adapter.search_with_governance(
        query="Market trends in renewable energy sector",
        model=PerplexityModel.SONAR_PRO,
        search_context=SearchContext.HIGH,
        session_id=session.session_id
    )
    
    competitors = adapter.search_with_governance(
        query="Leading companies in solar energy innovation",
        model=PerplexityModel.SONAR_PRO,
        search_context=SearchContext.MEDIUM,
        session_id=session.session_id
    )
    
    # Automatic session cost tracking and reporting
    print(f"Session cost: ${session.total_cost:.6f}")
    print(f"Total queries: {session.total_queries}")
```

### 4. Batch Processing Pattern

Efficient processing of multiple related queries:

```python
research_queries = [
    "AI adoption trends in healthcare 2024",
    "Machine learning applications in medical diagnosis",
    "Regulatory challenges for AI in healthcare",
    "Future of AI-powered drug discovery",
    "Ethical considerations in medical AI"
]

# Batch processing with optimization
results = adapter.batch_search_with_governance(
    queries=research_queries,
    model=PerplexityModel.SONAR,
    search_context=SearchContext.MEDIUM,
    batch_optimization=True,  # Reduces costs through intelligent batching
    research_topic="healthcare_ai_research"
)

# Analyze batch results
total_cost = sum(result.cost for result in results)
total_citations = sum(len(result.citations) for result in results)

print(f"Batch processing completed:")
print(f"  Queries processed: {len(results)}")
print(f"  Total cost: ${total_cost:.6f}")
print(f"  Average cost per query: ${total_cost / len(results):.6f}")
print(f"  Total citations: {total_citations}")
```

### 5. Async Pattern (Advanced)

For high-performance applications:

```python
import asyncio

async def async_search_workflow():
    async with adapter.async_track_search_session("concurrent_research") as session:
        # Concurrent searches for performance
        tasks = [
            adapter.async_search_with_governance(
                query="AI trends in fintech",
                session_id=session.session_id
            ),
            adapter.async_search_with_governance(
                query="Blockchain applications in banking",
                session_id=session.session_id
            ),
            adapter.async_search_with_governance(
                query="Regulatory landscape for digital currencies",
                session_id=session.session_id
            )
        ]
        
        results = await asyncio.gather(*tasks)
        return results

# Run async workflow
results = asyncio.run(async_search_workflow())
```

## Cost Management

### Understanding Perplexity's Dual Pricing Model

Perplexity charges both **token costs** and **request fees**:

**Token Costs (Model-based):**
- `sonar`: $1-1 per 1M input/output tokens
- `sonar-pro`: $3-15 per 1M input/output tokens  
- `sonar-reasoning`: $20-20 per 1M input/output tokens

**Request Fees (Context-based):**
- `LOW` context: $5 per 1,000 requests
- `MEDIUM` context: $8 per 1,000 requests
- `HIGH` context: $12 per 1,000 requests

### Cost Optimization Strategies

**1. Model Selection Optimization:**
```python
from genops.providers.perplexity_pricing import PerplexityPricingCalculator

calculator = PerplexityPricingCalculator()

# Compare costs for different models
models = ["sonar", "sonar-pro", "sonar-reasoning"]
query_tokens = 500

for model in models:
    cost = calculator.calculate_search_cost(
        model=model,
        tokens_used=query_tokens,
        search_context=SearchContext.MEDIUM
    )
    print(f"{model}: ${cost:.6f}")
```

**2. Context Optimization:**
```python
# Test different contexts for cost vs quality
contexts = [SearchContext.LOW, SearchContext.MEDIUM, SearchContext.HIGH]

for context in contexts:
    result = adapter.search_with_governance(
        query="Machine learning best practices",
        model=PerplexityModel.SONAR,
        search_context=context,
        max_tokens=200
    )
    
    print(f"{context.value}: ${result.cost:.6f} - {len(result.citations)} citations")
```

**3. Budget Management:**
```python
# Strict budget enforcement
adapter = GenOpsPerplexityAdapter(
    daily_budget_limit=50.0,
    governance_policy="enforced",  # Blocks operations when budget exceeded
    enable_cost_alerts=True
)

# Cost-aware search with budget checking
try:
    result = adapter.search_with_governance(
        query="Expensive research query",
        model=PerplexityModel.SONAR_PRO,
        search_context=SearchContext.HIGH,
        check_budget=True  # Validates budget before operation
    )
except BudgetExceededException as e:
    print(f"Operation blocked: {e}")
    # Implement fallback strategy
    result = adapter.search_with_governance(
        query="Same query but cost-optimized",
        model=PerplexityModel.SONAR,
        search_context=SearchContext.LOW
    )
```

**4. Cost Analytics and Forecasting:**
```python
# Get comprehensive cost analysis
analysis = adapter.get_search_cost_analysis(
    projected_queries=1000,  # Monthly volume
    model="sonar-pro",
    average_tokens_per_query=400
)

print("Cost Analysis:")
print(f"  Projected monthly cost: ${analysis['projected_total_cost']:.4f}")
print(f"  Cost per query: ${analysis['cost_per_query']:.6f}")

# Optimization recommendations
for opt in analysis['optimization_opportunities']:
    print(f"  💡 {opt['optimization_type']}: ${opt['potential_savings_total']:.4f} savings")
```

### Volume Pricing Strategies

**For High-Volume Applications:**
```python
# Implement intelligent query routing
def intelligent_search(query: str, urgency: str = "normal"):
    if urgency == "high":
        # High-quality for urgent requests
        model = PerplexityModel.SONAR_PRO
        context = SearchContext.HIGH
    elif urgency == "low":
        # Cost-optimized for non-urgent requests
        model = PerplexityModel.SONAR
        context = SearchContext.LOW
    else:
        # Balanced approach
        model = PerplexityModel.SONAR
        context = SearchContext.MEDIUM
    
    return adapter.search_with_governance(
        query=query,
        model=model,
        search_context=context
    )

# Usage-based model selection
result = intelligent_search("AI trends 2024", urgency="high")
```

## Advanced Features

### 1. Multi-Step Research Workflows

```python
class ResearchWorkflow:
    def __init__(self, adapter, topic: str):
        self.adapter = adapter
        self.topic = topic
        self.findings = {}
    
    def execute_research_pipeline(self):
        with self.adapter.track_search_session(f"research_{self.topic}") as session:
            # Step 1: Background research
            self.findings['background'] = self.adapter.search_with_governance(
                query=f"Background and overview of {self.topic}",
                model=PerplexityModel.SONAR_PRO,
                search_context=SearchContext.HIGH,
                session_id=session.session_id,
                research_phase="background"
            )
            
            # Step 2: Current challenges
            self.findings['challenges'] = self.adapter.search_with_governance(
                query=f"Current challenges and limitations in {self.topic}",
                model=PerplexityModel.SONAR_PRO,
                search_context=SearchContext.HIGH,
                session_id=session.session_id,
                research_phase="challenges"
            )
            
            # Step 3: Solutions and innovations
            self.findings['solutions'] = self.adapter.search_with_governance(
                query=f"Latest solutions and innovations in {self.topic}",
                model=PerplexityModel.SONAR_PRO,
                search_context=SearchContext.HIGH,
                session_id=session.session_id,
                research_phase="solutions"
            )
            
            # Step 4: Future trends
            self.findings['future'] = self.adapter.search_with_governance(
                query=f"Future trends and predictions for {self.topic}",
                model=PerplexityModel.SONAR_PRO,
                search_context=SearchContext.MEDIUM,
                session_id=session.session_id,
                research_phase="future"
            )
            
            return {
                'findings': self.findings,
                'session_cost': session.total_cost,
                'total_citations': sum(len(f.citations) for f in self.findings.values())
            }

# Usage
workflow = ResearchWorkflow(adapter, "sustainable AI computing")
research_report = workflow.execute_research_pipeline()
```

### 2. Citation Analysis and Quality Assessment

```python
def analyze_citation_quality(citations: List[Dict]) -> Dict[str, Any]:
    """Analyze citation sources for quality and domain distribution."""
    
    domain_analysis = {
        'academic': 0,
        'news': 0, 
        'technical': 0,
        'government': 0,
        'other': 0
    }
    
    quality_indicators = {
        'peer_reviewed': 0,
        'recent': 0,  # Less than 6 months old
        'authoritative': 0
    }
    
    academic_domains = {'arxiv.org', 'scholar.google.com', 'ieee.org', 'acm.org'}
    news_domains = {'reuters.com', 'bbc.com', 'techcrunch.com', 'wired.com'}
    technical_domains = {'github.com', 'stackoverflow.com', 'medium.com'}
    gov_domains = {'.gov', '.edu'}
    
    for citation in citations:
        url = citation.get('url', '').lower()
        
        # Domain classification
        if any(domain in url for domain in academic_domains):
            domain_analysis['academic'] += 1
            quality_indicators['peer_reviewed'] += 1
        elif any(domain in url for domain in news_domains):
            domain_analysis['news'] += 1
        elif any(domain in url for domain in technical_domains):
            domain_analysis['technical'] += 1
        elif any(domain in url for domain in gov_domains):
            domain_analysis['government'] += 1
            quality_indicators['authoritative'] += 1
        else:
            domain_analysis['other'] += 1
        
        # Recency analysis (if date available)
        if 'date' in citation:
            # Implementation depends on date format
            quality_indicators['recent'] += 1
    
    return {
        'domain_distribution': domain_analysis,
        'quality_score': sum(quality_indicators.values()),
        'total_citations': len(citations),
        'quality_percentage': (sum(quality_indicators.values()) / len(citations)) * 100
    }

# Usage with search results
result = adapter.search_with_governance(
    query="Climate change impact on renewable energy",
    model=PerplexityModel.SONAR_PRO,
    search_context=SearchContext.HIGH,
    return_citations=True
)

citation_analysis = analyze_citation_quality(result.citations)
print(f"Citation Quality Report:")
print(f"  Quality Score: {citation_analysis['quality_score']}/{citation_analysis['total_citations']}")
print(f"  Academic Sources: {citation_analysis['domain_distribution']['academic']}")
print(f"  Quality Percentage: {citation_analysis['quality_percentage']:.1f}%")
```

### 3. Domain Filtering and Source Control

```python
# Academic-only research
academic_result = adapter.search_with_governance(
    query="Machine learning interpretability methods research",
    model=PerplexityModel.SONAR_PRO,
    search_context=SearchContext.HIGH,
    search_domain_filter=[
        'arxiv.org',
        'scholar.google.com', 
        'ieee.org',
        'acm.org',
        'springer.com'
    ],
    max_tokens=500
)

# News and current events
news_result = adapter.search_with_governance(
    query="Latest AI industry developments",
    model=PerplexityModel.SONAR,
    search_context=SearchContext.MEDIUM,
    search_domain_filter=[
        'techcrunch.com',
        'venturebeat.com',
        'reuters.com',
        'bloomberg.com'
    ]
)

# Technical documentation
docs_result = adapter.search_with_governance(
    query="Python machine learning library comparison",
    model=PerplexityModel.SONAR,
    search_context=SearchContext.LOW,
    search_domain_filter=[
        'docs.python.org',
        'scikit-learn.org',
        'pytorch.org',
        'tensorflow.org'
    ]
)
```

### 4. Performance Monitoring and Optimization

```python
from genops.monitoring import PerformanceMonitor

# Enable performance monitoring
monitor = PerformanceMonitor(adapter)

with monitor.track_performance("search_performance_test"):
    # Measure search performance
    start_time = time.time()
    
    result = adapter.search_with_governance(
        query="Complex AI research query requiring extensive search",
        model=PerplexityModel.SONAR_PRO,
        search_context=SearchContext.HIGH
    )
    
    end_time = time.time()
    
    # Log performance metrics
    monitor.log_metrics({
        'search_latency': end_time - start_time,
        'token_efficiency': result.tokens_used / len(result.response),
        'cost_efficiency': result.cost / len(result.citations),
        'citation_quality': len([c for c in result.citations if 'arxiv' in c.get('url', '')])
    })

# Get performance insights
performance_report = monitor.get_performance_report()
print("Performance Report:")
print(f"  Average latency: {performance_report['avg_latency']:.2f}s")
print(f"  Cost efficiency: ${performance_report['avg_cost_per_token']:.8f}/token")
print(f"  Quality score: {performance_report['avg_citation_quality']:.1f}")
```

## Production Deployment

### 1. Enterprise Governance Patterns

```python
# Enterprise-grade adapter configuration
enterprise_adapter = GenOpsPerplexityAdapter(
    # Organization structure
    team="enterprise-ai-team",
    project="market-intelligence-platform",
    environment="production",
    customer_id="enterprise-corp-001",
    cost_center="strategic-research-division",
    
    # Strict governance
    governance_policy="strict",  # Maximum validation and controls
    daily_budget_limit=1000.0,
    monthly_budget_limit=25000.0,
    enable_cost_alerts=True,
    
    # Enterprise features
    enable_audit_trail=True,
    require_approval_for_high_cost=True,
    cost_approval_threshold=10.0,
    
    # Compliance settings
    data_classification="confidential",
    retention_policy="7_years",
    
    # Performance settings
    default_search_context=SearchContext.HIGH,
    enable_caching=True,
    cache_ttl_seconds=1800,  # 30 minutes
    
    # Monitoring and alerting
    enable_performance_monitoring=True,
    alert_on_budget_threshold=0.8,
    alert_on_performance_degradation=True,
    
    tags={
        "deployment": "production",
        "compliance_required": "true",
        "cost_attribution": "mandatory",
        "governance_level": "enterprise"
    }
)
```

### 2. Multi-Tenant Architecture

```python
class MultiTenantPerplexityService:
    def __init__(self):
        self.base_adapter = GenOpsPerplexityAdapter(
            team="platform-services",
            project="multi-tenant-search-service",
            environment="production"
        )
        self.tenant_configs = {}
    
    def register_tenant(self, tenant_id: str, config: Dict[str, Any]):
        """Register a new tenant with custom configuration."""
        self.tenant_configs[tenant_id] = {
            'budget_limit': config.get('budget_limit', 100.0),
            'governance_policy': config.get('governance_policy', 'enforced'),
            'allowed_models': config.get('allowed_models', ['sonar']),
            'cost_center': config.get('cost_center', f'tenant-{tenant_id}'),
            'tags': config.get('tags', {})
        }
    
    def search_for_tenant(self, tenant_id: str, query: str, **kwargs):
        """Execute search with tenant-specific governance."""
        if tenant_id not in self.tenant_configs:
            raise ValueError(f"Tenant {tenant_id} not registered")
        
        tenant_config = self.tenant_configs[tenant_id]
        
        # Apply tenant-specific settings
        kwargs.update({
            'customer_id': tenant_id,
            'cost_center': tenant_config['cost_center'],
            'governance_policy': tenant_config['governance_policy'],
            'tags': {**kwargs.get('tags', {}), **tenant_config['tags']}
        })
        
        # Budget validation
        tenant_usage = self.get_tenant_usage(tenant_id)
        if tenant_usage >= tenant_config['budget_limit']:
            raise BudgetExceededException(f"Tenant {tenant_id} budget exceeded")
        
        # Execute search with tenant context
        return self.base_adapter.search_with_governance(
            query=query,
            **kwargs
        )
    
    def get_tenant_usage(self, tenant_id: str) -> float:
        """Get current usage for a tenant."""
        cost_summary = self.base_adapter.get_cost_summary()
        return cost_summary.get('customer_costs', {}).get(tenant_id, 0.0)

# Usage
service = MultiTenantPerplexityService()

# Register tenants
service.register_tenant('client-a', {
    'budget_limit': 500.0,
    'governance_policy': 'enforced',
    'allowed_models': ['sonar', 'sonar-pro'],
    'tags': {'tier': 'enterprise', 'region': 'us-east'}
})

# Search for tenant
result = service.search_for_tenant(
    'client-a',
    "Market analysis for renewable energy sector",
    model=PerplexityModel.SONAR_PRO
)
```

### 3. Error Handling and Resilience

```python
from genops.resilience import CircuitBreaker, RetryPolicy

class ResilientPerplexityAdapter:
    def __init__(self, adapter: GenOpsPerplexityAdapter):
        self.adapter = adapter
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            timeout=60,
            expected_exception=Exception
        )
        self.retry_policy = RetryPolicy(
            max_retries=3,
            backoff_factor=2.0,
            max_delay=30.0
        )
    
    @circuit_breaker
    @retry_policy
    def resilient_search(self, query: str, **kwargs):
        """Search with circuit breaker and retry logic."""
        try:
            return self.adapter.search_with_governance(query, **kwargs)
        except RateLimitException as e:
            # Handle rate limiting with exponential backoff
            wait_time = min(60, 2 ** kwargs.get('retry_attempt', 0))
            time.sleep(wait_time)
            raise
        except NetworkTimeoutException as e:
            # Log timeout and retry
            logger.warning(f"Network timeout for query: {query[:50]}")
            raise
        except BudgetExceededException as e:
            # Don't retry budget errors
            logger.error(f"Budget exceeded: {e}")
            raise BudgetExceededException("Budget limit reached") from None
    
    def search_with_fallback(self, query: str, **kwargs):
        """Search with fallback strategies."""
        try:
            # Try primary search
            return self.resilient_search(query, **kwargs)
        except Exception as e:
            logger.warning(f"Primary search failed: {e}")
            
            # Fallback 1: Simpler model
            if kwargs.get('model') == PerplexityModel.SONAR_PRO:
                kwargs['model'] = PerplexityModel.SONAR
                try:
                    return self.resilient_search(query, **kwargs)
                except Exception:
                    pass
            
            # Fallback 2: Lower context
            if kwargs.get('search_context') == SearchContext.HIGH:
                kwargs['search_context'] = SearchContext.MEDIUM
                try:
                    return self.resilient_search(query, **kwargs)
                except Exception:
                    pass
            
            # Fallback 3: Cached results or error response
            return self._get_fallback_response(query)
    
    def _get_fallback_response(self, query: str):
        """Return cached results or graceful error response."""
        # Check cache first
        cached_result = self._get_cached_result(query)
        if cached_result:
            return cached_result
        
        # Return graceful error response
        return SearchResult(
            response=f"Unable to search for '{query}' at this time. Please try again later.",
            cost=0.0,
            tokens_used=0,
            citations=[],
            error_mode=True
        )

# Usage
resilient_adapter = ResilientPerplexityAdapter(enterprise_adapter)
result = resilient_adapter.search_with_fallback(
    "Complex query that might fail",
    model=PerplexityModel.SONAR_PRO,
    search_context=SearchContext.HIGH
)
```

### 4. Monitoring and Alerting Integration

```python
# Prometheus metrics integration
from prometheus_client import Counter, Histogram, Gauge

search_counter = Counter('perplexity_searches_total', 'Total searches', ['team', 'model'])
search_duration = Histogram('perplexity_search_duration_seconds', 'Search duration')
search_cost = Histogram('perplexity_search_cost_dollars', 'Search cost in dollars')
budget_utilization = Gauge('perplexity_budget_utilization_ratio', 'Budget utilization', ['team'])

class MonitoredPerplexityAdapter:
    def __init__(self, adapter: GenOpsPerplexityAdapter):
        self.adapter = adapter
    
    def search_with_monitoring(self, query: str, **kwargs):
        """Search with comprehensive monitoring."""
        start_time = time.time()
        
        try:
            # Execute search
            result = self.adapter.search_with_governance(query, **kwargs)
            
            # Record metrics
            search_counter.labels(
                team=self.adapter.team,
                model=kwargs.get('model', 'unknown').value
            ).inc()
            
            search_duration.observe(time.time() - start_time)
            search_cost.observe(float(result.cost))
            
            # Update budget utilization
            cost_summary = self.adapter.get_cost_summary()
            budget_utilization.labels(team=self.adapter.team).set(
                cost_summary['daily_budget_utilization'] / 100
            )
            
            # Custom alerts
            if result.cost > 1.0:  # High cost alert
                self._send_alert(f"High cost search: ${result.cost:.4f} for query: {query[:50]}")
            
            return result
            
        except Exception as e:
            # Error metrics
            search_counter.labels(
                team=self.adapter.team,
                model='error'
            ).inc()
            
            self._send_alert(f"Search error: {e}")
            raise

# DataDog integration
import datadog

def setup_datadog_monitoring(adapter: GenOpsPerplexityAdapter):
    """Setup DataDog monitoring for Perplexity operations."""
    
    @datadog.statsd.timed('perplexity.search.duration')
    def monitored_search(query: str, **kwargs):
        result = adapter.search_with_governance(query, **kwargs)
        
        # Custom metrics
        datadog.statsd.increment('perplexity.search.count', tags=[
            f'team:{adapter.team}',
            f'model:{kwargs.get("model", "unknown")}',
            f'environment:{adapter.environment}'
        ])
        
        datadog.statsd.histogram('perplexity.search.cost', float(result.cost), tags=[
            f'team:{adapter.team}'
        ])
        
        datadog.statsd.histogram('perplexity.search.tokens', result.tokens_used, tags=[
            f'team:{adapter.team}'
        ])
        
        return result
    
    return monitored_search
```

## Troubleshooting

### Common Issues and Solutions

**1. API Key Issues**
```python
# Validate API key format and connectivity
from genops.providers.perplexity_validation import validate_setup, print_validation_result

result = validate_setup()
print_validation_result(result)

# Common fixes:
# - Ensure key starts with 'pplx-'
# - Check key is active at https://www.perplexity.ai/settings/api  
# - Verify environment variable: echo $PERPLEXITY_API_KEY
```

**2. Budget Exceeded Errors**
```python
try:
    result = adapter.search_with_governance(query="expensive query")
except BudgetExceededException as e:
    print(f"Budget exceeded: {e}")
    
    # Check current usage
    cost_summary = adapter.get_cost_summary()
    print(f"Daily usage: ${cost_summary['daily_costs']:.4f}")
    print(f"Daily limit: ${cost_summary['daily_budget_limit']}")
    
    # Options:
    # 1. Increase budget limit
    adapter.daily_budget_limit = 100.0
    
    # 2. Use cost-optimized search
    result = adapter.search_with_governance(
        query="same query but cheaper",
        model=PerplexityModel.SONAR,  # Cheaper model
        search_context=SearchContext.LOW  # Cheaper context
    )
```

**3. Rate Limiting**
```python
import time
from genops.exceptions import RateLimitException

def search_with_backoff(adapter, query: str, **kwargs):
    """Search with exponential backoff for rate limits."""
    max_retries = 3
    base_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            return adapter.search_with_governance(query, **kwargs)
        except RateLimitException as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"Rate limited. Waiting {delay}s before retry...")
                time.sleep(delay)
            else:
                raise
```

**4. Network Connectivity Issues**
```python
import requests
from genops.exceptions import NetworkException

def test_perplexity_connectivity():
    """Test network connectivity to Perplexity API."""
    try:
        response = requests.get("https://api.perplexity.ai/health", timeout=10)
        if response.status_code == 200:
            print("✅ Perplexity API is reachable")
            return True
        else:
            print(f"⚠️ Perplexity API returned {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"❌ Network error: {e}")
        return False

# Usage
if not test_perplexity_connectivity():
    print("Check your internet connection and proxy settings")
```

**5. Import and Dependency Issues**
```python
# Check all dependencies
def check_dependencies():
    """Check if all required dependencies are available."""
    dependencies = [
        ('genops', 'GenOps core package'),
        ('openai', 'OpenAI client (required for Perplexity)'),
        ('opentelemetry', 'OpenTelemetry SDK'),
        ('pydantic', 'Configuration validation (optional)')
    ]
    
    for package, description in dependencies:
        try:
            __import__(package)
            print(f"✅ {package}: {description}")
        except ImportError:
            print(f"❌ {package}: {description}")
            print(f"   Install with: pip install {package}")

check_dependencies()
```

### Performance Troubleshooting

**1. Slow Search Performance**
```python
# Profile search performance
import time

def profile_search_performance(adapter, query: str):
    """Profile search performance components."""
    
    # Measure total time
    total_start = time.time()
    
    # Pre-request validation
    validation_start = time.time()
    # (Internal validation happens here)
    validation_time = time.time() - validation_start
    
    # API request time
    api_start = time.time()
    result = adapter.search_with_governance(query)
    api_time = time.time() - api_start
    
    # Post-processing time
    processing_start = time.time()
    # (Citation processing, cost calculation, telemetry)
    processing_time = time.time() - processing_start
    
    total_time = time.time() - total_start
    
    print(f"Performance Profile:")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  API time: {api_time:.3f}s ({api_time/total_time*100:.1f}%)")
    print(f"  Validation: {validation_time:.3f}s")
    print(f"  Processing: {processing_time:.3f}s")
    
    # Performance recommendations
    if api_time > 5.0:
        print("⚠️ Slow API response. Consider using lower search context or simpler model.")
    if processing_time > 0.5:
        print("⚠️ Slow post-processing. Check citation processing settings.")

# Usage
profile_search_performance(adapter, "Complex research query")
```

**2. High Cost Issues**
```python
# Analyze cost drivers
def analyze_cost_efficiency(adapter, queries: List[str]):
    """Analyze what's driving high costs."""
    
    cost_breakdown = {
        'token_costs': 0.0,
        'request_costs': 0.0,
        'total_tokens': 0,
        'total_requests': 0
    }
    
    for query in queries:
        result = adapter.search_with_governance(query)
        
        # Get detailed cost breakdown
        from genops.providers.perplexity_pricing import PerplexityPricingCalculator
        calculator = PerplexityPricingCalculator()
        
        breakdown = calculator.get_detailed_cost_breakdown(
            model=result.model_used,
            tokens_used=result.tokens_used,
            search_context=result.search_context
        )
        
        cost_breakdown['token_costs'] += breakdown.token_cost
        cost_breakdown['request_costs'] += breakdown.request_cost
        cost_breakdown['total_tokens'] += result.tokens_used
        cost_breakdown['total_requests'] += 1
    
    print("Cost Analysis:")
    print(f"  Token costs: ${cost_breakdown['token_costs']:.6f}")
    print(f"  Request costs: ${cost_breakdown['request_costs']:.6f}")
    print(f"  Average tokens per query: {cost_breakdown['total_tokens'] / len(queries):.0f}")
    
    # Optimization suggestions
    token_ratio = cost_breakdown['token_costs'] / (cost_breakdown['token_costs'] + cost_breakdown['request_costs'])
    if token_ratio > 0.7:
        print("💡 Token costs dominate. Consider using a cheaper model or reducing max_tokens.")
    else:
        print("💡 Request costs dominate. Consider using lower search context or batching queries.")

# Usage
test_queries = [
    "AI trends 2024",
    "Machine learning best practices", 
    "Future of automation"
]
analyze_cost_efficiency(adapter, test_queries)
```

### Debug Mode and Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('genops.providers.perplexity')

# Create adapter with debug mode
debug_adapter = GenOpsPerplexityAdapter(
    team="debug-team",
    project="troubleshooting",
    debug_mode=True,  # Enables detailed logging
    log_requests=True,  # Log all API requests
    log_responses=True  # Log all API responses (truncated)
)

# Debug search with detailed logging
result = debug_adapter.search_with_governance(
    query="Debug test query",
    model=PerplexityModel.SONAR,
    debug_context={'test_id': 'debug_001'}
)
```

## API Reference

### Core Classes

#### GenOpsPerplexityAdapter

Main adapter class for Perplexity AI integration with GenOps governance.

```python
class GenOpsPerplexityAdapter:
    def __init__(
        self,
        # Basic identification
        team: str,
        project: str = "default",
        environment: str = "development",
        
        # Enterprise attribution  
        customer_id: Optional[str] = None,
        cost_center: Optional[str] = None,
        
        # Budget management
        daily_budget_limit: float = 100.0,
        monthly_budget_limit: float = 3000.0,
        governance_policy: str = "advisory",  # advisory, enforced, strict
        enable_cost_alerts: bool = False,
        
        # Search defaults
        default_model: str = "sonar",
        default_search_context: str = "medium",
        max_tokens_default: int = 500,
        
        # Performance settings
        enable_caching: bool = False,
        cache_ttl_seconds: int = 300,
        retry_attempts: int = 3,
        timeout_seconds: int = 30,
        
        # Custom tags and metadata
        tags: Optional[Dict[str, str]] = None,
        
        # Advanced configuration
        debug_mode: bool = False,
        enable_telemetry: bool = True,
        telemetry_endpoint: Optional[str] = None
    )
```

#### Methods

**search_with_governance()**
```python
def search_with_governance(
    self,
    query: str,
    model: Union[PerplexityModel, str] = None,
    search_context: Union[SearchContext, str] = None,
    session_id: Optional[str] = None,
    max_tokens: int = None,
    return_citations: bool = True,
    
    # Governance options
    customer_id: Optional[str] = None,
    cost_center: Optional[str] = None,
    governance_tags: Optional[Dict[str, str]] = None,
    
    # Search filtering
    search_domain_filter: Optional[List[str]] = None,
    exclude_domains: Optional[List[str]] = None,
    
    # Performance options
    timeout: Optional[int] = None,
    enable_caching: Optional[bool] = None,
    cache_key: Optional[str] = None
) -> SearchResult
```

**batch_search_with_governance()**
```python
def batch_search_with_governance(
    self,
    queries: List[str],
    model: Union[PerplexityModel, str] = None,
    search_context: Union[SearchContext, str] = None,
    batch_optimization: bool = True,
    
    # Governance options
    session_id: Optional[str] = None,
    research_topic: Optional[str] = None,
    
    # Performance options
    max_concurrent: int = 5,
    batch_delay: float = 1.0
) -> List[SearchResult]
```

**track_search_session()**
```python
@contextmanager
def track_search_session(
    self,
    session_name: str,
    session_metadata: Optional[Dict[str, Any]] = None
) -> SearchSession
```

**Cost Management Methods**
```python
def get_cost_summary(self) -> Dict[str, Any]

def get_search_cost_analysis(
    self,
    projected_queries: int,
    model: str = "sonar",
    average_tokens_per_query: int = 400
) -> Dict[str, Any]

def reset_daily_budget(self) -> None

def set_budget_alert_threshold(self, threshold: float) -> None
```

### Data Classes

#### SearchResult
```python
@dataclass
class SearchResult:
    response: str                           # AI-generated response
    cost: Decimal                          # Total cost (tokens + requests)
    tokens_used: int                       # Number of tokens consumed
    citations: List[Dict[str, Any]]        # Source citations
    search_time_seconds: float             # Time taken for search
    model_used: str                        # Model that processed the request
    search_context: str                    # Context level used
    session_id: Optional[str] = None       # Session identifier
    governance_applied: bool = True        # Whether governance was applied
    cache_hit: bool = False                # Whether result came from cache
    error_mode: bool = False               # Whether this is an error response
```

#### SearchSession  
```python
@dataclass 
class SearchSession:
    session_id: str                        # Unique session identifier
    session_name: str                      # Human-readable session name
    total_cost: Decimal                    # Accumulated session cost
    total_queries: int                     # Number of queries in session
    start_time: datetime                   # Session start timestamp
    end_time: Optional[datetime] = None    # Session end timestamp
    metadata: Dict[str, Any] = None        # Custom session metadata
```

### Enums

#### PerplexityModel
```python
class PerplexityModel(Enum):
    SONAR = "sonar"                        # Cost-effective general search
    SONAR_PRO = "sonar-pro"                # Enhanced accuracy and citations
    SONAR_REASONING = "sonar-reasoning"     # Advanced reasoning capabilities
    SONAR_REASONING_PRO = "sonar-reasoning-pro"  # Premium reasoning model
```

#### SearchContext
```python  
class SearchContext(Enum):
    LOW = "low"           # Basic search, $5/1K requests
    MEDIUM = "medium"     # Balanced approach, $8/1K requests  
    HIGH = "high"         # Comprehensive search, $12/1K requests
```

### Utility Functions

#### Auto-instrumentation
```python
def auto_instrument(
    team: str,
    project: str = "default",
    environment: str = "development",
    daily_budget_limit: float = 50.0,
    governance_policy: str = "advisory",
    **kwargs
) -> GenOpsPerplexityAdapter
```

#### Validation
```python
from genops.providers.perplexity_validation import (
    validate_setup,
    print_validation_result,
    interactive_setup_wizard
)

def validate_setup() -> ValidationResult
def print_validation_result(result: ValidationResult) -> None
def interactive_setup_wizard() -> Dict[str, Any]
```

#### Pricing Utilities
```python
from genops.providers.perplexity_pricing import PerplexityPricingCalculator

calculator = PerplexityPricingCalculator()

def calculate_search_cost(
    model: str,
    tokens_used: int, 
    search_context: SearchContext
) -> Decimal

def estimate_search_cost(
    model: str,
    estimated_tokens: int,
    search_context: SearchContext  
) -> Decimal

def get_detailed_cost_breakdown(
    model: str,
    tokens_used: int,
    search_context: SearchContext
) -> CostBreakdown
```

## Examples

### Complete Working Examples

The `examples/perplexity/` directory contains comprehensive examples:

1. **[setup_validation.py](../../examples/perplexity/setup_validation.py)** - Validate your setup (2 min)
2. **[basic_search.py](../../examples/perplexity/basic_search.py)** - Basic real-time search (5 min)  
3. **[auto_instrumentation.py](../../examples/perplexity/auto_instrumentation.py)** - Zero-code integration (3 min)
4. **[advanced_search.py](../../examples/perplexity/advanced_search.py)** - Advanced patterns (15 min)
5. **[cost_optimization.py](../../examples/perplexity/cost_optimization.py)** - Cost optimization (10 min)
6. **[production_patterns.py](../../examples/perplexity/production_patterns.py)** - Production deployment (20 min)
7. **[interactive_setup_wizard.py](../../examples/perplexity/interactive_setup_wizard.py)** - Guided setup (10 min)

### Quick Example Snippets

**Basic Search:**
```bash
curl -O https://raw.githubusercontent.com/KoshiHQ/GenOps-AI/main/examples/perplexity/basic_search.py
python basic_search.py
```

**Cost Optimization:**
```bash
curl -O https://raw.githubusercontent.com/KoshiHQ/GenOps-AI/main/examples/perplexity/cost_optimization.py  
python cost_optimization.py
```

**Production Patterns:**
```bash
curl -O https://raw.githubusercontent.com/KoshiHQ/GenOps-AI/main/examples/perplexity/production_patterns.py
python production_patterns.py
```

## Support and Community

### Documentation
- **[5-Minute Quickstart](../perplexity-quickstart.md)** - Get started in under 5 minutes
- **[Cost Optimization Guide](cost-optimization/perplexity.md)** - Master dual pricing model  
- **[Production Deployment Guide](production/perplexity.md)** - Enterprise patterns

### Community Resources
- **GitHub Issues**: [Report bugs and request features](https://github.com/KoshiHQ/GenOps-AI/issues)
- **Discussions**: [Community Q&A and best practices](https://github.com/KoshiHQ/GenOps-AI/discussions)
- **Examples**: [Browse 20+ working examples](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/perplexity)

### Enterprise Support
- **Professional Services**: Implementation assistance and custom integration
- **Training Programs**: Team training on GenOps best practices
- **Priority Support**: Dedicated support channels for enterprise customers

---

**🎉 You now have complete Perplexity AI integration with governance!** 

Cost tracking, team attribution, and budget controls work automatically across all your searches, with comprehensive monitoring and optimization capabilities.