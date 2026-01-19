"""
Cost attribution and tracking in Grafana Tempo.

This example demonstrates:
- Cost tracking across multiple AI providers
- Team and customer cost attribution
- Cost aggregation and analysis via TraceQL
- Budget tracking patterns

Prerequisites:
    - Tempo running at http://localhost:3200
    - OpenAI API key (for cost simulation)
"""

import time
import random
from typing import Dict, Any
from opentelemetry import trace

from genops import track_usage
from genops.integrations.tempo import configure_tempo


def simulate_ai_operation(
    provider: str,
    model: str,
    tokens: int,
    cost_per_1k_tokens: float
) -> Dict[str, Any]:
    """
    Simulate an AI operation with cost tracking.

    Args:
        provider: AI provider name (e.g., "openai", "anthropic")
        model: Model name
        tokens: Total tokens used
        cost_per_1k_tokens: Cost per 1000 tokens

    Returns:
        Operation result with cost information
    """
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span(f"{provider}_operation") as span:
        # Set cost attributes
        total_cost = (tokens / 1000) * cost_per_1k_tokens

        span.set_attribute("genops.provider", provider)
        span.set_attribute("genops.model", model)
        span.set_attribute("genops.cost.total_tokens", tokens)
        span.set_attribute("genops.cost.total_cost", total_cost)
        span.set_attribute("genops.cost.currency", "USD")

        # Breakdown by token type
        prompt_tokens = int(tokens * 0.6)
        completion_tokens = tokens - prompt_tokens

        span.set_attribute("genops.cost.prompt_tokens", prompt_tokens)
        span.set_attribute("genops.cost.completion_tokens", completion_tokens)

        # Simulate operation time
        time.sleep(random.uniform(0.1, 0.3))

        return {
            "provider": provider,
            "model": model,
            "tokens": tokens,
            "cost": total_cost,
            "status": "success"
        }


def main():
    """
    Demonstrate cost attribution patterns in Tempo.
    """
    print("=" * 70)
    print("Grafana Tempo Cost Attribution Example")
    print("=" * 70)
    print()

    # Configure Tempo
    print("Configuring Tempo for cost tracking...")
    configure_tempo(
        endpoint="http://localhost:3200",
        service_name="cost-attribution-example",
        environment="development"
    )
    print("✅ Tempo configured\n")

    # ========================================================================
    # Scenario 1: Single Team Cost Tracking
    # ========================================================================

    print("=" * 70)
    print("Scenario 1: Single Team Cost Tracking")
    print("=" * 70)
    print()

    @track_usage(
        team="customer-support",
        project="ai-chatbot",
        feature="customer-query"
    )
    def customer_support_query():
        """Customer support AI query."""
        return simulate_ai_operation(
            provider="openai",
            model="gpt-4",
            tokens=1500,
            cost_per_1k_tokens=0.03
        )

    print("Executing customer support queries...")
    for i in range(3):
        result = customer_support_query()
        print(f"  Query {i+1}: {result['tokens']} tokens, ${result['cost']:.4f}")

    print()

    # ========================================================================
    # Scenario 2: Multi-Customer Cost Attribution
    # ========================================================================

    print("=" * 70)
    print("Scenario 2: Multi-Customer Cost Attribution")
    print("=" * 70)
    print()

    customers = ["acme-corp", "globex-inc", "initech-ltd"]

    @track_usage(
        team="sales",
        project="ai-sales-assistant"
    )
    def sales_assistant_query(customer_id: str):
        """Sales assistant query with customer attribution."""
        tracer = trace.get_tracer(__name__)

        with tracer.start_as_current_span("sales_query") as span:
            span.set_attribute("customer_id", customer_id)

            return simulate_ai_operation(
                provider="anthropic",
                model="claude-3-sonnet",
                tokens=random.randint(800, 2000),
                cost_per_1k_tokens=0.015
            )

    print("Executing sales queries for multiple customers...")
    for customer in customers:
        result = sales_assistant_query(customer)
        print(f"  {customer}: {result['tokens']} tokens, ${result['cost']:.4f}")

    print()

    # ========================================================================
    # Scenario 3: Multi-Provider Cost Comparison
    # ========================================================================

    print("=" * 70)
    print("Scenario 3: Multi-Provider Cost Comparison")
    print("=" * 70)
    print()

    @track_usage(
        team="ml-research",
        project="model-evaluation",
        feature="benchmark"
    )
    def run_multi_provider_benchmark():
        """Run same query across multiple providers."""
        providers = [
            ("openai", "gpt-4", 0.03),
            ("anthropic", "claude-3-sonnet", 0.015),
            ("google", "gemini-pro", 0.00125),
        ]

        results = []

        for provider, model, cost_per_1k in providers:
            result = simulate_ai_operation(
                provider=provider,
                model=model,
                tokens=1200,  # Same tokens for comparison
                cost_per_1k_tokens=cost_per_1k
            )
            results.append(result)

        return results

    print("Running multi-provider benchmark...")
    benchmark_results = run_multi_provider_benchmark()

    for result in benchmark_results:
        print(f"  {result['provider']:12} ({result['model']:20}): ${result['cost']:.4f}")

    print()

    # ========================================================================
    # Scenario 4: Budget-Constrained Operations
    # ========================================================================

    print("=" * 70)
    print("Scenario 4: Budget-Constrained Operations")
    print("=" * 70)
    print()

    MONTHLY_BUDGET = 1000.0  # $1000/month
    current_spend = 0.0

    @track_usage(
        team="content-generation",
        project="blog-writer",
        feature="article-generation"
    )
    def generate_content(budget_remaining: float):
        """Generate content within budget constraints."""
        tracer = trace.get_tracer(__name__)

        with tracer.start_as_current_span("content_generation") as span:
            # Set budget attributes
            span.set_attribute("genops.budget.monthly_limit", MONTHLY_BUDGET)
            span.set_attribute("genops.budget.remaining", budget_remaining)

            # Choose model based on budget
            if budget_remaining > 100:
                provider, model, tokens, cost_rate = "openai", "gpt-4", 2000, 0.03
            else:
                provider, model, tokens, cost_rate = "google", "gemini-pro", 2000, 0.00125

            span.set_attribute("genops.budget.model_selection", model)

            result = simulate_ai_operation(provider, model, tokens, cost_rate)

            return result

    print("Generating content with budget awareness...")
    budget_remaining = 150.0

    for i in range(3):
        result = generate_content(budget_remaining)
        budget_remaining -= result["cost"]

        print(f"  Article {i+1}: {result['model']:20} ${result['cost']:.4f} (remaining: ${budget_remaining:.2f})")

    print()

    # ========================================================================
    # Scenario 5: Cost Center Attribution
    # ========================================================================

    print("=" * 70)
    print("Scenario 5: Cost Center Attribution")
    print("=" * 70)
    print()

    @track_usage(
        team="engineering",
        project="code-assistant",
        cost_center="R&D",
        feature="code-generation"
    )
    def engineering_code_assistant():
        """Engineering team code generation."""
        return simulate_ai_operation(
            provider="openai",
            model="gpt-4",
            tokens=1800,
            cost_per_1k_tokens=0.03
        )

    @track_usage(
        team="marketing",
        project="content-assistant",
        cost_center="Marketing",
        feature="content-generation"
    )
    def marketing_content_assistant():
        """Marketing team content generation."""
        return simulate_ai_operation(
            provider="anthropic",
            model="claude-3-sonnet",
            tokens=1500,
            cost_per_1k_tokens=0.015
        )

    print("Executing operations for different cost centers...")

    eng_result = engineering_code_assistant()
    print(f"  R&D Cost Center: ${eng_result['cost']:.4f}")

    mkt_result = marketing_content_assistant()
    print(f"  Marketing Cost Center: ${mkt_result['cost']:.4f}")

    print()

    # Wait for spans to export
    print("⏳ Waiting for spans to export to Tempo...")
    time.sleep(2)

    # ========================================================================
    # Query Examples for Cost Analysis
    # ========================================================================

    print("=" * 70)
    print("Cost Analysis via TraceQL")
    print("=" * 70)
    print("""
Now query your cost data in Tempo using TraceQL:

1. **Total Cost by Team**
   {.team = "customer-support"} | sum(.genops.cost.total_cost) by (.team)

2. **Cost by Customer**
   {} | sum(.genops.cost.total_cost) by (.customer_id)

3. **High Cost Operations**
   {.genops.cost.total_cost > 0.05}

4. **Provider Cost Comparison**
   {} | avg(.genops.cost.total_cost) by (.genops.provider)

5. **Budget Utilization**
   {.genops.budget.monthly_limit > 0} | rate()

6. **Cost Center Breakdown**
   {} | sum(.genops.cost.total_cost) by (.cost_center)

7. **Expensive Slow Operations**
   {duration > 500ms && .genops.cost.total_cost > 0.04}

8. **Token Usage by Model**
   {} | sum(.genops.cost.total_tokens) by (.genops.model)

Run these queries at:
  http://localhost:3000 → Explore → Tempo → TraceQL

Or via CLI:
  curl "http://localhost:3200/api/search?q={.team=\\"customer-support\\"}&limit=10"
    """)

    print("=" * 70)
    print("Cost Attribution Patterns Summary")
    print("=" * 70)
    print("""
Key Cost Attribution Patterns:

1. **Team Attribution**
   - Track costs by team for chargeback/showback
   - Identify high-spending teams

2. **Customer Attribution**
   - Per-customer cost tracking for billing
   - Customer profitability analysis

3. **Multi-Provider Tracking**
   - Compare costs across OpenAI, Anthropic, Google, etc.
   - Optimize provider selection

4. **Budget Management**
   - Track against budget limits
   - Model selection based on budget remaining

5. **Cost Center Allocation**
   - Finance-aligned cost tracking
   - Departmental budget attribution

All cost data flows to Tempo as trace attributes,
queryable via TraceQL for powerful analysis!
    """)

    print("=" * 70)
    print("✅ Cost attribution example completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
