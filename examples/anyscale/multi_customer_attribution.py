#!/usr/bin/env python3
"""
Multi-Customer Cost Attribution - 30 Minute Tutorial

Learn how to track and attribute costs across multiple customers in SaaS applications.

Demonstrates:
- Per-customer cost tracking
- Team and project-level attribution
- Cost center allocation
- Feature-level cost breakdown
- Monthly billing report generation

Prerequisites:
- export ANYSCALE_API_KEY='your-api-key'
- pip install genops-ai
"""

import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from genops.providers.anyscale import calculate_completion_cost, instrument_anyscale

# Check API key
if not os.getenv("ANYSCALE_API_KEY"):
    print("❌ ERROR: ANYSCALE_API_KEY not set")
    exit(1)

print("=" * 70)
print("GenOps Anyscale - Multi-Customer Cost Attribution")
print("=" * 70 + "\n")


# Cost tracking data structure
@dataclass
class CostTracker:
    """Track costs across multiple dimensions."""

    by_customer: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    by_team: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    by_project: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    by_feature: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    by_model: dict[str, float] = field(default_factory=lambda: defaultdict(float))

    total_requests: int = 0
    total_cost: float = 0.0

    def record_cost(
        self,
        cost: float,
        customer_id: str = None,
        team: str = None,
        project: str = None,
        feature: str = None,
        model: str = None,
    ):
        """Record cost with all attribution dimensions."""
        self.total_requests += 1
        self.total_cost += cost

        if customer_id:
            self.by_customer[customer_id] += cost
        if team:
            self.by_team[team] += cost
        if project:
            self.by_project[project] += cost
        if feature:
            self.by_feature[feature] += cost
        if model:
            self.by_model[model] += cost

    def print_report(self):
        """Print comprehensive cost report."""
        print("=" * 70)
        print(f"COST ATTRIBUTION REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("=" * 70)
        print("\n📊 OVERALL SUMMARY:")
        print(f"   Total Requests: {self.total_requests}")
        print(f"   Total Cost: ${self.total_cost:.6f}")
        print(f"   Avg Cost/Request: ${self.total_cost / self.total_requests:.8f}")

        # By Customer
        if self.by_customer:
            print("\n💼 BY CUSTOMER:")
            sorted_customers = sorted(
                self.by_customer.items(), key=lambda x: x[1], reverse=True
            )
            for customer, cost in sorted_customers:
                pct = (cost / self.total_cost) * 100
                print(f"   {customer:30s} ${cost:10.6f} ({pct:5.1f}%)")

        # By Team
        if self.by_team:
            print("\n👥 BY TEAM:")
            sorted_teams = sorted(
                self.by_team.items(), key=lambda x: x[1], reverse=True
            )
            for team, cost in sorted_teams:
                pct = (cost / self.total_cost) * 100
                print(f"   {team:30s} ${cost:10.6f} ({pct:5.1f}%)")

        # By Project
        if self.by_project:
            print("\n📁 BY PROJECT:")
            sorted_projects = sorted(
                self.by_project.items(), key=lambda x: x[1], reverse=True
            )
            for project, cost in sorted_projects:
                pct = (cost / self.total_cost) * 100
                print(f"   {project:30s} ${cost:10.6f} ({pct:5.1f}%)")

        # By Feature
        if self.by_feature:
            print("\n🎯 BY FEATURE:")
            sorted_features = sorted(
                self.by_feature.items(), key=lambda x: x[1], reverse=True
            )
            for feature, cost in sorted_features:
                pct = (cost / self.total_cost) * 100
                print(f"   {feature:30s} ${cost:10.6f} ({pct:5.1f}%)")

        # By Model
        if self.by_model:
            print("\n🤖 BY MODEL:")
            sorted_models = sorted(
                self.by_model.items(), key=lambda x: x[1], reverse=True
            )
            for model, cost in sorted_models:
                pct = (cost / self.total_cost) * 100
                print(f"   {model:30s} ${cost:10.6f} ({pct:5.1f}%)")

        print("\n" + "=" * 70)


# Initialize cost tracker
cost_tracker = CostTracker()

# Create SaaS platform adapter
adapter = instrument_anyscale(
    team="saas-platform", project="ai-features", environment="production"
)

print("Simulating SaaS platform with multiple customers...\n")


# Scenario 1: Enterprise Customer - High Volume
print("=" * 70)
print("SCENARIO 1: Enterprise Customer (High Volume)")
print("=" * 70 + "\n")

enterprise_customer = "acme-corp-enterprise"

print(f"Processing requests for: {enterprise_customer}")
print("Features: Chat completion, Document analysis, Summarization\n")

# Chat completion requests
for i in range(5):
    response = adapter.completion_create(
        model="meta-llama/Llama-2-70b-chat-hf",  # Premium model
        messages=[
            {
                "role": "user",
                "content": f"Enterprise query {i + 1}: Analyze quarterly results",
            }
        ],
        max_tokens=200,
        customer_id=enterprise_customer,
        feature="chat-completion",
        cost_center="Enterprise-Sales",
    )

    usage = response["usage"]
    cost = calculate_completion_cost(
        model="meta-llama/Llama-2-70b-chat-hf",
        input_tokens=usage["prompt_tokens"],
        output_tokens=usage["completion_tokens"],
    )

    cost_tracker.record_cost(
        cost=cost,
        customer_id=enterprise_customer,
        team="saas-platform",
        project="ai-features",
        feature="chat-completion",
        model="meta-llama/Llama-2-70b-chat-hf",
    )

    print(f"   ✅ Chat request {i + 1}: ${cost:.8f}")

# Document analysis
for i in range(3):
    response = adapter.completion_create(
        model="meta-llama/Llama-2-70b-chat-hf",
        messages=[
            {"role": "user", "content": f"Analyze contract document section {i + 1}"}
        ],
        max_tokens=500,
        customer_id=enterprise_customer,
        feature="document-analysis",
        cost_center="Enterprise-Sales",
    )

    usage = response["usage"]
    cost = calculate_completion_cost(
        model="meta-llama/Llama-2-70b-chat-hf",
        input_tokens=usage["prompt_tokens"],
        output_tokens=usage["completion_tokens"],
    )

    cost_tracker.record_cost(
        cost=cost,
        customer_id=enterprise_customer,
        team="saas-platform",
        project="ai-features",
        feature="document-analysis",
        model="meta-llama/Llama-2-70b-chat-hf",
    )

    print(f"   ✅ Document analysis {i + 1}: ${cost:.8f}")

print()


# Scenario 2: Startup Customer - Cost Sensitive
print("=" * 70)
print("SCENARIO 2: Startup Customer (Cost Sensitive)")
print("=" * 70 + "\n")

startup_customer = "techstartup-basic"

print(f"Processing requests for: {startup_customer}")
print("Features: Basic chat, Classification\n")

# Using cheaper model for cost-sensitive customer
for i in range(10):
    response = adapter.completion_create(
        model="meta-llama/Llama-2-7b-chat-hf",  # Budget model
        messages=[{"role": "user", "content": f"Simple query {i + 1}"}],
        max_tokens=100,
        customer_id=startup_customer,
        feature="basic-chat",
        cost_center="Self-Serve",
    )

    usage = response["usage"]
    cost = calculate_completion_cost(
        model="meta-llama/Llama-2-7b-chat-hf",
        input_tokens=usage["prompt_tokens"],
        output_tokens=usage["completion_tokens"],
    )

    cost_tracker.record_cost(
        cost=cost,
        customer_id=startup_customer,
        team="saas-platform",
        project="ai-features",
        feature="basic-chat",
        model="meta-llama/Llama-2-7b-chat-hf",
    )

    print(f"   ✅ Chat request {i + 1}: ${cost:.8f}")

print()


# Scenario 3: Mid-Market Customer - Balanced
print("=" * 70)
print("SCENARIO 3: Mid-Market Customer (Balanced)")
print("=" * 70 + "\n")

midmarket_customer = "midsize-company-pro"

print(f"Processing requests for: {midmarket_customer}")
print("Features: Chat, Summarization, Q&A\n")

# Mix of models for different use cases
features_and_models = [
    ("chat-completion", "meta-llama/Llama-2-13b-chat-hf", 4),
    ("summarization", "meta-llama/Llama-2-13b-chat-hf", 3),
    ("qa-system", "meta-llama/Llama-2-13b-chat-hf", 3),
]

for feature, model, count in features_and_models:
    for i in range(count):
        response = adapter.completion_create(
            model=model,
            messages=[{"role": "user", "content": f"{feature} request {i + 1}"}],
            max_tokens=150,
            customer_id=midmarket_customer,
            feature=feature,
            cost_center="Mid-Market",
        )

        usage = response["usage"]
        cost = calculate_completion_cost(
            model=model,
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
        )

        cost_tracker.record_cost(
            cost=cost,
            customer_id=midmarket_customer,
            team="saas-platform",
            project="ai-features",
            feature=feature,
            model=model,
        )

        print(f"   ✅ {feature} {i + 1}: ${cost:.8f}")

print()


# Scenario 4: Internal Testing Team
print("=" * 70)
print("SCENARIO 4: Internal Testing Team")
print("=" * 70 + "\n")

internal_team = "internal-qa-team"

print(f"Processing requests for: {internal_team}")
print("Features: Testing, Validation\n")

for i in range(5):
    response = adapter.completion_create(
        model="meta-llama/Llama-2-7b-chat-hf",
        messages=[{"role": "user", "content": f"Test case {i + 1}"}],
        max_tokens=50,
        customer_id=internal_team,
        feature="testing",
        cost_center="Engineering",
    )

    usage = response["usage"]
    cost = calculate_completion_cost(
        model="meta-llama/Llama-2-7b-chat-hf",
        input_tokens=usage["prompt_tokens"],
        output_tokens=usage["completion_tokens"],
    )

    cost_tracker.record_cost(
        cost=cost,
        customer_id=internal_team,
        team="saas-platform",
        project="ai-features",
        feature="testing",
        model="meta-llama/Llama-2-7b-chat-hf",
    )

    print(f"   ✅ Test request {i + 1}: ${cost:.8f}")

print()


# Generate comprehensive cost report
cost_tracker.print_report()


# Monthly projection
print("\n📈 MONTHLY PROJECTION:")
print("(Assuming current usage pattern)\n")

daily_cost = cost_tracker.total_cost
monthly_cost = daily_cost * 30

print(f"Current sample cost: ${cost_tracker.total_cost:.6f}")
print(f"Requests in sample: {cost_tracker.total_requests}")
print("\nMonthly projections (30 days):")
print(f"   Total cost: ${monthly_cost:.2f}")
print(f"   Total requests: {cost_tracker.total_requests * 30:,}")
print()

# Per-customer monthly projections
print("Customer monthly billing estimates:")
for customer, cost in sorted(
    cost_tracker.by_customer.items(), key=lambda x: x[1], reverse=True
):
    monthly_customer_cost = cost * 30
    print(f"   {customer:30s} ${monthly_customer_cost:10.2f}/month")

print()


# Recommendations
print("=" * 70)
print("💡 OPTIMIZATION RECOMMENDATIONS")
print("=" * 70)

# Identify high-cost customers
high_cost_customers = [
    (customer, cost)
    for customer, cost in cost_tracker.by_customer.items()
    if cost > cost_tracker.total_cost * 0.3
]

if high_cost_customers:
    print("\n🔍 High-Cost Customers:")
    for customer, cost in high_cost_customers:
        pct = (cost / cost_tracker.total_cost) * 100
        print(f"   • {customer}: ${cost:.6f} ({pct:.1f}% of total)")
        print("     Consider: Enterprise pricing tier, volume discounts")

# Feature-level optimization
expensive_features = [
    (feature, cost)
    for feature, cost in cost_tracker.by_feature.items()
    if cost > cost_tracker.total_cost * 0.2
]

if expensive_features:
    print("\n🎯 Expensive Features:")
    for feature, cost in expensive_features:
        pct = (cost / cost_tracker.total_cost) * 100
        print(f"   • {feature}: ${cost:.6f} ({pct:.1f}% of total)")
        print("     Consider: Model optimization, caching, rate limiting")

# Model optimization
print("\n🤖 Model Usage Optimization:")
for model, cost in sorted(
    cost_tracker.by_model.items(), key=lambda x: x[1], reverse=True
):
    pct = (cost / cost_tracker.total_cost) * 100
    print(f"   • {model}")
    print(f"     Cost: ${cost:.6f} ({pct:.1f}% of total)")

    if "70b" in model.lower():
        print("     💡 Consider: Use 13B or 7B models for simpler tasks")
    elif "13b" in model.lower():
        print("     ✅ Good balance of cost and capability")
    elif "7b" in model.lower():
        print("     ✅ Cost-optimized for simple tasks")

print()
print("=" * 70)
print("✅ Multi-customer attribution complete!")
print("=" * 70)

print("\n🎯 NEXT STEPS:")
print("   • Export cost data to your billing system")
print("   • Set up alerts for cost anomalies")
print("   • Implement tiered pricing based on usage")
print("   • Use governance attributes for chargeback")
print("   • Monitor customer-level cost trends")
print()

print("📊 INTEGRATION:")
print("   • Query observability platform: SUM(cost) GROUP BY customer_id")
print("   • Create dashboards for cost tracking")
print("   • Set up automated monthly billing reports")
print("   • Implement budget alerts per customer")
print()

print("📚 Next Steps:")
print("   • Try context_manager_patterns.py for workflow management")
print("   • See production_deployment.py for scaling patterns")
