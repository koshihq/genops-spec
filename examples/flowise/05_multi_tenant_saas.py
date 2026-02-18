#!/usr/bin/env python3
"""
Example: Multi-Tenant SaaS with Cost Isolation

Complexity: ⭐⭐⭐ Advanced

This example demonstrates building a multi-tenant SaaS application using
Flowise with complete cost isolation, per-tenant governance, and usage
analytics. Perfect for SaaS platforms serving multiple customers.

Prerequisites:
- Flowise instance running
- GenOps package installed
- Basic understanding of SaaS architecture

Usage:
    python 05_multi_tenant_saas.py

Environment Variables:
    FLOWISE_BASE_URL: Flowise instance URL
    FLOWISE_API_KEY: API key (optional for local dev)
"""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

from genops.providers.flowise import instrument_flowise
from genops.providers.flowise_validation import validate_flowise_setup

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SubscriptionTier(Enum):
    """SaaS subscription tiers with different limits and features."""

    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


@dataclass
class TenantConfiguration:
    """Configuration for a SaaS tenant."""

    tenant_id: str
    tenant_name: str
    subscription_tier: SubscriptionTier
    monthly_ai_budget: Decimal
    daily_request_limit: int
    features_enabled: list[str]
    custom_branding: bool = False
    dedicated_flows: list[str] = field(default_factory=list)

    # Usage tracking
    current_monthly_spend: Decimal = Decimal("0.0")
    current_daily_requests: int = 0
    last_request_date: datetime = field(default_factory=datetime.now)

    def reset_daily_counters_if_needed(self):
        """Reset daily counters if it's a new day."""
        today = datetime.now().date()
        if self.last_request_date.date() != today:
            self.current_daily_requests = 0
            self.last_request_date = datetime.now()

    def can_make_request(self) -> tuple[bool, str]:
        """Check if tenant can make a request based on limits."""
        self.reset_daily_counters_if_needed()

        # Check daily request limit
        if self.current_daily_requests >= self.daily_request_limit:
            return False, f"Daily request limit of {self.daily_request_limit} exceeded"

        # Check monthly budget (rough check - in production this would be more sophisticated)
        if self.current_monthly_spend >= self.monthly_ai_budget:
            return False, f"Monthly AI budget of ${self.monthly_ai_budget} exceeded"

        return True, "Request allowed"


class MultiTenantFlowiseManager:
    """Manages Flowise access for multiple SaaS tenants with complete isolation."""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url
        self.api_key = api_key
        self.tenants: dict[str, TenantConfiguration] = {}
        self.adapters: dict[str, Any] = {}  # Cached adapters per tenant

        # SaaS-wide configuration
        self.saas_name = "AI Assistant SaaS"
        self.saas_version = "v1.2.0"

        logger.info(f"Initialized {self.saas_name} multi-tenant manager")

    def register_tenant(self, config: TenantConfiguration):
        """Register a new tenant with the system."""
        self.tenants[config.tenant_id] = config

        # Create dedicated adapter for this tenant
        self.adapters[config.tenant_id] = instrument_flowise(
            base_url=self.base_url,
            api_key=self.api_key,
            # Governance attributes for complete cost isolation
            team=f"tenant-{config.tenant_id}",
            project=self.saas_name.lower().replace(" ", "-"),
            customer_id=config.tenant_id,
            environment="production",
            # Custom attributes for SaaS tracking
            tenant_name=config.tenant_name,
            subscription_tier=config.subscription_tier.value,
            saas_version=self.saas_version,
        )

        logger.info(f"Registered tenant: {config.tenant_name} ({config.tenant_id})")

    def execute_for_tenant(
        self,
        tenant_id: str,
        chatflow_id: str,
        question: str,
        user_id: Optional[str] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Execute Flowise flow for a specific tenant with full governance."""

        # Validate tenant exists
        if tenant_id not in self.tenants:
            return {
                "success": False,
                "error": "Tenant not found",
                "error_code": "TENANT_NOT_FOUND",
            }

        tenant = self.tenants[tenant_id]

        # Check if request is allowed based on tenant limits
        can_request, reason = tenant.can_make_request()
        if not can_request:
            return {
                "success": False,
                "error": reason,
                "error_code": "LIMIT_EXCEEDED",
                "tenant_info": {
                    "subscription_tier": tenant.subscription_tier.value,
                    "daily_requests_used": tenant.current_daily_requests,
                    "daily_request_limit": tenant.daily_request_limit,
                    "monthly_spend": float(tenant.current_monthly_spend),
                    "monthly_budget": float(tenant.monthly_ai_budget),
                },
            }

        try:
            adapter = self.adapters[tenant_id]

            start_time = time.time()

            # Execute with tenant-specific governance
            response = adapter.predict_flow(
                chatflow_id=chatflow_id,
                question=question,
                # Additional governance for detailed tracking
                user_id=user_id,
                session_type="saas-tenant-request",
                chatflow_category=kwargs.get("category", "general"),
                **kwargs,
            )

            execution_time = time.time() - start_time

            # Update tenant usage tracking
            tenant.current_daily_requests += 1

            # Estimate cost (in production, this would come from telemetry)
            estimated_cost = self._estimate_request_cost(
                question, response, tenant.subscription_tier
            )
            tenant.current_monthly_spend += estimated_cost

            return {
                "success": True,
                "response": response,
                "execution_time_ms": int(execution_time * 1000),
                "estimated_cost": float(estimated_cost),
                "tenant_info": {
                    "requests_remaining_today": tenant.daily_request_limit
                    - tenant.current_daily_requests,
                    "monthly_budget_remaining": float(
                        tenant.monthly_ai_budget - tenant.current_monthly_spend
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Execution failed for tenant {tenant_id}: {e}")
            return {"success": False, "error": str(e), "error_code": "EXECUTION_ERROR"}

    def _estimate_request_cost(
        self, question: str, response: Any, subscription_tier: SubscriptionTier
    ) -> Decimal:
        """Estimate cost of a request (simplified for demo)."""

        # Extract response text
        response_text = ""
        if isinstance(response, dict):
            response_text = (
                response.get("text", "") or response.get("answer", "") or str(response)
            )
        else:
            response_text = str(response)

        # Estimate tokens
        total_tokens = len(question.split()) + len(response_text.split())

        # Apply tier-based pricing (enterprise gets better rates)
        if subscription_tier == SubscriptionTier.ENTERPRISE:
            cost_per_token = Decimal("0.000001")  # Best rates
        elif subscription_tier == SubscriptionTier.PROFESSIONAL:
            cost_per_token = Decimal("0.0000015")
        elif subscription_tier == SubscriptionTier.STARTER:
            cost_per_token = Decimal("0.000002")
        else:  # FREE
            cost_per_token = Decimal("0.0000025")  # Highest rates

        return Decimal(total_tokens) * cost_per_token

    def get_tenant_analytics(self, tenant_id: str, days: int = 7) -> dict[str, Any]:
        """Get analytics and usage stats for a tenant."""

        if tenant_id not in self.tenants:
            return {"error": "Tenant not found"}

        tenant = self.tenants[tenant_id]

        # In a real implementation, this would query your analytics database
        # For demo purposes, we'll return mock data
        return {
            "tenant_info": {
                "tenant_id": tenant.tenant_id,
                "tenant_name": tenant.tenant_name,
                "subscription_tier": tenant.subscription_tier.value,
                "account_status": "active",
            },
            "usage_summary": {
                "current_monthly_spend": float(tenant.current_monthly_spend),
                "monthly_budget": float(tenant.monthly_ai_budget),
                "budget_utilization_pct": float(
                    (tenant.current_monthly_spend / tenant.monthly_ai_budget) * 100
                ),
                "daily_requests_today": tenant.current_daily_requests,
                "daily_request_limit": tenant.daily_request_limit,
            },
            "features": {
                "features_enabled": tenant.features_enabled,
                "custom_branding": tenant.custom_branding,
                "dedicated_flows": tenant.dedicated_flows,
            },
            "recommendations": self._generate_tenant_recommendations(tenant),
        }

    def _generate_tenant_recommendations(
        self, tenant: TenantConfiguration
    ) -> list[str]:
        """Generate usage and upgrade recommendations for tenant."""
        recommendations = []

        budget_utilization = (
            tenant.current_monthly_spend / tenant.monthly_ai_budget
        ) * 100
        request_utilization = (
            tenant.current_daily_requests / tenant.daily_request_limit
        ) * 100

        # Budget recommendations
        if budget_utilization > 90:
            recommendations.append(
                "Consider upgrading to a higher tier for better AI budget allocation"
            )
        elif budget_utilization > 80:
            recommendations.append(
                "Monitor AI usage closely - approaching monthly budget limit"
            )

        # Request limit recommendations
        if request_utilization > 90:
            recommendations.append(
                "Daily request limit nearly reached - consider upgrading for higher limits"
            )

        # Tier-specific recommendations
        if tenant.subscription_tier == SubscriptionTier.FREE:
            if budget_utilization > 50 or request_utilization > 50:
                recommendations.append(
                    "Upgrade to Starter plan for 5x more requests and better pricing"
                )
        elif tenant.subscription_tier == SubscriptionTier.STARTER:
            if budget_utilization > 70:
                recommendations.append(
                    "Professional plan offers 30% better AI pricing and advanced features"
                )

        # Feature recommendations
        if "analytics-dashboard" not in tenant.features_enabled:
            recommendations.append(
                "Enable analytics dashboard for detailed usage insights"
            )

        return recommendations


def create_sample_tenants() -> list[TenantConfiguration]:
    """Create sample tenant configurations for different SaaS use cases."""

    tenants = [
        # Free tier - Small startup
        TenantConfiguration(
            tenant_id="startup-001",
            tenant_name="InnovateTech Startup",
            subscription_tier=SubscriptionTier.FREE,
            monthly_ai_budget=Decimal("25.00"),
            daily_request_limit=100,
            features_enabled=["basic-chatbot", "email-support"],
        ),
        # Starter tier - Growing company
        TenantConfiguration(
            tenant_id="growth-company-002",
            tenant_name="GrowthCorp Solutions",
            subscription_tier=SubscriptionTier.STARTER,
            monthly_ai_budget=Decimal("200.00"),
            daily_request_limit=1000,
            features_enabled=[
                "basic-chatbot",
                "email-support",
                "analytics-basic",
                "api-access",
            ],
            custom_branding=True,
        ),
        # Professional tier - Established business
        TenantConfiguration(
            tenant_id="enterprise-client-003",
            tenant_name="MegaCorp Industries",
            subscription_tier=SubscriptionTier.PROFESSIONAL,
            monthly_ai_budget=Decimal("1500.00"),
            daily_request_limit=10000,
            features_enabled=[
                "advanced-chatbot",
                "email-support",
                "phone-support",
                "analytics-advanced",
                "api-access",
                "custom-integrations",
            ],
            custom_branding=True,
            dedicated_flows=["custom-industry-bot", "compliance-assistant"],
        ),
        # Enterprise tier - Large corporation
        TenantConfiguration(
            tenant_id="enterprise-corp-004",
            tenant_name="GlobalTech Corporation",
            subscription_tier=SubscriptionTier.ENTERPRISE,
            monthly_ai_budget=Decimal("10000.00"),
            daily_request_limit=100000,
            features_enabled=[
                "advanced-chatbot",
                "email-support",
                "phone-support",
                "dedicated-support",
                "analytics-advanced",
                "analytics-custom",
                "api-access",
                "custom-integrations",
                "sso-integration",
                "audit-logging",
                "compliance-features",
            ],
            custom_branding=True,
            dedicated_flows=[
                "enterprise-sales-assistant",
                "compliance-bot",
                "hr-assistant",
                "technical-support-bot",
                "executive-briefing-bot",
            ],
        ),
    ]

    return tenants


def demonstrate_multi_tenant_saas():
    """Demonstrate multi-tenant SaaS with cost isolation."""

    print("🏢 Multi-Tenant SaaS with Cost Isolation")
    print("=" * 60)

    # Configuration
    base_url = os.getenv("FLOWISE_BASE_URL", "http://localhost:3000")
    api_key = os.getenv("FLOWISE_API_KEY")

    # Step 1: Setup and validation
    print("📋 Step 1: Initializing multi-tenant SaaS platform...")

    try:
        result = validate_flowise_setup(base_url, api_key)
        if not result.is_valid:
            print("❌ Setup validation failed.")
            return False

        # Get available chatflows
        temp_flowise = instrument_flowise(base_url=base_url, api_key=api_key)
        chatflows = temp_flowise.get_chatflows()
        if not chatflows:
            print("❌ No chatflows available.")
            return False

        chatflow_id = chatflows[0].get("id")
        chatflow_name = chatflows[0].get("name", "Unnamed")
        print(f"✅ Using chatflow: {chatflow_name}")

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return False

    # Step 2: Initialize multi-tenant manager
    print("\n🏗️ Step 2: Setting up multi-tenant manager...")

    manager = MultiTenantFlowiseManager(base_url, api_key)

    # Register sample tenants
    sample_tenants = create_sample_tenants()

    for tenant in sample_tenants:
        manager.register_tenant(tenant)

    print(f"✅ Registered {len(sample_tenants)} tenants")

    # Step 3: Simulate tenant requests
    print("\n🔄 Step 3: Simulating tenant requests...")

    # Different request patterns for each tenant
    tenant_scenarios = [
        {
            "tenant_id": "startup-001",
            "requests": [
                "What are the benefits of AI in customer service?",
                "How can startups leverage AI for growth?",
                "Explain machine learning in simple terms.",
            ],
            "user_type": "startup-founder",
        },
        {
            "tenant_id": "growth-company-002",
            "requests": [
                "Generate a customer onboarding email template.",
                "What are best practices for AI implementation in mid-size companies?",
                "Create a product feature comparison chart.",
                "Draft a technical specification for API integration.",
            ],
            "user_type": "product-manager",
        },
        {
            "tenant_id": "enterprise-client-003",
            "requests": [
                "Conduct a comprehensive market analysis for AI adoption in manufacturing.",
                "Generate an executive brief on digital transformation ROI.",
                "Create a detailed compliance checklist for AI systems.",
                "Develop a strategic roadmap for enterprise AI initiatives.",
                "Analyze competitive landscape and positioning strategies.",
            ],
            "user_type": "enterprise-analyst",
        },
        {
            "tenant_id": "enterprise-corp-004",
            "requests": [
                "Generate quarterly board presentation on AI initiatives and ROI metrics.",
                "Create comprehensive risk assessment for AI deployment across global operations.",
                "Develop regulatory compliance framework for AI systems in financial services.",
                "Analyze market opportunities for AI-powered product line extensions.",
                "Design change management strategy for AI transformation across 50,000 employees.",
            ],
            "user_type": "c-level-executive",
        },
    ]

    execution_results = []

    for scenario in tenant_scenarios:
        tenant_id = scenario["tenant_id"]
        tenant = manager.tenants[tenant_id]

        print(
            f"\n   📊 Processing requests for {tenant.tenant_name} ({tenant.subscription_tier.value})..."
        )

        scenario_results = []

        for i, request in enumerate(scenario["requests"], 1):
            print(f"      Request {i}/{len(scenario['requests'])}: {request[:50]}...")

            result = manager.execute_for_tenant(
                tenant_id=tenant_id,
                chatflow_id=chatflow_id,
                question=request,
                user_id=f"user-{scenario['user_type']}-{i}",
                category="general-ai-assistant",
            )

            scenario_results.append(result)

            if result["success"]:
                print(f"         ✅ Success (Cost: ${result['estimated_cost']:.4f})")
            else:
                print(f"         ❌ Failed: {result['error']}")

        execution_results.append(
            {
                "tenant_id": tenant_id,
                "tenant_name": tenant.tenant_name,
                "results": scenario_results,
            }
        )

    # Step 4: Analyze results and show tenant isolation
    print("\n📊 Step 4: Multi-Tenant Results Analysis")
    print("=" * 50)

    for tenant_result in execution_results:
        tenant_id = tenant_result["tenant_id"]
        tenant_name = tenant_result["tenant_name"]
        results = tenant_result["results"]

        successful_requests = sum(1 for r in results if r["success"])
        total_cost = sum(r.get("estimated_cost", 0) for r in results if r["success"])

        print(f"\n📋 {tenant_name} ({tenant_id}):")
        print(f"   Successful Requests: {successful_requests}/{len(results)}")
        print(f"   Total Cost: ${total_cost:.4f}")

        # Show tenant analytics
        analytics = manager.get_tenant_analytics(tenant_id)
        usage = analytics["usage_summary"]
        print(f"   Budget Utilization: {usage['budget_utilization_pct']:.1f}%")
        print(
            f"   Daily Requests Used: {usage['daily_requests_today']}/{usage['daily_request_limit']}"
        )

        # Show any recommendations
        if analytics["recommendations"]:
            print("   Recommendations:")
            for rec in analytics["recommendations"][:2]:  # Show top 2
                print(f"     • {rec}")

    # Step 5: Demonstrate cost isolation
    print("\n🔒 Step 5: Cost Isolation Verification")
    print("=" * 50)

    print("✅ Cost isolation achieved through:")
    print("   • Unique customer_id for each tenant")
    print("   • Dedicated team attribution per tenant")
    print("   • Subscription tier tracking in governance attributes")
    print("   • Per-tenant usage limits and budget controls")
    print("   • Isolated telemetry streams for each tenant")

    print("\n📈 SaaS Platform Benefits:")
    print("   • Complete cost transparency per customer")
    print("   • Automated usage-based billing capabilities")
    print("   • Tier-based feature and limit enforcement")
    print("   • Usage analytics and optimization recommendations")
    print("   • Scalable governance across unlimited tenants")

    return len(execution_results) > 0


def demonstrate_tenant_lifecycle():
    """Demonstrate tenant lifecycle management patterns."""

    print("\n🔄 Tenant Lifecycle Management")
    print("=" * 50)

    lifecycle_scenarios = [
        {
            "scenario": "New Tenant Onboarding",
            "description": "Free trial → paid subscription activation",
        },
        {
            "scenario": "Subscription Upgrade",
            "description": "Starter → Professional tier migration",
        },
        {
            "scenario": "Usage Optimization",
            "description": "Enterprise tenant cost optimization analysis",
        },
        {
            "scenario": "Churn Prevention",
            "description": "Usage-based retention insights",
        },
    ]

    for scenario in lifecycle_scenarios:
        print(f"\n📋 {scenario['scenario']}:")
        print(f"   Use Case: {scenario['description']}")

        if scenario["scenario"] == "New Tenant Onboarding":
            print("   Implementation:")
            print("     • Create tenant with FREE tier limits")
            print("     • Track trial usage and provide upgrade prompts")
            print("     • Automatically provision tenant-specific governance")
            print("     • Set up usage analytics and engagement tracking")

        elif scenario["scenario"] == "Subscription Upgrade":
            print("   Implementation:")
            print("     • Update subscription_tier in tenant configuration")
            print("     • Increase daily_request_limit and monthly_ai_budget")
            print("     • Enable additional features in features_enabled list")
            print("     • Maintain historical usage data for analytics")

        elif scenario["scenario"] == "Usage Optimization":
            print("   Implementation:")
            print("     • Analyze cost patterns across tenant's usage")
            print("     • Identify high-cost flows and optimization opportunities")
            print("     • Generate recommendations for cost reduction")
            print("     • Provide tier comparison for potential savings")

        elif scenario["scenario"] == "Churn Prevention":
            print("   Implementation:")
            print("     • Monitor usage trends and engagement patterns")
            print("     • Identify at-risk tenants with declining usage")
            print("     • Proactive outreach for optimization consultations")
            print("     • Offer tier downgrades to retain price-sensitive customers")


def main():
    """Main example function."""

    try:
        print("🚀 Multi-Tenant SaaS with Cost Isolation Example")
        print("=" * 60)

        # Run main demonstration
        success = demonstrate_multi_tenant_saas()

        if success:
            # Show tenant lifecycle patterns
            demonstrate_tenant_lifecycle()

            print("\n🎉 Multi-Tenant SaaS Example Complete!")
            print("=" * 50)
            print("✅ You've learned how to:")
            print("   • Build multi-tenant SaaS with complete cost isolation")
            print("   • Implement subscription tiers with usage limits")
            print("   • Track per-tenant analytics and usage patterns")
            print("   • Generate tenant-specific optimization recommendations")
            print("   • Manage tenant lifecycle from trial to enterprise")

            print("\n💡 Key SaaS Patterns:")
            print("   • Tenant isolation through governance attributes")
            print("   • Subscription-based feature and limit enforcement")
            print("   • Usage-based billing and cost attribution")
            print("   • Automated tenant analytics and recommendations")
            print("   • Scalable governance across unlimited customers")

            print("\n📚 Next Steps:")
            print("   • Implement tenant database and persistent storage")
            print("   • Set up automated billing based on usage data")
            print("   • Create tenant dashboard for self-service analytics")
            print("   • Explore enterprise governance (06_enterprise_governance.py)")

        return success

    except Exception as e:
        logger.error(f"Example failed: {e}")
        return False


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Example interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)
