#!/usr/bin/env python3
"""
📊 Cost Per Customer Attribution - Complete Governance Scenario

This example demonstrates how GenOps AI enables precise cost attribution
across customers, teams, and projects for multi-tenant AI applications.

BUSINESS PROBLEM:
Your SaaS platform uses AI for customer features, but finance has no visibility
into AI costs per customer. They want to implement usage-based pricing but
can't track actual costs or calculate profit margins accurately.

GENOPS SOLUTION:
- Automatic cost attribution to customers, teams, projects, and features
- Real-time cost tracking with multi-dimensional breakdowns
- Usage-based billing integration and chargeback calculations
- Cost optimization insights and customer profitability analysis

Run this example to see cost attribution in action!
"""

import os
import time
import logging
import random
from typing import Dict, Any, List
from datetime import datetime, timedelta

# GenOps imports
from genops.core.telemetry import GenOpsTelemetry
from genops.providers.openai import instrument_openai

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Simulate customer and usage data
CUSTOMERS = {
    "enterprise-123": {
        "name": "Acme Corporation", 
        "tier": "enterprise",
        "billing_rate": 1.25,  # 25% markup on AI costs
        "monthly_budget": 500.0
    },
    "startup-456": {
        "name": "InnovateCorp",
        "tier": "startup", 
        "billing_rate": 1.15,  # 15% markup
        "monthly_budget": 150.0
    },
    "premium-789": {
        "name": "TechGiant Ltd",
        "tier": "premium",
        "billing_rate": 1.30,  # 30% markup
        "monthly_budget": 1000.0
    }
}

FEATURES = {
    "chat_assistant": {"base_cost_per_request": 0.02},
    "document_analysis": {"base_cost_per_request": 0.15},
    "content_generation": {"base_cost_per_request": 0.08},
    "data_insights": {"base_cost_per_request": 0.25},
    "translation_service": {"base_cost_per_request": 0.05}
}

def demonstrate_multi_tenant_cost_tracking():
    """
    Show cost attribution across multiple customers and features.
    """
    print("\n💰 DEMONSTRATING MULTI-TENANT COST ATTRIBUTION")
    print("=" * 60)
    
    # Initialize telemetry
    telemetry = GenOpsTelemetry()
    
    # Simulate AI operations for different customers and features
    operations = [
        {
            "customer": "enterprise-123",
            "feature": "chat_assistant",
            "operation": "customer_support_chat",
            "team": "support-team",
            "project": "ai-assistant",
            "requests": 5
        },
        {
            "customer": "startup-456", 
            "feature": "document_analysis",
            "operation": "contract_analysis",
            "team": "legal-ai-team",
            "project": "contract-analyzer",
            "requests": 3
        },
        {
            "customer": "premium-789",
            "feature": "content_generation", 
            "operation": "marketing_copy",
            "team": "content-team",
            "project": "ai-writer",
            "requests": 8
        },
        {
            "customer": "enterprise-123",
            "feature": "data_insights",
            "operation": "analytics_summary",
            "team": "analytics-team", 
            "project": "insights-engine",
            "requests": 2
        },
        {
            "customer": "startup-456",
            "feature": "translation_service",
            "operation": "multilingual_support",
            "team": "localization-team",
            "project": "translator",
            "requests": 10
        }
    ]
    
    total_costs_by_customer = {}
    
    for op in operations:
        customer_id = op["customer"]
        customer_info = CUSTOMERS[customer_id]
        feature_info = FEATURES[op["feature"]]
        
        print(f"\n🏢 Processing: {customer_info['name']} ({customer_info['tier']})")
        print(f"   Feature: {op['feature']} | Operation: {op['operation']}")
        print(f"   Requests: {op['requests']}")
        
        # Calculate costs for this operation batch
        base_cost_per_request = feature_info["base_cost_per_request"]
        # Add some realistic variance
        actual_cost_per_request = base_cost_per_request * random.uniform(0.8, 1.3)
        total_cost = actual_cost_per_request * op["requests"]
        
        # Track cumulative costs
        if customer_id not in total_costs_by_customer:
            total_costs_by_customer[customer_id] = 0
        total_costs_by_customer[customer_id] += total_cost
        
        print(f"   💰 Cost: ${total_cost:.4f} (${actual_cost_per_request:.4f} per request)")
        
        # Record detailed telemetry for each request batch
        with telemetry.trace_operation(
            operation_name=op["operation"],
            operation_type="ai.inference",
            team=op["team"],
            project=op["project"],
            customer=customer_id,
            customer_name=customer_info["name"],
            customer_tier=customer_info["tier"],
            feature=op["feature"]
        ) as span:
            
            # Record cost telemetry with customer attribution
            telemetry.record_cost(
                span=span,
                cost=total_cost,
                currency="USD",
                provider="openai",
                model="gpt-4",
                input_tokens=op["requests"] * 200,  # Estimated tokens
                output_tokens=op["requests"] * 75,
                # Custom attributes for attribution
                cost_per_unit=actual_cost_per_request,
                request_count=op["requests"],
                billing_rate=customer_info["billing_rate"]
            )
            
            # Record customer budget tracking
            monthly_budget = customer_info["monthly_budget"]
            budget_used_pct = (total_costs_by_customer[customer_id] / monthly_budget) * 100
            
            telemetry.record_budget(
                span=span,
                budget_name=f"{customer_id}_monthly_ai_budget",
                allocated=monthly_budget,
                consumed=total_costs_by_customer[customer_id],
                period="monthly",
                customer_tier=customer_info["tier"]
            )
            
            if budget_used_pct > 80:
                print(f"   ⚠️  Budget Warning: {budget_used_pct:.1f}% of monthly budget used")
            else:
                print(f"   ✅ Budget: {budget_used_pct:.1f}% of monthly budget used")
    
    # Show cost summary
    print(f"\n📊 COST ATTRIBUTION SUMMARY")
    print("=" * 60)
    
    for customer_id, total_cost in total_costs_by_customer.items():
        customer_info = CUSTOMERS[customer_id]
        billing_cost = total_cost * customer_info["billing_rate"]
        profit = billing_cost - total_cost
        margin = (profit / billing_cost) * 100 if billing_cost > 0 else 0
        
        print(f"\n🏢 {customer_info['name']} ({customer_info['tier']})")
        print(f"   AI Cost: ${total_cost:.4f}")
        print(f"   Customer Bill: ${billing_cost:.4f} ({customer_info['billing_rate']}x markup)")
        print(f"   Profit: ${profit:.4f} ({margin:.1f}% margin)")
        print(f"   Budget Used: {(total_cost/customer_info['monthly_budget']*100):.1f}%")

def demonstrate_real_time_cost_tracking():
    """
    Show real-time cost tracking and attribution with actual API calls.
    """
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️ Skipping real-time tracking demo (no API key)")
        return
    
    print("\n🔗 REAL-TIME COST TRACKING WITH OPENAI API")
    print("=" * 60)
    
    try:
        # Instrument OpenAI client
        client = instrument_openai(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Simulate real customer operations
        customer_operations = [
            {
                "customer_id": "enterprise-123",
                "prompt": "Summarize the quarterly sales performance in 2 sentences",
                "feature": "data_insights",
                "model": "gpt-3.5-turbo"
            },
            {
                "customer_id": "startup-456", 
                "prompt": "Generate a product description for an AI-powered analytics tool",
                "feature": "content_generation",
                "model": "gpt-3.5-turbo"
            }
        ]
        
        for op in customer_operations:
            customer_info = CUSTOMERS[op["customer_id"]]
            
            print(f"\n🤖 Processing for {customer_info['name']}:")
            print(f"   Feature: {op['feature']}")
            print(f"   Model: {op['model']}")
            
            # Make real API call with full attribution
            response = client.chat_completions_create(
                model=op["model"],
                messages=[{"role": "user", "content": op["prompt"]}],
                max_tokens=100,
                # Governance attribution
                team="api-team",
                project="customer-api",
                customer_id=op["customer_id"],
                customer_name=customer_info["name"],
                customer_tier=customer_info["tier"],
                feature=op["feature"],
                billing_rate=customer_info["billing_rate"]
            )
            
            print(f"   ✅ Response generated and costs attributed to {customer_info['name']}")
            print(f"   📊 Real-time telemetry sent to observability platform")
    
    except Exception as e:
        print(f"❌ Error: {e}")

def show_cost_attribution_analytics():
    """
    Show the kind of analytics and insights enabled by cost attribution.
    """
    print("\n📈 COST ATTRIBUTION ANALYTICS & INSIGHTS")
    print("=" * 60)
    
    # Simulate analytics that would be available in your dashboard
    print("💡 Analytics enabled by GenOps cost attribution:")
    
    analytics_examples = [
        {
            "metric": "Customer Profitability Analysis",
            "value": "Enterprise customers: 25% avg margin, Startup customers: 15% avg margin",
            "action": "Consider tiered pricing optimization"
        },
        {
            "metric": "Feature Cost Efficiency", 
            "value": "Data Insights: $0.25/request, Chat Assistant: $0.02/request",
            "action": "Focus optimization efforts on high-cost features"
        },
        {
            "metric": "Usage Pattern Insights",
            "value": "70% of AI costs come from 20% of customers", 
            "action": "Implement usage-based pricing for heavy users"
        },
        {
            "metric": "Budget Utilization",
            "value": "3 customers approaching monthly budget limits",
            "action": "Proactive customer success outreach needed"
        },
        {
            "metric": "Cost Trend Analysis",
            "value": "AI costs growing 15% month-over-month",
            "action": "Review pricing strategy and cost optimization"
        }
    ]
    
    for i, analytic in enumerate(analytics_examples, 1):
        print(f"\n{i}. 📊 {analytic['metric']}")
        print(f"   📈 Insight: {analytic['value']}")
        print(f"   💡 Action: {analytic['action']}")

def show_telemetry_for_cost_attribution():
    """
    Show the telemetry data structure that enables cost attribution.
    """
    print("\n📊 COST ATTRIBUTION TELEMETRY DATA")
    print("=" * 60)
    
    sample_telemetry = {
        "genops.operation.name": "customer_support_chat",
        "genops.operation.type": "ai.inference",
        "genops.team": "support-team",
        "genops.project": "ai-assistant",
        "genops.feature": "chat_assistant",
        "genops.customer.id": "enterprise-123",
        "genops.customer.name": "Acme Corporation", 
        "genops.customer.tier": "enterprise",
        "genops.cost.total": 0.0234,
        "genops.cost.currency": "USD",
        "genops.cost.provider": "openai",
        "genops.cost.model": "gpt-4",
        "genops.cost.per_unit": 0.0234,
        "genops.cost.request_count": 1,
        "genops.billing.rate": 1.25,
        "genops.billing.amount": 0.0293,
        "genops.tokens.input": 200,
        "genops.tokens.output": 75,
        "genops.tokens.total": 275,
        "genops.budget.name": "enterprise-123_monthly_ai_budget",
        "genops.budget.allocated": 500.0,
        "genops.budget.consumed": 45.67,
        "genops.budget.utilization_percent": 9.13
    }
    
    print("📈 Sample cost attribution telemetry:")
    for key, value in sample_telemetry.items():
        print(f"   {key}: {value}")
    
    print(f"\n💡 This enables in your dashboards:")
    print(f"   • Cost breakdown by customer, team, project, feature")
    print(f"   • Real-time profit margin calculations")
    print(f"   • Budget utilization alerts and forecasting")
    print(f"   • Usage-based billing automation")
    print(f"   • Customer profitability analysis")

def show_integration_examples():
    """
    Show how this integrates with billing and analytics systems.
    """
    print("\n🔗 INTEGRATION WITH BUSINESS SYSTEMS")
    print("=" * 60)
    
    integrations = {
        "Billing Systems": [
            "Stripe - Usage-based billing with metered API costs",
            "Chargebee - Subscription billing with AI usage add-ons", 
            "Zuora - Enterprise billing with detailed cost attribution",
            "Custom billing - API-driven cost allocation and invoicing"
        ],
        "Analytics Platforms": [
            "Datadog - Cost dashboards and customer profitability metrics",
            "Grafana - Real-time cost visualization and budget alerts",
            "Tableau - Customer analytics and cost trend reporting", 
            "Custom dashboards - OpenTelemetry data to any visualization tool"
        ],
        "Business Intelligence": [
            "Looker - Customer LTV analysis with AI cost components",
            "PowerBI - Executive reporting on AI cost efficiency",
            "Amplitude - Product analytics with AI feature cost correlation",
            "Mixpanel - User behavior analysis with cost attribution"
        ]
    }
    
    for category, tools in integrations.items():
        print(f"\n🔧 {category}:")
        for tool in tools:
            print(f"   • {tool}")

def main():
    """
    Run the complete cost attribution demonstration.
    """
    print("📊 GenOps AI: Cost Per Customer Attribution Demo")
    print("=" * 80)
    print("\nThis demo shows how GenOps AI enables precise cost attribution")
    print("across customers, teams, and features for multi-tenant AI applications.")
    
    # Demonstrate multi-tenant tracking
    demonstrate_multi_tenant_cost_tracking()
    
    # Real-time tracking with API
    demonstrate_real_time_cost_tracking()
    
    # Show analytics capabilities
    show_cost_attribution_analytics()
    
    # Show telemetry structure
    show_telemetry_for_cost_attribution()
    
    # Show integration examples
    show_integration_examples()
    
    print(f"\n🎯 KEY TAKEAWAYS")
    print("=" * 60)
    print("✅ Precise cost attribution to customers, teams, projects, and features")
    print("✅ Real-time profit margin and customer profitability analysis")
    print("✅ Automated budget tracking and utilization alerts")
    print("✅ Usage-based billing integration and chargeback automation")
    print("✅ Complete cost visibility for pricing strategy optimization")
    
    print(f"\n📚 NEXT STEPS")
    print("=" * 60)
    print("1. Implement customer attribution in your AI application calls")
    print("2. Set up cost dashboards in your observability platform")
    print("3. Configure budget alerts for high-usage customers") 
    print("4. Integrate with your billing system for usage-based pricing")
    print("5. Analyze customer profitability and optimize pricing strategy")
    
    print(f"\n🔗 Learn more: https://docs.genopsai.org/cost-attribution")

if __name__ == "__main__":
    main()