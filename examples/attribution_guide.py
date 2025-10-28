#!/usr/bin/env python3
"""
🏷️ Complete Guide to Usage Attribution and Tagging in GenOps AI

This example demonstrates all the ways developers can tag and associate
AI usage by teams, projects, customers, features, and other dimensions.

ATTRIBUTION DIMENSIONS SUPPORTED:
✅ Teams & Projects - Internal organization and cost centers
✅ Customers - Multi-tenant cost attribution and billing  
✅ Features - Granular product feature usage tracking
✅ Users - Individual user activity and usage patterns
✅ Environment - Production, staging, development separation
✅ Custom - Any business-specific dimensions you need

Run this example to see all tagging patterns in action!
"""

import os
from typing import Dict, Any

# Import GenOps attribution and instrumentation
import genops
from genops.providers.openai import instrument_openai

def demonstrate_global_defaults():
    """
    Show how to set global default attributes to avoid repetitive tagging.
    
    This is the most developer-friendly approach for consistent attribution.
    """
    print("\n🌍 GLOBAL DEFAULT ATTRIBUTION")
    print("=" * 60)
    
    # Set defaults once at application startup
    genops.set_default_attributes(
        team="platform-engineering",
        project="ai-services",
        environment="production",
        cost_center="engineering"
    )
    
    print("✅ Set global defaults:")
    defaults = genops.get_default_attributes()
    for key, value in defaults.items():
        print(f"   {key}: {value}")
    
    print("\n💡 Now ALL AI operations inherit these defaults automatically!")

def demonstrate_provider_tagging():
    """
    Show provider-level tagging with automatic inheritance of defaults.
    """
    print("\n🤖 PROVIDER-LEVEL TAGGING (with defaults inherited)")
    print("=" * 60)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY to see real API calls")
        print("Showing tagging pattern without actual API call:")
        
        print("\n🏷️ Example: Customer support chat")
        print("Code:")
        print("""
        client = instrument_openai(api_key="your-key")
        response = client.chat_completions_create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello!"}],
            # Only specify what's unique to this operation
            customer_id="enterprise-123",
            feature="live-chat", 
            user_id="user_456"
            # team, project, environment automatically inherited!
        )
        """)
        
        print("📊 Resulting telemetry attributes:")
        print("   genops.team: platform-engineering (from defaults)")
        print("   genops.project: ai-services (from defaults)") 
        print("   genops.environment: production (from defaults)")
        print("   genops.cost_center: engineering (from defaults)")
        print("   genops.customer_id: enterprise-123 (operation-specific)")
        print("   genops.feature: live-chat (operation-specific)")
        print("   genops.user_id: user_456 (operation-specific)")
        print("   + cost, tokens, model data automatically recorded")
        return
    
    try:
        # Real API example with inheritance
        client = instrument_openai(api_key=os.getenv("OPENAI_API_KEY"))
        
        print("🏷️ Making AI call with mixed attribution...")
        response = client.chat_completions_create(
            model="gpt-3.5-turbo", 
            messages=[{"role": "user", "content": "What is AI governance?"}],
            # Only specify operation-specific attributes
            customer_id="enterprise-123",
            feature="ai-assistant",
            user_id="demo_user"
            # team, project, environment, cost_center inherited from defaults
        )
        
        print(f"✅ Response: {response.choices[0].message.content[:100]}...")
        print("📊 Complete attribution telemetry automatically recorded!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def demonstrate_context_scoping():
    """
    Show context-based attribution for request/session-scoped tagging.
    """
    print("\n🎯 CONTEXT-BASED ATTRIBUTION")  
    print("=" * 60)
    
    print("📝 Scenario: Web request handler with automatic user/customer context")
    print("\nCode pattern:")
    print("""
    @app.route('/api/chat')
    def chat_endpoint():
        # Set context for this request
        genops.set_context(
            user_id=request.user.id,
            customer_id=request.headers.get('X-Customer-ID'), 
            request_id=request.id,
            session_id=request.session.id
        )
        
        # All AI operations in this request inherit the context
        response = ai_chat(request.json['message'])
        return response
    """)
    
    # Simulate request context
    genops.set_context(
        user_id="user_789",
        customer_id="startup-456", 
        request_id="req_abc123",
        session_id="sess_def456"
    )
    
    print("✅ Context set for current operation scope:")
    context = genops.get_context()
    for key, value in context.items():
        print(f"   {key}: {value}")
    
    print("\n💡 All AI calls in this scope automatically get these attributes!")
    
    # Show effective attributes
    print("\n📊 Effective attributes for an AI operation:")
    effective = genops.get_effective_attributes(feature="chat", priority="high")
    for key, value in effective.items():
        print(f"   {key}: {value}")
    
    # Clear context (important in web apps)
    genops.clear_context()
    print("\n🧹 Context cleared (important at end of request)")

def demonstrate_convenience_functions():
    """
    Show convenience functions for common attribution patterns.
    """
    print("\n🎛️ CONVENIENCE FUNCTIONS FOR COMMON PATTERNS")
    print("=" * 60)
    
    # Team-based defaults
    print("1. 🏢 Setting team defaults:")
    genops.set_team_defaults(
        team="ml-engineering",
        project="recommendation-engine",
        cost_center="product-engineering" 
    )
    print("   ✅ Team defaults set for ml-engineering")
    
    # Customer context
    print("\n2. 👥 Setting customer context:")
    genops.set_customer_context(
        customer_id="premium-789",
        customer_name="TechGiant Ltd",
        tier="enterprise"
    )
    print("   ✅ Customer context set for TechGiant Ltd")
    
    # User context  
    print("\n3. 👤 Setting user context:")
    genops.set_user_context(user_id="admin_123", role="administrator")
    print("   ✅ User context set for admin_123")
    
    print("\n📊 Final effective attributes:")
    effective = genops.get_effective_attributes(feature="admin-panel", action="user-query")
    for key, value in sorted(effective.items()):
        print(f"   {key}: {value}")

def demonstrate_attribution_hierarchy():
    """
    Show how attribution priority works: operation > context > defaults.
    """
    print("\n🏆 ATTRIBUTION PRIORITY HIERARCHY")
    print("=" * 60)
    
    # Set up different levels
    genops.set_default_attributes(
        team="default-team",
        environment="development",
        cost_center="default-cost"
    )
    
    genops.set_context(
        team="context-team",  # Overrides default
        customer_id="context-customer",
        user_id="context-user"
    )
    
    # Operation-specific overrides
    operation_attrs = {
        "team": "operation-team",  # Highest priority
        "feature": "specific-feature"
    }
    
    print("🔄 Priority demonstration:")
    print("   1. Defaults: team='default-team', environment='development'")
    print("   2. Context: team='context-team' (overrides default), customer_id='context-customer'")  
    print("   3. Operation: team='operation-team' (overrides context), feature='specific-feature'")
    
    effective = genops.get_effective_attributes(**operation_attrs)
    
    print("\n🏆 Final effective attributes (highest priority wins):")
    for key, value in sorted(effective.items()):
        priority = "OPERATION" if key in operation_attrs else \
                  "CONTEXT" if key in genops.get_context() else "DEFAULT"
        print(f"   {key}: {value} ({priority})")

def demonstrate_multi_tenant_patterns():
    """
    Show common multi-tenant SaaS attribution patterns.
    """
    print("\n🏢 MULTI-TENANT SAAS ATTRIBUTION PATTERNS")
    print("=" * 60)
    
    # Pattern 1: Enterprise customer with teams
    print("1. 🏢 Enterprise customer with internal teams:")
    enterprise_attrs = genops.get_effective_attributes(
        customer_id="enterprise-456",
        customer_name="Acme Corporation", 
        customer_tier="enterprise",
        customer_team="acme-engineering",
        customer_project="ai-automation",
        feature="document-analysis"
    )
    
    for key, value in sorted(enterprise_attrs.items()):
        print(f"   {key}: {value}")
    
    # Pattern 2: Individual user in freemium model
    print("\n2. 👤 Individual user (freemium model):")
    individual_attrs = genops.get_effective_attributes(
        user_id="user_123",
        user_tier="freemium",
        feature="chat-assistant",
        usage_limit="20_per_month"
    )
    
    for key, value in sorted(individual_attrs.items()):
        print(f"   {key}: {value}")
    
    # Pattern 3: API customer with rate limiting
    print("\n3. 🔌 API customer with rate limiting:")
    api_attrs = genops.get_effective_attributes(
        api_key="ak_prod_abc123",
        customer_id="api-customer-789", 
        rate_limit_tier="pro",
        feature="api-inference",
        quota_remaining="5000_requests"
    )
    
    for key, value in sorted(api_attrs.items()):
        print(f"   {key}: {value}")

def show_observability_integration():
    """
    Show how attributed data appears in observability platforms.
    """
    print("\n📊 OBSERVABILITY PLATFORM INTEGRATION")
    print("=" * 60)
    
    print("🎯 All attributed data automatically exports to your observability stack:")
    print("\n📈 Sample telemetry data structure:")
    
    sample_telemetry = {
        # Core operation info  
        "genops.operation.type": "ai.inference",
        "genops.operation.name": "openai.chat.completions.create",
        "genops.timestamp": 1640995200,
        
        # Attribution dimensions
        "genops.team": "platform-engineering",
        "genops.project": "ai-services", 
        "genops.customer_id": "enterprise-123",
        "genops.customer": "Acme Corporation",
        "genops.customer_tier": "enterprise",
        "genops.feature": "chat-assistant",
        "genops.user_id": "user_456", 
        "genops.environment": "production",
        "genops.cost_center": "engineering",
        
        # Cost and usage data
        "genops.cost.total": 0.0234,
        "genops.cost.currency": "USD",
        "genops.cost.provider": "openai", 
        "genops.cost.model": "gpt-3.5-turbo",
        "genops.tokens.input": 150,
        "genops.tokens.output": 75,
        "genops.tokens.total": 225
    }
    
    for key, value in sample_telemetry.items():
        print(f"   {key}: {value}")
    
    print(f"\n💡 This enables powerful queries in your observability platform:")
    print(f"   • Cost by customer: WHERE genops.customer_id = 'enterprise-123'")
    print(f"   • Team usage: WHERE genops.team = 'platform-engineering'") 
    print(f"   • Feature costs: WHERE genops.feature = 'chat-assistant'")
    print(f"   • Environment breakdown: WHERE genops.environment = 'production'")
    print(f"   • User activity: WHERE genops.user_id = 'user_456'")

def show_framework_integration_examples():
    """
    Show integration patterns with popular web frameworks.
    """
    print("\n🔧 WEB FRAMEWORK INTEGRATION PATTERNS")
    print("=" * 60)
    
    print("🌟 Flask Integration:")
    print("""
from flask import Flask, request, g
import genops

app = Flask(__name__)

@app.before_request
def set_genops_context():
    genops.set_context(
        user_id=getattr(g, 'user_id', None),
        customer_id=request.headers.get('X-Customer-ID'),
        request_id=request.id,
        endpoint=request.endpoint
    )

@app.after_request  
def clear_genops_context(response):
    genops.clear_context()
    return response
    """)
    
    print("\n🚀 FastAPI Integration:")
    print("""
from fastapi import FastAPI, Depends, Request
import genops

app = FastAPI()

async def set_genops_context(request: Request):
    genops.set_context(
        user_id=request.headers.get('X-User-ID'),
        customer_id=request.headers.get('X-Customer-ID'),
        request_id=request.headers.get('X-Request-ID'),
        endpoint=request.url.path
    )
    return request

@app.middleware("http")
async def genops_middleware(request: Request, call_next):
    await set_genops_context(request)
    response = await call_next(request)
    genops.clear_context()
    return response
    """)
    
    print("\n🎸 Django Integration:")
    print("""
# middleware.py
import genops

class GenOpsAttributionMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        genops.set_context(
            user_id=getattr(request.user, 'id', None),
            customer_id=request.headers.get('X-Customer-ID'),
            session_id=request.session.session_key,
            view_name=request.resolver_match.view_name if request.resolver_match else None
        )
        
        response = self.get_response(request)
        genops.clear_context()
        return response

# settings.py 
MIDDLEWARE = [
    # ... other middleware
    'myapp.middleware.GenOpsAttributionMiddleware',
]
    """)

def main():
    """
    Run the complete attribution and tagging demonstration.
    """
    print("🏷️ GenOps AI: Complete Attribution and Tagging Guide")
    print("=" * 80)
    print("\nThis guide shows all the ways to tag and associate AI usage with")
    print("teams, projects, customers, features, and other business dimensions.")
    
    # Run all demonstrations
    demonstrate_global_defaults()
    demonstrate_provider_tagging() 
    demonstrate_context_scoping()
    demonstrate_convenience_functions()
    demonstrate_attribution_hierarchy()
    demonstrate_multi_tenant_patterns()
    show_observability_integration()
    show_framework_integration_examples()
    
    print(f"\n🎯 KEY TAKEAWAYS")
    print("=" * 60)
    print("✅ Set global defaults once to avoid repetitive tagging")
    print("✅ Use context for request/session-scoped attribution") 
    print("✅ Operation-specific tags override context and defaults")
    print("✅ All attribution automatically exports via OpenTelemetry")
    print("✅ Supports any business dimension: teams, customers, features, etc.")
    print("✅ Framework middleware handles web app attribution automatically")
    
    print(f"\n📚 NEXT STEPS")
    print("=" * 60)
    print("1. Set up global defaults for your application's attribution needs")
    print("2. Implement context middleware for your web framework")
    print("3. Configure your observability platform to query attributed data")
    print("4. Build dashboards showing cost/usage by attribution dimensions") 
    print("5. Set up alerts and budgets based on team/customer/feature usage")
    
    print(f"\n🔗 Learn more: https://github.com/KoshiHQ/GenOps-AI/tree/main/docs")

if __name__ == "__main__":
    main()