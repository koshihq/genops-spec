#!/usr/bin/env python3
"""
LiteLLM Basic Tracking Patterns with GenOps

Demonstrates manual tracking patterns and context managers for fine-grained
control over GenOps governance telemetry in LiteLLM applications. This shows
alternative approaches to auto-instrumentation for cases requiring explicit
control over tracking.

Usage:
    export OPENAI_API_KEY="your_key_here"
    python basic_tracking.py

Features:
    - Manual context managers for explicit tracking control
    - Custom attribution and tagging per request
    - Conditional tracking based on business logic
    - Performance-optimized tracking patterns
    - Request-level governance policies
"""

import os
import sys
import time
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import contextmanager

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def check_setup():
    """Check if required packages and API keys are available."""
    print("🔍 Checking setup for basic tracking patterns...")
    
    # Check imports
    try:
        import litellm
        from genops.providers.litellm import track_completion, get_usage_stats
        print("✅ LiteLLM and GenOps available")
    except ImportError as e:
        print(f"❌ Import error: [Error details redacted for security]")
        print("💡 Install: pip install litellm genops[litellm]")
        return False
    
    # Check API keys
    api_keys_found = []
    api_checks = {
        'OpenAI': 'OPENAI_API_KEY',
        'Anthropic': 'ANTHROPIC_API_KEY',
        'Google': 'GOOGLE_API_KEY',
        'Cohere': 'COHERE_API_KEY'
    }
    
    for provider, env_var in api_checks.items():
        if os.getenv(env_var):
            api_keys_found.append(provider)
            print(f"✅ {provider} API key configured")
    
    if not api_keys_found:
        print("⚠️  No API keys configured")
        print("💡 Set at least one: export OPENAI_API_KEY=your_key")
        print("   Will use demo mode for tracking patterns demonstration")
    else:
        print(f"🎯 Ready with {len(api_keys_found)} provider(s): {', '.join(api_keys_found)}")
    
    return True


def demo_basic_context_manager():
    """Demonstrate basic context manager usage for tracking."""
    print("\n" + "="*60)
    print("🎯 Demo: Basic Context Manager Tracking")
    print("="*60)
    
    import litellm
    from genops.providers.litellm import track_completion, get_usage_stats
    
    print("Manual tracking gives you explicit control over when and how")
    print("to track LiteLLM requests, with custom attribution per request.")
    
    # Example 1: Basic tracking with context manager
    print("\n📋 Example 1: Basic context manager usage")
    
    try:
        with track_completion(
            model="gpt-3.5-turbo",
            team="analytics-team", 
            project="user-insights",
            customer_id="customer-123"
        ) as context:
            
            print("   🔄 Making request with explicit tracking...")
            
            response = litellm.completion(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "What is machine learning? Be brief."}
                ],
                max_tokens=50,
                timeout=10
            )
            
            # Context provides immediate access to tracking data
            print(f"   ✅ Request completed")
            print(f"   📊 Cost: ${context.cost:.6f}")
            print(f"   🎫 Tokens: {context.total_tokens}")
            print(f"   ⏱️ Duration: {context.duration_ms:.0f}ms")
            print(f"   🏷️ Team: {context.team}, Project: {context.project}")
            
    except Exception as e:
        print(f"   ⚠️ Request failed: [Error details redacted for security]")
        print("   (This is expected if no API key configured)")


def demo_conditional_tracking():
    """Demonstrate conditional tracking based on business logic."""
    print("\n" + "="*60)
    print("🧠 Demo: Conditional Tracking Patterns")
    print("="*60)
    
    import litellm
    from genops.providers.litellm import track_completion
    
    print("Track requests conditionally based on business logic:")
    print("• High-value customers get detailed tracking")
    print("• Internal testing uses lightweight tracking")
    print("• Production requests include compliance metadata")
    
    # Simulate different user scenarios
    user_scenarios = [
        {
            "user_type": "enterprise_customer",
            "customer_id": "enterprise-456", 
            "tier": "premium",
            "track_detailed": True
        },
        {
            "user_type": "internal_testing",
            "customer_id": None,
            "tier": "internal", 
            "track_detailed": False
        },
        {
            "user_type": "freemium_user",
            "customer_id": "free-789",
            "tier": "free",
            "track_detailed": False
        }
    ]
    
    for scenario in user_scenarios:
        print(f"\n📋 Scenario: {scenario['user_type']} ({scenario['tier']} tier)")
        
        # Conditional tracking based on user tier
        if scenario['track_detailed']:
            # Detailed tracking for premium customers
            tracking_context = {
                "team": "premium-support",
                "project": "enterprise-ai",
                "customer_id": scenario['customer_id'],
                "custom_tags": {
                    "tier": scenario['tier'],
                    "tracking_level": "detailed",
                    "compliance_required": True
                }
            }
            print("   🔍 Using detailed tracking with compliance metadata")
        else:
            # Lightweight tracking for others
            tracking_context = {
                "team": "general-support", 
                "project": "community-ai",
                "customer_id": scenario['customer_id'],
                "custom_tags": {
                    "tier": scenario['tier'],
                    "tracking_level": "basic"
                }
            }
            print("   ⚡ Using lightweight tracking")
        
        try:
            with track_completion(model="gpt-3.5-turbo", **tracking_context) as context:
                # Simulate API call
                print(f"   🔄 Simulating request for {scenario['user_type']}...")
                
                # In a real scenario, you'd make the actual API call here
                response = litellm.completion(
                    model="gpt-3.5-turbo", 
                    messages=[{"role": "user", "content": "Hello!"}],
                    max_tokens=5,
                    timeout=5
                )
                
                print(f"   ✅ Tracked with tags: {context.custom_tags}")
                
        except Exception as e:
            print(f"   ⚠️ Request simulation failed: [Error details redacted for security]")


def demo_performance_patterns():
    """Demonstrate performance-optimized tracking patterns."""
    print("\n" + "="*60)
    print("⚡ Demo: Performance-Optimized Tracking")
    print("="*60)
    
    from genops.providers.litellm import track_completion
    
    print("Performance patterns for high-volume applications:")
    print("• Sampling-based tracking for cost efficiency")  
    print("• Batch processing for reduced overhead")
    print("• Asynchronous tracking for minimal latency impact")
    
    # Example 1: Sampling-based tracking
    print("\n📋 Example 1: Sampling-based tracking (10% sample rate)")
    
    import random
    
    requests_processed = 0
    requests_tracked = 0
    
    for request_id in range(20):  # Simulate 20 requests
        should_track = random.random() < 0.1  # 10% sampling
        
        if should_track:
            try:
                with track_completion(
                    model="gpt-3.5-turbo",
                    team="high-volume-service",
                    project="api-gateway",
                    custom_tags={
                        "request_id": f"req-{request_id}",
                        "sampling": True,
                        "sample_rate": 0.1
                    }
                ) as context:
                    # Simulate minimal API call
                    print(f"   📊 Tracking request {request_id} (sampled)")
                    requests_tracked += 1
            except Exception:
                pass
        else:
            # Process without detailed tracking
            print(f"   ⚡ Processing request {request_id} (no tracking)")
        
        requests_processed += 1
    
    print(f"\n   📈 Results: {requests_tracked}/{requests_processed} requests tracked")
    print(f"   💰 Tracking overhead reduced by {((requests_processed - requests_tracked) / requests_processed) * 100:.0f}%")


def demo_custom_attribution():
    """Demonstrate custom attribution and tagging patterns."""  
    print("\n" + "="*60)
    print("🏷️ Demo: Custom Attribution & Tagging")
    print("="*60)
    
    from genops.providers.litellm import track_completion
    
    print("Custom attribution enables detailed cost allocation:")
    print("• Multi-dimensional cost attribution")
    print("• Feature-specific tracking") 
    print("• A/B test measurement")
    
    # Example: Multi-dimensional attribution
    attribution_examples = [
        {
            "scenario": "Feature development",
            "team": "product-ai",
            "project": "recommendation-engine", 
            "feature": "personalization-v2",
            "environment": "development",
            "cost_center": "engineering",
            "experiment_id": None
        },
        {
            "scenario": "A/B testing",
            "team": "growth-team",
            "project": "onboarding-optimization",
            "feature": "ai-guided-setup",
            "environment": "production", 
            "cost_center": "marketing",
            "experiment_id": "exp-onboard-123"
        },
        {
            "scenario": "Customer support",
            "team": "support-ai", 
            "project": "automated-responses",
            "feature": "ticket-classification",
            "environment": "production",
            "cost_center": "operations", 
            "experiment_id": None
        }
    ]
    
    for example in attribution_examples:
        print(f"\n📋 {example['scenario']}:")
        
        # Build comprehensive tracking context
        tracking_context = {
            "team": example['team'],
            "project": example['project'],
            "environment": example['environment'],
            "custom_tags": {
                "feature": example['feature'],
                "cost_center": example['cost_center'],
                "scenario": example['scenario']
            }
        }
        
        if example['experiment_id']:
            tracking_context['custom_tags']['experiment_id'] = example['experiment_id']
            tracking_context['custom_tags']['is_experiment'] = True
        
        try:
            with track_completion(model="gpt-3.5-turbo", **tracking_context) as context:
                print(f"   🏷️  Team: {context.team}")
                print(f"   📁 Project: {context.project}")
                print(f"   🌍 Environment: {context.environment}")
                print(f"   🎯 Feature: {context.custom_tags['feature']}")
                
                if context.custom_tags.get('experiment_id'):
                    print(f"   🧪 Experiment: {context.custom_tags['experiment_id']}")
                
                # Simulate tracking
                print(f"   ✅ Attribution configured")
                
        except Exception as e:
            print(f"   ⚠️ Attribution setup failed: [Error details redacted for security]")


def demo_usage_analytics():
    """Demonstrate usage analytics and reporting patterns."""
    print("\n" + "="*60)
    print("📊 Demo: Usage Analytics & Reporting")
    print("="*60)
    
    from genops.providers.litellm import get_usage_stats, get_cost_summary
    
    print("Analyze usage patterns across teams, projects, and features:")
    
    # Get comprehensive usage statistics
    usage_stats = get_usage_stats()
    
    print(f"\n📈 Current Session Statistics:")
    print(f"   Total requests: {usage_stats.get('total_requests', 0)}")
    print(f"   Total cost: ${usage_stats.get('total_cost', 0):.6f}")
    print(f"   Average cost per request: ${usage_stats.get('avg_cost_per_request', 0):.6f}")
    
    if usage_stats.get('provider_usage'):
        print(f"\n🔌 Provider Usage Breakdown:")
        for provider, stats in usage_stats['provider_usage'].items():
            print(f"   • {provider}: {stats.get('requests', 0)} requests, ${stats.get('cost', 0):.6f}")
    
    # Get cost summary with different groupings
    cost_by_team = get_cost_summary(group_by="team")
    cost_by_project = get_cost_summary(group_by="project")
    
    if cost_by_team.get('cost_by_team'):
        print(f"\n👥 Cost by Team:")
        for team, cost in cost_by_team['cost_by_team'].items():
            percentage = (cost / cost_by_team['total_cost']) * 100 if cost_by_team['total_cost'] > 0 else 0
            print(f"   • {team}: ${cost:.6f} ({percentage:.1f}%)")
    
    if cost_by_project.get('cost_by_project'):
        print(f"\n📁 Cost by Project:")  
        for project, cost in cost_by_project['cost_by_project'].items():
            percentage = (cost / cost_by_project['total_cost']) * 100 if cost_by_project['total_cost'] > 0 else 0
            print(f"   • {project}: ${cost:.6f} ({percentage:.1f}%)")
    
    print(f"\n💡 Analytics Insights:")
    if usage_stats.get('total_requests', 0) > 0:
        print(f"   • Average request cost optimized for tracking efficiency")
        print(f"   • Multi-dimensional attribution enables cost allocation")
        print(f"   • Performance patterns minimize operational overhead")
    else:
        print(f"   • No tracked requests in current session")
        print(f"   • Run with valid API keys for live tracking data")


@contextmanager
def custom_tracking_context(
    model: str,
    business_context: Dict[str, Any],
    performance_mode: str = "balanced"
):
    """
    Custom context manager demonstrating advanced tracking patterns.
    
    Args:
        model: LLM model to track
        business_context: Business metadata for attribution
        performance_mode: "detailed", "balanced", or "minimal"
    """
    from genops.providers.litellm import track_completion
    
    # Adjust tracking based on performance mode
    if performance_mode == "detailed":
        tracking_config = {
            "enable_cost_tracking": True,
            "enable_performance_metrics": True,
            "custom_tags": business_context
        }
    elif performance_mode == "minimal":
        tracking_config = {
            "enable_cost_tracking": True,
            "enable_performance_metrics": False,
            "custom_tags": {"mode": "minimal"}
        }
    else:  # balanced
        tracking_config = {
            "enable_cost_tracking": True,
            "enable_performance_metrics": True,
            "custom_tags": {**business_context, "mode": "balanced"}
        }
    
    with track_completion(model=model, **tracking_config) as context:
        yield context


def demo_advanced_patterns():
    """Demonstrate advanced tracking patterns."""
    print("\n" + "="*60)
    print("🚀 Demo: Advanced Tracking Patterns")
    print("="*60)
    
    print("Advanced patterns for enterprise scenarios:")
    print("• Custom context managers")
    print("• Dynamic configuration")
    print("• Business-aware tracking")
    
    business_scenarios = [
        {
            "name": "High-value customer interaction",
            "context": {
                "customer_tier": "enterprise",
                "support_level": "premium", 
                "interaction_type": "technical_support"
            },
            "performance_mode": "detailed"
        },
        {
            "name": "Bulk processing job",
            "context": {
                "job_type": "data_processing",
                "batch_size": 1000,
                "priority": "low"
            },
            "performance_mode": "minimal"
        }
    ]
    
    for scenario in business_scenarios:
        print(f"\n📋 {scenario['name']}:")
        
        try:
            with custom_tracking_context(
                model="gpt-3.5-turbo",
                business_context=scenario['context'],
                performance_mode=scenario['performance_mode']
            ) as context:
                
                print(f"   🎯 Performance mode: {scenario['performance_mode']}")
                print(f"   🏷️ Business context: {scenario['context']}")
                print(f"   ✅ Custom tracking context active")
                
        except Exception as e:
            print(f"   ⚠️ Advanced pattern failed: [Error details redacted for security]")


def main():
    """Run the complete basic tracking demonstration."""
    
    print("🎯 LiteLLM + GenOps: Basic Tracking Patterns")
    print("=" * 60)
    print("Manual tracking patterns for fine-grained governance control")
    print("Alternative to auto-instrumentation for explicit request management")
    
    # Check setup
    if not check_setup():
        print("\n❌ Setup incomplete. Please resolve issues above.")
        return 1
    
    try:
        # Run demonstrations
        demo_basic_context_manager()
        demo_conditional_tracking()
        demo_performance_patterns()
        demo_custom_attribution()
        demo_usage_analytics()
        demo_advanced_patterns()
        
        print("\n" + "="*60)
        print("🎉 Basic Tracking Patterns Complete!")
        
        print("\n🚀 Key Patterns Demonstrated:")
        print("   ✅ Manual context managers for explicit control")
        print("   ✅ Conditional tracking based on business logic")
        print("   ✅ Performance optimization for high-volume usage")
        print("   ✅ Custom attribution and multi-dimensional tagging")
        print("   ✅ Usage analytics and cost reporting")
        print("   ✅ Advanced enterprise patterns")
        
        print("\n💡 When to Use Manual Tracking:")
        print("   • Need explicit control over tracking lifecycle")
        print("   • Conditional tracking based on business rules")
        print("   • Performance-critical applications requiring optimization")
        print("   • Complex attribution requirements")
        print("   • Integration with existing monitoring systems")
        
        print("\n📖 Next Steps:")
        print("   • Compare with auto_instrumentation.py for automatic patterns")
        print("   • Explore production_patterns.py for scaling strategies")  
        print("   • Try multi_provider_costs.py for cost optimization")
        print("   • Choose the tracking pattern that fits your use case!")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Demo failed: [Error details redacted for security]")
        print("💡 For debugging, check your API key configuration")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)