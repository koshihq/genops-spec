#!/usr/bin/env python3
"""
Multi-Provider Cost Tracking Example

Demonstrates how GenOps tracks costs across multiple underlying providers 
when using OpenRouter's intelligent routing system.

Usage:
    export OPENROUTER_API_KEY="your-key"
    python multi_provider_costs.py

Key features demonstrated:
- Cost attribution across multiple backend providers (OpenAI, Anthropic, Meta, etc.)
- Provider routing and fallback monitoring
- Unified cost aggregation and reporting
- Cross-provider budget tracking
"""

import os
from typing import Dict, List, Any
import time

def multi_provider_cost_demo():
    """Demonstrate multi-provider cost tracking with OpenRouter."""
    
    print("🌐 Multi-Provider Cost Tracking with OpenRouter")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Missing API key. Set OPENROUTER_API_KEY environment variable.")
        return
    
    try:
        from genops.providers.openrouter import instrument_openrouter
        from genops.providers.openrouter_pricing import get_cost_breakdown
        
        # Create instrumented client
        print("🔧 Setting up instrumented OpenRouter client...")
        client = instrument_openrouter(openrouter_api_key=api_key)
        print("   ✅ Client ready for multi-provider tracking")
        
        # Define test scenarios across different providers
        provider_scenarios = [
            {
                "provider_family": "OpenAI",
                "models": ["openai/gpt-4o", "openai/gpt-4o-mini", "openai/gpt-3.5-turbo"],
                "task": "Code a simple Python function to calculate fibonacci numbers.",
                "expected_provider": "openai"
            },
            {
                "provider_family": "Anthropic", 
                "models": ["anthropic/claude-3-5-sonnet", "anthropic/claude-3-haiku"],
                "task": "Explain the philosophical implications of artificial consciousness.",
                "expected_provider": "anthropic"
            },
            {
                "provider_family": "Meta",
                "models": ["meta-llama/llama-3.2-3b-instruct", "meta-llama/llama-3.1-8b-instruct"],
                "task": "Summarize the key benefits of open source software.",
                "expected_provider": "meta"
            },
            {
                "provider_family": "Google",
                "models": ["google/gemini-1.5-flash", "google/gemma-2-9b-it"],
                "task": "What are the latest developments in quantum computing?",
                "expected_provider": "google"
            },
            {
                "provider_family": "Mistral",
                "models": ["mistralai/mistral-small", "mistralai/mixtral-8x7b-instruct"],
                "task": "Design a marketing strategy for a sustainable energy company.",
                "expected_provider": "mistral"
            }
        ]
        
        # Track results across all providers
        provider_costs = {}
        model_costs = {}
        total_cost = 0.0
        governance_attrs = {
            "team": "multi-provider-research",
            "project": "cost-optimization-study",
            "customer_id": "research-division",
            "environment": "analysis"
        }
        
        print(f"\n🔄 Testing models across {len(provider_scenarios)} provider families...")
        
        for scenario in provider_scenarios:
            provider_name = scenario["provider_family"]
            print(f"\n🏢 Testing {provider_name} Models")
            print(f"   Task: {scenario['task']}")
            
            provider_total = 0.0
            
            for model in scenario["models"]:
                print(f"\n   📡 Model: {model}")
                
                try:
                    # Make request with governance attributes
                    start_time = time.time()
                    response = client.chat_completions_create(
                        model=model,
                        messages=[{"role": "user", "content": scenario["task"]}],
                        max_tokens=150,
                        **governance_attrs
                    )
                    request_time = time.time() - start_time
                    
                    # Extract usage information
                    usage = response.usage
                    if usage:
                        # Get detailed cost breakdown
                        cost_breakdown = get_cost_breakdown(
                            model,
                            actual_provider=scenario["expected_provider"],
                            input_tokens=usage.prompt_tokens,
                            output_tokens=usage.completion_tokens
                        )
                        
                        cost = cost_breakdown["total_cost"]
                        actual_provider = cost_breakdown["provider"]
                        
                        print(f"      ✅ Success! Cost: ${cost:.6f}")
                        print(f"         Tokens: {usage.prompt_tokens} in, {usage.completion_tokens} out")
                        print(f"         Provider: {actual_provider}")
                        print(f"         Latency: {request_time:.2f}s")
                        
                        # Accumulate costs by provider
                        if actual_provider not in provider_costs:
                            provider_costs[actual_provider] = 0.0
                        provider_costs[actual_provider] += cost
                        
                        # Track model costs
                        model_costs[model] = cost
                        provider_total += cost
                        total_cost += cost
                    else:
                        print(f"      ⚠️  No usage data available")
                        
                except Exception as e:
                    print(f"      ❌ Error: {str(e)}")
            
            if provider_total > 0:
                print(f"   💰 {provider_name} Total: ${provider_total:.6f}")
        
        # Display comprehensive cost analysis
        print("\n" + "=" * 60)
        print("📊 Multi-Provider Cost Analysis")
        print("=" * 60)
        
        if total_cost > 0:
            print(f"💰 Grand Total Cost: ${total_cost:.6f}")
            print(f"🏢 Providers Used: {len(provider_costs)}")
            print(f"🤖 Models Tested: {len(model_costs)}")
            
            print(f"\n📈 Cost Breakdown by Provider:")
            sorted_providers = sorted(provider_costs.items(), key=lambda x: x[1], reverse=True)
            for provider, cost in sorted_providers:
                percentage = (cost / total_cost) * 100
                print(f"   • {provider}: ${cost:.6f} ({percentage:.1f}%)")
            
            print(f"\n🎯 Most/Least Expensive Models:")
            sorted_models = sorted(model_costs.items(), key=lambda x: x[1], reverse=True)
            if sorted_models:
                print(f"   💸 Most expensive: {sorted_models[0][0]} (${sorted_models[0][1]:.6f})")
                print(f"   💰 Least expensive: {sorted_models[-1][0]} (${sorted_models[-1][1]:.6f})")
            
            print(f"\n🔍 GenOps Multi-Provider Features:")
            print("   ✅ Automatic provider detection and attribution")
            print("   ✅ Unified cost aggregation across all providers")  
            print("   ✅ Per-provider cost breakdown in telemetry")
            print("   ✅ Model-level cost granularity")
            print("   ✅ Governance attributes propagated to all requests")
            
            print(f"\n📊 Telemetry Attributes Captured:")
            print("   • genops.cost.total: Per-request and aggregated costs")
            print("   • genops.openrouter.actual_provider: Backend provider used")
            print("   • genops.openrouter.predicted_provider: Initial provider prediction")
            print("   • genops.team: multi-provider-research")
            print("   • genops.project: cost-optimization-study")
            print("   • genops.customer_id: research-division")
            
        else:
            print("❌ No successful requests completed")
        
        print(f"\n🎯 Business Value:")
        print("   • Unified billing across 400+ models from 60+ providers")
        print("   • Cost optimization through provider comparison")
        print("   • Budget controls with multi-provider awareness")
        print("   • Vendor-neutral governance and compliance")
        
        print(f"\n✨ Next Steps:")
        print("   • Set up provider-specific budget alerts")
        print("   • Implement cost-aware model selection strategies")
        print("   • Try advanced_features.py for routing control")
        print("   • Review cost_optimization.py for intelligent routing")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Install: pip install genops-ai openai")
    except Exception as e:
        print(f"❌ Error: {e}")


def show_cost_attribution_example():
    """Show how costs are attributed across different dimensions."""
    print("\n📋 Cost Attribution Dimensions")
    print("=" * 40)
    print("GenOps tracks costs across multiple dimensions simultaneously:")
    print()
    
    dimensions = [
        ("🏢 Provider", "openai, anthropic, meta, google, mistral, etc."),
        ("🤖 Model", "gpt-4o, claude-3-sonnet, llama-3.2-3b, etc."),
        ("👥 Team", "ml-team, product-team, research-team"),
        ("📁 Project", "chatbot, content-generation, code-assistant"),
        ("👤 Customer", "customer-123, enterprise-client, internal"),
        ("🌍 Environment", "development, staging, production"),
        ("💼 Cost Center", "R&D, Marketing, Engineering")
    ]
    
    for dimension, examples in dimensions:
        print(f"{dimension}: {examples}")
    
    print("\n🎯 Example Multi-Dimensional Query:")
    print("'Show me costs for ml-team using Anthropic models in production for customer-123'")
    print("→ Precise cost attribution across all dimensions simultaneously")


if __name__ == "__main__":
    multi_provider_cost_demo()
    show_cost_attribution_example()