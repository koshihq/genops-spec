#!/usr/bin/env python3
"""
OpenRouter Advanced Features Example

Demonstrates advanced OpenRouter capabilities with GenOps governance:
- Provider selection and routing strategies
- Fallback handling and monitoring
- Custom routing preferences
- Advanced telemetry capture

Usage:
    export OPENROUTER_API_KEY="your-key"
    python advanced_features.py

Key features demonstrated:
- Explicit provider selection
- Routing strategy configuration
- Fallback detection and monitoring
- Advanced governance controls
"""

import os
import time
from typing import Dict, List, Any

def advanced_features_demo():
    """Demonstrate advanced OpenRouter features with GenOps."""
    
    print("🚀 Advanced OpenRouter Features with GenOps")
    print("=" * 55)
    
    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Missing API key. Set OPENROUTER_API_KEY environment variable.")
        return
    
    try:
        from genops.providers.openrouter import instrument_openrouter
        
        print("🔧 Setting up advanced OpenRouter client...")
        client = instrument_openrouter(openrouter_api_key=api_key)
        print("   ✅ Client configured for advanced features")
        
        # Demo 1: Explicit Provider Selection
        print(f"\n🎯 Feature 1: Explicit Provider Selection")
        print("=" * 40)
        
        provider_preferences = [
            {
                "name": "Force Anthropic",
                "model": "anthropic/claude-3-sonnet", 
                "provider": "anthropic",
                "task": "Explain quantum entanglement in simple terms."
            },
            {
                "name": "Prefer OpenAI",
                "model": "openai/gpt-4o",
                "provider": "openai", 
                "task": "Write a Python function to sort a list."
            },
            {
                "name": "Any Provider (OpenRouter decides)",
                "model": "meta-llama/llama-3.1-8b-instruct",
                "provider": None,  # Let OpenRouter route automatically
                "task": "What are the benefits of renewable energy?"
            }
        ]
        
        for pref in provider_preferences:
            print(f"\n   🧪 Test: {pref['name']}")
            print(f"      Model: {pref['model']}")
            print(f"      Provider preference: {pref['provider'] or 'Auto-route'}")
            
            try:
                # Build request with optional provider preference
                request_params = {
                    "model": pref["model"],
                    "messages": [{"role": "user", "content": pref["task"]}],
                    "max_tokens": 100,
                    # Governance attributes
                    "team": "advanced-features-team",
                    "project": "routing-experiments",
                    "experiment_id": f"provider-{pref['name'].lower().replace(' ', '-')}"
                }
                
                # Add provider preference if specified
                if pref["provider"]:
                    request_params["provider"] = pref["provider"]
                
                response = client.chat_completions_create(**request_params)
                
                usage = response.usage
                if usage:
                    print(f"      ✅ Success! Tokens: {usage.total_tokens}")
                    print(f"         Response: {response.choices[0].message.content[:60]}...")
                else:
                    print(f"      ⚠️  No usage data")
                    
            except Exception as e:
                print(f"      ❌ Error: {str(e)}")
        
        # Demo 2: Routing Strategies
        print(f"\n⚡ Feature 2: Routing Strategies")
        print("=" * 35)
        
        routing_strategies = [
            {
                "name": "Least Cost",
                "route": "least-cost",
                "description": "Route to cheapest available provider"
            },
            {
                "name": "Fastest Response", 
                "route": "fastest",
                "description": "Route to fastest provider based on latency"
            },
            {
                "name": "Fallback Chain",
                "route": "fallback",
                "description": "Try multiple providers if first fails"
            }
        ]
        
        # Test same task with different routing strategies
        test_task = "Explain machine learning in one paragraph."
        test_model = "openai/gpt-3.5-turbo"  # Available on multiple providers
        
        for strategy in routing_strategies:
            print(f"\n   🎯 Strategy: {strategy['name']}")
            print(f"      Route: {strategy['route']}")
            print(f"      Goal: {strategy['description']}")
            
            try:
                start_time = time.time()
                response = client.chat_completions_create(
                    model=test_model,
                    messages=[{"role": "user", "content": test_task}],
                    max_tokens=80,
                    route=strategy["route"],  # OpenRouter routing strategy
                    team="routing-optimization",
                    project="strategy-comparison",
                    routing_strategy=strategy["name"]  # Custom governance attribute
                )
                response_time = time.time() - start_time
                
                usage = response.usage
                if usage:
                    print(f"      ✅ Success! Latency: {response_time:.2f}s")
                    print(f"         Tokens: {usage.total_tokens}")
                else:
                    print(f"      ⚠️  No usage data, Latency: {response_time:.2f}s")
                    
            except Exception as e:
                print(f"      ❌ Error: {str(e)}")
        
        # Demo 3: Fallback Monitoring
        print(f"\n🔄 Feature 3: Fallback Detection & Monitoring")
        print("=" * 45)
        
        # Test with models that might trigger fallbacks
        fallback_tests = [
            {
                "name": "High-demand model (might fallback)",
                "model": "openai/gpt-4",  # Popular model, might be rate limited
                "fallbacks": ["openai/gpt-4o", "anthropic/claude-3-sonnet"]
            },
            {
                "name": "Specific provider (with fallback)",
                "model": "anthropic/claude-3-opus",
                "provider": "anthropic",
                "fallbacks": ["anthropic/claude-3-sonnet", "openai/gpt-4o"]
            }
        ]
        
        for test in fallback_tests:
            print(f"\n   🧪 {test['name']}")
            print(f"      Primary model: {test['model']}")
            print(f"      Fallback options: {', '.join(test.get('fallbacks', []))}")
            
            try:
                request_params = {
                    "model": test["model"],
                    "messages": [{"role": "user", "content": "What is artificial intelligence?"}],
                    "max_tokens": 60,
                    # Add fallback models if specified  
                    "fallbacks": test.get("fallbacks", []),
                    # Governance
                    "team": "reliability-team",
                    "project": "fallback-monitoring",
                    "test_scenario": test["name"]
                }
                
                if "provider" in test:
                    request_params["provider"] = test["provider"]
                
                response = client.chat_completions_create(**request_params)
                
                # In a real scenario, GenOps would capture if fallback was used
                print(f"      ✅ Request successful")
                print(f"         Note: GenOps automatically tracks fallback events")
                
            except Exception as e:
                print(f"      ❌ Error (might indicate fallback needed): {str(e)}")
        
        # Demo 4: Advanced Governance Controls
        print(f"\n🏛️ Feature 4: Advanced Governance Controls")
        print("=" * 42)
        
        governance_scenarios = [
            {
                "name": "Multi-tenant request",
                "model": "anthropic/claude-3-haiku",
                "governance": {
                    "team": "platform-team",
                    "project": "multi-tenant-saas", 
                    "customer_id": "enterprise-customer-001",
                    "tenant_id": "tenant-abc-123",
                    "cost_center": "customer-success",
                    "billing_tier": "enterprise"
                }
            },
            {
                "name": "Compliance-sensitive request",
                "model": "openai/gpt-4o",
                "governance": {
                    "team": "compliance-team",
                    "project": "financial-analysis",
                    "compliance_level": "high",
                    "data_classification": "confidential",
                    "audit_required": "true",
                    "region": "us-east"
                }
            },
            {
                "name": "Development experiment",
                "model": "meta-llama/llama-3.2-3b-instruct",
                "governance": {
                    "team": "research-team",
                    "project": "model-evaluation", 
                    "experiment_id": "exp-2024-001",
                    "researcher": "alice-smith",
                    "hypothesis": "cost-vs-quality",
                    "environment": "development"
                }
            }
        ]
        
        for scenario in governance_scenarios:
            print(f"\n   📋 {scenario['name']}")
            print(f"      Model: {scenario['model']}")
            print(f"      Governance attrs: {len(scenario['governance'])} attributes")
            
            try:
                response = client.chat_completions_create(
                    model=scenario["model"],
                    messages=[{"role": "user", "content": "Hello, how are you?"}],
                    max_tokens=30,
                    **scenario["governance"]  # All governance attributes
                )
                
                print(f"      ✅ Request successful with full governance tracking")
                print(f"         All {len(scenario['governance'])} attributes captured in telemetry")
                
            except Exception as e:
                print(f"      ❌ Error: {str(e)}")
        
        # Summary
        print("\n" + "=" * 55)
        print("📊 Advanced Features Summary")
        print("=" * 55)
        
        print("🎯 Provider Selection:")
        print("   • Explicit provider preferences (provider='anthropic')")
        print("   • Automatic routing with intelligent fallbacks")
        print("   • Cost vs. performance trade-off controls")
        
        print("\n⚡ Routing Strategies:")
        print("   • route='least-cost' - Optimize for price")
        print("   • route='fastest' - Optimize for latency")  
        print("   • route='fallback' - Maximize reliability")
        
        print("\n🔍 Monitoring & Telemetry:")
        print("   • Automatic fallback detection and logging")
        print("   • Provider routing decision capture")
        print("   • Performance metrics (latency, tokens, cost)")
        print("   • Rich governance attribute propagation")
        
        print("\n🏛️ Governance Controls:")
        print("   • Multi-dimensional cost attribution")
        print("   • Compliance and audit trail automation")
        print("   • Custom attribute support (unlimited)")
        print("   • Cross-provider policy enforcement")
        
        print("\n✨ Next Steps:")
        print("   • Set up alerting on fallback events")
        print("   • Implement cost-based routing policies")
        print("   • Try production_patterns.py for deployment guidance")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Install: pip install genops-ai openai")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    advanced_features_demo()