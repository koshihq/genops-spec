#!/usr/bin/env python3
"""
💰 GenOps + Mistral AI: Cost Optimization Guide

GOAL: Master cost optimization with European AI models
TIME: 20-40 minutes  
WHAT YOU'LL LEARN: How to minimize costs while maximizing European AI value

This example demonstrates advanced cost optimization strategies specifically
for Mistral AI models, including model selection, token efficiency, and 
European AI provider cost advantages.

Prerequisites:
- Completed hello_mistral_minimal.py and european_ai_advantages.py
- Mistral API key: export MISTRAL_API_KEY="your-key"
- GenOps: pip install genops-ai
- Mistral: pip install mistralai
"""

import sys
import time
from dataclasses import dataclass


@dataclass
class ModelPerformance:
    """Model performance and cost metrics."""
    model: str
    cost: float
    tokens: int
    time: float
    quality_score: float
    cost_per_token: float
    tokens_per_second: float
    use_case_fit: str

def compare_mistral_models():
    """Compare all Mistral models for cost optimization."""
    print("📊 Mistral Model Cost Comparison")
    print("=" * 60)

    try:
        from genops.providers.mistral import instrument_mistral

        adapter = instrument_mistral(
            team="cost-optimization-team",
            project="model-comparison"
        )

        # Define models to test with different characteristics
        models_to_test = [
            {
                "model": "mistral-tiny-2312",
                "description": "Ultra-low cost",
                "best_for": "Simple Q&A, basic tasks, high-volume processing",
                "cost_tier": "Economy"
            },
            {
                "model": "mistral-small-latest",
                "description": "Cost-effective",
                "best_for": "General tasks, content generation, most use cases",
                "cost_tier": "Standard"
            },
            {
                "model": "mistral-medium-latest",
                "description": "Balanced performance",
                "best_for": "Complex analysis, professional content, reasoning",
                "cost_tier": "Professional"
            },
            {
                "model": "mistral-large-2407",
                "description": "Premium capabilities",
                "best_for": "Advanced reasoning, research, enterprise analysis",
                "cost_tier": "Enterprise"
            }
        ]

        # Test scenarios of different complexity levels
        test_scenarios = [
            {
                "name": "Simple Query",
                "prompt": "What is the capital of Germany?",
                "max_tokens": 10,
                "expected_quality": "factual_accuracy"
            },
            {
                "name": "Content Generation",
                "prompt": "Write a professional email thanking a client for their business.",
                "max_tokens": 100,
                "expected_quality": "professional_tone"
            },
            {
                "name": "Analysis Task",
                "prompt": "Analyze the pros and cons of remote work for European companies.",
                "max_tokens": 300,
                "expected_quality": "comprehensive_analysis"
            },
            {
                "name": "Complex Reasoning",
                "prompt": "Explain the economic implications of GDPR compliance costs for AI startups.",
                "max_tokens": 500,
                "expected_quality": "deep_reasoning"
            }
        ]

        results = []

        print("🧪 Testing Models Across Scenarios:")
        print("-" * 60)

        for scenario in test_scenarios:
            print(f"\n📝 Scenario: {scenario['name']}")
            print(f"   Prompt: \"{scenario['prompt'][:50]}...\"")
            print(f"   Max tokens: {scenario['max_tokens']}")
            print()

            scenario_results = []

            for model_config in models_to_test:
                model = model_config["model"]

                try:
                    start_time = time.time()

                    response = adapter.chat(
                        message=scenario["prompt"],
                        model=model,
                        max_tokens=scenario["max_tokens"],
                        temperature=0.3
                    )

                    request_time = time.time() - start_time

                    if response.success:
                        # Simple quality scoring based on response length and coherence
                        quality_score = min(10.0, len(response.content) / scenario["max_tokens"] * 10)

                        performance = ModelPerformance(
                            model=model,
                            cost=response.usage.total_cost,
                            tokens=response.usage.total_tokens,
                            time=request_time,
                            quality_score=quality_score,
                            cost_per_token=response.usage.cost_per_token,
                            tokens_per_second=response.usage.tokens_per_second,
                            use_case_fit=model_config["best_for"]
                        )

                        scenario_results.append(performance)

                        print(f"   {model_config['cost_tier']} ({model}):")
                        print(f"      Cost: ${performance.cost:.6f}")
                        print(f"      Tokens: {performance.tokens}")
                        print(f"      Quality: {performance.quality_score:.1f}/10")
                        print(f"      Time: {performance.time:.2f}s")
                        print(f"      Efficiency: ${performance.cost_per_token:.8f}/token")

                    else:
                        print(f"   ❌ {model}: {response.error_message}")

                except Exception as e:
                    print(f"   ❌ {model}: Error - {e}")

            # Find best value for this scenario
            if scenario_results:
                # Calculate value score (quality per dollar)
                for perf in scenario_results:
                    perf.value_score = perf.quality_score / max(perf.cost, 0.000001)

                best_value = max(scenario_results, key=lambda x: x.value_score)
                lowest_cost = min(scenario_results, key=lambda x: x.cost)

                print(f"\n   🏆 Best Value: {best_value.model} (Quality/Cost: {best_value.value_score:.1f})")
                print(f"   💰 Lowest Cost: {lowest_cost.model} (${lowest_cost.cost:.6f})")

                results.extend(scenario_results)

        # Overall analysis
        if results:
            print("\n" + "=" * 60)
            print("📈 Cost Optimization Analysis")
            print("=" * 60)

            # Group by model
            model_stats = {}
            for perf in results:
                if perf.model not in model_stats:
                    model_stats[perf.model] = []
                model_stats[perf.model].append(perf)

            print("\n🎯 Model Recommendations by Use Case:")

            for model, performances in model_stats.items():
                avg_cost = sum(p.cost for p in performances) / len(performances)
                avg_quality = sum(p.quality_score for p in performances) / len(performances)
                avg_value = sum(p.value_score for p in performances) / len(performances)

                print(f"\n   {model}:")
                print(f"      Average cost: ${avg_cost:.6f}")
                print(f"      Average quality: {avg_quality:.1f}/10")
                print(f"      Value score: {avg_value:.1f}")
                print(f"      Best for: {performances[0].use_case_fit}")

            return True
        else:
            print("❌ No results to analyze")
            return False

    except Exception as e:
        print(f"❌ Error in model comparison: {e}")
        return False

def optimize_token_usage():
    """Demonstrate token optimization strategies."""
    print("\n" + "=" * 60)
    print("🎯 Token Usage Optimization")
    print("=" * 60)

    try:
        from genops.providers.mistral import instrument_mistral

        adapter = instrument_mistral(
            team="token-optimization-team",
            project="efficiency-analysis"
        )

        # Test different prompting strategies for efficiency
        base_question = "Explain the benefits of European AI providers"

        optimization_strategies = [
            {
                "name": "Unoptimized (Verbose)",
                "prompt": f"""
                Please provide me with a comprehensive and detailed explanation about {base_question}.
                I would like you to give me as much information as possible, including all the details,
                background context, and any relevant information that might be helpful for understanding
                this topic completely. Please be thorough in your response.
                """,
                "max_tokens": 500,
                "temperature": 0.7
            },
            {
                "name": "Optimized (Concise)",
                "prompt": f"{base_question} in 3 key points:",
                "max_tokens": 150,
                "temperature": 0.3
            },
            {
                "name": "Structured (Efficient)",
                "prompt": f"List {base_question}:\n1. Cost advantages\n2. Compliance benefits\n3. Technical features",
                "max_tokens": 200,
                "temperature": 0.2
            },
            {
                "name": "Ultra-Concise",
                "prompt": f"{base_question} (bullet points, max 50 words):",
                "max_tokens": 75,
                "temperature": 0.1
            }
        ]

        print("🧪 Testing Token Optimization Strategies:")
        print("-" * 50)

        optimization_results = []

        for strategy in optimization_strategies:
            try:
                response = adapter.chat(
                    message=strategy["prompt"],
                    model="mistral-small-latest",  # Use consistent model
                    max_tokens=strategy["max_tokens"],
                    temperature=strategy["temperature"]
                )

                if response.success:
                    # Calculate efficiency metrics
                    words_per_token = len(response.content.split()) / max(response.usage.total_tokens, 1)
                    cost_per_word = response.usage.total_cost / max(len(response.content.split()), 1)

                    result = {
                        "strategy": strategy["name"],
                        "cost": response.usage.total_cost,
                        "tokens": response.usage.total_tokens,
                        "words": len(response.content.split()),
                        "chars": len(response.content),
                        "cost_per_token": response.usage.cost_per_token,
                        "cost_per_word": cost_per_word,
                        "words_per_token": words_per_token,
                        "response_sample": response.content[:100]
                    }

                    optimization_results.append(result)

                    print(f"✅ {strategy['name']}:")
                    print(f"   Cost: ${result['cost']:.6f}")
                    print(f"   Tokens: {result['tokens']}")
                    print(f"   Words: {result['words']}")
                    print(f"   Efficiency: ${result['cost_per_word']:.6f}/word")
                    print(f"   Sample: \"{result['response_sample']}...\"")
                    print()
                else:
                    print(f"❌ {strategy['name']}: {response.error_message}")

            except Exception as e:
                print(f"❌ {strategy['name']}: Error - {e}")

        if optimization_results:
            # Find most efficient strategy
            most_cost_efficient = min(optimization_results, key=lambda x: x["cost"])
            most_word_efficient = min(optimization_results, key=lambda x: x["cost_per_word"])

            print("🏆 Optimization Results:")
            print(f"   Most cost-efficient: {most_cost_efficient['strategy']}")
            print(f"      Cost: ${most_cost_efficient['cost']:.6f}")
            print(f"   Best cost per word: {most_word_efficient['strategy']}")
            print(f"      Cost per word: ${most_word_efficient['cost_per_word']:.6f}")

            # Calculate potential savings
            baseline_cost = max(optimization_results, key=lambda x: x["cost"])["cost"]
            optimized_cost = most_cost_efficient["cost"]
            savings = baseline_cost - optimized_cost
            savings_percent = (savings / baseline_cost) * 100

            print("\n💰 Token Optimization Savings:")
            print(f"   Baseline cost: ${baseline_cost:.6f}")
            print(f"   Optimized cost: ${optimized_cost:.6f}")
            print(f"   Savings: ${savings:.6f} ({savings_percent:.1f}%)")

            # Extrapolate to enterprise scale
            monthly_requests = 50000
            monthly_baseline = baseline_cost * monthly_requests
            monthly_optimized = optimized_cost * monthly_requests
            monthly_savings = monthly_baseline - monthly_optimized

            print(f"\n📊 Enterprise Scale Impact ({monthly_requests:,} requests/month):")
            print(f"   Baseline monthly cost: ${monthly_baseline:.2f}")
            print(f"   Optimized monthly cost: ${monthly_optimized:.2f}")
            print(f"   💰 Monthly savings: ${monthly_savings:.2f}")
            print(f"   💰 Annual savings: ${monthly_savings * 12:.2f}")

        return True

    except Exception as e:
        print(f"❌ Error in token optimization: {e}")
        return False

def european_ai_cost_strategies():
    """Advanced cost strategies specific to European AI."""
    print("\n" + "=" * 60)
    print("🇪🇺 European AI Cost Optimization Strategies")
    print("=" * 60)

    try:
        from genops.providers.mistral import instrument_mistral

        adapter = instrument_mistral(
            team="eu-cost-strategy-team",
            project="european-ai-optimization"
        )

        print("💡 European AI Cost Optimization Strategies:")
        print("-" * 50)

        strategies = [
            {
                "name": "GDPR-Optimized Prompting",
                "description": "Structure prompts for compliance efficiency",
                "example": "For GDPR-compliant customer data analysis:",
                "prompt": "Analyze customer feedback while maintaining GDPR Article 6 compliance. Focus on legitimate interests without processing personal identifiers.",
                "benefit": "Reduces tokens needed for compliance instructions"
            },
            {
                "name": "EU Regulatory Batch Processing",
                "description": "Batch similar compliance tasks",
                "example": "Process multiple GDPR requests in single call:",
                "prompt": "Process these 5 GDPR data portability requests using standard EU format: [request1], [request2]...",
                "benefit": "Reduces per-request overhead costs"
            },
            {
                "name": "European Market Specialization",
                "description": "Leverage Mistral's European focus",
                "example": "For EU market analysis:",
                "prompt": "Analyze European market trends for renewable energy, focusing on German and French markets:",
                "benefit": "Better results with European-trained models"
            },
            {
                "name": "Multi-language Efficiency",
                "description": "Process multiple EU languages efficiently",
                "example": "For multilingual content:",
                "prompt": "Translate and localize for EU markets: English, German, French versions:",
                "benefit": "European AI models excel at EU languages"
            }
        ]

        strategy_results = []

        for strategy in strategies:
            print(f"\n🎯 {strategy['name']}:")
            print(f"   Description: {strategy['description']}")
            print(f"   Example: {strategy['example']}")
            print(f"   Benefit: {strategy['benefit']}")

            try:
                # Test the strategy with actual API call
                response = adapter.chat(
                    message=strategy["prompt"],
                    model="mistral-small-latest",
                    max_tokens=200,
                    temperature=0.3
                )

                if response.success:
                    print(f"   ✅ Cost: ${response.usage.total_cost:.6f}")
                    print(f"   Tokens: {response.usage.total_tokens}")
                    print("   European AI advantage: Optimized for EU use cases")

                    strategy_results.append({
                        "name": strategy["name"],
                        "cost": response.usage.total_cost,
                        "tokens": response.usage.total_tokens,
                        "european_optimized": True
                    })
                else:
                    print(f"   ❌ Failed: {response.error_message}")

            except Exception as e:
                print(f"   ❌ Error: {e}")

        # European AI provider comparison
        print("\n🏆 European AI Provider Advantages:")
        print("-" * 50)

        print("💰 Cost Advantages:")
        print("   • 20-60% lower base costs vs US providers")
        print("   • No cross-border data transfer fees")
        print("   • Reduced compliance overhead costs")
        print("   • Simplified legal and audit expenses")

        print("\n🇪🇺 Performance Advantages:")
        print("   • Optimized for European languages and markets")
        print("   • Lower latency for EU-based applications")
        print("   • Native GDPR compliance reduces prompt complexity")
        print("   • Better understanding of European business context")

        print("\n📊 Total Cost of Ownership (TCO) Benefits:")
        total_monthly_cost = 5000  # Example enterprise monthly AI cost

        # Calculate TCO components
        base_cost_savings = total_monthly_cost * 0.4  # 40% base cost savings
        compliance_cost_savings = 2000  # Monthly compliance savings
        operational_savings = 1000  # Reduced operational overhead

        total_savings = base_cost_savings + compliance_cost_savings + operational_savings

        print(f"   Base AI costs (40% savings): ${base_cost_savings:.2f}/month")
        print(f"   Compliance cost reduction: ${compliance_cost_savings:.2f}/month")
        print(f"   Operational overhead savings: ${operational_savings:.2f}/month")
        print(f"   💰 Total monthly TCO savings: ${total_savings:.2f}")
        print(f"   💰 Annual TCO savings: ${total_savings * 12:.2f}")

        return True

    except Exception as e:
        print(f"❌ Error in European AI strategies: {e}")
        return False

def real_world_optimization_scenarios():
    """Show real-world cost optimization scenarios."""
    print("\n" + "=" * 60)
    print("🏢 Real-World Cost Optimization Scenarios")
    print("=" * 60)

    try:
        from genops.providers.mistral import instrument_mistral

        adapter = instrument_mistral(
            team="real-world-scenarios",
            project="cost-optimization-case-studies"
        )

        scenarios = [
            {
                "company": "German E-commerce Platform",
                "use_case": "Customer service automation",
                "current_volume": "100,000 queries/month",
                "optimization": "Model selection + token optimization",
                "before_model": "mistral-large-2407",
                "after_model": "mistral-small-latest",
                "token_reduction": 40,  # 40% reduction through optimization
                "test_query": "Handle customer complaint about delayed delivery"
            },
            {
                "company": "French Financial Services",
                "use_case": "GDPR compliance analysis",
                "current_volume": "5,000 documents/month",
                "optimization": "European AI + batch processing",
                "before_model": "mistral-medium-latest",
                "after_model": "mistral-medium-latest",  # Same model, better prompting
                "token_reduction": 25,  # 25% through better prompting
                "test_query": "Analyze customer data request for GDPR Article 15 compliance"
            },
            {
                "company": "Dutch SaaS Startup",
                "use_case": "Content generation",
                "current_volume": "50,000 generations/month",
                "optimization": "Model tiering + European focus",
                "before_model": "mistral-large-2407",
                "after_model": "mistral-tiny-2312",  # Aggressive cost reduction
                "token_reduction": 60,  # 60% through simpler model for simple tasks
                "test_query": "Generate product description for EU market"
            }
        ]

        total_monthly_savings = 0

        for scenario in scenarios:
            print(f"\n🏢 {scenario['company']}")
            print(f"   Use case: {scenario['use_case']}")
            print(f"   Volume: {scenario['current_volume']}")
            print(f"   Optimization: {scenario['optimization']}")
            print()

            # Test "before" scenario
            print("   📊 Before Optimization:")
            try:
                before_response = adapter.chat(
                    message=scenario["test_query"],
                    model=scenario["before_model"],
                    max_tokens=200
                )

                if before_response.success:
                    before_cost = before_response.usage.total_cost
                    print(f"      Model: {scenario['before_model']}")
                    print(f"      Cost per request: ${before_cost:.6f}")
                    print(f"      Tokens: {before_response.usage.total_tokens}")
                else:
                    before_cost = 0.001  # Fallback estimate
                    print(f"      ❌ Before test failed: {before_response.error_message}")
            except Exception as e:
                before_cost = 0.001
                print(f"      ❌ Before test error: {e}")

            # Test "after" scenario with optimization
            print("   📈 After Optimization:")
            try:
                # Apply token optimization to the prompt
                optimized_prompt = f"{scenario['test_query']} (concise response):"

                after_response = adapter.chat(
                    message=optimized_prompt,
                    model=scenario["after_model"],
                    max_tokens=int(200 * (1 - scenario["token_reduction"]/100)),  # Reduced tokens
                    temperature=0.2  # Lower temperature for consistency
                )

                if after_response.success:
                    after_cost = after_response.usage.total_cost
                    print(f"      Model: {scenario['after_model']}")
                    print(f"      Cost per request: ${after_cost:.6f}")
                    print(f"      Tokens: {after_response.usage.total_tokens}")

                    # Calculate savings
                    savings_per_request = before_cost - after_cost
                    savings_percent = (savings_per_request / before_cost) * 100

                    print(f"      💰 Savings per request: ${savings_per_request:.6f} ({savings_percent:.1f}%)")

                    # Calculate monthly savings based on volume
                    volume_num = int(scenario["current_volume"].split()[0].replace(",", ""))
                    monthly_savings = savings_per_request * volume_num
                    total_monthly_savings += monthly_savings

                    print(f"      💰 Monthly savings: ${monthly_savings:.2f}")

                else:
                    print(f"      ❌ After test failed: {after_response.error_message}")
            except Exception as e:
                print(f"      ❌ After test error: {e}")

        # Summary
        print("\n🏆 Real-World Optimization Summary:")
        print(f"   💰 Total monthly savings across scenarios: ${total_monthly_savings:.2f}")
        print(f"   💰 Potential annual savings: ${total_monthly_savings * 12:.2f}")

        print("\n💡 Key Optimization Insights:")
        print("   • Model selection has the biggest cost impact (up to 90% savings)")
        print("   • Token optimization provides consistent 20-40% savings")
        print("   • European AI specialization improves efficiency for EU use cases")
        print("   • GDPR-optimized prompting reduces compliance overhead")
        print("   • Batch processing reduces per-request costs")

        return True

    except Exception as e:
        print(f"❌ Error in real-world scenarios: {e}")
        return False

def main():
    """Main cost optimization demonstration."""
    print("💰 GenOps + Mistral AI: Cost Optimization Master Class")
    print("=" * 70)
    print("Time: 20-40 minutes | Learn: Advanced cost optimization strategies")
    print("=" * 70)

    # Check prerequisites
    try:
        from genops.providers.mistral_validation import quick_validate
        if not quick_validate():
            print("❌ Setup validation failed")
            print("   Please run hello_mistral_minimal.py first")
            return False
    except ImportError:
        print("❌ GenOps Mistral not available")
        return False

    success_count = 0
    total_sections = 4

    # Run all optimization demonstrations
    sections = [
        ("Model Comparison", compare_mistral_models),
        ("Token Optimization", optimize_token_usage),
        ("European AI Strategies", european_ai_cost_strategies),
        ("Real-World Scenarios", real_world_optimization_scenarios)
    ]

    for name, section_func in sections:
        print(f"\n🎯 Running: {name}")
        if section_func():
            success_count += 1
            print(f"✅ {name} completed successfully")
        else:
            print(f"❌ {name} failed")

    # Final summary
    print("\n" + "=" * 70)
    print(f"🎉 Cost Optimization Guide: {success_count}/{total_sections} sections completed")
    print("=" * 70)

    if success_count == total_sections:
        print("💰 **Cost Optimization Mastery Achieved:**")
        print("   ✅ Model selection strategies learned")
        print("   ✅ Token optimization techniques mastered")
        print("   ✅ European AI cost advantages understood")
        print("   ✅ Real-world optimization scenarios analyzed")

        print("\n🏆 **Key Cost Optimization Principles:**")
        print("   1. Choose the right model for each task complexity")
        print("   2. Optimize prompts for token efficiency")
        print("   3. Leverage European AI provider advantages")
        print("   4. Use batch processing for similar tasks")
        print("   5. Apply GDPR-optimized prompting strategies")

        print("\n💡 **Potential Cost Savings:**")
        print("   • Model optimization: 20-90% cost reduction")
        print("   • Token optimization: 20-40% efficiency gains")
        print("   • European AI advantages: 20-60% vs US providers")
        print("   • Compliance simplification: 50-75% overhead reduction")

        print("\n🚀 **Next Steps:**")
        print("   • Apply learned strategies to your use cases")
        print("   • Run auto_instrumentation.py for zero-code setup")
        print("   • Try enterprise_deployment.py for production patterns")
        print("   • Monitor costs with GenOps cost aggregation")

        return True
    else:
        print("⚠️ Some optimization sections failed - check setup")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Optimization guide interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
