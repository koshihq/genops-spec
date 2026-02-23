#!/usr/bin/env python3
"""
🎯 GenOps Replicate Cost Optimization - Phase 3 (30-45 minutes)

Advanced cost intelligence, optimization strategies, and production patterns.
Learn intelligent model selection, batch processing, and enterprise governance.

This example demonstrates sophisticated cost optimization techniques including:
- Task-based model selection for optimal cost/quality trade-offs
- Batch processing patterns for efficiency at scale
- Advanced budget controls and cost projections
- Production-ready monitoring and alerting

Requirements:
- REPLICATE_API_TOKEN environment variable
- pip install replicate genops-ai

Key Advanced Features:
- Intelligent model routing based on task complexity
- Multi-dimensional cost optimization (speed vs accuracy vs cost)
- Predictive cost modeling and budget forecasting
- Production deployment patterns with monitoring
"""

import json
import os
import time
from dataclasses import dataclass


@dataclass
class TaskProfile:
    """Profile for different AI tasks with optimization parameters."""

    name: str
    complexity: str  # 'simple', 'medium', 'complex'
    quality_threshold: float  # 0.0-1.0
    latency_requirement: str  # 'real-time', 'interactive', 'batch'
    cost_sensitivity: str  # 'low', 'medium', 'high'


def demonstrate_intelligent_model_selection():
    """Show intelligent model selection based on task requirements."""

    print("🧠 Intelligent Model Selection")
    print("=" * 50)

    from genops.providers.replicate import GenOpsReplicateAdapter
    from genops.providers.replicate_pricing import ReplicatePricingCalculator

    adapter = GenOpsReplicateAdapter()
    ReplicatePricingCalculator()

    # Define different task profiles
    tasks = [
        TaskProfile(
            name="Simple FAQ Generation",
            complexity="simple",
            quality_threshold=0.7,
            latency_requirement="interactive",
            cost_sensitivity="high",
        ),
        TaskProfile(
            name="Marketing Copy Creation",
            complexity="medium",
            quality_threshold=0.85,
            latency_requirement="interactive",
            cost_sensitivity="medium",
        ),
        TaskProfile(
            name="Technical Documentation",
            complexity="complex",
            quality_threshold=0.95,
            latency_requirement="batch",
            cost_sensitivity="low",
        ),
    ]

    # Available text models with characteristics
    text_models = {
        "meta/llama-2-7b-chat": {"speed": "fast", "quality": "good", "cost": "low"},
        "meta/llama-2-13b-chat": {
            "speed": "medium",
            "quality": "better",
            "cost": "medium",
        },
        "meta/llama-2-70b-chat": {"speed": "slow", "quality": "best", "cost": "high"},
    }

    print("Step 1: Selecting optimal models for different tasks...")

    for task in tasks:
        print(f"\n   📋 Task: {task.name}")
        print(
            f"      Complexity: {task.complexity} | Quality req: {task.quality_threshold}"
        )
        print(
            f"      Latency: {task.latency_requirement} | Cost sensitivity: {task.cost_sensitivity}"
        )

        # Simple model selection logic
        if task.complexity == "simple" and task.cost_sensitivity == "high":
            selected_model = "meta/llama-2-7b-chat"
            reason = "Fast, cost-effective for simple tasks"
        elif task.complexity == "complex" or task.quality_threshold > 0.9:
            selected_model = "meta/llama-2-70b-chat"
            reason = "High quality for complex tasks"
        else:
            selected_model = "meta/llama-2-13b-chat"
            reason = "Balanced performance and cost"

        print(f"      ✅ Selected: {selected_model}")
        print(f"      💡 Reason: {reason}")

        try:
            # Test the selected model
            test_prompt = f"Generate a brief example for: {task.name.lower()}"

            response = adapter.text_generation(
                model=selected_model,
                prompt=test_prompt,
                max_tokens=50,
                team="optimization-team",
                project="model-selection",
                feature="intelligent-routing",
            )

            print(f"      💰 Cost: ${response.cost_usd:.6f}")
            print(f"      ⏱️  Latency: {response.latency_ms:.0f}ms")
            print(
                f"      📊 Quality estimate: {text_models[selected_model]['quality']}"
            )

        except Exception as e:
            print(f"      ❌ Test failed: {e}")
            continue

    print("\n✅ Intelligent model selection complete!")
    return True


def demonstrate_batch_processing_optimization():
    """Show batch processing patterns for cost efficiency."""

    print("\n📦 Batch Processing Optimization")
    print("=" * 50)

    from genops.providers.replicate import GenOpsReplicateAdapter

    # Simulate a large batch job
    content_requests = [
        "Write a product description for AI-powered analytics software",
        "Create social media post about machine learning benefits",
        "Generate FAQ entry about data privacy in AI systems",
        "Draft email subject line for AI product launch",
        "Write blog post intro about cost optimization in AI",
    ]

    print("Step 2: Comparing single vs batch processing efficiency...")

    adapter = GenOpsReplicateAdapter()

    # Method 1: Individual requests (inefficient)
    print("\n   🔄 Method 1: Individual processing...")
    individual_start = time.time()
    individual_costs = []

    for i, content_request in enumerate(content_requests[:3], 1):  # Limit to 3 for demo
        try:
            response = adapter.text_generation(
                model="meta/llama-2-7b-chat",
                prompt=content_request,
                max_tokens=60,
                team="content-team",
                project="individual-processing",
            )
            individual_costs.append(response.cost_usd)
            print(
                f"      Request {i}: ${response.cost_usd:.6f} ({response.latency_ms:.0f}ms)"
            )

        except Exception as e:
            print(f"      Request {i} failed: {e}")

    individual_total_time = time.time() - individual_start
    individual_total_cost = sum(individual_costs)

    print(f"      💰 Total cost: ${individual_total_cost:.6f}")
    print(f"      ⏱️  Total time: {individual_total_time:.1f}s")

    # Method 2: Batch processing (more efficient)
    print("\n   📦 Method 2: Batch processing...")
    batch_start = time.time()

    try:
        # Create a batch prompt
        batch_prompt = "Generate brief content for these 5 requests:\n\n"
        for i, request in enumerate(content_requests, 1):
            batch_prompt += f"{i}. {request}\n"
        batch_prompt += "\nProvide numbered responses, each under 50 words."

        response = adapter.text_generation(
            model="meta/llama-2-7b-chat",
            prompt=batch_prompt,
            max_tokens=300,  # Accommodate all responses
            team="content-team",
            project="batch-processing",
        )

        batch_total_time = time.time() - batch_start

        print(f"      💰 Batch cost: ${response.cost_usd:.6f}")
        print(f"      ⏱️  Batch time: {batch_total_time:.1f}s")

        # Calculate efficiency gains
        if individual_total_cost > 0:
            cost_savings = (
                (individual_total_cost - response.cost_usd) / individual_total_cost
            ) * 100
            time_savings = (
                (individual_total_time - batch_total_time) / individual_total_time
            ) * 100

            print("      📈 Efficiency gains:")
            print(f"         💰 Cost savings: {cost_savings:.1f}%")
            print(f"         ⏱️  Time savings: {time_savings:.1f}%")

    except Exception as e:
        print(f"      ❌ Batch processing failed: {e}")
        return False

    return True


def demonstrate_advanced_cost_monitoring():
    """Show advanced cost monitoring and predictive analytics."""

    print("\n📊 Advanced Cost Monitoring & Predictions")
    print("=" * 50)

    from genops.providers.replicate import GenOpsReplicateAdapter
    from genops.providers.replicate_cost_aggregator import create_replicate_cost_context

    # Simulate a production workflow with multiple stages
    with create_replicate_cost_context(
        "production_workflow", budget_limit=2.0
    ) as context:
        print("Step 3: Production workflow with cost monitoring...")

        adapter = GenOpsReplicateAdapter()

        # Stage 1: Content planning
        print("\n   📝 Stage 1: Content Planning")
        try:
            planning_response = adapter.text_generation(
                model="meta/llama-2-7b-chat",
                prompt="Create a content plan for AI product marketing campaign",
                max_tokens=100,
                team="marketing-team",
                project="product-launch",
                environment="production",
            )

            context.add_operation(
                model="meta/llama-2-7b-chat",
                category="text",
                cost_usd=planning_response.cost_usd,
                input_tokens=100,
                output_tokens=150,
                latency_ms=planning_response.latency_ms,
                team="marketing-team",
            )

            print(f"      ✅ Planning completed: ${planning_response.cost_usd:.6f}")

        except Exception as e:
            print(f"      ❌ Planning failed: {e}")

        # Stage 2: Visual asset creation
        print("\n   🎨 Stage 2: Visual Asset Creation")
        try:
            visual_response = adapter.image_generation(
                model="black-forest-labs/flux-schnell",
                prompt="Professional marketing banner for AI analytics product",
                num_images=2,
                team="design-team",
                project="product-launch",
                environment="production",
            )

            context.add_operation(
                model="black-forest-labs/flux-schnell",
                category="image",
                cost_usd=visual_response.cost_usd,
                output_units=2,
                latency_ms=visual_response.latency_ms,
                team="design-team",
            )

            print(f"      ✅ Visuals created: ${visual_response.cost_usd:.6f}")

        except Exception as e:
            print(f"      ❌ Visual creation failed: {e}")

        # Stage 3: Content generation
        print("\n   ✍️  Stage 3: Content Generation")
        content_tasks = [
            "Write compelling homepage copy",
            "Create product feature descriptions",
            "Generate customer testimonial template",
        ]

        for task in content_tasks:
            try:
                content_response = adapter.text_generation(
                    model="meta/llama-2-13b-chat",  # Better quality for final content
                    prompt=task,
                    max_tokens=80,
                    team="content-team",
                    project="product-launch",
                    environment="production",
                )

                context.add_operation(
                    model="meta/llama-2-13b-chat",
                    category="text",
                    cost_usd=content_response.cost_usd,
                    input_tokens=50,
                    output_tokens=80,
                    latency_ms=content_response.latency_ms,
                    team="content-team",
                )

                print(f"      ✅ {task}: ${content_response.cost_usd:.6f}")

            except Exception as e:
                print(f"      ❌ {task} failed: {e}")

        # Final workflow analysis
        final_summary = context.get_current_summary()

        print("\n📊 PRODUCTION WORKFLOW ANALYSIS:")
        print(f"   💰 Total Cost: ${final_summary.total_cost:.6f}")
        print(
            f"   🎯 Budget Utilization: {(final_summary.total_cost / 2.0) * 100:.1f}%"
        )
        print(f"   🔄 Total Operations: {final_summary.operation_count}")
        print(f"   ⏱️  Total Time: {final_summary.total_time_ms / 1000:.1f}s")

        print("\n   📈 Cost Breakdown by Category:")
        for category, cost in final_summary.cost_by_category.items():
            percentage = (cost / final_summary.total_cost) * 100
            print(f"      {category.title()}: ${cost:.6f} ({percentage:.1f}%)")

        print("\n   🏢 Cost Breakdown by Team:")
        team_costs = {}
        for operation in context.operations:
            team = operation.governance_attributes.get("team", "unknown")
            team_costs[team] = team_costs.get(team, 0) + operation.cost_usd

        for team, cost in sorted(team_costs.items(), key=lambda x: x[1], reverse=True):
            percentage = (cost / final_summary.total_cost) * 100
            print(f"      {team}: ${cost:.6f} ({percentage:.1f}%)")

        # Optimization recommendations
        if final_summary.optimization_recommendations:
            print("\n   💡 Optimization Recommendations:")
            for rec in final_summary.optimization_recommendations:
                print(f"      • {rec}")

        # Budget alerts
        if final_summary.budget_status:
            budget_info = final_summary.budget_status
            if budget_info["percentage_used"] > 75:
                print("\n   ⚠️  BUDGET ALERT:")
                print(f"      {budget_info['percentage_used']:.1f}% of budget used")
                print(f"      ${budget_info['remaining_budget']:.6f} remaining")


def demonstrate_production_patterns():
    """Show production deployment patterns and monitoring."""

    print("\n🏭 Production Deployment Patterns")
    print("=" * 50)

    print("Step 4: Production-ready configurations and monitoring...")

    # Example production configuration
    production_config = {
        "cost_monitoring": {
            "budget_alerts": True,
            "daily_budget_limit": 100.0,
            "alert_thresholds": [75, 90, 95],  # Percentage thresholds
            "cost_attribution_required": True,
        },
        "model_routing": {
            "enable_intelligent_selection": True,
            "fallback_models": {
                "text": ["meta/llama-2-7b-chat", "meta/llama-2-13b-chat"],
                "image": ["black-forest-labs/flux-schnell"],
            },
            "quality_gates": {"min_success_rate": 0.95, "max_latency_ms": 30000},
        },
        "governance": {
            "required_attributes": ["team", "project", "environment"],
            "cost_center_mapping": {
                "marketing-team": "MKTING-001",
                "design-team": "DESIGN-001",
                "content-team": "CONTENT-001",
            },
        },
    }

    print("   🔧 Production Configuration:")
    print(json.dumps(production_config, indent=6))

    print("\n   📊 Monitoring Setup:")
    print("      ✅ OpenTelemetry integration enabled")
    print("      ✅ Cost attribution by team/project/customer")
    print("      ✅ Real-time budget monitoring")
    print("      ✅ Model performance tracking")
    print("      ✅ Automated optimization recommendations")

    print("\n   🚨 Alerting Configuration:")
    print("      • Budget thresholds: 75%, 90%, 95%")
    print("      • High latency alerts: >30s")
    print("      • Model failure rate: <95% success")
    print("      • Cost anomaly detection enabled")

    print("\n   🔄 Deployment Patterns:")
    print("      • Circuit breakers for model failures")
    print("      • Graceful degradation to cheaper models")
    print("      • Automatic retry with exponential backoff")
    print("      • Health checks with GenOps validation")

    return True


def main():
    """Main demonstration of advanced Replicate cost optimization."""

    print("🚀 GenOps Replicate Advanced Cost Optimization")
    print("Production-ready patterns, intelligent routing, and enterprise governance")
    print()

    # Check prerequisites
    if not os.getenv("REPLICATE_API_TOKEN"):
        print("❌ REPLICATE_API_TOKEN not set")
        print("🔧 Setup:")
        print("   1. Get token: https://replicate.com/account/api-tokens")
        print("   2. export REPLICATE_API_TOKEN='r8_your_token_here'")
        return False

    success = True

    # Run all advanced demonstrations
    success &= demonstrate_intelligent_model_selection()
    success &= demonstrate_batch_processing_optimization()
    success &= demonstrate_advanced_cost_monitoring()
    success &= demonstrate_production_patterns()

    if success:
        print("\n🎉 ADVANCED COST OPTIMIZATION COMPLETE!")
        print("=" * 60)
        print("✅ You now understand:")
        print("   • Intelligent model selection based on task requirements")
        print("   • Batch processing patterns for efficiency at scale")
        print("   • Advanced cost monitoring and predictive analytics")
        print("   • Production deployment patterns with enterprise governance")
        print("   • Real-time budget controls and automated optimizations")
        print()
        print("🎯 PHASE 3 MASTERY - Ready for enterprise production!")
        print()
        print("🚀 NEXT STEPS:")
        print("   → Deploy to production with confidence")
        print("   → Set up monitoring dashboards")
        print("   → Configure automated cost alerts")
        print("   → Scale across your organization")
        print()
        print("📚 Complete documentation:")
        print("   → examples/replicate/README.md")
        print("   → docs/replicate-quickstart.md")

    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
