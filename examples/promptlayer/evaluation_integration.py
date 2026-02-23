#!/usr/bin/env python3
"""
PromptLayer Evaluation Integration with GenOps

This example demonstrates comprehensive evaluation workflows with PromptLayer and GenOps,
including A/B testing with governance attribution, quality metrics, and cost analysis.

This is the Level 2 (30-minute) example - A/B testing with governance attribution.

Usage:
    python evaluation_integration.py

Prerequisites:
    pip install genops[promptlayer]  # Includes PromptLayer SDK
    export PROMPTLAYER_API_KEY="pl-your-api-key"
    export OPENAI_API_KEY="your-openai-key"  # For actual LLM calls

    # Required for governance attribution
    export GENOPS_TEAM="your-team"
    export GENOPS_PROJECT="your-project"
"""

import asyncio
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics with governance context."""

    variant_id: str
    prompt_name: str
    prompt_version: str

    # Quality metrics
    accuracy_score: float
    coherence_score: float
    relevance_score: float
    safety_score: float
    overall_quality: float

    # Performance metrics
    avg_latency_ms: float
    success_rate: float
    error_rate: float

    # Cost metrics
    avg_cost_per_request: float
    total_cost: float
    cost_per_quality_point: float

    # Governance metrics
    team: str
    project: str
    environment: str
    customer_attribution: Optional[str] = None

    # Statistical significance
    sample_size: int = 0
    confidence_level: float = 0.95
    p_value: Optional[float] = None

    execution_count: int = 0
    total_tokens: int = 0


@dataclass
class EvaluationSuite:
    """Complete evaluation suite with multiple test scenarios."""

    suite_name: str
    test_scenarios: list[dict[str, Any]] = field(default_factory=list)
    variants: list[str] = field(default_factory=list)
    metrics: dict[str, EvaluationMetrics] = field(default_factory=dict)
    governance_constraints: dict[str, Any] = field(default_factory=dict)


def comprehensive_ab_testing():
    """
    Demonstrates comprehensive A/B testing with governance attribution.

    Shows how GenOps enables sophisticated evaluation workflows with
    cost attribution, team tracking, and governance-aware result analysis.
    """
    print("🧪 Comprehensive A/B Testing with Governance Attribution")
    print("=" * 55)

    try:
        from genops.providers.promptlayer import (
            GovernancePolicy,
            instrument_promptlayer,
        )

        print("✅ GenOps PromptLayer adapter loaded successfully")

        # Initialize with evaluation-specific governance
        adapter = instrument_promptlayer(
            promptlayer_api_key=os.getenv("PROMPTLAYER_API_KEY"),
            team=os.getenv("GENOPS_TEAM", "evaluation-team"),
            project=os.getenv("GENOPS_PROJECT", "ab-testing-suite"),
            environment="evaluation",
            enable_cost_alerts=True,
            daily_budget_limit=15.0,  # $15 budget for evaluation
            governance_policy=GovernancePolicy.ADVISORY,  # Don't block during evaluation
        )
        print("✅ Evaluation governance configured")

    except ImportError as e:
        print(f"❌ Failed to import GenOps PromptLayer adapter: {e}")
        print("💡 Fix: Run 'pip install genops[promptlayer]'")
        return False

    print("\n🚀 Running Comprehensive A/B Testing Suite...")
    print("-" * 50)

    # Define comprehensive test suite
    evaluation_suite = EvaluationSuite(
        suite_name="customer_support_optimization",
        test_scenarios=[
            {
                "scenario": "billing_inquiry",
                "input": {
                    "query": "I was charged twice for my subscription",
                    "urgency": "high",
                },
                "expected_quality": 0.90,
            },
            {
                "scenario": "technical_support",
                "input": {
                    "query": "My app keeps crashing on startup",
                    "urgency": "medium",
                },
                "expected_quality": 0.85,
            },
            {
                "scenario": "feature_request",
                "input": {
                    "query": "Can you add dark mode to the mobile app?",
                    "urgency": "low",
                },
                "expected_quality": 0.75,
            },
            {
                "scenario": "account_management",
                "input": {
                    "query": "How do I change my email address?",
                    "urgency": "medium",
                },
                "expected_quality": 0.88,
            },
        ],
        variants=["control_v1", "empathetic_v2", "concise_v3", "detailed_v4"],
        governance_constraints={
            "max_cost_per_variant": 3.0,  # $3 per variant
            "max_execution_time": 300,  # 5 minutes
            "min_sample_size": 4,  # 4 scenarios per variant
        },
    )

    # Execute A/B testing suite
    print(f"🎯 Executing A/B Test: {evaluation_suite.suite_name}")
    print(f"   Variants: {len(evaluation_suite.variants)}")
    print(f"   Scenarios: {len(evaluation_suite.test_scenarios)}")
    print()

    try:
        with adapter.track_prompt_operation(
            prompt_name=evaluation_suite.suite_name,
            operation_type="ab_test_suite",
            operation_name="comprehensive_evaluation",
            tags={
                "evaluation_type": "ab_testing",
                "variants_count": str(len(evaluation_suite.variants)),
                "scenarios_count": str(len(evaluation_suite.test_scenarios)),
            },
        ) as suite_span:
            # Execute each variant across all scenarios
            for variant in evaluation_suite.variants:
                print(f"📊 Testing variant: {variant}")

                variant_metrics = EvaluationMetrics(
                    variant_id=variant,
                    prompt_name=f"customer_support_{variant}",
                    prompt_version=variant,
                    accuracy_score=0.0,
                    coherence_score=0.0,
                    relevance_score=0.0,
                    safety_score=0.0,
                    overall_quality=0.0,
                    avg_latency_ms=0.0,
                    success_rate=0.0,
                    error_rate=0.0,
                    avg_cost_per_request=0.0,
                    total_cost=0.0,
                    cost_per_quality_point=0.0,
                    team=adapter.team,
                    project=adapter.project,
                    environment="evaluation",
                )

                scenario_results = []
                total_variant_cost = 0.0

                with adapter.track_prompt_operation(
                    prompt_name=f"customer_support_{variant}",
                    prompt_version=variant,
                    operation_type="variant_evaluation",
                    operation_name=f"evaluate_{variant}",
                    tags={"ab_variant": variant, "evaluation_phase": "comprehensive"},
                    max_cost=evaluation_suite.governance_constraints[
                        "max_cost_per_variant"
                    ],
                ) as variant_span:
                    for scenario in evaluation_suite.test_scenarios:
                        scenario_name = scenario["scenario"]

                        with adapter.track_prompt_operation(
                            prompt_name=f"customer_support_{variant}_{scenario_name}",
                            operation_type="scenario_execution",
                            operation_name=f"{variant}_{scenario_name}",
                            tags={
                                "scenario": scenario_name,
                                "variant": variant,
                                "urgency": scenario["input"]["urgency"],
                            },
                        ) as scenario_span:
                            # Execute prompt with timing
                            start_time = time.time()

                            adapter.run_prompt_with_governance(
                                prompt_name=f"customer_support_{variant}",
                                input_variables={
                                    **scenario["input"],
                                    "variant": variant,
                                    "evaluation_mode": True,
                                },
                                tags=[
                                    f"scenario_{scenario_name}",
                                    f"variant_{variant}",
                                ],
                            )

                            execution_time = (time.time() - start_time) * 1000

                            # Simulate realistic metrics based on variant characteristics
                            base_cost = 0.015
                            cost_multipliers = {
                                "control_v1": 1.0,
                                "empathetic_v2": 1.3,  # More detailed responses
                                "concise_v3": 0.7,  # Shorter responses
                                "detailed_v4": 1.8,  # Comprehensive responses
                            }

                            scenario_cost = base_cost * cost_multipliers.get(
                                variant, 1.0
                            )

                            # Quality simulation based on scenario and variant
                            quality_base = scenario["expected_quality"]
                            quality_adjustments = {
                                "control_v1": 0.0,
                                "empathetic_v2": 0.05,  # Better for sensitive issues
                                "concise_v3": -0.02,  # Lower quality but faster
                                "detailed_v4": 0.08,  # Highest quality but expensive
                            }

                            scenario_quality = min(
                                0.98,
                                quality_base + quality_adjustments.get(variant, 0.0),
                            )
                            scenario_quality += random.uniform(
                                -0.03, 0.03
                            )  # Add realistic variance

                            # Update scenario span
                            scenario_span.update_cost(scenario_cost)
                            scenario_span.update_token_usage(
                                input_tokens=45 + len(str(scenario["input"])),
                                output_tokens=int(
                                    120 * cost_multipliers.get(variant, 1.0)
                                ),
                                model="gpt-3.5-turbo",
                            )

                            scenario_result = {
                                "scenario": scenario_name,
                                "cost": scenario_cost,
                                "quality": scenario_quality,
                                "latency_ms": execution_time,
                                "success": True,
                            }
                            scenario_results.append(scenario_result)
                            total_variant_cost += scenario_cost

                            print(
                                f"   • {scenario_name}: Quality {scenario_quality:.3f}, Cost ${scenario_cost:.6f}, Latency {execution_time:.0f}ms"
                            )

                    # Calculate aggregate metrics
                    if scenario_results:
                        variant_metrics.sample_size = len(scenario_results)
                        variant_metrics.execution_count = len(scenario_results)
                        variant_metrics.total_cost = total_variant_cost
                        variant_metrics.avg_cost_per_request = total_variant_cost / len(
                            scenario_results
                        )
                        variant_metrics.overall_quality = sum(
                            r["quality"] for r in scenario_results
                        ) / len(scenario_results)
                        variant_metrics.avg_latency_ms = sum(
                            r["latency_ms"] for r in scenario_results
                        ) / len(scenario_results)
                        variant_metrics.success_rate = sum(
                            1 for r in scenario_results if r["success"]
                        ) / len(scenario_results)
                        variant_metrics.error_rate = 1.0 - variant_metrics.success_rate
                        variant_metrics.cost_per_quality_point = (
                            variant_metrics.avg_cost_per_request
                            / variant_metrics.overall_quality
                        )

                        # Individual quality component simulation
                        variant_metrics.accuracy_score = (
                            variant_metrics.overall_quality
                            + random.uniform(-0.02, 0.02)
                        )
                        variant_metrics.coherence_score = (
                            variant_metrics.overall_quality
                            + random.uniform(-0.03, 0.01)
                        )
                        variant_metrics.relevance_score = (
                            variant_metrics.overall_quality
                            + random.uniform(-0.01, 0.03)
                        )
                        variant_metrics.safety_score = min(
                            0.99, variant_metrics.overall_quality + 0.05
                        )

                        # Update variant span
                        variant_span.update_cost(total_variant_cost)
                        variant_span.add_attributes(
                            {
                                "variant_quality": variant_metrics.overall_quality,
                                "scenarios_executed": len(scenario_results),
                                "cost_efficiency": variant_metrics.cost_per_quality_point,
                                "avg_latency": variant_metrics.avg_latency_ms,
                            }
                        )

                        evaluation_suite.metrics[variant] = variant_metrics

                        print(
                            f"   ✅ Variant {variant}: Overall Quality {variant_metrics.overall_quality:.3f}, CPQ ${variant_metrics.cost_per_quality_point:.6f}"
                        )

                print()

            # Analysis and comparison
            print("📈 A/B Testing Results Analysis")
            print("-" * 40)

            if evaluation_suite.metrics:
                # Quality champion
                quality_leader = max(
                    evaluation_suite.metrics.values(), key=lambda x: x.overall_quality
                )

                # Cost efficiency champion
                cost_leader = min(
                    evaluation_suite.metrics.values(),
                    key=lambda x: x.cost_per_quality_point,
                )

                # Performance champion
                speed_leader = min(
                    evaluation_suite.metrics.values(), key=lambda x: x.avg_latency_ms
                )

                print(f"🏆 Quality Leader: {quality_leader.variant_id}")
                print(f"   Overall Quality: {quality_leader.overall_quality:.3f}")
                print("   Quality Breakdown:")
                print(f"     • Accuracy: {quality_leader.accuracy_score:.3f}")
                print(f"     • Coherence: {quality_leader.coherence_score:.3f}")
                print(f"     • Relevance: {quality_leader.relevance_score:.3f}")
                print(f"     • Safety: {quality_leader.safety_score:.3f}")

                print(f"\n💰 Cost Efficiency Leader: {cost_leader.variant_id}")
                print(
                    f"   Cost per Quality Point: ${cost_leader.cost_per_quality_point:.6f}"
                )
                print(f"   Total Cost: ${cost_leader.total_cost:.6f}")
                print(
                    f"   Average Cost per Request: ${cost_leader.avg_cost_per_request:.6f}"
                )

                print(f"\n⚡ Performance Leader: {speed_leader.variant_id}")
                print(f"   Average Latency: {speed_leader.avg_latency_ms:.0f}ms")
                print(f"   Success Rate: {speed_leader.success_rate:.1%}")

                # Governance-aware recommendation
                print("\n🎯 Governance-Aware Recommendation:")

                # Calculate composite score considering governance priorities
                for variant_id, metrics in evaluation_suite.metrics.items():
                    # Weighted score: 40% quality, 35% cost efficiency, 25% speed
                    quality_normalized = metrics.overall_quality
                    cost_normalized = 1.0 / (
                        metrics.cost_per_quality_point * 1000
                    )  # Invert and normalize
                    speed_normalized = 1.0 / (
                        metrics.avg_latency_ms / 1000
                    )  # Invert and normalize

                    composite_score = (
                        0.40 * quality_normalized
                        + 0.35 * cost_normalized
                        + 0.25 * speed_normalized
                    )

                    print(f"   {variant_id}: Composite Score {composite_score:.3f}")
                    metrics.governance_score = composite_score

                # Select overall winner
                overall_winner = max(
                    evaluation_suite.metrics.values(),
                    key=lambda x: getattr(x, "governance_score", 0),
                )

                print(f"\n🌟 RECOMMENDED VARIANT: {overall_winner.variant_id}")
                print("   Balanced performance across quality, cost, and speed")
                print(f"   Quality: {overall_winner.overall_quality:.3f}")
                print(
                    f"   Cost efficiency: ${overall_winner.cost_per_quality_point:.6f}"
                )
                print(f"   Team attribution: {overall_winner.team}")
                print(
                    f"   Total evaluation cost: ${sum(m.total_cost for m in evaluation_suite.metrics.values()):.6f}"
                )

                # Update suite span with results
                suite_span.add_attributes(
                    {
                        "recommended_variant": overall_winner.variant_id,
                        "quality_leader": quality_leader.variant_id,
                        "cost_leader": cost_leader.variant_id,
                        "speed_leader": speed_leader.variant_id,
                        "total_evaluation_cost": sum(
                            m.total_cost for m in evaluation_suite.metrics.values()
                        ),
                        "variants_tested": len(evaluation_suite.metrics),
                    }
                )

    except Exception as e:
        print(f"❌ A/B testing failed: {e}")
        return False

    return True


def demonstrate_continuous_evaluation():
    """Demonstrate continuous evaluation monitoring with governance."""
    print("\n📊 Continuous Evaluation Monitoring")
    print("-" * 40)

    try:
        from genops.providers.promptlayer import instrument_promptlayer

        adapter = instrument_promptlayer(
            team="quality-assurance",
            project="continuous-monitoring",
            environment="production",
        )

        # Simulate continuous monitoring scenarios
        monitoring_scenarios = [
            {
                "name": "quality_regression_detection",
                "threshold": 0.85,
                "current": 0.82,
            },
            {"name": "cost_drift_monitoring", "threshold": 0.02, "current": 0.025},
            {"name": "latency_performance_check", "threshold": 2000, "current": 1850},
            {"name": "error_rate_tracking", "threshold": 0.02, "current": 0.015},
        ]

        print("🔍 Continuous Quality Monitoring:")

        with adapter.track_prompt_operation(
            prompt_name="continuous_monitoring_suite",
            operation_type="quality_monitoring",
            operation_name="automated_quality_checks",
        ) as monitoring_span:
            alerts = []

            for scenario in monitoring_scenarios:
                name = scenario["name"]
                threshold = scenario["threshold"]
                current = scenario["current"]

                # Determine alert status
                if "regression" in name or "error_rate" in name:
                    # Lower is better
                    alert = current > threshold
                    trend = "DEGRADED" if alert else "HEALTHY"
                elif "cost_drift" in name:
                    # Cost increases are concerning
                    alert = current > threshold
                    trend = "ELEVATED" if alert else "STABLE"
                else:
                    # Higher is better for performance
                    alert = current < threshold
                    trend = "UNDERPERFORMING" if alert else "OPTIMAL"

                status = "🚨" if alert else "✅"

                if alert:
                    alerts.append(
                        {
                            "metric": name,
                            "threshold": threshold,
                            "current": current,
                            "severity": "warning",
                        }
                    )

                print(
                    f"   {status} {name.replace('_', ' ').title()}: {current} (threshold: {threshold}) - {trend}"
                )

            if alerts:
                print(f"\n⚠️ Governance Alerts Generated: {len(alerts)}")
                for alert in alerts:
                    print(
                        f"   • {alert['metric']}: Current {alert['current']} exceeds threshold {alert['threshold']}"
                    )

                monitoring_span.add_attributes(
                    {
                        "alerts_triggered": len(alerts),
                        "monitoring_status": "attention_required",
                        "alert_metrics": [a["metric"] for a in alerts],
                    }
                )
            else:
                print("\n✅ All metrics within acceptable ranges")
                monitoring_span.add_attributes(
                    {"alerts_triggered": 0, "monitoring_status": "healthy"}
                )

        print("\n🎯 Continuous Monitoring Benefits:")
        print("   • Automatic quality regression detection")
        print("   • Cost drift early warning system")
        print("   • Performance monitoring with governance context")
        print("   • Team attribution for quality accountability")

    except Exception as e:
        print(f"❌ Continuous evaluation demo failed: {e}")


def show_evaluation_best_practices():
    """Show evaluation best practices with governance integration."""
    print("\n📋 Evaluation Best Practices with GenOps")
    print("-" * 40)

    print("🎯 Statistical Significance:")
    print("   • Minimum sample sizes for reliable results")
    print("   • Confidence intervals and p-value tracking")
    print("   • Governance-aware stopping criteria")

    print("\n💰 Cost-Aware Evaluation:")
    print("   • Budget allocation across variants")
    print("   • Cost-per-quality optimization")
    print("   • ROI calculation for evaluation efforts")

    print("\n👥 Team Attribution:")
    print("   • Clear ownership of evaluation results")
    print("   • Cost attribution to requesting teams")
    print("   • Governance context preservation")

    print("\n🔄 Lifecycle Integration:")
    print("   • Development → Staging → Production evaluation flow")
    print("   • Automated governance policy enforcement")
    print("   • Continuous monitoring post-deployment")

    print("\n📊 Metrics Framework:")
    print("   • Quality: Accuracy, coherence, relevance, safety")
    print("   • Performance: Latency, throughput, error rates")
    print("   • Cost: Per-request cost, cost efficiency, budget utilization")
    print("   • Governance: Policy compliance, team attribution, audit trails")


async def main():
    """Main execution function."""
    print("🚀 Starting PromptLayer Evaluation Integration Demo")
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Check prerequisites
    if not os.getenv("PROMPTLAYER_API_KEY"):
        print("❌ PROMPTLAYER_API_KEY not found")
        print(
            "💡 Set your PromptLayer API key: export PROMPTLAYER_API_KEY='pl-your-key'"
        )
        print("📖 Get your API key from: https://promptlayer.com/")
        return False

    # Run demonstrations
    success = True

    # Comprehensive A/B testing
    if not comprehensive_ab_testing():
        success = False

    # Continuous evaluation monitoring
    if success:
        demonstrate_continuous_evaluation()

    # Best practices guide
    if success:
        show_evaluation_best_practices()

    if success:
        print("\n" + "🌟" * 60)
        print("🎉 PromptLayer Evaluation Integration Demo Complete!")
        print("\n📊 What You've Accomplished:")
        print("   ✅ Comprehensive A/B testing with governance attribution")
        print("   ✅ Quality, cost, and performance metrics analysis")
        print("   ✅ Governance-aware variant selection and recommendations")
        print("   ✅ Continuous evaluation monitoring with automated alerts")

        print("\n🔍 Your Evaluation Excellence Stack:")
        print("   • PromptLayer: Prompt versioning and evaluation platform")
        print("   • GenOps: Governance-aware evaluation and cost intelligence")
        print("   • OpenTelemetry: Comprehensive metrics export and observability")
        print("   • Multi-Metric: Quality, performance, and cost optimization")

        print("\n📚 Next Steps:")
        print("   • Advanced observability: python advanced_observability.py")
        print("   • Production deployment: python production_patterns.py")
        print("   • Complete test suite: pytest tests/promptlayer/")
        print("   • Run all examples: ./run_all_examples.sh")

        print("\n💡 Evaluation Integration Pattern:")
        print("   ```python")
        print("   # Governance-aware A/B testing")
        print(
            "   with adapter.track_prompt_operation(operation_type='ab_test') as span:"
        )
        print("       for variant in test_variants:")
        print("           result = evaluate_variant_with_governance(variant)")
        print("           span.add_variant_metrics(result)")
        print("       recommendation = span.select_optimal_variant()")
        print("   ```")

        print("🌟" * 60)
    else:
        print("\n❌ Demo encountered errors. Please check the output above.")

    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
