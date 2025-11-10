#!/usr/bin/env python3
"""
Langfuse LLM Evaluation Integration with GenOps Governance Example

This example demonstrates comprehensive LLM evaluation workflows with Langfuse
observability enhanced by GenOps governance. Perfect for teams that need to
evaluate LLM performance while maintaining cost attribution and compliance.

Usage:
    python evaluation_integration.py

Prerequisites:
    pip install genops[langfuse]
    export LANGFUSE_PUBLIC_KEY="pk-lf-your-public-key"
    export LANGFUSE_SECRET_KEY="sk-lf-your-secret-key"
    export OPENAI_API_KEY="your-openai-api-key"  # Or another provider
"""

import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class EvaluationMetrics:
    """Standard evaluation metrics with governance context."""
    accuracy: float
    relevance: float
    coherence: float
    cost_efficiency: float
    latency_score: float
    overall_score: float
    governance_compliance: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "relevance": self.relevance,
            "coherence": self.coherence,
            "cost_efficiency": self.cost_efficiency,
            "latency_score": self.latency_score,
            "overall_score": self.overall_score,
            "governance_compliance": self.governance_compliance
        }


class GovernanceEvaluator:
    """Evaluation framework with integrated governance intelligence."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.evaluation_history = []

    def evaluate_response_quality(
        self,
        prompt: str,
        response: str,
        model: str,
        cost: float,
        latency_ms: float,
        **governance_attrs
    ) -> EvaluationMetrics:
        """Evaluate response quality with governance context."""

        # Quality metrics (simplified for demo)
        accuracy = min(1.0, len(response.split()) / 20.0)  # Basic word count proxy
        relevance = 0.85 if any(word in response.lower() for word in prompt.lower().split()) else 0.6
        coherence = 0.80 if len(response) > 50 else 0.6

        # Governance-aware cost efficiency (lower cost = higher score)
        cost_efficiency = max(0.0, 1.0 - (cost / 0.10))  # Normalized to 10 cent baseline

        # Latency scoring (faster = better)
        latency_score = max(0.0, 1.0 - (latency_ms / 5000.0))  # Normalized to 5 second baseline

        # Overall score with governance weighting
        overall_score = (
            accuracy * 0.25 +
            relevance * 0.25 +
            coherence * 0.20 +
            cost_efficiency * 0.15 +
            latency_score * 0.15
        )

        # Governance compliance check
        governance_compliance = 1.0  # Perfect compliance in demo
        if governance_attrs.get('customer_id') and governance_attrs.get('team'):
            governance_compliance = 1.0
        else:
            governance_compliance = 0.8  # Reduced for incomplete governance

        return EvaluationMetrics(
            accuracy=accuracy,
            relevance=relevance,
            coherence=coherence,
            cost_efficiency=cost_efficiency,
            latency_score=latency_score,
            overall_score=overall_score * governance_compliance,
            governance_compliance=governance_compliance
        )

    def batch_evaluate(
        self,
        evaluation_dataset: List[Dict[str, Any]],
        **governance_attrs
    ) -> Dict[str, Any]:
        """Run batch evaluation with governance tracking."""

        batch_results = []
        total_cost = 0.0
        total_evaluations = len(evaluation_dataset)

        print(f"🔄 Running batch evaluation on {total_evaluations} examples...")

        for i, example in enumerate(evaluation_dataset, 1):
            print(f"   📊 Evaluating example {i}/{total_evaluations}: {example['prompt'][:40]}...")

            with self.adapter.trace_with_governance(
                name=f"evaluation_example_{i}",
                evaluation_batch=True,
                **governance_attrs
            ) as trace:

                # Generate response with cost tracking
                response = self.adapter.generation_with_cost_tracking(
                    prompt=example['prompt'],
                    model=example.get('model', 'gpt-3.5-turbo'),
                    max_cost=example.get('max_cost', 0.05),
                    evaluation_mode=True,
                    **governance_attrs
                )

                # Evaluate the response
                metrics = self.evaluate_response_quality(
                    prompt=example['prompt'],
                    response=response.content,
                    model=response.usage.model,
                    cost=response.usage.cost,
                    latency_ms=response.usage.latency_ms,
                    **governance_attrs
                )

                # Record evaluation in Langfuse
                evaluation_result = self.adapter.evaluate_with_governance(
                    trace_id=response.trace_id,
                    evaluation_name="response_quality",
                    evaluator_function=lambda: {
                        "score": metrics.overall_score,
                        "comment": f"Quality: {metrics.overall_score:.3f}, Cost-efficiency: {metrics.cost_efficiency:.3f}",
                        "metrics": metrics.to_dict()
                    },
                    **governance_attrs
                )

                batch_results.append({
                    "example_id": i,
                    "prompt": example['prompt'],
                    "response": response.content,
                    "metrics": metrics,
                    "evaluation_id": evaluation_result['evaluation_id'],
                    "cost": response.usage.cost,
                    "governance": governance_attrs
                })

                total_cost += response.usage.cost

        # Calculate batch summary
        avg_metrics = self._calculate_average_metrics(batch_results)

        print("✅ Batch evaluation complete!")
        print(f"   📊 Examples evaluated: {total_evaluations}")
        print(f"   💰 Total cost: ${total_cost:.6f}")
        print(f"   📈 Average quality score: {avg_metrics['overall_score']:.3f}")
        print(f"   💡 Average cost efficiency: {avg_metrics['cost_efficiency']:.3f}")

        return {
            "total_examples": total_evaluations,
            "total_cost": total_cost,
            "average_cost_per_example": total_cost / total_evaluations,
            "average_metrics": avg_metrics,
            "results": batch_results,
            "governance": governance_attrs
        }

    def _calculate_average_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate average metrics across batch results."""
        if not results:
            return {}

        metric_sums = {}
        for result in results:
            metrics_dict = result['metrics'].to_dict()
            for metric, value in metrics_dict.items():
                metric_sums[metric] = metric_sums.get(metric, 0.0) + value

        return {
            metric: total / len(results)
            for metric, total in metric_sums.items()
        }


def demonstrate_basic_evaluation():
    """Demonstrate basic LLM evaluation with governance tracking."""
    print("📊 Basic LLM Evaluation with Governance Tracking")
    print("=" * 50)

    try:
        from genops.providers.langfuse import instrument_langfuse

        # Initialize adapter with evaluation budget
        adapter = instrument_langfuse(
            team="evaluation-team",
            project="llm-quality-assessment",
            environment="development",
            budget_limits={"daily": 2.0}  # $2 daily evaluation budget
        )

        print("✅ GenOps Langfuse adapter initialized for evaluation")
        print(f"   🏷️  Team: {adapter.team}")
        print(f"   📊 Project: {adapter.project}")
        print(f"   💰 Daily evaluation budget: ${adapter.budget_limits['daily']:.2f}")

        # Initialize evaluator
        evaluator = GovernanceEvaluator(adapter)

        # Example evaluation scenarios
        evaluation_scenarios = [
            {
                "name": "Technical Documentation",
                "prompt": "Explain how machine learning models are trained and evaluated",
                "model": "gpt-3.5-turbo",
                "customer_id": "tech-docs-customer",
                "cost_center": "documentation"
            },
            {
                "name": "Customer Support",
                "prompt": "How do I reset my password and update my account settings?",
                "model": "gpt-3.5-turbo",
                "customer_id": "support-customer",
                "cost_center": "customer-service"
            },
            {
                "name": "Creative Writing",
                "prompt": "Write a short story about artificial intelligence helping solve climate change",
                "model": "gpt-3.5-turbo",
                "customer_id": "creative-customer",
                "cost_center": "content-creation"
            }
        ]

        scenario_results = []

        for scenario in evaluation_scenarios:
            print(f"\n🧪 Evaluating scenario: {scenario['name']}")
            print("-" * 35)

            with adapter.trace_with_governance(
                name=f"evaluation_{scenario['name'].lower().replace(' ', '_')}",
                customer_id=scenario['customer_id'],
                cost_center=scenario['cost_center'],
                evaluation_type="single_response"
            ) as trace:

                # Generate response
                response = adapter.generation_with_cost_tracking(
                    prompt=scenario['prompt'],
                    model=scenario['model'],
                    max_cost=0.10,
                    operation=f"{scenario['name']}_evaluation",
                    customer_id=scenario['customer_id'],
                    cost_center=scenario['cost_center']
                )

                # Evaluate response quality
                metrics = evaluator.evaluate_response_quality(
                    prompt=scenario['prompt'],
                    response=response.content,
                    model=response.usage.model,
                    cost=response.usage.cost,
                    latency_ms=response.usage.latency_ms,
                    customer_id=scenario['customer_id'],
                    cost_center=scenario['cost_center']
                )

                # Record evaluation in Langfuse
                evaluation_result = adapter.evaluate_with_governance(
                    trace_id=response.trace_id,
                    evaluation_name=f"{scenario['name']}_quality",
                    evaluator_function=lambda m=metrics: {
                        "score": m.overall_score,
                        "comment": f"Quality: {m.overall_score:.3f} | Cost-efficiency: {m.cost_efficiency:.3f}",
                        "breakdown": m.to_dict()
                    },
                    customer_id=scenario['customer_id'],
                    cost_center=scenario['cost_center']
                )

                print(f"   📝 Response: {response.content[:80]}...")
                print(f"   📊 Overall Quality Score: {metrics.overall_score:.3f}")
                print(f"   💰 Cost: ${response.usage.cost:.6f}")
                print(f"   ⏱️  Latency: {response.usage.latency_ms:.0f}ms")
                print(f"   💡 Cost Efficiency: {metrics.cost_efficiency:.3f}")
                print(f"   🛡️  Governance Compliance: {metrics.governance_compliance:.3f}")
                print(f"   📈 Evaluation ID: {evaluation_result['evaluation_id']}")

                scenario_results.append({
                    "scenario": scenario['name'],
                    "metrics": metrics,
                    "cost": response.usage.cost,
                    "evaluation_id": evaluation_result['evaluation_id']
                })

        return scenario_results

    except Exception as e:
        print(f"❌ Basic evaluation failed: {e}")
        return None


def demonstrate_batch_evaluation():
    """Demonstrate batch evaluation with cost optimization."""
    print("\n📊 Batch Evaluation with Cost Optimization")
    print("=" * 42)

    try:
        from genops.providers.langfuse import instrument_langfuse

        # Initialize adapter for batch evaluation
        adapter = instrument_langfuse(
            team="batch-evaluation-team",
            project="llm-performance-testing",
            environment="testing",
            budget_limits={"daily": 5.0}  # Higher budget for batch evaluation
        )

        evaluator = GovernanceEvaluator(adapter)

        # Create evaluation dataset
        evaluation_dataset = [
            {
                "prompt": "Summarize the key benefits of renewable energy sources",
                "expected_topics": ["solar", "wind", "environmental", "cost"],
                "model": "gpt-3.5-turbo",
                "max_cost": 0.03
            },
            {
                "prompt": "Explain the basics of machine learning in simple terms",
                "expected_topics": ["data", "algorithms", "patterns", "predictions"],
                "model": "gpt-3.5-turbo",
                "max_cost": 0.04
            },
            {
                "prompt": "Describe best practices for remote work productivity",
                "expected_topics": ["schedule", "communication", "workspace", "tools"],
                "model": "gpt-3.5-turbo",
                "max_cost": 0.03
            },
            {
                "prompt": "What are the main components of a healthy diet?",
                "expected_topics": ["nutrition", "balance", "variety", "portions"],
                "model": "gpt-3.5-turbo",
                "max_cost": 0.03
            },
            {
                "prompt": "How does cloud computing benefit modern businesses?",
                "expected_topics": ["scalability", "cost", "accessibility", "security"],
                "model": "gpt-3.5-turbo",
                "max_cost": 0.04
            }
        ]

        print(f"📋 Dataset prepared: {len(evaluation_dataset)} examples")
        print("🎯 Running comprehensive batch evaluation...")

        # Run batch evaluation with governance
        batch_results = evaluator.batch_evaluate(
            evaluation_dataset,
            customer_id="batch-eval-customer",
            cost_center="quality-assurance",
            evaluation_type="batch_performance",
            feature="content-generation"
        )

        # Analyze results
        print("\n📈 Batch Evaluation Results Summary:")
        print(f"   📊 Total examples: {batch_results['total_examples']}")
        print(f"   💰 Total cost: ${batch_results['total_cost']:.6f}")
        print(f"   💡 Average cost per example: ${batch_results['average_cost_per_example']:.6f}")

        avg_metrics = batch_results['average_metrics']
        print("\n📊 Average Quality Metrics:")
        print(f"   🎯 Overall Score: {avg_metrics['overall_score']:.3f}")
        print(f"   ✅ Accuracy: {avg_metrics['accuracy']:.3f}")
        print(f"   🔍 Relevance: {avg_metrics['relevance']:.3f}")
        print(f"   📝 Coherence: {avg_metrics['coherence']:.3f}")
        print(f"   💰 Cost Efficiency: {avg_metrics['cost_efficiency']:.3f}")
        print(f"   ⚡ Latency Score: {avg_metrics['latency_score']:.3f}")
        print(f"   🛡️  Governance Compliance: {avg_metrics['governance_compliance']:.3f}")

        # Cost optimization insights
        print("\n💡 Cost Optimization Insights:")
        if avg_metrics['cost_efficiency'] < 0.7:
            print("   ⚠️  Consider using more cost-effective models for simpler tasks")
        if avg_metrics['latency_score'] < 0.8:
            print("   ⚠️  High latency detected - consider caching or optimization")
        if avg_metrics['overall_score'] > 0.85:
            print("   ✅ Excellent performance - current setup is well optimized")

        return batch_results

    except Exception as e:
        print(f"❌ Batch evaluation failed: {e}")
        return None


def demonstrate_model_comparison():
    """Demonstrate model comparison with governance-aware evaluation."""
    print("\n📊 Model Comparison with Governance Intelligence")
    print("=" * 48)

    try:
        from genops.providers.langfuse import instrument_langfuse

        # Initialize adapter for model comparison
        adapter = instrument_langfuse(
            team="model-comparison-team",
            project="llm-benchmarking",
            environment="testing",
            budget_limits={"daily": 3.0}
        )

        evaluator = GovernanceEvaluator(adapter)

        # Models to compare
        models_to_compare = ["gpt-3.5-turbo", "claude-3-haiku"]

        # Test prompt for comparison
        test_prompt = "Write a professional email explaining a project delay and proposing solutions"

        comparison_results = {}

        for model in models_to_compare:
            print(f"\n🔬 Testing model: {model}")
            print("-" * 25)

            with adapter.trace_with_governance(
                name=f"model_comparison_{model.replace('-', '_')}",
                customer_id="comparison-customer",
                cost_center="research",
                model_comparison=True,
                test_model=model
            ) as trace:

                # Generate response
                response = adapter.generation_with_cost_tracking(
                    prompt=test_prompt,
                    model=model,
                    max_cost=0.15,
                    operation="model_comparison",
                    customer_id="comparison-customer",
                    cost_center="research"
                )

                # Evaluate response
                metrics = evaluator.evaluate_response_quality(
                    prompt=test_prompt,
                    response=response.content,
                    model=model,
                    cost=response.usage.cost,
                    latency_ms=response.usage.latency_ms,
                    customer_id="comparison-customer",
                    cost_center="research"
                )

                # Record evaluation
                evaluation_result = adapter.evaluate_with_governance(
                    trace_id=response.trace_id,
                    evaluation_name=f"{model}_comparison",
                    evaluator_function=lambda m=metrics: {
                        "score": m.overall_score,
                        "comment": f"Model: {model} | Score: {m.overall_score:.3f}",
                        "model_metrics": m.to_dict()
                    },
                    customer_id="comparison-customer",
                    cost_center="research",
                    model_comparison=True
                )

                comparison_results[model] = {
                    "response": response,
                    "metrics": metrics,
                    "evaluation_id": evaluation_result['evaluation_id']
                }

                print(f"   📝 Response length: {len(response.content)} chars")
                print(f"   📊 Quality Score: {metrics.overall_score:.3f}")
                print(f"   💰 Cost: ${response.usage.cost:.6f}")
                print(f"   ⏱️  Latency: {response.usage.latency_ms:.0f}ms")
                print(f"   💡 Cost Efficiency: {metrics.cost_efficiency:.3f}")

        # Compare results
        print("\n🏆 Model Comparison Results:")
        print("=" * 28)

        best_quality = max(comparison_results.items(), key=lambda x: x[1]['metrics'].overall_score)
        best_cost = min(comparison_results.items(), key=lambda x: x[1]['response'].usage.cost)
        best_speed = min(comparison_results.items(), key=lambda x: x[1]['response'].usage.latency_ms)

        print(f"🥇 Best Quality: {best_quality[0]} (Score: {best_quality[1]['metrics'].overall_score:.3f})")
        print(f"💰 Most Cost Effective: {best_cost[0]} (${best_cost[1]['response'].usage.cost:.6f})")
        print(f"⚡ Fastest: {best_speed[0]} ({best_speed[1]['response'].usage.latency_ms:.0f}ms)")

        # Detailed comparison table
        print("\n📊 Detailed Comparison:")
        print("Model               | Quality | Cost     | Latency | Cost Eff.")
        print("-" * 60)

        for model, result in comparison_results.items():
            metrics = result['metrics']
            cost = result['response'].usage.cost
            latency = result['response'].usage.latency_ms

            print(f"{model:<18} | {metrics.overall_score:>7.3f} | ${cost:>7.6f} | {latency:>6.0f}ms | {metrics.cost_efficiency:>7.3f}")

        return comparison_results

    except Exception as e:
        print(f"❌ Model comparison failed: {e}")
        return None


def demonstrate_evaluation_automation():
    """Demonstrate automated evaluation workflows with governance."""
    print("\n🤖 Automated Evaluation Workflows with Governance")
    print("=" * 48)

    try:
        from genops.providers.langfuse import instrument_langfuse

        # Initialize adapter for automation
        adapter = instrument_langfuse(
            team="automation-team",
            project="continuous-evaluation",
            environment="production",
            budget_limits={"daily": 10.0, "monthly": 200.0}
        )

        print("🔄 Setting up automated evaluation pipeline...")
        print("   🎯 Continuous quality monitoring")
        print("   💰 Budget-aware evaluation scheduling")
        print("   📊 Governance-integrated metrics collection")

        # Simulate automated evaluation scenarios
        automation_scenarios = [
            {
                "scenario": "Hourly Content Quality Check",
                "frequency": "hourly",
                "budget_per_run": 0.50,
                "customer_id": "automation-customer",
                "priority": "high"
            },
            {
                "scenario": "Daily Model Performance Baseline",
                "frequency": "daily",
                "budget_per_run": 2.00,
                "customer_id": "baseline-customer",
                "priority": "medium"
            },
            {
                "scenario": "Weekly Comprehensive Evaluation",
                "frequency": "weekly",
                "budget_per_run": 5.00,
                "customer_id": "weekly-customer",
                "priority": "high"
            }
        ]

        automation_results = []

        for scenario in automation_scenarios:
            print(f"\n🤖 Running: {scenario['scenario']}")
            print(f"   ⏰ Frequency: {scenario['frequency']}")
            print(f"   💰 Budget: ${scenario['budget_per_run']:.2f}")

            with adapter.trace_with_governance(
                name=f"automated_{scenario['scenario'].lower().replace(' ', '_')}",
                customer_id=scenario['customer_id'],
                cost_center="automation",
                automation_type=scenario['frequency'],
                priority=scenario['priority']
            ) as trace:

                # Simulate automated evaluation
                start_time = time.time()

                # Mock evaluation tasks
                evaluation_tasks = [
                    "Response quality assessment",
                    "Cost efficiency analysis",
                    "Latency performance check",
                    "Governance compliance validation"
                ]

                task_results = []
                total_cost = 0.0

                for task in evaluation_tasks:
                    print(f"     ✅ {task}")

                    # Simulate task cost and time
                    task_cost = scenario['budget_per_run'] / len(evaluation_tasks)
                    total_cost += task_cost
                    time.sleep(0.05)  # Simulate processing

                duration_ms = (time.time() - start_time) * 1000

                # Record automation results
                automation_result = {
                    "scenario": scenario['scenario'],
                    "frequency": scenario['frequency'],
                    "total_cost": total_cost,
                    "duration_ms": duration_ms,
                    "tasks_completed": len(evaluation_tasks),
                    "budget_utilization": total_cost / scenario['budget_per_run'],
                    "governance": {
                        "customer_id": scenario['customer_id'],
                        "cost_center": "automation",
                        "priority": scenario['priority']
                    }
                }

                automation_results.append(automation_result)

                print(f"   ✅ Completed in {duration_ms:.0f}ms")
                print(f"   💰 Cost: ${total_cost:.6f}")
                print(f"   📊 Budget utilization: {automation_result['budget_utilization']:.1%}")

        # Summary of automation results
        total_automation_cost = sum(result['total_cost'] for result in automation_results)

        print("\n📈 Automation Pipeline Summary:")
        print(f"   🤖 Scenarios executed: {len(automation_results)}")
        print(f"   💰 Total automation cost: ${total_automation_cost:.6f}")
        print(f"   📊 Average cost per scenario: ${total_automation_cost / len(automation_results):.6f}")

        print("\n💡 Governance Benefits of Automation:")
        print("   ✅ Consistent evaluation quality across all scenarios")
        print("   ✅ Budget tracking and utilization optimization")
        print("   ✅ Customer and cost center attribution for all evaluations")
        print("   ✅ Automated compliance validation and reporting")
        print("   ✅ Scalable evaluation pipeline with governance controls")

        return automation_results

    except Exception as e:
        print(f"❌ Evaluation automation failed: {e}")
        return None


def show_next_steps():
    """Show next steps for advanced evaluation patterns."""
    print("\n🚀 Advanced Evaluation Patterns & Next Steps")
    print("=" * 43)

    advanced_patterns = [
        ("🎯 A/B Testing", "Compare model versions with statistical significance",
         "Set up controlled experiments with governance attribution"),
        ("📊 Custom Metrics", "Define domain-specific evaluation metrics",
         "Create evaluators for your specific use case"),
        ("🔄 Continuous Integration", "Integrate evaluations into CI/CD pipelines",
         "Automated quality gates with governance compliance"),
        ("📈 Performance Monitoring", "Real-time evaluation in production",
         "Monitor model drift and performance degradation"),
        ("🏭 Enterprise Deployment", "Scale evaluation workflows across teams",
         "python production_patterns.py")
    ]

    for title, description, example in advanced_patterns:
        print(f"   {title}")
        print(f"     Purpose: {description}")
        print(f"     Next Step: {example}")
        print()

    print("📚 Resources for Advanced Evaluation:")
    print("   • Prompt Management: python prompt_management.py")
    print("   • Advanced Observability: python advanced_observability.py")
    print("   • Production Patterns: python production_patterns.py")
    print("   • Comprehensive Guide: docs/integrations/langfuse.md")


def main():
    """Main function to run the evaluation integration example."""
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check prerequisites
    if not os.getenv('LANGFUSE_PUBLIC_KEY'):
        print("❌ Missing LANGFUSE_PUBLIC_KEY environment variable")
        print("💡 Get your keys at: https://cloud.langfuse.com/")
        return False

    if not os.getenv('LANGFUSE_SECRET_KEY'):
        print("❌ Missing LANGFUSE_SECRET_KEY environment variable")
        print("💡 Get your keys at: https://cloud.langfuse.com/")
        return False

    if not any([os.getenv('OPENAI_API_KEY'), os.getenv('ANTHROPIC_API_KEY')]):
        print("❌ No AI provider API keys found")
        print("💡 Set at least one:")
        print("   export OPENAI_API_KEY='your_openai_key'")
        print("   export ANTHROPIC_API_KEY='your_anthropic_key'")
        return False

    # Run evaluation demonstrations
    success = True

    # Basic evaluation
    basic_results = demonstrate_basic_evaluation()
    success &= basic_results is not None

    # Batch evaluation
    batch_results = demonstrate_batch_evaluation()
    success &= batch_results is not None

    # Model comparison
    comparison_results = demonstrate_model_comparison()
    success &= comparison_results is not None

    # Evaluation automation
    automation_results = demonstrate_evaluation_automation()
    success &= automation_results is not None

    if success:
        show_next_steps()
        print("\n" + "📊" * 20)
        print("LLM Evaluation + GenOps Governance integration complete!")
        print("Comprehensive evaluation workflows with cost intelligence!")
        print("Enterprise-ready governance for all evaluation processes!")
        print("📊" * 20)
        return True
    else:
        print("\n❌ Some evaluations failed. Check the errors above.")
        return False


if __name__ == "__main__":
    """Main entry point."""
    success = main()
    sys.exit(0 if success else 1)
