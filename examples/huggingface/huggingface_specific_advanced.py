#!/usr/bin/env python3
"""
Hugging Face Specific Advanced Features Example

This example showcases advanced Hugging Face-specific features and patterns
unique to the Hugging Face ecosystem that demonstrate the full capabilities
of GenOps AI governance integration.

Example usage:
    python huggingface_specific_advanced.py

Features demonstrated:
- Advanced multi-task AI operation workflows
- Cross-provider model comparison and optimization
- Task-specific cost optimization strategies
- Pipeline composition with cost aggregation
- Model hub integration patterns
- Advanced cost context management
- Provider detection and intelligent routing
"""

import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Structured result for AI task operations."""

    task_name: str
    result: Any
    provider: str
    model: str
    cost: float
    tokens_input: int
    tokens_output: int
    execution_time: float
    metadata: dict[str, Any]


def demonstrate_advanced_multi_task_pipeline():
    """
    Demonstrate advanced multi-task AI pipeline with cost optimization.

    This showcases Hugging Face-specific patterns for complex AI workflows
    that span multiple tasks and providers with unified cost tracking.
    """

    print("🤗 Advanced Hugging Face Multi-Task Pipeline")
    print("=" * 60)
    print("Demonstrating complex AI workflows with cost optimization...")
    print()

    try:
        from genops.providers.huggingface import (
            GenOpsHuggingFaceAdapter,
            create_huggingface_cost_context,
            production_workflow_context,
        )

        # Advanced workflow with multiple tasks and intelligent provider selection
        with production_workflow_context(
            workflow_name="content_intelligence_pipeline",
            customer_id="enterprise_client_001",
            team="ai_content_team",
            project="intelligent_content_system",
            environment="production",
            cost_center="product_development",
            business_unit="content_ai",
        ) as (workflow, workflow_id):
            print(f"🚀 Started workflow: {workflow_id}")
            workflow.record_step("workflow_initialization", {"total_tasks_planned": 6})

            # Initialize the adapter for advanced operations
            adapter = GenOpsHuggingFaceAdapter()
            results = []

            # Task 1: Content Generation with Provider Optimization
            workflow.record_step("content_generation_start")
            content_models = [
                "gpt-3.5-turbo",
                "claude-3-haiku",
                "microsoft/DialoGPT-medium",
            ]

            best_content_result = None
            best_content_cost = float("inf")

            for model in content_models:
                try:
                    start_time = time.time()

                    result = adapter.text_generation(
                        prompt="Create a comprehensive guide about sustainable energy solutions for small businesses",
                        model=model,
                        max_new_tokens=300,
                        temperature=0.7,
                        team="ai_content_team",
                        project="intelligent_content_system",
                        feature="content_generation",
                        task_complexity="high",
                    )

                    execution_time = time.time() - start_time

                    # Record the operation in workflow context
                    detected_provider = adapter._detect_provider(model)
                    estimated_cost = adapter._calculate_cost(
                        provider=detected_provider,
                        model=model,
                        input_tokens=adapter._estimate_tokens(
                            "Create a comprehensive guide about sustainable energy solutions for small businesses"
                        ),
                        output_tokens=adapter._estimate_tokens(str(result)),
                        task="text-generation",
                    )

                    workflow.record_hf_operation(
                        operation_name=f"content_generation_{model}",
                        provider=detected_provider,
                        model=model,
                        tokens_input=adapter._estimate_tokens(
                            "Create a comprehensive guide about sustainable energy solutions for small businesses"
                        ),
                        tokens_output=adapter._estimate_tokens(str(result)),
                        task="text-generation",
                    )

                    # Track best performer for cost optimization
                    if estimated_cost < best_content_cost:
                        best_content_cost = estimated_cost
                        best_content_result = TaskResult(
                            task_name="content_generation",
                            result=result,
                            provider=detected_provider,
                            model=model,
                            cost=estimated_cost,
                            tokens_input=adapter._estimate_tokens(
                                "Create a comprehensive guide about sustainable energy solutions for small businesses"
                            ),
                            tokens_output=adapter._estimate_tokens(str(result)),
                            execution_time=execution_time,
                            metadata={"optimization_rank": "best_cost"},
                        )

                    print(
                        f"✅ Content generation with {model} ({detected_provider}): ${estimated_cost:.4f}"
                    )

                except Exception as e:
                    print(f"❌ Content generation failed with {model}: {e}")
                    workflow.record_alert("content_generation_error", str(e), "warning")
                    continue

            if best_content_result:
                results.append(best_content_result)
                workflow.record_checkpoint(
                    "content_generation_complete",
                    {
                        "best_model": best_content_result.model,
                        "best_cost": best_content_result.cost,
                    },
                )
                print(
                    f"🎯 Best content model: {best_content_result.model} (${best_content_result.cost:.4f})"
                )

            # Task 2: Advanced Multi-Document Embedding Pipeline
            workflow.record_step("embedding_pipeline_start")

            documents = [
                "Sustainable energy solutions for small businesses",
                "Renewable energy cost analysis and ROI calculations",
                "Green technology implementation strategies",
                "Environmental impact assessment methodologies",
                "Energy efficiency optimization techniques",
            ]

            embedding_models = [
                "sentence-transformers/all-MiniLM-L6-v2",
                "text-embedding-ada-002",
            ]

            embedding_results = {}

            for model in embedding_models:
                try:
                    start_time = time.time()

                    # Process documents in batch for efficiency
                    embeddings = adapter.feature_extraction(
                        inputs=documents,
                        model=model,
                        team="ai_content_team",
                        project="intelligent_content_system",
                        feature="document_embedding",
                        batch_size=len(documents),
                    )

                    execution_time = time.time() - start_time

                    detected_provider = adapter._detect_provider(model)
                    total_input_tokens = sum(
                        adapter._estimate_tokens(doc) for doc in documents
                    )

                    workflow.record_hf_operation(
                        operation_name=f"batch_embedding_{model}",
                        provider=detected_provider,
                        model=model,
                        tokens_input=total_input_tokens,
                        tokens_output=0,
                        task="feature-extraction",
                    )

                    embedding_results[model] = {
                        "embeddings": embeddings,
                        "provider": detected_provider,
                        "execution_time": execution_time,
                        "documents_processed": len(documents),
                        "embedding_dimension": len(embeddings[0]) if embeddings else 0,
                    }

                    print(
                        f"✅ Embedding with {model}: {len(documents)} docs, {len(embeddings[0]) if embeddings else 0}D"
                    )

                except Exception as e:
                    print(f"❌ Embedding failed with {model}: {e}")
                    workflow.record_alert("embedding_error", str(e), "warning")
                    continue

            workflow.record_checkpoint(
                "embedding_pipeline_complete",
                {
                    "models_tested": len(embedding_models),
                    "successful_models": len(embedding_results),
                },
            )

            # Task 3: Cross-Task Intelligence with Cost Optimization Context
            workflow.record_step("cross_task_intelligence")

            with create_huggingface_cost_context(
                f"{workflow_id}_intelligence_analysis"
            ) as intelligence_context:
                # Analyze content and embeddings for insights
                if best_content_result and embedding_results:
                    # Advanced prompt that leverages both content and embeddings
                    analysis_prompt = f"""
                    Based on the generated content: "{str(best_content_result.result)[:200]}..."
                    And document embeddings from {len(documents)} related documents,
                    provide strategic recommendations for content optimization and cost efficiency.

                    Consider: content quality, audience engagement, and production cost efficiency.
                    """

                    adapter.text_generation(
                        prompt=analysis_prompt,
                        model="claude-3-haiku",  # Cost-optimized choice for analysis
                        max_new_tokens=250,
                        temperature=0.3,
                        team="ai_content_team",
                        project="intelligent_content_system",
                        feature="cross_task_analysis",
                    )

                    intelligence_summary = intelligence_context.get_current_summary()

                    workflow.record_step(
                        "intelligence_analysis_complete",
                        {
                            "analysis_cost": intelligence_summary.total_cost
                            if intelligence_summary
                            else 0,
                            "content_source_cost": best_content_result.cost,
                            "total_intelligence_cost": (
                                intelligence_summary.total_cost
                                if intelligence_summary
                                else 0
                            )
                            + best_content_result.cost,
                        },
                    )

                    print("🧠 Cross-task intelligence analysis complete")
                    print(f"   Content cost: ${best_content_result.cost:.4f}")
                    print(
                        f"   Analysis cost: ${intelligence_summary.total_cost if intelligence_summary else 0:.4f}"
                    )

            # Task 4: Advanced Image Generation with Model Hub Integration
            workflow.record_step("image_generation_start")

            try:
                image_result = adapter.text_to_image(
                    prompt="Professional infographic showing sustainable energy solutions for small businesses, modern design",
                    model="runwayml/stable-diffusion-v1-5",
                    team="ai_content_team",
                    project="intelligent_content_system",
                    feature="visual_content_generation",
                )

                workflow.record_hf_operation(
                    operation_name="professional_infographic_generation",
                    provider="huggingface_hub",
                    model="runwayml/stable-diffusion-v1-5",
                    tokens_input=adapter._estimate_tokens(
                        "Professional infographic showing sustainable energy solutions for small businesses, modern design"
                    ),
                    tokens_output=0,
                    task="text-to-image",
                )

                print(
                    f"✅ Generated professional infographic (size: {len(image_result) if isinstance(image_result, bytes) else 'unknown'})"
                )

                workflow.record_checkpoint(
                    "image_generation_complete",
                    {
                        "image_generated": True,
                        "model_used": "runwayml/stable-diffusion-v1-5",
                    },
                )

            except Exception as e:
                print(f"❌ Image generation failed: {e}")
                workflow.record_alert("image_generation_error", str(e), "error")

            # Task 5: Cost Optimization Analysis and Recommendations
            workflow.record_step("cost_optimization_analysis")

            current_cost_summary = workflow.get_current_cost_summary()
            if current_cost_summary:
                print("\n💰 Workflow Cost Analysis:")
                print(f"   Total Cost: ${current_cost_summary.total_cost:.4f}")
                print(
                    f"   Providers Used: {list(current_cost_summary.unique_providers)}"
                )
                print(f"   Models Used: {len(current_cost_summary.unique_models)}")
                print(
                    f"   Tasks Performed: {list(current_cost_summary.tasks_performed)}"
                )

                # Record performance metrics
                workflow.record_performance_metric(
                    "total_workflow_cost", current_cost_summary.total_cost, "USD"
                )
                workflow.record_performance_metric(
                    "provider_diversity",
                    len(current_cost_summary.unique_providers),
                    "count",
                )
                workflow.record_performance_metric(
                    "model_diversity", len(current_cost_summary.unique_models), "count"
                )

                # Cost optimization recommendations
                provider_breakdown = current_cost_summary.get_provider_breakdown()
                most_expensive_provider = max(
                    provider_breakdown.keys(),
                    key=lambda p: provider_breakdown[p]["cost"],
                )

                workflow.record_alert(
                    "cost_optimization_opportunity",
                    f"Provider {most_expensive_provider} accounts for ${provider_breakdown[most_expensive_provider]['cost']:.4f} - consider alternatives",
                    "info",
                )

                print(
                    f"   Most Expensive Provider: {most_expensive_provider} (${provider_breakdown[most_expensive_provider]['cost']:.4f})"
                )

                # Generate cost optimization recommendations
                if (
                    current_cost_summary.total_cost > 0.01
                ):  # Threshold for expensive operations
                    workflow.record_alert(
                        "high_cost_workflow",
                        f"Workflow cost ${current_cost_summary.total_cost:.4f} exceeds recommended threshold",
                        "warning",
                    )

            # Task 6: Final Workflow Optimization and Reporting
            workflow.record_step("final_optimization_and_reporting")

            workflow_metadata = workflow.get_workflow_metadata()

            print("\n📊 Final Workflow Report:")
            print(f"   Workflow ID: {workflow_metadata['workflow_id']}")
            print(f"   Steps Completed: {workflow_metadata['step_count']}")
            print(f"   Operations Performed: {workflow_metadata['operation_count']}")
            print(f"   Checkpoints Recorded: {workflow_metadata['checkpoint_count']}")
            print(f"   Alerts Generated: {workflow_metadata['alert_count']}")
            print(f"   Final Cost: ${workflow_metadata['current_cost']:.4f}")
            print(f"   Providers: {', '.join(workflow_metadata['providers_used'])}")

            # Set final governance attributes
            workflow.set_governance_attribute("workflow_success", True)
            workflow.set_governance_attribute(
                "final_cost", workflow_metadata["current_cost"]
            )
            workflow.set_governance_attribute(
                "efficiency_score",
                min(100, max(0, 100 - (workflow_metadata["current_cost"] * 1000))),
            )

            workflow.record_checkpoint("workflow_complete", workflow_metadata)

            print("✅ Advanced multi-task pipeline completed successfully!")
            print(f"   Total operations: {workflow_metadata['operation_count']}")
            print(f"   Total cost: ${workflow_metadata['current_cost']:.4f}")

    except ImportError as e:
        print(f"❌ Required GenOps components not available: {e}")
        print(
            "Please ensure GenOps is properly installed with: pip install genops-ai[huggingface]"
        )
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")


def demonstrate_advanced_provider_detection_and_routing():
    """
    Demonstrate advanced provider detection and intelligent routing.

    This showcases Hugging Face-specific provider detection patterns
    and cost-aware routing strategies.
    """

    print("\n🔍 Advanced Provider Detection and Routing")
    print("=" * 50)
    print("Demonstrating intelligent provider detection and routing...")
    print()

    try:
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        from genops.providers.huggingface_pricing import (
            compare_model_costs,
            detect_model_provider,
            get_cost_optimization_suggestions,
        )

        GenOpsHuggingFaceAdapter()

        # Test models from different providers
        test_models = [
            "gpt-4",
            "gpt-3.5-turbo",
            "claude-3-opus",
            "claude-3-haiku",
            "command-r",
            "llama-2-7b-chat",
            "mistral-7b-instruct",
            "microsoft/DialoGPT-medium",
            "sentence-transformers/all-MiniLM-L6-v2",
            "runwayml/stable-diffusion-v1-5",
        ]

        print("🕵️ Provider Detection Analysis:")
        detection_results = {}

        for model in test_models:
            detected_provider = detect_model_provider(model)
            detection_results[model] = detected_provider
            print(f"   {model:<40} → {detected_provider}")

        # Cost comparison for text generation task
        print(
            "\n💰 Cost Comparison for Text Generation (1000 input, 500 output tokens):"
        )

        text_generation_models = [
            "gpt-3.5-turbo",
            "claude-3-haiku",
            "microsoft/DialoGPT-medium",
            "llama-2-7b-chat",
        ]

        cost_comparison = compare_model_costs(
            models=text_generation_models,
            input_tokens=1000,
            output_tokens=500,
            task="text-generation",
        )

        for model, cost_data in cost_comparison.items():
            print(
                f"   {model:<30} ${cost_data['cost']:.6f} ({cost_data['provider']}) - {cost_data['relative_cost']:.1f}x"
            )

        # Cost optimization suggestions
        print("\n🎯 Cost Optimization Suggestions for GPT-4:")
        optimization = get_cost_optimization_suggestions("gpt-4", "text-generation")

        print(
            f"   Current: {optimization['current_model']['model']} - ${optimization['current_model']['cost_per_1k']['input']:.6f}/1k input"
        )
        print("   Alternatives:")
        for alt in optimization["alternatives"][:3]:  # Top 3 alternatives
            print(
                f"     {alt['model']:<25} - ${alt['cost_per_1k']['input']:.6f}/1k input ({alt['savings']:.1f}% savings)"
            )

        print("\n💡 Optimization Tips:")
        for tip in optimization["optimization_tips"][:3]:  # Top 3 tips
            print(f"   • {tip}")

    except ImportError as e:
        print(f"❌ Advanced pricing components not available: {e}")
    except Exception as e:
        print(f"❌ Provider detection demo failed: {e}")


def demonstrate_huggingface_hub_integration_patterns():
    """
    Demonstrate advanced Hugging Face Hub integration patterns.

    This showcases Hub-specific features like model discovery,
    task classification, and community model integration.
    """

    print("\n🤗 Hub Integration Patterns")
    print("=" * 35)
    print("Demonstrating Hub-specific integration patterns...")
    print()

    try:
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter

        adapter = GenOpsHuggingFaceAdapter()

        # Hub model categories with examples
        hub_model_categories = {
            "conversational": [
                "microsoft/DialoGPT-medium",
                "facebook/blenderbot-400M-distill",
            ],
            "text_classification": [
                "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "ProsusAI/finbert",
            ],
            "text_generation": ["gpt2", "distilgpt2"],
            "feature_extraction": [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
            ],
            "text_to_image": [
                "runwayml/stable-diffusion-v1-5",
                "CompVis/stable-diffusion-v1-4",
            ],
        }

        print("🏷️ Hub Model Categories and Cost Analysis:")

        for category, models in hub_model_categories.items():
            print(f"\n   {category.upper()}:")

            for model in models:
                try:
                    detected_provider = adapter._detect_provider(model)

                    # Estimate cost for typical operation
                    if category in ["conversational", "text_generation"]:
                        estimated_cost = adapter._calculate_cost(
                            provider=detected_provider,
                            model=model,
                            input_tokens=100,
                            output_tokens=50,
                            task="text-generation",
                        )
                        cost_desc = f"${estimated_cost:.6f} (100 in, 50 out)"

                    elif category == "feature_extraction":
                        estimated_cost = adapter._calculate_cost(
                            provider=detected_provider,
                            model=model,
                            input_tokens=100,
                            output_tokens=0,
                            task="feature-extraction",
                        )
                        cost_desc = f"${estimated_cost:.6f} (100 tokens)"

                    elif category == "text_to_image":
                        estimated_cost = adapter._calculate_cost(
                            provider=detected_provider,
                            model=model,
                            input_tokens=20,
                            output_tokens=0,
                            task="text-to-image",
                        )
                        cost_desc = f"${estimated_cost:.6f} (image gen)"

                    else:
                        cost_desc = "Cost estimation not available"

                    print(f"     {model:<45} → {detected_provider:<15} {cost_desc}")

                except Exception as e:
                    print(f"     {model:<45} → Error: {e}")

        # Demonstrate task-specific optimization
        print("\n🎯 Task-Specific Optimization Recommendations:")

        task_recommendations = {
            "High-volume text classification": {
                "recommended": [
                    "distilbert-base-uncased",
                    "cardiffnlp/twitter-roberta-base-sentiment-latest",
                ],
                "reason": "Optimized for speed and cost efficiency",
            },
            "High-quality content generation": {
                "recommended": ["gpt-3.5-turbo", "claude-3-haiku"],
                "reason": "Best quality-to-cost ratio",
            },
            "Batch document embedding": {
                "recommended": ["sentence-transformers/all-MiniLM-L6-v2"],
                "reason": "Free Hub model with good performance",
            },
            "Creative image generation": {
                "recommended": ["runwayml/stable-diffusion-v1-5"],
                "reason": "Community-proven model with reasonable cost",
            },
        }

        for task, recommendation in task_recommendations.items():
            print(f"\n   {task}:")
            print(f"     Recommended: {', '.join(recommendation['recommended'])}")
            print(f"     Reason: {recommendation['reason']}")

    except Exception as e:
        print(f"❌ Hub integration demo failed: {e}")


def main():
    """Main demonstration function."""

    print("🤗 Hugging Face Advanced Features Demonstration")
    print("=" * 60)
    print("This example showcases advanced Hugging Face-specific features")
    print("and integration patterns unique to the Hugging Face ecosystem.")
    print("=" * 60)
    print()

    # Run all demonstrations
    demonstrate_advanced_multi_task_pipeline()
    demonstrate_advanced_provider_detection_and_routing()
    demonstrate_huggingface_hub_integration_patterns()

    print("\n" + "=" * 60)
    print("✅ All advanced Hugging Face demonstrations completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
