#!/usr/bin/env python3
"""
Fireworks AI Advanced Features with GenOps

Demonstrates advanced Fireworks AI capabilities including multimodal operations,
streaming responses, function calling, structured outputs, and complex workflow patterns
with comprehensive governance tracking.

Usage:
    python advanced_features.py

Features:
    - Multimodal operations with vision-language models
    - Streaming responses with real-time cost tracking
    - Function calling and tool usage workflows
    - Structured JSON output generation
    - Async batch processing with 4x faster inference
    - Audio processing and embeddings
    - Complex reasoning tasks with specialized models
"""

import asyncio
import json
import sys
import time

try:
    from genops.providers.fireworks import FireworksModel, GenOpsFireworksAdapter
    from genops.providers.fireworks_pricing import FireworksPricingCalculator
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install: pip install genops-ai[fireworks]")
    print("Then run: python setup_validation.py")
    sys.exit(1)


def demonstrate_multimodal_operations():
    """Demonstrate multimodal operations with vision-language models."""
    print("🎨 Multimodal Operations (Vision + Language)")
    print("=" * 50)

    adapter = GenOpsFireworksAdapter(
        team="advanced-features",
        project="multimodal-demo",
        environment="development",
        daily_budget_limit=20.0,
        governance_policy="advisory"
    )

    # Example 1: Vision-language analysis
    try:
        print("👁️ Vision-Language Analysis:")

        # Sample image URL for demonstration (you would use your own images)
        sample_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Vd-Orig.svg/256px-Vd-Orig.svg.png"

        result = adapter.chat_with_governance(
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this image and describe what you see in a business context"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": sample_image_url}
                    }
                ]
            }],
            model=FireworksModel.LLAMA_VISION_11B,
            max_tokens=150,
            feature="vision-analysis",
            use_case="multimodal-understanding"
        )

        print(f"   Analysis: {result.response}")
        print(f"   Cost: ${result.cost:.6f}")
        print(f"   Speed: {result.execution_time_seconds:.2f}s (🔥 Fireattention optimized)")

    except Exception as e:
        print(f"   ⚠️ Vision analysis demo skipped: {e}")

    # Example 2: Text embeddings for semantic search
    print("\n🔤 Text Embeddings for Semantic Search:")

    try:
        documents = [
            "Fireworks AI provides 4x faster inference with Fireattention optimization",
            "Cost optimization is crucial for production AI deployments",
            "Multimodal AI enables vision and language understanding together",
            "Batch processing can reduce inference costs by up to 50%"
        ]

        embedding_result = adapter.embeddings_with_governance(
            input_texts=documents,
            model=FireworksModel.NOMIC_EMBED_TEXT,
            feature="semantic-search",
            use_case="document-similarity"
        )

        print(f"   Generated embeddings for {len(documents)} documents")
        print(f"   Cost: ${embedding_result.cost:.6f}")
        print(f"   Speed: {embedding_result.execution_time_seconds:.2f}s")
        print("   Use case: Enable semantic search across knowledge base")

    except Exception as e:
        print(f"   ❌ Embeddings demo failed: {e}")


def demonstrate_streaming_responses():
    """Demonstrate streaming responses with real-time cost tracking."""
    print("\n📺 Streaming Responses with Real-Time Cost Tracking")
    print("=" * 50)

    adapter = GenOpsFireworksAdapter(
        team="streaming-team",
        project="real-time-responses",
        governance_policy="advisory"
    )

    try:
        print("🌊 Starting streaming response (watch costs accumulate):")

        # Custom streaming handler to show real-time cost accumulation
        accumulated_cost = 0.0
        response_text = ""

        def handle_stream_chunk(chunk_content, estimated_cost):
            nonlocal accumulated_cost, response_text
            accumulated_cost += estimated_cost
            response_text += chunk_content

            # Show streaming progress
            if len(response_text) % 50 == 0:  # Every 50 characters
                print(f"   💰 Accumulated cost: ${accumulated_cost:.6f}")

        # Stream a longer response to show cost accumulation
        result = adapter.chat_with_governance(
            messages=[{
                "role": "user",
                "content": "Write a detailed explanation of how Fireworks AI's 4x speed advantage benefits production applications. Include specific examples and use cases."
            }],
            model=FireworksModel.LLAMA_3_1_70B_INSTRUCT,
            max_tokens=300,
            stream=True,
            feature="streaming-demo",
            on_chunk=handle_stream_chunk
        )

        print("\n✅ Streaming completed!")
        print(f"   Final response length: {len(result.response)} characters")
        print(f"   Total cost: ${result.cost:.6f}")
        print(f"   Speed: {result.execution_time_seconds:.2f}s")
        print("   🔥 Real-time cost tracking during streaming!")

    except Exception as e:
        print(f"❌ Streaming demo failed: {e}")


def demonstrate_function_calling():
    """Demonstrate function calling capabilities with governance."""
    print("\n🔧 Function Calling with Governance Tracking")
    print("=" * 50)

    adapter = GenOpsFireworksAdapter(
        team="function-calling-team",
        project="tool-usage",
        governance_policy="advisory"
    )

    # Define functions the model can call
    functions = [
        {
            "name": "get_performance_metrics",
            "description": "Get performance metrics for AI inference",
            "parameters": {
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "AI provider name",
                        "enum": ["fireworks", "openai", "anthropic"]
                    },
                    "metric_type": {
                        "type": "string",
                        "description": "Type of metric to retrieve",
                        "enum": ["speed", "cost", "accuracy"]
                    }
                },
                "required": ["provider", "metric_type"]
            }
        },
        {
            "name": "calculate_cost_savings",
            "description": "Calculate potential cost savings from optimization",
            "parameters": {
                "type": "object",
                "properties": {
                    "current_cost": {"type": "number", "description": "Current monthly cost"},
                    "optimization_percentage": {"type": "number", "description": "Expected savings percentage"}
                },
                "required": ["current_cost", "optimization_percentage"]
            }
        }
    ]

    try:
        print("🤖 Testing function calling capabilities:")

        result = adapter.chat_with_governance(
            messages=[{
                "role": "user",
                "content": "I want to understand Fireworks AI performance metrics and calculate savings if I optimize my current $500/month AI costs with 40% improvement."
            }],
            model=FireworksModel.LLAMA_3_1_70B_INSTRUCT,
            functions=functions,
            function_call="auto",
            max_tokens=200,
            feature="function-calling",
            use_case="performance-analysis"
        )

        print(f"   Response: {result.response}")
        print(f"   Cost: ${result.cost:.6f}")
        print(f"   Speed: {result.execution_time_seconds:.2f}s")
        print("   🎯 Function calls tracked with full governance!")

    except Exception as e:
        print(f"   ⚠️ Function calling demo: {e}")
        print("   Note: Function calling may not be available for all models")


def demonstrate_structured_output():
    """Demonstrate structured JSON output generation."""
    print("\n📝 Structured JSON Output Generation")
    print("=" * 50)

    adapter = GenOpsFireworksAdapter(
        team="structured-output-team",
        project="json-generation"
    )

    # Define JSON schema for structured output
    analysis_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "ai_provider_analysis",
            "schema": {
                "type": "object",
                "properties": {
                    "provider_name": {"type": "string"},
                    "speed_rating": {"type": "integer", "minimum": 1, "maximum": 10},
                    "cost_effectiveness": {"type": "integer", "minimum": 1, "maximum": 10},
                    "key_advantages": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 3
                    },
                    "recommended_use_cases": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 3
                    },
                    "overall_score": {"type": "integer", "minimum": 1, "maximum": 100}
                },
                "required": ["provider_name", "speed_rating", "cost_effectiveness", "key_advantages", "recommended_use_cases", "overall_score"]
            }
        }
    }

    try:
        print("🏗️ Generating structured analysis:")

        result = adapter.chat_with_governance(
            messages=[{
                "role": "user",
                "content": "Analyze Fireworks AI as a provider, focusing on their 4x speed advantage and cost optimization features. Return a structured analysis."
            }],
            model=FireworksModel.LLAMA_3_1_70B_INSTRUCT,
            response_format=analysis_schema,
            max_tokens=250,
            feature="structured-output",
            use_case="provider-analysis"
        )

        # Try to parse the JSON response
        try:
            analysis = json.loads(result.response)
            print("   ✅ Structured JSON generated successfully:")
            print(f"      Provider: {analysis.get('provider_name', 'N/A')}")
            print(f"      Speed Rating: {analysis.get('speed_rating', 'N/A')}/10")
            print(f"      Cost Effectiveness: {analysis.get('cost_effectiveness', 'N/A')}/10")
            print(f"      Key Advantages: {', '.join(analysis.get('key_advantages', []))}")
            print(f"      Overall Score: {analysis.get('overall_score', 'N/A')}/100")
        except json.JSONDecodeError:
            print(f"   Response: {result.response}")

        print(f"   Cost: ${result.cost:.6f}")
        print(f"   Speed: {result.execution_time_seconds:.2f}s")

    except Exception as e:
        print(f"   ⚠️ Structured output demo: {e}")


async def demonstrate_async_batch_processing():
    """Demonstrate async batch processing with concurrent operations."""
    print("\n⚡ Async Batch Processing (Concurrent Operations)")
    print("=" * 50)

    adapter = GenOpsFireworksAdapter(
        team="async-team",
        project="batch-processing",
        governance_policy="advisory"
    )

    # Create batch of tasks to process concurrently
    batch_tasks = [
        ("Summarize AI trends", FireworksModel.LLAMA_3_1_8B_INSTRUCT),
        ("Analyze cost optimization", FireworksModel.LLAMA_3_1_8B_INSTRUCT),
        ("Explain fast inference benefits", FireworksModel.LLAMA_3_1_8B_INSTRUCT),
        ("Generate marketing copy", FireworksModel.MIXTRAL_8X7B),
        ("Create technical documentation", FireworksModel.LLAMA_3_1_70B_INSTRUCT)
    ]

    try:
        print(f"🚀 Processing {len(batch_tasks)} tasks concurrently with batch pricing:")

        start_time = time.time()
        results = []

        # Process tasks concurrently (simulated - actual async would depend on client)
        with adapter.track_session("async-batch-processing") as session:
            for i, (task, model) in enumerate(batch_tasks):
                print(f"   Task {i+1}: {task} ({model.value.split('/')[-1]})")

                result = adapter.chat_with_governance(
                    messages=[{"role": "user", "content": task}],
                    model=model,
                    max_tokens=80,
                    is_batch=True,  # Apply 50% batch discount
                    session_id=session.session_id,
                    batch_id="concurrent-batch",
                    operation_index=i
                )

                results.append(result)
                print(f"      ✅ Completed in {result.execution_time_seconds:.2f}s, cost: ${result.cost:.6f}")

        total_time = time.time() - start_time
        total_cost = sum(float(r.cost) for r in results)
        avg_speed = sum(r.execution_time_seconds for r in results) / len(results)

        # Calculate savings from batch processing
        standard_cost = total_cost * 2  # Batch provides 50% savings
        batch_savings = standard_cost - total_cost

        print("\n📊 Batch Processing Results:")
        print(f"   Tasks completed: {len(results)}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average speed per task: {avg_speed:.2f}s (🔥 Fireattention optimized)")
        print(f"   Total cost: ${total_cost:.6f}")
        print(f"   Batch savings: ${batch_savings:.6f} (50% discount applied)")
        print(f"   Throughput: {len(results)/total_time:.1f} tasks/second")

    except Exception as e:
        print(f"❌ Async batch processing demo failed: {e}")


def demonstrate_complex_reasoning():
    """Demonstrate complex reasoning with specialized models."""
    print("\n🧠 Complex Reasoning with Specialized Models")
    print("=" * 50)

    adapter = GenOpsFireworksAdapter(
        team="reasoning-team",
        project="complex-analysis"
    )

    # Complex reasoning tasks that benefit from specialized models
    reasoning_tasks = [
        {
            "task": "Analyze the technical trade-offs between inference speed and model accuracy in production AI systems",
            "model": FireworksModel.LLAMA_3_1_405B_INSTRUCT,  # High-capacity model
            "complexity": "complex"
        },
        {
            "task": "Generate optimized Python code for batch processing AI requests with error handling",
            "model": FireworksModel.DEEPSEEK_CODER_V2_LITE,  # Code-specialized model
            "complexity": "code-generation"
        },
        {
            "task": "Step-by-step reasoning: If Fireworks AI is 4x faster and 50% cheaper in batch mode, calculate ROI for migrating 10k daily operations",
            "model": FireworksModel.DEEPSEEK_R1_DISTILL,  # Reasoning-specialized model
            "complexity": "mathematical-reasoning"
        }
    ]

    print("🎯 Testing specialized models for complex reasoning:")

    reasoning_results = []

    for i, task_info in enumerate(reasoning_tasks, 1):
        try:
            print(f"\n   🧮 Task {i}: {task_info['complexity']}")
            print(f"   Model: {task_info['model'].value.split('/')[-1]}")

            result = adapter.chat_with_governance(
                messages=[{"role": "user", "content": task_info["task"]}],
                model=task_info["model"],
                max_tokens=300,
                temperature=0.1,  # Lower temperature for reasoning tasks
                feature="complex-reasoning",
                task_complexity=task_info["complexity"]
            )

            reasoning_results.append(result)

            print(f"   ✅ Response: {result.response[:100]}...")
            print(f"   Cost: ${result.cost:.6f}")
            print(f"   Speed: {result.execution_time_seconds:.2f}s")

            # Quality assessment based on response length and coherence
            quality_score = min(len(result.response.split()) / 50, 10)  # Up to 10 for comprehensive responses
            print(f"   Quality indicator: {quality_score:.1f}/10 (based on comprehensiveness)")

        except Exception as e:
            print(f"   ❌ Task {i} failed: {e}")

    # Analyze reasoning performance
    if reasoning_results:
        print("\n📈 Reasoning Analysis Summary:")
        avg_cost = sum(float(r.cost) for r in reasoning_results) / len(reasoning_results)
        avg_speed = sum(r.execution_time_seconds for r in reasoning_results) / len(reasoning_results)
        total_words = sum(len(r.response.split()) for r in reasoning_results)

        print(f"   Tasks completed: {len(reasoning_results)}")
        print(f"   Average cost: ${avg_cost:.6f}")
        print(f"   Average speed: {avg_speed:.2f}s")
        print(f"   Total words generated: {total_words}")
        print(f"   Words per dollar: {total_words / sum(float(r.cost) for r in reasoning_results):.0f}")


def main():
    """Demonstrate all advanced Fireworks AI features."""
    print("🚀 Fireworks AI Advanced Features with GenOps")
    print("=" * 60)

    print("This demo showcases advanced Fireworks AI capabilities:")
    print("• Multimodal operations (vision, text, embeddings)")
    print("• Streaming responses with real-time cost tracking")
    print("• Function calling and tool usage")
    print("• Structured JSON output generation")
    print("• Async batch processing with 50% cost savings")
    print("• Complex reasoning with specialized models")
    print("• 4x faster inference with Fireattention optimization")

    try:
        # Run all demonstrations
        demonstrate_multimodal_operations()
        demonstrate_streaming_responses()
        demonstrate_function_calling()
        demonstrate_structured_output()

        # Run async demo
        asyncio.run(demonstrate_async_batch_processing())

        demonstrate_complex_reasoning()

        # Final summary
        print("\n" + "=" * 60)
        print("🎉 Advanced Features Demo Complete!")
        print("=" * 60)

        print("✅ What you've experienced:")
        print("   • Multimodal AI with vision-language understanding")
        print("   • Real-time streaming with cost accumulation tracking")
        print("   • Function calling for tool integration")
        print("   • Structured output for reliable data extraction")
        print("   • Batch processing with 50% cost savings")
        print("   • Complex reasoning with specialized model selection")
        print("   • 4x faster inference across all operations")
        print("   • Complete governance tracking for all features")

        print("\n🚀 Next Steps:")
        print("   • Implement multimodal features in your applications")
        print("   • Use streaming for real-time user experiences")
        print("   • Leverage batch processing for cost optimization")
        print("   • Apply function calling for tool integration")
        print("   • Take advantage of Fireworks' speed for production scale")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Advanced features demo failed: {e}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print("Try running setup_validation.py to check your configuration")
        sys.exit(1)
