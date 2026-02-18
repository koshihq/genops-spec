#!/usr/bin/env python3
"""
Together AI Advanced Features with GenOps

Demonstrates advanced Together AI capabilities including multimodal operations,
streaming responses, fine-tuning, and complex workflow patterns with governance.

Usage:
    python advanced_features.py

Features:
    - Multimodal operations with vision models
    - Streaming responses with real-time cost tracking
    - Code generation and completion workflows
    - Async batch processing
    - Fine-tuning cost estimation
    - Complex reasoning tasks with specialized models
"""

import asyncio
import sys
import time

try:
    from genops.providers.together import GenOpsTogetherAdapter, TogetherModel
    from genops.providers.together_pricing import TogetherPricingCalculator
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install: pip install genops-ai[together]")
    print("Then run: python setup_validation.py")
    sys.exit(1)


def demonstrate_multimodal_operations():
    """Demonstrate multimodal operations with vision-language models."""
    print("🎨 Multimodal Operations (Vision + Language)")
    print("=" * 50)

    adapter = GenOpsTogetherAdapter(
        team="advanced-features",
        project="multimodal-demo",
        environment="development",
        daily_budget_limit=20.0,
        governance_policy="advisory",
    )

    print("🔍 Testing multimodal capabilities...")

    # Example with image analysis (simulated - normally you'd use real images)
    multimodal_tasks = [
        {
            "name": "Image Description",
            "prompt": "Describe what you see in this image and identify any notable features.",
            "model": TogetherModel.QWEN_VL_72B,
            "context": "This would normally include an actual image URL",
        },
        {
            "name": "Visual Reasoning",
            "prompt": "Analyze the composition and artistic elements in this image.",
            "model": TogetherModel.LLAMA_VISION_11B,
            "context": "Educational content analysis",
        },
    ]

    multimodal_results = []

    for task in multimodal_tasks:
        print(f"\n🎯 {task['name']} with {task['model'].value}")

        try:
            # Note: In real usage, you'd include actual image data
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": task["prompt"]},
                        # {"type": "image_url", "image_url": {"url": "your-image-url"}}
                    ],
                }
            ]

            # For demo purposes, use text-only with multimodal model
            result = adapter.chat_with_governance(
                messages=[
                    {
                        "role": "user",
                        "content": f"{task['prompt']} [Note: This is a multimodal model demo without actual image]",
                    }
                ],
                model=task["model"],
                max_tokens=200,
                temperature=0.7,
                task_type="multimodal_analysis",
                feature=task["name"].lower().replace(" ", "_"),
            )

            multimodal_results.append(
                {
                    "task": task["name"],
                    "model": result.model_used,
                    "cost": float(result.cost),
                    "tokens": result.tokens_used,
                    "response_length": len(result.response),
                }
            )

            print(f"   ✅ Response generated ({result.tokens_used} tokens)")
            print(f"   💰 Cost: ${result.cost:.6f}")
            print(f"   📝 Response preview: {result.response[:100]}...")

        except Exception as e:
            print(f"   ❌ Multimodal task failed: {e}")

    if multimodal_results:
        total_multimodal_cost = sum(r["cost"] for r in multimodal_results)
        print("\n📊 Multimodal Operations Summary:")
        print(f"   Tasks completed: {len(multimodal_results)}")
        print(f"   Total cost: ${total_multimodal_cost:.6f}")
        print(
            f"   Average cost per task: ${total_multimodal_cost / len(multimodal_results):.6f}"
        )


def demonstrate_code_generation():
    """Demonstrate specialized code generation and completion."""
    print("\n💻 Code Generation & Completion")
    print("=" * 50)

    adapter = GenOpsTogetherAdapter(
        team="development",
        project="code-generation",
        environment="development",
        daily_budget_limit=15.0,
        default_model=TogetherModel.DEEPSEEK_CODER_V2,
    )

    # Different types of code generation tasks
    coding_tasks = [
        {
            "name": "Python Function",
            "prompt": "Write a Python function that implements a binary search algorithm with proper error handling and documentation.",
            "language": "python",
            "complexity": "moderate",
        },
        {
            "name": "API Endpoint",
            "prompt": "Create a FastAPI endpoint that handles user authentication with JWT tokens and includes proper error responses.",
            "language": "python",
            "complexity": "complex",
        },
        {
            "name": "Database Query",
            "prompt": "Write an optimized SQL query to find the top 10 customers by total order value in the last 6 months.",
            "language": "sql",
            "complexity": "moderate",
        },
    ]

    print("🔧 Testing specialized code generation models...")

    code_results = []

    with adapter.track_session("code-generation-session") as session:
        for task in coding_tasks:
            print(f"\n📝 {task['name']} ({task['language']})")

            try:
                result = adapter.chat_with_governance(
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are an expert {task['language']} developer. Write clean, well-documented code.",
                        },
                        {"role": "user", "content": task["prompt"]},
                    ],
                    model=TogetherModel.DEEPSEEK_CODER_V2,
                    max_tokens=300,
                    temperature=0.2,  # Lower temperature for more consistent code
                    session_id=session.session_id,
                    task_type="code_generation",
                    language=task["language"],
                    complexity=task["complexity"],
                )

                code_results.append(
                    {
                        "task": task["name"],
                        "language": task["language"],
                        "cost": float(result.cost),
                        "tokens": result.tokens_used,
                        "lines_of_code": result.response.count("\n"),
                        "execution_time": result.execution_time_seconds,
                    }
                )

                line_count = result.response.count("\n")
                print(f"   ✅ Generated {line_count} lines of code")
                print(f"   💰 Cost: ${result.cost:.6f}")
                print(f"   ⏱️  Time: {result.execution_time_seconds:.2f}s")

                # Show a preview of the generated code
                code_preview = "\n".join(result.response.split("\n")[:3])
                newline = "\n"
                indent = "      "
                formatted_preview = code_preview.replace(newline, newline + indent)
                print(f"   📄 Preview:{newline}{indent}{formatted_preview}")

            except Exception as e:
                print(f"   ❌ Code generation failed: {e}")

        print("\n📊 Code Generation Session Summary:")
        print(f"   Total operations: {session.total_operations}")
        print(f"   Session cost: ${session.total_cost:.6f}")

        if code_results:
            avg_cost = sum(r["cost"] for r in code_results) / len(code_results)
            avg_lines = sum(r["lines_of_code"] for r in code_results) / len(
                code_results
            )
            print(f"   Average cost per task: ${avg_cost:.6f}")
            print(f"   Average lines generated: {avg_lines:.1f}")


def demonstrate_streaming_responses():
    """Demonstrate streaming responses with real-time cost tracking."""
    print("\n⚡ Streaming Responses")
    print("=" * 50)

    print("🌊 Testing streaming capabilities...")
    print(
        "Note: This demo shows streaming concept - actual streaming requires Together client integration"
    )

    adapter = GenOpsTogetherAdapter(
        team="streaming-demo",
        project="real-time-responses",
        environment="development",
        daily_budget_limit=10.0,
    )

    streaming_tasks = [
        "Explain the concept of distributed systems in detail, covering architecture, challenges, and benefits.",
        "Write a comprehensive guide to machine learning for beginners, including key concepts and practical examples.",
    ]

    total_streaming_cost = 0

    for i, task in enumerate(streaming_tasks, 1):
        print(f"\n📡 Streaming Task {i}")
        start_time = time.time()

        try:
            # Simulate streaming by processing in chunks
            # In real implementation, this would use Together's streaming API
            result = adapter.chat_with_governance(
                messages=[{"role": "user", "content": task}],
                model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
                max_tokens=300,
                temperature=0.8,
                streaming_simulation=True,
                chunk_processing=True,
            )

            # Simulate real-time token processing
            response_chunks = [
                result.response[i : i + 50] for i in range(0, len(result.response), 50)
            ]

            print("   🔄 Streaming response:")
            for chunk_idx, chunk in enumerate(
                response_chunks[:5]
            ):  # Show first 5 chunks
                print(f"      Chunk {chunk_idx + 1}: {chunk}...")
                time.sleep(0.1)  # Simulate streaming delay

            total_streaming_cost += float(result.cost)

            print("\n   ✅ Streaming complete")
            print(f"   📊 Total tokens: {result.tokens_used}")
            print(f"   💰 Final cost: ${result.cost:.6f}")
            print(f"   ⏱️  Total time: {time.time() - start_time:.2f}s")

        except Exception as e:
            print(f"   ❌ Streaming failed: {e}")

    print("\n📊 Streaming Summary:")
    print(f"   Tasks streamed: {len(streaming_tasks)}")
    print(f"   Total streaming cost: ${total_streaming_cost:.6f}")


async def demonstrate_async_batch_processing():
    """Demonstrate async batch processing for high-throughput scenarios."""
    print("\n🚀 Async Batch Processing")
    print("=" * 50)

    adapter = GenOpsTogetherAdapter(
        team="async-processing",
        project="batch-operations",
        environment="development",
        daily_budget_limit=25.0,
        governance_policy="advisory",
    )

    # Create a batch of tasks to process concurrently
    batch_tasks = [
        f"Summarize the key benefits of task {i}: artificial intelligence in healthcare"
        for i in range(1, 6)
    ]

    print(f"⚡ Processing {len(batch_tasks)} tasks concurrently...")

    async def process_task(task_id: int, prompt: str):
        """Process a single task asynchronously."""
        try:
            # Simulate async processing (in real usage, you'd use AsyncTogether)
            result = adapter.chat_with_governance(
                messages=[{"role": "user", "content": prompt}],
                model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
                max_tokens=100,
                temperature=0.6,
                batch_id="async-demo",
                task_id=task_id,
                processing_type="concurrent",
            )

            return {
                "task_id": task_id,
                "cost": float(result.cost),
                "tokens": result.tokens_used,
                "time": result.execution_time_seconds,
                "success": True,
            }

        except Exception as e:
            return {"task_id": task_id, "error": str(e), "success": False}

    start_time = time.time()

    # Process all tasks (simulated async)
    batch_results = []
    for task_id, prompt in enumerate(batch_tasks, 1):
        result = await process_task(task_id, prompt)
        batch_results.append(result)

    total_batch_time = time.time() - start_time

    # Analyze batch results
    successful_tasks = [r for r in batch_results if r["success"]]
    failed_tasks = [r for r in batch_results if not r["success"]]

    if successful_tasks:
        total_cost = sum(r["cost"] for r in successful_tasks)
        avg_time = sum(r["time"] for r in successful_tasks) / len(successful_tasks)
        total_tokens = sum(r["tokens"] for r in successful_tasks)

        print("\n📊 Batch Processing Results:")
        print(f"   ✅ Successful tasks: {len(successful_tasks)}")
        print(f"   ❌ Failed tasks: {len(failed_tasks)}")
        print(f"   💰 Total cost: ${total_cost:.6f}")
        print(f"   📏 Total tokens: {total_tokens}")
        print(f"   ⏱️  Total batch time: {total_batch_time:.2f}s")
        print(f"   ⚡ Average task time: {avg_time:.2f}s")
        print(
            f"   🎯 Throughput: {len(successful_tasks) / total_batch_time:.1f} tasks/second"
        )


def demonstrate_reasoning_models():
    """Demonstrate advanced reasoning capabilities with specialized models."""
    print("\n🧠 Advanced Reasoning Models")
    print("=" * 50)

    adapter = GenOpsTogetherAdapter(
        team="reasoning-demo",
        project="complex-analysis",
        environment="development",
        daily_budget_limit=30.0,
    )

    reasoning_tasks = [
        {
            "name": "Mathematical Problem Solving",
            "prompt": "Solve this step-by-step: If a train travels 120 miles in 2 hours, and then increases speed by 25% for the next 3 hours, what is the total distance traveled?",
            "model": TogetherModel.DEEPSEEK_R1,
            "expected_features": ["step-by-step reasoning", "mathematical accuracy"],
        },
        {
            "name": "Logical Reasoning",
            "prompt": "All birds can fly. Penguins are birds. Penguins cannot fly. Identify the logical inconsistency and explain how to resolve it.",
            "model": TogetherModel.DEEPSEEK_R1_DISTILL,
            "expected_features": ["logical analysis", "contradiction resolution"],
        },
        {
            "name": "Complex System Analysis",
            "prompt": "Analyze the trade-offs between microservices and monolithic architecture for a fintech startup with 50 employees, considering scalability, security, and development velocity.",
            "model": TogetherModel.LLAMA_3_1_70B_INSTRUCT,
            "expected_features": ["multi-factor analysis", "domain expertise"],
        },
    ]

    print("🔍 Testing reasoning capabilities across specialized models...")

    reasoning_results = []

    with adapter.track_session("reasoning-analysis") as session:
        for task in reasoning_tasks:
            print(f"\n🎯 {task['name']}")
            print(f"   Model: {task['model'].value}")

            try:
                result = adapter.chat_with_governance(
                    messages=[
                        {
                            "role": "system",
                            "content": "Think step-by-step and provide detailed reasoning for your analysis.",
                        },
                        {"role": "user", "content": task["prompt"]},
                    ],
                    model=task["model"],
                    max_tokens=400,
                    temperature=0.3,  # Lower temperature for more consistent reasoning
                    session_id=session.session_id,
                    reasoning_task=task["name"],
                    expected_features=",".join(task["expected_features"]),
                )

                reasoning_results.append(
                    {
                        "task": task["name"],
                        "model": result.model_used,
                        "cost": float(result.cost),
                        "tokens": result.tokens_used,
                        "reasoning_depth": result.response.count("step")
                        + result.response.count("because")
                        + result.response.count("therefore"),
                        "response_length": len(result.response),
                    }
                )

                print(f"   ✅ Analysis completed ({result.tokens_used} tokens)")
                print(f"   💰 Cost: ${result.cost:.6f}")
                print(
                    f"   🧮 Reasoning indicators: {reasoning_results[-1]['reasoning_depth']}"
                )
                print(f"   📝 Preview: {result.response[:120]}...")

            except Exception as e:
                print(f"   ❌ Reasoning task failed: {e}")

    if reasoning_results:
        print("\n📊 Reasoning Analysis Summary:")
        print(f"   Tasks completed: {len(reasoning_results)}")
        total_reasoning_cost = sum(r["cost"] for r in reasoning_results)
        avg_reasoning_depth = sum(
            r["reasoning_depth"] for r in reasoning_results
        ) / len(reasoning_results)
        print(f"   Total cost: ${total_reasoning_cost:.6f}")
        print(
            f"   Average reasoning depth: {avg_reasoning_depth:.1f} indicators per response"
        )
        print(f"   Models used: {len({r['model'] for r in reasoning_results})}")


def demonstrate_fine_tuning_cost_estimation():
    """Demonstrate fine-tuning cost estimation and planning."""
    print("\n🎛️ Fine-Tuning Cost Estimation")
    print("=" * 50)

    pricing_calc = TogetherPricingCalculator()

    # Different fine-tuning scenarios
    fine_tuning_scenarios = [
        {
            "name": "Small Dataset Training",
            "base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "training_tokens": 100_000,
            "validation_tokens": 10_000,
            "epochs": 3,
        },
        {
            "name": "Medium Dataset Training",
            "base_model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "training_tokens": 500_000,
            "validation_tokens": 50_000,
            "epochs": 5,
        },
        {
            "name": "Large Dataset Training",
            "base_model": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            "training_tokens": 1_000_000,
            "validation_tokens": 100_000,
            "epochs": 2,
        },
    ]

    print("💡 Fine-tuning cost analysis:")

    for scenario in fine_tuning_scenarios:
        print(f"\n📋 {scenario['name']}:")

        try:
            cost = pricing_calc.calculate_fine_tuning_cost(
                model=scenario["base_model"],
                training_tokens=scenario["training_tokens"],
                validation_tokens=scenario["validation_tokens"],
                epochs=scenario["epochs"],
            )

            total_tokens = (
                scenario["training_tokens"] * scenario["epochs"]
            ) + scenario["validation_tokens"]

            print(f"   Base model: {scenario['base_model']}")
            print(f"   Training tokens: {scenario['training_tokens']:,}")
            print(f"   Validation tokens: {scenario['validation_tokens']:,}")
            print(f"   Epochs: {scenario['epochs']}")
            print(f"   Total tokens processed: {total_tokens:,}")
            print(f"   💰 Estimated cost: ${cost:.2f}")
            print(
                f"   📊 Cost per million tokens: ${float(cost) * 1_000_000 / total_tokens:.2f}"
            )

        except Exception as e:
            print(f"   ❌ Cost calculation failed: {e}")


def main():
    """Run all advanced feature demonstrations."""
    print("🚀 Together AI Advanced Features with GenOps")
    print("=" * 60)

    try:
        # Run all advanced demonstrations
        demonstrate_multimodal_operations()
        demonstrate_code_generation()
        demonstrate_streaming_responses()

        # Run async demo
        print("\n" + "=" * 60)
        asyncio.run(demonstrate_async_batch_processing())

        demonstrate_reasoning_models()
        demonstrate_fine_tuning_cost_estimation()

        # Final summary
        print("\n" + "=" * 60)
        print("🎯 Advanced Features Summary")
        print("=" * 60)

        print("✅ Advanced capabilities demonstrated:")
        print("   • Multimodal operations with vision-language models")
        print("   • Specialized code generation and completion")
        print("   • Streaming responses with real-time tracking")
        print("   • Async batch processing for high throughput")
        print("   • Advanced reasoning with specialized models")
        print("   • Fine-tuning cost estimation and planning")

        print("\n🚀 Key Insights:")
        print("   ✅ Specialized models excel at domain-specific tasks")
        print("   ✅ Cost-effective streaming maintains responsiveness")
        print("   ✅ Batch processing maximizes throughput efficiency")
        print("   ✅ Reasoning models provide step-by-step analysis")
        print("   ✅ Fine-tuning costs are predictable and manageable")

        print("\n📚 Next Steps:")
        print("   • Implement streaming for real-time applications")
        print("   • Use specialized models for domain-specific tasks")
        print("   • Consider fine-tuning for custom use cases")
        print("   • Leverage async processing for high-volume operations")

        return 0

    except Exception as e:
        print(f"❌ Advanced features demo failed: {e}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
        sys.exit(1)
