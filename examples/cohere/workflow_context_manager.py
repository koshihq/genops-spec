#!/usr/bin/env python3
"""
🔄 GenOps + Cohere: Workflow Context Manager Example

GOAL: Demonstrate advanced multi-operation workflow tracking with automatic cost aggregation
TIME: 15 minutes
WHAT YOU'LL LEARN: How to use context managers for complex Cohere workflows

This example shows how to use the GenOps workflow context manager for
intelligent document processing workflows that combine multiple Cohere operations.

Prerequisites:
- Cohere API key: export CO_API_KEY="your-key"
- GenOps: pip install genops-ai
- Cohere: pip install cohere
"""

import sys


def intelligent_document_workflow():
    """Demonstrate intelligent document processing workflow."""
    print("🔄 GenOps Cohere Workflow Context Manager Demo")
    print("=" * 60)

    try:
        from genops.providers.cohere import cohere_workflow_context

        # Sample documents to process
        documents = [
            "Machine learning revolutionizes medical diagnosis by analyzing vast datasets to identify patterns humans might miss.",
            "Artificial intelligence in healthcare enables personalized treatment plans based on patient-specific data analysis.",
            "Deep learning algorithms process medical images with accuracy that often exceeds human radiologist performance.",
            "Natural language processing helps extract insights from electronic health records for better patient care.",
            "Computer vision applications in medicine include automated analysis of X-rays, MRIs, and other diagnostic images."
        ]

        query = "AI applications in medical diagnosis and treatment"

        print(f"\n📋 Processing {len(documents)} documents about: '{query}'")

        # Execute intelligent workflow with automatic cost tracking
        with cohere_workflow_context(
            "intelligent_document_processing",
            team="ai-research",
            project="medical-ai-analysis",
            customer_id="healthcare-enterprise",
            environment="production"
        ) as (ctx, workflow_id):

            print(f"🚀 Starting workflow: {workflow_id}")

            # Step 1: Create query embedding for semantic similarity
            print("\n📊 Step 1: Creating query embedding...")
            query_embedding = ctx.embed(
                texts=[query],
                model="embed-english-v4.0",
                input_type="search_query"
            )

            if query_embedding.success:
                print(f"✅ Query embedding created: ${query_embedding.usage.total_cost:.6f}")
                print(f"   Vector dimensions: {len(query_embedding.embeddings[0])}")

            # Step 2: Create document embeddings
            print("\n📚 Step 2: Creating document embeddings...")
            doc_embeddings = ctx.embed(
                texts=documents,
                model="embed-english-v4.0",
                input_type="search_document"
            )

            if doc_embeddings.success:
                print(f"✅ Document embeddings created: ${doc_embeddings.usage.total_cost:.6f}")
                print(f"   Documents processed: {len(documents)}")
                print(f"   Cost per document: ${doc_embeddings.usage.total_cost / len(documents):.6f}")

            # Step 3: Rerank documents by relevance
            print("\n🔍 Step 3: Reranking documents by relevance...")
            rerank_result = ctx.rerank(
                query=query,
                documents=documents,
                model="rerank-english-v3.0",
                top_n=3
            )

            if rerank_result.success:
                print(f"✅ Document reranking completed: ${rerank_result.usage.total_cost:.6f}")
                print("   Top 3 most relevant documents:")
                for i, ranking in enumerate(rerank_result.rankings[:3]):
                    print(f"   {i+1}. Score: {ranking['relevance_score']:.3f}")
                    print(f"      Text: {ranking['document']['text'][:80]}...")

            # Step 4: Generate intelligent summary
            print("\n📝 Step 4: Generating intelligent summary...")
            top_docs = [r['document']['text'] for r in rerank_result.rankings[:3]]

            summary_prompt = f"""
            Based on these top medical AI documents about "{query}":
            
            {chr(10).join(f"{i+1}. {doc}" for i, doc in enumerate(top_docs))}
            
            Provide a concise executive summary highlighting key applications and benefits.
            """

            summary_result = ctx.chat(
                message=summary_prompt,
                model="command-r-08-2024",
                temperature=0.3,
                max_tokens=300
            )

            if summary_result.success:
                print(f"✅ Summary generated: ${summary_result.usage.total_cost:.6f}")
                print(f"   Response length: {len(summary_result.content)} characters")
                print(f"   Generation speed: {summary_result.usage.tokens_per_second:.1f} tokens/sec")

            # Step 5: Generate actionable insights
            print("\n💡 Step 5: Extracting actionable insights...")
            insights_result = ctx.chat(
                message=f"Based on the summary: '{summary_result.content[:200]}...', what are 3 specific actionable recommendations for healthcare organizations implementing AI?",
                model="command-light",  # Use faster model for simple task
                max_tokens=200
            )

            if insights_result.success:
                print(f"✅ Insights generated: ${insights_result.usage.total_cost:.6f}")

            # Display workflow results
            print("\n🎯 Workflow Results:")
            print(f"   Workflow ID: {workflow_id}")
            print(f"   Total Operations: {ctx.get_operation_count()}")
            print(f"   Total Cost: ${ctx.get_total_cost():.6f}")
            print(f"   Average Cost/Operation: ${ctx.get_total_cost() / ctx.get_operation_count():.6f}")

            # Cost breakdown by operation type
            cost_breakdown = ctx.get_cost_breakdown()
            print("\n💰 Cost Breakdown:")
            for operation, cost in cost_breakdown.items():
                percentage = (cost / ctx.get_total_cost()) * 100
                print(f"   {operation.title()}: ${cost:.6f} ({percentage:.1f}%)")

            # Display final outputs
            print("\n📋 Final Outputs:")
            print("   Executive Summary:")
            print(f"   {summary_result.content[:300]}...")

            print("\n   Key Insights:")
            print(f"   {insights_result.content[:300]}...")

        # Workflow automatically finalized with context manager
        print("\n✅ Workflow completed successfully!")
        print("🔧 All resources automatically cleaned up by context manager")

        return True

    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        return False


def cost_optimization_workflow():
    """Demonstrate cost optimization using workflow context."""
    print("\n" + "=" * 60)
    print("💰 Cost Optimization Workflow Example")
    print("=" * 60)

    try:
        from genops.providers.cohere import cohere_workflow_context

        # Compare different model strategies for the same task
        strategies = [
            {
                "name": "Premium Strategy",
                "chat_model": "command-r-plus-08-2024",
                "embed_model": "embed-english-v4.0",
                "rerank_model": "rerank-english-v3.0"
            },
            {
                "name": "Balanced Strategy",
                "chat_model": "command-r-08-2024",
                "embed_model": "embed-english-v4.0",
                "rerank_model": "rerank-english-v3.0"
            },
            {
                "name": "Cost-Effective Strategy",
                "chat_model": "command-light",
                "embed_model": "embed-english-v4.0",
                "rerank_model": "rerank-english-v3.0"
            }
        ]

        task = "Summarize key AI trends in healthcare"
        documents = ["AI diagnostic tools improve accuracy", "Machine learning predicts treatment outcomes"]

        strategy_results = []

        for strategy in strategies:
            print(f"\n🧪 Testing: {strategy['name']}")

            with cohere_workflow_context(
                f"cost_optimization_{strategy['name'].lower().replace(' ', '_')}",
                team="cost-optimization",
                project="model-comparison"
            ) as (ctx, workflow_id):

                # Execute same workflow with different models
                embed_result = ctx.embed(
                    texts=documents,
                    model=strategy['embed_model']
                )

                rerank_result = ctx.rerank(
                    query=task,
                    documents=documents,
                    model=strategy['rerank_model']
                )

                chat_result = ctx.chat(
                    message=f"Summarize: {' '.join(documents)}",
                    model=strategy['chat_model'],
                    max_tokens=100
                )

                # Collect results
                strategy_results.append({
                    'strategy': strategy['name'],
                    'total_cost': ctx.get_total_cost(),
                    'operations': ctx.get_operation_count(),
                    'cost_per_operation': ctx.get_total_cost() / ctx.get_operation_count(),
                    'breakdown': ctx.get_cost_breakdown(),
                    'quality_score': len(chat_result.content) if chat_result.success else 0  # Simple quality metric
                })

                print(f"   Total Cost: ${ctx.get_total_cost():.6f}")
                print(f"   Cost/Operation: ${ctx.get_total_cost() / ctx.get_operation_count():.6f}")

        # Compare strategies
        print("\n📊 Strategy Comparison:")
        print(f"{'Strategy':<20} {'Total Cost':<12} {'Cost/Op':<12} {'Quality':<10}")
        print("-" * 60)

        for result in strategy_results:
            print(f"{result['strategy']:<20} ${result['total_cost']:<11.6f} ${result['cost_per_operation']:<11.6f} {result['quality_score']:<10}")

        # Find best value
        best_value = min(strategy_results, key=lambda x: x['total_cost'] / max(x['quality_score'], 1))
        print(f"\n🏆 Best Value Strategy: {best_value['strategy']}")
        print(f"   Cost: ${best_value['total_cost']:.6f}")
        print(f"   Cost Efficiency: ${best_value['total_cost'] / max(best_value['quality_score'], 1):.6f} per quality unit")

        return True

    except Exception as e:
        print(f"❌ Cost optimization failed: {e}")
        return False


def error_handling_workflow():
    """Demonstrate error handling within workflow context."""
    print("\n" + "=" * 60)
    print("🛡️ Error Handling Workflow Example")
    print("=" * 60)

    try:
        from genops.providers.cohere import cohere_workflow_context

        print("\n🧪 Testing workflow with intentional errors...")

        with cohere_workflow_context(
            "error_handling_test",
            team="testing",
            project="error-scenarios"
        ) as (ctx, workflow_id):

            # Valid operation
            print("   ✅ Executing valid operation...")
            valid_result = ctx.chat(
                message="Test message",
                model="command-light"
            )

            print(f"   Valid operation cost: ${valid_result.usage.total_cost:.6f}")

            # Test with invalid model (should handle gracefully)
            print("   🧪 Testing invalid model handling...")
            try:
                invalid_result = ctx.chat(
                    message="Test with invalid model",
                    model="non-existent-model"
                )

                if not invalid_result.success:
                    print(f"   ✅ Error handled gracefully: {invalid_result.error_message[:50]}...")

            except Exception as e:
                print(f"   ✅ Exception caught by workflow: {str(e)[:50]}...")

            # Show partial results
            print("\n   Partial workflow results:")
            print(f"   Operations completed: {ctx.get_operation_count()}")
            print(f"   Total cost so far: ${ctx.get_total_cost():.6f}")

        print("✅ Workflow context manager handled errors gracefully")
        return True

    except Exception as e:
        print(f"✅ Expected error handled at workflow level: {e}")
        return True  # Expected behavior


def main():
    """Main demo function."""
    print("🚀 GenOps Cohere Workflow Context Manager Examples")
    print("=" * 60)

    # Check prerequisites
    try:
        from genops.providers.cohere_validation import quick_validate
        if not quick_validate():
            print("❌ Cohere setup validation failed")
            print("   Please ensure CO_API_KEY is set and cohere is installed")
            return False
    except ImportError:
        print("❌ GenOps not available")
        print("   Install with: pip install genops-ai")
        return False

    success_count = 0
    total_demos = 3

    # Run demonstrations
    demos = [
        ("Intelligent Document Workflow", intelligent_document_workflow),
        ("Cost Optimization Workflow", cost_optimization_workflow),
        ("Error Handling Workflow", error_handling_workflow)
    ]

    for name, demo_func in demos:
        print(f"\n🎯 Running: {name}")
        if demo_func():
            success_count += 1
            print(f"✅ {name} completed successfully")
        else:
            print(f"❌ {name} failed")

    # Summary
    print("\n" + "=" * 60)
    print(f"🎉 Demo Summary: {success_count}/{total_demos} workflows succeeded")
    print("=" * 60)

    if success_count == total_demos:
        print("✅ All workflow context manager examples completed successfully!")
        print("\n🚀 Key Benefits Demonstrated:")
        print("   • Automatic cost aggregation across multiple operations")
        print("   • Built-in error handling and recovery")
        print("   • OpenTelemetry span creation for observability")
        print("   • Resource cleanup and finalization")
        print("   • Cost optimization and model comparison")
        print("   • Enterprise governance integration")

        print("\n📚 Next Steps:")
        print("   • Use workflow context managers in your production code")
        print("   • Combine with cost aggregators for advanced analytics")
        print("   • Integrate with your observability stack via OpenTelemetry")
        print("   • Implement custom workflow patterns for your use cases")

        return True
    else:
        print("⚠️ Some examples failed - check your Cohere setup and API key")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
