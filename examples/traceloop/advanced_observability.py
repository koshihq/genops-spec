#!/usr/bin/env python3
"""
Advanced OpenLLMetry Observability Patterns with GenOps

This example demonstrates advanced observability patterns using OpenLLMetry as the foundation,
enhanced with GenOps governance for enterprise-grade monitoring, cost intelligence, and
policy enforcement.

Features demonstrated:
- Hierarchical tracing with parent-child relationships
- Multi-provider observability with unified governance
- Advanced cost optimization strategies
- Custom metrics and business intelligence
- Integration with enterprise observability stacks

Usage:
    python advanced_observability.py

Prerequisites:
    pip install genops[traceloop]
    export OPENAI_API_KEY="your-openai-api-key"
    export ANTHROPIC_API_KEY="your-anthropic-api-key"  # Optional for multi-provider demo
"""

import asyncio
import os
from datetime import datetime


def advanced_hierarchical_tracing():
    """Demonstrate hierarchical tracing with parent-child relationships."""
    print("🔍 Advanced Hierarchical Tracing")
    print("=" * 35)

    try:
        import openai

        from genops.providers.traceloop import instrument_traceloop

        adapter = instrument_traceloop(
            team="observability-team",
            project="advanced-patterns",
            environment="production",
            enable_governance=True,
            daily_budget_limit=5.0,
        )

        client = openai.OpenAI()

        # Parent workflow with nested operations
        with adapter.track_operation(
            operation_type="complex_workflow",
            operation_name="document_analysis_pipeline",
            tags={"pipeline": "document_analysis", "version": "v2.1"},
        ) as parent_span:
            # Step 1: Document preprocessing
            with adapter.track_operation(
                operation_type="preprocessing",
                operation_name="extract_key_sections",
                parent_span=parent_span,
                tags={"step": "preprocessing"},
            ) as prep_span:
                client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "user",
                            "content": "Extract key sections from this document: [document content]",
                        }
                    ],
                    max_tokens=100,
                )
                prep_span.update_cost(0.002)
                print("   ✅ Document preprocessing completed")

            # Step 2: Content analysis
            with adapter.track_operation(
                operation_type="analysis",
                operation_name="analyze_content",
                parent_span=parent_span,
                tags={"step": "analysis", "model": "gpt-4"},
            ) as analysis_span:
                client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {
                            "role": "user",
                            "content": "Analyze the extracted content for key insights",
                        }
                    ],
                    max_tokens=150,
                )
                analysis_span.update_cost(0.008)
                print("   ✅ Content analysis completed")

            # Step 3: Summary generation
            with adapter.track_operation(
                operation_type="generation",
                operation_name="generate_summary",
                parent_span=parent_span,
                tags={"step": "summary", "output_format": "executive"},
            ) as summary_span:
                client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": "Generate executive summary"}
                    ],
                    max_tokens=80,
                )
                summary_span.update_cost(0.003)
                print("   ✅ Summary generation completed")

            # Calculate total pipeline cost
            total_cost = parent_span.get_metrics()["estimated_cost"]
            print(f"   💰 Total pipeline cost: ${total_cost:.6f}")

            # Add pipeline-level metadata
            parent_span.add_attributes(
                {
                    "pipeline.steps_completed": 3,
                    "pipeline.success": True,
                    "pipeline.total_cost": total_cost,
                    "business.document_type": "contract",
                    "business.client_tier": "enterprise",
                }
            )

        print("✅ Hierarchical tracing with governance context completed")
        return True

    except Exception as e:
        print(f"❌ Advanced tracing failed: {e}")
        return False


def multi_provider_unified_governance():
    """Demonstrate unified governance across multiple AI providers."""
    print("\n🌐 Multi-Provider Unified Governance")
    print("-" * 35)

    try:
        from genops.providers.traceloop import instrument_traceloop

        # Unified adapter for multi-provider governance
        adapter = instrument_traceloop(
            team="multi-provider-team",
            project="unified-governance",
            environment="production",
            enable_governance=True,
            max_operation_cost=0.05,  # $0.05 per operation limit
        )

        # OpenAI operation
        import openai

        openai_client = openai.OpenAI()

        with adapter.track_operation(
            operation_type="openai_completion",
            operation_name="openai_analysis",
            tags={"provider": "openai", "model": "gpt-3.5-turbo"},
        ) as openai_span:
            openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Analyze market trends"}],
                max_tokens=100,
            )
            openai_span.update_cost(0.004)
            print("   ✅ OpenAI analysis: $0.004")

        # Anthropic operation (if available)
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            try:
                import anthropic

                anthropic_client = anthropic.Anthropic()

                with adapter.track_operation(
                    operation_type="anthropic_completion",
                    operation_name="anthropic_analysis",
                    tags={"provider": "anthropic", "model": "claude-3-haiku"},
                ) as anthropic_span:
                    anthropic_client.messages.create(
                        model="claude-3-haiku-20240307",
                        messages=[{"role": "user", "content": "Analyze market trends"}],
                        max_tokens=100,
                    )
                    anthropic_span.update_cost(0.003)
                    print("   ✅ Anthropic analysis: $0.003")

            except ImportError:
                print("   ⚠️ Anthropic not available")

        # Unified governance metrics
        metrics = adapter.get_metrics()
        print(f"   💰 Total multi-provider cost: ${metrics['daily_usage']:.6f}")
        print("   🛡️ Unified policy enforcement across all providers")

        return True

    except Exception as e:
        print(f"❌ Multi-provider governance failed: {e}")
        return False


def custom_business_metrics():
    """Demonstrate custom business metrics and intelligence."""
    print("\n📊 Custom Business Metrics & Intelligence")
    print("-" * 40)

    try:
        import openai

        from genops.providers.traceloop import instrument_traceloop

        adapter = instrument_traceloop(
            team="business-intelligence",
            project="custom-metrics",
            environment="production",
        )

        client = openai.OpenAI()

        # Business-critical operation with custom metrics
        with adapter.track_operation(
            operation_type="customer_interaction",
            operation_name="support_ticket_analysis",
            tags={
                "customer_tier": "enterprise",
                "priority": "high",
                "department": "support",
            },
        ) as span:
            client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "user",
                        "content": "Analyze this support ticket and provide resolution steps",
                    }
                ],
                max_tokens=200,
            )

            # Custom business metrics
            span.add_attributes(
                {
                    "business.customer_tier": "enterprise",
                    "business.ticket_priority": "high",
                    "business.resolution_complexity": "medium",
                    "business.customer_satisfaction_predicted": 0.87,
                    "business.revenue_impact": 2500.00,
                    "efficiency.time_saved_minutes": 45,
                    "efficiency.agent_productivity_gain": 0.3,
                    "quality.response_accuracy": 0.92,
                    "quality.customer_sentiment": "positive",
                }
            )

            metrics = span.get_metrics()
            print(f"   ✅ Support ticket analysis: ${metrics['estimated_cost']:.6f}")
            print("   📈 Custom business metrics captured:")
            print("      • Customer tier: Enterprise")
            print("      • Predicted satisfaction: 87%")
            print("      • Revenue impact: $2,500")
            print("      • Time saved: 45 minutes")

        return True

    except Exception as e:
        print(f"❌ Custom metrics demo failed: {e}")
        return False


async def main():
    """Main execution function."""
    print("🔍 Advanced OpenLLMetry Observability + GenOps Demo")
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Check prerequisites
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found")
        return False

    success = True

    # Run advanced patterns
    if not advanced_hierarchical_tracing():
        success = False

    if success and not multi_provider_unified_governance():
        success = False

    if success and not custom_business_metrics():
        success = False

    if success:
        print("\n" + "🔍" * 50)
        print("🎉 Advanced Observability Demo Complete!")
        print("\n📊 Advanced Patterns Demonstrated:")
        print("   ✅ Hierarchical tracing with parent-child relationships")
        print("   ✅ Multi-provider unified governance")
        print("   ✅ Custom business metrics and intelligence")
        print("   ✅ Enterprise-grade cost attribution")

        print("\n🏢 Production Benefits:")
        print("   • Complete observability across complex workflows")
        print("   • Unified governance regardless of AI provider")
        print("   • Business intelligence integrated with technical metrics")
        print("   • Cost optimization across multi-step operations")

        print("🔍" * 50)

    return success


if __name__ == "__main__":
    asyncio.run(main())
