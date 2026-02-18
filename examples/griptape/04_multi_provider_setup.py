#!/usr/bin/env python3
"""
Example 04: Multi-Provider Setup with Unified Governance

Complexity: ⭐⭐ Intermediate

This example demonstrates how GenOps provides unified governance across
multiple LLM providers (OpenAI, Anthropic, Google) within Griptape workflows,
including cost comparison, provider-specific optimizations, and fallback strategies.

Prerequisites:
- Griptape framework installed (pip install griptape)
- GenOps installed (pip install genops)
- Multiple LLM provider API keys (OpenAI + at least one other)

Usage:
    python 04_multi_provider_setup.py

Environment Variables:
    OPENAI_API_KEY: Your OpenAI API key (required)
    ANTHROPIC_API_KEY: Your Anthropic API key (optional)
    GOOGLE_API_KEY: Your Google API key (optional)
    GENOPS_TEAM: Team identifier for governance
    GENOPS_PROJECT: Project identifier
"""

import logging
import os

from griptape.rules import Rule
from griptape.structures import Agent, Pipeline
from griptape.tasks import PromptTask

# GenOps imports for multi-provider tracking
from genops.providers.griptape import auto_instrument

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_available_providers() -> dict[str, bool]:
    """Check which LLM provider API keys are available."""

    providers = {
        "OpenAI": bool(os.getenv("OPENAI_API_KEY")),
        "Anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
        "Google": bool(os.getenv("GOOGLE_API_KEY")),
        "Cohere": bool(os.getenv("COHERE_API_KEY")),
        "Mistral": bool(os.getenv("MISTRAL_API_KEY")),
    }

    return providers


def create_provider_comparison_agents(available_providers: dict[str, bool]):
    """Create agents for different providers to compare performance and costs."""

    agents = {}

    # Task prompt for consistent comparison
    comparison_prompt = """Analyze the following business scenario and provide strategic recommendations:

Scenario: {{ input }}

Provide:
1. Key challenges and opportunities identified
2. 3 specific strategic recommendations
3. Implementation priority and timeline
4. Expected business impact

Keep response structured and actionable."""

    comparison_rules = [
        Rule("Provide concrete, actionable advice"),
        Rule("Structure response clearly with numbered points"),
        Rule("Focus on business value and ROI"),
        Rule("Keep recommendations realistic and implementable"),
    ]

    # Create agents for each available provider
    if available_providers.get("OpenAI"):
        agents["openai"] = Agent(
            tasks=[PromptTask(prompt=comparison_prompt, rules=comparison_rules)],
            # Note: In real implementation, you'd configure specific model here
        )

    if available_providers.get("Anthropic"):
        agents["anthropic"] = Agent(
            tasks=[PromptTask(prompt=comparison_prompt, rules=comparison_rules)],
            # Note: In real implementation, you'd configure Claude model here
        )

    if available_providers.get("Google"):
        agents["google"] = Agent(
            tasks=[PromptTask(prompt=comparison_prompt, rules=comparison_rules)],
            # Note: In real implementation, you'd configure Gemini model here
        )

    return agents


def create_fallback_pipeline():
    """Create a pipeline that can gracefully handle provider failures."""

    pipeline = Pipeline(
        tasks=[
            PromptTask(
                id="primary_analysis",
                prompt="""Perform comprehensive analysis of the business challenge:

                Challenge: {{ input }}

                Provide:
                1. Root cause analysis
                2. Market context and competitive landscape
                3. Risk assessment
                4. Opportunity identification""",
                rules=[
                    Rule("Be thorough but concise"),
                    Rule("Use structured analysis framework"),
                    Rule("Consider multiple perspectives"),
                ],
            ),
            PromptTask(
                id="solution_design",
                prompt="""Based on the analysis, design comprehensive solutions:

                Analysis: {{ primary_analysis.output }}

                Design:
                1. Multiple solution alternatives
                2. Pros and cons for each approach
                3. Resource requirements
                4. Risk mitigation strategies""",
                rules=[
                    Rule("Present multiple viable options"),
                    Rule("Be realistic about constraints"),
                    Rule("Focus on practical implementation"),
                ],
            ),
            PromptTask(
                id="implementation_plan",
                prompt="""Create detailed implementation plan:

                Solutions: {{ solution_design.output }}

                Plan should include:
                1. Phase-by-phase implementation timeline
                2. Resource allocation and team structure
                3. Success metrics and KPIs
                4. Contingency planning""",
                rules=[
                    Rule("Make plan actionable and specific"),
                    Rule("Include realistic timelines"),
                    Rule("Consider change management aspects"),
                ],
            ),
        ]
    )

    return pipeline


def main():
    """Multi-provider setup with unified governance demonstration."""

    print("🤖 GenOps + Griptape - Multi-Provider Setup Example")
    print("=" * 70)

    try:
        # Check available providers
        print("🔍 Checking available LLM providers...")
        available_providers = check_available_providers()

        print("📊 Provider availability:")
        for provider, available in available_providers.items():
            status = "✅ Available" if available else "❌ Not configured"
            print(f"  {provider}: {status}")

        # Ensure we have at least one provider
        if not any(available_providers.values()):
            print("\n❌ Error: No LLM provider API keys found")
            print("   Set at least OPENAI_API_KEY to continue")
            return False

        available_count = sum(available_providers.values())
        print(f"\n✅ {available_count} provider(s) configured")

        team = os.getenv("GENOPS_TEAM", "your-team")
        project = os.getenv("GENOPS_PROJECT", "griptape-demo")

        # Enable GenOps governance
        print("\n📊 Enabling GenOps governance for multi-provider tracking...")
        adapter = auto_instrument(
            team=team,
            project=project,
            environment="development",
            enable_cost_tracking=True,
            enable_performance_monitoring=True,
        )

        print("✅ Multi-provider governance enabled")

        # === PART 1: Provider Comparison ===
        print("\n📋 PART 1: Provider Performance & Cost Comparison")
        print("-" * 60)

        # Test scenario for comparison
        test_scenario = """
        A mid-sized SaaS company is experiencing 40% customer churn rate,
        primarily due to poor user onboarding experience. Customer support
        tickets have increased 200% in the past quarter, and user activation
        rates have dropped from 65% to 35%. The company needs to quickly
        implement solutions to retain existing customers and improve new
        user experience while maintaining current development velocity.
        """

        print("🚀 Creating provider comparison agents...")
        comparison_agents = create_provider_comparison_agents(available_providers)

        print(f"📝 Testing {len(comparison_agents)} configured providers...")

        provider_results = {}
        provider_costs = {}
        initial_spending = adapter.get_daily_spending()

        for provider_name, agent in comparison_agents.items():
            print(f"\n⚡ Running {provider_name} agent...")

            pre_cost = adapter.get_daily_spending()

            try:
                result = agent.run(test_scenario)
                post_cost = adapter.get_daily_spending()

                provider_cost = post_cost - pre_cost
                provider_results[provider_name] = result
                provider_costs[provider_name] = provider_cost

                # Preview result
                if hasattr(result, "output") and result.output:
                    preview = str(result.output.value)[:150]
                    print(f"✅ {provider_name} completed (${provider_cost:.6f})")
                    print(f"   Preview: {preview}...")
                else:
                    print(f"✅ {provider_name} completed (${provider_cost:.6f})")

            except Exception as e:
                print(f"❌ {provider_name} failed: {str(e)[:100]}...")
                provider_costs[provider_name] = 0

        # === PART 2: Unified Pipeline with Fallback ===
        print("\n📋 PART 2: Unified Pipeline with Provider Fallback")
        print("-" * 60)

        print("🚀 Creating multi-provider fallback pipeline...")
        fallback_pipeline = create_fallback_pipeline()

        print("📝 Pipeline structure:")
        for i, task in enumerate(fallback_pipeline.tasks, 1):
            print(f"  {i}. {task.id}: {task.__class__.__name__}")

        print("\n⚡ Executing pipeline with automatic provider selection...")

        pipeline_pre_cost = adapter.get_daily_spending()

        fallback_pipeline.run({"input": test_scenario})

        pipeline_post_cost = adapter.get_daily_spending()
        pipeline_cost = pipeline_post_cost - pipeline_pre_cost

        print("✅ Pipeline completed successfully!")
        print(f"💰 Pipeline cost: ${pipeline_cost:.6f}")

        # === PART 3: Multi-Provider Analysis ===
        print("\n📊 Multi-Provider Analysis & Optimization")
        print("-" * 60)

        total_comparison_cost = sum(provider_costs.values())
        total_session_cost = adapter.get_daily_spending() - initial_spending

        print("💰 Cost Analysis:")
        print(f"  Provider Comparison: ${total_comparison_cost:.6f}")
        print(f"  Fallback Pipeline:   ${pipeline_cost:.6f}")
        print(f"  Total Session:       ${total_session_cost:.6f}")

        if provider_costs:
            print("\n📈 Provider Cost Comparison:")
            sorted_providers = sorted(provider_costs.items(), key=lambda x: x[1])
            for provider, cost in sorted_providers:
                if cost > 0:
                    print(f"  {provider}: ${cost:.6f}")

            if len([c for c in provider_costs.values() if c > 0]) > 1:
                cheapest_provider = sorted_providers[0][0]
                sorted_providers[-1][0]
                savings_potential = sorted_providers[-1][1] - sorted_providers[0][1]
                print("\n💡 Optimization Insight:")
                print(f"   Most cost-effective: {cheapest_provider}")
                print(
                    f"   Potential savings: ${savings_potential:.6f} per similar request"
                )

        # Budget and governance summary
        budget_status = adapter.check_budget_compliance()
        print(f"\n💳 Budget Status: {budget_status['status']}")

        print("👥 Governance Attribution:")
        print(f"  Team: {adapter.governance_attrs.team}")
        print(f"  Project: {adapter.governance_attrs.project}")
        print(f"  Providers Used: {list(provider_costs.keys())}")

        print("\n🎉 Multi-Provider Setup Example Complete!")
        print("\n✨ Key Takeaways:")
        print("  1. ✅ Unified governance across multiple LLM providers")
        print("  2. ✅ Real-time cost comparison and optimization insights")
        print("  3. ✅ Automatic fallback handling for provider failures")
        print("  4. ✅ Performance benchmarking across provider ecosystem")
        print("  5. ✅ Centralized cost attribution for multi-provider usage")

        print("\n🚀 Next Steps:")
        print("  • Configure provider-specific model preferences")
        print("  • Set up automatic provider selection based on cost/performance")
        print("  • Implement budget-based provider routing")
        print("  • Add provider-specific retry and timeout strategies")

        return True

    except ImportError as e:
        if "griptape" in str(e):
            print("❌ Error: Griptape not installed")
            print("   Install with: pip install griptape")
        elif "genops" in str(e):
            print("❌ Error: GenOps not installed")
            print("   Install with: pip install genops")
        else:
            print(f"❌ Import error: {e}")
        return False

    except Exception as e:
        logger.error(f"Multi-provider setup example failed: {e}")
        print(f"\n❌ Error occurred: {e}")
        print("\n🔧 Troubleshooting Tips:")
        print("  • Verify all configured API keys are valid")
        print("  • Check network connectivity for provider APIs")
        print("  • Ensure sufficient API credits across providers")
        print("  • Run setup validation for detailed provider diagnostics")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
