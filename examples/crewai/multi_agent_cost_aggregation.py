#!/usr/bin/env python3
"""
Multi-Agent Cost Aggregation and Optimization

Advanced cost tracking and analysis across multiple AI providers for CrewAI agents.
Demonstrates cost optimization, provider comparison, and budget management.

Usage:
    python multi_agent_cost_aggregation.py [--budget AMOUNT] [--provider PROVIDER]

Features:
    - Multi-provider cost aggregation (OpenAI, Anthropic, Google, etc.)
    - Real-time cost optimization recommendations
    - Provider performance vs cost analysis
    - Budget-aware agent selection and model switching
    - Cost attribution by agent, task, and crew
    - Migration cost analysis for switching providers

Time to Complete: ~15 minutes
Learning Outcomes: Advanced cost management for multi-agent systems
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from typing import Optional

# Core CrewAI imports
try:
    from crewai import Agent, Crew, Task
    from crewai.process import Process
except ImportError:
    print("❌ CrewAI not installed. Install with: pip install crewai")
    sys.exit(1)

# GenOps imports
try:
    from genops.providers.crewai import (
        CrewAICostAggregator,
        GenOpsCrewAIAdapter,
        analyze_crew_costs,
        multi_provider_cost_tracking,  # noqa: F401
        print_validation_result,
        validate_crewai_setup,
    )
except ImportError:
    print("❌ GenOps not installed. Install with: pip install genops-ai[crewai]")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProviderComparison:
    """Cost and performance comparison between providers."""

    provider: str
    total_cost: float
    avg_cost_per_operation: float
    operations_count: int
    avg_response_time: float
    quality_score: float  # Simulated quality metric
    cost_efficiency: float  # Cost per quality point


class MultiProviderCostDemo:
    """Demonstration of multi-provider cost tracking and optimization."""

    def __init__(
        self, budget_limit: float = 50.0, preferred_provider: Optional[str] = None
    ):
        self.budget_limit = budget_limit
        self.preferred_provider = preferred_provider
        self.adapter = GenOpsCrewAIAdapter(
            team="cost-optimization",
            project="multi-provider-demo",
            daily_budget_limit=budget_limit,
            enable_cost_tracking=True,
            governance_policy="advisory",
        )
        self.cost_aggregator = CrewAICostAggregator()

    def setup_validation(self) -> bool:
        """Validate setup for multi-provider cost tracking."""
        print("🔍 Validating multi-provider cost tracking setup...")

        result = validate_crewai_setup(quick=False)

        if result.is_valid:
            print("✅ Multi-provider setup validated")
            return True
        else:
            print("❌ Setup issues found:")
            print_validation_result(result)
            return False

    def create_diverse_crew(self, use_case: str) -> Crew:
        """Create a crew with agents that could use different providers."""
        print(f"\n🏗️ Creating diverse crew for {use_case}...")

        # Research agent (could use GPT-4 for deep analysis)
        researcher = Agent(
            role="Senior Research Analyst",
            goal="Conduct comprehensive research with high accuracy",
            backstory="""Expert researcher with access to vast knowledge bases.
                         Specializes in thorough analysis requiring advanced reasoning.""",
            verbose=True,
        )

        # Writing agent (could use Claude for creative writing)
        writer = Agent(
            role="Content Creator",
            goal="Transform research into engaging, accessible content",
            backstory="""Creative content specialist with expertise in making
                         complex topics understandable and compelling.""",
            verbose=True,
        )

        # Analyst agent (could use Gemini for data analysis)
        analyst = Agent(
            role="Data Analyst",
            goal="Extract insights and patterns from research data",
            backstory="""Analytical expert specializing in finding trends,
                         patterns, and actionable insights from complex data.""",
            verbose=True,
        )

        # Editor agent (could use cheaper model for final review)
        editor = Agent(
            role="Quality Editor",
            goal="Ensure accuracy, clarity, and consistency",
            backstory="""Experienced editor focused on quality assurance,
                         fact-checking, and content optimization.""",
            verbose=True,
        )

        # Define tasks with different complexity levels
        tasks = [
            Task(
                description=f"""Research the latest developments in {use_case}.
                              Focus on breakthrough innovations, market trends,
                              and future implications. Provide detailed analysis
                              with citations and evidence.""",
                agent=researcher,
            ),
            Task(
                description=f"""Create an engaging article about {use_case}
                              developments. Make it accessible to general audiences
                              while maintaining technical accuracy. Include
                              compelling examples and future predictions.""",
                agent=writer,
            ),
            Task(
                description=f"""Analyze the research data to identify key trends,
                              success patterns, and market opportunities in {use_case}.
                              Provide quantitative insights and recommendations.""",
                agent=analyst,
            ),
            Task(
                description="""Review and edit all content for accuracy, consistency,
                              and clarity. Ensure proper structure, flow, and
                              professional presentation standards.""",
                agent=editor,
            ),
        ]

        crew = Crew(
            agents=[researcher, writer, analyst, editor],
            tasks=tasks,
            process=Process.sequential,
            verbose=2,
        )

        print(f"✅ Created crew with {len(crew.agents)} agents for {use_case}")
        return crew

    def demonstrate_cost_tracking(self):
        """Demonstrate comprehensive cost tracking across multiple scenarios."""
        print("\n" + "=" * 70)
        print("📊 Multi-Provider Cost Tracking Demonstration")
        print("=" * 70)

        scenarios = [
            ("AI Safety Research", "artificial intelligence safety and alignment"),
            ("Climate Technology", "climate change mitigation technologies"),
            ("Healthcare Innovation", "digital health and medical AI applications"),
        ]

        scenario_results = []

        for i, (scenario_name, scenario_topic) in enumerate(scenarios, 1):
            print(f"\n🎬 Scenario {i}: {scenario_name}")
            print(f"   Topic: {scenario_topic}")

            crew = self.create_diverse_crew(scenario_topic)

            # Track with detailed cost attribution
            with self.adapter.track_crew(
                f"{scenario_name.lower().replace(' ', '-')}-crew",
                use_case=scenario_name,
                complexity_level="high",
            ) as context:
                print(f"\n🚀 Starting {scenario_name} crew execution...")
                start_time = time.time()

                # Execute crew
                result = crew.kickoff(
                    {
                        "focus_area": scenario_topic,
                        "target_length": "comprehensive analysis",
                        "audience": "technical professionals",
                    }
                )

                end_time = time.time()
                execution_time = end_time - start_time

                # Add custom metrics
                context.add_custom_metric("scenario", scenario_name)
                context.add_custom_metric("execution_time", execution_time)
                context.add_custom_metric("agents_count", len(crew.agents))
                context.add_custom_metric("tasks_count", len(crew.tasks))

                # Get metrics
                metrics = context.get_metrics()
                scenario_results.append(
                    {
                        "scenario": scenario_name,
                        "cost": metrics["total_cost"],
                        "time": execution_time,
                        "agents": len(crew.agents),
                        "result_length": len(str(result)),
                        "cost_per_agent": metrics["total_cost"] / len(crew.agents),
                        "providers_used": metrics.get("cost_by_provider", {}),
                    }
                )

                print(f"\n📊 {scenario_name} Results:")
                print(f"   💰 Total cost: ${metrics['total_cost']:.6f}")
                print(f"   ⏱️ Execution time: {execution_time:.2f} seconds")
                print(f"   👥 Agents: {len(crew.agents)}")
                print(
                    f"   💲 Cost per agent: ${metrics['total_cost'] / len(crew.agents):.6f}"
                )

                if metrics.get("cost_by_provider"):
                    print("   🏢 Providers used:")
                    for provider, cost in metrics["cost_by_provider"].items():
                        print(f"      • {provider}: ${cost:.6f}")

        return scenario_results

    def analyze_cost_optimization(self, scenario_results: list[dict]) -> dict:
        """Analyze cost optimization opportunities across scenarios."""
        print("\n" + "=" * 70)
        print("🔍 Cost Optimization Analysis")
        print("=" * 70)

        # Get comprehensive cost analysis
        analysis = analyze_crew_costs(self.adapter, time_period_hours=1)

        if "error" in analysis:
            print(f"❌ Cost analysis unavailable: {analysis['error']}")
            return {}

        print("\n📈 Overall Cost Analysis:")
        print(f"   💰 Total cost across all scenarios: ${analysis['total_cost']:.6f}")
        print(f"   🏢 Providers used: {len(analysis['cost_by_provider'])}")
        print(f"   👥 Unique agents: {len(analysis['cost_by_agent'])}")

        # Cost by provider analysis
        if analysis["cost_by_provider"]:
            print("\n💳 Cost by Provider:")
            sorted_providers = sorted(
                analysis["cost_by_provider"].items(), key=lambda x: x[1], reverse=True
            )
            for provider, cost in sorted_providers:
                percentage = (cost / analysis["total_cost"]) * 100
                print(f"   • {provider}: ${cost:.6f} ({percentage:.1f}%)")

        # Most expensive agent
        if analysis["most_expensive_agent"]:
            print(f"\n💸 Most expensive agent: {analysis['most_expensive_agent']}")

        # Optimization recommendations
        if analysis["recommendations"]:
            print("\n💡 Cost Optimization Recommendations:")
            for i, rec in enumerate(analysis["recommendations"], 1):
                savings_pct = (rec["potential_savings"] / analysis["total_cost"]) * 100
                print(f"   {i}. {rec['agent']}:")
                print(f"      • Current: {rec['current_provider']}")
                print(f"      • Recommended: {rec['recommended_provider']}")
                print(
                    f"      • Potential savings: ${rec['potential_savings']:.6f} ({savings_pct:.1f}%)"
                )
                print(f"      • Reasoning: {rec['reasoning']}")

        # Provider performance analysis
        if analysis["provider_summaries"]:
            print("\n📊 Provider Performance Analysis:")
            for provider, summary in analysis["provider_summaries"].items():
                efficiency = (
                    summary["total_cost"] / summary["total_operations"]
                    if summary["total_operations"] > 0
                    else 0
                )
                print(f"   • {provider}:")
                print(f"     - Total cost: ${summary['total_cost']:.6f}")
                print(f"     - Operations: {summary['total_operations']}")
                print(f"     - Cost per operation: ${efficiency:.6f}")
                print(f"     - Agents used: {len(summary['agents_used'])}")
                print(
                    f"     - Models used: {', '.join(summary['models_used']) if summary['models_used'] else 'N/A'}"
                )

        return analysis

    def demonstrate_budget_management(self):
        """Demonstrate budget-constrained operations and controls."""
        print("\n" + "=" * 70)
        print("💳 Budget Management & Controls")
        print("=" * 70)

        # Create budget-constrained adapter
        budget_adapter = GenOpsCrewAIAdapter(
            team="budget-demo",
            project="cost-control",
            daily_budget_limit=5.0,  # Low budget for demonstration
            governance_policy="enforced",  # Strict enforcement
            enable_cost_tracking=True,
        )

        print("📊 Budget Settings:")
        print(f"   💰 Daily budget limit: ${budget_adapter.daily_budget_limit}")
        print(f"   🚨 Policy: {budget_adapter.governance_policy}")

        # Create simple crew for budget testing
        budget_agent = Agent(
            role="Budget-Conscious Analyst",
            goal="Provide valuable insights within budget constraints",
            backstory="Expert at delivering maximum value with minimal resource usage",
        )

        budget_task = Task(
            description="""Provide a concise analysis of renewable energy trends.
                           Focus on key insights that provide maximum value.""",
            agent=budget_agent,
        )

        budget_crew = Crew(agents=[budget_agent], tasks=[budget_task], verbose=True)

        # Track budget usage
        try:
            with budget_adapter.track_crew(
                "budget-test", budget_conscious=True
            ) as context:
                print("\n🎬 Executing budget-constrained crew...")

                budget_crew.kickoff({"efficiency_mode": True, "budget_limit": 5.0})

                metrics = context.get_metrics()
                remaining_budget = (
                    budget_adapter.daily_budget_limit - metrics["total_cost"]
                )

                print("\n📊 Budget Usage Results:")
                print(f"   💰 Cost: ${metrics['total_cost']:.6f}")
                print(f"   💳 Budget limit: ${budget_adapter.daily_budget_limit}")
                print(f"   💰 Remaining: ${remaining_budget:.6f}")
                print(
                    f"   📈 Usage: {(metrics['total_cost'] / budget_adapter.daily_budget_limit) * 100:.1f}%"
                )

                if remaining_budget > 0:
                    print("   ✅ Within budget constraints")
                else:
                    print("   ⚠️ Budget limit reached")

        except Exception as e:
            print(f"❌ Budget enforcement triggered: {e}")
            print("   This demonstrates budget control in action!")

    def generate_cost_report(self, scenario_results: list[dict], analysis: dict):
        """Generate a comprehensive cost analysis report."""
        print("\n" + "=" * 70)
        print("📄 Comprehensive Cost Analysis Report")
        print("=" * 70)

        total_scenarios = len(scenario_results)
        total_cost = sum(result["cost"] for result in scenario_results)
        total_time = sum(result["time"] for result in scenario_results)
        total_agents = sum(result["agents"] for result in scenario_results)

        print("\n📊 Executive Summary:")
        print(f"   🎯 Scenarios analyzed: {total_scenarios}")
        print(f"   💰 Total cost: ${total_cost:.6f}")
        print(f"   ⏱️ Total execution time: {total_time:.2f} seconds")
        print(f"   👥 Total agent-tasks: {total_agents}")
        print(f"   💲 Average cost per scenario: ${total_cost / total_scenarios:.6f}")
        print(f"   ⚡ Average cost per second: ${total_cost / total_time:.6f}")

        # Scenario comparison
        print("\n🔍 Scenario Performance Comparison:")
        sorted_scenarios = sorted(scenario_results, key=lambda x: x["cost"])

        for result in sorted_scenarios:
            efficiency = result["cost"] / result["time"] if result["time"] > 0 else 0
            print(f"   • {result['scenario']}:")
            print(f"     - Cost: ${result['cost']:.6f}")
            print(f"     - Time: {result['time']:.2f}s")
            print(f"     - Efficiency: ${efficiency:.6f}/second")
            print(f"     - Cost per agent: ${result['cost_per_agent']:.6f}")

        # Recommendations
        print("\n💡 Cost Optimization Recommendations:")

        # Find most/least efficient scenarios
        most_efficient = min(scenario_results, key=lambda x: x["cost"] / x["time"])
        least_efficient = max(scenario_results, key=lambda x: x["cost"] / x["time"])

        print(f"   1. Most efficient scenario: {most_efficient['scenario']}")
        print(
            f"      - Cost efficiency: ${(most_efficient['cost'] / most_efficient['time']):.6f}/second"
        )
        print("      - Consider replicating this pattern for similar tasks")

        print(f"   2. Least efficient scenario: {least_efficient['scenario']}")
        print(
            f"      - Cost efficiency: ${(least_efficient['cost'] / least_efficient['time']):.6f}/second"
        )
        print("      - Investigate optimization opportunities")

        if analysis and "recommendations" in analysis:
            print("   3. Provider optimization potential:")
            total_savings = sum(
                rec["potential_savings"] for rec in analysis["recommendations"]
            )
            if total_savings > 0:
                savings_pct = (total_savings / analysis["total_cost"]) * 100
                print(
                    f"      - Potential savings: ${total_savings:.6f} ({savings_pct:.1f}%)"
                )
                print(
                    "      - Primary recommendation: Switch high-cost agents to optimal providers"
                )
            else:
                print("      - Current provider selection appears optimal")

        # Future predictions
        print("\n🔮 Future Cost Projections:")
        daily_rate = total_cost * (24 * 3600) / total_time if total_time > 0 else 0
        monthly_rate = daily_rate * 30

        print("   • If run continuously:")
        print(f"     - Daily cost: ${daily_rate:.2f}")
        print(f"     - Monthly cost: ${monthly_rate:.2f}")
        print("   • Budget planning recommendations:")

        if monthly_rate > 1000:
            print("     - Consider enterprise pricing tiers")
            print("     - Implement aggressive cost optimization")
        elif monthly_rate > 100:
            print("     - Monitor usage patterns closely")
            print("     - Set up budget alerts")
        else:
            print("     - Current usage appears cost-effective")

        return {
            "total_cost": total_cost,
            "scenarios": total_scenarios,
            "efficiency_leader": most_efficient["scenario"],
            "optimization_potential": total_savings if analysis else 0,
            "monthly_projection": monthly_rate,
        }


def main():
    """Run the comprehensive multi-provider cost aggregation demonstration."""
    parser = argparse.ArgumentParser(description="Multi-Provider Cost Aggregation Demo")
    parser.add_argument(
        "--budget",
        type=float,
        default=50.0,
        help="Daily budget limit in USD (default: 50.0)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        help="Preferred provider (openai, anthropic, google, etc.)",
    )
    args = parser.parse_args()

    print("💰 Multi-Agent Cost Aggregation and Optimization")
    print("=" * 60)
    print(f"Budget limit: ${args.budget}")
    if args.provider:
        print(f"Preferred provider: {args.provider}")

    # Initialize demo
    demo = MultiProviderCostDemo(
        budget_limit=args.budget, preferred_provider=args.provider
    )

    # Validate setup
    if not demo.setup_validation():
        print("\n❌ Please fix setup issues before proceeding")
        return 1

    try:
        # Run cost tracking demonstrations
        scenario_results = demo.demonstrate_cost_tracking()

        # Analyze optimization opportunities
        analysis = demo.analyze_cost_optimization(scenario_results)

        # Demonstrate budget controls
        demo.demonstrate_budget_management()

        # Generate comprehensive report
        demo.generate_cost_report(scenario_results, analysis)

        print("\n🎉 Multi-Provider Cost Analysis Complete!")
        print("\n🚀 Next Steps:")
        print("   • Review cost optimization recommendations")
        print("   • Implement budget controls for production usage")
        print("   • Try performance_optimization.py for speed improvements")
        print("   • Explore agent_workflow_governance.py for advanced monitoring")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️ Cost analysis interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Cost analysis failed: {e}", exc_info=True)
        print(f"\n❌ Demo failed: {e}")
        print("Try running setup_validation.py to check your configuration")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
