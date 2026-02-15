#!/usr/bin/env python3
"""
Basic CrewAI Crew Tracking with GenOps

Demonstrates simple crew execution with governance telemetry.
Perfect for getting started with GenOps CrewAI integration.

Usage:
    python basic_crew_tracking.py

Features:
    - Zero-code auto-instrumentation
    - Manual crew tracking
    - Basic cost attribution
    - Performance monitoring
    - Agent execution metrics
"""

import logging
import sys
import time

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
        GenOpsCrewAIAdapter,
        auto_instrument,
        print_validation_result,
        validate_crewai_setup,
    )
except ImportError:
    print("❌ GenOps not installed. Install with: pip install genops-ai[crewai]")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_environment() -> bool:
    """Verify required environment variables are set."""
    print("🔍 Checking environment setup...")

    # Run validation
    result = validate_crewai_setup(quick=True)

    if result.is_valid:
        print("✅ Environment setup validated")
        return True
    else:
        print("❌ Environment setup issues found:")
        print_validation_result(result)
        return False


def create_research_crew() -> Crew:
    """Create a simple research crew for demonstration."""
    print("\n🏗️ Creating research crew...")

    # Define research agent
    researcher = Agent(
        role="Senior Research Analyst",
        goal="Uncover cutting-edge developments in AI and machine learning",
        backstory="""You are a seasoned research analyst with expertise in artificial intelligence
                     and machine learning. Your specialty is identifying emerging trends and
                     breakthrough technologies that will shape the future.""",
        verbose=True,
    )

    # Define writer agent
    writer = Agent(
        role="Tech Content Strategist",
        goal="Craft compelling content on technology innovations",
        backstory="""You are a skilled content strategist with deep understanding of technology trends.
                     You excel at transforming complex technical research into engaging,
                     accessible content for diverse audiences.""",
        verbose=True,
    )

    # Define research task
    research_task = Task(
        description="""Conduct a comprehensive analysis of the latest developments in
                       multimodal AI systems. Focus on:
                       1. Recent breakthrough papers and models
                       2. Commercial applications and use cases
                       3. Technical challenges and limitations
                       4. Future research directions

                       Provide a structured summary with key insights.""",
        agent=researcher,
    )

    # Define writing task
    writing_task = Task(
        description="""Using the research analysis, create an engaging blog post about
                       multimodal AI developments. The post should:
                       1. Have an attention-grabbing introduction
                       2. Present complex concepts in accessible language
                       3. Include practical examples and implications
                       4. Conclude with future predictions

                       Target length: 800-1000 words.""",
        agent=writer,
    )

    # Create crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential,
        verbose=2,
    )

    print("✅ Research crew created with 2 agents and 2 tasks")
    return crew


def demo_zero_code_instrumentation():
    """Demonstrate zero-code auto-instrumentation."""
    print("\n" + "=" * 60)
    print("🚀 Demo 1: Zero-Code Auto-Instrumentation")
    print("=" * 60)

    # Enable auto-instrumentation
    print("Enabling auto-instrumentation...")
    success = auto_instrument(
        team="demo-team",
        project="basic-tracking",
        daily_budget_limit=20.0,
        governance_policy="advisory",
    )

    if not success:
        print("❌ Failed to enable auto-instrumentation")
        return

    print("✅ Auto-instrumentation enabled")

    # Create and run crew (automatically tracked)
    crew = create_research_crew()

    print("\n🎬 Starting crew execution (auto-instrumented)...")
    start_time = time.time()

    # This will be automatically tracked by GenOps
    result = crew.kickoff(
        {
            "topic": "multimodal AI systems",
            "target_audience": "technology professionals",
        }
    )

    end_time = time.time()

    print(f"\n📊 Execution completed in {end_time - start_time:.2f} seconds")
    print(f"📝 Result preview: {str(result)[:200]}...")

    # Get metrics from auto-instrumentation
    from genops.providers.crewai import get_cost_summary, get_execution_metrics

    cost_summary = get_cost_summary()
    if "error" not in cost_summary:
        print("\n💰 Auto-Instrumentation Metrics:")
        print(f"   Total cost: ${cost_summary.get('total_cost', 0):.6f}")
        print(f"   Agent executions: {cost_summary.get('agent_executions', 0)}")
        if cost_summary.get("cost_by_provider"):
            print(f"   Cost by provider: {cost_summary['cost_by_provider']}")

    execution_metrics = get_execution_metrics()
    if "error" not in execution_metrics:
        print(f"   Crew executions: {execution_metrics.get('total_executions', 0)}")
        print(f"   Success rate: {execution_metrics.get('success_rate', 0):.1%}")


def demo_manual_instrumentation():
    """Demonstrate manual crew tracking with full control."""
    print("\n" + "=" * 60)
    print("🎯 Demo 2: Manual Instrumentation with Full Control")
    print("=" * 60)

    # Create adapter with governance settings
    adapter = GenOpsCrewAIAdapter(
        team="manual-demo",
        project="crew-tracking",
        environment="development",
        daily_budget_limit=15.0,
        governance_policy="advisory",
        enable_cost_tracking=True,
    )

    print("✅ GenOps CrewAI adapter created")
    print(f"   Team: {adapter.team}")
    print(f"   Project: {adapter.project}")
    print(f"   Budget limit: ${adapter.daily_budget_limit}")

    # Create crew
    crew = create_research_crew()

    # Track with full governance
    with adapter.track_crew(
        "ai-research-crew", use_case="technology-analysis"
    ) as context:
        print("\n🎬 Starting tracked crew execution...")
        print(f"   Crew ID: {context.crew_id}")

        start_time = time.time()

        # Execute crew with tracking
        result = crew.kickoff(
            {
                "topic": "generative AI in enterprise applications",
                "focus_areas": ["productivity", "automation", "decision-making"],
            }
        )

        end_time = time.time()
        execution_time = end_time - start_time

        # Add custom business metrics
        context.add_custom_metric("research_domain", "enterprise_ai")
        context.add_custom_metric("content_type", "blog_post")
        context.add_custom_metric("execution_time", execution_time)

        print("\n📊 Execution Metrics:")
        print(f"   Execution time: {execution_time:.2f} seconds")
        print(f"   Result length: {len(str(result))} characters")

        # Get real-time metrics
        metrics = context.get_metrics()
        print(f"   Tracked agents: {metrics['total_agents']}")
        print(f"   Total cost: ${metrics['total_cost']:.6f}")

        if metrics["cost_by_provider"]:
            print(f"   Cost by provider: {metrics['cost_by_provider']}")

    print("\n✅ Crew execution completed and tracked")

    # Get adapter-level summary
    recent_results = adapter.get_crew_results(limit=1)
    if recent_results:
        latest = recent_results[0]
        print("\n📈 Summary:")
        print(f"   Total cost: ${latest['total_cost']:.6f}")
        print(f"   Execution time: {latest['execution_time_seconds']:.2f}s")
        print(f"   Success rate: {latest['success_rate']:.1%}")
        print(f"   Agents used: {latest['total_agents']}")


def demo_multi_crew_session():
    """Demonstrate session tracking with multiple crews."""
    print("\n" + "=" * 60)
    print("🔄 Demo 3: Multi-Crew Session Tracking")
    print("=" * 60)

    adapter = GenOpsCrewAIAdapter(
        team="session-demo", project="multi-crew-analysis", daily_budget_limit=25.0
    )

    # Create different crews for different tasks
    research_crew = create_research_crew()

    # Analysis crew (simplified for demo)
    analyst = Agent(
        role="Data Analyst",
        goal="Analyze research findings and extract insights",
        backstory="Expert at finding patterns and insights in research data",
    )

    analysis_task = Task(
        description="Analyze the research findings and provide 3 key insights",
        agent=analyst,
    )

    analysis_crew = Crew(
        agents=[analyst], tasks=[analysis_task], process=Process.sequential
    )

    # Track session with multiple crews
    with adapter.track_session("research-analysis-pipeline") as session:
        print(f"📋 Started session: {session.session_name}")

        # Execute multiple crews in sequence
        crews = [("research", research_crew), ("analysis", analysis_crew)]

        for crew_name, crew in crews:
            print(f"\n🔄 Executing {crew_name} crew...")

            with adapter.track_crew(f"{crew_name}-crew") as context:
                if crew_name == "research":
                    crew.kickoff({"topic": "AI safety research"})
                else:
                    crew.kickoff({"research_data": "placeholder findings"})

                # Add to session
                session.add_crew_result(context.get_metrics())

                print(f"   ✅ {crew_name} crew completed")

        print("\n📊 Session Summary:")
        print(f"   Total crews: {session.total_crews}")
        print(f"   Session cost: ${session.total_cost:.6f}")
        print(f"   Duration: {time.time() - session.start_time.timestamp():.1f}s")


def main():
    """Run the comprehensive CrewAI tracking demonstration."""
    print("🤖 Basic CrewAI Crew Tracking with GenOps")
    print("=" * 50)

    # Validate environment setup
    if not setup_environment():
        print("\n❌ Please fix environment issues before proceeding")
        return 1

    try:
        # Run demonstrations
        demo_zero_code_instrumentation()
        demo_manual_instrumentation()
        demo_multi_crew_session()

        print("\n🎉 All demonstrations completed successfully!")
        print("\n🚀 Next Steps:")
        print("   • Try multi_agent_cost_aggregation.py for advanced cost tracking")
        print("   • Run agent_workflow_governance.py for workflow analysis")
        print("   • Explore production_deployment_patterns.py for scaling")
        print("   • Integrate GenOps into your own CrewAI applications! 🌟")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️ Demonstration interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Demonstration failed: {e}", exc_info=True)
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
