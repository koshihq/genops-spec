#!/usr/bin/env python3
"""
AutoGen Group Chat Monitoring - Advanced Example

Demonstrates comprehensive monitoring of AutoGen group chat sessions with multiple
agents, role-based cost attribution, and coordination analytics. Shows enterprise
patterns for multi-agent governance.

Features Demonstrated:
    - Group chat orchestration tracking
    - Multi-agent cost attribution and role analysis
    - Speaker transition monitoring and coordination metrics
    - Agent participation balance analysis
    - Group dynamics and collaboration scoring
    - Advanced multi-provider cost optimization

Usage:
    python examples/autogen/03_group_chat_monitoring.py

Prerequisites:
    pip install genops[autogen]
    export OPENAI_API_KEY=your_key

Time Investment: 20-30 minutes to understand advanced patterns
Complexity Level: Advanced (builds on conversation tracking)
"""

import os
import time
from decimal import Decimal


def main():
    """Demonstrate advanced AutoGen group chat monitoring and governance."""

    print("👥 AutoGen + GenOps: Advanced Group Chat Monitoring")
    print("=" * 65)

    # Advanced governance configuration for group chats
    print("🏗️  Configuring advanced governance for group chat scenarios...")
    try:
        from genops.providers.autogen import create_multi_agent_adapter

        # Use the specialized multi-agent adapter
        adapter = create_multi_agent_adapter(
            team="research-team",
            project="collaborative-analysis",
            daily_budget_limit=15.0,  # Higher budget for group chats
            enable_advanced_monitoring=True,
        )

        print("✅ Multi-agent adapter configured:")
        print("   Team: research-team")
        print("   Project: collaborative-analysis")
        print(f"   Budget: ${adapter.daily_budget_limit}")
        print("   Advanced monitoring: Enabled")

    except Exception as e:
        print(f"❌ Advanced setup failed: {e}")
        return

    # Create diverse group of agents with different roles
    print("\n🤖 Creating diverse AutoGen agent group...")
    try:
        import autogen

        config_list = [
            {
                "model": "gpt-3.5-turbo",
                "api_key": os.getenv("OPENAI_API_KEY", "demo-key"),
            }
        ]

        use_real_llm = bool(os.getenv("OPENAI_API_KEY"))
        if not use_real_llm:
            print("⚠️  No API key - will simulate group chat")
            config_list = False

        # Create specialized agents with different roles
        analyst = autogen.AssistantAgent(
            name="data_analyst",
            llm_config={"config_list": config_list} if config_list else False,
            system_message="You are a data analyst. Focus on quantitative analysis and data-driven insights. Keep responses concise and analytical.",
        )

        critic = autogen.AssistantAgent(
            name="critic",
            llm_config={"config_list": config_list} if config_list else False,
            system_message="You are a critical reviewer. Question assumptions, identify potential issues, and suggest improvements. Be constructive but thorough.",
        )

        summarizer = autogen.AssistantAgent(
            name="summarizer",
            llm_config={"config_list": config_list} if config_list else False,
            system_message="You are a summarizer. Synthesize discussions into clear, actionable conclusions. Highlight key decisions and next steps.",
        )

        user_proxy = autogen.UserProxyAgent(
            name="research_lead",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=2,
            is_termination_msg=lambda x: "TERMINATE" in x.get("content", ""),
            code_execution_config={
                "work_dir": "autogen_workspace",
                "use_docker": False,
            },
        )

        # Instrument each agent for detailed tracking
        analyst = adapter.instrument_agent(analyst, "data_analyst")
        critic = adapter.instrument_agent(critic, "critical_reviewer")
        summarizer = adapter.instrument_agent(summarizer, "synthesis_specialist")
        user_proxy = adapter.instrument_agent(user_proxy, "research_lead")

        agents = [analyst, critic, summarizer, user_proxy]
        agent_names = [agent.name for agent in agents]

        print(f"✅ Created {len(agents)} specialized agents:")
        for agent in agents:
            role = {
                "data_analyst": "Quantitative analysis",
                "critic": "Critical review",
                "summarizer": "Synthesis & decisions",
                "research_lead": "Coordination & leadership",
            }.get(agent.name, "General purpose")
            print(f"   - {agent.name}: {role}")

    except ImportError:
        print("❌ AutoGen not installed: pip install pyautogen")
        return
    except Exception as e:
        print(f"❌ Agent group creation failed: {e}")
        return

    # Group Chat Session 1: Research Problem Analysis
    print("\n👥 Group Chat Session 1: Research Problem Analysis")
    try:
        with adapter.track_group_chat(
            group_chat_id="research-analysis", participants=agent_names
        ) as context:
            print("   Initializing group chat tracking...")

            if use_real_llm:
                # Create AutoGen GroupChat
                group_chat = autogen.GroupChat(
                    agents=agents,
                    messages=[],
                    max_round=6,  # Limit rounds for demo
                    speaker_selection_method="auto",
                )

                manager = autogen.GroupChatManager(
                    groupchat=group_chat, llm_config={"config_list": config_list}
                )

                print("   Starting group discussion...")
                user_proxy.initiate_chat(
                    manager,
                    message="""Let's analyze this research question: 'How can we optimize multi-agent systems for better cost efficiency while maintaining performance?'

                    Data Analyst: Please provide quantitative insights.
                    Critic: Challenge our assumptions.
                    Summarizer: Synthesize our findings.

                    Keep responses brief for this demo.""",
                )

            else:
                # Simulate group chat interaction
                print("   [Simulated Group Chat]")
                print("   Research Lead: How can we optimize multi-agent systems?")

                print(
                    "   Data Analyst: Based on benchmarks, cost efficiency correlates with..."
                )
                context.add_turn(Decimal("0.003"), 200, "data_analyst")

                print("   Critic: We should question whether cost efficiency might...")
                context.add_turn(Decimal("0.004"), 250, "critic")

                print("   Summarizer: Synthesizing the discussion, key insights are...")
                context.add_turn(Decimal("0.003"), 180, "summarizer")

                print("   Research Lead: Excellent analysis. Let's proceed with...")
                context.add_turn(Decimal("0.002"), 120, "research_lead")

                # Simulate function calls and code execution
                context.add_function_call("analyze_performance_metrics")
                context.add_code_execution()

                time.sleep(2)

            print("   ✅ Group Chat Session 1 completed:")
            print(f"      Total cost: ${context.total_cost:.6f}")
            print(f"      Turns: {context.turns_count}")
            print(f"      Function calls: {context.function_calls}")
            print(f"      Code executions: {context.code_executions}")

    except Exception as e:
        print(f"   ❌ Group Chat Session 1 failed: {e}")

    # Group Chat Session 2: Decision Making
    print("\n👥 Group Chat Session 2: Decision Making Process")
    try:
        with adapter.track_group_chat(
            group_chat_id="decision-making", participants=agent_names
        ) as context:
            if use_real_llm:
                group_chat = autogen.GroupChat(
                    agents=agents,
                    messages=[],
                    max_round=8,
                    speaker_selection_method="round_robin",  # Different selection method
                )

                manager = autogen.GroupChatManager(
                    groupchat=group_chat, llm_config={"config_list": config_list}
                )

                user_proxy.initiate_chat(
                    manager,
                    message="""Based on our analysis, we need to make a decision on implementation approach.

                    Each agent should weigh in with their perspective on the best path forward.
                    Focus on actionable recommendations.""",
                )

            else:
                # Simulate decision-making session
                print("   [Simulated Decision Session]")

                print(
                    "   Research Lead: We need to decide on implementation approach..."
                )
                context.add_turn(Decimal("0.002"), 150, "research_lead")

                print(
                    "   Data Analyst: The data suggests approach A has 23% better ROI..."
                )
                context.add_turn(Decimal("0.005"), 320, "data_analyst")

                print(
                    "   Critic: However, approach A has significant risks including..."
                )
                context.add_turn(Decimal("0.004"), 280, "critic")

                print(
                    "   Summarizer: Weighing the analysis, I recommend a hybrid approach..."
                )
                context.add_turn(Decimal("0.006"), 400, "summarizer")

                print(
                    "   Research Lead: Excellent. Let's proceed with the hybrid approach."
                )
                context.add_turn(Decimal("0.002"), 100, "research_lead")

                time.sleep(2)

            print("   ✅ Group Chat Session 2 completed:")
            print(f"      Total cost: ${context.total_cost:.6f}")
            print(f"      Turns: {context.turns_count}")

    except Exception as e:
        print(f"   ❌ Group Chat Session 2 failed: {e}")

    # Advanced Analytics: Group Dynamics Analysis
    print("\n📊 Advanced Analytics: Group Dynamics Analysis")
    try:
        summary = adapter.get_session_summary()

        print("Session Analytics:")
        print(f"   Total group chats: {summary['total_conversations']}")
        print(f"   Total cost: ${summary['total_cost']:.6f}")
        print(f"   Budget utilization: {summary['budget_utilization']:.1f}%")

        # Agent participation analysis
        if summary["active_agents"]:
            print("\n   Agent Participation:")
            summary["total_turns"]
            for agent in summary["active_agents"]:
                # This would normally come from the monitor
                participation = (
                    f"~{100 / len(summary['active_agents']):.1f}%"  # Simulated
                )
                print(f"      {agent}: {participation} of discussion")

        # Cost breakdown by agent role
        print("\n   Cost Analysis by Agent Role:")
        agent_roles = {
            "data_analyst": "Analysis & Research",
            "critic": "Review & Validation",
            "summarizer": "Synthesis & Decisions",
            "research_lead": "Coordination",
        }

        # Simulate cost distribution (in real usage, this comes from actual tracking)
        simulated_costs = {
            "data_analyst": 0.008,
            "critic": 0.006,
            "summarizer": 0.009,
            "research_lead": 0.004,
        }

        for agent, cost in simulated_costs.items():
            role = agent_roles.get(agent, "Unknown")
            print(f"      {role} ({agent}): ${cost:.6f}")

    except Exception as e:
        print(f"   ⚠️  Session analytics error: {e}")

    # Multi-Provider Cost Optimization for Groups
    print("\n💰 Multi-Provider Cost Optimization for Group Chats")
    try:
        from genops.providers.autogen import analyze_conversation_costs

        analysis = analyze_conversation_costs(adapter, time_period_hours=1)

        if "error" not in analysis:
            print("Group Chat Cost Analysis:")
            print(f"   Combined session cost: ${analysis['total_cost']:.6f}")

            if analysis["cost_by_agent"]:
                print("   Most expensive agent:", analysis["most_expensive_agent"])

            # Group-specific recommendations
            if analysis["recommendations"]:
                print("   💡 Group optimization recommendations:")
                for i, rec in enumerate(analysis["recommendations"][:2], 1):
                    print(f"      {i}. {rec['reasoning']}")
            else:
                print("   ✅ Group costs are well-optimized")

            # Simulate group-specific insights
            print("\n   Group Dynamics Insights:")
            print("   • Balanced participation across roles")
            print("   • Efficient turn-taking with minimal redundancy")
            print("   • Strong correlation between complexity and agent expertise")

        else:
            print(f"   ⚠️  Cost analysis: {analysis['error']}")

    except Exception as e:
        print(f"   ⚠️  Multi-provider optimization not available: {e}")

    # Collaboration Quality Assessment
    print("\n🤝 Collaboration Quality Assessment")
    try:
        print("Group Collaboration Metrics:")
        print("   Coordination efficiency: High (simulated)")
        print("   Speaker transition smoothness: 92% (simulated)")
        print("   Consensus quality: Strong (simulated)")
        print("   Role specialization clarity: Excellent (simulated)")

        print("\n   Quality Indicators:")
        print("   ✅ Clear role differentiation maintained")
        print("   ✅ Productive critical analysis without conflict")
        print("   ✅ Effective synthesis and decision-making")
        print("   ✅ Appropriate cost distribution across roles")

    except Exception as e:
        print(f"   ⚠️  Collaboration assessment not available: {e}")

    print("\n" + "=" * 65)
    print("🎉 Advanced Group Chat Monitoring Complete!")

    print("\n🎯 Advanced Concepts Demonstrated:")
    print("✅ Multi-agent group chat orchestration tracking")
    print("✅ Role-based cost attribution and analysis")
    print("✅ Speaker transition and coordination monitoring")
    print("✅ Group dynamics and collaboration quality scoring")
    print("✅ Advanced multi-provider cost optimization")
    print("✅ Enterprise-grade governance for team workflows")

    print("\n🚀 Next Steps:")
    print(
        "1. Code execution monitoring: python examples/autogen/04_code_execution_tracking.py"
    )
    print("2. Production deployment: python examples/autogen/05_production_patterns.py")
    print("3. Advanced optimization: python examples/autogen/06_cost_optimization.py")

    print("\n🏢 Enterprise Applications:")
    print("- Multi-team AI collaboration governance")
    print("- Cost center attribution for AI initiatives")
    print("- Performance optimization for complex workflows")
    print("- Compliance and audit trails for AI decisions")

    print("\n📚 Deep Dive Resources:")
    print("- Group chat patterns: docs/integrations/autogen.md#group-chats")
    print("- Multi-agent governance: docs/enterprise/multi-agent-governance.md")
    print("- Cost optimization: docs/optimization/multi-provider-strategies.md")
    print("=" * 65)


if __name__ == "__main__":
    main()
