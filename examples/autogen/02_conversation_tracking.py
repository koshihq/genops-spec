#!/usr/bin/env python3
"""
AutoGen Conversation Tracking - Intermediate Example

This example demonstrates more detailed conversation tracking and cost analysis,
building on the basic quickstart pattern. Shows manual instrumentation alongside
auto-instrumentation for more granular control.

Features Demonstrated:
    - Manual conversation tracking with context managers
    - Real-time cost monitoring and budget alerts
    - Conversation analytics and performance metrics
    - Multiple conversation patterns in one session
    - Cost optimization insights and recommendations

Usage:
    python examples/autogen/02_conversation_tracking.py

Prerequisites:
    pip install genops[autogen]
    export OPENAI_API_KEY=your_key
    
Time Investment: 10-15 minutes to understand and run
"""

import os
import time
from decimal import Decimal

def main():
    """Demonstrate intermediate AutoGen conversation tracking patterns."""
    
    print("🔬 AutoGen + GenOps: Intermediate Conversation Tracking")
    print("=" * 60)
    
    # Setup with more detailed configuration
    print("⚙️  Setting up detailed governance configuration...")
    try:
        from genops.providers.autogen import GenOpsAutoGenAdapter
        
        # Manual adapter setup for more control
        adapter = GenOpsAutoGenAdapter(
            team="demo-team",
            project="conversation-analysis",
            environment="development", 
            daily_budget_limit=5.0,  # $5 limit for demo
            governance_policy="advisory",
            enable_conversation_tracking=True,
            enable_agent_tracking=True,
            enable_cost_tracking=True
        )
        
        print(f"✅ Governance adapter configured:")
        print(f"   Team: {adapter.team}")
        print(f"   Project: {adapter.project}")
        print(f"   Daily budget: ${adapter.daily_budget_limit}")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return
    
    # Create AutoGen agents with manual instrumentation
    print("\n🤖 Creating instrumented AutoGen agents...")
    try:
        import autogen
        
        config_list = [
            {
                "model": "gpt-3.5-turbo", 
                "api_key": os.getenv("OPENAI_API_KEY", "demo-key")
            }
        ]
        
        # Skip real LLM calls if no API key
        use_real_llm = bool(os.getenv("OPENAI_API_KEY"))
        if not use_real_llm:
            print("⚠️  No API key - will simulate conversations")
            config_list = False
        
        # Create agents
        assistant = autogen.AssistantAgent(
            name="assistant",
            llm_config={"config_list": config_list} if config_list else False,
            system_message="You are a knowledgeable AI assistant. Keep responses concise."
        )
        
        user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE")
        )
        
        # Manually instrument agents for detailed tracking
        assistant = adapter.instrument_agent(assistant, "knowledge_assistant")
        user_proxy = adapter.instrument_agent(user_proxy, "demo_user")
        
        print("✅ Created and instrumented AutoGen agents")
        
    except ImportError:
        print("❌ AutoGen not installed: pip install pyautogen")
        return
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return
    
    # Conversation 1: Basic question-answer
    print("\n💬 Conversation 1: Basic Question-Answer")
    try:
        with adapter.track_conversation(
            conversation_id="basic-qa",
            participants=["assistant", "user_proxy"]
        ) as context:
            
            print("   Starting conversation tracking...")
            
            if use_real_llm:
                user_proxy.initiate_chat(
                    assistant,
                    message="What are the main benefits of using AutoGen for multi-agent systems?"
                )
            else:
                # Simulate conversation for demo
                print("   [Simulated] User: What are the main benefits of AutoGen?")
                print("   [Simulated] Assistant: AutoGen enables multi-agent conversations...")
                context.add_turn(Decimal('0.002'), 150, "assistant")
                context.add_turn(Decimal('0.001'), 75, "user_proxy")
                time.sleep(1)  # Simulate processing time
            
            print(f"   ✅ Conversation 1 completed:")
            print(f"      Cost: ${context.total_cost:.6f}")
            print(f"      Turns: {context.turns_count}")
            
    except Exception as e:
        print(f"   ❌ Conversation 1 failed: {e}")
    
    # Conversation 2: More complex interaction
    print("\n💬 Conversation 2: Complex Problem Solving")
    try:
        with adapter.track_conversation(
            conversation_id="problem-solving",
            participants=["assistant", "user_proxy"]
        ) as context:
            
            if use_real_llm:
                user_proxy.initiate_chat(
                    assistant,
                    message="Can you help me design a simple multi-agent workflow for document analysis? Describe the agents and their roles."
                )
            else:
                # Simulate more complex conversation
                print("   [Simulated] User: Help design a multi-agent workflow...")
                print("   [Simulated] Assistant: I'll design a workflow with specialized agents...")
                print("   [Simulated] User: Can you elaborate on the coordination?")
                print("   [Simulated] Assistant: Here's how agents coordinate...")
                
                # Simulate higher costs for more complex conversation
                context.add_turn(Decimal('0.004'), 280, "assistant")  # Longer response
                context.add_turn(Decimal('0.002'), 120, "user_proxy")
                context.add_turn(Decimal('0.006'), 420, "assistant")  # Even longer response
                context.add_turn(Decimal('0.001'), 60, "user_proxy")
                time.sleep(2)  # Simulate longer processing
            
            print(f"   ✅ Conversation 2 completed:")
            print(f"      Cost: ${context.total_cost:.6f}")
            print(f"      Turns: {context.turns_count}")
            
    except Exception as e:
        print(f"   ❌ Conversation 2 failed: {e}")
    
    # Session analysis and insights
    print("\n📊 Session Analysis & Insights")
    try:
        summary = adapter.get_session_summary()
        
        print(f"Session Summary:")
        print(f"   Total conversations: {summary['total_conversations']}")
        print(f"   Total cost: ${summary['total_cost']:.6f}")
        print(f"   Budget utilization: {summary['budget_utilization']:.1f}%")
        print(f"   Average cost per conversation: ${summary['avg_cost_per_conversation']:.6f}")
        print(f"   Average cost per turn: ${summary['avg_cost_per_turn']:.6f}")
        print(f"   Unique agents used: {len(summary['active_agents'])}")
        print(f"   Session duration: {summary['session_duration']:.1f} seconds")
        
        # Budget status
        if summary['budget_utilization'] > 80:
            print("   ⚠️  High budget utilization - consider monitoring")
        elif summary['budget_utilization'] > 50:
            print("   ✅ Moderate budget usage - healthy level")
        else:
            print("   ✅ Low budget usage - plenty of headroom")
            
    except Exception as e:
        print(f"   ⚠️  Could not get session summary: {e}")
    
    # Cost optimization insights
    print("\n💰 Cost Optimization Insights")
    try:
        from genops.providers.autogen import analyze_conversation_costs
        
        analysis = analyze_conversation_costs(adapter, time_period_hours=1)
        
        if 'error' not in analysis:
            print("Cost Analysis:")
            print(f"   Total cost: ${analysis['total_cost']:.6f}")
            
            if analysis['cost_by_agent']:
                print("   Cost by agent:")
                for agent, cost in analysis['cost_by_agent'].items():
                    print(f"      {agent}: ${cost:.6f}")
            
            if analysis['recommendations']:
                print("   💡 Optimization recommendations:")
                for i, rec in enumerate(analysis['recommendations'][:3], 1):
                    print(f"      {i}. {rec['reasoning']}")
            else:
                print("   ✅ No optimization recommendations - costs look optimal")
        else:
            print(f"   ⚠️  Cost analysis: {analysis['error']}")
            
    except Exception as e:
        print(f"   ⚠️  Cost optimization analysis not available: {e}")
    
    # Conversation insights
    print("\n🔍 Conversation Quality Insights")
    try:
        from genops.providers.autogen import get_conversation_insights
        
        monitor = adapter.conversation_monitor
        if monitor:
            # Get insights for our conversations
            for conv_id in ["basic-qa", "problem-solving"]:
                insights = get_conversation_insights(monitor, conv_id)
                if 'error' not in insights:
                    print(f"   {conv_id}:")
                    print(f"      Turns: {insights.get('turns_count', 0)}")
                    print(f"      Avg response time: {insights.get('avg_response_time_ms', 0):.1f}ms")
                    print(f"      Quality score: {insights.get('conversation_quality_score', 0):.2f}")
                else:
                    print(f"   {conv_id}: {insights['error']}")
        else:
            print("   ⚠️  Conversation monitor not available")
            
    except Exception as e:
        print(f"   ⚠️  Conversation insights not available: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Intermediate Conversation Tracking Complete!")
    
    print("\n🎯 Key Learnings:")
    print("✅ Manual conversation tracking with context managers")
    print("✅ Real-time cost monitoring and budget awareness") 
    print("✅ Session analytics and conversation quality metrics")
    print("✅ Cost optimization insights and recommendations")
    print("✅ Agent instrumentation for detailed tracking")
    
    print("\n🚀 Next Steps:")
    print("1. Try group conversations: python examples/autogen/03_group_chat_monitoring.py")
    print("2. Explore code execution: python examples/autogen/04_code_execution_tracking.py")
    print("3. Production patterns: python examples/autogen/05_production_patterns.py")
    print("4. Advanced optimization: python examples/autogen/06_cost_optimization.py")
    
    print("\n📚 Learn More:")
    print("- Comprehensive guide: docs/integrations/autogen.md")
    print("- All examples: examples/autogen/")
    print("- Community: https://github.com/KoshiHQ/GenOps-AI/discussions")
    print("=" * 60)


if __name__ == "__main__":
    main()