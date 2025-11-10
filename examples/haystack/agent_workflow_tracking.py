#!/usr/bin/env python3
"""
Agent Workflow Tracking with GenOps and Haystack

Demonstrates specialized agent workflow monitoring with GenOps governance,
including decision tracking, tool usage monitoring, iterative process governance,
and comprehensive agent-specific analytics.

Usage:
    python agent_workflow_tracking.py

Features:
    - Agent-optimized GenOps adapter with decision and tool tracking
    - Multi-step agent workflow simulation with decision points
    - Tool usage monitoring and cost attribution
    - Agent iteration tracking with performance analysis
    - Complex multi-agent coordination governance
    - Agent performance insights and optimization recommendations
"""

import logging
import os
import sys
import time
import random
from decimal import Decimal
from typing import List, Dict, Any, Optional

# Core Haystack imports for agent workflows
try:
    from haystack import Pipeline
    from haystack.components.generators import OpenAIGenerator
    from haystack.components.builders import PromptBuilder
    from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
    from haystack.components.writers import DocumentWriter
except ImportError as e:
    print(f"❌ Haystack not installed: {e}")
    print("Please install Haystack: pip install haystack-ai")
    sys.exit(1)

# GenOps imports
try:
    from genops.providers.haystack import (
        create_agent_adapter,
        GenOpsHaystackAdapter,
        validate_haystack_setup,
        print_validation_result,
        get_agent_insights,
        analyze_pipeline_costs
    )
except ImportError as e:
    print(f"❌ GenOps not installed: {e}")
    print("Please install GenOps: pip install genops-ai[haystack]")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentToolSimulator:
    """Simulate various agent tools with cost and performance tracking."""
    
    def __init__(self, adapter):
        self.adapter = adapter
        self.tool_costs = {
            "web_search": 0.005,
            "document_analysis": 0.008,
            "data_extraction": 0.003,
            "code_generation": 0.012,
            "api_call": 0.002,
            "calculation": 0.001,
            "translation": 0.004
        }
    
    def use_tool(self, tool_name: str, input_data: str, complexity: str = "medium") -> Dict[str, Any]:
        """Simulate using an agent tool."""
        
        # Simulate tool execution time based on complexity
        complexity_multipliers = {"simple": 0.5, "medium": 1.0, "complex": 2.0}
        base_time = random.uniform(0.5, 2.0)
        execution_time = base_time * complexity_multipliers.get(complexity, 1.0)
        
        # Simulate processing
        time.sleep(min(execution_time / 10, 0.2))  # Reduced for demo
        
        # Calculate cost
        base_cost = self.tool_costs.get(tool_name, 0.005)
        actual_cost = base_cost * complexity_multipliers.get(complexity, 1.0)
        
        # Simulate tool results based on tool type
        result = self.generate_tool_result(tool_name, input_data)
        
        return {
            "tool_name": tool_name,
            "input": input_data,
            "result": result,
            "execution_time": execution_time,
            "cost": actual_cost,
            "complexity": complexity,
            "success": random.random() > 0.05  # 95% success rate
        }
    
    def generate_tool_result(self, tool_name: str, input_data: str) -> str:
        """Generate realistic tool results for demonstration."""
        
        results = {
            "web_search": f"Found 15 relevant results for '{input_data[:50]}...'. Top results include recent articles and documentation.",
            "document_analysis": f"Analyzed document content. Key themes: machine learning, AI governance, cost optimization. Confidence: 0.87",
            "data_extraction": f"Extracted 42 data points from source material. Structured format available.",
            "code_generation": f"Generated Python code solution for '{input_data[:30]}...'. 45 lines, includes error handling.",
            "api_call": f"API call completed successfully. Retrieved data for {input_data}. Status: 200, Response time: 245ms",
            "calculation": f"Computation completed: Result = {random.randint(100, 999)}. Confidence: 0.99",
            "translation": f"Translated text to target language. Quality score: 0.92. {len(input_data)} characters processed."
        }
        
        return results.get(tool_name, f"Tool {tool_name} executed successfully with input: {input_data[:50]}...")


def create_agent_decision_pipeline() -> Pipeline:
    """Create pipeline for agent decision making."""
    print("🤖 Creating Agent Decision Pipeline")
    
    pipeline = Pipeline()
    
    # Decision maker component
    pipeline.add_component("decision_maker", PromptBuilder(
        template="""
        You are an AI agent that needs to make decisions about how to solve a task.
        
        Task: {{task}}
        Available tools: {{available_tools}}
        Previous results: {{previous_results}}
        
        Analyze the task and decide:
        1. What is the next best action to take?
        2. Which tool should be used?
        3. What input should be provided to the tool?
        4. Is this task complete, or are more steps needed?
        
        Provide your decision in this format:
        DECISION: [continue/complete]
        TOOL: [tool_name]
        INPUT: [tool_input]
        REASONING: [your_reasoning]
        """
    ))
    
    pipeline.add_component("llm", OpenAIGenerator(
        model="gpt-3.5-turbo",
        generation_kwargs={"max_tokens": 200, "temperature": 0.7}
    ))
    
    pipeline.connect("decision_maker", "llm")
    
    print("✅ Agent decision pipeline created")
    return pipeline


def create_agent_synthesis_pipeline() -> Pipeline:
    """Create pipeline for synthesizing agent results."""
    print("🧠 Creating Agent Synthesis Pipeline")
    
    pipeline = Pipeline()
    
    pipeline.add_component("synthesizer", PromptBuilder(
        template="""
        You are an AI agent synthesizing results from multiple tools and steps.
        
        Original Task: {{original_task}}
        
        Tool Results:
        {% for result in tool_results %}
        - {{result.tool_name}}: {{result.result}}
        {% endfor %}
        
        Provide a comprehensive final answer that synthesizes all the tool results
        to address the original task. Be specific and cite which tools provided
        which information.
        
        Final Answer:
        """
    ))
    
    pipeline.add_component("llm", OpenAIGenerator(
        model="gpt-3.5-turbo",
        generation_kwargs={"max_tokens": 300, "temperature": 0.5}
    ))
    
    pipeline.connect("synthesizer", "llm")
    
    print("✅ Agent synthesis pipeline created")
    return pipeline


def demo_agent_workflow_tracking():
    """Demonstrate comprehensive agent workflow tracking."""
    print("\n" + "="*70)
    print("🤖 Agent Workflow Tracking with GenOps")
    print("="*70)
    
    # Create agent-specialized adapter
    agent_adapter = create_agent_adapter(
        team="ai-agents",
        project="research-assistant",
        daily_budget_limit=100.0,
        enable_decision_tracking=True,
        enable_tool_tracking=True
    )
    
    print("✅ Agent-specialized GenOps adapter created")
    print(f"   Team: {agent_adapter.team}")
    print(f"   Project: {agent_adapter.project}")
    print(f"   Daily budget: ${agent_adapter.daily_budget_limit}")
    
    # Initialize agent components
    decision_pipeline = create_agent_decision_pipeline()
    synthesis_pipeline = create_agent_synthesis_pipeline()
    tool_simulator = AgentToolSimulator(agent_adapter)
    
    # Complex agent tasks for demonstration
    agent_tasks = [
        {
            "task": "Research the latest trends in AI governance and cost optimization",
            "complexity": "complex",
            "expected_tools": ["web_search", "document_analysis", "data_extraction"],
            "max_iterations": 4
        },
        {
            "task": "Create a Python script to analyze CSV data and generate visualizations",
            "complexity": "medium", 
            "expected_tools": ["code_generation", "document_analysis", "calculation"],
            "max_iterations": 3
        },
        {
            "task": "Translate technical documentation and summarize key points",
            "complexity": "medium",
            "expected_tools": ["translation", "document_analysis", "data_extraction"],
            "max_iterations": 3
        }
    ]
    
    # Execute agent tasks with comprehensive tracking
    session_results = []
    
    with agent_adapter.track_session("agent-research-session", use_case="multi-agent-workflow") as session:
        print(f"\n📋 Started agent session: {session.session_name}")
        
        for task_num, task_config in enumerate(agent_tasks, 1):
            task_description = task_config["task"]
            max_iterations = task_config["max_iterations"]
            
            print(f"\n🎯 Task {task_num}/{len(agent_tasks)}: {task_config['complexity']} complexity")
            print(f"   Description: {task_description}")
            
            # Track individual agent task execution
            with agent_adapter.track_pipeline(
                f"agent-task-{task_num}",
                customer_id="demo-customer", 
                task_complexity=task_config["complexity"],
                expected_iterations=max_iterations
            ) as context:
                
                # Initialize task state
                task_state = {
                    "task": task_description,
                    "tool_results": [],
                    "decisions": [],
                    "iterations": 0,
                    "completed": False
                }
                
                available_tools = list(tool_simulator.tool_costs.keys())
                
                # Agent iteration loop
                while not task_state["completed"] and task_state["iterations"] < max_iterations:
                    iteration = task_state["iterations"] + 1
                    print(f"\n   🔄 Iteration {iteration}/{max_iterations}")
                    
                    # Agent decision making
                    decision_result = decision_pipeline.run({
                        "decision_maker": {
                            "task": task_description,
                            "available_tools": ", ".join(available_tools),
                            "previous_results": [r["result"] for r in task_state["tool_results"][-2:]]
                        }
                    })
                    
                    decision_text = decision_result["llm"]["replies"][0]
                    print(f"      🧠 Decision: {decision_text[:100]}...")
                    
                    # Parse decision (simplified for demo)
                    if "COMPLETE" in decision_text.upper() or "complete" in decision_text.lower():
                        task_state["completed"] = True
                        print(f"      ✅ Agent decided task is complete")
                        break
                    
                    # Select tool based on decision
                    selected_tool = None
                    for tool in available_tools:
                        if tool.replace("_", " ") in decision_text.lower():
                            selected_tool = tool
                            break
                    
                    if not selected_tool:
                        selected_tool = random.choice(task_config["expected_tools"])
                    
                    # Use selected tool
                    tool_input = f"Process: {task_description[:50]}... (iteration {iteration})"
                    tool_result = tool_simulator.use_tool(
                        selected_tool, 
                        tool_input,
                        task_config["complexity"]
                    )
                    
                    print(f"      🛠️ Used tool: {selected_tool}")
                    print(f"      💰 Tool cost: ${tool_result['cost']:.6f}")
                    print(f"      ⏱️ Tool time: {tool_result['execution_time']:.2f}s")
                    print(f"      📊 Result: {tool_result['result'][:80]}...")
                    
                    # Track tool usage
                    context.add_custom_metric(f"tool_{selected_tool}_used", 1)
                    context.add_custom_metric(f"iteration_{iteration}_cost", tool_result['cost'])
                    context.add_custom_metric("tool_execution_time", tool_result['execution_time'])
                    
                    task_state["tool_results"].append(tool_result)
                    task_state["decisions"].append(decision_text)
                    task_state["iterations"] += 1
                
                # Synthesize final result
                if task_state["tool_results"]:
                    print(f"\n   🧬 Synthesizing results from {len(task_state['tool_results'])} tools...")
                    
                    synthesis_result = synthesis_pipeline.run({
                        "synthesizer": {
                            "original_task": task_description,
                            "tool_results": task_state["tool_results"]
                        }
                    })
                    
                    final_answer = synthesis_result["llm"]["replies"][0]
                    print(f"   🎯 Final Answer: {final_answer[:150]}...")
                
                # Calculate agent-specific metrics
                total_tool_cost = sum(r["cost"] for r in task_state["tool_results"])
                total_tools_used = len(task_state["tool_results"])
                success_rate = sum(1 for r in task_state["tool_results"] if r["success"]) / max(total_tools_used, 1)
                
                context.add_custom_metric("total_iterations", task_state["iterations"])
                context.add_custom_metric("tools_used", total_tools_used)
                context.add_custom_metric("tool_success_rate", success_rate)
                context.add_custom_metric("task_completed", task_state["completed"])
                context.add_custom_metric("total_tool_cost", total_tool_cost)
                
                # Get execution metrics
                metrics = context.get_metrics()
                print(f"   📊 Task Summary:")
                print(f"      Total cost: ${metrics.total_cost:.6f}")
                print(f"      Iterations: {task_state['iterations']}")
                print(f"      Tools used: {total_tools_used}")
                print(f"      Success rate: {success_rate:.1%}")
                print(f"      Completed: {'✅' if task_state['completed'] else '⏸️'}")
                
                # Store results for analysis
                session_results.append({
                    "task": task_description,
                    "complexity": task_config["complexity"],
                    "iterations": task_state["iterations"],
                    "tools_used": total_tools_used,
                    "success_rate": success_rate,
                    "cost": float(metrics.total_cost),
                    "time": metrics.total_execution_time_seconds,
                    "completed": task_state["completed"],
                    "tool_breakdown": {r["tool_name"]: r["cost"] for r in task_state["tool_results"]},
                    "pipeline_id": context.pipeline_id
                })
            
            session.add_pipeline_result(context.get_metrics())
        
        print(f"\n📊 Agent Session Summary:")
        print(f"   Total tasks: {session.total_pipelines}")
        print(f"   Total cost: ${session.total_cost:.6f}")
        print(f"   Average cost per task: ${session.total_cost / session.total_pipelines:.6f}")
    
    return agent_adapter, session_results


def analyze_agent_performance(agent_adapter, session_results):
    """Analyze agent performance with specialized insights."""
    print("\n" + "="*70)
    print("🔬 Agent Performance Analysis")
    print("="*70)
    
    # Get overall cost analysis
    cost_analysis = analyze_pipeline_costs(agent_adapter, time_period_hours=1)
    
    print("💰 Cost Analysis:")
    print(f"   Total cost: ${cost_analysis['total_cost']:.6f}")
    print(f"   Cost by provider: {cost_analysis['cost_by_provider']}")
    
    # Agent-specific performance metrics
    if session_results:
        total_iterations = sum(r["iterations"] for r in session_results)
        total_tools_used = sum(r["tools_used"] for r in session_results)
        avg_success_rate = sum(r["success_rate"] for r in session_results) / len(session_results)
        completed_tasks = sum(1 for r in session_results if r["completed"])
        
        print(f"\n🤖 Agent-Specific Metrics:")
        print(f"   Total iterations across all tasks: {total_iterations}")
        print(f"   Total tools used: {total_tools_used}")
        print(f"   Average tool success rate: {avg_success_rate:.1%}")
        print(f"   Task completion rate: {completed_tasks}/{len(session_results)} ({completed_tasks/len(session_results):.1%})")
        print(f"   Average iterations per task: {total_iterations/len(session_results):.1f}")
        
        # Performance by task complexity
        complexity_performance = {}
        for result in session_results:
            complexity = result["complexity"]
            if complexity not in complexity_performance:
                complexity_performance[complexity] = {
                    "costs": [], "iterations": [], "tools": [], 
                    "success_rates": [], "completion_rates": []
                }
            
            complexity_performance[complexity]["costs"].append(result["cost"])
            complexity_performance[complexity]["iterations"].append(result["iterations"])
            complexity_performance[complexity]["tools"].append(result["tools_used"])
            complexity_performance[complexity]["success_rates"].append(result["success_rate"])
            complexity_performance[complexity]["completion_rates"].append(1 if result["completed"] else 0)
        
        print(f"\n📊 Performance by Task Complexity:")
        for complexity, perf_data in complexity_performance.items():
            avg_cost = sum(perf_data["costs"]) / len(perf_data["costs"])
            avg_iterations = sum(perf_data["iterations"]) / len(perf_data["iterations"])
            avg_tools = sum(perf_data["tools"]) / len(perf_data["tools"])
            avg_success = sum(perf_data["success_rates"]) / len(perf_data["success_rates"])
            completion_rate = sum(perf_data["completion_rates"]) / len(perf_data["completion_rates"])
            
            print(f"   {complexity.title()} Tasks:")
            print(f"     Average cost: ${avg_cost:.6f}")
            print(f"     Average iterations: {avg_iterations:.1f}")
            print(f"     Average tools used: {avg_tools:.1f}")
            print(f"     Average success rate: {avg_success:.1%}")
            print(f"     Completion rate: {completion_rate:.1%}")
        
        # Tool usage analysis
        all_tools_used = {}
        for result in session_results:
            for tool_name, tool_cost in result["tool_breakdown"].items():
                if tool_name not in all_tools_used:
                    all_tools_used[tool_name] = {"count": 0, "total_cost": 0}
                all_tools_used[tool_name]["count"] += 1
                all_tools_used[tool_name]["total_cost"] += tool_cost
        
        print(f"\n🛠️ Tool Usage Analysis:")
        for tool_name, usage_data in sorted(all_tools_used.items(), 
                                          key=lambda x: x[1]["total_cost"], reverse=True):
            avg_cost = usage_data["total_cost"] / usage_data["count"]
            print(f"   {tool_name}:")
            print(f"     Times used: {usage_data['count']}")
            print(f"     Total cost: ${usage_data['total_cost']:.6f}")
            print(f"     Average cost per use: ${avg_cost:.6f}")
    
    # Get agent-specific insights (if available)
    print(f"\n🔍 Detailed Agent Insights:")
    for i, result in enumerate(session_results, 1):
        print(f"   Task {i} ({result['complexity']}):")
        print(f"     Decision-making efficiency: {result['tools_used']/result['iterations']:.1f} tools per iteration")
        print(f"     Cost efficiency: ${result['cost']/result['tools_used']:.6f} per tool")
        print(f"     Time efficiency: {result['time']/result['iterations']:.2f}s per iteration")


def demo_multi_agent_coordination():
    """Demonstrate multi-agent coordination and collaboration."""
    print("\n" + "="*70)
    print("🤝 Multi-Agent Coordination")
    print("="*70)
    
    # Create specialized adapters for different agent types
    coordinator_adapter = create_agent_adapter(
        team="agent-coordination",
        project="multi-agent-system",
        daily_budget_limit=75.0
    )
    
    # Simulate coordinated multi-agent workflow
    coordination_tasks = [
        {
            "agent_type": "researcher",
            "task": "Gather information about AI cost optimization strategies",
            "role": "information_gathering"
        },
        {
            "agent_type": "analyzer", 
            "task": "Analyze gathered information and identify key patterns",
            "role": "data_analysis"
        },
        {
            "agent_type": "synthesizer",
            "task": "Synthesize analysis into actionable recommendations",
            "role": "synthesis"
        }
    ]
    
    print("🎭 Simulating multi-agent coordination...")
    
    with coordinator_adapter.track_session("multi-agent-coordination", 
                                         use_case="collaborative-research") as session:
        
        agent_results = {}
        
        for i, agent_config in enumerate(coordination_tasks, 1):
            agent_type = agent_config["agent_type"]
            task = agent_config["task"]
            role = agent_config["role"]
            
            print(f"\n   🤖 Agent {i}: {agent_type.title()} Agent")
            print(f"      Role: {role}")
            print(f"      Task: {task}")
            
            with coordinator_adapter.track_pipeline(
                f"agent-{agent_type}",
                agent_type=agent_type,
                agent_role=role,
                coordination_step=i
            ) as context:
                
                # Simulate agent work based on previous results
                previous_context = ""
                if agent_results:
                    previous_context = "\n\nPrevious agent results:\n" + "\n".join([
                        f"{prev_agent}: {result[:100]}..."
                        for prev_agent, result in agent_results.items()
                    ])
                
                # Create simple pipeline for this agent
                agent_pipeline = Pipeline()
                agent_pipeline.add_component("agent_prompt", PromptBuilder(
                    template=f"""
                    You are a {agent_type} agent working on: {task}
                    
                    Your role: {role}
                    {previous_context}
                    
                    Provide your contribution to this collaborative task:
                    """
                ))
                agent_pipeline.add_component("llm", OpenAIGenerator(
                    model="gpt-3.5-turbo",
                    generation_kwargs={"max_tokens": 150}
                ))
                agent_pipeline.connect("agent_prompt", "llm")
                
                # Execute agent work
                result = agent_pipeline.run({"agent_prompt": {}})
                agent_output = result["llm"]["replies"][0]
                
                agent_results[agent_type] = agent_output
                
                print(f"      📝 Output: {agent_output[:100]}...")
                
                # Add agent-specific metrics
                context.add_custom_metric("agent_type", agent_type)
                context.add_custom_metric("coordination_step", i)
                context.add_custom_metric("depends_on_previous", len(agent_results) > 1)
                
                metrics = context.get_metrics()
                print(f"      💰 Cost: ${metrics.total_cost:.6f}")
            
            session.add_pipeline_result(context.get_metrics())
        
        print(f"\n🎯 Multi-Agent Coordination Summary:")
        print(f"   Agents coordinated: {session.total_pipelines}")
        print(f"   Total coordination cost: ${session.total_cost:.6f}")
        print(f"   Average cost per agent: ${session.total_cost / session.total_pipelines:.6f}")
    
    return coordination_tasks, agent_results


def main():
    """Run the comprehensive agent workflow tracking demonstration."""
    print("🤖 Agent Workflow Tracking with Haystack + GenOps")
    print("="*70)
    
    # Validate environment setup
    print("🔍 Validating setup...")
    result = validate_haystack_setup()
    
    if not result.is_valid:
        print("❌ Setup validation failed!")
        print_validation_result(result)
        return 1
    else:
        print("✅ Environment validated and ready")
    
    try:
        # Main agent workflow demonstration
        agent_adapter, session_results = demo_agent_workflow_tracking()
        
        # Analyze agent performance
        analyze_agent_performance(agent_adapter, session_results)
        
        # Multi-agent coordination
        coordination_tasks, agent_results = demo_multi_agent_coordination()
        
        print("\n🎉 Agent Workflow Tracking demonstration completed!")
        print("\n🚀 Next Steps:")
        print("   • Try multi_provider_cost_aggregation.py for cost optimization")
        print("   • Run enterprise_governance_patterns.py for advanced governance")
        print("   • Explore production_deployment_patterns.py for scaling")
        print("   • Build your own agent system with complete governance! 🤖")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Demonstration interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Demonstration failed: {e}", exc_info=True)
        print(f"\n❌ Demo failed: {e}")
        print("Try running the setup validation to check your configuration")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)