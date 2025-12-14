#!/usr/bin/env python3
"""
SkyRouter Multi-Agent Workflow Routing Example

This example demonstrates advanced multi-agent workflow patterns with SkyRouter
and GenOps governance. Learn how to orchestrate complex AI workflows across
multiple agents, models, and routing strategies with comprehensive cost tracking
and optimization.

Features demonstrated:
- Complex multi-agent workflow orchestration
- Cross-agent cost attribution and optimization
- Workflow-level governance and budget management
- Agent specialization and model selection strategies
- Performance monitoring across multi-step workflows
- Workflow optimization and efficiency analysis

Usage:
    export SKYROUTER_API_KEY="your-api-key"
    export GENOPS_TEAM="your-team"
    export GENOPS_PROJECT="your-project"
    python agent_workflows.py

Author: GenOps AI Contributors
"""

import os
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

@dataclass
class AgentWorkflowStep:
    """Represents a single step in a multi-agent workflow."""
    agent_name: str
    model: str
    task_description: str
    input_data: Dict[str, Any]
    routing_strategy: str
    complexity: str
    depends_on: Optional[List[str]] = None
    expected_output: Optional[str] = None

@dataclass
class WorkflowResult:
    """Results from a completed workflow execution."""
    workflow_name: str
    total_cost: float
    total_duration: float
    steps_completed: int
    steps_failed: int
    agent_costs: Dict[str, float]
    model_usage: Dict[str, int]
    optimization_opportunities: List[str]

def demonstrate_customer_support_workflow():
    """Demonstrate a comprehensive customer support workflow with multiple agents."""
    
    print("🤖 Multi-Agent Customer Support Workflow")
    print("-" * 42)
    
    try:
        from genops.providers.skyrouter import GenOpsSkyRouterAdapter
        
        adapter = GenOpsSkyRouterAdapter(
            team="customer-support",
            project="multi-agent-workflow",
            environment="production",
            daily_budget_limit=150.0,
            governance_policy="enforced"
        )
        
        # Define the customer support workflow
        customer_query = {
            "customer_id": "CUST_12345",
            "message": "I've been charged twice for my subscription this month but only received one service activation. Can you help me understand what happened and get this resolved?",
            "priority": "high",
            "channel": "email",
            "customer_tier": "premium"
        }
        
        # Define workflow steps with specialized agents
        workflow_steps = [
            AgentWorkflowStep(
                agent_name="intent_classifier",
                model="gpt-3.5-turbo",
                task_description="Classify customer intent and extract key entities",
                input_data={
                    "task": "intent_classification",
                    "customer_message": customer_query["message"],
                    "customer_context": {
                        "tier": customer_query["customer_tier"],
                        "priority": customer_query["priority"]
                    }
                },
                routing_strategy="cost_optimized",
                complexity="simple"
            ),
            AgentWorkflowStep(
                agent_name="knowledge_retriever",
                model="claude-3-haiku",
                task_description="Retrieve relevant knowledge base articles",
                input_data={
                    "task": "knowledge_retrieval",
                    "intent": "billing_dispute",
                    "entities": ["subscription", "double_charge", "service_activation"],
                    "customer_tier": customer_query["customer_tier"]
                },
                routing_strategy="latency_optimized",
                complexity="moderate",
                depends_on=["intent_classifier"]
            ),
            AgentWorkflowStep(
                agent_name="solution_generator",
                model="claude-3-sonnet",
                task_description="Generate comprehensive solution with empathetic tone",
                input_data={
                    "task": "solution_generation",
                    "customer_issue": "billing_dispute",
                    "knowledge_context": "billing_policies_and_procedures",
                    "customer_tier": customer_query["customer_tier"],
                    "tone_requirements": ["empathetic", "professional", "solution_focused"]
                },
                routing_strategy="balanced",
                complexity="complex",
                depends_on=["intent_classifier", "knowledge_retriever"]
            ),
            AgentWorkflowStep(
                agent_name="quality_reviewer",
                model="gpt-4",
                task_description="Review solution quality and compliance",
                input_data={
                    "task": "quality_review",
                    "generated_solution": "comprehensive_billing_resolution",
                    "quality_criteria": [
                        "accuracy", "completeness", "empathy", 
                        "compliance", "actionable_steps"
                    ],
                    "customer_tier": customer_query["customer_tier"]
                },
                routing_strategy="reliability_first",
                complexity="enterprise",
                depends_on=["solution_generator"]
            ),
            AgentWorkflowStep(
                agent_name="escalation_detector",
                model="gpt-3.5-turbo",
                task_description="Detect if escalation is needed",
                input_data={
                    "task": "escalation_detection",
                    "issue_complexity": "billing_dispute",
                    "customer_tier": customer_query["customer_tier"],
                    "solution_confidence": "high",
                    "policy_exceptions": []
                },
                routing_strategy="cost_optimized",
                complexity="simple",
                depends_on=["quality_reviewer"]
            )
        ]
        
        print(f"📋 Customer Query: {customer_query['customer_id']}")
        print(f"   💬 Message: {customer_query['message'][:100]}...")
        print(f"   🎯 Priority: {customer_query['priority']}")
        print(f"   ⭐ Tier: {customer_query['customer_tier']}")
        print()
        
        # Execute the workflow
        workflow_results = {}
        total_workflow_cost = 0
        workflow_start = time.time()
        
        with adapter.track_routing_session("customer-support-workflow") as session:
            print("🔄 Executing Multi-Agent Workflow:")
            print()
            
            for i, step in enumerate(workflow_steps, 1):
                step_start = time.time()
                
                print(f"   Step {i}: {step.agent_name}")
                print(f"      🤖 Model: {step.model}")
                print(f"      🎯 Task: {step.task_description}")
                print(f"      🔀 Strategy: {step.routing_strategy}")
                
                # Simulate step execution with cost tracking
                step_result = session.track_agent_workflow(
                    workflow_name=f"step_{i}_{step.agent_name}",
                    agent_steps=[{
                        "model": step.model,
                        "input": step.input_data,
                        "complexity": step.complexity,
                        "optimization": step.routing_strategy
                    }]
                )
                
                step_duration = time.time() - step_start
                
                workflow_results[step.agent_name] = {
                    "cost": float(step_result.total_cost),
                    "duration": step_duration,
                    "model": step.model,
                    "strategy": step.routing_strategy,
                    "status": "completed"
                }
                
                total_workflow_cost += float(step_result.total_cost)
                
                print(f"      ✅ Completed in {step_duration:.1f}s")
                print(f"      💰 Cost: ${step_result.total_cost:.4f}")
                print()
        
        workflow_duration = time.time() - workflow_start
        
        # Analyze workflow results
        print("📊 Workflow Execution Summary:")
        print("-" * 32)
        
        print(f"⏱️  Total duration: {workflow_duration:.1f}s")
        print(f"💰 Total cost: ${total_workflow_cost:.4f}")
        print(f"🔄 Steps completed: {len(workflow_steps)}")
        print(f"📉 Average cost per step: ${total_workflow_cost / len(workflow_steps):.4f}")
        print()
        
        # Cost breakdown by agent
        print("🤖 Cost by Agent:")
        sorted_agents = sorted(workflow_results.items(), key=lambda x: x[1]["cost"], reverse=True)
        for agent_name, result in sorted_agents:
            percentage = (result["cost"] / total_workflow_cost) * 100 if total_workflow_cost > 0 else 0
            print(f"   • {agent_name}: ${result['cost']:.4f} ({percentage:.1f}%)")
        
        print()
        
        # Model usage analysis
        model_usage = {}
        for result in workflow_results.values():
            model = result["model"]
            model_usage[model] = model_usage.get(model, 0) + 1
        
        print("🔧 Model Usage:")
        for model, count in sorted(model_usage.items(), key=lambda x: x[1], reverse=True):
            print(f"   • {model}: {count} step(s)")
        
        return WorkflowResult(
            workflow_name="customer_support",
            total_cost=total_workflow_cost,
            total_duration=workflow_duration,
            steps_completed=len(workflow_steps),
            steps_failed=0,
            agent_costs={name: result["cost"] for name, result in workflow_results.items()},
            model_usage=model_usage,
            optimization_opportunities=[]
        )
        
    except Exception as e:
        print(f"❌ Customer support workflow failed: {e}")
        return None

def demonstrate_content_creation_pipeline():
    """Demonstrate a content creation pipeline with specialized agents."""
    
    print("✍️ Multi-Agent Content Creation Pipeline")
    print("-" * 40)
    
    try:
        from genops.providers.skyrouter import GenOpsSkyRouterAdapter
        
        adapter = GenOpsSkyRouterAdapter(
            team="content-creation",
            project="multi-agent-pipeline",
            daily_budget_limit=200.0
        )
        
        # Content creation request
        content_request = {
            "topic": "The Future of AI in Healthcare",
            "target_audience": "healthcare_professionals",
            "content_type": "technical_blog_post",
            "word_count": 1500,
            "tone": "professional_authoritative",
            "deadline": "2_days",
            "seo_keywords": ["AI healthcare", "medical AI", "healthcare technology"]
        }
        
        print(f"📝 Content Request: {content_request['topic']}")
        print(f"   🎯 Audience: {content_request['target_audience']}")
        print(f"   📄 Type: {content_request['content_type']}")
        print(f"   📏 Length: {content_request['word_count']} words")
        print()
        
        # Define content creation pipeline
        pipeline_steps = [
            {
                "agent": "research_specialist",
                "model": "gpt-4",
                "task": "comprehensive_research",
                "strategy": "reliability_first",
                "complexity": "enterprise"
            },
            {
                "agent": "outline_creator",
                "model": "claude-3-sonnet",
                "task": "structure_planning",
                "strategy": "balanced",
                "complexity": "complex"
            },
            {
                "agent": "content_writer",
                "model": "gpt-4",
                "task": "content_generation",
                "strategy": "reliability_first",
                "complexity": "enterprise"
            },
            {
                "agent": "seo_optimizer",
                "model": "gpt-3.5-turbo",
                "task": "seo_enhancement",
                "strategy": "cost_optimized",
                "complexity": "moderate"
            },
            {
                "agent": "fact_checker",
                "model": "claude-3-opus",
                "task": "accuracy_verification",
                "strategy": "reliability_first",
                "complexity": "enterprise"
            },
            {
                "agent": "editor",
                "model": "gpt-4",
                "task": "final_editing",
                "strategy": "balanced",
                "complexity": "complex"
            }
        ]
        
        pipeline_results = {}
        total_pipeline_cost = 0
        pipeline_start = time.time()
        
        with adapter.track_routing_session("content-creation-pipeline") as session:
            print("🔄 Executing Content Creation Pipeline:")
            print()
            
            for i, step in enumerate(pipeline_steps, 1):
                print(f"   Stage {i}: {step['agent']}")
                print(f"      🎯 Task: {step['task']}")
                print(f"      🤖 Model: {step['model']}")
                
                # Execute pipeline step
                step_result = session.track_agent_workflow(
                    workflow_name=f"content_pipeline_step_{i}",
                    agent_steps=[{
                        "model": step["model"],
                        "input": {
                            "task": step["task"],
                            "content_request": content_request,
                            "previous_outputs": list(pipeline_results.keys())
                        },
                        "complexity": step["complexity"],
                        "optimization": step["strategy"]
                    }]
                )
                
                pipeline_results[step["agent"]] = {
                    "cost": float(step_result.total_cost),
                    "model": step["model"],
                    "strategy": step["strategy"],
                    "task": step["task"]
                }
                
                total_pipeline_cost += float(step_result.total_cost)
                
                print(f"      ✅ ${step_result.total_cost:.4f}")
                print()
        
        pipeline_duration = time.time() - pipeline_start
        
        print("📊 Pipeline Execution Results:")
        print("-" * 33)
        print(f"⏱️  Total duration: {pipeline_duration:.1f}s")
        print(f"💰 Total cost: ${total_pipeline_cost:.4f}")
        print(f"🔄 Stages completed: {len(pipeline_steps)}")
        print()
        
        # Analyze cost distribution
        print("💰 Cost Distribution by Stage:")
        for agent, result in pipeline_results.items():
            percentage = (result["cost"] / total_pipeline_cost) * 100
            print(f"   • {agent}: ${result['cost']:.4f} ({percentage:.1f}%)")
        
        # Strategy effectiveness analysis
        strategy_costs = {}
        for result in pipeline_results.values():
            strategy = result["strategy"]
            strategy_costs[strategy] = strategy_costs.get(strategy, 0) + result["cost"]
        
        print()
        print("🎯 Strategy Cost Analysis:")
        for strategy, total_cost in sorted(strategy_costs.items(), key=lambda x: x[1], reverse=True):
            percentage = (total_cost / total_pipeline_cost) * 100
            print(f"   • {strategy}: ${total_cost:.4f} ({percentage:.1f}%)")
        
        return {
            "pipeline": "content_creation",
            "total_cost": total_pipeline_cost,
            "duration": pipeline_duration,
            "stages": len(pipeline_steps),
            "stage_costs": {k: v["cost"] for k, v in pipeline_results.items()},
            "strategy_distribution": strategy_costs
        }
        
    except Exception as e:
        print(f"❌ Content creation pipeline failed: {e}")
        return None

def demonstrate_parallel_agent_execution():
    """Demonstrate parallel agent execution for improved efficiency."""
    
    print("⚡ Parallel Agent Execution")
    print("-" * 28)
    
    try:
        from genops.providers.skyrouter import GenOpsSkyRouterAdapter
        
        adapter = GenOpsSkyRouterAdapter(
            team="parallel-execution",
            project="concurrent-agents",
            daily_budget_limit=100.0
        )
        
        # Define parallel analysis tasks
        analysis_tasks = [
            {
                "agent": "sentiment_analyzer",
                "model": "gpt-3.5-turbo",
                "task": "Analyze sentiment of customer feedback",
                "data_source": "customer_reviews",
                "strategy": "cost_optimized"
            },
            {
                "agent": "topic_extractor",
                "model": "claude-3-haiku",
                "task": "Extract key topics and themes",
                "data_source": "customer_reviews",
                "strategy": "latency_optimized"
            },
            {
                "agent": "trend_detector",
                "model": "gemini-pro",
                "task": "Detect emerging trends",
                "data_source": "customer_reviews",
                "strategy": "balanced"
            },
            {
                "agent": "competitor_analyzer",
                "model": "claude-3-sonnet",
                "task": "Analyze competitive mentions",
                "data_source": "customer_reviews",
                "strategy": "balanced"
            }
        ]
        
        print("🔄 Executing Parallel Agent Tasks:")
        print()
        
        # Sequential execution (for comparison)
        print("📈 Sequential Execution:")
        sequential_start = time.time()
        sequential_cost = 0
        
        with adapter.track_routing_session("sequential-analysis") as session:
            for i, task in enumerate(analysis_tasks, 1):
                print(f"   Task {i}: {task['agent']}")
                
                result = session.track_model_call(
                    model=task["model"],
                    input_data={
                        "task": task["task"],
                        "data_source": task["data_source"]
                    },
                    route_optimization=task["strategy"],
                    complexity="moderate"
                )
                
                sequential_cost += float(result.total_cost)
                print(f"      ✅ ${result.total_cost:.4f}")
        
        sequential_duration = time.time() - sequential_start
        print(f"   ⏱️  Total time: {sequential_duration:.1f}s")
        print(f"   💰 Total cost: ${sequential_cost:.4f}")
        print()
        
        # Simulated parallel execution
        print("⚡ Parallel Execution (Simulated):")
        parallel_start = time.time()
        parallel_cost = 0
        
        # In real implementation, these would run concurrently
        with adapter.track_routing_session("parallel-analysis") as session:
            parallel_results = []
            
            # Simulate concurrent execution with batch processing
            for task in analysis_tasks:
                result = session.track_model_call(
                    model=task["model"],
                    input_data={
                        "task": task["task"],
                        "data_source": task["data_source"],
                        "execution_mode": "parallel"
                    },
                    route_optimization=task["strategy"],
                    complexity="moderate"
                )
                
                parallel_results.append({
                    "agent": task["agent"],
                    "cost": float(result.total_cost),
                    "model": task["model"]
                })
                parallel_cost += float(result.total_cost)
        
        # Simulate parallel execution time (much faster)
        parallel_duration = max(sequential_duration * 0.3, 1.0)  # Simulate 70% time savings
        
        for result in parallel_results:
            print(f"   Task: {result['agent']} - ${result['cost']:.4f}")
        
        print(f"   ⏱️  Total time: {parallel_duration:.1f}s")
        print(f"   💰 Total cost: ${parallel_cost:.4f}")
        print()
        
        # Performance comparison
        print("📊 Execution Comparison:")
        print("-" * 24)
        
        time_savings = sequential_duration - parallel_duration
        time_improvement = (time_savings / sequential_duration) * 100
        
        print(f"⚡ Time savings: {time_savings:.1f}s ({time_improvement:.1f}% faster)")
        print(f"💰 Cost difference: ${abs(parallel_cost - sequential_cost):.4f}")
        print(f"📈 Efficiency gain: {time_improvement:.1f}% faster execution")
        
        # Parallel execution benefits
        print()
        print("🎯 Parallel Execution Benefits:")
        print("   • Faster overall workflow completion")
        print("   • Better resource utilization")
        print("   • Improved user experience with faster results")
        print("   • Same cost with significantly better performance")
        
        return {
            "sequential_duration": sequential_duration,
            "parallel_duration": parallel_duration,
            "time_savings_percent": time_improvement,
            "sequential_cost": sequential_cost,
            "parallel_cost": parallel_cost
        }
        
    except Exception as e:
        print(f"❌ Parallel execution demo failed: {e}")
        return None

def demonstrate_workflow_optimization():
    """Demonstrate workflow optimization techniques."""
    
    print("🔧 Workflow Optimization Techniques")
    print("-" * 36)
    
    optimization_techniques = [
        {
            "name": "Agent Specialization",
            "description": "Use specialized models for specific agent roles",
            "example": "Use GPT-4 for complex reasoning, GPT-3.5 for simple classification",
            "savings": "20-40% cost reduction"
        },
        {
            "name": "Strategic Route Selection",
            "description": "Choose routing strategies based on step criticality",
            "example": "cost_optimized for preprocessing, reliability_first for final output",
            "savings": "15-30% cost reduction"
        },
        {
            "name": "Conditional Execution",
            "description": "Skip unnecessary steps based on intermediate results",
            "example": "Skip human review if quality score > 95%",
            "savings": "10-25% cost reduction"
        },
        {
            "name": "Batch Processing",
            "description": "Process multiple items together for efficiency",
            "example": "Batch similar customer queries for classification",
            "savings": "25-50% time reduction"
        },
        {
            "name": "Caching Strategies",
            "description": "Cache results for frequently repeated operations",
            "example": "Cache knowledge base searches, model responses",
            "savings": "30-70% for repeated queries"
        }
    ]
    
    print("💡 Workflow Optimization Strategies:")
    print()
    
    for i, technique in enumerate(optimization_techniques, 1):
        print(f"{i}. **{technique['name']}**")
        print(f"   📝 {technique['description']}")
        print(f"   💡 Example: {technique['example']}")
        print(f"   💰 Potential savings: {technique['savings']}")
        print()
    
    # Demonstrate optimization implementation
    print("🧪 Optimization Implementation Example:")
    print()
    
    print("**Before Optimization:**")
    print("```python")
    print("# Basic workflow - all steps use same strategy")
    print("for step in workflow_steps:")
    print("    result = track_agent_workflow(")
    print("        model='gpt-4',  # Expensive for all steps")
    print("        strategy='reliability_first'  # Conservative for all")
    print("    )")
    print("```")
    print()
    
    print("**After Optimization:**")
    print("```python")
    print("# Optimized workflow - strategic model and route selection")
    print("optimization_config = {")
    print("    'classification': {'model': 'gpt-3.5-turbo', 'strategy': 'cost_optimized'},")
    print("    'generation': {'model': 'claude-3-sonnet', 'strategy': 'balanced'},")
    print("    'review': {'model': 'gpt-4', 'strategy': 'reliability_first'}")
    print("}")
    print("")
    print("for step in workflow_steps:")
    print("    config = optimization_config[step.type]")
    print("    result = track_agent_workflow(")
    print("        model=config['model'],")
    print("        strategy=config['strategy']")
    print("    )")
    print("```")
    print()
    
    # Show potential savings
    unoptimized_cost = 0.15  # Example cost for unoptimized workflow
    optimized_cost = 0.09    # Example cost for optimized workflow
    savings = unoptimized_cost - optimized_cost
    savings_percent = (savings / unoptimized_cost) * 100
    
    print(f"📊 **Optimization Impact Example:**")
    print(f"   💰 Before: ${unoptimized_cost:.3f} per workflow")
    print(f"   💰 After: ${optimized_cost:.3f} per workflow")
    print(f"   💾 Savings: ${savings:.3f} ({savings_percent:.1f}% reduction)")
    print(f"   📈 At 1000 workflows/month: ${savings * 1000:.2f} monthly savings")
    
    return True

def main():
    """Main execution function."""
    
    print("🤖 SkyRouter Multi-Agent Workflow Routing Demo")
    print("=" * 50)
    print()
    
    print("This example demonstrates advanced multi-agent workflow patterns")
    print("with intelligent routing, cost optimization, and governance across")
    print("complex AI agent orchestration scenarios.")
    print()
    
    # Check prerequisites
    api_key = os.getenv("SKYROUTER_API_KEY")
    if not api_key:
        print("❌ Missing required environment variables:")
        print("   SKYROUTER_API_KEY - Your SkyRouter API key")
        print()
        print("💡 Set up your environment:")
        print("   export SKYROUTER_API_KEY='your-api-key'")
        print("   export GENOPS_TEAM='agent-workflow-team'")
        print("   export GENOPS_PROJECT='multi-agent-routing'")
        return
    
    try:
        success = True
        results = {}
        
        # Customer support workflow
        if success:
            customer_result = demonstrate_customer_support_workflow()
            if customer_result:
                results["customer_support"] = customer_result
            else:
                success = False
        
        # Content creation pipeline
        if success:
            print("\n" + "="*60 + "\n")
            content_result = demonstrate_content_creation_pipeline()
            if content_result:
                results["content_creation"] = content_result
            else:
                success = False
        
        # Parallel execution
        if success:
            print("\n" + "="*60 + "\n")
            parallel_result = demonstrate_parallel_agent_execution()
            if parallel_result:
                results["parallel_execution"] = parallel_result
            else:
                success = False
        
        # Workflow optimization
        if success:
            print("\n" + "="*60 + "\n")
            demonstrate_workflow_optimization()
        
        if success:
            print("\n" + "="*60 + "\n")
            print("🎉 Multi-Agent Workflow demonstration completed!")
            
            # Overall summary
            if results:
                total_cost = sum([
                    results.get("customer_support", WorkflowResult("", 0, 0, 0, 0, {}, {}, [])).total_cost,
                    results.get("content_creation", {}).get("total_cost", 0)
                ])
                
                print()
                print("📊 **Overall Demo Summary:**")
                if "customer_support" in results:
                    cs_result = results["customer_support"]
                    print(f"   🤖 Customer Support: ${cs_result.total_cost:.4f} ({cs_result.steps_completed} steps)")
                
                if "content_creation" in results:
                    cc_result = results["content_creation"]
                    print(f"   ✍️  Content Creation: ${cc_result['total_cost']:.4f} ({cc_result['stages']} stages)")
                
                if "parallel_execution" in results:
                    pe_result = results["parallel_execution"]
                    print(f"   ⚡ Parallel Execution: {pe_result['time_savings_percent']:.1f}% time savings")
                
                print(f"   💰 Total demo cost: ${total_cost:.4f}")
            
            print()
            print("🔑 **Key Takeaways:**")
            print("• Multi-agent workflows enable sophisticated AI automation")
            print("• Strategic model selection optimizes cost vs performance")
            print("• Parallel execution dramatically improves workflow speed")
            print("• Proper governance ensures accountability across complex workflows")
            print("• Optimization techniques can reduce costs by 20-50%")
            print()
            print("🚀 **Next Steps:**")
            print("1. Design your own multi-agent workflows for your use cases")
            print("2. Implement agent specialization strategies")
            print("3. Set up parallel execution for independent tasks")
            print("4. Try enterprise_patterns.py for production deployment patterns")
            print("5. Explore workflow optimization techniques for your scenarios")
            print()
            print("🏭 **Production Considerations:**")
            print("• Implement proper error handling and retry logic")
            print("• Add workflow state management for long-running processes")
            print("• Set up monitoring and alerting for workflow health")
            print("• Consider workflow versioning for iterative improvements")
            print("• Implement workflow caching for repeated patterns")
        
    except KeyboardInterrupt:
        print()
        print("👋 Demo cancelled.")
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        print()
        print("🔧 Troubleshooting tips:")
        print("1. Verify your SKYROUTER_API_KEY is correct")
        print("2. Check your internet connection")
        print("3. Ensure GenOps is properly installed: pip install genops[skyrouter]")
        print("4. Verify sufficient API credits for multi-step workflows")

if __name__ == "__main__":
    main()