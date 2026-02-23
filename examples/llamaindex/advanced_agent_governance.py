#!/usr/bin/env python3
"""
🤖 GenOps LlamaIndex Advanced Agent Governance - Phase 3 (45 minutes)

This example demonstrates comprehensive agent workflow governance with GenOps.
Track multi-step agent operations, tool usage costs, and complex workflow attribution.

What you'll learn:
- Agent workflow cost tracking across multiple tools and LLM calls
- Multi-step operation governance with nested attribution
- Tool usage monitoring and optimization
- Budget-constrained agent operations
- Agent performance analysis and optimization
- Complex workflow orchestration with cost visibility

Requirements:
- API key: OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY
- pip install llama-index genops-ai

Usage:
    python advanced_agent_governance.py
"""

import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


def setup_llm_provider():
    """Configure LLM provider for agent operations."""
    from llama_index.core import Settings

    provider_info = {}

    if os.getenv("OPENAI_API_KEY"):
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.llms.openai import OpenAI

        Settings.llm = OpenAI(
            model="gpt-4", temperature=0.1
        )  # Use GPT-4 for better agent reasoning
        Settings.embed_model = OpenAIEmbedding()
        provider_info = {
            "name": "OpenAI",
            "llm_model": "gpt-4",
            "embedding_model": "text-embedding-ada-002",
            "reasoning_quality": "high",
            "cost_profile": "premium",
        }
    elif os.getenv("ANTHROPIC_API_KEY"):
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.llms.anthropic import Anthropic

        Settings.llm = Anthropic(model="claude-3-sonnet-20240229", temperature=0.1)
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        provider_info = {
            "name": "Anthropic",
            "llm_model": "claude-3-sonnet",
            "embedding_model": "all-MiniLM-L6-v2",
            "reasoning_quality": "high",
            "cost_profile": "balanced",
        }
    elif os.getenv("GOOGLE_API_KEY"):
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.llms.gemini import Gemini

        Settings.llm = Gemini(model="gemini-pro", temperature=0.1)
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        provider_info = {
            "name": "Google",
            "llm_model": "gemini-pro",
            "embedding_model": "all-MiniLM-L6-v2",
            "reasoning_quality": "medium",
            "cost_profile": "cost_effective",
        }
    else:
        raise ValueError(
            "No API key found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY"
        )

    return provider_info


@dataclass
class AgentOperationMetrics:
    """Comprehensive metrics for agent operations."""

    operation_id: str
    agent_name: str
    start_time: datetime
    end_time: Optional[datetime] = None

    # Cost tracking
    total_cost: float = 0.0
    llm_calls: int = 0
    llm_cost: float = 0.0
    tool_calls: int = 0
    tool_cost: float = 0.0
    embedding_calls: int = 0
    embedding_cost: float = 0.0

    # Performance metrics
    steps_executed: int = 0
    reasoning_time_ms: float = 0.0
    tool_execution_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # Quality metrics
    success: bool = False
    reasoning_quality: float = 0.0
    tool_usage_efficiency: float = 0.0

    # Attribution
    team: Optional[str] = None
    project: Optional[str] = None
    customer_id: Optional[str] = None
    workflow_type: Optional[str] = None

    def finalize(self):
        """Finalize metrics calculation."""
        if self.end_time and self.start_time:
            self.total_time_ms = (
                self.end_time - self.start_time
            ).total_seconds() * 1000

        self.total_cost = self.llm_cost + self.tool_cost + self.embedding_cost

        # Calculate efficiency metrics
        if self.tool_calls > 0:
            self.tool_usage_efficiency = min(
                1.0, self.steps_executed / (self.tool_calls * 2)
            )  # Ideal: 2 tool calls per meaningful step

        # Simple reasoning quality heuristic
        if self.total_time_ms > 0:
            self.reasoning_quality = min(
                1.0, (self.reasoning_time_ms / self.total_time_ms) * 2
            )  # Prefer more reasoning time


class MockCalculatorTool:
    """Mock calculator tool for agent demonstrations."""

    def __init__(self, cost_per_operation: float = 0.001):
        self.cost_per_operation = cost_per_operation
        self.operations_count = 0

    def calculate(self, expression: str) -> dict[str, Any]:
        """Perform calculation and return result with cost tracking."""
        self.operations_count += 1

        try:
            # Simple expression evaluation (DEMO ONLY - NOT SECURE)
            result = eval(expression.replace("^", "**"))  # Convert ^ to ** for Python

            return {
                "result": result,
                "expression": expression,
                "cost": self.cost_per_operation,
                "operation_id": f"calc_{self.operations_count}",
            }
        except Exception as e:
            return {
                "error": str(e),
                "expression": expression,
                "cost": self.cost_per_operation,
                "operation_id": f"calc_error_{self.operations_count}",
            }


class MockDocumentSearchTool:
    """Mock document search tool for agent demonstrations."""

    def __init__(self, cost_per_search: float = 0.002):
        self.cost_per_search = cost_per_search
        self.search_count = 0
        self.mock_database = {
            "revenue": "Q3 2024 revenue was $2.3M, up 23% from Q2",
            "expenses": "Q3 2024 total expenses were $1.8M, including $400K in new hiring",
            "customers": "Customer base grew to 1,240 active customers, with 15% churn rate",
            "products": "Three products launched: Analytics Pro, Data Sync, and Mobile Dashboard",
            "team": "Engineering team expanded from 12 to 18 people, Sales team added 3 reps",
            "market": "Competitive landscape shows 5 major competitors, we hold 12% market share",
        }

    def search(self, query: str) -> dict[str, Any]:
        """Search documents and return relevant information."""
        self.search_count += 1

        # Simple keyword matching
        query_lower = query.lower()
        results = []

        for key, content in self.mock_database.items():
            if any(word in content.lower() for word in query_lower.split()):
                results.append(
                    {
                        "document": key,
                        "content": content,
                        "relevance": 0.8,  # Mock relevance score
                    }
                )

        return {
            "results": results,
            "query": query,
            "total_results": len(results),
            "cost": self.cost_per_search,
            "search_id": f"search_{self.search_count}",
        }


class MockWebSearchTool:
    """Mock web search tool for agent demonstrations."""

    def __init__(self, cost_per_search: float = 0.005):
        self.cost_per_search = cost_per_search
        self.search_count = 0

    def search_web(self, query: str) -> dict[str, Any]:
        """Perform web search and return mock results."""
        self.search_count += 1

        # Mock web search results
        mock_results = [
            {
                "title": f"Industry Analysis: {query.title()}",
                "url": f"https://example.com/industry-{query.replace(' ', '-').lower()}",
                "snippet": f"Comprehensive analysis of {query} trends and market data...",
                "source": "MarketResearch.com",
            },
            {
                "title": f"Latest {query} Statistics and Insights",
                "url": f"https://stats.example.com/{query.replace(' ', '-').lower()}",
                "snippet": f"Recent statistics and key insights about {query} performance...",
                "source": "BusinessStats.org",
            },
        ]

        return {
            "results": mock_results,
            "query": query,
            "total_results": len(mock_results),
            "cost": self.cost_per_search,
            "search_id": f"web_{self.search_count}",
        }


class GenOpsAgentWorkflowTracker:
    """Advanced agent workflow tracking with GenOps integration."""

    def __init__(self, workflow_name: str, budget_limit: Optional[float] = None):
        self.workflow_name = workflow_name
        self.budget_limit = budget_limit
        self.active_operations: dict[str, AgentOperationMetrics] = {}
        self.completed_operations: list[AgentOperationMetrics] = []
        self.total_cost = 0.0

        # Tool instances
        self.calculator = MockCalculatorTool()
        self.doc_search = MockDocumentSearchTool()
        self.web_search = MockWebSearchTool()

    def start_operation(
        self, operation_id: str, agent_name: str, **governance_attrs
    ) -> AgentOperationMetrics:
        """Start tracking a new agent operation."""
        metrics = AgentOperationMetrics(
            operation_id=operation_id,
            agent_name=agent_name,
            start_time=datetime.now(),
            **governance_attrs,
        )

        self.active_operations[operation_id] = metrics
        return metrics

    def record_llm_call(
        self, operation_id: str, cost: float, reasoning_time_ms: float = 0
    ):
        """Record an LLM call within an operation."""
        if operation_id in self.active_operations:
            metrics = self.active_operations[operation_id]
            metrics.llm_calls += 1
            metrics.llm_cost += cost
            metrics.reasoning_time_ms += reasoning_time_ms

    def record_tool_call(
        self,
        operation_id: str,
        tool_name: str,
        cost: float,
        execution_time_ms: float = 0,
    ) -> dict[str, Any]:
        """Record and execute a tool call."""
        if operation_id in self.active_operations:
            metrics = self.active_operations[operation_id]
            metrics.tool_calls += 1
            metrics.tool_cost += cost
            metrics.tool_execution_time_ms += execution_time_ms
            metrics.steps_executed += 1

        # Execute tool based on type
        if tool_name == "calculator":
            return {"tool": "calculator", "available": True}
        elif tool_name == "document_search":
            return {"tool": "document_search", "available": True}
        elif tool_name == "web_search":
            return {"tool": "web_search", "available": True}
        else:
            return {"tool": tool_name, "available": False, "error": "Tool not found"}

    def finish_operation(
        self, operation_id: str, success: bool = True
    ) -> AgentOperationMetrics:
        """Complete an operation and calculate final metrics."""
        if operation_id not in self.active_operations:
            raise ValueError(f"Operation {operation_id} not found")

        metrics = self.active_operations[operation_id]
        metrics.end_time = datetime.now()
        metrics.success = success
        metrics.finalize()

        self.total_cost += metrics.total_cost
        self.completed_operations.append(metrics)
        del self.active_operations[operation_id]

        # Check budget constraints
        if self.budget_limit and self.total_cost > self.budget_limit:
            print(
                f"⚠️  Budget limit exceeded: ${self.total_cost:.6f} > ${self.budget_limit:.6f}"
            )

        return metrics

    def get_workflow_summary(self) -> dict[str, Any]:
        """Get comprehensive workflow summary."""
        total_operations = len(self.completed_operations)
        if total_operations == 0:
            return {"error": "No completed operations"}

        # Aggregate metrics
        total_llm_calls = sum(op.llm_calls for op in self.completed_operations)
        total_tool_calls = sum(op.tool_calls for op in self.completed_operations)
        total_steps = sum(op.steps_executed for op in self.completed_operations)

        avg_cost = self.total_cost / total_operations
        success_rate = (
            sum(1 for op in self.completed_operations if op.success) / total_operations
        )

        # Cost breakdown
        total_llm_cost = sum(op.llm_cost for op in self.completed_operations)
        total_tool_cost = sum(op.tool_cost for op in self.completed_operations)
        total_embedding_cost = sum(
            op.embedding_cost for op in self.completed_operations
        )

        return {
            "workflow_name": self.workflow_name,
            "total_operations": total_operations,
            "total_cost": self.total_cost,
            "average_cost_per_operation": avg_cost,
            "success_rate": success_rate,
            "budget_utilization": self.total_cost / self.budget_limit
            if self.budget_limit
            else None,
            # Operation stats
            "total_llm_calls": total_llm_calls,
            "total_tool_calls": total_tool_calls,
            "total_steps": total_steps,
            "avg_steps_per_operation": total_steps / total_operations,
            # Cost breakdown
            "cost_breakdown": {
                "llm_cost": total_llm_cost,
                "tool_cost": total_tool_cost,
                "embedding_cost": total_embedding_cost,
            },
            # Performance metrics
            "avg_reasoning_quality": sum(
                op.reasoning_quality for op in self.completed_operations
            )
            / total_operations,
            "avg_tool_efficiency": sum(
                op.tool_usage_efficiency for op in self.completed_operations
            )
            / total_operations,
        }


def simulate_research_agent_workflow(tracker: GenOpsAgentWorkflowTracker) -> None:
    """Simulate a comprehensive research agent workflow."""
    print("🔍 RESEARCH AGENT WORKFLOW")
    print("=" * 50)

    # Research task: Analyze Q3 business performance
    operation_id = "research_q3_analysis"

    print("🤖 Agent: Business Research Assistant")
    print("📋 Task: Analyze Q3 2024 performance and create recommendations")

    # Start operation tracking
    tracker.start_operation(
        operation_id,
        "BusinessResearchAgent",
        team="business-intelligence",
        project="quarterly-analysis",
        customer_id="internal",
        workflow_type="research",
    )

    # Step 1: Initial planning (LLM reasoning)
    print("\n🧠 Step 1: Planning research approach...")
    start_time = time.time()

    # Simulate LLM planning call
    time.sleep(0.5)  # Simulate processing time
    planning_time = (time.time() - start_time) * 1000
    tracker.record_llm_call(operation_id, 0.015, planning_time)  # $0.015 for planning

    print(f"   ✅ Research plan created (${0.015:.3f}, {planning_time:.0f}ms)")

    # Step 2: Search company documents
    print("\n📄 Step 2: Searching internal documents...")
    start_time = time.time()

    search_results = tracker.doc_search.search("Q3 2024 revenue expenses")
    execution_time = (time.time() - start_time) * 1000
    tracker.record_tool_call(
        operation_id, "document_search", search_results["cost"], execution_time
    )

    print(f"   📊 Found {search_results['total_results']} relevant documents")
    for result in search_results["results"][:2]:  # Show first 2
        print(f"      • {result['document']}: {result['content'][:60]}...")
    print(f"   💰 Cost: ${search_results['cost']:.3f}, Time: {execution_time:.0f}ms")

    # Step 3: Analyze document data (LLM reasoning)
    print("\n🧠 Step 3: Analyzing document data...")
    start_time = time.time()

    time.sleep(0.7)  # Simulate analysis time
    analysis_time = (time.time() - start_time) * 1000
    tracker.record_llm_call(operation_id, 0.025, analysis_time)  # $0.025 for analysis

    print(f"   ✅ Document analysis complete (${0.025:.3f}, {analysis_time:.0f}ms)")

    # Step 4: Perform calculations
    print("\n🧮 Step 4: Calculating key metrics...")
    start_time = time.time()

    # Calculate profit margin
    calc_result = tracker.calculator.calculate("(2.3 - 1.8) / 2.3 * 100")
    execution_time = (time.time() - start_time) * 1000
    tracker.record_tool_call(
        operation_id, "calculator", calc_result["cost"], execution_time
    )

    print(f"   📊 Profit Margin: {calc_result['result']:.1f}%")
    print(f"   💰 Cost: ${calc_result['cost']:.3f}, Time: {execution_time:.0f}ms")

    # Step 5: Market research
    print("\n🌐 Step 5: Gathering market intelligence...")
    start_time = time.time()

    web_results = tracker.web_search.search_web("SaaS market trends Q3 2024")
    execution_time = (time.time() - start_time) * 1000
    tracker.record_tool_call(
        operation_id, "web_search", web_results["cost"], execution_time
    )

    print(f"   🔍 Found {web_results['total_results']} market insights")
    for result in web_results["results"]:
        print(f"      • {result['title']} - {result['source']}")
    print(f"   💰 Cost: ${web_results['cost']:.3f}, Time: {execution_time:.0f}ms")

    # Step 6: Final synthesis (LLM reasoning)
    print("\n🧠 Step 6: Synthesizing recommendations...")
    start_time = time.time()

    time.sleep(0.8)  # Simulate synthesis time
    synthesis_time = (time.time() - start_time) * 1000
    tracker.record_llm_call(operation_id, 0.030, synthesis_time)  # $0.030 for synthesis

    print(f"   📝 Research report generated (${0.030:.3f}, {synthesis_time:.0f}ms)")

    # Complete operation
    final_metrics = tracker.finish_operation(operation_id, success=True)

    # Display operation summary
    print("\n📊 OPERATION SUMMARY:")
    print(f"   Total Cost: ${final_metrics.total_cost:.6f}")
    print(f"   LLM Calls: {final_metrics.llm_calls} (${final_metrics.llm_cost:.6f})")
    print(f"   Tool Calls: {final_metrics.tool_calls} (${final_metrics.tool_cost:.6f})")
    print(f"   Steps Executed: {final_metrics.steps_executed}")
    print(f"   Total Time: {final_metrics.total_time_ms:.0f}ms")
    print(f"   Reasoning Quality: {final_metrics.reasoning_quality:.2f}")
    print(f"   Tool Efficiency: {final_metrics.tool_usage_efficiency:.2f}")


def simulate_customer_support_agent_workflow(
    tracker: GenOpsAgentWorkflowTracker,
) -> None:
    """Simulate customer support agent handling complex inquiry."""
    print("\n" + "=" * 50)
    print("🎧 CUSTOMER SUPPORT AGENT WORKFLOW")
    print("=" * 50)

    operation_id = "support_pricing_inquiry"

    print("🤖 Agent: Customer Support Assistant")
    print(
        "🎫 Ticket: Enterprise customer asking about pricing tiers and feature comparison"
    )

    # Start operation tracking
    tracker.start_operation(
        operation_id,
        "CustomerSupportAgent",
        team="customer-success",
        project="tier1-support",
        customer_id="enterprise-customer-456",
        workflow_type="support",
    )

    # Step 1: Understand customer inquiry (LLM reasoning)
    print("\n🧠 Step 1: Understanding customer inquiry...")
    start_time = time.time()

    time.sleep(0.3)
    reasoning_time = (time.time() - start_time) * 1000
    tracker.record_llm_call(operation_id, 0.008, reasoning_time)

    print(f"   ✅ Customer intent classified (${0.008:.3f}, {reasoning_time:.0f}ms)")

    # Step 2: Search product documentation
    print("\n📚 Step 2: Searching product information...")
    start_time = time.time()

    search_results = tracker.doc_search.search("products pricing features")
    execution_time = (time.time() - start_time) * 1000
    tracker.record_tool_call(
        operation_id, "document_search", search_results["cost"], execution_time
    )

    print(
        f"   📊 Found product documentation: {search_results['total_results']} results"
    )
    print(f"   💰 Cost: ${search_results['cost']:.3f}, Time: {execution_time:.0f}ms")

    # Step 3: Calculate pricing scenarios
    print("\n🧮 Step 3: Calculating pricing scenarios...")
    start_time = time.time()

    calc_result = tracker.calculator.calculate("149 * 12")  # Annual pricing
    execution_time = (time.time() - start_time) * 1000
    tracker.record_tool_call(
        operation_id, "calculator", calc_result["cost"], execution_time
    )

    print(f"   💰 Annual price: ${calc_result['result']}")
    print(f"   💰 Tool cost: ${calc_result['cost']:.3f}, Time: {execution_time:.0f}ms")

    # Step 4: Generate personalized response (LLM reasoning)
    print("\n🧠 Step 4: Crafting personalized response...")
    start_time = time.time()

    time.sleep(0.5)
    response_time = (time.time() - start_time) * 1000
    tracker.record_llm_call(operation_id, 0.012, response_time)

    print(f"   📝 Personalized response created (${0.012:.3f}, {response_time:.0f}ms)")

    # Complete operation
    final_metrics = tracker.finish_operation(operation_id, success=True)

    print("\n📊 OPERATION SUMMARY:")
    print(f"   Total Cost: ${final_metrics.total_cost:.6f}")
    print(f"   Customer ID: {final_metrics.customer_id}")
    print(f"   Resolution Time: {final_metrics.total_time_ms:.0f}ms")
    print(f"   Tool Efficiency: {final_metrics.tool_usage_efficiency:.2f}")


def simulate_budget_constrained_workflow(tracker: GenOpsAgentWorkflowTracker) -> None:
    """Simulate agent workflow with budget constraints and optimization."""
    print("\n" + "=" * 50)
    print("💰 BUDGET-CONSTRAINED WORKFLOW")
    print("=" * 50)

    # Set strict budget
    remaining_budget = 0.020  # $0.02 budget
    print(f"📊 Budget Limit: ${remaining_budget:.3f}")
    print(f"📊 Current Workflow Spend: ${tracker.total_cost:.6f}")
    print(f"📊 Available Budget: ${remaining_budget - tracker.total_cost:.6f}")

    if tracker.total_cost >= remaining_budget:
        print("⚠️  Budget exhausted - cannot execute workflow")
        return

    operation_id = "budget_constrained_analysis"

    # Start operation with budget monitoring
    tracker.start_operation(
        operation_id,
        "BudgetOptimizedAgent",
        team="cost-optimization",
        project="budget-demo",
        customer_id="demo",
        workflow_type="constrained",
    )

    print("\n🤖 Agent: Budget-Optimized Research Assistant")
    print("📋 Task: Quick market analysis with strict cost controls")

    # Step 1: Lightweight analysis
    print("\n🧠 Step 1: Lightweight analysis (cost-optimized)...")

    # Check budget before expensive operation
    if tracker.total_cost + 0.005 > remaining_budget:
        print("   ⚠️  Skipping expensive LLM reasoning - using cached patterns")
        tracker.record_llm_call(operation_id, 0.002, 100)  # Cheap cached response
    else:
        tracker.record_llm_call(operation_id, 0.005, 300)  # Normal reasoning

    # Step 2: Single focused search
    print("\n📄 Step 2: Focused document search...")
    search_results = tracker.doc_search.search("market share")
    tracker.record_tool_call(
        operation_id, "document_search", search_results["cost"], 150
    )

    print(
        f"   📊 Found {search_results['total_results']} results (${search_results['cost']:.3f})"
    )

    # Check if we can afford final step
    projected_final_cost = tracker.total_cost + 0.008  # Estimated final synthesis cost

    if projected_final_cost > remaining_budget:
        print(
            f"\n⚠️  Budget constraint: Projected cost ${projected_final_cost:.6f} > Budget ${remaining_budget:.3f}"
        )
        print("   🔄 Switching to template-based response...")
        tracker.record_llm_call(operation_id, 0.001, 50)  # Template response
        print(f"   📝 Template response generated (${0.001:.3f})")
    else:
        print("\n🧠 Step 3: Full synthesis within budget...")
        tracker.record_llm_call(operation_id, 0.008, 400)
        print(f"   📝 Complete analysis generated (${0.008:.3f})")

    # Complete operation
    final_metrics = tracker.finish_operation(operation_id, success=True)

    print("\n📊 BUDGET-CONSTRAINED RESULTS:")
    print(f"   Operation Cost: ${final_metrics.total_cost:.6f}")
    print(f"   Total Workflow Cost: ${tracker.total_cost:.6f}")
    print(f"   Budget Utilization: {tracker.total_cost / remaining_budget * 100:.1f}%")
    print(
        f"   Under Budget: {'✅ Yes' if tracker.total_cost <= remaining_budget else '❌ No'}"
    )


def main():
    """Main demonstration of advanced agent governance."""
    print("🤖 GenOps LlamaIndex Advanced Agent Governance")
    print("=" * 60)

    try:
        # Setup
        provider_info = setup_llm_provider()
        print(f"✅ Provider: {provider_info['name']}")
        print(
            f"✅ LLM Model: {provider_info['llm_model']} ({provider_info['reasoning_quality']} reasoning)"
        )
        print(f"✅ Cost Profile: {provider_info['cost_profile']}")

        # Initialize workflow tracker with budget
        workflow_tracker = GenOpsAgentWorkflowTracker(
            "multi_agent_demo",
            budget_limit=0.100,  # $0.10 budget for demo
        )

        print(
            f"✅ Agent Workflow Tracker initialized with ${workflow_tracker.budget_limit:.3f} budget"
        )

        # Demo 1: Research Agent Workflow
        simulate_research_agent_workflow(workflow_tracker)

        # Demo 2: Customer Support Agent Workflow
        simulate_customer_support_agent_workflow(workflow_tracker)

        # Demo 3: Budget-Constrained Workflow
        simulate_budget_constrained_workflow(workflow_tracker)

        # Final workflow summary
        workflow_summary = workflow_tracker.get_workflow_summary()

        print("\n" + "=" * 60)
        print("🎉 ADVANCED AGENT GOVERNANCE COMPLETE!")
        print("=" * 60)

        print("📊 WORKFLOW ANALYTICS:")
        print(f"   Total Operations: {workflow_summary['total_operations']}")
        print(f"   Total Cost: ${workflow_summary['total_cost']:.6f}")
        print(f"   Success Rate: {workflow_summary['success_rate']:.1%}")
        print(f"   Budget Utilization: {workflow_summary['budget_utilization']:.1%}")
        print()
        print(
            f"   LLM Calls: {workflow_summary['total_llm_calls']} (${workflow_summary['cost_breakdown']['llm_cost']:.6f})"
        )
        print(
            f"   Tool Calls: {workflow_summary['total_tool_calls']} (${workflow_summary['cost_breakdown']['tool_cost']:.6f})"
        )
        print(
            f"   Avg Steps/Operation: {workflow_summary['avg_steps_per_operation']:.1f}"
        )
        print()
        print(f"   Reasoning Quality: {workflow_summary['avg_reasoning_quality']:.2f}")
        print(f"   Tool Efficiency: {workflow_summary['avg_tool_efficiency']:.2f}")

        print("\n✅ WHAT YOU ACCOMPLISHED:")
        print("   • Multi-step agent workflow cost tracking")
        print("   • Tool usage monitoring and optimization")
        print("   • Budget-constrained agent operations")
        print("   • Cross-operation governance and attribution")
        print("   • Agent performance analysis and optimization")
        print("   • Complex workflow orchestration with cost visibility")

        print("\n🎯 KEY INSIGHTS:")
        print("   • Agent workflows require multi-component cost tracking")
        print("   • Budget constraints enable dynamic optimization strategies")
        print("   • Tool efficiency metrics help optimize agent performance")
        print("   • Customer attribution enables per-client cost analysis")
        print("   • Reasoning quality correlates with operation success")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")

        if "api key" in str(e).lower():
            print("\n🔧 API KEY ISSUE:")
            print("   Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY")
            print("   Note: GPT-4 or Claude-3 recommended for advanced agent reasoning")
        else:
            print("\n🔧 For detailed diagnostics run:")
            print(
                '   python -c "from genops.providers.llamaindex.validation import validate_setup, print_validation_result; print_validation_result(validate_setup(), detailed=True)"'
            )

        return False


if __name__ == "__main__":
    success = main()

    if success:
        print("\n🚀 CONTINUE WITH ADVANCED PHASE 3:")
        print(
            "   → python multi_modal_rag.py                   # Multi-modal RAG workflows"
        )
        print(
            "   → python production_rag_deployment.py         # Enterprise deployment"
        )
        print()
        print("🔄 Or revisit earlier phases:")
        print(
            "   → python rag_pipeline_tracking.py             # Complete RAG monitoring"
        )
        print(
            "   → python embedding_cost_optimization.py       # Embedding optimization"
        )
    else:
        print("\n💡 Need help?")
        print("   → examples/llamaindex/README.md#troubleshooting")

    exit(0 if success else 1)
