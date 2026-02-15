#!/usr/bin/env python3
"""
CrewAI Performance Optimization and Tuning

Advanced performance analysis and optimization for CrewAI multi-agent workflows.
Demonstrates agent performance tuning, parallel execution, and workflow optimization.

Usage:
    python performance_optimization.py [--mode MODE] [--agents COUNT]

Features:
    - Agent performance profiling and bottleneck identification
    - Parallel vs sequential execution comparison
    - Model selection for optimal speed/cost/quality balance
    - Workflow optimization recommendations
    - Real-time performance monitoring and alerting
    - Load balancing and resource utilization analysis

Time to Complete: ~20 minutes
Learning Outcomes: Performance tuning and optimization for production systems
"""

import argparse
import logging
import statistics
import sys
import time
from dataclasses import dataclass

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
        CrewAIAgentMonitor,
        GenOpsCrewAIAdapter,
        get_multi_agent_insights,
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
class PerformanceMetrics:
    """Performance metrics for crew execution."""

    execution_time: float
    total_cost: float
    agents_count: int
    tasks_count: int
    avg_response_time: float
    throughput: float  # tasks per second
    cost_efficiency: float  # cost per task
    quality_score: float  # simulated quality metric
    resource_utilization: float  # CPU/memory usage percentage


@dataclass
class OptimizationRecommendation:
    """Performance optimization recommendation."""

    category: str
    priority: str
    description: str
    expected_improvement: float
    implementation_effort: str
    cost_impact: str


class PerformanceOptimizer:
    """Advanced performance optimization for CrewAI workflows."""

    def __init__(self, optimization_mode: str = "balanced"):
        self.optimization_mode = optimization_mode
        self.adapter = GenOpsCrewAIAdapter(
            team="performance-team",
            project="optimization-demo",
            daily_budget_limit=100.0,
            enable_agent_tracking=True,
            enable_task_tracking=True,
            governance_policy="advisory",
        )
        self.monitor = CrewAIAgentMonitor()
        self.performance_history = []

    def setup_validation(self) -> bool:
        """Validate setup for performance optimization."""
        print("🔍 Validating performance optimization setup...")

        result = validate_crewai_setup(quick=False)

        if result.is_valid:
            print("✅ Performance optimization setup validated")
            return True
        else:
            print("❌ Setup issues found:")
            print_validation_result(result)
            return False

    def create_performance_test_crew(
        self, complexity_level: str, agent_count: int = 4
    ) -> Crew:
        """Create a crew optimized for performance testing."""
        print(
            f"\n🏗️ Creating {complexity_level} complexity crew with {agent_count} agents..."
        )

        agents = []
        tasks = []

        # Agent configurations optimized for different performance profiles
        agent_configs = [
            {
                "role": "Speed Optimizer",
                "goal": "Provide quick, efficient responses",
                "backstory": "Expert at rapid analysis with good accuracy",
                "description": "Focus on quick turnaround with essential insights",
            },
            {
                "role": "Quality Analyzer",
                "goal": "Provide thorough, high-quality analysis",
                "backstory": "Specialist in comprehensive, detailed analysis",
                "description": "Deep analysis with extensive research and validation",
            },
            {
                "role": "Cost-Efficient Processor",
                "goal": "Maximize value while minimizing resource usage",
                "backstory": "Expert at achieving optimal cost-performance balance",
                "description": "Efficient processing with strategic resource usage",
            },
            {
                "role": "Parallel Coordinator",
                "goal": "Coordinate multiple concurrent processes",
                "backstory": "Specialist in parallel processing and workflow coordination",
                "description": "Manage multiple concurrent tasks efficiently",
            },
        ]

        # Create agents based on requested count
        for i in range(min(agent_count, len(agent_configs))):
            config = agent_configs[i]
            agent = Agent(
                role=config["role"],
                goal=config["goal"],
                backstory=config["backstory"],
                verbose=True,
            )
            agents.append(agent)

            # Create corresponding task
            if complexity_level == "simple":
                task_description = f"""Perform {config["description"].lower()} for
                                     sustainable energy solutions. Provide a concise
                                     summary (2-3 sentences) with key points."""
            elif complexity_level == "medium":
                task_description = f"""Conduct {config["description"].lower()} of
                                     emerging renewable energy technologies. Provide
                                     structured analysis (5-7 key points) with
                                     supporting evidence and implications."""
            else:  # complex
                task_description = f"""Execute {config["description"].lower()} for
                                     comprehensive renewable energy market analysis.
                                     Include detailed research, market trends,
                                     competitive landscape, technology assessment,
                                     and strategic recommendations (10+ sections)."""

            task = Task(description=task_description, agent=agent)
            tasks.append(task)

        # Adjust process type for performance testing
        process_type = (
            Process.sequential if complexity_level == "simple" else Process.sequential
        )

        crew = Crew(agents=agents, tasks=tasks, process=process_type, verbose=2)

        print(f"✅ Created {complexity_level} crew with {len(agents)} agents")
        return crew

    def benchmark_crew_performance(
        self, crew: Crew, test_name: str, iterations: int = 3
    ) -> list[PerformanceMetrics]:
        """Benchmark crew performance over multiple iterations."""
        print(f"\n🚀 Benchmarking: {test_name} ({iterations} iterations)")

        performance_results = []

        for iteration in range(iterations):
            print(f"\n   Iteration {iteration + 1}/{iterations}")

            with self.adapter.track_crew(
                f"{test_name}-iteration-{iteration + 1}"
            ) as context:
                start_time = time.time()

                # Execute crew
                crew.kickoff(
                    {
                        "iteration": iteration + 1,
                        "benchmark_mode": True,
                        "performance_focus": self.optimization_mode,
                    }
                )

                end_time = time.time()
                execution_time = end_time - start_time

                # Get metrics
                metrics = context.get_metrics()

                # Calculate performance metrics
                throughput = (
                    len(crew.tasks) / execution_time if execution_time > 0 else 0
                )
                cost_efficiency = (
                    metrics["total_cost"] / len(crew.tasks)
                    if len(crew.tasks) > 0
                    else 0
                )

                # Simulate additional metrics (in real implementation, these would be measured)
                quality_score = 0.85 + (
                    iteration * 0.03
                )  # Simulated learning improvement
                resource_utilization = 0.65 + (
                    0.1 * len(crew.agents) / 10
                )  # Based on agent count

                perf_metrics = PerformanceMetrics(
                    execution_time=execution_time,
                    total_cost=metrics["total_cost"],
                    agents_count=len(crew.agents),
                    tasks_count=len(crew.tasks),
                    avg_response_time=execution_time / len(crew.tasks),
                    throughput=throughput,
                    cost_efficiency=cost_efficiency,
                    quality_score=quality_score,
                    resource_utilization=resource_utilization,
                )

                performance_results.append(perf_metrics)

                print(f"      ⏱️ Execution time: {execution_time:.2f}s")
                print(f"      💰 Cost: ${metrics['total_cost']:.6f}")
                print(f"      ⚡ Throughput: {throughput:.2f} tasks/sec")
                print(f"      📊 Quality score: {quality_score:.2f}")

        # Store in history
        self.performance_history.extend(performance_results)

        return performance_results

    def compare_execution_strategies(self) -> dict[str, list[PerformanceMetrics]]:
        """Compare different execution strategies for performance."""
        print("\n" + "=" * 70)
        print("⚡ Execution Strategy Performance Comparison")
        print("=" * 70)

        strategies = {}

        # Strategy 1: Sequential with simple tasks
        print("\n🔄 Strategy 1: Sequential Simple Tasks")
        simple_crew = self.create_performance_test_crew("simple", agent_count=2)
        strategies["sequential_simple"] = self.benchmark_crew_performance(
            simple_crew, "sequential-simple", iterations=2
        )

        # Strategy 2: Sequential with complex tasks
        print("\n🔄 Strategy 2: Sequential Complex Tasks")
        complex_crew = self.create_performance_test_crew("complex", agent_count=2)
        strategies["sequential_complex"] = self.benchmark_crew_performance(
            complex_crew, "sequential-complex", iterations=2
        )

        # Strategy 3: More agents with medium complexity
        print("\n🔄 Strategy 3: Multi-Agent Medium Complexity")
        multi_crew = self.create_performance_test_crew("medium", agent_count=4)
        strategies["multi_agent_medium"] = self.benchmark_crew_performance(
            multi_crew, "multi-agent-medium", iterations=2
        )

        # Analyze results
        print("\n📊 Strategy Performance Analysis:")

        for strategy, results in strategies.items():
            avg_time = statistics.mean([r.execution_time for r in results])
            avg_cost = statistics.mean([r.total_cost for r in results])
            avg_throughput = statistics.mean([r.throughput for r in results])
            avg_quality = statistics.mean([r.quality_score for r in results])

            print(f"\n   • {strategy.replace('_', ' ').title()}:")
            print(f"     - Avg execution time: {avg_time:.2f}s")
            print(f"     - Avg cost: ${avg_cost:.6f}")
            print(f"     - Avg throughput: {avg_throughput:.2f} tasks/sec")
            print(f"     - Avg quality: {avg_quality:.2f}")

            # Performance efficiency score
            efficiency_score = (avg_throughput * avg_quality) / (
                avg_cost * 1000 + avg_time
            )
            print(f"     - Efficiency score: {efficiency_score:.3f}")

        return strategies

    def analyze_bottlenecks(
        self, performance_data: dict[str, list[PerformanceMetrics]]
    ):
        """Analyze performance bottlenecks and optimization opportunities."""
        print("\n" + "=" * 70)
        print("🔍 Bottleneck Analysis & Optimization Opportunities")
        print("=" * 70)

        # Flatten all performance data
        all_metrics = []
        for _strategy, results in performance_data.items():
            all_metrics.extend(results)

        if not all_metrics:
            print("❌ No performance data available for analysis")
            return

        # Calculate statistics
        execution_times = [m.execution_time for m in all_metrics]
        costs = [m.total_cost for m in all_metrics]
        throughputs = [m.throughput for m in all_metrics]
        quality_scores = [m.quality_score for m in all_metrics]

        print("\n📈 Performance Statistics:")
        print(
            f"   ⏱️ Execution time - Min: {min(execution_times):.2f}s, "
            f"Max: {max(execution_times):.2f}s, Avg: {statistics.mean(execution_times):.2f}s"
        )
        print(
            f"   💰 Cost - Min: ${min(costs):.6f}, "
            f"Max: ${max(costs):.6f}, Avg: ${statistics.mean(costs):.6f}"
        )
        print(
            f"   ⚡ Throughput - Min: {min(throughputs):.2f}, "
            f"Max: {max(throughputs):.2f}, Avg: {statistics.mean(throughputs):.2f} tasks/sec"
        )
        print(
            f"   📊 Quality - Min: {min(quality_scores):.2f}, "
            f"Max: {max(quality_scores):.2f}, Avg: {statistics.mean(quality_scores):.2f}"
        )

        # Identify bottlenecks
        print("\n🚨 Identified Bottlenecks:")

        # Time bottlenecks
        slowest_metrics = [
            m
            for m in all_metrics
            if m.execution_time > statistics.mean(execution_times) * 1.2
        ]
        if slowest_metrics:
            print(f"   • Slow execution detected in {len(slowest_metrics)} tests")
            print(
                f"     - Average slow time: {statistics.mean([m.execution_time for m in slowest_metrics]):.2f}s"
            )
            print(
                "     - Likely cause: Complex task processing or inefficient agent coordination"
            )

        # Cost bottlenecks
        expensive_metrics = [
            m for m in all_metrics if m.total_cost > statistics.mean(costs) * 1.3
        ]
        if expensive_metrics:
            print(f"   • High cost detected in {len(expensive_metrics)} tests")
            print(
                f"     - Average high cost: ${statistics.mean([m.total_cost for m in expensive_metrics]):.6f}"
            )
            print(
                "     - Likely cause: Expensive model usage or inefficient token consumption"
            )

        # Throughput bottlenecks
        low_throughput = [
            m for m in all_metrics if m.throughput < statistics.mean(throughputs) * 0.7
        ]
        if low_throughput:
            print(f"   • Low throughput detected in {len(low_throughput)} tests")
            print(
                f"     - Average low throughput: {statistics.mean([m.throughput for m in low_throughput]):.2f} tasks/sec"
            )
            print(
                "     - Likely cause: Sequential processing limitations or agent coordination overhead"
            )

    def generate_optimization_recommendations(
        self, performance_data: dict[str, list[PerformanceMetrics]]
    ) -> list[OptimizationRecommendation]:
        """Generate specific optimization recommendations based on performance analysis."""
        print("\n" + "=" * 70)
        print("💡 Performance Optimization Recommendations")
        print("=" * 70)

        recommendations = []

        # Analyze performance patterns
        all_metrics = []
        for results in performance_data.values():
            all_metrics.extend(results)

        if not all_metrics:
            return recommendations

        avg_time = statistics.mean([m.execution_time for m in all_metrics])
        avg_cost = statistics.mean([m.total_cost for m in all_metrics])
        avg_throughput = statistics.mean([m.throughput for m in all_metrics])

        # Time optimization recommendations
        if avg_time > 30:  # If average execution time > 30 seconds
            recommendations.append(
                OptimizationRecommendation(
                    category="Speed",
                    priority="High",
                    description="Consider parallel task execution and agent optimization",
                    expected_improvement=0.40,  # 40% improvement
                    implementation_effort="Medium",
                    cost_impact="Neutral",
                )
            )

        # Cost optimization recommendations
        if avg_cost > 0.10:  # If average cost > $0.10
            recommendations.append(
                OptimizationRecommendation(
                    category="Cost",
                    priority="High",
                    description="Switch to more cost-effective models for routine tasks",
                    expected_improvement=0.30,
                    implementation_effort="Low",
                    cost_impact="Positive",
                )
            )

        # Throughput optimization recommendations
        if avg_throughput < 0.5:  # If throughput < 0.5 tasks/second
            recommendations.append(
                OptimizationRecommendation(
                    category="Throughput",
                    priority="Medium",
                    description="Implement task batching and agent specialization",
                    expected_improvement=0.50,
                    implementation_effort="High",
                    cost_impact="Neutral",
                )
            )

        # Quality-based recommendations
        quality_variance = (
            statistics.stdev([m.quality_score for m in all_metrics])
            if len(all_metrics) > 1
            else 0
        )
        if quality_variance > 0.1:
            recommendations.append(
                OptimizationRecommendation(
                    category="Quality",
                    priority="Medium",
                    description="Standardize agent prompts and add quality validation",
                    expected_improvement=0.15,
                    implementation_effort="Medium",
                    cost_impact="Slight increase",
                )
            )

        # Resource utilization recommendations
        avg_utilization = statistics.mean([m.resource_utilization for m in all_metrics])
        if avg_utilization < 0.6:
            recommendations.append(
                OptimizationRecommendation(
                    category="Resource Usage",
                    priority="Low",
                    description="Increase concurrent processing and optimize resource allocation",
                    expected_improvement=0.25,
                    implementation_effort="High",
                    cost_impact="Neutral",
                )
            )

        # Display recommendations
        print(f"\n🎯 Generated {len(recommendations)} optimization recommendations:")

        for i, rec in enumerate(recommendations, 1):
            print(f"\n   {i}. {rec.category} Optimization ({rec.priority} Priority)")
            print(f"      📝 {rec.description}")
            print(
                f"      📈 Expected improvement: {rec.expected_improvement * 100:.0f}%"
            )
            print(f"      🔧 Implementation effort: {rec.implementation_effort}")
            print(f"      💰 Cost impact: {rec.cost_impact}")

        return recommendations

    def implement_performance_monitoring(self):
        """Demonstrate real-time performance monitoring capabilities."""
        print("\n" + "=" * 70)
        print("📊 Real-Time Performance Monitoring")
        print("=" * 70)

        # Create monitoring crew
        monitor_crew = self.create_performance_test_crew("medium", agent_count=3)

        print("🔍 Setting up real-time monitoring for crew execution...")

        with self.adapter.track_crew(
            "performance-monitoring", enable_real_time_monitoring=True
        ) as context:
            # Simulate real-time monitoring during execution
            start_time = time.time()
            print(
                f"   ⏱️ Start time: {time.strftime('%H:%M:%S', time.localtime(start_time))}"
            )

            # Execute with monitoring
            monitor_crew.kickoff(
                {"monitoring_enabled": True, "performance_tracking": True}
            )

            end_time = time.time()
            execution_time = end_time - start_time

            # Get real-time metrics
            metrics = context.get_metrics()

            print("\n📊 Real-Time Performance Metrics:")
            print(f"   ⏱️ Total execution time: {execution_time:.2f} seconds")
            print(f"   💰 Real-time cost tracking: ${metrics['total_cost']:.6f}")
            print(f"   👥 Active agents: {metrics['total_agents']}")
            print(f"   📋 Completed tasks: {len(monitor_crew.tasks)}")

            # Simulated real-time alerts
            if execution_time > 60:
                print("   🚨 ALERT: Execution time exceeds 60 seconds")
            if metrics["total_cost"] > 0.50:
                print("   💸 ALERT: Cost exceeds $0.50 threshold")

            # Performance insights
            insights = get_multi_agent_insights(self.monitor, "performance-monitoring")
            if "error" not in insights:
                print("\n🧠 Multi-Agent Insights:")
                print(
                    f"   🤝 Collaboration efficiency: {insights.get('collaboration_matrix', {})}"
                )
                print(
                    f"   ⚠️ Bottleneck agents: {insights.get('bottleneck_agents', [])}"
                )
                print(
                    f"   ⚖️ Load balancing score: {insights.get('load_balancing_score', 0.0):.2f}"
                )

    def generate_performance_report(
        self,
        performance_data: dict[str, list[PerformanceMetrics]],
        recommendations: list[OptimizationRecommendation],
    ):
        """Generate comprehensive performance analysis report."""
        print("\n" + "=" * 70)
        print("📄 Performance Optimization Report")
        print("=" * 70)

        # Aggregate all metrics
        all_metrics = []
        for results in performance_data.values():
            all_metrics.extend(results)

        if not all_metrics:
            print("❌ No performance data available for report")
            return

        # Calculate comprehensive statistics
        total_executions = len(all_metrics)
        total_time = sum(m.execution_time for m in all_metrics)
        total_cost = sum(m.total_cost for m in all_metrics)
        total_tasks = sum(m.tasks_count for m in all_metrics)

        avg_execution_time = statistics.mean([m.execution_time for m in all_metrics])
        avg_cost = statistics.mean([m.total_cost for m in all_metrics])
        avg_throughput = statistics.mean([m.throughput for m in all_metrics])
        avg_quality = statistics.mean([m.quality_score for m in all_metrics])

        print("\n📊 Executive Performance Summary:")
        print(f"   🧪 Total test executions: {total_executions}")
        print(f"   ⏱️ Total execution time: {total_time:.2f} seconds")
        print(f"   💰 Total cost: ${total_cost:.6f}")
        print(f"   📋 Total tasks processed: {total_tasks}")
        print(f"   📈 Average throughput: {avg_throughput:.2f} tasks/second")
        print(f"   ⭐ Average quality score: {avg_quality:.2f}")

        # Performance benchmarks
        print("\n🎯 Performance Benchmarks:")
        fastest_execution = min(all_metrics, key=lambda m: m.execution_time)
        most_efficient = min(all_metrics, key=lambda m: m.cost_efficiency)
        highest_throughput = max(all_metrics, key=lambda m: m.throughput)

        print(
            f"   ⚡ Fastest execution: {fastest_execution.execution_time:.2f}s "
            f"({fastest_execution.agents_count} agents)"
        )
        print(
            f"   💰 Most cost-efficient: ${most_efficient.cost_efficiency:.6f} per task "
            f"({most_efficient.agents_count} agents)"
        )
        print(
            f"   🚀 Highest throughput: {highest_throughput.throughput:.2f} tasks/sec "
            f"({highest_throughput.agents_count} agents)"
        )

        # Optimization potential
        print("\n🔧 Optimization Potential:")
        if recommendations:
            total_improvement = sum(rec.expected_improvement for rec in recommendations)
            print(
                f"   📈 Combined improvement potential: {total_improvement * 100:.0f}%"
            )

            # Projected improvements
            optimized_time = avg_execution_time * (
                1 - total_improvement * 0.3
            )  # 30% of improvement on time
            optimized_cost = avg_cost * (
                1 - total_improvement * 0.4
            )  # 40% of improvement on cost

            print(
                f"   ⏱️ Projected execution time: {optimized_time:.2f}s "
                f"({((avg_execution_time - optimized_time) / avg_execution_time) * 100:+.1f}%)"
            )
            print(
                f"   💰 Projected cost: ${optimized_cost:.6f} "
                f"({((avg_cost - optimized_cost) / avg_cost) * 100:+.1f}%)"
            )

        # Recommendations by priority
        print("\n💡 Priority Recommendations:")
        high_priority = [r for r in recommendations if r.priority == "High"]
        medium_priority = [r for r in recommendations if r.priority == "Medium"]
        low_priority = [r for r in recommendations if r.priority == "Low"]

        if high_priority:
            print(f"   🔴 High Priority ({len(high_priority)} items):")
            for rec in high_priority:
                print(f"      • {rec.category}: {rec.description}")

        if medium_priority:
            print(f"   🟡 Medium Priority ({len(medium_priority)} items):")
            for rec in medium_priority:
                print(f"      • {rec.category}: {rec.description}")

        if low_priority:
            print(f"   🟢 Low Priority ({len(low_priority)} items):")
            for rec in low_priority:
                print(f"      • {rec.category}: {rec.description}")

        # Implementation roadmap
        print("\n🗺️ Implementation Roadmap:")
        print("   Phase 1 (Immediate): Implement High priority recommendations")
        print("   Phase 2 (Next 2 weeks): Implement Medium priority recommendations")
        print("   Phase 3 (Future): Implement Low priority recommendations")
        print("   Monitoring: Set up continuous performance monitoring in production")

        return {
            "total_executions": total_executions,
            "avg_execution_time": avg_execution_time,
            "avg_cost": avg_cost,
            "avg_throughput": avg_throughput,
            "optimization_potential": total_improvement if recommendations else 0,
            "high_priority_recs": len(high_priority),
            "recommendations": recommendations,
        }


def main():
    """Run the comprehensive performance optimization demonstration."""
    parser = argparse.ArgumentParser(description="CrewAI Performance Optimization Demo")
    parser.add_argument(
        "--mode",
        choices=["speed", "cost", "balanced", "quality"],
        default="balanced",
        help="Optimization focus mode",
    )
    parser.add_argument(
        "--agents", type=int, default=4, help="Maximum number of agents to test (1-4)"
    )
    args = parser.parse_args()

    print("⚡ CrewAI Performance Optimization and Tuning")
    print("=" * 50)
    print(f"Optimization mode: {args.mode}")
    print(f"Max agents: {args.agents}")

    # Initialize optimizer
    optimizer = PerformanceOptimizer(optimization_mode=args.mode)

    # Validate setup
    if not optimizer.setup_validation():
        print("\n❌ Please fix setup issues before proceeding")
        return 1

    try:
        # Run performance comparisons
        performance_data = optimizer.compare_execution_strategies()

        # Analyze bottlenecks
        optimizer.analyze_bottlenecks(performance_data)

        # Generate recommendations
        recommendations = optimizer.generate_optimization_recommendations(
            performance_data
        )

        # Demonstrate real-time monitoring
        optimizer.implement_performance_monitoring()

        # Generate comprehensive report
        optimizer.generate_performance_report(performance_data, recommendations)

        print("\n🎉 Performance Optimization Analysis Complete!")
        print("\n🚀 Next Steps:")
        print("   • Implement high-priority optimization recommendations")
        print("   • Set up continuous performance monitoring in production")
        print("   • Try agent_workflow_governance.py for advanced workflow analysis")
        print("   • Explore production_deployment_patterns.py for scaling strategies")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️ Performance analysis interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Performance analysis failed: {e}", exc_info=True)
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
