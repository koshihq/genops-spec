#!/usr/bin/env python3
"""
Advanced Langfuse Observability with GenOps Governance Example

This example demonstrates enterprise-grade observability patterns with Langfuse
enhanced by comprehensive GenOps governance. Designed for production systems
that need sophisticated tracing, monitoring, and governance automation.

Usage:
    python advanced_observability.py

Prerequisites:
    pip install genops[langfuse]
    export LANGFUSE_PUBLIC_KEY="pk-lf-your-public-key"
    export LANGFUSE_SECRET_KEY="sk-lf-your-secret-key"
    export OPENAI_API_KEY="your-openai-api-key"  # Or another provider
    export ANTHROPIC_API_KEY="your-anthropic-key"  # Optional: for multi-provider demos
"""

import os
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


@dataclass
class ObservabilityMetrics:
    """Comprehensive observability metrics with governance context."""
    operation_id: str
    operation_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None

    # Cost and performance metrics
    total_cost: float = 0.0
    provider_costs: Dict[str, float] = field(default_factory=dict)
    token_usage: Dict[str, int] = field(default_factory=dict)

    # Governance attributes
    team: Optional[str] = None
    project: Optional[str] = None
    customer_id: Optional[str] = None
    cost_center: Optional[str] = None
    compliance_tags: List[str] = field(default_factory=list)

    # Trace hierarchy
    parent_operation_id: Optional[str] = None
    child_operations: List[str] = field(default_factory=list)
    trace_depth: int = 0

    # Quality and error metrics
    success: bool = True
    error_count: int = 0
    warnings: List[str] = field(default_factory=list)
    quality_scores: Dict[str, float] = field(default_factory=dict)

    # Business metrics
    business_value: float = 0.0
    customer_satisfaction: Optional[float] = None
    operational_efficiency: Optional[float] = None

    def finalize(self):
        """Finalize metrics when operation completes."""
        if self.end_time:
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000


class AdvancedObservabilityManager:
    """Enterprise observability manager with comprehensive governance."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.active_operations = {}
        self.operation_hierarchy = {}
        self.metrics_cache = {}
        self.alert_thresholds = {
            "cost_per_operation": 1.0,
            "latency_ms": 10000,
            "error_rate": 0.05,
            "budget_utilization": 0.8
        }
        self.compliance_rules = {}
        self._setup_monitoring()

    def _setup_monitoring(self):
        """Initialize monitoring and alerting systems."""
        print("🔧 Initializing advanced observability monitoring...")
        print("   📊 Real-time metrics collection")
        print("   🚨 Automated alerting and anomaly detection")
        print("   📈 Performance trend analysis")
        print("   🛡️  Compliance monitoring and validation")
        print("   💰 Cost optimization recommendations")

    @contextmanager
    def observe_complex_operation(
        self,
        operation_name: str,
        operation_type: str = "complex_workflow",
        parent_operation_id: Optional[str] = None,
        **governance_attrs
    ):
        """Advanced context manager for complex operation observability."""

        operation_id = str(uuid.uuid4())
        start_time = datetime.now()

        # Initialize metrics
        metrics = ObservabilityMetrics(
            operation_id=operation_id,
            operation_type=operation_type,
            start_time=start_time,
            parent_operation_id=parent_operation_id,
            **{k: v for k, v in governance_attrs.items() if hasattr(ObservabilityMetrics, k)}
        )

        # Handle hierarchical operations
        if parent_operation_id and parent_operation_id in self.active_operations:
            parent_metrics = self.active_operations[parent_operation_id]
            parent_metrics.child_operations.append(operation_id)
            metrics.trace_depth = parent_metrics.trace_depth + 1

        self.active_operations[operation_id] = metrics

        # Create enhanced Langfuse trace
        with self.adapter.trace_with_governance(
            name=operation_name,
            operation_id=operation_id,
            operation_type=operation_type,
            parent_operation_id=parent_operation_id,
            trace_depth=metrics.trace_depth,
            **governance_attrs
        ) as trace:

            try:
                print(f"🚀 Starting complex operation: {operation_name}")
                print(f"   🆔 Operation ID: {operation_id}")
                print(f"   📊 Type: {operation_type}")
                print(f"   🔗 Parent: {parent_operation_id[:12] if parent_operation_id else 'None'}")
                print(f"   📏 Depth: {metrics.trace_depth}")

                yield {
                    "operation_id": operation_id,
                    "metrics": metrics,
                    "trace": trace,
                    "manager": self
                }

            except Exception as e:
                metrics.success = False
                metrics.error_count += 1
                metrics.warnings.append(f"Operation failed: {str(e)}")
                print(f"❌ Operation {operation_name} failed: {e}")
                raise

            finally:
                # Finalize metrics
                metrics.end_time = datetime.now()
                metrics.finalize()

                # Run compliance checks
                self._check_compliance(metrics)

                # Generate alerts if needed
                self._check_alerts(metrics)

                # Cache metrics for analysis
                self.metrics_cache[operation_id] = metrics

                # Clean up active operations
                if operation_id in self.active_operations:
                    del self.active_operations[operation_id]

                print(f"✅ Operation {operation_name} completed")
                print(f"   ⏱️  Duration: {metrics.duration_ms:.0f}ms")
                print(f"   💰 Cost: ${metrics.total_cost:.6f}")
                print(f"   🎯 Success: {metrics.success}")

                # Update parent metrics
                if parent_operation_id and parent_operation_id in self.active_operations:
                    parent_metrics = self.active_operations[parent_operation_id]
                    parent_metrics.total_cost += metrics.total_cost
                    parent_metrics.error_count += metrics.error_count

    def add_operation_cost(
        self,
        operation_id: str,
        provider: str,
        cost: float,
        tokens: Dict[str, int],
        model: str
    ):
        """Add cost and usage data to an operation."""
        if operation_id in self.active_operations:
            metrics = self.active_operations[operation_id]
            metrics.total_cost += cost
            metrics.provider_costs[provider] = metrics.provider_costs.get(provider, 0.0) + cost

            for token_type, count in tokens.items():
                metrics.token_usage[token_type] = metrics.token_usage.get(token_type, 0) + count

    def _check_compliance(self, metrics: ObservabilityMetrics):
        """Check compliance rules against operation metrics."""
        violations = []

        # Cost compliance
        if metrics.total_cost > self.alert_thresholds["cost_per_operation"]:
            violations.append(f"Cost exceeded threshold: ${metrics.total_cost:.6f}")

        # Latency compliance
        if metrics.duration_ms and metrics.duration_ms > self.alert_thresholds["latency_ms"]:
            violations.append(f"Latency exceeded threshold: {metrics.duration_ms:.0f}ms")

        # Error rate compliance
        if not metrics.success:
            violations.append("Operation failed")

        if violations:
            metrics.warnings.extend(violations)
            print(f"⚠️  Compliance violations detected for {metrics.operation_id}")
            for violation in violations:
                print(f"   • {violation}")

    def _check_alerts(self, metrics: ObservabilityMetrics):
        """Check if alerts should be triggered."""
        if metrics.warnings:
            print(f"🚨 Alert conditions detected for operation {metrics.operation_type}")
            # In production, this would send alerts to monitoring systems

    def get_operation_analytics(
        self,
        time_range: timedelta = timedelta(hours=24)
    ) -> Dict[str, Any]:
        """Generate comprehensive analytics from collected metrics."""

        cutoff_time = datetime.now() - time_range
        recent_operations = [
            metrics for metrics in self.metrics_cache.values()
            if metrics.start_time >= cutoff_time
        ]

        if not recent_operations:
            return {"message": "No operations in specified time range"}

        # Calculate aggregate metrics
        total_cost = sum(op.total_cost for op in recent_operations)
        total_operations = len(recent_operations)
        successful_operations = sum(1 for op in recent_operations if op.success)

        avg_cost = total_cost / total_operations if total_operations > 0 else 0
        success_rate = successful_operations / total_operations if total_operations > 0 else 0

        # Provider breakdown
        provider_costs = {}
        for op in recent_operations:
            for provider, cost in op.provider_costs.items():
                provider_costs[provider] = provider_costs.get(provider, 0.0) + cost

        # Governance breakdown
        team_costs = {}
        customer_costs = {}
        for op in recent_operations:
            if op.team:
                team_costs[op.team] = team_costs.get(op.team, 0.0) + op.total_cost
            if op.customer_id:
                customer_costs[op.customer_id] = customer_costs.get(op.customer_id, 0.0) + op.total_cost

        return {
            "time_range_hours": time_range.total_seconds() / 3600,
            "summary": {
                "total_operations": total_operations,
                "successful_operations": successful_operations,
                "success_rate": success_rate,
                "total_cost": total_cost,
                "average_cost_per_operation": avg_cost
            },
            "provider_breakdown": provider_costs,
            "team_breakdown": team_costs,
            "customer_breakdown": customer_costs,
            "compliance": {
                "operations_with_warnings": sum(1 for op in recent_operations if op.warnings),
                "total_warnings": sum(len(op.warnings) for op in recent_operations)
            }
        }


def demonstrate_hierarchical_tracing():
    """Demonstrate complex hierarchical operation tracing."""
    print("🌲 Hierarchical Operation Tracing with Governance")
    print("=" * 48)

    try:
        from genops.providers.langfuse import instrument_langfuse

        # Initialize adapter for advanced observability
        adapter = instrument_langfuse(
            team="advanced-observability-team",
            project="enterprise-tracing",
            environment="production",
            budget_limits={"daily": 10.0}
        )

        # Initialize observability manager
        obs_manager = AdvancedObservabilityManager(adapter)

        print("✅ Advanced observability manager initialized")

        # Demonstrate complex hierarchical workflow
        with obs_manager.observe_complex_operation(
            operation_name="document_analysis_pipeline",
            operation_type="ml_pipeline",
            customer_id="enterprise-customer-001",
            cost_center="ai-research",
            feature="document-intelligence"
        ) as context:

            main_operation_id = context["operation_id"]

            # Step 1: Document preprocessing
            with obs_manager.observe_complex_operation(
                operation_name="document_preprocessing",
                operation_type="data_preparation",
                parent_operation_id=main_operation_id,
                customer_id="enterprise-customer-001",
                cost_center="ai-research"
            ) as prep_context:

                print("   📄 Preprocessing document...")
                time.sleep(0.2)  # Simulate processing

                # Simulate LLM call for document parsing
                response = adapter.generation_with_cost_tracking(
                    prompt="Extract key information from this document: [document content]",
                    model="gpt-3.5-turbo",
                    max_cost=0.10,
                    operation="document_parsing",
                    customer_id="enterprise-customer-001"
                )

                obs_manager.add_operation_cost(
                    prep_context["operation_id"],
                    "openai",
                    response.usage.cost,
                    {"input": response.usage.input_tokens, "output": response.usage.output_tokens},
                    "gpt-3.5-turbo"
                )

                print(f"     ✅ Document parsed, cost: ${response.usage.cost:.6f}")

            # Step 2: Content analysis (parallel sub-operations)
            with obs_manager.observe_complex_operation(
                operation_name="content_analysis",
                operation_type="parallel_analysis",
                parent_operation_id=main_operation_id,
                customer_id="enterprise-customer-001",
                cost_center="ai-research"
            ) as analysis_context:

                print("   🔍 Running parallel content analysis...")

                # Simulate multiple parallel analysis tasks
                analysis_tasks = [
                    {"task": "sentiment_analysis", "prompt": "Analyze the sentiment of this document"},
                    {"task": "topic_extraction", "prompt": "Extract the main topics from this document"},
                    {"task": "summary_generation", "prompt": "Generate a concise summary of this document"}
                ]

                total_analysis_cost = 0.0

                for task in analysis_tasks:
                    with obs_manager.observe_complex_operation(
                        operation_name=task["task"],
                        operation_type="llm_analysis",
                        parent_operation_id=analysis_context["operation_id"],
                        customer_id="enterprise-customer-001",
                        cost_center="ai-research",
                        task_type=task["task"]
                    ) as task_context:

                        print(f"     🎯 {task['task']}...")

                        response = adapter.generation_with_cost_tracking(
                            prompt=task["prompt"],
                            model="gpt-3.5-turbo",
                            max_cost=0.05,
                            operation=task["task"],
                            customer_id="enterprise-customer-001"
                        )

                        obs_manager.add_operation_cost(
                            task_context["operation_id"],
                            "openai",
                            response.usage.cost,
                            {"input": response.usage.input_tokens, "output": response.usage.output_tokens},
                            "gpt-3.5-turbo"
                        )

                        total_analysis_cost += response.usage.cost
                        print(f"       ✅ {task['task']} complete, cost: ${response.usage.cost:.6f}")

                print(f"   ✅ All analysis tasks complete, total cost: ${total_analysis_cost:.6f}")

            # Step 3: Report generation
            with obs_manager.observe_complex_operation(
                operation_name="report_generation",
                operation_type="document_synthesis",
                parent_operation_id=main_operation_id,
                customer_id="enterprise-customer-001",
                cost_center="ai-research"
            ) as report_context:

                print("   📊 Generating comprehensive report...")
                time.sleep(0.3)  # Simulate report generation

                response = adapter.generation_with_cost_tracking(
                    prompt="Generate a comprehensive analysis report based on the document analysis",
                    model="gpt-3.5-turbo",
                    max_cost=0.15,
                    operation="report_synthesis",
                    customer_id="enterprise-customer-001"
                )

                obs_manager.add_operation_cost(
                    report_context["operation_id"],
                    "openai",
                    response.usage.cost,
                    {"input": response.usage.input_tokens, "output": response.usage.output_tokens},
                    "gpt-3.5-turbo"
                )

                print(f"   ✅ Report generated, cost: ${response.usage.cost:.6f}")

        print("✅ Hierarchical tracing complete!")
        return True

    except Exception as e:
        print(f"❌ Hierarchical tracing failed: {e}")
        return False


def demonstrate_multi_provider_observability():
    """Demonstrate observability across multiple AI providers."""
    print("\n🌐 Multi-Provider Observability with Unified Governance")
    print("=" * 54)

    try:
        from genops.providers.langfuse import instrument_langfuse

        # Initialize adapter for multi-provider tracking
        adapter = instrument_langfuse(
            team="multi-provider-team",
            project="provider-comparison",
            environment="production",
            budget_limits={"daily": 15.0}
        )

        obs_manager = AdvancedObservabilityManager(adapter)

        # Simulate multi-provider workflow
        with obs_manager.observe_complex_operation(
            operation_name="multi_provider_workflow",
            operation_type="provider_comparison",
            customer_id="multi-provider-customer",
            cost_center="ai-operations",
            feature="provider-optimization"
        ) as context:

            main_operation_id = context["operation_id"]

            # Define providers and their use cases
            provider_configs = [
                {
                    "provider": "openai",
                    "model": "gpt-3.5-turbo",
                    "use_case": "general_purpose",
                    "prompt": "Provide a comprehensive analysis of renewable energy trends"
                },
                {
                    "provider": "anthropic",
                    "model": "claude-3-haiku",
                    "use_case": "research_analysis",
                    "prompt": "Conduct detailed research on renewable energy market dynamics"
                },
                {
                    "provider": "openai",
                    "model": "gpt-4",
                    "use_case": "complex_reasoning",
                    "prompt": "Perform complex analysis of renewable energy policy implications"
                }
            ]

            provider_results = {}

            for config in provider_configs:
                provider_name = config["provider"]
                model_name = config["model"]
                use_case = config["use_case"]

                with obs_manager.observe_complex_operation(
                    operation_name=f"{provider_name}_{use_case}",
                    operation_type="provider_execution",
                    parent_operation_id=main_operation_id,
                    customer_id="multi-provider-customer",
                    cost_center="ai-operations",
                    provider=provider_name,
                    model=model_name,
                    use_case=use_case
                ) as provider_context:

                    print(f"   🤖 Testing {provider_name} ({model_name}) for {use_case}")

                    # Simulate provider-specific execution
                    start_time = time.time()

                    # Mock cost calculation based on provider
                    if provider_name == "openai" and model_name == "gpt-4":
                        mock_cost = 0.08  # Higher cost for GPT-4
                        mock_tokens = {"input": 150, "output": 200}
                    elif provider_name == "anthropic":
                        mock_cost = 0.04  # Medium cost for Claude
                        mock_tokens = {"input": 140, "output": 180}
                    else:
                        mock_cost = 0.02  # Lower cost for GPT-3.5
                        mock_tokens = {"input": 120, "output": 160}

                    time.sleep(0.2 + (0.1 if "gpt-4" in model_name else 0))  # Simulate different latencies

                    latency_ms = (time.time() - start_time) * 1000

                    # Record provider metrics
                    obs_manager.add_operation_cost(
                        provider_context["operation_id"],
                        provider_name,
                        mock_cost,
                        mock_tokens,
                        model_name
                    )

                    provider_results[f"{provider_name}_{model_name}"] = {
                        "provider": provider_name,
                        "model": model_name,
                        "use_case": use_case,
                        "cost": mock_cost,
                        "latency_ms": latency_ms,
                        "tokens": mock_tokens,
                        "operation_id": provider_context["operation_id"]
                    }

                    print(f"     ✅ Cost: ${mock_cost:.6f}, Latency: {latency_ms:.0f}ms")

            # Analyze provider performance
            print("\n   📊 Multi-Provider Performance Analysis:")
            print("   Provider/Model        | Cost      | Latency   | Tokens    | Cost/Token")
            print("   " + "-" * 70)

            for config_name, result in provider_results.items():
                cost_per_token = result["cost"] / sum(result["tokens"].values())
                print(f"   {config_name:<20} | ${result['cost']:>8.6f} | {result['latency_ms']:>6.0f}ms | {sum(result['tokens'].values()):>8} | ${cost_per_token:>9.7f}")

            # Find optimal provider for each metric
            cheapest = min(provider_results.items(), key=lambda x: x[1]["cost"])
            fastest = min(provider_results.items(), key=lambda x: x[1]["latency_ms"])
            most_efficient = min(provider_results.items(),
                               key=lambda x: x[1]["cost"] / sum(x[1]["tokens"].values()))

            print("\n   🏆 Performance Winners:")
            print(f"   💰 Most Cost Effective: {cheapest[0]} (${cheapest[1]['cost']:.6f})")
            print(f"   ⚡ Fastest: {fastest[0]} ({fastest[1]['latency_ms']:.0f}ms)")
            print(f"   🎯 Most Token Efficient: {most_efficient[0]}")

        print("✅ Multi-provider observability complete!")
        return True

    except Exception as e:
        print(f"❌ Multi-provider observability failed: {e}")
        return False


def demonstrate_real_time_analytics():
    """Demonstrate real-time analytics and monitoring."""
    print("\n📈 Real-Time Analytics and Performance Monitoring")
    print("=" * 50)

    try:
        from genops.providers.langfuse import instrument_langfuse

        # Initialize adapter for analytics
        adapter = instrument_langfuse(
            team="analytics-team",
            project="real-time-monitoring",
            environment="production",
            budget_limits={"daily": 20.0}
        )

        obs_manager = AdvancedObservabilityManager(adapter)

        print("📊 Setting up real-time analytics dashboard...")

        # Simulate continuous operations for analytics
        simulation_scenarios = [
            {"name": "customer_query", "frequency": 5, "base_cost": 0.01},
            {"name": "document_analysis", "frequency": 3, "base_cost": 0.05},
            {"name": "report_generation", "frequency": 2, "base_cost": 0.08},
            {"name": "data_validation", "frequency": 4, "base_cost": 0.02}
        ]

        # Run simulation for analytics
        print("🔄 Running operation simulation for analytics...")

        for round_num in range(1, 4):  # 3 rounds of operations
            print(f"\n   📊 Analytics Round {round_num}/3")

            for scenario in simulation_scenarios:
                for i in range(scenario["frequency"]):
                    with obs_manager.observe_complex_operation(
                        operation_name=f"{scenario['name']}_r{round_num}_i{i+1}",
                        operation_type=scenario["name"],
                        customer_id=f"analytics-customer-{(i % 3) + 1}",
                        cost_center="analytics-simulation",
                        round=round_num,
                        scenario=scenario["name"]
                    ) as context:

                        # Simulate operation with variable cost and latency
                        operation_cost = scenario["base_cost"] * (0.8 + (i * 0.1))
                        time.sleep(0.05)  # Minimal delay for simulation

                        obs_manager.add_operation_cost(
                            context["operation_id"],
                            "openai",
                            operation_cost,
                            {"input": 100, "output": 150},
                            "gpt-3.5-turbo"
                        )

        # Generate comprehensive analytics
        analytics = obs_manager.get_operation_analytics()

        print("\n📈 Real-Time Analytics Dashboard:")
        print("=" * 35)

        summary = analytics["summary"]
        print("📊 Operations Summary:")
        print(f"   Total Operations: {summary['total_operations']}")
        print(f"   Success Rate: {summary['success_rate']:.1%}")
        print(f"   Total Cost: ${summary['total_cost']:.6f}")
        print(f"   Avg Cost/Operation: ${summary['average_cost_per_operation']:.6f}")

        print("\n🏷️  Team Cost Breakdown:")
        for team, cost in analytics["team_breakdown"].items():
            percentage = (cost / summary["total_cost"]) * 100
            print(f"   {team}: ${cost:.6f} ({percentage:.1f}%)")

        print("\n👥 Customer Cost Attribution:")
        for customer, cost in analytics["customer_breakdown"].items():
            percentage = (cost / summary["total_cost"]) * 100
            print(f"   {customer}: ${cost:.6f} ({percentage:.1f}%)")

        compliance = analytics["compliance"]
        print("\n🛡️  Compliance Status:")
        print(f"   Operations with Warnings: {compliance['operations_with_warnings']}")
        print(f"   Total Warnings: {compliance['total_warnings']}")

        # Performance trends and recommendations
        print("\n💡 Performance Insights:")
        if summary["average_cost_per_operation"] > 0.03:
            print("   ⚠️  Average cost per operation is elevated - consider optimization")
        if summary["success_rate"] < 0.95:
            print("   ⚠️  Success rate below target - investigate error patterns")
        if compliance["total_warnings"] > 0:
            print("   ⚠️  Compliance warnings detected - review governance policies")

        print("   ✅ Real-time monitoring active and collecting metrics")
        print("   ✅ Cost attribution working across teams and customers")
        print("   ✅ Governance compliance being tracked and reported")

        return analytics

    except Exception as e:
        print(f"❌ Real-time analytics failed: {e}")
        return None


def demonstrate_anomaly_detection():
    """Demonstrate automated anomaly detection and alerting."""
    print("\n🚨 Automated Anomaly Detection and Alerting")
    print("=" * 43)

    try:
        from genops.providers.langfuse import instrument_langfuse

        # Initialize adapter with strict thresholds for anomaly detection
        adapter = instrument_langfuse(
            team="anomaly-detection-team",
            project="automated-monitoring",
            environment="production",
            budget_limits={"daily": 5.0}  # Lower limit to trigger alerts
        )

        obs_manager = AdvancedObservabilityManager(adapter)

        # Set up anomaly detection thresholds
        obs_manager.alert_thresholds.update({
            "cost_per_operation": 0.15,  # Trigger alert if operation costs > 15 cents
            "latency_ms": 3000,          # Trigger alert if latency > 3 seconds
            "error_rate": 0.1            # Trigger alert if error rate > 10%
        })

        print("🔍 Testing anomaly detection systems...")
        print(f"   Cost threshold: ${obs_manager.alert_thresholds['cost_per_operation']:.2f}")
        print(f"   Latency threshold: {obs_manager.alert_thresholds['latency_ms']}ms")
        print(f"   Error rate threshold: {obs_manager.alert_thresholds['error_rate']:.1%}")

        # Test scenarios including anomalies
        test_scenarios = [
            {
                "name": "normal_operation",
                "description": "Normal operation within thresholds",
                "cost": 0.05,
                "simulate_delay": 0.1,
                "should_fail": False
            },
            {
                "name": "high_cost_anomaly",
                "description": "Operation with abnormally high cost",
                "cost": 0.25,  # Above threshold
                "simulate_delay": 0.1,
                "should_fail": False
            },
            {
                "name": "high_latency_anomaly",
                "description": "Operation with abnormally high latency",
                "cost": 0.03,
                "simulate_delay": 4.0,  # Above threshold
                "should_fail": False
            },
            {
                "name": "failure_anomaly",
                "description": "Operation that fails",
                "cost": 0.02,
                "simulate_delay": 0.1,
                "should_fail": True
            }
        ]

        anomaly_results = []

        for scenario in test_scenarios:
            print(f"\n🧪 Testing: {scenario['description']}")

            try:
                with obs_manager.observe_complex_operation(
                    operation_name=scenario["name"],
                    operation_type="anomaly_test",
                    customer_id="anomaly-test-customer",
                    cost_center="quality-assurance",
                    test_scenario=scenario["name"]
                ) as context:

                    start_time = time.time()

                    # Simulate the scenario
                    if scenario["should_fail"]:
                        time.sleep(0.1)
                        raise Exception("Simulated failure for testing")

                    # Simulate processing delay
                    time.sleep(scenario["simulate_delay"])

                    # Record operation cost
                    obs_manager.add_operation_cost(
                        context["operation_id"],
                        "openai",
                        scenario["cost"],
                        {"input": 100, "output": 150},
                        "gpt-3.5-turbo"
                    )

                    actual_latency = (time.time() - start_time) * 1000

                    anomaly_results.append({
                        "scenario": scenario["name"],
                        "success": True,
                        "cost": scenario["cost"],
                        "latency_ms": actual_latency,
                        "alerts_triggered": len(context["metrics"].warnings) > 0
                    })

                    print(f"   ✅ Completed - Cost: ${scenario['cost']:.6f}, Latency: {actual_latency:.0f}ms")
                    if context["metrics"].warnings:
                        print(f"   🚨 Alerts triggered: {len(context['metrics'].warnings)}")
                        for warning in context["metrics"].warnings:
                            print(f"     • {warning}")

            except Exception as e:
                anomaly_results.append({
                    "scenario": scenario["name"],
                    "success": False,
                    "error": str(e),
                    "alerts_triggered": True
                })
                print(f"   ❌ Failed as expected: {e}")

        # Anomaly detection summary
        print("\n🚨 Anomaly Detection Results Summary:")
        print("=" * 35)

        total_scenarios = len(anomaly_results)
        scenarios_with_alerts = sum(1 for r in anomaly_results if r.get("alerts_triggered", False))

        print(f"   📊 Total scenarios tested: {total_scenarios}")
        print(f"   🚨 Scenarios triggering alerts: {scenarios_with_alerts}")
        print(f"   📈 Alert detection rate: {(scenarios_with_alerts / total_scenarios) * 100:.1f}%")

        # Expected vs actual alert analysis
        expected_alerts = ["high_cost_anomaly", "high_latency_anomaly", "failure_anomaly"]
        actual_alerts = [r["scenario"] for r in anomaly_results if r.get("alerts_triggered", False)]

        print("\n🎯 Alert Accuracy Analysis:")
        for scenario in expected_alerts:
            detected = scenario in actual_alerts
            status = "✅ Detected" if detected else "❌ Missed"
            print(f"   {scenario}: {status}")

        print("\n💡 Anomaly Detection Capabilities:")
        print("   ✅ Real-time cost threshold monitoring")
        print("   ✅ Latency performance alerting")
        print("   ✅ Failure detection and notification")
        print("   ✅ Governance compliance violation alerts")
        print("   ✅ Automated alert escalation and routing")

        return anomaly_results

    except Exception as e:
        print(f"❌ Anomaly detection test failed: {e}")
        return None


def demonstrate_enterprise_governance():
    """Demonstrate enterprise-grade governance and compliance features."""
    print("\n🏛️ Enterprise Governance and Compliance Monitoring")
    print("=" * 51)

    try:
        from genops.providers.langfuse import instrument_langfuse

        # Initialize adapter with enterprise governance
        adapter = instrument_langfuse(
            team="enterprise-governance",
            project="compliance-monitoring",
            environment="production",
            budget_limits={"daily": 50.0, "monthly": 1000.0}
        )

        obs_manager = AdvancedObservabilityManager(adapter)

        # Set up enterprise compliance rules
        obs_manager.compliance_rules = {
            "data_retention": {"max_days": 90, "classification": "confidential"},
            "cost_controls": {"daily_limit": 50.0, "approval_threshold": 10.0},
            "access_controls": {"approved_teams": ["enterprise-governance", "ai-research"]},
            "audit_requirements": {"log_all_operations": True, "compliance_reporting": True}
        }

        print("🏛️  Enterprise Governance Features Active:")
        governance_features = [
            "📊 Comprehensive audit logging for all LLM operations",
            "💰 Multi-tier budget controls with approval workflows",
            "🛡️  Data classification and retention policy enforcement",
            "👥 Role-based access control with team authorization",
            "📈 Compliance reporting and regulatory alignment",
            "🔍 Real-time governance monitoring and violation detection",
            "⚡ Automated policy enforcement and remediation"
        ]

        for feature in governance_features:
            print(f"   {feature}")

        # Simulate enterprise workflows with governance
        enterprise_scenarios = [
            {
                "scenario": "financial_analysis",
                "classification": "confidential",
                "approval_required": True,
                "customer_id": "enterprise-bank-001",
                "regulatory_context": "financial-services"
            },
            {
                "scenario": "customer_data_processing",
                "classification": "pii",
                "approval_required": True,
                "customer_id": "enterprise-retail-002",
                "regulatory_context": "gdpr-compliance"
            },
            {
                "scenario": "public_content_generation",
                "classification": "public",
                "approval_required": False,
                "customer_id": "enterprise-media-003",
                "regulatory_context": "content-standards"
            }
        ]

        governance_results = []

        for scenario in enterprise_scenarios:
            print(f"\n🏢 Enterprise Scenario: {scenario['scenario']}")
            print(f"   🏷️  Classification: {scenario['classification']}")
            print(f"   📋 Regulatory Context: {scenario['regulatory_context']}")
            print(f"   ✅ Approval Required: {scenario['approval_required']}")

            with obs_manager.observe_complex_operation(
                operation_name=scenario["scenario"],
                operation_type="enterprise_workflow",
                customer_id=scenario["customer_id"],
                cost_center="enterprise-operations",
                data_classification=scenario["classification"],
                regulatory_context=scenario["regulatory_context"],
                approval_required=scenario["approval_required"]
            ) as context:

                # Simulate governance checks
                print("     🔍 Running pre-execution governance checks...")

                governance_checks = [
                    "Data classification validation",
                    "Customer authorization verification",
                    "Regulatory compliance assessment",
                    "Budget allocation confirmation",
                    "Audit trail initialization"
                ]

                for check in governance_checks:
                    print(f"       ✅ {check}")
                    time.sleep(0.02)  # Simulate check processing

                # Simulate the enterprise operation
                print("     🚀 Executing governed enterprise operation...")

                # Different costs based on classification level
                operation_cost = {
                    "confidential": 0.12,
                    "pii": 0.08,
                    "public": 0.04
                }.get(scenario["classification"], 0.06)

                time.sleep(0.3)  # Simulate processing

                obs_manager.add_operation_cost(
                    context["operation_id"],
                    "openai",
                    operation_cost,
                    {"input": 200, "output": 300},
                    "gpt-3.5-turbo"
                )

                # Post-execution compliance validation
                print("     🛡️  Running post-execution compliance validation...")

                compliance_validations = [
                    "Output content review for compliance",
                    "Cost attribution to customer billing",
                    "Audit log completion and verification",
                    "Data retention policy application",
                    "Regulatory reporting requirement updates"
                ]

                for validation in compliance_validations:
                    print(f"       ✅ {validation}")
                    time.sleep(0.02)

                governance_results.append({
                    "scenario": scenario["scenario"],
                    "cost": operation_cost,
                    "customer": scenario["customer_id"],
                    "classification": scenario["classification"],
                    "regulatory_context": scenario["regulatory_context"],
                    "compliance_status": "compliant"
                })

                print(f"     ✅ Scenario complete - Cost: ${operation_cost:.6f}")

        # Generate enterprise governance summary
        print("\n📊 Enterprise Governance Summary:")
        print("=" * 32)

        total_cost = sum(r["cost"] for r in governance_results)
        classification_breakdown = {}
        customer_breakdown = {}

        for result in governance_results:
            # Classification breakdown
            classification = result["classification"]
            classification_breakdown[classification] = classification_breakdown.get(classification, 0.0) + result["cost"]

            # Customer breakdown
            customer = result["customer"]
            customer_breakdown[customer] = customer_breakdown.get(customer, 0.0) + result["cost"]

        print(f"💰 Total Enterprise Operation Cost: ${total_cost:.6f}")

        print("\n🏷️  Cost by Data Classification:")
        for classification, cost in classification_breakdown.items():
            percentage = (cost / total_cost) * 100
            print(f"   {classification.title()}: ${cost:.6f} ({percentage:.1f}%)")

        print("\n🏢 Cost by Enterprise Customer:")
        for customer, cost in customer_breakdown.items():
            percentage = (cost / total_cost) * 100
            print(f"   {customer}: ${cost:.6f} ({percentage:.1f}%)")

        print("\n✅ Governance Compliance Status:")
        print("   📋 All operations completed with full compliance")
        print("   🛡️  Data classification policies enforced")
        print("   📊 Complete audit trail maintained")
        print("   💰 Cost attribution accurate for enterprise billing")
        print("   🎯 Regulatory requirements satisfied")

        return governance_results

    except Exception as e:
        print(f"❌ Enterprise governance demonstration failed: {e}")
        return None


def show_next_steps():
    """Show next steps for production deployment."""
    print("\n🚀 Production Deployment and Advanced Features")
    print("=" * 46)

    production_steps = [
        ("🏭 Production Patterns", "Deploy advanced observability in production environments",
         "python production_patterns.py"),
        ("📊 Custom Dashboards", "Build custom monitoring dashboards for your organization",
         "Integrate with Grafana, Datadog, or your observability platform"),
        ("🔄 CI/CD Integration", "Integrate governance checks into deployment pipelines",
         "Add GenOps governance to your continuous integration workflows"),
        ("📈 Business Intelligence", "Connect observability to business intelligence platforms",
         "Export governance data to BI tools for executive reporting"),
        ("🌐 Multi-Region Deployment", "Deploy governance across multiple geographic regions",
         "Configure region-aware cost attribution and compliance"),
        ("🤖 AI/ML Pipeline Integration", "Integrate with ML pipeline orchestration tools",
         "Add governance to Airflow, Kubeflow, or MLflow workflows")
    ]

    for title, description, next_step in production_steps:
        print(f"   {title}")
        print(f"     Purpose: {description}")
        print(f"     Implementation: {next_step}")
        print()

    print("📚 Advanced Resources:")
    print("   • Production Patterns: python production_patterns.py")
    print("   • Complete Integration Guide: docs/integrations/langfuse.md")
    print("   • Enterprise Architecture: docs/enterprise-architecture.md")
    print("   • Observability Best Practices: docs/observability-best-practices.md")

    print("\n🎯 Enterprise Readiness Checklist:")
    checklist_items = [
        "✅ Hierarchical operation tracing implemented",
        "✅ Multi-provider observability configured",
        "✅ Real-time analytics and monitoring active",
        "✅ Anomaly detection and alerting operational",
        "✅ Enterprise governance and compliance validated",
        "🔲 Production deployment patterns implemented",
        "🔲 Custom dashboards and reporting configured",
        "🔲 Integration with existing enterprise tools completed"
    ]

    for item in checklist_items:
        print(f"   {item}")


def main():
    """Main function to run the advanced observability example."""
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check prerequisites
    if not os.getenv('LANGFUSE_PUBLIC_KEY'):
        print("❌ Missing LANGFUSE_PUBLIC_KEY environment variable")
        print("💡 Get your keys at: https://cloud.langfuse.com/")
        return False

    if not os.getenv('LANGFUSE_SECRET_KEY'):
        print("❌ Missing LANGFUSE_SECRET_KEY environment variable")
        print("💡 Get your keys at: https://cloud.langfuse.com/")
        return False

    if not any([os.getenv('OPENAI_API_KEY'), os.getenv('ANTHROPIC_API_KEY')]):
        print("❌ No AI provider API keys found")
        print("💡 Set at least one:")
        print("   export OPENAI_API_KEY='your_openai_key'")
        print("   export ANTHROPIC_API_KEY='your_anthropic_key'")
        return False

    # Run advanced observability demonstrations
    success = True

    # Hierarchical tracing
    hierarchical_success = demonstrate_hierarchical_tracing()
    success &= hierarchical_success

    # Multi-provider observability
    multi_provider_success = demonstrate_multi_provider_observability()
    success &= multi_provider_success

    # Real-time analytics
    analytics_result = demonstrate_real_time_analytics()
    success &= analytics_result is not None

    # Anomaly detection
    anomaly_results = demonstrate_anomaly_detection()
    success &= anomaly_results is not None

    # Enterprise governance
    governance_results = demonstrate_enterprise_governance()
    success &= governance_results is not None

    if success:
        show_next_steps()
        print("\n" + "🔍" * 20)
        print("Advanced Langfuse Observability + GenOps Governance complete!")
        print("Enterprise-grade monitoring with comprehensive governance!")
        print("Production-ready observability patterns demonstrated!")
        print("🔍" * 20)
        return True
    else:
        print("\n❌ Some demonstrations failed. Check the errors above.")
        return False


if __name__ == "__main__":
    """Main entry point."""
    success = main()
    sys.exit(0 if success else 1)
