#!/usr/bin/env python3
"""
PromptLayer Advanced Observability with GenOps

This example demonstrates advanced observability patterns for PromptLayer operations,
including distributed tracing, custom metrics, dashboard integration, and real-time
monitoring with comprehensive governance intelligence.

This is the Level 3 (2-hour) example - Advanced observability and monitoring.

Usage:
    python advanced_observability.py

Prerequisites:
    pip install genops[promptlayer]  # Includes PromptLayer SDK
    export PROMPTLAYER_API_KEY="pl-your-api-key"
    export OPENAI_API_KEY="your-openai-key"  # For actual LLM calls
    
    # Required for governance attribution
    export GENOPS_TEAM="your-team"
    export GENOPS_PROJECT="your-project"
    
    # Optional: OTLP observability backend
    export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
"""

import os
import asyncio
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ObservabilityMetrics:
    """Advanced observability metrics for PromptLayer operations."""
    operation_id: str
    operation_type: str
    prompt_name: str
    
    # Timing metrics
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    
    # Resource metrics
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    memory_usage_mb: Optional[float] = None
    cpu_time_ms: Optional[float] = None
    
    # Quality metrics
    quality_score: Optional[float] = None
    safety_score: Optional[float] = None
    relevance_score: Optional[float] = None
    
    # Business metrics
    cost_usd: float = 0.0
    customer_satisfaction: Optional[float] = None
    business_value: Optional[float] = None
    
    # Governance context
    team: str = ""
    project: str = ""
    environment: str = ""
    customer_id: Optional[str] = None
    
    # Error tracking
    error_count: int = 0
    error_types: List[str] = field(default_factory=list)
    retry_count: int = 0
    
    # Custom dimensions
    custom_attributes: Dict[str, Any] = field(default_factory=dict)

class AdvancedObservabilityManager:
    """Advanced observability manager for comprehensive monitoring."""
    
    def __init__(self, adapter, enable_detailed_tracing: bool = True):
        self.adapter = adapter
        self.enable_detailed_tracing = enable_detailed_tracing
        self.metrics_buffer: List[ObservabilityMetrics] = []
        self.active_traces: Dict[str, ObservabilityMetrics] = {}
        
        # Custom metric collectors
        self.metric_collectors: List[Callable] = []
        
        logger.info("Advanced observability manager initialized")
    
    @contextmanager
    def trace_operation(self, operation_name: str, **kwargs):
        """Enhanced tracing context manager with detailed observability."""
        metrics = ObservabilityMetrics(
            operation_id=kwargs.get('operation_id', f"op_{int(time.time() * 1000)}"),
            operation_type=kwargs.get('operation_type', 'prompt_execution'),
            prompt_name=kwargs.get('prompt_name', operation_name),
            start_time=time.time(),
            team=self.adapter.team or "unknown",
            project=self.adapter.project or "unknown",
            environment=kwargs.get('environment', 'production'),
            customer_id=kwargs.get('customer_id')
        )
        
        self.active_traces[metrics.operation_id] = metrics
        
        try:
            if self.enable_detailed_tracing:
                logger.info(f"Starting traced operation: {operation_name} (ID: {metrics.operation_id})")
            
            yield metrics
            
        except Exception as e:
            metrics.error_count += 1
            metrics.error_types.append(type(e).__name__)
            logger.error(f"Operation {operation_name} failed: {e}")
            raise
            
        finally:
            metrics.end_time = time.time()
            metrics.duration_ms = (metrics.end_time - metrics.start_time) * 1000
            
            # Collect additional metrics
            for collector in self.metric_collectors:
                try:
                    collector(metrics)
                except Exception as e:
                    logger.warning(f"Metric collector failed: {e}")
            
            self.metrics_buffer.append(metrics)
            
            if metrics.operation_id in self.active_traces:
                del self.active_traces[metrics.operation_id]
            
            if self.enable_detailed_tracing:
                logger.info(f"Completed traced operation: {operation_name} "
                           f"(Duration: {metrics.duration_ms:.2f}ms, Cost: ${metrics.cost_usd:.6f})")
    
    def add_metric_collector(self, collector: Callable):
        """Add custom metric collector function."""
        self.metric_collectors.append(collector)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        if not self.metrics_buffer:
            return {"message": "No metrics collected yet"}
        
        total_operations = len(self.metrics_buffer)
        total_cost = sum(m.cost_usd for m in self.metrics_buffer)
        avg_duration = sum(m.duration_ms or 0 for m in self.metrics_buffer) / total_operations
        
        error_count = sum(m.error_count for m in self.metrics_buffer)
        success_rate = (total_operations - error_count) / total_operations if total_operations > 0 else 0
        
        return {
            "summary": {
                "total_operations": total_operations,
                "total_cost": total_cost,
                "average_duration_ms": avg_duration,
                "success_rate": success_rate,
                "error_rate": 1.0 - success_rate
            },
            "cost_breakdown": {
                "total_cost_usd": total_cost,
                "average_cost_per_operation": total_cost / total_operations,
                "cost_by_team": {self.adapter.team: total_cost}
            },
            "performance_metrics": {
                "avg_duration_ms": avg_duration,
                "p95_duration_ms": self._calculate_percentile([m.duration_ms or 0 for m in self.metrics_buffer], 0.95),
                "p99_duration_ms": self._calculate_percentile([m.duration_ms or 0 for m in self.metrics_buffer], 0.99)
            },
            "governance_context": {
                "team": self.adapter.team,
                "project": self.adapter.project,
                "environment": self.metrics_buffer[-1].environment if self.metrics_buffer else "unknown",
                "active_operations": len(self.active_traces)
            }
        }
    
    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        return sorted_values[min(index, len(sorted_values) - 1)]

def demonstrate_distributed_tracing():
    """
    Demonstrates distributed tracing with comprehensive observability.
    
    Shows how GenOps enables detailed tracing of PromptLayer operations
    with governance context, performance metrics, and error tracking.
    """
    print("🔍 Distributed Tracing with Advanced Observability")
    print("=" * 55)
    
    try:
        from genops.providers.promptlayer import instrument_promptlayer
        print("✅ GenOps PromptLayer adapter loaded successfully")
        
        # Initialize adapter with observability focus
        adapter = instrument_promptlayer(
            promptlayer_api_key=os.getenv('PROMPTLAYER_API_KEY'),
            team=os.getenv('GENOPS_TEAM', 'observability-team'),
            project=os.getenv('GENOPS_PROJECT', 'tracing-demo'),
            environment="production",
            enable_cost_alerts=True
        )
        
        # Initialize advanced observability manager
        obs_manager = AdvancedObservabilityManager(adapter)
        print("✅ Advanced observability manager configured")
        
    except ImportError as e:
        print(f"❌ Failed to import GenOps PromptLayer adapter: {e}")
        print("💡 Fix: Run 'pip install genops[promptlayer]'")
        return False
    
    # Add custom metric collectors
    def cost_efficiency_collector(metrics: ObservabilityMetrics):
        """Custom collector for cost efficiency metrics."""
        if metrics.quality_score and metrics.cost_usd > 0:
            metrics.custom_attributes['cost_per_quality_point'] = metrics.cost_usd / metrics.quality_score
    
    def performance_collector(metrics: ObservabilityMetrics):
        """Custom collector for performance metrics."""
        if metrics.duration_ms:
            if metrics.duration_ms < 1000:
                metrics.custom_attributes['performance_tier'] = 'fast'
            elif metrics.duration_ms < 3000:
                metrics.custom_attributes['performance_tier'] = 'normal'
            else:
                metrics.custom_attributes['performance_tier'] = 'slow'
    
    obs_manager.add_metric_collector(cost_efficiency_collector)
    obs_manager.add_metric_collector(performance_collector)
    
    print("\n🚀 Running Distributed Tracing Scenarios...")
    print("-" * 50)
    
    # Scenario 1: Complex multi-step workflow tracing
    print("\n1️⃣ Complex Multi-Step Workflow Tracing")
    try:
        with obs_manager.trace_operation(
            "customer_journey_workflow",
            operation_type="complex_workflow",
            customer_id="enterprise_customer_001"
        ) as workflow_metrics:
            
            # Step 1: Customer intent analysis
            with obs_manager.trace_operation(
                "intent_analysis",
                operation_type="prompt_execution",
                prompt_name="intent_classifier_v3"
            ) as intent_metrics:
                
                with adapter.track_prompt_operation(
                    prompt_name="intent_classifier_v3",
                    operation_type="classification",
                    operation_name="analyze_customer_intent",
                    customer_id="enterprise_customer_001"
                ) as span:
                    
                    result = adapter.run_prompt_with_governance(
                        prompt_name="intent_classifier_v3",
                        input_variables={
                            "customer_message": "I'm having trouble with my premium subscription billing",
                            "customer_tier": "enterprise",
                            "previous_interactions": 3
                        },
                        tags=["intent_analysis", "billing_category"]
                    )
                    
                    # Simulate metrics
                    intent_metrics.cost_usd = 0.008
                    intent_metrics.input_tokens = 85
                    intent_metrics.output_tokens = 45
                    intent_metrics.quality_score = 0.92
                    intent_metrics.custom_attributes['intent_confidence'] = 0.87
                    
                    span.update_cost(intent_metrics.cost_usd)
                    span.update_token_usage(intent_metrics.input_tokens, intent_metrics.output_tokens, "gpt-3.5-turbo")
                    
                    print("   ✅ Intent Analysis: Billing issue detected (Confidence: 87%)")
            
            # Step 2: Context enrichment
            with obs_manager.trace_operation(
                "context_enrichment",
                operation_type="prompt_execution",
                prompt_name="context_enricher_v2"
            ) as context_metrics:
                
                with adapter.track_prompt_operation(
                    prompt_name="context_enricher_v2",
                    operation_type="enrichment",
                    operation_name="enrich_customer_context"
                ) as span:
                    
                    result = adapter.run_prompt_with_governance(
                        prompt_name="context_enricher_v2",
                        input_variables={
                            "customer_id": "enterprise_customer_001",
                            "intent": "billing_inquiry",
                            "account_type": "enterprise"
                        },
                        tags=["context_enrichment", "customer_data"]
                    )
                    
                    context_metrics.cost_usd = 0.012
                    context_metrics.input_tokens = 120
                    context_metrics.output_tokens = 80
                    context_metrics.quality_score = 0.89
                    context_metrics.custom_attributes['context_completeness'] = 0.94
                    
                    span.update_cost(context_metrics.cost_usd)
                    span.update_token_usage(context_metrics.input_tokens, context_metrics.output_tokens, "gpt-3.5-turbo")
                    
                    print("   ✅ Context Enrichment: Customer profile enhanced (Completeness: 94%)")
            
            # Step 3: Response generation
            with obs_manager.trace_operation(
                "response_generation",
                operation_type="prompt_execution",
                prompt_name="customer_response_v4"
            ) as response_metrics:
                
                with adapter.track_prompt_operation(
                    prompt_name="customer_response_v4",
                    operation_type="generation",
                    operation_name="generate_customer_response"
                ) as span:
                    
                    result = adapter.run_prompt_with_governance(
                        prompt_name="customer_response_v4",
                        input_variables={
                            "intent": "billing_inquiry",
                            "context": "enterprise customer, premium support tier",
                            "urgency": "medium"
                        },
                        tags=["response_generation", "customer_service"]
                    )
                    
                    response_metrics.cost_usd = 0.018
                    response_metrics.input_tokens = 150
                    response_metrics.output_tokens = 200
                    response_metrics.quality_score = 0.91
                    response_metrics.custom_attributes['response_completeness'] = 0.96
                    response_metrics.custom_attributes['tone_appropriateness'] = 0.93
                    
                    span.update_cost(response_metrics.cost_usd)
                    span.update_token_usage(response_metrics.input_tokens, response_metrics.output_tokens, "gpt-3.5-turbo")
                    
                    print("   ✅ Response Generation: Personalized response created (Quality: 91%)")
            
            # Aggregate workflow metrics
            workflow_metrics.cost_usd = intent_metrics.cost_usd + context_metrics.cost_usd + response_metrics.cost_usd
            workflow_metrics.input_tokens = intent_metrics.input_tokens + context_metrics.input_tokens + response_metrics.input_tokens
            workflow_metrics.output_tokens = intent_metrics.output_tokens + context_metrics.output_tokens + response_metrics.output_tokens
            workflow_metrics.quality_score = (intent_metrics.quality_score + context_metrics.quality_score + response_metrics.quality_score) / 3
            workflow_metrics.custom_attributes['workflow_steps'] = 3
            workflow_metrics.custom_attributes['total_operations'] = 3
            
            print(f"\n   📊 Workflow Complete:")
            print(f"      Total Cost: ${workflow_metrics.cost_usd:.6f}")
            print(f"      Total Tokens: {workflow_metrics.input_tokens + workflow_metrics.output_tokens}")
            print(f"      Average Quality: {workflow_metrics.quality_score:.3f}")
            print(f"      Duration: {workflow_metrics.duration_ms:.0f}ms")
    
    except Exception as e:
        print(f"❌ Workflow tracing failed: {e}")
        return False
    
    # Scenario 2: Real-time performance monitoring
    print("\n2️⃣ Real-Time Performance Monitoring")
    try:
        performance_scenarios = [
            {"name": "quick_response", "expected_duration": 800, "load_factor": 1.0},
            {"name": "normal_processing", "expected_duration": 1500, "load_factor": 1.5},
            {"name": "complex_analysis", "expected_duration": 3000, "load_factor": 2.5},
            {"name": "batch_processing", "expected_duration": 5000, "load_factor": 4.0}
        ]
        
        performance_results = []
        
        for scenario in performance_scenarios:
            with obs_manager.trace_operation(
                f"perf_test_{scenario['name']}",
                operation_type="performance_test",
                prompt_name=f"perf_prompt_{scenario['name']}"
            ) as perf_metrics:
                
                # Simulate operation with realistic timing
                start = time.time()
                
                with adapter.track_prompt_operation(
                    prompt_name=f"perf_prompt_{scenario['name']}",
                    operation_type="performance_benchmark",
                    operation_name=f"benchmark_{scenario['name']}"
                ) as span:
                    
                    # Simulate processing time
                    processing_delay = scenario['expected_duration'] / 1000
                    await asyncio.sleep(processing_delay * 0.1)  # Scale down for demo
                    
                    result = adapter.run_prompt_with_governance(
                        prompt_name=f"perf_prompt_{scenario['name']}",
                        input_variables={
                            "complexity": scenario['load_factor'],
                            "scenario": scenario['name']
                        },
                        tags=["performance_test", f"complexity_{scenario['load_factor']}"]
                    )
                    
                    actual_duration = (time.time() - start) * 1000
                    
                    perf_metrics.cost_usd = 0.005 * scenario['load_factor']
                    perf_metrics.input_tokens = int(50 * scenario['load_factor'])
                    perf_metrics.output_tokens = int(100 * scenario['load_factor'])
                    perf_metrics.quality_score = min(0.95, 0.80 + (0.15 / scenario['load_factor']))
                    perf_metrics.custom_attributes['load_factor'] = scenario['load_factor']
                    perf_metrics.custom_attributes['expected_duration'] = scenario['expected_duration']
                    
                    span.update_cost(perf_metrics.cost_usd)
                    span.update_token_usage(perf_metrics.input_tokens, perf_metrics.output_tokens, "gpt-3.5-turbo")
                    
                    # Performance analysis
                    performance_ratio = actual_duration / scenario['expected_duration']
                    if performance_ratio <= 1.1:
                        performance_status = "✅ OPTIMAL"
                    elif performance_ratio <= 1.3:
                        performance_status = "⚠️ ACCEPTABLE"
                    else:
                        performance_status = "🚨 DEGRADED"
                    
                    performance_results.append({
                        "scenario": scenario['name'],
                        "expected": scenario['expected_duration'],
                        "actual": actual_duration,
                        "ratio": performance_ratio,
                        "status": performance_status,
                        "cost": perf_metrics.cost_usd
                    })
                    
                    print(f"   {performance_status} {scenario['name']}: "
                          f"{actual_duration:.0f}ms (expected: {scenario['expected_duration']}ms)")
        
        # Performance summary
        print(f"\n   📊 Performance Monitoring Summary:")
        avg_ratio = sum(r['ratio'] for r in performance_results) / len(performance_results)
        total_cost = sum(r['cost'] for r in performance_results)
        
        print(f"      Average Performance Ratio: {avg_ratio:.2f}x expected")
        print(f"      Total Monitoring Cost: ${total_cost:.6f}")
        print(f"      Performance Tier Distribution:")
        
        optimal_count = sum(1 for r in performance_results if "OPTIMAL" in r['status'])
        acceptable_count = sum(1 for r in performance_results if "ACCEPTABLE" in r['status'])
        degraded_count = sum(1 for r in performance_results if "DEGRADED" in r['status'])
        
        print(f"        • Optimal: {optimal_count} scenarios")
        print(f"        • Acceptable: {acceptable_count} scenarios")
        print(f"        • Degraded: {degraded_count} scenarios")
        
    except Exception as e:
        print(f"❌ Performance monitoring failed: {e}")
        return False
    
    # Scenario 3: Comprehensive metrics dashboard
    print("\n3️⃣ Comprehensive Metrics Dashboard")
    try:
        metrics_summary = obs_manager.get_metrics_summary()
        
        print("   📊 Real-Time Metrics Dashboard:")
        print("   " + "=" * 40)
        
        # Summary metrics
        summary = metrics_summary.get('summary', {})
        print(f"   Operations: {summary.get('total_operations', 0)}")
        print(f"   Success Rate: {summary.get('success_rate', 0):.2%}")
        print(f"   Total Cost: ${summary.get('total_cost', 0):.6f}")
        print(f"   Avg Duration: {summary.get('average_duration_ms', 0):.0f}ms")
        
        # Cost breakdown
        cost_breakdown = metrics_summary.get('cost_breakdown', {})
        print(f"\n   💰 Cost Analysis:")
        print(f"   Avg Cost/Op: ${cost_breakdown.get('average_cost_per_operation', 0):.6f}")
        
        # Performance metrics
        perf_metrics = metrics_summary.get('performance_metrics', {})
        print(f"\n   ⚡ Performance Metrics:")
        print(f"   P95 Duration: {perf_metrics.get('p95_duration_ms', 0):.0f}ms")
        print(f"   P99 Duration: {perf_metrics.get('p99_duration_ms', 0):.0f}ms")
        
        # Governance context
        governance = metrics_summary.get('governance_context', {})
        print(f"\n   🛡️ Governance Context:")
        print(f"   Team: {governance.get('team', 'unknown')}")
        print(f"   Project: {governance.get('project', 'unknown')}")
        print(f"   Environment: {governance.get('environment', 'unknown')}")
        print(f"   Active Operations: {governance.get('active_operations', 0)}")
        
        # Custom metrics from collectors
        if obs_manager.metrics_buffer:
            custom_metrics = []
            for metrics in obs_manager.metrics_buffer:
                custom_metrics.extend(metrics.custom_attributes.keys())
            
            if custom_metrics:
                print(f"\n   🔧 Custom Metrics Available:")
                unique_metrics = set(custom_metrics)
                for metric in sorted(unique_metrics):
                    print(f"   • {metric}")
        
    except Exception as e:
        print(f"❌ Metrics dashboard failed: {e}")
        return False
    
    return True

def demonstrate_alerting_integration():
    """Demonstrate alerting and notification integration."""
    print("\n🚨 Advanced Alerting and Notification Integration")
    print("-" * 45)
    
    try:
        from genops.providers.promptlayer import instrument_promptlayer
        
        adapter = instrument_promptlayer(
            team="sre-team",
            project="alerting-demo",
            daily_budget_limit=1.0,  # Low limit to trigger alerts
            max_operation_cost=0.05,
            enable_cost_alerts=True
        )
        
        # Simulate alert scenarios
        alert_scenarios = [
            {
                "name": "cost_threshold_exceeded",
                "operation_cost": 0.08,  # Exceeds 0.05 limit
                "expected_alert": "cost_limit_violation"
            },
            {
                "name": "quality_degradation",
                "quality_score": 0.65,   # Below 0.75 threshold
                "expected_alert": "quality_degradation"
            },
            {
                "name": "error_rate_spike",
                "error_rate": 0.15,      # Above 0.05 threshold
                "expected_alert": "error_rate_spike"
            },
            {
                "name": "latency_anomaly",
                "duration_ms": 8000,     # Above 5000ms threshold
                "expected_alert": "latency_anomaly"
            }
        ]
        
        alerts_generated = []
        
        print("🔔 Alert Scenario Testing:")
        
        for scenario in alert_scenarios:
            scenario_name = scenario["name"]
            
            with adapter.track_prompt_operation(
                prompt_name=f"alert_test_{scenario_name}",
                operation_type="alert_testing",
                operation_name=f"test_{scenario_name}"
            ) as span:
                
                # Simulate scenario conditions
                if "cost_threshold" in scenario_name:
                    span.update_cost(scenario["operation_cost"])
                    
                    if scenario["operation_cost"] > 0.05:
                        alert = {
                            "type": scenario["expected_alert"],
                            "severity": "warning",
                            "message": f"Operation cost ${scenario['operation_cost']:.6f} exceeds limit $0.05",
                            "team": adapter.team,
                            "project": adapter.project
                        }
                        alerts_generated.append(alert)
                        print(f"   🚨 ALERT: {alert['message']}")
                    else:
                        print(f"   ✅ {scenario_name}: Within cost limits")
                
                elif "quality_degradation" in scenario_name:
                    quality_score = scenario["quality_score"]
                    
                    if quality_score < 0.75:
                        alert = {
                            "type": scenario["expected_alert"],
                            "severity": "critical",
                            "message": f"Quality score {quality_score:.3f} below threshold 0.750",
                            "team": adapter.team,
                            "project": adapter.project
                        }
                        alerts_generated.append(alert)
                        print(f"   🚨 CRITICAL ALERT: {alert['message']}")
                    else:
                        print(f"   ✅ {scenario_name}: Quality within acceptable range")
                
                elif "error_rate" in scenario_name:
                    error_rate = scenario["error_rate"]
                    
                    if error_rate > 0.05:
                        alert = {
                            "type": scenario["expected_alert"],
                            "severity": "critical",
                            "message": f"Error rate {error_rate:.1%} exceeds threshold 5%",
                            "team": adapter.team,
                            "project": adapter.project
                        }
                        alerts_generated.append(alert)
                        print(f"   🚨 CRITICAL ALERT: {alert['message']}")
                    else:
                        print(f"   ✅ {scenario_name}: Error rate within limits")
                
                elif "latency_anomaly" in scenario_name:
                    duration_ms = scenario["duration_ms"]
                    
                    if duration_ms > 5000:
                        alert = {
                            "type": scenario["expected_alert"],
                            "severity": "warning",
                            "message": f"Operation latency {duration_ms}ms exceeds threshold 5000ms",
                            "team": adapter.team,
                            "project": adapter.project
                        }
                        alerts_generated.append(alert)
                        print(f"   🚨 ALERT: {alert['message']}")
                    else:
                        print(f"   ✅ {scenario_name}: Latency within acceptable range")
        
        # Alert summary
        print(f"\n   📊 Alert Summary:")
        print(f"      Total Alerts Generated: {len(alerts_generated)}")
        print(f"      Alert Types:")
        
        alert_types = {}
        severity_counts = {}
        
        for alert in alerts_generated:
            alert_type = alert["type"]
            severity = alert["severity"]
            
            alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        for alert_type, count in alert_types.items():
            print(f"        • {alert_type}: {count}")
        
        print(f"      Severity Distribution:")
        for severity, count in severity_counts.items():
            icon = "🚨" if severity == "critical" else "⚠️"
            print(f"        • {icon} {severity}: {count}")
        
        # Governance integration
        print(f"\n   🛡️ Governance Integration:")
        print(f"      • All alerts attributed to team: {adapter.team}")
        print(f"      • Project context preserved: {adapter.project}")
        print(f"      • Cost attribution enabled for budget tracking")
        print(f"      • Policy violations logged for compliance audit")
        
    except Exception as e:
        print(f"❌ Alerting integration demo failed: {e}")

async def main():
    """Main execution function."""
    print("🚀 Starting PromptLayer Advanced Observability Demo")
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check prerequisites
    if not os.getenv('PROMPTLAYER_API_KEY'):
        print("❌ PROMPTLAYER_API_KEY not found")
        print("💡 Set your PromptLayer API key: export PROMPTLAYER_API_KEY='pl-your-key'")
        print("📖 Get your API key from: https://promptlayer.com/")
        return False
    
    # Run demonstrations
    success = True
    
    # Distributed tracing
    if not await demonstrate_distributed_tracing():
        success = False
    
    # Alerting integration
    if success:
        demonstrate_alerting_integration()
    
    if success:
        print("\n" + "🌟" * 60)
        print("🎉 PromptLayer Advanced Observability Demo Complete!")
        print("\n📊 What You've Mastered:")
        print("   ✅ Distributed tracing with comprehensive governance context")
        print("   ✅ Real-time performance monitoring and alerting")
        print("   ✅ Custom metrics collection and analysis")
        print("   ✅ Advanced dashboard integration with OpenTelemetry")
        
        print("\n🔍 Your Advanced Observability Stack:")
        print("   • PromptLayer: Prompt management and execution platform")
        print("   • GenOps: Advanced governance and cost intelligence")
        print("   • OpenTelemetry: Distributed tracing and metrics export")
        print("   • Custom Collectors: Extensible metric collection framework")
        
        print("\n📚 Next Steps:")
        print("   • Production deployment: python production_patterns.py")
        print("   • Complete test suite: pytest tests/promptlayer/")
        print("   • Integration with your observability stack (Datadog, Grafana, etc.)")
        print("   • Run all examples: ./run_all_examples.sh")
        
        print("\n💡 Observability Integration Pattern:")
        print("   ```python")
        print("   # Advanced tracing with custom metrics")
        print("   with obs_manager.trace_operation('complex_workflow') as metrics:")
        print("       metrics.custom_attributes['business_metric'] = calculate_value()")
        print("       result = execute_with_governance()")
        print("       metrics.quality_score = evaluate_quality(result)")
        print("   ```")
        
        print("\n🔗 Export Integration:")
        print("   • OTLP Protocol: Standard observability platform integration")
        print("   • Custom Exporters: Datadog, Grafana, Prometheus, Honeycomb")
        print("   • Real-time Dashboards: Cost, performance, and quality metrics")
        print("   • Alerting: Proactive monitoring with governance context")
        
        print("🌟" * 60)
    else:
        print("\n❌ Demo encountered errors. Please check the output above.")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)