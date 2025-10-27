#!/usr/bin/env python3
"""
🍯 Honeycomb Integration for GenOps AI Observability

This example demonstrates how to integrate GenOps AI telemetry with Honeycomb
for high-cardinality AI governance observability and analysis.

Features:
✅ OpenTelemetry OTLP export to Honeycomb  
✅ High-cardinality attribution analysis
✅ AI operation performance analysis
✅ Cost attribution with flexible grouping
✅ Compliance monitoring with drill-down
✅ Example queries for common use cases
"""

import os
import time
from typing import Dict, Any, Optional

import genops

# OpenTelemetry imports for Honeycomb integration
try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False


class HoneycombGenOpsIntegration:
    """Integration class for sending GenOps AI telemetry to Honeycomb."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        dataset: str = "genops-ai",
        service_name: str = "genops-ai",
        environment: str = "production"
    ):
        self.api_key = api_key or os.getenv("HONEYCOMB_API_KEY")
        self.dataset = dataset
        self.service_name = service_name
        self.environment = environment
        
        if not self.api_key:
            print("⚠️ HONEYCOMB_API_KEY not set. Using console export for demo.")
        
        self._setup_opentelemetry()
    
    def _setup_opentelemetry(self):
        """Set up OpenTelemetry exporters for Honeycomb."""
        
        if not HAS_OPENTELEMETRY:
            print("❌ OpenTelemetry not available.")
            return
        
        # Create resource
        resource = Resource.create({
            "service.name": self.service_name,
            "service.version": "1.0.0",
            "deployment.environment": self.environment,
            "honeycomb.dataset": self.dataset
        })
        
        # Set up tracing
        trace_provider = TracerProvider(resource=resource)
        
        if self.api_key:
            # Honeycomb OTLP endpoint
            span_exporter = OTLPSpanExporter(
                endpoint="https://api.honeycomb.io/v1/traces",
                headers={"X-Honeycomb-Team": self.api_key}
            )
        else:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            span_exporter = ConsoleSpanExporter()
        
        trace_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        trace.set_tracer_provider(trace_provider)
        
        print(f"✅ Honeycomb integration configured")
        print(f"   Dataset: {self.dataset}")
        print(f"   Service: {self.service_name}")


def demonstrate_honeycomb_telemetry():
    """Demonstrate GenOps AI telemetry flowing to Honeycomb."""
    
    print("\n🍯 HONEYCOMB TELEMETRY DEMONSTRATION")
    print("=" * 60)
    
    # Initialize Honeycomb integration
    honeycomb = HoneycombGenOpsIntegration(
        dataset="genops-ai-demo", 
        service_name="genops-demo",
        environment="development"
    )
    
    # Set up attribution
    genops.set_default_attributes(
        team="ai-platform",
        project="honeycomb-integration",
        environment="development"
    )
    
    # Generate sample AI operations
    operations = [
        {"customer_id": "enterprise-123", "feature": "chat", "user_tier": "premium"},
        {"customer_id": "startup-456", "feature": "analysis", "user_tier": "basic"}, 
        {"customer_id": "enterprise-789", "feature": "search", "user_tier": "enterprise"}
    ]
    
    print("🤖 Generating operations with high-cardinality attributes...")
    
    for i, op_attrs in enumerate(operations):
        genops.set_context(**op_attrs)
        
        # Simulate operation
        time.sleep(0.1)
        
        # Record with effective attributes
        effective = genops.get_effective_attributes()
        cost = 0.0234 * (i + 1)
        
        print(f"   Operation {i+1}: {effective.get('feature')} (${cost:.4f})")
        print(f"      Customer: {effective.get('customer_id')}")
        print(f"      Tier: {effective.get('user_tier')}")
        
        genops.clear_context()
    
    print("\n✅ High-cardinality telemetry sent to Honeycomb!")


def show_honeycomb_queries():
    """Show example Honeycomb queries for GenOps AI governance."""
    
    print("\n🔍 HONEYCOMB QUERY EXAMPLES")
    print("=" * 60)
    
    queries = {
        "Cost Analysis": [
            "WHERE genops.operation.type = 'ai.inference' | GROUP BY genops.customer_id | SUM(genops.cost.total)",
            "WHERE genops.cost.provider = 'openai' | HEATMAP(genops.cost.total, genops.tokens.total)",
            "GROUP BY genops.team, genops.feature | AVG(genops.cost.total) | ORDER BY AVG DESC"
        ],
        "Performance Analysis": [
            "WHERE genops.operation.name CONTAINS 'chat' | P95(duration_ms) | GROUP BY genops.customer_tier", 
            "GROUP BY genops.cost.provider | HEATMAP(duration_ms, genops.tokens.total)",
            "WHERE genops.eval.safety < 0.9 | COUNT | GROUP BY genops.team"
        ],
        "Attribution Analysis": [
            "GROUP BY genops.customer_id, genops.feature | SUM(genops.cost.total) | ORDER BY SUM DESC",
            "WHERE genops.environment = 'production' | GROUP BY genops.user_tier | COUNT",
            "GROUP BY genops.data.classification | AVG(genops.eval.privacy) | ORDER BY AVG ASC"
        ]
    }
    
    for category, query_list in queries.items():
        print(f"\n🍯 {category}:")
        for query in query_list:
            print(f"   {query}")
    
    print(f"\n💡 Honeycomb Query Tips:")
    print("• Use WHERE to filter by any attribute")
    print("• GROUP BY enables multi-dimensional analysis") 
    print("• HEATMAP shows correlation between metrics")
    print("• Use P50, P95, P99 for performance percentiles")
    print("• CONTAINS enables substring matching")


def main():
    """Run the Honeycomb integration demonstration."""
    
    print("🍯 GenOps AI: Honeycomb Integration Guide")
    print("=" * 80)
    
    try:
        demonstrate_honeycomb_telemetry()
        show_honeycomb_queries()
        
        print(f"\n🎯 HONEYCOMB INTEGRATION BENEFITS")
        print("=" * 60)
        print("✅ High-cardinality attribution analysis")
        print("✅ Flexible grouping and filtering")
        print("✅ Real-time AI governance insights") 
        print("✅ Cost optimization queries")
        print("✅ Performance correlation analysis")
        
        print(f"\n🔧 SETUP INSTRUCTIONS")
        print("=" * 60)
        print("1. Set HONEYCOMB_API_KEY environment variable")
        print("2. Install: pip install opentelemetry-exporter-otlp")
        print("3. Configure dataset and service names")
        print("4. Start sending telemetry data")
        print("5. Create custom queries and boards")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    main()