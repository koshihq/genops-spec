"""
GenOps Elasticsearch Integration Example

Demonstrates complete integration of GenOps governance telemetry with Elasticsearch,
including multi-provider cost tracking, policy enforcement, and budget management.

Prerequisites:
    - Elasticsearch 8.x or 9.x running (local or cloud)
    - Environment variables set (ELASTIC_URL or ELASTIC_CLOUD_ID)
    - GenOps AI installed with Elasticsearch support:
      pip install 'genops-ai[elastic]'

Usage:
    # Set environment variables
    export ELASTIC_URL=http://localhost:9200
    export ELASTIC_API_KEY=your-api-key  # Optional

    # Run example
    python elastic_integration.py

    # View results in Kibana
    # Navigate to Discover and search: genops.team: "ml-platform"
"""

import os
import time
from typing import Dict, Any

from genops.providers.elastic import (
    auto_instrument,
    validate_setup,
    print_validation_result,
)


class ElasticGenOpsIntegration:
    """
    Example integration demonstrating GenOps Elasticsearch telemetry export.

    Features:
    - Multi-provider cost tracking (OpenAI, Anthropic, Bedrock)
    - Policy enforcement recording
    - Budget management
    - Batch and realtime export modes
    - KQL query examples
    """

    def __init__(
        self,
        elastic_url: str = None,
        team: str = "ml-platform",
        project: str = "recommendations",
        environment: str = "development",
    ):
        """
        Initialize Elasticsearch integration.

        Args:
            elastic_url: Elasticsearch URL (defaults to ELASTIC_URL env var)
            team: Team for governance attribution
            project: Project for cost tracking
            environment: Environment (development/staging/production)
        """
        self.team = team
        self.project = project
        self.environment = environment

        # Validate setup before initialization
        print("=" * 70)
        print("Step 1: Validating Elasticsearch Setup")
        print("=" * 70)
        validation_result = validate_setup(elastic_url=elastic_url)
        print_validation_result(validation_result)

        if not validation_result.valid:
            raise RuntimeError(
                "Elasticsearch setup validation failed. "
                "Please fix the errors above and try again."
            )

        # Auto-instrument with batch mode
        print("\n" + "=" * 70)
        print("Step 2: Initializing GenOps Elasticsearch Adapter")
        print("=" * 70)

        self.adapter = auto_instrument(
            team=team,
            project=project,
            environment=environment,
            export_mode="batch",
            batch_size=10,  # Small batch for demo purposes
            batch_interval_seconds=5,  # Fast flush for demo
        )

        print(f"✅ Adapter initialized:")
        print(f"   • Team: {team}")
        print(f"   • Project: {project}")
        print(f"   • Environment: {environment}")
        print(f"   • Export Mode: batch")
        print(f"   • Cluster: {validation_result.cluster_name} ({validation_result.cluster_version})")

    def demonstrate_elastic_telemetry(self):
        """
        Demonstrate telemetry export for various AI operations.

        Simulates:
        - OpenAI GPT-4 completions
        - Anthropic Claude operations
        - AWS Bedrock operations
        - Policy enforcement
        - Budget tracking
        """
        print("\n" + "=" * 70)
        print("Step 3: Tracking AI Operations")
        print("=" * 70)

        # Example 1: OpenAI GPT-4 with cost tracking
        print("\n[1/5] OpenAI GPT-4 Completion")
        with self.adapter.track_ai_operation(
            "gpt4-completion",
            customer_id="acme-corp",
            feature="personalization"
        ) as span:
            # Simulate AI operation
            time.sleep(0.1)

            # Record cost telemetry
            self.adapter.record_cost(
                span,
                cost=0.05,
                provider="openai",
                model="gpt-4",
                tokens_input=50,
                tokens_output=150,
                cost_input=0.015,
                cost_output=0.035
            )

            print("✅ Tracked GPT-4 completion:")
            print("   • Cost: $0.05")
            print("   • Tokens: 50 input + 150 output")
            print("   • Customer: acme-corp")

        # Example 2: Anthropic Claude with policy enforcement
        print("\n[2/5] Anthropic Claude with Policy Check")
        with self.adapter.track_ai_operation(
            "claude-completion",
            customer_id="techcorp",
            feature="content-generation"
        ) as span:
            time.sleep(0.1)

            # Record cost
            self.adapter.record_cost(
                span,
                cost=0.03,
                provider="anthropic",
                model="claude-3-sonnet",
                tokens_input=100,
                tokens_output=200
            )

            # Record policy enforcement
            self.adapter.record_policy(
                span,
                policy_name="budget-constraint",
                result="allowed",
                reason="Within monthly budget"
            )

            print("✅ Tracked Claude completion:")
            print("   • Cost: $0.03")
            print("   • Policy: budget-constraint -> allowed")
            print("   • Customer: techcorp")

        # Example 3: AWS Bedrock with policy violation
        print("\n[3/5] AWS Bedrock with Policy Violation")
        with self.adapter.track_ai_operation(
            "bedrock-completion",
            customer_id="startup-xyz",
            feature="chatbot"
        ) as span:
            time.sleep(0.1)

            # Record cost
            self.adapter.record_cost(
                span,
                cost=0.02,
                provider="bedrock",
                model="anthropic.claude-v2",
                tokens_input=75,
                tokens_output=125
            )

            # Record policy violation
            self.adapter.record_policy(
                span,
                policy_name="pii-detection",
                result="warning",
                reason="Potential PII detected in prompt"
            )

            print("✅ Tracked Bedrock completion:")
            print("   • Cost: $0.02")
            print("   • Policy: pii-detection -> warning")
            print("   • Customer: startup-xyz")

        # Example 4: Budget tracking
        print("\n[4/5] Budget Tracking")
        with self.adapter.track_ai_operation(
            "gpt4-with-budget",
            customer_id="enterprise-co"
        ) as span:
            time.sleep(0.1)

            # Record cost
            self.adapter.record_cost(
                span,
                cost=0.08,
                provider="openai",
                model="gpt-4",
                tokens_input=100,
                tokens_output=300
            )

            # Record budget tracking
            self.adapter.record_budget(
                span,
                budget_id="team-monthly",
                limit=1000.0,
                consumed=750.0,
                remaining=250.0
            )

            print("✅ Tracked GPT-4 with budget:")
            print("   • Cost: $0.08")
            print("   • Budget: $750/$1000 consumed ($250 remaining)")
            print("   • Customer: enterprise-co")

        # Example 5: High-cost operation
        print("\n[5/5] High-Cost Operation")
        with self.adapter.track_ai_operation(
            "gpt4-large-context",
            customer_id="data-corp",
            feature="document-analysis"
        ) as span:
            time.sleep(0.1)

            # Record high cost
            self.adapter.record_cost(
                span,
                cost=1.25,
                provider="openai",
                model="gpt-4",
                tokens_input=5000,
                tokens_output=2000
            )

            print("✅ Tracked high-cost operation:")
            print("   • Cost: $1.25 (high-cost alert threshold)")
            print("   • Tokens: 5000 input + 2000 output")
            print("   • Customer: data-corp")

        # Force flush to Elasticsearch
        print("\n" + "=" * 70)
        print("Step 4: Flushing Data to Elasticsearch")
        print("=" * 70)

        print("\nFlushing batch buffer...")
        exported = self.adapter.flush()
        print(f"✅ Exported {exported} operations to Elasticsearch")

        # Wait a moment for indexing
        time.sleep(2)

        # Show export statistics
        stats = self.adapter.get_export_summary()
        print("\n📊 Export Statistics:")
        print(f"   • Total Exported: {stats['total_exported']}")
        print(f"   • Total Failed: {stats['total_failed']}")
        print(f"   • Total Batches: {stats['total_batches']}")
        print(f"   • Last Batch Size: {stats['last_batch_size']}")
        print(f"   • Last Export Duration: {stats['last_export_duration_ms']:.2f}ms")

    def show_elastic_queries(self):
        """Display useful Kibana KQL queries for analyzing the telemetry."""
        print("\n" + "=" * 70)
        print("Step 5: Kibana Query Examples")
        print("=" * 70)

        queries = [
            {
                "name": "All operations for your team",
                "query": f'genops.team: "{self.team}"',
                "description": "View all AI operations for your team"
            },
            {
                "name": "Cost attribution by customer",
                "query": 'genops.cost.total > 0 | stats sum(genops.cost.total) by genops.customer_id',
                "description": "Sum total costs grouped by customer"
            },
            {
                "name": "Policy violations",
                "query": 'genops.policy.result: "blocked" OR genops.policy.result: "warning"',
                "description": "Find all policy violations and warnings"
            },
            {
                "name": "High-cost operations (>$1)",
                "query": 'genops.cost.total > 1.0 | sort genops.cost.total desc',
                "description": "Find expensive operations"
            },
            {
                "name": "Operations by model",
                "query": 'genops.cost.model: * | stats count(), sum(genops.cost.total) by genops.cost.model',
                "description": "Compare usage and costs across models"
            },
            {
                "name": "Budget tracking",
                "query": 'genops.budget.id: * | stats latest(genops.budget.consumed), latest(genops.budget.remaining) by genops.budget.id',
                "description": "Monitor budget consumption"
            },
            {
                "name": "Provider comparison",
                "query": 'genops.cost.provider: * | stats sum(genops.cost.total), avg(genops.cost.total), count() by genops.cost.provider',
                "description": "Compare costs across OpenAI, Anthropic, Bedrock"
            },
        ]

        print("\n📋 Copy these KQL queries into Kibana Discover:\n")

        for i, q in enumerate(queries, 1):
            print(f"[{i}] {q['name']}")
            print(f"    {q['description']}")
            print(f"    Query: {q['query']}\n")

        print("💡 Tips:")
        print("   • Create index pattern: genops-ai-* (with timestamp field)")
        print("   • Import pre-built dashboards from: observability/elastic/dashboards/")
        print("   • Set time range to 'Last 1 hour' in Kibana (top-right corner)")

    def create_dashboards(self):
        """
        Demonstrate programmatic dashboard creation (optional).

        Note: This is a simplified example. For production, use the pre-built
        dashboard NDJSON files in observability/elastic/dashboards/
        """
        print("\n" + "=" * 70)
        print("Step 6: Dashboard Creation (Optional)")
        print("=" * 70)

        print("\n📊 Pre-built dashboards available:")
        print("   1. AI Operations Overview")
        print("      • Request volume over time")
        print("      • Success/error rates")
        print("      • Latency percentiles")
        print("\n   2. Cost Attribution")
        print("      • Total cost by team/project")
        print("      • Cost by model and provider")
        print("      • Cost trends over time")
        print("\n   3. Governance & Compliance")
        print("      • Policy violations by type")
        print("      • Budget consumption tracking")
        print("      • Compliance status by team")

        print("\n💡 To import dashboards:")
        print("   1. Navigate to: Management → Saved Objects")
        print("   2. Click 'Import'")
        print("   3. Select dashboard NDJSON file")
        print("   4. Click 'Import'")

        print("\n📁 Dashboard files located at:")
        print("   observability/elastic/dashboards/")

    def cleanup(self):
        """Gracefully shutdown adapter."""
        print("\n" + "=" * 70)
        print("Step 7: Cleanup")
        print("=" * 70)

        print("\nShutting down adapter...")
        self.adapter.shutdown()
        print("✅ Adapter shutdown complete")


def main():
    """Run the Elasticsearch integration example."""
    print("\n")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  GenOps Elasticsearch Integration Example                        ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")

    # Check environment variables
    elastic_url = os.getenv("ELASTIC_URL")
    cloud_id = os.getenv("ELASTIC_CLOUD_ID")

    if not elastic_url and not cloud_id:
        print("\n❌ Error: No Elasticsearch connection configured")
        print("\nPlease set environment variables:")
        print("   export ELASTIC_URL=http://localhost:9200")
        print("   # OR")
        print("   export ELASTIC_CLOUD_ID=<your-cloud-id>")
        print("   export ELASTIC_API_KEY=<your-api-key>")
        print("\nFor more help, run:")
        print("   python -m genops.providers.elastic.validation")
        return 1

    try:
        # Initialize integration
        integration = ElasticGenOpsIntegration(
            elastic_url=elastic_url,
            team="ml-platform",
            project="recommendations",
            environment="development"
        )

        # Demonstrate telemetry export
        integration.demonstrate_elastic_telemetry()

        # Show useful queries
        integration.show_elastic_queries()

        # Show dashboard info
        integration.create_dashboards()

        # Cleanup
        integration.cleanup()

        print("\n" + "=" * 70)
        print("✅ Example Complete!")
        print("=" * 70)
        print("\n🎉 Success! Your telemetry is now in Elasticsearch.")
        print("\n📊 Next steps:")
        print("   1. Open Kibana: http://localhost:5601")
        print("   2. Create index pattern: genops-ai-*")
        print("   3. Navigate to Discover and explore your data")
        print("   4. Import pre-built dashboards from: observability/elastic/dashboards/")
        print("\n📚 Documentation:")
        print("   • Quickstart: docs/quickstarts/elastic-quickstart.md")
        print("   • Full guide: docs/integrations/elastic.md")
        print("\n")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("   1. Verify Elasticsearch is running: curl http://localhost:9200")
        print("   2. Check environment variables: echo $ELASTIC_URL")
        print("   3. Run validation: python -m genops.providers.elastic.validation")
        return 1


if __name__ == "__main__":
    exit(main())
