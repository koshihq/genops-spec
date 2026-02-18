#!/usr/bin/env python3
"""
📊 Splunk Integration for GenOps AI Observability

This example demonstrates how to integrate GenOps AI telemetry with Splunk
for comprehensive AI governance observability, compliance monitoring, and cost analytics.

Features:
✅ OpenTelemetry OTLP export to Splunk HEC (HTTP Event Collector)
✅ SPL (Search Processing Language) query templates
✅ Dashboard configuration examples (XML)
✅ Cost attribution analytics
✅ Policy compliance monitoring
✅ Budget threshold alerting
✅ Audit trail for regulated industries
✅ Cribl routing path documentation

Integration Paths:
• Direct: GenOps → OTLP → Splunk HEC
• Pipeline: GenOps → OTLP → Cribl → Splunk

Splunk is ideal for:
• Enterprise log analytics and SIEM
• Compliance and audit trail requirements
• Complex ad-hoc governance queries with SPL
• Long-term retention for regulated industries
"""

import os
import time
from typing import Optional

import genops

# OpenTelemetry imports for Splunk integration
try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False
    print(
        "⚠️ OpenTelemetry not installed. Install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp"
    )


class SplunkGenOpsIntegration:
    """
    Integration class for sending GenOps AI telemetry to Splunk.

    This class sets up OpenTelemetry exporters for Splunk HEC and provides
    utilities for creating SPL queries, dashboards, and alerts.

    Splunk HEC Configuration:
    - Endpoint: https://splunk.example.com:8088/services/collector/raw
    - Authentication: Bearer token (HEC token)
    - Index: genops_ai (recommended)
    - Sourcetype: genops:telemetry
    """

    def __init__(
        self,
        splunk_hec_endpoint: Optional[str] = None,
        splunk_hec_token: Optional[str] = None,
        splunk_index: str = "genops_ai",
        splunk_sourcetype: str = "genops:telemetry",
        service_name: str = "genops-ai",
        environment: str = "production",
        **config,
    ):
        """
        Initialize Splunk GenOps integration.

        Args:
            splunk_hec_endpoint: Splunk HEC endpoint (e.g., https://splunk.example.com:8088)
            splunk_hec_token: HEC authentication token
            splunk_index: Target Splunk index for telemetry data
            splunk_sourcetype: Sourcetype for telemetry events
            service_name: Service name for OpenTelemetry resource
            environment: Deployment environment (production, staging, development)
            **config: Additional configuration options
        """
        self.splunk_hec_endpoint = splunk_hec_endpoint or os.getenv(
            "SPLUNK_HEC_ENDPOINT"
        )
        self.splunk_hec_token = splunk_hec_token or os.getenv("SPLUNK_HEC_TOKEN")
        self.splunk_index = splunk_index or os.getenv("SPLUNK_INDEX", "genops_ai")
        self.splunk_sourcetype = splunk_sourcetype
        self.service_name = service_name
        self.environment = environment
        self.config = config

        if not self.splunk_hec_endpoint:
            print("⚠️ SPLUNK_HEC_ENDPOINT not set. Using console export for demo.")
        if not self.splunk_hec_token:
            print("⚠️ SPLUNK_HEC_TOKEN not set. Using console export for demo.")

        # Set up OpenTelemetry for Splunk HEC
        self._setup_opentelemetry()

    def _setup_opentelemetry(self):
        """Set up OpenTelemetry exporters for Splunk HEC."""

        if not HAS_OPENTELEMETRY:
            print("❌ OpenTelemetry not available. Telemetry will not be exported.")
            return

        # Create resource with service information and Splunk-specific attributes
        resource = Resource.create(
            {
                "service.name": self.service_name,
                "service.version": "1.0.0",
                "deployment.environment": self.environment,
                "genops.framework": "splunk-integration",
                "splunk.index": self.splunk_index,
                "splunk.sourcetype": self.splunk_sourcetype,
            }
        )

        # Set up tracing
        trace_provider = TracerProvider(resource=resource)

        if self.splunk_hec_endpoint and self.splunk_hec_token:
            # Splunk HEC OTLP endpoint
            # HEC supports OTLP via /services/collector/raw endpoint
            hec_otlp_endpoint = f"{self.splunk_hec_endpoint}/services/collector/raw"

            # Splunk HEC authentication header
            headers = {
                "Authorization": f"Splunk {self.splunk_hec_token}",
                "X-Splunk-Request-Channel": os.getenv("SPLUNK_CHANNEL", ""),
            }

            # Set up OTLP span exporter for Splunk HEC
            span_exporter = OTLPSpanExporter(
                endpoint=hec_otlp_endpoint, headers=headers
            )

            print("✅ Splunk HEC OTLP exporter configured")
            print(f"   Endpoint: {self.splunk_hec_endpoint}")
            print(f"   Index: {self.splunk_index}")
            print(f"   Sourcetype: {self.splunk_sourcetype}")
        else:
            # Console export for demo
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter

            span_exporter = ConsoleSpanExporter()
            print("✅ Console exporter configured (demo mode)")

        # Add span processor
        trace_provider.add_span_processor(BatchSpanProcessor(span_exporter))

        # Set global tracer provider
        trace.set_tracer_provider(trace_provider)

        print("✅ OpenTelemetry configured for Splunk export")
        print(f"   Service: {self.service_name}")
        print(f"   Environment: {self.environment}")

    def create_spl_query(self, use_case: str, **kwargs) -> str:
        """
        Generate SPL queries for common GenOps governance use cases.

        Args:
            use_case: Query use case (cost_by_team, policy_violations, budget_alerts, etc.)
            **kwargs: Additional parameters for query customization

        Returns:
            SPL query string ready to run in Splunk Search
        """
        index = kwargs.get("index", self.splunk_index)

        queries = {
            "cost_by_team": f"""index={index} genops.cost.total=*
| stats sum(genops.cost.total) as total_cost by genops.team
| sort -total_cost
| eval total_cost_formatted=printf("$%.4f", total_cost)""",
            "cost_by_model": f"""index={index} genops.cost.model=*
| stats sum(genops.cost.total) as total_cost by genops.cost.model, genops.cost.provider
| sort -total_cost
| eval total_cost_formatted=printf("$%.4f", total_cost)""",
            "cost_trends": f"""index={index} genops.cost.total=*
| timechart span=1h sum(genops.cost.total) as total_cost by genops.project
| fillnull value=0""",
            "policy_violations": f"""index={index} genops.policy.result="blocked"
| table _time genops.policy.name genops.policy.reason genops.team genops.customer_id
| sort -_time""",
            "budget_alerts": f"""index={index} genops.budget.utilization=*
| where genops.budget.utilization > 80
| table _time genops.budget.name genops.budget.limit genops.budget.used genops.budget.utilization genops.team
| sort -genops.budget.utilization""",
            "compliance_audit": f"""index={index} genops.policy.* OR genops.eval.*
| table _time genops.operation.name genops.customer_id genops.team genops.policy.result genops.eval.safety genops.data.classification
| sort -_time""",
            "customer_cost_attribution": f"""index={index} genops.cost.total=* genops.customer_id=*
| stats sum(genops.cost.total) as total_cost count as request_count by genops.customer_id
| eval avg_cost_per_request=total_cost/request_count
| eval total_cost_formatted=printf("$%.4f", total_cost)
| eval avg_cost_formatted=printf("$%.4f", avg_cost_per_request)
| sort -total_cost""",
            "model_performance": f"""index={index} genops.eval.*
| stats avg(genops.eval.quality) as avg_quality avg(genops.eval.safety) as avg_safety count by genops.cost.model
| eval avg_quality_pct=round(avg_quality*100, 2)
| eval avg_safety_pct=round(avg_safety*100, 2)
| sort -avg_quality""",
            "realtime_cost_monitor": f"""index={index} genops.cost.total=*
| bin _time span=5m
| stats sum(genops.cost.total) as cost_5min by _time, genops.team
| eval cost_formatted=printf("$%.4f", cost_5min)""",
        }

        if use_case in queries:
            return queries[use_case]
        else:
            # Return a generic query template
            return f"""index={index} genops.*
| table _time genops.*
| sort -_time
| head 100"""

    def create_cost_dashboard(self) -> str:
        """
        Create a Splunk XML dashboard configuration for AI cost governance.

        Returns:
            XML dashboard configuration string
        """
        dashboard_xml = f"""<dashboard version="2">
  <label>GenOps AI - Cost Governance</label>
  <description>AI cost attribution, trend analysis, and optimization insights</description>

  <row>
    <panel>
      <title>Total Cost (Last 24h)</title>
      <single>
        <search>
          <query>index={self.splunk_index} genops.cost.total=* earliest=-24h
| stats sum(genops.cost.total) as total_cost
| eval total_cost_formatted="$" + tostring(round(total_cost, 2))</query>
        </search>
        <option name="drilldown">none</option>
        <option name="numberPrecision">0.00</option>
        <option name="rangeColors">["0x53a051","0x0877a6","0xf8be34","0xf1813f","0xdc4e41"]</option>
      </single>
    </panel>

    <panel>
      <title>Total Requests (Last 24h)</title>
      <single>
        <search>
          <query>index={self.splunk_index} genops.cost.total=* earliest=-24h
| stats count as total_requests</query>
        </search>
        <option name="drilldown">none</option>
      </single>
    </panel>

    <panel>
      <title>Average Cost Per Request</title>
      <single>
        <search>
          <query>index={self.splunk_index} genops.cost.total=* earliest=-24h
| stats sum(genops.cost.total) as total_cost count as requests
| eval avg_cost=total_cost/requests
| eval avg_cost_formatted="$" + tostring(round(avg_cost, 4))</query>
        </search>
        <option name="drilldown">none</option>
        <option name="numberPrecision">0.0000</option>
      </single>
    </panel>
  </row>

  <row>
    <panel>
      <title>Cost by Team</title>
      <chart>
        <search>
          <query>index={self.splunk_index} genops.cost.total=* earliest=-24h
| stats sum(genops.cost.total) as total_cost by genops.team
| sort -total_cost</query>
        </search>
        <option name="charting.chart">pie</option>
        <option name="charting.drilldown">all</option>
        <option name="charting.legend.placement">right</option>
      </chart>
    </panel>

    <panel>
      <title>Cost by Model</title>
      <chart>
        <search>
          <query>index={self.splunk_index} genops.cost.model=* earliest=-24h
| stats sum(genops.cost.total) as total_cost by genops.cost.model
| sort -total_cost</query>
        </search>
        <option name="charting.chart">bar</option>
        <option name="charting.axisTitleX.text">Model</option>
        <option name="charting.axisTitleY.text">Total Cost ($)</option>
      </chart>
    </panel>
  </row>

  <row>
    <panel>
      <title>Cost Trend Over Time</title>
      <chart>
        <search>
          <query>index={self.splunk_index} genops.cost.total=* earliest=-24h
| timechart span=1h sum(genops.cost.total) as total_cost by genops.project</query>
        </search>
        <option name="charting.chart">area</option>
        <option name="charting.chart.stackMode">stacked</option>
        <option name="charting.axisTitleX.text">Time</option>
        <option name="charting.axisTitleY.text">Cost ($)</option>
        <option name="charting.legend.placement">bottom</option>
      </chart>
    </panel>
  </row>

  <row>
    <panel>
      <title>Top 10 Customers by Cost</title>
      <table>
        <search>
          <query>index={self.splunk_index} genops.customer_id=* genops.cost.total=* earliest=-24h
| stats sum(genops.cost.total) as total_cost count as requests by genops.customer_id
| eval avg_cost=total_cost/requests
| eval total_cost_formatted=printf("$%.2f", total_cost)
| eval avg_cost_formatted=printf("$%.4f", avg_cost)
| sort -total_cost
| head 10
| fields genops.customer_id total_cost_formatted requests avg_cost_formatted</query>
        </search>
        <option name="drilldown">row</option>
        <option name="count">10</option>
      </table>
    </panel>
  </row>
</dashboard>"""
        return dashboard_xml

    def create_compliance_dashboard(self) -> str:
        """
        Create a Splunk XML dashboard configuration for policy compliance monitoring.

        Returns:
            XML dashboard configuration string
        """
        dashboard_xml = f"""<dashboard version="2">
  <label>GenOps AI - Compliance Monitoring</label>
  <description>Policy violations, audit trails, and compliance metrics</description>

  <row>
    <panel>
      <title>Policy Violations (Last 24h)</title>
      <single>
        <search>
          <query>index={self.splunk_index} genops.policy.result="blocked" earliest=-24h
| stats count as violations</query>
        </search>
        <option name="drilldown">all</option>
        <option name="rangeColors">["0x53a051","0xf8be34","0xdc4e41"]</option>
        <option name="rangeValues">[0,10]</option>
      </single>
    </panel>

    <panel>
      <title>Compliance Rate</title>
      <single>
        <search>
          <query>index={self.splunk_index} genops.policy.result=* earliest=-24h
| stats count(eval(genops.policy.result="allowed")) as allowed count as total
| eval compliance_rate=round((allowed/total)*100, 2)
| eval compliance_formatted=tostring(compliance_rate) + "%"</query>
        </search>
        <option name="drilldown">none</option>
        <option name="numberPrecision">0.00</option>
        <option name="rangeColors">["0xdc4e41","0xf8be34","0x53a051"]</option>
        <option name="rangeValues">[90,95]</option>
      </single>
    </panel>

    <panel>
      <title>Average Safety Score</title>
      <single>
        <search>
          <query>index={self.splunk_index} genops.eval.safety=* earliest=-24h
| stats avg(genops.eval.safety) as avg_safety
| eval avg_safety_pct=round(avg_safety*100, 2)
| eval safety_formatted=tostring(avg_safety_pct) + "%"</query>
        </search>
        <option name="drilldown">none</option>
        <option name="rangeColors">["0xdc4e41","0xf8be34","0x53a051"]</option>
        <option name="rangeValues">[0.8,0.9]</option>
      </single>
    </panel>
  </row>

  <row>
    <panel>
      <title>Violations by Policy Type</title>
      <chart>
        <search>
          <query>index={self.splunk_index} genops.policy.result="blocked" earliest=-24h
| stats count by genops.policy.name
| sort -count</query>
        </search>
        <option name="charting.chart">bar</option>
        <option name="charting.axisTitleX.text">Policy Type</option>
        <option name="charting.axisTitleY.text">Violation Count</option>
      </chart>
    </panel>

    <panel>
      <title>Violations by Team</title>
      <chart>
        <search>
          <query>index={self.splunk_index} genops.policy.result="blocked" genops.team=* earliest=-24h
| stats count by genops.team
| sort -count</query>
        </search>
        <option name="charting.chart">pie</option>
        <option name="charting.legend.placement">right</option>
      </chart>
    </panel>
  </row>

  <row>
    <panel>
      <title>Violation Trend Over Time</title>
      <chart>
        <search>
          <query>index={self.splunk_index} genops.policy.result="blocked" earliest=-24h
| timechart span=1h count as violations by genops.policy.name</query>
        </search>
        <option name="charting.chart">line</option>
        <option name="charting.axisTitleX.text">Time</option>
        <option name="charting.axisTitleY.text">Violations</option>
        <option name="charting.legend.placement">bottom</option>
      </chart>
    </panel>
  </row>

  <row>
    <panel>
      <title>Recent Policy Violations</title>
      <table>
        <search>
          <query>index={self.splunk_index} genops.policy.result="blocked" earliest=-24h
| table _time genops.policy.name genops.policy.reason genops.team genops.customer_id genops.operation.name
| sort -_time
| head 50</query>
        </search>
        <option name="drilldown">row</option>
        <option name="count">20</option>
      </table>
    </panel>
  </row>

  <row>
    <panel>
      <title>Compliance Audit Trail</title>
      <table>
        <search>
          <query>index={self.splunk_index} (genops.policy.* OR genops.eval.*) earliest=-24h
| table _time genops.operation.name genops.customer_id genops.team genops.policy.result genops.eval.safety genops.data.classification
| sort -_time
| head 100</query>
        </search>
        <option name="drilldown">row</option>
        <option name="count">20</option>
      </table>
    </panel>
  </row>
</dashboard>"""
        return dashboard_xml

    def create_budget_dashboard(self) -> str:
        """
        Create a Splunk XML dashboard configuration for budget monitoring and alerting.

        Returns:
            XML dashboard configuration string
        """
        dashboard_xml = f"""<dashboard version="2">
  <label>GenOps AI - Budget Monitoring</label>
  <description>Budget utilization, thresholds, and cost alerts</description>

  <row>
    <panel>
      <title>Budgets Over 80% Utilized</title>
      <single>
        <search>
          <query>index={self.splunk_index} genops.budget.utilization=* earliest=-1h
| stats max(genops.budget.utilization) as max_util by genops.budget.name
| where max_util > 80
| stats count as over_threshold</query>
        </search>
        <option name="drilldown">all</option>
        <option name="rangeColors">["0x53a051","0xf8be34","0xdc4e41"]</option>
        <option name="rangeValues">[0,1]</option>
      </single>
    </panel>

    <panel>
      <title>Total Budget Allocated</title>
      <single>
        <search>
          <query>index={self.splunk_index} genops.budget.limit=* earliest=-1h
| stats max(genops.budget.limit) as limit by genops.budget.name
| stats sum(limit) as total_budget
| eval total_budget_formatted="$" + tostring(round(total_budget, 2))</query>
        </search>
        <option name="drilldown">none</option>
      </single>
    </panel>

    <panel>
      <title>Total Budget Consumed</title>
      <single>
        <search>
          <query>index={self.splunk_index} genops.budget.used=* earliest=-1h
| stats max(genops.budget.used) as used by genops.budget.name
| stats sum(used) as total_used
| eval total_used_formatted="$" + tostring(round(total_used, 2))</query>
        </search>
        <option name="drilldown">none</option>
      </single>
    </panel>
  </row>

  <row>
    <panel>
      <title>Budget Utilization by Team</title>
      <chart>
        <search>
          <query>index={self.splunk_index} genops.budget.utilization=* genops.team=* earliest=-1h
| stats max(genops.budget.utilization) as utilization by genops.team
| eval utilization_pct=round(utilization, 1)
| sort -utilization_pct</query>
        </search>
        <option name="charting.chart">bar</option>
        <option name="charting.axisTitleX.text">Team</option>
        <option name="charting.axisTitleY.text">Utilization (%)</option>
        <option name="charting.chart.rangeValues">[0,50,80,100]</option>
      </chart>
    </panel>
  </row>

  <row>
    <panel>
      <title>Budget Status Details</title>
      <table>
        <search>
          <query>index={self.splunk_index} genops.budget.* earliest=-1h
| stats max(genops.budget.limit) as limit max(genops.budget.used) as used max(genops.budget.remaining) as remaining max(genops.budget.utilization) as utilization by genops.budget.name, genops.team
| eval limit_formatted=printf("$%.2f", limit)
| eval used_formatted=printf("$%.2f", used)
| eval remaining_formatted=printf("$%.2f", remaining)
| eval utilization_pct=round(utilization, 1) + "%"
| eval status=case(utilization >= 90, "CRITICAL", utilization >= 80, "WARNING", utilization >= 0, "OK")
| sort -utilization
| fields genops.budget.name genops.team limit_formatted used_formatted remaining_formatted utilization_pct status</query>
        </search>
        <option name="drilldown">row</option>
        <option name="count">20</option>
      </table>
    </panel>
  </row>

  <row>
    <panel>
      <title>Budget Utilization Trend</title>
      <chart>
        <search>
          <query>index={self.splunk_index} genops.budget.utilization=* earliest=-24h
| timechart span=1h max(genops.budget.utilization) as utilization by genops.budget.name</query>
        </search>
        <option name="charting.chart">line</option>
        <option name="charting.axisTitleX.text">Time</option>
        <option name="charting.axisTitleY.text">Utilization (%)</option>
        <option name="charting.legend.placement">bottom</option>
      </chart>
    </panel>
  </row>
</dashboard>"""
        return dashboard_xml

    def validate_configuration(self):
        """
        Validate current Splunk HEC configuration.

        This method checks:
        - Environment variables are set correctly
        - HEC endpoint is accessible
        - HEC token authentication works
        - Index write permissions
        - OpenTelemetry dependencies

        Returns:
            SplunkValidationResult with validation details

        Example:
            >>> splunk = SplunkGenOpsIntegration()
            >>> result = splunk.validate_configuration()
            >>> if result.valid:
            ...     print("Configuration is valid!")
        """
        try:
            from splunk_validation import validate_setup
        except ImportError:
            print("❌ splunk_validation module not found")
            print("   Ensure splunk_validation.py is in the same directory")
            return None

        return validate_setup(
            splunk_hec_endpoint=self.splunk_hec_endpoint,
            splunk_hec_token=self.splunk_hec_token,
            splunk_index=self.splunk_index,
        )

    def print_validation(self) -> bool:
        """
        Validate and print configuration status.

        Returns:
            True if validation passed, False otherwise

        Example:
            >>> splunk = SplunkGenOpsIntegration()
            >>> if splunk.print_validation():
            ...     print("Ready to send telemetry!")
        """
        try:
            from splunk_validation import print_validation_result
        except ImportError:
            print("❌ splunk_validation module not found")
            print("   Ensure splunk_validation.py is in the same directory")
            return False

        result = self.validate_configuration()
        if result:
            print_validation_result(result)
            return result.valid
        return False


def demonstrate_splunk_telemetry():
    """Demonstrate GenOps AI telemetry flowing to Splunk HEC."""

    print("\n📊 SPLUNK HEC TELEMETRY DEMONSTRATION")
    print("=" * 70)

    # Initialize Splunk integration
    splunk = SplunkGenOpsIntegration(
        service_name="genops-demo", environment="development"
    )

    # Validate configuration before proceeding
    print("\n🔍 Validating Splunk HEC configuration...")
    if not splunk.print_validation():
        print("\n❌ Validation failed. Fix configuration errors before proceeding.")
        print("   Set environment variables:")
        print("     export SPLUNK_HEC_ENDPOINT='https://splunk.example.com:8088'")
        print("     export SPLUNK_HEC_TOKEN='your-hec-token'")
        return

    print("\n✅ Configuration validated! Proceeding with demo...\n")

    # Set up default attribution
    genops.set_default_attributes(
        team="ai-platform", project="splunk-integration-demo", environment="development"
    )

    print("\n🤖 Generating sample AI operations with governance telemetry...")

    # Example 1: Cost tracking
    print("\n1. Cost Tracking Example:")
    with genops.track_enhanced(
        operation_name="ai.chat.completion",
        customer_id="enterprise-123",
        feature="customer-support-chat",
    ) as span:
        # Simulate AI operation
        time.sleep(0.1)

        # Record cost
        genops.record_cost(
            span,
            provider="openai",
            model="gpt-4",
            input_tokens=1500,
            output_tokens=500,
            total_cost=0.0325,
        )
        print("   ✅ Recorded: $0.0325 (GPT-4, 1500 in / 500 out tokens)")

    # Example 2: Policy violation
    print("\n2. Policy Compliance Example:")
    with genops.track_enhanced(
        operation_name="ai.content.moderation",
        customer_id="startup-456",
        feature="content-filter",
    ) as span:
        time.sleep(0.1)

        # Record policy evaluation
        genops.record_policy(
            span,
            policy_name="content_safety",
            policy_result="blocked",
            policy_reason="Potentially harmful content detected",
            metadata={"confidence": 0.95},
        )
        print("   ⚠️  Blocked: Content safety policy violation (confidence: 95%)")

    # Example 3: Budget tracking
    print("\n3. Budget Monitoring Example:")
    with genops.track_enhanced(
        operation_name="ai.budget.check", team="ai-platform"
    ) as span:
        time.sleep(0.1)

        # Record budget status
        genops.record_budget(
            span,
            budget_name="team-daily-budget",
            budget_limit=100.0,
            budget_used=87.50,
            budget_remaining=12.50,
            metadata={"utilization_percent": 87.5},
        )
        print("   💰 Budget: $87.50 / $100.00 (87.5% utilized)")

    print("\n✅ Sample telemetry sent to Splunk HEC!")
    print("   Check Splunk Search: index=" + splunk.splunk_index)


def show_splunk_queries():
    """Show example SPL queries for GenOps AI governance."""

    print("\n🔍 SPLUNK SPL QUERY EXAMPLES")
    print("=" * 70)

    splunk = SplunkGenOpsIntegration()

    query_examples = {
        "Cost Analysis": [
            ("Total cost by team", splunk.create_spl_query("cost_by_team")),
            ("Cost by model", splunk.create_spl_query("cost_by_model")),
            ("Cost trends over time", splunk.create_spl_query("cost_trends")),
        ],
        "Policy Compliance": [
            ("Recent policy violations", splunk.create_spl_query("policy_violations")),
            ("Compliance audit trail", splunk.create_spl_query("compliance_audit")),
        ],
        "Budget Monitoring": [
            ("Budgets over 80% threshold", splunk.create_spl_query("budget_alerts")),
            (
                "Real-time cost monitoring",
                splunk.create_spl_query("realtime_cost_monitor"),
            ),
        ],
        "Customer Analytics": [
            (
                "Customer cost attribution",
                splunk.create_spl_query("customer_cost_attribution"),
            ),
            ("Model performance metrics", splunk.create_spl_query("model_performance")),
        ],
    }

    for category, queries in query_examples.items():
        print(f"\n📊 {category}:")
        for title, query in queries:
            print(f"\n   {title}:")
            # Print first 2 lines of query
            query_lines = query.strip().split("\n")
            for line in query_lines[:2]:
                print(f"      {line}")
            if len(query_lines) > 2:
                print(f"      ... ({len(query_lines) - 2} more lines)")

    print("\n💡 SPL Query Tips:")
    print("• Use 'index=genops_ai' to search AI governance telemetry")
    print("• Filter with 'genops.cost.* OR genops.policy.* OR genops.budget.*'")
    print("• Use '| stats' for aggregations (sum, avg, count, max)")
    print("• Use '| timechart' for time-series visualizations")
    print("• Use '| where' for conditional filtering")


def show_cribl_routing_path():
    """Document GenOps → Cribl → Splunk routing path."""

    print("\n🔄 CRIBL ROUTING PATH")
    print("=" * 70)
    print("\nGenOps can route telemetry to Splunk via Cribl Stream for:")
    print("• Multi-destination routing (Splunk + Datadog + S3 simultaneously)")
    print("• Intelligent sampling (reduce volume by 90%+)")
    print("• Data enrichment and transformation")
    print("• Cost optimization with conditional routing")

    print("\n📋 Configuration Steps:")
    print("1. Configure GenOps → Cribl OTLP endpoint")
    print("   export CRIBL_OTLP_ENDPOINT='http://cribl-stream:4318'")

    print("\n2. Add Splunk HEC destination in Cribl")
    print("   - Navigate to: Data → Destinations → Splunk HEC")
    print("   - Configure endpoint, token, index")

    print("\n3. Create routing rule in Cribl")
    print("   - Route filter: __inputId == 'genops_otlp_source'")
    print("   - Destination: splunk_hec")

    print("\n4. Optional: Add sampling/filtering pipeline")
    print("   - Sample 10% of low-cost operations")
    print("   - Route all policy violations to Splunk")
    print("   - Enrich with additional metadata")

    print("\n✅ Benefits:")
    print("• Single GenOps configuration routes to multiple destinations")
    print("• Cribl handles retries, buffering, and backpressure")
    print("• Transform GenOps attributes to Splunk-specific fields")
    print("• Apply governance-specific routing logic")


def main():
    """Run the Splunk integration demonstration."""

    print("📊 GenOps AI: Splunk Integration Guide")
    print("=" * 80)

    try:
        # Demonstrate telemetry flow
        demonstrate_splunk_telemetry()

        # Show SPL query examples
        show_splunk_queries()

        # Document Cribl routing
        show_cribl_routing_path()

        print("\n🎯 SPLUNK INTEGRATION BENEFITS")
        print("=" * 70)
        print("✅ Enterprise log analytics and SIEM capabilities")
        print("✅ Complex ad-hoc governance queries with SPL")
        print("✅ Compliance audit trails for regulated industries")
        print("✅ Long-term retention and archival")
        print("✅ Cost attribution analytics across teams/customers")
        print("✅ Policy violation monitoring and alerting")
        print("✅ Budget threshold enforcement")

        print("\n🔧 SETUP INSTRUCTIONS")
        print("=" * 70)
        print("1. Enable Splunk HTTP Event Collector (HEC)")
        print("   Settings → Data Inputs → HTTP Event Collector → New Token")

        print("\n2. Set environment variables:")
        print("   export SPLUNK_HEC_ENDPOINT='https://splunk.example.com:8088'")
        print("   export SPLUNK_HEC_TOKEN='your-hec-token'")
        print("   export SPLUNK_INDEX='genops_ai'")

        print("\n3. Install GenOps AI:")
        print("   pip install genops-ai")
        print("   pip install opentelemetry-exporter-otlp")

        print("\n4. Import dashboards:")
        print("   splunk import dashboard cost_governance.xml")
        print("   splunk import dashboard compliance_monitoring.xml")
        print("   splunk import dashboard budget_alerting.xml")

        print("\n5. Start sending telemetry:")
        print("   python examples/observability/splunk_integration.py")

        print("\n📚 DOCUMENTATION")
        print("=" * 70)
        print("• Quickstart Guide: docs/splunk-quickstart.md")
        print("• Full Integration Guide: docs/integrations/splunk.md")
        print("• Cribl Routing: docs/integrations/cribl.md")
        print("• SPL Query Reference: docs/integrations/splunk.md#spl-queries")
        print("• Dashboard Templates: examples/observability/splunk_dashboards/")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
