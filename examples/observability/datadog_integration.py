#!/usr/bin/env python3
"""
📊 Datadog Integration for GenOps AI Observability

This example demonstrates how to integrate GenOps AI telemetry with Datadog
for comprehensive AI governance observability and monitoring.

Features:
✅ OpenTelemetry OTLP export to Datadog
✅ Custom metrics for AI governance
✅ Dashboard configuration examples
✅ Alerting rules for compliance violations
✅ Cost attribution queries and dashboards
✅ Performance monitoring and SLIs
✅ Multi-tenant observability isolation
"""

import json
import os
import socket
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

import genops

# OpenTelemetry imports for Datadog integration
try:
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False
    print("⚠️ OpenTelemetry not installed. Install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp")

# Optional Datadog-specific integrations
try:
    from datadog import initialize, statsd
    HAS_DATADOG = True
except ImportError:
    HAS_DATADOG = False


class DatadogGenOpsIntegration:
    """
    Integration class for sending GenOps AI telemetry to Datadog.

    This class sets up OpenTelemetry exporters for Datadog and provides
    utilities for creating dashboards, alerts, and queries.
    """

    def __init__(
        self,
        datadog_api_key: Optional[str] = None,
        datadog_app_key: Optional[str] = None,
        datadog_site: str = "datadoghq.com",
        service_name: str = "genops-ai",
        environment: str = "production",
        **config
    ):
        self.datadog_api_key = datadog_api_key or os.getenv("DATADOG_API_KEY")
        self.datadog_app_key = datadog_app_key or os.getenv("DATADOG_APP_KEY")
        self.datadog_site = datadog_site
        self.service_name = service_name
        self.environment = environment
        self.config = config

        if not self.datadog_api_key:
            print("⚠️ DATADOG_API_KEY not set. Using console export for demo.")

        # Set up OpenTelemetry for Datadog
        self._setup_opentelemetry()

        # Set up Datadog direct integration if available
        if HAS_DATADOG and self.datadog_api_key and self.datadog_app_key:
            self._setup_datadog_direct()

    def _setup_opentelemetry(self):
        """Set up OpenTelemetry exporters for Datadog."""

        if not HAS_OPENTELEMETRY:
            print("❌ OpenTelemetry not available. Telemetry will not be exported.")
            return

        # Create resource with service information
        resource = Resource.create({
            "service.name": self.service_name,
            "service.version": "1.0.0",
            "deployment.environment": self.environment,
            "genops.framework": "datadog-integration"
        })

        # Set up tracing
        trace_provider = TracerProvider(resource=resource)

        # Datadog OTLP endpoint
        if self.datadog_api_key:
            otlp_endpoint = f"https://otlp.{self.datadog_site}"
            headers = {"DD-API-KEY": self.datadog_api_key}

            # Set up OTLP span exporter
            span_exporter = OTLPSpanExporter(
                endpoint=f"{otlp_endpoint}/v1/traces",
                headers=headers
            )
        else:
            # Console export for demo
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter
            span_exporter = ConsoleSpanExporter()

        # Add span processor
        trace_provider.add_span_processor(
            BatchSpanProcessor(span_exporter)
        )

        # Set global tracer provider
        trace.set_tracer_provider(trace_provider)

        # Set up metrics
        if self.datadog_api_key:
            metric_exporter = OTLPMetricExporter(
                endpoint=f"{otlp_endpoint}/v1/metrics",
                headers=headers
            )
        else:
            from opentelemetry.sdk.metrics.export import ConsoleMetricExporter
            metric_exporter = ConsoleMetricExporter()

        # Create metric reader
        metric_reader = PeriodicExportingMetricReader(
            exporter=metric_exporter,
            export_interval_millis=10000  # 10 seconds
        )

        # Set up metrics provider
        metrics_provider = MeterProvider(
            resource=resource,
            metric_readers=[metric_reader]
        )

        # Set global metrics provider
        metrics.set_meter_provider(metrics_provider)

        print("✅ OpenTelemetry configured for Datadog export")
        print(f"   Service: {self.service_name}")
        print(f"   Environment: {self.environment}")
        if self.datadog_api_key:
            print(f"   Datadog Site: {self.datadog_site}")
        else:
            print("   Export Mode: Console (demo)")

    def _setup_datadog_direct(self):
        """Set up direct Datadog integration for custom metrics."""

        initialize(
            api_key=self.datadog_api_key,
            app_key=self.datadog_app_key,
            host_name=f"{self.service_name}-{self.environment}"
        )

        print("✅ Datadog direct integration configured")

    def send_custom_metric(
        self,
        metric_name: str,
        value: float,
        tags: Optional[dict[str, str]] = None
    ):
        """Send a custom metric to Datadog via StatsD."""

        if not HAS_DATADOG:
            print(f"📊 Custom Metric (demo): {metric_name} = {value}")
            return

        # Convert tags to Datadog format
        tag_list = []
        if tags:
            for key, val in tags.items():
                tag_list.append(f"{key}:{val}")

        # Send metric
        statsd.gauge(f"genops.{metric_name}", value, tags=tag_list)

        print(f"📊 Sent custom metric: genops.{metric_name} = {value} {tag_list}")

    def create_ai_cost_dashboard(self) -> dict[str, Any]:
        """Create a Datadog dashboard configuration for AI cost monitoring."""

        dashboard_config = {
            "title": "GenOps AI - Cost Attribution & Governance",
            "description": "Comprehensive AI cost tracking and governance monitoring",
            "widgets": [
                {
                    "id": "ai-cost-overview",
                    "definition": {
                        "title": "AI Cost Overview",
                        "type": "timeseries",
                        "requests": [
                            {
                                "q": "sum:genops.cost.total{*} by {genops.team,genops.project}",
                                "display_type": "line"
                            }
                        ]
                    }
                },
                {
                    "id": "cost-by-customer",
                    "definition": {
                        "title": "Cost by Customer",
                        "type": "toplist",
                        "requests": [
                            {
                                "q": "sum:genops.cost.total{*} by {genops.customer_id}",
                                "limit": 20
                            }
                        ]
                    }
                },
                {
                    "id": "token-usage",
                    "definition": {
                        "title": "Token Usage by Provider",
                        "type": "query_value",
                        "requests": [
                            {
                                "q": "sum:genops.tokens.total{*} by {genops.cost.provider}",
                                "aggregator": "sum"
                            }
                        ]
                    }
                },
                {
                    "id": "policy-violations",
                    "definition": {
                        "title": "Policy Violations",
                        "type": "timeseries",
                        "requests": [
                            {
                                "q": "sum:genops.policy.violation{*} by {genops.policy.name}",
                                "display_type": "bars"
                            }
                        ]
                    }
                },
                {
                    "id": "evaluation-scores",
                    "definition": {
                        "title": "AI Evaluation Scores",
                        "type": "heatmap",
                        "requests": [
                            {
                                "q": "avg:genops.eval.safety{*} by {genops.team,genops.feature}",
                            }
                        ]
                    }
                },
                {
                    "id": "cost-per-operation",
                    "definition": {
                        "title": "Average Cost per Operation",
                        "type": "query_value",
                        "requests": [
                            {
                                "q": "avg:genops.cost.total{*}",
                                "aggregator": "avg"
                            }
                        ]
                    }
                }
            ],
            "template_variables": [
                {
                    "name": "team",
                    "prefix": "genops.team",
                    "available_values": []
                },
                {
                    "name": "environment",
                    "prefix": "genops.environment",
                    "available_values": ["production", "staging", "development"]
                },
                {
                    "name": "customer_id",
                    "prefix": "genops.customer_id",
                    "available_values": []
                }
            ],
            "layout_type": "ordered"
        }

        return dashboard_config

    def create_compliance_dashboard(self) -> dict[str, Any]:
        """Create a dashboard for compliance monitoring."""

        dashboard_config = {
            "title": "GenOps AI - Compliance & Governance",
            "description": "AI compliance monitoring and audit trail visualization",
            "widgets": [
                {
                    "id": "compliance-score",
                    "definition": {
                        "title": "Overall Compliance Score",
                        "type": "query_value",
                        "requests": [
                            {
                                "q": "avg:genops.eval.safety{*}",
                                "aggregator": "avg"
                            }
                        ],
                        "custom_unit": "%"
                    }
                },
                {
                    "id": "policy-enforcement",
                    "definition": {
                        "title": "Policy Enforcement Results",
                        "type": "distribution",
                        "requests": [
                            {
                                "q": "sum:genops.policy.result{*} by {genops.policy.enforcement}"
                            }
                        ]
                    }
                },
                {
                    "id": "audit-trail-volume",
                    "definition": {
                        "title": "Audit Trail Volume",
                        "type": "timeseries",
                        "requests": [
                            {
                                "q": "sum:genops.audit.event{*} by {genops.compliance.framework}",
                                "display_type": "area"
                            }
                        ]
                    }
                },
                {
                    "id": "data-classification",
                    "definition": {
                        "title": "Operations by Data Classification",
                        "type": "sunburst",
                        "requests": [
                            {
                                "q": "sum:genops.operation{*} by {genops.data.classification}"
                            }
                        ]
                    }
                }
            ]
        }

        return dashboard_config

    def create_performance_alerts(self) -> list[dict[str, Any]]:
        """Create performance and cost alerting rules."""

        alerts = [
            {
                "name": "High AI Cost per Hour",
                "type": "metric alert",
                "query": "sum(last_1h):sum:genops.cost.total{*} > 100",
                "message": """
AI costs are unusually high (>${value}) in the last hour.

**Investigation Steps:**
1. Check cost by team: `sum:genops.cost.total{*} by {genops.team}`
2. Check cost by customer: `sum:genops.cost.total{*} by {genops.customer_id}`
3. Check for unusual token usage patterns

@slack-ai-governance-channel
""",
                "tags": ["team:ai-governance", "severity:high"],
                "options": {
                    "notify_audit": True,
                    "include_tags": True,
                    "new_host_delay": 300
                }
            },
            {
                "name": "Policy Violation Rate High",
                "type": "metric alert",
                "query": "sum(last_15m):sum:genops.policy.violation{*} > 10",
                "message": """
High rate of policy violations detected (${value} in 15 minutes).

**Check for:**
- Budget limit violations
- Content safety failures
- Compliance policy breaches

Dashboard: [AI Compliance](https://app.datadoghq.com/dashboard/genops-compliance)

@pagerduty-ai-governance
""",
                "tags": ["team:compliance", "severity:critical"]
            },
            {
                "name": "AI Safety Score Below Threshold",
                "type": "metric alert",
                "query": "avg(last_5m):avg:genops.eval.safety{*} < 0.85",
                "message": """
AI safety evaluation scores have dropped below acceptable threshold.

Current average: ${value}
Required minimum: 0.85

**Immediate Actions:**
1. Review recent AI operations for safety concerns
2. Check if new models or prompts were deployed
3. Consider temporarily increasing human review requirements

@slack-ai-safety-team
""",
                "tags": ["team:ai-safety", "severity:high"]
            },
            {
                "name": "Token Usage Anomaly",
                "type": "anomaly",
                "query": "avg(last_4h):sum:genops.tokens.total{*}",
                "message": """
Unusual token usage pattern detected.

This could indicate:
- Inefficient prompts or models
- Unexpected traffic spikes
- Potential misuse or abuse

Review: [Token Usage Dashboard](https://app.datadoghq.com/dashboard/genops-tokens)

@slack-ai-platform-team
""",
                "tags": ["team:ai-platform", "severity:medium"]
            }
        ]

        return alerts

    def create_sli_monitors(self) -> list[dict[str, Any]]:
        """Create SLI (Service Level Indicator) monitors for AI governance."""

        sli_monitors = [
            {
                "name": "AI Operation Success Rate SLI",
                "type": "service_check",
                "query": "\"genops.operation.success\".over(\"*\").last(2).count_by_status()",
                "message": "AI operation success rate SLI",
                "tags": ["sli", "ai-operations"],
                "options": {
                    "thresholds": {
                        "critical": 95.0,  # 95% success rate minimum
                        "warning": 98.0    # 98% target
                    }
                }
            },
            {
                "name": "Compliance Evaluation Coverage SLI",
                "type": "metric alert",
                "query": "sum(last_1h):sum:genops.eval.performed{*} / sum:genops.operation.total{*} * 100 < 95",
                "message": "Compliance evaluation coverage below target",
                "tags": ["sli", "compliance-coverage"]
            },
            {
                "name": "Policy Response Time SLI",
                "type": "metric alert",
                "query": "avg(last_5m):avg:genops.policy.response_time{*} > 500",
                "message": "Policy evaluation response time above target (500ms)",
                "tags": ["sli", "policy-performance"]
            }
        ]

        return sli_monitors


# ============================================================================
# Validation Utilities (Following GenOps Standards)
# ============================================================================

@dataclass
class ValidationIssue:
    """Individual validation issue with severity and fix suggestion."""
    severity: str  # "error", "warning", "info"
    component: str
    message: str
    fix_suggestion: str


@dataclass
class ValidationResult:
    """Comprehensive validation result."""
    success: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    environment_info: dict = field(default_factory=dict)


def validate_datadog_setup(
    api_key: Optional[str] = None,
    site: str = "datadoghq.com"
) -> ValidationResult:
    """
    Comprehensive Datadog setup validation.

    Validates:
    - Environment variables (DATADOG_API_KEY, DATADOG_SITE)
    - OpenTelemetry dependencies
    - Network connectivity to OTLP endpoint
    - Datadog API key validity (basic format check)
    - Configuration completeness

    Args:
        api_key: Datadog API key (or uses DATADOG_API_KEY env var)
        site: Datadog site (default: datadoghq.com)

    Returns:
        ValidationResult with actionable fix suggestions
    """
    issues = []

    # Check API key
    effective_api_key = api_key or os.getenv("DATADOG_API_KEY")
    if not effective_api_key:
        issues.append(ValidationIssue(
            severity="error",
            component="environment",
            message="DATADOG_API_KEY not set",
            fix_suggestion="Set environment variable: export DATADOG_API_KEY='your_32_char_api_key'"
        ))
    elif len(effective_api_key) != 32:
        issues.append(ValidationIssue(
            severity="warning",
            component="environment",
            message=f"DATADOG_API_KEY has unexpected length ({len(effective_api_key)} chars, expected 32)",
            fix_suggestion="Verify your Datadog API key is correct (should be 32 characters)"
        ))

    # Check Datadog site
    valid_sites = [
        "datadoghq.com",
        "us5.datadoghq.com",
        "datadoghq.eu",
        "us3.datadoghq.com",
        "ddog-gov.com"
    ]
    if site not in valid_sites:
        issues.append(ValidationIssue(
            severity="warning",
            component="configuration",
            message=f"Datadog site '{site}' is not a standard site",
            fix_suggestion=f"Valid sites: {', '.join(valid_sites)}"
        ))

    # Check OpenTelemetry dependencies
    if not HAS_OPENTELEMETRY:
        issues.append(ValidationIssue(
            severity="error",
            component="dependencies",
            message="OpenTelemetry not installed",
            fix_suggestion="Install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http"
        ))

    # Check network connectivity to Datadog OTLP endpoint
    if effective_api_key:  # Only test if API key is set
        otlp_host = f"otlp.{site}"
        try:
            socket.create_connection((otlp_host, 443), timeout=5)
        except socket.timeout:
            issues.append(ValidationIssue(
                severity="warning",
                component="network",
                message=f"Connection to {otlp_host}:443 timed out (5 seconds)",
                fix_suggestion="Check network connectivity, firewall rules, and proxy configuration"
            ))
        except socket.gaierror:
            issues.append(ValidationIssue(
                severity="error",
                component="network",
                message=f"Cannot resolve hostname: {otlp_host}",
                fix_suggestion="Verify DATADOG_SITE is correct and DNS is working"
            ))
        except Exception as e:
            issues.append(ValidationIssue(
                severity="warning",
                component="network",
                message=f"Cannot connect to {otlp_host}:443: {type(e).__name__}",
                fix_suggestion="Check network connectivity and firewall rules"
            ))

    # Check service name configuration
    service_name = os.getenv("OTEL_SERVICE_NAME")
    if not service_name:
        issues.append(ValidationIssue(
            severity="info",
            component="configuration",
            message="OTEL_SERVICE_NAME not set (will use default: genops-ai)",
            fix_suggestion="Set environment variable: export OTEL_SERVICE_NAME='your-service-name'"
        ))

    # Environment info
    environment_info = {
        "api_key_set": bool(effective_api_key),
        "api_key_length": len(effective_api_key) if effective_api_key else 0,
        "site": site,
        "has_opentelemetry": HAS_OPENTELEMETRY,
        "has_datadog_sdk": HAS_DATADOG,
        "service_name": service_name or "genops-ai (default)",
        "otlp_endpoint": f"https://otlp.{site}"
    }

    # Determine success (no errors)
    success = len([i for i in issues if i.severity == "error"]) == 0

    return ValidationResult(
        success=success,
        issues=issues,
        environment_info=environment_info
    )


def print_validation_result(result: ValidationResult):
    """
    User-friendly validation result display.

    Args:
        result: ValidationResult from validate_datadog_setup()
    """
    print("\n" + "=" * 60)
    print("  DATADOG SETUP VALIDATION")
    print("=" * 60)

    if result.success:
        print("\n✅ Datadog setup validation PASSED!")
        print(f"\n📋 Configuration:")
        print(f"   Site: {result.environment_info['site']}")
        print(f"   OTLP Endpoint: {result.environment_info['otlp_endpoint']}")
        print(f"   Service Name: {result.environment_info['service_name']}")
        print(f"   API Key: {'✅ Set (%d chars)' % result.environment_info['api_key_length'] if result.environment_info['api_key_set'] else '❌ Not Set'}")
        print(f"   OpenTelemetry: {'✅ Installed' if result.environment_info['has_opentelemetry'] else '❌ Missing'}")
        print(f"   Datadog SDK: {'✅ Installed' if result.environment_info['has_datadog_sdk'] else '⚠️  Not installed (optional)'}")
    else:
        print("\n❌ Datadog setup validation FAILED!")
        print("\nYou must fix the errors below before exporting telemetry.\n")

    # Group issues by severity
    errors = [i for i in result.issues if i.severity == "error"]
    warnings = [i for i in result.issues if i.severity == "warning"]
    info = [i for i in result.issues if i.severity == "info"]

    if errors:
        print("\n🚨 ERRORS (must fix):")
        for issue in errors:
            print(f"\n   [{issue.component.upper()}] {issue.message}")
            print(f"   💡 Fix: {issue.fix_suggestion}")

    if warnings:
        print("\n⚠️  WARNINGS:")
        for issue in warnings:
            print(f"\n   [{issue.component.upper()}] {issue.message}")
            print(f"   💡 Suggestion: {issue.fix_suggestion}")

    if info:
        print("\n📌 INFORMATION:")
        for issue in info:
            print(f"\n   [{issue.component.upper()}] {issue.message}")
            print(f"   💡 Tip: {issue.fix_suggestion}")

    print("\n" + "=" * 60)

    if result.success:
        print("✅ Ready to export telemetry to Datadog!")
    else:
        print("❌ Please fix errors above before proceeding.")

    print("=" * 60 + "\n")


# ============================================================================
# Standard Entry Points (Following GenOps Naming Conventions)
# ============================================================================

def instrument_datadog(
    api_key: Optional[str] = None,
    app_key: Optional[str] = None,
    site: str = "datadoghq.com",
    service_name: Optional[str] = None,
    environment: str = "production",
    auto_configure: bool = True,
    validate: bool = True,
    **config
) -> DatadogGenOpsIntegration:
    """
    Main entry point for Datadog instrumentation.

    Follows GenOps standard naming convention for provider adapters.

    Args:
        api_key: Datadog API key (or use DATADOG_API_KEY env var)
        app_key: Datadog App key (or use DATADOG_APP_KEY env var)
        site: Datadog site (default: datadoghq.com)
        service_name: Service name (or use OTEL_SERVICE_NAME env var)
        environment: Environment (default: production)
        auto_configure: Automatically configure OpenTelemetry (default: True)
        validate: Run validation before setup (default: True)
        **config: Additional configuration options

    Returns:
        Configured DatadogGenOpsIntegration instance

    Raises:
        RuntimeError: If validation fails with errors

    Example:
        >>> from genops.exporters.datadog import instrument_datadog
        >>> integration = instrument_datadog()  # Uses environment variables
        >>> # Or with explicit configuration:
        >>> integration = instrument_datadog(
        ...     api_key="your_api_key",
        ...     service_name="my-ai-app",
        ...     environment="production"
        ... )
    """
    # Run validation if requested
    if validate:
        result = validate_datadog_setup(api_key=api_key, site=site)
        print_validation_result(result)

        if not result.success:
            raise RuntimeError(
                "Datadog setup validation failed. Fix errors above before proceeding."
            )

    # Create and return integration
    return DatadogGenOpsIntegration(
        datadog_api_key=api_key,
        datadog_app_key=app_key,
        datadog_site=site,
        service_name=service_name or os.getenv("OTEL_SERVICE_NAME", "genops-ai"),
        environment=environment,
        **config
    )


def auto_instrument(
    validate: bool = True
) -> DatadogGenOpsIntegration:
    """
    Zero-code auto-instrumentation for Datadog export.

    Automatically detects Datadog environment variables and
    configures OpenTelemetry export with no code changes required.

    Environment variables used:
    - DATADOG_API_KEY (required)
    - DATADOG_APP_KEY (optional)
    - DATADOG_SITE (default: datadoghq.com)
    - OTEL_SERVICE_NAME (default: genops-ai)
    - OTEL_ENVIRONMENT (default: production)

    Args:
        validate: Run validation before setup (default: True)

    Returns:
        Configured DatadogGenOpsIntegration instance

    Raises:
        RuntimeError: If validation fails with errors

    Example:
        >>> from genops.exporters.datadog import auto_instrument
        >>> auto_instrument()
        >>> # All GenOps operations now export to Datadog!
    """
    return instrument_datadog(
        api_key=os.getenv("DATADOG_API_KEY"),
        app_key=os.getenv("DATADOG_APP_KEY"),
        site=os.getenv("DATADOG_SITE", "datadoghq.com"),
        service_name=os.getenv("OTEL_SERVICE_NAME"),
        environment=os.getenv("OTEL_ENVIRONMENT", "production"),
        auto_configure=True,
        validate=validate
    )


# ============================================================================
# Demonstration Functions
# ============================================================================

def demonstrate_datadog_telemetry():
    """Demonstrate GenOps AI telemetry flowing to Datadog."""

    print("\n📊 DATADOG TELEMETRY DEMONSTRATION")
    print("=" * 60)

    # Initialize Datadog integration
    datadog_integration = DatadogGenOpsIntegration(
        service_name="genops-ai-demo",
        environment="development"
    )

    # Set up GenOps with attribution
    genops.set_default_attributes(
        team="ai-platform",
        project="datadog-integration",
        environment="development",
        cost_center="engineering"
    )

    # Demonstrate various AI operations with telemetry
    operations = [
        {
            "name": "customer_support_chat",
            "customer_id": "enterprise-123",
            "feature": "ai-assistant",
            "data_classification": "internal"
        },
        {
            "name": "document_analysis",
            "customer_id": "startup-456",
            "feature": "document-processing",
            "data_classification": "confidential"
        },
        {
            "name": "financial_analysis",
            "customer_id": "enterprise-789",
            "feature": "risk-assessment",
            "data_classification": "restricted"
        }
    ]

    print("🤖 Generating AI operations with full telemetry...")

    for i, op in enumerate(operations):
        print(f"\n   Operation {i+1}: {op['name']}")

        # Set operation context
        genops.set_context(**op)

        # Simulate AI operation
        start_time = time.time()

        # Simulate processing
        time.sleep(0.2)

        duration = time.time() - start_time

        # Record telemetry with effective attributes
        effective_attrs = genops.get_effective_attributes()

        # Send custom metrics to Datadog
        datadog_integration.send_custom_metric(
            "operation.duration",
            duration * 1000,  # milliseconds
            tags=effective_attrs
        )

        datadog_integration.send_custom_metric(
            "operation.count",
            1,
            tags=effective_attrs
        )

        # Simulate cost and token usage
        cost = 0.0234 * (i + 1)  # Varying costs
        tokens = 150 * (i + 2)   # Varying token usage

        datadog_integration.send_custom_metric(
            "cost.total",
            cost,
            tags=effective_attrs
        )

        datadog_integration.send_custom_metric(
            "tokens.total",
            tokens,
            tags=effective_attrs
        )

        # Simulate evaluation scores
        safety_score = 0.92 - (i * 0.02)  # Varying scores
        accuracy_score = 0.88 + (i * 0.01)

        datadog_integration.send_custom_metric(
            "eval.safety",
            safety_score,
            tags=effective_attrs
        )

        datadog_integration.send_custom_metric(
            "eval.accuracy",
            accuracy_score,
            tags=effective_attrs
        )

        print(f"      Cost: ${cost:.4f} | Tokens: {tokens:,}")
        print(f"      Safety: {safety_score:.3f} | Accuracy: {accuracy_score:.3f}")
        print(f"      Duration: {duration*1000:.1f}ms")

        genops.clear_context()

    print("\n✅ All telemetry sent to Datadog!")


def demonstrate_dashboard_creation():
    """Demonstrate creating Datadog dashboards for GenOps AI."""

    print("\n📈 DATADOG DASHBOARD CREATION")
    print("=" * 60)

    # Initialize integration
    datadog_integration = DatadogGenOpsIntegration(
        service_name="genops-ai",
        environment="production"
    )

    # Create cost dashboard
    cost_dashboard = datadog_integration.create_ai_cost_dashboard()
    print("📊 AI Cost Dashboard Configuration:")
    print(f"   Title: {cost_dashboard['title']}")
    print(f"   Widgets: {len(cost_dashboard['widgets'])} widgets")
    print(f"   Template Variables: {len(cost_dashboard['template_variables'])} variables")

    # Widget details
    for widget in cost_dashboard['widgets']:
        print(f"      • {widget['definition']['title']} ({widget['definition']['type']})")

    # Create compliance dashboard
    compliance_dashboard = datadog_integration.create_compliance_dashboard()
    print("\n🛡️ Compliance Dashboard Configuration:")
    print(f"   Title: {compliance_dashboard['title']}")
    print(f"   Widgets: {len(compliance_dashboard['widgets'])} widgets")

    for widget in compliance_dashboard['widgets']:
        print(f"      • {widget['definition']['title']} ({widget['definition']['type']})")

    # Save dashboard configurations
    with open("datadog_cost_dashboard.json", "w") as f:
        json.dump(cost_dashboard, f, indent=2)

    with open("datadog_compliance_dashboard.json", "w") as f:
        json.dump(compliance_dashboard, f, indent=2)

    print("\n📄 Dashboard configurations saved:")
    print("   • datadog_cost_dashboard.json")
    print("   • datadog_compliance_dashboard.json")


def demonstrate_alerting_setup():
    """Demonstrate creating alerts and SLIs for GenOps AI monitoring."""

    print("\n🚨 DATADOG ALERTING SETUP")
    print("=" * 60)

    # Initialize integration
    datadog_integration = DatadogGenOpsIntegration(
        service_name="genops-ai",
        environment="production"
    )

    # Create performance alerts
    performance_alerts = datadog_integration.create_performance_alerts()
    print(f"⚡ Performance Alerts ({len(performance_alerts)} alerts):")

    for alert in performance_alerts:
        print(f"   • {alert['name']}")
        print(f"     Query: {alert['query']}")
        print(f"     Tags: {', '.join(alert['tags'])}")

    # Create SLI monitors
    sli_monitors = datadog_integration.create_sli_monitors()
    print(f"\n📊 SLI Monitors ({len(sli_monitors)} monitors):")

    for monitor in sli_monitors:
        print(f"   • {monitor['name']}")
        print(f"     Tags: {', '.join(monitor['tags'])}")

    # Save alerting configurations
    alerting_config = {
        "performance_alerts": performance_alerts,
        "sli_monitors": sli_monitors
    }

    with open("datadog_alerting_config.json", "w") as f:
        json.dump(alerting_config, f, indent=2)

    print("\n📄 Alerting configuration saved to: datadog_alerting_config.json")


def show_datadog_queries():
    """Show example Datadog queries for GenOps AI governance."""

    print("\n🔍 DATADOG QUERY EXAMPLES")
    print("=" * 60)

    queries = {
        "Cost Analysis": [
            "sum:genops.cost.total{*} by {genops.team}",
            "sum:genops.cost.total{*} by {genops.customer_id}",
            "avg:genops.cost.total{*} by {genops.cost.provider}",
            "sum:genops.cost.total{genops.environment:production} by {genops.feature}"
        ],
        "Token Usage": [
            "sum:genops.tokens.total{*} by {genops.cost.provider}",
            "avg:genops.tokens.input{*} by {genops.team}",
            "rate(sum:genops.tokens.total{*})",
            "sum:genops.tokens.total{genops.feature:chat-assistant}"
        ],
        "Performance Monitoring": [
            "avg:genops.operation.duration{*} by {genops.operation.name}",
            "p95:genops.operation.duration{*}",
            "rate(sum:genops.operation.count{*})",
            "sum:genops.operation.error{*} by {genops.error.type}"
        ],
        "Compliance & Governance": [
            "avg:genops.eval.safety{*} by {genops.team}",
            "sum:genops.policy.violation{*} by {genops.policy.name}",
            "count:genops.audit.event{*} by {genops.compliance.framework}",
            "avg:genops.eval.accuracy{genops.data.classification:restricted}"
        ],
        "Business Intelligence": [
            "sum:genops.cost.total{*} by {genops.customer_id,genops.feature}",
            "avg:genops.tokens.total{*} by {genops.customer_tier}",
            "sum:genops.operation.count{*} by {genops.project,genops.environment}",
            "rate(sum:genops.cost.total{*}) by {genops.cost_center}"
        ]
    }

    for category, query_list in queries.items():
        print(f"\n📊 {category}:")
        for query in query_list:
            print(f"   {query}")

    print("\n💡 Query Tips:")
    print("• Use .rollup(sum, 3600) for hourly aggregation")
    print("• Use .as_count() for rate calculations")
    print("• Use by {*} to group by all available tags")
    print("• Filter with {tag:value} syntax")
    print("• Use p50, p95, p99 for percentile calculations")


def main():
    """Run the complete Datadog integration demonstration."""

    print("📊 GenOps AI: Datadog Integration Guide")
    print("=" * 80)
    print("\nThis guide demonstrates comprehensive integration between")
    print("GenOps AI telemetry and Datadog observability platform.")

    # Run validation first
    print("\n🔍 STEP 1: Validating Datadog Setup")
    print("=" * 60)
    validation_result = validate_datadog_setup()
    print_validation_result(validation_result)

    if not validation_result.success:
        print("\n⚠️ Setup validation failed. Please fix errors above.")
        print("Continuing with demonstration mode (no actual export)...\n")

    # Check dependencies
    if not HAS_OPENTELEMETRY:
        print("\n⚠️ OpenTelemetry not installed. Install with:")
        print("pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp")
        print("\nContinuing with limited functionality...")

    try:
        # Run demonstrations
        demonstrate_datadog_telemetry()
        demonstrate_dashboard_creation()
        demonstrate_alerting_setup()
        show_datadog_queries()

        print("\n🎯 INTEGRATION SUMMARY")
        print("=" * 60)
        print("✅ OpenTelemetry OTLP export to Datadog configured")
        print("✅ Custom metrics for AI governance operations")
        print("✅ Cost attribution dashboards with multi-tenant views")
        print("✅ Compliance monitoring and audit trail visualization")
        print("✅ Performance alerting with SLI/SLO monitoring")
        print("✅ Business intelligence queries for cost optimization")

        print("\n📚 DATADOG FEATURES UTILIZED")
        print("=" * 60)
        print("🔍 APM: Distributed tracing for AI operations")
        print("📊 Metrics: Custom metrics for cost, tokens, evaluations")
        print("📈 Dashboards: Pre-built governance dashboards")
        print("🚨 Alerts: Cost, performance, and compliance monitoring")
        print("📋 Logs: Audit trail and policy decision logging")
        print("🎯 SLIs: Service level indicators for AI governance")

        print("\n🔧 SETUP INSTRUCTIONS")
        print("=" * 60)
        print("1. Set environment variables: DATADOG_API_KEY, DATADOG_APP_KEY")
        print("2. Install dependencies: pip install datadog opentelemetry-exporter-otlp")
        print("3. Import dashboard configurations into Datadog UI")
        print("4. Configure alerts and SLI monitors")
        print("5. Set up log ingestion for audit trails")
        print("6. Create custom notebooks for cost analysis")

        print("\n🔗 Next Steps")
        print("=" * 60)
        print("• Customize dashboards for your specific use cases")
        print("• Set up team-specific alert channels and escalations")
        print("• Create SLO targets based on your governance requirements")
        print("• Integrate with Datadog Watchdog for anomaly detection")
        print("• Set up cost attribution reports for FinOps workflows")

    except Exception as e:
        print(f"\n❌ Datadog integration demo failed: {e}")
        raise


if __name__ == "__main__":
    main()
