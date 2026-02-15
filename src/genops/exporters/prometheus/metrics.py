"""Governance metric definitions for Prometheus export.

This module defines standardized Prometheus metrics for GenOps governance telemetry:
- Cost metrics: Track AI operation costs across providers and models
- Token metrics: Monitor token usage and efficiency
- Policy metrics: Track policy violations and enforcement
- Evaluation metrics: Monitor quality scores and compliance
- Budget metrics: Track budget utilization and constraints

Metric Naming Convention:
    genops_<subsystem>_<metric>_<unit>

Standard Labels:
    - provider: AI provider (openai, anthropic, bedrock, etc.)
    - model: Model name (gpt-4, claude-3-sonnet, etc.)
    - team: Team identifier for cost attribution
    - customer_id: Customer identifier for multi-tenant tracking
    - environment: Environment (production, staging, development)
    - feature: Feature identifier for feature-level tracking
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MetricType(Enum):
    """Prometheus metric types."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricDefinition:
    """Definition for a Prometheus metric.

    Attributes:
        name: Metric name (without namespace prefix)
        metric_type: Type of Prometheus metric
        description: Human-readable description
        unit: Unit of measurement (e.g., "usd", "tokens", "seconds")
        labels: Standard labels for this metric
    """

    name: str
    metric_type: MetricType
    description: str
    unit: str
    labels: set[str]


# Standard label sets
STANDARD_LABELS = {"provider", "model", "team", "customer_id", "environment", "feature"}

OPERATION_LABELS = STANDARD_LABELS | {"operation_type", "operation_id"}
POLICY_LABELS = STANDARD_LABELS | {"policy_name", "policy_type"}
EVALUATION_LABELS = STANDARD_LABELS | {"evaluation_type", "evaluator"}


# ======================
# Cost Metrics
# ======================

COST_TOTAL = MetricDefinition(
    name="cost_total",
    metric_type=MetricType.COUNTER,
    description="Total cost of AI operations in USD",
    unit="usd",
    labels=STANDARD_LABELS,
)

COST_BY_OPERATION = MetricDefinition(
    name="cost_by_operation",
    metric_type=MetricType.COUNTER,
    description="Cost per operation type",
    unit="usd",
    labels=OPERATION_LABELS,
)


# ======================
# Token Metrics
# ======================

TOKENS_INPUT_TOTAL = MetricDefinition(
    name="tokens_input_total",
    metric_type=MetricType.COUNTER,
    description="Total input tokens consumed",
    unit="tokens",
    labels=STANDARD_LABELS,
)

TOKENS_OUTPUT_TOTAL = MetricDefinition(
    name="tokens_output_total",
    metric_type=MetricType.COUNTER,
    description="Total output tokens generated",
    unit="tokens",
    labels=STANDARD_LABELS,
)

TOKENS_TOTAL = MetricDefinition(
    name="tokens_total",
    metric_type=MetricType.COUNTER,
    description="Total tokens (input + output)",
    unit="tokens",
    labels=STANDARD_LABELS,
)

TOKEN_EFFICIENCY = MetricDefinition(
    name="token_efficiency",
    metric_type=MetricType.GAUGE,
    description="Tokens per dollar (cost efficiency)",
    unit="tokens_per_usd",
    labels=STANDARD_LABELS,
)


# ======================
# Policy Metrics
# ======================

POLICY_VIOLATIONS_TOTAL = MetricDefinition(
    name="policy_violations_total",
    metric_type=MetricType.COUNTER,
    description="Total number of policy violations",
    unit="violations",
    labels=POLICY_LABELS,
)

POLICY_EVALUATIONS_TOTAL = MetricDefinition(
    name="policy_evaluations_total",
    metric_type=MetricType.COUNTER,
    description="Total number of policy evaluations",
    unit="evaluations",
    labels=POLICY_LABELS,
)

POLICY_ENFORCEMENT_ACTIONS = MetricDefinition(
    name="policy_enforcement_actions",
    metric_type=MetricType.COUNTER,
    description="Number of policy enforcement actions taken",
    unit="actions",
    labels=POLICY_LABELS | {"action_type"},
)

POLICY_COMPLIANCE_RATE = MetricDefinition(
    name="policy_compliance_rate",
    metric_type=MetricType.GAUGE,
    description="Policy compliance rate (0-1)",
    unit="ratio",
    labels=POLICY_LABELS,
)


# ======================
# Evaluation Metrics
# ======================

EVALUATION_SCORE = MetricDefinition(
    name="evaluation_score",
    metric_type=MetricType.HISTOGRAM,
    description="Distribution of evaluation scores",
    unit="score",
    labels=EVALUATION_LABELS,
)

EVALUATION_LATENCY = MetricDefinition(
    name="evaluation_latency_seconds",
    metric_type=MetricType.HISTOGRAM,
    description="Evaluation execution latency",
    unit="seconds",
    labels=EVALUATION_LABELS,
)

EVALUATION_FAILURES = MetricDefinition(
    name="evaluation_failures_total",
    metric_type=MetricType.COUNTER,
    description="Number of failed evaluations",
    unit="failures",
    labels=EVALUATION_LABELS,
)


# ======================
# Budget Metrics
# ======================

BUDGET_UTILIZATION = MetricDefinition(
    name="budget_utilization_ratio",
    metric_type=MetricType.GAUGE,
    description="Budget utilization ratio (0-1)",
    unit="ratio",
    labels=STANDARD_LABELS | {"budget_period"},
)

BUDGET_REMAINING = MetricDefinition(
    name="budget_remaining_usd",
    metric_type=MetricType.GAUGE,
    description="Remaining budget in USD",
    unit="usd",
    labels=STANDARD_LABELS | {"budget_period"},
)

BUDGET_EXCEEDED = MetricDefinition(
    name="budget_exceeded_total",
    metric_type=MetricType.COUNTER,
    description="Number of times budget was exceeded",
    unit="events",
    labels=STANDARD_LABELS | {"budget_period"},
)


# ======================
# Performance Metrics
# ======================

OPERATION_LATENCY = MetricDefinition(
    name="operation_latency_seconds",
    metric_type=MetricType.HISTOGRAM,
    description="AI operation latency",
    unit="seconds",
    labels=OPERATION_LABELS,
)

OPERATION_ERRORS = MetricDefinition(
    name="operation_errors_total",
    metric_type=MetricType.COUNTER,
    description="Total number of operation errors",
    unit="errors",
    labels=OPERATION_LABELS | {"error_type"},
)

OPERATIONS_TOTAL = MetricDefinition(
    name="operations_total",
    metric_type=MetricType.COUNTER,
    description="Total number of AI operations",
    unit="operations",
    labels=OPERATION_LABELS,
)


# ======================
# Registry
# ======================

# All metric definitions for easy iteration
ALL_METRICS: dict[str, MetricDefinition] = {
    # Cost
    "cost_total": COST_TOTAL,
    "cost_by_operation": COST_BY_OPERATION,
    # Tokens
    "tokens_input_total": TOKENS_INPUT_TOTAL,
    "tokens_output_total": TOKENS_OUTPUT_TOTAL,
    "tokens_total": TOKENS_TOTAL,
    "token_efficiency": TOKEN_EFFICIENCY,
    # Policy
    "policy_violations_total": POLICY_VIOLATIONS_TOTAL,
    "policy_evaluations_total": POLICY_EVALUATIONS_TOTAL,
    "policy_enforcement_actions": POLICY_ENFORCEMENT_ACTIONS,
    "policy_compliance_rate": POLICY_COMPLIANCE_RATE,
    # Evaluation
    "evaluation_score": EVALUATION_SCORE,
    "evaluation_latency_seconds": EVALUATION_LATENCY,
    "evaluation_failures_total": EVALUATION_FAILURES,
    # Budget
    "budget_utilization_ratio": BUDGET_UTILIZATION,
    "budget_remaining_usd": BUDGET_REMAINING,
    "budget_exceeded_total": BUDGET_EXCEEDED,
    # Performance
    "operation_latency_seconds": OPERATION_LATENCY,
    "operation_errors_total": OPERATION_ERRORS,
    "operations_total": OPERATIONS_TOTAL,
}


def get_metric_definition(name: str) -> MetricDefinition:
    """Get metric definition by name.

    Args:
        name: Metric name (without namespace)

    Returns:
        MetricDefinition for the metric

    Raises:
        KeyError: If metric not found
    """
    return ALL_METRICS[name]


def get_full_metric_name(name: str, namespace: str = "genops") -> str:
    """Get full metric name with namespace prefix.

    Args:
        name: Metric name
        namespace: Namespace prefix (default: genops)

    Returns:
        Full metric name (e.g., "genops_cost_total_usd")
    """
    metric = get_metric_definition(name)
    base_name = f"{namespace}_{metric.name}"

    # Append unit if not already in name
    if metric.unit and metric.unit not in base_name:
        return f"{base_name}_{metric.unit}"

    return base_name


def filter_labels(
    labels: dict[str, str],
    include: set[str] | None = None,
    exclude: set[str] | None = None,
) -> dict[str, str]:
    """Filter labels based on include/exclude sets.

    Args:
        labels: Original label dictionary
        include: If provided, only include these labels (empty set = include all)
        exclude: Labels to exclude

    Returns:
        Filtered label dictionary
    """
    filtered = dict(labels)

    # Apply include filter
    if include:
        filtered = {k: v for k, v in filtered.items() if k in include}

    # Apply exclude filter
    if exclude:
        filtered = {k: v for k, v in filtered.items() if k not in exclude}

    return filtered
