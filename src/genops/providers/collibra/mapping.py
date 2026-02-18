"""Bidirectional data mapping between GenOps and Collibra."""

from __future__ import annotations

from typing import Any

# GenOps → Collibra Asset Type Mapping
GENOPS_TO_COLLIBRA_ASSET_TYPES = {
    "cost": "AI Operation Cost",
    "policy": "Policy Evaluation Event",
    "evaluation": "Model Evaluation",
    "budget": "Budget Allocation",
    "operation": "AI Workflow Execution",
}

# Collibra → GenOps Asset Type Mapping (reverse)
COLLIBRA_TO_GENOPS_ASSET_TYPES = {
    v: k for k, v in GENOPS_TO_COLLIBRA_ASSET_TYPES.items()
}


# GenOps Attribute → Collibra Attribute Mapping
GENOPS_TO_COLLIBRA_ATTRIBUTES = {
    # Cost attributes
    "genops.cost.total": "cost_amount",
    "genops.cost.currency": "currency",
    "genops.cost.provider": "ai_provider",
    "genops.cost.model": "ai_model",
    "genops.tokens.input": "tokens_input",
    "genops.tokens.output": "tokens_output",
    "genops.tokens.total": "tokens_total",
    # Policy attributes
    "genops.policy.name": "policy_name",
    "genops.policy.result": "policy_result",
    "genops.policy.reason": "policy_reason",
    # Evaluation attributes
    "genops.eval.metric": "quality_metric",
    "genops.eval.score": "metric_score",
    "genops.eval.threshold": "metric_threshold",
    "genops.eval.passed": "evaluation_passed",
    # Budget attributes
    "genops.budget.name": "budget_name",
    "genops.budget.allocated": "budget_allocated",
    "genops.budget.consumed": "budget_consumed",
    "genops.budget.remaining": "budget_remaining",
    "genops.budget.utilization_percent": "budget_utilization",
    # Governance attribution
    "genops.team": "team",
    "genops.project": "project",
    "genops.customer_id": "customer_identifier",
    "genops.environment": "environment",
    "genops.cost_center": "cost_center",
    "genops.feature": "feature",
    # Operation attributes
    "genops.operation.name": "operation_name",
    "genops.operation.type": "operation_type",
    "genops.operation.status": "operation_status",
    "genops.operation.duration_ms": "duration_milliseconds",
    # Span attributes
    "span.name": "span_name",
    "span.kind": "span_kind",
    "trace.id": "trace_id",
    "span.id": "span_id",
}

# Collibra → GenOps Attribute Mapping (reverse)
COLLIBRA_TO_GENOPS_ATTRIBUTES = {v: k for k, v in GENOPS_TO_COLLIBRA_ATTRIBUTES.items()}


def map_genops_to_collibra_asset_type(genops_category: str) -> str:
    """
    Map GenOps telemetry category to Collibra asset type.

    Args:
        genops_category: GenOps category (cost, policy, evaluation, budget, operation)

    Returns:
        Collibra asset type name

    Example:
        >>> map_genops_to_collibra_asset_type("cost")
        'AI Operation Cost'
    """
    return GENOPS_TO_COLLIBRA_ASSET_TYPES.get(genops_category, "AI Workflow Execution")


def map_collibra_to_genops_asset_type(collibra_asset_type: str) -> str:
    """
    Map Collibra asset type to GenOps category.

    Args:
        collibra_asset_type: Collibra asset type name

    Returns:
        GenOps telemetry category

    Example:
        >>> map_collibra_to_genops_asset_type("AI Operation Cost")
        'cost'
    """
    return COLLIBRA_TO_GENOPS_ASSET_TYPES.get(collibra_asset_type, "operation")


def map_genops_attributes_to_collibra(
    genops_attributes: dict[str, Any],
) -> dict[str, Any]:
    """
    Map GenOps telemetry attributes to Collibra asset attributes.

    Args:
        genops_attributes: GenOps span attributes

    Returns:
        Collibra asset attributes

    Example:
        >>> attrs = {
        ...     "genops.cost.total": 0.05,
        ...     "genops.cost.provider": "openai",
        ...     "genops.team": "ml-platform"
        ... }
        >>> map_genops_attributes_to_collibra(attrs)
        {'cost_amount': 0.05, 'ai_provider': 'openai', 'team': 'ml-platform'}
    """
    collibra_attrs = {}

    for genops_key, value in genops_attributes.items():
        # Map known attributes
        if genops_key in GENOPS_TO_COLLIBRA_ATTRIBUTES:
            collibra_key = GENOPS_TO_COLLIBRA_ATTRIBUTES[genops_key]
            collibra_attrs[collibra_key] = value
        # Pass through unknown attributes with prefix
        elif genops_key.startswith("genops."):
            # Strip genops. prefix and use as-is
            collibra_key = genops_key.replace("genops.", "")
            collibra_attrs[collibra_key] = value

    return collibra_attrs


def map_collibra_attributes_to_genops(
    collibra_attributes: dict[str, Any],
) -> dict[str, Any]:
    """
    Map Collibra asset attributes to GenOps telemetry attributes.

    Args:
        collibra_attributes: Collibra asset attributes

    Returns:
        GenOps telemetry attributes

    Example:
        >>> attrs = {
        ...     "cost_amount": 0.05,
        ...     "ai_provider": "openai",
        ...     "team": "ml-platform"
        ... }
        >>> map_collibra_attributes_to_genops(attrs)
        {'genops.cost.total': 0.05, 'genops.cost.provider': 'openai', 'genops.team': 'ml-platform'}
    """
    genops_attrs = {}

    for collibra_key, value in collibra_attributes.items():
        # Map known attributes
        if collibra_key in COLLIBRA_TO_GENOPS_ATTRIBUTES:
            genops_key = COLLIBRA_TO_GENOPS_ATTRIBUTES[collibra_key]
            genops_attrs[genops_key] = value
        # Unknown attributes get genops. prefix
        else:
            genops_key = f"genops.{collibra_key}"
            genops_attrs[genops_key] = value

    return genops_attrs


def infer_asset_type_from_attributes(attributes: dict[str, Any]) -> str:
    """
    Infer Collibra asset type from GenOps attributes.

    Args:
        attributes: GenOps span attributes

    Returns:
        Inferred Collibra asset type

    Example:
        >>> attrs = {"genops.cost.total": 0.05, "genops.cost.provider": "openai"}
        >>> infer_asset_type_from_attributes(attrs)
        'AI Operation Cost'
    """
    # Check for cost attributes
    if any(k.startswith("genops.cost.") for k in attributes.keys()):
        return "AI Operation Cost"

    # Check for policy attributes
    if any(k.startswith("genops.policy.") for k in attributes.keys()):
        return "Policy Evaluation Event"

    # Check for evaluation attributes
    if any(k.startswith("genops.eval.") for k in attributes.keys()):
        return "Model Evaluation"

    # Check for budget attributes
    if any(k.startswith("genops.budget.") for k in attributes.keys()):
        return "Budget Allocation"

    # Default to workflow execution
    return "AI Workflow Execution"


def create_collibra_asset_name(attributes: dict[str, Any], asset_type: str) -> str:
    """
    Create a descriptive asset name from GenOps attributes.

    Args:
        attributes: GenOps span attributes
        asset_type: Collibra asset type

    Returns:
        Human-readable asset name

    Example:
        >>> attrs = {
        ...     "genops.operation.name": "gpt-4-completion",
        ...     "genops.team": "ml-platform",
        ...     "genops.cost.total": 0.05
        ... }
        >>> create_collibra_asset_name(attrs, "AI Operation Cost")
        'gpt-4-completion (ml-platform) - $0.05'
    """
    # Get operation name
    operation_name = attributes.get(
        "genops.operation.name", attributes.get("span.name", "ai-operation")
    )

    # Get team for context
    team = attributes.get("genops.team")

    # Create base name
    if team:
        name = f"{operation_name} ({team})"
    else:
        name = operation_name

    # Add type-specific suffix
    if asset_type == "AI Operation Cost":
        cost = attributes.get("genops.cost.total")
        if cost is not None:
            currency = attributes.get("genops.cost.currency", "USD")
            if currency == "USD":
                name += f" - ${cost:.4f}"
            else:
                name += f" - {cost:.4f} {currency}"

    elif asset_type == "Policy Evaluation Event":
        policy_name = attributes.get("genops.policy.name")
        policy_result = attributes.get("genops.policy.result")
        if policy_name and policy_result:
            name += f" - {policy_name} ({policy_result})"

    elif asset_type == "Model Evaluation":
        metric = attributes.get("genops.eval.metric")
        score = attributes.get("genops.eval.score")
        if metric and score is not None:
            name += f" - {metric}: {score:.3f}"

    elif asset_type == "Budget Allocation":
        budget_name = attributes.get("genops.budget.name")
        if budget_name:
            name += f" - {budget_name}"

    return name


def create_collibra_asset_from_span(
    span_attributes: dict[str, Any],
    domain_id: str,
    asset_type: str | None = None,
) -> dict[str, Any]:
    """
    Create a complete Collibra asset structure from GenOps span attributes.

    Args:
        span_attributes: GenOps span attributes
        domain_id: Target Collibra domain ID
        asset_type: Override asset type (optional, will be inferred if not provided)

    Returns:
        Collibra asset creation payload

    Example:
        >>> attrs = {
        ...     "genops.cost.total": 0.05,
        ...     "genops.cost.provider": "openai",
        ...     "genops.operation.name": "completion",
        ...     "genops.team": "ml-platform"
        ... }
        >>> create_collibra_asset_from_span(attrs, "domain-123")
        {
            'domainId': 'domain-123',
            'typeId': 'AI Operation Cost',
            'name': 'completion (ml-platform) - $0.0500',
            'attributes': {
                'cost_amount': 0.05,
                'ai_provider': 'openai',
                'team': 'ml-platform',
                ...
            }
        }
    """
    # Infer asset type if not provided
    if asset_type is None:
        asset_type = infer_asset_type_from_attributes(span_attributes)

    # Create asset name
    asset_name = create_collibra_asset_name(span_attributes, asset_type)

    # Map attributes
    collibra_attributes = map_genops_attributes_to_collibra(span_attributes)

    # Create asset payload
    asset_payload = {
        "domainId": domain_id,
        "typeId": asset_type,
        "name": asset_name,
        "displayName": asset_name,
        "attributes": collibra_attributes,
    }

    return asset_payload


def extract_governance_metadata(attributes: dict[str, Any]) -> dict[str, Any]:
    """
    Extract governance metadata from GenOps attributes.

    Args:
        attributes: GenOps span attributes

    Returns:
        Dictionary with governance metadata (team, project, customer_id, etc.)

    Example:
        >>> attrs = {
        ...     "genops.team": "ml-platform",
        ...     "genops.project": "chatbot",
        ...     "genops.customer_id": "enterprise-123",
        ...     "genops.cost.total": 0.05
        ... }
        >>> extract_governance_metadata(attrs)
        {
            'team': 'ml-platform',
            'project': 'chatbot',
            'customer_id': 'enterprise-123'
        }
    """
    governance_keys = [
        "genops.team",
        "genops.project",
        "genops.customer_id",
        "genops.environment",
        "genops.cost_center",
        "genops.feature",
    ]

    metadata = {}
    for key in governance_keys:
        if key in attributes:
            # Strip genops. prefix for cleaner keys
            clean_key = key.replace("genops.", "")
            metadata[clean_key] = attributes[key]

    return metadata
