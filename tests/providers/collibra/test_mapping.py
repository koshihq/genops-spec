"""Unit tests for Collibra data mapping."""

import pytest

from genops.providers.collibra.mapping import (
    create_collibra_asset_from_span,
    create_collibra_asset_name,
    extract_governance_metadata,
    infer_asset_type_from_attributes,
    map_collibra_attributes_to_genops,
    map_collibra_to_genops_asset_type,
    map_genops_attributes_to_collibra,
    map_genops_to_collibra_asset_type,
)


def test_map_genops_to_collibra_asset_type():
    """Test GenOps to Collibra asset type mapping."""
    assert map_genops_to_collibra_asset_type("cost") == "AI Operation Cost"
    assert map_genops_to_collibra_asset_type("policy") == "Policy Evaluation Event"
    assert map_genops_to_collibra_asset_type("evaluation") == "Model Evaluation"
    assert map_genops_to_collibra_asset_type("budget") == "Budget Allocation"
    assert map_genops_to_collibra_asset_type("operation") == "AI Workflow Execution"
    assert map_genops_to_collibra_asset_type("unknown") == "AI Workflow Execution"


def test_map_collibra_to_genops_asset_type():
    """Test Collibra to GenOps asset type mapping."""
    assert map_collibra_to_genops_asset_type("AI Operation Cost") == "cost"
    assert (
        map_collibra_to_genops_asset_type("Policy Evaluation Event") == "policy"
    )
    assert map_collibra_to_genops_asset_type("Model Evaluation") == "evaluation"
    assert map_collibra_to_genops_asset_type("Budget Allocation") == "budget"
    assert (
        map_collibra_to_genops_asset_type("AI Workflow Execution") == "operation"
    )
    assert map_collibra_to_genops_asset_type("Unknown Type") == "operation"


def test_map_genops_attributes_to_collibra():
    """Test mapping GenOps attributes to Collibra."""
    genops_attrs = {
        "genops.cost.total": 0.05,
        "genops.cost.provider": "openai",
        "genops.team": "ml-platform",
        "genops.tokens.input": 150,
        "genops.custom.field": "value",
    }

    collibra_attrs = map_genops_attributes_to_collibra(genops_attrs)

    assert collibra_attrs["cost_amount"] == 0.05
    assert collibra_attrs["ai_provider"] == "openai"
    assert collibra_attrs["team"] == "ml-platform"
    assert collibra_attrs["tokens_input"] == 150
    assert collibra_attrs["custom.field"] == "value"


def test_map_collibra_attributes_to_genops():
    """Test mapping Collibra attributes to GenOps."""
    collibra_attrs = {
        "cost_amount": 0.05,
        "ai_provider": "openai",
        "team": "ml-platform",
        "tokens_input": 150,
        "custom_field": "value",
    }

    genops_attrs = map_collibra_attributes_to_genops(collibra_attrs)

    assert genops_attrs["genops.cost.total"] == 0.05
    assert genops_attrs["genops.cost.provider"] == "openai"
    assert genops_attrs["genops.team"] == "ml-platform"
    assert genops_attrs["genops.tokens.input"] == 150
    assert genops_attrs["genops.custom_field"] == "value"


def test_infer_asset_type_from_cost_attributes():
    """Test inferring asset type from cost attributes."""
    attributes = {"genops.cost.total": 0.05, "genops.cost.provider": "openai"}

    asset_type = infer_asset_type_from_attributes(attributes)

    assert asset_type == "AI Operation Cost"


def test_infer_asset_type_from_policy_attributes():
    """Test inferring asset type from policy attributes."""
    attributes = {
        "genops.policy.name": "cost_limit",
        "genops.policy.result": "allowed",
    }

    asset_type = infer_asset_type_from_attributes(attributes)

    assert asset_type == "Policy Evaluation Event"


def test_infer_asset_type_from_evaluation_attributes():
    """Test inferring asset type from evaluation attributes."""
    attributes = {"genops.eval.metric": "accuracy", "genops.eval.score": 0.95}

    asset_type = infer_asset_type_from_attributes(attributes)

    assert asset_type == "Model Evaluation"


def test_infer_asset_type_from_budget_attributes():
    """Test inferring asset type from budget attributes."""
    attributes = {
        "genops.budget.name": "team-monthly",
        "genops.budget.allocated": 1000.0,
    }

    asset_type = infer_asset_type_from_attributes(attributes)

    assert asset_type == "Budget Allocation"


def test_infer_asset_type_default():
    """Test default asset type inference."""
    attributes = {"genops.operation.name": "test-operation"}

    asset_type = infer_asset_type_from_attributes(attributes)

    assert asset_type == "AI Workflow Execution"


def test_create_collibra_asset_name_cost():
    """Test creating asset name for cost type."""
    attributes = {
        "genops.operation.name": "gpt-4-completion",
        "genops.team": "ml-platform",
        "genops.cost.total": 0.05,
        "genops.cost.currency": "USD",
    }

    name = create_collibra_asset_name(attributes, "AI Operation Cost")

    assert "gpt-4-completion" in name
    assert "ml-platform" in name
    assert "$0.05" in name


def test_create_collibra_asset_name_policy():
    """Test creating asset name for policy type."""
    attributes = {
        "genops.operation.name": "completion",
        "genops.team": "data-science",
        "genops.policy.name": "rate_limit",
        "genops.policy.result": "blocked",
    }

    name = create_collibra_asset_name(attributes, "Policy Evaluation Event")

    assert "completion" in name
    assert "data-science" in name
    assert "rate_limit" in name
    assert "blocked" in name


def test_create_collibra_asset_name_evaluation():
    """Test creating asset name for evaluation type."""
    attributes = {
        "genops.operation.name": "model-evaluation",
        "genops.eval.metric": "accuracy",
        "genops.eval.score": 0.927,
    }

    name = create_collibra_asset_name(attributes, "Model Evaluation")

    assert "model-evaluation" in name
    assert "accuracy" in name
    assert "0.927" in name


def test_create_collibra_asset_from_span():
    """Test creating complete Collibra asset from span attributes."""
    span_attributes = {
        "genops.cost.total": 0.05,
        "genops.cost.provider": "openai",
        "genops.cost.model": "gpt-4",
        "genops.operation.name": "completion",
        "genops.team": "ml-platform",
        "genops.project": "chatbot",
        "genops.tokens.input": 150,
        "genops.tokens.output": 200,
    }

    asset = create_collibra_asset_from_span(span_attributes, "domain-123")

    assert asset["domainId"] == "domain-123"
    assert asset["typeId"] == "AI Operation Cost"
    assert "completion" in asset["name"]
    assert "ml-platform" in asset["name"]
    assert asset["attributes"]["cost_amount"] == 0.05
    assert asset["attributes"]["ai_provider"] == "openai"
    assert asset["attributes"]["team"] == "ml-platform"
    assert asset["attributes"]["project"] == "chatbot"


def test_create_collibra_asset_with_override_type():
    """Test creating asset with override type."""
    span_attributes = {
        "genops.cost.total": 0.05,
        "genops.operation.name": "test-operation",
    }

    asset = create_collibra_asset_from_span(
        span_attributes, "domain-123", asset_type="Model Evaluation"
    )

    assert asset["typeId"] == "Model Evaluation"


def test_extract_governance_metadata():
    """Test extracting governance metadata."""
    attributes = {
        "genops.team": "ml-platform",
        "genops.project": "chatbot",
        "genops.customer_id": "enterprise-123",
        "genops.environment": "production",
        "genops.cost_center": "engineering",
        "genops.feature": "chat-completion",
        "genops.cost.total": 0.05,  # Should not be included
    }

    metadata = extract_governance_metadata(attributes)

    assert metadata["team"] == "ml-platform"
    assert metadata["project"] == "chatbot"
    assert metadata["customer_id"] == "enterprise-123"
    assert metadata["environment"] == "production"
    assert metadata["cost_center"] == "engineering"
    assert metadata["feature"] == "chat-completion"
    assert "cost.total" not in metadata


def test_extract_governance_metadata_empty():
    """Test extracting governance metadata when none present."""
    attributes = {"genops.cost.total": 0.05, "genops.operation.name": "test"}

    metadata = extract_governance_metadata(attributes)

    assert metadata == {}
