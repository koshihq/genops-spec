"""Unit tests for Collibra policy importer."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from genops.core.policy import PolicyConfig, PolicyResult
from genops.providers.collibra.policy_importer import PolicyImporter, PolicySyncStats


@pytest.fixture
def mock_client():
    """Create mock Collibra client."""
    client = MagicMock()
    client.list_domains.return_value = [{"id": "domain-123", "name": "Test Domain"}]
    client.list_assets.return_value = []
    return client


@pytest.fixture
def sample_collibra_policies():
    """Create sample Collibra policy assets."""
    return [
        {
            "id": "policy-001",
            "name": "Cost Limit Policy",
            "typeId": "AI Cost Limit",
            "domainId": "domain-123",
            "attributes": {
                "enforcement_level": "block",
                "enabled": True,
                "description": "Maximum cost per operation",
                "max_cost": 10.0,
            },
        },
        {
            "id": "policy-002",
            "name": "Rate Limit Policy",
            "typeId": "AI Rate Limit",
            "domainId": "domain-123",
            "attributes": {
                "enforcement_level": "rate_limit",
                "enabled": True,
                "description": "Request rate throttling",
                "max_requests_per_minute": 100,
            },
        },
        {
            "id": "policy-003",
            "name": "Content Filter",
            "typeId": "Content Filter",
            "domainId": "domain-123",
            "attributes": {
                "enforcement_level": "warn",
                "enabled": True,
                "description": "Blocked content patterns",
                "blocked_patterns": "sensitive,confidential,secret",
            },
        },
    ]


def test_policy_importer_initialization(mock_client):
    """Test policy importer initialization."""
    importer = PolicyImporter(
        client=mock_client,
        domain_id="domain-123",
        sync_interval_minutes=5,
        enable_background_sync=False,
    )

    assert importer.client == mock_client
    assert importer.domain_id == "domain-123"
    assert importer.sync_interval_minutes == 5
    assert importer.background_sync_enabled is False
    assert importer.background_thread is None


def test_policy_importer_with_background_sync(mock_client):
    """Test policy importer with background sync enabled."""
    importer = PolicyImporter(
        client=mock_client,
        domain_id="domain-123",
        sync_interval_minutes=1,
        enable_background_sync=True,
    )

    # Background thread should be started
    assert importer.background_thread is not None
    assert importer.background_thread.is_alive()

    # Clean up
    importer.shutdown()


def test_fetch_policies_from_domain(mock_client, sample_collibra_policies):
    """Test fetching policies from specific domain."""
    mock_client.list_assets.return_value = sample_collibra_policies

    importer = PolicyImporter(
        client=mock_client, domain_id="domain-123", enable_background_sync=False
    )

    policies = importer.fetch_policies(domain_id="domain-123")

    assert len(policies) == 3
    assert mock_client.list_assets.called


def test_fetch_policies_from_all_domains(mock_client, sample_collibra_policies):
    """Test fetching policies from all domains."""
    mock_client.list_domains.return_value = [
        {"id": "domain-1", "name": "Domain 1"},
        {"id": "domain-2", "name": "Domain 2"},
    ]
    mock_client.list_assets.return_value = sample_collibra_policies

    importer = PolicyImporter(
        client=mock_client, domain_id=None, enable_background_sync=False
    )

    policies = importer.fetch_policies()

    # Should fetch from both domains
    assert mock_client.list_assets.call_count == 2


def test_translate_cost_limit_policy(mock_client):
    """Test translating cost limit policy."""
    importer = PolicyImporter(
        client=mock_client, domain_id="domain-123", enable_background_sync=False
    )

    collibra_policy = {
        "id": "policy-001",
        "name": "Cost Limit",
        "typeId": "AI Cost Limit",
        "attributes": {
            "enforcement_level": "block",
            "enabled": True,
            "description": "Max cost policy",
            "max_cost": 5.0,
        },
    }

    policy_config = importer.translate_policy(collibra_policy)

    assert policy_config is not None
    assert "cost_limit" in policy_config.name
    assert policy_config.enabled is True
    assert policy_config.enforcement_level == PolicyResult.BLOCKED
    assert policy_config.conditions["max_cost"] == 5.0


def test_translate_rate_limit_policy(mock_client):
    """Test translating rate limit policy."""
    importer = PolicyImporter(
        client=mock_client, domain_id="domain-123", enable_background_sync=False
    )

    collibra_policy = {
        "id": "policy-002",
        "name": "Rate Limit",
        "typeId": "AI Rate Limit",
        "attributes": {
            "enforcement_level": "throttle",
            "enabled": True,
            "max_requests_per_minute": 100,
        },
    }

    policy_config = importer.translate_policy(collibra_policy)

    assert policy_config is not None
    assert "rate_limit" in policy_config.name
    assert policy_config.enforcement_level == PolicyResult.RATE_LIMITED
    assert policy_config.conditions["max_requests_per_minute"] == 100


def test_translate_content_filter_policy(mock_client):
    """Test translating content filter policy."""
    importer = PolicyImporter(
        client=mock_client, domain_id="domain-123", enable_background_sync=False
    )

    collibra_policy = {
        "id": "policy-003",
        "name": "Content Filter",
        "typeId": "Content Filter",
        "attributes": {
            "enforcement_level": "warn",
            "enabled": True,
            "blocked_patterns": "secret,confidential",
        },
    }

    policy_config = importer.translate_policy(collibra_policy)

    assert policy_config is not None
    assert "content_filter" in policy_config.name
    assert policy_config.enforcement_level == PolicyResult.WARNING
    assert "blocked_patterns" in policy_config.conditions
    assert len(policy_config.conditions["blocked_patterns"]) == 2


def test_translate_team_access_policy(mock_client):
    """Test translating team access policy."""
    importer = PolicyImporter(
        client=mock_client, domain_id="domain-123", enable_background_sync=False
    )

    collibra_policy = {
        "id": "policy-004",
        "name": "Team Access",
        "typeId": "Team Access Control",
        "attributes": {
            "enforcement_level": "block",
            "enabled": True,
            "allowed_teams": "ml-platform,data-science",
        },
    }

    policy_config = importer.translate_policy(collibra_policy)

    assert policy_config is not None
    assert "team_access" in policy_config.name
    assert policy_config.enforcement_level == PolicyResult.BLOCKED
    assert "allowed_teams" in policy_config.conditions
    assert len(policy_config.conditions["allowed_teams"]) == 2


def test_translate_budget_constraint_policy(mock_client):
    """Test translating budget constraint policy."""
    importer = PolicyImporter(
        client=mock_client, domain_id="domain-123", enable_background_sync=False
    )

    collibra_policy = {
        "id": "policy-005",
        "name": "Budget Constraint",
        "typeId": "Budget Constraint",
        "attributes": {
            "enforcement_level": "block",
            "enabled": True,
            "daily_budget": 100.0,
            "monthly_budget": 3000.0,
        },
    }

    policy_config = importer.translate_policy(collibra_policy)

    assert policy_config is not None
    assert "budget_limit" in policy_config.name
    assert policy_config.conditions["daily_budget"] == 100.0
    assert policy_config.conditions["monthly_budget"] == 3000.0


def test_translate_model_governance_policy(mock_client):
    """Test translating model governance policy."""
    importer = PolicyImporter(
        client=mock_client, domain_id="domain-123", enable_background_sync=False
    )

    collibra_policy = {
        "id": "policy-006",
        "name": "Model Governance",
        "typeId": "Model Governance",
        "attributes": {
            "enforcement_level": "block",
            "enabled": True,
            "allowed_models": "gpt-4,claude-3",
            "blocked_models": "gpt-3.5-turbo",
        },
    }

    policy_config = importer.translate_policy(collibra_policy)

    assert policy_config is not None
    assert "model_governance" in policy_config.name
    assert "allowed_models" in policy_config.conditions
    assert "blocked_models" in policy_config.conditions


def test_custom_policy_transformer(mock_client):
    """Test custom policy transformation function."""

    def custom_transformer(collibra_policy):
        """Custom transformer that always creates a warning policy."""
        return PolicyConfig(
            name="custom_policy",
            description="Custom transformed policy",
            enabled=True,
            enforcement_level=PolicyResult.WARNING,
            conditions={"custom": True},
        )

    importer = PolicyImporter(
        client=mock_client,
        domain_id="domain-123",
        enable_background_sync=False,
        policy_transformer=custom_transformer,
    )

    collibra_policy = {
        "id": "policy-custom",
        "name": "Any Policy",
        "typeId": "Any Type",
        "attributes": {},
    }

    policy_config = importer.translate_policy(collibra_policy)

    assert policy_config.name == "custom_policy"
    assert policy_config.enforcement_level == PolicyResult.WARNING
    assert policy_config.conditions["custom"] is True


@patch("genops.providers.collibra.policy_importer.register_policy")
def test_import_policies_with_registration(
    mock_register, mock_client, sample_collibra_policies
):
    """Test importing policies with registration."""
    mock_client.list_assets.return_value = sample_collibra_policies

    importer = PolicyImporter(
        client=mock_client, domain_id="domain-123", enable_background_sync=False
    )

    policies = importer.import_policies(register=True)

    assert len(policies) == 3
    assert mock_register.call_count == 3
    assert importer.stats.policies_imported == 3


@patch("genops.providers.collibra.policy_importer.register_policy")
def test_import_policies_without_registration(
    mock_register, mock_client, sample_collibra_policies
):
    """Test importing policies without registration."""
    mock_client.list_assets.return_value = sample_collibra_policies

    importer = PolicyImporter(
        client=mock_client, domain_id="domain-123", enable_background_sync=False
    )

    policies = importer.import_policies(register=False)

    assert len(policies) == 3
    assert mock_register.call_count == 0


@patch("genops.providers.collibra.policy_importer.register_policy")
def test_sync_policies(mock_register, mock_client, sample_collibra_policies):
    """Test policy synchronization."""
    mock_client.list_assets.return_value = sample_collibra_policies

    importer = PolicyImporter(
        client=mock_client, domain_id="domain-123", enable_background_sync=False
    )

    sync_result = importer.sync_policies()

    assert sync_result["imported"] == 3
    assert sync_result["failed"] == 0
    assert "timestamp" in sync_result


def test_get_imported_policies(mock_client, sample_collibra_policies):
    """Test getting imported policies."""
    mock_client.list_assets.return_value = sample_collibra_policies

    with patch("genops.providers.collibra.policy_importer.register_policy"):
        importer = PolicyImporter(
            client=mock_client, domain_id="domain-123", enable_background_sync=False
        )

        importer.import_policies(register=True)

        imported = importer.get_imported_policies()

        assert len(imported) == 3
        assert all(isinstance(p, PolicyConfig) for p in imported.values())


def test_policy_sync_stats_tracking(mock_client, sample_collibra_policies):
    """Test policy sync statistics tracking."""
    mock_client.list_assets.return_value = sample_collibra_policies

    with patch("genops.providers.collibra.policy_importer.register_policy"):
        importer = PolicyImporter(
            client=mock_client, domain_id="domain-123", enable_background_sync=False
        )

        importer.import_policies(register=True)

        stats = importer.get_stats()

        assert stats.policies_imported == 3
        assert stats.policies_failed == 0
        assert stats.last_sync_time is not None


def test_policy_import_failure_handling(mock_client):
    """Test handling of policy import failures."""
    # Configure client to return invalid policy
    mock_client.list_assets.return_value = [
        {
            "id": "policy-invalid",
            "name": "Invalid Policy",
            # Missing typeId
            "attributes": {},
        }
    ]

    with patch("genops.providers.collibra.policy_importer.register_policy"):
        importer = PolicyImporter(
            client=mock_client, domain_id="domain-123", enable_background_sync=False
        )

        policies = importer.import_policies(register=True)

        # Should handle gracefully
        assert len(policies) <= 1  # May skip invalid policy


def test_shutdown_stops_background_sync(mock_client):
    """Test shutdown stops background sync thread."""
    importer = PolicyImporter(
        client=mock_client,
        domain_id="domain-123",
        sync_interval_minutes=60,
        enable_background_sync=True,
    )

    assert importer.background_thread.is_alive()

    # Shutdown
    importer.shutdown(timeout=2.0)

    # Background thread should be stopped
    assert not importer.background_thread.is_alive()


def test_enforcement_level_mapping(mock_client):
    """Test enforcement level mapping from Collibra to GenOps."""
    importer = PolicyImporter(
        client=mock_client, domain_id="domain-123", enable_background_sync=False
    )

    test_cases = [
        ("block", PolicyResult.BLOCKED),
        ("blocked", PolicyResult.BLOCKED),
        ("enforce", PolicyResult.BLOCKED),
        ("warn", PolicyResult.WARNING),
        ("warning", PolicyResult.WARNING),
        ("alert", PolicyResult.WARNING),
        ("rate_limit", PolicyResult.RATE_LIMITED),
        ("throttle", PolicyResult.RATE_LIMITED),
        ("allow", PolicyResult.ALLOWED),
        ("allowed", PolicyResult.ALLOWED),
    ]

    for collibra_level, expected_result in test_cases:
        collibra_policy = {
            "id": "policy-test",
            "name": "Test Policy",
            "typeId": "AI Cost Limit",
            "attributes": {
                "enforcement_level": collibra_level,
                "enabled": True,
                "max_cost": 10.0,
            },
        }

        policy_config = importer.translate_policy(collibra_policy)
        assert policy_config.enforcement_level == expected_result


def test_disabled_policy_import(mock_client):
    """Test importing disabled policies."""
    collibra_policy = {
        "id": "policy-disabled",
        "name": "Disabled Policy",
        "typeId": "AI Cost Limit",
        "attributes": {
            "enforcement_level": "block",
            "enabled": False,  # Disabled
            "max_cost": 10.0,
        },
    }

    with patch("genops.providers.collibra.policy_importer.register_policy"):
        importer = PolicyImporter(
            client=mock_client, domain_id="domain-123", enable_background_sync=False
        )

        policy_config = importer.translate_policy(collibra_policy)

        assert policy_config.enabled is False
