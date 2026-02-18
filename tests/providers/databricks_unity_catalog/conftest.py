"""Shared fixtures and utilities for Databricks Unity Catalog tests."""

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Test constants
TEST_WORKSPACE_URL = "https://test-workspace.cloud.databricks.com"
TEST_METASTORE_ID = "test-metastore-12345"
TEST_CATALOG_NAME = "test_catalog"
TEST_SCHEMA_NAME = "test_schema"
TEST_TABLE_NAME = "test_table"
TEST_SQL_WAREHOUSE_ID = "test-warehouse-small"

# Sample governance attributes
SAMPLE_GOVERNANCE_ATTRS = {
    "team": "test-data-team",
    "project": "unity-catalog-testing",
    "environment": "test",
    "customer_id": "test-customer-123",
    "cost_center": "engineering",
    "user_id": "test-user@example.com",
}


@pytest.fixture
def workspace_url():
    """Provide test workspace URL."""
    return TEST_WORKSPACE_URL


@pytest.fixture
def metastore_id():
    """Provide test metastore ID."""
    return TEST_METASTORE_ID


@pytest.fixture
def catalog_name():
    """Provide test catalog name."""
    return TEST_CATALOG_NAME


@pytest.fixture
def schema_name():
    """Provide test schema name."""
    return TEST_SCHEMA_NAME


@pytest.fixture
def table_name():
    """Provide test table name."""
    return TEST_TABLE_NAME


@pytest.fixture
def sql_warehouse_id():
    """Provide test SQL warehouse ID."""
    return TEST_SQL_WAREHOUSE_ID


@pytest.fixture
def sample_governance_attrs():
    """Provide sample governance attributes."""
    return SAMPLE_GOVERNANCE_ATTRS.copy()


@pytest.fixture
def mock_databricks_client():
    """Mock Databricks WorkspaceClient."""
    mock_client = MagicMock()

    # Mock catalogs operations
    mock_catalog = MagicMock()
    mock_catalog.name = TEST_CATALOG_NAME
    mock_catalog.metastore_id = TEST_METASTORE_ID
    mock_catalog.created_at = datetime.now()

    mock_client.catalogs.list.return_value = [mock_catalog]
    mock_client.catalogs.get.return_value = mock_catalog

    # Mock schemas operations
    mock_schema = MagicMock()
    mock_schema.name = TEST_SCHEMA_NAME
    mock_schema.catalog_name = TEST_CATALOG_NAME
    mock_schema.created_at = datetime.now()

    mock_client.schemas.list.return_value = [mock_schema]
    mock_client.schemas.get.return_value = mock_schema

    # Mock tables operations
    mock_table = MagicMock()
    mock_table.name = TEST_TABLE_NAME
    mock_table.catalog_name = TEST_CATALOG_NAME
    mock_table.schema_name = TEST_SCHEMA_NAME
    mock_table.table_type = "MANAGED"
    mock_table.data_source_format = "DELTA"
    mock_table.created_at = datetime.now()

    mock_client.tables.list.return_value = [mock_table]
    mock_client.tables.get.return_value = mock_table

    # Mock SQL warehouses operations
    mock_warehouse = MagicMock()
    mock_warehouse.id = TEST_SQL_WAREHOUSE_ID
    mock_warehouse.name = "Test Warehouse Small"
    mock_warehouse.cluster_size = "Small"
    mock_warehouse.state = "RUNNING"

    mock_client.warehouses.list.return_value = [mock_warehouse]
    mock_client.warehouses.get.return_value = mock_warehouse

    # Mock current user
    mock_user = MagicMock()
    mock_user.user_name = "test-user@example.com"
    mock_user.id = "test-user-id-123"

    mock_client.current_user.me.return_value = mock_user

    return mock_client


@pytest.fixture
def mock_databricks_sdk():
    """Mock entire Databricks SDK module."""
    with patch("databricks.sdk.WorkspaceClient") as mock_client_class:
        mock_client = mock_databricks_client()
        mock_client_class.return_value = mock_client
        yield mock_client_class


@pytest.fixture
def mock_table_operation_result():
    """Mock table operation result data."""
    return {
        "operation": "query",
        "catalog_name": TEST_CATALOG_NAME,
        "schema_name": TEST_SCHEMA_NAME,
        "table_name": TEST_TABLE_NAME,
        "row_count": 25000,
        "data_size_bytes": 85 * 1024 * 1024,  # 85 MB
        "query_duration_ms": 2500.0,
        "cost_usd": 0.0045,
        "governance_attributes": SAMPLE_GOVERNANCE_ATTRS,
    }


@pytest.fixture
def mock_sql_warehouse_operation_result():
    """Mock SQL warehouse operation result data."""
    return {
        "sql_warehouse_id": TEST_SQL_WAREHOUSE_ID,
        "query_type": "analytics",
        "query_duration_ms": 3500.0,
        "compute_units": 1.2,
        "cost_usd": 0.0078,
        "governance_attributes": SAMPLE_GOVERNANCE_ATTRS,
    }


@pytest.fixture
def mock_cost_summary_data():
    """Mock cost aggregation summary data."""
    return {
        "total_cost_usd": 0.0523,
        "operation_count": 8,
        "unique_workspaces": {"test-workspace"},
        "cost_by_team": {"test-data-team": 0.0523},
        "cost_by_project": {"unity-catalog-testing": 0.0523},
        "cost_by_resource_type": {
            "sql_warehouse": 0.0312,
            "compute_cluster": 0.0156,
            "storage": 0.0055,
        },
        "cost_by_workspace": {"test-workspace": 0.0523},
    }


@pytest.fixture
def mock_lineage_data():
    """Mock data lineage tracking data."""
    return {
        "lineage_type": "transform",
        "source_catalog": "raw_data",
        "source_schema": "events",
        "source_table": "user_sessions",
        "target_catalog": "analytics",
        "target_schema": "aggregated",
        "target_table": "session_metrics",
        "transformation_logic": "GROUP BY user_id, DATE(session_start)",
        "data_classification": "internal",
        "user_id": "data-engineer@example.com",
        "timestamp": datetime.now(),
    }


@pytest.fixture
def mock_governance_summary_data():
    """Mock governance operation summary data."""
    return {
        "lineage_events": 12,
        "policy_evaluations": 8,
        "compliance_checks": 5,
        "data_classifications": {"internal": 8, "confidential": 3, "public": 1},
        "schema_validation_pass": 11,
        "schema_validation_fail": 1,
        "last_updated": datetime.now(),
    }


@pytest.fixture
def mock_validation_result():
    """Mock setup validation result."""
    return {
        "is_valid": True,
        "overall_status": "PASSED",
        "checks": {
            "databricks_connectivity": {
                "status": "PASSED",
                "message": "Connected successfully",
            },
            "unity_catalog_access": {
                "status": "PASSED",
                "message": "Unity Catalog accessible",
            },
            "environment_variables": {
                "status": "PASSED",
                "message": "All required variables set",
            },
            "governance_config": {
                "status": "PASSED",
                "message": "Governance attributes valid",
            },
        },
        "warnings": [],
        "errors": [],
        "summary": "All validation checks passed",
    }


@pytest.fixture
def sample_catalog_operations():
    """Sample catalog operations for testing."""
    return [
        {
            "operation": "list",
            "catalog_name": "production",
            "expected_result": "success",
        },
        {
            "operation": "create",
            "catalog_name": "new_catalog",
            "expected_result": "success",
        },
        {
            "operation": "delete",
            "catalog_name": "temp_catalog",
            "expected_result": "success",
        },
        {
            "operation": "read",
            "catalog_name": "analytics",
            "expected_result": "success",
        },
    ]


@pytest.fixture
def sample_table_operations():
    """Sample table operations for testing."""
    return [
        {
            "operation": "query",
            "catalog": "production",
            "schema": "events",
            "table": "user_actions",
            "rows": 50000,
            "size_mb": 125,
        },
        {
            "operation": "write",
            "catalog": "analytics",
            "schema": "aggregated",
            "table": "daily_metrics",
            "rows": 1000,
            "size_mb": 5,
        },
        {
            "operation": "read",
            "catalog": "raw_data",
            "schema": "ingestion",
            "table": "sensor_data",
            "rows": 100000,
            "size_mb": 250,
        },
    ]


@pytest.fixture
def sample_sql_warehouse_operations():
    """Sample SQL warehouse operations for testing."""
    return [
        {
            "warehouse_id": "analytics-small",
            "query_type": "select",
            "duration_ms": 1500,
            "compute_units": 0.5,
        },
        {
            "warehouse_id": "analytics-medium",
            "query_type": "transform",
            "duration_ms": 5000,
            "compute_units": 2.0,
        },
        {
            "warehouse_id": "production-large",
            "query_type": "aggregation",
            "duration_ms": 12000,
            "compute_units": 4.5,
        },
    ]


class MockEnvironment:
    """Mock environment variables for testing."""

    def __init__(self):
        self.env_vars = {
            "DATABRICKS_HOST": TEST_WORKSPACE_URL,
            "DATABRICKS_TOKEN": "test-token-12345",
            "DATABRICKS_METASTORE_ID": TEST_METASTORE_ID,
            "GENOPS_TEAM": "test-data-team",
            "GENOPS_PROJECT": "unity-catalog-testing",
            "GENOPS_ENVIRONMENT": "test",
            "GENOPS_COST_CENTER": "engineering",
        }

    def set_env(self, key: str, value: str):
        """Set environment variable."""
        self.env_vars[key] = value

    def get_env(self, key: str, default: str = None):
        """Get environment variable."""
        return self.env_vars.get(key, default)

    def clear_env(self, key: str):
        """Clear environment variable."""
        self.env_vars.pop(key, None)

    def as_patch(self):
        """Return as patch context manager."""
        return patch.dict("os.environ", self.env_vars)


@pytest.fixture
def mock_environment():
    """Provide mock environment for testing."""
    return MockEnvironment()


class DatabricksTestHelpers:
    """Helper utilities for Databricks testing."""

    @staticmethod
    def create_mock_operation_result(
        operation_type: str, success: bool = True, cost_usd: float = 0.001, **kwargs
    ) -> dict[str, Any]:
        """Create mock operation result."""
        base_result = {
            "operation_type": operation_type,
            "success": success,
            "cost_usd": cost_usd,
            "timestamp": datetime.now(),
            "governance_attributes": SAMPLE_GOVERNANCE_ATTRS.copy(),
        }
        base_result.update(kwargs)
        return base_result

    @staticmethod
    def assert_valid_governance_tracking(result: dict[str, Any]):
        """Assert result has valid governance tracking."""
        assert "governance_attributes" in result
        gov_attrs = result["governance_attributes"]
        assert "team" in gov_attrs
        assert "project" in gov_attrs
        assert gov_attrs["team"] is not None
        assert gov_attrs["project"] is not None

    @staticmethod
    def assert_valid_cost_calculation(result: dict[str, Any]):
        """Assert result has valid cost calculation."""
        assert "cost_usd" in result
        assert isinstance(result["cost_usd"], (int, float))
        assert result["cost_usd"] >= 0

    @staticmethod
    def assert_valid_telemetry_attributes(span_attributes: dict[str, Any]):
        """Assert span has required telemetry attributes."""
        required_attrs = [
            "genops.provider",
            "genops.framework_type",
            "genops.operation_type",
        ]
        for attr in required_attrs:
            assert attr in span_attributes

        assert span_attributes["genops.provider"] == "databricks_unity_catalog"
        assert span_attributes["genops.framework_type"] == "data_platform"


@pytest.fixture
def test_helpers():
    """Provide test helper utilities."""
    return DatabricksTestHelpers()


@pytest.fixture
def mock_adapter_with_dependencies():
    """Mock adapter with all dependencies set up."""
    from unittest.mock import patch

    with patch("databricks.sdk.WorkspaceClient"):
        with patch(
            "genops.providers.databricks_unity_catalog.adapter.GenOpsDatabricksUnityCatalogAdapter"
        ) as mock_adapter_class:
            mock_adapter = MagicMock()
            mock_adapter_class.return_value = mock_adapter

            # Configure mock adapter methods
            mock_adapter.track_catalog_operation.return_value = {
                "operation": "test",
                "cost_usd": 0.001,
                "governance_attributes": SAMPLE_GOVERNANCE_ATTRS,
            }

            mock_adapter.track_table_operation.return_value = {
                "operation": "query",
                "cost_usd": 0.005,
                "governance_attributes": SAMPLE_GOVERNANCE_ATTRS,
            }

            mock_adapter.track_sql_warehouse_operation.return_value = {
                "operation": "analytics",
                "cost_usd": 0.012,
                "governance_attributes": SAMPLE_GOVERNANCE_ATTRS,
            }

            yield mock_adapter


@pytest.fixture
def error_scenarios():
    """Common error scenarios for testing."""
    return {
        "connection_failed": {
            "error_type": "ConnectionError",
            "message": "Failed to connect to Databricks workspace",
            "status_code": None,
        },
        "authentication_failed": {
            "error_type": "AuthenticationError",
            "message": "Invalid Databricks token",
            "status_code": 401,
        },
        "unity_catalog_not_enabled": {
            "error_type": "PermissionError",
            "message": "Unity Catalog not enabled for this workspace",
            "status_code": 403,
        },
        "catalog_not_found": {
            "error_type": "NotFoundError",
            "message": "Catalog not found",
            "status_code": 404,
        },
        "rate_limit_exceeded": {
            "error_type": "RateLimitError",
            "message": "Rate limit exceeded",
            "status_code": 429,
        },
    }
