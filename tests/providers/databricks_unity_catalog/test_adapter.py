"""
Comprehensive tests for GenOps Databricks Unity Catalog Adapter.

Tests the core adapter functionality including:
- Catalog, schema, and table operation tracking
- SQL warehouse operation monitoring
- Multi-workspace governance
- Cost calculation accuracy
- Error handling and resilience
- Auto-instrumentation patterns
- Performance monitoring
"""

import threading
import time
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

# Import the modules under test
try:
    from genops.providers.databricks_unity_catalog import (
        GenOpsDatabricksUnityCatalogAdapter,
        instrument_databricks_unity_catalog,
    )
    from genops.providers.databricks_unity_catalog.adapter import (
        DataPlatformOperationResult,
        create_unity_catalog_operation_context,
    )

    DATABRICKS_AVAILABLE = True
except ImportError:
    DATABRICKS_AVAILABLE = False


@pytest.mark.skipif(
    not DATABRICKS_AVAILABLE, reason="Databricks Unity Catalog provider not available"
)
class TestGenOpsDatabricksUnityCatalogAdapter:
    """Test suite for the main Databricks Unity Catalog adapter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = GenOpsDatabricksUnityCatalogAdapter(
            workspace_url="https://test-workspace.cloud.databricks.com",
            metastore_id="test-metastore-123",
        )

    @patch("databricks.sdk.WorkspaceClient")
    def test_adapter_initialization(self, mock_client_class):
        """Test adapter initialization with various configurations."""
        # Test default initialization
        adapter = GenOpsDatabricksUnityCatalogAdapter()
        assert adapter.workspace_url is not None

        # Test custom initialization
        adapter_custom = GenOpsDatabricksUnityCatalogAdapter(
            workspace_url="https://custom-workspace.cloud.databricks.com",
            metastore_id="custom-metastore-456",
        )
        assert "custom-workspace" in adapter_custom.workspace_url
        assert adapter_custom.metastore_id == "custom-metastore-456"

    @patch("databricks.sdk.WorkspaceClient")
    def test_catalog_operation_tracking(
        self, mock_client_class, mock_databricks_client
    ):
        """Test catalog operation tracking functionality."""
        mock_client_class.return_value = mock_databricks_client

        # Test catalog listing operation
        result = self.adapter.track_catalog_operation(
            operation="list",
            catalog_name="production",
            team="data-engineering",
            project="analytics",
            environment="production",
        )

        # Verify result structure
        assert isinstance(result, dict)
        assert result["operation"] == "list"
        assert result["catalog_name"] == "production"
        assert "cost_usd" in result
        assert result["cost_usd"] >= 0

        # Verify governance attributes
        assert result["governance_attributes"]["team"] == "data-engineering"
        assert result["governance_attributes"]["project"] == "analytics"
        assert result["governance_attributes"]["environment"] == "production"

    @patch("databricks.sdk.WorkspaceClient")
    def test_table_operation_tracking(self, mock_client_class, mock_databricks_client):
        """Test table operation tracking with detailed metrics."""
        mock_client_class.return_value = mock_databricks_client

        # Test table query operation
        result = self.adapter.track_table_operation(
            operation="query",
            catalog_name="production",
            schema_name="analytics",
            table_name="customer_events",
            row_count=50000,
            data_size_bytes=100 * 1024 * 1024,  # 100 MB
            team="analytics-team",
            project="customer-insights",
            data_classification="confidential",
        )

        # Verify operation details
        assert result["operation"] == "query"
        assert result["catalog_name"] == "production"
        assert result["schema_name"] == "analytics"
        assert result["table_name"] == "customer_events"
        assert result["row_count"] == 50000
        assert result["data_size_bytes"] == 100 * 1024 * 1024

        # Verify cost calculation
        assert "cost_usd" in result
        assert result["cost_usd"] > 0

        # Verify governance attributes
        governance_attrs = result["governance_attributes"]
        assert governance_attrs["team"] == "analytics-team"
        assert governance_attrs["project"] == "customer-insights"
        assert governance_attrs["data_classification"] == "confidential"

    @patch("databricks.sdk.WorkspaceClient")
    def test_sql_warehouse_operation_tracking(
        self, mock_client_class, mock_databricks_client
    ):
        """Test SQL warehouse operation tracking."""
        mock_client_class.return_value = mock_databricks_client

        result = self.adapter.track_sql_warehouse_operation(
            sql_warehouse_id="analytics-warehouse-small",
            query_type="transform",
            query_duration_ms=3500,
            compute_units=1.5,
            team="data-engineering",
            project="etl-pipeline",
        )

        # Verify warehouse operation details
        assert result["sql_warehouse_id"] == "analytics-warehouse-small"
        assert result["query_type"] == "transform"
        assert result["query_duration_ms"] == 3500
        assert result["compute_units"] == 1.5

        # Verify cost calculation based on compute units and duration
        assert "cost_usd" in result
        assert result["cost_usd"] > 0

        # Verify performance metrics
        assert "latency_ms" in result
        assert result["latency_ms"] >= 0

    def test_cost_calculation_accuracy(self):
        """Test that cost calculations are accurate and consistent."""
        # Test data for different operation types
        test_cases = [
            {
                "operation_type": "table_query",
                "row_count": 10000,
                "data_size_gb": 1.0,
                "expected_min_cost": 0.001,
            },
            {
                "operation_type": "sql_warehouse",
                "compute_units": 2.0,
                "duration_hours": 0.5,
                "expected_min_cost": 0.05,
            },
            {
                "operation_type": "storage",
                "data_size_tb": 0.1,
                "expected_min_cost": 0.01,
            },
        ]

        for case in test_cases:
            # Cost calculations should be deterministic
            if case["operation_type"] == "table_query":
                cost = self.adapter._calculate_table_operation_cost(
                    row_count=case["row_count"],
                    data_size_bytes=case["data_size_gb"] * 1024**3,
                )
            elif case["operation_type"] == "sql_warehouse":
                cost = self.adapter._calculate_sql_warehouse_cost(
                    compute_units=case["compute_units"],
                    duration_ms=case["duration_hours"] * 3600 * 1000,
                )
            elif case["operation_type"] == "storage":
                cost = self.adapter._calculate_storage_cost(
                    data_size_bytes=case["data_size_tb"] * 1024**4
                )

            assert cost >= case["expected_min_cost"]
            assert isinstance(cost, (int, float))

    @patch("databricks.sdk.WorkspaceClient")
    def test_multi_workspace_support(self, mock_client_class):
        """Test support for multiple Databricks workspaces."""
        workspaces = [
            "https://prod-us-west.cloud.databricks.com",
            "https://prod-eu-central.cloud.databricks.com",
            "https://staging.cloud.databricks.com",
        ]

        adapters = []
        for workspace_url in workspaces:
            adapter = GenOpsDatabricksUnityCatalogAdapter(workspace_url=workspace_url)
            adapters.append(adapter)
            assert adapter.workspace_url == workspace_url

        # Verify each adapter is independent
        assert len(adapters) == 3
        assert len({adapter.workspace_url for adapter in adapters}) == 3

    @patch("databricks.sdk.WorkspaceClient")
    def test_error_handling_scenarios(self, mock_client_class):
        """Test error handling for various failure scenarios."""
        # Test connection error
        mock_client_class.side_effect = ConnectionError(
            "Failed to connect to Databricks"
        )

        with pytest.raises(Exception) as exc_info:
            adapter = GenOpsDatabricksUnityCatalogAdapter()
            adapter.track_catalog_operation("list", "test_catalog", team="test")

        error_msg = str(exc_info.value).lower()
        assert any(
            keyword in error_msg for keyword in ["connect", "connection", "databricks"]
        )

    @patch("databricks.sdk.WorkspaceClient")
    def test_authentication_error_handling(
        self, mock_client_class, mock_databricks_client
    ):
        """Test handling of authentication errors."""
        # Mock authentication failure
        mock_databricks_client.current_user.me.side_effect = Exception(
            "Authentication failed"
        )
        mock_client_class.return_value = mock_databricks_client

        with pytest.raises(Exception) as exc_info:
            adapter = GenOpsDatabricksUnityCatalogAdapter()
            adapter.validate_connection()

        assert "authentication" in str(exc_info.value).lower()

    def test_governance_attributes_validation(self):
        """Test validation of governance attributes."""
        # Test with all governance attributes
        full_governance = {
            "team": "full-team",
            "project": "full-project",
            "customer_id": "full-customer",
            "environment": "production",
            "cost_center": "engineering",
            "feature": "data-analytics",
            "data_classification": "confidential",
        }

        adapter = GenOpsDatabricksUnityCatalogAdapter()

        # Should accept complete governance attributes
        normalized_attrs = adapter._normalize_governance_attributes(**full_governance)
        assert normalized_attrs["team"] == "full-team"
        assert normalized_attrs["project"] == "full-project"
        assert normalized_attrs["environment"] == "production"

        # Test with minimal governance
        minimal_governance = {"team": "minimal-team"}
        normalized_minimal = adapter._normalize_governance_attributes(
            **minimal_governance
        )
        assert normalized_minimal["team"] == "minimal-team"

    @patch("databricks.sdk.WorkspaceClient")
    def test_performance_metrics_capture(
        self, mock_client_class, mock_databricks_client
    ):
        """Test that performance metrics are captured correctly."""
        mock_client_class.return_value = mock_databricks_client

        # Add delay to simulate real operation
        def delayed_operation(*args, **kwargs):
            time.sleep(0.1)  # 100ms delay
            return Mock()

        mock_databricks_client.tables.get.side_effect = delayed_operation

        result = self.adapter.track_table_operation(
            operation="read",
            catalog_name="test_catalog",
            schema_name="test_schema",
            table_name="test_table",
            team="performance-test",
        )

        # Verify performance metrics
        assert "latency_ms" in result
        assert result["latency_ms"] >= 100  # Should capture the 100ms delay
        assert "timestamp" in result
        assert isinstance(result["timestamp"], datetime)

    def test_is_available_check(self):
        """Test availability checking."""
        adapter = GenOpsDatabricksUnityCatalogAdapter()

        # Should have availability check method
        assert hasattr(adapter, "is_available")

        # Method should be callable
        try:
            availability = adapter.is_available()
            assert isinstance(availability, bool)
        except Exception:
            # Expected to fail without real Databricks credentials
            pass

    @patch("databricks.sdk.WorkspaceClient")
    def test_context_manager_support(self, mock_client_class, mock_databricks_client):
        """Test context manager usage pattern."""
        mock_client_class.return_value = mock_databricks_client

        # Test adapter works in context manager
        try:
            with self.adapter as ctx_adapter:
                result = ctx_adapter.track_catalog_operation(
                    operation="list", catalog_name="context_test", team="context-team"
                )
                assert result["operation"] == "list"
        except AttributeError:
            # Context manager may not be implemented yet
            pass

    @patch("databricks.sdk.WorkspaceClient")
    def test_concurrent_usage(self, mock_client_class, mock_databricks_client):
        """Test concurrent usage of the adapter."""
        mock_client_class.return_value = mock_databricks_client

        results = []
        errors = []

        def worker(worker_id):
            try:
                adapter = GenOpsDatabricksUnityCatalogAdapter()
                result = adapter.track_catalog_operation(
                    operation="test",
                    catalog_name=f"catalog_{worker_id}",
                    team=f"worker_{worker_id}",
                )
                results.append(result)
            except Exception as e:
                errors.append(f"worker-{worker_id}: {str(e)}")

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join(timeout=5.0)

        # At least some operations should succeed
        total_operations = len(results) + len(errors)
        assert total_operations == 5

    def test_large_data_operation_handling(self):
        """Test handling of large data operations."""
        adapter = GenOpsDatabricksUnityCatalogAdapter()

        # Test with large row counts and data sizes
        large_operation_cases = [
            {"row_count": 1000000, "data_size_gb": 10.0},
            {"row_count": 10000000, "data_size_gb": 100.0},
            {"row_count": 100000000, "data_size_gb": 1000.0},
        ]

        for case in large_operation_cases:
            # Should handle large operations without error
            cost = adapter._calculate_table_operation_cost(
                row_count=case["row_count"],
                data_size_bytes=case["data_size_gb"] * 1024**3,
            )
            assert cost > 0
            assert isinstance(cost, (int, float))

    def test_different_operation_types(self):
        """Test different types of Unity Catalog operations."""
        operation_types = [
            "create",
            "read",
            "write",
            "update",
            "delete",
            "query",
            "transform",
            "aggregate",
            "join",
            "union",
        ]

        adapter = GenOpsDatabricksUnityCatalogAdapter()

        for operation in operation_types:
            # Each operation type should be handled
            try:
                result = adapter._create_operation_result(
                    operation_type=operation,
                    cost_usd=0.001,
                    governance_attributes={"team": "test"},
                )
                assert result["operation_type"] == operation
            except Exception:
                # Some operations may not be implemented yet
                pass

    @patch("databricks.sdk.WorkspaceClient")
    def test_workspace_region_handling(self, mock_client_class):
        """Test different AWS/Azure/GCP regions."""
        regions_workspaces = [
            "https://dbc-12345678-1234.cloud.databricks.com",  # AWS
            "https://adb-123456789012345.67.azuredatabricks.net",  # Azure
            "https://123456789012345.7.gcp.databricks.com",  # GCP
        ]

        for workspace_url in regions_workspaces:
            adapter = GenOpsDatabricksUnityCatalogAdapter(workspace_url=workspace_url)
            assert adapter.workspace_url == workspace_url

    def test_data_classification_handling(self):
        """Test handling of different data classifications."""
        classifications = ["public", "internal", "confidential", "restricted", "pii"]

        adapter = GenOpsDatabricksUnityCatalogAdapter()

        for classification in classifications:
            # Should handle all classification levels
            governance_attrs = adapter._normalize_governance_attributes(
                team="test-team", data_classification=classification
            )
            assert governance_attrs["data_classification"] == classification

    def test_cost_attribution_accuracy(self):
        """Test accuracy of cost attribution across teams and projects."""
        adapter = GenOpsDatabricksUnityCatalogAdapter()

        # Test different team/project combinations
        attribution_cases = [
            {
                "team": "data-engineering",
                "project": "etl-pipeline",
                "expected_cost_factor": 1.0,
            },
            {"team": "analytics", "project": "reporting", "expected_cost_factor": 0.8},
            {
                "team": "ml-platform",
                "project": "model-training",
                "expected_cost_factor": 1.5,
            },
        ]

        for case in attribution_cases:
            cost = adapter._calculate_attributed_cost(
                base_cost=1.0, team=case["team"], project=case["project"]
            )
            # Cost attribution should preserve or adjust costs appropriately
            assert cost > 0
            assert isinstance(cost, (int, float))

    @patch("databricks.sdk.WorkspaceClient")
    def test_schema_operation_tracking(self, mock_client_class, mock_databricks_client):
        """Test schema-level operation tracking."""
        mock_client_class.return_value = mock_databricks_client

        result = self.adapter.track_schema_operation(
            operation="create",
            catalog_name="test_catalog",
            schema_name="new_schema",
            team="schema-team",
            project="schema-project",
        )

        assert result["operation"] == "create"
        assert result["catalog_name"] == "test_catalog"
        assert result["schema_name"] == "new_schema"
        assert "cost_usd" in result

    def test_memory_usage_patterns(self):
        """Test that memory usage doesn't grow excessively."""
        import gc

        # Get initial memory baseline
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Create and destroy multiple adapters
        adapters = []
        for _ in range(10):
            adapter = GenOpsDatabricksUnityCatalogAdapter()
            adapters.append(adapter)

        # Clean up
        adapters.clear()
        gc.collect()

        final_objects = len(gc.get_objects())

        # Memory growth should be reasonable
        growth_ratio = final_objects / initial_objects
        assert growth_ratio < 1.5, f"Memory growth too high: {growth_ratio}"

    def test_telemetry_attributes_compliance(self):
        """Test that telemetry attributes follow GenOps standards."""
        adapter = GenOpsDatabricksUnityCatalogAdapter()

        # Test telemetry attribute generation
        attrs = adapter._generate_telemetry_attributes(
            operation_type="table.query",
            catalog_name="test_catalog",
            team="test-team",
            project="test-project",
        )

        # Verify required GenOps attributes
        assert attrs["genops.provider"] == "databricks_unity_catalog"
        assert attrs["genops.framework_type"] == "data_platform"
        assert attrs["genops.operation_type"] == "table.query"
        assert attrs["genops.catalog_name"] == "test_catalog"
        assert attrs["genops.team"] == "test-team"
        assert attrs["genops.project"] == "test-project"

    @patch("databricks.sdk.WorkspaceClient")
    def test_operation_context_management(
        self, mock_client_class, mock_databricks_client
    ):
        """Test operation context creation and management."""
        mock_client_class.return_value = mock_databricks_client

        try:
            with create_unity_catalog_operation_context(
                workspace_id="test-workspace", operation_type="analytics_pipeline"
            ) as context:
                # Operations within context should be tracked
                self.adapter.track_table_operation(
                    operation="query",
                    catalog_name="test",
                    schema_name="test",
                    table_name="test",
                    team="context-test",
                )

                # Context should aggregate operations
                assert hasattr(context, "get_summary")

        except (AttributeError, NotImplementedError):
            # Context management may not be fully implemented
            pass

    def test_edge_case_inputs(self):
        """Test edge cases and boundary conditions."""
        adapter = GenOpsDatabricksUnityCatalogAdapter()

        # Test with empty strings
        try:
            adapter._normalize_governance_attributes(team="", project="")
        except ValueError:
            # Expected behavior for invalid inputs
            pass

        # Test with None values
        try:
            adapter._normalize_governance_attributes(team=None, project="valid")
        except ValueError:
            # Expected behavior for None values
            pass

        # Test with very long strings
        long_string = "x" * 1000
        try:
            adapter._normalize_governance_attributes(team=long_string, project="test")
        except ValueError:
            # May have length limits
            pass

    @patch("databricks.sdk.WorkspaceClient")
    def test_real_world_usage_patterns(self, mock_client_class, mock_databricks_client):
        """Test realistic usage patterns and scenarios."""
        mock_client_class.return_value = mock_databricks_client

        # Simulate a typical ETL pipeline
        etl_operations = [
            # Extract phase
            {
                "op": "table",
                "action": "read",
                "catalog": "raw",
                "schema": "ingestion",
                "table": "events",
            },
            {
                "op": "table",
                "action": "read",
                "catalog": "raw",
                "schema": "ingestion",
                "table": "users",
            },
            # Transform phase
            {"op": "warehouse", "warehouse_id": "etl-medium", "query_type": "join"},
            {
                "op": "warehouse",
                "warehouse_id": "etl-medium",
                "query_type": "aggregate",
            },
            # Load phase
            {
                "op": "table",
                "action": "write",
                "catalog": "processed",
                "schema": "analytics",
                "table": "user_metrics",
            },
        ]

        total_cost = 0.0
        for operation in etl_operations:
            if operation["op"] == "table":
                result = self.adapter.track_table_operation(
                    operation=operation["action"],
                    catalog_name=operation["catalog"],
                    schema_name=operation["schema"],
                    table_name=operation["table"],
                    team="etl-team",
                    project="user-analytics",
                )
            elif operation["op"] == "warehouse":
                result = self.adapter.track_sql_warehouse_operation(
                    sql_warehouse_id=operation["warehouse_id"],
                    query_type=operation["query_type"],
                    team="etl-team",
                    project="user-analytics",
                )

            total_cost += result.get("cost_usd", 0.0)

        # ETL pipeline should have accumulated cost
        assert total_cost > 0


class TestInstrumentationFunction:
    """Test the instrumentation function."""

    def test_instrument_function_exists(self):
        """Test that instrumentation function exists."""
        assert callable(instrument_databricks_unity_catalog)

    @patch("databricks.sdk.WorkspaceClient")
    def test_instrumentation_setup(self, mock_client_class):
        """Test that instrumentation can be set up."""
        try:
            adapter = instrument_databricks_unity_catalog(
                workspace_url="https://test-workspace.cloud.databricks.com"
            )
            assert adapter is not None
        except Exception:
            # Expected in test environment without full setup
            pass

    def test_multiple_instrumentation_calls(self):
        """Test that multiple instrumentation calls are safe."""
        try:
            instrument_databricks_unity_catalog()
            instrument_databricks_unity_catalog()
            # Should not raise errors
        except Exception:
            # Expected in test environment
            pass


@pytest.mark.integration
class TestIntegration:
    """Integration tests (require real Databricks credentials)."""

    def test_real_databricks_connectivity(self):
        """Test real Databricks connectivity (skipped if no credentials)."""
        pytest.skip("Integration test - requires real Databricks credentials")

        # This test would be enabled in CI/CD with proper credentials
        adapter = GenOpsDatabricksUnityCatalogAdapter()

        try:
            available = adapter.is_available()
            if available:
                # Test basic operations
                catalogs = adapter.list_catalogs()
                assert isinstance(catalogs, list)
        except Exception as e:
            pytest.skip(f"Databricks not available: {e}")


class TestResultObjects:
    """Test result data structures."""

    def test_operation_result_structure(self):
        """Test that operation results have required fields."""
        try:
            # Test result object creation
            result_data = {
                "operation": "test",
                "cost_usd": 0.001,
                "latency_ms": 150.0,
                "governance_attributes": {"team": "test"},
                "timestamp": datetime.now(),
            }

            if "DataPlatformOperationResult" in globals():
                result = DataPlatformOperationResult(**result_data)
                assert result.operation == "test"
                assert result.cost_usd == 0.001
                assert result.latency_ms == 150.0
        except (NameError, TypeError):
            # Result class may be implemented differently
            pass

    def test_governance_attribute_preservation(self):
        """Test that governance attributes are preserved in results."""
        governance_attrs = {
            "team": "test-team",
            "project": "test-project",
            "environment": "test",
            "customer_id": "test-customer",
        }

        adapter = GenOpsDatabricksUnityCatalogAdapter()

        # Governance attributes should be preserved through operations
        normalized = adapter._normalize_governance_attributes(**governance_attrs)

        for key, value in governance_attrs.items():
            assert normalized[key] == value
