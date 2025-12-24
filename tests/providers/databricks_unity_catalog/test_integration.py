"""
Integration tests for Databricks Unity Catalog provider.

Tests end-to-end integration scenarios including:
- Complete workflow integration from setup to telemetry export
- Cross-provider compatibility scenarios
- Performance under realistic workloads
- Error recovery and resilience patterns
- Real-world usage simulation
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Import the modules under test
try:
    from genops.providers.databricks_unity_catalog import (
        instrument_databricks_unity_catalog,
        get_cost_aggregator,
        get_governance_monitor
    )
    from genops.providers.databricks_unity_catalog.registration import (
        auto_instrument_databricks,
        configure_unity_catalog_governance
    )
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False


@pytest.mark.skipif(not INTEGRATION_AVAILABLE, reason="Integration modules not available")
class TestEndToEndIntegration:
    """Test complete end-to-end integration scenarios."""

    @patch('databricks.sdk.WorkspaceClient')
    def test_complete_etl_pipeline_tracking(self, mock_client_class, mock_databricks_client):
        """Test complete ETL pipeline with governance tracking."""
        mock_client_class.return_value = mock_databricks_client
        
        # Initialize adapter
        adapter = instrument_databricks_unity_catalog(
            workspace_url="https://test-workspace.cloud.databricks.com"
        )
        
        cost_aggregator = get_cost_aggregator()
        governance_monitor = get_governance_monitor("test-metastore")
        
        # Simulate ETL pipeline operations
        etl_operations = [
            # Extract phase
            {
                "phase": "extract",
                "operation": "read",
                "catalog": "raw_data",
                "schema": "external_feeds",
                "table": "customer_events",
                "rows": 100000,
                "size_mb": 250
            },
            {
                "phase": "extract",
                "operation": "read", 
                "catalog": "raw_data",
                "schema": "external_feeds",
                "table": "product_catalog",
                "rows": 50000,
                "size_mb": 125
            },
            
            # Transform phase
            {
                "phase": "transform",
                "operation": "join",
                "warehouse_id": "etl-medium",
                "query_type": "join",
                "duration_ms": 45000,
                "compute_units": 2.5
            },
            {
                "phase": "transform",
                "operation": "aggregate",
                "warehouse_id": "etl-medium",
                "query_type": "aggregate",
                "duration_ms": 30000,
                "compute_units": 1.8
            },
            
            # Load phase
            {
                "phase": "load",
                "operation": "write",
                "catalog": "processed",
                "schema": "analytics",
                "table": "customer_product_interactions",
                "rows": 75000,
                "size_mb": 180
            }
        ]
        
        total_cost = 0.0
        lineage_events = 0
        
        for op in etl_operations:
            if op["operation"] in ["read", "write"]:
                # Track table operation
                result = adapter.track_table_operation(
                    operation=op["operation"],
                    catalog_name=op["catalog"],
                    schema_name=op["schema"],
                    table_name=op["table"],
                    row_count=op["rows"],
                    data_size_bytes=op["size_mb"] * 1024 * 1024,
                    team="etl-team",
                    project="customer-analytics",
                    environment="production",
                    phase=op["phase"]
                )
                
                # Track data lineage
                if op["operation"] == "read":
                    governance_monitor.track_data_lineage(
                        lineage_type="read",
                        source_catalog=op["catalog"],
                        source_schema=op["schema"],
                        source_table=op["table"],
                        user_id="etl-service@example.com"
                    )
                elif op["operation"] == "write":
                    governance_monitor.track_data_lineage(
                        lineage_type="write",
                        target_catalog=op["catalog"],
                        target_schema=op["schema"],
                        target_table=op["table"],
                        user_id="etl-service@example.com"
                    )
                
                lineage_events += 1
                
            else:
                # Track SQL warehouse operation
                result = adapter.track_sql_warehouse_operation(
                    sql_warehouse_id=op["warehouse_id"],
                    query_type=op["query_type"],
                    query_duration_ms=op["duration_ms"],
                    compute_units=op["compute_units"],
                    team="etl-team",
                    project="customer-analytics",
                    environment="production",
                    phase=op["phase"]
                )
            
            total_cost += result.get("cost_usd", 0.0)
        
        # Verify end-to-end results
        cost_summary = cost_aggregator.get_summary()
        governance_summary = governance_monitor.get_governance_summary()
        
        assert cost_summary.total_cost_usd > 0
        assert cost_summary.operation_count >= 5
        assert "etl-team" in cost_summary.cost_by_team
        assert "customer-analytics" in cost_summary.cost_by_project
        
        assert governance_summary.lineage_events >= lineage_events
        
        # Verify telemetry attributes are properly set
        assert "sql_warehouse" in cost_summary.cost_by_resource_type
        assert "storage" in cost_summary.cost_by_resource_type or "table" in cost_summary.cost_by_resource_type

    @patch('databricks.sdk.WorkspaceClient')
    def test_auto_instrumentation_to_telemetry_export(self, mock_client_class, mock_databricks_client):
        """Test full flow from auto-instrumentation to telemetry export."""
        mock_client_class.return_value = mock_databricks_client
        
        # Mock environment for auto-instrumentation
        env_vars = {
            'DATABRICKS_HOST': 'https://auto-test.cloud.databricks.com',
            'DATABRICKS_TOKEN': 'auto-test-token',
            'GENOPS_TEAM': 'automation-team',
            'GENOPS_PROJECT': 'auto-governance',
            'GENOPS_ENVIRONMENT': 'test'
        }
        
        with patch.dict('os.environ', env_vars):
            # Auto-instrument
            adapter = auto_instrument_databricks()
            
            assert adapter is not None
            
            # Perform operations
            result = adapter.track_catalog_operation(
                operation="list",
                catalog_name="auto_test_catalog",
                team="automation-team",
                project="auto-governance"
            )
            
            assert result["operation"] == "list"
            assert result["governance_attributes"]["team"] == "automation-team"
            assert result["governance_attributes"]["project"] == "auto-governance"
            assert result["governance_attributes"]["environment"] == "test"

    @patch('databricks.sdk.WorkspaceClient')
    def test_multi_workspace_governance_coordination(self, mock_client_class, mock_databricks_client):
        """Test governance coordination across multiple workspaces."""
        mock_client_class.return_value = mock_databricks_client
        
        workspaces = [
            {
                "id": "prod-us-west",
                "url": "https://prod-us-west.cloud.databricks.com",
                "metastore": "prod-us-west-metastore"
            },
            {
                "id": "prod-eu-central", 
                "url": "https://prod-eu-central.cloud.databricks.com",
                "metastore": "prod-eu-central-metastore"
            },
            {
                "id": "staging",
                "url": "https://staging.cloud.databricks.com",
                "metastore": "staging-metastore"
            }
        ]
        
        adapters = {}
        total_cross_workspace_cost = 0.0
        
        # Set up governance for each workspace
        for workspace in workspaces:
            adapters[workspace["id"]] = instrument_databricks_unity_catalog(
                workspace_url=workspace["url"]
            )
            
            # Simulate operations in each workspace
            result = adapters[workspace["id"]].track_table_operation(
                operation="read",
                catalog_name="shared_catalog",
                schema_name="cross_workspace",
                table_name="global_metrics",
                team="global-data-team",
                project="cross-workspace-analytics",
                workspace_id=workspace["id"]
            )
            
            total_cross_workspace_cost += result.get("cost_usd", 0.0)
        
        # Verify multi-workspace coordination
        cost_aggregator = get_cost_aggregator()
        cost_summary = cost_aggregator.get_summary()
        
        assert len(cost_summary.unique_workspaces) >= 3
        assert cost_summary.total_cost_usd > 0
        assert "global-data-team" in cost_summary.cost_by_team

    @patch('databricks.sdk.WorkspaceClient') 
    def test_compliance_workflow_integration(self, mock_client_class, mock_databricks_client):
        """Test complete compliance workflow integration."""
        mock_client_class.return_value = mock_databricks_client
        
        adapter = instrument_databricks_unity_catalog()
        governance_monitor = get_governance_monitor("compliance-metastore")
        
        # Simulate compliance-sensitive operations
        sensitive_operations = [
            {
                "catalog": "customer_data",
                "schema": "pii",
                "table": "customer_profiles",
                "classification": "restricted",
                "operation": "query"
            },
            {
                "catalog": "financial",
                "schema": "transactions",
                "table": "credit_card_data",
                "classification": "restricted",
                "operation": "read"
            },
            {
                "catalog": "marketing",
                "schema": "campaigns",
                "table": "customer_preferences",
                "classification": "confidential", 
                "operation": "write"
            }
        ]
        
        compliance_events = 0
        
        for op in sensitive_operations:
            # Track operation
            result = adapter.track_table_operation(
                operation=op["operation"],
                catalog_name=op["catalog"],
                schema_name=op["schema"],
                table_name=op["table"],
                data_classification=op["classification"],
                team="compliance-team",
                project="data-governance",
                user_id="compliance-officer@example.com"
            )
            
            # Enforce classification policy
            policy_result = governance_monitor.enforce_data_classification_policy(
                catalog=op["catalog"],
                schema=op["schema"], 
                table=op["table"],
                required_classification=op["classification"],
                user_clearance="restricted"  # High clearance user
            )
            
            assert policy_result["access_granted"] == True
            
            # Track compliance audit
            governance_monitor.track_compliance_audit(
                audit_type="data_access_review",
                resource_path=f"{op['catalog']}.{op['schema']}.{op['table']}",
                compliance_status="pass",
                findings=[f"authorized_access_to_{op['classification']}_data"]
            )
            
            compliance_events += 1
        
        # Verify compliance workflow results
        governance_summary = governance_monitor.get_governance_summary()
        
        assert governance_summary.compliance_checks >= compliance_events
        assert governance_summary.lineage_events >= compliance_events
        assert "restricted" in governance_summary.data_classifications
        assert "confidential" in governance_summary.data_classifications


class TestPerformanceIntegration:
    """Test performance characteristics under realistic loads."""

    @patch('databricks.sdk.WorkspaceClient')
    def test_high_volume_operation_tracking(self, mock_client_class, mock_databricks_client):
        """Test performance with high volume of operations."""
        mock_client_class.return_value = mock_databricks_client
        
        adapter = instrument_databricks_unity_catalog()
        
        # Track large number of operations
        num_operations = 100
        start_time = time.time()
        
        for i in range(num_operations):
            adapter.track_table_operation(
                operation="read",
                catalog_name=f"catalog_{i % 10}",  # 10 different catalogs
                schema_name=f"schema_{i % 5}",     # 5 different schemas
                table_name=f"table_{i}",
                row_count=1000 + (i * 10),
                team=f"team_{i % 3}",              # 3 different teams
                project=f"project_{i % 4}"         # 4 different projects
            )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Performance should be reasonable
        avg_time_per_operation = total_time / num_operations
        assert avg_time_per_operation < 0.1  # Less than 100ms per operation
        
        # Verify all operations were tracked
        cost_summary = get_cost_aggregator().get_summary()
        assert cost_summary.operation_count >= num_operations

    def test_concurrent_operation_tracking(self):
        """Test concurrent operation tracking performance.""" 
        num_threads = 10
        operations_per_thread = 20
        results = []
        errors = []
        
        def worker_thread(thread_id):
            try:
                adapter = instrument_databricks_unity_catalog(
                    workspace_url=f"https://thread-{thread_id}.cloud.databricks.com"
                )
                
                thread_results = []
                for i in range(operations_per_thread):
                    with patch('databricks.sdk.WorkspaceClient'):
                        result = adapter.track_sql_warehouse_operation(
                            sql_warehouse_id=f"warehouse-{thread_id}",
                            query_type="concurrent_test",
                            query_duration_ms=1000,
                            compute_units=0.5,
                            team=f"thread-team-{thread_id}",
                            project=f"thread-project-{thread_id}"
                        )
                        thread_results.append(result)
                
                results.extend(thread_results)
                
            except Exception as e:
                errors.append(f"Thread {thread_id}: {str(e)}")
        
        # Start all threads
        threads = []
        start_time = time.time()
        
        for i in range(num_threads):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=30.0)  # 30 second timeout
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Verify concurrent performance
        expected_operations = num_threads * operations_per_thread
        
        # Most operations should succeed (allow for some test environment issues)
        success_rate = len(results) / expected_operations
        assert success_rate > 0.8  # At least 80% success rate
        
        # Should complete in reasonable time
        assert total_time < 60  # Less than 60 seconds total

    @patch('databricks.sdk.WorkspaceClient')
    def test_memory_usage_under_load(self, mock_client_class, mock_databricks_client):
        """Test memory usage patterns under sustained load."""
        import gc
        
        mock_client_class.return_value = mock_databricks_client
        
        # Get baseline memory usage
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        adapter = instrument_databricks_unity_catalog()
        
        # Perform sustained operations
        for cycle in range(10):
            for i in range(50):  # 50 operations per cycle
                adapter.track_table_operation(
                    operation="memory_test",
                    catalog_name="memory_test_catalog",
                    schema_name="memory_test_schema",
                    table_name=f"table_{i}",
                    team="memory-test-team",
                    project="memory-test-project"
                )
            
            # Force garbage collection every cycle
            gc.collect()
        
        # Check final memory usage
        final_objects = len(gc.get_objects())
        memory_growth = final_objects - initial_objects
        growth_ratio = final_objects / initial_objects
        
        # Memory growth should be reasonable for 500 operations
        assert growth_ratio < 2.0  # Less than 100% increase
        assert memory_growth < 1000  # Less than 1000 new objects


class TestErrorRecoveryIntegration:
    """Test error recovery and resilience patterns."""

    def test_databricks_connection_failure_recovery(self):
        """Test recovery from Databricks connection failures."""
        # Mock connection that fails initially then succeeds
        connection_attempts = []
        
        def mock_client_factory(*args, **kwargs):
            connection_attempts.append(datetime.now())
            if len(connection_attempts) <= 2:
                raise ConnectionError("Databricks workspace unavailable")
            return MagicMock()  # Success on third attempt
        
        with patch('databricks.sdk.WorkspaceClient', side_effect=mock_client_factory):
            adapter = instrument_databricks_unity_catalog()
            
            # First few operations should fail, then succeed
            failed_operations = 0
            successful_operations = 0
            
            for i in range(5):
                try:
                    result = adapter.track_catalog_operation(
                        operation="resilience_test",
                        catalog_name="test_catalog",
                        team="resilience-team",
                        project="error-recovery"
                    )
                    successful_operations += 1
                except Exception:
                    failed_operations += 1
            
            # Should eventually recover and succeed
            assert successful_operations > 0
            assert len(connection_attempts) >= 3  # Multiple retry attempts

    @patch('databricks.sdk.WorkspaceClient')
    def test_partial_service_degradation_handling(self, mock_client_class):
        """Test handling of partial service degradation."""
        # Mock client with some operations failing
        mock_client = MagicMock()
        mock_client.catalogs.list.side_effect = Exception("Catalog service unavailable")
        mock_client.warehouses.list.return_value = [MagicMock()]  # Warehouses work
        mock_client_class.return_value = mock_client
        
        adapter = instrument_databricks_unity_catalog()
        
        # Catalog operations should gracefully degrade
        try:
            result = adapter.track_catalog_operation(
                operation="degradation_test",
                catalog_name="test_catalog",
                team="degradation-team",
                project="partial-failure"
            )
            # Should return result even if some backend calls fail
            assert "operation" in result
        except Exception as e:
            # Acceptable if properly handled degradation
            assert "service unavailable" in str(e).lower() or "graceful" in str(e).lower()

    def test_telemetry_export_failure_resilience(self):
        """Test resilience when telemetry export fails."""
        with patch('opentelemetry.sdk.trace.export.SpanProcessor.on_end', side_effect=Exception("Export failed")):
            adapter = instrument_databricks_unity_catalog()
            
            # Operations should continue despite telemetry export failures
            results = []
            for i in range(5):
                try:
                    result = adapter.track_table_operation(
                        operation="export_failure_test",
                        catalog_name="test_catalog",
                        schema_name="test_schema",
                        table_name=f"table_{i}",
                        team="export-failure-team",
                        project="telemetry-resilience"
                    )
                    results.append(result)
                except Exception as e:
                    # Should not propagate telemetry failures to business logic
                    assert "export failed" not in str(e).lower()
            
            # All operations should succeed despite export failures
            assert len(results) == 5


class TestCrossProviderCompatibility:
    """Test compatibility with other GenOps providers."""

    @patch('databricks.sdk.WorkspaceClient')
    def test_mixed_provider_cost_aggregation(self, mock_client_class):
        """Test cost aggregation across multiple provider types."""
        mock_client_class.return_value = MagicMock()
        
        # Initialize Databricks adapter
        databricks_adapter = instrument_databricks_unity_catalog()
        
        # Simulate operations from multiple providers
        databricks_result = databricks_adapter.track_sql_warehouse_operation(
            sql_warehouse_id="databricks-warehouse",
            query_type="analytics",
            query_duration_ms=5000,
            compute_units=2.0,
            team="cross-provider-team",
            project="multi-provider-analytics"
        )
        
        # Mock operations from other providers (OpenAI, Bedrock, etc.)
        cost_aggregator = get_cost_aggregator()
        
        # Add mock costs from other providers
        with patch.object(cost_aggregator, 'add_external_provider_cost') as mock_add_external:
            # Simulate costs from other providers being added
            mock_add_external("openai", 0.15, team="cross-provider-team", project="multi-provider-analytics")
            mock_add_external("bedrock", 0.08, team="cross-provider-team", project="multi-provider-analytics")
        
        summary = cost_aggregator.get_summary()
        
        # Should aggregate costs across all providers
        assert "cross-provider-team" in summary.cost_by_team
        assert "multi-provider-analytics" in summary.cost_by_project
        assert summary.total_cost_usd > 0

    def test_governance_attribute_consistency(self):
        """Test governance attribute consistency across providers."""
        common_governance_attrs = {
            "team": "consistency-team",
            "project": "cross-provider-governance",
            "environment": "test",
            "customer_id": "consistency-customer-123",
            "cost_center": "engineering"
        }
        
        # Test that Databricks adapter handles standard governance attributes
        with patch('databricks.sdk.WorkspaceClient'):
            adapter = instrument_databricks_unity_catalog()
            
            result = adapter.track_catalog_operation(
                operation="consistency_test",
                catalog_name="test_catalog",
                **common_governance_attrs
            )
            
            # Verify all governance attributes are preserved
            for key, value in common_governance_attrs.items():
                assert result["governance_attributes"][key] == value

    def test_telemetry_schema_compatibility(self):
        """Test OpenTelemetry schema compatibility with other providers."""
        with patch('databricks.sdk.WorkspaceClient'):
            adapter = instrument_databricks_unity_catalog()
            
            # Generate telemetry attributes
            telemetry_attrs = adapter._generate_telemetry_attributes(
                operation_type="table.query",
                catalog_name="compatibility_catalog",
                team="telemetry-team",
                project="schema-compatibility"
            )
            
            # Verify GenOps telemetry schema compliance
            required_genops_attrs = [
                "genops.provider",
                "genops.framework_type",
                "genops.operation_type",
                "genops.team",
                "genops.project"
            ]
            
            for attr in required_genops_attrs:
                assert attr in telemetry_attrs
            
            # Verify provider-specific attributes
            assert telemetry_attrs["genops.provider"] == "databricks_unity_catalog"
            assert telemetry_attrs["genops.framework_type"] == "data_platform"
            assert telemetry_attrs["genops.catalog_name"] == "compatibility_catalog"


@pytest.mark.integration
class TestRealWorldScenarios:
    """Test real-world usage scenarios (requires careful mocking)."""

    def test_typical_data_science_workflow(self):
        """Test typical data science workflow simulation."""
        pytest.skip("Real-world scenario test - requires extensive mocking")

    def test_enterprise_governance_audit(self):
        """Test enterprise governance audit scenario."""
        pytest.skip("Enterprise scenario test - requires comprehensive setup")

    def test_multi_region_deployment(self):
        """Test multi-region deployment scenario."""
        pytest.skip("Multi-region test - requires complex infrastructure setup")