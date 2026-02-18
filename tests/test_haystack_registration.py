#!/usr/bin/env python3
"""
Comprehensive test suite for Haystack auto-instrumentation registration.

Tests cover auto-instrumentation functionality, monkey patching, component registration,
and temporary instrumentation scenarios as required by CLAUDE.md standards.
"""

from unittest.mock import Mock, patch

from genops.providers.haystack_registration import (
    TemporaryInstrumentation,
    auto_instrument,
    configure_auto_instrumentation,
    disable_auto_instrumentation,
    get_cost_summary,
    get_current_adapter,
    get_current_monitor,
    get_execution_metrics,
    get_instrumentation_stats,
    is_instrumented,
)


class TestAutoInstrumentation:
    """Auto-instrumentation core functionality tests."""

    def setup_method(self):
        """Setup for each test - reset global state."""
        # Reset global variables
        import genops.providers.haystack_registration as reg_module

        reg_module._global_adapter = None
        reg_module._global_monitor = None
        reg_module._instrumentation_active = False

    def test_auto_instrument_basic_setup(self):
        """Test basic auto-instrumentation setup."""
        success = auto_instrument(team="test-team", project="test-project")

        assert success is True
        assert is_instrumented() is True

        adapter = get_current_adapter()
        assert adapter is not None
        assert adapter.team == "test-team"
        assert adapter.project == "test-project"

    def test_auto_instrument_with_all_parameters(self):
        """Test auto-instrumentation with all parameters."""
        success = auto_instrument(
            team="advanced-team",
            project="advanced-project",
            environment="production",
            daily_budget_limit=500.0,
            monthly_budget_limit=10000.0,
            governance_policy="enforcing",
            enable_cost_alerts=True,
        )

        assert success is True

        adapter = get_current_adapter()
        assert adapter.team == "advanced-team"
        assert adapter.project == "advanced-project"
        assert adapter.environment == "production"
        assert float(adapter.daily_budget_limit) == 500.0
        assert adapter.governance_policy == "enforcing"

    def test_auto_instrument_duplicate_calls(self):
        """Test auto-instrumentation handles duplicate calls gracefully."""
        # First call
        success1 = auto_instrument(team="team1", project="project1")
        assert success1 is True

        adapter1 = get_current_adapter()
        assert adapter1.team == "team1"

        # Second call with different parameters
        success2 = auto_instrument(team="team2", project="project2")
        assert success2 is True

        adapter2 = get_current_adapter()
        # Should use new configuration
        assert adapter2.team == "team2"

    @patch("genops.providers.haystack_registration.HAS_HAYSTACK", False)
    def test_auto_instrument_without_haystack(self):
        """Test auto-instrumentation when Haystack is not available."""
        success = auto_instrument(team="test-team", project="test-project")

        # Should still succeed but with limited functionality
        assert success is True
        assert is_instrumented() is True

    def test_auto_instrument_invalid_parameters(self):
        """Test auto-instrumentation with invalid parameters."""
        success = auto_instrument(
            team="test-team", project="test-project", governance_policy="invalid-policy"
        )

        # Should handle invalid parameters gracefully
        assert success is False
        assert is_instrumented() is False


class TestInstrumentationManagement:
    """Instrumentation lifecycle management tests."""

    def setup_method(self):
        """Setup for each test."""
        import genops.providers.haystack_registration as reg_module

        reg_module._global_adapter = None
        reg_module._global_monitor = None
        reg_module._instrumentation_active = False

    def test_disable_auto_instrumentation(self):
        """Test disabling auto-instrumentation."""
        # Enable first
        auto_instrument(team="test-team", project="test-project")
        assert is_instrumented() is True

        # Disable
        disable_auto_instrumentation()
        assert is_instrumented() is False

        adapter = get_current_adapter()
        assert adapter is None

    def test_configure_auto_instrumentation(self):
        """Test configuring auto-instrumentation."""
        # Enable with basic config
        auto_instrument(team="initial-team", project="initial-project")

        # Reconfigure
        success = configure_auto_instrumentation(
            team="configured-team",
            project="configured-project",
            daily_budget_limit=250.0,
            governance_policy="advisory",
        )

        assert success is True

        adapter = get_current_adapter()
        assert adapter.team == "configured-team"
        assert adapter.project == "configured-project"
        assert float(adapter.daily_budget_limit) == 250.0

    def test_configure_without_active_instrumentation(self):
        """Test configuring when instrumentation is not active."""
        success = configure_auto_instrumentation(
            team="config-team", project="config-project"
        )

        # Should fail if no active instrumentation
        assert success is False

    def test_instrumentation_status_check(self):
        """Test instrumentation status checking."""
        # Initially not instrumented
        assert is_instrumented() is False

        # Enable instrumentation
        auto_instrument(team="test-team", project="test-project")
        assert is_instrumented() is True

        # Disable instrumentation
        disable_auto_instrumentation()
        assert is_instrumented() is False


class TestInstrumentationStats:
    """Instrumentation statistics tests."""

    def setup_method(self):
        """Setup for each test."""
        import genops.providers.haystack_registration as reg_module

        reg_module._global_adapter = None
        reg_module._global_monitor = None
        reg_module._instrumentation_active = False

    def test_get_instrumentation_stats_active(self):
        """Test getting instrumentation stats when active."""
        auto_instrument(team="stats-team", project="stats-project")

        stats = get_instrumentation_stats()

        assert isinstance(stats, dict)
        assert "active" in stats
        assert "team" in stats
        assert "project" in stats
        assert stats["active"] is True
        assert stats["team"] == "stats-team"
        assert stats["project"] == "stats-project"

    def test_get_instrumentation_stats_inactive(self):
        """Test getting instrumentation stats when inactive."""
        stats = get_instrumentation_stats()

        assert isinstance(stats, dict)
        assert stats["active"] is False

    def test_get_current_adapter_active(self):
        """Test getting current adapter when active."""
        auto_instrument(team="adapter-team", project="adapter-project")

        adapter = get_current_adapter()
        assert adapter is not None
        assert adapter.team == "adapter-team"
        assert adapter.project == "adapter-project"

    def test_get_current_adapter_inactive(self):
        """Test getting current adapter when inactive."""
        adapter = get_current_adapter()
        assert adapter is None

    def test_get_current_monitor_active(self):
        """Test getting current monitor when active."""
        auto_instrument(team="monitor-team", project="monitor-project")

        monitor = get_current_monitor()
        assert monitor is not None
        assert monitor.team == "monitor-team"
        assert monitor.project == "monitor-project"

    def test_get_current_monitor_inactive(self):
        """Test getting current monitor when inactive."""
        monitor = get_current_monitor()
        assert monitor is None


class TestCostAndMetricsIntegration:
    """Cost tracking and metrics integration tests."""

    def setup_method(self):
        """Setup for each test."""
        import genops.providers.haystack_registration as reg_module

        reg_module._global_adapter = None
        reg_module._global_monitor = None
        reg_module._instrumentation_active = False

    def test_get_cost_summary_active(self):
        """Test getting cost summary when instrumentation is active."""
        auto_instrument(team="cost-team", project="cost-project")

        # Mock the adapter's get_cost_summary method
        adapter = get_current_adapter()
        adapter.get_cost_summary = Mock(
            return_value={
                "daily_costs": 0.025,
                "daily_budget_utilization": 25.0,
                "total_operations": 10,
            }
        )

        summary = get_cost_summary()

        assert isinstance(summary, dict)
        assert "daily_costs" in summary
        assert summary["daily_costs"] == 0.025

    def test_get_cost_summary_inactive(self):
        """Test getting cost summary when instrumentation is inactive."""
        summary = get_cost_summary()

        assert isinstance(summary, dict)
        assert "error" in summary
        assert "not active" in summary["error"]

    def test_get_execution_metrics_active(self):
        """Test getting execution metrics when active."""
        auto_instrument(team="metrics-team", project="metrics-project")

        # Mock the adapter's get_execution_metrics method
        adapter = get_current_adapter()
        adapter.get_execution_metrics = Mock(
            return_value={
                "total_executions": 5,
                "avg_execution_time": 2.5,
                "success_rate": 1.0,
            }
        )

        metrics = get_execution_metrics()

        assert isinstance(metrics, dict)
        assert "total_executions" in metrics
        assert metrics["total_executions"] == 5

    def test_get_execution_metrics_inactive(self):
        """Test getting execution metrics when inactive."""
        metrics = get_execution_metrics()

        assert isinstance(metrics, dict)
        assert "error" in metrics
        assert "not active" in metrics["error"]


class TestTemporaryInstrumentation:
    """Temporary instrumentation context manager tests."""

    def setup_method(self):
        """Setup for each test."""
        import genops.providers.haystack_registration as reg_module

        reg_module._global_adapter = None
        reg_module._global_monitor = None
        reg_module._instrumentation_active = False

    def test_temporary_instrumentation_basic(self):
        """Test basic temporary instrumentation."""
        # Initially not instrumented
        assert is_instrumented() is False

        with TemporaryInstrumentation(team="temp-team", project="temp-project"):
            # Should be instrumented inside context
            assert is_instrumented() is True

            adapter = get_current_adapter()
            assert adapter is not None
            assert adapter.team == "temp-team"
            assert adapter.project == "temp-project"

        # Should be disabled after context
        assert is_instrumented() is False
        assert get_current_adapter() is None

    def test_temporary_instrumentation_with_existing(self):
        """Test temporary instrumentation when already instrumented."""
        # Enable global instrumentation
        auto_instrument(team="global-team", project="global-project")
        assert is_instrumented() is True

        get_current_adapter()

        with TemporaryInstrumentation(team="temp-team", project="temp-project"):
            # Should use temporary configuration
            temp_adapter = get_current_adapter()
            assert temp_adapter.team == "temp-team"
            assert temp_adapter.project == "temp-project"

        # Should restore original configuration
        restored_adapter = get_current_adapter()
        assert restored_adapter.team == "global-team"
        assert restored_adapter.project == "global-project"

    def test_temporary_instrumentation_exception_handling(self):
        """Test temporary instrumentation handles exceptions properly."""
        assert is_instrumented() is False

        try:
            with TemporaryInstrumentation(team="temp-team", project="temp-project"):
                assert is_instrumented() is True
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected exception

        # Should still restore state after exception
        assert is_instrumented() is False

    def test_temporary_instrumentation_nested(self):
        """Test nested temporary instrumentation."""
        with TemporaryInstrumentation(team="outer-team", project="outer-project"):
            assert get_current_adapter().team == "outer-team"

            with TemporaryInstrumentation(team="inner-team", project="inner-project"):
                assert get_current_adapter().team == "inner-team"

            # Should restore outer context
            assert get_current_adapter().team == "outer-team"

        # Should be completely disabled
        assert is_instrumented() is False


class TestMonkeyPatchingIntegration:
    """Monkey patching integration tests."""

    def setup_method(self):
        """Setup for each test."""
        import genops.providers.haystack_registration as reg_module

        reg_module._global_adapter = None
        reg_module._global_monitor = None
        reg_module._instrumentation_active = False

    @patch("genops.providers.haystack_registration.HAS_HAYSTACK", True)
    @patch("genops.providers.haystack_registration._patch_haystack_components")
    def test_auto_instrument_applies_patches(self, mock_patch_components):
        """Test auto-instrumentation applies component patches."""
        mock_patch_components.return_value = True

        success = auto_instrument(team="patch-team", project="patch-project")

        assert success is True
        mock_patch_components.assert_called_once()

    @patch("genops.providers.haystack_registration.HAS_HAYSTACK", True)
    @patch("genops.providers.haystack_registration._unpatch_haystack_components")
    def test_disable_removes_patches(self, mock_unpatch_components):
        """Test disabling auto-instrumentation removes patches."""
        # Enable first (mocked)
        auto_instrument(team="patch-team", project="patch-project")

        # Disable
        disable_auto_instrumentation()

        mock_unpatch_components.assert_called_once()

    @patch("genops.providers.haystack_registration.HAS_HAYSTACK", True)
    def test_patch_component_registration(self):
        """Test component registration through patches."""
        # This would test the actual patching logic if we had access to Haystack components
        # For now, we test that the patching functions exist and can be called

        from genops.providers.haystack_registration import (
            _patch_haystack_components,
            _unpatch_haystack_components,
        )

        # Functions should exist and be callable
        assert callable(_patch_haystack_components)
        assert callable(_unpatch_haystack_components)


class TestErrorHandlingAndEdgeCases:
    """Error handling and edge case tests."""

    def setup_method(self):
        """Setup for each test."""
        import genops.providers.haystack_registration as reg_module

        reg_module._global_adapter = None
        reg_module._global_monitor = None
        reg_module._instrumentation_active = False

    def test_auto_instrument_adapter_creation_failure(self):
        """Test auto-instrumentation handles adapter creation failure."""
        with patch(
            "genops.providers.haystack_registration.GenOpsHaystackAdapter"
        ) as mock_adapter_class:
            mock_adapter_class.side_effect = Exception("Adapter creation failed")

            success = auto_instrument(team="fail-team", project="fail-project")

            assert success is False
            assert is_instrumented() is False

    def test_auto_instrument_monitor_creation_failure(self):
        """Test auto-instrumentation handles monitor creation failure."""
        with patch(
            "genops.providers.haystack_registration.HaystackMonitor"
        ) as mock_monitor_class:
            mock_monitor_class.side_effect = Exception("Monitor creation failed")

            success = auto_instrument(team="fail-team", project="fail-project")

            # Should still succeed with adapter only
            assert success is True
            assert is_instrumented() is True
            assert (
                get_current_monitor() is not None
            )  # Fallback monitor should be created

    def test_configure_with_invalid_parameters(self):
        """Test configuration with invalid parameters."""
        auto_instrument(team="base-team", project="base-project")

        success = configure_auto_instrumentation(
            team="new-team",
            project="new-project",
            daily_budget_limit=-100.0,  # Invalid negative budget
        )

        assert success is False

        # Original configuration should remain
        adapter = get_current_adapter()
        assert adapter.team == "base-team"

    def test_get_cost_summary_adapter_error(self):
        """Test cost summary when adapter method fails."""
        auto_instrument(team="error-team", project="error-project")

        adapter = get_current_adapter()
        adapter.get_cost_summary = Mock(
            side_effect=Exception("Cost calculation failed")
        )

        summary = get_cost_summary()

        assert isinstance(summary, dict)
        assert "error" in summary
        assert "failed to retrieve" in summary["error"].lower()

    def test_get_execution_metrics_monitor_error(self):
        """Test execution metrics when monitor method fails."""
        auto_instrument(team="error-team", project="error-project")

        adapter = get_current_adapter()
        adapter.get_execution_metrics = Mock(
            side_effect=Exception("Metrics calculation failed")
        )

        metrics = get_execution_metrics()

        assert isinstance(metrics, dict)
        assert "error" in metrics
        assert "failed to retrieve" in metrics["error"].lower()

    def test_temporary_instrumentation_creation_failure(self):
        """Test temporary instrumentation handles creation failures."""
        with patch(
            "genops.providers.haystack_registration.GenOpsHaystackAdapter"
        ) as mock_adapter_class:
            mock_adapter_class.side_effect = Exception(
                "Temporary adapter creation failed"
            )

            try:
                with TemporaryInstrumentation(team="fail-temp", project="fail-temp"):
                    pass  # Should not reach here
                assert False, "Should have raised exception"
            except Exception:
                pass  # Expected

            # Should still be not instrumented
            assert is_instrumented() is False


class TestComponentLifecycleIntegration:
    """Component lifecycle integration tests."""

    def setup_method(self):
        """Setup for each test."""
        import genops.providers.haystack_registration as reg_module

        reg_module._global_adapter = None
        reg_module._global_monitor = None
        reg_module._instrumentation_active = False

    def test_instrumentation_with_component_tracking(self):
        """Test instrumentation enables component tracking."""
        auto_instrument(
            team="tracking-team",
            project="tracking-project",
            enable_component_tracking=True,
        )

        adapter = get_current_adapter()
        assert adapter.enable_component_tracking is True

    def test_instrumentation_without_component_tracking(self):
        """Test instrumentation can disable component tracking."""
        auto_instrument(
            team="no-tracking-team",
            project="no-tracking-project",
            enable_component_tracking=False,
        )

        adapter = get_current_adapter()
        assert adapter.enable_component_tracking is False

    def test_instrumentation_cost_alerts_configuration(self):
        """Test instrumentation cost alerts configuration."""
        auto_instrument(
            team="alerts-team", project="alerts-project", enable_cost_alerts=True
        )

        adapter = get_current_adapter()
        assert adapter.enable_cost_alerts is True
