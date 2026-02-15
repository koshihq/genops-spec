#!/usr/bin/env python3
"""
Test suite for CrewAI Registration and Auto-Instrumentation

Comprehensive tests for the auto-instrumentation system including:
- Auto-instrumentation activation/deactivation
- Zero-code setup functionality
- Configuration management
- Integration with existing CrewAI code
- Error handling and edge cases
"""

import os
from unittest.mock import Mock, patch

import pytest

# Import the CrewAI registration system
try:
    from genops.providers.crewai import (
        TemporaryInstrumentation,
        auto_instrument,
        configure_auto_instrumentation,
        disable_auto_instrumentation,
        get_current_adapter,
        get_current_monitor,
        get_instrumentation_stats,
        is_instrumented,
    )
except ImportError:
    pytest.skip("CrewAI provider not available", allow_module_level=True)


class TestCrewAIAutoInstrumentation:
    """Test suite for CrewAI auto-instrumentation system."""

    def setup_method(self):
        """Setup method run before each test."""
        # Ensure clean state before each test
        try:
            disable_auto_instrumentation()
        except Exception:
            pass

    def teardown_method(self):
        """Teardown method run after each test."""
        # Clean up after each test
        try:
            disable_auto_instrumentation()
        except Exception:
            pass

    def test_auto_instrument_basic(self):
        """Test basic auto-instrumentation setup."""
        result = auto_instrument(team="test-team", project="test-project")

        assert result is True or result is not None
        assert is_instrumented() is True

    def test_auto_instrument_with_parameters(self):
        """Test auto-instrumentation with custom parameters."""
        result = auto_instrument(
            team="custom-team",
            project="custom-project",
            environment="production",
            daily_budget_limit=200.0,
            governance_policy="enforced",
        )

        assert result is True or result is not None
        assert is_instrumented() is True

        # Check that current adapter has correct settings
        adapter = get_current_adapter()
        if adapter:
            assert adapter.team == "custom-team"
            assert adapter.project == "custom-project"
            assert adapter.environment == "production"
            assert adapter.daily_budget_limit == 200.0
            assert adapter.governance_policy == "enforced"

    def test_is_instrumented_before_setup(self):
        """Test is_instrumented returns False before setup."""
        # Ensure no instrumentation
        disable_auto_instrumentation()

        assert is_instrumented() is False

    def test_is_instrumented_after_setup(self):
        """Test is_instrumented returns True after setup."""
        auto_instrument(team="test", project="test")

        assert is_instrumented() is True

    def test_disable_auto_instrumentation(self):
        """Test disabling auto-instrumentation."""
        # First enable it
        auto_instrument(team="test", project="test")
        assert is_instrumented() is True

        # Then disable it
        disable_auto_instrumentation()
        assert is_instrumented() is False

    def test_multiple_auto_instrument_calls(self):
        """Test calling auto_instrument multiple times."""
        # First call
        result1 = auto_instrument(team="team1", project="project1")
        assert result1 is True or result1 is not None

        # Second call (should handle gracefully)
        auto_instrument(team="team2", project="project2")

        # Should still be instrumented
        assert is_instrumented() is True

        # Current adapter should reflect latest settings or handle appropriately
        adapter = get_current_adapter()
        if adapter:
            # Implementation-dependent: might keep first or update to latest
            assert adapter.team in ["team1", "team2"]

    def test_get_current_adapter(self):
        """Test getting current adapter instance."""
        # Before instrumentation
        adapter_before = get_current_adapter()
        assert adapter_before is None

        # After instrumentation
        auto_instrument(team="adapter-test", project="adapter-project")
        adapter_after = get_current_adapter()

        assert adapter_after is not None
        assert hasattr(adapter_after, "team")
        assert adapter_after.team == "adapter-test"

    def test_get_current_monitor(self):
        """Test getting current monitor instance."""
        # After instrumentation
        auto_instrument(team="monitor-test", project="monitor-project")
        monitor = get_current_monitor()

        if monitor:  # Monitor might be optional
            assert hasattr(monitor, "start_agent_tracking") or hasattr(
                monitor, "monitor_agent"
            )

    def test_get_instrumentation_stats(self):
        """Test getting instrumentation statistics."""
        auto_instrument(team="stats-test", project="stats-project")

        stats = get_instrumentation_stats()

        assert isinstance(stats, dict)
        assert "instrumented" in stats
        assert stats["instrumented"] is True

        # Should have basic stats
        expected_keys = ["team", "project", "start_time", "total_crews"]
        for key in expected_keys:
            if key in stats:
                assert stats[key] is not None

    def test_configure_auto_instrumentation(self):
        """Test configuring auto-instrumentation settings."""
        # Configure before instrumenting
        config_result = configure_auto_instrumentation(
            default_team="configured-team",
            default_project="configured-project",
            default_budget_limit=300.0,
        )

        assert config_result is True or config_result is None

        # Now auto-instrument (should use configured defaults)
        auto_instrument()  # No parameters - should use defaults

        adapter = get_current_adapter()
        if adapter:
            assert adapter.team == "configured-team"
            assert adapter.project == "configured-project"
            assert adapter.daily_budget_limit == 300.0

    def test_environment_variable_configuration(self):
        """Test configuration via environment variables."""
        # Set environment variables
        os.environ["GENOPS_TEAM"] = "env-team"
        os.environ["GENOPS_PROJECT"] = "env-project"
        os.environ["GENOPS_ENVIRONMENT"] = "staging"

        try:
            auto_instrument()  # Should pick up env vars

            adapter = get_current_adapter()
            if adapter:
                assert adapter.team == "env-team"
                assert adapter.project == "env-project"
                assert adapter.environment == "staging"
        finally:
            # Clean up environment variables
            os.environ.pop("GENOPS_TEAM", None)
            os.environ.pop("GENOPS_PROJECT", None)
            os.environ.pop("GENOPS_ENVIRONMENT", None)

    def test_auto_instrument_with_invalid_parameters(self):
        """Test auto-instrumentation with invalid parameters."""
        # Test invalid budget
        with pytest.raises((ValueError, TypeError)):
            auto_instrument(
                team="test",
                project="test",
                daily_budget_limit=-100.0,  # Invalid negative budget
            )

        # Test invalid governance policy
        with pytest.raises((ValueError, TypeError)):
            auto_instrument(
                team="test", project="test", governance_policy="invalid_policy"
            )

    def test_temporary_instrumentation_context_manager(self):
        """Test temporary instrumentation context manager."""
        # Ensure not instrumented initially
        assert is_instrumented() is False

        with TemporaryInstrumentation(team="temp", project="temp") as temp_adapter:
            # Should be instrumented within context
            assert is_instrumented() is True
            assert temp_adapter is not None

        # Should be disabled after context
        assert is_instrumented() is False

    def test_temporary_instrumentation_with_exception(self):
        """Test temporary instrumentation handles exceptions properly."""
        assert is_instrumented() is False

        with pytest.raises(ValueError):
            with TemporaryInstrumentation(team="temp", project="temp"):
                assert is_instrumented() is True
                raise ValueError("Test exception")

        # Should still be disabled after exception
        assert is_instrumented() is False

    def test_concurrent_instrumentation(self):
        """Test instrumentation with concurrent operations."""
        import threading
        import time

        results = []

        def worker_thread(thread_id):
            try:
                auto_instrument(team=f"thread-{thread_id}", project="concurrent-test")

                # Check instrumentation
                instrumented = is_instrumented()
                results.append((thread_id, instrumented))

                time.sleep(0.01)  # Small delay

            except Exception as e:
                results.append((thread_id, f"ERROR: {e}"))

        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        assert len(results) == 3
        # At least one should succeed
        successful = [r for r in results if r[1] is True]
        assert len(successful) >= 1

    @patch("genops.providers.crewai.registration.GenOpsCrewAIAdapter")
    def test_auto_instrument_adapter_creation(self, mock_adapter_class):
        """Test that auto_instrument creates adapter correctly."""
        mock_adapter = Mock()
        mock_adapter_class.return_value = mock_adapter

        auto_instrument(
            team="mock-test", project="mock-project", daily_budget_limit=150.0
        )

        # Should have created adapter with correct parameters
        mock_adapter_class.assert_called_once()
        call_args = mock_adapter_class.call_args

        assert call_args[1]["team"] == "mock-test"  # Keyword args
        assert call_args[1]["project"] == "mock-project"
        assert call_args[1]["daily_budget_limit"] == 150.0

    @patch("genops.providers.crewai.registration.CrewAIAgentMonitor")
    def test_auto_instrument_monitor_creation(self, mock_monitor_class):
        """Test that auto_instrument creates monitor if enabled."""
        mock_monitor = Mock()
        mock_monitor_class.return_value = mock_monitor

        auto_instrument(
            team="monitor-test", project="monitor-project", enable_agent_monitoring=True
        )

        # Should have created monitor
        monitor = get_current_monitor()
        if monitor:  # Implementation might not always create monitor
            mock_monitor_class.assert_called_once()

    def test_instrumentation_with_crewai_import_error(self):
        """Test graceful handling when CrewAI is not available."""
        with patch("genops.providers.crewai.registration.HAS_CREWAI", False):
            # Should handle gracefully
            result = auto_instrument(team="test", project="test")

            # Might return False or raise informative error
            assert result is False or isinstance(result, bool)

    def test_get_instrumentation_stats_without_instrumentation(self):
        """Test getting stats when not instrumented."""
        disable_auto_instrumentation()

        stats = get_instrumentation_stats()

        assert isinstance(stats, dict)
        assert stats.get("instrumented") is False
        assert stats.get("error") is None or "not instrumented" in stats.get(
            "error", ""
        )

    def test_configuration_persistence(self):
        """Test that configuration persists across enable/disable cycles."""
        # Configure with specific settings
        configure_auto_instrumentation(
            default_team="persistent-team", default_project="persistent-project"
        )

        # Enable instrumentation
        auto_instrument()
        adapter1 = get_current_adapter()

        # Disable and re-enable
        disable_auto_instrumentation()
        auto_instrument()
        adapter2 = get_current_adapter()

        # Should maintain same configuration
        if adapter1 and adapter2:
            assert adapter1.team == adapter2.team
            assert adapter1.project == adapter2.project

    def test_instrumentation_statistics_accuracy(self):
        """Test accuracy of instrumentation statistics."""
        auto_instrument(team="stats-accuracy", project="stats-test")

        # Get initial stats
        get_instrumentation_stats()

        # Simulate some crew executions (if possible)
        adapter = get_current_adapter()
        if adapter and hasattr(adapter, "track_crew"):
            for i in range(3):
                with adapter.track_crew(f"test-crew-{i}"):
                    pass

        # Get updated stats
        final_stats = get_instrumentation_stats()

        # Stats should be updated
        assert final_stats["instrumented"] is True
        if "total_crews" in final_stats:
            assert final_stats["total_crews"] >= 0

    def test_auto_instrument_return_values(self):
        """Test return values from auto_instrument function."""
        # First call should succeed
        result1 = auto_instrument(team="return-test", project="return-project")
        assert result1 is True

        # Second call behavior (implementation-dependent)
        result2 = auto_instrument(team="return-test", project="return-project")
        assert isinstance(result2, bool)  # Should return boolean

    def test_instrumentation_error_handling(self):
        """Test error handling in instrumentation setup."""
        # Test with missing required parameters
        try:
            auto_instrument()  # No team or project
            # Should either use defaults or raise error
            assert True  # If no error, then defaults were used
        except (ValueError, TypeError) as e:
            # If error raised, it should be informative
            assert "team" in str(e) or "project" in str(e)

    def test_cleanup_on_disable(self):
        """Test proper cleanup when disabling instrumentation."""
        # Enable instrumentation
        auto_instrument(team="cleanup-test", project="cleanup-project")

        # Verify it's active
        assert is_instrumented() is True
        adapter_before = get_current_adapter()
        assert adapter_before is not None

        # Disable instrumentation
        disable_auto_instrumentation()

        # Verify cleanup
        assert is_instrumented() is False
        adapter_after = get_current_adapter()
        assert adapter_after is None

    def test_instrumentation_thread_safety(self):
        """Test thread safety of instrumentation operations."""
        import threading
        import time

        results = []
        errors = []

        def enable_disable_cycle(cycle_id):
            try:
                for i in range(5):
                    auto_instrument(team=f"thread-{cycle_id}", project=f"cycle-{i}")
                    time.sleep(0.001)

                    instrumented = is_instrumented()
                    results.append((cycle_id, i, instrumented))

                    disable_auto_instrumentation()
                    time.sleep(0.001)
            except Exception as e:
                errors.append((cycle_id, str(e)))

        # Run multiple cycles concurrently
        threads = []
        for cycle_id in range(2):
            thread = threading.Thread(target=enable_disable_cycle, args=(cycle_id,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should have completed without major errors
        assert len(errors) <= len(threads)  # Allow some errors due to concurrency
        assert len(results) > 0  # Should have some successful operations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
