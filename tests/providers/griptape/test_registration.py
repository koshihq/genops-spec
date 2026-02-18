#!/usr/bin/env python3
"""
Test suite for Griptape Auto-Instrumentation Registration

Tests auto-instrumentation functionality, import hooks, class wrapping,
and instrumentation management for Griptape framework integration.
"""

import sys
from unittest.mock import Mock, patch

import pytest

from genops.providers.griptape.registration import (
    _detect_griptape_version,
    _instrumentation_registry,
    _is_griptape_available,
    auto_instrument,
    disable_auto_instrument,
    get_instrumentation_adapter,
    instrument_griptape,
    is_instrumented,
    validate_griptape_setup,
)


class TestGriptapeDetection:
    """Test Griptape framework detection utilities."""

    @patch.dict(sys.modules, {"griptape": Mock(__version__="1.0.0")})
    def test_is_griptape_available_true(self):
        """Test Griptape detection when available."""
        assert _is_griptape_available() is True

    def test_is_griptape_available_false(self):
        """Test Griptape detection when not available."""
        # Remove griptape from sys.modules if present
        griptape_module = sys.modules.pop("griptape", None)

        try:
            with patch.dict(sys.modules, {}, clear=False):
                # Import should fail
                result = _is_griptape_available()
                assert result is False
        finally:
            # Restore module if it was there
            if griptape_module:
                sys.modules["griptape"] = griptape_module

    @patch.dict(sys.modules, {"griptape": Mock(__version__="1.2.3")})
    def test_detect_griptape_version(self):
        """Test Griptape version detection."""
        version = _detect_griptape_version()
        assert version == "1.2.3"

    @patch.dict(sys.modules, {"griptape": Mock(spec=[])})  # No __version__ attribute
    def test_detect_griptape_version_unknown(self):
        """Test version detection when version is unknown."""
        version = _detect_griptape_version()
        assert version == "unknown"

    def test_detect_griptape_version_not_available(self):
        """Test version detection when Griptape not available."""
        griptape_module = sys.modules.pop("griptape", None)

        try:
            with patch.dict(sys.modules, {}, clear=False):
                version = _detect_griptape_version()
                assert version is None
        finally:
            if griptape_module:
                sys.modules["griptape"] = griptape_module


class TestInstrumentationRegistry:
    """Test instrumentation registry management."""

    def setup_method(self):
        """Reset registry state before each test."""
        with _instrumentation_registry["lock"]:
            _instrumentation_registry["enabled"] = False
            _instrumentation_registry["adapter"] = None
            _instrumentation_registry["original_classes"] = {}
            _instrumentation_registry["wrapped_classes"] = {}

    def test_is_instrumented_false(self):
        """Test instrumentation status when disabled."""
        assert is_instrumented() is False

    def test_get_instrumentation_adapter_none(self):
        """Test getting adapter when not instrumented."""
        adapter = get_instrumentation_adapter()
        assert adapter is None


class TestAutoInstrumentation:
    """Test auto-instrumentation functionality."""

    def setup_method(self):
        """Reset instrumentation state."""
        try:
            disable_auto_instrument()
        except Exception:
            pass

    def teardown_method(self):
        """Clean up instrumentation state."""
        try:
            disable_auto_instrument()
        except Exception:
            pass

    @patch(
        "genops.providers.griptape.registration._is_griptape_available",
        return_value=False,
    )
    def test_auto_instrument_griptape_not_available(self, mock_available):
        """Test auto-instrumentation when Griptape is not available."""
        with pytest.raises(ImportError, match="Griptape framework not found"):
            auto_instrument(team="test-team")

    @patch(
        "genops.providers.griptape.registration._is_griptape_available",
        return_value=True,
    )
    @patch("genops.providers.griptape.registration._apply_instrumentation")
    def test_auto_instrument_success(self, mock_apply, mock_available):
        """Test successful auto-instrumentation."""
        adapter = auto_instrument(
            team="test-team", project="test-project", enable_cost_tracking=True
        )

        assert adapter is not None
        assert adapter.governance_attrs.team == "test-team"
        assert adapter.governance_attrs.project == "test-project"
        assert is_instrumented() is True

        # Check adapter is stored in registry
        stored_adapter = get_instrumentation_adapter()
        assert stored_adapter is adapter

    @patch(
        "genops.providers.griptape.registration._is_griptape_available",
        return_value=True,
    )
    @patch("genops.providers.griptape.registration._apply_instrumentation")
    def test_auto_instrument_already_enabled(self, mock_apply, mock_available):
        """Test auto-instrumentation when already enabled."""
        # Enable first time
        adapter1 = auto_instrument(team="team1")

        # Try to enable again
        adapter2 = auto_instrument(team="team2")

        # Should return the same adapter
        assert adapter1 is adapter2
        assert adapter1.governance_attrs.team == "team1"  # Original team preserved

    @patch(
        "genops.providers.griptape.registration._is_griptape_available",
        return_value=True,
    )
    @patch("genops.providers.griptape.registration._apply_instrumentation")
    def test_disable_auto_instrument(self, mock_apply, mock_available):
        """Test disabling auto-instrumentation."""
        # Enable instrumentation
        auto_instrument(team="test-team")
        assert is_instrumented() is True

        # Disable instrumentation
        disable_auto_instrument()
        assert is_instrumented() is False
        assert get_instrumentation_adapter() is None

    def test_disable_auto_instrument_not_enabled(self):
        """Test disabling when not enabled."""
        assert is_instrumented() is False

        # Should not raise exception
        disable_auto_instrument()
        assert is_instrumented() is False

    @patch(
        "genops.providers.griptape.registration._is_griptape_available",
        return_value=True,
    )
    @patch(
        "genops.providers.griptape.registration._apply_instrumentation",
        side_effect=Exception("Apply failed"),
    )
    def test_auto_instrument_apply_failure(self, mock_apply, mock_available):
        """Test auto-instrumentation when apply fails."""
        with pytest.raises(Exception, match="Apply failed"):
            auto_instrument(team="test-team")

        # Should not be marked as instrumented
        assert is_instrumented() is False


class TestManualInstrumentation:
    """Test manual instrumentation wrapper."""

    @patch(
        "genops.providers.griptape.registration._is_griptape_available",
        return_value=False,
    )
    def test_instrument_griptape_not_available(self, mock_available):
        """Test manual instrumentation when Griptape not available."""
        with pytest.raises(ImportError, match="Griptape framework not available"):
            instrument_griptape(team="test-team")

    @patch(
        "genops.providers.griptape.registration._is_griptape_available",
        return_value=True,
    )
    @patch("griptape.structures.Agent")
    @patch("griptape.structures.Pipeline")
    @patch("griptape.structures.Workflow")
    def test_instrument_griptape_success(
        self, mock_workflow, mock_pipeline, mock_agent, mock_available
    ):
        """Test successful manual instrumentation."""
        # Mock Griptape structures
        mock_agent.__name__ = "Agent"
        mock_pipeline.__name__ = "Pipeline"
        mock_workflow.__name__ = "Workflow"

        with (
            patch("griptape.structures.Agent", mock_agent),
            patch("griptape.structures.Pipeline", mock_pipeline),
            patch("griptape.structures.Workflow", mock_workflow),
        ):
            instrumented = instrument_griptape(
                team="test-team", project="test-project", daily_budget_limit=100.0
            )

            assert instrumented is not None
            assert instrumented.adapter.governance_attrs.team == "test-team"
            assert instrumented.adapter.governance_attrs.project == "test-project"
            assert instrumented.adapter.daily_budget_limit == 100.0


class TestClassWrapping:
    """Test Griptape class wrapping functionality."""

    @patch(
        "genops.providers.griptape.registration._is_griptape_available",
        return_value=True,
    )
    def test_wrap_structure_class(self, mock_available):
        """Test wrapping of Griptape structure classes."""
        from genops.providers.griptape.adapter import GenOpsGriptapeAdapter
        from genops.providers.griptape.registration import _wrap_structure_class

        # Create mock original class
        class MockAgent:
            def __init__(self, *args, **kwargs):
                self.id = "test-agent"

            def run(self, *args, **kwargs):
                return "original result"

        # Create adapter
        adapter = GenOpsGriptapeAdapter(team="test-team")

        # Wrap the class
        WrappedAgent = _wrap_structure_class(MockAgent, "agent", adapter)

        # Test wrapped class
        wrapped_instance = WrappedAgent()
        assert hasattr(wrapped_instance, "_genops_adapter")
        assert wrapped_instance._genops_adapter is adapter
        assert wrapped_instance._genops_structure_type == "agent"

        # Test that original functionality is preserved
        assert wrapped_instance.id == "test-agent"

    @patch(
        "genops.providers.griptape.registration._is_griptape_available",
        return_value=True,
    )
    def test_wrap_structure_method(self, mock_available):
        """Test wrapping of structure methods."""
        from genops.providers.griptape.adapter import GenOpsGriptapeAdapter
        from genops.providers.griptape.registration import _wrap_structure_method

        # Create mock method
        def original_method(self, *args, **kwargs):
            return "method result"

        # Create adapter
        adapter = GenOpsGriptapeAdapter(team="test-team")

        # Mock the track_agent context manager
        mock_request = Mock()
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_request)
        mock_context.__exit__ = Mock(return_value=None)

        with patch.object(adapter, "track_agent", return_value=mock_context):
            # Wrap the method
            wrapped_method = _wrap_structure_method(
                original_method, "agent", "run", adapter
            )

            # Create mock self object
            mock_self = Mock()
            mock_self.id = "test-agent-123"

            # Call wrapped method
            result = wrapped_method(mock_self, "test_arg")

            # Verify tracking was called
            adapter.track_agent.assert_called_once()

            # Verify original method behavior preserved
            assert result == "method result"


class TestValidationFunctionality:
    """Test Griptape setup validation."""

    @patch(
        "genops.providers.griptape.registration._is_griptape_available",
        return_value=True,
    )
    @patch(
        "genops.providers.griptape.registration._detect_griptape_version",
        return_value="1.0.0",
    )
    @patch("griptape.structures.Agent")
    @patch("griptape.structures.Pipeline")
    @patch("griptape.structures.Workflow")
    def test_validate_griptape_setup_success(
        self, mock_workflow, mock_pipeline, mock_agent, mock_version, mock_available
    ):
        """Test successful Griptape setup validation."""
        result = validate_griptape_setup()

        assert result["griptape_available"] is True
        assert result["griptape_version"] == "1.0.0"
        assert "Agent" in result["supported_structures"]
        assert "Pipeline" in result["supported_structures"]
        assert "Workflow" in result["supported_structures"]
        assert len(result["issues"]) == 0

    @patch(
        "genops.providers.griptape.registration._is_griptape_available",
        return_value=False,
    )
    def test_validate_griptape_setup_not_available(self, mock_available):
        """Test validation when Griptape is not available."""
        result = validate_griptape_setup()

        assert result["griptape_available"] is False
        assert result["griptape_version"] is None
        assert len(result["supported_structures"]) == 0
        assert "Griptape framework not installed" in result["issues"]
        assert "Install Griptape: pip install griptape" in result["recommendations"]

    @patch(
        "genops.providers.griptape.registration._is_griptape_available",
        return_value=True,
    )
    @patch(
        "genops.providers.griptape.registration._detect_griptape_version",
        return_value="unknown",
    )
    @patch("griptape.structures.Agent")
    def test_validate_griptape_setup_unknown_version(
        self, mock_agent, mock_version, mock_available
    ):
        """Test validation with unknown version."""
        result = validate_griptape_setup()

        assert result["griptape_available"] is True
        assert result["griptape_version"] == "unknown"
        assert "Cannot determine Griptape version" in result["issues"]

    @patch(
        "genops.providers.griptape.registration._is_griptape_available",
        return_value=True,
    )
    @patch(
        "genops.providers.griptape.registration._detect_griptape_version",
        return_value="1.0.0",
    )
    @patch("griptape.structures.Agent", side_effect=ImportError("Failed to import"))
    def test_validate_griptape_setup_import_failure(
        self, mock_agent, mock_version, mock_available
    ):
        """Test validation when structure import fails."""
        result = validate_griptape_setup()

        assert result["griptape_available"] is True
        assert "Failed to import core structures" in str(result["issues"])

    @patch(
        "genops.providers.griptape.registration._is_griptape_available",
        return_value=True,
    )
    @patch("genops.providers.griptape.registration.is_instrumented", return_value=True)
    def test_validate_griptape_setup_with_instrumentation(
        self, mock_instrumented, mock_available
    ):
        """Test validation when instrumentation is enabled."""
        result = validate_griptape_setup()

        assert result["instrumentation_enabled"] is True


class TestIntegrationScenarios:
    """Test complex integration scenarios."""

    def setup_method(self):
        """Reset state before each test."""
        try:
            disable_auto_instrument()
        except Exception:
            pass

    def teardown_method(self):
        """Clean up after each test."""
        try:
            disable_auto_instrument()
        except Exception:
            pass

    @patch(
        "genops.providers.griptape.registration._is_griptape_available",
        return_value=True,
    )
    @patch("genops.providers.griptape.registration._apply_instrumentation")
    @patch("genops.providers.griptape.registration._remove_instrumentation")
    def test_multiple_enable_disable_cycles(
        self, mock_remove, mock_apply, mock_available
    ):
        """Test multiple instrumentation enable/disable cycles."""
        # Cycle 1
        auto_instrument(team="team1")
        assert is_instrumented() is True

        disable_auto_instrument()
        assert is_instrumented() is False

        # Cycle 2
        adapter2 = auto_instrument(team="team2")
        assert is_instrumented() is True
        assert adapter2.governance_attrs.team == "team2"

        disable_auto_instrument()
        assert is_instrumented() is False

        # Check methods were called appropriately
        assert mock_apply.call_count == 2
        assert mock_remove.call_count == 2

    @patch(
        "genops.providers.griptape.registration._is_griptape_available",
        return_value=True,
    )
    @patch("genops.providers.griptape.registration._apply_instrumentation")
    def test_concurrent_instrumentation_attempts(self, mock_apply, mock_available):
        """Test concurrent instrumentation attempts."""
        import threading

        results = []
        errors = []

        def try_instrument(thread_id):
            try:
                adapter = auto_instrument(team=f"team-{thread_id}")
                results.append(adapter)
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=try_instrument, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Should have 5 results (some may be the same adapter due to already-enabled logic)
        assert len(results) == 5
        assert len(errors) == 0

        # All adapters should be the same instance (first one wins)
        first_adapter = results[0]
        for adapter in results[1:]:
            assert adapter is first_adapter


if __name__ == "__main__":
    pytest.main([__file__])
