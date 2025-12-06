#!/usr/bin/env python3
"""
Unit tests for GenOps Raindrop AI Adapter

This test suite provides comprehensive coverage for the Raindrop AI integration
including adapter initialization, session management, cost tracking, and error handling.

Test Categories:
- Adapter initialization and configuration
- Session lifecycle management  
- Cost calculation and aggregation
- Error handling and edge cases
- Validation framework
- OpenTelemetry integration

Author: GenOps AI Contributors
"""

import pytest
import os
import time
import uuid
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from genops.providers.raindrop import (
    GenOpsRaindropAdapter,
    RaindropMonitoringSession,
    RaindropGovernanceAttributes,
    auto_instrument,
    restore_raindrop
)
from genops.providers.raindrop_pricing import RaindropCostResult, RaindropPricingConfig
from genops.providers.raindrop_validation import ValidationResult, ValidationIssue

class TestRaindropGovernanceAttributes:
    """Test governance attributes data structure and methods."""
    
    def test_governance_attributes_initialization(self):
        """Test basic governance attributes initialization."""
        attrs = RaindropGovernanceAttributes(
            team="test-team",
            project="test-project"
        )
        
        assert attrs.team == "test-team"
        assert attrs.project == "test-project"
        assert attrs.environment == "production"  # default
        assert attrs.customer_id is None
        assert attrs.cost_center is None
        assert attrs.feature is None
        assert len(attrs.session_id) > 0
    
    def test_governance_attributes_with_all_fields(self):
        """Test governance attributes with all fields specified."""
        attrs = RaindropGovernanceAttributes(
            team="ai-platform",
            project="agent-monitoring",
            environment="staging",
            customer_id="customer-123",
            cost_center="ai-operations",
            feature="fraud-detection"
        )
        
        assert attrs.team == "ai-platform"
        assert attrs.project == "agent-monitoring"
        assert attrs.environment == "staging"
        assert attrs.customer_id == "customer-123"
        assert attrs.cost_center == "ai-operations"
        assert attrs.feature == "fraud-detection"
    
    def test_governance_attributes_to_dict(self):
        """Test conversion to dictionary format."""
        attrs = RaindropGovernanceAttributes(
            team="test-team",
            project="test-project",
            customer_id="customer-456"
        )
        
        result = attrs.to_dict()
        
        assert result["genops.team"] == "test-team"
        assert result["genops.project"] == "test-project"
        assert result["genops.environment"] == "production"
        assert result["genops.customer_id"] == "customer-456"
        assert result["genops.provider"] == "raindrop"
        assert "genops.session_id" in result
        assert len(result["genops.session_id"]) > 0
    
    def test_governance_attributes_session_id_uniqueness(self):
        """Test that session IDs are unique across instances."""
        attrs1 = RaindropGovernanceAttributes(team="team1", project="project1")
        attrs2 = RaindropGovernanceAttributes(team="team2", project="project2")
        
        assert attrs1.session_id != attrs2.session_id
        assert len(attrs1.session_id) > 10
        assert len(attrs2.session_id) > 10

class TestGenOpsRaindropAdapter:
    """Test the main GenOps Raindrop AI adapter class."""
    
    def test_adapter_initialization_minimal(self):
        """Test adapter initialization with minimal parameters."""
        with patch('genops.providers.raindrop.validate_setup') as mock_validate:
            mock_validate.return_value = Mock(is_valid=True)
            
            adapter = GenOpsRaindropAdapter(
                raindrop_api_key="test-key",
                governance_policy="advisory"  # Skip validation in tests
            )
            
            assert adapter.raindrop_api_key == "test-key"
            assert adapter.governance_attrs.team == "default"
            assert adapter.governance_attrs.project == "default"
            assert adapter.daily_budget_limit is None
            assert adapter.enable_cost_alerts is True
            assert adapter.governance_policy == "advisory"
    
    def test_adapter_initialization_full_config(self):
        """Test adapter initialization with full configuration."""
        with patch('genops.providers.raindrop.validate_setup') as mock_validate:
            mock_validate.return_value = Mock(is_valid=True)
            
            adapter = GenOpsRaindropAdapter(
                raindrop_api_key="test-key",
                team="ai-platform",
                project="agent-monitoring",
                environment="staging",
                customer_id="customer-789",
                cost_center="ai-operations",
                feature="chatbot",
                daily_budget_limit=100.0,
                enable_cost_alerts=True,
                governance_policy="advisory",
                export_telemetry=False
            )
            
            assert adapter.raindrop_api_key == "test-key"
            assert adapter.governance_attrs.team == "ai-platform"
            assert adapter.governance_attrs.project == "agent-monitoring"
            assert adapter.governance_attrs.environment == "staging"
            assert adapter.governance_attrs.customer_id == "customer-789"
            assert adapter.governance_attrs.cost_center == "ai-operations"
            assert adapter.governance_attrs.feature == "chatbot"
            assert adapter.daily_budget_limit == 100.0
    
    def test_adapter_environment_variable_fallback(self):
        """Test that adapter falls back to environment variables."""
        with patch.dict(os.environ, {"RAINDROP_API_KEY": "env-key"}):
            with patch('genops.providers.raindrop.validate_setup') as mock_validate:
                mock_validate.return_value = Mock(is_valid=True)
                
                adapter = GenOpsRaindropAdapter(governance_policy="advisory")
                
                assert adapter.raindrop_api_key == "env-key"
    
    def test_adapter_validation_enforced_mode_success(self):
        """Test successful validation in enforced mode."""
        with patch('genops.providers.raindrop.validate_setup') as mock_validate:
            mock_validate.return_value = Mock(is_valid=True)
            
            adapter = GenOpsRaindropAdapter(
                raindrop_api_key="test-key",
                governance_policy="enforced"
            )
            
            assert adapter.governance_policy == "enforced"
            mock_validate.assert_called_once_with("test-key")
    
    def test_adapter_validation_enforced_mode_failure(self):
        """Test validation failure in enforced mode raises error."""
        with patch('genops.providers.raindrop.validate_setup') as mock_validate:
            mock_error = ValidationIssue(
                category="test",
                severity="error", 
                message="Test validation error",
                fix_suggestion="Fix the test error"
            )
            mock_validate.return_value = Mock(is_valid=False, errors=[mock_error])
            
            with pytest.raises(ValueError, match="Raindrop AI setup validation failed"):
                GenOpsRaindropAdapter(
                    raindrop_api_key="invalid-key",
                    governance_policy="enforced"
                )
    
    def test_adapter_telemetry_setup_disabled(self):
        """Test adapter with telemetry disabled."""
        with patch('genops.providers.raindrop.validate_setup') as mock_validate:
            mock_validate.return_value = Mock(is_valid=True)
            
            adapter = GenOpsRaindropAdapter(
                raindrop_api_key="test-key",
                governance_policy="advisory",
                export_telemetry=False
            )
            
            assert adapter.tracer is None
    
    @patch('genops.providers.raindrop.trace')
    def test_adapter_telemetry_setup_success(self, mock_trace):
        """Test successful telemetry setup."""
        # Mock OpenTelemetry components
        mock_tracer_provider = Mock()
        mock_tracer = Mock()
        mock_trace.get_tracer_provider.return_value = mock_tracer_provider
        mock_trace.get_tracer.return_value = mock_tracer
        
        with patch('genops.providers.raindrop.validate_setup') as mock_validate:
            mock_validate.return_value = Mock(is_valid=True)
            
            adapter = GenOpsRaindropAdapter(
                raindrop_api_key="test-key",
                governance_policy="advisory",
                export_telemetry=True
            )
            
            assert adapter.tracer == mock_tracer
    
    def test_adapter_pricing_calculator_initialization(self):
        """Test that pricing calculator is properly initialized."""
        with patch('genops.providers.raindrop.validate_setup') as mock_validate:
            mock_validate.return_value = Mock(is_valid=True)
            
            adapter = GenOpsRaindropAdapter(
                raindrop_api_key="test-key",
                governance_policy="advisory"
            )
            
            assert adapter.pricing_calculator is not None
            assert hasattr(adapter.pricing_calculator, 'calculate_interaction_cost')
    
    def test_adapter_cost_aggregator_initialization(self):
        """Test that cost aggregator is properly initialized."""
        with patch('genops.providers.raindrop.validate_setup') as mock_validate:
            mock_validate.return_value = Mock(is_valid=True)
            
            adapter = GenOpsRaindropAdapter(
                raindrop_api_key="test-key",
                governance_policy="advisory"
            )
            
            assert adapter.cost_aggregator is not None
            assert hasattr(adapter.cost_aggregator, 'add_session')

class TestRaindropMonitoringSession:
    """Test the monitoring session context manager and operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('genops.providers.raindrop.validate_setup') as mock_validate:
            mock_validate.return_value = Mock(is_valid=True)
            
            self.adapter = GenOpsRaindropAdapter(
                raindrop_api_key="test-key",
                team="test-team",
                project="test-project",
                governance_policy="advisory",
                export_telemetry=False
            )
    
    def test_session_initialization(self):
        """Test session initialization with basic parameters."""
        session = RaindropMonitoringSession(
            name="test-session",
            adapter=self.adapter,
            governance_attrs=self.adapter.governance_attrs
        )
        
        assert session.name == "test-session"
        assert session.adapter == self.adapter
        assert session.governance_attrs == self.adapter.governance_attrs
        assert session.total_cost == Decimal('0.00')
        assert session.operation_count == 0
        assert session.finalized is False
        assert session.start_time > 0
    
    def test_session_context_manager(self):
        """Test session as context manager."""
        session_name = "context-test"
        
        with self.adapter.track_agent_monitoring_session(session_name) as session:
            assert session.name == session_name
            assert session.adapter == self.adapter
            assert not session.finalized
        
        # Session should be finalized after context exit
        assert session.finalized
    
    def test_session_track_agent_interaction(self):
        """Test tracking agent interactions."""
        with self.adapter.track_agent_monitoring_session("test-session") as session:
            interaction_data = {
                "input": "test query",
                "output": "test response",
                "performance_metrics": {"latency": 200}
            }
            
            cost_result = session.track_agent_interaction(
                agent_id="test-agent",
                interaction_data=interaction_data,
                cost=0.01
            )
            
            assert isinstance(cost_result, RaindropCostResult)
            assert cost_result.total_cost == Decimal('0.01')
            assert session.operation_count == 1
            assert session.total_cost == Decimal('0.01')
    
    def test_session_track_performance_signal(self):
        """Test tracking performance signals."""
        with self.adapter.track_agent_monitoring_session("test-session") as session:
            signal_data = {
                "threshold": 0.85,
                "current_value": 0.92,
                "monitoring_frequency": "high"
            }
            
            cost_result = session.track_performance_signal(
                signal_name="accuracy_monitoring",
                signal_data=signal_data,
                cost=0.02
            )
            
            assert isinstance(cost_result, RaindropCostResult)
            assert cost_result.total_cost == Decimal('0.02')
            assert session.operation_count == 1
            assert session.total_cost == Decimal('0.02')
    
    def test_session_create_alert(self):
        """Test creating alerts."""
        with self.adapter.track_agent_monitoring_session("test-session") as session:
            alert_config = {
                "conditions": [{"metric": "accuracy", "operator": "<", "threshold": 0.8}],
                "notification_channels": ["email"],
                "severity": "warning"
            }
            
            cost_result = session.create_alert(
                alert_name="performance_alert",
                alert_config=alert_config,
                cost=0.05
            )
            
            assert isinstance(cost_result, RaindropCostResult)
            assert cost_result.total_cost == Decimal('0.05')
            assert session.operation_count == 1
            assert session.total_cost == Decimal('0.05')
    
    def test_session_multiple_operations(self):
        """Test session with multiple operations."""
        with self.adapter.track_agent_monitoring_session("multi-op-session") as session:
            # Track multiple operations
            session.track_agent_interaction("agent-1", {"test": "data1"}, cost=0.01)
            session.track_performance_signal("signal-1", {"test": "data2"}, cost=0.02)
            session.create_alert("alert-1", {"test": "config"}, cost=0.03)
            
            assert session.operation_count == 3
            assert session.total_cost == Decimal('0.06')
            assert len(session.operations) == 3
    
    def test_session_budget_enforcement(self):
        """Test budget enforcement during session operations."""
        # Create adapter with low budget limit
        with patch('genops.providers.raindrop.validate_setup') as mock_validate:
            mock_validate.return_value = Mock(is_valid=True)
            
            adapter = GenOpsRaindropAdapter(
                raindrop_api_key="test-key",
                team="budget-test",
                project="test-project",
                daily_budget_limit=0.05,  # Very low budget
                governance_policy="enforced",
                export_telemetry=False
            )
        
        with pytest.raises(ValueError, match="exceed daily budget limit"):
            with adapter.track_agent_monitoring_session("budget-test") as session:
                session.track_agent_interaction("agent-1", {"test": "data"}, cost=0.10)
    
    def test_session_properties(self):
        """Test session computed properties."""
        with self.adapter.track_agent_monitoring_session("properties-test") as session:
            # Add some operations with delays to test duration
            session.track_agent_interaction("agent-1", {"test": "data1"}, cost=0.01)
            time.sleep(0.1)  # Small delay
            session.track_performance_signal("signal-1", {"test": "data2"}, cost=0.02)
            
            # Test properties
            assert session.operation_count == 2
            assert session.duration_seconds > 0.05
            assert session.operations_per_hour > 0

class TestAutoInstrumentation:
    """Test auto-instrumentation functionality."""
    
    def test_auto_instrument_basic(self):
        """Test basic auto-instrumentation setup."""
        with patch('genops.providers.raindrop.validate_setup') as mock_validate:
            mock_validate.return_value = Mock(is_valid=True)
            
            adapter = auto_instrument(
                raindrop_api_key="test-key",
                team="auto-team",
                project="auto-project"
            )
            
            assert isinstance(adapter, GenOpsRaindropAdapter)
            assert adapter.governance_attrs.team == "auto-team"
            assert adapter.governance_attrs.project == "auto-project"
    
    def test_auto_instrument_with_kwargs(self):
        """Test auto-instrumentation with additional parameters."""
        with patch('genops.providers.raindrop.validate_setup') as mock_validate:
            mock_validate.return_value = Mock(is_valid=True)
            
            adapter = auto_instrument(
                raindrop_api_key="test-key",
                team="auto-team", 
                project="auto-project",
                daily_budget_limit=200.0,
                governance_policy="advisory"
            )
            
            assert adapter.daily_budget_limit == 200.0
            assert adapter.governance_policy == "advisory"
    
    @patch('genops.providers.raindrop.logger')
    def test_auto_instrument_sdk_not_available(self, mock_logger):
        """Test auto-instrumentation when Raindrop SDK is not available."""
        with patch('genops.providers.raindrop.validate_setup') as mock_validate:
            mock_validate.return_value = Mock(is_valid=True)
            
            with patch('builtins.__import__', side_effect=ImportError("No module named 'raindrop'")):
                adapter = auto_instrument(raindrop_api_key="test-key")
                
                assert isinstance(adapter, GenOpsRaindropAdapter)
                # Should log that SDK is not found
                mock_logger.info.assert_called()
    
    def test_auto_instrument_already_enabled_warning(self):
        """Test warning when auto-instrumentation is already enabled."""
        with patch('genops.providers.raindrop.validate_setup') as mock_validate:
            mock_validate.return_value = Mock(is_valid=True)
            
            # Enable first time
            auto_instrument(raindrop_api_key="test-key")
            
            # Enable second time should warn
            with patch('genops.providers.raindrop.logger') as mock_logger:
                auto_instrument(raindrop_api_key="test-key")
                mock_logger.warning.assert_called_with("Raindrop AI auto-instrumentation already enabled")
    
    def test_restore_raindrop(self):
        """Test disabling auto-instrumentation."""
        with patch('genops.providers.raindrop.validate_setup') as mock_validate:
            mock_validate.return_value = Mock(is_valid=True)
            
            # Enable instrumentation
            auto_instrument(raindrop_api_key="test-key")
            
            # Restore should work without errors
            restore_raindrop()

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_governance_policy(self):
        """Test handling of invalid governance policy."""
        with patch('genops.providers.raindrop.validate_setup') as mock_validate:
            mock_validate.return_value = Mock(is_valid=True)
            
            # This should still work as the policy is just stored
            adapter = GenOpsRaindropAdapter(
                raindrop_api_key="test-key",
                governance_policy="invalid-policy"
            )
            assert adapter.governance_policy == "invalid-policy"
    
    def test_missing_api_key_enforced_mode(self):
        """Test enforced mode with missing API key."""
        with patch('genops.providers.raindrop.validate_setup') as mock_validate:
            mock_error = ValidationIssue(
                category="auth",
                severity="error",
                message="API key not found",
                fix_suggestion="Set RAINDROP_API_KEY"
            )
            mock_validate.return_value = Mock(is_valid=False, errors=[mock_error])
            
            with pytest.raises(ValueError, match="setup validation failed"):
                GenOpsRaindropAdapter(governance_policy="enforced")
    
    def test_session_double_finalization(self):
        """Test that double finalization is handled gracefully."""
        with patch('genops.providers.raindrop.validate_setup') as mock_validate:
            mock_validate.return_value = Mock(is_valid=True)
            
            adapter = GenOpsRaindropAdapter(
                raindrop_api_key="test-key",
                governance_policy="advisory",
                export_telemetry=False
            )
        
        session = RaindropMonitoringSession(
            name="double-final-test",
            adapter=adapter,
            governance_attrs=adapter.governance_attrs
        )
        
        # Finalize once
        session._finalize()
        assert session.finalized
        
        # Finalize again (should be safe)
        session._finalize()
        assert session.finalized

class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        with patch('genops.providers.raindrop.validate_setup') as mock_validate:
            mock_validate.return_value = Mock(is_valid=True)
            
            # Initialize adapter
            adapter = GenOpsRaindropAdapter(
                raindrop_api_key="integration-key",
                team="integration-team",
                project="integration-project",
                daily_budget_limit=10.0,
                governance_policy="advisory",
                export_telemetry=False
            )
            
            # Run monitoring session
            with adapter.track_agent_monitoring_session("integration-test") as session:
                # Perform various operations
                interaction_cost = session.track_agent_interaction(
                    agent_id="integration-agent",
                    interaction_data={"type": "integration_test"},
                    cost=0.01
                )
                
                signal_cost = session.track_performance_signal(
                    signal_name="integration_signal",
                    signal_data={"performance": "good"},
                    cost=0.02
                )
                
                alert_cost = session.create_alert(
                    alert_name="integration_alert",
                    alert_config={"severity": "info"},
                    cost=0.03
                )
                
                # Verify results
                assert session.operation_count == 3
                assert session.total_cost == Decimal('0.06')
                
            # Verify session was added to aggregator
            summary = adapter.cost_aggregator.get_summary()
            assert summary.session_count == 1
            assert summary.total_cost == Decimal('0.06')
    
    def test_multiple_concurrent_sessions(self):
        """Test multiple concurrent monitoring sessions."""
        with patch('genops.providers.raindrop.validate_setup') as mock_validate:
            mock_validate.return_value = Mock(is_valid=True)
            
            adapter = GenOpsRaindropAdapter(
                raindrop_api_key="concurrent-key",
                governance_policy="advisory",
                export_telemetry=False
            )
            
            # Note: This tests the ability to have multiple sessions,
            # though they won't actually be concurrent in this test
            sessions_data = []
            
            for i in range(3):
                with adapter.track_agent_monitoring_session(f"session-{i}") as session:
                    session.track_agent_interaction(f"agent-{i}", {"test": f"data-{i}"}, cost=0.01)
                    sessions_data.append({
                        "name": session.name,
                        "cost": float(session.total_cost),
                        "operations": session.operation_count
                    })
            
            # Verify all sessions were tracked
            summary = adapter.cost_aggregator.get_summary()
            assert summary.session_count == 3
            assert summary.total_cost == Decimal('0.03')
            assert len(sessions_data) == 3

if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__])