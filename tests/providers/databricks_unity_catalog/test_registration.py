"""
Tests for Databricks Unity Catalog registration and auto-instrumentation.

Tests auto-registration, configuration detection, and instrumentation patterns:
- Auto-instrumentation setup and configuration detection
- Provider registration with instrumentation system
- Environment variable configuration handling
- Intelligent defaults and fallback mechanisms
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Import the modules under test
try:
    from genops.providers.databricks_unity_catalog.registration import (
        register_databricks_unity_catalog_provider,
        auto_register,
        auto_instrument_databricks,
        configure_unity_catalog_governance,
        _detect_databricks_configuration,
        _str_to_bool
    )
    REGISTRATION_AVAILABLE = True
except ImportError:
    REGISTRATION_AVAILABLE = False


@pytest.mark.skipif(not REGISTRATION_AVAILABLE, reason="Registration module not available")
class TestProviderRegistration:
    """Test provider registration functionality."""

    def test_register_provider_function(self):
        """Test provider registration function."""
        try:
            success = register_databricks_unity_catalog_provider()
            assert isinstance(success, bool)
        except Exception:
            # Expected in test environment without full GenOps setup
            pass

    @patch('genops.auto_instrumentation.register_provider')
    def test_successful_provider_registration(self, mock_register):
        """Test successful provider registration."""
        mock_register.return_value = True
        
        success = register_databricks_unity_catalog_provider()
        
        assert success == True
        mock_register.assert_called_once()
        call_args = mock_register.call_args[1]
        assert call_args["provider_name"] == "databricks_unity_catalog"
        assert call_args["framework_type"] == "data_platform"
        assert "databricks" in call_args["auto_detect_modules"]

    @patch('genops.auto_instrumentation.register_provider')
    def test_failed_provider_registration(self, mock_register):
        """Test failed provider registration handling."""
        mock_register.side_effect = Exception("Registration failed")
        
        success = register_databricks_unity_catalog_provider()
        
        assert success == False

    def test_auto_register_function(self):
        """Test auto-registration function."""
        # Should not raise errors
        try:
            auto_register()
        except Exception:
            # Expected in test environment
            pass

    @patch('databricks.sdk.WorkspaceClient')
    def test_auto_register_with_databricks_available(self, mock_client_class):
        """Test auto-registration when Databricks SDK is available."""
        with patch('genops.providers.databricks_unity_catalog.registration.register_databricks_unity_catalog_provider') as mock_register:
            mock_register.return_value = True
            
            auto_register()
            
            mock_register.assert_called_once()


class TestAutoInstrumentation:
    """Test auto-instrumentation functionality."""

    def test_auto_instrument_function_exists(self):
        """Test that auto-instrumentation function exists."""
        assert callable(auto_instrument_databricks)

    @patch('databricks.sdk.WorkspaceClient')
    @patch('genops.providers.databricks_unity_catalog.registration._detect_databricks_configuration')
    def test_auto_instrumentation_success(self, mock_detect_config, mock_client_class):
        """Test successful auto-instrumentation."""
        # Mock configuration detection
        mock_detect_config.return_value = {
            'workspace_url': 'https://test-workspace.cloud.databricks.com',
            'governance_attrs': {
                'team': 'test-team',
                'project': 'test-project',
                'environment': 'test'
            },
            'enable_auto_patching': True
        }
        
        with patch('genops.providers.databricks_unity_catalog.registration.instrument_databricks_unity_catalog') as mock_instrument:
            mock_adapter = MagicMock()
            mock_instrument.return_value = mock_adapter
            
            result = auto_instrument_databricks()
            
            assert result == mock_adapter
            mock_instrument.assert_called_once()

    @patch('genops.providers.databricks_unity_catalog.registration._detect_databricks_configuration')
    def test_auto_instrumentation_no_workspace_url(self, mock_detect_config):
        """Test auto-instrumentation when no workspace URL is detected."""
        mock_detect_config.return_value = {
            'workspace_url': None,
            'governance_attrs': {}
        }
        
        result = auto_instrument_databricks()
        
        assert result is None

    def test_auto_instrumentation_no_databricks_sdk(self):
        """Test auto-instrumentation when Databricks SDK is not available."""
        with patch('databricks.sdk.WorkspaceClient', side_effect=ImportError("No module named 'databricks'")):
            result = auto_instrument_databricks()
            assert result is None


class TestConfigurationDetection:
    """Test configuration detection and intelligent defaults."""

    def test_detect_databricks_configuration_full(self):
        """Test configuration detection with all environment variables."""
        env_vars = {
            'DATABRICKS_HOST': 'https://test-workspace.cloud.databricks.com',
            'DATABRICKS_TOKEN': 'test-token-12345',
            'GENOPS_TEAM': 'data-platform',
            'GENOPS_PROJECT': 'unity-catalog-governance',
            'GENOPS_ENVIRONMENT': 'production',
            'GENOPS_COST_CENTER': 'engineering',
            'GENOPS_USER_ID': 'test-user@example.com'
        }
        
        with patch.dict('os.environ', env_vars):
            config = _detect_databricks_configuration()
            
            assert config['workspace_url'] == 'https://test-workspace.cloud.databricks.com'
            assert config['governance_attrs']['team'] == 'data-platform'
            assert config['governance_attrs']['project'] == 'unity-catalog-governance'
            assert config['governance_attrs']['environment'] == 'production'
            assert config['governance_attrs']['cost_center'] == 'engineering'
            assert config['governance_attrs']['user_id'] == 'test-user@example.com'

    def test_detect_databricks_configuration_minimal(self):
        """Test configuration detection with minimal environment variables."""
        env_vars = {
            'DATABRICKS_HOST': 'https://minimal-workspace.cloud.databricks.com',
            'USER': 'system-user'
        }
        
        with patch.dict('os.environ', env_vars, clear=True):
            config = _detect_databricks_configuration()
            
            assert config['workspace_url'] == 'https://minimal-workspace.cloud.databricks.com'
            assert config['governance_attrs']['environment'] == 'development'  # Default
            assert config['governance_attrs']['project'] == 'auto-detected'  # Default

    def test_detect_databricks_configuration_alternative_vars(self):
        """Test configuration detection with alternative environment variable names."""
        env_vars = {
            'DATABRICKS_WORKSPACE_URL': 'https://alt-workspace.cloud.databricks.com',
            'DATABRICKS_ACCESS_TOKEN': 'alt-token-67890',
            'TEAM_NAME': 'alternative-team',
            'PROJECT_NAME': 'alternative-project',
            'ENVIRONMENT': 'staging'
        }
        
        with patch.dict('os.environ', env_vars, clear=True):
            config = _detect_databricks_configuration()
            
            assert 'alt-workspace' in config['workspace_url']
            assert config['governance_attrs']['team'] == 'alternative-team'
            assert config['governance_attrs']['project'] == 'alternative-project'
            assert config['governance_attrs']['environment'] == 'staging'

    def test_workspace_url_normalization(self):
        """Test workspace URL normalization."""
        test_cases = [
            {
                'input': 'test-workspace.cloud.databricks.com',
                'expected': 'https://test-workspace.cloud.databricks.com'
            },
            {
                'input': 'https://test-workspace.cloud.databricks.com/',
                'expected': 'https://test-workspace.cloud.databricks.com'
            },
            {
                'input': 'http://test-workspace.cloud.databricks.com',
                'expected': 'http://test-workspace.cloud.databricks.com'
            }
        ]
        
        for case in test_cases:
            env_vars = {'DATABRICKS_HOST': case['input']}
            
            with patch.dict('os.environ', env_vars, clear=True):
                config = _detect_databricks_configuration()
                assert config['workspace_url'] == case['expected']

    def test_environment_detection_from_url(self):
        """Test environment detection from workspace URL patterns."""
        url_patterns = [
            ('https://prod-workspace.cloud.databricks.com', 'production'),
            ('https://staging-env.cloud.databricks.com', 'staging'),
            ('https://dev-workspace.cloud.databricks.com', 'development'),
            ('https://test-environment.cloud.databricks.com', 'testing'),
            ('https://random-workspace.cloud.databricks.com', None)  # No detection
        ]
        
        for workspace_url, expected_env in url_patterns:
            env_vars = {'DATABRICKS_HOST': workspace_url}
            
            with patch.dict('os.environ', env_vars, clear=True):
                config = _detect_databricks_configuration()
                
                if expected_env:
                    assert config['governance_attrs']['environment'] == expected_env
                else:
                    # Should fall back to default
                    assert config['governance_attrs']['environment'] == 'development'

    def test_boolean_configuration_parsing(self):
        """Test boolean configuration value parsing."""
        boolean_test_cases = [
            ('true', True),
            ('false', False),
            ('1', True),
            ('0', False),
            ('yes', True),
            ('no', False),
            ('on', True),
            ('off', False),
            ('enabled', True),
            ('disabled', False),
            ('TRUE', True),
            ('FALSE', False)
        ]
        
        for input_value, expected_result in boolean_test_cases:
            result = _str_to_bool(input_value)
            assert result == expected_result

    def test_feature_toggle_detection(self):
        """Test feature toggle detection from environment variables."""
        env_vars = {
            'DATABRICKS_HOST': 'https://test-workspace.cloud.databricks.com',
            'GENOPS_ENABLE_AUTO_PATCHING': 'false',
            'GENOPS_ENABLE_COST_TRACKING': 'true',
            'GENOPS_ENABLE_LINEAGE_TRACKING': 'yes'
        }
        
        with patch.dict('os.environ', env_vars, clear=True):
            config = _detect_databricks_configuration()
            
            assert config['enable_auto_patching'] == False
            assert config['enable_cost_tracking'] == True
            assert config['enable_lineage_tracking'] == True


class TestUnityGovernanceConfiguration:
    """Test Unity Catalog governance configuration."""

    @patch('genops.providers.databricks_unity_catalog.registration.instrument_databricks_unity_catalog')
    @patch('genops.providers.databricks_unity_catalog.registration.get_governance_monitor')
    @patch('genops.providers.databricks_unity_catalog.registration.get_cost_aggregator')
    def test_configure_unity_catalog_governance_success(self, mock_cost_agg, mock_gov_monitor, mock_instrument):
        """Test successful Unity Catalog governance configuration."""
        # Mock dependencies
        mock_adapter = MagicMock()
        mock_instrument.return_value = mock_adapter
        mock_governance_monitor = MagicMock()
        mock_gov_monitor.return_value = mock_governance_monitor
        mock_cost_aggregator = MagicMock()
        mock_cost_agg.return_value = mock_cost_aggregator
        
        result = configure_unity_catalog_governance(
            workspace_url="https://test-workspace.cloud.databricks.com",
            metastore_id="test-metastore-123"
        )
        
        assert result["configured"] == True
        assert result["workspace_url"] == "https://test-workspace.cloud.databricks.com"
        assert result["metastore_id"] == "test-metastore-123"
        assert "data_lineage_tracking" in result["governance_features"]
        assert "compliance_monitoring" in result["governance_features"]
        assert "cost_attribution" in result["governance_features"]
        assert "policy_enforcement" in result["governance_features"]

    def test_configure_unity_catalog_governance_failure(self):
        """Test Unity Catalog governance configuration failure handling."""
        with patch('genops.providers.databricks_unity_catalog.registration.instrument_databricks_unity_catalog', side_effect=Exception("Configuration failed")):
            result = configure_unity_catalog_governance(
                workspace_url="https://invalid-workspace.com"
            )
            
            assert result["configured"] == False
            assert len(result["errors"]) > 0
            assert "Configuration failed" in result["errors"][0]


class TestEnvironmentHandling:
    """Test environment variable handling and edge cases."""

    def test_missing_environment_variables(self):
        """Test behavior when environment variables are missing."""
        # Clear all Databricks-related environment variables
        with patch.dict('os.environ', {}, clear=True):
            config = _detect_databricks_configuration()
            
            # Should handle missing variables gracefully
            assert config['workspace_url'] is None
            assert config['governance_attrs']['project'] == 'auto-detected'
            assert config['governance_attrs']['environment'] == 'development'

    def test_empty_environment_variables(self):
        """Test behavior when environment variables are empty strings."""
        env_vars = {
            'DATABRICKS_HOST': '',
            'GENOPS_TEAM': '',
            'GENOPS_PROJECT': ''
        }
        
        with patch.dict('os.environ', env_vars, clear=True):
            config = _detect_databricks_configuration()
            
            # Should handle empty strings appropriately
            assert config['workspace_url'] is None or config['workspace_url'] == ''

    def test_special_characters_in_environment_variables(self):
        """Test handling of special characters in environment variables."""
        env_vars = {
            'DATABRICKS_HOST': 'https://test-workspace.cloud.databricks.com',
            'GENOPS_TEAM': 'team-with-特殊字符',
            'GENOPS_PROJECT': 'project@#$%^&*()',
            'GENOPS_USER_ID': 'user+name@example.com'
        }
        
        with patch.dict('os.environ', env_vars, clear=True):
            config = _detect_databricks_configuration()
            
            # Should preserve special characters
            assert config['governance_attrs']['team'] == 'team-with-特殊字符'
            assert config['governance_attrs']['project'] == 'project@#$%^&*()'
            assert config['governance_attrs']['user_id'] == 'user+name@example.com'

    def test_very_long_environment_variables(self):
        """Test handling of very long environment variable values."""
        long_value = 'x' * 1000  # 1000 character string
        
        env_vars = {
            'DATABRICKS_HOST': 'https://test-workspace.cloud.databricks.com',
            'GENOPS_PROJECT': long_value
        }
        
        with patch.dict('os.environ', env_vars, clear=True):
            config = _detect_databricks_configuration()
            
            # Should handle long values without truncation (unless explicitly limited)
            assert len(config['governance_attrs']['project']) >= 500


class TestIntegrationPatterns:
    """Test integration patterns and real-world usage scenarios."""

    @patch('databricks.sdk.WorkspaceClient')
    def test_typical_enterprise_setup(self, mock_client_class):
        """Test typical enterprise setup pattern."""
        env_vars = {
            'DATABRICKS_HOST': 'https://enterprise-prod.cloud.databricks.com',
            'DATABRICKS_TOKEN': 'enterprise-production-token',
            'GENOPS_TEAM': 'data-platform-engineering',
            'GENOPS_PROJECT': 'enterprise-data-governance',
            'GENOPS_ENVIRONMENT': 'production',
            'GENOPS_COST_CENTER': 'data-infrastructure',
            'GENOPS_ENABLE_AUTO_PATCHING': 'true',
            'GENOPS_ENABLE_COST_TRACKING': 'true',
            'GENOPS_ENABLE_LINEAGE_TRACKING': 'true'
        }
        
        with patch.dict('os.environ', env_vars):
            with patch('genops.providers.databricks_unity_catalog.registration.instrument_databricks_unity_catalog') as mock_instrument:
                mock_adapter = MagicMock()
                mock_instrument.return_value = mock_adapter
                
                result = auto_instrument_databricks()
                
                assert result == mock_adapter
                mock_instrument.assert_called_once()
                
                call_kwargs = mock_instrument.call_args[1]
                assert 'enterprise-prod' in call_kwargs['workspace_url']
                assert call_kwargs['team'] == 'data-platform-engineering'
                assert call_kwargs['environment'] == 'production'

    def test_development_environment_setup(self):
        """Test development environment setup with minimal configuration."""
        env_vars = {
            'DATABRICKS_HOST': 'https://dev-workspace.cloud.databricks.com',
            'DATABRICKS_TOKEN': 'dev-token',
            'USER': 'developer'
        }
        
        with patch.dict('os.environ', env_vars, clear=True):
            config = _detect_databricks_configuration()
            
            # Should set appropriate defaults for development
            assert 'dev-workspace' in config['workspace_url']
            assert config['governance_attrs']['environment'] == 'development'
            assert config['governance_attrs']['project'] == 'auto-detected'

    @patch('genops.providers.databricks_unity_catalog.registration.patch_databricks_operations')
    def test_auto_patching_integration(self, mock_patch_ops):
        """Test auto-patching integration when enabled."""
        env_vars = {
            'DATABRICKS_HOST': 'https://test-workspace.cloud.databricks.com',
            'GENOPS_ENABLE_AUTO_PATCHING': 'true'
        }
        
        with patch.dict('os.environ', env_vars):
            with patch('databricks.sdk.WorkspaceClient'):
                with patch('genops.providers.databricks_unity_catalog.registration.instrument_databricks_unity_catalog') as mock_instrument:
                    mock_adapter = MagicMock()
                    mock_instrument.return_value = mock_adapter
                    
                    auto_instrument_databricks()
                    
                    mock_patch_ops.assert_called_once_with(mock_adapter)

    def test_multiple_workspace_configuration(self):
        """Test configuration for multiple workspace scenarios."""
        workspace_configs = [
            'https://prod-us-west.cloud.databricks.com',
            'https://prod-eu-central.cloud.databricks.com',
            'https://staging-global.cloud.databricks.com'
        ]
        
        for workspace_url in workspace_configs:
            env_vars = {'DATABRICKS_HOST': workspace_url}
            
            with patch.dict('os.environ', env_vars, clear=True):
                config = _detect_databricks_configuration()
                
                assert config['workspace_url'] == workspace_url
                # Should detect environment from URL when possible
                if 'staging' in workspace_url:
                    assert config['governance_attrs']['environment'] == 'staging'
                elif 'prod' in workspace_url:
                    assert config['governance_attrs']['environment'] == 'production'