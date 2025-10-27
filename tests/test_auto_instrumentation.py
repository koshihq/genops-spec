"""Tests for GenOps AI auto-instrumentation system."""

import pytest
from unittest.mock import MagicMock, patch, call
import importlib

from genops.auto_instrumentation import (
    GenOpsInstrumentor,
    init,
    uninstrument,
    status,
    get_default_attributes,
)


class TestGenOpsInstrumentor:
    """Test the GenOpsInstrumentor class."""
    
    def test_instrumentor_singleton(self, cleanup_test_state):
        """Test that GenOpsInstrumentor follows singleton pattern."""
        instrumentor1 = GenOpsInstrumentor()
        instrumentor2 = GenOpsInstrumentor()
        
        assert instrumentor1 is instrumentor2
        assert GenOpsInstrumentor._instance is instrumentor1
    
    def test_instrumentor_initialization(self, cleanup_test_state):
        """Test instrumentor initialization sets up provider registry."""
        instrumentor = GenOpsInstrumentor()
        
        assert hasattr(instrumentor, 'patched_providers')
        assert hasattr(instrumentor, 'available_providers')
        assert hasattr(instrumentor, 'provider_patches')
        assert len(instrumentor.provider_patches) >= 2  # At least OpenAI and Anthropic
        
        # Check that provider patches are registered
        assert 'openai' in instrumentor.provider_patches
        assert 'anthropic' in instrumentor.provider_patches
    
    def test_check_provider_availability(self, cleanup_test_state):
        """Test provider availability checking."""
        instrumentor = GenOpsInstrumentor()
        
        # Mock successful import
        with patch('importlib.import_module') as mock_import:
            mock_import.return_value = MagicMock()
            available = instrumentor._check_provider_availability('openai')
            assert available is True
            mock_import.assert_called_once_with('openai')
        
        # Mock failed import
        with patch('importlib.import_module') as mock_import:
            mock_import.side_effect = ImportError("No module named 'openai'")
            available = instrumentor._check_provider_availability('openai')
            assert available is False
    
    def test_setup_opentelemetry_console_exporter(self, cleanup_test_state):
        """Test OpenTelemetry setup with console exporter."""
        instrumentor = GenOpsInstrumentor()
        
        with patch('opentelemetry.trace.set_tracer_provider') as mock_set_provider:
            with patch('genops.auto_instrumentation.TracerProvider') as mock_tracer_provider:
                with patch('genops.auto_instrumentation.ConsoleSpanExporter') as mock_console:
                    with patch('genops.auto_instrumentation.BatchSpanProcessor') as mock_processor:
                        
                        instrumentor._setup_opentelemetry(
                            service_name="test-service",
                            exporter_type="console"
                        )
                        
                        # Verify TracerProvider was created and set
                        mock_tracer_provider.assert_called_once()
                        mock_set_provider.assert_called_once()
                        
                        # Verify console exporter was used
                        mock_console.assert_called_once()
                        mock_processor.assert_called()
    
    def test_setup_opentelemetry_otlp_exporter(self, cleanup_test_state):
        """Test OpenTelemetry setup with OTLP exporter."""
        instrumentor = GenOpsInstrumentor()
        
        with patch('opentelemetry.trace.set_tracer_provider') as mock_set_provider:
            with patch('genops.auto_instrumentation.TracerProvider') as mock_tracer_provider:
                with patch('genops.auto_instrumentation.OTLPSpanExporter') as mock_otlp:
                    with patch('genops.auto_instrumentation.BatchSpanProcessor') as mock_processor:
                        
                        instrumentor._setup_opentelemetry(
                            service_name="test-service",
                            exporter_type="otlp",
                            otlp_endpoint="https://api.honeycomb.io",
                            otlp_headers={"x-honeycomb-team": "test-key"}
                        )
                        
                        # Verify OTLP exporter was configured
                        mock_otlp.assert_called_once_with(
                            endpoint="https://api.honeycomb.io",
                            headers={"x-honeycomb-team": "test-key"}
                        )
    
    def test_instrument_provider_success(self, cleanup_test_state):
        """Test successful provider instrumentation."""
        instrumentor = GenOpsInstrumentor()
        
        # Mock provider availability and patch function
        mock_patch_func = MagicMock()
        instrumentor.provider_patches['test_provider'] = {
            'patch': mock_patch_func,
            'unpatch': MagicMock(),
            'module': 'test_module'
        }
        
        with patch.object(instrumentor, '_check_provider_availability', return_value=True):
            result = instrumentor._instrument_provider('test_provider')
            
            assert result is True
            mock_patch_func.assert_called_once()
            assert 'test_provider' in instrumentor.patched_providers
    
    def test_instrument_provider_unavailable(self, cleanup_test_state):
        """Test provider instrumentation when provider is unavailable."""
        instrumentor = GenOpsInstrumentor()
        
        with patch.object(instrumentor, '_check_provider_availability', return_value=False):
            result = instrumentor._instrument_provider('openai')
            
            assert result is False
            assert 'openai' not in instrumentor.patched_providers
    
    def test_instrument_provider_failure(self, cleanup_test_state):
        """Test provider instrumentation when patching fails."""
        instrumentor = GenOpsInstrumentor()
        
        # Mock provider availability but patch function fails
        mock_patch_func = MagicMock(side_effect=Exception("Patch failed"))
        instrumentor.provider_patches['test_provider'] = {
            'patch': mock_patch_func,
            'unpatch': MagicMock(),
            'module': 'test_module'
        }
        
        with patch.object(instrumentor, '_check_provider_availability', return_value=True):
            result = instrumentor._instrument_provider('test_provider')
            
            assert result is False
            assert 'test_provider' not in instrumentor.patched_providers
    
    def test_instrument_all_providers(self, cleanup_test_state):
        """Test instrumenting all available providers."""
        instrumentor = GenOpsInstrumentor()
        
        # Mock some providers as available, others not
        availability_map = {
            'openai': True,
            'anthropic': False,
        }
        
        with patch.object(instrumentor, '_check_provider_availability', side_effect=lambda p: availability_map.get(p, False)):
            with patch.object(instrumentor, '_instrument_provider', return_value=True) as mock_instrument:
                
                instrumented = instrumentor._instrument_providers()
                
                # Should try to instrument all providers
                expected_calls = [call('openai'), call('anthropic')]
                mock_instrument.assert_has_calls(expected_calls, any_order=True)
    
    def test_instrument_specific_providers(self, cleanup_test_state):
        """Test instrumenting specific providers only."""
        instrumentor = GenOpsInstrumentor()
        
        with patch.object(instrumentor, '_instrument_provider', return_value=True) as mock_instrument:
            instrumented = instrumentor._instrument_providers(['openai'])
            
            # Should only instrument OpenAI
            mock_instrument.assert_called_once_with('openai')
    
    def test_uninstrument_providers(self, cleanup_test_state):
        """Test uninstrumenting providers."""
        instrumentor = GenOpsInstrumentor()
        
        # Mock some patched providers
        mock_unpatch1 = MagicMock()
        mock_unpatch2 = MagicMock()
        
        instrumentor.patched_providers = {
            'openai': {'unpatch': mock_unpatch1},
            'anthropic': {'unpatch': mock_unpatch2},
        }
        
        instrumentor._uninstrument_providers()
        
        # Both unpatch functions should be called
        mock_unpatch1.assert_called_once()
        mock_unpatch2.assert_called_once()
        
        # Patched providers should be cleared
        assert len(instrumentor.patched_providers) == 0
    
    def test_instrument_full_workflow(self, cleanup_test_state, mock_otel_setup):
        """Test full instrumentation workflow."""
        instrumentor = GenOpsInstrumentor()
        
        # Mock provider availability
        with patch.object(instrumentor, '_check_provider_availability', return_value=True):
            with patch.object(instrumentor, '_instrument_provider', return_value=True):
                
                result = instrumentor.instrument(
                    service_name="test-app",
                    environment="testing",
                    exporter_type="console",
                    default_team="test-team"
                )
                
                assert result == instrumentor
                assert instrumentor._initialized is True
                assert instrumentor.default_attributes['team'] == "test-team"
                assert instrumentor.default_attributes['environment'] == "testing"
    
    def test_status_method(self, cleanup_test_state):
        """Test status method returns correct information."""
        instrumentor = GenOpsInstrumentor()
        
        # Before initialization
        status_info = instrumentor.status()
        assert status_info['initialized'] is False
        assert status_info['instrumented_providers'] == []
        assert status_info['default_attributes'] == {}
        
        # Mock initialization
        instrumentor._initialized = True
        instrumentor.patched_providers = {'openai': {}, 'anthropic': {}}
        instrumentor.default_attributes = {'team': 'test-team'}
        instrumentor.available_providers = {'openai': True, 'anthropic': False}
        
        status_info = instrumentor.status()
        assert status_info['initialized'] is True
        assert set(status_info['instrumented_providers']) == {'openai', 'anthropic'}
        assert status_info['default_attributes']['team'] == 'test-team'
        assert status_info['available_providers']['openai'] is True
        assert status_info['available_providers']['anthropic'] is False


class TestGlobalAutoInstrumentationFunctions:
    """Test global auto-instrumentation functions."""
    
    def test_init_function(self, cleanup_test_state):
        """Test global init function."""
        with patch('genops.auto_instrumentation.GenOpsInstrumentor') as mock_instrumentor_class:
            mock_instrumentor = MagicMock()
            mock_instrumentor_class.return_value = mock_instrumentor
            
            result = init(
                service_name="test-service",
                environment="production",
                default_team="platform-team"
            )
            
            assert result == mock_instrumentor
            mock_instrumentor.instrument.assert_called_once_with(
                service_name="test-service",
                environment="production",
                default_team="platform-team"
            )
    
    def test_uninstrument_function(self, cleanup_test_state):
        """Test global uninstrument function."""
        with patch('genops.auto_instrumentation.GenOpsInstrumentor') as mock_instrumentor_class:
            mock_instrumentor = MagicMock()
            mock_instrumentor_class.return_value = mock_instrumentor
            
            uninstrument()
            
            mock_instrumentor.uninstrument.assert_called_once()
    
    def test_status_function(self, cleanup_test_state):
        """Test global status function."""
        with patch('genops.auto_instrumentation.GenOpsInstrumentor') as mock_instrumentor_class:
            mock_instrumentor = MagicMock()
            mock_instrumentor.status.return_value = {"initialized": True}
            mock_instrumentor_class.return_value = mock_instrumentor
            
            result = status()
            
            assert result == {"initialized": True}
            mock_instrumentor.status.assert_called_once()
    
    def test_get_default_attributes_function(self, cleanup_test_state):
        """Test global get_default_attributes function."""
        with patch('genops.auto_instrumentation.GenOpsInstrumentor') as mock_instrumentor_class:
            mock_instrumentor = MagicMock()
            mock_instrumentor.get_default_attributes.return_value = {"team": "test"}
            mock_instrumentor_class.return_value = mock_instrumentor
            
            result = get_default_attributes()
            
            assert result == {"team": "test"}
            mock_instrumentor.get_default_attributes.assert_called_once()


class TestAutoInstrumentationIntegration:
    """Integration tests for auto-instrumentation system."""
    
    def test_end_to_end_instrumentation_workflow(self, cleanup_test_state, mock_otel_setup):
        """Test complete instrumentation workflow from init to uninstrument."""
        # Initialize instrumentation
        with patch('genops.auto_instrumentation.GenOpsInstrumentor._check_provider_availability') as mock_check:
            with patch('genops.auto_instrumentation.GenOpsInstrumentor._instrument_provider') as mock_instrument:
                # Mock OpenAI available, Anthropic not
                mock_check.side_effect = lambda p: p == 'openai'
                mock_instrument.return_value = True
                
                # Initialize
                instrumentor = init(
                    service_name="integration-test",
                    environment="testing",
                    exporter_type="console",
                    default_team="integration-team",
                    default_project="test-project"
                )
                
                # Check status after init
                status_info = status()
                assert status_info['initialized'] is True
                assert 'integration-team' in str(status_info['default_attributes'])
                
                # Get default attributes
                defaults = get_default_attributes()
                assert defaults['team'] == 'integration-team'
                assert defaults['project'] == 'test-project'
                
                # Uninstrument
                with patch.object(instrumentor, '_uninstrument_providers') as mock_uninstrument:
                    uninstrument()
                    mock_uninstrument.assert_called_once()
    
    def test_multiple_initialization_calls(self, cleanup_test_state):
        """Test that multiple init calls work properly."""
        with patch('genops.auto_instrumentation.GenOpsInstrumentor._setup_opentelemetry'):
            with patch('genops.auto_instrumentation.GenOpsInstrumentor._instrument_provider', return_value=True):
                
                # First initialization
                instrumentor1 = init(service_name="service1")
                
                # Second initialization should return same instance but update config
                instrumentor2 = init(service_name="service2", default_team="new-team")
                
                assert instrumentor1 is instrumentor2
                
                # Should have updated configuration
                defaults = get_default_attributes()
                assert defaults.get('team') == 'new-team'
    
    def test_provider_specific_instrumentation(self, cleanup_test_state):
        """Test instrumentation with specific providers."""
        with patch('genops.auto_instrumentation.GenOpsInstrumentor._check_provider_availability', return_value=True):
            with patch('genops.auto_instrumentation.GenOpsInstrumentor._instrument_provider', return_value=True) as mock_instrument:
                
                # Initialize with only OpenAI
                instrumentor = init(
                    service_name="openai-only",
                    providers=["openai"]
                )
                
                # Should only instrument OpenAI
                mock_instrument.assert_called_once_with('openai')
    
    def test_configuration_inheritance_and_override(self, cleanup_test_state):
        """Test configuration inheritance and override behavior."""
        with patch('genops.auto_instrumentation.GenOpsInstrumentor._setup_opentelemetry'):
            with patch('genops.auto_instrumentation.GenOpsInstrumentor._instrument_provider', return_value=True):
                
                # Initialize with default configuration
                init(
                    service_name="test-service",
                    default_team="platform-team",
                    default_project="main-project",
                    default_environment="staging"
                )
                
                defaults = get_default_attributes()
                
                # Verify all defaults are set
                assert defaults['team'] == 'platform-team'
                assert defaults['project'] == 'main-project'
                assert defaults['environment'] == 'staging'
    
    def test_error_handling_in_initialization(self, cleanup_test_state):
        """Test error handling during initialization."""
        with patch('genops.auto_instrumentation.GenOpsInstrumentor._setup_opentelemetry') as mock_setup:
            mock_setup.side_effect = Exception("OpenTelemetry setup failed")
            
            with patch('genops.auto_instrumentation.GenOpsInstrumentor._instrument_provider', return_value=True):
                
                # Initialization should handle the error gracefully
                instrumentor = init(service_name="error-test")
                
                # Should still return an instrumentor instance
                assert instrumentor is not None
                
                # Status should reflect the error state
                status_info = status()
                # Implementation should handle this gracefully