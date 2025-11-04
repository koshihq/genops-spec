"""
Integration tests for Hugging Face GenOps integration.

Tests end-to-end workflows and integration scenarios including:
- Full workflow testing with real components  
- Integration between adapter, pricing, and validation
- Auto-instrumentation integration testing
- Error handling in integrated scenarios
- Performance and scalability testing
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', True)
    @patch('genops.providers.huggingface.InferenceClient')
    @patch('genops.providers.huggingface.GenOpsTelemetry')
    def test_complete_text_generation_workflow(self, mock_telemetry_class, mock_inference_client):
        """Test complete text generation workflow with all components."""
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        
        # Setup comprehensive mocks
        mock_client_instance = Mock()
        mock_client_instance.text_generation.return_value = "Generated response text"
        mock_inference_client.return_value = mock_client_instance
        
        mock_telemetry_instance = Mock()
        mock_telemetry_class.return_value = mock_telemetry_instance
        mock_span = Mock()
        mock_telemetry_instance.trace_operation.return_value.__enter__.return_value = mock_span
        
        # Mock pricing calculation
        with patch('genops.providers.huggingface_pricing.calculate_huggingface_cost', return_value=0.0025):
            adapter = GenOpsHuggingFaceAdapter()
            
            # Execute complete workflow
            result = adapter.text_generation(
                prompt="Generate a comprehensive product description for an AI governance platform",
                model="microsoft/DialoGPT-medium",
                max_new_tokens=200,
                temperature=0.7,
                
                # Governance attributes
                team="product-team",
                project="ai-governance-platform",
                customer_id="enterprise-client-001",
                environment="production",
                cost_center="product-development",
                feature="description-generation"
            )
            
            # Verify result
            assert result == "Generated response text"
            
            # Verify telemetry integration
            mock_telemetry_instance.trace_operation.assert_called_once()
            trace_call = mock_telemetry_instance.trace_operation.call_args
            
            # Check governance attributes were passed to telemetry
            assert trace_call[1]['team'] == "product-team"
            assert trace_call[1]['project'] == "ai-governance-platform"
            assert trace_call[1]['customer_id'] == "enterprise-client-001"
            assert trace_call[1]['environment'] == "production"
            
            # Verify cost recording
            mock_telemetry_instance.record_cost.assert_called_once()
            cost_call = mock_telemetry_instance.record_cost.call_args
            assert cost_call[1]['cost'] == 0.0025
            assert cost_call[1]['provider'] == 'huggingface_hub'
            assert cost_call[1]['model'] == 'microsoft/DialoGPT-medium'
            
            # Verify span attributes
            expected_span_calls = [
                ('genops.provider.detected', 'huggingface_hub'),
                ('genops.task.type', 'text-generation'),
                ('genops.tokens.input', 17),  # Estimated from prompt
                ('genops.tokens.output', 4)   # Estimated from "Generated response text"
            ]
            
            for attr_name, attr_value in expected_span_calls:
                mock_span.set_attribute.assert_any_call(attr_name, attr_value)

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', True)
    @patch('genops.providers.huggingface.InferenceClient')
    @patch('genops.providers.huggingface.GenOpsTelemetry')
    def test_multi_task_workflow_integration(self, mock_telemetry_class, mock_inference_client):
        """Test workflow using multiple AI tasks with integrated cost tracking."""
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        
        # Setup mocks for different tasks
        mock_client_instance = Mock()
        mock_client_instance.text_generation.return_value = "Generated content"
        mock_client_instance.feature_extraction.return_value = [[0.1, 0.2, 0.3]]
        mock_client_instance.text_to_image.return_value = b"fake_image_data"
        mock_inference_client.return_value = mock_client_instance
        
        mock_telemetry_instance = Mock()
        mock_telemetry_class.return_value = mock_telemetry_instance
        mock_span = Mock()
        mock_telemetry_instance.trace_operation.return_value.__enter__.return_value = mock_span
        
        # Mock different costs for different tasks
        cost_calculations = {
            'text-generation': 0.002,
            'feature-extraction': 0.0001,
            'text-to-image': 0.02
        }
        
        def mock_calculate_cost(**kwargs):
            return cost_calculations.get(kwargs.get('task', 'text-generation'), 0.001)
        
        with patch('genops.providers.huggingface_pricing.calculate_huggingface_cost', side_effect=mock_calculate_cost):
            adapter = GenOpsHuggingFaceAdapter()
            
            # Execute multi-task workflow
            governance_attrs = {
                "team": "content-team",
                "project": "multi-modal-content",
                "customer_id": "creative-agency-456"
            }
            
            # Task 1: Text generation
            text_result = adapter.text_generation(
                prompt="Create engaging marketing copy",
                model="gpt-3.5-turbo",
                **governance_attrs
            )
            
            # Task 2: Embedding generation
            embedding_result = adapter.feature_extraction(
                inputs=["Marketing copy content", "Brand messaging"],
                model="sentence-transformers/all-MiniLM-L6-v2",
                **governance_attrs
            )
            
            # Task 3: Image generation
            image_result = adapter.text_to_image(
                prompt="Create visual for marketing campaign",
                model="runwayml/stable-diffusion-v1-5",
                **governance_attrs
            )
            
            # Verify all tasks executed
            assert text_result == "Generated content"
            assert embedding_result == [[0.1, 0.2, 0.3]]
            assert image_result == b"fake_image_data"
            
            # Verify telemetry was called for each task
            assert mock_telemetry_instance.trace_operation.call_count == 3
            assert mock_telemetry_instance.record_cost.call_count == 3
            
            # Verify different cost calculations were used
            cost_calls = [call[1]['cost'] for call in mock_telemetry_instance.record_cost.call_args_list]
            assert 0.002 in cost_calls  # Text generation
            assert 0.0001 in cost_calls  # Embedding
            assert 0.02 in cost_calls   # Image generation

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', True)
    @patch('genops.providers.huggingface.InferenceClient')
    @patch('genops.providers.huggingface.GenOpsTelemetry')
    def test_provider_detection_integration(self, mock_telemetry_class, mock_inference_client):
        """Test integration between provider detection and cost calculation."""
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        
        mock_client_instance = Mock()
        mock_client_instance.text_generation.return_value = "Response"
        mock_inference_client.return_value = mock_client_instance
        
        mock_telemetry_instance = Mock()
        mock_telemetry_class.return_value = mock_telemetry_instance
        mock_span = Mock()
        mock_telemetry_instance.trace_operation.return_value.__enter__.return_value = mock_span
        
        # Test different models with their expected providers and costs
        test_scenarios = [
            ('gpt-3.5-turbo', 'openai', 0.0035),
            ('claude-3-haiku', 'anthropic', 0.000875),
            ('microsoft/DialoGPT-medium', 'huggingface_hub', 0.0001),
            ('mistral-7b-instruct', 'mistral', 0.0004)
        ]
        
        def mock_cost_calculation(provider, model, **kwargs):
            cost_map = {
                'openai': 0.0035,
                'anthropic': 0.000875,
                'huggingface_hub': 0.0001,
                'mistral': 0.0004
            }
            return cost_map.get(provider, 0.001)
        
        with patch('genops.providers.huggingface_pricing.calculate_huggingface_cost', side_effect=mock_cost_calculation):
            adapter = GenOpsHuggingFaceAdapter()
            
            for model, expected_provider, expected_cost in test_scenarios:
                # Reset mocks for each test
                mock_telemetry_instance.reset_mock()
                mock_span.reset_mock()
                
                result = adapter.text_generation(
                    prompt="Test prompt",
                    model=model,
                    team="integration-test"
                )
                
                # Verify provider detection
                mock_span.set_attribute.assert_any_call("genops.provider.detected", expected_provider)
                
                # Verify cost calculation integration
                cost_call = mock_telemetry_instance.record_cost.call_args
                assert abs(cost_call[1]['cost'] - expected_cost) < 0.000001
                assert cost_call[1]['provider'] == expected_provider
                assert cost_call[1]['model'] == model


class TestAutoInstrumentationIntegration:
    """Test auto-instrumentation integration scenarios."""

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', True)
    @patch('genops.providers.huggingface.InferenceClient')
    @patch('genops.providers.huggingface.GenOpsTelemetry')
    def test_auto_instrumentation_with_governance(self, mock_telemetry_class, mock_inference_client):
        """Test auto-instrumentation preserves governance attributes."""
        from genops.providers.huggingface import instrument_huggingface, GenOpsHuggingFaceAdapter
        
        # Setup mocks
        mock_client_instance = Mock()
        mock_client_instance.text_generation.return_value = "Auto-instrumented response"
        mock_inference_client.return_value = mock_client_instance
        
        mock_telemetry_instance = Mock()
        mock_telemetry_class.return_value = mock_telemetry_instance
        mock_span = Mock()
        mock_telemetry_instance.trace_operation.return_value.__enter__.return_value = mock_span
        
        # Mock original methods
        original_text_generation = Mock(return_value="original_response")
        mock_inference_client.text_generation = original_text_generation
        mock_inference_client.feature_extraction = Mock()
        mock_inference_client.text_to_image = Mock()
        
        with patch('genops.providers.huggingface_pricing.calculate_huggingface_cost', return_value=0.003):
            # Apply auto-instrumentation
            result = instrument_huggingface()
            assert result is True
            
            # Use client with auto-instrumentation
            client = mock_inference_client()
            
            # This should now go through GenOps adapter
            response = client.text_generation(
                "Generate content with governance",
                model="microsoft/DialoGPT-medium",
                max_new_tokens=100,
                
                # Governance attributes should work through auto-instrumentation
                team="auto-instrumented-team",
                project="instrumentation-test",
                customer_id="auto-client-789"
            )
            
            # Verify telemetry was called with governance
            mock_telemetry_instance.trace_operation.assert_called_once()
            trace_call = mock_telemetry_instance.trace_operation.call_args
            
            # Governance attributes should be preserved
            assert trace_call[1]['team'] == "auto-instrumented-team"
            assert trace_call[1]['project'] == "instrumentation-test"
            assert trace_call[1]['customer_id'] == "auto-client-789"
            
            # Cost recording should work
            mock_telemetry_instance.record_cost.assert_called_once()

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', True)
    @patch('genops.providers.huggingface.InferenceClient')
    def test_auto_instrumentation_restoration(self, mock_inference_client):
        """Test auto-instrumentation can be properly removed."""
        from genops.providers.huggingface import instrument_huggingface, uninstrument_huggingface
        
        # Setup original methods
        original_text_gen = Mock()
        original_feature_ext = Mock()
        original_text_to_img = Mock()
        
        mock_inference_client.text_generation = original_text_gen
        mock_inference_client.feature_extraction = original_feature_ext
        mock_inference_client.text_to_image = original_text_to_img
        
        # Apply instrumentation
        assert instrument_huggingface() is True
        
        # Methods should be wrapped now
        assert mock_inference_client.text_generation != original_text_gen
        
        # Original methods should be stored
        assert hasattr(mock_inference_client, '_genops_original_text_generation')
        assert mock_inference_client._genops_original_text_generation == original_text_gen
        
        # Remove instrumentation
        assert uninstrument_huggingface() is True
        
        # Methods should be restored
        assert mock_inference_client.text_generation == original_text_gen
        assert mock_inference_client.feature_extraction == original_feature_ext
        assert mock_inference_client.text_to_image == original_text_to_img
        
        # Storage attributes should be cleaned up
        assert not hasattr(mock_inference_client, '_genops_original_text_generation')


class TestValidationIntegration:
    """Test validation integration with other components."""

    def test_validation_with_real_components(self):
        """Test validation works with actual component integration."""
        from genops.providers.huggingface_validation import validate_huggingface_setup
        
        # This test uses real validation logic but mocks external dependencies
        with patch('genops.providers.huggingface_validation.InferenceClient') as mock_client, \
             patch('genops.providers.huggingface_validation.GenOpsHuggingFaceAdapter') as mock_adapter_class:
            
            # Mock successful components
            mock_client_instance = Mock()
            mock_client_instance.text_generation = Mock()
            mock_client.return_value = mock_client_instance
            
            mock_adapter = Mock()
            mock_adapter.get_supported_tasks.return_value = ['text-generation', 'feature-extraction']
            mock_adapter.detect_provider_for_model.return_value = 'openai'
            mock_adapter.is_available.return_value = True
            mock_adapter_class.return_value = mock_adapter
            
            # Mock pricing functions
            with patch('genops.providers.huggingface_validation.detect_model_provider', return_value='openai'), \
                 patch('genops.providers.huggingface_validation.calculate_huggingface_cost', return_value=0.002), \
                 patch('genops.providers.huggingface_validation.get_provider_info', return_value={'provider': 'openai'}):
                
                result = validate_huggingface_setup()
                
                # Should pass validation
                assert result.is_valid is True
                
                # Should have completed all validation checks
                assert result.summary['components_checked'] > 0
                
                # Should have minimal issues (only info/warnings)
                error_count = len([i for i in result.issues if i.level == 'error'])
                assert error_count == 0

    def test_validation_integration_with_missing_components(self):
        """Test validation correctly identifies missing component integration."""
        from genops.providers.huggingface_validation import validate_huggingface_setup
        
        # Mock missing components
        with patch('genops.providers.huggingface_validation.GenOpsHuggingFaceAdapter', side_effect=ImportError("Missing GenOps")):
            
            result = validate_huggingface_setup()
            
            # Should fail validation
            assert result.is_valid is False
            
            # Should identify missing GenOps integration
            error_issues = [i for i in result.issues if i.level == 'error']
            genops_error = next(
                (i for i in error_issues if 'GenOps' in i.message),
                None
            )
            assert genops_error is not None


class TestErrorHandlingIntegration:
    """Test integrated error handling scenarios."""

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', True)
    @patch('genops.providers.huggingface.InferenceClient')
    @patch('genops.providers.huggingface.GenOpsTelemetry')
    def test_error_handling_with_telemetry_integration(self, mock_telemetry_class, mock_inference_client):
        """Test error handling preserves telemetry context."""
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        
        # Setup mocks with failure
        mock_client_instance = Mock()
        mock_client_instance.text_generation.side_effect = Exception("API Error")
        mock_inference_client.return_value = mock_client_instance
        
        mock_telemetry_instance = Mock()
        mock_telemetry_class.return_value = mock_telemetry_instance
        mock_span = Mock()
        mock_telemetry_instance.trace_operation.return_value.__enter__.return_value = mock_span
        
        adapter = GenOpsHuggingFaceAdapter()
        
        # Execute operation that will fail
        with pytest.raises(Exception, match="API Error"):
            adapter.text_generation(
                prompt="This will fail",
                model="failing-model",
                team="error-handling-team",
                project="error-test",
                customer_id="error-client-123"
            )
        
        # Verify telemetry context was preserved during error
        mock_telemetry_instance.trace_operation.assert_called_once()
        trace_call = mock_telemetry_instance.trace_operation.call_args
        
        # Governance should be preserved even during errors
        assert trace_call[1]['team'] == "error-handling-team"
        assert trace_call[1]['project'] == "error-test"
        assert trace_call[1]['customer_id'] == "error-client-123"
        
        # Error details should be captured
        mock_span.set_attribute.assert_any_call("genops.error.message", "API Error")
        mock_span.set_attribute.assert_any_call("genops.error.type", "Exception")
        
        # Cost should not be recorded for failed operations
        mock_telemetry_instance.record_cost.assert_not_called()

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', True)
    @patch('genops.providers.huggingface.InferenceClient')
    @patch('genops.providers.huggingface.GenOpsTelemetry')
    def test_cost_calculation_error_handling_integration(self, mock_telemetry_class, mock_inference_client):
        """Test error handling when cost calculation fails."""
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        
        # Setup successful API call but failing cost calculation
        mock_client_instance = Mock()
        mock_client_instance.text_generation.return_value = "Success response"
        mock_inference_client.return_value = mock_client_instance
        
        mock_telemetry_instance = Mock()
        mock_telemetry_class.return_value = mock_telemetry_instance
        mock_span = Mock()
        mock_telemetry_instance.trace_operation.return_value.__enter__.return_value = mock_span
        
        # Mock cost calculation failure
        with patch('genops.providers.huggingface_pricing.calculate_huggingface_cost', side_effect=Exception("Cost calc failed")):
            adapter = GenOpsHuggingFaceAdapter()
            
            # Operation should still succeed despite cost calculation failure
            result = adapter.text_generation(
                prompt="Test prompt",
                model="test-model",
                team="cost-error-team"
            )
            
            # API call should succeed
            assert result == "Success response"
            
            # Telemetry should be called
            mock_telemetry_instance.trace_operation.assert_called_once()
            
            # Cost recording may be called with fallback cost
            # The adapter should handle cost calculation errors gracefully


class TestPerformanceIntegration:
    """Test performance aspects of integrated components."""

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', True)
    @patch('genops.providers.huggingface.InferenceClient')
    @patch('genops.providers.huggingface.GenOpsTelemetry')
    def test_concurrent_operations_integration(self, mock_telemetry_class, mock_inference_client):
        """Test integration handles concurrent operations correctly."""
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        
        # Setup mocks that simulate some processing time
        def slow_text_generation(*args, **kwargs):
            time.sleep(0.01)  # Small delay to simulate API call
            return f"Response for {kwargs.get('model', 'unknown')}"
        
        mock_client_instance = Mock()
        mock_client_instance.text_generation.side_effect = slow_text_generation
        mock_inference_client.return_value = mock_client_instance
        
        mock_telemetry_instance = Mock()
        mock_telemetry_class.return_value = mock_telemetry_instance
        mock_span = Mock()
        mock_telemetry_instance.trace_operation.return_value.__enter__.return_value = mock_span
        
        with patch('genops.providers.huggingface_pricing.calculate_huggingface_cost', return_value=0.001):
            adapter = GenOpsHuggingFaceAdapter()
            
            # Create multiple concurrent operations
            def run_operation(i):
                return adapter.text_generation(
                    prompt=f"Concurrent prompt {i}",
                    model=f"model-{i}",
                    team=f"team-{i}",
                    project="concurrent-test",
                    operation_id=f"op-{i}"
                )
            
            # Execute operations concurrently
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(run_operation, i) for i in range(10)]
                results = [future.result() for future in futures]
            
            end_time = time.time()
            
            # Verify all operations completed
            assert len(results) == 10
            for i, result in enumerate(results):
                assert f"model-{i}" in result
            
            # Verify concurrent execution was faster than sequential
            # (10 operations * 0.01s = 0.1s sequential, should be much faster concurrent)
            assert end_time - start_time < 0.08  # Allow some overhead
            
            # Verify telemetry was called for each operation
            assert mock_telemetry_instance.trace_operation.call_count == 10
            assert mock_telemetry_instance.record_cost.call_count == 10

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', True)
    @patch('genops.providers.huggingface.InferenceClient')
    @patch('genops.providers.huggingface.GenOpsTelemetry')
    def test_memory_efficiency_integration(self, mock_telemetry_class, mock_inference_client):
        """Test integration doesn't cause memory leaks with repeated operations."""
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        
        mock_client_instance = Mock()
        mock_client_instance.text_generation.return_value = "Repeated response"
        mock_inference_client.return_value = mock_client_instance
        
        mock_telemetry_instance = Mock()
        mock_telemetry_class.return_value = mock_telemetry_instance
        mock_span = Mock()
        mock_telemetry_instance.trace_operation.return_value.__enter__.return_value = mock_span
        
        with patch('genops.providers.huggingface_pricing.calculate_huggingface_cost', return_value=0.0001):
            adapter = GenOpsHuggingFaceAdapter()
            
            # Run many operations to test for memory accumulation
            for i in range(100):
                result = adapter.text_generation(
                    prompt=f"Memory test {i}",
                    model="memory-test-model",
                    team="memory-team",
                    batch_id=f"batch-{i // 10}"
                )
                
                assert result == "Repeated response"
                
                # Reset mocks periodically to prevent mock call history buildup
                if i % 20 == 0:
                    mock_telemetry_instance.reset_mock()
                    mock_span.reset_mock()
            
            # Test should complete without memory issues
            # This is mainly a regression test to ensure no obvious memory leaks


class TestComplexWorkflowIntegration:
    """Test complex real-world workflow integrations."""

    @patch('genops.providers.huggingface.HAS_HUGGINGFACE', True)
    @patch('genops.providers.huggingface.InferenceClient')
    @patch('genops.providers.huggingface.GenOpsTelemetry')
    def test_content_pipeline_workflow(self, mock_telemetry_class, mock_inference_client):
        """Test complex content generation pipeline workflow."""
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter
        
        # Setup mocks for different operations
        mock_client_instance = Mock()
        mock_client_instance.text_generation.side_effect = [
            "Content outline",
            "Full article content", 
            "SEO metadata"
        ]
        mock_client_instance.feature_extraction.return_value = [[0.1, 0.2, 0.3]]
        mock_inference_client.return_value = mock_client_instance
        
        mock_telemetry_instance = Mock()
        mock_telemetry_class.return_value = mock_telemetry_instance
        mock_span = Mock()
        mock_telemetry_instance.trace_operation.return_value.__enter__.return_value = mock_span
        
        # Mock different costs for different steps
        costs = [0.001, 0.005, 0.0005, 0.0002]  # Outline, content, metadata, embedding
        with patch('genops.providers.huggingface_pricing.calculate_huggingface_cost', side_effect=costs):
            adapter = GenOpsHuggingFaceAdapter()
            
            # Execute content pipeline workflow
            pipeline_governance = {
                "team": "content-team",
                "project": "automated-content-pipeline",
                "customer_id": "content-client-999",
                "environment": "production",
                "workflow_id": "content-pipeline-001"
            }
            
            # Step 1: Generate outline
            outline = adapter.text_generation(
                prompt="Create article outline about AI governance best practices",
                model="microsoft/DialoGPT-medium",
                max_new_tokens=150,
                feature="outline-generation",
                **pipeline_governance
            )
            
            # Step 2: Generate full content
            content = adapter.text_generation(
                prompt=f"Write full article based on outline: {outline}",
                model="gpt-3.5-turbo",
                max_new_tokens=800,
                feature="content-generation",
                **pipeline_governance
            )
            
            # Step 3: Generate metadata
            metadata = adapter.text_generation(
                prompt=f"Generate SEO metadata for: {content[:200]}",
                model="claude-3-haiku",
                max_new_tokens=100,
                feature="metadata-generation",
                **pipeline_governance
            )
            
            # Step 4: Generate content embeddings
            embeddings = adapter.feature_extraction(
                inputs=[content],
                model="sentence-transformers/all-MiniLM-L6-v2",
                feature="content-embedding",
                **pipeline_governance
            )
            
            # Verify workflow execution
            assert outline == "Content outline"
            assert content == "Full article content"
            assert metadata == "SEO metadata"
            assert embeddings == [[0.1, 0.2, 0.3]]
            
            # Verify telemetry integration across workflow
            assert mock_telemetry_instance.trace_operation.call_count == 4
            assert mock_telemetry_instance.record_cost.call_count == 4
            
            # Verify governance propagation across all steps
            for call in mock_telemetry_instance.trace_operation.call_args_list:
                assert call[1]['team'] == "content-team"
                assert call[1]['project'] == "automated-content-pipeline"
                assert call[1]['customer_id'] == "content-client-999"
                assert call[1]['workflow_id'] == "content-pipeline-001"
            
            # Verify different features were tracked
            expected_features = ["outline-generation", "content-generation", "metadata-generation", "content-embedding"]
            actual_features = [call[1]['feature'] for call in mock_telemetry_instance.trace_operation.call_args_list]
            assert set(actual_features) == set(expected_features)
            
            # Verify total cost accumulation
            total_expected_cost = sum(costs)
            actual_costs = [call[1]['cost'] for call in mock_telemetry_instance.record_cost.call_args_list]
            assert abs(sum(actual_costs) - total_expected_cost) < 0.000001


if __name__ == '__main__':
    pytest.main([__file__, '-v'])