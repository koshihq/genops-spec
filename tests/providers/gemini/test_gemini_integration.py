#!/usr/bin/env python3
"""
Integration tests for GenOps Gemini provider.

This module tests the full integration of all Gemini components including:
- End-to-end workflow testing
- Auto-instrumentation functionality
- Cross-component interaction
- Real-world usage scenarios (with mocking)
"""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest

# Mock all external dependencies before importing our modules
genai_mock = MagicMock()
genai_mock.Client = MagicMock()

with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": genai_mock}):
    from genops.providers.gemini import (
        GenOpsGeminiAdapter,
        auto_instrument_gemini,
        print_validation_result,
        validate_setup,
    )
    from genops.providers.gemini_cost_aggregator import create_gemini_cost_context
    from genops.providers.gemini_pricing import (
        compare_gemini_models,
    )


class TestGeminiEndToEndWorkflow:
    """Test complete end-to-end Gemini workflows."""

    @patch("genops.providers.gemini.GEMINI_AVAILABLE", True)
    @patch("genops.providers.gemini.GENOPS_AVAILABLE", True)
    @patch("genops.providers.gemini.calculate_gemini_cost")
    def test_complete_text_generation_workflow(self, mock_calculate_cost):
        """Test complete text generation workflow with telemetry."""
        mock_calculate_cost.return_value = 0.001234

        # Mock Gemini client and response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is a generated response from Gemini."

        # Mock usage metadata
        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 15
        mock_usage.candidates_token_count = 25
        mock_response.usage_metadata = mock_usage

        mock_client.models.generate_content.return_value = mock_response

        # Mock telemetry
        mock_telemetry = MagicMock()
        mock_span = MagicMock()
        mock_telemetry.trace_operation.return_value.__enter__ = Mock(
            return_value=mock_span
        )
        mock_telemetry.trace_operation.return_value.__exit__ = Mock(return_value=None)

        with patch("genops.providers.gemini.genai.Client", return_value=mock_client):
            with patch(
                "genops.providers.gemini.GenOpsTelemetry", return_value=mock_telemetry
            ):
                # Initialize adapter
                adapter = GenOpsGeminiAdapter(api_key="test_key_123")

                # Perform text generation with governance attributes
                result = adapter.text_generation(
                    prompt="Explain artificial intelligence in simple terms",
                    model="gemini-2.5-flash",
                    temperature=0.7,
                    max_tokens=100,
                    team="ai-education",
                    project="content-generation",
                    customer_id="edu-platform-456",
                    environment="production",
                )

                # Verify result structure
                assert result.content == "This is a generated response from Gemini."
                assert result.model_id == "gemini-2.5-flash"
                assert result.input_tokens == 15
                assert result.output_tokens == 25
                assert result.cost_usd == 0.001234
                assert result.latency_ms > 0

                # Verify governance attributes
                assert result.governance_attributes["team"] == "ai-education"
                assert result.governance_attributes["project"] == "content-generation"
                assert result.governance_attributes["customer_id"] == "edu-platform-456"
                assert result.governance_attributes["environment"] == "production"

                # Verify API call was made correctly
                mock_client.models.generate_content.assert_called_once()
                call_args = mock_client.models.generate_content.call_args
                assert call_args[1]["model"] == "gemini-2.5-flash"
                assert (
                    call_args[1]["contents"]
                    == "Explain artificial intelligence in simple terms"
                )

                # Verify generation config
                gen_config = call_args[1]["generation_config"]
                assert gen_config["temperature"] == 0.7
                assert gen_config["max_output_tokens"] == 100

                # Verify telemetry was called
                mock_telemetry.trace_operation.assert_called_once()

                # Verify span attributes
                mock_span.set_attributes.assert_called()
                span_attrs = mock_span.set_attributes.call_args[0][0]
                assert span_attrs["genops.provider"] == "gemini"
                assert span_attrs["genops.model"] == "gemini-2.5-flash"
                assert span_attrs["genops.cost.total"] == 0.001234
                assert span_attrs["genops.tokens.input"] == 15
                assert span_attrs["genops.tokens.output"] == 25

    @patch("genops.providers.gemini.GEMINI_AVAILABLE", True)
    @patch("genops.providers.gemini.GENOPS_AVAILABLE", True)
    @patch("genops.providers.gemini_cost_aggregator.calculate_gemini_cost")
    def test_cost_aggregation_workflow(self, mock_calculate_cost):
        """Test complete cost aggregation workflow."""
        # Mock different costs for different operations
        mock_calculate_cost.side_effect = [0.001, 0.005, 0.002]

        # Mock Gemini client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Generated response"
        mock_client.models.generate_content.return_value = mock_response

        with patch("genops.providers.gemini.genai.Client", return_value=mock_client):
            # Initialize adapter
            adapter = GenOpsGeminiAdapter(api_key="test_key_123")

            # Use cost aggregation context
            with create_gemini_cost_context(
                "multi_operation_workflow",
                budget_limit=0.01,  # $0.01 budget
                enable_optimization=True,
                team="content-team",
                project="article-generation",
            ) as context:
                # Operation 1: Generate headline
                result1 = adapter.text_generation(
                    prompt="Generate a catchy headline for AI article",
                    model="gemini-2.5-flash-lite",
                    feature="headline-generation",
                )

                # Add to context
                context.add_operation(
                    operation_id="headline_gen",
                    model_id="gemini-2.5-flash-lite",
                    input_tokens=result1.input_tokens,
                    output_tokens=result1.output_tokens,
                    latency_ms=result1.latency_ms,
                    operation_type="headline_generation",
                )

                # Operation 2: Generate article content (more expensive)
                result2 = adapter.text_generation(
                    prompt="Write a comprehensive article about AI advances",
                    model="gemini-2.5-pro",
                    max_tokens=500,
                    feature="content-generation",
                )

                context.add_operation(
                    operation_id="content_gen",
                    model_id="gemini-2.5-pro",
                    input_tokens=result2.input_tokens,
                    output_tokens=result2.output_tokens,
                    latency_ms=result2.latency_ms,
                    operation_type="content_generation",
                )

                # Operation 3: Generate summary
                result3 = adapter.text_generation(
                    prompt="Create a brief summary of the article",
                    model="gemini-2.5-flash",
                    feature="summarization",
                )

                context.add_operation(
                    operation_id="summary_gen",
                    model_id="gemini-2.5-flash",
                    input_tokens=result3.input_tokens,
                    output_tokens=result3.output_tokens,
                    latency_ms=result3.latency_ms,
                    operation_type="summarization",
                )

                # Get aggregated summary
                summary = context.get_current_summary()

            # Verify aggregated results
            assert summary.total_cost == 0.008  # 0.001 + 0.005 + 0.002
            assert summary.total_operations == 3
            assert len(summary.unique_models) == 3  # Three different models

            # Verify model cost breakdown
            assert "gemini-2.5-flash-lite" in summary.cost_by_model
            assert "gemini-2.5-pro" in summary.cost_by_model
            assert "gemini-2.5-flash" in summary.cost_by_model

            # Verify operation type breakdown
            assert "headline_generation" in summary.cost_by_operation_type
            assert "content_generation" in summary.cost_by_operation_type
            assert "summarization" in summary.cost_by_operation_type

            # Verify governance attributes
            assert summary.governance_attributes["team"] == "content-team"
            assert summary.governance_attributes["project"] == "article-generation"

            # Verify optimization recommendations were generated
            assert len(summary.optimization_recommendations) > 0

    @patch("genops.providers.gemini.GEMINI_AVAILABLE", True)
    @patch("genops.providers.gemini.GENOPS_AVAILABLE", False)
    def test_workflow_without_telemetry(self):
        """Test workflow when telemetry is not available."""
        # Mock Gemini client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Generated response without telemetry"
        mock_client.models.generate_content.return_value = mock_response

        with patch("genops.providers.gemini.genai.Client", return_value=mock_client):
            with patch(
                "genops.providers.gemini.calculate_gemini_cost", return_value=0.001
            ):
                # Initialize adapter (should work without telemetry)
                adapter = GenOpsGeminiAdapter(api_key="test_key_123")

                # Perform text generation
                result = adapter.text_generation(
                    prompt="Test without telemetry",
                    model="gemini-2.5-flash",
                    team="test-team",
                )

                # Should still work and return result
                assert result.content == "Generated response without telemetry"
                assert result.model_id == "gemini-2.5-flash"
                assert result.governance_attributes["team"] == "test-team"

                # Cost should be 0 without pricing module
                # (since calculate_gemini_cost might not be available)


class TestGeminiAutoInstrumentation:
    """Test auto-instrumentation functionality."""

    @patch("genops.providers.gemini.GEMINI_AVAILABLE", True)
    @patch("genops.providers.gemini.GENOPS_AVAILABLE", True)
    def test_auto_instrumentation_setup(self):
        """Test auto-instrumentation setup."""
        with patch("genops.providers.gemini.genai"):
            # Test instrumentation
            success = auto_instrument_gemini()

            assert success is True

            # Verify that the original method was patched
            # (This is a simplified test - actual patching logic would be more complex)

    @patch("genops.providers.gemini.GEMINI_AVAILABLE", False)
    def test_auto_instrumentation_without_sdk(self):
        """Test auto-instrumentation when SDK is not available."""
        success = auto_instrument_gemini()

        assert success is False

    @patch("genops.providers.gemini.GEMINI_AVAILABLE", True)
    @patch("genops.providers.gemini.GENOPS_AVAILABLE", False)
    def test_auto_instrumentation_without_genops_core(self):
        """Test auto-instrumentation when GenOps core is not available."""
        success = auto_instrument_gemini()

        assert success is False


class TestGeminiChatCompletion:
    """Test chat completion functionality."""

    @patch("genops.providers.gemini.GEMINI_AVAILABLE", True)
    @patch("genops.providers.gemini.GENOPS_AVAILABLE", False)
    def test_chat_completion_workflow(self):
        """Test chat completion with message conversion."""
        # Mock Gemini client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "I'm doing well, thank you for asking!"
        mock_client.models.generate_content.return_value = mock_response

        with patch("genops.providers.gemini.genai.Client", return_value=mock_client):
            with patch(
                "genops.providers.gemini.calculate_gemini_cost", return_value=0.001
            ):
                adapter = GenOpsGeminiAdapter(api_key="test_key_123")

                # Test chat completion with multiple messages
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello there!"},
                    {
                        "role": "assistant",
                        "content": "Hello! How can I help you today?",
                    },
                    {"role": "user", "content": "How are you doing?"},
                ]

                result = adapter.chat_completion(
                    messages=messages,
                    model="gemini-2.5-flash",
                    temperature=0.8,
                    team="chat-team",
                    project="conversational-ai",
                )

                # Verify result
                assert result.content == "I'm doing well, thank you for asking!"
                assert result.model_id == "gemini-2.5-flash"
                assert result.governance_attributes["team"] == "chat-team"

                # Verify that messages were converted to prompt
                call_args = mock_client.models.generate_content.call_args
                combined_prompt = call_args[1]["contents"]

                # Should contain all message types
                assert "System: You are a helpful assistant." in combined_prompt
                assert "User: Hello there!" in combined_prompt
                assert "Assistant: Hello! How can I help you today?" in combined_prompt
                assert "User: How are you doing?" in combined_prompt


class TestGeminiModelComparison:
    """Test model comparison integration."""

    def test_model_comparison_integration(self):
        """Test integration of model comparison with adapter."""
        # Test model comparison
        models = ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"]

        comparison = compare_gemini_models(
            models=models, input_tokens=1000, output_tokens=500, sort_by="total_cost"
        )

        # Verify comparison results
        assert len(comparison) == 3

        # Should be sorted by cost (ascending)
        costs = [result["total_cost"] for result in comparison]
        assert costs == sorted(costs)

        # Verify all required fields are present
        for result in comparison:
            assert "model_id" in result
            assert "display_name" in result
            assert "total_cost" in result
            assert "cost_per_1k_tokens" in result
            assert "tier" in result

        # Flash-Lite should be cheapest, Pro most expensive
        cheapest = comparison[0]
        most_expensive = comparison[-1]

        assert "flash-lite" in cheapest["model_id"].lower()
        assert "pro" in most_expensive["model_id"].lower()


class TestGeminiValidationIntegration:
    """Test validation integration with other components."""

    @patch("genops.providers.gemini_validation.GEMINI_AVAILABLE", True)
    @patch("genops.providers.gemini_validation.GENOPS_AVAILABLE", True)
    @patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyDVWsKuP8_test_key_format"})
    def test_validation_integration_success(self):
        """Test successful validation integration."""
        with patch(
            "genops.providers.gemini_validation.genai.Client"
        ) as mock_client_class:
            # Mock successful API responses
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "Hello from validation"
            mock_client.models.generate_content.return_value = mock_response
            mock_client_class.return_value = mock_client

            # Run validation
            result = validate_setup()

            # Should pass all checks
            assert result.success is True
            assert result.get_error_count() == 0

            # Should have generated recommendations
            assert len(result.recommendations) > 0

            # Should have environment info
            assert result.environment_info["gemini_sdk_available"] is True
            assert result.environment_info["genops_available"] is True
            assert result.environment_info["api_key_env_set"] is True

    def test_validation_print_integration(self, capsys):
        """Test validation result printing integration."""
        # Create a realistic validation result
        from genops.providers.gemini_validation import (
            GeminiValidationResult,
            ValidationCheck,
            ValidationLevel,
        )

        result = GeminiValidationResult(
            success=True,
            checks=[
                ValidationCheck("gemini_sdk", ValidationLevel.SUCCESS, "SDK available"),
                ValidationCheck(
                    "api_key", ValidationLevel.SUCCESS, "API key configured"
                ),
                ValidationCheck(
                    "connectivity", ValidationLevel.SUCCESS, "API connectivity OK"
                ),
            ],
            recommendations=["✅ Gemini setup is ready for production use"],
            performance_metrics={"connectivity_latency_ms": 650},
            environment_info={"api_key_env_set": True},
        )

        print_validation_result(result, detailed=True)

        captured = capsys.readouterr()

        # Verify key elements are printed
        assert "OVERALL STATUS: PASSED" in captured.out
        assert "SDK available" in captured.out
        assert "API key configured" in captured.out
        assert "PERFORMANCE METRICS" in captured.out
        assert "connectivity_latency_ms: 650" in captured.out
        assert "production use" in captured.out


class TestGeminiErrorHandling:
    """Test error handling across components."""

    @patch("genops.providers.gemini.GEMINI_AVAILABLE", True)
    @patch("genops.providers.gemini.GENOPS_AVAILABLE", True)
    def test_adapter_error_handling(self):
        """Test adapter error handling."""
        # Mock client that raises exception
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception(
            "API rate limit exceeded"
        )

        with patch("genops.providers.gemini.genai.Client", return_value=mock_client):
            adapter = GenOpsGeminiAdapter(api_key="test_key")

            # Should propagate exception
            with pytest.raises(Exception, match="API rate limit exceeded"):
                adapter.text_generation(prompt="Test prompt")

    @patch("genops.providers.gemini.GEMINI_AVAILABLE", True)
    @patch("genops.providers.gemini.GENOPS_AVAILABLE", True)
    def test_telemetry_error_handling(self):
        """Test telemetry error handling."""
        # Mock client that works
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Success response"
        mock_client.models.generate_content.return_value = mock_response

        # Mock telemetry that fails
        mock_telemetry = MagicMock()
        mock_span = MagicMock()
        mock_span.set_attributes.side_effect = Exception("Telemetry error")
        mock_telemetry.trace_operation.return_value.__enter__ = Mock(
            return_value=mock_span
        )
        mock_telemetry.trace_operation.return_value.__exit__ = Mock(return_value=None)

        with patch("genops.providers.gemini.genai.Client", return_value=mock_client):
            with patch(
                "genops.providers.gemini.GenOpsTelemetry", return_value=mock_telemetry
            ):
                with patch(
                    "genops.providers.gemini.calculate_gemini_cost", return_value=0.001
                ):
                    adapter = GenOpsGeminiAdapter(api_key="test_key")

                    # Should handle telemetry error gracefully
                    result = adapter.text_generation(prompt="Test prompt")

                    # Should still return valid result despite telemetry failure
                    assert result.content == "Success response"

    @patch("genops.providers.gemini_cost_aggregator.calculate_gemini_cost")
    def test_cost_context_error_handling(self, mock_calculate_cost):
        """Test cost context error handling."""
        mock_calculate_cost.side_effect = Exception("Cost calculation failed")

        context_id = "error_test_context"

        # Should handle cost calculation errors gracefully
        try:
            with create_gemini_cost_context(context_id) as context:
                # This should not fail even if cost calculation fails
                context.add_operation(
                    operation_id="test_op",
                    model_id="gemini-2.5-flash",
                    input_tokens=1000,
                    output_tokens=500,
                    latency_ms=800.0,
                )
        except Exception as e:
            # Should not propagate cost calculation errors
            pytest.fail(f"Context manager should handle cost calculation errors: {e}")


class TestGeminiRealWorldScenarios:
    """Test realistic usage scenarios."""

    @patch("genops.providers.gemini.GEMINI_AVAILABLE", True)
    @patch("genops.providers.gemini.GENOPS_AVAILABLE", True)
    def test_content_generation_pipeline(self):
        """Test realistic content generation pipeline."""
        # Mock responses for different steps
        responses = [
            "AI Revolution: Transforming Industries",  # Title
            "Artificial intelligence is rapidly transforming...",  # Content
            "Key takeaways: AI adoption is accelerating...",  # Summary
            "#AI #Technology #Innovation",  # Tags
        ]

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [
            MagicMock(text=resp) for resp in responses
        ]

        with patch("genops.providers.gemini.genai.Client", return_value=mock_client):
            with patch(
                "genops.providers.gemini.calculate_gemini_cost",
                side_effect=[0.0005, 0.003, 0.001, 0.0003],
            ):
                with create_gemini_cost_context(
                    "content_pipeline",
                    budget_limit=0.01,
                    team="content-marketing",
                    project="ai-blog-series",
                ) as context:
                    adapter = GenOpsGeminiAdapter(api_key="test_key")

                    # Step 1: Generate title
                    title = adapter.text_generation(
                        prompt="Create a compelling title for an AI article",
                        model="gemini-2.5-flash-lite",
                        customer_id="tech-blog",
                    )
                    context.add_operation(
                        "title_gen", "gemini-2.5-flash-lite", 50, 10, 400.0
                    )

                    # Step 2: Generate content
                    content = adapter.text_generation(
                        prompt=f"Write article content for: {title.content}",
                        model="gemini-2.5-pro",
                        max_tokens=800,
                        customer_id="tech-blog",
                    )
                    context.add_operation(
                        "content_gen", "gemini-2.5-pro", 200, 600, 2500.0
                    )

                    # Step 3: Generate summary
                    summary = adapter.text_generation(
                        prompt=f"Summarize this article: {content.content[:200]}...",
                        model="gemini-2.5-flash",
                        customer_id="tech-blog",
                    )
                    context.add_operation(
                        "summary_gen", "gemini-2.5-flash", 100, 50, 800.0
                    )

                    # Step 4: Generate tags
                    tags = adapter.text_generation(
                        prompt=f"Generate hashtags for: {title.content}",
                        model="gemini-2.5-flash-lite",
                        max_tokens=20,
                        customer_id="tech-blog",
                    )
                    context.add_operation(
                        "tags_gen", "gemini-2.5-flash-lite", 30, 8, 350.0
                    )

                    # Get final summary
                    final_summary = context.get_current_summary()

                # Verify pipeline results
                assert title.content == "AI Revolution: Transforming Industries"
                assert content.content.startswith("Artificial intelligence is rapidly")
                assert summary.content.startswith("Key takeaways:")
                assert tags.content == "#AI #Technology #Innovation"

                # Verify cost tracking
                assert final_summary.total_operations == 4
                assert final_summary.total_cost == 0.0048  # Sum of all costs

                # Verify model usage
                assert len(final_summary.unique_models) == 3
                assert "gemini-2.5-flash-lite" in final_summary.unique_models
                assert "gemini-2.5-pro" in final_summary.unique_models
                assert "gemini-2.5-flash" in final_summary.unique_models

                # Verify governance attributes
                assert (
                    final_summary.governance_attributes["team"] == "content-marketing"
                )
                assert (
                    final_summary.governance_attributes["project"] == "ai-blog-series"
                )


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
