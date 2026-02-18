#!/usr/bin/env python3
"""
Test Suite for Replicate Integration End-to-End Workflows

Integration tests covering complete workflows including:
- Full adapter initialization and configuration
- Multi-modal operations with cost tracking
- Integration with cost aggregator and validation
- Real-world scenario simulation
- Performance and scalability testing
- Cross-component interaction validation

Target: ~17 tests covering end-to-end integration scenarios
"""

import time
from unittest.mock import Mock, patch

import pytest
from src.genops.providers.replicate import (
    GenOpsReplicateAdapter,
    ReplicateResponse,
    auto_instrument,
    instrument_replicate,
)
from src.genops.providers.replicate_cost_aggregator import (
    create_replicate_cost_context,
)
from src.genops.providers.replicate_validation import quick_validate, validate_setup


class TestFullWorkflowIntegration:
    """Test complete multi-modal workflows with cost aggregation."""

    @pytest.fixture
    def mock_replicate_environment(self):
        """Setup complete mock environment for integration tests."""
        with (
            patch("src.genops.providers.replicate.replicate") as mock_replicate,
            patch(
                "src.genops.providers.replicate_validation.replicate"
            ) as mock_val_replicate,
        ):
            # Mock successful API responses
            mock_replicate.run.return_value = "Integration test response"
            mock_replicate.stream.return_value = iter(
                ["Integration", " test", " streaming"]
            )

            # Mock client for validation
            mock_client = Mock()
            mock_models = Mock()
            mock_models.list.return_value = ["model1", "model2"]
            mock_models.get.return_value = Mock()
            mock_client.models = mock_models
            mock_val_replicate.Client.return_value = mock_client

            # Mock pricing calculations
            pricing_patch = patch(
                "src.genops.providers.replicate_pricing.ReplicatePricingCalculator"
            )
            mock_pricing = pricing_patch.start()
            mock_pricing_instance = Mock()

            # Setup different pricing for different model types
            def mock_get_model_info(model_name):
                from src.genops.providers.replicate import ReplicateModelInfo

                if "llama" in model_name.lower():
                    return ReplicateModelInfo(
                        name=model_name,
                        pricing_type="token",
                        base_cost=0.0,
                        input_cost=0.5,
                        output_cost=0.5,
                        category="text",
                        official=True,
                    )
                elif "flux" in model_name.lower():
                    return ReplicateModelInfo(
                        name=model_name,
                        pricing_type="output",
                        base_cost=0.003,
                        category="image",
                        official=True,
                    )
                elif "whisper" in model_name.lower():
                    return ReplicateModelInfo(
                        name=model_name,
                        pricing_type="time",
                        base_cost=0.0001,
                        category="audio",
                        official=True,
                    )
                else:
                    return ReplicateModelInfo(
                        name=model_name,
                        pricing_type="time",
                        base_cost=0.001,
                        category="unknown",
                        official=False,
                    )

            mock_pricing_instance.get_model_info.side_effect = mock_get_model_info
            mock_pricing_instance.calculate_cost.return_value = 0.001234
            mock_pricing.return_value = mock_pricing_instance

            yield {
                "replicate": mock_replicate,
                "validation_replicate": mock_val_replicate,
                "pricing": mock_pricing_instance,
            }

            pricing_patch.stop()

    def test_complete_multimodal_workflow(self, mock_replicate_environment):
        """Test complete multi-modal workflow with cost aggregation."""

        with create_replicate_cost_context(
            "integration-workflow", budget_limit=1.0
        ) as context:
            adapter = GenOpsReplicateAdapter(api_token="r8_integration_test_token")

            # Text generation task
            with patch("time.time", side_effect=[1000, 1002]):
                text_response = adapter.text_generation(
                    model="meta/llama-2-7b-chat",
                    prompt="Generate marketing copy for AI platform",
                    max_tokens=100,
                    team="marketing-team",
                    project="ai-platform-launch",
                )

            assert isinstance(text_response, ReplicateResponse)
            assert text_response.model == "meta/llama-2-7b-chat"
            assert text_response.cost_usd == 0.001234

            # Add to cost context
            context.add_operation(
                model=text_response.model,
                category="text",
                cost_usd=text_response.cost_usd,
                latency_ms=text_response.latency_ms,
                team="marketing-team",
            )

            # Image generation task
            with patch("time.time", side_effect=[2000, 2003]):
                image_response = adapter.image_generation(
                    model="black-forest-labs/flux-schnell",
                    prompt="AI platform logo design",
                    num_images=2,
                    team="design-team",
                    project="ai-platform-launch",
                )

            assert isinstance(image_response, ReplicateResponse)
            assert image_response.model == "black-forest-labs/flux-schnell"

            # Add to cost context
            context.add_operation(
                model=image_response.model,
                category="image",
                cost_usd=image_response.cost_usd,
                output_units=2,
                latency_ms=image_response.latency_ms,
                team="design-team",
            )

            # Audio processing task
            with patch("time.time", side_effect=[3000, 3002.5]):
                audio_response = adapter.audio_processing(
                    model="openai/whisper",
                    audio_input="marketing_voiceover.wav",
                    team="content-team",
                    project="ai-platform-launch",
                )

            assert isinstance(audio_response, ReplicateResponse)
            assert audio_response.model == "openai/whisper"

            # Add to cost context
            context.add_operation(
                model=audio_response.model,
                category="audio",
                cost_usd=audio_response.cost_usd,
                latency_ms=audio_response.latency_ms,
                team="content-team",
            )

            # Verify complete workflow summary
            summary = context.get_current_summary()

            assert summary.operation_count == 3
            assert len(summary.unique_categories) == 3
            assert "text" in summary.unique_categories
            assert "image" in summary.unique_categories
            assert "audio" in summary.unique_categories

            # Should have cost breakdown by team
            team_costs = {}
            for operation in context.operations:
                team = operation.governance_attributes.get("team", "unknown")
                team_costs[team] = team_costs.get(team, 0) + operation.cost_usd

            assert "marketing-team" in team_costs
            assert "design-team" in team_costs
            assert "content-team" in team_costs

    def test_auto_instrumentation_integration(self, mock_replicate_environment):
        """Test auto-instrumentation integration with cost tracking."""

        # Enable auto-instrumentation
        auto_instrument()

        # Use raw replicate.run calls (should be automatically tracked)
        mock_replicate = mock_replicate_environment["replicate"]

        with patch("time.time", side_effect=[1000, 1001, 1002, 1003]):
            # These calls should be automatically instrumented
            result1 = mock_replicate.run(
                "meta/llama-2-7b-chat",
                input={"prompt": "Test auto-instrumentation", "max_length": 50},
                team="engineering-team",
                project="auto-instrumentation-test",
            )

            result2 = mock_replicate.run(
                "black-forest-labs/flux-schnell",
                input={"prompt": "Test image generation"},
                team="design-team",
            )

        # Verify calls were made (content returned from mocked responses)
        assert result1 == "Integration test response"
        assert result2 == "Integration test response"

        # Verify instrumentation was applied
        assert hasattr(mock_replicate, "_original_run")

    def test_streaming_integration(self, mock_replicate_environment):
        """Test streaming integration with cost tracking."""

        adapter = GenOpsReplicateAdapter(api_token="r8_streaming_test_token")

        with patch("time.time", side_effect=[1000, 1005]):
            # Test streaming text generation
            streaming_result = adapter.text_generation(
                model="meta/llama-2-13b-chat",
                prompt="Stream a detailed analysis of AI cost management",
                stream=True,
                team="research-team",
                project="streaming-analysis",
            )

            # Collect streaming chunks
            chunks = list(streaming_result)

            assert len(chunks) == 3
            assert chunks == ["Integration", " test", " streaming"]


class TestValidationIntegration:
    """Test integration between validation and other components."""

    @pytest.fixture
    def validation_environment(self):
        """Setup environment for validation integration tests."""
        with patch.dict(
            "os.environ", {"REPLICATE_API_TOKEN": "r8_validation_test_token"}
        ):
            yield

    @patch("src.genops.providers.replicate_validation.replicate")
    @patch("requests.get")
    def test_complete_validation_workflow(
        self, mock_requests, mock_replicate, validation_environment
    ):
        """Test complete validation workflow integration."""

        # Mock successful authentication
        mock_response = Mock()
        mock_response.status_code = 200
        mock_requests.return_value = mock_response

        # Mock API connectivity
        mock_client = Mock()
        mock_models = Mock()
        mock_models.list.return_value = ["model1", "model2", "model3"]
        mock_client.models = mock_models
        mock_replicate.Client.return_value = mock_client

        # Mock model availability
        mock_client.models.get.return_value = Mock()

        # Run complete validation
        with patch("time.time", side_effect=[1000, 1001, 1002, 1003, 1004, 1005]):
            result = validate_setup()

        assert result.success is True
        assert len(result.errors) == 0

        # Should have performance metrics
        assert result.performance_metrics is not None
        assert "api_latency_ms" in result.performance_metrics

        # Should have environment info
        assert result.environment_info is not None
        assert result.environment_info["replicate_token_set"] is True

        # Should have model availability results
        assert result.model_availability is not None
        assert len(result.model_availability) > 0

    @patch("src.genops.providers.replicate_validation.replicate", None)
    def test_validation_missing_dependencies(self, validation_environment):
        """Test validation with missing dependencies."""

        result = validate_setup()

        assert result.success is False
        assert any("not installed" in error for error in result.errors)

    def test_quick_validate_integration(self):
        """Test quick validation integration."""

        with patch(
            "src.genops.providers.replicate_validation.validate_setup"
        ) as mock_validate:
            # Test successful quick validation
            mock_validate.return_value = Mock(success=True)

            result = quick_validate()
            assert result is True

            # Test failed quick validation
            mock_validate.return_value = Mock(success=False)

            result = quick_validate()
            assert result is False


class TestErrorHandlingIntegration:
    """Test error handling across integrated components."""

    def test_adapter_error_propagation(self):
        """Test error propagation from adapter through aggregator."""

        with patch("src.genops.providers.replicate.replicate") as mock_replicate:
            # Mock API error
            mock_replicate.run.side_effect = Exception("API Rate Limit Exceeded")

            adapter = GenOpsReplicateAdapter(api_token="r8_error_test_token")

            with pytest.raises(Exception) as exc_info:
                adapter.text_generation(
                    model="meta/llama-2-7b-chat", prompt="Test error handling"
                )

            assert "API Rate Limit Exceeded" in str(exc_info.value)

    def test_cost_context_error_handling(self):
        """Test error handling within cost context manager."""

        with patch("src.genops.providers.replicate.replicate") as mock_replicate:
            mock_replicate.run.side_effect = Exception("Network Error")

            adapter = GenOpsReplicateAdapter(api_token="r8_context_error_test")

            with pytest.raises(Exception):  # noqa: B017
                with create_replicate_cost_context("error-context"):
                    # This should propagate the error
                    adapter.text_generation(
                        model="meta/llama-2-7b-chat",
                        prompt="Test context error handling",
                    )

    def test_graceful_degradation_without_pricing(self):
        """Test graceful degradation when pricing calculator unavailable."""

        with (
            patch("src.genops.providers.replicate.replicate") as mock_replicate,
            patch(
                "src.genops.providers.replicate.ReplicatePricingCalculator",
                side_effect=ImportError,
            ),
        ):
            mock_replicate.run.return_value = "Fallback response"

            adapter = GenOpsReplicateAdapter(api_token="r8_fallback_test")

            # Should still work with fallback pricing
            with patch("time.time", side_effect=[1000, 1001]):
                response = adapter.text_generation(
                    model="unknown/community-model", prompt="Test fallback behavior"
                )

            assert isinstance(response, ReplicateResponse)
            assert response.content == "Fallback response"
            # Should have some cost (fallback calculation)
            assert response.cost_usd > 0


class TestPerformanceIntegration:
    """Test performance characteristics of integrated components."""

    @pytest.fixture
    def performance_environment(self):
        """Setup high-performance test environment."""
        with patch("src.genops.providers.replicate.replicate") as mock_replicate:
            # Mock fast responses
            mock_replicate.run.return_value = "Fast response"

            # Mock pricing for performance
            with patch(
                "src.genops.providers.replicate_pricing.ReplicatePricingCalculator"
            ) as mock_calc:
                mock_instance = Mock()

                def fast_model_info(model_name):
                    from src.genops.providers.replicate import ReplicateModelInfo

                    return ReplicateModelInfo(
                        name=model_name,
                        pricing_type="token",
                        base_cost=0.001,
                        category="text",
                    )

                mock_instance.get_model_info.side_effect = fast_model_info
                mock_instance.calculate_cost.return_value = 0.001
                mock_calc.return_value = mock_instance

                yield mock_replicate

    def test_high_volume_operations(self, performance_environment):
        """Test performance with high volume of operations."""

        adapter = GenOpsReplicateAdapter(api_token="r8_performance_test")

        with create_replicate_cost_context("high-volume-test") as context:
            start_time = time.time()

            # Simulate 50 operations
            for i in range(50):
                with patch("time.time", side_effect=[i * 10, i * 10 + 0.1]):
                    response = adapter.text_generation(
                        model="meta/llama-2-7b-chat",
                        prompt=f"Operation {i}",
                        team=f"team-{i % 5}",  # Distribute across 5 teams
                        project="performance-test",
                    )

                context.add_operation(
                    model=response.model,
                    category="text",
                    cost_usd=response.cost_usd,
                    team=f"team-{i % 5}",
                )

            end_time = time.time()
            processing_time = end_time - start_time

            # Verify all operations completed
            summary = context.get_current_summary()
            assert summary.operation_count == 50
            assert len(summary.unique_models) == 1

            # Performance should be reasonable (< 5 seconds for mocked operations)
            assert processing_time < 5.0

            # Should have cost breakdown by team
            assert len(summary.cost_by_category) > 0

            # Verify efficiency metrics are calculated
            assert summary.efficiency_metrics is not None

    def test_memory_usage_with_large_context(self, performance_environment):
        """Test memory efficiency with large cost context."""

        GenOpsReplicateAdapter(api_token="r8_memory_test")

        with create_replicate_cost_context("large-context-test") as context:
            # Add many operations with varying data sizes
            for i in range(200):
                # Vary the governance attributes to test memory usage
                governance_attrs = {
                    "team": f"team-{i % 10}",
                    "project": f"project-{i % 20}",
                    "customer_id": f"customer-{i % 5}",
                    "environment": "performance-test",
                }

                context.add_operation(
                    model=f"model-{i % 3}",  # 3 different models
                    category="text",
                    cost_usd=0.001,
                    input_tokens=100 + (i % 50),  # Varying token counts
                    output_tokens=150 + (i % 75),
                    latency_ms=1000 + (i % 500),
                    **governance_attrs,
                )

            # Generate summary (this exercises memory-intensive operations)
            summary = context.get_current_summary()

            assert summary.operation_count == 200
            assert len(summary.cost_by_model) == 3
            assert len(summary.unique_models) == 3

            # Export should complete without memory issues
            export_data = context.export_summary()
            assert len(export_data["operations"]) == 200
            assert len(export_data["model_performance"]) == 3


class TestRealWorldScenarios:
    """Test realistic usage scenarios and workflows."""

    @pytest.fixture
    def realistic_environment(self):
        """Setup realistic test environment with varied responses."""
        with patch("src.genops.providers.replicate.replicate") as mock_replicate:
            # Define realistic responses for different model types
            responses = {
                "meta/llama-2-7b-chat": "AI cost management is essential for scaling AI operations efficiently and maintaining budget control across teams and projects.",
                "meta/llama-2-70b-chat": "Comprehensive AI cost management involves implementing governance frameworks, establishing team attribution systems, setting budget controls with real-time monitoring, and optimizing model selection based on task complexity versus cost trade-offs.",
                "black-forest-labs/flux-schnell": [
                    "https://example.com/generated_image_1.png"
                ],
                "black-forest-labs/flux-pro": [
                    "https://example.com/professional_image.png"
                ],
                "openai/whisper": "This is the transcribed content from the audio file containing important business information.",
            }

            def mock_run(model, input, **kwargs):
                return responses.get(model, f"Default response for {model}")

            mock_replicate.run.side_effect = mock_run

            # Mock pricing with realistic values
            with patch(
                "src.genops.providers.replicate_pricing.ReplicatePricingCalculator"
            ) as mock_calc:
                mock_instance = Mock()

                def realistic_model_info(model_name):
                    from src.genops.providers.replicate import ReplicateModelInfo

                    pricing_map = {
                        "meta/llama-2-7b-chat": ("token", 0.5, 0.5, "text", 0.0),
                        "meta/llama-2-70b-chat": ("token", 1.0, 1.0, "text", 0.0),
                        "black-forest-labs/flux-schnell": (
                            "output",
                            0.003,
                            None,
                            "image",
                            0.003,
                        ),
                        "black-forest-labs/flux-pro": (
                            "output",
                            0.04,
                            None,
                            "image",
                            0.04,
                        ),
                        "openai/whisper": ("time", 0.0001, None, "audio", 0.0001),
                    }

                    if model_name in pricing_map:
                        pricing_type, input_cost, output_cost, category, base_cost = (
                            pricing_map[model_name]
                        )
                        return ReplicateModelInfo(
                            name=model_name,
                            pricing_type=pricing_type,
                            base_cost=base_cost,
                            input_cost=input_cost,
                            output_cost=output_cost,
                            category=category,
                            official=True,
                        )
                    else:
                        return ReplicateModelInfo(
                            name=model_name,
                            pricing_type="time",
                            base_cost=0.001,
                            category="unknown",
                            official=False,
                        )

                def realistic_cost_calculation(
                    model_info, input_data, output, latency_ms
                ):
                    """Calculate realistic costs based on model type."""
                    if model_info.pricing_type == "token":
                        # Estimate tokens
                        prompt_tokens = len(str(input_data.get("prompt", ""))) // 4
                        output_tokens = len(str(output)) // 4 if output else 100

                        input_cost = (prompt_tokens / 1000) * (
                            model_info.input_cost or 0
                        )
                        output_cost = (output_tokens / 1000) * (
                            model_info.output_cost or 0
                        )
                        return input_cost + output_cost

                    elif model_info.pricing_type == "output":
                        num_outputs = input_data.get("num_outputs", 1)
                        return model_info.base_cost * num_outputs

                    elif model_info.pricing_type == "time":
                        time_seconds = latency_ms / 1000
                        return model_info.base_cost * time_seconds

                    return 0.001  # Fallback

                mock_instance.get_model_info.side_effect = realistic_model_info
                mock_instance.calculate_cost.side_effect = realistic_cost_calculation
                mock_calc.return_value = mock_instance

                yield mock_replicate

    def test_marketing_campaign_workflow(self, realistic_environment):
        """Test realistic marketing campaign workflow."""

        with create_replicate_cost_context(
            "marketing-campaign", budget_limit=5.0
        ) as context:
            adapter = GenOpsReplicateAdapter(api_token="r8_marketing_test")

            # Phase 1: Content strategy planning
            with patch("time.time", side_effect=[1000, 1002]):
                strategy_response = adapter.text_generation(
                    model="meta/llama-2-70b-chat",  # Use high-quality model for strategy
                    prompt="Create a comprehensive marketing strategy for an AI cost management platform targeting enterprise clients",
                    max_tokens=200,
                    team="marketing-strategy",
                    project="ai-platform-launch",
                    customer_id="internal-campaign",
                )

            context.add_operation(
                model=strategy_response.model,
                category="text",
                cost_usd=strategy_response.cost_usd,
                team="marketing-strategy",
            )

            # Phase 2: Visual asset creation
            visual_tasks = [
                "Professional banner for AI cost management platform",
                "Infographic showing cost savings with AI governance",
                "Social media visual highlighting key benefits",
            ]

            for i, visual_task in enumerate(visual_tasks):
                with patch("time.time", side_effect=[2000 + i, 2003 + i]):
                    visual_response = adapter.image_generation(
                        model="black-forest-labs/flux-pro",  # High quality for professional assets
                        prompt=visual_task,
                        num_images=1,
                        team="creative-design",
                        project="ai-platform-launch",
                        customer_id="internal-campaign",
                    )

                context.add_operation(
                    model=visual_response.model,
                    category="image",
                    cost_usd=visual_response.cost_usd,
                    output_units=1,
                    team="creative-design",
                )

            # Phase 3: Copy creation for different channels
            copy_tasks = [
                ("Website homepage copy", "meta/llama-2-13b-chat"),
                ("Email campaign subject lines", "meta/llama-2-7b-chat"),
                ("Blog post outline", "meta/llama-2-13b-chat"),
                ("Social media captions", "meta/llama-2-7b-chat"),
            ]

            for i, (copy_task, model) in enumerate(copy_tasks):
                with patch("time.time", side_effect=[3000 + i, 3002 + i]):
                    copy_response = adapter.text_generation(
                        model=model,
                        prompt=f"Write {copy_task.lower()} for AI cost management platform",
                        max_tokens=80,
                        team="content-creation",
                        project="ai-platform-launch",
                        customer_id="internal-campaign",
                    )

                context.add_operation(
                    model=copy_response.model,
                    category="text",
                    cost_usd=copy_response.cost_usd,
                    team="content-creation",
                )

            # Analyze campaign cost breakdown
            summary = context.get_current_summary()

            # Verify campaign structure
            assert (
                summary.operation_count == 8
            )  # 1 strategy + 3 visuals + 4 copy pieces
            assert len(summary.unique_categories) == 2  # text and image
            assert "text" in summary.unique_categories
            assert "image" in summary.unique_categories

            # Verify team attribution
            team_operations = {}
            for operation in context.operations:
                team = operation.governance_attributes.get("team", "unknown")
                team_operations[team] = team_operations.get(team, 0) + 1

            assert "marketing-strategy" in team_operations
            assert "creative-design" in team_operations
            assert "content-creation" in team_operations

            # Verify budget management
            assert summary.total_cost < 5.0  # Should stay within budget
            if summary.budget_status:
                assert summary.budget_status["budget_limit"] == 5.0
                assert summary.budget_status["percentage_used"] < 100

            # Should have optimization recommendations
            assert len(summary.optimization_recommendations) > 0

    def test_development_team_workflow(self, realistic_environment):
        """Test realistic development team workflow."""

        with create_replicate_cost_context("dev-team-workflow") as context:
            adapter = GenOpsReplicateAdapter(api_token="r8_dev_test")

            # Documentation generation
            doc_tasks = [
                "API documentation for cost tracking endpoints",
                "User guide for team attribution setup",
                "Troubleshooting guide for common issues",
            ]

            for i, doc_task in enumerate(doc_tasks):
                with patch("time.time", side_effect=[1000 + i * 10, 1002 + i * 10]):
                    doc_response = adapter.text_generation(
                        model="meta/llama-2-13b-chat",
                        prompt=f"Generate technical documentation: {doc_task}",
                        max_tokens=150,
                        team="engineering",
                        project="documentation-sprint",
                        environment="development",
                    )

                context.add_operation(
                    model=doc_response.model,
                    category="text",
                    cost_usd=doc_response.cost_usd,
                    team="engineering",
                )

            # Code review assistance
            code_review_tasks = [
                "Review cost calculation logic for accuracy",
                "Suggest improvements for token counting algorithm",
                "Identify potential performance bottlenecks",
            ]

            for i, review_task in enumerate(code_review_tasks):
                with patch("time.time", side_effect=[2000 + i * 10, 2003 + i * 10]):
                    review_response = adapter.text_generation(
                        model="meta/llama-2-70b-chat",  # Use more capable model for code review
                        prompt=f"Code review task: {review_task}",
                        max_tokens=120,
                        team="engineering",
                        project="code-quality",
                        environment="development",
                    )

                context.add_operation(
                    model=review_response.model,
                    category="text",
                    cost_usd=review_response.cost_usd,
                    team="engineering",
                )

            # Verify development workflow summary
            summary = context.get_current_summary()

            assert summary.operation_count == 6
            assert len(summary.unique_models) >= 2  # Different models used
            assert summary.cost_by_category["text"] > 0

            # All operations should be attributed to engineering team
            engineering_cost = 0
            for operation in context.operations:
                if operation.governance_attributes.get("team") == "engineering":
                    engineering_cost += operation.cost_usd

            assert engineering_cost == summary.total_cost


class TestInstrumentReplicateFunction:
    """Test the instrument_replicate convenience function."""

    def test_instrument_replicate_basic(self):
        """Test basic instrument_replicate function usage."""

        adapter = instrument_replicate(api_token="r8_convenience_test")

        assert isinstance(adapter, GenOpsReplicateAdapter)
        assert adapter.api_token == "r8_convenience_test"

    def test_instrument_replicate_with_options(self):
        """Test instrument_replicate with additional options."""

        adapter = instrument_replicate(
            api_token="r8_options_test", telemetry_enabled=False, debug=True
        )

        assert adapter.telemetry_enabled is False
        assert adapter.debug is True

    def test_instrument_replicate_env_token(self):
        """Test instrument_replicate using environment token."""

        with patch.dict("os.environ", {"REPLICATE_API_TOKEN": "r8_env_token_test"}):
            adapter = instrument_replicate()

            assert adapter.api_token == "r8_env_token_test"
