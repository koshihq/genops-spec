#!/usr/bin/env python3
"""
Test Suite for ReplicatePricingCalculator

Unit tests covering pricing calculations for all model types including:
- Official model pricing database accuracy
- Multi-modal cost calculations (text, image, video, audio)
- Hardware-based pricing models
- Cost estimation for unknown/community models
- Optimization recommendations
- Model alternatives and comparisons

Target: ~30 tests covering all pricing scenarios
"""


import pytest
from src.genops.providers.replicate import ReplicateModelInfo
from src.genops.providers.replicate_pricing import (
    ReplicatePricingCalculator,
)


class TestReplicatePricingCalculatorInitialization:
    """Test pricing calculator initialization and setup."""

    def test_calculator_initialization(self):
        """Test basic calculator initialization."""
        calculator = ReplicatePricingCalculator()

        assert calculator is not None
        assert calculator.use_cache is True
        assert len(calculator._pricing_cache) > 0  # Should have official models loaded

    def test_calculator_with_cache_disabled(self):
        """Test calculator initialization with cache disabled."""
        calculator = ReplicatePricingCalculator(use_cache=False)

        assert calculator.use_cache is False

    def test_official_models_loaded(self):
        """Test that official models are loaded into cache."""
        calculator = ReplicatePricingCalculator()

        # Check for key official models
        expected_models = [
            "meta/llama-2-70b-chat",
            "meta/llama-2-13b-chat",
            "black-forest-labs/flux-pro",
            "black-forest-labs/flux-schnell",
            "google/veo-2",
            "openai/whisper"
        ]

        for model in expected_models:
            assert model in calculator._pricing_cache

    def test_hardware_pricing_loaded(self):
        """Test that hardware pricing is available."""
        calculator = ReplicatePricingCalculator()

        expected_hardware = ["cpu", "t4", "a100-40gb", "a100-80gb", "h100"]

        for hardware in expected_hardware:
            assert hardware in calculator.hardware_pricing
            assert calculator.hardware_pricing[hardware] > 0

class TestOfficialModelPricing:
    """Test pricing for official Replicate models."""

    @pytest.fixture
    def calculator(self):
        return ReplicatePricingCalculator()

    def test_llama_2_70b_pricing(self, calculator):
        """Test Llama 2 70B model pricing."""
        model_info = calculator.get_model_info("meta/llama-2-70b-chat")

        assert model_info.name == "meta/llama-2-70b-chat"
        assert model_info.pricing_type == "token"
        assert model_info.input_cost == 1.0  # $1.00 per 1K tokens
        assert model_info.output_cost == 1.0  # $1.00 per 1K tokens
        assert model_info.category == "text"
        assert model_info.official is True

    def test_flux_pro_pricing(self, calculator):
        """Test FLUX Pro image model pricing."""
        model_info = calculator.get_model_info("black-forest-labs/flux-pro")

        assert model_info.name == "black-forest-labs/flux-pro"
        assert model_info.pricing_type == "output"
        assert model_info.base_cost == 0.04  # $0.04 per image
        assert model_info.category == "image"
        assert model_info.official is True

    def test_veo_2_pricing(self, calculator):
        """Test Google Veo-2 video model pricing."""
        model_info = calculator.get_model_info("google/veo-2")

        assert model_info.name == "google/veo-2"
        assert model_info.pricing_type == "output"
        assert model_info.base_cost == 0.5  # $0.50 per second of video
        assert model_info.category == "video"
        assert model_info.official is True

    def test_whisper_pricing(self, calculator):
        """Test OpenAI Whisper audio model pricing."""
        model_info = calculator.get_model_info("openai/whisper")

        assert model_info.name == "openai/whisper"
        assert model_info.pricing_type == "time"
        assert model_info.base_cost == 0.0001
        assert model_info.category == "audio"
        assert model_info.official is True

class TestModelEstimation:
    """Test cost estimation for unknown/community models."""

    @pytest.fixture
    def calculator(self):
        return ReplicatePricingCalculator()

    def test_unknown_text_model_estimation(self, calculator):
        """Test estimation for unknown text model."""
        model_info = calculator.get_model_info("community/unknown-chat-model")

        assert model_info.name == "community/unknown-chat-model"
        assert model_info.category == "text"
        assert model_info.pricing_type == "token"
        assert model_info.base_cost == 0.5  # Default text model cost
        assert model_info.official is False

    def test_unknown_image_model_estimation(self, calculator):
        """Test estimation for unknown image model."""
        model_info = calculator.get_model_info("community/custom-diffusion-model")

        assert model_info.name == "community/custom-diffusion-model"
        assert model_info.category == "image"
        assert model_info.pricing_type == "output"
        assert model_info.base_cost == 0.01  # Default image cost
        assert model_info.official is False

    def test_model_size_based_hardware_estimation(self, calculator):
        """Test hardware estimation based on model size indicators."""
        # Large model should get high-end hardware
        large_model = calculator.get_model_info("community/huge-70b-model")
        assert large_model.hardware_type == "a100-40gb"

        # Medium model should get mid-tier hardware
        medium_model = calculator.get_model_info("community/medium-13b-model")
        assert medium_model.hardware_type == "t4"

        # Small/unknown model should get basic hardware
        small_model = calculator.get_model_info("community/simple-model")
        assert small_model.hardware_type == "cpu"

    def test_category_pattern_matching(self, calculator):
        """Test model category detection from name patterns."""
        test_cases = [
            ("community/video-generator-pro", "video"),
            ("user/audio-transcriber", "audio"),
            ("org/stable-diffusion-xl", "image"),
            ("creator/claude-alternative", "text")
        ]

        for model_name, expected_category in test_cases:
            model_info = calculator.get_model_info(model_name)
            assert model_info.category == expected_category

class TestTokenBasedCostCalculation:
    """Test cost calculations for token-based models."""

    @pytest.fixture
    def calculator(self):
        return ReplicatePricingCalculator()

    @pytest.fixture
    def text_model_info(self):
        return ReplicateModelInfo(
            name="meta/llama-2-13b-chat",
            pricing_type="token",
            base_cost=0.0,
            input_cost=0.5,  # $0.50 per 1K input tokens
            output_cost=0.5,  # $0.50 per 1K output tokens
            category="text"
        )

    def test_basic_token_cost_calculation(self, calculator, text_model_info):
        """Test basic token-based cost calculation."""
        input_data = {"prompt": "This is a test prompt with multiple words"}
        output = "This is a generated response with several words as well"
        latency_ms = 2000.0

        breakdown = calculator.calculate_cost_breakdown(
            text_model_info, input_data, output, latency_ms
        )

        assert breakdown.total_cost > 0
        assert breakdown.input_cost > 0
        assert breakdown.output_cost > 0
        assert breakdown.input_tokens > 0
        assert breakdown.output_tokens > 0

        # Verify cost calculation logic
        expected_input_cost = (breakdown.input_tokens / 1000) * 0.5
        expected_output_cost = (breakdown.output_tokens / 1000) * 0.5

        assert abs(breakdown.input_cost - expected_input_cost) < 0.0001
        assert abs(breakdown.output_cost - expected_output_cost) < 0.0001

    def test_empty_output_token_estimation(self, calculator, text_model_info):
        """Test token estimation with empty or None output."""
        input_data = {"prompt": "Test prompt"}
        output = None
        latency_ms = 1000.0

        breakdown = calculator.calculate_cost_breakdown(
            text_model_info, input_data, output, latency_ms
        )

        # Should use default token estimate
        assert breakdown.output_tokens == 100
        assert breakdown.output_cost > 0

    def test_large_token_count_handling(self, calculator, text_model_info):
        """Test handling of large token counts."""
        # Create large input/output
        large_prompt = "word " * 2000  # ~2000 tokens
        large_output = "response " * 1500  # ~1500 tokens

        input_data = {"prompt": large_prompt}
        latency_ms = 10000.0

        breakdown = calculator.calculate_cost_breakdown(
            text_model_info, input_data, large_output, latency_ms
        )

        # Should handle large numbers correctly
        assert breakdown.input_tokens > 1500  # Rough estimate
        assert breakdown.output_tokens > 1000  # Rough estimate
        assert breakdown.total_cost > 1.0  # Should be expensive

class TestOutputBasedCostCalculation:
    """Test cost calculations for output-based models (images, videos)."""

    @pytest.fixture
    def calculator(self):
        return ReplicatePricingCalculator()

    @pytest.fixture
    def image_model_info(self):
        return ReplicateModelInfo(
            name="black-forest-labs/flux-schnell",
            pricing_type="output",
            base_cost=0.003,  # $0.003 per image
            category="image"
        )

    @pytest.fixture
    def video_model_info(self):
        return ReplicateModelInfo(
            name="google/veo-2",
            pricing_type="output",
            base_cost=0.5,  # $0.50 per second
            category="video"
        )

    def test_single_image_cost(self, calculator, image_model_info):
        """Test cost calculation for single image generation."""
        input_data = {"prompt": "Generate a test image", "num_outputs": 1}
        output = ["generated_image_url.png"]
        latency_ms = 3000.0

        breakdown = calculator.calculate_cost_breakdown(
            image_model_info, input_data, output, latency_ms
        )

        assert breakdown.total_cost == 0.003  # $0.003 per image
        assert breakdown.output_cost == 0.003
        assert breakdown.output_units == 1

    def test_multiple_image_cost(self, calculator, image_model_info):
        """Test cost calculation for multiple images."""
        input_data = {"prompt": "Generate test images", "num_outputs": 5}
        output = ["img1.png", "img2.png", "img3.png", "img4.png", "img5.png"]
        latency_ms = 8000.0

        breakdown = calculator.calculate_cost_breakdown(
            image_model_info, input_data, output, latency_ms
        )

        assert breakdown.total_cost == 0.015  # 5 * $0.003
        assert breakdown.output_units == 5

    def test_video_duration_cost(self, calculator, video_model_info):
        """Test cost calculation for video generation by duration."""
        input_data = {"prompt": "Generate test video", "duration": 10.0}
        output = ["generated_video.mp4"]
        latency_ms = 30000.0

        breakdown = calculator.calculate_cost_breakdown(
            video_model_info, input_data, output, latency_ms
        )

        assert breakdown.total_cost == 5.0  # 10 seconds * $0.50/second
        assert breakdown.output_units == 10  # Duration in seconds

    def test_video_default_duration(self, calculator, video_model_info):
        """Test video cost with default duration when not specified."""
        input_data = {"prompt": "Generate test video"}  # No duration specified
        output = ["generated_video.mp4"]
        latency_ms = 15000.0

        breakdown = calculator.calculate_cost_breakdown(
            video_model_info, input_data, output, latency_ms
        )

        assert breakdown.total_cost == 2.5  # Default 5 seconds * $0.50/second
        assert breakdown.output_units == 5

class TestTimeBasedCostCalculation:
    """Test cost calculations for time-based models."""

    @pytest.fixture
    def calculator(self):
        return ReplicatePricingCalculator()

    @pytest.fixture
    def audio_model_info(self):
        return ReplicateModelInfo(
            name="openai/whisper",
            pricing_type="time",
            base_cost=0.0001,  # $0.0001 per second
            hardware_type="cpu",
            category="audio"
        )

    def test_time_based_cost_calculation(self, calculator, audio_model_info):
        """Test time-based cost calculation."""
        input_data = {"audio": "audio_file.wav"}
        output = "Transcribed audio content"
        latency_ms = 5000.0  # 5 seconds processing

        breakdown = calculator.calculate_cost_breakdown(
            audio_model_info, input_data, output, latency_ms
        )

        assert breakdown.total_cost == 0.0005  # 5 seconds * $0.0001/second
        assert breakdown.hardware_cost == 0.0005
        assert breakdown.time_seconds == 5.0

    def test_hardware_fallback_pricing(self, calculator):
        """Test fallback to hardware pricing when model cost not available."""
        model_info = ReplicateModelInfo(
            name="unknown/audio-model",
            pricing_type="time",
            base_cost=None,  # No specific cost
            hardware_type="t4",
            category="audio"
        )

        input_data = {"audio": "test.wav"}
        output = "Transcription"
        latency_ms = 3000.0  # 3 seconds

        breakdown = calculator.calculate_cost_breakdown(
            model_info, input_data, output, latency_ms
        )

        # Should use T4 hardware rate
        t4_rate = calculator.hardware_pricing["t4"]
        expected_cost = 3.0 * t4_rate

        assert abs(breakdown.total_cost - expected_cost) < 0.0001
        assert breakdown.hardware_cost == expected_cost

class TestHybridCostCalculation:
    """Test cost calculations for hybrid pricing models."""

    @pytest.fixture
    def calculator(self):
        return ReplicatePricingCalculator()

    @pytest.fixture
    def hybrid_model_info(self):
        return ReplicateModelInfo(
            name="custom/hybrid-model",
            pricing_type="hybrid",
            base_cost=0.0,
            input_cost=1.0,
            output_cost=1.0,
            hardware_type="a100-40gb",
            category="text"
        )

    def test_hybrid_cost_calculation(self, calculator, hybrid_model_info):
        """Test hybrid pricing with both token and time costs."""
        input_data = {"prompt": "Complex analysis request requiring both tokens and processing time"}
        output = "Detailed analysis response with comprehensive information and insights"
        latency_ms = 8000.0  # 8 seconds processing

        breakdown = calculator.calculate_cost_breakdown(
            hybrid_model_info, input_data, output, latency_ms
        )

        # Should have both token costs and hardware costs
        assert breakdown.input_cost > 0
        assert breakdown.output_cost > 0
        assert breakdown.hardware_cost > 0
        assert breakdown.total_cost == breakdown.input_cost + breakdown.output_cost + breakdown.hardware_cost

class TestOptimizationRecommendations:
    """Test cost optimization recommendation generation."""

    @pytest.fixture
    def calculator(self):
        return ReplicatePricingCalculator()

    def test_high_token_usage_recommendations(self, calculator):
        """Test recommendations for high token usage."""
        model_info = ReplicateModelInfo(
            name="meta/llama-2-70b-chat",
            pricing_type="token",
            base_cost=0.0,
            input_cost=1.0,
            output_cost=1.0,
            category="text"
        )

        # Large input/output to trigger recommendations
        large_input = {"prompt": "word " * 3000}  # Very large prompt
        large_output = "response " * 2000  # Very large response
        latency_ms = 45000.0  # Long processing time

        breakdown = calculator.calculate_cost_breakdown(
            model_info, large_input, large_output, latency_ms
        )

        recommendations = breakdown.optimization_suggestions

        # Should suggest optimizations for large inputs/outputs
        assert len(recommendations) > 0
        assert any("large prompts" in rec.lower() for rec in recommendations)
        assert any("limit response" in rec.lower() or "max_tokens" in rec.lower() for rec in recommendations)

    def test_high_cost_operation_recommendations(self, calculator):
        """Test recommendations for expensive operations."""
        model_info = ReplicateModelInfo(
            name="expensive/premium-model",
            pricing_type="output",
            base_cost=2.0,  # Very expensive per output
            category="video"
        )

        input_data = {"prompt": "Generate expensive video", "duration": 30}
        output = ["expensive_video.mp4"]
        latency_ms = 60000.0

        breakdown = calculator.calculate_cost_breakdown(
            model_info, input_data, output, latency_ms
        )

        recommendations = breakdown.optimization_suggestions

        # Should warn about high cost
        assert any("high cost" in rec.lower() for rec in recommendations)

    def test_batch_processing_recommendations(self, calculator):
        """Test recommendations for batch processing efficiency."""
        model_info = ReplicateModelInfo(
            name="black-forest-labs/flux-schnell",
            pricing_type="output",
            base_cost=0.003,
            category="image"
        )

        # Many images to trigger batch recommendation
        input_data = {"prompt": "Generate images", "num_outputs": 10}
        output = [f"image_{i}.png" for i in range(10)]
        latency_ms = 15000.0

        breakdown = calculator.calculate_cost_breakdown(
            model_info, input_data, output, latency_ms
        )

        recommendations = breakdown.optimization_suggestions

        # Should suggest batch processing for multiple outputs
        assert any("batch" in rec.lower() for rec in recommendations)

class TestModelAlternatives:
    """Test model alternative suggestions for cost optimization."""

    @pytest.fixture
    def calculator(self):
        return ReplicatePricingCalculator()

    def test_get_cheaper_alternatives(self, calculator):
        """Test finding cheaper alternatives to expensive models."""
        expensive_model = "meta/llama-2-70b-chat"  # Expensive model

        alternatives = calculator.get_model_alternatives(expensive_model, "text")

        # Should find cheaper text models
        assert len(alternatives) > 0

        for model_name, cost_ratio, reason in alternatives:
            assert cost_ratio < 1.0  # Should be cheaper
            assert "cost reduction" in reason.lower()

    def test_alternatives_same_category(self, calculator):
        """Test that alternatives are in the same category."""
        model = "black-forest-labs/flux-pro"  # Image model

        alternatives = calculator.get_model_alternatives(model, "image")

        # All alternatives should be image models
        for model_name, cost_ratio, reason in alternatives:
            alternative_info = calculator.get_model_info(model_name)
            assert alternative_info.category == "image"

    def test_no_alternatives_for_cheapest_model(self, calculator):
        """Test that cheapest models return fewer/no alternatives."""
        cheap_model = "black-forest-labs/flux-schnell"  # Already cheap

        alternatives = calculator.get_model_alternatives(cheap_model, "image")

        # Should have fewer alternatives since it's already cheap
        assert len(alternatives) <= 1

class TestCostPrecision:
    """Test cost calculation precision and rounding."""

    @pytest.fixture
    def calculator(self):
        return ReplicatePricingCalculator()

    def test_cost_precision_rounding(self, calculator):
        """Test that costs are properly rounded to reasonable precision."""
        model_info = ReplicateModelInfo(
            name="test-model",
            pricing_type="token",
            base_cost=0.0,
            input_cost=0.001234567,  # Very precise cost
            output_cost=0.001234567,
            category="text"
        )

        input_data = {"prompt": "test"}
        output = "response"
        latency_ms = 1000.0

        breakdown = calculator.calculate_cost_breakdown(
            model_info, input_data, output, latency_ms
        )

        # Total cost should be rounded to 6 decimal places
        cost_str = f"{breakdown.total_cost:.6f}"
        decimal_places = len(cost_str.split('.')[-1])
        assert decimal_places <= 6

    def test_zero_cost_handling(self, calculator):
        """Test handling of zero or very small costs."""
        model_info = ReplicateModelInfo(
            name="free-model",
            pricing_type="token",
            base_cost=0.0,
            input_cost=0.0,
            output_cost=0.0,
            category="text"
        )

        input_data = {"prompt": "test"}
        output = "response"
        latency_ms = 1000.0

        breakdown = calculator.calculate_cost_breakdown(
            model_info, input_data, output, latency_ms
        )

        assert breakdown.total_cost == 0.0
        assert breakdown.input_cost == 0.0
        assert breakdown.output_cost == 0.0
