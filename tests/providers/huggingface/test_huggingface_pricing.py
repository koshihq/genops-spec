"""
Unit tests for Hugging Face pricing and cost calculation.

Tests the cost calculation engine including:
- Provider detection accuracy
- Model size categorization
- Cost calculation for different providers and tasks
- Model comparison and optimization features
- Edge cases and error handling
"""

import os
import sys

import pytest

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))


class TestProviderDetection:
    """Test suite for provider detection functionality."""

    def test_detect_openai_models(self):
        """Test detection of OpenAI models."""
        from genops.providers.huggingface_pricing import detect_model_provider

        openai_models = [
            'gpt-3.5-turbo',
            'gpt-4',
            'gpt-4-turbo',
            'dall-e-2',
            'dall-e-3',
            'text-embedding-ada-002',
            'whisper-1'
        ]

        for model in openai_models:
            provider = detect_model_provider(model)
            assert provider == 'openai', f"Failed to detect OpenAI for {model}"

    def test_detect_anthropic_models(self):
        """Test detection of Anthropic models."""
        from genops.providers.huggingface_pricing import detect_model_provider

        anthropic_models = [
            'claude-3-opus',
            'claude-3-sonnet',
            'claude-3-haiku',
            'claude-2.1',
            'claude-instant-1.2'
        ]

        for model in anthropic_models:
            provider = detect_model_provider(model)
            assert provider == 'anthropic', f"Failed to detect Anthropic for {model}"

    def test_detect_cohere_models(self):
        """Test detection of Cohere models."""
        from genops.providers.huggingface_pricing import detect_model_provider

        cohere_models = [
            'command-r',
            'command-light',
            'embed-english-v3.0',
            'embed-multilingual-v3.0'
        ]

        for model in cohere_models:
            provider = detect_model_provider(model)
            assert provider == 'cohere', f"Failed to detect Cohere for {model}"

    def test_detect_meta_models(self):
        """Test detection of Meta/Facebook models."""
        from genops.providers.huggingface_pricing import detect_model_provider

        meta_models = [
            'llama-2-7b-chat',
            'llama-3-8b-instruct',
            'meta-llama/Llama-2-70b-hf',
            'code-llama-34b-instruct'
        ]

        for model in meta_models:
            provider = detect_model_provider(model)
            assert provider == 'meta', f"Failed to detect Meta for {model}"

    def test_detect_mistral_models(self):
        """Test detection of Mistral models."""
        from genops.providers.huggingface_pricing import detect_model_provider

        mistral_models = [
            'mistral-7b-instruct',
            'mixtral-8x7b-instruct',
            'mistral-large'
        ]

        for model in mistral_models:
            provider = detect_model_provider(model)
            assert provider == 'mistral', f"Failed to detect Mistral for {model}"

    def test_detect_google_models(self):
        """Test detection of Google models."""
        from genops.providers.huggingface_pricing import detect_model_provider

        google_models = [
            'gemma-7b-it',
            'flan-t5-xxl'
        ]

        for model in google_models:
            provider = detect_model_provider(model)
            assert provider == 'google', f"Failed to detect Google for {model}"

    def test_detect_huggingface_hub_models(self):
        """Test detection of Hugging Face Hub models."""
        from genops.providers.huggingface_pricing import detect_model_provider

        hub_models = [
            'microsoft/DialoGPT-medium',
            'sentence-transformers/all-MiniLM-L6-v2',
            'facebook/bart-large-cnn',
            'google/flan-t5-base',
            'runwayml/stable-diffusion-v1-5'
        ]

        for model in hub_models:
            provider = detect_model_provider(model)
            assert provider == 'huggingface_hub', f"Failed to detect Hub model for {model}"

    def test_detect_unknown_models(self):
        """Test detection defaults to huggingface_hub for unknown models."""
        from genops.providers.huggingface_pricing import detect_model_provider

        unknown_models = [
            'unknown-model-123',
            'some-random-model',
            ''
        ]

        for model in unknown_models:
            provider = detect_model_provider(model)
            assert provider == 'huggingface_hub', f"Should default to huggingface_hub for {model}"


class TestModelSizeCategorization:
    """Test suite for model size categorization."""

    def test_large_model_detection(self):
        """Test detection of large models."""
        from genops.providers.huggingface_pricing import estimate_model_size_category

        large_models = [
            'llama-3.1-405b-instruct',
            'gpt-3-175b',
            'llama-2-70b-chat',
            'model-65b-base'
        ]

        for model in large_models:
            category = estimate_model_size_category(model)
            assert category == 'large_models', f"Failed to categorize {model} as large"

    def test_medium_model_detection(self):
        """Test detection of medium models."""
        from genops.providers.huggingface_pricing import estimate_model_size_category

        medium_models = [
            'llama-2-13b-chat',
            'code-llama-34b-instruct',
            'model-20b-base',
            'llama-3-8b-instruct'
        ]

        for model in medium_models:
            category = estimate_model_size_category(model)
            assert category == 'medium_models', f"Failed to categorize {model} as medium"

    def test_small_model_detection(self):
        """Test detection of small models."""
        from genops.providers.huggingface_pricing import estimate_model_size_category

        small_models = [
            'gemma-2b-it',
            'model-3b-base',
            'tiny-1b-model'
        ]

        for model in small_models:
            category = estimate_model_size_category(model)
            assert category == 'small_models', f"Failed to categorize {model} as small"

    def test_embedding_model_detection(self):
        """Test detection of embedding models."""
        from genops.providers.huggingface_pricing import estimate_model_size_category

        embedding_models = [
            'sentence-transformers/all-MiniLM-L6-v2',
            'text-embedding-ada-002',
            'embed-english-v3.0'
        ]

        for model in embedding_models:
            category = estimate_model_size_category(model)
            assert category == 'embedding_models', f"Failed to categorize {model} as embedding"

    def test_image_model_detection(self):
        """Test detection of image models."""
        from genops.providers.huggingface_pricing import estimate_model_size_category

        image_models = [
            'runwayml/stable-diffusion-v1-5',
            'dall-e-2',
            'midjourney-v4',
            'stable-diffusion-xl'
        ]

        for model in image_models:
            category = estimate_model_size_category(model)
            assert category == 'image_models', f"Failed to categorize {model} as image model"

    def test_default_categorization(self):
        """Test default categorization for unclear model names."""
        from genops.providers.huggingface_pricing import estimate_model_size_category

        unclear_models = [
            'gpt-3.5-turbo',
            'claude-3-haiku',
            'mysterious-model'
        ]

        for model in unclear_models:
            category = estimate_model_size_category(model)
            assert category in ['small_models', 'medium_models'], f"Unexpected category for {model}: {category}"


class TestCostCalculation:
    """Test suite for cost calculation functionality."""

    def test_openai_cost_calculation(self):
        """Test cost calculation for OpenAI models."""
        from genops.providers.huggingface_pricing import calculate_huggingface_cost

        # Test GPT-3.5-turbo pricing
        cost = calculate_huggingface_cost(
            provider='openai',
            model='gpt-3.5-turbo',
            input_tokens=1000,
            output_tokens=500,
            task='text-generation'
        )

        # Expected: (1000/1000) * 0.0015 + (500/1000) * 0.002 = 0.0015 + 0.001 = 0.0025
        expected_cost = 0.0025
        assert abs(cost - expected_cost) < 0.000001, f"Expected {expected_cost}, got {cost}"

    def test_anthropic_cost_calculation(self):
        """Test cost calculation for Anthropic models."""
        from genops.providers.huggingface_pricing import calculate_huggingface_cost

        # Test Claude-3-haiku pricing
        cost = calculate_huggingface_cost(
            provider='anthropic',
            model='claude-3-haiku',
            input_tokens=1000,
            output_tokens=500,
            task='text-generation'
        )

        # Expected: (1000/1000) * 0.00025 + (500/1000) * 0.00125 = 0.00025 + 0.000625 = 0.000875
        expected_cost = 0.000875
        assert abs(cost - expected_cost) < 0.000001, f"Expected {expected_cost}, got {cost}"

    def test_huggingface_hub_cost_calculation(self):
        """Test cost calculation for Hugging Face Hub models."""
        from genops.providers.huggingface_pricing import calculate_huggingface_cost

        # Test medium Hub model pricing
        cost = calculate_huggingface_cost(
            provider='huggingface_hub',
            model='microsoft/DialoGPT-medium',
            input_tokens=1000,
            output_tokens=500,
            task='text-generation'
        )

        # Should use medium_models pricing: (1000/1000) * 0.00005 + (500/1000) * 0.0001 = 0.00005 + 0.00005 = 0.0001
        expected_cost = 0.0001
        assert abs(cost - expected_cost) < 0.000001, f"Expected {expected_cost}, got {cost}"

    def test_task_multiplier_application(self):
        """Test task-specific cost multipliers."""
        from genops.providers.huggingface_pricing import calculate_huggingface_cost

        base_cost = calculate_huggingface_cost(
            provider='openai',
            model='gpt-3.5-turbo',
            input_tokens=1000,
            output_tokens=500,
            task='text-generation'
        )

        # Test feature extraction (0.5x multiplier)
        embedding_cost = calculate_huggingface_cost(
            provider='openai',
            model='text-embedding-ada-002',
            input_tokens=1000,
            output_tokens=0,
            task='feature-extraction'
        )

        # Test text-to-image (10.0x multiplier)
        image_cost = calculate_huggingface_cost(
            provider='openai',
            model='dall-e-2',
            input_tokens=100,
            output_tokens=0,
            task='text-to-image'
        )

        # Embedding should be cheaper due to multiplier
        assert embedding_cost < base_cost

        # Image generation should be more expensive due to multiplier
        assert image_cost > base_cost

    def test_cost_calculation_edge_cases(self):
        """Test cost calculation edge cases."""
        from genops.providers.huggingface_pricing import calculate_huggingface_cost

        # Test zero tokens
        zero_cost = calculate_huggingface_cost(
            provider='openai',
            model='gpt-3.5-turbo',
            input_tokens=0,
            output_tokens=0
        )
        assert zero_cost == 0.0

        # Test unknown provider (should not fail)
        unknown_cost = calculate_huggingface_cost(
            provider='unknown-provider',
            model='unknown-model',
            input_tokens=100,
            output_tokens=50
        )
        assert isinstance(unknown_cost, float)
        assert unknown_cost >= 0

    def test_cost_calculation_with_images_and_audio(self):
        """Test cost calculation with image and audio parameters."""
        from genops.providers.huggingface_pricing import calculate_huggingface_cost

        # Test image generation cost
        image_cost = calculate_huggingface_cost(
            provider='openai',
            model='dall-e-2',
            input_tokens=50,
            output_tokens=0,
            task='text-to-image',
            images_generated=3
        )

        # Should consider both token cost and image count
        assert image_cost > 0

        # Test audio processing cost
        audio_cost = calculate_huggingface_cost(
            provider='openai',
            model='whisper-1',
            input_tokens=0,
            output_tokens=100,
            task='speech-to-text',
            audio_minutes=5
        )

        assert audio_cost > 0


class TestModelComparison:
    """Test suite for model comparison functionality."""

    def test_compare_model_costs_basic(self):
        """Test basic model cost comparison."""
        from genops.providers.huggingface_pricing import compare_model_costs

        models = ['gpt-3.5-turbo', 'claude-3-haiku', 'microsoft/DialoGPT-medium']

        comparison = compare_model_costs(
            models=models,
            input_tokens=1000,
            output_tokens=500
        )

        assert len(comparison) == 3

        for model in models:
            assert model in comparison
            assert 'cost' in comparison[model]
            assert 'provider' in comparison[model]
            assert 'relative_cost' in comparison[model]
            assert comparison[model]['cost'] >= 0

    def test_compare_model_costs_relative_calculation(self):
        """Test relative cost calculation in comparison."""
        from genops.providers.huggingface_pricing import compare_model_costs

        models = ['gpt-4', 'gpt-3.5-turbo']  # GPT-4 should be more expensive

        comparison = compare_model_costs(
            models=models,
            input_tokens=1000,
            output_tokens=500
        )

        gpt4_cost = comparison['gpt-4']['cost']
        gpt35_cost = comparison['gpt-3.5-turbo']['cost']

        # GPT-4 should be more expensive
        assert gpt4_cost > gpt35_cost

        # Relative costs should be calculated correctly
        min_cost = min(gpt4_cost, gpt35_cost)
        assert comparison['gpt-4']['relative_cost'] == gpt4_cost / min_cost
        assert comparison['gpt-3.5-turbo']['relative_cost'] == gpt35_cost / min_cost

    def test_compare_model_costs_with_task_type(self):
        """Test model comparison with different task types."""
        from genops.providers.huggingface_pricing import compare_model_costs

        embedding_models = [
            'text-embedding-ada-002',
            'sentence-transformers/all-MiniLM-L6-v2'
        ]

        comparison = compare_model_costs(
            models=embedding_models,
            input_tokens=1000,
            output_tokens=0,
            task='feature-extraction'
        )

        assert len(comparison) == 2

        # All costs should be relatively low for embedding tasks
        for model_info in comparison.values():
            assert model_info['cost'] < 0.01  # Should be less than 1 cent


class TestCostOptimization:
    """Test suite for cost optimization functionality."""

    def test_get_cost_optimization_suggestions_basic(self):
        """Test basic cost optimization suggestions."""
        from genops.providers.huggingface_pricing import (
            get_cost_optimization_suggestions,
        )

        suggestions = get_cost_optimization_suggestions('gpt-4', 'text-generation')

        assert 'current_model' in suggestions
        assert 'alternatives' in suggestions
        assert 'optimization_tips' in suggestions

        assert suggestions['current_model']['model'] == 'gpt-4'
        assert suggestions['current_model']['provider'] == 'openai'
        assert 'cost_per_1k' in suggestions['current_model']

        assert isinstance(suggestions['alternatives'], list)
        assert isinstance(suggestions['optimization_tips'], list)
        assert len(suggestions['optimization_tips']) > 0

    def test_get_cost_optimization_alternatives(self):
        """Test cost optimization alternatives generation."""
        from genops.providers.huggingface_pricing import (
            get_cost_optimization_suggestions,
        )

        # Use expensive model to ensure alternatives are found
        suggestions = get_cost_optimization_suggestions('gpt-4', 'text-generation')

        # Should find cheaper alternatives
        assert len(suggestions['alternatives']) > 0

        for alternative in suggestions['alternatives']:
            assert 'model' in alternative
            assert 'cost_per_1k' in alternative
            assert 'savings' in alternative
            assert alternative['savings'] > 0  # Should show positive savings

    def test_optimization_tips_content(self):
        """Test optimization tips content."""
        from genops.providers.huggingface_pricing import (
            get_cost_optimization_suggestions,
        )

        suggestions = get_cost_optimization_suggestions('gpt-3.5-turbo')

        tips = suggestions['optimization_tips']

        # Check for expected tip categories
        tips_text = ' '.join(tips).lower()

        expected_keywords = ['cost', 'model', 'hugging face', 'cache', 'usage']
        for keyword in expected_keywords:
            assert keyword in tips_text, f"Expected keyword '{keyword}' not found in tips"


class TestProviderInfo:
    """Test suite for provider information functionality."""

    def test_get_provider_info_basic(self):
        """Test basic provider information retrieval."""
        from genops.providers.huggingface_pricing import get_provider_info

        info = get_provider_info('gpt-3.5-turbo')

        assert 'provider' in info
        assert 'is_third_party' in info
        assert 'supports_streaming' in info
        assert 'supports_function_calling' in info
        assert 'cost_per_1k_tokens' in info
        assert 'cost_estimates' in info

        assert info['provider'] == 'openai'
        assert info['is_third_party'] is True

    def test_get_provider_info_huggingface_hub(self):
        """Test provider info for Hugging Face Hub models."""
        from genops.providers.huggingface_pricing import get_provider_info

        info = get_provider_info('microsoft/DialoGPT-medium')

        assert info['provider'] == 'huggingface_hub'
        assert info['is_third_party'] is False

    def test_provider_info_cost_estimates(self):
        """Test cost estimates in provider info."""
        from genops.providers.huggingface_pricing import get_provider_info

        info = get_provider_info('gpt-3.5-turbo')

        estimates = info['cost_estimates']

        assert 'short_chat' in estimates
        assert 'long_generation' in estimates
        assert 'embedding' in estimates

        # All estimates should be positive numbers
        for estimate_type, cost in estimates.items():
            assert isinstance(cost, (int, float))
            assert cost >= 0

    def test_provider_capabilities(self):
        """Test provider capability detection."""
        from genops.providers.huggingface_pricing import get_provider_info

        # Test OpenAI capabilities
        openai_info = get_provider_info('gpt-4')
        assert openai_info['supports_function_calling'] is True

        # Test Anthropic capabilities
        anthropic_info = get_provider_info('claude-3-sonnet')
        assert anthropic_info['supports_function_calling'] is True

        # Test Hub model capabilities
        hub_info = get_provider_info('microsoft/DialoGPT-medium')
        assert hub_info['supports_function_calling'] is False


class TestErrorHandling:
    """Test suite for error handling in pricing functions."""

    def test_calculate_cost_with_exception(self):
        """Test cost calculation handles exceptions gracefully."""
        from genops.providers.huggingface_pricing import calculate_huggingface_cost

        # Test with invalid parameters that might cause issues
        cost = calculate_huggingface_cost(
            provider='invalid-provider',
            model='invalid-model',
            input_tokens=-1,  # Negative tokens
            output_tokens=1000000,  # Very large number
            task='invalid-task'
        )

        # Should not raise exception, should return conservative estimate
        assert isinstance(cost, float)
        assert cost >= 0

    def test_get_model_pricing_fallback(self):
        """Test model pricing fallback for unknown models."""
        from genops.providers.huggingface_pricing import get_model_pricing

        # Test completely unknown provider and model
        pricing = get_model_pricing(
            provider='unknown-provider',
            model='unknown-model',
            task='unknown-task'
        )

        assert 'input' in pricing
        assert 'output' in pricing
        assert pricing['input'] >= 0
        assert pricing['output'] >= 0

    def test_provider_detection_edge_cases(self):
        """Test provider detection with edge cases."""
        from genops.providers.huggingface_pricing import detect_model_provider

        edge_cases = [
            None,
            '',
            '   ',  # Whitespace
            'https://example.com/model',  # URL
            'model with spaces',
            'model/with/many/slashes',
            '🤗/emoji-model'  # Unicode
        ]

        for model in edge_cases:
            # Should not raise exception
            provider = detect_model_provider(model)
            assert provider == 'huggingface_hub'  # Default fallback


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
