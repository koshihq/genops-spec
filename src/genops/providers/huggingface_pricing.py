"""Hugging Face cost calculation engine with multi-provider support."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Hugging Face Inference API pricing (per 1K tokens)
# Based on Hugging Face documentation and typical provider rates
HUGGINGFACE_PRICING = {
    # OpenAI models available through HF
    "openai": {
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "gpt-3.5-turbo-0125": {"input": 0.0005, "output": 0.0015},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-0125-preview": {"input": 0.01, "output": 0.03},
        "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
        "text-embedding-ada-002": {"input": 0.0001, "output": 0.0},
        "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
        "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
        "whisper-1": {
            "input": 0.006,
            "output": 0.0,
        },  # per minute, converted to token equiv
        # DALL-E pricing (per image, converted to token equivalent)
        "dall-e-2": {"input": 0.02, "output": 0.0},  # $0.020/image ≈ 10 tokens
        "dall-e-3": {"input": 0.04, "output": 0.0},  # $0.040/image ≈ 20 tokens
    },
    # Anthropic models
    "anthropic": {
        "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
        "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
        "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        "claude-2.1": {"input": 0.008, "output": 0.024},
        "claude-2.0": {"input": 0.008, "output": 0.024},
        "claude-instant-1.2": {"input": 0.0008, "output": 0.0024},
    },
    # Cohere models
    "cohere": {
        "command": {"input": 0.0015, "output": 0.002},
        "command-light": {"input": 0.0003, "output": 0.0006},
        "command-r": {"input": 0.0005, "output": 0.0015},
        "command-r-plus": {"input": 0.003, "output": 0.015},
        "embed-english-v3.0": {"input": 0.0001, "output": 0.0},
        "embed-multilingual-v3.0": {"input": 0.0001, "output": 0.0},
    },
    # Meta/Facebook models
    "meta": {
        "llama-2-7b-chat": {"input": 0.0002, "output": 0.0002},
        "llama-2-13b-chat": {"input": 0.0003, "output": 0.0003},
        "llama-2-70b-chat": {"input": 0.0007, "output": 0.0008},
        "code-llama-34b-instruct": {"input": 0.0005, "output": 0.0005},
        "llama-3-8b-instruct": {"input": 0.0002, "output": 0.0002},
        "llama-3-70b-instruct": {"input": 0.0009, "output": 0.0009},
        "llama-3.1-8b-instruct": {"input": 0.0002, "output": 0.0002},
        "llama-3.1-70b-instruct": {"input": 0.0009, "output": 0.0009},
        "llama-3.1-405b-instruct": {"input": 0.005, "output": 0.015},
    },
    # Mistral models
    "mistral": {
        "mistral-7b-instruct": {"input": 0.0002, "output": 0.0002},
        "mixtral-8x7b-instruct": {"input": 0.0007, "output": 0.0007},
        "mixtral-8x22b-instruct": {"input": 0.002, "output": 0.006},
        "mistral-small": {"input": 0.001, "output": 0.003},
        "mistral-medium": {"input": 0.0027, "output": 0.0081},
        "mistral-large": {"input": 0.004, "output": 0.012},
    },
    # Google models
    "google": {
        "gemma-7b-it": {"input": 0.0002, "output": 0.0002},
        "gemma-2b-it": {"input": 0.0001, "output": 0.0001},
        "flan-t5-xxl": {"input": 0.0003, "output": 0.0003},
        "flan-ul2": {"input": 0.0003, "output": 0.0003},
    },
    # Hugging Face Hub models (free tier + compute costs)
    "huggingface_hub": {
        # Free inference for popular models, minimal compute costs for others
        "small_models": {"input": 0.00001, "output": 0.00002},  # <1B params
        "medium_models": {"input": 0.00005, "output": 0.0001},  # 1B-10B params
        "large_models": {"input": 0.0001, "output": 0.0002},  # 10B+ params
        "embedding_models": {"input": 0.00001, "output": 0.0},
        "image_models": {"input": 0.001, "output": 0.0},  # per image generation
    },
}

# Task-specific pricing adjustments
TASK_MULTIPLIERS = {
    "text-generation": 1.0,
    "chat-completion": 1.0,
    "feature-extraction": 0.5,  # Embeddings typically cheaper
    "text-to-image": 10.0,  # Image generation more expensive
    "speech-to-text": 2.0,  # Audio processing premium
    "text-to-speech": 2.0,  # Audio synthesis premium
    "image-classification": 0.3,  # Classification usually cheaper
    "sentiment-analysis": 0.2,  # Simple NLP tasks
    "summarization": 1.2,  # Slightly more than basic generation
    "translation": 1.1,  # Slightly more than basic generation
}


def detect_model_provider(model: str) -> str:
    """Detect the provider based on model name patterns."""
    if not model:
        return "huggingface_hub"

    model_lower = model.lower()

    # Provider detection patterns
    patterns = {
        "openai": r"(gpt-|dall-e|whisper|text-embedding)",
        "anthropic": r"claude-",
        "cohere": r"(command-|embed-)",
        "meta": r"(llama|meta-llama|code-llama)",
        "mistral": r"mistral",
        "google": r"(gemma|flan-)",
    }

    for provider, pattern in patterns.items():
        if re.search(pattern, model_lower):
            return provider

    # Default to Hugging Face Hub for org/model format
    if "/" in model and not model.startswith("http"):
        return "huggingface_hub"

    return "huggingface_hub"


def estimate_model_size_category(model: str) -> str:
    """Estimate model size category for Hub models."""
    model_lower = model.lower()

    # Size indicators in model names
    if any(size in model_lower for size in ["405b", "175b", "70b", "65b"]):
        return "large_models"
    elif any(size in model_lower for size in ["13b", "20b", "30b", "34b"]):
        return "medium_models"
    elif any(size in model_lower for size in ["7b", "8b", "11b"]):
        return "medium_models"
    elif any(size in model_lower for size in ["1b", "2b", "3b"]):
        return "small_models"
    elif "embed" in model_lower or "sentence" in model_lower:
        return "embedding_models"
    elif any(
        img_type in model_lower
        for img_type in ["diffus", "dalle", "imagen", "midjourney"]
    ):
        return "image_models"
    else:
        # Default based on common patterns
        if any(term in model_lower for term in ["base", "small", "mini"]):
            return "small_models"
        else:
            return "medium_models"


def get_model_pricing(
    provider: str, model: str, task: str = "text-generation"
) -> dict[str, float]:
    """Get pricing information for a specific model and task."""
    pricing_data = HUGGINGFACE_PRICING.get(provider, {})

    if provider == "huggingface_hub":
        # For Hub models, use size-based category
        size_category = estimate_model_size_category(model)
        base_pricing = pricing_data.get(
            size_category,
            pricing_data.get("medium_models", {"input": 0.0001, "output": 0.0002}),
        )
    else:
        # For third-party models, try exact match first, then fallback
        model_key = model.lower().replace("/", "-").replace(":", "-")
        base_pricing = None

        # Try exact match
        for key, pricing in pricing_data.items():
            if key in model_key or model_key in key:
                base_pricing = pricing
                break

        # Fallback to first available pricing for the provider
        if not base_pricing and pricing_data:
            base_pricing = next(iter(pricing_data.values()))

        if not base_pricing:
            # Ultimate fallback
            base_pricing = {"input": 0.001, "output": 0.002}

    # Apply task multiplier
    task_multiplier = TASK_MULTIPLIERS.get(task, 1.0)

    return {
        "input": base_pricing.get("input", 0.0) * task_multiplier,
        "output": base_pricing.get("output", 0.0) * task_multiplier,
    }


def calculate_huggingface_cost(
    provider: str,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    task: str = "text-generation",
    images_generated: int = 0,
    audio_minutes: int = 0,
) -> float:
    """
    Calculate cost for Hugging Face inference operations.

    Args:
        provider: Detected provider (openai, anthropic, cohere, etc.)
        model: Model identifier
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        task: Task type (text-generation, chat-completion, etc.)
        images_generated: Number of images generated (for image tasks)
        audio_minutes: Minutes of audio processed (for audio tasks)

    Returns:
        Estimated cost in USD
    """
    try:
        pricing = get_model_pricing(provider, model, task)

        # Base token costs
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]

        total_cost = input_cost + output_cost

        # Additional costs for specific tasks
        if task == "text-to-image" and images_generated > 0:
            # For image generation, pricing["input"] represents cost per image
            image_cost = images_generated * pricing["input"]
            total_cost = max(
                total_cost, image_cost
            )  # Use higher of token-based or image-based cost

        if task in ["speech-to-text", "text-to-speech"] and audio_minutes > 0:
            # Audio tasks often priced per minute
            audio_cost = audio_minutes * pricing["input"]
            total_cost = max(total_cost, audio_cost)

        return round(total_cost, 6)  # Round to 6 decimal places for precision

    except Exception as e:
        logger.warning(f"Cost calculation failed for {provider}/{model}: {e}")
        # Return conservative estimate
        return (input_tokens + output_tokens) / 1000 * 0.002


def get_provider_info(model: str) -> dict[str, any]:
    """Get comprehensive provider information for a model."""
    provider = detect_model_provider(model)

    info = {
        "provider": provider,
        "is_third_party": provider != "huggingface_hub",
        "supports_streaming": True,  # Most providers support streaming
        "supports_function_calling": provider in ["openai", "anthropic"],
    }

    # Add cost estimates for common scenarios
    pricing = get_model_pricing(provider, model)
    info["cost_per_1k_tokens"] = {
        "input": pricing["input"],
        "output": pricing["output"],
    }

    # Add typical use case cost estimates
    info["cost_estimates"] = {
        "short_chat": calculate_huggingface_cost(
            provider, model, 100, 50, "chat-completion"
        ),
        "long_generation": calculate_huggingface_cost(
            provider, model, 500, 2000, "text-generation"
        ),
        "embedding": calculate_huggingface_cost(
            provider, model, 1000, 0, "feature-extraction"
        ),
    }

    return info


def compare_model_costs(
    models: list[str],
    input_tokens: int = 1000,
    output_tokens: int = 500,
    task: str = "text-generation",
) -> dict[str, dict[str, any]]:
    """Compare costs across multiple models for the same workload."""
    comparison = {}

    for model in models:
        provider = detect_model_provider(model)
        cost = calculate_huggingface_cost(
            provider, model, input_tokens, output_tokens, task
        )

        comparison[model] = {
            "provider": provider,
            "cost": cost,
            "cost_per_1k_tokens": get_model_pricing(provider, model, task),
            "relative_cost": 1.0,  # Will be updated after all costs calculated
        }

    # Calculate relative costs
    if comparison:
        min_cost = min(info["cost"] for info in comparison.values())  # type: ignore[type-var]
        for model_info in comparison.values():
            model_info["relative_cost"] = (
                model_info["cost"] / min_cost if min_cost > 0 else 1.0
            )

    return comparison


def get_cost_optimization_suggestions(
    model: str, task: str = "text-generation"
) -> dict[str, any]:
    """Get cost optimization suggestions for a given model and task."""
    provider = detect_model_provider(model)
    current_pricing = get_model_pricing(provider, model, task)

    suggestions = {
        "current_model": {
            "model": model,
            "provider": provider,
            "cost_per_1k": current_pricing,
        },
        "alternatives": [],
        "optimization_tips": [],
    }

    # Suggest cheaper alternatives within the same provider
    if provider in HUGGINGFACE_PRICING:
        provider_models = HUGGINGFACE_PRICING[provider]
        for alt_model, alt_pricing in provider_models.items():
            if alt_pricing["input"] < current_pricing["input"]:
                suggestions["alternatives"].append(
                    {
                        "model": alt_model,
                        "provider": provider,
                        "cost_per_1k": alt_pricing,
                        "savings": round(
                            (current_pricing["input"] - alt_pricing["input"])
                            / current_pricing["input"]
                            * 100,
                            1,
                        ),
                    }
                )

    # General optimization tips
    suggestions["optimization_tips"] = [
        "Consider using Hugging Face Hub models for significant cost savings",
        "Use embeddings/feature extraction for similarity tasks instead of full text generation",
        "Implement response caching to avoid repeated inference costs",
        "Use streaming for better user experience without additional costs",
        "Monitor usage patterns to identify the most cost-effective models for your use case",
    ]

    return suggestions
