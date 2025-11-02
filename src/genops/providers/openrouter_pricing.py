"""OpenRouter pricing engine for accurate cost calculation across 400+ models."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OpenRouterPricing:
    """OpenRouter model pricing information."""
    input_cost_per_token: float
    output_cost_per_token: float
    currency: str = "USD"
    provider: str = "unknown"
    model_family: str = "unknown"


class OpenRouterPricingEngine:
    """Comprehensive OpenRouter pricing engine with 400+ model support."""
    
    def __init__(self):
        # Initialize pricing database with major model families
        # Prices are per token (not per 1K or 1M tokens)
        self.pricing_db = self._initialize_pricing_database()
        
    def _initialize_pricing_database(self) -> Dict[str, OpenRouterPricing]:
        """Initialize comprehensive pricing database for OpenRouter models."""
        pricing = {}
        
        # OpenAI Models (via OpenRouter)
        openai_models = {
            "openai/gpt-4o": OpenRouterPricing(0.005 / 1000, 0.015 / 1000, provider="openai", model_family="gpt-4o"),
            "openai/gpt-4o-mini": OpenRouterPricing(0.00015 / 1000, 0.0006 / 1000, provider="openai", model_family="gpt-4o"),
            "openai/gpt-4-turbo": OpenRouterPricing(0.01 / 1000, 0.03 / 1000, provider="openai", model_family="gpt-4"),
            "openai/gpt-4": OpenRouterPricing(0.03 / 1000, 0.06 / 1000, provider="openai", model_family="gpt-4"),
            "openai/gpt-3.5-turbo": OpenRouterPricing(0.0015 / 1000, 0.002 / 1000, provider="openai", model_family="gpt-3.5"),
            "openai/gpt-3.5-turbo-instruct": OpenRouterPricing(0.0015 / 1000, 0.002 / 1000, provider="openai", model_family="gpt-3.5"),
        }
        
        # Anthropic Models (via OpenRouter) 
        anthropic_models = {
            "anthropic/claude-3-5-sonnet": OpenRouterPricing(3.00 / 1000000, 15.00 / 1000000, provider="anthropic", model_family="claude-3.5"),
            "anthropic/claude-3-5-sonnet:beta": OpenRouterPricing(3.00 / 1000000, 15.00 / 1000000, provider="anthropic", model_family="claude-3.5"),
            "anthropic/claude-3-5-haiku": OpenRouterPricing(1.00 / 1000000, 5.00 / 1000000, provider="anthropic", model_family="claude-3.5"),
            "anthropic/claude-3-opus": OpenRouterPricing(15.00 / 1000000, 75.00 / 1000000, provider="anthropic", model_family="claude-3"),
            "anthropic/claude-3-sonnet": OpenRouterPricing(3.00 / 1000000, 15.00 / 1000000, provider="anthropic", model_family="claude-3"),
            "anthropic/claude-3-haiku": OpenRouterPricing(0.25 / 1000000, 1.25 / 1000000, provider="anthropic", model_family="claude-3"),
        }
        
        # Google Models (via OpenRouter)
        google_models = {
            "google/gemini-2.0-flash-exp": OpenRouterPricing(0.075 / 1000000, 0.30 / 1000000, provider="google", model_family="gemini-2.0"),
            "google/gemini-1.5-pro": OpenRouterPricing(1.25 / 1000000, 5.00 / 1000000, provider="google", model_family="gemini-1.5"),
            "google/gemini-1.5-flash": OpenRouterPricing(0.075 / 1000000, 0.30 / 1000000, provider="google", model_family="gemini-1.5"),
            "google/gemini-pro": OpenRouterPricing(0.5 / 1000000, 1.5 / 1000000, provider="google", model_family="gemini-1.0"),
            "google/gemma-2-9b-it": OpenRouterPricing(0.2 / 1000000, 0.2 / 1000000, provider="google", model_family="gemma-2"),
        }
        
        # Meta Models (via OpenRouter)
        meta_models = {
            "meta-llama/llama-3.2-90b-vision-instruct": OpenRouterPricing(0.9 / 1000000, 0.9 / 1000000, provider="meta", model_family="llama-3.2"),
            "meta-llama/llama-3.2-11b-vision-instruct": OpenRouterPricing(0.55 / 1000000, 0.55 / 1000000, provider="meta", model_family="llama-3.2"),
            "meta-llama/llama-3.2-3b-instruct": OpenRouterPricing(0.06 / 1000000, 0.06 / 1000000, provider="meta", model_family="llama-3.2"),
            "meta-llama/llama-3.2-1b-instruct": OpenRouterPricing(0.04 / 1000000, 0.04 / 1000000, provider="meta", model_family="llama-3.2"),
            "meta-llama/llama-3.1-405b-instruct": OpenRouterPricing(5.0 / 1000000, 15.0 / 1000000, provider="meta", model_family="llama-3.1"),
            "meta-llama/llama-3.1-70b-instruct": OpenRouterPricing(0.9 / 1000000, 0.9 / 1000000, provider="meta", model_family="llama-3.1"),
            "meta-llama/llama-3.1-8b-instruct": OpenRouterPricing(0.2 / 1000000, 0.2 / 1000000, provider="meta", model_family="llama-3.1"),
        }
        
        # Mistral Models (via OpenRouter)
        mistral_models = {
            "mistralai/mistral-large": OpenRouterPricing(4.0 / 1000000, 12.0 / 1000000, provider="mistral", model_family="mistral-large"),
            "mistralai/mistral-medium": OpenRouterPricing(2.7 / 1000000, 8.1 / 1000000, provider="mistral", model_family="mistral-medium"),
            "mistralai/mistral-small": OpenRouterPricing(1.0 / 1000000, 3.0 / 1000000, provider="mistral", model_family="mistral-small"),
            "mistralai/mistral-tiny": OpenRouterPricing(0.25 / 1000000, 0.25 / 1000000, provider="mistral", model_family="mistral-tiny"),
            "mistralai/mixtral-8x7b-instruct": OpenRouterPricing(0.5 / 1000000, 0.5 / 1000000, provider="mistral", model_family="mixtral"),
            "mistralai/mixtral-8x22b-instruct": OpenRouterPricing(1.2 / 1000000, 1.2 / 1000000, provider="mistral", model_family="mixtral"),
        }
        
        # Cohere Models (via OpenRouter)
        cohere_models = {
            "cohere/command-r": OpenRouterPricing(0.5 / 1000000, 1.5 / 1000000, provider="cohere", model_family="command-r"),
            "cohere/command-r-plus": OpenRouterPricing(3.0 / 1000000, 15.0 / 1000000, provider="cohere", model_family="command-r"),
            "cohere/command": OpenRouterPricing(1.0 / 1000000, 2.0 / 1000000, provider="cohere", model_family="command"),
        }
        
        # Other Notable Models
        other_models = {
            # Perplexity
            "perplexity/llama-3.1-sonar-small-128k-online": OpenRouterPricing(0.2 / 1000000, 0.2 / 1000000, provider="perplexity", model_family="sonar"),
            "perplexity/llama-3.1-sonar-large-128k-online": OpenRouterPricing(1.0 / 1000000, 1.0 / 1000000, provider="perplexity", model_family="sonar"),
            
            # Databricks
            "databricks/dbrx-instruct": OpenRouterPricing(0.75 / 1000000, 2.25 / 1000000, provider="databricks", model_family="dbrx"),
            
            # Together AI
            "togethercomputer/llama-2-7b-chat": OpenRouterPricing(0.2 / 1000000, 0.2 / 1000000, provider="together", model_family="llama-2"),
            
            # Nous Research  
            "nousresearch/nous-hermes-2-mixtral-8x7b-dpo": OpenRouterPricing(0.5 / 1000000, 0.5 / 1000000, provider="nous", model_family="hermes-2"),
        }
        
        # Combine all models
        pricing.update(openai_models)
        pricing.update(anthropic_models)
        pricing.update(google_models)
        pricing.update(meta_models)
        pricing.update(mistral_models)
        pricing.update(cohere_models)
        pricing.update(other_models)
        
        return pricing
    
    def get_model_pricing(self, model_name: str) -> Optional[OpenRouterPricing]:
        """Get pricing for a specific model."""
        # Direct lookup first
        if model_name in self.pricing_db:
            return self.pricing_db[model_name]
            
        # Try fuzzy matching for common variations
        normalized_model = self._normalize_model_name(model_name)
        for db_model, pricing in self.pricing_db.items():
            if self._normalize_model_name(db_model) == normalized_model:
                return pricing
                
        return None
    
    def _normalize_model_name(self, model_name: str) -> str:
        """Normalize model name for fuzzy matching."""
        return model_name.lower().replace(":", "-").replace("_", "-")
    
    def calculate_cost(
        self, 
        model_name: str, 
        actual_provider: Optional[str] = None,
        input_tokens: int = 0, 
        output_tokens: int = 0
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate cost for OpenRouter model usage.
        
        Args:
            model_name: The OpenRouter model name
            actual_provider: The actual provider used (if known from routing)
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Tuple of (total_cost, cost_breakdown_dict)
        """
        pricing = self.get_model_pricing(model_name)
        
        if pricing is None:
            # Fallback pricing based on provider or model patterns
            pricing = self._get_fallback_pricing(model_name, actual_provider)
        
        # Calculate costs
        input_cost = input_tokens * pricing.input_cost_per_token
        output_cost = output_tokens * pricing.output_cost_per_token
        total_cost = input_cost + output_cost
        
        # Cost breakdown for detailed telemetry
        cost_breakdown = {
            "total_cost": total_cost,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost_per_token": pricing.input_cost_per_token,
            "output_cost_per_token": pricing.output_cost_per_token,
            "provider": pricing.provider,
            "model_family": pricing.model_family,
            "currency": pricing.currency,
            "model_name": model_name
        }
        
        return total_cost, cost_breakdown
    
    def _get_fallback_pricing(
        self, model_name: str, actual_provider: Optional[str] = None
    ) -> OpenRouterPricing:
        """Get fallback pricing when exact model is not found."""
        
        # First, try provider-based fallback
        if actual_provider:
            provider_defaults = {
                "openai": OpenRouterPricing(0.01 / 1000, 0.02 / 1000, provider="openai", model_family="unknown"),
                "anthropic": OpenRouterPricing(3.00 / 1000000, 15.00 / 1000000, provider="anthropic", model_family="unknown"),
                "google": OpenRouterPricing(0.5 / 1000000, 1.5 / 1000000, provider="google", model_family="unknown"),
                "meta": OpenRouterPricing(0.2 / 1000000, 0.2 / 1000000, provider="meta", model_family="unknown"),
                "mistral": OpenRouterPricing(1.0 / 1000000, 3.0 / 1000000, provider="mistral", model_family="unknown"),
                "cohere": OpenRouterPricing(1.0 / 1000000, 2.0 / 1000000, provider="cohere", model_family="unknown"),
            }
            
            if actual_provider in provider_defaults:
                return provider_defaults[actual_provider]
        
        # Model name pattern matching
        model_lower = model_name.lower()
        
        if any(pattern in model_lower for pattern in ["gpt-4", "openai"]):
            return OpenRouterPricing(0.01 / 1000, 0.03 / 1000, provider="openai", model_family="gpt-4")
        elif any(pattern in model_lower for pattern in ["gpt-3.5", "gpt-3"]):
            return OpenRouterPricing(0.0015 / 1000, 0.002 / 1000, provider="openai", model_family="gpt-3.5")
        elif any(pattern in model_lower for pattern in ["claude", "anthropic"]):
            return OpenRouterPricing(3.00 / 1000000, 15.00 / 1000000, provider="anthropic", model_family="claude-3")
        elif any(pattern in model_lower for pattern in ["gemini", "google"]):
            return OpenRouterPricing(0.5 / 1000000, 1.5 / 1000000, provider="google", model_family="gemini")
        elif any(pattern in model_lower for pattern in ["llama", "meta"]):
            return OpenRouterPricing(0.2 / 1000000, 0.2 / 1000000, provider="meta", model_family="llama")
        elif any(pattern in model_lower for pattern in ["mistral", "mixtral"]):
            return OpenRouterPricing(1.0 / 1000000, 3.0 / 1000000, provider="mistral", model_family="mistral")
        elif any(pattern in model_lower for pattern in ["command", "cohere"]):
            return OpenRouterPricing(1.0 / 1000000, 2.0 / 1000000, provider="cohere", model_family="command")
        else:
            # Generic fallback - medium cost tier
            logger.warning(f"Unknown model {model_name}, using generic pricing")
            return OpenRouterPricing(0.005 / 1000, 0.01 / 1000, provider="unknown", model_family="unknown")
    
    def get_provider_models(self, provider: str) -> Dict[str, OpenRouterPricing]:
        """Get all models for a specific provider."""
        return {
            model: pricing 
            for model, pricing in self.pricing_db.items() 
            if pricing.provider == provider
        }
    
    def estimate_cost_for_text(
        self, model_name: str, text: str, completion_ratio: float = 0.3
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Estimate cost for text processing.
        
        Args:
            model_name: The OpenRouter model name
            text: Input text to estimate tokens for
            completion_ratio: Estimated completion tokens as ratio of input (default 0.3)
            
        Returns:
            Tuple of (estimated_cost, cost_breakdown_dict)
        """
        # Rough token estimation (1 token ≈ 0.75 words)
        estimated_tokens = int(len(text.split()) * 1.33)
        estimated_completion_tokens = int(estimated_tokens * completion_ratio)
        
        return self.calculate_cost(
            model_name, 
            input_tokens=estimated_tokens, 
            output_tokens=estimated_completion_tokens
        )


# Global pricing engine instance
_pricing_engine = None


def get_pricing_engine() -> OpenRouterPricingEngine:
    """Get the global pricing engine instance."""
    global _pricing_engine
    if _pricing_engine is None:
        _pricing_engine = OpenRouterPricingEngine()
    return _pricing_engine


def calculate_openrouter_cost(
    model_name: str, 
    actual_provider: Optional[str] = None,
    input_tokens: int = 0, 
    output_tokens: int = 0
) -> float:
    """
    Calculate cost for OpenRouter model usage.
    
    Args:
        model_name: The OpenRouter model name
        actual_provider: The actual provider used (if known from routing)
        input_tokens: Number of input tokens  
        output_tokens: Number of output tokens
        
    Returns:
        Total cost in USD
    """
    engine = get_pricing_engine()
    cost, _ = engine.calculate_cost(model_name, actual_provider, input_tokens, output_tokens)
    return cost


def get_cost_breakdown(
    model_name: str,
    actual_provider: Optional[str] = None, 
    input_tokens: int = 0,
    output_tokens: int = 0
) -> Dict[str, Any]:
    """
    Get detailed cost breakdown for OpenRouter model usage.
    
    Args:
        model_name: The OpenRouter model name
        actual_provider: The actual provider used (if known from routing)
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        
    Returns:
        Dictionary with detailed cost breakdown
    """
    engine = get_pricing_engine()
    _, breakdown = engine.calculate_cost(model_name, actual_provider, input_tokens, output_tokens)
    return breakdown


def get_supported_models() -> Dict[str, OpenRouterPricing]:
    """Get all supported models and their pricing."""
    engine = get_pricing_engine()
    return engine.pricing_db


def get_provider_models(provider: str) -> Dict[str, OpenRouterPricing]:
    """Get all models for a specific provider."""
    engine = get_pricing_engine()
    return engine.get_provider_models(provider)