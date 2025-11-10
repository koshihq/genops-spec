#!/usr/bin/env python3
"""
GenOps Replicate Pricing Calculator

Comprehensive pricing calculations for all Replicate model categories including
text, image, video, audio, and custom models. Supports multiple billing patterns:
- Time-based billing (hardware usage)
- Token-based billing (input/output tokens)
- Output-based billing (per image/video/audio)
- Hybrid billing (combination of above)

Features:
- Official model pricing database with regular updates
- Community model cost estimation
- Hardware-specific pricing (CPU, GPU types)
- Multi-modal cost optimization recommendations
- Batch processing cost calculations

Usage:
    from genops.providers.replicate_pricing import ReplicatePricingCalculator
    
    calculator = ReplicatePricingCalculator()
    cost = calculator.calculate_cost(model_info, input_data, output, latency_ms)
"""

import logging
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

@dataclass
class ModelPricing:
    """Pricing information for a specific Replicate model."""

    model_name: str
    pricing_type: str  # 'time', 'token', 'output', 'hybrid'
    base_cost: float
    input_cost: Optional[float] = None  # Per 1K tokens or per input unit
    output_cost: Optional[float] = None  # Per 1K tokens or per output unit
    hardware_type: Optional[str] = None  # 'cpu', 't4', 'a100-40gb', 'a100-80gb'
    hardware_cost_per_second: Optional[float] = None
    category: str = 'unknown'  # 'text', 'image', 'video', 'audio', 'multimodal'
    official: bool = False
    min_cost: Optional[float] = None  # Minimum billing amount
    free_tier: Optional[int] = None  # Free requests/tokens per month

@dataclass
class CostBreakdown:
    """Detailed cost breakdown for a Replicate operation."""

    total_cost: float
    base_cost: float = 0.0
    input_cost: float = 0.0
    output_cost: float = 0.0
    hardware_cost: float = 0.0
    time_seconds: float = 0.0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    output_units: Optional[int] = None  # Images, videos, etc.
    hardware_type: Optional[str] = None
    optimization_suggestions: List[str] = None

    def __post_init__(self):
        if self.optimization_suggestions is None:
            self.optimization_suggestions = []

class ReplicatePricingCalculator:
    """
    Comprehensive pricing calculator for all Replicate models.
    
    Maintains an up-to-date database of official model pricing and provides
    intelligent cost estimation for community models.
    """

    def __init__(self, use_cache: bool = True):
        """
        Initialize the pricing calculator.
        
        Args:
            use_cache: Cache pricing data for performance
        """
        self.use_cache = use_cache
        self._pricing_cache: Dict[str, ModelPricing] = {}
        self._load_official_pricing()

    def _load_official_pricing(self):
        """Load official Replicate model pricing data."""

        # Official Text Models (as of 2025)
        text_models = {
            "meta/llama-2-70b-chat": ModelPricing(
                model_name="meta/llama-2-70b-chat",
                pricing_type="token",
                base_cost=0.0,
                input_cost=1.0,   # $1.00 per 1K input tokens
                output_cost=1.0,  # $1.00 per 1K output tokens
                category="text",
                official=True
            ),
            "meta/llama-2-13b-chat": ModelPricing(
                model_name="meta/llama-2-13b-chat",
                pricing_type="token",
                base_cost=0.0,
                input_cost=0.5,   # $0.50 per 1K input tokens
                output_cost=0.5,  # $0.50 per 1K output tokens
                category="text",
                official=True
            ),
            "replicate/llama-2-70b-chat": ModelPricing(
                model_name="replicate/llama-2-70b-chat",
                pricing_type="token",
                base_cost=0.0,
                input_cost=1.0,
                output_cost=1.0,
                category="text",
                official=True
            ),
            "anthropic/claude-3-5-sonnet": ModelPricing(
                model_name="anthropic/claude-3-5-sonnet",
                pricing_type="token",
                base_cost=0.0,
                input_cost=3.0,   # $3.00 per 1K input tokens
                output_cost=15.0,  # $15.00 per 1K output tokens
                category="text",
                official=True
            ),
            "anthropic/claude-3-haiku": ModelPricing(
                model_name="anthropic/claude-3-haiku",
                pricing_type="token",
                base_cost=0.0,
                input_cost=0.25,   # $0.25 per 1K input tokens
                output_cost=1.25,  # $1.25 per 1K output tokens
                category="text",
                official=True
            )
        }

        # Official Image Models
        image_models = {
            "black-forest-labs/flux-pro": ModelPricing(
                model_name="black-forest-labs/flux-pro",
                pricing_type="output",
                base_cost=0.04,  # $0.04 per image
                output_cost=0.04,
                category="image",
                official=True
            ),
            "black-forest-labs/flux-schnell": ModelPricing(
                model_name="black-forest-labs/flux-schnell",
                pricing_type="output",
                base_cost=0.003,  # $0.003 per image
                output_cost=0.003,
                category="image",
                official=True
            ),
            "black-forest-labs/flux-dev": ModelPricing(
                model_name="black-forest-labs/flux-dev",
                pricing_type="output",
                base_cost=0.025,  # $0.025 per image
                output_cost=0.025,
                category="image",
                official=True
            ),
            "stability-ai/sdxl": ModelPricing(
                model_name="stability-ai/sdxl",
                pricing_type="output",
                base_cost=0.002,  # $0.002 per image
                output_cost=0.002,
                category="image",
                official=True
            )
        }

        # Official Video Models
        video_models = {
            "google/veo-2": ModelPricing(
                model_name="google/veo-2",
                pricing_type="output",
                base_cost=0.50,  # $0.50 per second of video
                output_cost=0.50,
                category="video",
                official=True
            ),
            "runwayml/gen-3-alpha-turbo": ModelPricing(
                model_name="runwayml/gen-3-alpha-turbo",
                pricing_type="output",
                base_cost=0.1,   # $0.10 per second of video
                output_cost=0.1,
                category="video",
                official=True
            )
        }

        # Official Audio Models
        audio_models = {
            "openai/whisper": ModelPricing(
                model_name="openai/whisper",
                pricing_type="time",
                base_cost=0.0001,  # Based on processing time
                hardware_cost_per_second=0.0001,
                category="audio",
                official=True
            ),
            "meta/musicgen": ModelPricing(
                model_name="meta/musicgen",
                pricing_type="time",
                base_cost=0.002,
                hardware_cost_per_second=0.002,
                category="audio",
                official=True
            )
        }

        # Hardware pricing (fallback for non-official models)
        self.hardware_pricing = {
            "cpu": 0.000025,      # $0.000025/sec ($0.09/hr)
            "t4": 0.000225,       # $0.000225/sec ($0.81/hr)
            "a100-40gb": 0.001000, # $0.001000/sec ($3.60/hr)
            "a100-80gb": 0.001400, # $0.001400/sec ($5.04/hr)
            "h100": 0.002000,     # $0.002000/sec ($7.20/hr)
        }

        # Combine all models
        all_models = {**text_models, **image_models, **video_models, **audio_models}

        for model_name, pricing in all_models.items():
            self._pricing_cache[model_name] = pricing

    def get_model_info(self, model_name: str) -> 'ReplicateModelInfo':
        """
        Get model information for cost calculation.
        
        Returns model info with pricing details, falling back to estimation
        for unknown models.
        """
        from .replicate import ReplicateModelInfo

        # Check cache first
        if model_name in self._pricing_cache:
            pricing = self._pricing_cache[model_name]
            return ReplicateModelInfo(
                name=model_name,
                pricing_type=pricing.pricing_type,
                base_cost=pricing.base_cost,
                input_cost=pricing.input_cost,
                output_cost=pricing.output_cost,
                hardware_type=pricing.hardware_type,
                official=pricing.official,
                category=pricing.category
            )

        # Estimate for unknown models
        return self._estimate_model_info(model_name)

    def _estimate_model_info(self, model_name: str) -> 'ReplicateModelInfo':
        """Estimate model info for unknown/community models."""
        from .replicate import ReplicateModelInfo

        # Pattern matching for model categories
        category = "unknown"
        pricing_type = "time"
        base_cost = 0.001  # Default $0.001/second

        model_lower = model_name.lower()

        if any(term in model_lower for term in ["llama", "chat", "gpt", "claude", "mistral", "falcon"]):
            category = "text"
            pricing_type = "token"
            base_cost = 0.5  # Default $0.50 per 1K tokens
        elif any(term in model_lower for term in ["flux", "sdxl", "stable", "diffusion", "midjourney", "dalle"]):
            category = "image"
            pricing_type = "output"
            base_cost = 0.01  # Default $0.01 per image
        elif any(term in model_lower for term in ["video", "gen-", "runway", "veo", "pika"]):
            category = "video"
            pricing_type = "output"
            base_cost = 0.2  # Default $0.20 per second of video
        elif any(term in model_lower for term in ["whisper", "music", "audio", "speech", "voice"]):
            category = "audio"
            pricing_type = "time"
            base_cost = 0.001  # Default $0.001/second

        # Estimate hardware type based on model size/complexity
        hardware_type = "cpu"
        if any(term in model_lower for term in ["70b", "large", "xl", "pro"]):
            hardware_type = "a100-40gb"
        elif any(term in model_lower for term in ["13b", "medium", "base"]):
            hardware_type = "t4"

        return ReplicateModelInfo(
            name=model_name,
            pricing_type=pricing_type,
            base_cost=base_cost,
            hardware_type=hardware_type,
            official=False,
            category=category
        )

    def calculate_cost(
        self,
        model_info: 'ReplicateModelInfo',
        input_data: Dict[str, Any],
        output: Any,
        latency_ms: float
    ) -> float:
        """
        Calculate comprehensive cost for a Replicate operation.
        
        Args:
            model_info: Model information with pricing details
            input_data: Input parameters sent to the model
            output: Output received from the model
            latency_ms: Processing time in milliseconds
            
        Returns:
            Total cost in USD
        """
        breakdown = self.calculate_cost_breakdown(model_info, input_data, output, latency_ms)
        return breakdown.total_cost

    def calculate_cost_breakdown(
        self,
        model_info: 'ReplicateModelInfo',
        input_data: Dict[str, Any],
        output: Any,
        latency_ms: float
    ) -> CostBreakdown:
        """
        Calculate detailed cost breakdown for a Replicate operation.
        
        Returns:
            CostBreakdown with detailed cost components and optimization suggestions
        """
        time_seconds = latency_ms / 1000
        breakdown = CostBreakdown(
            total_cost=0.0,
            time_seconds=time_seconds,
            hardware_type=model_info.hardware_type
        )

        # Calculate based on pricing type
        if model_info.pricing_type == "token":
            breakdown = self._calculate_token_cost(model_info, input_data, output, breakdown)
        elif model_info.pricing_type == "output":
            breakdown = self._calculate_output_cost(model_info, input_data, output, breakdown)
        elif model_info.pricing_type == "time":
            breakdown = self._calculate_time_cost(model_info, time_seconds, breakdown)
        elif model_info.pricing_type == "hybrid":
            breakdown = self._calculate_hybrid_cost(model_info, input_data, output, time_seconds, breakdown)
        else:
            # Fallback to time-based
            breakdown.hardware_cost = model_info.base_cost * time_seconds
            breakdown.total_cost = breakdown.hardware_cost

        # Add optimization suggestions
        breakdown.optimization_suggestions = self._get_optimization_suggestions(model_info, breakdown)

        # Round to reasonable precision
        breakdown.total_cost = float(Decimal(str(breakdown.total_cost)).quantize(
            Decimal('0.000001'), rounding=ROUND_HALF_UP
        ))

        return breakdown

    def _calculate_token_cost(
        self,
        model_info: 'ReplicateModelInfo',
        input_data: Dict[str, Any],
        output: Any,
        breakdown: CostBreakdown
    ) -> CostBreakdown:
        """Calculate cost for token-based models."""

        # Estimate input tokens
        prompt = str(input_data.get('prompt', ''))
        input_tokens = self._estimate_tokens(prompt)
        breakdown.input_tokens = input_tokens

        # Estimate output tokens
        if output and isinstance(output, (str, list)):
            output_text = str(output) if isinstance(output, str) else ' '.join(map(str, output))
            output_tokens = self._estimate_tokens(output_text)
            breakdown.output_tokens = output_tokens
        else:
            output_tokens = 100  # Default estimate
            breakdown.output_tokens = output_tokens

        # Calculate costs
        if model_info.input_cost:
            breakdown.input_cost = (input_tokens / 1000) * model_info.input_cost
        if model_info.output_cost:
            breakdown.output_cost = (output_tokens / 1000) * model_info.output_cost

        breakdown.total_cost = breakdown.input_cost + breakdown.output_cost
        return breakdown

    def _calculate_output_cost(
        self,
        model_info: 'ReplicateModelInfo',
        input_data: Dict[str, Any],
        output: Any,
        breakdown: CostBreakdown
    ) -> CostBreakdown:
        """Calculate cost for output-based models (images, videos, etc.)."""

        # Determine number of outputs
        num_outputs = 1

        if model_info.category == "image":
            num_outputs = input_data.get('num_outputs', input_data.get('num_images', 1))
        elif model_info.category == "video":
            # For video, cost is often per second of output
            duration = input_data.get('duration', input_data.get('length', 5.0))  # Default 5 seconds
            num_outputs = duration
        elif isinstance(output, list):
            num_outputs = len(output)

        breakdown.output_units = int(num_outputs)
        breakdown.output_cost = num_outputs * model_info.base_cost
        breakdown.total_cost = breakdown.output_cost

        return breakdown

    def _calculate_time_cost(
        self,
        model_info: 'ReplicateModelInfo',
        time_seconds: float,
        breakdown: CostBreakdown
    ) -> CostBreakdown:
        """Calculate cost for time-based models."""

        # Use model-specific rate or hardware rate
        if model_info.base_cost:
            rate_per_second = model_info.base_cost
        elif model_info.hardware_type and model_info.hardware_type in self.hardware_pricing:
            rate_per_second = self.hardware_pricing[model_info.hardware_type]
        else:
            rate_per_second = self.hardware_pricing['cpu']  # Default fallback

        breakdown.hardware_cost = time_seconds * rate_per_second
        breakdown.total_cost = breakdown.hardware_cost

        return breakdown

    def _calculate_hybrid_cost(
        self,
        model_info: 'ReplicateModelInfo',
        input_data: Dict[str, Any],
        output: Any,
        time_seconds: float,
        breakdown: CostBreakdown
    ) -> CostBreakdown:
        """Calculate cost for models with hybrid pricing (tokens + time)."""

        # Calculate token costs
        breakdown = self._calculate_token_cost(model_info, input_data, output, breakdown)

        # Add hardware/time costs
        if model_info.hardware_type and model_info.hardware_type in self.hardware_pricing:
            hardware_rate = self.hardware_pricing[model_info.hardware_type]
            breakdown.hardware_cost = time_seconds * hardware_rate
            breakdown.total_cost += breakdown.hardware_cost

        return breakdown

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        if not text:
            return 0

        # Rough approximation: ~4 characters per token for English text
        return max(1, len(text) // 4)

    def _get_optimization_suggestions(
        self,
        model_info: 'ReplicateModelInfo',
        breakdown: CostBreakdown
    ) -> List[str]:
        """Generate cost optimization suggestions."""

        suggestions = []

        # Token-based optimizations
        if model_info.pricing_type == "token":
            if breakdown.input_tokens and breakdown.input_tokens > 2000:
                suggestions.append("Consider breaking large prompts into smaller chunks")
            if breakdown.output_tokens and breakdown.output_tokens > 1000:
                suggestions.append("Use max_tokens parameter to limit response length")

        # Time-based optimizations
        if breakdown.time_seconds > 30:
            suggestions.append("Consider using a faster model variant for time-sensitive tasks")

        # Hardware optimizations
        if model_info.hardware_type == "a100-80gb" and breakdown.time_seconds < 5:
            suggestions.append("Consider using smaller GPU for short tasks to reduce costs")

        # Model-specific suggestions
        if model_info.category == "image" and breakdown.output_units and breakdown.output_units > 1:
            suggestions.append("Batch multiple images in single request to reduce overhead")

        if model_info.category == "video" and breakdown.output_units and breakdown.output_units > 10:
            suggestions.append("Consider shorter video clips - cost scales linearly with duration")

        # Cost threshold suggestions
        if breakdown.total_cost > 1.0:
            suggestions.append(f"High cost operation (${breakdown.total_cost:.2f}) - verify necessity")

        return suggestions

    def get_model_alternatives(self, model_name: str, category: Optional[str] = None) -> List[Tuple[str, float, str]]:
        """
        Get alternative models for cost optimization.
        
        Returns:
            List of (model_name, estimated_cost_ratio, reason) tuples
        """
        alternatives = []

        current_info = self.get_model_info(model_name)
        target_category = category or current_info.category

        # Find models in same category
        for cached_model, pricing in self._pricing_cache.items():
            if (pricing.category == target_category and
                cached_model != model_name and
                pricing.official):

                cost_ratio = pricing.base_cost / max(current_info.base_cost, 0.001)

                if cost_ratio < 0.8:  # Significantly cheaper
                    reason = f"~{int((1-cost_ratio)*100)}% cost reduction"
                    alternatives.append((cached_model, cost_ratio, reason))

        # Sort by cost ratio (cheapest first)
        alternatives.sort(key=lambda x: x[1])

        return alternatives[:3]  # Return top 3 alternatives

# Export main classes
__all__ = [
    'ReplicatePricingCalculator',
    'ModelPricing',
    'CostBreakdown'
]
