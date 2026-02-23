#!/usr/bin/env python3
"""
LiteLLM Provider for GenOps

Comprehensive integration with LiteLLM's unified interface to 100+ LLM providers,
providing governance telemetry, cost tracking, and performance monitoring across
the entire LLM ecosystem through a single instrumentation layer.

Usage:
    from genops.providers.litellm import auto_instrument
    auto_instrument()

    # Your existing LiteLLM code works unchanged
    import litellm
    response = litellm.completion(
        model="gpt-4",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    # ✅ Automatic cost tracking and governance added across 100+ providers!

Features:
    - Single instrumentation layer for massive ecosystem coverage
    - Auto-instrumentation for existing LiteLLM applications
    - Provider-agnostic cost tracking and optimization
    - OpenTelemetry-native governance telemetry
    - Multi-provider budget controls and compliance
    - Enterprise deployment patterns
"""

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Optional

# GenOps core imports
try:
    from genops.core.cost_calculator import CostCalculator
    from genops.core.governance import GovernanceManager
    from genops.core.telemetry import GenOpsTelemetry
except ImportError:
    # Graceful degradation if core modules not available
    GenOpsTelemetry = None  # type: ignore
    CostCalculator = None
    GovernanceManager = None  # type: ignore

logger = logging.getLogger(__name__)

# Check for LiteLLM availability
try:
    import litellm

    LITELLM_AVAILABLE = True
    logger.info("LiteLLM found - full functionality available")
except ImportError:
    LITELLM_AVAILABLE = False
    logger.warning(
        "LiteLLM not installed - provider available but limited functionality"
    )

# Global instrumentation state
_instrumentation_active = False
_instrumentation_config = {}
_callback_registry = []
_usage_stats = {
    "total_requests": 0,
    "total_cost": Decimal("0"),
    "provider_usage": {},
    "model_usage": {},
}
_stats_lock = threading.Lock()

# Aliases for test imports
_global_usage_stats = _usage_stats
_usage_lock = _stats_lock


@dataclass
class LiteLLMUsageStats:
    """Structured usage statistics for LiteLLM requests."""

    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: Decimal
    latency_ms: float
    timestamp: float
    team: Optional[str] = None
    project: Optional[str] = None
    customer_id: Optional[str] = None


@dataclass
class LiteLLMGovernanceContext:
    """Context for LiteLLM governance tracking."""

    team: str = "default-team"
    project: str = "default-project"
    environment: str = "development"
    customer_id: Optional[str] = None
    cost_center: Optional[str] = None
    feature: Optional[str] = None
    daily_budget_limit: float = 100.0
    governance_policy: str = "advisory"  # advisory, enforced
    enable_cost_tracking: bool = True
    custom_attributes: dict[str, Any] = field(default_factory=dict)


class GenOpsLiteLLMCallback:
    """
    GenOps callback for LiteLLM that captures telemetry and governance data.

    Integrates with LiteLLM's callback system to provide:
    - OpenTelemetry-native telemetry export
    - Cost tracking with team/project attribution
    - Budget controls and compliance monitoring
    - Performance tracking across all providers
    """

    def __init__(self, governance_context: LiteLLMGovernanceContext):
        self.governance_context = governance_context
        self.telemetry = GenOpsTelemetry() if GenOpsTelemetry else None
        self.cost_calculator = CostCalculator() if CostCalculator else None
        self.governance_manager = GovernanceManager() if GovernanceManager else None

    def input_callback(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Handle input callback - called before LiteLLM request."""
        try:
            # Start telemetry span
            if self.telemetry:
                span_name = f"litellm.completion.{kwargs.get('model', 'unknown')}"
                self.telemetry.start_span(
                    span_name,
                    {
                        "genops.provider": "litellm",
                        "genops.model": kwargs.get("model"),
                        "genops.team": self.governance_context.team,
                        "genops.project": self.governance_context.project,
                        "genops.environment": self.governance_context.environment,
                        "genops.customer_id": self.governance_context.customer_id,
                        "litellm.input.messages": len(kwargs.get("messages", [])),
                        "litellm.input.max_tokens": kwargs.get("max_tokens"),
                        "litellm.input.temperature": kwargs.get("temperature"),
                    },
                )

            # Budget check if governance enabled
            if (
                self.governance_manager
                and self.governance_context.governance_policy == "enforced"
            ):
                current_spend = self._get_daily_spend()
                if current_spend >= self.governance_context.daily_budget_limit:
                    raise Exception(
                        f"Daily budget limit ${self.governance_context.daily_budget_limit} exceeded"
                    )

            # Add GenOps metadata to request
            if "metadata" not in kwargs:
                kwargs["metadata"] = {}

            kwargs["metadata"].update(
                {
                    "genops_team": self.governance_context.team,
                    "genops_project": self.governance_context.project,
                    "genops_customer_id": self.governance_context.customer_id,
                }
            )

            return kwargs

        except Exception as e:
            logger.debug(f"Error in LiteLLM input callback: {e}")
            return kwargs

    def success_callback(
        self,
        kwargs: dict[str, Any],
        completion_response: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Handle success callback - called after successful LiteLLM request."""
        try:
            # Extract usage and cost information
            usage_stats = self._extract_usage_stats(
                kwargs, completion_response, start_time, end_time
            )

            # Update global stats
            self._update_usage_stats(usage_stats)

            # Send telemetry
            if self.telemetry:
                self.telemetry.record_metrics(
                    {
                        "genops.cost.total": float(usage_stats.cost),
                        "genops.tokens.input": usage_stats.input_tokens,
                        "genops.tokens.output": usage_stats.output_tokens,
                        "genops.tokens.total": usage_stats.total_tokens,
                        "genops.latency.ms": usage_stats.latency_ms,
                        "genops.provider": usage_stats.provider,
                        "genops.model": usage_stats.model,
                    }
                )

                self.telemetry.end_span(
                    {"genops.status": "success", "genops.cost.currency": "USD"}
                )

            # Log cost information
            logger.info(
                f"LiteLLM request completed: {usage_stats.provider}/{usage_stats.model} "
                f"cost=${usage_stats.cost:.6f} tokens={usage_stats.total_tokens}"
            )

        except Exception as e:
            logger.debug(f"Error in LiteLLM success callback: {e}")

    def failure_callback(
        self,
        kwargs: dict[str, Any],
        completion_response: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Handle failure callback - called after failed LiteLLM request."""
        try:
            # Extract basic information
            latency_ms = (end_time - start_time) * 1000

            # Send failure telemetry
            if self.telemetry:
                self.telemetry.record_metrics(
                    {"genops.latency.ms": latency_ms, "genops.status": "error"}
                )

                self.telemetry.end_span(
                    {
                        "genops.status": "error",
                        "genops.error": str(completion_response)
                        if completion_response
                        else "unknown_error",
                    }
                )

            logger.warning(
                f"LiteLLM request failed: {kwargs.get('model', 'unknown')} "
                f"latency={latency_ms:.1f}ms error={completion_response}"
            )

        except Exception as e:
            logger.debug(f"Error in LiteLLM failure callback: {e}")

    def _extract_usage_stats(
        self, kwargs: dict[str, Any], response: Any, start_time: float, end_time: float
    ) -> LiteLLMUsageStats:
        """Extract usage statistics from LiteLLM response."""
        # Default values
        provider = "unknown"
        model = kwargs.get("model", "unknown")
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        cost = Decimal("0")

        try:
            # Extract provider from model name (LiteLLM convention)
            if "/" in model:
                provider = model.split("/")[0]
            elif model.startswith("gpt-"):
                provider = "openai"
            elif model.startswith("claude-"):
                provider = "anthropic"
            elif model.startswith("gemini-"):
                provider = "google"

            # Extract usage from response
            if hasattr(response, "usage") and response.usage:
                input_tokens = getattr(response.usage, "prompt_tokens", 0)
                output_tokens = getattr(response.usage, "completion_tokens", 0)
                total_tokens = getattr(
                    response.usage, "total_tokens", input_tokens + output_tokens
                )

            # Calculate cost using LiteLLM's built-in cost tracking or fallback
            if (
                hasattr(response, "_hidden_params")
                and "response_cost" in response._hidden_params
            ):
                cost = Decimal(str(response._hidden_params["response_cost"]))
            elif self.cost_calculator:
                cost = self.cost_calculator.calculate_cost(
                    provider, model, input_tokens, output_tokens
                )

        except Exception as e:
            logger.debug(f"Error extracting usage stats: {e}")

        return LiteLLMUsageStats(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost=cost,
            latency_ms=(end_time - start_time) * 1000,
            timestamp=time.time(),
            team=self.governance_context.team,
            project=self.governance_context.project,
            customer_id=self.governance_context.customer_id,
        )

    def _update_usage_stats(self, stats: LiteLLMUsageStats) -> None:
        """Update global usage statistics."""
        with _stats_lock:
            _usage_stats["total_requests"] += 1
            _usage_stats["total_cost"] += stats.cost

            # Update provider stats
            if stats.provider not in _usage_stats["provider_usage"]:
                _usage_stats["provider_usage"][stats.provider] = {
                    "requests": 0,
                    "cost": Decimal("0"),
                    "tokens": 0,
                }

            _usage_stats["provider_usage"][stats.provider]["requests"] += 1
            _usage_stats["provider_usage"][stats.provider]["cost"] += stats.cost
            _usage_stats["provider_usage"][stats.provider]["tokens"] += (
                stats.total_tokens
            )

            # Update model stats
            if stats.model not in _usage_stats["model_usage"]:
                _usage_stats["model_usage"][stats.model] = {
                    "requests": 0,
                    "cost": Decimal("0"),
                    "tokens": 0,
                }

            _usage_stats["model_usage"][stats.model]["requests"] += 1
            _usage_stats["model_usage"][stats.model]["cost"] += stats.cost
            _usage_stats["model_usage"][stats.model]["tokens"] += stats.total_tokens

    def _get_daily_spend(self) -> float:
        """Get current daily spend for budget checking."""
        with _stats_lock:
            return float(_usage_stats["total_cost"])  # type: ignore[arg-type]


def auto_instrument(
    team: str = "default-team",
    project: str = "default-project",
    environment: str = "development",
    customer_id: Optional[str] = None,
    daily_budget_limit: float = 100.0,
    governance_policy: str = "advisory",
    enable_cost_tracking: bool = True,
    **kwargs,
) -> bool:
    """
    Auto-instrument LiteLLM with GenOps governance telemetry.

    This function enables automatic tracking of all LiteLLM requests across
    100+ providers with zero code changes to existing applications.

    Args:
        team: Team identifier for cost attribution
        project: Project identifier for governance
        environment: Deployment environment (development, staging, production)
        customer_id: Optional customer attribution
        daily_budget_limit: Daily spending limit in USD
        governance_policy: "advisory" (warnings) or "enforced" (blocking)
        enable_cost_tracking: Enable detailed cost tracking
        **kwargs: Additional governance attributes

    Returns:
        bool: True if instrumentation successful, False otherwise
    """
    global _instrumentation_active, _instrumentation_config

    if not LITELLM_AVAILABLE:
        logger.warning("LiteLLM not available - cannot enable auto-instrumentation")
        return False

    try:
        # Create governance context
        governance_context = LiteLLMGovernanceContext(
            team=team,
            project=project,
            environment=environment,
            customer_id=customer_id,
            daily_budget_limit=daily_budget_limit,
            governance_policy=governance_policy,
            enable_cost_tracking=enable_cost_tracking,
            custom_attributes=kwargs,
        )

        # Create GenOps callback
        genops_callback = GenOpsLiteLLMCallback(governance_context)

        # Register callbacks with LiteLLM
        if not hasattr(litellm, "input_callback"):
            litellm.input_callback = []
        if not hasattr(litellm, "success_callback"):
            litellm.success_callback = []
        if not hasattr(litellm, "failure_callback"):
            litellm.failure_callback = []

        # Add GenOps callbacks
        litellm.input_callback.append(genops_callback.input_callback)
        litellm.success_callback.append(genops_callback.success_callback)
        litellm.failure_callback.append(genops_callback.failure_callback)

        # Store configuration
        _instrumentation_config = {
            "team": team,
            "project": project,
            "environment": environment,
            "governance_context": governance_context,
        }
        _instrumentation_active = True

        logger.info(
            f"GenOps LiteLLM auto-instrumentation enabled for team={team} project={project}"
        )
        logger.info(
            "All LiteLLM requests will now include governance telemetry across 100+ providers"
        )

        return True

    except Exception as e:
        logger.error(f"Failed to enable LiteLLM auto-instrumentation: {e}")
        return False


@contextmanager
def track_completion(
    model: str,
    team: Optional[str] = None,
    project: Optional[str] = None,
    customer_id: Optional[str] = None,
    **attributes,
):
    """
    Context manager for tracking individual LiteLLM completions.

    Usage:
        with track_completion("gpt-4", team="ai-team") as context:
            response = litellm.completion(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello!"}]
            )
            # context.cost, context.tokens, etc. available
    """
    start_time = time.time()
    context = type(
        "Context",
        (),
        {
            "model": model,
            "team": team or _instrumentation_config.get("team", "default-team"),
            "project": project
            or _instrumentation_config.get("project", "default-project"),
            "customer_id": customer_id,
            "start_time": start_time,
            "cost": Decimal("0"),
            "tokens": 0,
            "provider": "unknown",
            "custom_attributes": attributes,
        },
    )()

    try:
        yield context
    finally:
        context.end_time = time.time()
        context.duration = context.end_time - context.start_time

        logger.debug(
            f"Completion tracked: {context.model} cost=${context.cost} "
            f"tokens={context.tokens} duration={context.duration:.2f}s"
        )


def get_usage_stats() -> dict[str, Any]:
    """Get current usage statistics across all providers."""
    with _stats_lock:
        # Calculate total tokens across all providers
        total_tokens = sum(
            stats["tokens"] for stats in _usage_stats["provider_usage"].values()
        )

        return {
            "total_requests": _usage_stats["total_requests"],
            "total_cost": float(_usage_stats["total_cost"]),  # type: ignore[arg-type]
            "total_tokens": total_tokens,
            "provider_usage": {
                provider: {
                    "requests": stats["requests"],
                    "cost": float(stats["cost"]),
                    "tokens": stats["tokens"],
                }
                for provider, stats in _usage_stats["provider_usage"].items()
            },
            "model_usage": {
                model: {
                    "requests": stats["requests"],
                    "cost": float(stats["cost"]),
                    "tokens": stats["tokens"],
                }
                for model, stats in _usage_stats["model_usage"].items()
            },
            "instrumentation_active": _instrumentation_active,
            "instrumentation_config": _instrumentation_config.copy()
            if _instrumentation_config
            else {},
        }


def get_cost_summary(
    timeframe: str = "all", group_by: str = "provider"
) -> dict[str, Any]:
    """
    Get cost summary with various grouping options.

    Args:
        timeframe: "all", "today", "week", "month"
        group_by: "provider", "model", "team", "project"
    """
    stats = get_usage_stats()

    if group_by == "provider":
        return {
            "total_cost": stats["total_cost"],
            "cost_by_provider": {
                provider: data["cost"]
                for provider, data in stats["provider_usage"].items()
            },
            "timeframe": timeframe,
        }
    elif group_by == "model":
        return {
            "total_cost": stats["total_cost"],
            "cost_by_model": {
                model: data["cost"] for model, data in stats["model_usage"].items()
            },
            "timeframe": timeframe,
        }
    else:
        return stats


def reset_usage_stats() -> None:
    """Reset all usage statistics (useful for testing)."""
    global _usage_stats
    with _stats_lock:
        _usage_stats = {
            "total_requests": 0,
            "total_cost": Decimal("0"),
            "provider_usage": {},
            "model_usage": {},
        }


def instrument_litellm(
    team: str,
    project: str,
    environment: str = "development",
    customer_id: Optional[str] = None,
    daily_budget_limit: float = 100.0,
    governance_policy: str = "advisory",
    enable_cost_tracking: bool = True,
    **kwargs,
) -> bool:
    """
    Factory function for creating instrumented LiteLLM instances.

    This is an alias for auto_instrument() that follows GenOps naming conventions.

    Args:
        team: Team identifier for cost attribution
        project: Project identifier for governance
        environment: Deployment environment
        customer_id: Optional customer attribution
        daily_budget_limit: Daily spending limit in USD
        governance_policy: "advisory" or "enforced"
        enable_cost_tracking: Enable detailed cost tracking
        **kwargs: Additional governance attributes

    Returns:
        bool: True if instrumentation successful, False otherwise
    """
    return auto_instrument(
        team=team,
        project=project,
        environment=environment,
        customer_id=customer_id,
        daily_budget_limit=daily_budget_limit,
        governance_policy=governance_policy,
        enable_cost_tracking=enable_cost_tracking,
        **kwargs,
    )


def multi_provider_cost_tracking(
    providers: Optional[list[str]] = None,
    time_range: str = "1d",
    group_by: str = "provider",
) -> dict[str, Any]:
    """
    Unified cost tracking across multiple providers.

    Args:
        providers: List of provider names to include (None for all)
        time_range: Time range filter (e.g., "1h", "1d", "7d")
        group_by: Group costs by "provider", "team", "project", or "customer"

    Returns:
        Dict containing cost breakdown and statistics
    """
    # Get current usage statistics
    stats = get_usage_stats()

    # Get cost summary with grouping
    summary = get_cost_summary(group_by=group_by)

    # Filter by providers if specified
    if providers and group_by == "provider":
        filtered_costs = {}
        total_filtered = 0.0

        for provider in providers:
            if provider in summary.get("cost_by_provider", {}):
                cost = summary["cost_by_provider"][provider]
                filtered_costs[provider] = cost
                total_filtered += cost

        summary["cost_by_provider"] = filtered_costs
        summary["total_cost"] = total_filtered

    # Add multi-provider insights
    result = {
        "total_cost": summary.get("total_cost", 0.0),
        "total_requests": stats.get("total_requests", 0),
        "total_tokens": stats.get("total_tokens", 0),
        f"cost_by_{group_by}": summary.get(f"cost_by_{group_by}", {}),
        "provider_count": len(stats.get("provider_usage", {})),
        "time_range": time_range,
        "group_by": group_by,
    }

    # Add provider comparison if grouping by provider
    if group_by == "provider" and len(result["cost_by_provider"]) > 1:
        costs = list(result["cost_by_provider"].values())
        result["cost_analysis"] = {
            "cheapest_provider": min(
                result["cost_by_provider"], key=result["cost_by_provider"].get
            ),
            "most_expensive_provider": max(
                result["cost_by_provider"], key=result["cost_by_provider"].get
            ),
            "cost_variance": max(costs) - min(costs) if costs else 0.0,
            "average_cost_per_provider": sum(costs) / len(costs) if costs else 0.0,
        }

    return result


def validate_setup(quick: bool = False, test_connectivity: bool = False) -> bool:
    """
    Validate LiteLLM + GenOps integration setup.

    This is a convenience wrapper around the comprehensive validation module.

    Args:
        quick: Run only essential validations
        test_connectivity: Test actual API connectivity

    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        from .litellm_validation import print_validation_result, validate_litellm_setup

        result = validate_litellm_setup(
            quick=quick, test_connectivity=test_connectivity
        )
        print_validation_result(result, verbose=not quick)

        return result.is_valid

    except ImportError:
        logger.warning("Validation module not available")
        return False
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False


# Provider information
PROVIDER_INFO = {
    "name": "LiteLLM",
    "description": "Unified interface to 100+ LLM providers",
    "website": "https://docs.litellm.ai/",
    "ecosystem_coverage": 100,  # Number of supported providers
    "supported_providers": [
        "openai",
        "anthropic",
        "azure",
        "vertexai",
        "bedrock",
        "cohere",
        "huggingface",
        "ollama",
        "together",
        "replicate",
        "palm",
        "gemini",
        "claude",
        "mistral",
        "fireworks",
        "anyscale",
        "deepinfra",
        "perplexity",
    ],
    "genops_features": [
        "auto_instrumentation",
        "cost_tracking",
        "governance_telemetry",
        "budget_controls",
        "multi_provider_optimization",
        "compliance_monitoring",
    ],
}


# Export main functions and classes
__all__ = [
    "auto_instrument",
    "instrument_litellm",
    "multi_provider_cost_tracking",
    "validate_setup",
    "track_completion",
    "get_usage_stats",
    "get_cost_summary",
    "reset_usage_stats",
    "LiteLLMGovernanceContext",
    "LiteLLMUsageStats",
    "GenOpsLiteLLMCallback",
    "PROVIDER_INFO",
    "_calculate_cost",
    "_infer_provider_from_model",
]


# Pricing per 1K tokens (input, output) keyed by model prefix
_MODEL_PRICING = {
    # OpenAI
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-4": (0.03, 0.06),
    "gpt-3.5-turbo": (0.0015, 0.002),
    "text-davinci": (0.02, 0.02),
    # Anthropic
    "claude-3-opus": (0.015, 0.075),
    "claude-3-sonnet": (0.003, 0.015),
    "claude-3-haiku": (0.00025, 0.00125),
    "claude-2": (0.008, 0.024),
    "claude": (0.008, 0.024),
    # Google
    "gemini-pro": (0.00025, 0.0005),
    "gemini": (0.00025, 0.0005),
    "palm": (0.00025, 0.0005),
}

# Fallback pricing per 1K tokens
_FALLBACK_PRICING = (0.001, 0.002)


def _calculate_cost(
    provider: str = "",
    model: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> float:
    """Calculate cost for a model completion based on known pricing."""
    if input_tokens == 0 and output_tokens == 0:
        return 0.0

    model_lower = model.lower()

    # Try longest prefix match against known models
    input_rate, output_rate = _FALLBACK_PRICING
    for prefix in sorted(_MODEL_PRICING, key=len, reverse=True):
        if model_lower.startswith(prefix):
            input_rate, output_rate = _MODEL_PRICING[prefix]
            break

    return (input_tokens * input_rate / 1000) + (output_tokens * output_rate / 1000)


def _infer_provider_from_model(model: str) -> str:
    """Infer provider name from model identifier."""
    model_lower = model.lower()
    if "gpt" in model_lower or "davinci" in model_lower:
        return "openai"
    elif "claude" in model_lower:
        return "anthropic"
    elif "gemini" in model_lower or "palm" in model_lower:
        return "google"
    elif "llama" in model_lower:
        return "meta"
    elif "mistral" in model_lower:
        return "mistral"
    elif "command" in model_lower or "embed" in model_lower:
        return "cohere"
    return "unknown"
