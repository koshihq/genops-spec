#!/usr/bin/env python3
"""
GenOps Langfuse Observability Platform Integration

This module provides comprehensive Langfuse integration for GenOps AI governance,
cost intelligence, and policy enforcement. Langfuse is a powerful LLM engineering
platform that provides observability, evaluation, and prompt management.

Features:
- Enhanced Langfuse traces with GenOps governance attributes
- Cost attribution and budget enforcement for LLM operations
- Policy compliance tracking integrated with Langfuse observations
- LLM evaluation with governance oversight
- Prompt management with cost optimization insights
- Zero-code auto-instrumentation with instrument_langfuse()
- Enterprise-ready governance patterns for production deployments

Example usage:

    # Zero-code auto-instrumentation
    from genops.providers.langfuse import instrument_langfuse
    instrument_langfuse(
        langfuse_public_key="pk-lf-...",
        langfuse_secret_key="sk-lf-...",
        team="ai-team"
    )

    # Your existing Langfuse code now includes GenOps governance
    from langfuse import observe

    @observe()
    def my_llm_function():
        # Automatically tracked with cost attribution and governance
        return openai.chat.completions.create(...)

    # Manual adapter usage for advanced governance
    from genops.providers.langfuse import GenOpsLangfuseAdapter

    adapter = GenOpsLangfuseAdapter(
        langfuse_public_key="pk-lf-...",
        langfuse_secret_key="sk-lf-...",
        team="research-team",
        project="llm-evaluation"
    )

    # Enhanced tracing with governance
    with adapter.trace_with_governance(
        name="research_analysis",
        customer_id="enterprise_123",
        cost_center="research"
    ) as trace:
        result = adapter.generation_with_cost_tracking(
            prompt="Analyze the market trends...",
            model="gpt-4",
            max_cost=0.50
        )
"""

import logging
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Import Langfuse with graceful failure
try:
    from langfuse import Langfuse
    from langfuse.client import StatefulClient
    from langfuse.decorators import observe

    HAS_LANGFUSE = True
except ImportError:
    HAS_LANGFUSE = False
    Langfuse = None
    observe = None
    StatefulClient = None
    logger.warning("Langfuse not installed. Install with: pip install langfuse")


class LangfuseObservationType(Enum):
    """Langfuse observation types for different operations."""

    GENERATION = "generation"
    TRACE = "trace"
    SPAN = "span"
    EVENT = "event"


class GovernancePolicy(Enum):
    """Governance policy enforcement levels."""

    ADVISORY = "advisory"  # Log policy violations but continue
    ENFORCED = "enforced"  # Block operations that violate policy
    AUDIT_ONLY = "audit_only"  # Track for compliance reporting


@dataclass
class LangfuseUsage:
    """Usage statistics from Langfuse operations with GenOps governance."""

    operation_id: str
    observation_type: str
    model: Optional[str]
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float
    latency_ms: float

    # GenOps governance attributes
    team: Optional[str] = None
    project: Optional[str] = None
    customer_id: Optional[str] = None
    cost_center: Optional[str] = None
    environment: str = "production"

    # Budget and policy tracking
    budget_remaining: Optional[float] = None
    policy_violations: list[str] = field(default_factory=list)
    governance_tags: dict[str, str] = field(default_factory=dict)


@dataclass
class LangfuseResponse:
    """Standardized response from Langfuse operations with governance."""

    content: str
    usage: LangfuseUsage
    trace_id: str
    observation_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    governance_status: str = "compliant"
    cost_optimization_suggestions: list[str] = field(default_factory=list)


class GenOpsLangfuseAdapter:
    """
    GenOps adapter for Langfuse with comprehensive governance integration.

    This adapter enhances Langfuse's observability capabilities with GenOps
    governance features including cost attribution, budget enforcement, and
    policy compliance tracking.
    """

    def __init__(
        self,
        langfuse_public_key: Optional[str] = None,
        langfuse_secret_key: Optional[str] = None,
        langfuse_base_url: str = "https://cloud.langfuse.com",
        team: Optional[str] = None,
        project: Optional[str] = None,
        environment: str = "production",
        enable_governance: bool = True,
        budget_limits: Optional[dict[str, float]] = None,
        policy_mode: GovernancePolicy = GovernancePolicy.ADVISORY,
    ):
        """
        Initialize GenOps Langfuse adapter with governance capabilities.

        Args:
            langfuse_public_key: Langfuse public API key
            langfuse_secret_key: Langfuse secret API key
            langfuse_base_url: Langfuse instance URL
            team: Team identifier for cost attribution
            project: Project identifier for tracking
            environment: Environment (dev/staging/production)
            enable_governance: Enable GenOps governance features
            budget_limits: Budget limits for cost enforcement
            policy_mode: Governance policy enforcement level
        """
        if not HAS_LANGFUSE:
            raise ImportError(
                "Langfuse package not found. Install with: pip install langfuse"
            )

        # Initialize Langfuse client
        self.langfuse = Langfuse(
            public_key=langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=langfuse_secret_key or os.getenv("LANGFUSE_SECRET_KEY"),
            host=langfuse_base_url
            or os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
        )

        # GenOps governance configuration
        self.team = team
        self.project = project
        self.environment = environment
        self.enable_governance = enable_governance
        self.budget_limits = budget_limits or {}
        self.policy_mode = policy_mode

        # Governance tracking
        self.current_costs = {"daily": 0.0, "monthly": 0.0}
        self.operation_count = 0
        self.policy_violations = []

        # Cost tracking configuration
        self.cost_per_token = {
            "gpt-4": {"input": 0.00003, "output": 0.00006},
            "gpt-3.5-turbo": {"input": 0.000001, "output": 0.000002},
            "claude-3-opus": {"input": 0.000015, "output": 0.000075},
            "claude-3-sonnet": {"input": 0.000003, "output": 0.000015},
            "claude-3-haiku": {"input": 0.00000025, "output": 0.00000125},
        }

        logger.info(
            f"GenOps Langfuse adapter initialized for team='{team}', project='{project}'"
        )

    def _calculate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate cost for LLM operation."""
        if model not in self.cost_per_token:
            # Default cost estimation for unknown models
            return (input_tokens + output_tokens) * 0.00001

        pricing = self.cost_per_token[model]
        input_cost = input_tokens * pricing["input"]
        output_cost = output_tokens * pricing["output"]
        return input_cost + output_cost

    def _check_budget_compliance(self, estimated_cost: float) -> bool:
        """Check if operation complies with budget limits."""
        if not self.budget_limits:
            return True

        daily_limit = self.budget_limits.get("daily", float("inf"))
        monthly_limit = self.budget_limits.get("monthly", float("inf"))

        if self.current_costs["daily"] + estimated_cost > daily_limit:
            self.policy_violations.append(
                f"Daily budget exceeded: ${self.current_costs['daily'] + estimated_cost:.6f} > ${daily_limit:.6f}"
            )
            return False

        if self.current_costs["monthly"] + estimated_cost > monthly_limit:
            self.policy_violations.append(
                f"Monthly budget exceeded: ${self.current_costs['monthly'] + estimated_cost:.6f} > ${monthly_limit:.6f}"
            )
            return False

        return True

    def _extract_governance_attributes(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Extract GenOps governance attributes from kwargs."""
        governance_attrs = {}

        # Standard governance attributes
        for attr in [
            "team",
            "project",
            "customer_id",
            "cost_center",
            "environment",
            "feature",
            "user_id",
        ]:
            if attr in kwargs:
                governance_attrs[attr] = kwargs.pop(attr)

        # Use defaults if not provided
        governance_attrs.setdefault("team", self.team)
        governance_attrs.setdefault("project", self.project)
        governance_attrs.setdefault("environment", self.environment)

        return governance_attrs

    @contextmanager
    def trace_with_governance(self, name: str, **kwargs) -> Iterator[Any]:
        """
        Create Langfuse trace enhanced with GenOps governance.

        Args:
            name: Trace name
            **kwargs: Additional parameters including governance attributes

        Yields:
            Langfuse trace with governance capabilities
        """
        governance_attrs = self._extract_governance_attributes(kwargs)

        # Create enhanced Langfuse trace
        trace = self.langfuse.trace(
            name=name,
            metadata={
                **kwargs.get("metadata", {}),
                "genops_governance": governance_attrs,
                "genops_version": "1.0.0",
                "governance_enabled": self.enable_governance,
            },
            tags=kwargs.get("tags", [])
            + [f"team:{governance_attrs.get('team', 'unknown')}"],
            **{k: v for k, v in kwargs.items() if k not in ["metadata", "tags"]},
        )

        start_time = time.time()
        operation_id = str(uuid.uuid4())

        try:
            logger.info(f"Starting governed trace: {name} (ID: {operation_id})")
            yield trace

        except Exception as e:
            # Log governance violation if applicable
            if self.enable_governance:
                trace.update(
                    metadata={
                        **trace.metadata,
                        "governance_error": str(e),
                        "policy_violations": self.policy_violations,
                    }
                )
            logger.error(f"Trace {name} failed: {e}")
            raise

        finally:
            duration_ms = (time.time() - start_time) * 1000

            # Update trace with governance metrics
            if self.enable_governance:
                trace.update(
                    metadata={
                        **trace.metadata,
                        "genops_duration_ms": duration_ms,
                        "genops_operation_count": self.operation_count,
                        "genops_policy_violations": len(self.policy_violations),
                    }
                )

            self.operation_count += 1
            logger.info(f"Trace {name} completed in {duration_ms:.2f}ms")

    def generation_with_cost_tracking(
        self, prompt: str, model: str, max_cost: Optional[float] = None, **kwargs
    ) -> LangfuseResponse:
        """
        Execute LLM generation with comprehensive cost tracking and governance.

        Args:
            prompt: Input prompt for the LLM
            model: Model identifier
            max_cost: Maximum allowed cost for this operation
            **kwargs: Additional parameters including governance attributes

        Returns:
            LangfuseResponse with usage and governance information
        """
        governance_attrs = self._extract_governance_attributes(kwargs)

        # Estimate cost for budget check
        estimated_input_tokens = len(prompt.split()) * 1.3  # Rough estimation
        estimated_cost = self._calculate_cost(model, int(estimated_input_tokens), 100)

        # Budget compliance check
        if max_cost and estimated_cost > max_cost:
            raise ValueError(
                f"Estimated cost ${estimated_cost:.6f} exceeds max_cost ${max_cost:.6f}"
            )

        if self.enable_governance and not self._check_budget_compliance(estimated_cost):
            if self.policy_mode == GovernancePolicy.ENFORCED:
                raise ValueError(
                    f"Budget limit exceeded. Violations: {self.policy_violations}"
                )

        # Create Langfuse generation with governance metadata
        start_time = time.time()
        operation_id = str(uuid.uuid4())

        generation = self.langfuse.generation(
            name=f"{model}_generation",
            model=model,
            input=prompt,
            metadata={
                **kwargs.get("metadata", {}),
                "genops_operation_id": operation_id,
                "genops_governance": governance_attrs,
                "genops_max_cost": max_cost,
                "genops_estimated_cost": estimated_cost,
            },
            tags=kwargs.get("tags", [])
            + [
                f"team:{governance_attrs.get('team', 'unknown')}",
                f"model:{model}",
                "genops_tracked",
            ],
        )

        # Simulate LLM call (in real implementation, this would call actual LLM)
        # For demo purposes, we'll create a mock response
        latency_ms = (time.time() - start_time) * 1000

        # Mock response data - in real implementation this would come from actual LLM
        mock_response = f"Generated response for: {prompt[:50]}..."
        input_tokens = len(prompt.split())
        output_tokens = len(mock_response.split())
        actual_cost = self._calculate_cost(model, input_tokens, output_tokens)

        # Update generation with results
        generation.end(
            output=mock_response,
            usage={
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens,
                "unit": "TOKENS",
            },
            metadata={
                **generation.metadata,
                "genops_actual_cost": actual_cost,
                "genops_duration_ms": latency_ms,
                "genops_cost_difference": actual_cost - estimated_cost,
            },
        )

        # Update governance tracking
        self.current_costs["daily"] += actual_cost
        self.current_costs["monthly"] += actual_cost

        # Create usage object
        usage = LangfuseUsage(
            operation_id=operation_id,
            observation_type=LangfuseObservationType.GENERATION.value,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost=actual_cost,
            latency_ms=latency_ms,
            **governance_attrs,
        )

        # Generate cost optimization suggestions
        suggestions = []
        if actual_cost > estimated_cost * 1.5:
            suggestions.append("Consider using a smaller model for similar tasks")
        if latency_ms > 5000:
            suggestions.append(
                "High latency detected - consider caching for repeated queries"
            )

        return LangfuseResponse(
            content=mock_response,
            usage=usage,
            trace_id=getattr(generation, "trace_id", ""),
            observation_id=generation.id,
            metadata=generation.metadata,
            cost_optimization_suggestions=suggestions,
        )

    def evaluate_with_governance(
        self,
        trace_id: str,
        evaluation_name: str,
        evaluator_function: Callable,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Run LLM evaluation with governance tracking.

        Args:
            trace_id: ID of trace to evaluate
            evaluation_name: Name of the evaluation
            evaluator_function: Function to run evaluation
            **kwargs: Additional parameters

        Returns:
            Evaluation results with governance metadata
        """
        governance_attrs = self._extract_governance_attributes(kwargs)

        start_time = time.time()

        try:
            # Run evaluation
            evaluation_result = evaluator_function()

            # Create Langfuse score with governance
            score = self.langfuse.score(
                trace_id=trace_id,
                name=evaluation_name,
                value=evaluation_result.get("score", 0.0),
                comment=evaluation_result.get("comment", ""),
                metadata={
                    **kwargs.get("metadata", {}),
                    "genops_governance": governance_attrs,
                    "genops_evaluation_duration_ms": (time.time() - start_time) * 1000,
                    "genops_evaluator": evaluator_function.__name__,
                },
            )

            return {
                "score": evaluation_result.get("score", 0.0),
                "evaluation_id": score.id,
                "governance": governance_attrs,
                "duration_ms": (time.time() - start_time) * 1000,
            }

        except Exception as e:
            logger.error(f"Evaluation {evaluation_name} failed: {e}")
            raise

    def get_cost_summary(self, time_period: str = "daily") -> dict[str, Any]:
        """
        Get cost summary with governance breakdown.

        Args:
            time_period: Time period for summary (daily/monthly)

        Returns:
            Cost summary with governance details
        """
        return {
            "period": time_period,
            "total_cost": self.current_costs.get(time_period, 0.0),
            "operation_count": self.operation_count,
            "average_cost_per_operation": (
                self.current_costs.get(time_period, 0.0) / max(self.operation_count, 1)
            ),
            "budget_limit": self.budget_limits.get(time_period),
            "budget_remaining": (
                self.budget_limits.get(time_period, 0.0)
                - self.current_costs.get(time_period, 0.0)
            ),
            "policy_violations": len(self.policy_violations),
            "governance": {
                "team": self.team,
                "project": self.project,
                "environment": self.environment,
                "policy_mode": self.policy_mode.value,
            },
        }


def instrument_langfuse(
    langfuse_public_key: Optional[str] = None,
    langfuse_secret_key: Optional[str] = None,
    langfuse_base_url: str = "https://cloud.langfuse.com",
    team: Optional[str] = None,
    project: Optional[str] = None,
    environment: str = "production",
    auto_instrument: bool = True,
    budget_limits: Optional[dict[str, float]] = None,
) -> GenOpsLangfuseAdapter:
    """
    Zero-code instrumentation for Langfuse with GenOps governance.

    This function enables automatic GenOps governance for all Langfuse-traced
    operations with zero code changes required.

    Args:
        langfuse_public_key: Langfuse public API key
        langfuse_secret_key: Langfuse secret API key
        langfuse_base_url: Langfuse instance URL
        team: Team identifier for cost attribution
        project: Project identifier for tracking
        environment: Environment (dev/staging/production)
        auto_instrument: Auto-instrument Langfuse decorators
        budget_limits: Budget limits for governance

    Returns:
        GenOpsLangfuseAdapter instance for manual usage
    """
    adapter = GenOpsLangfuseAdapter(
        langfuse_public_key=langfuse_public_key,
        langfuse_secret_key=langfuse_secret_key,
        langfuse_base_url=langfuse_base_url,
        team=team,
        project=project,
        environment=environment,
        budget_limits=budget_limits,
    )

    if auto_instrument:
        _auto_instrument_langfuse(adapter)

    logger.info(
        f"GenOps Langfuse instrumentation enabled for team='{team}', project='{project}'"
    )
    return adapter


def _auto_instrument_langfuse(adapter: GenOpsLangfuseAdapter):
    """Automatically enhance Langfuse decorators with GenOps governance."""
    try:
        if not HAS_LANGFUSE:
            logger.warning("Langfuse not available for auto-instrumentation")
            return

        # Enhance Langfuse observe decorator with governance
        original_observe = observe

        def enhanced_observe(*args, **kwargs):
            """Enhanced observe decorator with GenOps governance."""
            # Add governance metadata to all observations
            if "metadata" not in kwargs:
                kwargs["metadata"] = {}

            kwargs["metadata"].update(
                {
                    "genops_enabled": True,
                    "genops_team": adapter.team,
                    "genops_project": adapter.project,
                    "genops_environment": adapter.environment,
                }
            )

            return original_observe(*args, **kwargs)

        # Replace the observe decorator globally
        import langfuse.decorators

        langfuse.decorators.observe = enhanced_observe

        logger.info("Langfuse observe decorator enhanced with GenOps governance")

    except Exception as e:
        logger.warning(f"Failed to auto-instrument Langfuse: {e}")


# Convenience function for creating adapter
def create_langfuse_adapter(**kwargs) -> GenOpsLangfuseAdapter:
    """Create GenOps Langfuse adapter with configuration."""
    return GenOpsLangfuseAdapter(**kwargs)
