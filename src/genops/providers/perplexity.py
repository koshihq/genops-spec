"""
Perplexity AI Provider Adapter for GenOps AI Governance

Provides comprehensive governance for Perplexity AI operations including:
- Real-time web search with citation tracking
- Dual pricing model support (token costs + request fees)
- Search context optimization and cost intelligence
- Enterprise governance with multi-tenant support
- Zero-code auto-instrumentation for existing Perplexity integrations
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from genops.core.exceptions import (
    GenOpsBudgetExceededError,
    GenOpsConfigurationError,
)

# Core GenOps imports
from genops.core.telemetry import GenOpsTelemetry

# Import Perplexity pricing calculator
from .perplexity_pricing import PerplexityPricingCalculator

logger = logging.getLogger(__name__)

# Optional Perplexity dependencies
try:
    import openai  # Perplexity uses OpenAI-compatible client

    HAS_OPENAI_CLIENT = True
except ImportError:
    HAS_OPENAI_CLIENT = False
    logger.warning("OpenAI client not installed. Install with: pip install openai")

try:
    import requests  # noqa: F401

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("Requests not installed. Install with: pip install requests")


class SearchContext(Enum):
    """Perplexity search context depth levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PerplexityModel(Enum):
    """Available Perplexity models with their characteristics."""

    SONAR = "sonar"
    SONAR_PRO = "sonar-pro"
    SONAR_REASONING = "sonar-reasoning"
    SONAR_REASONING_PRO = "sonar-reasoning-pro"
    SONAR_DEEP_RESEARCH = "sonar-deep-research"


@dataclass
class SearchResult:
    """Search result with governance metadata."""

    query: str
    response: str
    citations: list[dict[str, Any]]
    search_context: SearchContext
    model_used: str
    tokens_used: int
    cost: Decimal
    search_time_seconds: float
    governance_metadata: dict[str, Any]
    session_id: str | None = None


@dataclass
class PerplexitySearchSession:
    """Search session with cost tracking and governance."""

    session_id: str
    session_name: str
    start_time: datetime
    end_time: datetime | None = None
    total_queries: int = 0
    total_cost: Decimal = Decimal("0")
    governance_attributes: dict[str, Any] = None  # type: ignore[assignment]
    search_results: list[SearchResult] = None  # type: ignore

    def __post_init__(self):
        if self.governance_attributes is None:
            self.governance_attributes = {}
        if self.search_results is None:
            self.search_results = []


class GenOpsPerplexityAdapter:
    """
    Perplexity AI adapter with GenOps governance for real-time web search.

    Provides comprehensive governance for Perplexity AI operations including:
    - Real-time web search with citation tracking and governance
    - Dual pricing model support (token costs + request fees)
    - Search context optimization and cost intelligence
    - Multi-tenant search operations with governance controls
    - Zero-code auto-instrumentation for existing integrations
    """

    def __init__(
        self,
        perplexity_api_key: str | None = None,
        team: str = "default",
        project: str = "default",
        environment: str = "production",
        customer_id: str | None = None,
        cost_center: str | None = None,
        daily_budget_limit: float = 1000.0,
        monthly_budget_limit: float | None = None,
        enable_governance: bool = True,
        enable_cost_alerts: bool = True,
        governance_policy: str = "advisory",  # advisory, enforced, strict
        default_search_context: SearchContext = SearchContext.MEDIUM,
        perplexity_base_url: str = "https://api.perplexity.ai",
        tags: dict[str, str] | None = None,
        **kwargs,
    ):
        """
        Initialize Perplexity adapter with governance configuration.

        Args:
            perplexity_api_key: Perplexity API key (or use PERPLEXITY_API_KEY env var)
            team: Team name for cost attribution and governance
            project: Project name for cost tracking
            environment: Environment (production, staging, development)
            customer_id: Customer ID for multi-tenant attribution
            cost_center: Cost center for financial reporting
            daily_budget_limit: Daily budget limit in USD
            monthly_budget_limit: Monthly budget limit in USD
            enable_governance: Enable governance controls
            enable_cost_alerts: Enable cost alerting
            governance_policy: Governance enforcement level
            default_search_context: Default search context depth
            perplexity_base_url: Perplexity API base URL
            tags: Additional tags for governance metadata
            **kwargs: Additional configuration options
        """
        # Configuration
        self.perplexity_api_key = perplexity_api_key or os.getenv("PERPLEXITY_API_KEY")
        self.team = team or os.getenv("GENOPS_TEAM", "default")
        self.project = project or os.getenv("GENOPS_PROJECT", "default")
        self.environment = environment
        self.customer_id = customer_id
        self.cost_center = cost_center
        self.daily_budget_limit = Decimal(str(daily_budget_limit))
        self.monthly_budget_limit = (
            Decimal(str(monthly_budget_limit)) if monthly_budget_limit else None
        )
        self.enable_governance = enable_governance
        self.enable_cost_alerts = enable_cost_alerts
        self.governance_policy = governance_policy
        self.default_search_context = default_search_context
        self.perplexity_base_url = perplexity_base_url
        self.tags = tags or {}

        # Cost tracking
        self.pricing_calculator = PerplexityPricingCalculator()
        self.daily_costs = Decimal("0")
        self.monthly_costs = Decimal("0")

        # Telemetry
        self.telemetry = GenOpsTelemetry(tracer_name="perplexity")

        # Active sessions
        self._active_sessions: dict[str, PerplexitySearchSession] = {}

        # Validation
        if not self.perplexity_api_key:
            raise GenOpsConfigurationError(
                "Perplexity API key required. Set PERPLEXITY_API_KEY environment variable or pass perplexity_api_key parameter."
            )

        # Initialize OpenAI client for Perplexity (compatible API)
        if HAS_OPENAI_CLIENT:
            self.client = openai.OpenAI(
                api_key=self.perplexity_api_key, base_url=self.perplexity_base_url
            )
        else:
            self.client = None
            logger.warning("OpenAI client not available. Some features may be limited.")

        logger.info(
            f"GenOps Perplexity adapter initialized for team='{self.team}', project='{self.project}'"
        )

    def _build_base_tags(
        self, additional_tags: dict[str, str] | None = None
    ) -> dict[str, str]:
        """Build base governance tags for telemetry."""
        base_tags = {
            "provider": "perplexity",
            "team": self.team,
            "project": self.project,
            "environment": self.environment,
            "governance_enabled": str(self.enable_governance),
            "governance_policy": self.governance_policy,
        }

        if self.customer_id:
            base_tags["customer_id"] = self.customer_id
        if self.cost_center:
            base_tags["cost_center"] = self.cost_center

        # Merge with instance tags and additional tags
        base_tags.update(self.tags)
        if additional_tags:
            base_tags.update(additional_tags)

        return base_tags

    def _check_budget_limits(self, estimated_cost: Decimal) -> None:
        """Check if operation would exceed budget limits."""
        if not self.enable_governance or self.governance_policy == "advisory":
            return

        projected_daily = self.daily_costs + estimated_cost
        if projected_daily > self.daily_budget_limit:
            if self.governance_policy in ["enforced", "strict"]:
                raise GenOpsBudgetExceededError(
                    f"Operation would exceed daily budget limit. "
                    f"Projected: ${projected_daily:.4f}, Limit: ${self.daily_budget_limit:.4f}"
                )

        if self.monthly_budget_limit:
            projected_monthly = self.monthly_costs + estimated_cost
            if projected_monthly > self.monthly_budget_limit:
                if self.governance_policy in ["enforced", "strict"]:
                    raise GenOpsBudgetExceededError(
                        f"Operation would exceed monthly budget limit. "
                        f"Projected: ${projected_monthly:.4f}, Limit: ${self.monthly_budget_limit:.4f}"
                    )

    def _update_costs(self, cost: Decimal) -> None:
        """Update cost tracking."""
        self.daily_costs += cost
        self.monthly_costs += cost

        # Cost alerting
        if self.enable_cost_alerts:
            daily_utilization = (self.daily_costs / self.daily_budget_limit) * 100
            if daily_utilization > 80:
                logger.warning(
                    f"Perplexity costs approaching daily limit: {daily_utilization:.1f}% "
                    f"(${self.daily_costs:.4f}/${self.daily_budget_limit:.4f})"
                )

    @contextmanager
    def track_search_session(
        self,
        session_name: str,
        customer_id: str | None = None,
        cost_center: str | None = None,
        environment: str | None = None,
        **governance_attributes,
    ) -> Iterator[PerplexitySearchSession]:
        """
        Context manager for tracking search sessions with governance.

        Args:
            session_name: Name of the search session
            customer_id: Customer ID override
            cost_center: Cost center override
            environment: Environment override
            **governance_attributes: Additional governance attributes

        Returns:
            PerplexitySearchSession: Session object for tracking

        Example:
            with adapter.track_search_session("competitive_analysis") as session:
                result = adapter.search_with_governance(
                    query="AI market trends 2024",
                    session_id=session.session_id
                )
        """
        session_id = str(uuid.uuid4())

        # Build governance attributes
        governance_attrs = self._build_base_tags()
        governance_attrs.update(
            {
                "session_name": session_name,
                "customer_id": customer_id or self.customer_id,
                "cost_center": cost_center or self.cost_center,
                "environment": environment or self.environment,
            }
        )
        governance_attrs.update(governance_attributes)

        # Create session
        session = PerplexitySearchSession(
            session_id=session_id,
            session_name=session_name,
            start_time=datetime.now(timezone.utc),
            governance_attributes=governance_attrs,
        )

        self._active_sessions[session_id] = session

        try:
            logger.info(
                f"Starting Perplexity search session '{session_name}' ({session_id})"
            )
            yield session
        finally:
            # Finalize session
            session.end_time = datetime.now(timezone.utc)
            session_duration = (session.end_time - session.start_time).total_seconds()

            logger.info(
                f"Completed Perplexity search session '{session_name}': "
                f"{session.total_queries} queries, ${session.total_cost:.4f} cost, "
                f"{session_duration:.1f}s duration"
            )

            # Remove from active sessions
            if session_id in self._active_sessions:
                del self._active_sessions[session_id]

    def search_with_governance(
        self,
        query: str,
        model: str | PerplexityModel = PerplexityModel.SONAR,
        search_context: SearchContext | None = None,
        session_id: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        return_citations: bool = True,
        return_images: bool = False,
        search_domain_filter: list[str] | None = None,
        search_recency_filter: str | None = None,
        **governance_attributes,
    ) -> SearchResult:
        """
        Perform web search with Perplexity AI and comprehensive governance.

        Args:
            query: Search query
            model: Perplexity model to use
            search_context: Search context depth (affects pricing)
            session_id: Optional session ID for tracking
            max_tokens: Maximum tokens in response
            temperature: Response temperature (0.0-1.0)
            return_citations: Include citations in response
            return_images: Include images in response
            search_domain_filter: Restrict search to specific domains
            search_recency_filter: Filter results by recency
            **governance_attributes: Additional governance metadata

        Returns:
            SearchResult: Search result with governance metadata

        Example:
            result = adapter.search_with_governance(
                query="Latest AI developments in healthcare",
                model=PerplexityModel.SONAR_PRO,
                search_context=SearchContext.HIGH,
                return_citations=True,
                team="research-team",
                project="ai-healthcare-analysis"
            )
        """
        if not HAS_OPENAI_CLIENT:
            raise GenOpsConfigurationError(
                "OpenAI client required for Perplexity integration"
            )

        start_time = time.time()

        # Normalize model
        if isinstance(model, PerplexityModel):
            model_name = model.value
        else:
            model_name = str(model)

        search_context = search_context or self.default_search_context

        # Estimate cost before operation
        estimated_cost = self.pricing_calculator.estimate_search_cost(
            model=model_name,
            estimated_tokens=max_tokens,
            search_context=search_context,  # type: ignore[arg-type]
        )

        # Budget check
        self._check_budget_limits(estimated_cost)

        # Build governance attributes
        operation_attrs = self._build_base_tags()
        operation_attrs.update(governance_attributes)
        operation_attrs.update(
            {
                "operation": "search",
                "model": model_name,
                "search_context": search_context.value,
                "query_length": len(query),
                "max_tokens": max_tokens,
                "estimated_cost": str(estimated_cost),
            }
        )

        # Prepare request
        messages = [{"role": "user", "content": query}]

        request_params = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

        # Add search-specific parameters
        if return_citations:
            request_params["return_citations"] = True
        if return_images:
            request_params["return_images"] = True
        if search_domain_filter:
            request_params["search_domain_filter"] = search_domain_filter
        if search_recency_filter:
            request_params["search_recency_filter"] = search_recency_filter

        try:
            # Execute search with telemetry
            with self.telemetry.trace_operation(
                "perplexity.search", **operation_attrs
            ) as span:
                response = self.client.chat.completions.create(**request_params)

                # Extract response data
                response_text = response.choices[0].message.content
                tokens_used = (
                    response.usage.total_tokens
                    if hasattr(response, "usage")
                    else max_tokens
                )

                # Extract citations (Perplexity-specific)
                citations = []
                if hasattr(response, "citations") and response.citations:
                    citations = [
                        {
                            "url": citation.get("url", ""),
                            "title": citation.get("title", ""),
                            "snippet": citation.get("snippet", ""),
                        }
                        for citation in response.citations
                    ]

                # Calculate actual cost
                actual_cost = self.pricing_calculator.calculate_search_cost(
                    model=model_name,
                    tokens_used=tokens_used,
                    search_context=search_context,  # type: ignore[arg-type]
                )

                # Update cost tracking
                self._update_costs(actual_cost)

                # Update telemetry
                span.set_attributes(
                    {
                        "perplexity.tokens_used": tokens_used,
                        "perplexity.actual_cost": str(actual_cost),
                        "perplexity.citations_count": len(citations),
                        "perplexity.search_time_seconds": time.time() - start_time,
                    }
                )

                # Create result
                search_result = SearchResult(
                    query=query,
                    response=response_text,
                    citations=citations,
                    search_context=search_context,
                    model_used=model_name,
                    tokens_used=tokens_used,
                    cost=actual_cost,
                    search_time_seconds=time.time() - start_time,
                    governance_metadata=operation_attrs,
                    session_id=session_id,
                )

                # Update session if provided
                if session_id and session_id in self._active_sessions:
                    session = self._active_sessions[session_id]
                    session.total_queries += 1
                    session.total_cost += actual_cost
                    session.search_results.append(search_result)

                logger.info(
                    f"Perplexity search completed: {tokens_used} tokens, "
                    f"${actual_cost:.4f} cost, {len(citations)} citations"
                )

                return search_result

        except Exception as e:
            logger.error(f"Perplexity search failed: {e}")
            # Update telemetry with error
            if "span" in locals():
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))  # type: ignore  # noqa: F821
            raise

    def batch_search_with_governance(
        self,
        queries: list[str],
        model: str | PerplexityModel = PerplexityModel.SONAR,
        search_context: SearchContext | None = None,
        session_id: str | None = None,
        **governance_attributes,
    ) -> list[SearchResult]:
        """
        Perform batch search operations with cost optimization.

        Args:
            queries: List of search queries
            model: Perplexity model to use
            search_context: Search context depth
            session_id: Optional session ID for tracking
            **governance_attributes: Additional governance metadata

        Returns:
            List[SearchResult]: List of search results
        """
        results = []

        for i, query in enumerate(queries):
            try:
                result = self.search_with_governance(
                    query=query,
                    model=model,
                    search_context=search_context,
                    session_id=session_id,
                    batch_index=i,
                    batch_total=len(queries),
                    **governance_attributes,
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Batch search query {i + 1}/{len(queries)} failed: {e}")
                # Continue with remaining queries
                continue

        return results

    def get_cost_summary(self) -> dict[str, Any]:
        """
        Get comprehensive cost summary and analytics.

        Returns:
            Dict with cost summary, budget utilization, and recommendations
        """
        summary = {
            "daily_costs": float(self.daily_costs),
            "monthly_costs": float(self.monthly_costs),
            "daily_budget_limit": float(self.daily_budget_limit),
            "monthly_budget_limit": float(self.monthly_budget_limit)
            if self.monthly_budget_limit
            else None,
            "daily_budget_utilization": (
                self.daily_costs / self.daily_budget_limit * 100
            )
            if self.daily_budget_limit > 0
            else 0,
            "monthly_budget_utilization": (
                (self.monthly_costs / self.monthly_budget_limit * 100)
                if self.monthly_budget_limit and self.monthly_budget_limit > 0
                else 0
            ),
            "governance_enabled": self.enable_governance,
            "governance_policy": self.governance_policy,
            "active_sessions": len(self._active_sessions),
            "team": self.team,
            "project": self.project,
            "environment": self.environment,
        }

        return summary

    def get_search_cost_analysis(
        self, projected_queries: int, model: str = "sonar"
    ) -> dict[str, Any]:
        """
        Analyze projected search costs and provide optimization recommendations.

        Args:
            projected_queries: Number of queries to analyze
            model: Model to analyze costs for

        Returns:
            Cost analysis with optimization recommendations
        """
        return self.pricing_calculator.analyze_search_costs(  # type: ignore[return-value]
            projected_queries=projected_queries,
            model=model,
            current_daily_costs=self.daily_costs,
            daily_budget_limit=self.daily_budget_limit,
        )


# Auto-instrumentation functions
_current_adapter: GenOpsPerplexityAdapter | None = None


def auto_instrument(
    perplexity_api_key: str | None = None,
    team: str = "auto-instrumented",
    project: str = "default",
    **adapter_kwargs,
) -> GenOpsPerplexityAdapter:
    """
        Enable automatic instrumentation for Perplexity AI operations.

        This function enables zero-code governance for existing Perplexity integrations.
        Once called, all Perplexity operations will be automatically tracked with cost
        attribution and governance controls.

        Args:
            perplexity_api_key: Perplexity API key (or use PERPLEXITY_API_KEY env var)
            team: Team name for cost attribution
            project: Project name for cost tracking
            **adapter_kwargs: Additional adapter configuration

        Returns:
            GenOpsPerplexityAdapter: The configured adapter instance

        Example:
            # Add ONE line to enable governance for all Perplexity operations
            from genops.providers.perplexity import auto_instrument
            auto_instrument()

            # Your existing code works unchanged with governance
            import openai
    from opentelemetry import trace
            client = openai.OpenAI(
                api_key="your-perplexity-key",
                base_url="https://api.perplexity.ai"
            )
            response = client.chat.completions.create(
                model="sonar-pro",
                messages=[{"role": "user", "content": "AI trends 2024"}]
            )
    """
    global _current_adapter

    _current_adapter = GenOpsPerplexityAdapter(
        perplexity_api_key=perplexity_api_key,
        team=team,
        project=project,
        **adapter_kwargs,
    )

    logger.info("Perplexity auto-instrumentation enabled")
    return _current_adapter


def instrument_perplexity(
    perplexity_api_key: str | None = None,
    team: str = "default",
    project: str = "default",
    **kwargs,
) -> GenOpsPerplexityAdapter:
    """
    Create instrumented Perplexity adapter.

    Alternative entry point for creating a GenOps Perplexity adapter with
    governance controls and cost tracking.

    Args:
        perplexity_api_key: Perplexity API key
        team: Team name for attribution
        project: Project name for tracking
        **kwargs: Additional configuration

    Returns:
        GenOpsPerplexityAdapter: Configured adapter
    """
    return GenOpsPerplexityAdapter(
        perplexity_api_key=perplexity_api_key, team=team, project=project, **kwargs
    )


def get_current_adapter() -> GenOpsPerplexityAdapter | None:
    """Get the current auto-instrumented adapter instance."""
    return _current_adapter


# Export key classes and functions
__all__ = [
    "GenOpsPerplexityAdapter",
    "PerplexitySearchSession",
    "SearchResult",
    "SearchContext",
    "PerplexityModel",
    "auto_instrument",
    "instrument_perplexity",
    "get_current_adapter",
]
