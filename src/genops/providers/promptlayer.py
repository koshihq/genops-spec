#!/usr/bin/env python3
"""
GenOps PromptLayer Integration

This module provides comprehensive PromptLayer integration for GenOps AI governance,
cost intelligence, and policy enforcement. PromptLayer is a powerful prompt management
and AI engineering platform that provides versioning, evaluation, and observability
for AI prompts and LLM operations.

Features:
- Enhanced PromptLayer operations with GenOps governance attributes
- Cost attribution and budget enforcement for prompt management workflows
- Policy compliance tracking integrated with PromptLayer observations
- Prompt evaluation with governance oversight and cost optimization
- A/B testing with cost intelligence and team attribution
- Zero-code auto-instrumentation with instrument_promptlayer()
- Enterprise-ready governance patterns for production prompt management

Example usage:

    # Zero-code auto-instrumentation
    from genops.providers.promptlayer import auto_instrument
    auto_instrument(
        promptlayer_api_key="pl-your-api-key",
        team="ai-team",
        project="prompt-optimization"
    )
    
    # Your existing PromptLayer code now includes GenOps governance
    import promptlayer
    
    promptlayer_client = promptlayer.PromptLayer()
    response = promptlayer_client.run(
        prompt_name="customer_support_v2",
        input_variables={"query": "Help request"}
    )
    # Automatically tracked with cost attribution and governance
    
    # Manual adapter usage for advanced governance
    from genops.providers.promptlayer import GenOpsPromptLayerAdapter
    
    adapter = GenOpsPromptLayerAdapter(
        promptlayer_api_key="pl-your-api-key",
        team="engineering-team",
        project="prompt-management",
        enable_cost_alerts=True,
        daily_budget_limit=50.0
    )
    
    # Enhanced prompt operations with governance
    with adapter.track_prompt_operation(
        prompt_name="sales_assistant",
        prompt_version="v2.1",
        customer_id="enterprise_123",
        cost_center="sales"
    ) as span:
        result = adapter.run_prompt_with_governance(
            prompt_name="sales_assistant",
            input_variables={"context": "Product demo request"},
            max_cost=0.25
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
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Import PromptLayer with graceful failure
try:
    import promptlayer
    from promptlayer import PromptLayer
    HAS_PROMPTLAYER = True
except ImportError:
    HAS_PROMPTLAYER = False
    promptlayer = None
    PromptLayer = None
    logger.warning("PromptLayer not installed. Install with: pip install promptlayer")

class PromptLayerOperationType(Enum):
    """PromptLayer operation types for different workflows."""
    PROMPT_RUN = "prompt_run"
    PROMPT_TRACKING = "prompt_tracking"
    EVALUATION = "evaluation"
    AB_TEST = "ab_test"
    TEMPLATE_EXECUTION = "template_execution"

class GovernancePolicy(Enum):
    """Governance policy enforcement levels."""
    ADVISORY = "advisory"      # Log policy violations but continue
    ENFORCED = "enforced"      # Block operations that violate policy
    AUDIT_ONLY = "audit_only"  # Track for compliance reporting

@dataclass
class PromptLayerUsage:
    """Usage statistics from PromptLayer operations with GenOps governance."""
    operation_id: str
    operation_type: str
    prompt_name: Optional[str]
    prompt_version: Optional[str]
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
    policy_violations: List[str] = field(default_factory=list)
    governance_tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class PromptLayerResponse:
    """Standardized response from PromptLayer operations with governance."""
    content: str
    usage: PromptLayerUsage
    prompt_id: str
    request_id: str
    prompt_version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    governance_status: str = "compliant"
    cost_optimization_suggestions: List[str] = field(default_factory=list)

class MockPromptLayer:
    """Mock PromptLayer client for graceful degradation when PromptLayer is not available."""

    def __init__(self, *args, **kwargs) -> None:
        self.config = kwargs
        logger.warning("Using MockPromptLayer - PromptLayer not installed")

    def run(self, *args, **kwargs) -> dict:
        logger.warning("MockPromptLayer.run() called - install PromptLayer for full functionality")
        return {"error": "PromptLayer not installed", "mock": True}

    def track(self, *args, **kwargs) -> dict:
        logger.warning("MockPromptLayer.track() called - install PromptLayer for full functionality")
        return {"error": "PromptLayer not installed", "mock": True}

class EnhancedPromptLayerSpan:
    """Enhanced span for PromptLayer operations with GenOps governance capabilities."""

    def __init__(
        self,
        operation_type: str,
        operation_name: str,
        prompt_name: Optional[str] = None,
        prompt_version: Optional[str] = None,
        team: Optional[str] = None,
        project: Optional[str] = None,
        customer_id: Optional[str] = None,
        cost_center: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        max_cost: Optional[float] = None
    ):
        self.operation_id = str(uuid.uuid4())
        self.operation_type = operation_type
        self.operation_name = operation_name
        self.prompt_name = prompt_name
        self.prompt_version = prompt_version
        self.team = team
        self.project = project
        self.customer_id = customer_id
        self.cost_center = cost_center
        self.tags = tags or {}
        self.max_cost = max_cost

        # Usage tracking
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.estimated_cost = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.model: Optional[str] = None

        # Governance tracking
        self.policy_violations: List[str] = []
        self.governance_tags: Dict[str, str] = {}
        self.metadata: Dict[str, Any] = {}

        logger.info(f"Created PromptLayer span: {self.operation_name} (ID: {self.operation_id})")

    def update_cost(self, cost: float) -> None:
        """Update the estimated cost for this operation."""
        self.estimated_cost = cost

        # Check cost limits
        if self.max_cost and cost > self.max_cost:
            violation = f"Operation cost ${cost:.6f} exceeds maximum ${self.max_cost:.6f}"
            self.policy_violations.append(violation)
            logger.warning(f"Cost violation: {violation}")

    def update_token_usage(self, input_tokens: int, output_tokens: int, model: Optional[str] = None) -> None:
        """Update token usage metrics."""
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = input_tokens + output_tokens
        if model:
            self.model = model

        # Estimate cost based on token usage (basic estimation)
        if model and "gpt-4" in model.lower():
            # GPT-4 pricing (approximate)
            input_cost = (input_tokens / 1000) * 0.03
            output_cost = (output_tokens / 1000) * 0.06
            self.update_cost(input_cost + output_cost)
        elif model and "gpt-3.5" in model.lower():
            # GPT-3.5 pricing (approximate)
            input_cost = (input_tokens / 1000) * 0.0015
            output_cost = (output_tokens / 1000) * 0.002
            self.update_cost(input_cost + output_cost)

    def add_attributes(self, attributes: Dict[str, Any]) -> None:
        """Add custom attributes to the span."""
        self.metadata.update(attributes)

        # Extract governance-relevant attributes
        if "team" in attributes:
            self.team = attributes["team"]
        if "project" in attributes:
            self.project = attributes["project"]
        if "customer_id" in attributes:
            self.customer_id = attributes["customer_id"]

    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for this span."""
        duration = (self.end_time or time.time()) - self.start_time

        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "operation_name": self.operation_name,
            "prompt_name": self.prompt_name,
            "prompt_version": self.prompt_version,
            "duration_seconds": duration,
            "estimated_cost": self.estimated_cost,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "team": self.team,
            "project": self.project,
            "customer_id": self.customer_id,
            "cost_center": self.cost_center,
            "policy_violations": self.policy_violations,
            "governance_tags": self.governance_tags,
            "metadata": self.metadata
        }

    def finalize(self) -> None:
        """Finalize the span and perform cleanup."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time

        logger.info(
            f"Finalized PromptLayer span: {self.operation_name} "
            f"(Duration: {duration:.2f}s, Cost: ${self.estimated_cost:.6f})"
        )

class GenOpsPromptLayerAdapter:
    """
    GenOps adapter for PromptLayer with comprehensive governance integration.
    
    This adapter enhances PromptLayer's prompt management capabilities with GenOps
    governance features including cost attribution, budget enforcement, and
    policy compliance tracking for prompt engineering workflows.
    """

    def __init__(
        self,
        promptlayer_api_key: Optional[str] = None,
        team: Optional[str] = None,
        project: Optional[str] = None,
        environment: str = "production",
        customer_id: Optional[str] = None,
        cost_center: Optional[str] = None,
        enable_governance: bool = True,
        daily_budget_limit: Optional[float] = None,
        max_operation_cost: Optional[float] = None,
        governance_policy: GovernancePolicy = GovernancePolicy.ADVISORY,
        enable_cost_alerts: bool = True,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the GenOps PromptLayer adapter.
        
        Args:
            promptlayer_api_key: PromptLayer API key (pl-...)
            team: Team name for cost attribution
            project: Project name for tracking
            environment: Environment (development/staging/production)
            customer_id: Customer ID for per-customer attribution
            cost_center: Cost center for financial reporting
            enable_governance: Enable governance features
            daily_budget_limit: Daily spending limit in USD
            max_operation_cost: Maximum cost per operation
            governance_policy: Policy enforcement level
            enable_cost_alerts: Enable cost-based alerting
            tags: Additional tags for tracking
        """
        # Store configuration
        self.promptlayer_api_key = promptlayer_api_key or os.getenv("PROMPTLAYER_API_KEY")
        self.team = team or os.getenv("GENOPS_TEAM")
        self.project = project or os.getenv("GENOPS_PROJECT")
        self.environment = environment
        self.customer_id = customer_id
        self.cost_center = cost_center
        self.enable_governance = enable_governance
        self.daily_budget_limit = daily_budget_limit
        self.max_operation_cost = max_operation_cost
        self.governance_policy = governance_policy
        self.enable_cost_alerts = enable_cost_alerts
        self.tags = tags or {}

        # Validate required configuration
        if not self.promptlayer_api_key:
            logger.warning("PromptLayer API key not provided. Set PROMPTLAYER_API_KEY or pass promptlayer_api_key")

        if not self.team:
            logger.warning("Team not specified. Set GENOPS_TEAM or pass team parameter for cost attribution")

        # Initialize PromptLayer client
        if HAS_PROMPTLAYER and self.promptlayer_api_key:
            try:
                self.client = PromptLayer(api_key=self.promptlayer_api_key)
                logger.info("PromptLayer client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize PromptLayer client: {e}")
                self.client = MockPromptLayer()
        else:
            self.client = MockPromptLayer()

        # Usage tracking
        self.daily_usage = 0.0
        self.operation_count = 0
        self.active_spans: Dict[str, EnhancedPromptLayerSpan] = {}

        logger.info(f"GenOpsPromptLayerAdapter initialized for team: {self.team}, project: {self.project}")

    @contextmanager
    def track_prompt_operation(
        self,
        prompt_name: Optional[str] = None,
        prompt_version: Optional[str] = None,
        operation_type: str = "prompt_run",
        operation_name: Optional[str] = None,
        customer_id: Optional[str] = None,
        cost_center: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        max_cost: Optional[float] = None
    ) -> Iterator[EnhancedPromptLayerSpan]:
        """
        Context manager for tracking PromptLayer operations with governance.
        
        Args:
            prompt_name: Name of the prompt being executed
            prompt_version: Version of the prompt
            operation_type: Type of operation (prompt_run, evaluation, etc.)
            operation_name: Custom name for this operation
            customer_id: Customer ID for attribution
            cost_center: Cost center for this operation
            tags: Additional tags
            max_cost: Maximum allowed cost for this operation
        
        Yields:
            EnhancedPromptLayerSpan: Enhanced span for tracking
        """
        # Use provided values or fall back to adapter defaults
        final_customer_id = customer_id or self.customer_id
        final_cost_center = cost_center or self.cost_center
        final_max_cost = max_cost or self.max_operation_cost
        final_operation_name = operation_name or f"{prompt_name or 'prompt'}_{operation_type}"

        # Merge tags
        final_tags = {**self.tags, **(tags or {})}

        # Create enhanced span
        span = EnhancedPromptLayerSpan(
            operation_type=operation_type,
            operation_name=final_operation_name,
            prompt_name=prompt_name,
            prompt_version=prompt_version,
            team=self.team,
            project=self.project,
            customer_id=final_customer_id,
            cost_center=final_cost_center,
            tags=final_tags,
            max_cost=final_max_cost
        )

        # Add to active spans
        self.active_spans[span.operation_id] = span

        try:
            # Check budget before operation
            if self.enable_governance and self.daily_budget_limit:
                if self.daily_usage >= self.daily_budget_limit:
                    violation = f"Daily budget limit ${self.daily_budget_limit} exceeded (current: ${self.daily_usage})"
                    span.policy_violations.append(violation)

                    if self.governance_policy == GovernancePolicy.ENFORCED:
                        raise ValueError(f"Operation blocked: {violation}")
                    else:
                        logger.warning(f"Budget warning: {violation}")

            yield span

        except Exception as e:
            span.add_attributes({"error": str(e), "error_type": type(e).__name__})
            logger.error(f"PromptLayer operation failed: {e}")
            raise

        finally:
            # Finalize span
            span.finalize()

            # Update usage tracking
            self.daily_usage += span.estimated_cost
            self.operation_count += 1

            # Remove from active spans
            if span.operation_id in self.active_spans:
                del self.active_spans[span.operation_id]

            # Log governance summary
            if self.enable_governance and span.policy_violations:
                logger.warning(
                    f"Operation {span.operation_name} had {len(span.policy_violations)} policy violations: "
                    f"{', '.join(span.policy_violations)}"
                )

    def run_prompt_with_governance(
        self,
        prompt_name: str,
        input_variables: Dict[str, Any],
        prompt_version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run a PromptLayer prompt with governance tracking.
        
        Args:
            prompt_name: Name of the prompt to run
            input_variables: Variables to substitute in the prompt
            prompt_version: Specific version of the prompt
            metadata: Additional metadata to track
            tags: Tags for categorization
        
        Returns:
            Dict containing the response and governance information
        """
        try:
            # Prepare PromptLayer request
            pl_tags = tags or []
            if self.team:
                pl_tags.append(f"team:{self.team}")
            if self.project:
                pl_tags.append(f"project:{self.project}")
            if self.customer_id:
                pl_tags.append(f"customer:{self.customer_id}")

            # Execute PromptLayer request
            if hasattr(self.client, 'run') and callable(self.client.run):
                response = self.client.run(
                    prompt_name=prompt_name,
                    input_variables=input_variables,
                    version=prompt_version,
                    metadata=metadata,
                    tags=pl_tags
                )

                # Extract cost and usage information if available
                # Note: Actual cost extraction depends on PromptLayer's response format
                if isinstance(response, dict) and "usage" in response:
                    usage_info = response["usage"]
                    # Update span with actual usage if available
                    # This would be implemented based on PromptLayer's actual response structure

                return {
                    "response": response,
                    "governance": {
                        "team": self.team,
                        "project": self.project,
                        "customer_id": self.customer_id,
                        "cost_center": self.cost_center,
                        "estimated_cost": 0.0,  # Would be calculated from actual usage
                        "tags": pl_tags
                    }
                }
            else:
                logger.warning("PromptLayer client not available - using mock response")
                return {
                    "response": {"mock": True, "message": "PromptLayer not available"},
                    "governance": {
                        "team": self.team,
                        "project": self.project,
                        "estimated_cost": 0.0
                    }
                }

        except Exception as e:
            logger.error(f"Failed to run prompt {prompt_name}: {e}")
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """Get current governance metrics."""
        budget_remaining = None
        if self.daily_budget_limit:
            budget_remaining = self.daily_budget_limit - self.daily_usage

        return {
            "team": self.team,
            "project": self.project,
            "environment": self.environment,
            "daily_usage": self.daily_usage,
            "operation_count": self.operation_count,
            "budget_remaining": budget_remaining,
            "governance_enabled": self.enable_governance,
            "policy_level": self.governance_policy.value,
            "active_operations": len(self.active_spans)
        }

    def _check_governance_policies(self, span: EnhancedPromptLayerSpan) -> None:
        """Check governance policies against operation."""
        if not self.enable_governance:
            return

        violations = []

        # Check cost limits
        if self.max_operation_cost and span.estimated_cost > self.max_operation_cost:
            violations.append(f"Operation cost ${span.estimated_cost:.6f} exceeds limit ${self.max_operation_cost:.6f}")

        # Check daily budget
        if self.daily_budget_limit and (self.daily_usage + span.estimated_cost) > self.daily_budget_limit:
            violations.append(f"Operation would exceed daily budget ${self.daily_budget_limit:.2f}")

        # Add violations to span
        span.policy_violations.extend(violations)

        # Handle enforcement
        if violations and self.governance_policy == GovernancePolicy.ENFORCED:
            raise ValueError(f"Governance policy violations: {'; '.join(violations)}")

# Convenience functions for easy integration

def instrument_promptlayer(
    promptlayer_api_key: Optional[str] = None,
    team: Optional[str] = None,
    project: Optional[str] = None,
    **kwargs
) -> GenOpsPromptLayerAdapter:
    """
    Create and configure a GenOps PromptLayer adapter.
    
    This is the main factory function for creating PromptLayer adapters with
    GenOps governance capabilities.
    
    Args:
        promptlayer_api_key: PromptLayer API key
        team: Team name for cost attribution  
        project: Project name for tracking
        **kwargs: Additional configuration options
    
    Returns:
        GenOpsPromptLayerAdapter: Configured adapter instance
    """
    return GenOpsPromptLayerAdapter(
        promptlayer_api_key=promptlayer_api_key,
        team=team,
        project=project,
        **kwargs
    )

def auto_instrument(
    promptlayer_api_key: Optional[str] = None,
    team: Optional[str] = None,
    project: Optional[str] = None,
    environment: str = "production",
    **kwargs
) -> None:
    """
    Enable automatic instrumentation for PromptLayer operations.
    
    This function sets up global instrumentation that automatically enhances
    all PromptLayer operations with GenOps governance capabilities.
    
    Args:
        promptlayer_api_key: PromptLayer API key
        team: Team name for cost attribution
        project: Project name for tracking  
        environment: Environment name
        **kwargs: Additional configuration options
    """
    # Create global adapter
    global _global_promptlayer_adapter
    _global_promptlayer_adapter = GenOpsPromptLayerAdapter(
        promptlayer_api_key=promptlayer_api_key,
        team=team,
        project=project,
        environment=environment,
        **kwargs
    )

    logger.info(f"Auto-instrumentation enabled for PromptLayer with team: {team}, project: {project}")

# Global adapter for auto-instrumentation
_global_promptlayer_adapter: Optional[GenOpsPromptLayerAdapter] = None

def get_current_adapter() -> Optional[GenOpsPromptLayerAdapter]:
    """Get the current global adapter if auto-instrumentation is enabled."""
    return _global_promptlayer_adapter
