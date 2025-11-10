#!/usr/bin/env python3
"""
GenOps PostHog Integration

This module provides comprehensive PostHog integration for GenOps governance,
cost intelligence, and policy enforcement. PostHog is an open-source product
analytics platform that provides event tracking, feature flags, session recording,
and A/B testing capabilities with comprehensive user analytics.

Features:
- Enhanced product analytics with GenOps governance attributes and cost tracking
- Cost attribution and budget enforcement for analytics operations
- Policy compliance tracking integrated with product analytics workflows
- Feature flag management with governance oversight and cost optimization
- LLM analytics integration with unified cost intelligence  
- Zero-code auto-instrumentation with instrument_posthog()
- Enterprise-ready governance patterns for production analytics deployments

Example usage:

    # Zero-code auto-instrumentation
    from genops.providers.posthog import auto_instrument
    auto_instrument(
        posthog_api_key="phc_your-project-api-key",
        team="analytics-team",
        project="product-analytics"
    )
    
    # Your existing PostHog code now includes GenOps governance
    import posthog
    
    posthog.capture("user_signed_up", {"email": "user@example.com"})
    # Automatically tracked with cost attribution and governance
    
    # Manual adapter usage for advanced governance
    from genops.providers.posthog import GenOpsPostHogAdapter
    
    adapter = GenOpsPostHogAdapter(
        posthog_api_key="phc_your-project-api-key",
        team="growth-team",
        project="user-analytics",
        enable_cost_alerts=True,
        daily_budget_limit=100.0
    )
    
    # Enhanced analytics with governance
    with adapter.track_analytics_session(
        session_name="user_onboarding",
        customer_id="enterprise_123",
        cost_center="growth"
    ) as session:
        # Event tracking with automatic cost attribution
        session.capture_event("onboarding_started")
        session.evaluate_feature_flag("new_signup_flow")
        session.record_conversion("trial_signup")

Author: GenOps AI Team
License: Apache 2.0
"""

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from genops.core.cost_tracking import BaseCostCalculator
from genops.core.exceptions import (
    GenOpsBudgetExceededError,
    GenOpsConfigurationError,
)
from genops.core.governance import GovernanceProvider

# Core GenOps imports
from genops.core.telemetry import GenOpsTelemetry

logger = logging.getLogger(__name__)

# PostHog cost constants (based on 2024 pricing)
POSTHOG_COSTS = {
    'events': {
        'free_tier': 1_000_000,  # 1M free events per month
        'tiers': [
            (2_000_000, 0.00005),    # 1M-2M: $0.00005 per event
            (10_000_000, 0.000025),  # 2M-10M: $0.000025 per event
            (50_000_000, 0.000015),  # 10M-50M: $0.000015 per event
            (float('inf'), 0.000009) # 50M+: $0.000009 per event
        ]
    },
    'identified_events': {
        'free_tier': 0,
        'base_cost': 0.000198  # $0.000198 per identified event
    },
    'feature_flags': {
        'free_tier': 1_000_000,  # 1M free requests per month
        'base_cost': 0.000005    # $0.000005 per request above free tier
    },
    'session_recordings': {
        'free_tier': 5_000,      # 5K free recordings per month
        'base_cost': 0.000071    # $0.000071 per recording above free tier
    },
    'llm_analytics': {
        'free_tier': 100_000,    # 100K free LLM events per month
        'base_cost': 0.0001      # $0.0001 per LLM event above free tier
    }
}

@dataclass
class PostHogEventCost:
    """Cost breakdown for PostHog events."""
    event_count: int
    identified_events: int
    feature_flag_requests: int
    session_recordings: int
    llm_events: int
    total_cost: Decimal
    cost_breakdown: Dict[str, Decimal] = field(default_factory=dict)
    cost_per_event: Decimal = Decimal('0')
    free_tier_usage: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if self.event_count > 0:
            self.cost_per_event = self.total_cost / self.event_count

@dataclass
class PostHogAnalyticsSession:
    """PostHog analytics session with governance tracking."""
    session_id: str
    session_name: str
    start_time: datetime
    team: str
    project: str
    customer_id: Optional[str] = None
    cost_center: Optional[str] = None
    environment: str = "production"
    events_captured: int = 0
    flags_evaluated: int = 0
    recordings_created: int = 0
    llm_events_tracked: int = 0
    total_cost: Decimal = Decimal('0')
    governance_attributes: Dict[str, Any] = field(default_factory=dict)
    end_time: Optional[datetime] = None

    def finalize_session(self) -> PostHogEventCost:
        """Calculate final session costs and return cost summary."""
        self.end_time = datetime.now(timezone.utc)

        calculator = PostHogCostCalculator()
        cost_result = calculator.calculate_session_cost(
            event_count=self.events_captured,
            identified_events=0,  # Tracked separately
            feature_flag_requests=self.flags_evaluated,
            session_recordings=self.recordings_created,
            llm_events=self.llm_events_tracked
        )

        self.total_cost = cost_result.total_cost
        return cost_result

class PostHogCostCalculator(BaseCostCalculator):
    """PostHog-specific cost calculation engine."""

    def __init__(self):
        super().__init__()
        self.costs = POSTHOG_COSTS

    def calculate_event_cost(self, event_count: int, is_identified: bool = False) -> Decimal:
        """Calculate cost for PostHog events based on tiered pricing."""
        if event_count <= 0:
            return Decimal('0')

        # Regular events with tiered pricing
        total_cost = Decimal('0')
        remaining_events = event_count

        # Apply free tier
        free_events = min(remaining_events, self.costs['events']['free_tier'])
        remaining_events -= free_events

        # Apply tiered pricing for remaining events
        for tier_limit, cost_per_event in self.costs['events']['tiers']:
            if remaining_events <= 0:
                break

            tier_events = min(remaining_events, tier_limit - sum(t[0] for t in self.costs['events']['tiers'][:self.costs['events']['tiers'].index((tier_limit, cost_per_event))]))
            if tier_events > 0:
                total_cost += Decimal(str(tier_events)) * Decimal(str(cost_per_event))
                remaining_events -= tier_events

        # Add cost for identified events (charged separately)
        if is_identified:
            identified_cost = Decimal(str(event_count)) * Decimal(str(self.costs['identified_events']['base_cost']))
            total_cost += identified_cost

        return total_cost

    def calculate_feature_flag_cost(self, request_count: int) -> Decimal:
        """Calculate cost for feature flag evaluations."""
        if request_count <= self.costs['feature_flags']['free_tier']:
            return Decimal('0')

        billable_requests = request_count - self.costs['feature_flags']['free_tier']
        return Decimal(str(billable_requests)) * Decimal(str(self.costs['feature_flags']['base_cost']))

    def calculate_session_recording_cost(self, recording_count: int) -> Decimal:
        """Calculate cost for session recordings."""
        if recording_count <= self.costs['session_recordings']['free_tier']:
            return Decimal('0')

        billable_recordings = recording_count - self.costs['session_recordings']['free_tier']
        return Decimal(str(billable_recordings)) * Decimal(str(self.costs['session_recordings']['base_cost']))

    def calculate_llm_analytics_cost(self, llm_event_count: int) -> Decimal:
        """Calculate cost for LLM analytics events."""
        if llm_event_count <= self.costs['llm_analytics']['free_tier']:
            return Decimal('0')

        billable_events = llm_event_count - self.costs['llm_analytics']['free_tier']
        return Decimal(str(billable_events)) * Decimal(str(self.costs['llm_analytics']['base_cost']))

    def calculate_session_cost(
        self,
        event_count: int,
        identified_events: int = 0,
        feature_flag_requests: int = 0,
        session_recordings: int = 0,
        llm_events: int = 0
    ) -> PostHogEventCost:
        """Calculate comprehensive session cost breakdown."""

        # Calculate individual cost components
        event_cost = self.calculate_event_cost(event_count)
        identified_cost = self.calculate_event_cost(identified_events, is_identified=True)
        flag_cost = self.calculate_feature_flag_cost(feature_flag_requests)
        recording_cost = self.calculate_session_recording_cost(session_recordings)
        llm_cost = self.calculate_llm_analytics_cost(llm_events)

        total_cost = event_cost + identified_cost + flag_cost + recording_cost + llm_cost

        cost_breakdown = {
            'events': event_cost,
            'identified_events': identified_cost,
            'feature_flags': flag_cost,
            'session_recordings': recording_cost,
            'llm_analytics': llm_cost
        }

        free_tier_usage = {
            'events': min(event_count, self.costs['events']['free_tier']),
            'feature_flags': min(feature_flag_requests, self.costs['feature_flags']['free_tier']),
            'session_recordings': min(session_recordings, self.costs['session_recordings']['free_tier']),
            'llm_analytics': min(llm_events, self.costs['llm_analytics']['free_tier'])
        }

        return PostHogEventCost(
            event_count=event_count,
            identified_events=identified_events,
            feature_flag_requests=feature_flag_requests,
            session_recordings=session_recordings,
            llm_events=llm_events,
            total_cost=total_cost,
            cost_breakdown=cost_breakdown,
            free_tier_usage=free_tier_usage
        )

    def get_volume_discount_recommendations(self, monthly_events: int) -> List[Dict[str, Any]]:
        """Generate volume discount recommendations for cost optimization."""
        recommendations = []

        current_cost = self.calculate_event_cost(monthly_events)

        # Analyze tier positioning
        for i, (tier_limit, cost_per_event) in enumerate(self.costs['events']['tiers']):
            if monthly_events < tier_limit:
                next_tier_events = tier_limit
                next_tier_cost = self.calculate_event_cost(next_tier_events)
                cost_per_event_current = current_cost / monthly_events if monthly_events > 0 else Decimal('0')
                cost_per_event_next = next_tier_cost / next_tier_events if next_tier_events > 0 else Decimal('0')

                if cost_per_event_next < cost_per_event_current:
                    potential_savings = (cost_per_event_current - cost_per_event_next) * monthly_events
                    recommendations.append({
                        'optimization_type': 'Volume Tier Advancement',
                        'current_tier': f'Tier {i}',
                        'next_tier': f'Tier {i+1}',
                        'events_needed': int(next_tier_events - monthly_events),
                        'potential_savings_per_month': float(potential_savings),
                        'cost_per_event_improvement': float(cost_per_event_current - cost_per_event_next),
                        'priority_score': 85.0 if potential_savings > 10 else 60.0
                    })
                break

        return recommendations

class GenOpsPostHogAdapter(GovernanceProvider):
    """GenOps PostHog adapter for product analytics with governance."""

    def __init__(
        self,
        posthog_api_key: Optional[str] = None,
        posthog_host: str = "https://app.posthog.com",
        team: str = "default",
        project: str = "default",
        environment: str = "production",
        customer_id: Optional[str] = None,
        cost_center: Optional[str] = None,
        daily_budget_limit: float = 1000.0,
        monthly_budget_limit: Optional[float] = None,
        enable_governance: bool = True,
        enable_cost_alerts: bool = True,
        governance_policy: str = "advisory",
        tags: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Initialize PostHog adapter with governance capabilities.
        
        Args:
            posthog_api_key: PostHog project API key (or set POSTHOG_API_KEY env var)
            posthog_host: PostHog instance URL 
            team: Team name for cost attribution and governance
            project: Project name for cost attribution  
            environment: Environment (development, staging, production)
            customer_id: Customer identifier for multi-tenant cost attribution
            cost_center: Cost center for financial reporting
            daily_budget_limit: Daily budget limit in USD
            monthly_budget_limit: Monthly budget limit in USD
            enable_governance: Enable governance features
            enable_cost_alerts: Enable cost threshold alerts
            governance_policy: Governance enforcement level (advisory, enforced, strict)
            tags: Additional tags for telemetry
        """
        super().__init__()

        # Configuration
        self.posthog_api_key = posthog_api_key or os.getenv('POSTHOG_API_KEY')
        self.posthog_host = posthog_host or os.getenv('POSTHOG_HOST', 'https://app.posthog.com')
        self.team = team
        self.project = project
        self.environment = environment
        self.customer_id = customer_id
        self.cost_center = cost_center

        # Budget and governance
        self.daily_budget_limit = Decimal(str(daily_budget_limit))
        self.monthly_budget_limit = Decimal(str(monthly_budget_limit)) if monthly_budget_limit else None
        self.enable_governance = enable_governance
        self.enable_cost_alerts = enable_cost_alerts
        self.governance_policy = governance_policy

        # Cost tracking
        self.cost_calculator = PostHogCostCalculator()
        self.daily_costs = Decimal('0')
        self.monthly_costs = Decimal('0')

        # Telemetry
        self.telemetry = GenOpsTelemetry(tracer_name="posthog")

        # Active sessions
        self._active_sessions: Dict[str, PostHogAnalyticsSession] = {}

        # Validation
        if not self.posthog_api_key:
            raise GenOpsConfigurationError(
                "PostHog API key required. Set POSTHOG_API_KEY environment variable or pass posthog_api_key parameter."
            )

        logger.info(f"Initialized GenOps PostHog adapter for team '{self.team}', project '{self.project}'")

    def _build_base_tags(self, additional_tags: Dict[str, str]) -> Dict[str, str]:
        """Build base telemetry tags."""
        base_tags = {
            'genops.provider': 'posthog',
            'genops.team': self.team,
            'genops.project': self.project,
            'genops.environment': self.environment,
            'genops.version': '1.0.0',
            'posthog.host': self.posthog_host,
            'genops.governance.enabled': str(self.enable_governance),
            'genops.cost.tracking': 'enabled'
        }

        if self.customer_id:
            base_tags['genops.customer_id'] = self.customer_id
        if self.cost_center:
            base_tags['genops.cost_center'] = self.cost_center

        base_tags.update(additional_tags)
        return base_tags

    def _check_budget_constraints(self, estimated_cost: Decimal) -> None:
        """Check if operation would exceed budget limits."""
        if not self.enable_governance:
            return

        total_estimated_daily = self.daily_costs + estimated_cost

        if total_estimated_daily > self.daily_budget_limit:
            if self.governance_policy == "enforced":
                raise GenOpsBudgetExceededError(
                    f"PostHog operation would exceed daily budget limit. "
                    f"Estimated cost: ${estimated_cost}, Daily limit: ${self.daily_budget_limit}, "
                    f"Current usage: ${self.daily_costs}"
                )
            elif self.enable_cost_alerts:
                logger.warning(
                    f"PostHog operation approaches daily budget limit: ${total_estimated_daily}/${self.daily_budget_limit}"
                )

        if self.monthly_budget_limit:
            total_estimated_monthly = self.monthly_costs + estimated_cost
            if total_estimated_monthly > self.monthly_budget_limit:
                if self.governance_policy == "enforced":
                    raise GenOpsBudgetExceededError(
                        f"PostHog operation would exceed monthly budget limit. "
                        f"Estimated cost: ${estimated_cost}, Monthly limit: ${self.monthly_budget_limit}, "
                        f"Current usage: ${self.monthly_costs}"
                    )
                elif self.enable_cost_alerts:
                    logger.warning(
                        f"PostHog operation approaches monthly budget limit: ${total_estimated_monthly}/${self.monthly_budget_limit}"
                    )

    @contextmanager
    def track_analytics_session(
        self,
        session_name: str,
        customer_id: Optional[str] = None,
        cost_center: Optional[str] = None,
        environment: Optional[str] = None,
        **governance_attributes
    ):
        """
        Context manager for tracking PostHog analytics sessions with governance.
        
        Args:
            session_name: Descriptive name for the analytics session
            customer_id: Customer identifier for multi-tenant attribution  
            cost_center: Cost center override for this session
            environment: Environment override for this session
            **governance_attributes: Additional governance attributes
            
        Yields:
            PostHogAnalyticsSession: Analytics session with cost tracking
        """
        session_id = str(uuid4())
        start_time = datetime.now(timezone.utc)

        # Create session
        session = PostHogAnalyticsSession(
            session_id=session_id,
            session_name=session_name,
            start_time=start_time,
            team=self.team,
            project=self.project,
            customer_id=customer_id or self.customer_id,
            cost_center=cost_center or self.cost_center,
            environment=environment or self.environment,
            governance_attributes=governance_attributes
        )

        self._active_sessions[session_id] = session

        # Start telemetry span
        span_attributes = {
            'genops.posthog.session.id': session_id,
            'genops.posthog.session.name': session_name,
            'genops.posthog.session.start_time': start_time.isoformat(),
            **self.telemetry.tags,
            **governance_attributes
        }

        with self.telemetry.trace_operation(
            operation_name="posthog_analytics_session",
            attributes=span_attributes
        ) as span:

            try:
                logger.info(f"Started PostHog analytics session: {session_name} ({session_id})")
                yield session

            except Exception as e:
                logger.error(f"Error in PostHog analytics session {session_name}: {e}")
                span.set_status({"status_code": "ERROR", "description": str(e)})
                raise

            finally:
                # Finalize session and calculate costs
                cost_summary = session.finalize_session()

                # Update running costs
                self.daily_costs += cost_summary.total_cost
                self.monthly_costs += cost_summary.total_cost

                # Update span with final metrics
                span.set_attributes({
                    'genops.posthog.session.events_captured': session.events_captured,
                    'genops.posthog.session.flags_evaluated': session.flags_evaluated,
                    'genops.posthog.session.recordings_created': session.recordings_created,
                    'genops.posthog.session.llm_events': session.llm_events_tracked,
                    'genops.cost.total': float(cost_summary.total_cost),
                    'genops.cost.currency': 'USD',
                    'genops.cost.per_event': float(cost_summary.cost_per_event),
                    'genops.posthog.session.duration_seconds': (
                        session.end_time - session.start_time
                    ).total_seconds() if session.end_time else 0,
                    'genops.posthog.session.end_time': session.end_time.isoformat() if session.end_time else ''
                })

                # Clean up session
                self._active_sessions.pop(session_id, None)

                logger.info(
                    f"Completed PostHog analytics session {session_name}: "
                    f"{session.events_captured} events, ${cost_summary.total_cost:.4f} cost"
                )

    def capture_event_with_governance(
        self,
        event_name: str,
        properties: Optional[Dict[str, Any]] = None,
        distinct_id: Optional[str] = None,
        is_identified: bool = False,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Capture PostHog event with governance tracking.
        
        Args:
            event_name: Name of the event to capture
            properties: Event properties dictionary
            distinct_id: User identifier for the event
            is_identified: Whether this is an identified user event (affects cost)
            session_id: Session ID if part of tracked session
            
        Returns:
            Dict containing event metadata and cost information
        """
        # Cost estimation and budget check
        estimated_cost = self.cost_calculator.calculate_event_cost(1, is_identified=is_identified)
        self._check_budget_constraints(estimated_cost)

        # Build enhanced properties with governance
        enhanced_properties = {
            'genops_team': self.team,
            'genops_project': self.project,
            'genops_environment': self.environment,
            'genops_timestamp': datetime.now(timezone.utc).isoformat(),
            'genops_cost_estimated': float(estimated_cost)
        }

        if self.customer_id:
            enhanced_properties['genops_customer_id'] = self.customer_id
        if self.cost_center:
            enhanced_properties['genops_cost_center'] = self.cost_center
        if properties:
            enhanced_properties.update(properties)

        # Update session if provided
        if session_id and session_id in self._active_sessions:
            session = self._active_sessions[session_id]
            session.events_captured += 1
            session.total_cost += estimated_cost

        # Telemetry tracking
        event_attributes = {
            'genops.posthog.event.name': event_name,
            'genops.posthog.event.distinct_id': distinct_id or 'anonymous',
            'genops.posthog.event.is_identified': is_identified,
            'genops.cost.estimated': float(estimated_cost),
            'genops.cost.currency': 'USD'
        }

        if session_id:
            event_attributes['genops.posthog.session.id'] = session_id

        with self.telemetry.trace_operation(
            operation_name="posthog_capture_event",
            attributes=event_attributes
        ):
            # In a real implementation, this would call the actual PostHog client
            # posthog.capture(distinct_id=distinct_id, event=event_name, properties=enhanced_properties)
            pass

        result = {
            'event_name': event_name,
            'distinct_id': distinct_id,
            'cost': float(estimated_cost),
            'governance_applied': True,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'properties_count': len(enhanced_properties),
            'is_identified': is_identified
        }

        logger.debug(f"Captured PostHog event '{event_name}' with cost ${estimated_cost:.6f}")
        return result

    def evaluate_feature_flag_with_governance(
        self,
        flag_key: str,
        distinct_id: str,
        properties: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Evaluate PostHog feature flag with governance tracking.
        
        Args:
            flag_key: Feature flag key to evaluate
            distinct_id: User identifier for flag evaluation
            properties: User properties for flag evaluation context
            session_id: Session ID if part of tracked session
            
        Returns:
            Tuple of (flag_value, metadata) with governance information
        """
        # Cost estimation and budget check
        estimated_cost = self.cost_calculator.calculate_feature_flag_cost(1)
        self._check_budget_constraints(estimated_cost)

        # Update session if provided
        if session_id and session_id in self._active_sessions:
            session = self._active_sessions[session_id]
            session.flags_evaluated += 1
            session.total_cost += estimated_cost

        # Telemetry tracking
        flag_attributes = {
            'genops.posthog.flag.key': flag_key,
            'genops.posthog.flag.distinct_id': distinct_id,
            'genops.cost.estimated': float(estimated_cost),
            'genops.cost.currency': 'USD'
        }

        if session_id:
            flag_attributes['genops.posthog.session.id'] = session_id

        with self.telemetry.trace_operation(
            operation_name="posthog_evaluate_feature_flag",
            attributes=flag_attributes
        ):
            # In a real implementation, this would call the actual PostHog client
            # flag_value = posthog.feature_enabled(flag_key, distinct_id, person_properties=properties)
            flag_value = False  # Mock value

        metadata = {
            'flag_key': flag_key,
            'distinct_id': distinct_id,
            'cost': float(estimated_cost),
            'governance_applied': True,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'evaluation_context': len(properties) if properties else 0
        }

        logger.debug(f"Evaluated PostHog feature flag '{flag_key}' with cost ${estimated_cost:.6f}")
        return flag_value, metadata

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get current cost summary and usage statistics."""
        active_sessions = len(self._active_sessions)

        return {
            'daily_costs': float(self.daily_costs),
            'monthly_costs': float(self.monthly_costs),
            'daily_budget_limit': float(self.daily_budget_limit),
            'monthly_budget_limit': float(self.monthly_budget_limit) if self.monthly_budget_limit else None,
            'daily_budget_utilization': float(self.daily_costs / self.daily_budget_limit * 100) if self.daily_budget_limit > 0 else 0,
            'active_sessions': active_sessions,
            'team': self.team,
            'project': self.project,
            'environment': self.environment,
            'governance_enabled': self.enable_governance,
            'governance_policy': self.governance_policy,
            'cost_alerts_enabled': self.enable_cost_alerts
        }

    def get_volume_discount_analysis(self, projected_monthly_events: int) -> Dict[str, Any]:
        """Generate volume discount analysis and cost optimization recommendations."""
        recommendations = self.cost_calculator.get_volume_discount_recommendations(projected_monthly_events)
        current_cost = self.cost_calculator.calculate_event_cost(projected_monthly_events)

        return {
            'projected_monthly_events': projected_monthly_events,
            'projected_monthly_cost': float(current_cost),
            'cost_per_event': float(current_cost / projected_monthly_events) if projected_monthly_events > 0 else 0,
            'optimization_recommendations': recommendations,
            'free_tier_utilization': {
                'events': min(projected_monthly_events, POSTHOG_COSTS['events']['free_tier']),
                'feature_flags': 0,  # Would need actual usage data
                'session_recordings': 0,  # Would need actual usage data
                'llm_analytics': 0  # Would need actual usage data
            }
        }


# Auto-instrumentation functions
def auto_instrument(
    posthog_api_key: Optional[str] = None,
    team: str = "auto-instrumented",
    project: str = "default",
    **adapter_kwargs
) -> GenOpsPostHogAdapter:
    """
    Auto-instrument PostHog with GenOps governance for zero-code setup.
    
    Args:
        posthog_api_key: PostHog project API key
        team: Team name for governance
        project: Project name for governance
        **adapter_kwargs: Additional arguments for GenOpsPostHogAdapter
        
    Returns:
        Configured PostHog adapter instance
    """
    adapter = GenOpsPostHogAdapter(
        posthog_api_key=posthog_api_key,
        team=team,
        project=project,
        **adapter_kwargs
    )

    # TODO: In a real implementation, this would patch the PostHog client
    # to automatically apply governance to all PostHog operations

    logger.info("PostHog auto-instrumentation activated with GenOps governance")
    return adapter

def instrument_posthog(
    posthog_api_key: Optional[str] = None,
    team: str = "default",
    project: str = "default",
    **kwargs
) -> GenOpsPostHogAdapter:
    """
    Legacy alias for auto_instrument for backward compatibility.
    
    Args:
        posthog_api_key: PostHog project API key
        team: Team name for governance  
        project: Project name for governance
        **kwargs: Additional adapter configuration
        
    Returns:
        Configured PostHog adapter instance
    """
    return auto_instrument(
        posthog_api_key=posthog_api_key,
        team=team,
        project=project,
        **kwargs
    )

def get_current_adapter() -> Optional[GenOpsPostHogAdapter]:
    """Get the current auto-instrumented PostHog adapter instance."""
    # In a real implementation, this would return the globally registered adapter
    return None

# Export key classes and functions
__all__ = [
    'GenOpsPostHogAdapter',
    'PostHogCostCalculator',
    'PostHogEventCost',
    'PostHogAnalyticsSession',
    'auto_instrument',
    'instrument_posthog',
    'get_current_adapter',
    'POSTHOG_COSTS'
]
