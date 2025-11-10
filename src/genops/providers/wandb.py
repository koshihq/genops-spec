#!/usr/bin/env python3
"""
GenOps Weights & Biases Integration

This module provides comprehensive Weights & Biases (W&B) integration for GenOps AI governance,
cost intelligence, and policy enforcement. W&B is a powerful machine learning experiment 
tracking, model versioning, and MLOps platform that provides comprehensive experiment 
management and artifact tracking for AI workflows.

Features:
- Enhanced W&B experiment tracking with GenOps governance attributes
- Cost attribution and budget enforcement for ML experiments and model training
- Policy compliance tracking integrated with W&B runs and artifacts
- Experiment lifecycle management with governance oversight and cost optimization
- Multi-run campaign tracking with unified cost intelligence
- Zero-code auto-instrumentation with instrument_wandb()
- Enterprise-ready governance patterns for production ML workflows

Example usage:

    # Zero-code auto-instrumentation
    from genops.providers.wandb import auto_instrument
    auto_instrument(
        wandb_api_key="your-wandb-api-key",
        team="ml-team",
        project="experiment-optimization"
    )
    
    # Your existing W&B code now includes GenOps governance
    import wandb
    
    wandb.init(project="my-project", name="experiment-1")
    wandb.log({"accuracy": 0.95, "loss": 0.05})
    wandb.finish()
    # Automatically tracked with cost attribution and governance
    
    # Manual adapter usage for advanced governance
    from genops.providers.wandb import GenOpsWandbAdapter
    
    adapter = GenOpsWandbAdapter(
        wandb_api_key="your-wandb-api-key",
        team="ml-engineering-team",
        project="model-training",
        enable_cost_alerts=True,
        daily_budget_limit=100.0
    )
    
    # Enhanced experiment operations with governance
    with adapter.track_experiment_lifecycle("model-training-v2") as experiment:
        run = wandb.init(project="my-project", name="experiment-1")
        
        # Training loop with cost tracking
        for epoch in range(10):
            metrics = train_epoch()
            wandb.log(metrics)
            experiment.update_compute_cost(calculate_epoch_cost())
        
        # Model artifact tracking
        model_artifact = wandb.Artifact("trained-model", type="model")
        model_artifact.add_file("model.pkl")
        experiment.log_governed_artifact(model_artifact)
        
        run.finish()

Dependencies:
    - wandb: Weights & Biases Python SDK (pip install wandb)
    - opentelemetry-api: For telemetry export
    - Optional: wandb[sweeps] for hyperparameter optimization support

Environment Variables:
    - WANDB_API_KEY: Your Weights & Biases API key
    - GENOPS_TEAM: Team attribution (recommended)
    - GENOPS_PROJECT: Project attribution (recommended)
    - GENOPS_DAILY_BUDGET_LIMIT: Daily spending limit in USD
"""

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# Optional wandb import with graceful degradation
try:
    import wandb
    from wandb.sdk.wandb_run import Run as WandbRun
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None
    WandbRun = None

logger = logging.getLogger(__name__)


class GovernancePolicy(Enum):
    """Governance policy enforcement levels."""
    AUDIT_ONLY = "audit_only"
    ADVISORY = "advisory"
    ENFORCED = "enforced"


@dataclass
class ExperimentCostSummary:
    """Cost summary for W&B experiment runs."""
    total_cost: float
    compute_cost: float
    storage_cost: float
    data_transfer_cost: float
    cost_by_run: Dict[str, float]
    experiment_duration: float
    resource_efficiency: float


@dataclass
class WandbRunContext:
    """Context for tracking W&B run governance."""
    run_id: str
    run_name: str
    project: str
    team: str
    customer_id: Optional[str]
    start_time: datetime
    estimated_cost: float = 0.0
    compute_hours: float = 0.0
    storage_gb: float = 0.0
    policy_violations: List[str] = None

    def __post_init__(self):
        if self.policy_violations is None:
            self.policy_violations = []


class GenOpsWandbAdapter:
    """
    GenOps governance adapter for Weights & Biases experiment tracking.
    
    Provides comprehensive cost intelligence, policy enforcement, and team attribution
    for W&B experiments, runs, and artifacts with enterprise-grade governance features.
    """

    def __init__(
        self,
        wandb_api_key: Optional[str] = None,
        team: Optional[str] = None,
        project: Optional[str] = None,
        customer_id: Optional[str] = None,
        environment: str = "development",
        daily_budget_limit: float = 100.0,
        max_experiment_cost: float = 50.0,
        governance_policy: Union[GovernancePolicy, str] = GovernancePolicy.ADVISORY,
        enable_cost_alerts: bool = True,
        enable_governance: bool = True,
        cost_center: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the GenOps W&B adapter.
        
        Args:
            wandb_api_key: W&B API key (or set WANDB_API_KEY env var)
            team: Team name for cost attribution 
            project: Project name for cost attribution
            customer_id: Customer identifier for multi-tenant scenarios
            environment: Environment (development/staging/production)
            daily_budget_limit: Maximum daily spending limit in USD
            max_experiment_cost: Maximum cost per experiment in USD
            governance_policy: Policy enforcement level
            enable_cost_alerts: Enable cost threshold alerts
            enable_governance: Enable governance features
            cost_center: Cost center for financial reporting
            tags: Additional tags for telemetry
        """
        if not WANDB_AVAILABLE:
            raise ImportError(
                "Weights & Biases (wandb) is required for this integration. "
                "Install with: pip install wandb"
            )

        # Configuration
        self.wandb_api_key = wandb_api_key or os.getenv('WANDB_API_KEY')
        self.team = team or os.getenv('GENOPS_TEAM', 'default-team')
        self.project = project or os.getenv('GENOPS_PROJECT', 'default-project')
        self.customer_id = customer_id or os.getenv('GENOPS_CUSTOMER_ID')
        self.environment = environment
        self.cost_center = cost_center

        # Budget and policy settings
        self.daily_budget_limit = daily_budget_limit
        self.max_experiment_cost = max_experiment_cost
        if isinstance(governance_policy, str):
            governance_policy = GovernancePolicy(governance_policy)
        self.governance_policy = governance_policy
        self.enable_cost_alerts = enable_cost_alerts
        self.enable_governance = enable_governance

        # Tags
        self.tags = tags or {}

        # Runtime tracking
        self.active_runs: Dict[str, WandbRunContext] = {}
        self.daily_usage = 0.0
        self.operation_count = 0

        # Initialize tracer
        self.tracer = trace.get_tracer(__name__)

        # Initialize W&B if API key provided
        if self.wandb_api_key:
            os.environ['WANDB_API_KEY'] = self.wandb_api_key

        logger.info(f"GenOps W&B adapter initialized for team={self.team}, project={self.project}")

    @contextmanager
    def track_experiment_lifecycle(
        self,
        experiment_name: str,
        experiment_type: str = "training",
        max_cost: Optional[float] = None,
        **kwargs
    ):
        """
        Context manager for tracking complete experiment lifecycle with governance.
        
        Args:
            experiment_name: Name of the experiment
            experiment_type: Type of experiment (training, evaluation, inference, etc.)
            max_cost: Maximum cost limit for this experiment
            **kwargs: Additional attributes for telemetry
            
        Yields:
            WandbRunContext: Experiment context for cost tracking and governance
        """
        experiment_id = f"{experiment_name}_{int(time.time())}"
        max_cost = max_cost or self.max_experiment_cost

        # Create experiment context
        experiment_context = WandbRunContext(
            run_id=experiment_id,
            run_name=experiment_name,
            project=self.project,
            team=self.team,
            customer_id=self.customer_id,
            start_time=datetime.utcnow()
        )

        # Start OpenTelemetry span
        with self.tracer.start_as_current_span(
            f"wandb.experiment.{experiment_type}",
            attributes={
                "genops.provider": "wandb",
                "genops.team": self.team,
                "genops.project": self.project,
                "genops.customer_id": self.customer_id,
                "genops.environment": self.environment,
                "genops.experiment.name": experiment_name,
                "genops.experiment.type": experiment_type,
                "genops.experiment.id": experiment_id,
                "genops.cost.budget_limit": max_cost,
                **kwargs
            }
        ) as span:

            try:
                # Register active experiment
                self.active_runs[experiment_id] = experiment_context

                # Pre-experiment governance checks
                if self.enable_governance:
                    self._validate_experiment_budget(max_cost)

                logger.info(f"Starting experiment lifecycle tracking: {experiment_name}")

                yield experiment_context

                # Calculate final costs and metrics
                total_cost = experiment_context.estimated_cost
                duration = (datetime.utcnow() - experiment_context.start_time).total_seconds()

                # Update span with final metrics
                span.set_attributes({
                    "genops.cost.total": total_cost,
                    "genops.cost.currency": "USD",
                    "genops.experiment.duration_seconds": duration,
                    "genops.experiment.compute_hours": experiment_context.compute_hours,
                    "genops.experiment.storage_gb": experiment_context.storage_gb,
                    "genops.governance.violations": len(experiment_context.policy_violations)
                })

                # Update daily usage
                self.daily_usage += total_cost
                self.operation_count += 1

                # Log governance violations
                if experiment_context.policy_violations:
                    span.add_event("governance_violations", {
                        "violations": experiment_context.policy_violations,
                        "policy": self.governance_policy.value
                    })

                # Cost alerts
                if self.enable_cost_alerts and total_cost > max_cost * 0.8:
                    logger.warning(
                        f"Experiment {experiment_name} approaching cost limit: "
                        f"${total_cost:.4f} / ${max_cost:.2f}"
                    )

                span.set_status(Status(StatusCode.OK))
                logger.info(f"Experiment completed: {experiment_name}, cost: ${total_cost:.4f}")

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                logger.error(f"Error in experiment {experiment_name}: {e}")
                raise
            finally:
                # Cleanup
                self.active_runs.pop(experiment_id, None)

    def instrument_wandb_init(self, original_init):
        """
        Instrument wandb.init() with governance tracking.
        
        Args:
            original_init: Original wandb.init function
            
        Returns:
            Enhanced wandb.init function with governance
        """
        def enhanced_init(*args, **kwargs):
            # Extract run configuration
            project = kwargs.get('project', 'default-project')
            name = kwargs.get('name', f'run-{int(time.time())}')

            # Add governance tags
            tags = kwargs.get('tags', [])
            tags.extend([
                f"genops-team:{self.team}",
                f"genops-project:{self.project}",
                f"genops-env:{self.environment}"
            ])
            kwargs['tags'] = tags

            # Add governance config
            config = kwargs.get('config', {})
            config.update({
                'genops_team': self.team,
                'genops_project': self.project,
                'genops_customer_id': self.customer_id,
                'genops_environment': self.environment,
                'genops_governance_enabled': self.enable_governance
            })
            kwargs['config'] = config

            # Start OpenTelemetry tracking
            span = self.tracer.start_span(
                "wandb.init",
                attributes={
                    "genops.provider": "wandb",
                    "genops.team": self.team,
                    "genops.project": self.project,
                    "genops.wandb.project": project,
                    "genops.wandb.run_name": name,
                    "genops.environment": self.environment
                }
            )

            try:
                # Initialize W&B run
                run = original_init(*args, **kwargs)

                # Create run context
                if run:
                    run_context = WandbRunContext(
                        run_id=run.id,
                        run_name=run.name,
                        project=project,
                        team=self.team,
                        customer_id=self.customer_id,
                        start_time=datetime.utcnow()
                    )
                    self.active_runs[run.id] = run_context

                    # Enhance run with governance methods
                    run.genops_update_cost = lambda cost: self._update_run_cost(run.id, cost)
                    run.genops_log_violation = lambda violation: self._log_policy_violation(run.id, violation)
                    run.genops_get_context = lambda: self.active_runs.get(run.id)

                span.set_status(Status(StatusCode.OK))
                return run

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            finally:
                span.end()

        return enhanced_init

    def instrument_wandb_log(self, original_log):
        """
        Instrument wandb.log() with cost tracking.
        
        Args:
            original_log: Original wandb.log function
            
        Returns:
            Enhanced wandb.log function with cost tracking
        """
        def enhanced_log(*args, **kwargs):
            # Extract current run
            current_run = wandb.run
            if not current_run:
                return original_log(*args, **kwargs)

            # Track logging operation
            with self.tracer.start_as_current_span(
                "wandb.log",
                attributes={
                    "genops.provider": "wandb",
                    "genops.team": self.team,
                    "genops.wandb.run_id": current_run.id,
                    "genops.operation": "log_metrics"
                }
            ) as span:

                try:
                    # Call original log function
                    result = original_log(*args, **kwargs)

                    # Estimate cost for logging operation
                    log_data = args[0] if args else kwargs.get('data', {})
                    estimated_cost = self._estimate_log_cost(log_data)

                    # Update run cost
                    self._update_run_cost(current_run.id, estimated_cost)

                    span.set_attributes({
                        "genops.cost.estimated": estimated_cost,
                        "genops.metrics.count": len(log_data) if isinstance(log_data, dict) else 1
                    })

                    return result

                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        return enhanced_log

    def log_governed_artifact(
        self,
        artifact: Any,
        cost_estimate: Optional[float] = None,
        governance_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log W&B artifact with governance metadata and cost tracking.
        
        Args:
            artifact: W&B Artifact object
            cost_estimate: Estimated cost for storing artifact
            governance_metadata: Additional governance metadata
        """
        if not hasattr(artifact, 'metadata'):
            logger.error("Invalid artifact object provided")
            return

        # Add governance metadata
        governance_data = {
            'genops_team': self.team,
            'genops_project': self.project,
            'genops_customer_id': self.customer_id,
            'genops_environment': self.environment,
            'genops_logged_at': datetime.utcnow().isoformat(),
            'genops_cost_estimate': cost_estimate or 0.0
        }

        if governance_metadata:
            governance_data.update(governance_metadata)

        # Update artifact metadata
        artifact.metadata.update(governance_data)

        # Track in OpenTelemetry
        with self.tracer.start_as_current_span(
            "wandb.artifact.log",
            attributes={
                "genops.provider": "wandb",
                "genops.team": self.team,
                "genops.artifact.name": artifact.name,
                "genops.artifact.type": artifact.type,
                "genops.cost.estimated": cost_estimate or 0.0
            }
        ) as span:

            try:
                # Log artifact
                current_run = wandb.run
                if current_run:
                    current_run.log_artifact(artifact)

                    # Update run cost
                    if cost_estimate:
                        self._update_run_cost(current_run.id, cost_estimate)

                span.set_status(Status(StatusCode.OK))
                logger.info(f"Logged governed artifact: {artifact.name}")

            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise

    def get_experiment_cost_summary(self, experiment_id: str) -> Optional[ExperimentCostSummary]:
        """Get comprehensive cost summary for an experiment."""
        run_context = self.active_runs.get(experiment_id)
        if not run_context:
            return None

        duration = (datetime.utcnow() - run_context.start_time).total_seconds()

        return ExperimentCostSummary(
            total_cost=run_context.estimated_cost,
            compute_cost=run_context.compute_hours * 0.50,  # Estimated GPU cost
            storage_cost=run_context.storage_gb * 0.02,     # Estimated storage cost
            data_transfer_cost=0.0,  # To be calculated based on usage
            cost_by_run={experiment_id: run_context.estimated_cost},
            experiment_duration=duration,
            resource_efficiency=run_context.estimated_cost / max(duration / 3600, 0.01)  # Cost per hour
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get current governance metrics and status."""
        return {
            'team': self.team,
            'project': self.project,
            'customer_id': self.customer_id,
            'daily_usage': self.daily_usage,
            'daily_budget_limit': self.daily_budget_limit,
            'budget_remaining': max(0, self.daily_budget_limit - self.daily_usage),
            'operation_count': self.operation_count,
            'active_experiments': len(self.active_runs),
            'governance_policy': self.governance_policy.value,
            'cost_alerts_enabled': self.enable_cost_alerts
        }

    def _validate_experiment_budget(self, experiment_cost: float) -> None:
        """Validate experiment against budget limits."""
        if self.daily_usage + experiment_cost > self.daily_budget_limit:
            violation = f"Experiment would exceed daily budget: ${self.daily_usage + experiment_cost:.2f} > ${self.daily_budget_limit:.2f}"

            if self.governance_policy == GovernancePolicy.ENFORCED:
                raise ValueError(violation)
            else:
                logger.warning(f"Budget violation (advisory): {violation}")

    def _update_run_cost(self, run_id: str, cost: float) -> None:
        """Update cost for a specific run."""
        if run_id in self.active_runs:
            self.active_runs[run_id].estimated_cost += cost

    def _log_policy_violation(self, run_id: str, violation: str) -> None:
        """Log a policy violation for a specific run."""
        if run_id in self.active_runs:
            self.active_runs[run_id].policy_violations.append(violation)

    def _estimate_log_cost(self, log_data: Any) -> float:
        """Estimate cost for logging operation based on data size."""
        if isinstance(log_data, dict):
            # Rough estimate: $0.001 per metric logged
            return len(log_data) * 0.001
        return 0.001  # Default small cost


def instrument_wandb(
    wandb_api_key: Optional[str] = None,
    team: Optional[str] = None,
    project: Optional[str] = None,
    **kwargs
) -> GenOpsWandbAdapter:
    """
    Create and configure a GenOps W&B adapter for experiment governance.
    
    Args:
        wandb_api_key: W&B API key
        team: Team name for cost attribution
        project: Project name for cost attribution
        **kwargs: Additional configuration options
        
    Returns:
        Configured GenOpsWandbAdapter instance
    """
    return GenOpsWandbAdapter(
        wandb_api_key=wandb_api_key,
        team=team,
        project=project,
        **kwargs
    )


def auto_instrument(
    wandb_api_key: Optional[str] = None,
    team: Optional[str] = None,
    project: Optional[str] = None,
    **kwargs
) -> GenOpsWandbAdapter:
    """
    Enable zero-code auto-instrumentation for W&B with GenOps governance.
    
    This function patches wandb.init() and wandb.log() to automatically include
    governance tracking without requiring code changes to existing W&B usage.
    
    Args:
        wandb_api_key: W&B API key
        team: Team name for cost attribution
        project: Project name for cost attribution
        **kwargs: Additional configuration options
        
    Returns:
        Configured GenOpsWandbAdapter instance
    """
    if not WANDB_AVAILABLE:
        raise ImportError(
            "Weights & Biases (wandb) is required for auto-instrumentation. "
            "Install with: pip install wandb"
        )

    # Create adapter
    adapter = GenOpsWandbAdapter(
        wandb_api_key=wandb_api_key,
        team=team,
        project=project,
        **kwargs
    )

    # Patch wandb functions
    if hasattr(wandb, 'init'):
        original_init = wandb.init
        wandb.init = adapter.instrument_wandb_init(original_init)

    if hasattr(wandb, 'log'):
        original_log = wandb.log
        wandb.log = adapter.instrument_wandb_log(original_log)

    logger.info("W&B auto-instrumentation enabled with GenOps governance")

    return adapter


# Global adapter instance for convenience
_global_adapter: Optional[GenOpsWandbAdapter] = None


def get_current_adapter() -> Optional[GenOpsWandbAdapter]:
    """Get the current global GenOps W&B adapter instance."""
    return _global_adapter


def set_global_adapter(adapter: GenOpsWandbAdapter) -> None:
    """Set the global GenOps W&B adapter instance."""
    global _global_adapter
    _global_adapter = adapter


# Convenience exports
__all__ = [
    'GenOpsWandbAdapter',
    'WandbRunContext',
    'ExperimentCostSummary',
    'GovernancePolicy',
    'instrument_wandb',
    'auto_instrument',
    'get_current_adapter',
    'set_global_adapter',
    'WANDB_AVAILABLE'
]
