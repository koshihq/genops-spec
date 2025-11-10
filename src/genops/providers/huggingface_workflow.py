"""Production workflow context manager for Hugging Face operations."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple

from genops.core.telemetry import GenOpsTelemetry
from genops.providers.huggingface_cost_aggregator import create_huggingface_cost_context

logger = logging.getLogger(__name__)


@contextmanager
def production_workflow_context(
    workflow_name: str,
    customer_id: str,
    **kwargs
) -> Tuple[Any, str]:
    """
    Enterprise workflow template for complex Hugging Face operations.
    
    This follows the exact pattern specified in CLAUDE.md:
    
    with production_workflow_context(workflow_name, customer_id, **kwargs) as (span, workflow_id):
        # Multi-step operations with unified governance
        # Automatic cost attribution and error handling
        # Performance monitoring and alerting integration
    
    Args:
        workflow_name: Name of the workflow being executed
        customer_id: Customer identifier for billing attribution
        **kwargs: Additional governance attributes (team, project, environment, etc.)
    
    Yields:
        Tuple of (span, workflow_id) for operation tracking
    """
    # Generate unique workflow ID
    workflow_id = f"{workflow_name}_{int(time.time())}_{str(uuid.uuid4())[:8]}"

    # Extract governance attributes
    governance_attrs = {
        'workflow_name': workflow_name,
        'workflow_id': workflow_id,
        'customer_id': customer_id,
        **kwargs
    }

    # Initialize telemetry
    telemetry = GenOpsTelemetry()

    # Start workflow tracking with cost aggregation
    with create_huggingface_cost_context(workflow_id) as cost_context:
        with telemetry.trace_operation(
            operation_name=f"huggingface.workflow.{workflow_name}",
            **governance_attrs
        ) as span:

            # Set workflow-specific attributes
            span.set_attribute("genops.workflow.name", workflow_name)
            span.set_attribute("genops.workflow.id", workflow_id)
            span.set_attribute("genops.workflow.customer_id", customer_id)
            span.set_attribute("genops.workflow.start_time", time.time())

            # Add additional governance attributes
            for key, value in kwargs.items():
                if isinstance(value, (str, int, float, bool)):
                    span.set_attribute(f"genops.governance.{key}", value)

            logger.info(f"Started production workflow: {workflow_name} (ID: {workflow_id})")

            try:
                # Create enhanced span context with cost tracking
                enhanced_span = ProductionWorkflowSpan(
                    span=span,
                    workflow_id=workflow_id,
                    workflow_name=workflow_name,
                    cost_context=cost_context,
                    governance_attrs=governance_attrs
                )

                yield enhanced_span, workflow_id

                # Record successful completion
                span.set_attribute("genops.workflow.status", "completed")
                span.set_attribute("genops.workflow.end_time", time.time())

                # Get final cost summary
                final_summary = cost_context.get_final_summary()
                if final_summary:
                    span.set_attribute("genops.workflow.total_cost", final_summary.total_cost)
                    span.set_attribute("genops.workflow.providers_used", len(final_summary.unique_providers))
                    span.set_attribute("genops.workflow.models_used", len(final_summary.unique_models))
                    span.set_attribute("genops.workflow.total_tokens_input", final_summary.total_tokens_input)
                    span.set_attribute("genops.workflow.total_tokens_output", final_summary.total_tokens_output)

                logger.info(f"Completed production workflow: {workflow_name} (ID: {workflow_id})")

            except Exception as e:
                # Record error details
                span.set_attribute("genops.workflow.status", "error")
                span.set_attribute("genops.workflow.error_message", str(e))
                span.set_attribute("genops.workflow.error_type", type(e).__name__)
                span.set_attribute("genops.workflow.end_time", time.time())

                logger.error(f"Workflow {workflow_name} failed: {e}", exc_info=True)

                # Re-raise the exception to maintain error flow
                raise


class ProductionWorkflowSpan:
    """
    Enhanced span context for production workflows.
    
    Provides additional methods for workflow-specific operations like
    checkpoint recording, progress tracking, and cost monitoring.
    """

    def __init__(
        self,
        span: Any,
        workflow_id: str,
        workflow_name: str,
        cost_context: Any,
        governance_attrs: Dict[str, Any]
    ):
        self.span = span
        self.workflow_id = workflow_id
        self.workflow_name = workflow_name
        self.cost_context = cost_context
        self.governance_attrs = governance_attrs
        self._step_counter = 0
        self._checkpoints = []

    def record_step(self, step_name: str, metadata: Dict[str, Any] = None) -> None:
        """Record a workflow step with optional metadata."""
        self._step_counter += 1
        step_id = f"{self.workflow_id}_step_{self._step_counter}"

        # Record step in span
        self.span.set_attribute(f"genops.workflow.step.{self._step_counter}.name", step_name)
        self.span.set_attribute(f"genops.workflow.step.{self._step_counter}.timestamp", time.time())

        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool)):
                    self.span.set_attribute(f"genops.workflow.step.{self._step_counter}.{key}", value)

        logger.debug(f"Workflow {self.workflow_name} step {self._step_counter}: {step_name}")

    def record_checkpoint(self, checkpoint_name: str, data: Dict[str, Any] = None) -> str:
        """Record a workflow checkpoint for recovery purposes."""
        checkpoint_id = f"{self.workflow_id}_checkpoint_{len(self._checkpoints) + 1}"

        checkpoint_data = {
            'id': checkpoint_id,
            'name': checkpoint_name,
            'timestamp': time.time(),
            'step_count': self._step_counter,
            'data': data or {}
        }

        self._checkpoints.append(checkpoint_data)

        # Record checkpoint in span
        self.span.set_attribute(
            f"genops.workflow.checkpoint.{len(self._checkpoints)}.name",
            checkpoint_name
        )
        self.span.set_attribute(
            f"genops.workflow.checkpoint.{len(self._checkpoints)}.id",
            checkpoint_id
        )

        logger.info(f"Workflow {self.workflow_name} checkpoint: {checkpoint_name} (ID: {checkpoint_id})")
        return checkpoint_id

    def record_hf_operation(
        self,
        operation_name: str,
        provider: str,
        model: str,
        tokens_input: int,
        tokens_output: int,
        task: str = None,
        **metadata
    ) -> None:
        """Record a Hugging Face operation within this workflow."""
        # Add operation to cost tracking
        call_cost = self.cost_context.add_hf_call(
            provider=provider,
            model=model,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            task=task,
            operation_name=operation_name,
            **metadata
        )

        # Record operation in span
        operation_count = getattr(self, '_operation_count', 0) + 1
        self._operation_count = operation_count

        self.span.set_attribute(f"genops.workflow.operation.{operation_count}.name", operation_name)
        self.span.set_attribute(f"genops.workflow.operation.{operation_count}.provider", provider)
        self.span.set_attribute(f"genops.workflow.operation.{operation_count}.model", model)
        self.span.set_attribute(f"genops.workflow.operation.{operation_count}.task", task or "text-generation")

        if call_cost:
            self.span.set_attribute(f"genops.workflow.operation.{operation_count}.cost", call_cost.cost)

        logger.debug(f"Workflow {self.workflow_name} operation {operation_count}: {operation_name} ({provider}/{model})")

    def get_current_cost_summary(self) -> Optional[Any]:
        """Get current cost summary for the workflow."""
        return self.cost_context.get_current_summary()

    def record_performance_metric(self, metric_name: str, value: float, unit: str = None) -> None:
        """Record a performance metric for the workflow."""
        self.span.set_attribute(f"genops.workflow.metrics.{metric_name}", value)
        if unit:
            self.span.set_attribute(f"genops.workflow.metrics.{metric_name}.unit", unit)

        logger.debug(f"Workflow {self.workflow_name} metric: {metric_name} = {value} {unit or ''}")

    def record_alert(self, alert_type: str, message: str, severity: str = "info") -> None:
        """Record an alert or notification for the workflow."""
        alert_count = getattr(self, '_alert_count', 0) + 1
        self._alert_count = alert_count

        self.span.set_attribute(f"genops.workflow.alert.{alert_count}.type", alert_type)
        self.span.set_attribute(f"genops.workflow.alert.{alert_count}.message", message)
        self.span.set_attribute(f"genops.workflow.alert.{alert_count}.severity", severity)
        self.span.set_attribute(f"genops.workflow.alert.{alert_count}.timestamp", time.time())

        if severity in ['warning', 'error', 'critical']:
            logger.warning(f"Workflow {self.workflow_name} {severity}: {alert_type} - {message}")
        else:
            logger.info(f"Workflow {self.workflow_name} alert: {alert_type} - {message}")

    def set_governance_attribute(self, key: str, value: Any) -> None:
        """Set additional governance attributes during workflow execution."""
        if isinstance(value, (str, int, float, bool)):
            self.span.set_attribute(f"genops.governance.{key}", value)
            self.governance_attrs[key] = value
            logger.debug(f"Workflow {self.workflow_name} governance: {key} = {value}")

    def get_workflow_metadata(self) -> Dict[str, Any]:
        """Get comprehensive workflow metadata."""
        current_summary = self.get_current_cost_summary()

        return {
            'workflow_id': self.workflow_id,
            'workflow_name': self.workflow_name,
            'step_count': self._step_counter,
            'checkpoint_count': len(self._checkpoints),
            'operation_count': getattr(self, '_operation_count', 0),
            'alert_count': getattr(self, '_alert_count', 0),
            'governance_attributes': self.governance_attrs.copy(),
            'current_cost': current_summary.total_cost if current_summary else 0.0,
            'providers_used': list(current_summary.unique_providers) if current_summary else [],
            'models_used': list(current_summary.unique_models) if current_summary else []
        }
