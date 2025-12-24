"""Base framework provider interface for GenOps AI governance."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from genops.core.telemetry import GenOpsTelemetry

logger = logging.getLogger(__name__)


class BaseFrameworkProvider(ABC):
    """Abstract base class for GenOps framework providers."""

    # Framework types for categorization
    FRAMEWORK_TYPE_ORCHESTRATION = "orchestration"  # LangChain, Haystack
    FRAMEWORK_TYPE_TRAINING = "training"            # PyTorch, TensorFlow
    FRAMEWORK_TYPE_INFERENCE = "inference"          # HuggingFace Transformers
    FRAMEWORK_TYPE_VECTOR = "vector"                # Chroma, Pinecone
    FRAMEWORK_TYPE_MULTIMODAL = "multimodal"        # NeMo
    FRAMEWORK_TYPE_AUTOML = "automl"                # NNI, Optuna
    FRAMEWORK_TYPE_DISTRIBUTED = "distributed"     # Horovod, Ray
    FRAMEWORK_TYPE_DATA_PLATFORM = "data_platform"  # Databricks Unity Catalog, WandB

    def __init__(self, client: Any | None = None, **kwargs):
        """
        Initialize the framework provider.

        Args:
            client: Existing framework client/instance (optional)
            **kwargs: Additional configuration parameters
        """
        self.client = client
        self.telemetry = GenOpsTelemetry()
        self.config = kwargs

        # Standard governance attributes across all providers
        self.GOVERNANCE_ATTRIBUTES = {
            'team', 'project', 'feature', 'customer_id', 'customer',
            'environment', 'cost_center', 'user_id', 'experiment_id',
            'model_version', 'dataset_id', 'training_job_id'
        }

        # Framework-specific request attributes (to be defined by subclasses)
        self.REQUEST_ATTRIBUTES: set[str] = set()

        # Setup any framework-specific configuration
        self.setup_governance_attributes()

    @abstractmethod
    def setup_governance_attributes(self) -> None:
        """Setup framework-specific governance attributes. Override in subclasses."""
        pass

    def _extract_attributes(self, kwargs: dict) -> tuple[dict, dict, dict]:
        """
        Extract governance and request attributes from kwargs.

        Returns:
            Tuple of (governance_attrs, request_attrs, api_kwargs)
        """
        governance_attrs = {}
        request_attrs = {}
        api_kwargs = kwargs.copy()

        # Extract governance attributes
        for attr in self.GOVERNANCE_ATTRIBUTES:
            if attr in kwargs:
                governance_attrs[attr] = kwargs[attr]
                api_kwargs.pop(attr)

        # Extract request attributes
        for attr in self.REQUEST_ATTRIBUTES:
            if attr in kwargs:
                request_attrs[attr] = kwargs[attr]

        return governance_attrs, request_attrs, api_kwargs

    def _build_trace_attributes(
        self,
        operation_name: str,
        operation_type: str,
        governance_attrs: dict,
        **additional_attrs
    ) -> dict:
        """
        Build standardized trace attributes for telemetry.

        Args:
            operation_name: Name of the operation being traced
            operation_type: Type of operation (ai.inference, ai.training, etc.)
            governance_attrs: Governance attributes from request
            **additional_attrs: Additional attributes to include

        Returns:
            Dictionary of trace attributes
        """
        trace_attrs = {
            "operation_name": operation_name,
            "operation_type": operation_type,
            "framework": self.get_framework_name(),
            "framework_type": self.get_framework_type(),
        }

        # Add any additional framework-specific attributes
        trace_attrs.update(additional_attrs)

        # Add effective governance attributes (defaults + context + governance)
        try:
            from genops.core.context import get_effective_attributes
            effective_attrs = get_effective_attributes(**governance_attrs)
            trace_attrs.update(effective_attrs)
        except (ImportError, Exception) as e:
            logger.debug(f"Context integration not available: {e}")
            # Fallback to just governance attributes
            trace_attrs.update(governance_attrs)

        return trace_attrs

    @abstractmethod
    def get_framework_name(self) -> str:
        """Return the framework name (e.g., 'langchain', 'pytorch', 'tensorflow')."""
        pass

    @abstractmethod
    def get_framework_type(self) -> str:
        """Return the framework type (orchestration, training, inference, etc.)."""
        pass

    @abstractmethod
    def get_framework_version(self) -> str | None:
        """Return the installed framework version if available."""
        pass

    @abstractmethod
    def is_framework_available(self) -> bool:
        """Check if the framework is available and can be instrumented."""
        pass

    @abstractmethod
    def calculate_cost(self, operation_context: dict) -> float:
        """
        Calculate cost for framework-specific operations.

        Args:
            operation_context: Dictionary containing operation metadata
                             (tokens, model, duration, etc.)

        Returns:
            Estimated cost in USD
        """
        pass

    @abstractmethod
    def get_operation_mappings(self) -> dict[str, str]:
        """
        Return mapping of framework operations to instrumentation methods.

        Returns:
            Dictionary mapping operation names to method names
            e.g., {'chain.run': 'instrument_chain_run'}
        """
        pass

    def get_supported_operations(self) -> dict[str, str]:
        """
        Get list of supported operations for this framework.

        Returns:
            Dictionary of operation_name -> description
        """
        mappings = self.get_operation_mappings()
        return {op: f"Track {op} operations with governance telemetry"
                for op in mappings.keys()}

    def validate_operation_context(self, context: dict) -> bool:
        """
        Validate that operation context contains required fields.

        Args:
            context: Operation context dictionary

        Returns:
            True if context is valid, False otherwise
        """
        # Default validation - subclasses can override
        return isinstance(context, dict)

    def record_operation_telemetry(
        self,
        span: Any,
        operation_type: str,
        context: dict,
        **metadata
    ) -> None:
        """
        Record framework-specific telemetry on a span.

        Args:
            span: OpenTelemetry span
            operation_type: Type of operation
            context: Operation context data
            **metadata: Additional metadata to record
        """
        # Record cost if available
        if self.validate_operation_context(context):
            try:
                cost = self.calculate_cost(context)
                if cost > 0:
                    self.telemetry.record_cost(
                        span=span,
                        cost=cost,
                        currency="USD",
                        provider=self.get_framework_name(),
                        **metadata
                    )
            except Exception as e:
                logger.warning(f"Failed to calculate cost: {e}")

        # Record framework-specific metrics
        self._record_framework_metrics(span, operation_type, context)

    @abstractmethod
    def _record_framework_metrics(self, span: Any, operation_type: str, context: dict) -> None:
        """
        Record framework-specific metrics. Override in subclasses.

        Args:
            span: OpenTelemetry span
            operation_type: Type of operation
            context: Operation context data
        """
        # Default implementation - subclasses should override
        pass

    def instrument_framework(self, **config) -> bool:
        """
        Apply framework-specific instrumentation.

        Args:
            **config: Configuration options for instrumentation

        Returns:
            True if instrumentation was successful, False otherwise
        """
        if not self.is_framework_available():
            logger.warning(f"Framework {self.get_framework_name()} not available")
            return False

        try:
            self._apply_instrumentation(**config)
            logger.info(f"Successfully instrumented {self.get_framework_name()}")
            return True
        except Exception as e:
            logger.error(f"Failed to instrument {self.get_framework_name()}: {e}")
            return False

    @abstractmethod
    def _apply_instrumentation(self, **config) -> None:
        """Apply the actual instrumentation. Implemented by subclasses."""
        pass

    def uninstrument_framework(self) -> bool:
        """
        Remove framework instrumentation.

        Returns:
            True if uninstrumentation was successful, False otherwise
        """
        try:
            self._remove_instrumentation()
            logger.info(f"Successfully uninstrumented {self.get_framework_name()}")
            return True
        except Exception as e:
            logger.error(f"Failed to uninstrument {self.get_framework_name()}: {e}")
            return False

    @abstractmethod
    def _remove_instrumentation(self) -> None:
        """Remove the actual instrumentation. Implemented by subclasses."""
        pass
