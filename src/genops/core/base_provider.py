"""
Base provider interface for GenOps AI governance.

This module defines the common interface and patterns that all GenOps provider
adapters must implement for consistent behavior across different AI platforms.
"""

import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from genops.core.telemetry import GenOpsTelemetry


@dataclass
class OperationContext:
    """Context for a GenOps operation with timing and metadata."""
    operation_id: str
    operation_type: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    governance_attributes: Dict[str, str] = field(default_factory=dict)

    def finalize(self) -> None:
        """Mark the operation as completed."""
        self.end_time = time.time()

    @property
    def duration(self) -> float:
        """Get the operation duration in seconds."""
        end = self.end_time or time.time()
        return end - self.start_time


class BaseProvider(ABC):
    """
    Base class for all GenOps provider adapters.
    
    This class defines the common interface that all provider adapters
    (OpenAI, Anthropic, Bedrock, etc.) must implement to ensure consistent
    behavior and developer experience across platforms.
    """

    def __init__(self, **kwargs):
        """Initialize the base provider."""
        self.telemetry = GenOpsTelemetry()
        self._default_attributes = {}

    def set_default_attributes(self, **attributes: str) -> None:
        """Set default governance attributes for this provider instance."""
        self._default_attributes.update(attributes)

    def get_effective_attributes(self, **override_attributes: str) -> Dict[str, str]:
        """Get effective governance attributes, combining defaults and overrides."""
        effective = self._default_attributes.copy()
        effective.update(override_attributes)
        return effective

    @contextmanager
    def _create_operation_context(
        self,
        operation_type: str,
        **governance_attributes: str
    ):
        """Create a context manager for tracking an operation."""
        import uuid

        operation_id = str(uuid.uuid4())
        context = OperationContext(
            operation_id=operation_id,
            operation_type=operation_type,
            governance_attributes=self.get_effective_attributes(**governance_attributes)
        )

        try:
            yield context
        finally:
            context.finalize()

    @abstractmethod
    def validate_setup(self) -> Dict[str, Any]:
        """
        Validate that the provider is properly configured.
        
        Returns:
            Dict containing validation results with keys:
            - 'valid': bool indicating if setup is valid
            - 'errors': list of error messages
            - 'warnings': list of warning messages
            - 'recommendations': list of recommendations
        """
        pass

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about this provider.
        
        Returns:
            Dict containing provider information
        """
        return {
            "provider_name": self.__class__.__name__,
            "provider_type": getattr(self, "PROVIDER_TYPE", "unknown"),
            "supported_features": getattr(self, "SUPPORTED_FEATURES", []),
        }


class ProviderRegistry:
    """Registry for managing available GenOps providers."""

    def __init__(self):
        self._providers: Dict[str, type] = {}
        self._instances: Dict[str, BaseProvider] = {}

    def register(self, name: str, provider_class: type) -> None:
        """Register a provider class."""
        if not issubclass(provider_class, BaseProvider):
            raise ValueError(f"Provider {provider_class} must inherit from BaseProvider")
        self._providers[name] = provider_class

    def get_provider_class(self, name: str) -> Optional[type]:
        """Get a provider class by name."""
        return self._providers.get(name)

    def get_provider_instance(self, name: str, **kwargs) -> Optional[BaseProvider]:
        """Get or create a provider instance."""
        if name not in self._providers:
            return None

        if name not in self._instances:
            self._instances[name] = self._providers[name](**kwargs)

        return self._instances[name]

    def list_providers(self) -> List[str]:
        """List all registered provider names."""
        return list(self._providers.keys())


# Global provider registry instance
provider_registry = ProviderRegistry()
