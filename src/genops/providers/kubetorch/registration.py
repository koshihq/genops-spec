"""
Auto-instrumentation registration for Kubetorch.

This module provides zero-code setup for Kubetorch governance tracking.
It handles global registration and lifecycle management of instrumentation.

Example (Zero-Code Setup):
    >>> from genops.providers.kubetorch import auto_instrument_kubetorch
    >>> auto_instrument_kubetorch(team="ml-research", project="llm-training")
    >>> # All Kubetorch operations now automatically tracked!

Example (Manual Cleanup):
    >>> from genops.providers.kubetorch import uninstrument_kubetorch
    >>> uninstrument_kubetorch()
    >>> # Instrumentation removed, back to normal Kubetorch behavior
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Global state
_global_adapter: Optional[Any] = None
_global_monitor: Optional[Any] = None
_instrumentation_enabled: bool = False


def auto_instrument_kubetorch(
    team: Optional[str] = None,
    project: Optional[str] = None,
    customer_id: Optional[str] = None,
    environment: Optional[str] = None,
    cost_center: Optional[str] = None,
    enable_monitoring: bool = True,
    enable_cost_tracking: bool = True,
    **kwargs
) -> bool:
    """
    Enable zero-code auto-instrumentation for Kubetorch.

    This function sets up global instrumentation that automatically tracks
    all Kubetorch compute operations without requiring code changes.

    Args:
        team: Team name for governance attribution
        project: Project name for governance attribution
        customer_id: Customer ID for billing attribution
        environment: Environment (dev/staging/prod)
        cost_center: Cost center for financial reporting
        enable_monitoring: Enable operation monitoring
        enable_cost_tracking: Enable cost aggregation
        **kwargs: Additional governance attributes

    Returns:
        True if instrumentation was enabled, False if already enabled

    Example:
        >>> from genops.providers.kubetorch import auto_instrument_kubetorch
        >>> auto_instrument_kubetorch(
        ...     team="ml-research",
        ...     project="llm-training",
        ...     environment="production"
        ... )
        >>> # Your Kubetorch code here - automatically tracked!

    Note:
        This is idempotent - calling multiple times has no additional effect
        unless uninstrument_kubetorch() is called first.
    """
    global _global_adapter, _global_monitor, _instrumentation_enabled

    if _instrumentation_enabled:
        logger.warning("Kubetorch auto-instrumentation already enabled")
        return False

    logger.info("Enabling Kubetorch auto-instrumentation")

    try:
        # Import adapter
        from .adapter import instrument_kubetorch

        # Create adapter with governance attributes
        governance_attrs = {}
        if team:
            governance_attrs['team'] = team
        if project:
            governance_attrs['project'] = project
        if customer_id:
            governance_attrs['customer_id'] = customer_id
        if environment:
            governance_attrs['environment'] = environment
        if cost_center:
            governance_attrs['cost_center'] = cost_center

        # Merge additional kwargs
        governance_attrs.update(kwargs)

        # Create global adapter
        _global_adapter = instrument_kubetorch(
            cost_tracking_enabled=enable_cost_tracking,
            **governance_attrs
        )

        # Enable operation monitoring if requested
        if enable_monitoring:
            try:
                from .compute_monitor import create_compute_monitor

                _global_monitor = create_compute_monitor(_global_adapter)
                monitor_enabled = _global_monitor.enable_instrumentation()

                if not monitor_enabled:
                    logger.info(
                        "Kubetorch monitoring not available "
                        "(runhouse not installed). "
                        "Cost tracking and basic telemetry still active."
                    )

            except Exception as e:
                logger.warning(f"Failed to enable monitoring: {e}")
                _global_monitor = None

        _instrumentation_enabled = True

        logger.info(
            f"Kubetorch auto-instrumentation enabled "
            f"(monitoring={_global_monitor is not None})"
        )

        return True

    except Exception as e:
        logger.error(f"Failed to enable Kubetorch auto-instrumentation: {e}")
        _cleanup_global_state()
        raise


def uninstrument_kubetorch() -> bool:
    """
    Disable and remove Kubetorch auto-instrumentation.

    Restores all instrumented methods to their original behavior and
    cleans up global state.

    Returns:
        True if instrumentation was disabled, False if not enabled

    Example:
        >>> from genops.providers.kubetorch import uninstrument_kubetorch
        >>> uninstrument_kubetorch()
        >>> # Back to normal Kubetorch behavior
    """
    global _global_adapter, _global_monitor, _instrumentation_enabled

    if not _instrumentation_enabled:
        logger.warning("Kubetorch auto-instrumentation not enabled")
        return False

    logger.info("Disabling Kubetorch auto-instrumentation")

    try:
        # Disable monitoring first
        if _global_monitor is not None:
            try:
                _global_monitor.disable_instrumentation()
            except Exception as e:
                logger.warning(f"Failed to disable monitoring: {e}")

        # Clean up global state
        _cleanup_global_state()

        logger.info("Kubetorch auto-instrumentation disabled")
        return True

    except Exception as e:
        logger.error(f"Failed to disable Kubetorch auto-instrumentation: {e}")
        raise


def is_kubetorch_instrumented() -> bool:
    """
    Check if Kubetorch auto-instrumentation is currently enabled.

    Returns:
        True if instrumentation is active, False otherwise

    Example:
        >>> from genops.providers.kubetorch import is_kubetorch_instrumented
        >>> if is_kubetorch_instrumented():
        ...     print("Kubetorch is being tracked")
    """
    return _instrumentation_enabled


def get_global_adapter() -> Optional[Any]:
    """
    Get the global adapter instance (if instrumentation is enabled).

    Returns:
        GenOpsKubetorchAdapter instance or None

    Note:
        This is mainly for internal use and debugging.
    """
    return _global_adapter


def get_global_monitor() -> Optional[Any]:
    """
    Get the global monitor instance (if monitoring is enabled).

    Returns:
        KubetorchComputeMonitor instance or None

    Note:
        This is mainly for internal use and debugging.
    """
    return _global_monitor


def get_instrumentation_status() -> Dict[str, Any]:
    """
    Get detailed instrumentation status information.

    Returns:
        Dict with instrumentation status details

    Example:
        >>> from genops.providers.kubetorch import get_instrumentation_status
        >>> status = get_instrumentation_status()
        >>> print(status['enabled'])
        True
    """
    status = {
        'enabled': _instrumentation_enabled,
        'adapter': _global_adapter is not None,
        'monitor': _global_monitor is not None,
    }

    # Add monitor details if available
    if _global_monitor is not None:
        try:
            status['monitor_status'] = _global_monitor.get_instrumentation_status()
        except Exception as e:
            logger.debug(f"Failed to get monitor status: {e}")

    # Add adapter details if available
    if _global_adapter is not None:
        try:
            status['governance_attributes'] = {
                'team': getattr(_global_adapter, 'team', None),
                'project': getattr(_global_adapter, 'project', None),
                'customer_id': getattr(_global_adapter, 'customer_id', None),
                'environment': getattr(_global_adapter, 'environment', None),
            }
        except Exception as e:
            logger.debug(f"Failed to get adapter details: {e}")

    return status


def _cleanup_global_state() -> None:
    """Clean up global instrumentation state."""
    global _global_adapter, _global_monitor, _instrumentation_enabled

    _global_adapter = None
    _global_monitor = None
    _instrumentation_enabled = False


# Context manager for temporary instrumentation
class temporary_instrumentation:
    """
    Context manager for temporary auto-instrumentation.

    Enables instrumentation for the duration of the context and automatically
    disables it when exiting.

    Example:
        >>> from genops.providers.kubetorch.registration import temporary_instrumentation
        >>> with temporary_instrumentation(team="ml-research"):
        ...     # Kubetorch operations tracked here
        ...     pass
        >>> # Instrumentation automatically disabled
    """

    def __init__(self, **kwargs):
        """
        Initialize temporary instrumentation context.

        Args:
            **kwargs: Arguments to pass to auto_instrument_kubetorch()
        """
        self.kwargs = kwargs
        self.was_already_enabled = False

    def __enter__(self):
        """Enable instrumentation on context entry."""
        self.was_already_enabled = is_kubetorch_instrumented()

        if not self.was_already_enabled:
            auto_instrument_kubetorch(**self.kwargs)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Disable instrumentation on context exit."""
        # Only disable if we enabled it
        if not self.was_already_enabled and is_kubetorch_instrumented():
            uninstrument_kubetorch()
