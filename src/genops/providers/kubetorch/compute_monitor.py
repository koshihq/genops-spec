"""
Framework-specific instrumentation for Kubetorch operations.

This module provides instrumentation hooks for Kubetorch-specific operations:
- Resource allocation (.to(compute))
- Scaling operations (.autoscale())
- Checkpointing
- Fault recovery and migration

The instrumentation is designed to be:
- Non-invasive (preserves original behavior)
- Reversible (can be removed cleanly)
- Gracefully degrading (works without Kubetorch installed)
"""

import logging
import time
from typing import Any, Callable, Dict, Optional, Set
from functools import wraps

logger = logging.getLogger(__name__)


class KubetorchComputeMonitor:
    """
    Monitors and instruments Kubetorch compute operations.

    Provides hooks for tracking:
    - Resource allocation and compute placement
    - Dynamic scaling operations
    - Checkpoint creation and restoration
    - Fault recovery and migration events

    Example:
        >>> from genops.providers.kubetorch import GenOpsKubetorchAdapter
        >>> adapter = GenOpsKubetorchAdapter()
        >>> monitor = KubetorchComputeMonitor(adapter)
        >>> monitor.enable_instrumentation()
        >>> # Kubetorch operations now tracked
        >>> monitor.disable_instrumentation()
    """

    def __init__(
        self,
        adapter: Any,
        enable_resource_allocation: bool = True,
        enable_scaling: bool = True,
        enable_checkpointing: bool = True,
        enable_fault_recovery: bool = True,
    ):
        """
        Initialize compute monitor.

        Args:
            adapter: GenOpsKubetorchAdapter instance for telemetry
            enable_resource_allocation: Track resource allocation operations
            enable_scaling: Track scaling operations
            enable_checkpointing: Track checkpoint operations
            enable_fault_recovery: Track fault recovery operations
        """
        self.adapter = adapter
        self.enabled = False

        # Feature flags
        self.enable_resource_allocation = enable_resource_allocation
        self.enable_scaling = enable_scaling
        self.enable_checkpointing = enable_checkpointing
        self.enable_fault_recovery = enable_fault_recovery

        # Original methods storage for reversibility
        self._original_methods: Dict[str, Callable] = {}
        self._instrumented_classes: Set[str] = set()

        logger.debug("Initialized KubetorchComputeMonitor")

    def enable_instrumentation(self) -> bool:
        """
        Enable instrumentation of Kubetorch operations.

        Returns:
            True if instrumentation was enabled, False if Kubetorch not available

        Raises:
            RuntimeError: If instrumentation is already enabled
        """
        if self.enabled:
            raise RuntimeError("Instrumentation already enabled")

        # Check if Kubetorch is available
        try:
            import runhouse as rh
            self._kubetorch_available = True
        except ImportError:
            logger.warning(
                "Kubetorch (runhouse) not installed. "
                "Instrumentation will be no-op. "
                "Install with: pip install runhouse"
            )
            self._kubetorch_available = False
            return False

        logger.info("Enabling Kubetorch instrumentation")

        # Apply instrumentation hooks
        if self.enable_resource_allocation:
            self._instrument_resource_allocation()

        if self.enable_scaling:
            self._instrument_scaling_operations()

        if self.enable_checkpointing:
            self._instrument_checkpointing()

        if self.enable_fault_recovery:
            self._instrument_fault_recovery()

        self.enabled = True
        logger.info(
            f"Kubetorch instrumentation enabled "
            f"({len(self._instrumented_classes)} classes instrumented)"
        )
        return True

    def disable_instrumentation(self) -> None:
        """
        Disable and remove instrumentation.

        Restores all original methods to their pre-instrumentation state.
        """
        if not self.enabled:
            logger.warning("Instrumentation not enabled")
            return

        logger.info("Disabling Kubetorch instrumentation")

        # Restore original methods
        for method_path, original_method in self._original_methods.items():
            self._restore_method(method_path, original_method)

        self._original_methods.clear()
        self._instrumented_classes.clear()
        self.enabled = False

        logger.info("Kubetorch instrumentation disabled")

    def _instrument_resource_allocation(self) -> None:
        """
        Instrument resource allocation operations (.to(compute)).

        Intercepts calls to move computations to specific resources and
        tracks GPU/CPU allocation decisions.
        """
        if not self._kubetorch_available:
            return

        try:
            import runhouse as rh

            # Instrument Module.to() method
            if hasattr(rh, 'Module') and hasattr(rh.Module, 'to'):
                original_to = rh.Module.to
                method_path = 'runhouse.Module.to'

                @wraps(original_to)
                def instrumented_to(self, *args, **kwargs):
                    """Instrumented .to() method."""
                    start_time = time.time()

                    # Extract compute resource information
                    compute_resource = args[0] if args else kwargs.get('system')
                    resource_info = self._extract_resource_info(compute_resource)

                    # Track the allocation
                    operation_id = f"allocate-{id(self)}-{int(start_time * 1000)}"

                    logger.debug(
                        f"Resource allocation: {operation_id} -> "
                        f"{resource_info.get('instance_type', 'unknown')}"
                    )

                    # Call original method
                    try:
                        result = original_to(self, *args, **kwargs)
                        duration = time.time() - start_time

                        # Record telemetry
                        self.adapter.track_compute_deployment(
                            instance_type=resource_info.get('instance_type', 'unknown'),
                            num_devices=resource_info.get('num_devices', 1),
                            workload_type='resource_allocation',
                            duration_seconds=duration,
                            operation_id=operation_id,
                            metadata=resource_info
                        )

                        return result

                    except Exception as e:
                        logger.error(f"Resource allocation failed: {e}")
                        raise

                # Store original and apply instrumentation
                self._original_methods[method_path] = original_to
                rh.Module.to = instrumented_to
                self._instrumented_classes.add('runhouse.Module')

                logger.debug("Instrumented runhouse.Module.to()")

        except Exception as e:
            logger.warning(f"Failed to instrument resource allocation: {e}")

    def _instrument_scaling_operations(self) -> None:
        """
        Instrument scaling operations (.autoscale()).

        Tracks dynamic scaling events including scale-up and scale-down operations.
        """
        if not self._kubetorch_available:
            return

        try:
            import runhouse as rh

            # Instrument Cluster.autoscale() if available
            if hasattr(rh, 'Cluster') and hasattr(rh.Cluster, 'autoscale'):
                original_autoscale = rh.Cluster.autoscale
                method_path = 'runhouse.Cluster.autoscale'

                @wraps(original_autoscale)
                def instrumented_autoscale(self, *args, **kwargs):
                    """Instrumented .autoscale() method."""
                    start_time = time.time()

                    # Extract scaling parameters
                    min_workers = kwargs.get('min_workers', 0)
                    max_workers = kwargs.get('max_workers', 10)

                    logger.debug(
                        f"Autoscale triggered: min={min_workers}, max={max_workers}"
                    )

                    # Call original method
                    try:
                        result = original_autoscale(self, *args, **kwargs)
                        duration = time.time() - start_time

                        # Record telemetry
                        self.adapter.track_compute_deployment(
                            instance_type='autoscale',
                            num_devices=max_workers,
                            workload_type='scaling',
                            duration_seconds=duration,
                            metadata={
                                'action': 'autoscale',
                                'min_workers': min_workers,
                                'max_workers': max_workers,
                            }
                        )

                        return result

                    except Exception as e:
                        logger.error(f"Autoscale failed: {e}")
                        raise

                # Store original and apply instrumentation
                self._original_methods[method_path] = original_autoscale
                rh.Cluster.autoscale = instrumented_autoscale
                self._instrumented_classes.add('runhouse.Cluster')

                logger.debug("Instrumented runhouse.Cluster.autoscale()")

        except Exception as e:
            logger.warning(f"Failed to instrument scaling operations: {e}")

    def _instrument_checkpointing(self) -> None:
        """
        Instrument checkpoint operations.

        Tracks checkpoint creation, restoration, and storage costs.
        """
        if not self._kubetorch_available:
            return

        try:
            import runhouse as rh

            # Instrument checkpoint save/load if available
            if hasattr(rh, 'Module'):
                # Instrument save_checkpoint
                if hasattr(rh.Module, 'save_checkpoint'):
                    original_save = rh.Module.save_checkpoint
                    method_path = 'runhouse.Module.save_checkpoint'

                    @wraps(original_save)
                    def instrumented_save_checkpoint(self, *args, **kwargs):
                        """Instrumented save_checkpoint method."""
                        start_time = time.time()
                        checkpoint_path = args[0] if args else kwargs.get('path')

                        logger.debug(f"Checkpoint save: {checkpoint_path}")

                        try:
                            result = original_save(self, *args, **kwargs)
                            duration = time.time() - start_time

                            # Estimate checkpoint size (would need actual file size in production)
                            checkpoint_size_gb = kwargs.get('size_gb', 10.0)

                            # Record telemetry
                            self.adapter.track_compute_deployment(
                                instance_type='storage',
                                num_devices=1,
                                workload_type='checkpoint_save',
                                duration_seconds=duration,
                                metadata={
                                    'checkpoint_path': str(checkpoint_path),
                                    'checkpoint_size_gb': checkpoint_size_gb,
                                }
                            )

                            return result

                        except Exception as e:
                            logger.error(f"Checkpoint save failed: {e}")
                            raise

                    self._original_methods[method_path] = original_save
                    rh.Module.save_checkpoint = instrumented_save_checkpoint
                    self._instrumented_classes.add('runhouse.Module')

                    logger.debug("Instrumented runhouse.Module.save_checkpoint()")

        except Exception as e:
            logger.warning(f"Failed to instrument checkpointing: {e}")

    def _instrument_fault_recovery(self) -> None:
        """
        Instrument fault recovery and migration operations.

        Tracks retry attempts, job migrations, and failure recovery.
        """
        if not self._kubetorch_available:
            return

        try:
            import runhouse as rh

            # Instrument retry/migrate operations if available
            if hasattr(rh, 'Cluster') and hasattr(rh.Cluster, 'restart'):
                original_restart = rh.Cluster.restart
                method_path = 'runhouse.Cluster.restart'

                @wraps(original_restart)
                def instrumented_restart(self, *args, **kwargs):
                    """Instrumented restart method."""
                    start_time = time.time()

                    logger.debug("Cluster restart initiated")

                    try:
                        result = original_restart(self, *args, **kwargs)
                        duration = time.time() - start_time

                        # Record telemetry
                        self.adapter.track_compute_deployment(
                            instance_type='recovery',
                            num_devices=1,
                            workload_type='fault_recovery',
                            duration_seconds=duration,
                            metadata={
                                'action': 'restart',
                                'reason': kwargs.get('reason', 'unknown'),
                            }
                        )

                        return result

                    except Exception as e:
                        logger.error(f"Cluster restart failed: {e}")
                        raise

                self._original_methods[method_path] = original_restart
                rh.Cluster.restart = instrumented_restart
                self._instrumented_classes.add('runhouse.Cluster')

                logger.debug("Instrumented runhouse.Cluster.restart()")

        except Exception as e:
            logger.warning(f"Failed to instrument fault recovery: {e}")

    def _extract_resource_info(self, compute_resource: Any) -> Dict[str, Any]:
        """
        Extract resource information from compute resource object.

        Args:
            compute_resource: Kubetorch compute resource object

        Returns:
            Dict with resource information (instance_type, num_devices, etc.)
        """
        resource_info = {
            'instance_type': 'unknown',
            'num_devices': 1,
        }

        try:
            # Try to extract instance type
            if hasattr(compute_resource, 'instance_type'):
                resource_info['instance_type'] = compute_resource.instance_type
            elif hasattr(compute_resource, 'name'):
                resource_info['instance_type'] = compute_resource.name

            # Try to extract device count
            if hasattr(compute_resource, 'num_gpus'):
                resource_info['num_devices'] = compute_resource.num_gpus
            elif hasattr(compute_resource, 'gpus'):
                resource_info['num_devices'] = len(compute_resource.gpus)

        except Exception as e:
            logger.debug(f"Failed to extract full resource info: {e}")

        return resource_info

    def _restore_method(self, method_path: str, original_method: Callable) -> None:
        """
        Restore original method.

        Args:
            method_path: Path to method (e.g., 'runhouse.Module.to')
            original_method: Original method to restore
        """
        try:
            parts = method_path.split('.')
            module_name = '.'.join(parts[:-2])
            class_name = parts[-2]
            method_name = parts[-1]

            import importlib
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
            setattr(cls, method_name, original_method)

            logger.debug(f"Restored original method: {method_path}")

        except Exception as e:
            logger.warning(f"Failed to restore method {method_path}: {e}")

    def get_instrumentation_status(self) -> Dict[str, Any]:
        """
        Get current instrumentation status.

        Returns:
            Dict with instrumentation status information
        """
        return {
            'enabled': self.enabled,
            'kubetorch_available': self._kubetorch_available,
            'instrumented_classes': list(self._instrumented_classes),
            'feature_flags': {
                'resource_allocation': self.enable_resource_allocation,
                'scaling': self.enable_scaling,
                'checkpointing': self.enable_checkpointing,
                'fault_recovery': self.enable_fault_recovery,
            }
        }


def create_compute_monitor(adapter: Any, **kwargs) -> KubetorchComputeMonitor:
    """
    Create and configure a compute monitor.

    Args:
        adapter: GenOpsKubetorchAdapter instance
        **kwargs: Additional configuration options

    Returns:
        Configured KubetorchComputeMonitor instance

    Example:
        >>> from genops.providers.kubetorch import instrument_kubetorch
        >>> from genops.providers.kubetorch.compute_monitor import create_compute_monitor
        >>> adapter = instrument_kubetorch()
        >>> monitor = create_compute_monitor(adapter)
        >>> monitor.enable_instrumentation()
    """
    return KubetorchComputeMonitor(adapter, **kwargs)
