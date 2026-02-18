"""Registration and auto-instrumentation for MLflow provider."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def register_mlflow_provider() -> bool:
    """
    Register MLflow provider with GenOps instrumentation system.

    Returns:
        True if registration successful, False otherwise
    """
    try:
        from genops.auto_instrumentation import register_provider

        from .adapter import GenOpsMLflowAdapter

        # Register the provider
        register_provider(
            provider_name="mlflow",
            provider_class=GenOpsMLflowAdapter,
            framework_type="data_platform",
            auto_detect_modules=["mlflow", "mlflow.tracking"],
            description="MLflow experiment tracking and model registry governance",
        )

        logger.info("MLflow provider registered successfully")
        return True

    except ImportError as e:
        logger.warning(f"Could not register MLflow provider: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to register MLflow provider: {e}")
        return False


def auto_register() -> None:
    """
    Automatically register the MLflow provider if dependencies are available.

    This function is called when the provider module is imported.
    """
    try:
        # Check if MLflow is available
        import mlflow  # noqa: F401

        # Attempt registration
        success = register_mlflow_provider()
        if success:
            logger.debug("MLflow auto-registration completed")
        else:
            logger.debug("MLflow auto-registration failed")

    except ImportError:
        logger.debug(
            "MLflow not found, skipping auto-registration. "
            "Install MLflow to enable experiment tracking governance."
        )
    except Exception as e:
        logger.warning(f"MLflow auto-registration error: {e}")


def auto_instrument_mlflow() -> Any | None:
    """
    Automatically instrument existing MLflow operations with zero-code setup.

    Features:
    - Auto-detects MLflow configuration from environment
    - Enables governance tracking with intelligent defaults
    - Works with existing code without modification

    Returns:
        Instrumented adapter if successful, None otherwise

    Example:
        ```python
        from genops.providers.mlflow import auto_instrument_mlflow

        # Zero-code setup - just call this once
        auto_instrument_mlflow()

        # Your existing MLflow code works automatically with governance
        import mlflow

        mlflow.set_experiment("my-experiment")
        with mlflow.start_run():
            mlflow.log_param("param1", 5)
            mlflow.log_metric("metric1", 0.95)
        ```
    """
    try:
        # Import mlflow if available
        try:
            import mlflow  # noqa: F401
        except ImportError:
            logger.debug("MLflow not available for auto-instrumentation")
            return None

        # Import our adapter
        from .adapter import instrument_mlflow

        # Auto-detect configuration
        auto_config = _detect_mlflow_configuration()

        # Create adapter with auto-detected configuration
        adapter = instrument_mlflow(
            tracking_uri=auto_config["tracking_uri"],
            registry_uri=auto_config.get("registry_uri"),
            **auto_config.get("governance_attrs", {}),
        )

        # Enable auto-patching
        if auto_config.get("enable_auto_patching", True):
            adapter.instrument_framework()

        logger.info(
            f"MLflow auto-instrumentation enabled for "
            f"tracking URI: {auto_config['tracking_uri']}"
        )
        return adapter

    except Exception as e:
        logger.warning(f"MLflow auto-instrumentation failed: {e}")
        return None


def _detect_mlflow_configuration() -> dict[str, Any]:
    """
    Auto-detect MLflow configuration from environment.

    Returns:
        Dictionary with detected configuration
    """
    config = {}

    # Tracking URI detection
    tracking_uri = (
        os.getenv("MLFLOW_TRACKING_URI") or "file:///mlruns"  # Default local storage
    )
    config["tracking_uri"] = tracking_uri

    # Registry URI (optional)
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI")
    if registry_uri:
        config["registry_uri"] = registry_uri

    # Governance attributes with intelligent defaults
    governance_attrs = {}

    # Team attribution
    team = (
        os.getenv("GENOPS_TEAM")
        or os.getenv("MLFLOW_TEAM")
        or os.getenv("TEAM_NAME")
        or os.getenv("USER", "unknown-team")
    )
    if team and team != "unknown-team":
        governance_attrs["team"] = team

    # Project attribution
    project = (
        os.getenv("GENOPS_PROJECT")
        or os.getenv("MLFLOW_PROJECT")
        or os.getenv("PROJECT_NAME")
        or "auto-detected"
    )
    governance_attrs["project"] = project

    # Environment detection
    environment = (
        os.getenv("GENOPS_ENVIRONMENT")
        or os.getenv("MLFLOW_ENV")
        or os.getenv("ENVIRONMENT")
        or "development"
    )
    governance_attrs["environment"] = environment

    # Customer ID (optional)
    customer_id = os.getenv("GENOPS_CUSTOMER_ID")
    if customer_id:
        governance_attrs["customer_id"] = customer_id

    # Cost center (optional)
    cost_center = os.getenv("GENOPS_COST_CENTER")
    if cost_center:
        governance_attrs["cost_center"] = cost_center

    config["governance_attrs"] = governance_attrs  # type: ignore[assignment]

    # Feature toggles
    config["enable_auto_patching"] = _str_to_bool(  # type: ignore[assignment]
        os.getenv("GENOPS_ENABLE_AUTO_PATCHING", "true")
    )

    logger.debug(f"Auto-detected MLflow configuration: {config}")

    return config


def _str_to_bool(value: str) -> bool:
    """Convert string environment variable to boolean."""
    return value.lower() in ("true", "1", "yes", "on", "enabled")


def patch_mlflow_operations(adapter: Any) -> None:
    """
    Patch common MLflow operations to include GenOps governance tracking.

    Args:
        adapter: MLflow adapter instance
    """
    try:
        adapter.instrument_framework()
        logger.debug("MLflow operations patched for governance tracking")
    except Exception as e:
        logger.warning(f"Failed to patch MLflow operations: {e}")
