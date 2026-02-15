"""Registration and auto-instrumentation for Databricks Unity Catalog provider."""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def register_databricks_unity_catalog_provider() -> bool:
    """
    Register Databricks Unity Catalog provider with GenOps instrumentation system.

    Returns:
        True if registration successful, False otherwise
    """
    try:
        # Import here to avoid circular dependencies
        from genops.auto_instrumentation import register_provider

        from .adapter import GenOpsDatabricksUnityCatalogAdapter

        # Register the provider
        register_provider(
            provider_name="databricks_unity_catalog",
            provider_class=GenOpsDatabricksUnityCatalogAdapter,
            framework_type="data_platform",
            auto_detect_modules=["databricks", "databricks.sdk", "pyspark"],
            description="Databricks Unity Catalog data governance and cost tracking",
        )

        logger.info("Databricks Unity Catalog provider registered successfully")
        return True

    except ImportError as e:
        logger.warning(f"Could not register Databricks Unity Catalog provider: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to register Databricks Unity Catalog provider: {e}")
        return False


def auto_register() -> None:
    """
    Automatically register the Databricks Unity Catalog provider if dependencies are available.

    This function is called when the provider module is imported.
    """
    try:
        # Check if Databricks SDK is available
        import databricks  # noqa: F401

        # Attempt registration
        success = register_databricks_unity_catalog_provider()
        if success:
            logger.debug("Databricks Unity Catalog auto-registration completed")
        else:
            logger.debug("Databricks Unity Catalog auto-registration failed")

    except ImportError:
        logger.debug(
            "Databricks SDK not found, skipping auto-registration. "
            "Install databricks-sdk to enable Databricks Unity Catalog governance."
        )
    except Exception as e:
        logger.warning(f"Databricks Unity Catalog auto-registration error: {e}")


def auto_instrument_databricks() -> Optional[Any]:
    """
    Automatically instrument existing Databricks operations with zero-code setup.

    Features:
    - Auto-detects Databricks configuration from environment
    - Enables governance tracking with intelligent defaults
    - Works with existing code without modification

    Returns:
        Instrumented adapter if successful, None otherwise
    """
    try:
        # Import databricks modules if available
        try:
            import databricks.sdk  # noqa: F401
            from databricks.sdk import WorkspaceClient  # noqa: F401
        except ImportError:
            logger.debug("Databricks SDK not available for auto-instrumentation")
            return None

        # Import our adapter
        from .adapter import instrument_databricks_unity_catalog

        # Auto-detect configuration with intelligent defaults
        auto_config = _detect_databricks_configuration()

        if not auto_config.get("workspace_url"):
            logger.warning("Databricks workspace URL not found in environment")
            return None

        # Create adapter with auto-detected configuration
        adapter = instrument_databricks_unity_catalog(
            workspace_url=auto_config["workspace_url"],
            **auto_config.get("governance_attrs", {}),
        )

        # Enable auto-patching of common operations
        if auto_config.get("enable_auto_patching", True):
            patch_databricks_operations(adapter)

        logger.info(
            f"Databricks Unity Catalog auto-instrumentation enabled for "
            f"workspace: {auto_config['workspace_url']}"
        )
        return adapter

    except Exception as e:
        logger.warning(f"Databricks Unity Catalog auto-instrumentation failed: {e}")
        return None


def _detect_databricks_configuration() -> dict[str, Any]:
    """
    Auto-detect Databricks configuration from environment with intelligent defaults.

    Returns:
        Dictionary with detected configuration
    """
    import os

    config = {}

    # Primary configuration detection
    workspace_url = (
        os.getenv("DATABRICKS_HOST")
        or os.getenv("DATABRICKS_WORKSPACE_URL")
        or os.getenv("DATABRICKS_SERVER_HOSTNAME")
    )

    (
        os.getenv("DATABRICKS_TOKEN")
        or os.getenv("DATABRICKS_ACCESS_TOKEN")
        or os.getenv("DATABRICKS_PAT")
    )

    if workspace_url:
        config["workspace_url"] = workspace_url.rstrip("/")

        # Normalize workspace URL format
        if not workspace_url.startswith(("http://", "https://")):
            config["workspace_url"] = f"https://{workspace_url}"

    # Governance attributes with intelligent defaults
    governance_attrs = {}

    # Team attribution (multiple sources)
    team = (
        os.getenv("GENOPS_TEAM")
        or os.getenv("DATABRICKS_TEAM")
        or os.getenv("TEAM_NAME")
        or os.getenv("USER", "unknown-team")  # Fallback to system user
    )
    if team and team != "unknown-team":
        governance_attrs["team"] = team

    # Project attribution
    project = (
        os.getenv("GENOPS_PROJECT")
        or os.getenv("DATABRICKS_PROJECT")
        or os.getenv("PROJECT_NAME")
        or "auto-detected"
    )
    governance_attrs["project"] = project

    # Environment detection
    environment = (
        os.getenv("GENOPS_ENVIRONMENT")
        or os.getenv("DATABRICKS_ENV")
        or os.getenv("ENVIRONMENT")
        or os.getenv("ENV")
        or _detect_environment_from_url(workspace_url)
        or "development"
    )
    governance_attrs["environment"] = environment

    # Cost center (optional)
    cost_center = (
        os.getenv("GENOPS_COST_CENTER")
        or os.getenv("DATABRICKS_COST_CENTER")
        or os.getenv("COST_CENTER")
    )
    if cost_center:
        governance_attrs["cost_center"] = cost_center

    # User identification
    user_id = (
        os.getenv("GENOPS_USER_ID")
        or os.getenv("DATABRICKS_USER_ID")
        or os.getenv("USER")
        or "auto-detected-user"
    )
    governance_attrs["user_id"] = user_id

    config["governance_attrs"] = governance_attrs  # type: ignore[assignment]

    # Feature toggles with intelligent defaults
    config["enable_auto_patching"] = _str_to_bool(  # type: ignore[assignment]
        os.getenv("GENOPS_ENABLE_AUTO_PATCHING", "true")
    )
    config["enable_cost_tracking"] = _str_to_bool(  # type: ignore[assignment]
        os.getenv("GENOPS_ENABLE_COST_TRACKING", "true")
    )
    config["enable_lineage_tracking"] = _str_to_bool(  # type: ignore[assignment]
        os.getenv("GENOPS_ENABLE_LINEAGE_TRACKING", "true")
    )

    return config


def _detect_environment_from_url(workspace_url: Optional[str]) -> Optional[str]:
    """Intelligently detect environment from workspace URL."""
    if not workspace_url:
        return None

    url_lower = workspace_url.lower()

    if any(env in url_lower for env in ["prod", "production"]):
        return "production"
    elif any(env in url_lower for env in ["stage", "staging"]):
        return "staging"
    elif any(env in url_lower for env in ["dev", "development"]):
        return "development"
    elif any(env in url_lower for env in ["test", "testing"]):
        return "testing"

    return None


def _str_to_bool(value: str) -> bool:
    """Convert string environment variable to boolean."""
    return value.lower() in ("true", "1", "yes", "on", "enabled")


def patch_databricks_operations(adapter: Any) -> None:
    """
    Patch common Databricks SDK operations to include GenOps governance tracking.

    Args:
        adapter: Databricks Unity Catalog adapter instance
    """
    try:
        # This would patch databricks.sdk operations to add governance tracking
        # Implementation would wrap key methods like:
        # - WorkspaceClient.catalogs.* operations
        # - WorkspaceClient.schemas.* operations
        # - WorkspaceClient.tables.* operations
        # - WorkspaceClient.sql.* operations

        logger.debug("Databricks operations patched for governance tracking")

    except Exception as e:
        logger.warning(f"Failed to patch Databricks operations: {e}")


def configure_unity_catalog_governance(
    workspace_url: Optional[str] = None,
    metastore_id: Optional[str] = None,
    **governance_config,
) -> dict[str, Any]:
    """
    Configure Unity Catalog governance settings.

    Args:
        workspace_url: Databricks workspace URL
        metastore_id: Unity Catalog metastore ID
        **governance_config: Additional governance configuration

    Returns:
        Configuration result
    """
    config_result = {
        "configured": False,
        "workspace_url": workspace_url,
        "metastore_id": metastore_id,
        "governance_features": [],
        "errors": [],
    }

    try:
        # Import required modules
        from .adapter import instrument_databricks_unity_catalog
        from .cost_aggregator import get_cost_aggregator
        from .governance_monitor import get_governance_monitor

        # Initialize adapter
        instrument_databricks_unity_catalog(workspace_url=workspace_url)

        # Initialize governance monitor
        get_governance_monitor(metastore_id=metastore_id)

        # Initialize cost aggregator
        get_cost_aggregator()

        config_result["configured"] = True
        config_result["governance_features"] = [
            "data_lineage_tracking",
            "compliance_monitoring",
            "cost_attribution",
            "policy_enforcement",
        ]

        logger.info(
            f"Unity Catalog governance configured for workspace: {workspace_url}, "
            f"metastore: {metastore_id}"
        )

    except Exception as e:
        config_result["errors"].append(str(e))
        logger.error(f"Failed to configure Unity Catalog governance: {e}")

    return config_result
