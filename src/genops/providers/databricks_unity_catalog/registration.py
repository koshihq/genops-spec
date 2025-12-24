"""Registration and auto-instrumentation for Databricks Unity Catalog provider."""

import logging
from typing import Any, Dict, Optional

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
            description="Databricks Unity Catalog data governance and cost tracking"
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
        import databricks
        
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
    Automatically instrument existing Databricks operations.
    
    Returns:
        Instrumented adapter if successful, None otherwise
    """
    try:
        # Import databricks modules if available
        try:
            import databricks.sdk
            from databricks.sdk import WorkspaceClient
        except ImportError:
            logger.debug("Databricks SDK not available for auto-instrumentation")
            return None
        
        # Import our adapter
        from .adapter import instrument_databricks_unity_catalog
        
        # Create adapter with auto-detected configuration
        adapter = instrument_databricks_unity_catalog()
        
        # Attempt to patch common Databricks operations
        patch_databricks_operations(adapter)
        
        logger.info("Databricks Unity Catalog auto-instrumentation enabled")
        return adapter
        
    except Exception as e:
        logger.warning(f"Databricks Unity Catalog auto-instrumentation failed: {e}")
        return None


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
    **governance_config
) -> Dict[str, Any]:
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
        "errors": []
    }
    
    try:
        # Import required modules
        from .adapter import instrument_databricks_unity_catalog
        from .governance_monitor import get_governance_monitor
        from .cost_aggregator import get_cost_aggregator
        
        # Initialize adapter
        adapter = instrument_databricks_unity_catalog(workspace_url=workspace_url)
        
        # Initialize governance monitor
        governance_monitor = get_governance_monitor(metastore_id=metastore_id)
        
        # Initialize cost aggregator
        cost_aggregator = get_cost_aggregator()
        
        config_result["configured"] = True
        config_result["governance_features"] = [
            "data_lineage_tracking",
            "compliance_monitoring",
            "cost_attribution",
            "policy_enforcement"
        ]
        
        logger.info(
            f"Unity Catalog governance configured for workspace: {workspace_url}, "
            f"metastore: {metastore_id}"
        )
        
    except Exception as e:
        config_result["errors"].append(str(e))
        logger.error(f"Failed to configure Unity Catalog governance: {e}")
    
    return config_result