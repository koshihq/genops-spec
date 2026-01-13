"""Databricks Unity Catalog adapter for GenOps AI governance."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from genops.core.telemetry import GenOpsTelemetry
from genops.providers.base.provider import BaseFrameworkProvider

logger = logging.getLogger(__name__)


class GenOpsDatabricksUnityCatalogAdapter(BaseFrameworkProvider):
    """
    GenOps adapter for Databricks Unity Catalog data governance operations.
    
    Provides comprehensive governance telemetry, cost tracking, and policy enforcement
    for Unity Catalog data operations across multi-workspace environments.
    """

    def __init__(self, workspace_url: Optional[str] = None, **kwargs):
        """
        Initialize Databricks Unity Catalog adapter.

        Args:
            workspace_url: Databricks workspace URL (optional, can be set via env)
            **kwargs: Additional configuration parameters
        """
        super().__init__(**kwargs)
        
        self.framework_type = self.FRAMEWORK_TYPE_DATA_PLATFORM
        self.workspace_url = workspace_url or os.getenv('DATABRICKS_HOST')
        self.access_token = os.getenv('DATABRICKS_TOKEN')
        
        # Unity Catalog specific governance attributes
        self.UNITY_CATALOG_ATTRIBUTES = {
            'catalog_name', 'schema_name', 'table_name', 'metastore_id',
            'workspace_id', 'sql_warehouse_id', 'compute_cluster_id',
            'data_classification', 'retention_policy', 'access_control_list'
        }
        
        # Add Unity Catalog attributes to standard governance attributes
        self.REQUEST_ATTRIBUTES.update(self.UNITY_CATALOG_ATTRIBUTES)
        
        # Initialize telemetry with Unity Catalog context
        self.telemetry = GenOpsTelemetry()
        self.tracer = trace.get_tracer(__name__)

    @contextmanager
    def track_unity_catalog_operation(
        self,
        operation_type: str,
        catalog: Optional[str] = None,
        schema: Optional[str] = None,
        table: Optional[str] = None,
        **governance_attrs
    ):
        """
        Context manager for tracking Unity Catalog operations with governance telemetry.

        Args:
            operation_type: Type of operation (e.g., 'catalog.create', 'table.query')
            catalog: Unity Catalog name
            schema: Schema name within catalog
            table: Table name within schema
            **governance_attrs: Additional governance attributes
        """
        span_name = f"genops.databricks.unity_catalog.{operation_type}"
        
        with self.tracer.start_as_current_span(span_name) as span:
            try:
                # Set standard telemetry attributes
                span.set_attribute("genops.provider", "databricks_unity_catalog")
                span.set_attribute("genops.operation_type", operation_type)
                span.set_attribute("genops.framework_type", self.framework_type)
                
                # Set Unity Catalog specific attributes
                if catalog:
                    span.set_attribute("genops.catalog_name", catalog)
                if schema:
                    span.set_attribute("genops.schema_name", schema)
                if table:
                    span.set_attribute("genops.table_name", table)
                if self.workspace_url:
                    span.set_attribute("genops.workspace_url", self.workspace_url)
                
                # Set governance attributes
                for attr_name, attr_value in governance_attrs.items():
                    if attr_name in self.GOVERNANCE_ATTRIBUTES:
                        span.set_attribute(f"genops.{attr_name}", str(attr_value))
                
                logger.debug(f"Starting Unity Catalog operation: {operation_type}")
                
                yield span
                
                span.set_status(Status(StatusCode.OK))
                logger.debug(f"Completed Unity Catalog operation: {operation_type}")
                
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                logger.error(f"Failed Unity Catalog operation {operation_type}: {e}")
                raise

    def track_catalog_operation(
        self,
        operation: str,
        catalog_name: str,
        **governance_attrs
    ) -> Dict[str, Any]:
        """
        Track Unity Catalog catalog-level operations.

        Args:
            operation: Operation type (create, read, update, delete)
            catalog_name: Name of the catalog
            **governance_attrs: Governance attributes

        Returns:
            Operation metadata and telemetry information
        """
        with self.track_unity_catalog_operation(
            f"catalog.{operation}",
            catalog=catalog_name,
            **governance_attrs
        ) as span:
            metadata = {
                "operation": f"catalog.{operation}",
                "catalog_name": catalog_name,
                "span_id": span.get_span_context().span_id,
                "trace_id": span.get_span_context().trace_id,
            }
            
            # Add cost tracking
            span.set_attribute("genops.cost.operation", f"catalog.{operation}")
            span.set_attribute("genops.cost.resource_type", "catalog")
            
            return metadata

    def track_table_operation(
        self,
        operation: str,
        catalog_name: str,
        schema_name: str,
        table_name: str,
        row_count: Optional[int] = None,
        data_size_bytes: Optional[int] = None,
        **governance_attrs
    ) -> Dict[str, Any]:
        """
        Track Unity Catalog table-level operations.

        Args:
            operation: Operation type (create, read, update, delete, query)
            catalog_name: Name of the catalog
            schema_name: Name of the schema
            table_name: Name of the table
            row_count: Number of rows processed (optional)
            data_size_bytes: Size of data processed in bytes (optional)
            **governance_attrs: Governance attributes

        Returns:
            Operation metadata and telemetry information
        """
        with self.track_unity_catalog_operation(
            f"table.{operation}",
            catalog=catalog_name,
            schema=schema_name,
            table=table_name,
            **governance_attrs
        ) as span:
            metadata = {
                "operation": f"table.{operation}",
                "catalog_name": catalog_name,
                "schema_name": schema_name,
                "table_name": table_name,
                "span_id": span.get_span_context().span_id,
                "trace_id": span.get_span_context().trace_id,
            }
            
            # Add data processing metrics
            if row_count is not None:
                span.set_attribute("genops.data.row_count", row_count)
                metadata["row_count"] = row_count
                
            if data_size_bytes is not None:
                span.set_attribute("genops.data.size_bytes", data_size_bytes)
                metadata["data_size_bytes"] = data_size_bytes
            
            # Add cost tracking
            span.set_attribute("genops.cost.operation", f"table.{operation}")
            span.set_attribute("genops.cost.resource_type", "table")
            
            return metadata

    def track_sql_warehouse_operation(
        self,
        sql_warehouse_id: str,
        query_type: str,
        query_duration_ms: Optional[int] = None,
        compute_units: Optional[float] = None,
        **governance_attrs
    ) -> Dict[str, Any]:
        """
        Track SQL Warehouse operations with cost attribution.

        Args:
            sql_warehouse_id: SQL warehouse identifier
            query_type: Type of query (select, insert, update, etc.)
            query_duration_ms: Query duration in milliseconds
            compute_units: Compute units consumed
            **governance_attrs: Governance attributes

        Returns:
            Operation metadata and telemetry information
        """
        with self.track_unity_catalog_operation(
            f"sql_warehouse.{query_type}",
            **governance_attrs
        ) as span:
            span.set_attribute("genops.sql_warehouse_id", sql_warehouse_id)
            span.set_attribute("genops.query_type", query_type)
            
            metadata = {
                "operation": f"sql_warehouse.{query_type}",
                "sql_warehouse_id": sql_warehouse_id,
                "query_type": query_type,
                "span_id": span.get_span_context().span_id,
                "trace_id": span.get_span_context().trace_id,
            }
            
            # Add performance metrics
            if query_duration_ms is not None:
                span.set_attribute("genops.performance.duration_ms", query_duration_ms)
                metadata["query_duration_ms"] = query_duration_ms
                
            if compute_units is not None:
                span.set_attribute("genops.cost.compute_units", compute_units)
                metadata["compute_units"] = compute_units
            
            # Add cost tracking
            span.set_attribute("genops.cost.resource_type", "sql_warehouse")
            span.set_attribute("genops.cost.operation", f"sql_warehouse.{query_type}")
            
            return metadata

    def setup_governance_attributes(self) -> None:
        """Set up Unity Catalog-specific governance attributes."""
        # Add data governance attributes specific to Unity Catalog
        additional_attrs = {
            'data_owner', 'data_steward', 'security_classification',
            'compliance_tags', 'lineage_upstream', 'lineage_downstream'
        }
        self.GOVERNANCE_ATTRIBUTES.update(additional_attrs)

    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate Databricks Unity Catalog configuration.

        Returns:
            Validation results with configuration status
        """
        validation_result = {
            "valid": True,
            "issues": [],
            "configuration": {}
        }
        
        # Check workspace URL
        if not self.workspace_url:
            validation_result["valid"] = False
            validation_result["issues"].append(
                "DATABRICKS_HOST environment variable not set"
            )
        else:
            validation_result["configuration"]["workspace_url"] = self.workspace_url
        
        # Check access token
        if not self.access_token:
            validation_result["valid"] = False
            validation_result["issues"].append(
                "DATABRICKS_TOKEN environment variable not set"
            )
        else:
            validation_result["configuration"]["access_token"] = "***configured***"
        
        return validation_result


def instrument_databricks_unity_catalog(
    workspace_url: Optional[str] = None,
    **kwargs
) -> GenOpsDatabricksUnityCatalogAdapter:
    """
    Create and configure GenOps instrumentation for Databricks Unity Catalog.

    Args:
        workspace_url: Databricks workspace URL (optional)
        **kwargs: Additional configuration parameters

    Returns:
        Configured Databricks Unity Catalog adapter
    """
    adapter = GenOpsDatabricksUnityCatalogAdapter(
        workspace_url=workspace_url,
        **kwargs
    )
    
    logger.info("GenOps instrumentation enabled for Databricks Unity Catalog")
    return adapter