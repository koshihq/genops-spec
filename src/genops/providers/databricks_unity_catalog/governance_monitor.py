"""Governance monitoring for Databricks Unity Catalog operations."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DataLineageMetrics:
    """Metrics for data lineage tracking in Unity Catalog."""
    
    # Source and target information
    source_catalog: Optional[str] = None
    source_schema: Optional[str] = None
    source_table: Optional[str] = None
    target_catalog: Optional[str] = None
    target_schema: Optional[str] = None
    target_table: Optional[str] = None
    
    # Lineage metadata
    lineage_type: str = "unknown"  # read, write, transform, copy
    transformation_logic: Optional[str] = None
    data_classification: Optional[str] = None  # public, internal, confidential, restricted
    
    # Governance attributes
    data_owner: Optional[str] = None
    data_steward: Optional[str] = None
    compliance_tags: List[str] = field(default_factory=list)
    retention_policy: Optional[str] = None
    
    # Operation metadata
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: Optional[str] = None
    workspace_id: Optional[str] = None


@dataclass
class UnityMetastore:
    """Represents Unity Catalog metastore information."""
    
    metastore_id: str
    workspace_ids: Set[str] = field(default_factory=set)
    catalogs: Set[str] = field(default_factory=set)
    data_governance_enabled: bool = True
    
    # Governance policies
    default_classification: str = "internal"
    auto_tagging_enabled: bool = False
    lineage_tracking_enabled: bool = True
    
    def add_catalog(self, catalog_name: str) -> None:
        """Add a catalog to the metastore."""
        self.catalogs.add(catalog_name)
    
    def add_workspace(self, workspace_id: str) -> None:
        """Add a workspace to the metastore."""
        self.workspace_ids.add(workspace_id)


@dataclass
class GovernanceOperationSummary:
    """Summary of governance operations for Unity Catalog."""
    
    # Operation counts by type
    catalog_operations: int = 0
    schema_operations: int = 0
    table_operations: int = 0
    lineage_events: int = 0
    
    # Governance metrics
    data_classifications: Dict[str, int] = field(default_factory=dict)
    compliance_violations: List[str] = field(default_factory=list)
    access_patterns: Dict[str, int] = field(default_factory=dict)
    
    # Policy enforcement
    policies_applied: Set[str] = field(default_factory=set)
    access_grants: int = 0
    access_denials: int = 0
    
    # Data quality metrics
    schema_validation_pass: int = 0
    schema_validation_fail: int = 0
    data_quality_checks: int = 0
    
    def add_lineage_event(self, lineage_metrics: DataLineageMetrics) -> None:
        """Add a data lineage event to the summary."""
        self.lineage_events += 1
        
        if lineage_metrics.data_classification:
            self.data_classifications[lineage_metrics.data_classification] = (
                self.data_classifications.get(lineage_metrics.data_classification, 0) + 1
            )
    
    def add_policy_enforcement(self, policy_name: str, result: str) -> None:
        """Record policy enforcement result."""
        self.policies_applied.add(policy_name)
        
        if result == "granted":
            self.access_grants += 1
        elif result == "denied":
            self.access_denials += 1


class DatabricksGovernanceMonitor:
    """
    Monitors and tracks governance operations for Databricks Unity Catalog.
    
    Provides comprehensive data lineage, compliance, and policy enforcement tracking.
    """
    
    def __init__(self, metastore_id: Optional[str] = None):
        """
        Initialize governance monitor.
        
        Args:
            metastore_id: Unity Catalog metastore ID
        """
        self.metastore_id = metastore_id
        self.metastore: Optional[UnityMetastore] = None
        self.lineage_events: List[DataLineageMetrics] = []
        self.governance_policies: Dict[str, Dict] = {}
        self.operation_summary = GovernanceOperationSummary()
        
        if metastore_id:
            self.metastore = UnityMetastore(metastore_id=metastore_id)

    def track_data_lineage(
        self,
        lineage_type: str,
        source_catalog: Optional[str] = None,
        source_schema: Optional[str] = None,
        source_table: Optional[str] = None,
        target_catalog: Optional[str] = None,
        target_schema: Optional[str] = None,
        target_table: Optional[str] = None,
        **governance_attrs
    ) -> DataLineageMetrics:
        """
        Track data lineage for Unity Catalog operations.

        Args:
            lineage_type: Type of lineage (read, write, transform, copy)
            source_catalog: Source catalog name
            source_schema: Source schema name
            source_table: Source table name
            target_catalog: Target catalog name
            target_schema: Target schema name
            target_table: Target table name
            **governance_attrs: Additional governance attributes

        Returns:
            DataLineageMetrics object
        """
        lineage_metrics = DataLineageMetrics(
            source_catalog=source_catalog,
            source_schema=source_schema,
            source_table=source_table,
            target_catalog=target_catalog,
            target_schema=target_schema,
            target_table=target_table,
            lineage_type=lineage_type,
            data_owner=governance_attrs.get('data_owner'),
            data_steward=governance_attrs.get('data_steward'),
            data_classification=governance_attrs.get('data_classification', 'internal'),
            user_id=governance_attrs.get('user_id'),
            workspace_id=governance_attrs.get('workspace_id'),
        )
        
        # Add compliance tags if provided
        if 'compliance_tags' in governance_attrs:
            lineage_metrics.compliance_tags = governance_attrs['compliance_tags']
        
        self.lineage_events.append(lineage_metrics)
        self.operation_summary.add_lineage_event(lineage_metrics)
        
        logger.debug(
            f"Tracked data lineage: {lineage_type} from "
            f"{source_catalog}.{source_schema}.{source_table} to "
            f"{target_catalog}.{target_schema}.{target_table}"
        )
        
        return lineage_metrics

    def enforce_data_classification_policy(
        self,
        catalog: str,
        schema: str,
        table: str,
        required_classification: str,
        user_clearance: str,
        **governance_attrs
    ) -> Dict[str, any]:
        """
        Enforce data classification access policy.

        Args:
            catalog: Catalog name
            schema: Schema name
            table: Table name
            required_classification: Required data classification level
            user_clearance: User's clearance level
            **governance_attrs: Additional governance attributes

        Returns:
            Policy enforcement result
        """
        # Define classification hierarchy (higher number = more restrictive)
        classification_levels = {
            "public": 1,
            "internal": 2,
            "confidential": 3,
            "restricted": 4
        }
        
        required_level = classification_levels.get(required_classification, 2)
        user_level = classification_levels.get(user_clearance, 1)
        
        access_granted = user_level >= required_level
        
        result = {
            "policy": "data_classification",
            "resource": f"{catalog}.{schema}.{table}",
            "required_classification": required_classification,
            "user_clearance": user_clearance,
            "access_granted": access_granted,
            "enforcement_timestamp": datetime.now(),
        }
        
        # Record policy enforcement
        policy_result = "granted" if access_granted else "denied"
        self.operation_summary.add_policy_enforcement("data_classification", policy_result)
        
        if not access_granted:
            violation = (
                f"Access denied: User clearance '{user_clearance}' insufficient "
                f"for '{required_classification}' data in {catalog}.{schema}.{table}"
            )
            self.operation_summary.compliance_violations.append(violation)
            
            logger.warning(f"Data classification policy violation: {violation}")
        else:
            logger.debug(f"Data classification policy passed: {catalog}.{schema}.{table}")
        
        return result

    def track_compliance_audit(
        self,
        audit_type: str,
        resource_path: str,
        compliance_status: str,
        findings: Optional[List[str]] = None,
        **governance_attrs
    ) -> Dict[str, any]:
        """
        Track compliance audit events.

        Args:
            audit_type: Type of audit (pii_scan, retention_check, access_review)
            resource_path: Path to audited resource
            compliance_status: Compliance status (pass, fail, warning)
            findings: List of audit findings
            **governance_attrs: Additional governance attributes

        Returns:
            Audit tracking result
        """
        audit_result = {
            "audit_type": audit_type,
            "resource_path": resource_path,
            "compliance_status": compliance_status,
            "findings": findings or [],
            "timestamp": datetime.now(),
            "auditor": governance_attrs.get('user_id'),
            "workspace_id": governance_attrs.get('workspace_id'),
        }
        
        # Track compliance violations
        if compliance_status == "fail":
            violation = f"{audit_type} failed for {resource_path}: {findings}"
            self.operation_summary.compliance_violations.append(violation)
            logger.warning(f"Compliance audit failed: {violation}")
        
        # Record policy enforcement
        self.operation_summary.add_policy_enforcement(audit_type, compliance_status)
        
        logger.debug(f"Compliance audit tracked: {audit_type} on {resource_path} - {compliance_status}")
        
        return audit_result

    def validate_schema_compliance(
        self,
        catalog: str,
        schema: str,
        table: str,
        schema_definition: Dict,
        compliance_rules: List[str],
        **governance_attrs
    ) -> Dict[str, any]:
        """
        Validate schema compliance against governance rules.

        Args:
            catalog: Catalog name
            schema: Schema name
            table: Table name
            schema_definition: Schema definition to validate
            compliance_rules: List of compliance rules to check
            **governance_attrs: Additional governance attributes

        Returns:
            Schema validation result
        """
        validation_result = {
            "resource": f"{catalog}.{schema}.{table}",
            "compliance_status": "pass",
            "violations": [],
            "warnings": [],
            "validated_rules": compliance_rules,
            "timestamp": datetime.now(),
        }
        
        # Example compliance checks
        for rule in compliance_rules:
            if rule == "pii_detection":
                # Check for PII column names
                pii_patterns = ['ssn', 'email', 'phone', 'credit_card']
                for column in schema_definition.get('columns', []):
                    column_name = column.get('name', '').lower()
                    if any(pattern in column_name for pattern in pii_patterns):
                        if not column.get('encrypted', False):
                            violation = f"PII column '{column_name}' not encrypted"
                            validation_result["violations"].append(violation)
                            validation_result["compliance_status"] = "fail"
            
            elif rule == "required_columns":
                # Check for required audit columns
                required_cols = ['created_at', 'updated_at', 'created_by']
                schema_cols = [col.get('name', '') for col in schema_definition.get('columns', [])]
                missing_cols = [col for col in required_cols if col not in schema_cols]
                if missing_cols:
                    warning = f"Missing recommended audit columns: {missing_cols}"
                    validation_result["warnings"].append(warning)
        
        # Update operation summary
        if validation_result["compliance_status"] == "pass":
            self.operation_summary.schema_validation_pass += 1
        else:
            self.operation_summary.schema_validation_fail += 1
            # Add violations to summary
            self.operation_summary.compliance_violations.extend(validation_result["violations"])
        
        logger.debug(f"Schema compliance validated: {catalog}.{schema}.{table} - {validation_result['compliance_status']}")
        
        return validation_result

    def get_governance_summary(self) -> GovernanceOperationSummary:
        """Get comprehensive governance operation summary."""
        return self.operation_summary

    def get_lineage_graph(self, catalog: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Generate data lineage graph for visualization.

        Args:
            catalog: Optional catalog filter

        Returns:
            Dictionary representing lineage relationships
        """
        lineage_graph = {}
        
        for lineage in self.lineage_events:
            if catalog and lineage.source_catalog != catalog and lineage.target_catalog != catalog:
                continue
            
            # Build source and target identifiers
            if lineage.source_catalog and lineage.source_schema and lineage.source_table:
                source = f"{lineage.source_catalog}.{lineage.source_schema}.{lineage.source_table}"
            else:
                source = "external"
            
            if lineage.target_catalog and lineage.target_schema and lineage.target_table:
                target = f"{lineage.target_catalog}.{lineage.target_schema}.{lineage.target_table}"
            else:
                target = "external"
            
            if source not in lineage_graph:
                lineage_graph[source] = []
            lineage_graph[source].append(target)
        
        return lineage_graph


# Global governance monitor instance
_governance_monitor: Optional[DatabricksGovernanceMonitor] = None


def get_governance_monitor(metastore_id: Optional[str] = None) -> DatabricksGovernanceMonitor:
    """Get or create global governance monitor instance."""
    global _governance_monitor
    if _governance_monitor is None:
        _governance_monitor = DatabricksGovernanceMonitor(metastore_id=metastore_id)
    return _governance_monitor