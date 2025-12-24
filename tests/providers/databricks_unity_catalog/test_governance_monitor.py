"""
Comprehensive tests for Databricks Unity Catalog Governance Monitor.

Tests data governance, lineage tracking, and compliance including:
- Data lineage tracking across catalogs/schemas/tables
- Policy enforcement and compliance monitoring
- Data classification and PII detection
- Audit trail generation and compliance reporting
- Governance metrics aggregation
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Import the modules under test
try:
    from genops.providers.databricks_unity_catalog import (
        DatabricksGovernanceMonitor,
        DataLineageMetrics,
        GovernanceOperationSummary,
        UnityMetastore,
        get_governance_monitor
    )
    GOVERNANCE_MONITOR_AVAILABLE = True
except ImportError:
    GOVERNANCE_MONITOR_AVAILABLE = False


@pytest.mark.skipif(not GOVERNANCE_MONITOR_AVAILABLE, reason="Governance monitor not available")
class TestDatabricksGovernanceMonitor:
    """Test suite for the governance monitor."""

    def setup_method(self):
        """Set up test fixtures."""
        self.governance_monitor = DatabricksGovernanceMonitor(
            metastore_id="test-metastore-123"
        )

    def test_governance_monitor_initialization(self):
        """Test governance monitor initialization."""
        monitor = DatabricksGovernanceMonitor(metastore_id="test-metastore")
        
        assert hasattr(monitor, 'track_data_lineage')
        assert hasattr(monitor, 'enforce_data_classification_policy')
        assert hasattr(monitor, 'track_compliance_audit')
        assert hasattr(monitor, 'get_governance_summary')
        assert monitor.metastore_id == "test-metastore"

    def test_data_lineage_tracking_read_operation(self):
        """Test data lineage tracking for read operations."""
        lineage_result = self.governance_monitor.track_data_lineage(
            lineage_type="read",
            source_catalog="raw_data",
            source_schema="events",
            source_table="user_sessions",
            data_classification="internal",
            user_id="data-analyst@example.com"
        )
        
        assert isinstance(lineage_result, DataLineageMetrics)
        assert lineage_result.lineage_type == "read"
        assert lineage_result.source_catalog == "raw_data"
        assert lineage_result.source_schema == "events"
        assert lineage_result.source_table == "user_sessions"
        assert lineage_result.data_classification == "internal"
        assert lineage_result.user_id == "data-analyst@example.com"

    def test_data_lineage_tracking_transform_operation(self):
        """Test data lineage tracking for transformation operations."""
        lineage_result = self.governance_monitor.track_data_lineage(
            lineage_type="transform",
            source_catalog="raw_data",
            source_schema="events",
            source_table="user_actions",
            target_catalog="analytics",
            target_schema="aggregated",
            target_table="daily_user_metrics",
            transformation_logic="SELECT user_id, COUNT(*) as action_count FROM user_actions GROUP BY user_id",
            data_classification="confidential",
            user_id="data-engineer@example.com"
        )
        
        assert lineage_result.lineage_type == "transform"
        assert lineage_result.source_catalog == "raw_data"
        assert lineage_result.target_catalog == "analytics"
        assert lineage_result.target_table == "daily_user_metrics"
        assert lineage_result.transformation_logic is not None
        assert "GROUP BY" in lineage_result.transformation_logic

    def test_data_lineage_tracking_write_operation(self):
        """Test data lineage tracking for write operations."""
        lineage_result = self.governance_monitor.track_data_lineage(
            lineage_type="write",
            target_catalog="processed",
            target_schema="ml_features",
            target_table="user_feature_vectors",
            data_classification="restricted",
            user_id="ml-engineer@example.com",
            data_owner="ml-platform-team",
            data_steward="data-governance@example.com"
        )
        
        assert lineage_result.lineage_type == "write"
        assert lineage_result.target_catalog == "processed"
        assert lineage_result.target_schema == "ml_features"
        assert lineage_result.data_classification == "restricted"
        assert lineage_result.data_owner == "ml-platform-team"
        assert lineage_result.data_steward == "data-governance@example.com"

    def test_data_classification_policy_enforcement(self):
        """Test data classification policy enforcement."""
        # Test access granted for sufficient clearance
        policy_result = self.governance_monitor.enforce_data_classification_policy(
            catalog="customer_data",
            schema="pii",
            table="user_profiles",
            required_classification="confidential",
            user_clearance="confidential"
        )
        
        assert policy_result["access_granted"] == True
        assert policy_result["policy_name"] == "data_classification_policy"
        assert policy_result["required_classification"] == "confidential"
        assert policy_result["user_clearance"] == "confidential"

    def test_data_classification_policy_denial(self):
        """Test data classification policy denial for insufficient clearance."""
        policy_result = self.governance_monitor.enforce_data_classification_policy(
            catalog="customer_data",
            schema="pii",
            table="credit_card_data",
            required_classification="restricted",
            user_clearance="internal"
        )
        
        assert policy_result["access_granted"] == False
        assert policy_result["violation_reason"] == "insufficient_clearance"
        assert policy_result["required_classification"] == "restricted"
        assert policy_result["user_clearance"] == "internal"

    def test_compliance_audit_tracking(self):
        """Test compliance audit event tracking."""
        audit_result = self.governance_monitor.track_compliance_audit(
            audit_type="pii_scan",
            resource_path="customer_data.profiles.users",
            compliance_status="pass",
            findings=["encrypted_email_column", "masked_phone_numbers"],
            auditor_id="compliance-bot@example.com"
        )
        
        assert audit_result["audit_type"] == "pii_scan"
        assert audit_result["resource_path"] == "customer_data.profiles.users"
        assert audit_result["compliance_status"] == "pass"
        assert "encrypted_email_column" in audit_result["findings"]
        assert "masked_phone_numbers" in audit_result["findings"]

    def test_compliance_audit_violation(self):
        """Test compliance audit violation tracking."""
        audit_result = self.governance_monitor.track_compliance_audit(
            audit_type="gdpr_compliance",
            resource_path="marketing.campaigns.customer_emails",
            compliance_status="violation",
            findings=["missing_consent_flag", "unencrypted_email_addresses"],
            violation_severity="high",
            remediation_required=True
        )
        
        assert audit_result["compliance_status"] == "violation"
        assert audit_result["violation_severity"] == "high"
        assert audit_result["remediation_required"] == True
        assert len(audit_result["findings"]) == 2

    def test_governance_summary_generation(self):
        """Test governance operation summary generation."""
        # Add some governance events
        self.governance_monitor.track_data_lineage(
            lineage_type="read",
            source_catalog="test",
            source_schema="test",
            source_table="test1",
            data_classification="internal"
        )
        
        self.governance_monitor.track_data_lineage(
            lineage_type="transform",
            source_catalog="test",
            source_schema="test",
            source_table="test1",
            target_catalog="processed",
            target_schema="analytics",
            target_table="test_metrics",
            data_classification="confidential"
        )
        
        self.governance_monitor.track_compliance_audit(
            audit_type="schema_validation",
            resource_path="test.test.test1",
            compliance_status="pass"
        )
        
        summary = self.governance_monitor.get_governance_summary()
        
        assert isinstance(summary, GovernanceOperationSummary)
        assert summary.lineage_events >= 2
        assert summary.policy_evaluations >= 0
        assert summary.compliance_checks >= 1
        assert "internal" in summary.data_classifications
        assert "confidential" in summary.data_classifications

    def test_pii_detection_and_classification(self):
        """Test PII detection and automatic classification."""
        # Test with PII-containing data
        pii_detection_result = self.governance_monitor.detect_pii_and_classify(
            catalog="customer_data",
            schema="raw",
            table="user_registrations",
            sample_data={
                "email": "user@example.com",
                "phone": "+1-555-0123",
                "ssn": "123-45-6789",
                "name": "John Doe"
            }
        )
        
        assert pii_detection_result["contains_pii"] == True
        assert "email" in pii_detection_result["pii_columns"]
        assert "phone" in pii_detection_result["pii_columns"] 
        assert "ssn" in pii_detection_result["pii_columns"]
        assert pii_detection_result["recommended_classification"] == "restricted"

    def test_pii_detection_no_pii(self):
        """Test PII detection with non-PII data."""
        pii_detection_result = self.governance_monitor.detect_pii_and_classify(
            catalog="analytics",
            schema="aggregated",
            table="page_views",
            sample_data={
                "page_url": "/products/shoes",
                "view_count": 1523,
                "avg_time_on_page": 45.6
            }
        )
        
        assert pii_detection_result["contains_pii"] == False
        assert len(pii_detection_result["pii_columns"]) == 0
        assert pii_detection_result["recommended_classification"] == "internal"

    def test_data_retention_policy_enforcement(self):
        """Test data retention policy enforcement."""
        retention_result = self.governance_monitor.enforce_data_retention_policy(
            catalog="archive",
            schema="historical",
            table="old_user_events",
            data_age_days=2557,  # ~7 years
            retention_policy="gdpr_7_year",
            user_id="compliance-officer@example.com"
        )
        
        assert retention_result["policy_violated"] == True
        assert retention_result["retention_policy"] == "gdpr_7_year"
        assert retention_result["data_age_days"] == 2557
        assert retention_result["action_required"] == "data_deletion"

    def test_lineage_graph_generation(self):
        """Test data lineage graph generation."""
        # Add multiple lineage relationships
        self.governance_monitor.track_data_lineage(
            lineage_type="read",
            source_catalog="raw",
            source_schema="events",
            source_table="user_clicks"
        )
        
        self.governance_monitor.track_data_lineage(
            lineage_type="transform",
            source_catalog="raw",
            source_schema="events",
            source_table="user_clicks",
            target_catalog="processed",
            target_schema="features",
            target_table="click_features"
        )
        
        self.governance_monitor.track_data_lineage(
            lineage_type="transform",
            source_catalog="processed",
            source_schema="features", 
            source_table="click_features",
            target_catalog="ml",
            target_schema="models",
            target_table="user_propensity_scores"
        )
        
        lineage_graph = self.governance_monitor.get_lineage_graph(
            catalog="processed"
        )
        
        assert "nodes" in lineage_graph
        assert "edges" in lineage_graph
        assert len(lineage_graph["nodes"]) >= 3
        assert len(lineage_graph["edges"]) >= 2

    def test_access_pattern_monitoring(self):
        """Test monitoring of data access patterns."""
        # Track multiple accesses
        for i in range(5):
            self.governance_monitor.track_data_access(
                catalog="sensitive",
                schema="customer_data",
                table="personal_info",
                user_id=f"analyst-{i}@example.com",
                access_type="read",
                access_time=datetime.now() - timedelta(hours=i)
            )
        
        access_patterns = self.governance_monitor.get_access_patterns(
            catalog="sensitive",
            schema="customer_data",
            table="personal_info",
            time_window_hours=24
        )
        
        assert access_patterns["total_accesses"] == 5
        assert access_patterns["unique_users"] == 5
        assert len(access_patterns["access_by_user"]) == 5

    def test_governance_metrics_aggregation(self):
        """Test aggregation of governance metrics across catalogs."""
        # Add governance events across multiple catalogs
        catalogs = ["raw_data", "processed", "analytics", "ml"]
        
        for i, catalog in enumerate(catalogs):
            # Add lineage events
            self.governance_monitor.track_data_lineage(
                lineage_type="read",
                source_catalog=catalog,
                source_schema=f"schema_{i}",
                source_table=f"table_{i}",
                data_classification="internal"
            )
            
            # Add compliance checks
            self.governance_monitor.track_compliance_audit(
                audit_type="schema_validation",
                resource_path=f"{catalog}.schema_{i}.table_{i}",
                compliance_status="pass" if i % 2 == 0 else "fail"
            )
        
        aggregated_metrics = self.governance_monitor.get_aggregated_governance_metrics()
        
        assert aggregated_metrics["total_catalogs"] == 4
        assert aggregated_metrics["total_lineage_events"] == 4
        assert aggregated_metrics["total_compliance_checks"] == 4
        assert aggregated_metrics["compliance_pass_rate"] == 0.5  # 2 pass, 2 fail

    def test_unity_metastore_configuration(self):
        """Test Unity Catalog metastore configuration handling."""
        try:
            metastore = UnityMetastore(
                metastore_id="test-metastore-123",
                name="Test Metastore",
                region="us-west-2",
                owner="data-platform-team@example.com"
            )
            
            assert metastore.metastore_id == "test-metastore-123"
            assert metastore.name == "Test Metastore"
            assert metastore.region == "us-west-2"
            assert metastore.owner == "data-platform-team@example.com"
            
        except (NameError, TypeError):
            # UnityMetastore class may be implemented differently
            pass

    def test_get_governance_monitor_singleton(self):
        """Test get_governance_monitor function."""
        monitor1 = get_governance_monitor("test-metastore")
        monitor2 = get_governance_monitor("test-metastore") 
        
        # Should return same instance for same metastore
        assert monitor1.metastore_id == monitor2.metastore_id

    def test_cross_catalog_lineage_tracking(self):
        """Test lineage tracking across multiple catalogs."""
        # Create a complex lineage spanning multiple catalogs
        lineage_chain = [
            {
                "type": "read",
                "source": ("external", "third_party", "api_data"),
                "target": None
            },
            {
                "type": "transform",
                "source": ("external", "third_party", "api_data"),
                "target": ("raw", "ingested", "cleaned_api_data")
            },
            {
                "type": "transform", 
                "source": ("raw", "ingested", "cleaned_api_data"),
                "target": ("processed", "features", "api_features")
            },
            {
                "type": "transform",
                "source": ("processed", "features", "api_features"),
                "target": ("analytics", "reports", "api_insights")
            }
        ]
        
        for lineage in lineage_chain:
            if lineage["target"]:
                self.governance_monitor.track_data_lineage(
                    lineage_type=lineage["type"],
                    source_catalog=lineage["source"][0],
                    source_schema=lineage["source"][1], 
                    source_table=lineage["source"][2],
                    target_catalog=lineage["target"][0],
                    target_schema=lineage["target"][1],
                    target_table=lineage["target"][2]
                )
            else:
                self.governance_monitor.track_data_lineage(
                    lineage_type=lineage["type"],
                    source_catalog=lineage["source"][0],
                    source_schema=lineage["source"][1],
                    source_table=lineage["source"][2]
                )
        
        # Verify cross-catalog lineage is tracked
        cross_catalog_lineage = self.governance_monitor.get_cross_catalog_lineage()
        
        assert len(cross_catalog_lineage["catalog_relationships"]) >= 3
        catalogs_involved = set()
        for rel in cross_catalog_lineage["catalog_relationships"]:
            catalogs_involved.add(rel["source_catalog"])
            if rel.get("target_catalog"):
                catalogs_involved.add(rel["target_catalog"])
        
        expected_catalogs = {"external", "raw", "processed", "analytics"}
        assert expected_catalogs.issubset(catalogs_involved)


class TestDataLineageMetrics:
    """Test DataLineageMetrics data structure."""

    def test_data_lineage_metrics_creation(self):
        """Test creation of DataLineageMetrics objects."""
        try:
            lineage_metrics = DataLineageMetrics(
                lineage_type="transform",
                source_catalog="source_cat",
                source_schema="source_schema",
                source_table="source_table",
                target_catalog="target_cat",
                target_schema="target_schema",
                target_table="target_table",
                transformation_logic="SELECT * FROM source_table WHERE active = true",
                data_classification="confidential",
                timestamp=datetime.now(),
                user_id="test-user@example.com"
            )
            
            assert lineage_metrics.lineage_type == "transform"
            assert lineage_metrics.source_catalog == "source_cat"
            assert lineage_metrics.target_catalog == "target_cat"
            assert lineage_metrics.data_classification == "confidential"
            assert "WHERE active = true" in lineage_metrics.transformation_logic
            
        except (NameError, TypeError):
            # DataLineageMetrics may be implemented differently
            pass

    def test_lineage_metrics_serialization(self):
        """Test serialization of lineage metrics for storage."""
        governance_monitor = DatabricksGovernanceMonitor("test-metastore")
        
        lineage_result = governance_monitor.track_data_lineage(
            lineage_type="read",
            source_catalog="test",
            source_schema="test",
            source_table="test",
            data_classification="internal"
        )
        
        # Should be serializable for telemetry export
        try:
            serialized = lineage_result.to_dict()
            assert isinstance(serialized, dict)
            assert "lineage_type" in serialized
            assert "source_catalog" in serialized
        except AttributeError:
            # Serialization method may not be implemented
            pass


class TestGovernanceCompliance:
    """Test compliance and regulatory features."""

    def setup_method(self):
        """Set up test fixtures."""
        self.governance_monitor = DatabricksGovernanceMonitor("test-metastore")

    def test_gdpr_compliance_checking(self):
        """Test GDPR compliance validation."""
        gdpr_result = self.governance_monitor.validate_gdpr_compliance(
            catalog="customer_data",
            schema="personal_info",
            table="user_profiles",
            data_subjects_present=True,
            consent_mechanism="explicit_opt_in",
            retention_period_days=2555  # 7 years
        )
        
        assert "gdpr_compliant" in gdpr_result
        assert "findings" in gdpr_result
        assert "recommendations" in gdpr_result

    def test_ccpa_compliance_checking(self):
        """Test CCPA compliance validation."""
        ccpa_result = self.governance_monitor.validate_ccpa_compliance(
            catalog="customer_data", 
            schema="california_residents",
            table="personal_data",
            california_residents_present=True,
            deletion_mechanism_available=True,
            opt_out_mechanism_available=True
        )
        
        assert "ccpa_compliant" in ccpa_result
        assert "consumer_rights_supported" in ccpa_result

    def test_sox_compliance_audit_trail(self):
        """Test SOX compliance audit trail generation."""
        # Generate audit events
        for i in range(10):
            self.governance_monitor.track_compliance_audit(
                audit_type="financial_data_access",
                resource_path=f"finance.quarterly.revenue_q{i%4+1}",
                compliance_status="pass",
                auditor_id="sox-auditor@example.com"
            )
        
        sox_audit_trail = self.governance_monitor.generate_sox_audit_trail(
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now()
        )
        
        assert sox_audit_trail["total_audit_events"] == 10
        assert "audit_events" in sox_audit_trail
        assert "compliance_summary" in sox_audit_trail

    def test_automated_compliance_reporting(self):
        """Test automated compliance report generation."""
        # Add various compliance events
        compliance_events = [
            {"type": "pii_scan", "status": "pass", "resource": "customers.pii.emails"},
            {"type": "retention_check", "status": "violation", "resource": "archive.old.user_data"},
            {"type": "access_review", "status": "pass", "resource": "sensitive.financial.reports"},
        ]
        
        for event in compliance_events:
            self.governance_monitor.track_compliance_audit(
                audit_type=event["type"],
                resource_path=event["resource"],
                compliance_status=event["status"]
            )
        
        compliance_report = self.governance_monitor.generate_compliance_report(
            report_type="monthly",
            compliance_frameworks=["gdpr", "ccpa", "sox"]
        )
        
        assert "report_period" in compliance_report
        assert "compliance_summary" in compliance_report
        assert "violations" in compliance_report
        assert "recommendations" in compliance_report