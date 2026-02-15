#!/usr/bin/env python3
"""
🔍 Compliance Audit Trail Scenario for GenOps AI

This scenario demonstrates how to create comprehensive audit trails for
AI operations that meet enterprise compliance requirements including:

✅ SOX compliance for financial services
✅ GDPR compliance for EU data processing
✅ HIPAA compliance for healthcare applications
✅ SOC 2 compliance for service organizations
✅ Custom compliance frameworks

The audit trail captures all AI operations with evaluation metrics,
policy decisions, and complete traceability for regulatory reporting.

COMPLIANCE CAPABILITIES:
🛡️ Complete AI operation audit logs
📊 Evaluation metrics with thresholds
🔐 Policy enforcement tracking
📋 Compliance scope and classification
⏰ Immutable timestamp records
🏢 Multi-tenant compliance isolation
"""

import json
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import genops
from genops.core.policy import PolicyResult, register_policy
from genops.core.telemetry import GenOpsTelemetry


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""

    SOX = "sox"  # Sarbanes-Oxley Act
    GDPR = "gdpr"  # General Data Protection Regulation
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    SOC2 = "soc2"  # Service Organization Control 2
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard
    CUSTOM = "custom"  # Custom compliance requirements


class DataClassification(Enum):
    """Data classification levels for compliance."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"


class ComplianceAuditor:
    """
    Compliance auditor that tracks and evaluates AI operations for audit trails.

    This class provides comprehensive compliance monitoring including:
    - Policy evaluation and enforcement
    - Data classification tracking
    - Evaluation metrics with thresholds
    - Audit trail generation
    - Compliance reporting
    """

    def __init__(self, compliance_frameworks: list[ComplianceFramework], **config):
        self.frameworks = compliance_frameworks
        self.config = config
        self.telemetry = GenOpsTelemetry()
        self.audit_records = []

        # Set up compliance-specific validation rules
        self._setup_compliance_validation()

        # Register compliance policies
        self._register_compliance_policies()

    def _setup_compliance_validation(self):
        """Set up validation rules for compliance requirements."""

        # Require compliance framework specification
        genops.add_validation_rule(
            genops.ValidationRule(
                name="compliance_framework_required",
                attribute="compliance_framework",
                rule_type="required",
                severity=genops.ValidationSeverity.BLOCK,
                description="Compliance framework must be specified",
                error_message="compliance_framework is required for audit trail",
            )
        )

        # Require data classification
        genops.add_validation_rule(
            genops.ValidationRule(
                name="data_classification_required",
                attribute="data_classification",
                rule_type="required",
                severity=genops.ValidationSeverity.BLOCK,
                description="Data classification required for compliance",
                error_message="data_classification must be specified",
            )
        )

        # Validate compliance frameworks
        allowed_frameworks = {f.value for f in ComplianceFramework}
        genops.add_validation_rule(
            genops.create_enum_rule(
                "compliance_framework",
                allowed_frameworks,
                genops.ValidationSeverity.BLOCK,
            )
        )

        # Validate data classifications
        allowed_classifications = {d.value for d in DataClassification}
        genops.add_validation_rule(
            genops.create_enum_rule(
                "data_classification",
                allowed_classifications,
                genops.ValidationSeverity.BLOCK,
            )
        )

        # Require audit justification for restricted data
        def validate_audit_justification(value):
            context = genops.get_context()
            data_class = context.get("data_classification")
            if data_class in ["restricted", "top_secret"]:
                return value is not None and len(str(value).strip()) > 10
            return True

        genops.add_validation_rule(
            genops.ValidationRule(
                name="audit_justification_required",
                attribute="audit_justification",
                rule_type="custom",
                severity=genops.ValidationSeverity.BLOCK,
                description="Audit justification required for restricted data",
                validator_func=validate_audit_justification,
                error_message="audit_justification (min 10 chars) required for restricted/top_secret data",
            )
        )

    def _register_compliance_policies(self):
        """Register compliance-specific policies."""

        # Data retention policy
        register_policy(
            name="data_retention_compliance",
            enforcement_level=PolicyResult.WARNING,
            conditions={
                "max_retention_days": 2555,  # 7 years for SOX
                "sensitive_data_max_days": 90,
            },
        )

        # Access control policy
        register_policy(
            name="access_control_compliance",
            enforcement_level=PolicyResult.BLOCKED,
            conditions={
                "require_authentication": True,
                "require_authorization": True,
                "max_privilege_level": "standard",
            },
        )

        # Evaluation quality policy
        register_policy(
            name="evaluation_quality_compliance",
            enforcement_level=PolicyResult.WARNING,
            conditions={
                "min_safety_score": 0.85,
                "min_accuracy_score": 0.80,
                "require_human_review": True,
            },
        )

    def start_compliant_operation(
        self,
        operation_name: str,
        compliance_framework: ComplianceFramework,
        data_classification: DataClassification,
        purpose: str,
        legal_basis: Optional[str] = None,
        retention_period: Optional[int] = None,
        audit_justification: Optional[str] = None,
        **additional_context,
    ) -> str:
        """
        Start a compliance-tracked AI operation.

        Args:
            operation_name: Name of the AI operation
            compliance_framework: Applicable compliance framework
            data_classification: Classification of data being processed
            purpose: Business purpose of the operation
            legal_basis: Legal basis for data processing (GDPR requirement)
            retention_period: Data retention period in days
            audit_justification: Justification for audit trail
            **additional_context: Additional context attributes

        Returns:
            operation_id: Unique identifier for this operation
        """

        operation_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()

        # Build compliance context
        compliance_context = {
            "operation_id": operation_id,
            "operation_name": operation_name,
            "compliance_framework": compliance_framework.value,
            "data_classification": data_classification.value,
            "purpose": purpose,
            "timestamp": timestamp.isoformat(),
            "audit_required": True,
            **additional_context,
        }

        # Add optional fields
        if legal_basis:
            compliance_context["legal_basis"] = legal_basis
        if retention_period:
            compliance_context["retention_period_days"] = retention_period
        if audit_justification:
            compliance_context["audit_justification"] = audit_justification

        # Set context with validation
        try:
            genops.enforce_tags(compliance_context)
        except genops.TagValidationError as e:
            raise ValueError(f"Compliance validation failed: {e}") from e

        genops.set_context(**compliance_context)

        # Create initial audit record
        audit_record = {
            "operation_id": operation_id,
            "event_type": "operation_start",
            "timestamp": timestamp.isoformat(),
            "compliance_context": compliance_context,
            "status": "started",
        }

        self.audit_records.append(audit_record)

        print(f"🔍 Started compliant AI operation: {operation_id}")
        print(f"   Framework: {compliance_framework.value.upper()}")
        print(f"   Data Classification: {data_classification.value.upper()}")
        print(f"   Purpose: {purpose}")

        return operation_id

    def evaluate_compliance_metrics(
        self,
        operation_id: str,
        safety_score: float,
        accuracy_score: float,
        bias_score: float,
        privacy_score: float,
        human_reviewed: bool = False,
        reviewer_id: Optional[str] = None,
        **custom_metrics,
    ):
        """
        Record compliance evaluation metrics for an AI operation.

        Args:
            operation_id: Operation identifier
            safety_score: Safety evaluation score (0.0-1.0)
            accuracy_score: Accuracy evaluation score (0.0-1.0)
            bias_score: Bias evaluation score (0.0-1.0, lower is better)
            privacy_score: Privacy protection score (0.0-1.0)
            human_reviewed: Whether operation was human reviewed
            reviewer_id: ID of human reviewer if applicable
            **custom_metrics: Additional custom evaluation metrics
        """

        timestamp = datetime.utcnow()

        # Validate scores
        scores = {
            "safety_score": safety_score,
            "accuracy_score": accuracy_score,
            "bias_score": bias_score,
            "privacy_score": privacy_score,
        }

        for metric_name, score in scores.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"{metric_name} must be between 0.0 and 1.0")

        # Record evaluation metrics in telemetry
        with self.telemetry.trace_operation(
            operation_name="compliance_evaluation", operation_id=operation_id
        ) as span:
            # Record all evaluation metrics
            self.telemetry.record_evaluation(
                span,
                "safety",
                safety_score,
                threshold=0.85,
                passed=safety_score >= 0.85,
            )
            self.telemetry.record_evaluation(
                span,
                "accuracy",
                accuracy_score,
                threshold=0.80,
                passed=accuracy_score >= 0.80,
            )
            self.telemetry.record_evaluation(
                span, "bias", bias_score, threshold=0.2, passed=bias_score <= 0.2
            )
            self.telemetry.record_evaluation(
                span,
                "privacy",
                privacy_score,
                threshold=0.90,
                passed=privacy_score >= 0.90,
            )

            # Record custom metrics
            for metric_name, metric_value in custom_metrics.items():
                self.telemetry.record_evaluation(span, metric_name, metric_value)

            # Record human review status
            span.set_attribute("genops.compliance.human_reviewed", human_reviewed)
            if reviewer_id:
                span.set_attribute("genops.compliance.reviewer_id", reviewer_id)

        # Evaluate compliance policies
        policy_results = self._evaluate_compliance_policies(
            safety_score, accuracy_score, human_reviewed
        )

        # Create evaluation audit record
        audit_record = {
            "operation_id": operation_id,
            "event_type": "evaluation_completed",
            "timestamp": timestamp.isoformat(),
            "evaluation_metrics": {
                **scores,
                **custom_metrics,
                "human_reviewed": human_reviewed,
                "reviewer_id": reviewer_id,
            },
            "policy_results": policy_results,
            "compliance_status": self._determine_compliance_status(
                scores, policy_results
            ),
        }

        self.audit_records.append(audit_record)

        print(f"📊 Recorded compliance evaluation for {operation_id}")
        print(f"   Safety: {safety_score:.3f} | Accuracy: {accuracy_score:.3f}")
        print(f"   Bias: {bias_score:.3f} | Privacy: {privacy_score:.3f}")
        print(f"   Human Reviewed: {human_reviewed}")

    def _evaluate_compliance_policies(
        self, safety_score: float, accuracy_score: float, human_reviewed: bool
    ) -> list[dict[str, Any]]:
        """Evaluate compliance policies and return results."""

        # This would integrate with the policy engine in a real implementation
        policy_results = []

        # Safety threshold policy
        if safety_score < 0.85:
            policy_results.append(
                {
                    "policy_name": "evaluation_quality_compliance",
                    "rule": "min_safety_score",
                    "result": "violation",
                    "threshold": 0.85,
                    "actual": safety_score,
                    "severity": "warning",
                }
            )

        # Accuracy threshold policy
        if accuracy_score < 0.80:
            policy_results.append(
                {
                    "policy_name": "evaluation_quality_compliance",
                    "rule": "min_accuracy_score",
                    "result": "violation",
                    "threshold": 0.80,
                    "actual": accuracy_score,
                    "severity": "warning",
                }
            )

        # Human review requirement for sensitive data
        context = genops.get_context()
        data_class = context.get("data_classification")
        if data_class in ["restricted", "top_secret"] and not human_reviewed:
            policy_results.append(
                {
                    "policy_name": "evaluation_quality_compliance",
                    "rule": "require_human_review",
                    "result": "violation",
                    "reason": f"Human review required for {data_class} data",
                    "severity": "error",
                }
            )

        return policy_results

    def _determine_compliance_status(
        self, scores: dict[str, float], policy_results: list[dict[str, Any]]
    ) -> str:
        """Determine overall compliance status."""

        # Check for blocking violations
        blocking_violations = [
            r for r in policy_results if r.get("severity") == "error"
        ]
        if blocking_violations:
            return "non_compliant"

        # Check for warnings
        warnings = [r for r in policy_results if r.get("severity") == "warning"]
        if warnings:
            return "compliant_with_warnings"

        return "compliant"

    def complete_operation(
        self,
        operation_id: str,
        outcome: str,
        cost: Optional[float] = None,
        tokens_used: Optional[int] = None,
        **completion_metadata,
    ):
        """
        Complete a compliance-tracked AI operation.

        Args:
            operation_id: Operation identifier
            outcome: Operation outcome description
            cost: Total cost of operation
            tokens_used: Total tokens consumed
            **completion_metadata: Additional completion metadata
        """

        timestamp = datetime.utcnow()

        # Record completion in telemetry
        with self.telemetry.trace_operation(
            operation_name="compliance_completion", operation_id=operation_id
        ) as span:
            if cost is not None:
                self.telemetry.record_cost(span, cost=cost, currency="USD")

            span.set_attribute("genops.completion.outcome", outcome)
            if tokens_used:
                span.set_attribute("genops.tokens.total", tokens_used)

        # Create completion audit record
        audit_record = {
            "operation_id": operation_id,
            "event_type": "operation_completed",
            "timestamp": timestamp.isoformat(),
            "outcome": outcome,
            "cost": cost,
            "tokens_used": tokens_used,
            "completion_metadata": completion_metadata,
            "final_context": genops.get_context(),
        }

        self.audit_records.append(audit_record)

        # Clear operation context
        genops.clear_context()

        print(f"✅ Completed compliant AI operation: {operation_id}")
        if cost:
            print(f"   Cost: ${cost:.4f}")
        if tokens_used:
            print(f"   Tokens: {tokens_used:,}")

    def generate_audit_report(
        self,
        operation_ids: Optional[list[str]] = None,
        compliance_framework: Optional[ComplianceFramework] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """
        Generate compliance audit report.

        Args:
            operation_ids: Filter by specific operation IDs
            compliance_framework: Filter by compliance framework
            start_date: Filter operations after this date
            end_date: Filter operations before this date

        Returns:
            Comprehensive audit report dictionary
        """

        # Filter audit records based on criteria
        filtered_records = self.audit_records.copy()

        if operation_ids:
            filtered_records = [
                r for r in filtered_records if r["operation_id"] in operation_ids
            ]

        if compliance_framework:
            filtered_records = [
                r
                for r in filtered_records
                if r.get("compliance_context", {}).get("compliance_framework")
                == compliance_framework.value
            ]

        if start_date or end_date:

            def in_date_range(record):
                record_time = datetime.fromisoformat(record["timestamp"])
                if start_date and record_time < start_date:
                    return False
                if end_date and record_time > end_date:
                    return False
                return True

            filtered_records = [r for r in filtered_records if in_date_range(r)]

        # Aggregate statistics
        operations = {}
        total_cost = 0
        total_tokens = 0
        compliance_violations = []

        for record in filtered_records:
            op_id = record["operation_id"]

            if op_id not in operations:
                operations[op_id] = {
                    "operation_id": op_id,
                    "events": [],
                    "compliance_status": "unknown",
                    "cost": 0,
                    "tokens": 0,
                }

            operations[op_id]["events"].append(record)

            # Extract metrics
            if record["event_type"] == "evaluation_completed":
                operations[op_id]["compliance_status"] = record["compliance_status"]
                if record.get("policy_results"):
                    compliance_violations.extend(record["policy_results"])

            elif record["event_type"] == "operation_completed":
                if record.get("cost"):
                    operations[op_id]["cost"] = record["cost"]
                    total_cost += record["cost"]
                if record.get("tokens_used"):
                    operations[op_id]["tokens"] = record["tokens_used"]
                    total_tokens += record["tokens_used"]

        # Generate report
        report = {
            "report_metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "report_period": {
                    "start": start_date.isoformat() if start_date else None,
                    "end": end_date.isoformat() if end_date else None,
                },
                "filters": {
                    "operation_ids": operation_ids,
                    "compliance_framework": compliance_framework.value
                    if compliance_framework
                    else None,
                },
                "total_operations": len(operations),
                "total_events": len(filtered_records),
            },
            "compliance_summary": {
                "compliant_operations": len(
                    [
                        op
                        for op in operations.values()
                        if op["compliance_status"] == "compliant"
                    ]
                ),
                "non_compliant_operations": len(
                    [
                        op
                        for op in operations.values()
                        if op["compliance_status"] == "non_compliant"
                    ]
                ),
                "operations_with_warnings": len(
                    [
                        op
                        for op in operations.values()
                        if op["compliance_status"] == "compliant_with_warnings"
                    ]
                ),
                "total_violations": len(compliance_violations),
                "violation_types": list({v["rule"] for v in compliance_violations}),
            },
            "cost_analysis": {
                "total_cost": total_cost,
                "average_cost_per_operation": total_cost / max(len(operations), 1),
                "total_tokens": total_tokens,
                "average_tokens_per_operation": total_tokens // max(len(operations), 1),
            },
            "operations": list(operations.values()),
            "compliance_violations": compliance_violations,
            "audit_trail": filtered_records,
        }

        return report


def demonstrate_sox_compliance():
    """Demonstrate SOX compliance audit trail for financial services AI."""

    print("\n💰 SOX COMPLIANCE SCENARIO")
    print("=" * 60)
    print("Scenario: AI-powered financial risk assessment for loan approvals")

    # Initialize compliance auditor for SOX
    auditor = ComplianceAuditor([ComplianceFramework.SOX])

    # Set up global defaults for financial institution
    genops.set_default_attributes(
        team="risk-assessment",
        department="lending",
        business_unit="commercial_banking",
        cost_center="risk_management",
    )

    # Start compliant operation
    operation_id = auditor.start_compliant_operation(
        operation_name="loan_risk_assessment",
        compliance_framework=ComplianceFramework.SOX,
        data_classification=DataClassification.CONFIDENTIAL,
        purpose="Automated loan risk scoring for regulatory compliance",
        legal_basis="Contractual necessity for loan processing",
        retention_period=2555,  # 7 years for SOX
        customer_id="bank_customer_12345",
        loan_application_id="LA-2024-001234",
        loan_amount=250000,
        borrower_type="commercial",
    )

    # Simulate AI risk assessment with compliance tracking
    print("\n🤖 Performing AI-powered risk assessment...")
    time.sleep(0.5)  # Simulate processing

    # Record evaluation metrics
    auditor.evaluate_compliance_metrics(
        operation_id=operation_id,
        safety_score=0.92,  # High safety - good
        accuracy_score=0.88,  # High accuracy - good
        bias_score=0.15,  # Low bias - good
        privacy_score=0.94,  # High privacy - good
        human_reviewed=True,  # Required for SOX
        reviewer_id="risk_analyst_jane_doe",
        # Custom financial metrics
        credit_score_confidence=0.91,
        fraud_detection_score=0.97,
        regulatory_score=0.89,
    )

    # Complete operation
    auditor.complete_operation(
        operation_id=operation_id,
        outcome="Risk assessment completed - APPROVED with conditions",
        cost=0.0234,
        tokens_used=1250,
        risk_rating="Medium",
        approval_conditions=[
            "Collateral requirement: 20%",
            "Personal guarantee required",
        ],
        approver_id="senior_underwriter_john_smith",
    )

    return auditor, operation_id


def demonstrate_gdpr_compliance():
    """Demonstrate GDPR compliance audit trail for EU data processing."""

    print("\n🇪🇺 GDPR COMPLIANCE SCENARIO")
    print("=" * 60)
    print("Scenario: AI-powered customer service with EU personal data")

    # Initialize compliance auditor for GDPR
    auditor = ComplianceAuditor([ComplianceFramework.GDPR])

    # Set up global defaults for EU service
    genops.set_default_attributes(
        team="customer-service",
        data_center="eu-central-1",
        jurisdiction="EU",
        privacy_officer="dpo@company.eu",
    )

    # Start compliant operation
    operation_id = auditor.start_compliant_operation(
        operation_name="customer_support_ai",
        compliance_framework=ComplianceFramework.GDPR,
        data_classification=DataClassification.RESTRICTED,
        purpose="Automated customer support response generation",
        legal_basis="Legitimate interest for customer service improvement",
        retention_period=90,  # Short retention for personal data
        audit_justification="Customer explicitly requested AI assistance for faster support resolution",
        customer_id="eu_customer_67890",
        support_ticket_id="TICKET-EU-98765",
        data_subject_consent=True,
        processing_location="eu-central-1",
    )

    # Simulate AI customer service with GDPR considerations
    print("\n🤖 Processing customer support request with AI...")
    time.sleep(0.3)

    # Record evaluation metrics with GDPR focus
    auditor.evaluate_compliance_metrics(
        operation_id=operation_id,
        safety_score=0.89,  # Good safety
        accuracy_score=0.85,  # Good accuracy
        bias_score=0.12,  # Low bias
        privacy_score=0.96,  # Excellent privacy - critical for GDPR
        human_reviewed=True,  # Required for restricted data
        reviewer_id="privacy_specialist_maria_garcia",
        # GDPR-specific metrics
        data_minimization_score=0.93,
        purpose_limitation_score=0.91,
        consent_validity_score=1.0,
        right_to_explanation_score=0.88,
    )

    # Complete operation
    auditor.complete_operation(
        operation_id=operation_id,
        outcome="Customer support response generated and reviewed",
        cost=0.0156,
        tokens_used=890,
        response_type="product_information",
        personal_data_processed=True,
        data_retention_scheduled=True,
    )

    return auditor, operation_id


def demonstrate_hipaa_compliance():
    """Demonstrate HIPAA compliance audit trail for healthcare AI."""

    print("\n🏥 HIPAA COMPLIANCE SCENARIO")
    print("=" * 60)
    print("Scenario: AI medical diagnosis assistance with PHI protection")

    # Initialize compliance auditor for HIPAA
    auditor = ComplianceAuditor([ComplianceFramework.HIPAA])

    # Set up global defaults for healthcare
    genops.set_default_attributes(
        team="clinical-ai",
        department="radiology",
        facility="regional_medical_center",
        hipaa_covered_entity=True,
    )

    # Start compliant operation
    operation_id = auditor.start_compliant_operation(
        operation_name="medical_image_analysis",
        compliance_framework=ComplianceFramework.HIPAA,
        data_classification=DataClassification.TOP_SECRET,  # PHI is top secret
        purpose="AI-assisted medical diagnosis for patient care",
        legal_basis="Treatment - HIPAA permitted use",
        retention_period=365,  # 1 year medical record retention
        audit_justification="AI diagnostic assistance requested by attending physician for complex case requiring specialized analysis",
        patient_id="PATIENT_789123",
        medical_record_number="MRN-45678901",
        physician_id="DR_SMITH_MD",
        phi_present=True,
        minimum_necessary=True,
    )

    # Simulate medical AI analysis
    print("\n🤖 Performing AI medical image analysis...")
    time.sleep(0.7)

    # Record evaluation metrics with HIPAA focus
    auditor.evaluate_compliance_metrics(
        operation_id=operation_id,
        safety_score=0.95,  # Excellent safety - critical for healthcare
        accuracy_score=0.91,  # High accuracy for medical decisions
        bias_score=0.08,  # Very low bias
        privacy_score=0.98,  # Excellent privacy for PHI
        human_reviewed=True,  # Required for top secret/PHI data
        reviewer_id="radiologist_dr_johnson_md",
        # HIPAA-specific metrics
        phi_protection_score=0.99,
        minimum_necessary_score=0.94,
        audit_log_completeness=1.0,
        diagnostic_confidence=0.87,
    )

    # Complete operation
    auditor.complete_operation(
        operation_id=operation_id,
        outcome="Medical diagnosis assistance completed - findings documented",
        cost=0.0523,
        tokens_used=2100,
        diagnosis_suggestion="Preliminary findings suggest further cardiac evaluation needed",
        physician_review="Attending physician concurred with AI analysis",
        phi_disclosed=False,
    )

    return auditor, operation_id


def generate_comprehensive_audit_report(auditors: list[ComplianceAuditor]):
    """Generate a comprehensive audit report across all compliance frameworks."""

    print("\n📋 COMPREHENSIVE COMPLIANCE AUDIT REPORT")
    print("=" * 60)

    # Combine all audit records
    all_records = []
    for auditor in auditors:
        all_records.extend(auditor.audit_records)

    # Create consolidated auditor for reporting
    master_auditor = ComplianceAuditor([])
    master_auditor.audit_records = all_records

    # Generate comprehensive report
    report = master_auditor.generate_audit_report()

    # Display key findings
    print("📊 AUDIT SUMMARY")
    print(f"   Total Operations: {report['report_metadata']['total_operations']}")
    print(f"   Compliant: {report['compliance_summary']['compliant_operations']}")
    print(
        f"   Non-Compliant: {report['compliance_summary']['non_compliant_operations']}"
    )
    print(
        f"   With Warnings: {report['compliance_summary']['operations_with_warnings']}"
    )
    print(f"   Total Violations: {report['compliance_summary']['total_violations']}")

    print("\n💰 COST ANALYSIS")
    print(f"   Total Cost: ${report['cost_analysis']['total_cost']:.4f}")
    print(
        f"   Avg Cost/Operation: ${report['cost_analysis']['average_cost_per_operation']:.4f}"
    )
    print(f"   Total Tokens: {report['cost_analysis']['total_tokens']:,}")

    if report["compliance_violations"]:
        print("\n⚠️ COMPLIANCE VIOLATIONS")
        for violation in report["compliance_violations"]:
            print(
                f"   • {violation['policy_name']}: {violation['rule']} ({violation['severity']})"
            )

    # Save detailed report
    report_filename = (
        f"compliance_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(report_filename, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Detailed audit report saved to: {report_filename}")

    return report


def main():
    """Run the complete compliance audit trail demonstration."""

    print("🔍 GenOps AI: Compliance Audit Trail Scenarios")
    print("=" * 80)
    print("\nThis demonstration shows how GenOps AI creates comprehensive")
    print("audit trails for AI operations meeting enterprise compliance requirements.")

    auditors = []

    try:
        # Run compliance scenarios
        sox_auditor, sox_op_id = demonstrate_sox_compliance()
        auditors.append(sox_auditor)

        gdpr_auditor, gdpr_op_id = demonstrate_gdpr_compliance()
        auditors.append(gdpr_auditor)

        hipaa_auditor, hipaa_op_id = demonstrate_hipaa_compliance()
        auditors.append(hipaa_auditor)

        # Generate comprehensive report
        generate_comprehensive_audit_report(auditors)

        print("\n🎯 KEY TAKEAWAYS")
        print("=" * 60)
        print("✅ Complete audit trails for SOX, GDPR, and HIPAA compliance")
        print("✅ Evaluation metrics with compliance thresholds")
        print("✅ Policy enforcement and violation tracking")
        print("✅ Data classification and retention management")
        print("✅ Human review requirements for sensitive data")
        print("✅ Cost and token tracking for financial oversight")
        print("✅ Immutable audit records with timestamps")
        print("✅ Comprehensive compliance reporting")

        print("\n📚 COMPLIANCE FRAMEWORKS DEMONSTRATED")
        print("=" * 60)
        print("🏛️ SOX (Sarbanes-Oxley): Financial services risk assessment")
        print("🇪🇺 GDPR (EU Data Protection): Customer service with personal data")
        print("🏥 HIPAA (Healthcare Privacy): Medical diagnosis with PHI")
        print("🔒 Custom frameworks supported for industry-specific requirements")

        print("\n📋 AUDIT TRAIL COMPONENTS")
        print("=" * 60)
        print("• Operation lifecycle tracking (start → evaluate → complete)")
        print("• Compliance framework and data classification")
        print("• Evaluation metrics (safety, accuracy, bias, privacy)")
        print("• Policy evaluations and violation records")
        print("• Human review tracking and approver identification")
        print("• Cost attribution and resource consumption")
        print("• Legal basis and retention period documentation")
        print("• Complete context and metadata capture")

        print("\n🔗 Next Steps for Implementation")
        print("=" * 60)
        print("1. Define your organization's compliance requirements")
        print("2. Set up data classification and retention policies")
        print("3. Configure evaluation metrics and thresholds")
        print("4. Implement human review workflows")
        print("5. Establish audit report generation processes")
        print("6. Train teams on compliance attribution requirements")

    except Exception as e:
        print(f"\n❌ Compliance demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()
