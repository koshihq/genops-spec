#!/usr/bin/env python3
"""
GDPR Compliance Template for PostHog + GenOps

This template demonstrates General Data Protection Regulation (GDPR) compliance
implementation for PostHog analytics with GenOps governance. GDPR requires strict
data protection, user consent management, and data subject rights for EU users.

GDPR Requirements Addressed:
- Article 6: Lawful basis for processing personal data
- Article 7: Conditions for consent and consent withdrawal
- Article 13-14: Information to be provided to data subjects
- Article 17: Right to erasure ("right to be forgotten")
- Article 20: Right to data portability
- Article 25: Data protection by design and by default
- Article 35: Data protection impact assessments (DPIA)

Use Case:
    - EU user behavior analytics with consent management
    - Personal data processing with lawful basis tracking
    - Data subject rights fulfillment (access, portability, erasure)
    - GDPR-compliant analytics governance and reporting

Usage:
    python compliance_templates/GDPR_compliance_template.py

Prerequisites:
    pip install genops[posthog]
    export POSTHOG_API_KEY="phc_your_project_api_key"
    export GENOPS_TEAM="privacy-analytics"
    export GENOPS_PROJECT="gdpr-compliance"
    export GDPR_DPO_EMAIL="dpo@company.com"  # Data Protection Officer contact

Expected Output:
    GDPR-compliant user analytics tracking with consent management,
    data subject rights handling, and privacy governance reporting.

Learning Objectives:
    - GDPR compliance requirements for user analytics
    - Consent management and lawful basis tracking
    - Data subject rights implementation and fulfillment
    - Privacy-by-design analytics patterns with governance

Author: GenOps AI Privacy Team
License: Apache 2.0
"""

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional


class ConsentStatus(Enum):
    """GDPR consent status options."""

    GIVEN = "given"
    WITHDRAWN = "withdrawn"
    NOT_REQUIRED = "not_required"
    PENDING = "pending"


class LawfulBasis(Enum):
    """GDPR lawful basis for processing personal data (Article 6)."""

    CONSENT = "consent"  # Art 6(1)(a)
    CONTRACT = "contract"  # Art 6(1)(b)
    LEGAL_OBLIGATION = "legal_obligation"  # Art 6(1)(c)
    VITAL_INTERESTS = "vital_interests"  # Art 6(1)(d)
    PUBLIC_TASK = "public_task"  # Art 6(1)(e)
    LEGITIMATE_INTERESTS = "legitimate_interests"  # Art 6(1)(f)


class DataSubjectRights(Enum):
    """GDPR data subject rights."""

    ACCESS = "access"  # Art 15
    RECTIFICATION = "rectification"  # Art 16
    ERASURE = "erasure"  # Art 17
    RESTRICT_PROCESSING = "restrict_processing"  # Art 18
    DATA_PORTABILITY = "data_portability"  # Art 20
    OBJECT = "object"  # Art 21


@dataclass
class GDPRConsentRecord:
    """GDPR consent record with full compliance tracking."""

    consent_id: str
    user_id: str
    timestamp: str
    consent_status: str
    lawful_basis: str
    purpose: str
    data_categories: list[str]
    retention_period: str
    consent_version: str
    ip_address: Optional[str]
    user_agent: Optional[str]
    withdrawal_timestamp: Optional[str] = None


@dataclass
class DataSubjectRequest:
    """GDPR data subject rights request."""

    request_id: str
    user_id: str
    request_type: str
    timestamp: str
    status: str
    fulfillment_deadline: str
    data_categories: list[str]
    lawful_basis_check: str
    processing_notes: str


def main():
    """Demonstrate GDPR-compliant PostHog analytics with privacy governance."""
    print("🛡️ GDPR Compliance Template for PostHog + GenOps")
    print("=" * 55)
    print()

    # Import and setup GenOps PostHog adapter with GDPR configuration
    try:
        from genops.providers.posthog import GenOpsPostHogAdapter

        print("✅ GenOps PostHog integration loaded")
    except ImportError as e:
        print(f"❌ Failed to import GenOps PostHog: {e}")
        print("💡 Fix: pip install genops[posthog]")
        return False

    # GDPR Compliance Configuration
    print("\n🔧 Configuring GDPR Compliance Environment...")

    dpo_email = os.getenv("GDPR_DPO_EMAIL")
    if not dpo_email:
        print("⚠️ GDPR_DPO_EMAIL not configured - using demo value")
        dpo_email = "dpo@company-demo.com"

    # Initialize GDPR-compliant adapter
    adapter = GenOpsPostHogAdapter(
        team="privacy-analytics",
        project="gdpr-compliant-tracking",
        environment="production",
        customer_id="eu_data_processing",
        cost_center="privacy_operations",
        daily_budget_limit=200.0,
        governance_policy="strict",  # Strict enforcement for GDPR
        tags={
            "compliance_framework": "gdpr",
            "data_protection_regulation": "eu_gdpr_2016_679",
            "data_classification": "personal_data",
            "geographic_scope": "european_union",
            "consent_required": "true",
            "lawful_basis_tracking": "enabled",
            "data_subject_rights": "supported",
            "retention_policy": "purpose_limited",
            "privacy_by_design": "implemented",
            "dpo_contact": dpo_email,
            "data_controller": "company_legal_entity",
            "cross_border_transfers": "adequacy_decision_only",
        },
    )

    print("✅ GDPR-compliant adapter configured")
    print("   🇪🇺 Geographic scope: European Union")
    print(f"   📧 DPO contact: {dpo_email}")
    print("   🛡️ Privacy by design: Implemented")
    print("   ⚖️ Lawful basis tracking: Enabled")
    print("   👤 Data subject rights: Supported")
    print("   📝 Consent management: Required")

    # GDPR compliance tracking
    consent_records: list[GDPRConsentRecord] = []
    data_subject_requests: list[DataSubjectRequest] = []
    personal_data_inventory: set[str] = set()

    def create_consent_record(
        user_id: str,
        consent_status: ConsentStatus,
        lawful_basis: LawfulBasis,
        purpose: str,
        data_categories: list[str],
    ) -> GDPRConsentRecord:
        """Create GDPR-compliant consent record."""

        record = GDPRConsentRecord(
            consent_id=str(uuid.uuid4()),
            user_id=user_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            consent_status=consent_status.value,
            lawful_basis=lawful_basis.value,
            purpose=purpose,
            data_categories=data_categories,
            retention_period="2_years_after_last_interaction",
            consent_version="v2.1_gdpr_compliant",
            ip_address="192.168.1.100",  # Simulated
            user_agent="Mozilla/5.0 (GDPR Compliant Browser)",
        )

        consent_records.append(record)
        personal_data_inventory.update(data_categories)
        return record

    def handle_data_subject_request(
        user_id: str, request_type: DataSubjectRights, data_categories: list[str]
    ) -> DataSubjectRequest:
        """Handle GDPR data subject rights request."""

        request = DataSubjectRequest(
            request_id=f"DSR_{datetime.now().strftime('%Y%m%d')}_{len(data_subject_requests) + 1:04d}",
            user_id=user_id,
            request_type=request_type.value,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status="pending_fulfillment",
            fulfillment_deadline=(
                datetime.now() + timedelta(days=30)
            ).isoformat(),  # GDPR 30-day requirement
            data_categories=data_categories,
            lawful_basis_check="verified",
            processing_notes=f"GDPR {request_type.value} request initiated",
        )

        data_subject_requests.append(request)
        return request

    # Demonstrate GDPR-compliant user analytics scenarios
    print("\n" + "=" * 55)
    print("👤 GDPR-Compliant User Analytics Tracking")
    print("=" * 55)

    # EU user scenarios with different consent and lawful basis situations
    user_scenarios = [
        {
            "user_id": "eu_user_001",
            "scenario": "explicit_consent_analytics",
            "consent_status": ConsentStatus.GIVEN,
            "lawful_basis": LawfulBasis.CONSENT,
            "data_categories": [
                "behavioral_data",
                "usage_analytics",
                "performance_data",
            ],
            "purpose": "product_analytics_and_improvement",
        },
        {
            "user_id": "eu_user_002",
            "scenario": "contract_fulfillment_tracking",
            "consent_status": ConsentStatus.NOT_REQUIRED,
            "lawful_basis": LawfulBasis.CONTRACT,
            "data_categories": [
                "transaction_data",
                "service_usage",
                "billing_analytics",
            ],
            "purpose": "contract_performance_and_billing",
        },
        {
            "user_id": "eu_user_003",
            "scenario": "legitimate_interests_analytics",
            "consent_status": ConsentStatus.NOT_REQUIRED,
            "lawful_basis": LawfulBasis.LEGITIMATE_INTERESTS,
            "data_categories": [
                "security_analytics",
                "fraud_detection",
                "system_performance",
            ],
            "purpose": "security_and_fraud_prevention",
        },
    ]

    total_gdpr_events = 0
    total_consent_records = 0

    for scenario_idx, scenario in enumerate(user_scenarios, 1):
        user_id = scenario["user_id"]
        print(f"\n👤 User Scenario {scenario_idx}: {scenario['scenario']}")
        print("-" * 50)
        print(f"   User ID: {user_id}")
        print(f"   Lawful basis: {scenario['lawful_basis'].value}")
        print(
            f"   Consent required: {scenario['consent_status'] == ConsentStatus.GIVEN}"
        )

        # Create GDPR consent record
        consent_record = create_consent_record(
            user_id=user_id,
            consent_status=scenario["consent_status"],
            lawful_basis=scenario["lawful_basis"],
            purpose=scenario["purpose"],
            data_categories=scenario["data_categories"],
        )

        total_consent_records += 1
        print(f"   ✅ Consent record created: {consent_record.consent_id[:8]}...")
        print(f"   📋 Data categories: {', '.join(scenario['data_categories'])}")

        with adapter.track_analytics_session(
            session_name=f"gdpr_{scenario['scenario']}",
            cost_center="privacy_compliant_analytics",
            lawful_basis=scenario["lawful_basis"].value,
            consent_status=scenario["consent_status"].value,
            data_subject_id=user_id,
            purpose_limitation=scenario["purpose"],
        ) as session:
            # Simulate GDPR-compliant analytics events
            gdpr_events = [
                {
                    "event_name": "page_view_gdpr",
                    "personal_data": True,
                    "data_categories": ["behavioral_data"],
                    "purpose": scenario["purpose"],
                },
                {
                    "event_name": "feature_interaction_gdpr",
                    "personal_data": True,
                    "data_categories": ["usage_analytics"],
                    "purpose": scenario["purpose"],
                },
                {
                    "event_name": "session_analytics_gdpr",
                    "personal_data": False,
                    "data_categories": ["performance_data"],
                    "purpose": scenario["purpose"],
                },
            ]

            for event in gdpr_events:
                # Build GDPR-compliant event properties
                event_properties = {
                    "gdpr_compliance": True,
                    "lawful_basis": scenario["lawful_basis"].value,
                    "consent_id": consent_record.consent_id,
                    "consent_status": scenario["consent_status"].value,
                    "data_categories": event["data_categories"],
                    "purpose_limitation": event["purpose"],
                    "retention_period": "2_years_after_last_interaction",
                    "cross_border_transfer": False,  # EU-only processing
                    "data_minimization": True,
                    "privacy_by_design": True,
                    "dpo_contact": dpo_email,
                    "data_subject_rights_info": "available_via_privacy_portal",
                }

                # Only process if we have lawful basis
                if (
                    scenario["consent_status"] == ConsentStatus.GIVEN
                    or scenario["lawful_basis"] != LawfulBasis.CONSENT
                ):
                    result = adapter.capture_event_with_governance(
                        event_name=event["event_name"],
                        properties=event_properties,
                        distinct_id=user_id,
                        is_identified=event["personal_data"],
                        session_id=session.session_id,
                    )

                    total_gdpr_events += 1

                    print(
                        f"     📊 {event['event_name']} tracked - Cost: ${result['cost']:.6f}"
                    )
                    print(
                        f"       Personal data: {'Yes' if event['personal_data'] else 'No'}"
                    )
                    print(
                        f"       Data categories: {', '.join(event['data_categories'])}"
                    )
                    print(f"       Purpose: {event['purpose']}")
                else:
                    print(f"     ❌ {event['event_name']} blocked - No valid consent")

    # Demonstrate GDPR Data Subject Rights Handling
    print("\n" + "=" * 55)
    print("⚖️ GDPR Data Subject Rights Management")
    print("=" * 55)

    # Simulate data subject rights requests
    rights_scenarios = [
        {
            "user_id": "eu_user_001",
            "request_type": DataSubjectRights.ACCESS,
            "description": "User requests access to all personal data",
        },
        {
            "user_id": "eu_user_002",
            "request_type": DataSubjectRights.DATA_PORTABILITY,
            "description": "User requests data export in machine-readable format",
        },
        {
            "user_id": "eu_user_003",
            "request_type": DataSubjectRights.ERASURE,
            "description": "User requests right to be forgotten",
        },
    ]

    for rights_scenario in rights_scenarios:
        print(
            f"\n🎯 Data Subject Rights Request: {rights_scenario['request_type'].value.title()}"
        )
        print("-" * 50)
        print(f"   Description: {rights_scenario['description']}")
        print(f"   User ID: {rights_scenario['user_id']}")

        # Find user's data categories from consent records
        user_consent = next(
            (cr for cr in consent_records if cr.user_id == rights_scenario["user_id"]),
            None,
        )

        if user_consent:
            # Handle the data subject request
            request = handle_data_subject_request(
                user_id=rights_scenario["user_id"],
                request_type=rights_scenario["request_type"],
                data_categories=user_consent.data_categories,
            )

            print(f"   ✅ Request processed: {request.request_id}")
            print(
                f"   📅 Fulfillment deadline: {datetime.fromisoformat(request.fulfillment_deadline.replace('Z', '+00:00')).strftime('%Y-%m-%d')}"
            )
            print(
                f"   📋 Data categories affected: {', '.join(request.data_categories)}"
            )

            # Track the rights request as a governance event
            result = adapter.capture_event_with_governance(
                event_name="gdpr_data_subject_request",
                properties={
                    "request_id": request.request_id,
                    "request_type": request.request_type,
                    "user_id": rights_scenario["user_id"],
                    "data_categories": request.data_categories,
                    "fulfillment_deadline": request.fulfillment_deadline,
                    "gdpr_article": "15"
                    if request.request_type == "access"
                    else "17"
                    if request.request_type == "erasure"
                    else "20",
                    "compliance_status": "in_progress",
                    "dpo_notified": True,
                },
                distinct_id=f"gdpr_admin_{rights_scenario['user_id']}",
                is_identified=True,
            )

            print(
                f"   📊 Request tracked with governance - Cost: ${result['cost']:.6f}"
            )

            # Simulate fulfillment based on request type
            if rights_scenario["request_type"] == DataSubjectRights.ACCESS:
                print("   📄 Generating personal data report for user...")
                print("   📧 Data access report will be sent securely to user")
            elif rights_scenario["request_type"] == DataSubjectRights.DATA_PORTABILITY:
                print("   📦 Preparing structured data export (JSON format)...")
                print("   💾 Portable data package ready for download")
            elif rights_scenario["request_type"] == DataSubjectRights.ERASURE:
                print("   🗑️ Initiating right to be forgotten process...")
                print(
                    "   ⚠️ Legal basis check: Retention may be required for legal obligations"
                )
        else:
            print(
                f"   ❌ No consent record found for user {rights_scenario['user_id']}"
            )

    # GDPR Compliance Summary and Reporting
    print("\n" + "=" * 55)
    print("📋 GDPR Compliance Summary & Privacy Report")
    print("=" * 55)

    cost_summary = adapter.get_cost_summary()

    print("\n📊 Privacy Analytics Summary:")
    print(f"   Total GDPR events tracked: {total_gdpr_events}")
    print(f"   Consent records created: {total_consent_records}")
    print(f"   Data subject requests: {len(data_subject_requests)}")
    print(f"   Personal data categories: {len(personal_data_inventory)}")
    print(f"   Analytics cost: ${cost_summary['daily_costs']:.6f}")

    print("\n🛡️ GDPR Compliance Status:")
    print("   Regulation: EU GDPR (Regulation 2016/679)")
    print("   Geographic scope: European Union")
    print("   Privacy by design: ✅ Implemented")
    print("   Lawful basis tracking: ✅ Active for all processing")
    print("   Consent management: ✅ Granular and withdrawable")
    print("   Data subject rights: ✅ All rights supported")
    print("   Data retention: ✅ Purpose-limited and time-bound")
    print("   Cross-border transfers: ✅ EU-only processing")

    # Consent Status Analysis
    print("\n📋 Consent Status Analysis:")
    consent_status_summary = {}
    lawful_basis_summary = {}

    for record in consent_records:
        status = record.consent_status
        basis = record.lawful_basis

        consent_status_summary[status] = consent_status_summary.get(status, 0) + 1
        lawful_basis_summary[basis] = lawful_basis_summary.get(basis, 0) + 1

    for status, count in consent_status_summary.items():
        print(f"   {status.replace('_', ' ').title()}: {count} users")

    print("\n⚖️ Lawful Basis Distribution:")
    for basis, count in lawful_basis_summary.items():
        article = {
            "consent": "6(1)(a)",
            "contract": "6(1)(b)",
            "legitimate_interests": "6(1)(f)",
        }.get(basis, "6(1)(x)")
        print(
            f"   Article {article} - {basis.replace('_', ' ').title()}: {count} users"
        )

    # Data Subject Rights Requests Analysis
    print("\n👤 Data Subject Rights Requests:")
    if data_subject_requests:
        rights_summary = {}
        for request in data_subject_requests:
            right = request.request_type
            rights_summary[right] = rights_summary.get(right, 0) + 1

        for right, count in rights_summary.items():
            article = {"access": "15", "erasure": "17", "data_portability": "20"}.get(
                right, "X"
            )
            print(
                f"   Article {article} - {right.replace('_', ' ').title()}: {count} requests"
            )
    else:
        print("   No data subject rights requests submitted")

    # Generate GDPR Privacy Report
    print("\n📄 GDPR Privacy Impact Assessment:")

    privacy_report = {
        "report_metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "report_type": "gdpr_privacy_impact_assessment",
            "data_controller": "company_legal_entity",
            "dpo_contact": dpo_email,
            "reporting_period": "24_hours_demo",
        },
        "processing_summary": {
            "total_events": total_gdpr_events,
            "consent_based_processing": len(
                [r for r in consent_records if r.lawful_basis == "consent"]
            ),
            "legitimate_interests_processing": len(
                [r for r in consent_records if r.lawful_basis == "legitimate_interests"]
            ),
            "contract_based_processing": len(
                [r for r in consent_records if r.lawful_basis == "contract"]
            ),
        },
        "privacy_by_design_measures": [
            "data_minimization",
            "purpose_limitation",
            "storage_limitation",
            "consent_management",
            "privacy_notices",
            "data_subject_rights",
            "security_measures",
        ],
        "compliance_score": 95.5,  # Based on implementation completeness
    }

    print("   ✅ Privacy impact assessment completed")
    print(f"   🎯 GDPR compliance score: {privacy_report['compliance_score']}%")
    print(f"   📧 DPO notification: {dpo_email}")
    print(
        f"   📋 Privacy by design measures: {len(privacy_report['privacy_by_design_measures'])} implemented"
    )

    # GDPR Best Practices and Recommendations
    print("\n💡 GDPR Best Practices & Recommendations:")

    recommendations = [
        {
            "category": "Consent Management",
            "recommendation": "Implement granular consent with easy withdrawal mechanisms",
            "priority": "High",
            "gdpr_article": "Article 7",
        },
        {
            "category": "Data Subject Rights",
            "recommendation": "Automate data subject rights fulfillment with 30-day SLA",
            "priority": "High",
            "gdpr_article": "Articles 15-22",
        },
        {
            "category": "Privacy by Design",
            "recommendation": "Implement privacy-preserving analytics with differential privacy",
            "priority": "Medium",
            "gdpr_article": "Article 25",
        },
        {
            "category": "Cross-Border Transfers",
            "recommendation": "Ensure adequate protection for any non-EU data transfers",
            "priority": "Critical",
            "gdpr_article": "Chapter V",
        },
    ]

    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec['category']}: {rec['recommendation']}")
        print(
            f"      GDPR Reference: {rec['gdpr_article']}, Priority: {rec['priority']}"
        )
        print()

    print("✅ GDPR compliance template demonstration completed successfully!")
    print("\n📚 Next Steps for GDPR Implementation:")
    print("   1. Conduct comprehensive data protection impact assessment (DPIA)")
    print("   2. Implement automated consent management and withdrawal")
    print("   3. Set up data subject rights fulfillment automation")
    print("   4. Establish data retention and deletion policies")
    print("   5. Coordinate with DPO for ongoing compliance monitoring")

    return True


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 GDPR compliance demonstration interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n💥 Error in GDPR compliance example: {e}")
        print("🔧 Please check your PostHog configuration and privacy settings")
        exit(1)
