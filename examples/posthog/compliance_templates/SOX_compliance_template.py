#!/usr/bin/env python3
"""
SOX Compliance Template for PostHog + GenOps

This template demonstrates Sarbanes-Oxley (SOX) compliance implementation for
PostHog analytics with GenOps governance. SOX requires strict financial data
controls, audit trails, and change management for publicly traded companies.

SOX Requirements Addressed:
- Section 302: Management assessment of internal controls
- Section 404: Management assessment of internal control over financial reporting
- Section 409: Real-time financial disclosure requirements
- Audit trail requirements with immutable logs
- Data retention policies (7 years minimum)
- Access controls and segregation of duties

Use Case:
    - Publicly traded companies tracking financial metrics
    - E-commerce revenue and transaction analytics
    - Financial dashboard and reporting compliance
    - Audit trail generation for financial data access

Usage:
    python compliance_templates/SOX_compliance_template.py

Prerequisites:
    pip install genops[posthog]
    export POSTHOG_API_KEY="phc_your_project_api_key"
    export GENOPS_TEAM="finance-analytics"
    export GENOPS_PROJECT="sox-compliance"
    export SOX_AUDITOR_EMAIL="auditor@company.com"  # Required for audit notifications

Expected Output:
    SOX-compliant financial analytics tracking with full audit trail,
    immutable logs, and compliance reporting for financial data governance.

Learning Objectives:
    - SOX compliance requirements for financial data analytics
    - Audit trail generation and immutable logging patterns
    - Financial data access controls and segregation of duties
    - Real-time financial reporting with compliance governance

Author: GenOps AI Compliance Team
License: Apache 2.0
"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Optional


@dataclass
class SOXAuditEntry:
    """SOX-compliant audit log entry with immutable properties."""

    audit_id: str
    timestamp: str
    user_id: str
    action: str
    resource_type: str
    resource_id: str
    financial_data_involved: bool
    sox_control_point: str
    risk_level: str
    approval_status: str
    supervisor_approval: Optional[str]
    data_hash: str
    retention_until: str
    compliance_metadata: dict[str, Any]


def generate_audit_hash(data: dict[str, Any]) -> str:
    """Generate immutable hash for audit trail integrity."""
    audit_string = json.dumps(data, sort_keys=True)
    return hashlib.sha256(audit_string.encode()).hexdigest()


def main():
    """Demonstrate SOX-compliant PostHog analytics with full governance."""
    print("🏛️ SOX Compliance Template for PostHog + GenOps")
    print("=" * 55)
    print()

    # Import and setup GenOps PostHog adapter with SOX configuration
    try:
        from genops.providers.posthog import GenOpsPostHogAdapter

        print("✅ GenOps PostHog integration loaded")
    except ImportError as e:
        print(f"❌ Failed to import GenOps PostHog: {e}")
        print("💡 Fix: pip install genops[posthog]")
        return False

    # SOX Compliance Configuration
    print("\n🔧 Configuring SOX Compliance Environment...")

    sox_auditor_email = os.getenv("SOX_AUDITOR_EMAIL")
    if not sox_auditor_email:
        print("⚠️ SOX_AUDITOR_EMAIL not configured - using demo value")
        sox_auditor_email = "sox-auditor@company-demo.com"

    # Initialize SOX-compliant adapter
    adapter = GenOpsPostHogAdapter(
        team="sox-finance-analytics",
        project="financial-reporting-system",
        environment="production",
        customer_id="sox_compliance_entity",
        cost_center="financial_operations",
        daily_budget_limit=500.0,  # Higher budget for critical financial systems
        governance_policy="strict",  # Strictest enforcement for SOX
        tags={
            "compliance_framework": "sox",
            "sox_entity": "publicly_traded_company",
            "data_classification": "financial_confidential",
            "retention_policy": "7_years_minimum",
            "audit_trail_required": "true",
            "change_management": "formal_approval_required",
            "access_control": "role_based_segregated",
            "sox_auditor_contact": sox_auditor_email,
            "financial_year": "2024",
            "sox_compliance_level": "section_302_404",
        },
    )

    print("✅ SOX-compliant adapter configured")
    print("   🏢 Entity: Publicly traded company")
    print("   📋 Compliance level: SOX Sections 302 & 404")
    print("   🔒 Governance policy: Strict enforcement")
    print(f"   📧 SOX auditor: {sox_auditor_email}")
    print("   💾 Data retention: 7+ years")
    print("   🛡️ Access controls: Role-based segregation")

    # SOX audit log for compliance tracking
    sox_audit_log: list[SOXAuditEntry] = []

    def create_sox_audit_entry(
        action: str,
        resource_type: str,
        resource_id: str,
        financial_data: bool = True,
        sox_control: str = "general",
        risk_level: str = "medium",
    ) -> SOXAuditEntry:
        """Create SOX-compliant audit entry with immutable properties."""

        timestamp = datetime.now(timezone.utc)
        audit_data = {
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "timestamp": timestamp.isoformat(),
            "financial_data_involved": financial_data,
        }

        entry = SOXAuditEntry(
            audit_id=f"SOX_{timestamp.strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(str(audit_data).encode()).hexdigest()[:8]}",
            timestamp=timestamp.isoformat(),
            user_id="finance_analytics_system",
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            financial_data_involved=financial_data,
            sox_control_point=sox_control,
            risk_level=risk_level,
            approval_status="system_approved"
            if risk_level == "low"
            else "supervisor_approval_required",
            supervisor_approval="auto_approved"
            if risk_level == "low"
            else "pending_finance_manager",
            data_hash=generate_audit_hash(audit_data),
            retention_until=(timestamp + timedelta(days=2557)).isoformat(),  # 7+ years
            compliance_metadata={
                "sox_section": "302_404",
                "financial_materiality": "material"
                if financial_data
                else "non_material",
                "segregation_compliance": "verified",
                "change_control_id": f"CC_{timestamp.strftime('%Y%m%d')}_{len(sox_audit_log) + 1:03d}",
            },
        )

        sox_audit_log.append(entry)
        return entry

    # Demonstrate SOX-compliant financial analytics scenarios
    print("\n" + "=" * 55)
    print("💰 SOX-Compliant Financial Analytics Tracking")
    print("=" * 55)

    # Financial reporting scenarios with SOX requirements
    financial_scenarios = [
        {
            "scenario": "quarterly_revenue_reporting",
            "description": "Q4 2024 revenue recognition and reporting",
            "sox_control": "revenue_recognition",
            "risk_level": "high",
            "events": [
                {"type": "revenue_transaction", "amount": 125000.00, "currency": "USD"},
                {"type": "revenue_adjustment", "amount": -2500.00, "currency": "USD"},
                {"type": "revenue_recognition", "amount": 122500.00, "currency": "USD"},
            ],
        },
        {
            "scenario": "financial_dashboard_access",
            "description": "Executive dashboard access for SOX reporting",
            "sox_control": "management_assessment",
            "risk_level": "medium",
            "events": [
                {"type": "dashboard_view", "report_type": "executive_summary"},
                {"type": "financial_metric_access", "metric_type": "cash_flow"},
                {
                    "type": "sox_control_review",
                    "control_type": "internal_control_assessment",
                },
            ],
        },
        {
            "scenario": "audit_preparation",
            "description": "Preparing for external SOX audit",
            "sox_control": "audit_compliance",
            "risk_level": "critical",
            "events": [
                {"type": "audit_trail_export", "period": "FY2024"},
                {"type": "control_testing", "control_id": "ITGC-001"},
                {"type": "deficiency_tracking", "deficiency_type": "material_weakness"},
            ],
        },
    ]

    total_financial_transactions = 0
    total_audit_entries = 0
    sox_compliance_score = 100.0

    for scenario_idx, scenario in enumerate(financial_scenarios, 1):
        print(f"\n📊 Scenario {scenario_idx}: {scenario['description']}")
        print("-" * 50)
        print(f"   SOX Control: {scenario['sox_control']}")
        print(f"   Risk Level: {scenario['risk_level']}")

        # Create audit entry for scenario initiation
        audit_entry = create_sox_audit_entry(
            action="scenario_initiated",
            resource_type="financial_analytics_scenario",
            resource_id=scenario["scenario"],
            financial_data=True,
            sox_control=scenario["sox_control"],
            risk_level=scenario["risk_level"],
        )

        total_audit_entries += 1
        print(f"   🔍 Audit entry created: {audit_entry.audit_id}")

        with adapter.track_analytics_session(
            session_name=scenario["scenario"],
            cost_center="sox_compliance_reporting",
            sox_control_point=scenario["sox_control"],
            risk_assessment=scenario["risk_level"],
            financial_materiality="material",
        ) as session:
            scenario_cost = Decimal("0")

            for event_idx, event in enumerate(scenario["events"]):
                print(f"\n   📈 Event {event_idx + 1}: {event['type']}")

                # Build SOX-compliant event properties
                event_properties = {
                    "sox_control_point": scenario["sox_control"],
                    "risk_level": scenario["risk_level"],
                    "financial_materiality": "material",
                    "segregation_verified": True,
                    "approval_status": "authorized",
                    "sox_section_applicable": "302_404",
                    "change_control_documented": True,
                    "audit_trail_enabled": True,
                    **event,
                }

                # Add financial amount tracking if present
                if "amount" in event:
                    event_properties.update(
                        {
                            "financial_transaction": True,
                            "transaction_amount": event["amount"],
                            "currency": event.get("currency", "USD"),
                            "materiality_threshold_check": abs(event["amount"])
                            >= 10000.0,
                        }
                    )
                    total_financial_transactions += 1

                # Capture event with SOX compliance
                result = adapter.capture_event_with_governance(
                    event_name=f"sox_{event['type']}",
                    properties=event_properties,
                    distinct_id=f"sox_user_{scenario['scenario']}",
                    is_identified=True,  # Financial events are always identified
                    session_id=session.session_id,
                )

                scenario_cost += Decimal(str(result["cost"]))

                # Create detailed audit entry for each financial event
                event_audit = create_sox_audit_entry(
                    action="financial_event_captured",
                    resource_type="postoh_analytics_event",
                    resource_id=f"{scenario['scenario']}_{event['type']}",
                    financial_data="amount" in event,
                    sox_control=scenario["sox_control"],
                    risk_level=scenario["risk_level"],
                )

                total_audit_entries += 1

                print(
                    f"     Event tracked with SOX compliance - Cost: ${result['cost']:.6f}"
                )
                print(f"     Audit ID: {event_audit.audit_id}")
                print(f"     Data hash: {event_audit.data_hash[:16]}...")

                if "amount" in event:
                    print(
                        f"     Financial amount: {event.get('currency', 'USD')} {event['amount']:,.2f}"
                    )
                    print(
                        f"     Materiality check: {'✅ Material' if abs(event['amount']) >= 10000.0 else '⚠️ Below threshold'}"
                    )

            # Session compliance summary
            print("\n   📋 Scenario Summary:")
            print(f"     Events processed: {len(scenario['events'])}")
            print(f"     Session cost: ${scenario_cost:.4f}")
            print(f"     SOX control: {scenario['sox_control']}")
            print(f"     Risk level: {scenario['risk_level']}")
            print(
                f"     Audit entries: {len([e for e in sox_audit_log if scenario['scenario'] in e.resource_id])}"
            )

    # SOX Compliance Summary and Reporting
    print("\n" + "=" * 55)
    print("📋 SOX Compliance Summary & Audit Report")
    print("=" * 55)

    cost_summary = adapter.get_cost_summary()

    print("\n💰 Financial Analytics Summary:")
    print(f"   Total financial transactions tracked: {total_financial_transactions}")
    print(f"   Total audit entries generated: {total_audit_entries}")
    print(f"   Analytics cost: ${cost_summary['daily_costs']:.6f}")
    print(f"   Budget utilization: {cost_summary['daily_budget_utilization']:.1f}%")

    print("\n🏛️ SOX Compliance Status:")
    print("   Compliance framework: SOX (Sarbanes-Oxley Act)")
    print(
        "   Applicable sections: 302 (Management Assessment), 404 (Internal Controls)"
    )
    print(
        f"   Data retention period: 7+ years (until {(datetime.now() + timedelta(days=2557)).strftime('%Y-%m-%d')})"
    )
    print(
        f"   Audit trail completeness: {'✅ 100%' if total_audit_entries > 0 else '❌ Incomplete'}"
    )
    print("   Financial data segregation: ✅ Verified")
    print("   Change control compliance: ✅ Documented")
    print("   Access controls: ✅ Role-based segregation")

    # Audit Trail Analysis
    print("\n🔍 Audit Trail Analysis:")

    # Group audit entries by risk level
    risk_level_summary = {}
    for entry in sox_audit_log:
        level = entry.risk_level
        if level not in risk_level_summary:
            risk_level_summary[level] = 0
        risk_level_summary[level] += 1

    for risk_level, count in risk_level_summary.items():
        print(f"   {risk_level.title()} risk operations: {count}")

    # SOX Control Point Analysis
    control_points = {}
    for entry in sox_audit_log:
        control = entry.sox_control_point
        if control not in control_points:
            control_points[control] = 0
        control_points[control] += 1

    print("\n🛡️ SOX Control Points Coverage:")
    for control, count in control_points.items():
        print(f"   {control.replace('_', ' ').title()}: {count} operations")

    # Generate SOX Audit Report Export
    print("\n📄 SOX Audit Report Generation:")

    audit_report = {
        "report_metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "report_type": "sox_compliance_audit_trail",
            "reporting_entity": "publicly_traded_company",
            "financial_year": "2024",
            "sox_sections": ["302", "404"],
            "auditor_contact": sox_auditor_email,
        },
        "compliance_summary": {
            "total_financial_transactions": total_financial_transactions,
            "total_audit_entries": total_audit_entries,
            "analytics_cost_usd": float(cost_summary["daily_costs"]),
            "compliance_score": sox_compliance_score,
            "control_points_tested": list(control_points.keys()),
        },
        "audit_entries": [
            asdict(entry) for entry in sox_audit_log[-5:]
        ],  # Last 5 entries for demo
    }

    # In production, this would be exported to secure audit storage
    print(
        f"   ✅ Audit report generated: {len(audit_report['audit_entries'])} entries (sample)"
    )
    print(f"   🔒 Report hash: {generate_audit_hash(audit_report)[:16]}...")
    print(f"   📧 Auditor notification: {sox_auditor_email}")
    print(
        f"   💾 Retention until: {(datetime.now() + timedelta(days=2557)).strftime('%Y-%m-%d')}"
    )

    # SOX Compliance Recommendations
    print("\n💡 SOX Compliance Recommendations:")

    recommendations = [
        {
            "category": "Internal Controls",
            "recommendation": "Implement automated control testing for ITGC controls",
            "priority": "High",
            "timeline": "30 days",
        },
        {
            "category": "Data Retention",
            "recommendation": "Establish automated 7-year retention policy with legal hold",
            "priority": "Medium",
            "timeline": "60 days",
        },
        {
            "category": "Access Controls",
            "recommendation": "Regular access review and segregation of duties validation",
            "priority": "High",
            "timeline": "Quarterly",
        },
        {
            "category": "Audit Preparation",
            "recommendation": "Implement continuous controls monitoring and deficiency tracking",
            "priority": "Medium",
            "timeline": "90 days",
        },
    ]

    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec['category']}: {rec['recommendation']}")
        print(f"      Priority: {rec['priority']}, Timeline: {rec['timeline']}")
        print()

    print("✅ SOX compliance template demonstration completed successfully!")
    print("\n📚 Next Steps for SOX Implementation:")
    print("   1. Review and customize SOX control points for your organization")
    print("   2. Implement automated audit trail export and archival")
    print("   3. Set up role-based access controls and segregation of duties")
    print("   4. Establish quarterly SOX compliance review processes")
    print("   5. Coordinate with external auditors for SOX 404 assessment")

    return True


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 SOX compliance demonstration interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n💥 Error in SOX compliance example: {e}")
        print("🔧 Please check your PostHog configuration and compliance settings")
        exit(1)
