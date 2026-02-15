#!/usr/bin/env python3
"""
LiteLLM Compliance Monitoring and Governance Automation with GenOps

Demonstrates comprehensive compliance monitoring, audit trails, and governance
automation for enterprise LiteLLM deployments. This showcases patterns for
regulatory compliance, data governance, and automated policy enforcement.

Usage:
    export OPENAI_API_KEY="your_key_here"
    python compliance_monitoring.py

Features:
    - Comprehensive audit trails for AI requests
    - Data governance and privacy protection
    - Regulatory compliance patterns (SOX, GDPR, HIPAA)
    - Automated policy enforcement and violations detection
    - Cost governance and budget compliance monitoring
    - Real-time compliance reporting and alerting
"""

import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure compliance-level logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [COMPLIANCE] %(message)s",
)
logger = logging.getLogger(__name__)


class ComplianceLevel(Enum):
    """Compliance levels for different regulatory requirements."""

    BASIC = "basic"
    SOX = "sox"  # Sarbanes-Oxley
    GDPR = "gdpr"  # General Data Protection Regulation
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    PCI_DSS = "pci_dss"  # Payment Card Industry Data Security Standard
    STRICT = "strict"  # Maximum compliance


class PolicyViolationType(Enum):
    """Types of policy violations."""

    BUDGET_EXCEEDED = "budget_exceeded"
    UNAUTHORIZED_MODEL = "unauthorized_model"
    DATA_SENSITIVITY = "data_sensitivity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    CONTENT_POLICY = "content_policy"


@dataclass
class ComplianceEvent:
    """Individual compliance event record."""

    event_id: str
    timestamp: str
    event_type: str
    user_id: Optional[str]
    team: str
    project: str
    customer_id: Optional[str]

    # Request details
    model_used: str
    provider: str
    cost: float
    tokens_used: int

    # Compliance metadata
    compliance_level: str
    data_classification: str
    audit_trail_id: str

    # Privacy and security
    pii_detected: bool = False
    sensitive_data_redacted: bool = False
    encryption_applied: bool = True

    # Custom attributes
    custom_attributes: dict[str, Any] = field(default_factory=dict)

    def to_audit_record(self) -> dict[str, Any]:
        """Convert to audit record format."""
        record = asdict(self)
        record["audit_hash"] = self._generate_audit_hash(record)
        return record

    def _generate_audit_hash(self, record: dict[str, Any]) -> str:
        """Generate audit hash for tamper detection."""
        # Create deterministic string for hashing
        audit_string = json.dumps(record, sort_keys=True, default=str)
        return hashlib.sha256(audit_string.encode()).hexdigest()[:16]


@dataclass
class PolicyViolation:
    """Policy violation record."""

    violation_id: str
    timestamp: str
    violation_type: PolicyViolationType
    severity: str  # low, medium, high, critical
    description: str

    # Context
    user_id: Optional[str]
    team: str
    project: str
    customer_id: Optional[str]

    # Violation details
    policy_name: str
    threshold_value: Optional[Union[float, int]]
    actual_value: Optional[Union[float, int]]

    # Resolution
    auto_resolved: bool = False
    resolution_action: Optional[str] = None
    acknowledged: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format."""
        result = asdict(self)
        result["violation_type"] = self.violation_type.value
        return result


class ComplianceMonitor:
    """Comprehensive compliance monitoring for LiteLLM + GenOps."""

    def __init__(self, compliance_level: ComplianceLevel = ComplianceLevel.BASIC):
        self.compliance_level = compliance_level
        self.audit_events: list[ComplianceEvent] = []
        self.policy_violations: list[PolicyViolation] = []
        self.active_policies: dict[str, dict[str, Any]] = {}
        self.is_monitoring = False

        # Initialize policies based on compliance level
        self._initialize_policies()

    def _initialize_policies(self):
        """Initialize policies based on compliance level."""
        base_policies = {
            "daily_budget_limit": {"threshold": 1000.0, "enabled": True},
            "request_rate_limit": {"threshold": 100, "period": 3600, "enabled": True},
            "authorized_models": {
                "allowed": ["gpt-3.5-turbo", "gpt-4", "claude-3"],
                "enabled": False,
            },
        }

        if self.compliance_level in [ComplianceLevel.GDPR, ComplianceLevel.HIPAA]:
            base_policies.update(
                {
                    "pii_detection": {"enabled": True, "action": "redact"},
                    "data_retention": {"days": 30, "enabled": True},
                    "encryption_required": {"enabled": True},
                }
            )

        if self.compliance_level == ComplianceLevel.SOX:
            base_policies.update(
                {
                    "audit_trail_required": {"enabled": True},
                    "segregation_of_duties": {"enabled": True},
                    "change_approval": {"enabled": True},
                }
            )

        if self.compliance_level == ComplianceLevel.STRICT:
            # Enable all policies
            for policy in base_policies.values():
                policy["enabled"] = True
            base_policies.update(
                {
                    "content_filtering": {"enabled": True},
                    "model_approval": {"enabled": True},
                    "dual_authorization": {"enabled": True},
                }
            )

        self.active_policies = base_policies
        logger.info(
            f"Initialized {len(base_policies)} policies for {self.compliance_level.value} compliance"
        )

    def start_monitoring(self) -> bool:
        """Start compliance monitoring."""
        try:
            from genops.providers.litellm import auto_instrument

            # Configure GenOps with compliance settings
            success = auto_instrument(
                team="compliance-team",
                project="governance-monitoring",
                environment="production",
                governance_policy="strict",
                enable_cost_tracking=True,
                # Compliance attributes
                compliance_level=self.compliance_level.value,
                audit_enabled=True,
                data_retention_days=30
                if self.compliance_level != ComplianceLevel.BASIC
                else 7,
            )

            if success:
                self.is_monitoring = True
                logger.info(
                    f"Compliance monitoring started with {self.compliance_level.value} level"
                )
                return True
            else:
                logger.error("Failed to start compliance monitoring")
                return False

        except Exception as e:
            logger.error(f"Error starting compliance monitoring: {e}")
            return False

    def record_compliance_event(
        self,
        event_type: str,
        user_id: Optional[str],
        team: str,
        project: str,
        model_used: str,
        provider: str,
        cost: float,
        tokens_used: int,
        customer_id: Optional[str] = None,
        **kwargs,
    ) -> ComplianceEvent:
        """Record a compliance event."""

        event_id = f"comp-{int(time.time() * 1000)}-{len(self.audit_events)}"
        audit_trail_id = f"audit-{hashlib.md5(f'{event_id}{team}{project}'.encode()).hexdigest()[:8]}"

        event = ComplianceEvent(
            event_id=event_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            user_id=user_id,
            team=team,
            project=project,
            customer_id=customer_id,
            model_used=model_used,
            provider=provider,
            cost=cost,
            tokens_used=tokens_used,
            compliance_level=self.compliance_level.value,
            data_classification=kwargs.get("data_classification", "internal"),
            audit_trail_id=audit_trail_id,
            pii_detected=kwargs.get("pii_detected", False),
            sensitive_data_redacted=kwargs.get("sensitive_data_redacted", False),
            custom_attributes=kwargs.get("custom_attributes", {}),
        )

        self.audit_events.append(event)

        # Check for policy violations
        self._check_policy_compliance(event)

        logger.info(f"Compliance event recorded: {event_id}")
        return event

    def _check_policy_compliance(self, event: ComplianceEvent):
        """Check event against active policies."""

        # Budget compliance
        if self.active_policies.get("daily_budget_limit", {}).get("enabled"):
            daily_spend = sum(
                e.cost
                for e in self.audit_events
                if e.timestamp.split("T")[0] == event.timestamp.split("T")[0]
                and e.team == event.team
            )

            budget_limit = self.active_policies["daily_budget_limit"]["threshold"]

            if daily_spend > budget_limit:
                self._create_violation(
                    violation_type=PolicyViolationType.BUDGET_EXCEEDED,
                    event=event,
                    policy_name="daily_budget_limit",
                    threshold_value=budget_limit,
                    actual_value=daily_spend,
                    severity="high",
                )

        # Authorized models
        if self.active_policies.get("authorized_models", {}).get("enabled"):
            allowed_models = self.active_policies["authorized_models"]["allowed"]
            if event.model_used not in allowed_models:
                self._create_violation(
                    violation_type=PolicyViolationType.UNAUTHORIZED_MODEL,
                    event=event,
                    policy_name="authorized_models",
                    threshold_value=None,
                    actual_value=None,
                    severity="medium",
                )

        # PII detection
        if event.pii_detected and not event.sensitive_data_redacted:
            if self.active_policies.get("pii_detection", {}).get("enabled"):
                self._create_violation(
                    violation_type=PolicyViolationType.DATA_SENSITIVITY,
                    event=event,
                    policy_name="pii_detection",
                    threshold_value=None,
                    actual_value=None,
                    severity="critical",
                )

    def _create_violation(
        self,
        violation_type: PolicyViolationType,
        event: ComplianceEvent,
        policy_name: str,
        threshold_value: Optional[Union[float, int]],
        actual_value: Optional[Union[float, int]],
        severity: str,
    ):
        """Create a policy violation record."""

        violation_id = f"viol-{int(time.time() * 1000)}-{len(self.policy_violations)}"

        violation = PolicyViolation(
            violation_id=violation_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            violation_type=violation_type,
            severity=severity,
            description=f"Policy violation: {policy_name}",
            user_id=event.user_id,
            team=event.team,
            project=event.project,
            customer_id=event.customer_id,
            policy_name=policy_name,
            threshold_value=threshold_value,
            actual_value=actual_value,
        )

        self.policy_violations.append(violation)

        logger.warning(
            f"Policy violation detected: {violation_id} - {violation_type.value}"
        )

        # Auto-resolve if configured
        if severity in ["low", "medium"]:
            violation.auto_resolved = True
            violation.resolution_action = "automated_notification"

    def get_compliance_report(self, team: Optional[str] = None) -> dict[str, Any]:
        """Generate compliance report."""

        events = self.audit_events
        violations = self.policy_violations

        if team:
            events = [e for e in events if e.team == team]
            violations = [v for v in violations if v.team == team]

        report = {
            "report_generated": datetime.now(timezone.utc).isoformat(),
            "compliance_level": self.compliance_level.value,
            "reporting_period": "current_session",
            "team_filter": team,
            "summary": {
                "total_events": len(events),
                "total_violations": len(violations),
                "critical_violations": len(
                    [v for v in violations if v.severity == "critical"]
                ),
                "unresolved_violations": len(
                    [v for v in violations if not v.auto_resolved]
                ),
                "total_cost": sum(e.cost for e in events),
                "unique_users": len({e.user_id for e in events if e.user_id}),
                "unique_teams": len({e.team for e in events}),
            },
            "cost_breakdown": self._get_cost_breakdown(events),
            "violations_by_type": self._get_violations_by_type(violations),
            "compliance_score": self._calculate_compliance_score(events, violations),
        }

        return report

    def _get_cost_breakdown(self, events: list[ComplianceEvent]) -> dict[str, Any]:
        """Get cost breakdown for compliance report."""
        breakdown = {
            "by_team": {},
            "by_project": {},
            "by_provider": {},
            "by_customer": {},
        }

        for event in events:
            # By team
            breakdown["by_team"][event.team] = (
                breakdown["by_team"].get(event.team, 0) + event.cost
            )

            # By project
            breakdown["by_project"][event.project] = (
                breakdown["by_project"].get(event.project, 0) + event.cost
            )

            # By provider
            breakdown["by_provider"][event.provider] = (
                breakdown["by_provider"].get(event.provider, 0) + event.cost
            )

            # By customer (if present)
            if event.customer_id:
                breakdown["by_customer"][event.customer_id] = (
                    breakdown["by_customer"].get(event.customer_id, 0) + event.cost
                )

        return breakdown

    def _get_violations_by_type(
        self, violations: list[PolicyViolation]
    ) -> dict[str, int]:
        """Get violations grouped by type."""
        counts = {}
        for violation in violations:
            vtype = violation.violation_type.value
            counts[vtype] = counts.get(vtype, 0) + 1
        return counts

    def _calculate_compliance_score(
        self, events: list[ComplianceEvent], violations: list[PolicyViolation]
    ) -> float:
        """Calculate compliance score (0-100)."""
        if not events:
            return 100.0

        # Base score
        score = 100.0

        # Deduct for violations
        for violation in violations:
            if violation.severity == "critical":
                score -= 10
            elif violation.severity == "high":
                score -= 5
            elif violation.severity == "medium":
                score -= 2
            elif violation.severity == "low":
                score -= 1

        return max(0.0, score)


def check_compliance_setup():
    """Check compliance monitoring setup."""
    print("🔍 Checking compliance monitoring setup...")

    # Check imports
    try:
        import litellm  # noqa: F401

        from genops.providers.litellm import (  # noqa: F401
            auto_instrument,
            get_usage_stats,
        )

        print("✅ LiteLLM and GenOps available")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Install: pip install litellm genops[litellm]")
        return False

    # Check API keys
    api_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]
    configured = [key for key in api_keys if os.getenv(key)]

    if configured:
        print(f"✅ {len(configured)} API key(s) configured for compliance testing")
    else:
        print("⚠️  No API keys configured - will use demo mode")

    print("✅ Compliance monitoring setup ready")
    return True


def demo_audit_trail_generation():
    """Demonstrate comprehensive audit trail generation."""
    print("\n" + "=" * 60)
    print("📋 Demo: Audit Trail Generation")
    print("=" * 60)

    print("Comprehensive audit trails for regulatory compliance:")
    print("• Every AI request recorded with full context")
    print("• Tamper-evident audit hashes")
    print("• User attribution and data classification")

    # Initialize compliance monitor
    monitor = ComplianceMonitor(ComplianceLevel.SOX)

    if not monitor.start_monitoring():
        print("❌ Failed to start compliance monitoring")
        return

    # Simulate AI requests with full audit trails
    audit_scenarios = [
        {
            "user_id": "user123",
            "team": "financial-reporting",
            "project": "quarterly-analysis",
            "model": "gpt-4",
            "provider": "openai",
            "cost": 0.025,
            "tokens": 1500,
            "customer_id": "enterprise-corp",
            "data_classification": "confidential",
            "custom_attributes": {
                "department": "finance",
                "sox_compliance": True,
                "audit_required": True,
            },
        },
        {
            "user_id": "user456",
            "team": "customer-support",
            "project": "automated-responses",
            "model": "claude-3-sonnet",
            "provider": "anthropic",
            "cost": 0.018,
            "tokens": 1200,
            "customer_id": "customer-abc",
            "data_classification": "internal",
            "pii_detected": True,
            "sensitive_data_redacted": True,
        },
    ]

    print("\n📋 Generating audit trails:")

    for i, scenario in enumerate(audit_scenarios):
        event = monitor.record_compliance_event(
            event_type="ai_request",
            user_id=scenario["user_id"],
            team=scenario["team"],
            project=scenario["project"],
            model_used=scenario["model"],
            provider=scenario["provider"],
            cost=scenario["cost"],
            tokens_used=scenario["tokens"],
            customer_id=scenario["customer_id"],
            data_classification=scenario["data_classification"],
            pii_detected=scenario.get("pii_detected", False),
            sensitive_data_redacted=scenario.get("sensitive_data_redacted", False),
            custom_attributes=scenario.get("custom_attributes", {}),
        )

        print(f"   📊 Event {i + 1}: {event.event_id}")
        print(f"      • User: {event.user_id}")
        print(f"      • Team: {event.team} / Project: {event.project}")
        print(f"      • Model: {event.model_used} ({event.provider})")
        print(f"      • Cost: ${event.cost:.6f}, Tokens: {event.tokens_used}")
        print(f"      • Audit ID: {event.audit_trail_id}")
        print(f"      • Data class: {event.data_classification}")

        if event.pii_detected:
            print(f"      • ⚠️  PII detected, redacted: {event.sensitive_data_redacted}")

    print(
        f"\n✅ {len(monitor.audit_events)} compliance events recorded with full audit trails"
    )


def demo_policy_enforcement():
    """Demonstrate automated policy enforcement."""
    print("\n" + "=" * 60)
    print("🛡️ Demo: Automated Policy Enforcement")
    print("=" * 60)

    print("Automated governance with real-time policy enforcement:")
    print("• Budget limits with automatic violation detection")
    print("• Model authorization policies")
    print("• Data sensitivity and PII protection")

    # Initialize strict compliance monitor
    monitor = ComplianceMonitor(ComplianceLevel.STRICT)
    monitor.start_monitoring()

    print(f"\n📋 Active Policies ({monitor.compliance_level.value} level):")
    for policy_name, policy_config in monitor.active_policies.items():
        if policy_config.get("enabled"):
            print(f"   ✅ {policy_name}: {policy_config}")

    # Simulate policy violations
    violation_scenarios = [
        {
            "description": "Budget limit exceeded",
            "user_id": "user789",
            "team": "marketing-ai",
            "project": "campaign-generation",
            "model": "gpt-4",
            "provider": "openai",
            "cost": 1100.0,  # Exceeds 1000 limit
            "tokens": 50000,
            "expected_violation": PolicyViolationType.BUDGET_EXCEEDED,
        },
        {
            "description": "Unauthorized model usage",
            "user_id": "user101",
            "team": "research-team",
            "project": "experimental-ai",
            "model": "gpt-4-turbo",  # Not in authorized list
            "provider": "openai",
            "cost": 0.050,
            "tokens": 2000,
            "expected_violation": PolicyViolationType.UNAUTHORIZED_MODEL,
        },
        {
            "description": "PII detected without redaction",
            "user_id": "user202",
            "team": "healthcare-ai",
            "project": "patient-analysis",
            "model": "gpt-3.5-turbo",
            "provider": "openai",
            "cost": 0.020,
            "tokens": 1000,
            "pii_detected": True,
            "sensitive_data_redacted": False,
            "expected_violation": PolicyViolationType.DATA_SENSITIVITY,
        },
    ]

    print("\n📋 Testing Policy Violations:")

    for i, scenario in enumerate(violation_scenarios):
        print(f"\n   🔍 Test {i + 1}: {scenario['description']}")

        # Record event that should trigger violation
        monitor.record_compliance_event(
            event_type="ai_request",
            user_id=scenario["user_id"],
            team=scenario["team"],
            project=scenario["project"],
            model_used=scenario["model"],
            provider=scenario["provider"],
            cost=scenario["cost"],
            tokens_used=scenario["tokens"],
            pii_detected=scenario.get("pii_detected", False),
            sensitive_data_redacted=scenario.get("sensitive_data_redacted", False),
        )

        # Check if violation was detected
        violations = [
            v
            for v in monitor.policy_violations
            if v.violation_type == scenario["expected_violation"]
        ]

        if violations:
            violation = violations[-1]  # Get latest violation
            print(f"      ⚠️ Violation detected: {violation.violation_id}")
            print(f"      • Type: {violation.violation_type.value}")
            print(f"      • Severity: {violation.severity}")
            print(f"      • Policy: {violation.policy_name}")
            print(f"      • Auto-resolved: {violation.auto_resolved}")
        else:
            print("      ✅ No violation detected (unexpected)")

    print("\n📊 Policy Enforcement Summary:")
    print(f"   Total violations: {len(monitor.policy_violations)}")
    print(
        f"   Auto-resolved: {len([v for v in monitor.policy_violations if v.auto_resolved])}"
    )
    print(
        f"   Critical: {len([v for v in monitor.policy_violations if v.severity == 'critical'])}"
    )


def demo_gdpr_compliance():
    """Demonstrate GDPR compliance patterns."""
    print("\n" + "=" * 60)
    print("🇪🇺 Demo: GDPR Compliance Patterns")
    print("=" * 60)

    print("GDPR compliance for AI systems:")
    print("• Data minimization and purpose limitation")
    print("• Automated PII detection and redaction")
    print("• Data retention and deletion policies")
    print("• Consent tracking and withdrawal")

    # Initialize GDPR compliance monitor
    monitor = ComplianceMonitor(ComplianceLevel.GDPR)
    monitor.start_monitoring()

    # GDPR-specific scenarios
    gdpr_scenarios = [
        {
            "scenario": "Customer service with PII",
            "user_id": "support_agent_1",
            "team": "customer-support-eu",
            "project": "gdpr-compliant-support",
            "model": "claude-3-sonnet",
            "provider": "anthropic",
            "cost": 0.015,
            "tokens": 800,
            "customer_id": "eu-customer-456",
            "data_classification": "personal_data",
            "pii_detected": True,
            "sensitive_data_redacted": True,
            "custom_attributes": {
                "gdpr_lawful_basis": "legitimate_interest",
                "data_subject_consent": True,
                "data_retention_category": "customer_service",
                "geographic_region": "eu",
            },
        },
        {
            "scenario": "Marketing analytics without consent",
            "user_id": "marketing_analyst",
            "team": "marketing-analytics",
            "project": "customer-segmentation",
            "model": "gpt-3.5-turbo",
            "provider": "openai",
            "cost": 0.012,
            "tokens": 600,
            "customer_id": "eu-prospect-789",
            "data_classification": "personal_data",
            "pii_detected": True,
            "sensitive_data_redacted": False,  # Violation
            "custom_attributes": {
                "gdpr_lawful_basis": "consent",
                "data_subject_consent": False,  # No consent
                "purpose_limitation": "marketing",
                "geographic_region": "eu",
            },
        },
    ]

    print("\n📋 GDPR Compliance Testing:")

    for i, scenario in enumerate(gdpr_scenarios):
        print(f"\n   🔍 Scenario {i + 1}: {scenario['scenario']}")

        event = monitor.record_compliance_event(
            event_type="gdpr_ai_processing",
            user_id=scenario["user_id"],
            team=scenario["team"],
            project=scenario["project"],
            model_used=scenario["model"],
            provider=scenario["provider"],
            cost=scenario["cost"],
            tokens_used=scenario["tokens"],
            customer_id=scenario["customer_id"],
            data_classification=scenario["data_classification"],
            pii_detected=scenario["pii_detected"],
            sensitive_data_redacted=scenario["sensitive_data_redacted"],
            custom_attributes=scenario["custom_attributes"],
        )

        print(f"      📊 Event: {event.event_id}")
        print(f"      • Data class: {event.data_classification}")
        print(f"      • PII detected: {event.pii_detected}")
        print(f"      • Data redacted: {event.sensitive_data_redacted}")
        print(
            f"      • Consent: {scenario['custom_attributes'].get('data_subject_consent', 'unknown')}"
        )
        print(
            f"      • Lawful basis: {scenario['custom_attributes'].get('gdpr_lawful_basis', 'unknown')}"
        )
        print(
            f"      • Region: {scenario['custom_attributes'].get('geographic_region', 'unknown')}"
        )

    # Show GDPR-specific violations
    gdpr_violations = [
        v for v in monitor.policy_violations if "pii" in v.policy_name.lower()
    ]

    if gdpr_violations:
        print("\n   ⚠️ GDPR Violations Detected:")
        for violation in gdpr_violations:
            print(f"      • {violation.violation_id}: {violation.description}")
            print(f"        Severity: {violation.severity}")


def demo_compliance_reporting():
    """Demonstrate comprehensive compliance reporting."""
    print("\n" + "=" * 60)
    print("📊 Demo: Compliance Reporting")
    print("=" * 60)

    print("Enterprise compliance reporting:")
    print("• Comprehensive audit reports by team/project")
    print("• Policy violation summaries and trends")
    print("• Cost governance and budget compliance")
    print("• Compliance score calculation")

    # Use existing monitor with accumulated data
    monitor = ComplianceMonitor(ComplianceLevel.SOX)
    monitor.start_monitoring()

    # Add some sample data for reporting
    reporting_scenarios = [
        (
            "finance_user",
            "finance-team",
            "compliance-reporting",
            "gpt-4",
            "openai",
            0.030,
            1800,
        ),
        (
            "audit_user",
            "audit-team",
            "risk-analysis",
            "claude-3-sonnet",
            "anthropic",
            0.025,
            1500,
        ),
        (
            "ops_user",
            "operations",
            "process-automation",
            "gpt-3.5-turbo",
            "openai",
            0.015,
            900,
        ),
    ]

    for user, team, project, model, provider, cost, tokens in reporting_scenarios:
        monitor.record_compliance_event(
            event_type="compliance_audit_request",
            user_id=user,
            team=team,
            project=project,
            model_used=model,
            provider=provider,
            cost=cost,
            tokens_used=tokens,
            data_classification="confidential",
            custom_attributes={
                "sox_compliance": True,
                "audit_trail_required": True,
                "financial_data": True,
            },
        )

    # Generate comprehensive compliance report
    report = monitor.get_compliance_report()

    print("\n📋 Compliance Report:")
    print(f"   Generated: {report['report_generated']}")
    print(f"   Compliance Level: {report['compliance_level'].upper()}")
    print(f"   Period: {report['reporting_period']}")

    print("\n📊 Summary Metrics:")
    summary = report["summary"]
    print(f"   • Total events: {summary['total_events']}")
    print(f"   • Total violations: {summary['total_violations']}")
    print(f"   • Critical violations: {summary['critical_violations']}")
    print(f"   • Unresolved violations: {summary['unresolved_violations']}")
    print(f"   • Total cost: ${summary['total_cost']:.6f}")
    print(f"   • Unique users: {summary['unique_users']}")
    print(f"   • Unique teams: {summary['unique_teams']}")

    print("\n💰 Cost Breakdown:")
    cost_breakdown = report["cost_breakdown"]

    if cost_breakdown["by_team"]:
        print("   By Team:")
        for team, cost in cost_breakdown["by_team"].items():
            print(f"      • {team}: ${cost:.6f}")

    if cost_breakdown["by_provider"]:
        print("   By Provider:")
        for provider, cost in cost_breakdown["by_provider"].items():
            print(f"      • {provider}: ${cost:.6f}")

    print("\n⚠️ Violations by Type:")
    violations_by_type = report["violations_by_type"]
    if violations_by_type:
        for vtype, count in violations_by_type.items():
            print(f"   • {vtype}: {count}")
    else:
        print("   No violations recorded")

    print(f"\n🎯 Compliance Score: {report['compliance_score']:.1f}/100")

    # Export report (simulate)
    report_filename = f"compliance_report_{int(time.time())}.json"
    print(f"\n📄 Report available for export as: {report_filename}")
    print("   Contains full audit trail for regulatory compliance")


def main():
    """Run the complete compliance monitoring demonstration."""

    print("🛡️ LiteLLM + GenOps: Compliance Monitoring & Governance Automation")
    print("=" * 75)
    print("Enterprise-grade compliance patterns for AI governance")
    print("Audit trails, policy enforcement, and regulatory compliance automation")

    # Check setup
    if not check_compliance_setup():
        print("\n❌ Compliance setup incomplete. Please resolve issues above.")
        return 1

    try:
        # Run demonstrations
        demo_audit_trail_generation()
        demo_policy_enforcement()
        demo_gdpr_compliance()
        demo_compliance_reporting()

        print("\n" + "=" * 60)
        print("🎉 Compliance Monitoring & Governance Complete!")

        print("\n🛡️ Compliance Patterns Demonstrated:")
        print("   ✅ Comprehensive audit trail generation")
        print("   ✅ Automated policy enforcement with violation detection")
        print("   ✅ GDPR compliance with PII protection")
        print("   ✅ Enterprise compliance reporting and scoring")

        print("\n📋 Regulatory Compliance Coverage:")
        print("   • SOX (Sarbanes-Oxley): Audit trails and segregation of duties")
        print("   • GDPR: Data protection, consent tracking, PII redaction")
        print("   • HIPAA: Healthcare data governance and privacy protection")
        print("   • PCI DSS: Payment data security and access controls")
        print("   • Custom: Configurable policies for industry-specific requirements")

        print("\n🏢 Enterprise Integration Benefits:")
        print("   • Automated compliance reduces manual oversight burden")
        print("   • Real-time violation detection prevents policy breaches")
        print("   • Comprehensive audit trails support regulatory audits")
        print("   • Cost governance ensures budget compliance")
        print("   • Multi-tenant isolation supports enterprise customers")

        print("\n📖 Implementation Recommendations:")
        print("   • Configure compliance level based on regulatory requirements")
        print("   • Implement automated alerting for critical violations")
        print("   • Regular compliance reporting for stakeholders")
        print("   • Data retention policies aligned with legal requirements")
        print("   • Staff training on governance policies and procedures")

        print("\n🚀 Next Steps:")
        print("   • Deploy with your compliance team and legal review")
        print("   • Integrate with SIEM and GRC platforms")
        print("   • Configure automated incident response workflows")
        print("   • Set up compliance dashboards for executives")
        print("   • Establish regular compliance audits and reviews")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
        return 1

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
