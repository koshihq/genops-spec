#!/usr/bin/env python3
"""
Example: Enterprise Governance with Policy Enforcement

Complexity: ⭐⭐⭐ Advanced

This example demonstrates enterprise-grade governance patterns including
budget enforcement, policy compliance, audit logging, and comprehensive
compliance monitoring for Flowise deployments.

Prerequisites:
- Flowise instance running
- GenOps package installed
- Understanding of enterprise governance requirements

Usage:
    python 06_enterprise_governance.py

Environment Variables:
    FLOWISE_BASE_URL: Flowise instance URL
    FLOWISE_API_KEY: API key
    GENOPS_TEAM: Team name for governance
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Optional

from genops.providers.flowise import instrument_flowise
from genops.providers.flowise_validation import validate_flowise_setup

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PolicyViolationLevel(Enum):
    """Levels of policy violations."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class GovernanceAction(Enum):
    """Actions that can be taken on policy violations."""

    LOG_ONLY = "log_only"
    WARN_USER = "warn_user"
    THROTTLE_REQUEST = "throttle_request"
    BLOCK_REQUEST = "block_request"
    ESCALATE_ALERT = "escalate_alert"


@dataclass
class PolicyViolation:
    """Represents a governance policy violation."""

    policy_name: str
    violation_level: PolicyViolationLevel
    message: str
    suggested_action: GovernanceAction
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class GovernancePolicy:
    """Defines an enterprise governance policy."""

    name: str
    description: str
    validator: Callable[[dict[str, Any]], list[PolicyViolation]]
    enabled: bool = True
    applies_to: list[str] = field(
        default_factory=lambda: ["all"]
    )  # teams, projects, or "all"


@dataclass
class AuditLogEntry:
    """Audit log entry for governance tracking."""

    timestamp: datetime
    event_type: str
    user_id: Optional[str]
    team: str
    project: str
    resource_type: str
    resource_id: str
    action: str
    result: str
    cost: Optional[Decimal]
    policy_violations: list[PolicyViolation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class EnterpriseGovernanceEngine:
    """Enterprise governance engine for Flowise with policy enforcement."""

    def __init__(self, organization_name: str = "Enterprise Corp"):
        self.organization_name = organization_name
        self.policies: list[GovernancePolicy] = []
        self.audit_log: list[AuditLogEntry] = []

        # Budget and compliance tracking
        self.department_budgets: dict[str, dict[str, Any]] = {}
        self.compliance_frameworks = ["SOX", "GDPR", "HIPAA", "SOC2"]

        # Initialize default enterprise policies
        self._initialize_default_policies()

        logger.info(f"Initialized enterprise governance for {organization_name}")

    def _initialize_default_policies(self):
        """Initialize standard enterprise governance policies."""

        # Budget enforcement policy
        self.add_policy(
            GovernancePolicy(
                name="budget_enforcement",
                description="Enforce department and project budget limits",
                validator=self._validate_budget_compliance,
            )
        )

        # Data classification policy
        self.add_policy(
            GovernancePolicy(
                name="data_classification",
                description="Ensure proper handling of sensitive data",
                validator=self._validate_data_classification,
            )
        )

        # Access control policy
        self.add_policy(
            GovernancePolicy(
                name="access_control",
                description="Validate user permissions and access patterns",
                validator=self._validate_access_control,
            )
        )

        # Cost optimization policy
        self.add_policy(
            GovernancePolicy(
                name="cost_optimization",
                description="Flag inefficient or expensive usage patterns",
                validator=self._validate_cost_optimization,
            )
        )

        # Compliance policy
        self.add_policy(
            GovernancePolicy(
                name="compliance_monitoring",
                description="Monitor compliance with regulatory frameworks",
                validator=self._validate_compliance_requirements,
            )
        )

    def add_policy(self, policy: GovernancePolicy):
        """Add a governance policy to the engine."""
        self.policies.append(policy)
        logger.info(f"Added governance policy: {policy.name}")

    def set_department_budget(
        self,
        department: str,
        monthly_budget: Decimal,
        alert_threshold_pct: float = 80.0,
        hard_limit_pct: float = 100.0,
    ):
        """Set budget limits for a department."""
        self.department_budgets[department] = {
            "monthly_budget": monthly_budget,
            "current_spend": Decimal("0.0"),
            "alert_threshold_pct": alert_threshold_pct,
            "hard_limit_pct": hard_limit_pct,
            "last_reset": datetime.now().replace(day=1),  # First of month
        }
        logger.info(f"Set monthly budget for {department}: ${monthly_budget}")

    def evaluate_request(
        self,
        team: str,
        project: str,
        user_id: Optional[str],
        request_data: dict[str, Any],
    ) -> tuple[bool, list[PolicyViolation]]:
        """Evaluate a request against all governance policies."""

        all_violations = []
        request_blocked = False

        # Prepare context for policy evaluation
        context = {
            "team": team,
            "project": project,
            "user_id": user_id,
            "timestamp": datetime.now(),
            **request_data,
        }

        # Evaluate each enabled policy
        for policy in self.policies:
            if not policy.enabled:
                continue

            # Check if policy applies to this team/project
            if policy.applies_to != ["all"]:
                if team not in policy.applies_to and project not in policy.applies_to:
                    continue

            try:
                violations = policy.validator(context)
                all_violations.extend(violations)

                # Check if any critical violations require blocking
                for violation in violations:
                    if violation.suggested_action == GovernanceAction.BLOCK_REQUEST:
                        request_blocked = True

            except Exception as e:
                logger.error(f"Policy evaluation failed for {policy.name}: {e}")
                # Don't block on policy evaluation errors

        return not request_blocked, all_violations

    def log_audit_event(
        self,
        event_type: str,
        team: str,
        project: str,
        resource_type: str,
        resource_id: str,
        action: str,
        result: str,
        user_id: Optional[str] = None,
        cost: Optional[Decimal] = None,
        policy_violations: list[PolicyViolation] = None,
        **metadata,
    ):
        """Log an audit event for compliance tracking."""

        entry = AuditLogEntry(
            timestamp=datetime.now(),
            event_type=event_type,
            user_id=user_id,
            team=team,
            project=project,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            result=result,
            cost=cost,
            policy_violations=policy_violations or [],
            metadata=metadata,
        )

        self.audit_log.append(entry)

        # Log violations at appropriate levels
        for violation in policy_violations or []:
            log_func = {
                PolicyViolationLevel.INFO: logger.info,
                PolicyViolationLevel.WARNING: logger.warning,
                PolicyViolationLevel.ERROR: logger.error,
                PolicyViolationLevel.CRITICAL: logger.critical,
            }[violation.violation_level]

            log_func(f"Policy violation [{violation.policy_name}]: {violation.message}")

    # Policy validators

    def _validate_budget_compliance(
        self, context: dict[str, Any]
    ) -> list[PolicyViolation]:
        """Validate budget compliance for department/team."""
        violations = []
        team = context.get("team", "unknown")

        # Check if team has budget configuration
        if team in self.department_budgets:
            budget_info = self.department_budgets[team]

            # Reset monthly spend if new month
            current_month = datetime.now().replace(day=1)
            if budget_info["last_reset"] < current_month:
                budget_info["current_spend"] = Decimal("0.0")
                budget_info["last_reset"] = current_month

            # Calculate budget utilization
            budget_pct = (
                budget_info["current_spend"] / budget_info["monthly_budget"]
            ) * 100

            # Check thresholds
            if budget_pct >= budget_info["hard_limit_pct"]:
                violations.append(
                    PolicyViolation(
                        policy_name="budget_enforcement",
                        violation_level=PolicyViolationLevel.CRITICAL,
                        message=f"Team {team} has exceeded hard budget limit ({budget_pct:.1f}%)",
                        suggested_action=GovernanceAction.BLOCK_REQUEST,
                        context={"budget_utilization": budget_pct},
                    )
                )
            elif budget_pct >= budget_info["alert_threshold_pct"]:
                violations.append(
                    PolicyViolation(
                        policy_name="budget_enforcement",
                        violation_level=PolicyViolationLevel.WARNING,
                        message=f"Team {team} approaching budget limit ({budget_pct:.1f}%)",
                        suggested_action=GovernanceAction.WARN_USER,
                        context={"budget_utilization": budget_pct},
                    )
                )

        return violations

    def _validate_data_classification(
        self, context: dict[str, Any]
    ) -> list[PolicyViolation]:
        """Validate proper handling of sensitive data."""
        violations = []

        question = context.get("question", "").lower()

        # Check for sensitive data patterns
        sensitive_patterns = [
            ("ssn", "social security number"),
            ("credit card", "credit card information"),
            ("password", "authentication credentials"),
            ("medical", "healthcare information"),
            ("patient", "healthcare information"),
            ("diagnosis", "healthcare information"),
        ]

        for pattern, data_type in sensitive_patterns:
            if pattern in question:
                violations.append(
                    PolicyViolation(
                        policy_name="data_classification",
                        violation_level=PolicyViolationLevel.WARNING,
                        message=f"Request may contain {data_type} - ensure proper data handling",
                        suggested_action=GovernanceAction.LOG_ONLY,
                        context={"detected_pattern": pattern, "data_type": data_type},
                    )
                )

        return violations

    def _validate_access_control(
        self, context: dict[str, Any]
    ) -> list[PolicyViolation]:
        """Validate user access permissions."""
        violations = []

        user_id = context.get("user_id")
        context.get("team", "unknown")

        # Simulate access control validation
        if not user_id:
            violations.append(
                PolicyViolation(
                    policy_name="access_control",
                    violation_level=PolicyViolationLevel.ERROR,
                    message="Request missing user identification",
                    suggested_action=GovernanceAction.BLOCK_REQUEST,
                )
            )

        # Check for suspicious access patterns (simplified)
        hour = datetime.now().hour
        if hour < 6 or hour > 22:  # Outside business hours
            violations.append(
                PolicyViolation(
                    policy_name="access_control",
                    violation_level=PolicyViolationLevel.INFO,
                    message=f"After-hours access detected for user {user_id}",
                    suggested_action=GovernanceAction.LOG_ONLY,
                    context={"access_hour": hour},
                )
            )

        return violations

    def _validate_cost_optimization(
        self, context: dict[str, Any]
    ) -> list[PolicyViolation]:
        """Validate cost optimization and efficiency."""
        violations = []

        question = context.get("question", "")

        # Flag potentially expensive requests
        if len(question) > 2000:  # Very long requests
            violations.append(
                PolicyViolation(
                    policy_name="cost_optimization",
                    violation_level=PolicyViolationLevel.WARNING,
                    message=f"Large request detected ({len(question)} chars) - may incur high costs",
                    suggested_action=GovernanceAction.WARN_USER,
                    context={"request_length": len(question)},
                )
            )

        # Check for potentially inefficient patterns
        inefficient_patterns = [
            "summarize this entire document",
            "analyze all data",
            "process everything",
        ]
        for pattern in inefficient_patterns:
            if pattern in question.lower():
                violations.append(
                    PolicyViolation(
                        policy_name="cost_optimization",
                        violation_level=PolicyViolationLevel.INFO,
                        message=f"Potentially inefficient request pattern: {pattern}",
                        suggested_action=GovernanceAction.LOG_ONLY,
                        context={"inefficient_pattern": pattern},
                    )
                )

        return violations

    def _validate_compliance_requirements(
        self, context: dict[str, Any]
    ) -> list[PolicyViolation]:
        """Validate compliance with regulatory frameworks."""
        violations = []

        team = context.get("team", "unknown")
        question = context.get("question", "").lower()

        # GDPR compliance check
        if "personal data" in question or "pii" in question:
            violations.append(
                PolicyViolation(
                    policy_name="compliance_monitoring",
                    violation_level=PolicyViolationLevel.WARNING,
                    message="Request involves personal data - ensure GDPR compliance",
                    suggested_action=GovernanceAction.LOG_ONLY,
                    context={"compliance_framework": "GDPR"},
                )
            )

        # HIPAA compliance for healthcare teams
        if team.lower() in ["healthcare", "medical", "hospital"]:
            if any(
                term in question
                for term in ["patient", "medical", "health", "diagnosis"]
            ):
                violations.append(
                    PolicyViolation(
                        policy_name="compliance_monitoring",
                        violation_level=PolicyViolationLevel.WARNING,
                        message="Healthcare team accessing medical data - ensure HIPAA compliance",
                        suggested_action=GovernanceAction.LOG_ONLY,
                        context={"compliance_framework": "HIPAA"},
                    )
                )

        return violations

    def generate_compliance_report(self, days: int = 30) -> dict[str, Any]:
        """Generate compliance report for audit purposes."""

        cutoff_date = datetime.now() - timedelta(days=days)
        recent_entries = [
            entry for entry in self.audit_log if entry.timestamp >= cutoff_date
        ]

        # Aggregate statistics
        total_requests = len(recent_entries)
        total_violations = sum(len(entry.policy_violations) for entry in recent_entries)
        total_cost = sum(entry.cost for entry in recent_entries if entry.cost)

        # Violation breakdown
        violation_counts = {}
        for entry in recent_entries:
            for violation in entry.policy_violations:
                key = f"{violation.policy_name}_{violation.violation_level.value}"
                violation_counts[key] = violation_counts.get(key, 0) + 1

        # Team activity breakdown
        team_stats = {}
        for entry in recent_entries:
            team = entry.team
            if team not in team_stats:
                team_stats[team] = {
                    "requests": 0,
                    "violations": 0,
                    "cost": Decimal("0.0"),
                }
            team_stats[team]["requests"] += 1
            team_stats[team]["violations"] += len(entry.policy_violations)
            team_stats[team]["cost"] += entry.cost or Decimal("0.0")

        return {
            "report_period_days": days,
            "organization": self.organization_name,
            "summary": {
                "total_requests": total_requests,
                "total_violations": total_violations,
                "total_cost": float(total_cost),
                "violation_rate": (total_violations / total_requests * 100)
                if total_requests > 0
                else 0,
            },
            "violation_breakdown": violation_counts,
            "team_statistics": {
                team: {
                    "requests": stats["requests"],
                    "violations": stats["violations"],
                    "cost": float(stats["cost"]),
                    "violation_rate": (stats["violations"] / stats["requests"] * 100)
                    if stats["requests"] > 0
                    else 0,
                }
                for team, stats in team_stats.items()
            },
            "compliance_frameworks_monitored": self.compliance_frameworks,
            "active_policies": len([p for p in self.policies if p.enabled]),
            "generated_at": datetime.now().isoformat(),
        }


def demonstrate_enterprise_governance():
    """Demonstrate enterprise governance with policy enforcement."""

    print("🏛️ Enterprise Governance with Policy Enforcement")
    print("=" * 60)

    # Configuration
    base_url = os.getenv("FLOWISE_BASE_URL", "http://localhost:3000")
    api_key = os.getenv("FLOWISE_API_KEY")

    # Step 1: Setup and validation
    print("📋 Step 1: Initializing enterprise governance...")

    try:
        result = validate_flowise_setup(base_url, api_key)
        if not result.is_valid:
            print("❌ Setup validation failed.")
            return False

        # Create governance engine
        governance = EnterpriseGovernanceEngine("TechCorp Industries")

        # Set up department budgets
        governance.set_department_budget("engineering", Decimal("5000.00"), 80.0, 100.0)
        governance.set_department_budget("marketing", Decimal("2000.00"), 75.0, 95.0)
        governance.set_department_budget("healthcare", Decimal("3000.00"), 70.0, 90.0)

        flowise = instrument_flowise(
            base_url=base_url,
            api_key=api_key,
            team="governance-demo",
            project="enterprise-compliance",
            environment="production",
        )

        chatflows = flowise.get_chatflows()
        if not chatflows:
            print("❌ No chatflows available.")
            return False

        chatflow_id = chatflows[0].get("id")
        print("✅ Enterprise governance initialized")

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        return False

    # Step 2: Test policy enforcement scenarios
    print("\n🔒 Step 2: Testing policy enforcement scenarios...")

    test_scenarios = [
        {
            "name": "Compliant Request",
            "team": "engineering",
            "project": "ai-assistant",
            "user_id": "john.doe@techcorp.com",
            "question": "How can we optimize our API performance?",
            "expected_violations": 0,
        },
        {
            "name": "Budget Alert Scenario",
            "team": "marketing",
            "project": "campaign-ai",
            "user_id": "jane.smith@techcorp.com",
            "question": "Generate comprehensive market analysis for all our products with detailed competitive intelligence and consumer behavior insights across all demographics and geographic regions",
            "expected_violations": 1,  # Cost optimization warning
        },
        {
            "name": "Data Classification Alert",
            "team": "engineering",
            "project": "user-data",
            "user_id": "bob.wilson@techcorp.com",
            "question": "Help me process customer SSN and credit card data for analysis",
            "expected_violations": 2,  # Sensitive data warnings
        },
        {
            "name": "Healthcare Compliance",
            "team": "healthcare",
            "project": "patient-care",
            "user_id": "dr.johnson@techcorp.com",
            "question": "Analyze patient medical records for diagnosis patterns",
            "expected_violations": 1,  # HIPAA compliance warning
        },
        {
            "name": "Access Control Violation",
            "team": "engineering",
            "project": "security-test",
            "user_id": None,  # Missing user ID
            "question": "Show me all user data",
            "expected_violations": 1,  # Access control error
        },
    ]

    for scenario in test_scenarios:
        print(f"\n   🧪 Testing: {scenario['name']}")

        # Evaluate request against policies
        request_data = {
            "question": scenario["question"],
            "chatflow_id": chatflow_id,
            "estimated_cost": Decimal("0.05"),  # Simulated cost
        }

        allowed, violations = governance.evaluate_request(
            scenario["team"], scenario["project"], scenario["user_id"], request_data
        )

        print(f"      Request allowed: {'✅ Yes' if allowed else '❌ No'}")
        print(f"      Violations found: {len(violations)}")

        for violation in violations:
            level_emoji = {
                PolicyViolationLevel.INFO: "💡",
                PolicyViolationLevel.WARNING: "⚠️",
                PolicyViolationLevel.ERROR: "❌",
                PolicyViolationLevel.CRITICAL: "🚨",
            }[violation.violation_level]

            print(
                f"        {level_emoji} [{violation.policy_name}] {violation.message}"
            )

        # Log audit event
        governance.log_audit_event(
            event_type="flowise_request",
            team=scenario["team"],
            project=scenario["project"],
            resource_type="chatflow",
            resource_id=chatflow_id,
            action="predict_flow",
            result="allowed" if allowed else "blocked",
            user_id=scenario["user_id"],
            cost=request_data.get("estimated_cost"),
            policy_violations=violations,
        )

        # Execute request if allowed (simulated)
        if allowed:
            print("      ✅ Request executed successfully")
        else:
            print("      🚫 Request blocked by governance policies")

    # Step 3: Generate compliance report
    print("\n📊 Step 3: Generating compliance report...")

    report = governance.generate_compliance_report(days=30)

    print(f"\n📋 Compliance Report for {report['organization']}:")
    print(f"   Report Period: {report['report_period_days']} days")
    print(f"   Total Requests: {report['summary']['total_requests']}")
    print(f"   Total Violations: {report['summary']['total_violations']}")
    print(f"   Violation Rate: {report['summary']['violation_rate']:.2f}%")
    print(f"   Total Cost: ${report['summary']['total_cost']:.2f}")

    print("\n   Policy Violations by Type:")
    for violation_type, count in report["violation_breakdown"].items():
        policy_name, level = violation_type.rsplit("_", 1)
        print(f"     {policy_name} ({level}): {count}")

    print("\n   Team Statistics:")
    for team, stats in report["team_statistics"].items():
        print(f"     {team}:")
        print(f"       Requests: {stats['requests']}")
        print(
            f"       Violations: {stats['violations']} ({stats['violation_rate']:.1f}%)"
        )
        print(f"       Cost: ${stats['cost']:.2f}")

    print(
        f"\n   Compliance Frameworks Monitored: {', '.join(report['compliance_frameworks_monitored'])}"
    )
    print(f"   Active Policies: {report['active_policies']}")

    return True


def demonstrate_advanced_governance_patterns():
    """Show advanced enterprise governance patterns."""

    print("\n🔬 Advanced Enterprise Governance Patterns")
    print("=" * 60)

    patterns = [
        {
            "name": "Dynamic Policy Configuration",
            "description": "Policies that adapt based on context and risk assessment",
            "use_cases": [
                "Time-based access controls (stricter after hours)",
                "Risk-based authentication requirements",
                "Dynamic budget allocation based on business priority",
            ],
        },
        {
            "name": "Automated Compliance Reporting",
            "description": "Scheduled compliance reports for regulatory frameworks",
            "use_cases": [
                "SOX compliance quarterly reports",
                "GDPR data processing activity reports",
                "HIPAA access audit logs",
            ],
        },
        {
            "name": "Policy Exception Management",
            "description": "Structured process for handling policy exceptions",
            "use_cases": [
                "Emergency access during outages",
                "Executive override for critical business needs",
                "Temporary policy suspension with approval workflow",
            ],
        },
        {
            "name": "Cross-System Policy Enforcement",
            "description": "Consistent policies across all AI/ML systems",
            "use_cases": [
                "Unified data classification across platforms",
                "Consistent access controls for AI services",
                "Centralized budget management across tools",
            ],
        },
    ]

    for pattern in patterns:
        print(f"\n📋 {pattern['name']}:")
        print(f"   Description: {pattern['description']}")
        print("   Use Cases:")
        for use_case in pattern["use_cases"]:
            print(f"     • {use_case}")

    print("\n💡 Implementation Best Practices:")
    print("   • Start with basic policies and evolve based on violations")
    print("   • Implement graduated responses (warn → throttle → block)")
    print("   • Maintain comprehensive audit logs for compliance")
    print("   • Regular policy review and updates based on business needs")
    print("   • Integration with existing enterprise security systems")
    print("   • Automated alerting and escalation workflows")


def main():
    """Main example function."""

    try:
        print("🚀 Enterprise Governance with Policy Enforcement Example")
        print("=" * 70)

        # Run main demonstration
        success = demonstrate_enterprise_governance()

        if success:
            # Show advanced patterns
            demonstrate_advanced_governance_patterns()

            print("\n🎉 Enterprise Governance Example Complete!")
            print("=" * 50)
            print("✅ You've learned how to:")
            print("   • Implement enterprise-grade governance policies")
            print("   • Enforce budget limits and cost controls")
            print("   • Monitor compliance with regulatory frameworks")
            print("   • Generate comprehensive audit logs")
            print("   • Handle policy violations with graduated responses")

            print("\n🏛️ Enterprise Features Demonstrated:")
            print("   • Multi-tier policy enforcement engine")
            print("   • Automated compliance monitoring and reporting")
            print("   • Comprehensive audit logging for all activities")
            print("   • Budget enforcement with configurable thresholds")
            print("   • Data classification and access control policies")

            print("\n📚 Next Steps:")
            print("   • Integrate with enterprise identity management systems")
            print("   • Set up automated compliance reporting workflows")
            print("   • Configure alerting and escalation for policy violations")
            print("   • Explore production monitoring (07_production_monitoring.py)")

        return success

    except Exception as e:
        logger.error(f"Example failed: {e}")
        return False


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Example interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)
