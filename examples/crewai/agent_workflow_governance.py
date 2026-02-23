#!/usr/bin/env python3
"""
Agent Workflow Governance and Advanced Monitoring

Advanced governance for CrewAI multi-agent workflows with comprehensive monitoring,
compliance tracking, and intelligent decision analysis.

Usage:
    python agent_workflow_governance.py [--governance-mode MODE] [--compliance-level LEVEL]

Features:
    - Multi-agent decision tracking and audit trails
    - Compliance monitoring and policy enforcement
    - Agent collaboration pattern analysis
    - Workflow decision transparency and explainability
    - Real-time governance alerts and interventions
    - Cross-crew governance aggregation and reporting

Time to Complete: ~30 minutes
Learning Outcomes: Enterprise-grade governance and compliance for AI systems
"""

import argparse
import logging
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any

# Core CrewAI imports
try:
    from crewai import Agent, Crew, Task
    from crewai.process import Process
except ImportError:
    print("❌ CrewAI not installed. Install with: pip install crewai")
    sys.exit(1)

# GenOps imports
try:
    from genops.providers.crewai import (
        CrewAIAgentMonitor,
        GenOpsCrewAIAdapter,
        get_multi_agent_insights,  # noqa: F401
        print_validation_result,
        validate_crewai_setup,
    )
except ImportError:
    print("❌ GenOps not installed. Install with: pip install genops-ai[crewai]")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GovernanceLevel(Enum):
    """Governance enforcement levels."""

    MONITORING = "monitoring"
    ADVISORY = "advisory"
    ENFORCED = "enforced"
    STRICT = "strict"


class ComplianceStatus(Enum):
    """Compliance check status."""

    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    CRITICAL = "critical"


@dataclass
class GovernancePolicy:
    """Governance policy definition."""

    id: str
    name: str
    description: str
    category: str
    enforcement_level: GovernanceLevel
    rules: dict[str, Any]
    violation_actions: list[str]


@dataclass
class ComplianceCheck:
    """Compliance check result."""

    policy_id: str
    status: ComplianceStatus
    details: str
    timestamp: datetime
    agent_id: str
    task_id: str
    remediation_required: bool


@dataclass
class AgentDecision:
    """Agent decision audit record."""

    decision_id: str
    agent_id: str
    agent_role: str
    task_id: str
    decision_type: str
    decision_data: dict[str, Any]
    reasoning: str
    confidence_score: float
    timestamp: datetime
    governance_approval: bool


@dataclass
class WorkflowAuditEntry:
    """Workflow audit trail entry."""

    entry_id: str
    crew_id: str
    workflow_stage: str
    action: str
    actor: str  # agent or system
    details: dict[str, Any]
    timestamp: datetime
    compliance_impact: str


class GovernanceEngine:
    """Advanced governance engine for CrewAI workflows."""

    def __init__(
        self, governance_mode: str = "advisory", compliance_level: str = "standard"
    ):
        self.governance_mode = GovernanceLevel(governance_mode)
        self.compliance_level = compliance_level
        self.policies = self._initialize_policies()
        self.audit_trail = []
        self.compliance_history = []
        self.decision_log = []

        self.adapter = GenOpsCrewAIAdapter(
            team="governance-demo",
            project="workflow-governance",
            daily_budget_limit=75.0,
            governance_policy=governance_mode,
            enable_agent_tracking=True,
            enable_task_tracking=True,
        )
        self.monitor = CrewAIAgentMonitor()

    def _initialize_policies(self) -> list[GovernancePolicy]:
        """Initialize default governance policies."""
        policies = [
            GovernancePolicy(
                id="cost_control",
                name="Cost Control Policy",
                description="Ensure agent operations stay within budget limits",
                category="financial",
                enforcement_level=GovernanceLevel.ENFORCED,
                rules={
                    "max_task_cost": 0.50,
                    "daily_budget_limit": 50.0,
                    "cost_alert_threshold": 0.80,
                },
                violation_actions=["alert", "task_suspension", "workflow_pause"],
            ),
            GovernancePolicy(
                id="data_privacy",
                name="Data Privacy Policy",
                description="Protect sensitive information in agent interactions",
                category="security",
                enforcement_level=GovernanceLevel.STRICT,
                rules={
                    "no_pii_logging": True,
                    "data_retention_days": 30,
                    "anonymize_outputs": True,
                },
                violation_actions=[
                    "immediate_stop",
                    "data_redaction",
                    "incident_report",
                ],
            ),
            GovernancePolicy(
                id="quality_assurance",
                name="Quality Assurance Policy",
                description="Maintain quality standards for agent outputs",
                category="quality",
                enforcement_level=GovernanceLevel.ADVISORY,
                rules={
                    "min_confidence_score": 0.70,
                    "require_reasoning": True,
                    "output_validation": True,
                },
                violation_actions=["quality_warning", "human_review", "retry_task"],
            ),
            GovernancePolicy(
                id="ethical_guidelines",
                name="Ethical AI Guidelines",
                description="Ensure ethical AI usage and decision making",
                category="ethics",
                enforcement_level=GovernanceLevel.ENFORCED,
                rules={
                    "bias_detection": True,
                    "fairness_check": True,
                    "transparency_required": True,
                },
                violation_actions=["ethics_review", "decision_override", "escalation"],
            ),
            GovernancePolicy(
                id="operational_limits",
                name="Operational Limits Policy",
                description="Prevent excessive resource usage and system overload",
                category="operations",
                enforcement_level=GovernanceLevel.MONITORING,
                rules={
                    "max_execution_time": 300,  # 5 minutes
                    "max_concurrent_agents": 10,
                    "resource_utilization_limit": 0.80,
                },
                violation_actions=["performance_alert", "scaling_recommendation"],
            ),
        ]
        return policies

    def setup_validation(self) -> bool:
        """Validate setup for governance monitoring."""
        print("🔍 Validating governance and compliance setup...")

        result = validate_crewai_setup(quick=False)

        if result.is_valid:
            print("✅ Governance setup validated")
            print(f"   🛡️ Governance mode: {self.governance_mode.value}")
            print(f"   📋 Compliance level: {self.compliance_level}")
            print(f"   📜 Active policies: {len(self.policies)}")
            return True
        else:
            print("❌ Setup issues found:")
            print_validation_result(result)
            return False

    def create_governed_crew(self, use_case: str) -> Crew:
        """Create a crew with comprehensive governance monitoring."""
        print(f"\n🏗️ Creating governed crew for {use_case}...")

        # Compliance Officer Agent
        compliance_officer = Agent(
            role="Compliance Officer",
            goal="Ensure all actions comply with organizational policies",
            backstory="""Expert in organizational compliance with deep understanding
                         of policies, regulations, and ethical guidelines. Responsible
                         for monitoring and ensuring adherence to governance standards.""",
            verbose=True,
        )

        # Senior Analyst Agent
        senior_analyst = Agent(
            role="Senior Business Analyst",
            goal="Provide thorough analysis with documented reasoning",
            backstory="""Experienced business analyst with expertise in market research,
                         data analysis, and strategic recommendations. Focused on
                         delivering high-quality insights with clear methodology.""",
            verbose=True,
        )

        # Decision Maker Agent
        decision_maker = Agent(
            role="Strategic Decision Maker",
            goal="Make informed decisions based on comprehensive analysis",
            backstory="""Executive-level decision maker with extensive experience
                         in strategic planning and risk assessment. Responsible for
                         final decisions with full accountability and transparency.""",
            verbose=True,
        )

        # Quality Reviewer Agent
        quality_reviewer = Agent(
            role="Quality Assurance Reviewer",
            goal="Validate quality and accuracy of all work products",
            backstory="""Quality assurance specialist focused on maintaining high
                         standards for accuracy, completeness, and professional
                         presentation of all deliverables.""",
            verbose=True,
        )

        # Define governance-aware tasks
        tasks = [
            Task(
                description=f"""Review the proposed {use_case} initiative for compliance
                              with all organizational policies. Check for ethical considerations,
                              regulatory requirements, and risk factors. Document any
                              compliance concerns and provide recommendations.""",
                agent=compliance_officer,
            ),
            Task(
                description=f"""Conduct comprehensive analysis of {use_case}. Include
                              market research, competitive analysis, financial projections,
                              and risk assessment. Provide clear methodology and supporting
                              evidence for all conclusions.""",
                agent=senior_analyst,
            ),
            Task(
                description=f"""Based on compliance review and business analysis, make
                              strategic decisions regarding {use_case}. Document decision
                              rationale, consider alternative options, and identify key
                              success metrics and risk mitigation strategies.""",
                agent=decision_maker,
            ),
            Task(
                description="""Review all work products for quality, accuracy, and
                              completeness. Verify compliance with standards and ensure
                              professional presentation. Provide final quality assessment
                              and recommendations for improvement.""",
                agent=quality_reviewer,
            ),
        ]

        crew = Crew(
            agents=[
                compliance_officer,
                senior_analyst,
                decision_maker,
                quality_reviewer,
            ],
            tasks=tasks,
            process=Process.sequential,
            verbose=2,
        )

        print(f"✅ Created governed crew with {len(crew.agents)} specialized agents")
        return crew

    def monitor_compliance(
        self, crew_id: str, agent_id: str, task_id: str, action_data: dict[str, Any]
    ) -> list[ComplianceCheck]:
        """Monitor agent actions for compliance violations."""
        compliance_results = []

        for policy in self.policies:
            check_result = self._check_policy_compliance(
                policy, agent_id, task_id, action_data
            )
            compliance_results.append(check_result)

            # Log compliance check
            if check_result.status in [
                ComplianceStatus.VIOLATION,
                ComplianceStatus.CRITICAL,
            ]:
                self._log_audit_entry(
                    crew_id=crew_id,
                    workflow_stage="compliance_check",
                    action="policy_violation",
                    actor=f"agent_{agent_id}",
                    details={
                        "policy_id": policy.id,
                        "violation_type": check_result.status.value,
                        "details": check_result.details,
                    },
                    compliance_impact="negative",
                )

        return compliance_results

    def _check_policy_compliance(
        self,
        policy: GovernancePolicy,
        agent_id: str,
        task_id: str,
        action_data: dict[str, Any],
    ) -> ComplianceCheck:
        """Check specific policy compliance."""
        timestamp = datetime.now()

        # Cost control policy checks
        if policy.id == "cost_control":
            task_cost = action_data.get("estimated_cost", 0.0)
            if task_cost > policy.rules["max_task_cost"]:
                return ComplianceCheck(
                    policy_id=policy.id,
                    status=ComplianceStatus.VIOLATION,
                    details=f"Task cost ${task_cost:.4f} exceeds limit ${policy.rules['max_task_cost']}",
                    timestamp=timestamp,
                    agent_id=agent_id,
                    task_id=task_id,
                    remediation_required=True,
                )

        # Data privacy policy checks
        elif policy.id == "data_privacy":
            if action_data.get("contains_pii", False):
                return ComplianceCheck(
                    policy_id=policy.id,
                    status=ComplianceStatus.CRITICAL,
                    details="Personal identifiable information detected in agent output",
                    timestamp=timestamp,
                    agent_id=agent_id,
                    task_id=task_id,
                    remediation_required=True,
                )

        # Quality assurance policy checks
        elif policy.id == "quality_assurance":
            confidence = action_data.get("confidence_score", 1.0)
            if confidence < policy.rules["min_confidence_score"]:
                return ComplianceCheck(
                    policy_id=policy.id,
                    status=ComplianceStatus.WARNING,
                    details=f"Confidence score {confidence:.2f} below threshold {policy.rules['min_confidence_score']}",
                    timestamp=timestamp,
                    agent_id=agent_id,
                    task_id=task_id,
                    remediation_required=False,
                )

        # Default: compliant
        return ComplianceCheck(
            policy_id=policy.id,
            status=ComplianceStatus.COMPLIANT,
            details="No policy violations detected",
            timestamp=timestamp,
            agent_id=agent_id,
            task_id=task_id,
            remediation_required=False,
        )

    def log_agent_decision(
        self,
        agent_id: str,
        agent_role: str,
        task_id: str,
        decision_type: str,
        decision_data: dict[str, Any],
        reasoning: str,
        confidence: float,
    ) -> str:
        """Log agent decision for audit trail."""
        decision_id = str(uuid.uuid4())

        # Check governance approval
        governance_approval = self._evaluate_governance_approval(
            decision_data, confidence, agent_role
        )

        decision = AgentDecision(
            decision_id=decision_id,
            agent_id=agent_id,
            agent_role=agent_role,
            task_id=task_id,
            decision_type=decision_type,
            decision_data=decision_data,
            reasoning=reasoning,
            confidence_score=confidence,
            timestamp=datetime.now(),
            governance_approval=governance_approval,
        )

        self.decision_log.append(decision)

        print(f"📝 Decision logged: {decision_type} by {agent_role}")
        print(f"   🆔 Decision ID: {decision_id}")
        print(f"   🎯 Confidence: {confidence:.2f}")
        print(f"   ✅ Governance approval: {'Yes' if governance_approval else 'No'}")

        return decision_id

    def _evaluate_governance_approval(
        self, decision_data: dict[str, Any], confidence: float, agent_role: str
    ) -> bool:
        """Evaluate if decision meets governance approval criteria."""
        # High-confidence decisions from senior roles get automatic approval
        if confidence >= 0.9 and "senior" in agent_role.lower():
            return True

        # Medium-confidence decisions need additional checks
        if 0.7 <= confidence < 0.9:
            # Check for risk factors
            risk_score = decision_data.get("risk_score", 0.0)
            if risk_score < 0.3:  # Low risk
                return True

        # Low-confidence or high-risk decisions need human review
        return False

    def _log_audit_entry(
        self,
        crew_id: str,
        workflow_stage: str,
        action: str,
        actor: str,
        details: dict[str, Any],
        compliance_impact: str,
    ):
        """Log entry to workflow audit trail."""
        entry = WorkflowAuditEntry(
            entry_id=str(uuid.uuid4()),
            crew_id=crew_id,
            workflow_stage=workflow_stage,
            action=action,
            actor=actor,
            details=details,
            timestamp=datetime.now(),
            compliance_impact=compliance_impact,
        )

        self.audit_trail.append(entry)

    def demonstrate_governance_workflow(self):
        """Demonstrate end-to-end governance workflow."""
        print("\n" + "=" * 70)
        print("🛡️ Governance Workflow Demonstration")
        print("=" * 70)

        use_case = "AI-powered customer service automation"
        crew = self.create_governed_crew(use_case)

        with self.adapter.track_crew(
            "governance-workflow", use_case=use_case, governance_enabled=True
        ) as context:
            crew_id = context.crew_id
            print(f"\n🎬 Starting governed workflow for crew {crew_id}")

            # Simulate governance monitoring during execution
            start_time = time.time()

            # Log workflow start
            self._log_audit_entry(
                crew_id=crew_id,
                workflow_stage="initialization",
                action="workflow_started",
                actor="system",
                details={"use_case": use_case, "agents_count": len(crew.agents)},
                compliance_impact="positive",
            )

            # Execute crew with governance monitoring
            print("   🔍 Monitoring compliance in real-time...")

            # Simulate agent decisions and compliance checks
            for i, agent in enumerate(crew.agents):
                agent_id = f"agent_{i}"
                task_id = f"task_{i}"

                # Simulate decision data
                decision_data = {
                    "estimated_cost": 0.15 + (i * 0.05),  # Increasing cost
                    "confidence_score": 0.85 - (i * 0.05),  # Decreasing confidence
                    "risk_score": 0.2 + (i * 0.1),  # Increasing risk
                    "contains_pii": i == 1,  # Second agent has PII issue
                }

                # Check compliance
                compliance_results = self.monitor_compliance(
                    crew_id, agent_id, task_id, decision_data
                )

                # Log agent decision
                self.log_agent_decision(
                    agent_id=agent_id,
                    agent_role=agent.role,
                    task_id=task_id,
                    decision_type="task_execution",
                    decision_data=decision_data,
                    reasoning=f"Executing {agent.goal}",
                    confidence=decision_data["confidence_score"],
                )

                # Handle violations
                violations = [
                    c
                    for c in compliance_results
                    if c.status
                    in [ComplianceStatus.VIOLATION, ComplianceStatus.CRITICAL]
                ]

                if violations:
                    print(f"   🚨 Compliance violations detected for {agent.role}:")
                    for violation in violations:
                        print(f"      • {violation.details}")

                        # Apply enforcement actions
                        self._apply_enforcement_actions(violation, crew_id)

            # Execute actual crew (simplified for demo)
            result = crew.kickoff(
                {
                    "governance_mode": self.governance_mode.value,
                    "compliance_monitoring": True,
                    "use_case": use_case,
                }
            )

            end_time = time.time()
            execution_time = end_time - start_time

            # Log workflow completion
            self._log_audit_entry(
                crew_id=crew_id,
                workflow_stage="completion",
                action="workflow_completed",
                actor="system",
                details={
                    "execution_time": execution_time,
                    "result_length": len(str(result)),
                    "governance_status": "monitored",
                },
                compliance_impact="neutral",
            )

            # Get final metrics
            metrics = context.get_metrics()

            print("\n📊 Governance Workflow Results:")
            print(f"   ⏱️ Execution time: {execution_time:.2f} seconds")
            print(f"   💰 Total cost: ${metrics['total_cost']:.6f}")
            print(f"   👥 Agents monitored: {len(crew.agents)}")
            print(f"   📋 Decisions logged: {len(self.decision_log)}")
            print(f"   📝 Audit entries: {len(self.audit_trail)}")

    def _apply_enforcement_actions(self, violation: ComplianceCheck, crew_id: str):
        """Apply enforcement actions for policy violations."""
        policy = next(p for p in self.policies if p.id == violation.policy_id)

        print(f"   ⚖️ Applying enforcement for {policy.name}:")

        for action in policy.violation_actions:
            if action == "alert":
                print(f"      🚨 ALERT: {violation.details}")

            elif (
                action == "immediate_stop"
                and policy.enforcement_level == GovernanceLevel.STRICT
            ):
                print("      🛑 IMMEDIATE STOP: Critical violation detected")
                # In real implementation, would halt execution

            elif action == "human_review":
                print("      👤 HUMAN REVIEW: Flagging for manual review")

            elif action == "data_redaction":
                print("      🗑️ DATA REDACTION: Removing sensitive information")

            # Log enforcement action
            self._log_audit_entry(
                crew_id=crew_id,
                workflow_stage="enforcement",
                action=action,
                actor="governance_system",
                details={
                    "policy_id": policy.id,
                    "violation_details": violation.details,
                    "enforcement_level": policy.enforcement_level.value,
                },
                compliance_impact="corrective",
            )

    def analyze_governance_effectiveness(self):
        """Analyze governance effectiveness and generate insights."""
        print("\n" + "=" * 70)
        print("📈 Governance Effectiveness Analysis")
        print("=" * 70)

        if not self.audit_trail or not self.decision_log:
            print("❌ Insufficient governance data for analysis")
            return

        # Compliance rate analysis
        total_decisions = len(self.decision_log)
        approved_decisions = len(
            [d for d in self.decision_log if d.governance_approval]
        )
        compliance_rate = (
            (approved_decisions / total_decisions) * 100 if total_decisions > 0 else 0
        )

        print("📊 Compliance Metrics:")
        print(f"   ✅ Total decisions: {total_decisions}")
        print(f"   👍 Approved decisions: {approved_decisions}")
        print(f"   📈 Compliance rate: {compliance_rate:.1f}%")

        # Violation analysis
        violations = [
            entry for entry in self.audit_trail if "violation" in entry.action
        ]
        violation_rate = (
            (len(violations) / len(self.audit_trail)) * 100 if self.audit_trail else 0
        )

        print(f"   🚨 Policy violations: {len(violations)}")
        print(f"   📉 Violation rate: {violation_rate:.1f}%")

        # Policy effectiveness
        print("\n📜 Policy Effectiveness:")
        for policy in self.policies:
            policy_violations = [v for v in violations if policy.id in str(v.details)]
            effectiveness = (
                ((total_decisions - len(policy_violations)) / total_decisions * 100)
                if total_decisions > 0
                else 100
            )
            print(f"   • {policy.name}: {effectiveness:.1f}% effective")

        # Agent governance performance
        print("\n👥 Agent Governance Performance:")
        agent_performance = {}
        for decision in self.decision_log:
            role = decision.agent_role
            if role not in agent_performance:
                agent_performance[role] = {"total": 0, "approved": 0}
            agent_performance[role]["total"] += 1
            if decision.governance_approval:
                agent_performance[role]["approved"] += 1

        for role, perf in agent_performance.items():
            approval_rate = (
                (perf["approved"] / perf["total"]) * 100 if perf["total"] > 0 else 0
            )
            print(
                f"   • {role}: {approval_rate:.1f}% approval rate ({perf['approved']}/{perf['total']})"
            )

        # Recommendations
        print("\n💡 Governance Recommendations:")

        if compliance_rate < 80:
            print("   🔴 Low compliance rate detected - consider policy training")
        elif compliance_rate < 90:
            print("   🟡 Moderate compliance - review policy clarity")
        else:
            print("   🟢 Good compliance rate - maintain current practices")

        if violation_rate > 10:
            print("   🔴 High violation rate - strengthen enforcement")
        elif violation_rate > 5:
            print("   🟡 Moderate violations - review policy effectiveness")
        else:
            print("   🟢 Low violation rate - governance working well")

    def generate_audit_report(self):
        """Generate comprehensive audit and compliance report."""
        print("\n" + "=" * 70)
        print("📄 Governance Audit Report")
        print("=" * 70)

        report_data = {
            "report_id": str(uuid.uuid4()),
            "generated_at": datetime.now().isoformat(),
            "governance_mode": self.governance_mode.value,
            "compliance_level": self.compliance_level,
            "total_policies": len(self.policies),
            "audit_entries": len(self.audit_trail),
            "decisions_logged": len(self.decision_log),
            "policies": [asdict(policy) for policy in self.policies],
            "recent_violations": [
                asdict(entry)
                for entry in self.audit_trail[-10:]
                if "violation" in entry.action
            ],
            "governance_summary": {
                "total_workflows": len({entry.crew_id for entry in self.audit_trail}),
                "enforcement_actions": len(
                    [
                        entry
                        for entry in self.audit_trail
                        if entry.workflow_stage == "enforcement"
                    ]
                ),
                "compliance_positive": len(
                    [
                        entry
                        for entry in self.audit_trail
                        if entry.compliance_impact == "positive"
                    ]
                ),
                "compliance_negative": len(
                    [
                        entry
                        for entry in self.audit_trail
                        if entry.compliance_impact == "negative"
                    ]
                ),
            },
        }

        print("📋 Report Summary:")
        print(f"   🆔 Report ID: {report_data['report_id']}")
        print(f"   📅 Generated: {report_data['generated_at']}")
        print(f"   🛡️ Governance mode: {report_data['governance_mode']}")
        print(
            f"   📊 Workflows monitored: {report_data['governance_summary']['total_workflows']}"
        )
        print(
            f"   ⚖️ Enforcement actions: {report_data['governance_summary']['enforcement_actions']}"
        )
        print(
            f"   ✅ Positive compliance events: {report_data['governance_summary']['compliance_positive']}"
        )
        print(
            f"   ❌ Negative compliance events: {report_data['governance_summary']['compliance_negative']}"
        )

        # Export report (in real implementation, would save to file/database)
        print("\n💾 Audit report generated and ready for export")
        print("   📁 Contains: Policies, violations, decisions, recommendations")
        print("   🔗 Integration ready: JSON format for downstream systems")

        return report_data


def main():
    """Run the comprehensive governance and compliance demonstration."""
    parser = argparse.ArgumentParser(description="Agent Workflow Governance Demo")
    parser.add_argument(
        "--governance-mode",
        choices=["monitoring", "advisory", "enforced", "strict"],
        default="advisory",
        help="Governance enforcement level",
    )
    parser.add_argument(
        "--compliance-level",
        choices=["basic", "standard", "enhanced", "enterprise"],
        default="standard",
        help="Compliance monitoring level",
    )
    args = parser.parse_args()

    print("🛡️ Agent Workflow Governance and Advanced Monitoring")
    print("=" * 60)
    print(f"Governance mode: {args.governance_mode}")
    print(f"Compliance level: {args.compliance_level}")

    # Initialize governance engine
    governance = GovernanceEngine(
        governance_mode=args.governance_mode, compliance_level=args.compliance_level
    )

    # Validate setup
    if not governance.setup_validation():
        print("\n❌ Please fix setup issues before proceeding")
        return 1

    try:
        # Demonstrate governance workflow
        governance.demonstrate_governance_workflow()

        # Analyze effectiveness
        governance.analyze_governance_effectiveness()

        # Generate audit report
        governance.generate_audit_report()

        print("\n🎉 Governance and Compliance Demonstration Complete!")
        print("\n🚀 Next Steps:")
        print("   • Review governance policies and adjust as needed")
        print("   • Implement automated compliance monitoring in production")
        print("   • Set up regular audit report generation")
        print("   • Try production_deployment_patterns.py for scaling governance")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️ Governance demo interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Governance demo failed: {e}", exc_info=True)
        print(f"\n❌ Demo failed: {e}")
        print("Try running setup_validation.py to check your configuration")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
