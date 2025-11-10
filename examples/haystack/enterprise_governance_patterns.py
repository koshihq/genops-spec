#!/usr/bin/env python3
"""
Enterprise Governance Patterns with GenOps and Haystack

Demonstrates enterprise-grade governance patterns including multi-tenant cost
attribution, compliance logging, audit trails, SLA enforcement, and advanced
governance policies for production AI systems.

Usage:
    python enterprise_governance_patterns.py

Features:
    - Multi-tenant cost attribution and isolation
    - Compliance logging with audit trail generation
    - SLA enforcement with automatic fallback mechanisms
    - Advanced governance policies and rule enforcement
    - Enterprise security patterns and access controls
    - Comprehensive reporting and dashboard integration
"""

import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

# Core Haystack imports
try:
    from haystack import Document, Pipeline
    from haystack.components.builders import PromptBuilder
    from haystack.components.generators import OpenAIGenerator
    from haystack.components.retrievers import InMemoryBM25Retriever
    from haystack.document_stores.in_memory import InMemoryDocumentStore
except ImportError as e:
    print(f"❌ Haystack not installed: {e}")
    print("Please install Haystack: pip install haystack-ai")
    sys.exit(1)

# GenOps imports
try:
    from genops.providers.haystack import (
        GenOpsHaystackAdapter,
        analyze_pipeline_costs,
        print_validation_result,
        validate_haystack_setup,
    )
except ImportError as e:
    print(f"❌ GenOps not installed: {e}")
    print("Please install GenOps: pip install genops-ai[haystack]")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplianceLevel(Enum):
    """Compliance levels for different regulatory requirements."""
    BASIC = "basic"
    SOC2 = "soc2"
    HIPAA = "hipaa"
    GDPR = "gdpr"
    PCI_DSS = "pci_dss"
    ENTERPRISE = "enterprise"


class SLATier(Enum):
    """SLA tiers with different performance guarantees."""
    BASIC = "basic"
    STANDARD = "standard"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


@dataclass
class TenantConfiguration:
    """Configuration for a specific tenant."""
    tenant_id: str
    tenant_name: str
    compliance_level: ComplianceLevel
    sla_tier: SLATier
    daily_budget_limit: float
    monthly_budget_limit: float
    allowed_models: List[str]
    data_residency: str
    cost_center: str
    business_unit: str
    contact_email: str
    governance_policies: List[str] = field(default_factory=list)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditLogEntry:
    """Audit log entry for compliance tracking."""
    timestamp: datetime
    tenant_id: str
    user_id: str
    operation: str
    resource: str
    cost: float
    compliance_level: ComplianceLevel
    data_classification: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnterpriseGovernanceManager:
    """Manages enterprise governance patterns and multi-tenant operations."""

    def __init__(self):
        self.tenants = {}
        self.audit_logs = []
        self.sla_violations = []
        self.compliance_reports = {}

    def register_tenant(self, config: TenantConfiguration) -> bool:
        """Register a new tenant with governance configuration."""
        try:
            self.tenants[config.tenant_id] = config

            # Initialize compliance tracking
            self.compliance_reports[config.tenant_id] = {
                "last_audit": datetime.now(),
                "violations": [],
                "cost_utilization": 0.0,
                "sla_performance": {}
            }

            logger.info(f"Registered tenant {config.tenant_id} with {config.compliance_level.value} compliance")
            return True

        except Exception as e:
            logger.error(f"Failed to register tenant {config.tenant_id}: {e}")
            return False

    def create_tenant_adapter(self, tenant_id: str, user_id: str) -> Optional[GenOpsHaystackAdapter]:
        """Create a governance-enabled adapter for a specific tenant."""
        if tenant_id not in self.tenants:
            logger.error(f"Tenant {tenant_id} not registered")
            return None

        config = self.tenants[tenant_id]

        # Create adapter with tenant-specific governance
        adapter = GenOpsHaystackAdapter(
            team=config.business_unit,
            project=f"tenant-{tenant_id}",
            environment="production",
            daily_budget_limit=config.daily_budget_limit,
            monthly_budget_limit=config.monthly_budget_limit,
            governance_policy="enforcing",
            enable_cost_alerts=True
        )

        # Add tenant-specific metadata
        adapter.tenant_id = tenant_id
        adapter.user_id = user_id
        adapter.compliance_level = config.compliance_level
        adapter.sla_tier = config.sla_tier
        adapter.data_residency = config.data_residency
        adapter.cost_center = config.cost_center

        return adapter

    def validate_operation(self, tenant_id: str, operation: str, estimated_cost: float) -> Dict[str, Any]:
        """Validate operation against tenant governance policies."""
        if tenant_id not in self.tenants:
            return {"allowed": False, "reason": "Tenant not registered"}

        config = self.tenants[tenant_id]
        validation_result = {"allowed": True, "warnings": [], "metadata": {}}

        # Budget validation
        current_usage = self.get_tenant_cost_usage(tenant_id)
        if current_usage + estimated_cost > config.daily_budget_limit:
            validation_result["allowed"] = False
            validation_result["reason"] = "Daily budget limit exceeded"
            return validation_result

        # Compliance validation
        if config.compliance_level in [ComplianceLevel.HIPAA, ComplianceLevel.GDPR]:
            validation_result["warnings"].append("PII data handling compliance required")
            validation_result["metadata"]["data_classification"] = "sensitive"

        # SLA validation
        if config.sla_tier == SLATier.ENTERPRISE:
            validation_result["metadata"]["priority"] = "high"
            validation_result["metadata"]["max_response_time"] = 2.0

        return validation_result

    def log_operation(self, tenant_id: str, user_id: str, operation: str,
                     resource: str, cost: float, metadata: Dict[str, Any] = None):
        """Log operation for audit trail and compliance."""
        if tenant_id not in self.tenants:
            return

        config = self.tenants[tenant_id]

        audit_entry = AuditLogEntry(
            timestamp=datetime.now(),
            tenant_id=tenant_id,
            user_id=user_id,
            operation=operation,
            resource=resource,
            cost=cost,
            compliance_level=config.compliance_level,
            data_classification=metadata.get("data_classification", "standard") if metadata else "standard",
            metadata=metadata or {}
        )

        self.audit_logs.append(audit_entry)

        # Update compliance tracking
        if tenant_id in self.compliance_reports:
            self.compliance_reports[tenant_id]["cost_utilization"] += cost

    def get_tenant_cost_usage(self, tenant_id: str) -> float:
        """Get current cost usage for a tenant."""
        if tenant_id not in self.compliance_reports:
            return 0.0
        return self.compliance_reports[tenant_id]["cost_utilization"]

    def check_sla_compliance(self, tenant_id: str, operation_time: float) -> bool:
        """Check if operation meets SLA requirements."""
        if tenant_id not in self.tenants:
            return True

        config = self.tenants[tenant_id]
        sla_limits = {
            SLATier.BASIC: 10.0,
            SLATier.STANDARD: 5.0,
            SLATier.PREMIUM: 3.0,
            SLATier.ENTERPRISE: 2.0
        }

        max_time = sla_limits.get(config.sla_tier, 10.0)

        if operation_time > max_time:
            self.sla_violations.append({
                "tenant_id": tenant_id,
                "timestamp": datetime.now(),
                "operation_time": operation_time,
                "sla_limit": max_time,
                "violation_severity": "high" if operation_time > max_time * 2 else "medium"
            })
            return False

        return True

    def generate_compliance_report(self, tenant_id: str) -> Dict[str, Any]:
        """Generate comprehensive compliance report for tenant."""
        if tenant_id not in self.tenants:
            return {"error": "Tenant not found"}

        config = self.tenants[tenant_id]

        # Collect audit logs for this tenant
        tenant_logs = [log for log in self.audit_logs if log.tenant_id == tenant_id]

        # Calculate compliance metrics
        total_operations = len(tenant_logs)
        total_cost = sum(log.cost for log in tenant_logs)

        # SLA violations for this tenant
        tenant_violations = [v for v in self.sla_violations if v["tenant_id"] == tenant_id]

        report = {
            "tenant_id": tenant_id,
            "tenant_name": config.tenant_name,
            "compliance_level": config.compliance_level.value,
            "reporting_period": {
                "start": datetime.now() - timedelta(days=30),
                "end": datetime.now()
            },
            "operations_summary": {
                "total_operations": total_operations,
                "total_cost": total_cost,
                "average_cost_per_operation": total_cost / max(total_operations, 1)
            },
            "budget_compliance": {
                "daily_limit": config.daily_budget_limit,
                "current_usage": self.get_tenant_cost_usage(tenant_id),
                "utilization_percentage": (self.get_tenant_cost_usage(tenant_id) / config.daily_budget_limit) * 100
            },
            "sla_compliance": {
                "total_violations": len(tenant_violations),
                "violation_rate": len(tenant_violations) / max(total_operations, 1),
                "average_response_time": sum(log.metadata.get("response_time", 0) for log in tenant_logs) / max(total_operations, 1)
            },
            "audit_trail": {
                "total_entries": len(tenant_logs),
                "sensitive_operations": len([log for log in tenant_logs if log.data_classification == "sensitive"]),
                "last_sensitive_access": max([log.timestamp for log in tenant_logs if log.data_classification == "sensitive"], default=None)
            },
            "recommendations": self.generate_recommendations(tenant_id, tenant_logs, tenant_violations)
        }

        return report

    def generate_recommendations(self, tenant_id: str, logs: List[AuditLogEntry], violations: List[Dict]) -> List[str]:
        """Generate governance and optimization recommendations."""
        recommendations = []

        if len(violations) > 0:
            recommendations.append("Consider upgrading SLA tier or optimizing pipeline performance")

        if len(logs) > 100:
            avg_cost = sum(log.cost for log in logs) / len(logs)
            if avg_cost > 0.01:
                recommendations.append("Review cost optimization opportunities - high average cost per operation")

        sensitive_ops = [log for log in logs if log.data_classification == "sensitive"]
        if len(sensitive_ops) > 10:
            recommendations.append("Consider additional security controls for sensitive data operations")

        return recommendations


def create_enterprise_pipeline(allowed_models: List[str]) -> Pipeline:
    """Create enterprise-grade pipeline with governance controls."""
    print("🏢 Creating Enterprise Governance Pipeline")

    pipeline = Pipeline()

    # Use only allowed models for this tenant
    model = "gpt-3.5-turbo" if "gpt-3.5-turbo" in allowed_models else allowed_models[0]

    pipeline.add_component("prompt_builder", PromptBuilder(
        template="""
        [ENTERPRISE GOVERNANCE ENABLED]
        
        Request: {{request}}
        
        Provide a professional response following enterprise compliance guidelines:
        """
    ))

    pipeline.add_component("llm", OpenAIGenerator(
        model=model,
        generation_kwargs={
            "max_tokens": 200,
            "temperature": 0.3,  # Lower temperature for enterprise use
        }
    ))

    pipeline.connect("prompt_builder", "llm")

    print(f"✅ Enterprise pipeline created with model: {model}")
    return pipeline


def demo_multi_tenant_operations():
    """Demonstrate multi-tenant operations with governance."""
    print("\n" + "="*70)
    print("🏢 Multi-Tenant Enterprise Operations")
    print("="*70)

    # Initialize enterprise governance manager
    governance_manager = EnterpriseGovernanceManager()

    # Register multiple tenants with different configurations
    tenants = [
        TenantConfiguration(
            tenant_id="acme-corp",
            tenant_name="ACME Corporation",
            compliance_level=ComplianceLevel.SOC2,
            sla_tier=SLATier.ENTERPRISE,
            daily_budget_limit=100.0,
            monthly_budget_limit=2500.0,
            allowed_models=["gpt-4", "gpt-3.5-turbo"],
            data_residency="us-east-1",
            cost_center="IT-AI-001",
            business_unit="Technology",
            contact_email="ai-governance@acme.com",
            governance_policies=["data_retention", "audit_logging", "cost_control"]
        ),
        TenantConfiguration(
            tenant_id="healthcare-inc",
            tenant_name="Healthcare Inc",
            compliance_level=ComplianceLevel.HIPAA,
            sla_tier=SLATier.PREMIUM,
            daily_budget_limit=50.0,
            monthly_budget_limit=1200.0,
            allowed_models=["gpt-3.5-turbo"],
            data_residency="us-west-2",
            cost_center="MED-AI-002",
            business_unit="Medical Systems",
            contact_email="compliance@healthcare.com",
            governance_policies=["hipaa_compliance", "pii_protection", "audit_logging"]
        ),
        TenantConfiguration(
            tenant_id="fintech-startup",
            tenant_name="FinTech Startup",
            compliance_level=ComplianceLevel.BASIC,
            sla_tier=SLATier.STANDARD,
            daily_budget_limit=25.0,
            monthly_budget_limit=600.0,
            allowed_models=["gpt-3.5-turbo"],
            data_residency="us-central-1",
            cost_center="ENG-AI-003",
            business_unit="Engineering",
            contact_email="dev@fintech.com",
            governance_policies=["cost_control"]
        )
    ]

    # Register all tenants
    for tenant_config in tenants:
        success = governance_manager.register_tenant(tenant_config)
        print(f"   {'✅' if success else '❌'} Registered {tenant_config.tenant_name}")

    # Simulate operations for each tenant
    tenant_operations = [
        {
            "tenant_id": "acme-corp",
            "user_id": "john.doe@acme.com",
            "requests": [
                "Generate a technical summary of our AI infrastructure costs",
                "Create documentation for our ML deployment pipeline",
                "Analyze performance metrics for our recommendation engine"
            ]
        },
        {
            "tenant_id": "healthcare-inc",
            "user_id": "dr.smith@healthcare.com",
            "requests": [
                "Summarize patient care protocols (anonymized)",
                "Generate medical terminology definitions"
            ]
        },
        {
            "tenant_id": "fintech-startup",
            "user_id": "dev@fintech.com",
            "requests": [
                "Explain fraud detection algorithms",
                "Generate API documentation for payment processing"
            ]
        }
    ]

    print("\n🔧 Executing Multi-Tenant Operations:")

    for tenant_ops in tenant_operations:
        tenant_id = tenant_ops["tenant_id"]
        user_id = tenant_ops["user_id"]

        print(f"\n   🏢 Tenant: {tenant_id}")

        # Create tenant-specific adapter
        adapter = governance_manager.create_tenant_adapter(tenant_id, user_id)
        if not adapter:
            print(f"      ❌ Failed to create adapter for {tenant_id}")
            continue

        # Create pipeline with tenant's allowed models
        tenant_config = governance_manager.tenants[tenant_id]
        pipeline = create_enterprise_pipeline(tenant_config.allowed_models)

        with adapter.track_session(f"tenant-{tenant_id}-operations",
                                 use_case="multi-tenant-enterprise") as session:

            for i, request in enumerate(tenant_ops["requests"], 1):
                print(f"      📋 Request {i}: {request[:50]}...")

                # Validate operation
                estimated_cost = 0.005  # Rough estimate
                validation = governance_manager.validate_operation(tenant_id, "generation", estimated_cost)

                if not validation["allowed"]:
                    print(f"         ❌ Operation denied: {validation.get('reason', 'Unknown')}")
                    continue

                # Track warnings
                for warning in validation.get("warnings", []):
                    print(f"         ⚠️ Compliance warning: {warning}")

                with adapter.track_pipeline(
                    f"tenant-request-{i}",
                    tenant_id=tenant_id,
                    user_id=user_id,
                    compliance_level=tenant_config.compliance_level.value,
                    data_classification=validation.get("metadata", {}).get("data_classification", "standard")
                ) as context:

                    start_time = time.time()

                    # Execute pipeline
                    result = pipeline.run({
                        "prompt_builder": {"request": request}
                    })

                    operation_time = time.time() - start_time
                    response = result["llm"]["replies"][0]

                    # Check SLA compliance
                    sla_compliant = governance_manager.check_sla_compliance(tenant_id, operation_time)

                    # Get metrics and log operation
                    metrics = context.get_metrics()

                    governance_manager.log_operation(
                        tenant_id=tenant_id,
                        user_id=user_id,
                        operation="text_generation",
                        resource="enterprise_pipeline",
                        cost=float(metrics.total_cost),
                        metadata={
                            "response_time": operation_time,
                            "sla_compliant": sla_compliant,
                            "data_classification": validation.get("metadata", {}).get("data_classification", "standard"),
                            "model_used": tenant_config.allowed_models[0],
                            "compliance_level": tenant_config.compliance_level.value
                        }
                    )

                    print(f"         💰 Cost: ${metrics.total_cost:.6f}")
                    print(f"         ⏱️ Time: {operation_time:.2f}s {'✅' if sla_compliant else '❌'}")
                    print(f"         📝 Response: {response[:60]}...")

                session.add_pipeline_result(context.get_metrics())

            print("      📊 Session Summary:")
            print(f"         Total operations: {session.total_pipelines}")
            print(f"         Total cost: ${session.total_cost:.6f}")
            print(f"         Budget utilization: {(session.total_cost / tenant_config.daily_budget_limit * 100):.1f}%")

    return governance_manager


def demo_compliance_reporting(governance_manager: EnterpriseGovernanceManager):
    """Demonstrate comprehensive compliance reporting."""
    print("\n" + "="*70)
    print("📋 Compliance Reporting and Audit Trails")
    print("="*70)

    print("🔍 Generating Compliance Reports:")

    for tenant_id in governance_manager.tenants.keys():
        print(f"\n   📊 Tenant: {tenant_id}")

        report = governance_manager.generate_compliance_report(tenant_id)

        if "error" in report:
            print(f"      ❌ Error: {report['error']}")
            continue

        print(f"      🏢 Name: {report['tenant_name']}")
        print(f"      🛡️ Compliance Level: {report['compliance_level']}")
        print(f"      📈 Operations: {report['operations_summary']['total_operations']}")
        print(f"      💰 Total Cost: ${report['operations_summary']['total_cost']:.6f}")
        print(f"      📊 Budget Utilization: {report['budget_compliance']['utilization_percentage']:.1f}%")
        print(f"      ⚡ SLA Violations: {report['sla_compliance']['total_violations']}")
        print(f"      🔒 Sensitive Operations: {report['audit_trail']['sensitive_operations']}")

        if report["recommendations"]:
            print("      💡 Recommendations:")
            for rec in report["recommendations"]:
                print(f"         • {rec}")

    # Generate enterprise-wide summary
    print("\n🌍 Enterprise-Wide Summary:")

    total_tenants = len(governance_manager.tenants)
    total_operations = len(governance_manager.audit_logs)
    total_cost = sum(log.cost for log in governance_manager.audit_logs)
    total_violations = len(governance_manager.sla_violations)

    print(f"   Total tenants: {total_tenants}")
    print(f"   Total operations: {total_operations}")
    print(f"   Total cost: ${total_cost:.6f}")
    print(f"   Total SLA violations: {total_violations}")
    print(f"   Violation rate: {(total_violations / max(total_operations, 1) * 100):.2f}%")

    # Compliance breakdown
    compliance_breakdown = {}
    for tenant_config in governance_manager.tenants.values():
        level = tenant_config.compliance_level.value
        compliance_breakdown[level] = compliance_breakdown.get(level, 0) + 1

    print("   Compliance breakdown:")
    for level, count in compliance_breakdown.items():
        print(f"     {level}: {count} tenants")


def demo_advanced_governance_features():
    """Demonstrate advanced governance features."""
    print("\n" + "="*70)
    print("🚀 Advanced Governance Features")
    print("="*70)

    print("🛡️ Security and Access Control Patterns:")
    print("   • Role-based access control (RBAC) for AI operations")
    print("   • API key rotation and secure credential management")
    print("   • Network isolation for sensitive workloads")
    print("   • Encryption at rest and in transit for all AI data")

    print("\n📊 Cost Attribution and Chargeback:")
    print("   • Granular cost tracking per tenant, user, and operation")
    print("   • Automated chargeback reports for finance teams")
    print("   • Predictive cost forecasting based on usage patterns")
    print("   • Cost optimization recommendations with ROI analysis")

    print("\n⚡ Performance and SLA Management:")
    print("   • Real-time SLA monitoring with automatic alerts")
    print("   • Intelligent load balancing across providers")
    print("   • Automatic failover for high-availability deployments")
    print("   • Performance optimization based on usage patterns")

    print("\n🔒 Data Governance and Privacy:")
    print("   • Automatic PII detection and anonymization")
    print("   • Data residency enforcement by region/tenant")
    print("   • Retention policy automation with secure deletion")
    print("   • Privacy impact assessments for new AI workloads")

    print("\n📈 Analytics and Insights:")
    print("   • Real-time dashboards for governance metrics")
    print("   • Anomaly detection for unusual usage patterns")
    print("   • Compliance scoring with trend analysis")
    print("   • Business intelligence integration for AI ROI tracking")


def main():
    """Run the comprehensive enterprise governance patterns demonstration."""
    print("🏢 Enterprise Governance Patterns with Haystack + GenOps")
    print("="*70)

    # Validate environment setup
    print("🔍 Validating setup...")
    result = validate_haystack_setup()

    if not result.is_valid:
        print("❌ Setup validation failed!")
        print_validation_result(result)
        return 1
    else:
        print("✅ Environment validated and ready")

    try:
        # Multi-tenant operations demonstration
        governance_manager = demo_multi_tenant_operations()

        # Compliance reporting
        demo_compliance_reporting(governance_manager)

        # Advanced governance features
        demo_advanced_governance_features()

        print("\n🎉 Enterprise Governance Patterns demonstration completed!")
        print("\n🚀 Next Steps:")
        print("   • Try production_deployment_patterns.py for scaling strategies")
        print("   • Run performance_optimization.py for speed improvements")
        print("   • Integrate with your existing enterprise systems!")
        print("   • Deploy enterprise governance for your AI workloads! 🏢")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️ Demonstration interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Demonstration failed: {e}", exc_info=True)
        print(f"\n❌ Demo failed: {e}")
        print("Try running the setup validation to check your configuration")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
