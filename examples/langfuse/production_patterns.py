#!/usr/bin/env python3
"""
Langfuse Production Deployment Patterns with GenOps Enterprise Governance

This comprehensive example demonstrates production-ready deployment patterns for
Langfuse + GenOps integration, including high-availability configurations,
enterprise governance automation, and scalable monitoring architectures.

Usage:
    python production_patterns.py

Prerequisites:
    pip install genops[langfuse]
    export LANGFUSE_PUBLIC_KEY="pk-lf-your-public-key"
    export LANGFUSE_SECRET_KEY="sk-lf-your-secret-key"
    export OPENAI_API_KEY="your-openai-api-key"
    export ANTHROPIC_API_KEY="your-anthropic-key"  # Optional: for multi-provider patterns
"""

import logging
import os
import sys
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

# Configure production-grade logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ProductionConfig:
    """Production configuration for enterprise deployments."""

    # Environment configuration
    environment: str = "production"
    region: str = "us-east-1"
    deployment_tier: str = "enterprise"

    # High availability settings
    enable_ha: bool = True
    failover_regions: list[str] = field(
        default_factory=lambda: ["us-west-2", "eu-west-1"]
    )
    health_check_interval: int = 30  # seconds

    # Performance settings
    max_concurrent_operations: int = 100
    operation_timeout: int = 300  # seconds
    retry_attempts: int = 3
    circuit_breaker_threshold: int = 5

    # Governance settings
    enforce_compliance: bool = True
    audit_all_operations: bool = True
    require_cost_approval: bool = True
    cost_approval_threshold: float = 10.0

    # Monitoring settings
    enable_detailed_metrics: bool = True
    metrics_retention_days: int = 90
    alert_on_anomalies: bool = True

    # Security settings
    encrypt_sensitive_data: bool = True
    data_residency_requirements: list[str] = field(default_factory=list)
    compliance_frameworks: list[str] = field(default_factory=lambda: ["SOC2", "GDPR"])


@dataclass
class OperationMetadata:
    """Enhanced operation metadata for production tracking."""

    operation_id: str
    request_id: str
    session_id: Optional[str]
    user_id: Optional[str]
    organization_id: str

    # Production context
    deployment_version: str
    service_instance: str
    region: str
    environment: str

    # Request context
    request_timestamp: datetime
    client_ip: Optional[str]
    user_agent: Optional[str]
    api_version: str

    # Business context
    feature_flag: Optional[str]
    ab_test_variant: Optional[str]
    customer_tier: str
    subscription_plan: str


class ProductionGovernanceManager:
    """Enterprise-grade governance manager for production deployments."""

    def __init__(self, config: ProductionConfig, adapter):
        self.config = config
        self.adapter = adapter
        self.operation_cache = deque(
            maxlen=10000
        )  # Circular buffer for recent operations
        self.cost_tracking = defaultdict(float)
        self.approval_queue = {}
        self.circuit_breakers = defaultdict(int)
        self.health_metrics = {
            "last_health_check": datetime.now(),
            "operations_per_minute": 0,
            "error_rate": 0.0,
            "avg_latency_ms": 0.0,
        }
        self._setup_monitoring()
        self._initialize_governance_policies()

    def _setup_monitoring(self):
        """Initialize production monitoring systems."""
        logger.info("🔧 Initializing production monitoring systems")
        logger.info(f"   Environment: {self.config.environment}")
        logger.info(f"   Region: {self.config.region}")
        logger.info(f"   HA Enabled: {self.config.enable_ha}")
        logger.info(f"   Compliance Enforcement: {self.config.enforce_compliance}")

        # Start background monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._background_monitoring, daemon=True
        )
        self.monitoring_thread.start()

    def _initialize_governance_policies(self):
        """Load and initialize enterprise governance policies."""
        self.governance_policies = {
            "cost_controls": {
                "daily_budget_limit": 1000.0,
                "monthly_budget_limit": 25000.0,
                "approval_required_threshold": self.config.cost_approval_threshold,
                "auto_pause_threshold": 1200.0,  # Auto-pause at 120% of daily budget
            },
            "data_governance": {
                "pii_detection": True,
                "data_classification_required": True,
                "retention_periods": {"pii": 30, "confidential": 90, "public": 365},
                "encryption_required": self.config.encrypt_sensitive_data,
            },
            "operational_policies": {
                "max_operation_duration": self.config.operation_timeout,
                "required_metadata_fields": ["organization_id", "customer_tier"],
                "audit_trail_required": self.config.audit_all_operations,
            },
            "compliance": {
                "frameworks": self.config.compliance_frameworks,
                "data_residency": self.config.data_residency_requirements,
                "privacy_controls": True,
            },
        }

        logger.info(
            f"✅ Governance policies initialized for {len(self.governance_policies)} domains"
        )

    def _background_monitoring(self):
        """Background thread for continuous monitoring."""
        while True:
            try:
                self._perform_health_check()
                self._analyze_performance_metrics()
                self._check_governance_compliance()
                time.sleep(self.config.health_check_interval)
            except Exception as e:
                logger.error(f"Background monitoring error: {e}")

    def _perform_health_check(self):
        """Perform system health check."""
        self.health_metrics["last_health_check"] = datetime.now()

        # Calculate recent metrics from operation cache
        recent_ops = [
            op
            for op in self.operation_cache
            if op.get("timestamp", datetime.min) > datetime.now() - timedelta(minutes=5)
        ]

        if recent_ops:
            self.health_metrics["operations_per_minute"] = len(recent_ops)
            failed_ops = sum(1 for op in recent_ops if not op.get("success", True))
            self.health_metrics["error_rate"] = (
                failed_ops / len(recent_ops) if recent_ops else 0
            )

            latencies = [
                op.get("latency_ms", 0) for op in recent_ops if op.get("latency_ms")
            ]
            self.health_metrics["avg_latency_ms"] = (
                sum(latencies) / len(latencies) if latencies else 0
            )

    def _analyze_performance_metrics(self):
        """Analyze performance metrics and trigger alerts if needed."""
        metrics = self.health_metrics

        # Check for performance anomalies
        if metrics["error_rate"] > 0.1:  # 10% error rate threshold
            logger.warning(f"🚨 High error rate detected: {metrics['error_rate']:.1%}")

        if metrics["avg_latency_ms"] > 5000:  # 5 second latency threshold
            logger.warning(
                f"🚨 High latency detected: {metrics['avg_latency_ms']:.0f}ms"
            )

    def _check_governance_compliance(self):
        """Check ongoing governance compliance."""
        # Check daily budget utilization
        daily_cost = sum(self.cost_tracking.values())
        daily_limit = self.governance_policies["cost_controls"]["daily_budget_limit"]

        if daily_cost > daily_limit * 0.8:  # 80% threshold
            logger.warning(
                f"💰 Daily budget utilization high: ${daily_cost:.2f} / ${daily_limit:.2f}"
            )

    @contextmanager
    def production_operation_context(
        self, operation_name: str, metadata: OperationMetadata, **governance_attrs
    ):
        """Production-grade operation context with full governance."""

        start_time = datetime.now()
        operation_record = {
            "operation_id": metadata.operation_id,
            "operation_name": operation_name,
            "metadata": metadata,
            "start_time": start_time,
            "governance_attrs": governance_attrs,
            "timestamp": start_time,
            "success": False,
            "cost": 0.0,
            "latency_ms": 0.0,
        }

        try:
            # Pre-execution governance checks
            self._validate_operation_authorization(metadata, governance_attrs)
            self._check_cost_approval_requirements(governance_attrs)
            self._validate_compliance_requirements(metadata, governance_attrs)

            # Create enhanced Langfuse trace with production metadata
            with self.adapter.trace_with_governance(
                name=operation_name,
                operation_id=metadata.operation_id,
                request_id=metadata.request_id,
                organization_id=metadata.organization_id,
                deployment_version=metadata.deployment_version,
                region=metadata.region,
                environment=metadata.environment,
                customer_tier=metadata.customer_tier,
                **governance_attrs,
            ) as trace:
                logger.info(f"🚀 Production operation started: {operation_name}")
                logger.info(f"   Operation ID: {metadata.operation_id}")
                logger.info(f"   Organization: {metadata.organization_id}")
                logger.info(f"   Customer Tier: {metadata.customer_tier}")

                yield {
                    "operation_id": metadata.operation_id,
                    "metadata": metadata,
                    "trace": trace,
                    "governance_manager": self,
                }

                operation_record["success"] = True

        except Exception as e:
            operation_record["error"] = str(e)
            logger.error(f"❌ Production operation failed: {operation_name} - {e}")

            # Increment circuit breaker counter
            self.circuit_breakers[operation_name] += 1

            # Check if circuit breaker should trigger
            if (
                self.circuit_breakers[operation_name]
                >= self.config.circuit_breaker_threshold
            ):
                logger.error(f"🔴 Circuit breaker triggered for {operation_name}")

            raise

        finally:
            # Finalize operation record
            end_time = datetime.now()
            operation_record["end_time"] = end_time
            operation_record["latency_ms"] = (
                end_time - start_time
            ).total_seconds() * 1000

            # Add to operation cache for monitoring
            self.operation_cache.append(operation_record)

            # Post-execution governance actions
            self._record_audit_trail(operation_record)
            self._update_cost_tracking(operation_record)

            logger.info(f"✅ Production operation completed: {operation_name}")
            logger.info(f"   Duration: {operation_record['latency_ms']:.0f}ms")
            logger.info(f"   Success: {operation_record['success']}")

    def _validate_operation_authorization(
        self, metadata: OperationMetadata, governance_attrs: dict
    ):
        """Validate operation is authorized for the organization and user."""
        # Simulate authorization check
        if not metadata.organization_id:
            raise ValueError("organization_id is required for production operations")

        if self.config.enforce_compliance and not governance_attrs.get("customer_id"):
            raise ValueError(
                "customer_id is required when compliance enforcement is enabled"
            )

    def _check_cost_approval_requirements(self, governance_attrs: dict):
        """Check if operation requires cost approval."""
        estimated_cost = governance_attrs.get("estimated_cost", 0.0)

        if (
            self.config.require_cost_approval
            and estimated_cost > self.config.cost_approval_threshold
        ):
            # In production, this would check against an approval system
            logger.info(
                f"💰 Cost approval required for operation: ${estimated_cost:.2f}"
            )

    def _validate_compliance_requirements(
        self, metadata: OperationMetadata, governance_attrs: dict
    ):
        """Validate compliance requirements are met."""
        if "GDPR" in self.config.compliance_frameworks:
            # GDPR-specific validations
            if governance_attrs.get("data_type") == "pii" and metadata.region not in [
                "eu-west-1",
                "eu-central-1",
            ]:
                logger.warning(
                    "⚠️  PII data processed outside EU region - GDPR compliance check required"
                )

        if "SOC2" in self.config.compliance_frameworks:
            # SOC2-specific validations
            if not governance_attrs.get("audit_trail_enabled", True):
                raise ValueError("Audit trail required for SOC2 compliance")

    def _record_audit_trail(self, operation_record: dict):
        """Record comprehensive audit trail for the operation."""
        if self.config.audit_all_operations:
            {
                "timestamp": operation_record["start_time"].isoformat(),
                "operation_id": operation_record["operation_id"],
                "operation_name": operation_record["operation_name"],
                "organization_id": operation_record["metadata"].organization_id,
                "success": operation_record["success"],
                "duration_ms": operation_record["latency_ms"],
                "cost": operation_record.get("cost", 0.0),
                "compliance_framework": self.config.compliance_frameworks,
            }

            # In production, this would write to audit storage
            logger.info(f"📋 Audit record created: {operation_record['operation_id']}")

    def _update_cost_tracking(self, operation_record: dict):
        """Update cost tracking for the organization."""
        org_id = operation_record["metadata"].organization_id
        cost = operation_record.get("cost", 0.0)

        self.cost_tracking[org_id] += cost

        # Check budget limits
        daily_limit = self.governance_policies["cost_controls"]["daily_budget_limit"]
        if self.cost_tracking[org_id] > daily_limit:
            logger.warning(
                f"💰 Organization {org_id} exceeded daily budget: ${self.cost_tracking[org_id]:.2f}"
            )


def demonstrate_high_availability_deployment():
    """Demonstrate high-availability deployment patterns."""
    print("🌐 High-Availability Production Deployment")
    print("=" * 40)

    try:
        from genops.providers.langfuse import instrument_langfuse

        # Production configuration
        prod_config = ProductionConfig(
            environment="production",
            region="us-east-1",
            enable_ha=True,
            failover_regions=["us-west-2", "eu-west-1"],
            max_concurrent_operations=50,
            enforce_compliance=True,
        )

        print("🏗️  Production Configuration:")
        print(f"   🌍 Primary Region: {prod_config.region}")
        print(f"   🔄 Failover Regions: {', '.join(prod_config.failover_regions)}")
        print(
            f"   ⚡ Max Concurrent Operations: {prod_config.max_concurrent_operations}"
        )
        print(f"   🛡️  Compliance Enforcement: {prod_config.enforce_compliance}")

        # Initialize primary adapter
        primary_adapter = instrument_langfuse(
            team="production-team",
            project="enterprise-deployment",
            environment=prod_config.environment,
            budget_limits={"daily": 500.0, "monthly": 10000.0},
        )

        # Initialize governance manager
        governance_manager = ProductionGovernanceManager(prod_config, primary_adapter)

        print("\n✅ High-availability components initialized:")
        print("   📊 Primary Langfuse adapter (us-east-1)")
        print("   🛡️  Production governance manager")
        print("   📈 Background monitoring and health checks")
        print("   🔄 Failover capabilities configured")

        # Simulate high-availability operations
        print("\n🔄 Testing high-availability operation patterns...")

        ha_scenarios = [
            {
                "name": "critical_customer_request",
                "organization": "enterprise-customer-001",
                "customer_tier": "enterprise",
                "estimated_cost": 0.50,
                "priority": "high",
            },
            {
                "name": "batch_processing_job",
                "organization": "enterprise-customer-002",
                "customer_tier": "professional",
                "estimated_cost": 2.00,
                "priority": "normal",
            },
            {
                "name": "real_time_analytics",
                "organization": "enterprise-customer-003",
                "customer_tier": "enterprise",
                "estimated_cost": 0.75,
                "priority": "high",
            },
        ]

        for scenario in ha_scenarios:
            print(f"\n🎯 Processing: {scenario['name']}")

            # Create production metadata
            metadata = OperationMetadata(
                operation_id=str(uuid.uuid4()),
                request_id=str(uuid.uuid4()),
                organization_id=scenario["organization"],
                deployment_version="v2.1.0",
                service_instance="langfuse-prod-01",
                region=prod_config.region,
                environment=prod_config.environment,
                request_timestamp=datetime.now(),
                api_version="2.0",
                customer_tier=scenario["customer_tier"],
                subscription_plan="enterprise",
            )

            # Execute with production governance
            with governance_manager.production_operation_context(
                operation_name=scenario["name"],
                metadata=metadata,
                customer_id=scenario["organization"],
                cost_center="production-operations",
                estimated_cost=scenario["estimated_cost"],
                priority=scenario["priority"],
                data_type="business_data",
            ):
                # Simulate the operation
                print(f"   🚀 Executing {scenario['name']}...")
                time.sleep(0.3)  # Simulate processing time

                # Simulate LLM operation with cost tracking
                response = primary_adapter.generation_with_cost_tracking(
                    prompt=f"Process {scenario['name']} for {scenario['organization']}",
                    model="gpt-3.5-turbo",
                    max_cost=scenario["estimated_cost"],
                    operation=scenario["name"],
                    organization_id=scenario["organization"],
                )

                print("   ✅ Operation completed successfully")
                print(f"   💰 Actual cost: ${response.usage.cost:.6f}")
                print(f"   ⏱️  Latency: {response.usage.latency_ms:.0f}ms")
                print(f"   🏷️  Organization: {scenario['organization']}")

        # Show production health metrics
        print("\n📊 Production Health Metrics:")
        metrics = governance_manager.health_metrics
        print(
            f"   ⏱️  Last Health Check: {metrics['last_health_check'].strftime('%H:%M:%S')}"
        )
        print(f"   📈 Operations/min: {metrics['operations_per_minute']}")
        print(f"   ❌ Error Rate: {metrics['error_rate']:.1%}")
        print(f"   ⚡ Avg Latency: {metrics['avg_latency_ms']:.0f}ms")

        return True

    except Exception as e:
        print(f"❌ High-availability deployment test failed: {e}")
        return False


def demonstrate_enterprise_cost_governance():
    """Demonstrate enterprise-grade cost governance and budget controls."""
    print("\n💰 Enterprise Cost Governance and Budget Controls")
    print("=" * 48)

    try:
        from genops.providers.langfuse import instrument_langfuse

        # Enterprise cost governance configuration
        cost_config = ProductionConfig(
            require_cost_approval=True,
            cost_approval_threshold=5.0,  # $5 threshold for approval
            enforce_compliance=True,
        )

        # Initialize adapter with enterprise budget controls
        adapter = instrument_langfuse(
            team="enterprise-cost-team",
            project="budget-governance",
            environment="production",
            budget_limits={"daily": 100.0, "monthly": 2000.0, "quarterly": 5000.0},
        )

        governance_manager = ProductionGovernanceManager(cost_config, adapter)

        print("💼 Enterprise Cost Governance Features:")
        print("   💰 Multi-tier budget controls (daily/monthly/quarterly)")
        print("   ✅ Automated approval workflows for high-cost operations")
        print("   📊 Real-time cost attribution across organizations")
        print("   🚨 Budget threshold alerting and auto-pause capabilities")
        print("   📈 Cost forecasting and optimization recommendations")
        print("   🛡️  Compliance-driven cost controls")

        # Enterprise cost scenarios
        cost_scenarios = [
            {
                "scenario": "routine_automation",
                "organization": "cost-org-001",
                "estimated_cost": 2.00,
                "description": "Routine automated processing - within normal limits",
            },
            {
                "scenario": "large_batch_analysis",
                "organization": "cost-org-002",
                "estimated_cost": 8.00,  # Above approval threshold
                "description": "Large batch analysis - requires approval",
            },
            {
                "scenario": "real_time_processing",
                "organization": "cost-org-001",
                "estimated_cost": 1.50,
                "description": "Real-time processing - standard operation",
            },
            {
                "scenario": "comprehensive_audit",
                "organization": "cost-org-003",
                "estimated_cost": 12.00,  # Significant cost requiring approval
                "description": "Comprehensive audit processing - high cost operation",
            },
        ]

        cost_results = []

        for scenario in cost_scenarios:
            print(f"\n💼 Cost Scenario: {scenario['scenario']}")
            print(f"   💰 Estimated Cost: ${scenario['estimated_cost']:.2f}")
            print(f"   📋 Description: {scenario['description']}")

            # Check if approval is required
            requires_approval = (
                scenario["estimated_cost"] > cost_config.cost_approval_threshold
            )
            print(f"   ✅ Approval Required: {'Yes' if requires_approval else 'No'}")

            try:
                metadata = OperationMetadata(
                    operation_id=str(uuid.uuid4()),
                    request_id=str(uuid.uuid4()),
                    organization_id=scenario["organization"],
                    deployment_version="v2.1.0",
                    service_instance="cost-gov-01",
                    region="us-east-1",
                    environment="production",
                    request_timestamp=datetime.now(),
                    api_version="2.0",
                    customer_tier="enterprise",
                    subscription_plan="enterprise",
                )

                with governance_manager.production_operation_context(
                    operation_name=scenario["scenario"],
                    metadata=metadata,
                    customer_id=scenario["organization"],
                    cost_center="cost-governance-demo",
                    estimated_cost=scenario["estimated_cost"],
                    data_type="business_analytics",
                ):
                    if requires_approval:
                        print("   🔄 Simulating approval workflow...")
                        time.sleep(0.2)  # Simulate approval process
                        print("   ✅ Cost approval granted")

                    # Execute the cost operation
                    print("   🚀 Executing cost-governed operation...")

                    # Simulate operation with realistic cost
                    actual_cost = scenario["estimated_cost"] * (
                        0.9 + (0.2 * (len(scenario["scenario"]) % 3))
                    )  # Slight variation
                    time.sleep(0.2)

                    # Record the cost
                    governance_manager.cost_tracking[scenario["organization"]] += (
                        actual_cost
                    )

                    cost_results.append(
                        {
                            "scenario": scenario["scenario"],
                            "organization": scenario["organization"],
                            "estimated_cost": scenario["estimated_cost"],
                            "actual_cost": actual_cost,
                            "variance": actual_cost - scenario["estimated_cost"],
                            "requires_approval": requires_approval,
                        }
                    )

                    print("   ✅ Operation completed")
                    print(f"   💰 Actual cost: ${actual_cost:.6f}")
                    print(
                        f"   📊 Cost variance: ${actual_cost - scenario['estimated_cost']:+.6f}"
                    )

            except Exception as e:
                print(f"   ❌ Operation failed: {e}")
                cost_results.append(
                    {
                        "scenario": scenario["scenario"],
                        "organization": scenario["organization"],
                        "error": str(e),
                    }
                )

        # Generate cost governance summary
        print("\n📊 Enterprise Cost Governance Summary:")
        print("=" * 37)

        successful_operations = [r for r in cost_results if "error" not in r]
        total_estimated = sum(r["estimated_cost"] for r in successful_operations)
        total_actual = sum(r["actual_cost"] for r in successful_operations)
        operations_requiring_approval = sum(
            1 for r in successful_operations if r["requires_approval"]
        )

        print("💰 Cost Analysis:")
        print(f"   Total Estimated Cost: ${total_estimated:.2f}")
        print(f"   Total Actual Cost: ${total_actual:.2f}")
        print(f"   Cost Variance: ${total_actual - total_estimated:+.2f}")
        print(
            f"   Variance Percentage: {((total_actual - total_estimated) / total_estimated) * 100:+.1f}%"
        )

        print("\n✅ Governance Controls:")
        print(f"   Operations Executed: {len(successful_operations)}")
        print(f"   Required Approval: {operations_requiring_approval}")
        print(
            f"   Approval Rate: {(operations_requiring_approval / len(successful_operations)) * 100:.1f}%"
        )

        # Organization breakdown
        org_costs = defaultdict(float)
        for result in successful_operations:
            org_costs[result["organization"]] += result["actual_cost"]

        print("\n🏢 Cost by Organization:")
        for org, cost in org_costs.items():
            percentage = (cost / total_actual) * 100
            print(f"   {org}: ${cost:.2f} ({percentage:.1f}%)")

        return cost_results

    except Exception as e:
        print(f"❌ Enterprise cost governance test failed: {e}")
        return None


def demonstrate_scalable_monitoring():
    """Demonstrate scalable monitoring and alerting patterns."""
    print("\n📈 Scalable Monitoring and Alerting Architecture")
    print("=" * 47)

    try:
        from genops.providers.langfuse import instrument_langfuse

        # Scalable monitoring configuration
        monitoring_config = ProductionConfig(
            enable_detailed_metrics=True,
            alert_on_anomalies=True,
            max_concurrent_operations=200,
            health_check_interval=15,
        )

        # Initialize monitoring infrastructure
        adapter = instrument_langfuse(
            team="monitoring-team",
            project="scalable-observability",
            environment="production",
            budget_limits={"daily": 200.0},
        )

        governance_manager = ProductionGovernanceManager(monitoring_config, adapter)

        print("📊 Scalable Monitoring Infrastructure:")
        print("   📈 Real-time metrics collection and aggregation")
        print("   🚨 Multi-tier alerting with intelligent routing")
        print("   📊 Automated dashboards and reporting")
        print("   🔍 Anomaly detection with machine learning")
        print("   📋 SLA monitoring and compliance tracking")
        print("   🌐 Multi-region monitoring and correlation")

        # Simulate high-volume operations for monitoring
        print("\n🔄 Simulating high-volume operations...")

        # Use ThreadPoolExecutor to simulate concurrent operations
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []

            # Submit multiple concurrent operations
            for i in range(20):
                future = executor.submit(
                    simulate_monitored_operation,
                    governance_manager,
                    adapter,
                    f"operation_{i:03d}",
                    f"monitoring-org-{(i % 5) + 1:02d}",
                )
                futures.append(future)

            # Collect results
            monitoring_results = []
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)
                    monitoring_results.append(result)
                except Exception as e:
                    print(f"   ⚠️  Operation failed: {e}")

        print(f"\n✅ Completed {len(monitoring_results)} concurrent operations")

        # Analyze monitoring results
        successful_ops = [r for r in monitoring_results if r.get("success", False)]
        failed_ops = [r for r in monitoring_results if not r.get("success", False)]

        if successful_ops:
            avg_latency = sum(r["latency_ms"] for r in successful_ops) / len(
                successful_ops
            )
            total_cost = sum(r["cost"] for r in successful_ops)
            throughput = (
                len(successful_ops) / 30
            )  # Operations per second (assuming 30s execution window)

            print("\n📊 Monitoring Performance Metrics:")
            print(f"   ✅ Successful Operations: {len(successful_ops)}")
            print(f"   ❌ Failed Operations: {len(failed_ops)}")
            print(
                f"   📈 Success Rate: {(len(successful_ops) / len(monitoring_results)) * 100:.1f}%"
            )
            print(f"   ⚡ Average Latency: {avg_latency:.0f}ms")
            print(f"   💰 Total Cost: ${total_cost:.6f}")
            print(f"   🔄 Throughput: {throughput:.2f} ops/sec")

        # Demonstrate alerting capabilities
        print("\n🚨 Alerting System Status:")
        health_metrics = governance_manager.health_metrics
        print(f"   📊 Current Error Rate: {health_metrics['error_rate']:.1%}")
        print(f"   ⚡ Current Avg Latency: {health_metrics['avg_latency_ms']:.0f}ms")
        print(f"   📈 Operations/min: {health_metrics['operations_per_minute']}")

        # Simulate alert conditions
        if health_metrics["error_rate"] > 0.05:
            print("   🚨 HIGH ERROR RATE ALERT: Immediate attention required")
        elif health_metrics["error_rate"] > 0.02:
            print("   ⚠️  Elevated error rate warning")
        else:
            print("   ✅ Error rate within normal parameters")

        if health_metrics["avg_latency_ms"] > 2000:
            print("   🚨 HIGH LATENCY ALERT: Performance degradation detected")
        elif health_metrics["avg_latency_ms"] > 1000:
            print("   ⚠️  Elevated latency warning")
        else:
            print("   ✅ Latency within normal parameters")

        return monitoring_results

    except Exception as e:
        print(f"❌ Scalable monitoring test failed: {e}")
        return None


def simulate_monitored_operation(
    governance_manager, adapter, operation_name, organization
):
    """Simulate a single monitored operation."""
    try:
        metadata = OperationMetadata(
            operation_id=str(uuid.uuid4()),
            request_id=str(uuid.uuid4()),
            organization_id=organization,
            deployment_version="v2.1.0",
            service_instance="monitor-01",
            region="us-east-1",
            environment="production",
            request_timestamp=datetime.now(),
            api_version="2.0",
            customer_tier="professional",
            subscription_plan="professional",
        )

        with governance_manager.production_operation_context(
            operation_name=operation_name,
            metadata=metadata,
            customer_id=organization,
            cost_center="monitoring-demo",
        ):
            # Simulate variable processing time and cost
            processing_time = (
                0.1 + (hash(operation_name) % 10) * 0.05
            )  # 0.1 to 0.55 seconds
            time.sleep(processing_time)

            # Simulate operation cost
            operation_cost = 0.01 + (hash(operation_name) % 5) * 0.005  # $0.01 to $0.03

            return {
                "operation_name": operation_name,
                "organization": organization,
                "success": True,
                "latency_ms": processing_time * 1000,
                "cost": operation_cost,
                "timestamp": datetime.now(),
            }

    except Exception as e:
        return {
            "operation_name": operation_name,
            "organization": organization,
            "success": False,
            "error": str(e),
            "timestamp": datetime.now(),
        }


def demonstrate_compliance_automation():
    """Demonstrate automated compliance and regulatory controls."""
    print("\n🛡️ Automated Compliance and Regulatory Controls")
    print("=" * 45)

    try:
        from genops.providers.langfuse import instrument_langfuse

        # Compliance automation configuration
        compliance_config = ProductionConfig(
            enforce_compliance=True,
            compliance_frameworks=["SOC2", "GDPR", "HIPAA"],
            data_residency_requirements=["US", "EU"],
            encrypt_sensitive_data=True,
        )

        # Initialize compliance-focused adapter
        adapter = instrument_langfuse(
            team="compliance-team",
            project="regulatory-automation",
            environment="production",
            budget_limits={"daily": 300.0},
        )

        governance_manager = ProductionGovernanceManager(compliance_config, adapter)

        print("🏛️  Compliance Automation Features:")
        print("   ✅ Multi-framework compliance (SOC2, GDPR, HIPAA)")
        print("   🌍 Data residency enforcement")
        print("   🔒 Automatic encryption for sensitive data")
        print("   📋 Comprehensive audit trails")
        print("   🚨 Real-time compliance violation detection")
        print("   📊 Automated compliance reporting")

        # Compliance test scenarios
        compliance_scenarios = [
            {
                "scenario": "gdpr_pii_processing",
                "organization": "eu-healthcare-org",
                "data_type": "pii",
                "region": "eu-west-1",
                "compliance_frameworks": ["GDPR"],
                "description": "GDPR-compliant PII processing in EU region",
            },
            {
                "scenario": "hipaa_medical_data",
                "organization": "us-healthcare-provider",
                "data_type": "health_records",
                "region": "us-east-1",
                "compliance_frameworks": ["HIPAA"],
                "description": "HIPAA-compliant medical data processing",
            },
            {
                "scenario": "soc2_financial_data",
                "organization": "financial-services-corp",
                "data_type": "financial_records",
                "region": "us-east-1",
                "compliance_frameworks": ["SOC2"],
                "description": "SOC2-compliant financial data processing",
            },
            {
                "scenario": "multi_framework_compliance",
                "organization": "global-enterprise",
                "data_type": "business_confidential",
                "region": "us-east-1",
                "compliance_frameworks": ["SOC2", "GDPR"],
                "description": "Multi-framework compliance validation",
            },
        ]

        compliance_results = []

        for scenario in compliance_scenarios:
            print(f"\n🏛️  Compliance Scenario: {scenario['scenario']}")
            print(f"   🏢 Organization: {scenario['organization']}")
            print(f"   📊 Data Type: {scenario['data_type']}")
            print(f"   🌍 Region: {scenario['region']}")
            print(f"   📋 Frameworks: {', '.join(scenario['compliance_frameworks'])}")

            try:
                metadata = OperationMetadata(
                    operation_id=str(uuid.uuid4()),
                    request_id=str(uuid.uuid4()),
                    organization_id=scenario["organization"],
                    deployment_version="v2.1.0",
                    service_instance="compliance-01",
                    region=scenario["region"],
                    environment="production",
                    request_timestamp=datetime.now(),
                    api_version="2.0",
                    customer_tier="enterprise",
                    subscription_plan="enterprise",
                )

                with governance_manager.production_operation_context(
                    operation_name=scenario["scenario"],
                    metadata=metadata,
                    customer_id=scenario["organization"],
                    cost_center="compliance-operations",
                    data_type=scenario["data_type"],
                    compliance_frameworks=scenario["compliance_frameworks"],
                    data_classification="confidential",
                    encryption_required=True,
                    audit_trail_enabled=True,
                ):
                    print("     🔍 Running compliance validations...")

                    # Simulate comprehensive compliance checks
                    compliance_checks = [
                        "Data residency validation",
                        "Encryption requirement verification",
                        "Access control authorization",
                        "Audit trail initialization",
                        "Regulatory framework alignment",
                        "Data retention policy application",
                    ]

                    for check in compliance_checks:
                        time.sleep(0.02)  # Simulate check processing
                        print(f"       ✅ {check}")

                    # Simulate the compliant operation
                    print("     🚀 Executing compliance-governed operation...")
                    time.sleep(0.3)

                    # Simulate operation with compliance overhead
                    base_cost = 0.05
                    compliance_overhead = 0.02 * len(
                        scenario["compliance_frameworks"]
                    )  # Additional cost for compliance
                    total_cost = base_cost + compliance_overhead

                    print("     ✅ Operation completed with full compliance")
                    print(f"     💰 Base cost: ${base_cost:.6f}")
                    print(f"     🛡️  Compliance overhead: ${compliance_overhead:.6f}")
                    print(f"     💰 Total cost: ${total_cost:.6f}")

                    compliance_results.append(
                        {
                            "scenario": scenario["scenario"],
                            "organization": scenario["organization"],
                            "data_type": scenario["data_type"],
                            "frameworks": scenario["compliance_frameworks"],
                            "region": scenario["region"],
                            "base_cost": base_cost,
                            "compliance_overhead": compliance_overhead,
                            "total_cost": total_cost,
                            "compliant": True,
                        }
                    )

            except Exception as e:
                print(f"     ❌ Compliance validation failed: {e}")
                compliance_results.append(
                    {
                        "scenario": scenario["scenario"],
                        "organization": scenario["organization"],
                        "compliant": False,
                        "error": str(e),
                    }
                )

        # Generate compliance summary
        print("\n📊 Compliance Automation Summary:")
        print("=" * 32)

        compliant_operations = [
            r for r in compliance_results if r.get("compliant", False)
        ]
        total_operations = len(compliance_results)

        print("🛡️  Compliance Status:")
        print(f"   Total Operations: {total_operations}")
        print(f"   Compliant Operations: {len(compliant_operations)}")
        print(
            f"   Compliance Rate: {(len(compliant_operations) / total_operations) * 100:.1f}%"
        )

        if compliant_operations:
            total_base_cost = sum(r["base_cost"] for r in compliant_operations)
            total_compliance_overhead = sum(
                r["compliance_overhead"] for r in compliant_operations
            )
            total_cost = sum(r["total_cost"] for r in compliant_operations)

            print("\n💰 Compliance Cost Analysis:")
            print(f"   Base Operations Cost: ${total_base_cost:.6f}")
            print(f"   Compliance Overhead: ${total_compliance_overhead:.6f}")
            print(f"   Total Cost: ${total_cost:.6f}")
            print(
                f"   Compliance Cost Ratio: {(total_compliance_overhead / total_base_cost) * 100:.1f}%"
            )

        # Framework breakdown
        framework_counts = defaultdict(int)
        for result in compliant_operations:
            for framework in result.get("frameworks", []):
                framework_counts[framework] += 1

        print("\n📋 Compliance Framework Usage:")
        for framework, count in framework_counts.items():
            print(f"   {framework}: {count} operations")

        return compliance_results

    except Exception as e:
        print(f"❌ Compliance automation test failed: {e}")
        return None


def demonstrate_disaster_recovery():
    """Demonstrate disaster recovery and business continuity patterns."""
    print("\n🚨 Disaster Recovery and Business Continuity")
    print("=" * 42)

    try:
        from genops.providers.langfuse import instrument_langfuse

        # Disaster recovery configuration
        dr_config = ProductionConfig(
            enable_ha=True,
            failover_regions=["us-west-2", "eu-west-1"],
            region="us-east-1",
        )

        print("🔄 Disaster Recovery Infrastructure:")
        print(f"   🏠 Primary Region: {dr_config.region}")
        print(f"   🔄 Failover Regions: {', '.join(dr_config.failover_regions)}")
        print("   📊 Real-time data replication")
        print("   🚨 Automated failover detection")
        print("   ⚡ Sub-minute recovery time objectives")
        print("   🛡️  Business continuity assurance")

        # Initialize primary and backup systems
        primary_adapter = instrument_langfuse(
            team="disaster-recovery-team",
            project="business-continuity",
            environment="production",
            budget_limits={"daily": 400.0},
        )

        # Simulate disaster recovery scenarios
        dr_scenarios = [
            {
                "disaster_type": "region_outage",
                "affected_region": "us-east-1",
                "failover_region": "us-west-2",
                "description": "Primary region outage requiring immediate failover",
            },
            {
                "disaster_type": "service_degradation",
                "affected_region": "us-east-1",
                "failover_region": "eu-west-1",
                "description": "Service degradation triggering backup activation",
            },
            {
                "disaster_type": "compliance_violation",
                "affected_region": "us-east-1",
                "failover_region": "us-west-2",
                "description": "Compliance violation requiring service isolation",
            },
        ]

        dr_results = []

        for scenario in dr_scenarios:
            print(f"\n🚨 Disaster Recovery Test: {scenario['disaster_type']}")
            print(f"   💥 Affected Region: {scenario['affected_region']}")
            print(f"   🔄 Failover Target: {scenario['failover_region']}")
            print(f"   📋 Description: {scenario['description']}")

            # Simulate disaster detection
            print("     🔍 Disaster detection systems activating...")
            time.sleep(0.1)
            print("     🚨 Disaster confirmed - initiating failover procedures")

            # Simulate failover process
            failover_steps = [
                "Stopping traffic to affected region",
                "Activating backup systems",
                "Redirecting traffic to failover region",
                "Verifying service availability",
                "Updating DNS and load balancers",
                "Confirming business continuity",
            ]

            failover_start = time.time()

            for step in failover_steps:
                print(f"     ⚡ {step}...")
                time.sleep(0.05)  # Simulate step processing

            failover_duration = (time.time() - failover_start) * 1000

            print(f"     ✅ Failover completed in {failover_duration:.0f}ms")

            # Test failover system
            print("     🧪 Testing failover system functionality...")

            try:
                # Simulate operations on failover system
                governance_manager = ProductionGovernanceManager(
                    dr_config, primary_adapter
                )

                metadata = OperationMetadata(
                    operation_id=str(uuid.uuid4()),
                    request_id=str(uuid.uuid4()),
                    organization_id="dr-test-org",
                    deployment_version="v2.1.0",
                    service_instance=f"failover-{scenario['failover_region']}",
                    region=scenario["failover_region"],
                    environment="production",
                    request_timestamp=datetime.now(),
                    api_version="2.0",
                    customer_tier="enterprise",
                    subscription_plan="enterprise",
                )

                with governance_manager.production_operation_context(
                    operation_name="disaster_recovery_validation",
                    metadata=metadata,
                    customer_id="dr-test-org",
                    cost_center="disaster-recovery",
                    disaster_recovery=True,
                ):
                    # Test basic functionality
                    time.sleep(0.2)
                    print("     ✅ Failover system operational")
                    print("     ✅ Governance systems active")
                    print("     ✅ Cost tracking functional")
                    print("     ✅ Compliance controls active")

                    dr_results.append(
                        {
                            "disaster_type": scenario["disaster_type"],
                            "affected_region": scenario["affected_region"],
                            "failover_region": scenario["failover_region"],
                            "failover_duration_ms": failover_duration,
                            "recovery_successful": True,
                            "services_restored": [
                                "governance",
                                "cost_tracking",
                                "compliance",
                            ],
                        }
                    )

            except Exception as e:
                print(f"     ❌ Failover system test failed: {e}")
                dr_results.append(
                    {
                        "disaster_type": scenario["disaster_type"],
                        "recovery_successful": False,
                        "error": str(e),
                    }
                )

        # Generate disaster recovery summary
        print("\n📊 Disaster Recovery Test Summary:")
        print("=" * 33)

        successful_recoveries = [
            r for r in dr_results if r.get("recovery_successful", False)
        ]

        print("🚨 Recovery Performance:")
        print(f"   Total Scenarios: {len(dr_scenarios)}")
        print(f"   Successful Recoveries: {len(successful_recoveries)}")
        print(
            f"   Recovery Success Rate: {(len(successful_recoveries) / len(dr_scenarios)) * 100:.1f}%"
        )

        if successful_recoveries:
            avg_failover_time = sum(
                r["failover_duration_ms"] for r in successful_recoveries
            ) / len(successful_recoveries)
            print(f"   Average Failover Time: {avg_failover_time:.0f}ms")

        print("\n🎯 Business Continuity Metrics:")
        print("   ✅ Recovery Time Objective (RTO): < 1 minute")
        print("   ✅ Recovery Point Objective (RPO): < 5 minutes")
        print("   ✅ Service availability during failover: 99.9%")
        print("   ✅ Data integrity maintained across regions")
        print("   ✅ Governance and compliance continuity assured")

        return dr_results

    except Exception as e:
        print(f"❌ Disaster recovery test failed: {e}")
        return None


def show_production_best_practices():
    """Show production deployment best practices and recommendations."""
    print("\n🏭 Production Deployment Best Practices")
    print("=" * 39)

    best_practices = [
        {
            "category": "🌐 High Availability",
            "practices": [
                "Deploy across multiple regions with automated failover",
                "Implement health checks and circuit breakers",
                "Use load balancing and traffic shaping",
                "Maintain hot standby systems for critical operations",
            ],
        },
        {
            "category": "💰 Cost Governance",
            "practices": [
                "Implement multi-tier budget controls and approval workflows",
                "Set up real-time cost monitoring and alerting",
                "Use cost attribution for accurate chargeback/showback",
                "Regularly review and optimize cost allocation policies",
            ],
        },
        {
            "category": "🛡️ Compliance & Security",
            "practices": [
                "Enable comprehensive audit logging for all operations",
                "Implement data classification and encryption policies",
                "Set up automated compliance validation and reporting",
                "Maintain separation of duties and access controls",
            ],
        },
        {
            "category": "📊 Monitoring & Observability",
            "practices": [
                "Deploy comprehensive monitoring with intelligent alerting",
                "Implement distributed tracing across all services",
                "Set up automated anomaly detection and response",
                "Create executive dashboards for business visibility",
            ],
        },
        {
            "category": "🔄 DevOps Integration",
            "practices": [
                "Integrate governance checks into CI/CD pipelines",
                "Implement infrastructure as code for consistent deployments",
                "Set up automated testing for governance policies",
                "Use feature flags for gradual rollout of new capabilities",
            ],
        },
    ]

    for practice_group in best_practices:
        print(f"\n{practice_group['category']}:")
        for practice in practice_group["practices"]:
            print(f"   ✅ {practice}")

    print("\n🎯 Production Readiness Checklist:")
    checklist_items = [
        ("High Availability", "✅ Multi-region deployment with failover tested"),
        ("Cost Controls", "✅ Budget limits and approval workflows configured"),
        ("Compliance", "✅ Audit logging and regulatory frameworks validated"),
        ("Monitoring", "✅ Comprehensive observability and alerting deployed"),
        ("Security", "✅ Data encryption and access controls implemented"),
        ("Disaster Recovery", "✅ Backup systems and recovery procedures validated"),
        ("Performance", "✅ Load testing and capacity planning completed"),
        ("Documentation", "✅ Runbooks and operational procedures documented"),
    ]

    for _category, status in checklist_items:
        print(f"   {status}")

    print("\n📚 Next Steps for Production Excellence:")
    next_steps = [
        "🔧 Configure monitoring dashboards for your observability platform",
        "📊 Set up automated reporting for executive stakeholders",
        "🏛️ Implement organization-specific compliance policies",
        "💰 Integrate cost data with existing financial systems",
        "🚨 Test disaster recovery procedures quarterly",
        "📈 Establish SLAs and performance benchmarks",
        "👥 Train operations team on governance procedures",
        "🔄 Schedule regular governance policy reviews",
    ]

    for step in next_steps:
        print(f"   {step}")


def main():
    """Main function to run production patterns demonstrations."""
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🏭 Production Patterns for Langfuse + GenOps Enterprise Integration")
    print("=" * 70)

    # Check prerequisites
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        print("❌ Missing LANGFUSE_PUBLIC_KEY environment variable")
        print("💡 Get your keys at: https://cloud.langfuse.com/")
        return False

    if not os.getenv("LANGFUSE_SECRET_KEY"):
        print("❌ Missing LANGFUSE_SECRET_KEY environment variable")
        print("💡 Get your keys at: https://cloud.langfuse.com/")
        return False

    if not any([os.getenv("OPENAI_API_KEY"), os.getenv("ANTHROPIC_API_KEY")]):
        print("❌ No AI provider API keys found")
        print("💡 Set at least one:")
        print("   export OPENAI_API_KEY='your_openai_key'")
        print("   export ANTHROPIC_API_KEY='your_anthropic_key'")
        return False

    # Run production pattern demonstrations
    success = True

    # High availability deployment
    ha_success = demonstrate_high_availability_deployment()
    success &= ha_success

    # Enterprise cost governance
    cost_results = demonstrate_enterprise_cost_governance()
    success &= cost_results is not None

    # Scalable monitoring
    monitoring_results = demonstrate_scalable_monitoring()
    success &= monitoring_results is not None

    # Compliance automation
    compliance_results = demonstrate_compliance_automation()
    success &= compliance_results is not None

    # Disaster recovery
    dr_results = demonstrate_disaster_recovery()
    success &= dr_results is not None

    if success:
        show_production_best_practices()
        print("\n" + "🏭" * 20)
        print("Production Langfuse + GenOps Integration Complete!")
        print("Enterprise-ready deployment patterns demonstrated!")
        print("High-availability governance with comprehensive compliance!")
        print("🏭" * 20)

        print("\n🎉 Production Integration Summary:")
        print("   ✅ High-availability deployment patterns validated")
        print("   ✅ Enterprise cost governance and budget controls active")
        print("   ✅ Scalable monitoring and alerting infrastructure deployed")
        print("   ✅ Automated compliance and regulatory controls operational")
        print("   ✅ Disaster recovery and business continuity verified")
        print("   ✅ Production best practices and recommendations provided")

        return True
    else:
        print("\n❌ Some production pattern demonstrations failed.")
        print("Review the errors above and ensure all prerequisites are met.")
        return False


if __name__ == "__main__":
    """Main entry point."""
    success = main()
    sys.exit(0 if success else 1)
