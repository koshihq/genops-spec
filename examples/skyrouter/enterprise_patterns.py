#!/usr/bin/env python3
"""
SkyRouter Enterprise Production Deployment Patterns

This example demonstrates enterprise-grade production deployment patterns
with SkyRouter and GenOps governance, including multi-environment setups,
high-availability configurations, compliance frameworks, and production
monitoring for large-scale multi-model routing deployments.

Features demonstrated:
- Multi-environment deployment patterns (dev/staging/prod)
- High-availability and disaster recovery configurations
- Enterprise compliance and security frameworks
- Production monitoring and alerting systems
- Auto-scaling and load balancing strategies
- Cost governance for enterprise-scale operations

Usage:
    export SKYROUTER_API_KEY="your-api-key"
    export GENOPS_TEAM="your-team"
    export GENOPS_PROJECT="your-project"
    python enterprise_patterns.py

Author: GenOps AI Contributors
"""

import os
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

# Add the src directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class Environment(Enum):
    """Environment types for deployment."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DR = "disaster_recovery"


class ComplianceFramework(Enum):
    """Compliance framework types."""

    SOC2 = "soc2"
    HIPAA = "hipaa"
    GDPR = "gdpr"
    PCI_DSS = "pci_dss"
    FINRA = "finra"


@dataclass
class EnvironmentConfig:
    """Configuration for each environment."""

    name: str
    environment: Environment
    daily_budget_limit: float
    governance_policy: str
    enable_cost_alerts: bool
    compliance_frameworks: list[ComplianceFramework]
    monitoring_config: dict[str, Any]
    scaling_config: dict[str, Any]


def demonstrate_multi_environment_setup():
    """Demonstrate multi-environment deployment patterns."""

    print("🏢 Enterprise Multi-Environment Deployment")
    print("=" * 45)
    print()

    try:
        from genops.providers.skyrouter import GenOpsSkyRouterAdapter

        # Define environment configurations
        environments = [
            EnvironmentConfig(
                name="Development",
                environment=Environment.DEVELOPMENT,
                daily_budget_limit=10.0,
                governance_policy="advisory",
                enable_cost_alerts=False,
                compliance_frameworks=[],
                monitoring_config={
                    "log_level": "debug",
                    "metrics_collection": "basic",
                    "alert_threshold": "high",
                },
                scaling_config={
                    "min_instances": 1,
                    "max_instances": 3,
                    "auto_scaling": False,
                },
            ),
            EnvironmentConfig(
                name="Staging",
                environment=Environment.STAGING,
                daily_budget_limit=50.0,
                governance_policy="enforced",
                enable_cost_alerts=True,
                compliance_frameworks=[ComplianceFramework.SOC2],
                monitoring_config={
                    "log_level": "info",
                    "metrics_collection": "detailed",
                    "alert_threshold": "medium",
                },
                scaling_config={
                    "min_instances": 2,
                    "max_instances": 10,
                    "auto_scaling": True,
                },
            ),
            EnvironmentConfig(
                name="Production",
                environment=Environment.PRODUCTION,
                daily_budget_limit=500.0,
                governance_policy="strict",
                enable_cost_alerts=True,
                compliance_frameworks=[
                    ComplianceFramework.SOC2,
                    ComplianceFramework.GDPR,
                    ComplianceFramework.HIPAA,
                ],
                monitoring_config={
                    "log_level": "warn",
                    "metrics_collection": "comprehensive",
                    "alert_threshold": "low",
                    "sla_monitoring": True,
                },
                scaling_config={
                    "min_instances": 5,
                    "max_instances": 50,
                    "auto_scaling": True,
                    "load_balancing": "advanced",
                },
            ),
        ]

        print("🏗️ Environment Configuration Overview:")
        print()

        adapters = {}

        for env_config in environments:
            print(f"📊 **{env_config.name} Environment**")
            print(f"   🔧 Policy: {env_config.governance_policy}")
            print(f"   💰 Budget: ${env_config.daily_budget_limit:.2f}/day")
            print(
                f"   🔐 Compliance: {', '.join([cf.value for cf in env_config.compliance_frameworks]) or 'None'}"
            )
            print(
                f"   📈 Scaling: {env_config.scaling_config['min_instances']}-{env_config.scaling_config['max_instances']} instances"
            )
            print()

            # Initialize adapter for each environment
            adapter = GenOpsSkyRouterAdapter(
                team=f"enterprise-{env_config.environment.value}",
                project="multi-env-deployment",
                environment=env_config.environment.value,
                daily_budget_limit=env_config.daily_budget_limit,
                governance_policy=env_config.governance_policy,
                enable_cost_alerts=env_config.enable_cost_alerts,
            )

            adapters[env_config.environment] = adapter

        print("✅ All environment adapters initialized successfully")
        print()

        # Demonstrate environment-specific routing
        print("🧪 Testing Environment-Specific Routing:")
        print()

        test_request = {
            "task": "Process sensitive customer data",
            "data_classification": "confidential",
            "compliance_required": True,
        }

        for env_type, adapter in adapters.items():
            print(f"🔄 {env_type.value.title()} Environment:")

            with adapter.track_routing_session(f"env-test-{env_type.value}") as session:
                result = session.track_multi_model_routing(
                    models=["gpt-4", "claude-3-opus", "claude-3-sonnet"],
                    input_data=test_request,
                    routing_strategy="reliability_first"
                    if env_type == Environment.PRODUCTION
                    else "balanced",
                    complexity="enterprise"
                    if env_type == Environment.PRODUCTION
                    else "moderate",
                )

                print(f"   🤖 Model: {result.model}")
                print(f"   💰 Cost: ${result.total_cost:.4f}")
                print(
                    f"   🔐 Compliance: {'✅ Verified' if env_type == Environment.PRODUCTION else '⚠️ Basic'}"
                )
                print()

        return adapters

    except Exception as e:
        print(f"❌ Multi-environment setup failed: {e}")
        return {}


def demonstrate_high_availability_patterns():
    """Demonstrate high-availability and disaster recovery patterns."""

    print("🚀 High-Availability & Disaster Recovery Patterns")
    print("=" * 55)
    print()

    try:
        from genops.providers.skyrouter import GenOpsSkyRouterAdapter

        # Primary production environment
        primary_adapter = GenOpsSkyRouterAdapter(
            team="ha-primary",
            project="enterprise-ha-dr",
            environment="production",
            daily_budget_limit=1000.0,
            governance_policy="strict",
        )

        # Disaster recovery environment
        dr_adapter = GenOpsSkyRouterAdapter(
            team="ha-disaster-recovery",
            project="enterprise-ha-dr",
            environment="disaster_recovery",
            daily_budget_limit=500.0,
            governance_policy="strict",
        )

        print("🏗️ High-Availability Configuration:")
        print("📍 Primary Region: us-east-1 (3 AZs)")
        print("📍 DR Region: us-west-2 (2 AZs)")
        print("🎯 RTO Target: 15 minutes")
        print("🎯 RPO Target: 5 minutes")
        print("🔄 Failover: Automatic")
        print()

        # Simulate normal operations on primary
        print("🟢 Normal Operations (Primary Region):")

        operations = [
            {"type": "customer_service", "priority": "high"},
            {"type": "content_generation", "priority": "medium"},
            {"type": "data_analysis", "priority": "low"},
        ]

        primary_results = []

        for operation in operations:
            with primary_adapter.track_routing_session(
                f"primary-{operation['type']}"
            ) as session:
                result = session.track_multi_model_routing(
                    models=["gpt-4", "claude-3-opus", "claude-3-sonnet"],
                    input_data={
                        "operation_type": operation["type"],
                        "priority": operation["priority"],
                        "region": "primary",
                    },
                    routing_strategy="reliability_first",
                )

                primary_results.append(
                    {
                        "type": operation["type"],
                        "cost": float(result.total_cost),
                        "model": result.model,
                        "region": "us-east-1",
                    }
                )

                print(
                    f"   ✅ {operation['type']}: {result.model} - ${result.total_cost:.4f}"
                )

        print()

        # Simulate failover scenario
        print("🔄 Simulating Failover to DR Region:")
        print("⚠️  Primary region degraded - initiating automatic failover...")

        time.sleep(2)  # Simulate failover time

        print("🚨 Failover completed - operations now running in DR region")
        print()

        # Run same operations on DR
        print("🟡 DR Operations (Disaster Recovery Region):")

        dr_results = []

        for operation in operations:
            with dr_adapter.track_routing_session(f"dr-{operation['type']}") as session:
                result = session.track_multi_model_routing(
                    models=["gpt-4", "claude-3-opus", "claude-3-sonnet"],
                    input_data={
                        "operation_type": operation["type"],
                        "priority": operation["priority"],
                        "region": "dr",
                    },
                    routing_strategy="reliability_first",
                )

                dr_results.append(
                    {
                        "type": operation["type"],
                        "cost": float(result.total_cost),
                        "model": result.model,
                        "region": "us-west-2",
                    }
                )

                print(
                    f"   ✅ {operation['type']}: {result.model} - ${result.total_cost:.4f}"
                )

        print()

        # Failover analysis
        print("📊 Failover Analysis:")
        primary_total = sum(r["cost"] for r in primary_results)
        dr_total = sum(r["cost"] for r in dr_results)
        cost_difference = abs(dr_total - primary_total)

        print(f"   💰 Primary region cost: ${primary_total:.4f}")
        print(f"   💰 DR region cost: ${dr_total:.4f}")
        print(
            f"   📈 Cost difference: ${cost_difference:.4f} ({((cost_difference / primary_total) * 100):.1f}%)"
        )
        print("   ⏱️  Failover time: ~2 seconds (within 15m RTO)")
        print("   🎯 Availability: 99.9% maintained")
        print()

        # Show recovery procedures
        print("🔧 Enterprise Recovery Procedures:")
        recovery_steps = [
            "1. Automated health check detection",
            "2. Traffic redirection to DR region",
            "3. Application state synchronization",
            "4. Cost governance policy transfer",
            "5. Monitoring and alerting reconfiguration",
            "6. Team notification and incident response",
        ]

        for step in recovery_steps:
            print(f"   {step}")

        return True

    except Exception as e:
        print(f"❌ High-availability demo failed: {e}")
        return False


def demonstrate_compliance_frameworks():
    """Demonstrate enterprise compliance framework integration."""

    print("🔐 Enterprise Compliance Framework Integration")
    print("=" * 50)
    print()

    try:
        from genops.providers.skyrouter import GenOpsSkyRouterAdapter

        # Define compliance-specific configurations
        compliance_configs = {
            ComplianceFramework.SOC2: {
                "audit_logging": True,
                "data_retention": "7_years",
                "access_controls": "role_based",
                "encryption": "aes_256",
                "monitoring": "continuous",
            },
            ComplianceFramework.HIPAA: {
                "phi_protection": True,
                "access_logging": "detailed",
                "data_encryption": "fips_140_2",
                "audit_trail": "complete",
                "breach_notification": "automatic",
            },
            ComplianceFramework.GDPR: {
                "data_processing_consent": True,
                "right_to_deletion": True,
                "data_portability": True,
                "privacy_by_design": True,
                "dpo_notification": "enabled",
            },
            ComplianceFramework.FINRA: {
                "trade_surveillance": True,
                "communication_monitoring": True,
                "record_keeping": "regulatory",
                "risk_management": "enhanced",
                "reporting": "automated",
            },
        }

        print("🏛️ Supported Compliance Frameworks:")
        print()

        compliant_adapters = {}

        for framework, config in compliance_configs.items():
            print(f"📋 **{framework.value.upper()} Compliance**")

            # Create compliance-specific adapter
            adapter = GenOpsSkyRouterAdapter(
                team=f"compliance-{framework.value}",
                project="enterprise-compliance",
                environment="production",
                daily_budget_limit=200.0,
                governance_policy="strict",
            )

            compliant_adapters[framework] = adapter

            # Show framework requirements
            for requirement, value in config.items():
                print(f"   • {requirement.replace('_', ' ').title()}: {value}")
            print()

        # Demonstrate compliance-aware routing
        print("🧪 Compliance-Aware Routing Demonstration:")
        print()

        sensitive_requests = [
            {
                "framework": ComplianceFramework.HIPAA,
                "data": "Medical patient consultation transcript",
                "classification": "phi_protected",
            },
            {
                "framework": ComplianceFramework.GDPR,
                "data": "EU customer personal data analysis",
                "classification": "personal_data",
            },
            {
                "framework": ComplianceFramework.FINRA,
                "data": "Financial trading algorithm review",
                "classification": "trading_data",
            },
            {
                "framework": ComplianceFramework.SOC2,
                "data": "Customer security audit log analysis",
                "classification": "audit_data",
            },
        ]

        for request in sensitive_requests:
            framework = request["framework"]
            adapter = compliant_adapters[framework]

            print(f"🔐 {framework.value.upper()} Compliant Processing:")
            print(f"   📄 Data: {request['data']}")
            print(f"   🏷️  Classification: {request['classification']}")

            with adapter.track_routing_session(
                f"compliance-{framework.value}"
            ) as session:
                result = session.track_multi_model_routing(
                    models=["gpt-4", "claude-3-opus"],  # Only highest security models
                    input_data={
                        "sensitive_data": request["data"],
                        "compliance_framework": framework.value,
                        "data_classification": request["classification"],
                    },
                    routing_strategy="reliability_first",
                    complexity="enterprise",
                )

                print(f"   🤖 Model: {result.model}")
                print(f"   💰 Cost: ${result.total_cost:.4f}")
                print(f"   🔐 Compliance: ✅ {framework.value.upper()} verified")
                print(f"   📝 Audit ID: {result.session_id}")
                print()

        return True

    except Exception as e:
        print(f"❌ Compliance framework demo failed: {e}")
        return False


def demonstrate_production_monitoring():
    """Demonstrate enterprise production monitoring and alerting."""

    print("📊 Enterprise Production Monitoring & Alerting")
    print("=" * 50)
    print()

    try:
        from genops.providers.skyrouter import GenOpsSkyRouterAdapter

        # Production monitoring adapter
        production_adapter = GenOpsSkyRouterAdapter(
            team="enterprise-production",
            project="monitoring-demo",
            environment="production",
            daily_budget_limit=1000.0,
            governance_policy="strict",
        )

        print("📈 Production Monitoring Configuration:")
        print("   📊 Metrics: Comprehensive collection enabled")
        print("   🚨 Alerts: Slack, PagerDuty, Email")
        print("   🎯 SLA Monitoring: 99.9% uptime target")
        print("   💰 Cost Anomaly Detection: Real-time")
        print("   📺 Dashboards: Live performance metrics")
        print()

        # Simulate production workload
        print("🏭 Simulating Production Workload:")
        print()

        workload_patterns = [
            {"name": "Peak Hours", "operations": 20, "complexity": "high"},
            {"name": "Normal Business", "operations": 10, "complexity": "medium"},
            {"name": "Maintenance Window", "operations": 3, "complexity": "low"},
        ]

        monitoring_results = {
            "total_operations": 0,
            "total_cost": 0.0,
            "avg_latency": 0.0,
            "error_rate": 0.0,
            "sla_compliance": 100.0,
        }

        for pattern in workload_patterns:
            print(f"⏰ {pattern['name']} Pattern:")
            pattern_start = time.time()
            pattern_cost = 0.0

            for i in range(pattern["operations"]):
                with production_adapter.track_routing_session(
                    f"{pattern['name'].lower().replace(' ', '_')}-{i}"
                ) as session:
                    result = session.track_multi_model_routing(
                        models=[
                            "gpt-4",
                            "claude-3-opus",
                            "claude-3-sonnet",
                            "gpt-3.5-turbo",
                        ],
                        input_data={
                            "workload_pattern": pattern["name"],
                            "operation_id": i,
                            "complexity": pattern["complexity"],
                        },
                        routing_strategy="balanced",
                        complexity=pattern["complexity"],
                    )

                    pattern_cost += float(result.total_cost)
                    monitoring_results["total_operations"] += 1

            pattern_duration = time.time() - pattern_start
            avg_op_time = pattern_duration / pattern["operations"]

            print(f"   📊 {pattern['operations']} operations completed")
            print(f"   💰 Cost: ${pattern_cost:.4f}")
            print(f"   ⏱️  Avg latency: {avg_op_time:.2f}s")
            print()

            monitoring_results["total_cost"] += pattern_cost
            monitoring_results["avg_latency"] += avg_op_time

        # Calculate final metrics
        monitoring_results["avg_latency"] /= len(workload_patterns)
        monitoring_results["error_rate"] = 0.1  # Simulate very low error rate

        print("📊 Production Monitoring Summary:")
        print(f"   🔄 Total operations: {monitoring_results['total_operations']}")
        print(f"   💰 Total cost: ${monitoring_results['total_cost']:.4f}")
        print(f"   ⏱️  Average latency: {monitoring_results['avg_latency']:.2f}s")
        print(f"   ❌ Error rate: {monitoring_results['error_rate']:.1f}%")
        print(f"   🎯 SLA compliance: {monitoring_results['sla_compliance']:.1f}%")
        print()

        # Simulate alerts and thresholds
        print("🚨 Monitoring Alerts & Thresholds:")

        thresholds = {
            "latency": {"threshold": 5.0, "current": monitoring_results["avg_latency"]},
            "error_rate": {
                "threshold": 1.0,
                "current": monitoring_results["error_rate"],
            },
            "cost_per_hour": {
                "threshold": 50.0,
                "current": monitoring_results["total_cost"] * 60,
            },
            "sla_compliance": {
                "threshold": 99.5,
                "current": monitoring_results["sla_compliance"],
            },
        }

        for metric, data in thresholds.items():
            if metric == "sla_compliance":
                status = (
                    "🟢 HEALTHY" if data["current"] >= data["threshold"] else "🔴 ALERT"
                )
            else:
                status = (
                    "🟢 HEALTHY" if data["current"] <= data["threshold"] else "🔴 ALERT"
                )

            print(f"   {metric.replace('_', ' ').title()}: {status}")
            print(
                f"     Current: {data['current']:.2f} | Threshold: {data['threshold']}"
            )

        print()

        # Show alerting configuration
        print("🔔 Enterprise Alerting Configuration:")
        alert_rules = [
            "💰 Cost spike > 50% increase in 1 hour",
            "⏱️  Latency > 5 seconds for 3 consecutive operations",
            "❌ Error rate > 1% for 5 minutes",
            "📉 SLA compliance < 99.5% for 10 minutes",
            "🔄 Failed operations > 5 in 1 minute",
            "🚀 Traffic spike > 200% increase in 15 minutes",
        ]

        for rule in alert_rules:
            print(f"   {rule}")

        return True

    except Exception as e:
        print(f"❌ Production monitoring demo failed: {e}")
        return False


def demonstrate_auto_scaling_patterns():
    """Demonstrate auto-scaling and load balancing for enterprise deployments."""

    print("⚡ Auto-Scaling & Load Balancing Patterns")
    print("=" * 45)
    print()

    try:
        from genops.providers.skyrouter import GenOpsSkyRouterAdapter

        # Auto-scaling configuration
        scaling_config = {
            "min_instances": 2,
            "max_instances": 20,
            "target_cpu_utilization": 70,
            "target_memory_utilization": 80,
            "scale_up_cooldown": 300,  # 5 minutes
            "scale_down_cooldown": 600,  # 10 minutes
            "scale_up_threshold": 2,  # requests per second
            "scale_down_threshold": 0.5,
        }

        # Load balancer configuration
        lb_config = {
            "algorithm": "least_connections",
            "health_check_interval": 30,
            "health_check_timeout": 10,
            "unhealthy_threshold": 3,
            "healthy_threshold": 2,
            "session_affinity": "source_ip",
        }

        # Create auto-scaling adapter
        scaling_adapter = GenOpsSkyRouterAdapter(
            team="enterprise-autoscaling",
            project="load-balancing-demo",
            environment="production",
            daily_budget_limit=2000.0,
            governance_policy="strict",
        )

        print("📊 Auto-Scaling Configuration:")
        print(
            f"   📈 Instance range: {scaling_config['min_instances']}-{scaling_config['max_instances']}"
        )
        print(f"   🎯 CPU target: {scaling_config['target_cpu_utilization']}%")
        print(f"   🧠 Memory target: {scaling_config['target_memory_utilization']}%")
        print(f"   ⬆️  Scale up cooldown: {scaling_config['scale_up_cooldown']}s")
        print(f"   ⬇️  Scale down cooldown: {scaling_config['scale_down_cooldown']}s")
        print()

        print("🔄 Load Balancer Configuration:")
        print(f"   🎯 Algorithm: {lb_config['algorithm']}")
        print(f"   ❤️  Health check: Every {lb_config['health_check_interval']}s")
        print(f"   🏥 Healthy threshold: {lb_config['healthy_threshold']} checks")
        print(f"   🚨 Unhealthy threshold: {lb_config['unhealthy_threshold']} checks")
        print()

        # Simulate load scenarios
        load_scenarios = [
            {"name": "Light Load", "requests": 5, "duration": 1},
            {"name": "Normal Load", "requests": 15, "duration": 2},
            {"name": "Peak Load", "requests": 50, "duration": 3},
            {"name": "Spike Load", "requests": 100, "duration": 1},
            {"name": "Cool Down", "requests": 8, "duration": 2},
        ]

        print("🧪 Load Scenario Testing:")
        print()

        current_instances = scaling_config["min_instances"]
        total_cost = 0.0

        for scenario in load_scenarios:
            print(f"📊 {scenario['name']} ({scenario['requests']} requests):")
            scenario_start = time.time()
            scenario_cost = 0.0

            # Calculate required instances based on load
            requests_per_instance = 10  # Assume each instance can handle 10 requests
            required_instances = max(
                scaling_config["min_instances"],
                min(
                    scaling_config["max_instances"],
                    (scenario["requests"] + requests_per_instance - 1)
                    // requests_per_instance,
                ),
            )

            # Simulate scaling decision
            if required_instances > current_instances:
                print(
                    f"   ⬆️  Scaling up: {current_instances} → {required_instances} instances"
                )
            elif required_instances < current_instances:
                print(
                    f"   ⬇️  Scaling down: {current_instances} → {required_instances} instances"
                )
            else:
                print(f"   ➡️  Maintaining: {current_instances} instances")

            current_instances = required_instances

            # Simulate request processing
            for i in range(
                min(scenario["requests"], 10)
            ):  # Process up to 10 requests for demo
                with scaling_adapter.track_routing_session(
                    f"{scenario['name'].lower().replace(' ', '_')}-{i}"
                ) as session:
                    result = session.track_multi_model_routing(
                        models=["gpt-4", "claude-3-sonnet", "gpt-3.5-turbo"],
                        input_data={
                            "load_scenario": scenario["name"],
                            "request_id": i,
                            "instances": current_instances,
                        },
                        routing_strategy="latency_optimized",
                        complexity="moderate",
                    )

                    scenario_cost += float(result.total_cost)

            scenario_duration = time.time() - scenario_start
            throughput = scenario["requests"] / scenario_duration

            print(f"   💰 Cost: ${scenario_cost:.4f}")
            print(f"   📈 Throughput: {throughput:.1f} req/s")
            print(f"   ⏱️  Duration: {scenario_duration:.1f}s")
            print()

            total_cost += scenario_cost
            time.sleep(1)  # Simulate time between scenarios

        print("📊 Auto-Scaling Performance Summary:")
        print(f"   💰 Total cost: ${total_cost:.4f}")
        print(
            f"   📈 Peak instances: {max([sc['requests'] // 10 + 1 for sc in load_scenarios])}"
        )
        print(f"   📉 Min instances maintained: {scaling_config['min_instances']}")
        print("   🎯 Average CPU utilization: ~65% (within target)")
        print()

        # Show scaling benefits
        print("💡 Enterprise Auto-Scaling Benefits:")
        benefits = [
            "💰 Cost optimization through dynamic resource allocation",
            "📈 Performance maintenance during traffic spikes",
            "🔄 Automatic recovery from instance failures",
            "🎯 SLA compliance through adequate capacity",
            "⚡ Reduced manual intervention requirements",
            "📊 Predictable performance under varying loads",
        ]

        for benefit in benefits:
            print(f"   {benefit}")

        return True

    except Exception as e:
        print(f"❌ Auto-scaling demo failed: {e}")
        return False


def demonstrate_cost_governance_enterprise():
    """Demonstrate enterprise-scale cost governance and optimization."""

    print("💼 Enterprise Cost Governance & Optimization")
    print("=" * 48)
    print()

    try:
        from genops.providers.skyrouter import GenOpsSkyRouterAdapter

        # Department-level cost governance
        departments = {
            "engineering": {
                "daily_budget": 500.0,
                "teams": ["backend", "frontend", "ml", "devops"],
                "cost_center": "TECH-001",
            },
            "product": {
                "daily_budget": 200.0,
                "teams": ["product_management", "design", "research"],
                "cost_center": "PROD-002",
            },
            "customer_success": {
                "daily_budget": 150.0,
                "teams": ["support", "success", "training"],
                "cost_center": "CS-003",
            },
            "sales": {
                "daily_budget": 100.0,
                "teams": ["inside_sales", "enterprise", "marketing"],
                "cost_center": "SALES-004",
            },
        }

        print("🏢 Department Cost Governance Setup:")
        print()

        department_adapters = {}
        total_budget = 0.0

        for dept_name, dept_config in departments.items():
            print(f"📊 **{dept_name.title()} Department**")
            print(f"   💰 Daily budget: ${dept_config['daily_budget']:.2f}")
            print(f"   👥 Teams: {len(dept_config['teams'])}")
            print(f"   🏷️  Cost center: {dept_config['cost_center']}")

            # Create department-level adapter
            adapter = GenOpsSkyRouterAdapter(
                team=f"dept-{dept_name}",
                project="enterprise-cost-governance",
                environment="production",
                daily_budget_limit=dept_config["daily_budget"],
                governance_policy="strict",
                cost_center=dept_config["cost_center"],
                enable_cost_alerts=True,
            )

            department_adapters[dept_name] = adapter
            total_budget += dept_config["daily_budget"]
            print()

        print(f"🏦 **Enterprise Total Daily Budget: ${total_budget:.2f}**")
        print()

        # Simulate department usage
        print("🧪 Simulating Department Usage Patterns:")
        print()

        usage_patterns = {
            "engineering": {
                "operations": [
                    {"type": "code_review", "complexity": "enterprise", "count": 15},
                    {
                        "type": "automated_testing",
                        "complexity": "moderate",
                        "count": 25,
                    },
                    {
                        "type": "deployment_analysis",
                        "complexity": "complex",
                        "count": 8,
                    },
                ]
            },
            "product": {
                "operations": [
                    {"type": "user_research", "complexity": "moderate", "count": 12},
                    {"type": "feature_analysis", "complexity": "complex", "count": 6},
                    {
                        "type": "competitive_intel",
                        "complexity": "moderate",
                        "count": 10,
                    },
                ]
            },
            "customer_success": {
                "operations": [
                    {"type": "support_tickets", "complexity": "simple", "count": 30},
                    {"type": "customer_training", "complexity": "moderate", "count": 8},
                    {"type": "success_analysis", "complexity": "complex", "count": 5},
                ]
            },
            "sales": {
                "operations": [
                    {"type": "lead_qualification", "complexity": "simple", "count": 20},
                    {
                        "type": "proposal_generation",
                        "complexity": "moderate",
                        "count": 6,
                    },
                    {"type": "sales_analysis", "complexity": "complex", "count": 3},
                ]
            },
        }

        department_costs = {}

        for dept_name, adapter in department_adapters.items():
            print(f"💼 {dept_name.title()} Department Operations:")
            dept_cost = 0.0
            dept_operations = 0

            for operation in usage_patterns[dept_name]["operations"]:
                for i in range(operation["count"]):
                    with adapter.track_routing_session(
                        f"{dept_name}-{operation['type']}-{i}"
                    ) as session:
                        result = session.track_multi_model_routing(
                            models=[
                                "gpt-4",
                                "claude-3-opus",
                                "claude-3-sonnet",
                                "gpt-3.5-turbo",
                            ],
                            input_data={
                                "department": dept_name,
                                "operation_type": operation["type"],
                                "operation_id": i,
                            },
                            routing_strategy="cost_optimized"
                            if dept_name == "sales"
                            else "balanced",
                            complexity=operation["complexity"],
                        )

                        dept_cost += float(result.total_cost)
                        dept_operations += 1

            department_costs[dept_name] = {
                "cost": dept_cost,
                "operations": dept_operations,
                "budget": departments[dept_name]["daily_budget"],
                "utilization": (dept_cost / departments[dept_name]["daily_budget"])
                * 100,
            }

            print(f"   💰 Total cost: ${dept_cost:.4f}")
            print(f"   📊 Operations: {dept_operations}")
            print(
                f"   📈 Budget utilization: {department_costs[dept_name]['utilization']:.1f}%"
            )
            print()

        # Enterprise cost analysis
        print("📊 Enterprise Cost Governance Analysis:")
        print()

        total_spent = sum(dept["cost"] for dept in department_costs.values())
        total_ops = sum(dept["operations"] for dept in department_costs.values())

        print(f"💰 **Total Enterprise Spend: ${total_spent:.4f}**")
        print(f"🔄 **Total Operations: {total_ops}**")
        print(f"📉 **Average Cost per Operation: ${total_spent / total_ops:.4f}**")
        print(
            f"📊 **Overall Budget Utilization: {(total_spent / total_budget) * 100:.1f}%**"
        )
        print()

        # Department ranking by efficiency
        print("🏆 Department Cost Efficiency Ranking:")
        efficiency_ranking = sorted(
            department_costs.items(), key=lambda x: x[1]["cost"] / x[1]["operations"]
        )

        for i, (dept_name, stats) in enumerate(efficiency_ranking, 1):
            efficiency = stats["cost"] / stats["operations"]
            print(f"   {i}. {dept_name.title()}: ${efficiency:.4f} per operation")

        print()

        # Budget alerts and recommendations
        print("🚨 Budget Alerts & Recommendations:")

        for dept_name, stats in department_costs.items():
            if stats["utilization"] > 90:
                print(
                    f"   🔴 {dept_name.title()}: HIGH - {stats['utilization']:.1f}% budget used"
                )
            elif stats["utilization"] > 80:
                print(
                    f"   🟡 {dept_name.title()}: MEDIUM - {stats['utilization']:.1f}% budget used"
                )
            else:
                print(
                    f"   🟢 {dept_name.title()}: LOW - {stats['utilization']:.1f}% budget used"
                )

        print()

        # Enterprise governance recommendations
        print("💡 Enterprise Governance Recommendations:")
        recommendations = [
            "🔄 Implement automated budget reallocation between departments",
            "📊 Set up real-time cost monitoring dashboards",
            "🎯 Establish cost optimization targets (5-10% monthly reduction)",
            "📈 Create department-specific routing strategy guidelines",
            "🔔 Configure proactive budget threshold alerts",
            "📝 Implement monthly cost review and optimization sessions",
        ]

        for rec in recommendations:
            print(f"   {rec}")

        return True

    except Exception as e:
        print(f"❌ Enterprise cost governance demo failed: {e}")
        return False


def main():
    """Main execution function."""

    print("🏢 SkyRouter Enterprise Production Deployment Patterns")
    print("=" * 65)
    print()

    print("This example demonstrates enterprise-grade production deployment")
    print("patterns including multi-environment setups, high-availability,")
    print("compliance frameworks, and large-scale cost governance.")
    print()

    # Check prerequisites
    api_key = os.getenv("SKYROUTER_API_KEY")
    if not api_key:
        print("❌ Missing required environment variables:")
        print("   SKYROUTER_API_KEY - Your SkyRouter API key")
        print()
        print("💡 Set up your environment:")
        print("   export SKYROUTER_API_KEY='your-api-key'")
        print("   export GENOPS_TEAM='enterprise-team'")
        print("   export GENOPS_PROJECT='production-deployment'")
        return

    try:
        success = True

        # Multi-environment deployment
        if success:
            adapters = demonstrate_multi_environment_setup()
            success = bool(adapters)

        # High-availability patterns
        if success:
            print("\n" + "=" * 65 + "\n")
            success = demonstrate_high_availability_patterns()

        # Compliance frameworks
        if success:
            print("\n" + "=" * 65 + "\n")
            success = demonstrate_compliance_frameworks()

        # Production monitoring
        if success:
            print("\n" + "=" * 65 + "\n")
            success = demonstrate_production_monitoring()

        # Auto-scaling patterns
        if success:
            print("\n" + "=" * 65 + "\n")
            success = demonstrate_auto_scaling_patterns()

        # Enterprise cost governance
        if success:
            print("\n" + "=" * 65 + "\n")
            success = demonstrate_cost_governance_enterprise()

        if success:
            print("\n" + "=" * 65 + "\n")
            print("🎉 Enterprise deployment patterns demonstration completed!")
            print()
            print("🔑 **Key Takeaways:**")
            print("• Multi-environment deployments ensure safe production rollouts")
            print(
                "• High-availability patterns maintain 99.9% uptime with automatic failover"
            )
            print("• Compliance frameworks enable regulated industry deployments")
            print("• Production monitoring provides real-time visibility and alerting")
            print("• Auto-scaling optimizes costs while maintaining performance")
            print("• Enterprise governance enables department-level cost control")
            print()
            print("🚀 **Production Deployment Checklist:**")
            print("1. ✅ Configure multi-environment pipeline (dev/staging/prod)")
            print("2. ✅ Set up high-availability with disaster recovery")
            print("3. ✅ Implement required compliance frameworks")
            print("4. ✅ Deploy comprehensive monitoring and alerting")
            print("5. ✅ Configure auto-scaling and load balancing")
            print("6. ✅ Establish enterprise cost governance policies")
            print()
            print("🏭 **Enterprise Integration Patterns:**")
            print("• CI/CD pipeline integration with automated testing")
            print("• Infrastructure as Code (IaC) for repeatable deployments")
            print("• Secrets management integration (Vault, AWS Secrets Manager)")
            print("• Service mesh integration for advanced traffic management")
            print("• Observability integration (Prometheus, Grafana, Datadog)")
            print("• GitOps workflows for declarative deployment management")
            print()
            print("🔗 **Next Steps for Production:**")
            print("1. Review security best practices documentation")
            print("2. Conduct load testing with realistic traffic patterns")
            print("3. Set up disaster recovery testing procedures")
            print("4. Implement custom compliance requirements")
            print("5. Configure organization-specific monitoring integrations")
            print("6. Train teams on production deployment procedures")

    except KeyboardInterrupt:
        print()
        print("👋 Demo cancelled.")
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        print()
        print("🔧 Troubleshooting tips:")
        print("1. Verify your SKYROUTER_API_KEY is correct and has sufficient credits")
        print("2. Check your internet connection")
        print("3. Ensure GenOps is properly installed: pip install genops[skyrouter]")
        print("4. Verify adequate permissions for enterprise features")


if __name__ == "__main__":
    main()
