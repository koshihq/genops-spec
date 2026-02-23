#!/usr/bin/env python3
"""
Production Deployment Patterns for CrewAI + GenOps

Enterprise-ready deployment patterns, scaling strategies, and production best practices
for CrewAI multi-agent systems with comprehensive GenOps governance.

Usage:
    python production_deployment_patterns.py [--pattern PATTERN] [--scale SCALE]

Features:
    - Production-ready configuration patterns
    - Auto-scaling and load balancing strategies
    - Multi-environment deployment (dev/staging/prod)
    - Fault tolerance and disaster recovery
    - Enterprise security and compliance
    - Performance monitoring and alerting at scale

Time to Complete: ~60 minutes
Learning Outcomes: Production deployment and enterprise scaling patterns
"""

import argparse
import concurrent.futures
import logging
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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
        CrewAIAgentMonitor,  # noqa: F401
        CrewAICostAggregator,  # noqa: F401
        GenOpsCrewAIAdapter,
        auto_instrument,  # noqa: F401
        print_validation_result,
        validate_crewai_setup,
    )
except ImportError:
    print("❌ GenOps not installed. Install with: pip install genops-ai[crewai]")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeploymentEnvironment(Enum):
    """Deployment environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DISASTER_RECOVERY = "disaster_recovery"


class ScalingStrategy(Enum):
    """Scaling strategy types."""

    FIXED = "fixed"
    AUTO_SCALE = "auto_scale"
    PREDICTIVE = "predictive"
    BURST = "burst"


@dataclass
class ProductionConfig:
    """Production deployment configuration."""

    environment: DeploymentEnvironment
    scaling_strategy: ScalingStrategy
    max_concurrent_crews: int
    daily_budget_limit: float
    governance_policy: str
    monitoring_level: str
    security_config: dict[str, Any]
    performance_thresholds: dict[str, float]
    alert_settings: dict[str, Any]


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""

    cpu_usage: float
    memory_usage: float
    active_crews: int
    queued_requests: int
    response_time_p95: float
    error_rate: float
    cost_per_hour: float
    throughput: float


@dataclass
class ScalingEvent:
    """Auto-scaling event record."""

    event_id: str
    timestamp: datetime
    trigger: str
    action: str
    current_scale: int
    target_scale: int
    resource_metrics: ResourceMetrics
    decision_factors: dict[str, Any]


class ProductionDeploymentManager:
    """Manages production deployment patterns and scaling."""

    def __init__(self, deployment_pattern: str = "standard", scale_factor: int = 1):
        self.deployment_pattern = deployment_pattern
        self.scale_factor = scale_factor
        self.environments = self._initialize_environments()
        self.active_crews = {}
        self.resource_history = []
        self.scaling_events = []
        self.request_queue = []

        # Thread pool for concurrent operations
        self.executor = ThreadPoolExecutor(max_workers=10)

    def _initialize_environments(self) -> dict[DeploymentEnvironment, ProductionConfig]:
        """Initialize production environment configurations."""
        environments = {
            DeploymentEnvironment.DEVELOPMENT: ProductionConfig(
                environment=DeploymentEnvironment.DEVELOPMENT,
                scaling_strategy=ScalingStrategy.FIXED,
                max_concurrent_crews=2,
                daily_budget_limit=10.0,
                governance_policy="advisory",
                monitoring_level="basic",
                security_config={
                    "encryption": False,
                    "access_control": "basic",
                    "audit_logging": False,
                },
                performance_thresholds={
                    "max_response_time": 120.0,
                    "max_error_rate": 0.10,
                    "max_cpu_usage": 0.80,
                },
                alert_settings={"enabled": False, "channels": ["console"]},
            ),
            DeploymentEnvironment.STAGING: ProductionConfig(
                environment=DeploymentEnvironment.STAGING,
                scaling_strategy=ScalingStrategy.AUTO_SCALE,
                max_concurrent_crews=5,
                daily_budget_limit=50.0,
                governance_policy="enforced",
                monitoring_level="enhanced",
                security_config={
                    "encryption": True,
                    "access_control": "rbac",
                    "audit_logging": True,
                },
                performance_thresholds={
                    "max_response_time": 60.0,
                    "max_error_rate": 0.05,
                    "max_cpu_usage": 0.70,
                },
                alert_settings={"enabled": True, "channels": ["email", "slack"]},
            ),
            DeploymentEnvironment.PRODUCTION: ProductionConfig(
                environment=DeploymentEnvironment.PRODUCTION,
                scaling_strategy=ScalingStrategy.PREDICTIVE,
                max_concurrent_crews=20,
                daily_budget_limit=500.0,
                governance_policy="strict",
                monitoring_level="comprehensive",
                security_config={
                    "encryption": True,
                    "access_control": "rbac_with_mfa",
                    "audit_logging": True,
                    "data_classification": True,
                    "compliance_scanning": True,
                },
                performance_thresholds={
                    "max_response_time": 30.0,
                    "max_error_rate": 0.01,
                    "max_cpu_usage": 0.60,
                },
                alert_settings={
                    "enabled": True,
                    "channels": ["pagerduty", "email", "slack", "sms"],
                    "escalation_rules": True,
                },
            ),
        }

        return environments

    def setup_validation(self) -> bool:
        """Validate production deployment setup."""
        print("🔍 Validating production deployment setup...")

        result = validate_crewai_setup(quick=False)

        if result.is_valid:
            print("✅ Production deployment setup validated")
            print(f"   🏗️ Deployment pattern: {self.deployment_pattern}")
            print(f"   📏 Scale factor: {self.scale_factor}")
            print(f"   🌍 Environments configured: {len(self.environments)}")
            return True
        else:
            print("❌ Setup issues found:")
            print_validation_result(result)
            return False

    def create_production_crew(
        self, crew_type: str, environment: DeploymentEnvironment
    ) -> Crew:
        """Create a production-ready crew with appropriate configuration."""
        self.environments[environment]

        print(f"\n🏗️ Creating production crew for {environment.value} environment...")

        # Adjust agents based on environment and crew type
        if crew_type == "customer_service":
            agents = [
                Agent(
                    role="Customer Service Specialist",
                    goal="Provide excellent customer support with quick resolution",
                    backstory="Experienced customer service expert with deep product knowledge",
                    verbose=True,
                ),
                Agent(
                    role="Technical Support Engineer",
                    goal="Resolve technical issues and provide solutions",
                    backstory="Technical expert specializing in troubleshooting and problem resolution",
                    verbose=True,
                ),
                Agent(
                    role="Escalation Manager",
                    goal="Handle complex issues requiring senior intervention",
                    backstory="Senior manager with authority to make decisions and escalate issues",
                    verbose=True,
                ),
            ]

            tasks = [
                Task(
                    description="Analyze customer inquiry and provide initial assessment",
                    agent=agents[0],
                ),
                Task(
                    description="Provide technical resolution or escalate if needed",
                    agent=agents[1],
                ),
                Task(
                    description="Final review and ensure customer satisfaction",
                    agent=agents[2],
                ),
            ]

        elif crew_type == "content_generation":
            agents = [
                Agent(
                    role="Content Strategist",
                    goal="Develop content strategy aligned with business goals",
                    backstory="Strategic content expert with market research expertise",
                    verbose=True,
                ),
                Agent(
                    role="Content Creator",
                    goal="Create high-quality, engaging content",
                    backstory="Creative writer with expertise in various content formats",
                    verbose=True,
                ),
                Agent(
                    role="Quality Editor",
                    goal="Ensure content quality and brand compliance",
                    backstory="Editorial expert with brand guidelines knowledge",
                    verbose=True,
                ),
            ]

            tasks = [
                Task(
                    description="Research topic and develop content strategy",
                    agent=agents[0],
                ),
                Task(
                    description="Create content following strategic guidelines",
                    agent=agents[1],
                ),
                Task(
                    description="Review and edit content for quality and compliance",
                    agent=agents[2],
                ),
            ]

        elif crew_type == "data_analysis":
            agents = [
                Agent(
                    role="Data Analyst",
                    goal="Extract insights from data with statistical rigor",
                    backstory="Experienced data analyst with advanced statistical knowledge",
                    verbose=True,
                ),
                Agent(
                    role="Business Intelligence Specialist",
                    goal="Transform data insights into business recommendations",
                    backstory="BI expert with deep understanding of business operations",
                    verbose=True,
                ),
                Agent(
                    role="Report Generator",
                    goal="Create clear, actionable reports for stakeholders",
                    backstory="Communication specialist focused on data visualization and reporting",
                    verbose=True,
                ),
            ]

            tasks = [
                Task(
                    description="Analyze data and identify key patterns and trends",
                    agent=agents[0],
                ),
                Task(
                    description="Interpret findings and generate business insights",
                    agent=agents[1],
                ),
                Task(
                    description="Create comprehensive report with recommendations",
                    agent=agents[2],
                ),
            ]
        else:
            # Default generic crew
            agents = [
                Agent(
                    role="General Purpose Agent",
                    goal="Complete assigned tasks efficiently",
                    backstory="Versatile agent capable of handling various tasks",
                    verbose=True,
                )
            ]
            tasks = [
                Task(
                    description="Complete the assigned task with high quality",
                    agent=agents[0],
                )
            ]

        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=1 if environment == DeploymentEnvironment.PRODUCTION else 2,
        )

        print(
            f"✅ Created {crew_type} crew with {len(agents)} agents for {environment.value}"
        )
        return crew

    def deploy_to_environment(
        self, environment: DeploymentEnvironment
    ) -> GenOpsCrewAIAdapter:
        """Deploy GenOps adapter to specific environment."""
        config = self.environments[environment]

        print(f"\n🚀 Deploying to {environment.value} environment...")

        # Create environment-specific adapter
        adapter = GenOpsCrewAIAdapter(
            team=f"production-{environment.value}",
            project="enterprise-crews",
            environment=environment.value,
            daily_budget_limit=config.daily_budget_limit,
            governance_policy=config.governance_policy,
            enable_agent_tracking=True,
            enable_task_tracking=True,
            enable_cost_tracking=True,
        )

        # Configure monitoring level
        if config.monitoring_level == "comprehensive":
            adapter.enable_advanced_monitoring = True
            adapter.enable_real_time_alerts = True

        print(f"   ✅ GenOps adapter deployed to {environment.value}")
        print(f"   💰 Budget limit: ${config.daily_budget_limit}")
        print(f"   🛡️ Governance: {config.governance_policy}")
        print(f"   📊 Monitoring: {config.monitoring_level}")

        return adapter

    def simulate_production_workload(
        self, environment: DeploymentEnvironment, duration_minutes: int = 5
    ) -> list[ResourceMetrics]:
        """Simulate production workload with realistic patterns."""
        print(f"\n⚡ Simulating production workload in {environment.value}")
        print(f"   ⏱️ Duration: {duration_minutes} minutes")

        adapter = self.deploy_to_environment(environment)
        config = self.environments[environment]

        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        resource_metrics = []

        # Simulate different crew types
        crew_types = ["customer_service", "content_generation", "data_analysis"]

        while time.time() < end_time:
            current_time = time.time()
            elapsed_minutes = (current_time - start_time) / 60

            # Simulate varying load patterns
            if elapsed_minutes < 1:
                # Ramp up
                load_factor = elapsed_minutes
                target_crews = max(1, int(config.max_concurrent_crews * load_factor))
            elif elapsed_minutes < duration_minutes - 1:
                # Peak load
                target_crews = config.max_concurrent_crews
            else:
                # Ramp down
                remaining = duration_minutes - elapsed_minutes
                load_factor = remaining
                target_crews = max(1, int(config.max_concurrent_crews * load_factor))

            # Execute crews based on target load
            active_crew_count = 0
            crew_futures = []

            for i in range(min(target_crews, 3)):  # Limit for demo
                crew_type = crew_types[i % len(crew_types)]
                crew = self.create_production_crew(crew_type, environment)

                # Submit to thread pool for concurrent execution
                future = self.executor.submit(
                    self._execute_crew_with_tracking, adapter, crew, crew_type, i
                )
                crew_futures.append(future)
                active_crew_count += 1

            # Simulate resource metrics
            cpu_usage = min(
                0.95, 0.20 + (active_crew_count / config.max_concurrent_crews) * 0.60
            )
            memory_usage = min(
                0.90, 0.30 + (active_crew_count / config.max_concurrent_crews) * 0.50
            )
            response_time = (
                15.0 + (active_crew_count / config.max_concurrent_crews) * 30.0
            )
            error_rate = (
                max(0.001, 0.02 * (cpu_usage - 0.70)) if cpu_usage > 0.70 else 0.001
            )

            metrics = ResourceMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                active_crews=active_crew_count,
                queued_requests=max(0, target_crews - active_crew_count),
                response_time_p95=response_time,
                error_rate=error_rate,
                cost_per_hour=0.50 * active_crew_count,
                throughput=active_crew_count * 2.0,  # tasks per minute
            )

            resource_metrics.append(metrics)
            self.resource_history.append(metrics)

            # Check for scaling triggers
            self._evaluate_scaling_triggers(environment, metrics)

            print(
                f"   📊 Minute {elapsed_minutes:.1f}: "
                f"CPU {cpu_usage:.1%}, Active crews: {active_crew_count}, "
                f"Response time: {response_time:.1f}s"
            )

            # Wait for crews to complete or timeout
            for future in crew_futures:
                try:
                    future.result(timeout=30)  # 30 second timeout per crew
                except concurrent.futures.TimeoutError:
                    print("   ⏰ Crew execution timeout")
                except Exception as e:
                    print(f"   ❌ Crew execution error: {e}")

            time.sleep(10)  # 10 second intervals for demo

        print(f"\n✅ Workload simulation completed for {environment.value}")
        return resource_metrics

    def _execute_crew_with_tracking(
        self, adapter: GenOpsCrewAIAdapter, crew: Crew, crew_type: str, crew_index: int
    ) -> dict[str, Any]:
        """Execute crew with full production tracking."""
        crew_id = f"{crew_type}-{crew_index}-{int(time.time())}"

        try:
            with adapter.track_crew(crew_id, use_case=crew_type) as context:
                result = crew.kickoff(
                    {
                        "production_mode": True,
                        "crew_type": crew_type,
                        "crew_index": crew_index,
                    }
                )

                metrics = context.get_metrics()

                return {
                    "crew_id": crew_id,
                    "crew_type": crew_type,
                    "result_length": len(str(result)),
                    "total_cost": metrics["total_cost"],
                    "execution_successful": True,
                }

        except Exception as e:
            logger.error(f"Crew execution failed: {e}")
            return {
                "crew_id": crew_id,
                "crew_type": crew_type,
                "error": str(e),
                "execution_successful": False,
            }

    def _evaluate_scaling_triggers(
        self, environment: DeploymentEnvironment, metrics: ResourceMetrics
    ):
        """Evaluate if scaling actions are needed."""
        config = self.environments[environment]

        # Skip scaling for fixed strategy
        if config.scaling_strategy == ScalingStrategy.FIXED:
            return

        scaling_needed = False
        scaling_action = "none"
        trigger_reason = ""

        # Scale up triggers
        if metrics.cpu_usage > config.performance_thresholds["max_cpu_usage"]:
            scaling_needed = True
            scaling_action = "scale_up"
            trigger_reason = f"CPU usage {metrics.cpu_usage:.1%} exceeds threshold"

        elif (
            metrics.response_time_p95
            > config.performance_thresholds["max_response_time"]
        ):
            scaling_needed = True
            scaling_action = "scale_up"
            trigger_reason = (
                f"Response time {metrics.response_time_p95:.1f}s exceeds threshold"
            )

        elif metrics.queued_requests > 5:
            scaling_needed = True
            scaling_action = "scale_up"
            trigger_reason = f"Queue backlog: {metrics.queued_requests} requests"

        # Scale down triggers (only if no scale up needed)
        elif metrics.cpu_usage < 0.30 and metrics.active_crews > 1:
            scaling_needed = True
            scaling_action = "scale_down"
            trigger_reason = f"Low CPU usage {metrics.cpu_usage:.1%}, over-provisioned"

        if scaling_needed:
            current_scale = metrics.active_crews
            if scaling_action == "scale_up":
                target_scale = min(config.max_concurrent_crews, current_scale + 2)
            else:  # scale_down
                target_scale = max(1, current_scale - 1)

            scaling_event = ScalingEvent(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                trigger=trigger_reason,
                action=scaling_action,
                current_scale=current_scale,
                target_scale=target_scale,
                resource_metrics=metrics,
                decision_factors={
                    "cpu_threshold_exceeded": metrics.cpu_usage
                    > config.performance_thresholds["max_cpu_usage"],
                    "response_time_exceeded": metrics.response_time_p95
                    > config.performance_thresholds["max_response_time"],
                    "queue_backlog": metrics.queued_requests > 5,
                },
            )

            self.scaling_events.append(scaling_event)

            print(
                f"   🔄 Scaling trigger: {scaling_action} from {current_scale} to {target_scale}"
            )
            print(f"      Reason: {trigger_reason}")

    def demonstrate_multi_environment_deployment(self):
        """Demonstrate deployment across multiple environments."""
        print("\n" + "=" * 70)
        print("🌍 Multi-Environment Deployment Pattern")
        print("=" * 70)

        environments_to_deploy = [
            DeploymentEnvironment.DEVELOPMENT,
            DeploymentEnvironment.STAGING,
            DeploymentEnvironment.PRODUCTION,
        ]

        deployment_results = {}

        for env in environments_to_deploy:
            print(f"\n🚀 Deploying to {env.value} environment...")

            # Simulate deployment
            adapter = self.deploy_to_environment(env)
            config = self.environments[env]

            # Run a quick validation test
            test_crew = self.create_production_crew("customer_service", env)

            with adapter.track_crew(f"deployment-test-{env.value}") as context:
                start_time = time.time()

                # Quick test execution
                test_crew.kickoff({"test_mode": True, "environment": env.value})

                execution_time = time.time() - start_time
                metrics = context.get_metrics()

                deployment_results[env.value] = {
                    "deployment_successful": True,
                    "test_execution_time": execution_time,
                    "test_cost": metrics["total_cost"],
                    "governance_active": metrics.get("governance_enabled", True),
                    "monitoring_level": config.monitoring_level,
                    "security_config": config.security_config,
                }

                print(f"   ✅ {env.value} deployment successful")
                print(f"      Test execution: {execution_time:.2f}s")
                print(f"      Test cost: ${metrics['total_cost']:.6f}")
                print(f"      Monitoring: {config.monitoring_level}")

        # Environment comparison
        print("\n📊 Environment Deployment Summary:")
        for env_name, results in deployment_results.items():
            print(f"   • {env_name.title()}:")
            print(
                f"     - Status: {'✅ Active' if results['deployment_successful'] else '❌ Failed'}"
            )
            print(f"     - Test time: {results['test_execution_time']:.2f}s")
            print(f"     - Monitoring: {results['monitoring_level']}")

        return deployment_results

    def demonstrate_auto_scaling(
        self, environment: DeploymentEnvironment = DeploymentEnvironment.STAGING
    ):
        """Demonstrate auto-scaling capabilities."""
        print("\n" + "=" * 70)
        print("📈 Auto-Scaling Demonstration")
        print("=" * 70)

        print(f"Environment: {environment.value}")
        config = self.environments[environment]
        print(f"Strategy: {config.scaling_strategy.value}")
        print(f"Max concurrent crews: {config.max_concurrent_crews}")

        # Run workload simulation to trigger scaling
        resource_metrics = self.simulate_production_workload(
            environment, duration_minutes=3
        )

        # Analyze scaling events
        print("\n📊 Scaling Analysis:")
        print(f"   🔄 Scaling events triggered: {len(self.scaling_events)}")

        if self.scaling_events:
            scale_ups = [e for e in self.scaling_events if e.action == "scale_up"]
            scale_downs = [e for e in self.scaling_events if e.action == "scale_down"]

            print(f"   ⬆️ Scale-up events: {len(scale_ups)}")
            print(f"   ⬇️ Scale-down events: {len(scale_downs)}")

            # Show latest scaling events
            for event in self.scaling_events[-3:]:  # Last 3 events
                print(f"\n   📅 {event.timestamp.strftime('%H:%M:%S')}")
                print(f"      Action: {event.action}")
                print(f"      Trigger: {event.trigger}")
                print(f"      Scale: {event.current_scale} → {event.target_scale}")

        # Resource utilization summary
        if resource_metrics:
            avg_cpu = sum(m.cpu_usage for m in resource_metrics) / len(resource_metrics)
            avg_response_time = sum(
                m.response_time_p95 for m in resource_metrics
            ) / len(resource_metrics)
            max_active_crews = max(m.active_crews for m in resource_metrics)

            print("\n📈 Performance Summary:")
            print(f"   🖥️ Average CPU usage: {avg_cpu:.1%}")
            print(f"   ⏱️ Average response time: {avg_response_time:.1f}s")
            print(f"   👥 Peak concurrent crews: {max_active_crews}")

    def demonstrate_fault_tolerance(self):
        """Demonstrate fault tolerance and error handling."""
        print("\n" + "=" * 70)
        print("🛡️ Fault Tolerance & Error Handling")
        print("=" * 70)

        adapter = self.deploy_to_environment(DeploymentEnvironment.PRODUCTION)

        # Simulate various failure scenarios
        fault_scenarios = [
            {
                "name": "API Rate Limit",
                "description": "Simulate API rate limiting",
                "error_type": "rate_limit",
                "recovery_strategy": "exponential_backoff",
            },
            {
                "name": "Network Timeout",
                "description": "Simulate network connectivity issues",
                "error_type": "timeout",
                "recovery_strategy": "retry_with_fallback",
            },
            {
                "name": "Budget Exceeded",
                "description": "Simulate budget limit exceeded",
                "error_type": "budget_exceeded",
                "recovery_strategy": "graceful_degradation",
            },
        ]

        recovery_results = []

        for scenario in fault_scenarios:
            print(f"\n🔬 Testing: {scenario['name']}")
            print(f"   Description: {scenario['description']}")

            # Create test crew
            test_crew = self.create_production_crew(
                "customer_service", DeploymentEnvironment.PRODUCTION
            )

            try:
                with adapter.track_crew(
                    f"fault-test-{scenario['error_type']}"
                ) as context:
                    # Simulate the fault condition
                    start_time = time.time()

                    if scenario["error_type"] == "budget_exceeded":
                        # Temporarily lower budget to trigger limit
                        original_budget = adapter.daily_budget_limit
                        adapter.daily_budget_limit = 0.001  # Very low budget

                    # Execute with fault injection
                    test_crew.kickoff(
                        {
                            "fault_injection": scenario["error_type"],
                            "recovery_strategy": scenario["recovery_strategy"],
                        }
                    )

                    execution_time = time.time() - start_time
                    metrics = context.get_metrics()

                    # Restore original settings
                    if scenario["error_type"] == "budget_exceeded":
                        adapter.daily_budget_limit = original_budget

                    recovery_results.append(
                        {
                            "scenario": scenario["name"],
                            "success": True,
                            "execution_time": execution_time,
                            "recovery_strategy": scenario["recovery_strategy"],
                            "cost": metrics["total_cost"],
                        }
                    )

                    print("   ✅ Fault tolerance successful")
                    print(f"      Recovery time: {execution_time:.2f}s")
                    print(f"      Strategy: {scenario['recovery_strategy']}")

            except Exception as e:
                recovery_results.append(
                    {
                        "scenario": scenario["name"],
                        "success": False,
                        "error": str(e),
                        "recovery_strategy": scenario["recovery_strategy"],
                    }
                )

                print(f"   ⚠️ Fault tolerance test failed: {e}")

        # Summary
        successful_recoveries = [r for r in recovery_results if r.get("success", False)]
        success_rate = (
            len(successful_recoveries) / len(recovery_results) * 100
            if recovery_results
            else 0
        )

        print("\n📊 Fault Tolerance Summary:")
        print(f"   🎯 Scenarios tested: {len(fault_scenarios)}")
        print(f"   ✅ Successful recoveries: {len(successful_recoveries)}")
        print(f"   📈 Success rate: {success_rate:.1f}%")

        return recovery_results

    def generate_production_report(
        self,
        deployment_results: dict,
        scaling_events: list[ScalingEvent],
        fault_tolerance_results: list[dict],
    ) -> dict[str, Any]:
        """Generate comprehensive production deployment report."""
        print("\n" + "=" * 70)
        print("📄 Production Deployment Report")
        print("=" * 70)

        # Calculate overall metrics
        total_environments = len(deployment_results)
        successful_deployments = len(
            [
                r
                for r in deployment_results.values()
                if r.get("deployment_successful", False)
            ]
        )

        total_scaling_events = len(scaling_events)
        successful_fault_recoveries = len(
            [r for r in fault_tolerance_results if r.get("success", False)]
        )

        report = {
            "report_id": str(uuid.uuid4()),
            "generated_at": datetime.now().isoformat(),
            "deployment_pattern": self.deployment_pattern,
            "scale_factor": self.scale_factor,
            "summary": {
                "total_environments": total_environments,
                "successful_deployments": successful_deployments,
                "deployment_success_rate": (
                    successful_deployments / total_environments * 100
                )
                if total_environments > 0
                else 0,
                "scaling_events": total_scaling_events,
                "fault_tolerance_tests": len(fault_tolerance_results),
                "fault_recovery_rate": (
                    successful_fault_recoveries / len(fault_tolerance_results) * 100
                )
                if fault_tolerance_results
                else 0,
            },
            "environments": deployment_results,
            "scaling_analysis": {
                "total_events": total_scaling_events,
                "scale_up_events": len(
                    [e for e in scaling_events if e.action == "scale_up"]
                ),
                "scale_down_events": len(
                    [e for e in scaling_events if e.action == "scale_down"]
                ),
            },
            "fault_tolerance": {
                "scenarios_tested": len(fault_tolerance_results),
                "successful_recoveries": successful_fault_recoveries,
                "recovery_strategies": list(
                    {r.get("recovery_strategy", "") for r in fault_tolerance_results}
                ),
            },
            "recommendations": self._generate_production_recommendations(
                deployment_results, scaling_events, fault_tolerance_results
            ),
        }

        print("📊 Production Report Summary:")
        print(f"   🆔 Report ID: {report['report_id'][:8]}...")
        print(
            f"   🌍 Environments: {report['summary']['successful_deployments']}/{report['summary']['total_environments']} successful"
        )
        print(f"   📈 Scaling events: {report['summary']['scaling_events']}")
        print(
            f"   🛡️ Fault tolerance: {report['summary']['fault_recovery_rate']:.1f}% success rate"
        )

        return report

    def _generate_production_recommendations(
        self,
        deployment_results: dict,
        scaling_events: list[ScalingEvent],
        fault_tolerance_results: list[dict],
    ) -> list[str]:
        """Generate production recommendations based on results."""
        recommendations = []

        # Deployment recommendations
        failed_deployments = [
            env
            for env, result in deployment_results.items()
            if not result.get("deployment_successful", True)
        ]
        if failed_deployments:
            recommendations.append(
                f"Fix deployment issues in: {', '.join(failed_deployments)}"
            )

        # Scaling recommendations
        if len(scaling_events) > 10:
            recommendations.append(
                "High scaling activity detected - consider adjusting baseline capacity"
            )
        elif len(scaling_events) == 0:
            recommendations.append("No scaling events - monitor for under-provisioning")

        # Fault tolerance recommendations
        fault_success_rate = (
            len([r for r in fault_tolerance_results if r.get("success", False)])
            / len(fault_tolerance_results)
            * 100
            if fault_tolerance_results
            else 100
        )
        if fault_success_rate < 80:
            recommendations.append("Improve fault tolerance - success rate below 80%")

        # Performance recommendations
        if any(
            result.get("test_execution_time", 0) > 60
            for result in deployment_results.values()
        ):
            recommendations.append(
                "Optimize performance - some environments showing slow response times"
            )

        return recommendations


def main():
    """Run the comprehensive production deployment patterns demonstration."""
    parser = argparse.ArgumentParser(description="Production Deployment Patterns Demo")
    parser.add_argument(
        "--pattern",
        choices=["minimal", "standard", "enterprise", "global"],
        default="standard",
        help="Deployment pattern type",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=1,
        choices=[1, 2, 3, 5],
        help="Scale factor for deployment size",
    )
    args = parser.parse_args()

    print("🏗️ Production Deployment Patterns for CrewAI + GenOps")
    print("=" * 60)
    print(f"Deployment pattern: {args.pattern}")
    print(f"Scale factor: {args.scale}")

    # Initialize deployment manager
    deployment_manager = ProductionDeploymentManager(
        deployment_pattern=args.pattern, scale_factor=args.scale
    )

    # Validate setup
    if not deployment_manager.setup_validation():
        print("\n❌ Please fix setup issues before proceeding")
        return 1

    try:
        # Demonstrate multi-environment deployment
        deployment_results = (
            deployment_manager.demonstrate_multi_environment_deployment()
        )

        # Demonstrate auto-scaling
        deployment_manager.demonstrate_auto_scaling()

        # Demonstrate fault tolerance
        fault_tolerance_results = deployment_manager.demonstrate_fault_tolerance()

        # Generate production report
        deployment_manager.generate_production_report(
            deployment_results,
            deployment_manager.scaling_events,
            fault_tolerance_results,
        )

        print("\n🎉 Production Deployment Patterns Demonstration Complete!")
        print("\n🚀 Next Steps:")
        print("   • Review production deployment recommendations")
        print("   • Implement monitoring and alerting in your production environment")
        print("   • Set up CI/CD pipelines for automated deployment")
        print("   • Configure disaster recovery and backup strategies")
        print("   • Scale to your actual production requirements")

        print("\n📋 Key Takeaways:")
        print("   • Multi-environment deployment patterns validated")
        print("   • Auto-scaling mechanisms demonstrate load adaptability")
        print("   • Fault tolerance ensures production reliability")
        print("   • GenOps provides comprehensive governance at scale")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️ Production deployment demo interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Production deployment demo failed: {e}", exc_info=True)
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
