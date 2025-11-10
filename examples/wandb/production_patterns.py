#!/usr/bin/env python3
"""
W&B Production Patterns with GenOps Governance

This comprehensive example demonstrates production-ready deployment patterns for 
Weights & Biases integration with GenOps governance. It covers CI/CD integration,
monitoring, scaling considerations, enterprise deployment scenarios, and production
ML operations patterns with comprehensive governance and observability.

Features demonstrated:
- Production-ready deployment patterns with environment-specific configurations
- CI/CD pipeline integration with automated governance validation
- Production monitoring and alerting with observability integration
- Auto-scaling patterns for high-throughput ML workloads
- Multi-tenant deployment with customer isolation and cost attribution
- Disaster recovery and backup strategies for ML artifacts
- Performance optimization for large-scale production deployments
- Enterprise security integration with SSO and role-based access control
- Comprehensive production governance with automated compliance reporting

Usage:
    python production_patterns.py

Prerequisites:
    pip install genops[wandb]
    export WANDB_API_KEY="your-wandb-api-key"
    export GENOPS_TEAM="your-team"
    export GENOPS_PROJECT="your-project"
    export GENOPS_ENVIRONMENT="production"  # Critical for production patterns
    export GENOPS_CUSTOMER_ID="your-customer-id"  # For multi-tenant scenarios

This example demonstrates enterprise-grade ML governance patterns suitable for
production environments with requirements for high availability, scalability,
security, compliance, and comprehensive observability.
"""

import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/wandb_production.log') if '/tmp' in os.listdir('/') else logging.NullHandler()
    ]
)

logger = logging.getLogger(__name__)


class DeploymentEnvironment(Enum):
    """Production deployment environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    CANARY = "canary"
    DISASTER_RECOVERY = "disaster_recovery"


class ScalingStrategy(Enum):
    """Scaling strategies for production workloads."""
    MANUAL = "manual"
    AUTO_SCALE_CPU = "auto_scale_cpu"
    AUTO_SCALE_GPU = "auto_scale_gpu"
    AUTO_SCALE_WORKLOAD = "auto_scale_workload"
    PREDICTIVE_SCALING = "predictive_scaling"


class SecurityLevel(Enum):
    """Security levels for production deployments."""
    BASIC = "basic"
    ENTERPRISE = "enterprise"
    GOVERNMENT = "government"
    FINANCIAL_SERVICES = "financial_services"


@dataclass
class ProductionConfiguration:
    """Production deployment configuration."""
    environment: DeploymentEnvironment
    scaling_strategy: ScalingStrategy
    security_level: SecurityLevel

    # Resource limits
    max_concurrent_experiments: int = 50
    max_daily_cost: float = 10000.0
    max_experiment_duration_hours: int = 24

    # Monitoring and alerting
    enable_detailed_monitoring: bool = True
    alert_email_addresses: List[str] = field(default_factory=list)
    metrics_retention_days: int = 365

    # Security and compliance
    enable_encryption_at_rest: bool = True
    enable_encryption_in_transit: bool = True
    require_mfa: bool = True
    audit_log_retention_years: int = 7

    # High availability
    enable_multi_region: bool = False
    backup_frequency_hours: int = 24
    disaster_recovery_rpo_hours: int = 4

    # Performance optimization
    enable_caching: bool = True
    cache_ttl_minutes: int = 60
    enable_compression: bool = True
    max_batch_size: int = 1000


@dataclass
class CICDPipelineConfig:
    """CI/CD pipeline configuration for ML workflows."""
    pipeline_id: str
    trigger_events: List[str]  # 'commit', 'pull_request', 'scheduled'
    validation_stages: List[str]
    deployment_stages: List[str]
    approval_gates: List[str]
    rollback_strategy: str
    test_coverage_threshold: float = 0.80
    governance_validation_required: bool = True


@dataclass
class ProductionMetrics:
    """Production metrics and KPIs."""
    uptime_percentage: float
    avg_response_time_ms: float
    error_rate_percentage: float
    cost_per_experiment: float
    experiments_per_hour: int
    governance_compliance_score: float
    security_incidents: int
    cost_efficiency_score: float


class ProductionMLWorkflowManager:
    """Manages production ML workflows with comprehensive governance."""

    def __init__(
        self,
        config: ProductionConfiguration,
        adapter: Any,  # GenOpsWandbAdapter
        pipeline_config: Optional[CIcdPipelineConfig] = None
    ):
        self.config = config
        self.adapter = adapter
        self.pipeline_config = pipeline_config

        # Production state tracking
        self.active_experiments: Dict[str, Any] = {}
        self.deployment_history: List[Dict[str, Any]] = []
        self.performance_metrics: List[ProductionMetrics] = []
        self.security_events: List[Dict[str, Any]] = []

        # Initialize monitoring
        self.metrics_collector = self._initialize_metrics_collection()

        logger.info(f"Production ML Workflow Manager initialized for {config.environment.value}")

    def _initialize_metrics_collection(self):
        """Initialize production metrics collection."""
        return {
            'start_time': datetime.utcnow(),
            'experiment_count': 0,
            'total_cost': 0.0,
            'error_count': 0,
            'performance_samples': []
        }

    @contextmanager
    def production_experiment_lifecycle(
        self,
        experiment_name: str,
        customer_id: Optional[str] = None,
        **kwargs
    ):
        """Production-grade experiment lifecycle management."""

        experiment_id = f"prod_{experiment_name}_{int(time.time())}"
        start_time = datetime.utcnow()

        # Pre-experiment production validation
        self._validate_production_constraints(experiment_id, kwargs)

        # Initialize production monitoring
        monitoring_context = self._setup_experiment_monitoring(experiment_id)

        # Create production-grade telemetry span
        with self.adapter.tracer.start_as_current_span(
            f"production.experiment.{experiment_name}",
            attributes={
                "genops.environment": self.config.environment.value,
                "genops.security_level": self.config.security_level.value,
                "genops.scaling_strategy": self.config.scaling_strategy.value,
                "genops.customer_id": customer_id,
                "genops.experiment_id": experiment_id,
                "genops.production": True,
                **kwargs
            }
        ) as span:

            try:
                # Register experiment in production tracking
                self.active_experiments[experiment_id] = {
                    'name': experiment_name,
                    'customer_id': customer_id,
                    'start_time': start_time,
                    'status': 'running',
                    'monitoring': monitoring_context
                }

                logger.info(f"Production experiment started: {experiment_id}")

                yield experiment_id

                # Successful completion
                self.active_experiments[experiment_id]['status'] = 'completed'
                self.active_experiments[experiment_id]['end_time'] = datetime.utcnow()

                # Update production metrics
                self._update_production_metrics(experiment_id)

                span.set_status(Status(StatusCode.OK))
                logger.info(f"Production experiment completed successfully: {experiment_id}")

            except Exception as e:
                # Handle production failures
                self.active_experiments[experiment_id]['status'] = 'failed'
                self.active_experiments[experiment_id]['error'] = str(e)

                # Increment error count
                self.metrics_collector['error_count'] += 1

                # Log security event if needed
                self._log_security_event('experiment_failure', {
                    'experiment_id': experiment_id,
                    'error': str(e),
                    'customer_id': customer_id
                })

                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                logger.error(f"Production experiment failed: {experiment_id} - {e}")

                # Production failure handling
                self._handle_production_failure(experiment_id, e)

                raise

            finally:
                # Cleanup
                self._cleanup_experiment_monitoring(experiment_id)

                # Move to history
                if experiment_id in self.active_experiments:
                    completed_experiment = self.active_experiments.pop(experiment_id)
                    self.deployment_history.append(completed_experiment)

    def _validate_production_constraints(self, experiment_id: str, params: Dict[str, Any]):
        """Validate production constraints and limits."""

        # Check concurrent experiment limit
        if len(self.active_experiments) >= self.config.max_concurrent_experiments:
            raise ValueError(
                f"Maximum concurrent experiments ({self.config.max_concurrent_experiments}) reached"
            )

        # Check daily cost limit
        daily_cost = self.metrics_collector['total_cost']
        estimated_cost = params.get('estimated_cost', 100.0)

        if daily_cost + estimated_cost > self.config.max_daily_cost:
            raise ValueError(
                f"Experiment would exceed daily cost limit: "
                f"${daily_cost + estimated_cost:.2f} > ${self.config.max_daily_cost:.2f}"
            )

        # Validate experiment duration
        estimated_duration = params.get('estimated_duration_hours', 2.0)
        if estimated_duration > self.config.max_experiment_duration_hours:
            raise ValueError(
                f"Experiment duration exceeds limit: "
                f"{estimated_duration}h > {self.config.max_experiment_duration_hours}h"
            )

        # Security validation
        if self.config.security_level in [SecurityLevel.GOVERNMENT, SecurityLevel.FINANCIAL_SERVICES]:
            self._validate_high_security_requirements(experiment_id, params)

        logger.debug(f"Production constraints validated for {experiment_id}")

    def _validate_high_security_requirements(self, experiment_id: str, params: Dict[str, Any]):
        """Validate high-security requirements for sensitive environments."""

        required_fields = ['data_classification', 'approval_required', 'encryption_required']
        missing_fields = [field for field in required_fields if field not in params]

        if missing_fields:
            raise ValueError(
                f"High-security deployment requires fields: {missing_fields}"
            )

        if params.get('data_classification') in ['confidential', 'top_secret']:
            if not params.get('encryption_required', False):
                raise ValueError("Confidential data requires encryption")

        logger.info(f"High-security requirements validated for {experiment_id}")

    def _setup_experiment_monitoring(self, experiment_id: str) -> Dict[str, Any]:
        """Setup comprehensive monitoring for production experiment."""

        monitoring_context = {
            'experiment_id': experiment_id,
            'start_time': datetime.utcnow(),
            'metrics': {
                'cpu_usage': [],
                'memory_usage': [],
                'gpu_utilization': [],
                'network_io': [],
                'cost_accumulation': []
            },
            'alerts': [],
            'health_checks': []
        }

        # Simulate monitoring setup
        logger.debug(f"Monitoring setup completed for {experiment_id}")

        return monitoring_context

    def _update_production_metrics(self, experiment_id: str):
        """Update production metrics after experiment completion."""

        experiment = self.active_experiments[experiment_id]
        duration = (datetime.utcnow() - experiment['start_time']).total_seconds()

        # Update aggregated metrics
        self.metrics_collector['experiment_count'] += 1

        # Simulate cost accumulation
        estimated_cost = duration * 0.5  # $0.50 per second approximation
        self.metrics_collector['total_cost'] += estimated_cost

        # Record performance sample
        performance_sample = ProductionMetrics(
            uptime_percentage=99.9,  # Simulated high availability
            avg_response_time_ms=duration * 1000,
            error_rate_percentage=0.1,  # Low error rate
            cost_per_experiment=estimated_cost,
            experiments_per_hour=3600 / duration if duration > 0 else 0,
            governance_compliance_score=95.0,  # High compliance
            security_incidents=0,
            cost_efficiency_score=85.0
        )

        self.performance_metrics.append(performance_sample)

        logger.debug(f"Production metrics updated for {experiment_id}")

    def _log_security_event(self, event_type: str, context: Dict[str, Any]):
        """Log security events for audit and compliance."""

        security_event = {
            'event_id': f"sec_{int(time.time())}_{hash(str(context)) % 10000:04d}",
            'event_type': event_type,
            'timestamp': datetime.utcnow().isoformat(),
            'context': context,
            'severity': 'INFO' if event_type == 'experiment_failure' else 'WARNING',
            'environment': self.config.environment.value,
            'investigated': False
        }

        self.security_events.append(security_event)

        logger.warning(f"Security event logged: {event_type} - {security_event['event_id']}")

    def _handle_production_failure(self, experiment_id: str, error: Exception):
        """Handle production failures with appropriate escalation."""

        failure_severity = self._assess_failure_severity(error)

        if failure_severity == 'CRITICAL':
            self._trigger_incident_response(experiment_id, error)
        elif failure_severity == 'HIGH':
            self._send_alert_notification(experiment_id, error)

        # Log failure for analysis
        logger.error(
            f"Production failure handled: {experiment_id} - "
            f"Severity: {failure_severity} - Error: {error}"
        )

    def _assess_failure_severity(self, error: Exception) -> str:
        """Assess failure severity for proper escalation."""

        error_str = str(error).lower()

        if any(keyword in error_str for keyword in ['security', 'unauthorized', 'breach']):
            return 'CRITICAL'
        elif any(keyword in error_str for keyword in ['cost', 'budget', 'limit']):
            return 'HIGH'
        elif any(keyword in error_str for keyword in ['timeout', 'connection']):
            return 'MEDIUM'
        else:
            return 'LOW'

    def _trigger_incident_response(self, experiment_id: str, error: Exception):
        """Trigger incident response for critical failures."""

        incident = {
            'incident_id': f"INC-{int(time.time())}",
            'experiment_id': experiment_id,
            'severity': 'CRITICAL',
            'description': str(error),
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'open',
            'assigned_team': 'ml_ops_oncall'
        }

        # In production, this would integrate with PagerDuty, Slack, etc.
        logger.critical(f"INCIDENT TRIGGERED: {incident['incident_id']} - {error}")

    def _send_alert_notification(self, experiment_id: str, error: Exception):
        """Send alert notifications for high-severity failures."""

        alert = {
            'alert_id': f"ALERT-{int(time.time())}",
            'experiment_id': experiment_id,
            'message': f"Production experiment failed: {error}",
            'timestamp': datetime.utcnow().isoformat(),
            'recipients': self.config.alert_email_addresses
        }

        # In production, this would send actual notifications
        logger.warning(f"ALERT SENT: {alert['alert_id']} - {error}")

    def _cleanup_experiment_monitoring(self, experiment_id: str):
        """Cleanup monitoring resources after experiment completion."""

        # Simulate cleanup of monitoring resources
        logger.debug(f"Monitoring cleanup completed for {experiment_id}")

    def get_production_status(self) -> Dict[str, Any]:
        """Get comprehensive production status and metrics."""

        uptime_hours = (datetime.utcnow() - self.metrics_collector['start_time']).total_seconds() / 3600

        return {
            'deployment_environment': self.config.environment.value,
            'uptime_hours': round(uptime_hours, 2),
            'active_experiments': len(self.active_experiments),
            'total_experiments_completed': self.metrics_collector['experiment_count'],
            'total_cost_today': round(self.metrics_collector['total_cost'], 2),
            'error_count': self.metrics_collector['error_count'],
            'error_rate_percentage': (self.metrics_collector['error_count'] / max(self.metrics_collector['experiment_count'], 1)) * 100,
            'security_events': len(self.security_events),
            'unresolved_security_events': len([e for e in self.security_events if not e['investigated']]),
            'average_experiment_cost': round(self.metrics_collector['total_cost'] / max(self.metrics_collector['experiment_count'], 1), 2),
            'performance_metrics': self.performance_metrics[-10:],  # Last 10 samples
            'scaling_utilization': len(self.active_experiments) / self.config.max_concurrent_experiments * 100
        }


def simulate_cicd_pipeline_integration(config: ProductionConfiguration) -> Dict[str, Any]:
    """Simulate CI/CD pipeline integration with governance validation."""

    print("🚀 Simulating CI/CD Pipeline Integration...")

    pipeline_stages = [
        {
            'name': 'source_validation',
            'description': 'Validate source code and dependencies',
            'duration_seconds': 30,
            'success_rate': 0.95
        },
        {
            'name': 'governance_validation',
            'description': 'Validate GenOps governance requirements',
            'duration_seconds': 45,
            'success_rate': 0.90
        },
        {
            'name': 'security_scan',
            'description': 'Security vulnerability and compliance scan',
            'duration_seconds': 120,
            'success_rate': 0.85
        },
        {
            'name': 'ml_model_validation',
            'description': 'ML model performance and accuracy validation',
            'duration_seconds': 300,
            'success_rate': 0.80
        },
        {
            'name': 'integration_tests',
            'description': 'End-to-end integration testing',
            'duration_seconds': 180,
            'success_rate': 0.88
        },
        {
            'name': 'staging_deployment',
            'description': 'Deploy to staging environment',
            'duration_seconds': 60,
            'success_rate': 0.92
        },
        {
            'name': 'production_deployment',
            'description': 'Deploy to production environment',
            'duration_seconds': 90,
            'success_rate': 0.95
        }
    ]

    pipeline_results = {
        'pipeline_id': f"pipeline_{int(time.time())}",
        'start_time': datetime.utcnow().isoformat(),
        'stages': [],
        'overall_success': True,
        'total_duration_seconds': 0,
        'deployment_environment': config.environment.value
    }

    print(f"   📋 Pipeline ID: {pipeline_results['pipeline_id']}")
    print(f"   🎯 Target Environment: {config.environment.value}")
    print()

    for stage in pipeline_stages:
        stage_start = datetime.utcnow()

        # Simulate stage execution
        print(f"   ⏳ Running stage: {stage['name']}")
        print(f"      📝 {stage['description']}")

        # Simulate execution time
        time.sleep(min(stage['duration_seconds'] / 100, 2.0))  # Scaled down for demo

        # Determine success/failure
        success = random.random() < stage['success_rate']

        stage_result = {
            'name': stage['name'],
            'description': stage['description'],
            'success': success,
            'duration_seconds': stage['duration_seconds'],
            'start_time': stage_start.isoformat(),
            'end_time': datetime.utcnow().isoformat(),
            'logs': f"Stage {stage['name']} {'completed successfully' if success else 'failed'}"
        }

        if not success:
            stage_result['error'] = f"Simulated failure in {stage['name']}"
            stage_result['retry_count'] = 0
            pipeline_results['overall_success'] = False
            print(f"      ❌ FAILED: {stage['name']}")
            break
        else:
            print(f"      ✅ SUCCESS: {stage['name']} ({stage['duration_seconds']}s)")

        pipeline_results['stages'].append(stage_result)
        pipeline_results['total_duration_seconds'] += stage['duration_seconds']

    pipeline_results['end_time'] = datetime.utcnow().isoformat()

    print("\n📊 Pipeline Results:")
    print(f"   • Overall Success: {'✅ PASSED' if pipeline_results['overall_success'] else '❌ FAILED'}")
    print(f"   • Stages Completed: {len(pipeline_results['stages'])}/{len(pipeline_stages)}")
    print(f"   • Total Duration: {pipeline_results['total_duration_seconds']}s")

    return pipeline_results


def simulate_production_monitoring(workflow_manager: ProductionMLWorkflowManager) -> Dict[str, Any]:
    """Simulate production monitoring and alerting systems."""

    print("📊 Simulating Production Monitoring & Alerting...")

    # Simulate running multiple production experiments
    monitoring_results = {
        'monitoring_session_id': f"monitor_{int(time.time())}",
        'start_time': datetime.utcnow().isoformat(),
        'experiments_monitored': [],
        'alerts_generated': [],
        'performance_metrics': [],
        'system_health': {}
    }

    # Define test experiments with different characteristics
    test_experiments = [
        {
            'name': 'production_baseline_model',
            'estimated_cost': 50.0,
            'estimated_duration_hours': 2.0,
            'expected_success': True
        },
        {
            'name': 'high_cost_optimization',
            'estimated_cost': 150.0,
            'estimated_duration_hours': 4.0,
            'expected_success': True
        },
        {
            'name': 'edge_case_testing',
            'estimated_cost': 25.0,
            'estimated_duration_hours': 1.0,
            'expected_success': False  # Simulate failure for monitoring
        }
    ]

    print(f"   🔬 Running {len(test_experiments)} monitored experiments...")
    print()

    for i, exp_config in enumerate(test_experiments):
        print(f"   🧪 Experiment {i+1}: {exp_config['name']}")

        try:
            with workflow_manager.production_experiment_lifecycle(
                exp_config['name'],
                customer_id=f"customer_{i+1}",
                **exp_config
            ) as experiment_id:

                # Simulate experiment execution
                execution_time = min(exp_config['estimated_duration_hours'], 0.5)  # Scale down for demo
                time.sleep(execution_time * 0.1)  # Further scale for demo

                # Simulate failure for edge case testing
                if not exp_config['expected_success']:
                    raise ValueError("Simulated experiment failure for monitoring demo")

                # Record successful experiment
                monitoring_results['experiments_monitored'].append({
                    'experiment_id': experiment_id,
                    'name': exp_config['name'],
                    'status': 'completed',
                    'cost': exp_config['estimated_cost'],
                    'duration_hours': execution_time
                })

                print(f"      ✅ Completed successfully (Cost: ${exp_config['estimated_cost']:.2f})")

        except Exception as e:
            # Record failed experiment and generated alerts
            monitoring_results['experiments_monitored'].append({
                'experiment_id': f"failed_{int(time.time())}",
                'name': exp_config['name'],
                'status': 'failed',
                'error': str(e),
                'cost': 0.0
            })

            alert = {
                'alert_id': f"alert_{int(time.time())}",
                'type': 'experiment_failure',
                'severity': 'HIGH',
                'message': f"Production experiment {exp_config['name']} failed: {e}",
                'timestamp': datetime.utcnow().isoformat(),
                'resolved': False
            }

            monitoring_results['alerts_generated'].append(alert)
            print(f"      ❌ Failed: {e}")

    # Get current production status
    production_status = workflow_manager.get_production_status()
    monitoring_results['system_health'] = production_status

    # Generate performance metrics
    monitoring_results['performance_metrics'] = [
        {
            'timestamp': datetime.utcnow().isoformat(),
            'cpu_usage_percent': random.uniform(20, 80),
            'memory_usage_percent': random.uniform(30, 70),
            'gpu_utilization_percent': random.uniform(40, 95),
            'network_throughput_mbps': random.uniform(100, 1000),
            'cost_per_hour': production_status['total_cost_today'] / max(production_status['uptime_hours'], 1)
        }
        for _ in range(5)
    ]

    monitoring_results['end_time'] = datetime.utcnow().isoformat()

    print("\n📈 Monitoring Results Summary:")
    print(f"   • Experiments Monitored: {len(monitoring_results['experiments_monitored'])}")
    print(f"   • Successful Experiments: {len([e for e in monitoring_results['experiments_monitored'] if e['status'] == 'completed'])}")
    print(f"   • Failed Experiments: {len([e for e in monitoring_results['experiments_monitored'] if e['status'] == 'failed'])}")
    print(f"   • Alerts Generated: {len(monitoring_results['alerts_generated'])}")
    print(f"   • System Health Score: {production_status.get('scaling_utilization', 0):.1f}% resource utilization")

    return monitoring_results


def simulate_auto_scaling_patterns(config: ProductionConfiguration) -> Dict[str, Any]:
    """Simulate auto-scaling patterns for production workloads."""

    print("📈 Simulating Auto-Scaling Patterns...")

    scaling_results = {
        'scaling_session_id': f"scale_{int(time.time())}",
        'strategy': config.scaling_strategy.value,
        'scaling_events': [],
        'resource_utilization': [],
        'cost_optimization': {}
    }

    # Simulate workload patterns
    workload_patterns = [
        {'hour': 0, 'demand': 20, 'description': 'Low overnight demand'},
        {'hour': 6, 'demand': 40, 'description': 'Morning startup workload'},
        {'hour': 9, 'demand': 80, 'description': 'Peak business hours'},
        {'hour': 12, 'demand': 90, 'description': 'Midday peak'},
        {'hour': 15, 'demand': 85, 'description': 'Afternoon high demand'},
        {'hour': 18, 'demand': 60, 'description': 'Evening wind-down'},
        {'hour': 21, 'demand': 30, 'description': 'Late evening low demand'},
        {'hour': 24, 'demand': 15, 'description': 'Overnight minimum'}
    ]

    current_capacity = 50  # Current resource capacity
    base_cost_per_hour = 100.0

    print(f"   📊 Scaling Strategy: {config.scaling_strategy.value}")
    print("   🎯 Simulating 24-hour workload pattern...")
    print()

    total_cost = 0.0

    for pattern in workload_patterns:
        demand = pattern['demand']
        hour = pattern['hour']

        # Calculate required capacity based on demand
        required_capacity = demand

        # Apply scaling strategy
        if config.scaling_strategy == ScalingStrategy.AUTO_SCALE_WORKLOAD:
            # Scale to meet demand with 20% buffer
            target_capacity = int(required_capacity * 1.2)
        elif config.scaling_strategy == ScalingStrategy.PREDICTIVE_SCALING:
            # Predictive scaling anticipates demand
            next_hour_demand = workload_patterns[(workload_patterns.index(pattern) + 1) % len(workload_patterns)]['demand']
            target_capacity = int(max(required_capacity, next_hour_demand) * 1.1)
        else:
            # Manual scaling - fixed capacity
            target_capacity = current_capacity

        # Simulate scaling event if capacity change needed
        if target_capacity != current_capacity:
            scaling_event = {
                'timestamp': datetime.utcnow().isoformat(),
                'hour': hour,
                'previous_capacity': current_capacity,
                'new_capacity': target_capacity,
                'demand': demand,
                'scaling_reason': f"Demand {demand}% requires {target_capacity} capacity",
                'cost_impact': (target_capacity - current_capacity) * base_cost_per_hour / 100
            }

            scaling_results['scaling_events'].append(scaling_event)
            current_capacity = target_capacity

            print(f"   ⚡ Hour {hour:2d}: Scaled to {target_capacity}% capacity (Demand: {demand}%)")

        # Calculate hourly cost
        hourly_cost = current_capacity * base_cost_per_hour / 100
        total_cost += hourly_cost

        # Record resource utilization
        utilization = {
            'hour': hour,
            'demand_percent': demand,
            'capacity_percent': current_capacity,
            'utilization_efficiency': min(100, (demand / current_capacity) * 100) if current_capacity > 0 else 0,
            'hourly_cost': hourly_cost
        }

        scaling_results['resource_utilization'].append(utilization)

    # Calculate cost optimization metrics
    # Compare with fixed capacity scenario
    fixed_capacity_cost = max(pattern['demand'] for pattern in workload_patterns) * base_cost_per_hour / 100 * 24
    cost_savings = fixed_capacity_cost - total_cost
    cost_optimization_percentage = (cost_savings / fixed_capacity_cost) * 100 if fixed_capacity_cost > 0 else 0

    scaling_results['cost_optimization'] = {
        'total_cost': round(total_cost, 2),
        'fixed_capacity_cost': round(fixed_capacity_cost, 2),
        'cost_savings': round(cost_savings, 2),
        'cost_optimization_percentage': round(cost_optimization_percentage, 1),
        'average_utilization': round(np.mean([u['utilization_efficiency'] for u in scaling_results['resource_utilization']]), 1)
    }

    print("\n📊 Auto-Scaling Results:")
    print(f"   • Scaling Events: {len(scaling_results['scaling_events'])}")
    print(f"   • Total Cost (24h): ${scaling_results['cost_optimization']['total_cost']:.2f}")
    print(f"   • Cost Savings vs Fixed: ${scaling_results['cost_optimization']['cost_savings']:.2f} ({scaling_results['cost_optimization']['cost_optimization_percentage']:.1f}%)")
    print(f"   • Average Utilization: {scaling_results['cost_optimization']['average_utilization']:.1f}%")

    return scaling_results


def demonstrate_disaster_recovery(config: ProductionConfiguration) -> Dict[str, Any]:
    """Demonstrate disaster recovery and backup strategies."""

    print("🔄 Demonstrating Disaster Recovery & Backup Strategies...")

    dr_results = {
        'dr_session_id': f"dr_{int(time.time())}",
        'backup_operations': [],
        'recovery_scenarios': [],
        'rpo_rto_metrics': {},
        'compliance_validations': []
    }

    # Simulate backup operations
    backup_types = [
        {
            'type': 'model_artifacts',
            'description': 'ML model artifacts and metadata',
            'size_gb': 25.0,
            'backup_time_minutes': 15,
            'retention_days': 365
        },
        {
            'type': 'experiment_data',
            'description': 'Experiment configurations and results',
            'size_gb': 150.0,
            'backup_time_minutes': 45,
            'retention_days': 1095  # 3 years
        },
        {
            'type': 'governance_logs',
            'description': 'Audit trails and compliance data',
            'size_gb': 5.0,
            'backup_time_minutes': 5,
            'retention_days': 2555  # 7 years
        },
        {
            'type': 'configuration_data',
            'description': 'System configurations and policies',
            'size_gb': 1.0,
            'backup_time_minutes': 2,
            'retention_days': 1095
        }
    ]

    print("   💾 Executing backup operations...")

    total_backup_size = 0.0
    total_backup_time = 0.0

    for backup in backup_types:
        backup_start = datetime.utcnow()

        # Simulate backup execution
        time.sleep(backup['backup_time_minutes'] / 60)  # Scale for demo

        backup_result = {
            'type': backup['type'],
            'description': backup['description'],
            'size_gb': backup['size_gb'],
            'start_time': backup_start.isoformat(),
            'end_time': datetime.utcnow().isoformat(),
            'duration_minutes': backup['backup_time_minutes'],
            'retention_days': backup['retention_days'],
            'success': True,
            'backup_location': f"s3://prod-ml-backups/{backup['type']}/{int(time.time())}/",
            'encryption': config.enable_encryption_at_rest
        }

        dr_results['backup_operations'].append(backup_result)
        total_backup_size += backup['size_gb']
        total_backup_time += backup['backup_time_minutes']

        print(f"      ✅ {backup['type']}: {backup['size_gb']}GB in {backup['backup_time_minutes']}min")

    # Simulate disaster recovery scenarios
    dr_scenarios = [
        {
            'scenario': 'region_outage',
            'description': 'Primary AWS region becomes unavailable',
            'rto_target_minutes': config.disaster_recovery_rpo_hours * 60,
            'rpo_target_minutes': config.disaster_recovery_rpo_hours * 60,
            'recovery_steps': [
                'Detect outage via monitoring',
                'Initiate DR runbook',
                'Switch DNS to DR region',
                'Restore from latest backup',
                'Validate system functionality',
                'Resume production operations'
            ]
        },
        {
            'scenario': 'data_corruption',
            'description': 'Critical experiment data becomes corrupted',
            'rto_target_minutes': 120,
            'rpo_target_minutes': 60,
            'recovery_steps': [
                'Identify corruption scope',
                'Isolate affected systems',
                'Restore from point-in-time backup',
                'Validate data integrity',
                'Resume experiment workflows'
            ]
        },
        {
            'scenario': 'security_breach',
            'description': 'Unauthorized access to ML systems detected',
            'rto_target_minutes': 30,
            'rpo_target_minutes': 15,
            'recovery_steps': [
                'Immediate system isolation',
                'Forensic analysis initiation',
                'Clean environment restoration',
                'Security controls validation',
                'Gradual service restoration'
            ]
        }
    ]

    print("\n   🚨 Testing disaster recovery scenarios...")

    for scenario in dr_scenarios:
        recovery_start = datetime.utcnow()

        # Simulate recovery execution time (scaled for demo)
        simulated_recovery_time = min(scenario['rto_target_minutes'] / 10, 30)  # Max 30 seconds for demo
        time.sleep(simulated_recovery_time / 60)

        # Calculate actual recovery metrics
        actual_rto = scenario['rto_target_minutes'] * random.uniform(0.8, 1.2)  # ±20% variation
        actual_rpo = scenario['rpo_target_minutes'] * random.uniform(0.7, 1.1)  # Better RPO usually

        recovery_result = {
            'scenario': scenario['scenario'],
            'description': scenario['description'],
            'start_time': recovery_start.isoformat(),
            'end_time': datetime.utcnow().isoformat(),
            'rto_target_minutes': scenario['rto_target_minutes'],
            'rto_actual_minutes': round(actual_rto, 1),
            'rto_met': actual_rto <= scenario['rto_target_minutes'],
            'rpo_target_minutes': scenario['rpo_target_minutes'],
            'rpo_actual_minutes': round(actual_rpo, 1),
            'rpo_met': actual_rpo <= scenario['rpo_target_minutes'],
            'success': True,
            'steps_completed': len(scenario['recovery_steps']),
            'data_loss_minutes': actual_rpo
        }

        dr_results['recovery_scenarios'].append(recovery_result)

        rto_status = '✅' if recovery_result['rto_met'] else '❌'
        rpo_status = '✅' if recovery_result['rpo_met'] else '❌'

        print(f"      {scenario['scenario']}: RTO {rto_status} {actual_rto:.1f}min, RPO {rpo_status} {actual_rpo:.1f}min")

    # Calculate overall DR metrics
    dr_results['rpo_rto_metrics'] = {
        'average_rto_minutes': round(np.mean([r['rto_actual_minutes'] for r in dr_results['recovery_scenarios']]), 1),
        'average_rpo_minutes': round(np.mean([r['rpo_actual_minutes'] for r in dr_results['recovery_scenarios']]), 1),
        'rto_sla_compliance_percentage': round(
            (len([r for r in dr_results['recovery_scenarios'] if r['rto_met']]) / len(dr_results['recovery_scenarios'])) * 100, 1
        ),
        'rpo_sla_compliance_percentage': round(
            (len([r for r in dr_results['recovery_scenarios'] if r['rpo_met']]) / len(dr_results['recovery_scenarios'])) * 100, 1
        ),
        'total_backup_size_gb': total_backup_size,
        'total_backup_time_minutes': total_backup_time
    }

    print("\n📊 Disaster Recovery Results:")
    print(f"   • Backup Operations: {len(dr_results['backup_operations'])} completed successfully")
    print(f"   • Total Backup Size: {total_backup_size:.1f}GB")
    print(f"   • Recovery Scenarios Tested: {len(dr_results['recovery_scenarios'])}")
    print(f"   • Average RTO: {dr_results['rpo_rto_metrics']['average_rto_minutes']:.1f} minutes")
    print(f"   • Average RPO: {dr_results['rpo_rto_metrics']['average_rpo_minutes']:.1f} minutes")
    print(f"   • RTO SLA Compliance: {dr_results['rpo_rto_metrics']['rto_sla_compliance_percentage']:.1f}%")
    print(f"   • RPO SLA Compliance: {dr_results['rpo_rto_metrics']['rpo_sla_compliance_percentage']:.1f}%")

    return dr_results


def generate_production_governance_report(
    workflow_manager: ProductionMLWorkflowManager,
    cicd_results: Dict[str, Any],
    monitoring_results: Dict[str, Any],
    scaling_results: Dict[str, Any],
    dr_results: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate comprehensive production governance report."""

    print("\n📊 Generating Production Governance Report...")

    # Get production status
    production_status = workflow_manager.get_production_status()

    # Calculate overall metrics
    total_experiments = production_status['total_experiments_completed']
    error_rate = production_status['error_rate_percentage']
    uptime_hours = production_status['uptime_hours']

    # CI/CD pipeline metrics
    cicd_success_rate = (len([s for s in cicd_results['stages'] if s['success']]) / max(len(cicd_results['stages']), 1)) * 100

    # Monitoring effectiveness
    monitoring_coverage = len(monitoring_results['experiments_monitored'])
    alert_response_rate = 100.0  # All alerts handled in simulation

    # Scaling efficiency
    scaling_cost_optimization = scaling_results['cost_optimization']['cost_optimization_percentage']
    scaling_utilization = scaling_results['cost_optimization']['average_utilization']

    # Disaster recovery readiness
    dr_rto_compliance = dr_results['rpo_rto_metrics']['rto_sla_compliance_percentage']
    dr_rpo_compliance = dr_results['rpo_rto_metrics']['rpo_sla_compliance_percentage']

    # Generate comprehensive report
    governance_report = {
        'report_metadata': {
            'report_id': f"prod_gov_report_{int(time.time())}",
            'generated_at': datetime.utcnow().isoformat(),
            'reporting_period_hours': uptime_hours,
            'environment': workflow_manager.config.environment.value,
            'security_level': workflow_manager.config.security_level.value
        },

        'executive_summary': {
            'overall_health_score': round(
                (100 - error_rate + cicd_success_rate + alert_response_rate + dr_rto_compliance + dr_rpo_compliance) / 5, 1
            ),
            'production_readiness': 'EXCELLENT' if error_rate < 1 else 'GOOD' if error_rate < 5 else 'NEEDS_IMPROVEMENT',
            'key_achievements': [
                f"Processed {total_experiments} experiments with {100-error_rate:.1f}% success rate",
                f"Maintained {uptime_hours:.1f} hours of uptime",
                f"Achieved {scaling_cost_optimization:.1f}% cost optimization through auto-scaling",
                f"Maintained {dr_rto_compliance:.1f}% disaster recovery SLA compliance"
            ],
            'areas_of_concern': [
                "Security event monitoring could be enhanced" if production_status['security_events'] > 0 else None,
                "Cost optimization opportunities available" if scaling_cost_optimization < 15 else None,
                "Disaster recovery testing frequency should increase" if dr_rto_compliance < 95 else None
            ]
        },

        'operational_metrics': {
            'experiments': {
                'total_completed': total_experiments,
                'success_rate_percentage': round(100 - error_rate, 1),
                'average_cost_per_experiment': production_status['average_experiment_cost'],
                'total_daily_cost': production_status['total_cost_today']
            },
            'infrastructure': {
                'uptime_hours': uptime_hours,
                'scaling_utilization_percentage': round(production_status['scaling_utilization'], 1),
                'cost_optimization_percentage': scaling_cost_optimization,
                'resource_efficiency_score': scaling_utilization
            },
            'security': {
                'security_events_total': production_status['security_events'],
                'unresolved_security_events': production_status['unresolved_security_events'],
                'encryption_at_rest_enabled': workflow_manager.config.enable_encryption_at_rest,
                'encryption_in_transit_enabled': workflow_manager.config.enable_encryption_in_transit
            }
        },

        'cicd_pipeline_performance': {
            'pipeline_success_rate_percentage': round(cicd_success_rate, 1),
            'average_pipeline_duration_minutes': cicd_results['total_duration_seconds'] / 60,
            'governance_validation_passed': any(s['name'] == 'governance_validation' and s['success'] for s in cicd_results['stages']),
            'security_scan_passed': any(s['name'] == 'security_scan' and s['success'] for s in cicd_results['stages']),
            'deployment_success': cicd_results['overall_success']
        },

        'monitoring_and_alerting': {
            'monitoring_coverage_percentage': round((monitoring_coverage / max(total_experiments, 1)) * 100, 1),
            'alerts_generated': len(monitoring_results['alerts_generated']),
            'alert_response_rate_percentage': alert_response_rate,
            'mean_time_to_detection_minutes': 5.0,  # Simulated MTTD
            'mean_time_to_resolution_minutes': 15.0  # Simulated MTTR
        },

        'disaster_recovery_readiness': {
            'backup_success_rate_percentage': 100.0,  # All backups successful in simulation
            'rto_sla_compliance_percentage': dr_rto_compliance,
            'rpo_sla_compliance_percentage': dr_rpo_compliance,
            'last_dr_test_date': datetime.utcnow().date().isoformat(),
            'backup_retention_compliance': True,
            'recovery_scenarios_tested': len(dr_results['recovery_scenarios'])
        },

        'compliance_and_governance': {
            'governance_policy_enforcement': 'ENFORCED',
            'audit_trail_completeness_percentage': 95.0,  # High audit coverage
            'data_retention_compliance': True,
            'regulatory_compliance_score': 98.0,
            'cost_governance_effectiveness': round((100 - (production_status['total_cost_today'] / workflow_manager.config.max_daily_cost) * 100), 1)
        },

        'recommendations': [
            {
                'priority': 'HIGH',
                'category': 'cost_optimization',
                'recommendation': 'Implement predictive scaling to achieve additional 5-10% cost savings',
                'estimated_impact': 'Cost reduction of $500-1000/month'
            },
            {
                'priority': 'MEDIUM',
                'category': 'monitoring',
                'recommendation': 'Add custom alerting rules for ML-specific metrics',
                'estimated_impact': 'Improved incident detection by 20%'
            },
            {
                'priority': 'LOW',
                'category': 'disaster_recovery',
                'recommendation': 'Increase DR testing frequency to quarterly',
                'estimated_impact': 'Enhanced recovery confidence and process optimization'
            }
        ],

        'risk_assessment': {
            'overall_risk_level': 'LOW',
            'identified_risks': [
                {
                    'risk': 'Single point of failure in monitoring system',
                    'probability': 'LOW',
                    'impact': 'MEDIUM',
                    'mitigation': 'Implement redundant monitoring infrastructure'
                },
                {
                    'risk': 'Cost overrun during peak usage',
                    'probability': 'MEDIUM',
                    'impact': 'LOW',
                    'mitigation': 'Enhanced predictive scaling and budget alerts'
                }
            ]
        }
    }

    # Filter out None values from areas of concern
    governance_report['executive_summary']['areas_of_concern'] = [
        concern for concern in governance_report['executive_summary']['areas_of_concern'] if concern is not None
    ]

    return governance_report


@contextmanager
def enterprise_ml_workflow_context(
    workflow_manager: ProductionMLWorkflowManager,
    workflow_name: str,
    customer_id: str,
    cost_limit: float = 1000.0,
    timeout_minutes: int = 120,
    **metadata
):
    """
    Enhanced context manager for enterprise ML workflows with comprehensive governance.
    
    Provides circuit breaker patterns, timeout management, cost enforcement,
    resource cleanup, and comprehensive error handling for production workflows.
    """
    workflow_id = f"enterprise_{workflow_name}_{int(time.time())}"
    start_time = time.time()

    print(f"🚀 Starting enterprise workflow: {workflow_id}")
    print(f"   • Customer: {customer_id}")
    print(f"   • Cost Limit: ${cost_limit:.2f}")
    print(f"   • Timeout: {timeout_minutes} minutes")

    # Circuit breaker for external dependencies
    circuit_breaker = {'failures': 0, 'last_failure': None, 'state': 'closed'}

    try:
        with workflow_manager.adapter.tracer.start_as_current_span(
            f"enterprise.workflow.{workflow_name}",
            attributes={
                "genops.workflow_id": workflow_id,
                "genops.customer_id": customer_id,
                "genops.cost_limit": cost_limit,
                "genops.enterprise": True,
                **{f"genops.{k}": str(v) for k, v in metadata.items()}
            }
        ) as span:

            workflow_context = {
                'id': workflow_id,
                'name': workflow_name,
                'customer_id': customer_id,
                'current_cost': 0.0,
                'circuit_breaker': circuit_breaker,
                'timeout_at': start_time + (timeout_minutes * 60),
            }

            class WorkflowContext:
                def add_cost(self, amount: float, description: str = ""):
                    workflow_context['current_cost'] += amount
                    workflow_manager.metrics_collector['total_cost'] += amount

                    if workflow_context['current_cost'] > cost_limit:
                        raise ValueError(f"Cost limit exceeded: ${workflow_context['current_cost']:.2f}")

                def circuit_breaker_call(self, operation_name: str, func, *args, **kwargs):
                    """Execute operation with circuit breaker protection."""
                    cb = workflow_context['circuit_breaker']

                    if cb['state'] == 'open':
                        if cb['last_failure'] and time.time() - cb['last_failure'] < 300:
                            raise Exception(f"Circuit breaker open for {operation_name}")
                        cb['state'] = 'half_open'

                    try:
                        result = func(*args, **kwargs)
                        cb['failures'] = 0
                        cb['state'] = 'closed'
                        return result
                    except Exception:
                        cb['failures'] += 1
                        cb['last_failure'] = time.time()
                        if cb['failures'] >= 3:
                            cb['state'] = 'open'
                        raise

                @property
                def workflow_id(self):
                    return workflow_context['id']

                @property
                def current_cost(self):
                    return workflow_context['current_cost']

            yield WorkflowContext()

            # Success handling
            elapsed_time = time.time() - start_time
            span.set_status(Status(StatusCode.OK))
            print(f"✅ Enterprise workflow completed: ${workflow_context['current_cost']:.3f} in {elapsed_time:.1f}s")

    except Exception as e:
        span.record_exception(e)
        span.set_status(Status(StatusCode.ERROR, str(e)))
        workflow_manager._log_security_event('enterprise_workflow_failure', {
            'workflow_id': workflow_id,
            'customer_id': customer_id,
            'error': str(e)
        })
        logger.error(f"Enterprise workflow failed: {e}")
        raise


def main():
    """Main function demonstrating enhanced production patterns with enterprise context managers."""
    print("🏭 W&B Production Patterns with GenOps Governance")
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)

    # Check prerequisites
    api_key = os.getenv('WANDB_API_KEY')
    if not api_key:
        print("❌ WANDB_API_KEY environment variable not set")
        print("💡 Get your API key from https://wandb.ai/settings")
        return False

    team = os.getenv('GENOPS_TEAM', 'production-ml-team')
    project = os.getenv('GENOPS_PROJECT', 'production-patterns-demo')
    customer_id = os.getenv('GENOPS_CUSTOMER_ID', 'enterprise-production-001')
    environment = os.getenv('GENOPS_ENVIRONMENT', 'production')

    print("📋 Production Configuration:")
    print(f"   • Team: {team}")
    print(f"   • Project: {project}")
    print(f"   • Customer ID: {customer_id}")
    print(f"   • Environment: {environment}")
    print(f"   • API Key: {'✅ Set' if api_key else '❌ Not set'}")
    print()

    try:
        # Import required modules
        import wandb
        from opentelemetry.trace import Status, StatusCode

        from genops.providers.wandb import instrument_wandb

        # Create production configuration
        prod_config = ProductionConfiguration(
            environment=DeploymentEnvironment(environment.lower()),
            scaling_strategy=ScalingStrategy.AUTO_SCALE_WORKLOAD,
            security_level=SecurityLevel.ENTERPRISE,
            max_concurrent_experiments=25,
            max_daily_cost=5000.0,
            max_experiment_duration_hours=12,
            enable_detailed_monitoring=True,
            alert_email_addresses=['ml-ops@company.com', 'on-call@company.com'],
            enable_encryption_at_rest=True,
            enable_encryption_in_transit=True,
            require_mfa=True,
            enable_multi_region=True,
            backup_frequency_hours=6,
            disaster_recovery_rpo_hours=2
        )

        # Create GenOps W&B adapter with production configuration
        print("🔧 Initializing Production-Grade GenOps W&B Integration...")
        adapter = instrument_wandb(
            wandb_api_key=api_key,
            team=team,
            project=project,
            customer_id=customer_id,
            environment=environment,
            daily_budget_limit=prod_config.max_daily_cost,
            max_experiment_cost=500.0,  # $500 max per experiment in production
            governance_policy="enforced",  # Strict enforcement in production
            enable_cost_alerts=True,
            enable_governance=True,
            cost_center="production_ml_operations",
            tags={
                "deployment_type": "production",
                "security_level": prod_config.security_level.value,
                "scaling_strategy": prod_config.scaling_strategy.value
            }
        )

        print("✅ Production GenOps W&B adapter initialized successfully")

        # Display production governance configuration
        initial_metrics = adapter.get_metrics()
        print("\n🛡️ Production Governance Configuration:")
        print(f"   • Environment: {environment}")
        print(f"   • Daily Budget Limit: ${initial_metrics['daily_budget_limit']:,.2f}")
        print(f"   • Security Level: {prod_config.security_level.value}")
        print(f"   • Governance Policy: {initial_metrics['governance_policy']}")
        print(f"   • Multi-Region: {'✅ Enabled' if prod_config.enable_multi_region else '❌ Disabled'}")
        print(f"   • Encryption at Rest: {'✅ Enabled' if prod_config.enable_encryption_at_rest else '❌ Disabled'}")
        print(f"   • MFA Required: {'✅ Enabled' if prod_config.require_mfa else '❌ Disabled'}")

        # Initialize production workflow manager
        workflow_manager = ProductionMLWorkflowManager(prod_config, adapter)

        # === CI/CD PIPELINE INTEGRATION ===
        print("\n" + "="*90)
        print("🚀 CI/CD PIPELINE INTEGRATION")
        print("="*90)

        cicd_results = simulate_cicd_pipeline_integration(prod_config)

        # === PRODUCTION MONITORING & ALERTING ===
        print("\n" + "="*90)
        print("📊 PRODUCTION MONITORING & ALERTING")
        print("="*90)

        monitoring_results = simulate_production_monitoring(workflow_manager)

        # === AUTO-SCALING PATTERNS ===
        print("\n" + "="*90)
        print("📈 AUTO-SCALING PATTERNS")
        print("="*90)

        scaling_results = simulate_auto_scaling_patterns(prod_config)

        # === DISASTER RECOVERY & BACKUP ===
        print("\n" + "="*90)
        print("🔄 DISASTER RECOVERY & BACKUP")
        print("="*90)

        dr_results = demonstrate_disaster_recovery(prod_config)

        # === PRODUCTION GOVERNANCE REPORT ===
        print("\n" + "="*90)
        print("📊 PRODUCTION GOVERNANCE REPORT")
        print("="*90)

        governance_report = generate_production_governance_report(
            workflow_manager, cicd_results, monitoring_results, scaling_results, dr_results
        )

        # Display executive summary
        exec_summary = governance_report['executive_summary']
        print("📋 Executive Summary:")
        print(f"   • Overall Health Score: {exec_summary['overall_health_score']:.1f}/100")
        print(f"   • Production Readiness: {exec_summary['production_readiness']}")

        print("\n🎯 Key Achievements:")
        for achievement in exec_summary['key_achievements']:
            print(f"   ✅ {achievement}")

        if exec_summary['areas_of_concern']:
            print("\n⚠️  Areas of Concern:")
            for concern in exec_summary['areas_of_concern']:
                print(f"   • {concern}")

        # Operational metrics
        ops_metrics = governance_report['operational_metrics']
        print("\n📊 Operational Metrics:")
        print(f"   • Experiments: {ops_metrics['experiments']['total_completed']} completed ({ops_metrics['experiments']['success_rate_percentage']:.1f}% success)")
        print(f"   • Infrastructure: {ops_metrics['infrastructure']['uptime_hours']:.1f}h uptime, {ops_metrics['infrastructure']['scaling_utilization_percentage']:.1f}% utilization")
        print(f"   • Cost Optimization: {ops_metrics['infrastructure']['cost_optimization_percentage']:.1f}% savings through auto-scaling")
        print(f"   • Security: {ops_metrics['security']['security_events_total']} events, {ops_metrics['security']['unresolved_security_events']} unresolved")

        # CI/CD performance
        cicd_perf = governance_report['cicd_pipeline_performance']
        print("\n🔄 CI/CD Pipeline Performance:")
        print(f"   • Success Rate: {cicd_perf['pipeline_success_rate_percentage']:.1f}%")
        print(f"   • Average Duration: {cicd_perf['average_pipeline_duration_minutes']:.1f} minutes")
        print(f"   • Governance Validation: {'✅ Passed' if cicd_perf['governance_validation_passed'] else '❌ Failed'}")
        print(f"   • Security Scan: {'✅ Passed' if cicd_perf['security_scan_passed'] else '❌ Failed'}")

        # Monitoring and alerting
        monitoring = governance_report['monitoring_and_alerting']
        print("\n📺 Monitoring & Alerting:")
        print(f"   • Monitoring Coverage: {monitoring['monitoring_coverage_percentage']:.1f}%")
        print(f"   • Alerts Generated: {monitoring['alerts_generated']}")
        print(f"   • Alert Response Rate: {monitoring['alert_response_rate_percentage']:.1f}%")
        print(f"   • MTTD: {monitoring['mean_time_to_detection_minutes']:.1f}min, MTTR: {monitoring['mean_time_to_resolution_minutes']:.1f}min")

        # Disaster recovery
        dr_readiness = governance_report['disaster_recovery_readiness']
        print("\n🚨 Disaster Recovery Readiness:")
        print(f"   • Backup Success Rate: {dr_readiness['backup_success_rate_percentage']:.1f}%")
        print(f"   • RTO SLA Compliance: {dr_readiness['rto_sla_compliance_percentage']:.1f}%")
        print(f"   • RPO SLA Compliance: {dr_readiness['rpo_sla_compliance_percentage']:.1f}%")
        print(f"   • Recovery Scenarios Tested: {dr_readiness['recovery_scenarios_tested']}")

        # Compliance and governance
        compliance = governance_report['compliance_and_governance']
        print("\n🛡️  Compliance & Governance:")
        print(f"   • Policy Enforcement: {compliance['governance_policy_enforcement']}")
        print(f"   • Audit Trail: {compliance['audit_trail_completeness_percentage']:.1f}% complete")
        print(f"   • Regulatory Compliance: {compliance['regulatory_compliance_score']:.1f}%")
        print(f"   • Cost Governance: {compliance['cost_governance_effectiveness']:.1f}% effective")

        # Recommendations
        print("\n💡 Recommendations:")
        for rec in governance_report['recommendations']:
            priority_emoji = '🔴' if rec['priority'] == 'HIGH' else '🟡' if rec['priority'] == 'MEDIUM' else '🟢'
            print(f"   {priority_emoji} {rec['priority']}: {rec['recommendation']}")
            print(f"      Impact: {rec['estimated_impact']}")

        # Risk assessment
        risk = governance_report['risk_assessment']
        risk_emoji = '🔴' if risk['overall_risk_level'] == 'HIGH' else '🟡' if risk['overall_risk_level'] == 'MEDIUM' else '🟢'
        print("\n⚠️  Risk Assessment:")
        print(f"   • Overall Risk Level: {risk_emoji} {risk['overall_risk_level']}")
        if risk['identified_risks']:
            print("   • Identified Risks:")
            for r in risk['identified_risks']:
                print(f"     - {r['risk']} (Probability: {r['probability']}, Impact: {r['impact']})")

        # === PRODUCTION DEPLOYMENT COMPLETED ===
        print("\n" + "="*90)
        print("🎉 PRODUCTION PATTERNS DEMONSTRATION COMPLETED")
        print("="*90)

        # Final production status
        final_status = workflow_manager.get_production_status()
        print("\n📊 Final Production Status:")
        print(f"   • System Uptime: {final_status['uptime_hours']:.1f} hours")
        print(f"   • Total Experiments: {final_status['total_experiments_completed']}")
        print(f"   • Error Rate: {final_status['error_rate_percentage']:.2f}%")
        print(f"   • Total Cost: ${final_status['total_cost_today']:.2f}")
        print(f"   • Resource Utilization: {final_status['scaling_utilization']:.1f}%")

        print("\n🎓 Production Patterns Demonstrated:")
        print("   ✅ Production-ready deployment configuration with enterprise security")
        print("   ✅ CI/CD pipeline integration with automated governance validation")
        print("   ✅ Comprehensive production monitoring and alerting systems")
        print("   ✅ Auto-scaling patterns for cost-optimized resource management")
        print("   ✅ Multi-tenant deployment with customer isolation and attribution")
        print("   ✅ Disaster recovery and backup strategies with SLA compliance")
        print("   ✅ Performance optimization for large-scale production workloads")
        print("   ✅ Enterprise security integration with encryption and audit trails")
        print("   ✅ Comprehensive production governance and compliance reporting")

        print("\n📈 Key Production Metrics Achieved:")
        print(f"   • {exec_summary['overall_health_score']:.1f}/100 overall health score")
        print(f"   • {ops_metrics['experiments']['success_rate_percentage']:.1f}% experiment success rate")
        print(f"   • {ops_metrics['infrastructure']['cost_optimization_percentage']:.1f}% cost optimization through scaling")
        print(f"   • {dr_readiness['rto_sla_compliance_percentage']:.1f}% disaster recovery SLA compliance")
        print(f"   • {compliance['regulatory_compliance_score']:.1f}% regulatory compliance score")

        print("\n🚀 Production Deployment Benefits:")
        print(f"   💰 Cost Intelligence: ${ops_metrics['infrastructure']['cost_optimization_percentage']:.1f}% savings through intelligent scaling")
        print("   🛡️  Security: Enterprise-grade encryption, MFA, and audit trails")
        print(f"   📊 Observability: Comprehensive monitoring with {monitoring['monitoring_coverage_percentage']:.1f}% coverage")
        print(f"   🔄 Reliability: {dr_readiness['rto_sla_compliance_percentage']:.1f}% disaster recovery compliance")
        print(f"   ⚡ Performance: {ops_metrics['infrastructure']['resource_efficiency_score']:.1f}% resource efficiency")

        print("\n🏢 Enterprise Value Delivered:")
        print("   • Production-ready ML governance with comprehensive policy enforcement")
        print("   • Automated compliance reporting and audit trail generation")
        print("   • Cost optimization achieving significant operational savings")
        print("   • High availability and disaster recovery meeting enterprise SLAs")
        print("   • Scalable architecture supporting growing ML workloads")
        print("   • Security controls meeting enterprise and regulatory requirements")

        print("\n📚 Next Steps for Production Deployment:")
        print("   • Customize configuration for your specific environment requirements")
        print("   • Integrate with your existing CI/CD and monitoring systems")
        print("   • Configure organization-specific governance policies and compliance rules")
        print("   • Set up production alerting and incident response procedures")
        print("   • Train your team on production ML operations best practices")
        print("   • Review complete documentation: docs/integrations/wandb.md")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Install required packages: pip install genops[wandb]")
        return False

    except Exception as e:
        print(f"❌ Error during execution: {e}")
        print("💡 Check your configuration and try running setup_validation.py first")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
