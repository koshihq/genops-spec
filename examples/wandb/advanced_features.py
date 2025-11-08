#!/usr/bin/env python3
"""
W&B Advanced Features with GenOps Governance

This comprehensive example demonstrates advanced Weights & Biases integration patterns
enhanced with GenOps governance for complex ML workflows. It covers multi-run campaigns,
distributed training scenarios, advanced artifact management, comprehensive governance
features, and enterprise-grade ML operations patterns.

Features demonstrated:
- Multi-run campaign management with unified governance tracking
- Distributed training simulation with cost attribution across nodes
- Advanced artifact versioning and governance metadata management  
- Comprehensive policy enforcement and compliance monitoring
- Cross-team collaboration with fine-grained access controls
- Advanced cost intelligence with multi-dimensional tracking
- Integration with external ML pipeline orchestration
- Enterprise governance reporting and audit trail generation

Usage:
    python advanced_features.py

Prerequisites:
    pip install genops[wandb]
    export WANDB_API_KEY="your-wandb-api-key"
    export GENOPS_TEAM="your-team"
    export GENOPS_PROJECT="your-project"
    export GENOPS_CUSTOMER_ID="your-customer-id"  # Optional for multi-tenant scenarios

This example demonstrates advanced ML governance patterns suitable for production
environments with complex requirements for cost control, compliance, and collaboration.
"""

import os
import time
import json
import random
import hashlib
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, NamedTuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed


class DistributedTrainingMode(Enum):
    """Distributed training strategies."""
    SINGLE_NODE = "single_node"
    DATA_PARALLEL = "data_parallel" 
    MODEL_PARALLEL = "model_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    HYBRID_PARALLEL = "hybrid_parallel"


class PolicyViolationSeverity(Enum):
    """Severity levels for policy violations."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ArtifactGovernanceLevel(Enum):
    """Governance levels for artifact management."""
    BASIC = "basic"              # Basic metadata only
    STANDARD = "standard"        # Standard governance + lineage
    ENTERPRISE = "enterprise"    # Full governance + compliance
    REGULATORY = "regulatory"    # Regulatory compliance + audit trail


@dataclass
class DistributedNode:
    """Configuration for distributed training node."""
    node_id: str
    node_type: str  # 'master', 'worker', 'parameter_server'
    instance_type: str  # 'p3.2xlarge', 'p4d.24xlarge', etc.
    gpu_count: int
    cpu_count: int
    memory_gb: float
    cost_per_hour: float
    region: str = "us-east-1"


@dataclass
class PolicyViolation:
    """Represents a policy violation with full context."""
    violation_id: str
    severity: PolicyViolationSeverity
    policy_name: str
    description: str
    detected_at: datetime
    context: Dict[str, Any]
    remediation_suggested: Optional[str] = None
    auto_remediated: bool = False
    acknowledged_by: Optional[str] = None


@dataclass
class CampaignGovernanceConfig:
    """Governance configuration for multi-run campaigns."""
    campaign_id: str
    max_total_cost: float
    max_concurrent_runs: int
    cost_alert_thresholds: List[float]
    required_approvals: List[str]  # Team roles required for approval
    compliance_requirements: List[str]
    data_retention_policy: str
    access_control_policy: Dict[str, List[str]]


@dataclass
class ArtifactLineage:
    """Tracks artifact lineage and dependencies."""
    artifact_id: str
    parent_artifacts: List[str]
    derived_artifacts: List[str]
    creation_context: Dict[str, Any]
    processing_pipeline: List[str]
    data_sources: List[str]
    validation_results: Dict[str, Any]
    governance_approvals: List[str]


@dataclass 
class AdvancedMLCampaign:
    """Represents a complex ML campaign with governance."""
    campaign_id: str
    campaign_name: str
    team: str
    project: str
    customer_id: Optional[str]
    start_time: datetime
    governance_config: CampaignGovernanceConfig
    
    # Runtime state
    total_cost: float = 0.0
    active_runs: Dict[str, Any] = field(default_factory=dict)
    completed_runs: List[Dict[str, Any]] = field(default_factory=list)
    policy_violations: List[PolicyViolation] = field(default_factory=list)
    artifacts_created: List[str] = field(default_factory=list)
    compliance_checkpoints: List[Dict[str, Any]] = field(default_factory=list)


class AdvancedMLWorkflowSimulator:
    """Simulates advanced ML workflows with realistic complexity."""
    
    @staticmethod
    def simulate_distributed_training(
        config: Dict[str, Any],
        nodes: List[DistributedNode],
        epochs: int = 10
    ) -> Dict[str, Any]:
        """Simulate distributed training across multiple nodes."""
        
        print(f"🖥️  Simulating distributed training across {len(nodes)} nodes...")
        
        # Calculate total resources
        total_gpus = sum(node.gpu_count for node in nodes)
        total_cost_per_hour = sum(node.cost_per_hour for node in nodes)
        
        # Simulate training efficiency based on parallelism strategy
        training_mode = DistributedTrainingMode(config.get('distributed_mode', 'data_parallel'))
        
        efficiency_factors = {
            DistributedTrainingMode.SINGLE_NODE: 1.0,
            DistributedTrainingMode.DATA_PARALLEL: 0.85 * min(len(nodes), 8),  # Diminishing returns
            DistributedTrainingMode.MODEL_PARALLEL: 0.75 * np.sqrt(len(nodes)),
            DistributedTrainingMode.PIPELINE_PARALLEL: 0.90 * len(nodes) * 0.8,
            DistributedTrainingMode.HYBRID_PARALLEL: 0.80 * len(nodes) * 0.9
        }
        
        speedup_factor = efficiency_factors[training_mode]
        
        # Simulate training progression
        training_metrics = []
        node_costs = {node.node_id: 0.0 for node in nodes}
        
        base_epoch_time = 2.0  # Base time per epoch in minutes
        actual_epoch_time = base_epoch_time / speedup_factor
        
        for epoch in range(epochs):
            # Simulate convergence
            progress = (epoch + 1) / epochs
            base_accuracy = 0.70 + 0.20 * (1 - np.exp(-3 * progress))
            
            # Add distributed training artifacts (communication overhead, etc.)
            communication_noise = random.uniform(-0.01, 0.01) * len(nodes)
            accuracy = base_accuracy + communication_noise
            
            # Calculate loss
            loss = max(0.01, 2.0 * (1 - accuracy) + random.uniform(-0.05, 0.05))
            
            # Calculate per-node costs for this epoch
            epoch_duration_hours = actual_epoch_time / 60
            for node in nodes:
                epoch_cost = node.cost_per_hour * epoch_duration_hours
                # Add variability based on node utilization
                utilization = random.uniform(0.85, 1.0)
                node_costs[node.node_id] += epoch_cost * utilization
            
            # Simulate distributed metrics
            metrics = {
                'epoch': epoch,
                'accuracy': min(0.99, max(0.1, accuracy)),
                'loss': loss,
                'epoch_time_minutes': actual_epoch_time,
                'total_gpus_used': total_gpus,
                'communication_overhead': len(nodes) * 0.02,
                'resource_efficiency': speedup_factor / len(nodes),
                'cost_per_epoch': sum(node.cost_per_hour * epoch_duration_hours for node in nodes)
            }
            
            # Add per-node metrics
            for node in nodes:
                metrics[f'node_{node.node_id}_utilization'] = random.uniform(0.85, 1.0)
                metrics[f'node_{node.node_id}_temperature'] = random.uniform(65, 85)
            
            training_metrics.append(metrics)
            
            print(f"      📊 Epoch {epoch+1:2d}: accuracy={accuracy:.3f}, time={actual_epoch_time:.1f}min, cost=${sum(node.cost_per_hour * epoch_duration_hours for node in nodes):.2f}")
            
            time.sleep(0.1)
        
        total_training_cost = sum(node_costs.values())
        
        return {
            'final_accuracy': training_metrics[-1]['accuracy'],
            'final_loss': training_metrics[-1]['loss'],
            'total_training_time_hours': (actual_epoch_time * epochs) / 60,
            'total_cost': total_training_cost,
            'cost_by_node': node_costs,
            'training_mode': training_mode.value,
            'resource_efficiency': speedup_factor,
            'metrics_history': training_metrics,
            'distributed_summary': {
                'total_gpus': total_gpus,
                'nodes_used': len(nodes),
                'average_utilization': np.mean([m['resource_efficiency'] for m in training_metrics]),
                'communication_overhead': len(nodes) * 0.02
            }
        }
    
    @staticmethod
    def create_governed_artifact(
        name: str,
        artifact_type: str,
        governance_level: ArtifactGovernanceLevel,
        lineage: ArtifactLineage,
        compliance_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create an artifact with comprehensive governance metadata."""
        
        artifact_data = {
            'name': name,
            'type': artifact_type,
            'governance_level': governance_level.value,
            'created_at': datetime.utcnow().isoformat(),
            'lineage': asdict(lineage),
            'compliance': compliance_metadata or {},
            'governance_metadata': {
                'data_classification': 'internal',
                'retention_period_days': 365,
                'encryption_required': governance_level in [ArtifactGovernanceLevel.ENTERPRISE, ArtifactGovernanceLevel.REGULATORY],
                'audit_trail_required': governance_level == ArtifactGovernanceLevel.REGULATORY,
                'approval_required': governance_level in [ArtifactGovernanceLevel.ENTERPRISE, ArtifactGovernanceLevel.REGULATORY]
            }
        }
        
        # Add regulatory-specific metadata
        if governance_level == ArtifactGovernanceLevel.REGULATORY:
            artifact_data['regulatory'] = {
                'gdpr_compliant': True,
                'data_residency': 'EU',
                'privacy_impact_assessment': 'completed',
                'retention_justification': 'machine_learning_model_training'
            }
        
        return artifact_data


def run_multi_run_campaign(adapter, campaign_config: CampaignGovernanceConfig) -> AdvancedMLCampaign:
    """Execute a complex multi-run ML campaign with governance."""
    
    print(f"🚀 Starting Multi-Run Campaign: {campaign_config.campaign_id}")
    print(f"   • Max Total Cost: ${campaign_config.max_total_cost:.2f}")
    print(f"   • Max Concurrent Runs: {campaign_config.max_concurrent_runs}")
    print(f"   • Cost Alert Thresholds: {campaign_config.cost_alert_thresholds}")
    print()
    
    # Initialize campaign
    campaign = AdvancedMLCampaign(
        campaign_id=campaign_config.campaign_id,
        campaign_name=f"Advanced ML Campaign - {campaign_config.campaign_id}",
        team=adapter.team,
        project=adapter.project,
        customer_id=adapter.customer_id,
        start_time=datetime.utcnow(),
        governance_config=campaign_config
    )
    
    # Define experiment configurations for the campaign
    experiment_configs = [
        # Experiment 1: Single-node baseline
        {
            'name': 'baseline_single_node',
            'model_type': 'resnet50',
            'distributed_mode': 'single_node',
            'batch_size': 64,
            'learning_rate': 0.001,
            'epochs': 8,
            'priority': 'high'
        },
        
        # Experiment 2: Data parallel training
        {
            'name': 'data_parallel_optimization',
            'model_type': 'resnet50',
            'distributed_mode': 'data_parallel',
            'batch_size': 32,
            'learning_rate': 0.002,
            'epochs': 10,
            'priority': 'high'
        },
        
        # Experiment 3: Model parallel for large model
        {
            'name': 'large_model_parallel',
            'model_type': 'transformer_large',
            'distributed_mode': 'model_parallel',
            'batch_size': 16,
            'learning_rate': 0.0005,
            'epochs': 6,
            'priority': 'medium'
        },
        
        # Experiment 4: Hybrid parallel approach
        {
            'name': 'hybrid_parallel_advanced',
            'model_type': 'transformer_xlarge',
            'distributed_mode': 'hybrid_parallel',
            'batch_size': 8,
            'learning_rate': 0.0001,
            'epochs': 12,
            'priority': 'low'
        }
    ]
    
    # Execute experiments with governance oversight
    import wandb
    
    for exp_config in experiment_configs:
        
        # Check campaign budget before starting experiment
        if campaign.total_cost > campaign_config.max_total_cost * 0.9:
            print(f"   ⚠️  Campaign approaching budget limit, skipping remaining experiments")
            break
        
        # Check concurrent runs limit
        if len(campaign.active_runs) >= campaign_config.max_concurrent_runs:
            print(f"   ⏸️  Maximum concurrent runs reached, waiting...")
            time.sleep(1)  # In real scenario, would wait for runs to complete
        
        print(f"\n🧪 Starting Experiment: {exp_config['name']}")
        
        # Create distributed training setup
        if exp_config['distributed_mode'] == 'single_node':
            nodes = [DistributedNode(
                node_id='master',
                node_type='master',
                instance_type='p3.2xlarge',
                gpu_count=1,
                cpu_count=8,
                memory_gb=61,
                cost_per_hour=3.06
            )]
        elif exp_config['distributed_mode'] == 'data_parallel':
            nodes = [
                DistributedNode(f'worker_{i}', 'worker', 'p3.2xlarge', 1, 8, 61, 3.06)
                for i in range(4)
            ]
        else:  # Model/hybrid parallel
            nodes = [
                DistributedNode('master', 'master', 'p3.8xlarge', 4, 32, 244, 12.24),
                DistributedNode('worker_1', 'worker', 'p3.8xlarge', 4, 32, 244, 12.24)
            ]
        
        # Track experiment with governance
        with adapter.track_experiment_lifecycle(
            exp_config['name'],
            experiment_type='distributed_training',
            max_cost=campaign_config.max_total_cost * 0.3  # 30% of campaign budget per experiment
        ) as experiment_context:
            
            # Initialize W&B run
            run = wandb.init(
                project=f"genops-advanced-campaign-{campaign.campaign_id}",
                name=exp_config['name'],
                config=exp_config,
                tags=['advanced', 'multi-run', 'governance', exp_config['distributed_mode']],
                reinit=True
            )
            
            campaign.active_runs[run.id] = {
                'run_id': run.id,
                'name': exp_config['name'],
                'config': exp_config,
                'start_time': datetime.utcnow(),
                'nodes': nodes
            }
            
            try:
                # Run distributed training simulation
                training_results = AdvancedMLWorkflowSimulator.simulate_distributed_training(
                    exp_config, nodes, exp_config['epochs']
                )
                
                # Log comprehensive metrics to W&B
                for epoch_metrics in training_results['metrics_history']:
                    wandb.log(epoch_metrics)
                
                # Log distributed training summary
                wandb.log({
                    'final_accuracy': training_results['final_accuracy'],
                    'final_loss': training_results['final_loss'],
                    'total_training_time_hours': training_results['total_training_time_hours'],
                    'total_cost': training_results['total_cost'],
                    'resource_efficiency': training_results['resource_efficiency'],
                    'distributed_nodes': len(nodes),
                    'distributed_gpus': training_results['distributed_summary']['total_gpus']
                })
                
                # Create governed model artifact
                lineage = ArtifactLineage(
                    artifact_id=f"model_{exp_config['name']}_{run.id}",
                    parent_artifacts=[],
                    derived_artifacts=[],
                    creation_context={'experiment': exp_config['name'], 'campaign': campaign.campaign_id},
                    processing_pipeline=['data_loading', 'distributed_training', 'model_validation'],
                    data_sources=['imagenet_subset', 'custom_dataset'],
                    validation_results={'accuracy': training_results['final_accuracy']},
                    governance_approvals=['ml_engineer', 'data_scientist']
                )
                
                artifact_data = AdvancedMLWorkflowSimulator.create_governed_artifact(
                    f"model_{exp_config['name']}",
                    'model',
                    ArtifactGovernanceLevel.ENTERPRISE,
                    lineage,
                    {
                        'model_performance': training_results['final_accuracy'],
                        'training_cost': training_results['total_cost'],
                        'governance_approved': True
                    }
                )
                
                # Create W&B artifact
                model_artifact = wandb.Artifact(
                    artifact_data['name'],
                    type=artifact_data['type'],
                    metadata=artifact_data
                )
                
                # Add model files (simulated)
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(training_results, f, indent=2)
                    model_artifact.add_file(f.name, name='training_results.json')
                
                # Log governed artifact
                adapter.log_governed_artifact(
                    model_artifact,
                    cost_estimate=training_results['total_cost'] * 0.02,  # 2% storage cost
                    governance_metadata=artifact_data['governance_metadata']
                )
                
                campaign.artifacts_created.append(artifact_data['artifact_id'])
                
                # Update campaign cost
                experiment_context.estimated_cost = training_results['total_cost']
                campaign.total_cost += training_results['total_cost']
                
                # Check cost alert thresholds
                for threshold in campaign_config.cost_alert_thresholds:
                    if campaign.total_cost >= threshold and threshold not in [alert.get('threshold') for alert in campaign.compliance_checkpoints]:
                        print(f"   🚨 Campaign cost alert: ${campaign.total_cost:.2f} >= ${threshold:.2f}")
                        campaign.compliance_checkpoints.append({
                            'type': 'cost_threshold_reached',
                            'threshold': threshold,
                            'current_cost': campaign.total_cost,
                            'timestamp': datetime.utcnow().isoformat()
                        })
                
                # Move to completed runs
                completed_run = campaign.active_runs.pop(run.id)
                completed_run['end_time'] = datetime.utcnow()
                completed_run['results'] = training_results
                campaign.completed_runs.append(completed_run)
                
                print(f"   ✅ Completed: {exp_config['name']}")
                print(f"      • Final Accuracy: {training_results['final_accuracy']:.3f}")
                print(f"      • Training Cost: ${training_results['total_cost']:.2f}")
                print(f"      • Resource Efficiency: {training_results['resource_efficiency']:.2f}")
                print(f"      • Campaign Total Cost: ${campaign.total_cost:.2f}")
                
            except Exception as e:
                print(f"   ❌ Experiment failed: {e}")
                
                # Log policy violation
                violation = PolicyViolation(
                    violation_id=f"exp_failure_{run.id}",
                    severity=PolicyViolationSeverity.ERROR,
                    policy_name="experiment_execution_policy",
                    description=f"Experiment {exp_config['name']} failed: {str(e)}",
                    detected_at=datetime.utcnow(),
                    context={'experiment': exp_config, 'error': str(e)}
                )
                campaign.policy_violations.append(violation)
                
            finally:
                run.finish()
    
    # Generate campaign compliance report
    campaign_duration = datetime.utcnow() - campaign.start_time
    
    final_compliance_report = {
        'campaign_id': campaign.campaign_id,
        'duration_hours': campaign_duration.total_seconds() / 3600,
        'total_cost': campaign.total_cost,
        'budget_utilization': (campaign.total_cost / campaign_config.max_total_cost) * 100,
        'experiments_completed': len(campaign.completed_runs),
        'experiments_failed': len([v for v in campaign.policy_violations if v.severity == PolicyViolationSeverity.ERROR]),
        'artifacts_created': len(campaign.artifacts_created),
        'policy_violations': len(campaign.policy_violations),
        'compliance_checkpoints': len(campaign.compliance_checkpoints),
        'governance_score': max(0, 100 - len(campaign.policy_violations) * 10)
    }
    
    campaign.compliance_checkpoints.append({
        'type': 'final_compliance_report',
        'report': final_compliance_report,
        'timestamp': datetime.utcnow().isoformat()
    })
    
    return campaign


def demonstrate_advanced_governance_features(adapter) -> Dict[str, Any]:
    """Demonstrate advanced governance features like policy enforcement."""
    
    print(f"🛡️  Demonstrating Advanced Governance Features...")
    print()
    
    governance_results = {
        'policy_enforcement': [],
        'access_control': [],
        'audit_trail': [],
        'compliance_monitoring': []
    }
    
    # 1. Policy Enforcement Simulation
    print("📋 Policy Enforcement:")
    
    # Simulate different policy scenarios
    policies = [
        {
            'name': 'cost_limit_policy',
            'description': 'Experiments must not exceed $20',
            'max_cost': 20.0,
            'violation_action': 'block'
        },
        {
            'name': 'data_residency_policy', 
            'description': 'Data must remain in specified regions',
            'allowed_regions': ['us-east-1', 'eu-west-1'],
            'violation_action': 'warn'
        },
        {
            'name': 'artifact_approval_policy',
            'description': 'Production artifacts require approval',
            'required_approvers': ['senior_ml_engineer', 'data_science_lead'],
            'violation_action': 'require_approval'
        }
    ]
    
    for policy in policies:
        # Simulate policy check
        policy_result = {
            'policy_name': policy['name'],
            'description': policy['description'],
            'status': 'enforced',
            'violations_detected': random.randint(0, 2),
            'auto_remediation_applied': random.choice([True, False])
        }
        
        governance_results['policy_enforcement'].append(policy_result)
        
        status = '✅ Compliant' if policy_result['violations_detected'] == 0 else f'⚠️  {policy_result["violations_detected"]} violations'
        print(f"   • {policy['name']}: {status}")
    
    # 2. Access Control Demonstration
    print(f"\n🔐 Access Control:")
    
    access_scenarios = [
        {'user': 'data_scientist_a', 'resource': 'experiment_config', 'action': 'read', 'allowed': True},
        {'user': 'data_scientist_a', 'resource': 'production_model', 'action': 'deploy', 'allowed': False},
        {'user': 'ml_engineer_lead', 'resource': 'production_model', 'action': 'deploy', 'allowed': True},
        {'user': 'external_contractor', 'resource': 'sensitive_dataset', 'action': 'access', 'allowed': False}
    ]
    
    for scenario in access_scenarios:
        access_result = {
            'user': scenario['user'],
            'resource': scenario['resource'],
            'action': scenario['action'],
            'decision': 'allow' if scenario['allowed'] else 'deny',
            'reason': 'role_based_permissions'
        }
        
        governance_results['access_control'].append(access_result)
        
        print(f"   • {scenario['user']} → {scenario['action']} {scenario['resource']}: {'✅ Allowed' if scenario['allowed'] else '❌ Denied'}")
    
    # 3. Audit Trail Generation
    print(f"\n📝 Audit Trail Generation:")
    
    audit_events = [
        {'event': 'experiment_started', 'user': 'data_scientist_a', 'details': 'Started distributed training experiment'},
        {'event': 'model_deployed', 'user': 'ml_engineer_lead', 'details': 'Deployed model v2.1 to production'},
        {'event': 'data_accessed', 'user': 'data_scientist_b', 'details': 'Accessed customer dataset for analysis'},
        {'event': 'policy_violation', 'user': 'contractor_x', 'details': 'Attempted to access restricted resource'}
    ]
    
    for event in audit_events:
        audit_entry = {
            'event_id': f"audit_{hash(str(event)) % 10000:04d}",
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event['event'],
            'user': event['user'],
            'details': event['details'],
            'ip_address': f"10.0.{random.randint(1,255)}.{random.randint(1,255)}",
            'session_id': f"session_{random.randint(1000,9999)}"
        }
        
        governance_results['audit_trail'].append(audit_entry)
        
        print(f"   • {event['event']} by {event['user']}: {event['details']}")
    
    # 4. Compliance Monitoring
    print(f"\n📊 Compliance Monitoring:")
    
    compliance_checks = [
        {'requirement': 'GDPR Data Processing', 'status': 'compliant', 'last_check': datetime.utcnow()},
        {'requirement': 'SOX Financial Controls', 'status': 'compliant', 'last_check': datetime.utcnow()},
        {'requirement': 'HIPAA Data Protection', 'status': 'non_applicable', 'last_check': datetime.utcnow()},
        {'requirement': 'Internal ML Model Policy', 'status': 'compliant', 'last_check': datetime.utcnow()}
    ]
    
    for check in compliance_checks:
        compliance_result = {
            'requirement': check['requirement'],
            'status': check['status'],
            'last_assessment': check['last_check'].isoformat(),
            'next_review_due': (check['last_check'] + timedelta(days=90)).isoformat(),
            'risk_level': 'low' if check['status'] == 'compliant' else 'high'
        }
        
        governance_results['compliance_monitoring'].append(compliance_result)
        
        status_emoji = '✅' if check['status'] == 'compliant' else '⚠️' if check['status'] == 'non_applicable' else '❌'
        print(f"   • {check['requirement']}: {status_emoji} {check['status'].replace('_', ' ').title()}")
    
    return governance_results


def generate_enterprise_governance_report(campaign: AdvancedMLCampaign, governance_features: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive enterprise governance report."""
    
    print(f"\n📊 Generating Enterprise Governance Report...")
    
    # Calculate campaign statistics
    total_experiments = len(campaign.completed_runs)
    total_cost = campaign.total_cost
    avg_cost_per_experiment = total_cost / max(total_experiments, 1)
    
    # Calculate success rate
    failed_experiments = len([v for v in campaign.policy_violations if v.severity == PolicyViolationSeverity.ERROR])
    success_rate = ((total_experiments - failed_experiments) / max(total_experiments, 1)) * 100
    
    # Analyze cost distribution
    experiment_costs = [run['results']['total_cost'] for run in campaign.completed_runs if 'results' in run]
    cost_variance = np.var(experiment_costs) if experiment_costs else 0
    
    # Performance analysis
    accuracies = [run['results']['final_accuracy'] for run in campaign.completed_runs if 'results' in run]
    avg_accuracy = np.mean(accuracies) if accuracies else 0
    
    # Governance compliance score
    total_violations = len(campaign.policy_violations)
    compliance_score = max(0, 100 - (total_violations * 5))  # 5 points per violation
    
    # Policy enforcement effectiveness
    policy_violations_by_severity = {}
    for violation in campaign.policy_violations:
        severity = violation.severity.value
        policy_violations_by_severity[severity] = policy_violations_by_severity.get(severity, 0) + 1
    
    # Access control statistics
    access_attempts = len(governance_features['access_control'])
    access_denials = len([a for a in governance_features['access_control'] if a['decision'] == 'deny'])
    access_control_effectiveness = (access_denials / max(access_attempts, 1)) * 100
    
    # Audit trail completeness
    audit_events = len(governance_features['audit_trail'])
    audit_coverage = min(100, (audit_events / max(total_experiments * 3, 1)) * 100)  # Expect ~3 events per experiment
    
    # Compliance status summary
    compliance_checks = governance_features['compliance_monitoring']
    compliant_requirements = len([c for c in compliance_checks if c['status'] == 'compliant'])
    compliance_rate = (compliant_requirements / max(len(compliance_checks), 1)) * 100
    
    # Generate executive summary
    executive_summary = {
        'campaign_overview': {
            'campaign_id': campaign.campaign_id,
            'duration_days': (datetime.utcnow() - campaign.start_time).days,
            'total_experiments': total_experiments,
            'success_rate_percent': round(success_rate, 1),
            'total_cost_usd': round(total_cost, 2),
            'average_cost_per_experiment': round(avg_cost_per_experiment, 2)
        },
        
        'performance_metrics': {
            'average_model_accuracy': round(avg_accuracy, 3),
            'cost_efficiency_score': round((avg_accuracy / avg_cost_per_experiment) * 100, 1) if avg_cost_per_experiment > 0 else 0,
            'resource_utilization_rate': round(np.mean([run['results']['resource_efficiency'] for run in campaign.completed_runs if 'results' in run]), 2),
            'experiment_cost_variance': round(cost_variance, 4)
        },
        
        'governance_compliance': {
            'overall_compliance_score': round(compliance_score, 1),
            'policy_violations_total': total_violations,
            'policy_violations_by_severity': policy_violations_by_severity,
            'compliance_requirements_met': f"{compliant_requirements}/{len(compliance_checks)}",
            'compliance_rate_percent': round(compliance_rate, 1)
        },
        
        'security_and_access': {
            'access_control_effectiveness_percent': round(access_control_effectiveness, 1),
            'total_access_attempts': access_attempts,
            'access_denials': access_denials,
            'audit_trail_completeness_percent': round(audit_coverage, 1),
            'audit_events_captured': audit_events
        },
        
        'cost_governance': {
            'budget_utilization_percent': round((total_cost / campaign.governance_config.max_total_cost) * 100, 1),
            'cost_alert_threshold_breaches': len([cp for cp in campaign.compliance_checkpoints if cp['type'] == 'cost_threshold_reached']),
            'cost_optimization_opportunities': []
        },
        
        'artifact_management': {
            'total_artifacts_created': len(campaign.artifacts_created),
            'governed_artifacts_percent': 100,  # All artifacts in this demo are governed
            'artifact_lineage_tracked': True,
            'regulatory_compliance_artifacts': len([a for a in campaign.artifacts_created if 'regulatory' in str(a)])
        }
    }
    
    # Add cost optimization recommendations
    if cost_variance > avg_cost_per_experiment * 0.5:
        executive_summary['cost_governance']['cost_optimization_opportunities'].append(
            "High cost variance detected - standardize experiment configurations"
        )
    
    if success_rate < 90:
        executive_summary['cost_governance']['cost_optimization_opportunities'].append(
            "Improve experiment success rate to reduce wasted compute costs"
        )
    
    if avg_cost_per_experiment > 10:
        executive_summary['cost_governance']['cost_optimization_opportunities'].append(
            "Consider smaller model configurations or shorter training runs"
        )
    
    # Risk assessment
    risk_factors = []
    if total_violations > 5:
        risk_factors.append("HIGH: Multiple policy violations detected")
    if compliance_rate < 95:
        risk_factors.append("MEDIUM: Some compliance requirements not met")
    if access_control_effectiveness < 50:
        risk_factors.append("MEDIUM: Access control may be too permissive")
    if total_cost > campaign.governance_config.max_total_cost * 0.9:
        risk_factors.append("MEDIUM: Approaching budget limits")
    
    executive_summary['risk_assessment'] = {
        'overall_risk_level': 'HIGH' if any('HIGH' in rf for rf in risk_factors) else 'MEDIUM' if risk_factors else 'LOW',
        'risk_factors': risk_factors,
        'mitigation_recommended': len(risk_factors) > 0
    }
    
    return executive_summary


def main():
    """Main function demonstrating advanced W&B features with governance."""
    print("🚀 W&B Advanced Features with GenOps Governance")
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Check prerequisites
    api_key = os.getenv('WANDB_API_KEY')
    if not api_key:
        print("❌ WANDB_API_KEY environment variable not set")
        print("💡 Get your API key from https://wandb.ai/settings")
        return False
    
    team = os.getenv('GENOPS_TEAM', 'advanced-ml-team')
    project = os.getenv('GENOPS_PROJECT', 'advanced-features-demo')
    customer_id = os.getenv('GENOPS_CUSTOMER_ID', 'enterprise-client-001')
    
    print(f"📋 Configuration:")
    print(f"   • Team: {team}")
    print(f"   • Project: {project}")
    print(f"   • Customer ID: {customer_id}")
    print(f"   • API Key: {'✅ Set' if api_key else '❌ Not set'}")
    print()
    
    try:
        # Import required modules
        import wandb
        from genops.providers.wandb import instrument_wandb
        
        # Create GenOps W&B adapter with advanced configuration
        print("🔧 Creating GenOps W&B adapter for advanced features...")
        adapter = instrument_wandb(
            wandb_api_key=api_key,
            team=team,
            project=project,
            customer_id=customer_id,
            environment="production",
            daily_budget_limit=500.0,   # $500 daily budget for advanced workflows
            max_experiment_cost=100.0,  # $100 max per experiment
            governance_policy="enforced",
            enable_cost_alerts=True,
            enable_governance=True,
            cost_center="ml_research_and_development",
            tags={"workflow_type": "advanced_ml_operations"}
        )
        
        print("✅ GenOps W&B adapter created successfully")
        
        # Display advanced governance configuration
        initial_metrics = adapter.get_metrics()
        print(f"\n🛡️ Advanced Governance Configuration:")
        print(f"   • Daily Budget Limit: ${initial_metrics['daily_budget_limit']:.2f}")
        print(f"   • Current Usage: ${initial_metrics['daily_usage']:.2f}")
        print(f"   • Governance Policy: {initial_metrics['governance_policy']}")
        print(f"   • Cost Center: ml_research_and_development")
        print(f"   • Customer Attribution: {customer_id}")
        
        # === MULTI-RUN CAMPAIGN EXECUTION ===
        print(f"\n" + "="*80)
        print("🎯 MULTI-RUN CAMPAIGN EXECUTION")
        print("="*80)
        
        # Create campaign governance configuration
        campaign_config = CampaignGovernanceConfig(
            campaign_id="advanced_ml_campaign_001",
            max_total_cost=200.0,
            max_concurrent_runs=3,
            cost_alert_thresholds=[50.0, 100.0, 150.0, 180.0],
            required_approvals=["ml_engineer_lead", "data_science_manager"],
            compliance_requirements=["data_governance", "model_validation", "cost_approval"],
            data_retention_policy="retain_365_days",
            access_control_policy={
                "data_scientists": ["read", "create_experiment"],
                "ml_engineers": ["read", "create_experiment", "deploy_model"],
                "managers": ["read", "create_experiment", "deploy_model", "approve_budget"]
            }
        )
        
        # Execute multi-run campaign
        campaign = run_multi_run_campaign(adapter, campaign_config)
        
        print(f"\n📊 Campaign Results:")
        print(f"   • Campaign ID: {campaign.campaign_id}")
        print(f"   • Experiments Completed: {len(campaign.completed_runs)}")
        print(f"   • Total Campaign Cost: ${campaign.total_cost:.2f}")
        print(f"   • Budget Utilization: {(campaign.total_cost/campaign_config.max_total_cost)*100:.1f}%")
        print(f"   • Artifacts Created: {len(campaign.artifacts_created)}")
        print(f"   • Policy Violations: {len(campaign.policy_violations)}")
        print(f"   • Compliance Checkpoints: {len(campaign.compliance_checkpoints)}")
        
        # Show individual experiment results
        print(f"\n🧪 Individual Experiment Results:")
        for run in campaign.completed_runs:
            results = run.get('results', {})
            print(f"   • {run['name']}:")
            print(f"     - Final Accuracy: {results.get('final_accuracy', 0):.3f}")
            print(f"     - Training Cost: ${results.get('total_cost', 0):.2f}")
            print(f"     - Resource Efficiency: {results.get('resource_efficiency', 0):.2f}")
            print(f"     - Distributed Mode: {results.get('training_mode', 'N/A')}")
        
        # === ADVANCED GOVERNANCE FEATURES ===
        print(f"\n" + "="*80)
        print("🛡️  ADVANCED GOVERNANCE FEATURES")
        print("="*80)
        
        governance_features = demonstrate_advanced_governance_features(adapter)
        
        print(f"\n📈 Governance Features Summary:")
        print(f"   • Policy Enforcement Rules: {len(governance_features['policy_enforcement'])}")
        print(f"   • Access Control Decisions: {len(governance_features['access_control'])}")
        print(f"   • Audit Trail Events: {len(governance_features['audit_trail'])}")
        print(f"   • Compliance Checks: {len(governance_features['compliance_monitoring'])}")
        
        # === ENTERPRISE GOVERNANCE REPORT ===
        print(f"\n" + "="*80)
        print("📊 ENTERPRISE GOVERNANCE REPORT")
        print("="*80)
        
        governance_report = generate_enterprise_governance_report(campaign, governance_features)
        
        # Display executive summary
        print(f"📋 Executive Summary:")
        campaign_overview = governance_report['campaign_overview']
        print(f"   • Campaign: {campaign_overview['campaign_id']}")
        print(f"   • Duration: {campaign_overview['duration_days']} days")
        print(f"   • Experiments: {campaign_overview['total_experiments']} (Success Rate: {campaign_overview['success_rate_percent']}%)")
        print(f"   • Total Cost: ${campaign_overview['total_cost_usd']:.2f} (Avg: ${campaign_overview['average_cost_per_experiment']:.2f}/experiment)")
        
        performance = governance_report['performance_metrics']
        print(f"\n🎯 Performance Metrics:")
        print(f"   • Average Model Accuracy: {performance['average_model_accuracy']:.3f}")
        print(f"   • Cost Efficiency Score: {performance['cost_efficiency_score']:.1f}")
        print(f"   • Resource Utilization: {performance['resource_utilization_rate']:.2f}")
        
        compliance = governance_report['governance_compliance']
        print(f"\n🛡️  Governance Compliance:")
        print(f"   • Overall Compliance Score: {compliance['overall_compliance_score']:.1f}/100")
        print(f"   • Policy Violations: {compliance['policy_violations_total']}")
        print(f"   • Compliance Rate: {compliance['compliance_rate_percent']:.1f}%")
        
        security = governance_report['security_and_access']
        print(f"\n🔐 Security & Access:")
        print(f"   • Access Control Effectiveness: {security['access_control_effectiveness_percent']:.1f}%")
        print(f"   • Audit Trail Completeness: {security['audit_trail_completeness_percent']:.1f}%")
        
        cost_gov = governance_report['cost_governance']
        print(f"\n💰 Cost Governance:")
        print(f"   • Budget Utilization: {cost_gov['budget_utilization_percent']:.1f}%")
        print(f"   • Cost Alert Breaches: {cost_gov['cost_alert_threshold_breaches']}")
        
        # Risk assessment
        risk = governance_report['risk_assessment']
        risk_emoji = '🔴' if risk['overall_risk_level'] == 'HIGH' else '🟡' if risk['overall_risk_level'] == 'MEDIUM' else '🟢'
        print(f"\n⚠️  Risk Assessment:")
        print(f"   • Overall Risk Level: {risk_emoji} {risk['overall_risk_level']}")
        if risk['risk_factors']:
            print(f"   • Risk Factors:")
            for rf in risk['risk_factors']:
                print(f"     - {rf}")
        
        # Optimization opportunities
        if cost_gov['cost_optimization_opportunities']:
            print(f"\n💡 Cost Optimization Opportunities:")
            for opp in cost_gov['cost_optimization_opportunities']:
                print(f"   • {opp}")
        
        # === DEMONSTRATION SUMMARY ===
        print(f"\n" + "="*80)
        print("🎉 ADVANCED FEATURES DEMONSTRATION COMPLETED")
        print("="*80)
        
        # Final governance metrics
        final_metrics = adapter.get_metrics()
        print(f"\n📊 Final Governance Metrics:")
        print(f"   • Total Daily Usage: ${final_metrics['daily_usage']:.2f}")
        print(f"   • Budget Remaining: ${final_metrics['budget_remaining']:.2f}")
        print(f"   • Operations Tracked: {final_metrics['operation_count']}")
        print(f"   • Active Experiments: {final_metrics['active_experiments']}")
        
        print(f"\n🎓 Advanced Concepts Demonstrated:")
        print(f"   ✅ Multi-run campaign management with unified governance")
        print(f"   ✅ Distributed training simulation with cost attribution")
        print(f"   ✅ Advanced artifact management with lineage tracking")
        print(f"   ✅ Comprehensive policy enforcement and compliance monitoring")
        print(f"   ✅ Enterprise-grade governance reporting and risk assessment")
        print(f"   ✅ Cross-team collaboration with fine-grained access controls")
        print(f"   ✅ Advanced cost intelligence with multi-dimensional tracking")
        print(f"   ✅ Integration patterns for ML pipeline orchestration")
        
        print(f"\n📈 Key Achievement Metrics:")
        print(f"   • Managed {len(campaign.completed_runs)} distributed experiments across multiple nodes")
        print(f"   • Tracked ${campaign.total_cost:.2f} in compute costs with detailed attribution")
        print(f"   • Maintained {compliance['compliance_score']:.1f}% governance compliance score")
        print(f"   • Generated comprehensive audit trail with {len(governance_features['audit_trail'])} events")
        print(f"   • Created {len(campaign.artifacts_created)} governed artifacts with full lineage")
        
        print(f"\n🚀 Next Steps:")
        print(f"   • Deploy patterns: python production_patterns.py")
        print(f"   • Review complete documentation: docs/integrations/wandb.md")
        print(f"   • Implement in your production ML workflows")
        print(f"   • Customize governance policies for your organization")
        
        print(f"\n💼 Enterprise Value Delivered:")
        print(f"   💰 Cost Intelligence: Complete visibility into ML experiment costs")
        print(f"   🛡️  Governance: Policy enforcement and compliance automation")
        print(f"   📊 Insights: Performance vs cost optimization across distributed workloads")
        print(f"   🔐 Security: Role-based access control and comprehensive audit trails")
        print(f"   📈 Scalability: Enterprise-ready patterns for complex ML operations")
        
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