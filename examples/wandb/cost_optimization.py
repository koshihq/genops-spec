#!/usr/bin/env python3
"""
W&B Cost Optimization with GenOps Governance

This example demonstrates advanced cost optimization techniques for ML experiments
using Weights & Biases enhanced with GenOps governance. It covers cost-aware
experiment planning, budget monitoring, resource efficiency analysis, and cost
forecasting based on historical patterns.

Features demonstrated:
- Cost-aware experiment planning and optimization strategies
- Real-time budget monitoring with automatic alerts and interventions
- Resource efficiency analysis with cost-per-accuracy optimization
- Cost forecasting based on historical experiment patterns
- Cross-provider cost comparison and migration analysis
- Budget-constrained hyperparameter optimization strategies

Usage:
    python cost_optimization.py

Prerequisites:
    pip install genops[wandb]
    export WANDB_API_KEY="your-wandb-api-key"
    export GENOPS_TEAM="your-team"
    export GENOPS_PROJECT="your-project"

This example demonstrates intermediate-level cost intelligence features
suitable for teams looking to optimize ML experiment costs and maximize
resource efficiency within budget constraints.
"""

import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np


class OptimizationStrategy(Enum):
    """Cost optimization strategy for experiments."""
    COST_FOCUSED = "cost_focused"          # Minimize cost, accept lower performance
    BALANCED = "balanced"                  # Balance cost and performance
    PERFORMANCE_FOCUSED = "performance_focused"  # Maximize performance, higher cost ok


class ResourceProfile(Enum):
    """Resource allocation profiles for experiments."""
    MINIMAL = "minimal"         # CPU-only, small datasets
    STANDARD = "standard"       # Single GPU, medium datasets
    ACCELERATED = "accelerated" # Multi-GPU, large datasets
    DISTRIBUTED = "distributed" # Multi-node, very large datasets


@dataclass
class CostForecast:
    """Cost forecast for planned experiments."""
    estimated_total_cost: float
    cost_breakdown: Dict[str, float]
    confidence_interval: Tuple[float, float]
    optimization_recommendations: List[str]
    cost_drivers: List[str]


@dataclass
class ExperimentCostProfile:
    """Cost profile for a specific experiment configuration."""
    config: Dict[str, Any]
    estimated_cost: float
    expected_performance: float
    cost_efficiency: float  # performance / cost
    resource_profile: ResourceProfile
    estimated_duration: float


class CostOptimizedMLExperiment:
    """Simulates ML experiment with realistic cost and performance characteristics."""

    @staticmethod
    def estimate_experiment_cost(
        config: Dict[str, Any],
        resource_profile: ResourceProfile = ResourceProfile.STANDARD
    ) -> float:
        """Estimate cost for experiment based on configuration and resources."""

        # Base costs by resource profile (per hour)
        profile_costs = {
            ResourceProfile.MINIMAL: {"base": 0.10, "multiplier": 1.0},
            ResourceProfile.STANDARD: {"base": 0.75, "multiplier": 2.5},
            ResourceProfile.ACCELERATED: {"base": 2.50, "multiplier": 4.0},
            ResourceProfile.DISTRIBUTED: {"base": 8.00, "multiplier": 6.0}
        }

        profile_info = profile_costs[resource_profile]
        base_cost = profile_info["base"]
        multiplier = profile_info["multiplier"]

        # Extract experiment parameters
        epochs = config.get('epochs', 10)
        batch_size = config.get('batch_size', 32)
        model_size = config.get('model_size', 'medium')
        dataset_size = config.get('dataset_size', 'medium')

        # Model complexity factors
        model_factors = {
            'small': 0.7,
            'medium': 1.0,
            'large': 1.8,
            'xlarge': 3.2
        }

        # Dataset size factors
        dataset_factors = {
            'small': 0.5,
            'medium': 1.0,
            'large': 1.8,
            'xlarge': 3.5
        }

        # Calculate duration (hours) based on configuration
        model_factor = model_factors.get(model_size, 1.0)
        dataset_factor = dataset_factors.get(dataset_size, 1.0)
        batch_factor = 64 / max(batch_size, 16)  # Smaller batches = longer training

        estimated_hours = (epochs * model_factor * dataset_factor * batch_factor * multiplier) / 10

        # Add random variation (±20%)
        variation = random.uniform(0.8, 1.2)
        estimated_hours *= variation

        total_cost = base_cost * estimated_hours

        # Add storage and data transfer costs (5-15% of compute cost)
        overhead_factor = random.uniform(1.05, 1.15)
        total_cost *= overhead_factor

        return round(total_cost, 4)

    @staticmethod
    def estimate_performance(
        config: Dict[str, Any],
        resource_profile: ResourceProfile = ResourceProfile.STANDARD
    ) -> float:
        """Estimate expected performance based on configuration."""

        # Base performance by resource profile
        profile_performance = {
            ResourceProfile.MINIMAL: 0.75,
            ResourceProfile.STANDARD: 0.85,
            ResourceProfile.ACCELERATED: 0.90,
            ResourceProfile.DISTRIBUTED: 0.92
        }

        base_performance = profile_performance[resource_profile]

        # Extract parameters
        learning_rate = config.get('learning_rate', 0.001)
        batch_size = config.get('batch_size', 32)
        epochs = config.get('epochs', 10)
        model_size = config.get('model_size', 'medium')
        optimizer = config.get('optimizer', 'adam')

        # Model size impact
        model_performance = {
            'small': -0.05,
            'medium': 0.0,
            'large': 0.03,
            'xlarge': 0.05
        }

        # Optimizer impact
        optimizer_boost = {
            'sgd': -0.01,
            'adam': 0.0,
            'adamw': 0.015
        }

        # Learning rate optimization (peak at 0.001)
        lr_factor = 1.0 - abs(np.log10(learning_rate) + 3) * 0.02
        lr_factor = max(0.85, min(1.0, lr_factor))

        # Epochs impact (diminishing returns)
        epoch_factor = 1.0 - np.exp(-epochs / 15) * 0.1

        # Batch size impact (optimal around 32-64)
        if batch_size < 16:
            batch_factor = 0.95
        elif batch_size > 128:
            batch_factor = 0.97
        else:
            batch_factor = 1.0

        # Calculate final performance
        performance = (
            base_performance +
            model_performance.get(model_size, 0.0) +
            optimizer_boost.get(optimizer, 0.0)
        ) * lr_factor * epoch_factor * batch_factor

        # Add random variation (±5%)
        variation = random.uniform(0.95, 1.05)
        performance *= variation

        return round(min(0.99, max(0.5, performance)), 4)


def run_cost_optimization_analysis(adapter, budget_limit: float) -> Dict[str, Any]:
    """Run comprehensive cost optimization analysis."""

    print("🔬 Running Cost Optimization Analysis...")
    print(f"   • Budget Limit: ${budget_limit:.2f}")
    print("   • Optimization Target: Maximize performance within budget")
    print()

    # Define experiment configurations to evaluate
    base_configs = [
        # Cost-focused configurations
        {
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 5,
            'model_size': 'small',
            'dataset_size': 'medium',
            'optimizer': 'sgd'
        },
        # Balanced configurations
        {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 10,
            'model_size': 'medium',
            'dataset_size': 'medium',
            'optimizer': 'adam'
        },
        {
            'learning_rate': 0.0005,
            'batch_size': 32,
            'epochs': 15,
            'model_size': 'medium',
            'dataset_size': 'large',
            'optimizer': 'adamw'
        },
        # Performance-focused configurations
        {
            'learning_rate': 0.0001,
            'batch_size': 16,
            'epochs': 25,
            'model_size': 'large',
            'dataset_size': 'large',
            'optimizer': 'adamw'
        }
    ]

    # Evaluate configurations across different resource profiles
    experiment_profiles = []

    for config in base_configs:
        for resource_profile in ResourceProfile:

            # Estimate cost and performance
            estimated_cost = CostOptimizedMLExperiment.estimate_experiment_cost(
                config, resource_profile
            )

            estimated_performance = CostOptimizedMLExperiment.estimate_performance(
                config, resource_profile
            )

            # Calculate cost efficiency
            cost_efficiency = estimated_performance / max(estimated_cost, 0.001)

            # Estimate duration (hours)
            duration = estimated_cost / 0.75  # Rough estimate based on standard GPU cost

            profile = ExperimentCostProfile(
                config=config,
                estimated_cost=estimated_cost,
                expected_performance=estimated_performance,
                cost_efficiency=cost_efficiency,
                resource_profile=resource_profile,
                estimated_duration=duration
            )

            experiment_profiles.append(profile)

    # Filter profiles within budget
    affordable_profiles = [p for p in experiment_profiles if p.estimated_cost <= budget_limit]

    if not affordable_profiles:
        return {
            'error': f'No experiments fit within budget of ${budget_limit:.2f}',
            'min_cost_required': min(p.estimated_cost for p in experiment_profiles)
        }

    # Sort by cost efficiency (best first)
    affordable_profiles.sort(key=lambda x: x.cost_efficiency, reverse=True)

    # Analysis results
    best_profile = affordable_profiles[0]
    cheapest_profile = min(affordable_profiles, key=lambda x: x.estimated_cost)
    highest_performance = max(affordable_profiles, key=lambda x: x.expected_performance)

    # Budget utilization analysis
    total_budget_used = sum(p.estimated_cost for p in affordable_profiles[:3])  # Top 3
    budget_efficiency = total_budget_used / budget_limit if budget_limit > 0 else 0

    # Generate optimization recommendations
    recommendations = []

    # Resource profile analysis
    profile_counts = {}
    for profile in affordable_profiles[:10]:  # Top 10 affordable
        resource = profile.resource_profile.value
        profile_counts[resource] = profile_counts.get(resource, 0) + 1

    most_efficient_resource = max(profile_counts.keys(), key=lambda x: profile_counts[x])
    recommendations.append(
        f"🎯 Most cost-efficient resource profile: {most_efficient_resource} "
        f"(appears in {profile_counts[most_efficient_resource]} of top 10 configs)"
    )

    # Performance vs cost tradeoff analysis
    perf_cost_ratio = best_profile.expected_performance / best_profile.estimated_cost
    cheap_perf_ratio = cheapest_profile.expected_performance / cheapest_profile.estimated_cost

    if perf_cost_ratio > cheap_perf_ratio * 1.2:
        recommendations.append(
            f"💡 Best efficiency config is {perf_cost_ratio/cheap_perf_ratio:.1f}x more efficient than cheapest option"
        )

    # Budget optimization suggestions
    if budget_efficiency < 0.8:
        recommendations.append(
            "💰 Budget underutilized: Consider running multiple experiments or upgrading configurations"
        )
    elif budget_efficiency > 0.95:
        recommendations.append(
            "⚠️  Budget nearly exhausted: Consider smaller configurations or staged experiments"
        )

    # Model size recommendations
    model_sizes = [p.config['model_size'] for p in affordable_profiles[:5]]
    most_common_model = max(set(model_sizes), key=model_sizes.count)
    if most_common_model != 'medium':
        recommendations.append(
            f"🧠 Optimal model size for budget: {most_common_model}"
        )

    return {
        'total_configurations_evaluated': len(experiment_profiles),
        'affordable_configurations': len(affordable_profiles),
        'budget_limit': budget_limit,
        'budget_utilization': budget_efficiency,

        'best_efficiency_profile': {
            'config': best_profile.config,
            'estimated_cost': best_profile.estimated_cost,
            'expected_performance': best_profile.expected_performance,
            'cost_efficiency': best_profile.cost_efficiency,
            'resource_profile': best_profile.resource_profile.value,
            'estimated_duration': best_profile.estimated_duration
        },

        'cheapest_profile': {
            'config': cheapest_profile.config,
            'estimated_cost': cheapest_profile.estimated_cost,
            'expected_performance': cheapest_profile.expected_performance,
            'cost_efficiency': cheapest_profile.cost_efficiency,
            'resource_profile': cheapest_profile.resource_profile.value
        },

        'highest_performance_profile': {
            'config': highest_performance.config,
            'estimated_cost': highest_performance.estimated_cost,
            'expected_performance': highest_performance.expected_performance,
            'cost_efficiency': highest_performance.cost_efficiency,
            'resource_profile': highest_performance.resource_profile.value
        },

        'optimization_recommendations': recommendations,

        'cost_distribution': {
            'min_cost': min(p.estimated_cost for p in affordable_profiles),
            'max_cost': max(p.estimated_cost for p in affordable_profiles),
            'avg_cost': np.mean([p.estimated_cost for p in affordable_profiles]),
            'median_cost': np.median([p.estimated_cost for p in affordable_profiles])
        },

        'performance_distribution': {
            'min_performance': min(p.expected_performance for p in affordable_profiles),
            'max_performance': max(p.expected_performance for p in affordable_profiles),
            'avg_performance': np.mean([p.expected_performance for p in affordable_profiles])
        }
    }


def run_budget_monitoring_simulation(adapter, initial_budget: float) -> Dict[str, Any]:
    """Simulate real-time budget monitoring with alerts."""

    print("📊 Simulating Budget Monitoring & Alerts...")
    print(f"   • Initial Budget: ${initial_budget:.2f}")
    print("   • Monitoring Period: 5 simulated experiments")
    print()

    current_budget = initial_budget
    spent_amounts = []
    alert_events = []
    experiment_results = []

    # Simulate 5 experiments with varying costs
    for i in range(5):
        # Generate realistic experiment configuration
        config = {
            'experiment_id': f'exp_{i+1}',
            'learning_rate': random.choice([0.0001, 0.001, 0.01]),
            'batch_size': random.choice([16, 32, 64]),
            'epochs': random.choice([5, 10, 15, 20]),
            'model_size': random.choice(['small', 'medium', 'large']),
            'resource_profile': random.choice(list(ResourceProfile))
        }

        # Estimate experiment cost
        experiment_cost = CostOptimizedMLExperiment.estimate_experiment_cost(
            config, config['resource_profile']
        )

        # Check budget before running
        if experiment_cost > current_budget:
            alert_events.append({
                'type': 'budget_insufficient',
                'experiment_id': config['experiment_id'],
                'required': experiment_cost,
                'available': current_budget,
                'action': 'experiment_blocked'
            })

            print(f"   🚫 Experiment {i+1} blocked: Cost ${experiment_cost:.2f} > Budget ${current_budget:.2f}")
            break

        # Pre-experiment budget alerts
        if experiment_cost > current_budget * 0.8:
            alert_events.append({
                'type': 'budget_warning_high',
                'experiment_id': config['experiment_id'],
                'cost_percentage': (experiment_cost / current_budget) * 100,
                'message': f'Experiment will use {(experiment_cost / current_budget) * 100:.1f}% of remaining budget'
            })
            print(f"   ⚠️  High cost warning for Experiment {i+1}: {(experiment_cost / current_budget) * 100:.1f}% of budget")

        elif experiment_cost > current_budget * 0.5:
            alert_events.append({
                'type': 'budget_warning_medium',
                'experiment_id': config['experiment_id'],
                'cost_percentage': (experiment_cost / current_budget) * 100
            })
            print(f"   💡 Medium cost alert for Experiment {i+1}: {(experiment_cost / current_budget) * 100:.1f}% of budget")

        # Run experiment simulation
        estimated_performance = CostOptimizedMLExperiment.estimate_performance(
            config, config['resource_profile']
        )

        # Apply actual cost (with some variation)
        actual_cost = experiment_cost * random.uniform(0.9, 1.1)
        current_budget -= actual_cost
        spent_amounts.append(actual_cost)

        experiment_results.append({
            'experiment_id': config['experiment_id'],
            'config': config,
            'estimated_cost': experiment_cost,
            'actual_cost': actual_cost,
            'performance': estimated_performance,
            'remaining_budget': current_budget
        })

        print(f"   ✅ Experiment {i+1}: Cost ${actual_cost:.2f}, Performance {estimated_performance:.3f}")
        print(f"      Remaining Budget: ${current_budget:.2f}")

        # Post-experiment budget alerts
        budget_used_pct = ((initial_budget - current_budget) / initial_budget) * 100

        if current_budget < initial_budget * 0.1:
            alert_events.append({
                'type': 'budget_critical',
                'remaining_budget': current_budget,
                'budget_used_percentage': budget_used_pct,
                'action': 'restrict_future_experiments'
            })
            print(f"   🚨 CRITICAL: Only ${current_budget:.2f} remaining ({100-budget_used_pct:.1f}% of budget)")

        elif current_budget < initial_budget * 0.25:
            alert_events.append({
                'type': 'budget_low',
                'remaining_budget': current_budget,
                'budget_used_percentage': budget_used_pct
            })

        time.sleep(0.2)  # Simulate experiment time

    # Generate budget optimization insights
    total_spent = initial_budget - current_budget
    avg_cost_per_experiment = np.mean(spent_amounts) if spent_amounts else 0

    insights = []
    if len(spent_amounts) > 0:
        cost_variation = np.std(spent_amounts) / avg_cost_per_experiment if avg_cost_per_experiment > 0 else 0
        if cost_variation > 0.3:
            insights.append("High cost variation detected - consider standardizing experiment configurations")

        if avg_cost_per_experiment > initial_budget * 0.2:
            insights.append("Average experiment cost is high relative to budget - consider smaller configurations")

        if len(alert_events) > 3:
            insights.append("Multiple budget alerts triggered - implement tighter cost controls")

    return {
        'initial_budget': initial_budget,
        'final_budget': current_budget,
        'total_spent': total_spent,
        'budget_utilization': (total_spent / initial_budget) * 100 if initial_budget > 0 else 0,
        'experiments_completed': len(experiment_results),
        'experiments_blocked': len([a for a in alert_events if a.get('action') == 'experiment_blocked']),
        'average_cost_per_experiment': avg_cost_per_experiment,
        'alert_events': alert_events,
        'experiment_results': experiment_results,
        'budget_insights': insights,
        'alert_summary': {
            'total_alerts': len(alert_events),
            'warning_alerts': len([a for a in alert_events if 'warning' in a['type']]),
            'critical_alerts': len([a for a in alert_events if a['type'] == 'budget_critical']),
            'blocked_experiments': len([a for a in alert_events if a.get('action') == 'experiment_blocked'])
        }
    }


def generate_cost_forecast(adapter, historical_data: List[Dict], target_experiments: int) -> CostForecast:
    """Generate cost forecast based on historical patterns."""

    print("🔮 Generating Cost Forecast...")
    print(f"   • Historical Experiments: {len(historical_data)}")
    print(f"   • Target Future Experiments: {target_experiments}")
    print()

    if len(historical_data) < 2:
        # Generate synthetic historical data for demo
        historical_data = []
        for i in range(10):
            config = {
                'learning_rate': random.choice([0.0001, 0.001, 0.01]),
                'batch_size': random.choice([16, 32, 64]),
                'model_size': random.choice(['small', 'medium', 'large']),
                'epochs': random.choice([5, 10, 15])
            }
            cost = CostOptimizedMLExperiment.estimate_experiment_cost(
                config, random.choice(list(ResourceProfile))
            )
            historical_data.append({
                'config': config,
                'actual_cost': cost,
                'timestamp': datetime.now() - timedelta(days=30-i*3)
            })

    # Analyze historical patterns
    costs = [exp['actual_cost'] for exp in historical_data]
    avg_cost = np.mean(costs)
    cost_std = np.std(costs)

    # Identify cost drivers from historical data
    cost_drivers = []

    # Analyze model size impact
    model_costs = {}
    for exp in historical_data:
        model_size = exp['config'].get('model_size', 'medium')
        if model_size not in model_costs:
            model_costs[model_size] = []
        model_costs[model_size].append(exp['actual_cost'])

    if len(model_costs) > 1:
        avg_by_model = {size: np.mean(costs) for size, costs in model_costs.items()}
        max_model = max(avg_by_model.keys(), key=lambda x: avg_by_model[x])
        min_model = min(avg_by_model.keys(), key=lambda x: avg_by_model[x])
        cost_drivers.append(f"Model size: {max_model} models cost {avg_by_model[max_model]/avg_by_model[min_model]:.1f}x more than {min_model}")

    # Analyze epoch impact
    epoch_costs = [(exp['config'].get('epochs', 10), exp['actual_cost']) for exp in historical_data]
    if len(epoch_costs) > 5:
        high_epoch = [cost for epochs, cost in epoch_costs if epochs >= 15]
        low_epoch = [cost for epochs, cost in epoch_costs if epochs <= 10]
        if high_epoch and low_epoch:
            cost_drivers.append(f"Training epochs: 15+ epochs average ${np.mean(high_epoch):.2f} vs ≤10 epochs ${np.mean(low_epoch):.2f}")

    # Generate forecast
    base_forecast = avg_cost * target_experiments

    # Apply trend analysis (simple linear trend)
    if len(historical_data) >= 5:
        recent_costs = costs[-5:]
        early_costs = costs[:5]
        trend_factor = np.mean(recent_costs) / np.mean(early_costs)
        trend_adjusted_forecast = base_forecast * trend_factor
    else:
        trend_adjusted_forecast = base_forecast
        trend_factor = 1.0

    # Calculate confidence interval (based on historical variance)
    confidence_multiplier = 1.96  # 95% confidence
    margin_of_error = confidence_multiplier * (cost_std * np.sqrt(target_experiments))

    confidence_interval = (
        max(0, trend_adjusted_forecast - margin_of_error),
        trend_adjusted_forecast + margin_of_error
    )

    # Generate optimization recommendations
    recommendations = []

    if trend_factor > 1.2:
        recommendations.append("📈 Costs trending upward - review recent configuration changes")
    elif trend_factor < 0.8:
        recommendations.append("📉 Costs trending downward - good optimization progress")

    if cost_std > avg_cost * 0.5:
        recommendations.append("📊 High cost variability - standardize experiment configurations for predictable budgeting")

    if avg_cost > 2.0:
        recommendations.append("💰 High average experiment cost - consider smaller models or shorter training runs")

    # Breakdown by cost components (estimated)
    cost_breakdown = {
        'compute': trend_adjusted_forecast * 0.75,
        'storage': trend_adjusted_forecast * 0.15,
        'data_transfer': trend_adjusted_forecast * 0.05,
        'platform_fees': trend_adjusted_forecast * 0.05
    }

    return CostForecast(
        estimated_total_cost=trend_adjusted_forecast,
        cost_breakdown=cost_breakdown,
        confidence_interval=confidence_interval,
        optimization_recommendations=recommendations,
        cost_drivers=cost_drivers
    )


def main():
    """Main function demonstrating cost optimization with governance."""
    print("💰 W&B Cost Optimization with GenOps Governance")
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Check prerequisites
    api_key = os.getenv('WANDB_API_KEY')
    if not api_key:
        print("❌ WANDB_API_KEY environment variable not set")
        print("💡 Get your API key from https://wandb.ai/settings")
        return False

    team = os.getenv('GENOPS_TEAM', 'cost-optimization-team')
    project = os.getenv('GENOPS_PROJECT', 'cost-optimization-demo')

    print("📋 Configuration:")
    print(f"   • Team: {team}")
    print(f"   • Project: {project}")
    print(f"   • API Key: {'✅ Set' if api_key else '❌ Not set'}")
    print()

    try:
        # Import required modules
        import wandb

        from genops.providers.wandb import instrument_wandb

        # Create GenOps W&B adapter with cost optimization focus
        print("🔧 Creating GenOps W&B adapter for cost optimization...")
        adapter = instrument_wandb(
            wandb_api_key=api_key,
            team=team,
            project=project,
            daily_budget_limit=50.0,  # $50 daily budget
            max_experiment_cost=15.0,  # $15 max per experiment
            enable_cost_alerts=True,
            enable_governance=True
        )

        print("✅ GenOps W&B adapter created successfully")

        # Display initial governance configuration
        initial_metrics = adapter.get_metrics()
        print("\n🛡️ Cost Optimization Configuration:")
        print(f"   • Daily Budget Limit: ${initial_metrics['daily_budget_limit']:.2f}")
        print("   • Max Experiment Cost: $15.00")
        print(f"   • Current Usage: ${initial_metrics['daily_usage']:.2f}")
        print(f"   • Cost Alerts: {'Enabled' if initial_metrics['cost_alerts_enabled'] else 'Disabled'}")

        # === COST OPTIMIZATION ANALYSIS ===
        print("\n" + "="*70)
        print("📊 COST OPTIMIZATION ANALYSIS")
        print("="*70)

        budget_limit = 40.0  # Budget for optimization analysis
        optimization_analysis = run_cost_optimization_analysis(adapter, budget_limit)

        if 'error' in optimization_analysis:
            print(f"❌ {optimization_analysis['error']}")
            print(f"💡 Minimum budget required: ${optimization_analysis['min_cost_required']:.2f}")
            return False

        print("📈 Cost Optimization Results:")
        print(f"   • Configurations Evaluated: {optimization_analysis['total_configurations_evaluated']}")
        print(f"   • Affordable Options: {optimization_analysis['affordable_configurations']}")
        print(f"   • Budget Utilization: {optimization_analysis['budget_utilization']*100:.1f}%")

        # Best efficiency configuration
        best_config = optimization_analysis['best_efficiency_profile']
        print("\n🏆 Most Cost-Efficient Configuration:")
        print(f"   • Config: {best_config['config']}")
        print(f"   • Estimated Cost: ${best_config['estimated_cost']:.2f}")
        print(f"   • Expected Performance: {best_config['expected_performance']:.3f}")
        print(f"   • Cost Efficiency: {best_config['cost_efficiency']:.1f} performance/dollar")
        print(f"   • Resource Profile: {best_config['resource_profile']}")
        print(f"   • Estimated Duration: {best_config['estimated_duration']:.1f} hours")

        # Cheapest configuration
        cheapest_config = optimization_analysis['cheapest_profile']
        print("\n💸 Cheapest Configuration:")
        print(f"   • Config: {cheapest_config['config']}")
        print(f"   • Estimated Cost: ${cheapest_config['estimated_cost']:.2f}")
        print(f"   • Expected Performance: {cheapest_config['expected_performance']:.3f}")
        print(f"   • Resource Profile: {cheapest_config['resource_profile']}")

        # Highest performance configuration
        perf_config = optimization_analysis['highest_performance_profile']
        print("\n🎯 Highest Performance Configuration (within budget):")
        print(f"   • Config: {perf_config['config']}")
        print(f"   • Estimated Cost: ${perf_config['estimated_cost']:.2f}")
        print(f"   • Expected Performance: {perf_config['expected_performance']:.3f}")
        print(f"   • Cost Efficiency: {perf_config['cost_efficiency']:.1f} performance/dollar")

        # Optimization recommendations
        print("\n💡 Optimization Recommendations:")
        for i, rec in enumerate(optimization_analysis['optimization_recommendations'], 1):
            print(f"   {i}. {rec}")

        # === BUDGET MONITORING SIMULATION ===
        print("\n" + "="*70)
        print("📊 BUDGET MONITORING SIMULATION")
        print("="*70)

        budget_simulation = run_budget_monitoring_simulation(adapter, 25.0)

        print("\n📈 Budget Monitoring Results:")
        print(f"   • Initial Budget: ${budget_simulation['initial_budget']:.2f}")
        print(f"   • Final Budget: ${budget_simulation['final_budget']:.2f}")
        print(f"   • Total Spent: ${budget_simulation['total_spent']:.2f}")
        print(f"   • Budget Utilization: {budget_simulation['budget_utilization']:.1f}%")
        print(f"   • Experiments Completed: {budget_simulation['experiments_completed']}")
        print(f"   • Experiments Blocked: {budget_simulation['experiments_blocked']}")
        print(f"   • Average Cost/Experiment: ${budget_simulation['average_cost_per_experiment']:.2f}")

        # Alert summary
        alert_summary = budget_simulation['alert_summary']
        print("\n🚨 Alert Summary:")
        print(f"   • Total Alerts: {alert_summary['total_alerts']}")
        print(f"   • Warning Alerts: {alert_summary['warning_alerts']}")
        print(f"   • Critical Alerts: {alert_summary['critical_alerts']}")
        print(f"   • Blocked Experiments: {alert_summary['blocked_experiments']}")

        # Budget insights
        if budget_simulation['budget_insights']:
            print("\n💡 Budget Management Insights:")
            for i, insight in enumerate(budget_simulation['budget_insights'], 1):
                print(f"   {i}. {insight}")

        # === COST FORECASTING ===
        print("\n" + "="*70)
        print("🔮 COST FORECASTING")
        print("="*70)

        # Use simulation results as historical data
        historical_data = [
            {
                'config': result['config'],
                'actual_cost': result['actual_cost'],
                'timestamp': datetime.now() - timedelta(days=i)
            }
            for i, result in enumerate(budget_simulation['experiment_results'])
        ]

        forecast = generate_cost_forecast(adapter, historical_data, 20)

        print("📊 Cost Forecast for 20 Future Experiments:")
        print(f"   • Estimated Total Cost: ${forecast.estimated_total_cost:.2f}")
        print(f"   • Confidence Interval: ${forecast.confidence_interval[0]:.2f} - ${forecast.confidence_interval[1]:.2f}")

        print("\n💰 Cost Breakdown:")
        for component, cost in forecast.cost_breakdown.items():
            percentage = (cost / forecast.estimated_total_cost) * 100
            print(f"   • {component.replace('_', ' ').title()}: ${cost:.2f} ({percentage:.1f}%)")

        print("\n🔍 Cost Drivers:")
        for i, driver in enumerate(forecast.cost_drivers, 1):
            print(f"   {i}. {driver}")

        print("\n💡 Forecasting Recommendations:")
        for i, rec in enumerate(forecast.optimization_recommendations, 1):
            print(f"   {i}. {rec}")

        # === DEMONSTRATE COST-OPTIMIZED EXPERIMENT ===
        print("\n" + "="*70)
        print("🚀 RUNNING COST-OPTIMIZED EXPERIMENT")
        print("="*70)

        # Use the most cost-efficient configuration from analysis
        optimal_config = best_config['config']

        print("⚡ Running experiment with optimal cost-efficiency configuration...")
        print(f"   • Configuration: {optimal_config}")
        print(f"   • Expected Cost: ${best_config['estimated_cost']:.2f}")
        print(f"   • Expected Performance: {best_config['expected_performance']:.3f}")

        # Run actual W&B experiment with cost tracking
        with adapter.track_experiment_lifecycle(
            "cost-optimized-experiment",
            experiment_type="cost_optimization",
            max_cost=best_config['estimated_cost'] * 1.2  # 20% buffer
        ) as experiment:

            # Initialize W&B run with optimal config
            run = wandb.init(
                project="genops-cost-optimization",
                name="cost-optimized-run",
                config=optimal_config,
                tags=["cost-optimized", "genops", "efficiency-focused"]
            )

            # Simulate training with the optimal configuration
            epochs = optimal_config['epochs']
            for epoch in range(epochs):
                # Simulate training metrics based on config
                base_perf = best_config['expected_performance']
                progress = (epoch + 1) / epochs

                # Progressive improvement with diminishing returns
                accuracy = base_perf * (1 - np.exp(-3 * progress)) + random.uniform(-0.01, 0.01)
                loss = (2.0 * (1 - accuracy)) + random.uniform(-0.1, 0.1)

                # Log metrics to W&B
                wandb.log({
                    'epoch': epoch,
                    'accuracy': accuracy,
                    'loss': max(0.01, loss),
                    'cost_efficiency_target': best_config['cost_efficiency'],
                    'estimated_experiment_cost': best_config['estimated_cost'],
                    'learning_rate': optimal_config['learning_rate'],
                    'batch_size': optimal_config['batch_size']
                })

                # Update experiment cost (simulate actual resource usage)
                epoch_cost = best_config['estimated_cost'] / epochs * random.uniform(0.95, 1.05)
                experiment.estimated_cost += epoch_cost

                print(f"      📊 Epoch {epoch+1:2d}: accuracy={accuracy:.3f}, loss={loss:.3f}, cost=${epoch_cost:.3f}")

                time.sleep(0.1)  # Simulate training time

            # Create cost-optimized model artifact
            artifact = wandb.Artifact('cost-optimized-model', type='model')

            # Add cost optimization metadata to artifact
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                import json
                optimization_metadata = {
                    'cost_efficiency': best_config['cost_efficiency'],
                    'total_cost': experiment.estimated_cost,
                    'final_accuracy': accuracy,
                    'optimization_strategy': 'cost_efficiency_maximization',
                    'resource_profile': best_config['resource_profile']
                }
                json.dump(optimization_metadata, f, indent=2)
                artifact.add_file(f.name, name='optimization_metadata.json')

            # Log governed artifact with cost estimate
            adapter.log_governed_artifact(
                artifact,
                cost_estimate=0.02,  # Small storage cost
                governance_metadata={
                    'cost_optimization': True,
                    'efficiency_score': best_config['cost_efficiency']
                }
            )

            run.finish()

            print("   ✅ Cost-optimized experiment completed!")
            print(f"   📊 Final Cost: ${experiment.estimated_cost:.3f}")
            print(f"   🎯 Final Performance: {accuracy:.3f}")
            print(f"   💰 Cost Efficiency: {accuracy/experiment.estimated_cost:.1f} performance/dollar")
            print(f"   🔗 W&B URL: {run.url}")

        # Final governance metrics
        final_metrics = adapter.get_metrics()
        print("\n🛡️ Final Governance Status:")
        print(f"   • Total Daily Usage: ${final_metrics['daily_usage']:.3f}")
        print(f"   • Budget Remaining: ${final_metrics['budget_remaining']:.3f}")
        print(f"   • Operations Tracked: {final_metrics['operation_count']}")

        print("\n🎉 Cost optimization analysis completed successfully!")

        print("\n📚 What you learned:")
        print("   ✅ How to perform cost optimization analysis for ML experiments")
        print("   ✅ Real-time budget monitoring with alerts and interventions")
        print("   ✅ Resource efficiency analysis and cost-per-accuracy optimization")
        print("   ✅ Cost forecasting based on historical experiment patterns")
        print("   ✅ Budget-constrained experiment planning and execution")
        print("   ✅ Cost-aware ML workflow design and governance")

        print("\n🚀 Next Steps:")
        print("   • Explore advanced features: python advanced_features.py")
        print("   • Deploy in production: python production_patterns.py")
        print("   • Review complete documentation: docs/integrations/wandb.md")

        print("\n💡 Key Cost Optimization Insights:")
        print(f"   • Most efficient configuration achieved {best_config['cost_efficiency']:.1f} performance/dollar")
        print(f"   • Budget monitoring prevented {alert_summary['blocked_experiments']} over-budget experiments")
        print(f"   • Cost forecasting predicts ${forecast.estimated_total_cost:.2f} for 20 future experiments")
        print(f"   • Resource profile '{best_config['resource_profile']}' shows best cost efficiency")

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
