#!/usr/bin/env python3
"""
W&B Experiment Management with GenOps Governance

This example demonstrates complete experiment lifecycle management with Weights & Biases
enhanced with GenOps governance. It covers advanced experiment patterns including
hyperparameter sweeps, multi-run campaigns, and cost-aware experiment optimization.

Features demonstrated:
- Complete experiment lifecycle management with governance
- Hyperparameter sweep governance and budget enforcement
- Multi-run campaign tracking with unified cost intelligence
- Experiment comparison with cost-aware analysis
- Advanced cost attribution across experiment phases
- Policy compliance for long-running experiment campaigns

Usage:
    python experiment_management.py

Prerequisites:
    pip install genops[wandb]
    export WANDB_API_KEY="your-wandb-api-key"
    export GENOPS_TEAM="your-team"
    export GENOPS_PROJECT="your-project"

This example demonstrates intermediate-level W&B + GenOps integration patterns
suitable for ML teams managing complex experiment workflows.
"""

import os
import random
import time
from datetime import datetime
from typing import Any

import numpy as np


def simulate_model_training(config: dict[str, Any]) -> dict[str, float]:
    """
    Simulate realistic model training with hyperparameter sensitivity.

    This function simulates training a neural network with realistic
    performance characteristics based on hyperparameter choices.
    """
    # Extract hyperparameters
    learning_rate = config.get("learning_rate", 0.001)
    batch_size = config.get("batch_size", 32)
    model_size = config.get("model_size", "medium")
    optimizer = config.get("optimizer", "adam")
    epochs = config.get("epochs", 10)

    # Simulate model complexity impact on performance and cost
    model_complexity = {
        "small": {"params": 1e6, "cost_multiplier": 1.0, "base_accuracy": 0.80},
        "medium": {"params": 10e6, "cost_multiplier": 2.5, "base_accuracy": 0.85},
        "large": {"params": 100e6, "cost_multiplier": 8.0, "base_accuracy": 0.88},
    }

    model_info = model_complexity.get(model_size, model_complexity["medium"])

    # Simulate optimizer effects
    optimizer_effects = {
        "adam": {"convergence_speed": 1.0, "final_accuracy_boost": 0.02},
        "sgd": {"convergence_speed": 0.8, "final_accuracy_boost": 0.01},
        "adamw": {"convergence_speed": 1.1, "final_accuracy_boost": 0.025},
    }

    opt_info = optimizer_effects.get(optimizer, optimizer_effects["adam"])

    # Simulate training progression
    metrics_history = {
        "train_accuracy": [],
        "val_accuracy": [],
        "train_loss": [],
        "val_loss": [],
        "epoch_costs": [],
    }

    base_accuracy = model_info["base_accuracy"]
    convergence_speed = opt_info["convergence_speed"]

    for epoch in range(epochs):
        # Simulate learning rate impact on convergence
        lr_factor = min(1.0, learning_rate * 1000)  # Optimal around 0.001
        if learning_rate > 0.01:
            lr_factor *= 0.7  # Too high learning rate hurts performance
        elif learning_rate < 0.0001:
            lr_factor *= 0.8  # Too low learning rate slows convergence

        # Simulate batch size impact
        batch_factor = 1.0
        if batch_size < 16:
            batch_factor = 0.95  # Small batches are less stable
        elif batch_size > 128:
            batch_factor = 0.98  # Very large batches may hurt generalization

        # Progressive accuracy improvement with diminishing returns
        progress = (epoch + 1) / epochs
        accuracy_gain = (base_accuracy * 0.2) * (
            1 - np.exp(-3 * progress * convergence_speed)
        )

        train_acc = base_accuracy + accuracy_gain + random.uniform(-0.02, 0.02)
        val_acc = train_acc - random.uniform(0.01, 0.05)  # Validation slightly lower

        # Apply hyperparameter effects
        train_acc *= lr_factor * batch_factor
        val_acc *= lr_factor * batch_factor

        # Add final accuracy boost for good optimizers
        if epoch == epochs - 1:
            val_acc += opt_info["final_accuracy_boost"]

        # Calculate losses (inversely related to accuracy)
        train_loss = max(0.01, 2.0 * (1 - train_acc) + random.uniform(-0.1, 0.1))
        val_loss = max(0.01, 2.0 * (1 - val_acc) + random.uniform(-0.1, 0.1))

        # Clamp to realistic ranges
        train_acc = max(0.1, min(0.99, train_acc))
        val_acc = max(0.1, min(0.99, val_acc))

        # Calculate epoch cost based on model complexity and batch size
        base_epoch_cost = model_info["cost_multiplier"] * 0.02  # Base cost per epoch
        batch_cost_factor = batch_size / 64  # Cost scales with batch size
        epoch_cost = base_epoch_cost * batch_cost_factor + random.uniform(-0.005, 0.005)

        metrics_history["train_accuracy"].append(train_acc)
        metrics_history["val_accuracy"].append(val_acc)
        metrics_history["train_loss"].append(train_loss)
        metrics_history["val_loss"].append(val_loss)
        metrics_history["epoch_costs"].append(max(0.001, epoch_cost))

        # Simulate training time
        time.sleep(0.1)

    return {
        "final_train_accuracy": metrics_history["train_accuracy"][-1],
        "final_val_accuracy": metrics_history["val_accuracy"][-1],
        "final_train_loss": metrics_history["train_loss"][-1],
        "final_val_loss": metrics_history["val_loss"][-1],
        "total_cost": sum(metrics_history["epoch_costs"]),
        "cost_per_accuracy": sum(metrics_history["epoch_costs"]) / max(val_acc, 0.01),
        "model_parameters": model_info["params"],
        "metrics_history": metrics_history,
    }


def run_hyperparameter_sweep(
    adapter, sweep_config: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    Run a hyperparameter sweep with governance tracking.

    This demonstrates how to manage multi-run experiments with
    unified cost tracking and governance compliance.
    """
    from itertools import product

    import wandb

    print("🔬 Starting hyperparameter sweep with governance...")

    # Generate all parameter combinations
    param_names = list(sweep_config.keys())
    param_values = [sweep_config[name] for name in param_names]
    param_combinations = list(product(*param_values))

    print(f"   • Total combinations: {len(param_combinations)}")
    print(f"   • Parameters: {param_names}")

    sweep_results = []

    # Track the entire sweep as a campaign
    with adapter.track_experiment_lifecycle(
        experiment_name="hyperparameter_sweep",
        experiment_type="parameter_optimization",
        max_cost=len(param_combinations) * 2.0,  # $2 per run estimate
    ) as campaign:
        for i, param_combo in enumerate(param_combinations):
            # Create configuration for this run
            config = dict(zip(param_names, param_combo))
            config["run_id"] = i + 1

            print(f"\n   🏃 Run {i + 1}/{len(param_combinations)}: {config}")

            # Initialize W&B run for this configuration
            run_name = f"sweep_run_{i + 1}"
            run = wandb.init(
                project="genops-experiment-sweep",
                name=run_name,
                config=config,
                tags=["sweep", "hyperparameter_optimization", "genops"],
                reinit=True,  # Allow multiple inits in same process
            )

            try:
                # Run training simulation
                results = simulate_model_training(config)

                # Log metrics to W&B
                wandb.log(
                    {
                        "final_train_accuracy": results["final_train_accuracy"],
                        "final_val_accuracy": results["final_val_accuracy"],
                        "final_train_loss": results["final_train_loss"],
                        "final_val_loss": results["final_val_loss"],
                        "total_cost": results["total_cost"],
                        "cost_per_accuracy": results["cost_per_accuracy"],
                        "model_parameters": results["model_parameters"],
                    }
                )

                # Log training progression
                for epoch, (
                    train_acc,
                    val_acc,
                    train_loss,
                    val_loss,
                    cost,
                ) in enumerate(
                    zip(
                        results["metrics_history"]["train_accuracy"],
                        results["metrics_history"]["val_accuracy"],
                        results["metrics_history"]["train_loss"],
                        results["metrics_history"]["val_loss"],
                        results["metrics_history"]["epoch_costs"],
                    )
                ):
                    wandb.log(
                        {
                            "epoch": epoch,
                            "epoch_train_accuracy": train_acc,
                            "epoch_val_accuracy": val_acc,
                            "epoch_train_loss": train_loss,
                            "epoch_val_loss": val_loss,
                            "epoch_cost": cost,
                        }
                    )

                # Update campaign cost
                campaign.estimated_cost += results["total_cost"]

                # Store results for analysis
                result_summary = {
                    "run_id": i + 1,
                    "config": config,
                    "final_val_accuracy": results["final_val_accuracy"],
                    "total_cost": results["total_cost"],
                    "cost_efficiency": results["final_val_accuracy"]
                    / results["total_cost"],
                    "wandb_url": run.url,
                }

                sweep_results.append(result_summary)

                print(
                    f"      ✅ Accuracy: {results['final_val_accuracy']:.3f}, Cost: ${results['total_cost']:.3f}"
                )

            except Exception as e:
                print(f"      ❌ Run failed: {e}")

            finally:
                run.finish()

    return sweep_results


def analyze_sweep_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze hyperparameter sweep results with cost intelligence."""

    print("\n📊 Analyzing sweep results...")

    if not results:
        return {"error": "No successful runs to analyze"}

    # Sort by validation accuracy (best first)
    sorted_by_accuracy = sorted(
        results, key=lambda x: x["final_val_accuracy"], reverse=True
    )

    # Sort by cost efficiency (best accuracy per dollar)
    sorted_by_efficiency = sorted(
        results, key=lambda x: x["cost_efficiency"], reverse=True
    )

    # Sort by cost (cheapest first)
    sorted_by_cost = sorted(results, key=lambda x: x["total_cost"])

    best_accuracy = sorted_by_accuracy[0]
    most_efficient = sorted_by_efficiency[0]
    cheapest = sorted_by_cost[0]

    # Calculate statistics
    accuracies = [r["final_val_accuracy"] for r in results]
    costs = [r["total_cost"] for r in results]
    efficiencies = [r["cost_efficiency"] for r in results]

    analysis = {
        "total_runs": len(results),
        "best_accuracy_run": best_accuracy,
        "most_efficient_run": most_efficient,
        "cheapest_run": cheapest,
        "statistics": {
            "accuracy_mean": np.mean(accuracies),
            "accuracy_std": np.std(accuracies),
            "cost_mean": np.mean(costs),
            "cost_std": np.std(costs),
            "efficiency_mean": np.mean(efficiencies),
            "total_sweep_cost": sum(costs),
        },
        "recommendations": [],
    }

    # Generate recommendations
    if most_efficient != best_accuracy:
        analysis["recommendations"].append(
            f"💡 Most efficient config (accuracy/cost) differs from highest accuracy. "
            f"Consider config {most_efficient['config']} for better cost efficiency."
        )

    if analysis["statistics"]["cost_std"] > analysis["statistics"]["cost_mean"] * 0.5:
        analysis["recommendations"].append(
            "💰 High cost variation detected. Model size or batch size may be key cost drivers."
        )

    if best_accuracy["final_val_accuracy"] - cheapest["final_val_accuracy"] < 0.05:
        analysis["recommendations"].append(
            "🎯 Cheapest config performs within 5% of best. Consider using cheaper configuration."
        )

    return analysis


def main():
    """Main function demonstrating experiment management with governance."""
    print("🔬 W&B Experiment Management with GenOps Governance")
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Check prerequisites
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        print("❌ WANDB_API_KEY environment variable not set")
        print("💡 Get your API key from https://wandb.ai/settings")
        return False

    team = os.getenv("GENOPS_TEAM", "ml-research-team")
    project = os.getenv("GENOPS_PROJECT", "experiment-management-demo")

    print("📋 Configuration:")
    print(f"   • Team: {team}")
    print(f"   • Project: {project}")
    print(f"   • API Key: {'✅ Set' if api_key else '❌ Not set'}")
    print()

    try:
        # Import required modules
        import wandb

        from genops.providers.wandb import instrument_wandb

        # Create GenOps W&B adapter
        print("🔧 Creating GenOps W&B adapter for experiment management...")
        adapter = instrument_wandb(
            wandb_api_key=api_key,
            team=team,
            project=project,
            daily_budget_limit=25.0,  # $25 daily budget for experiments
            max_experiment_cost=20.0,  # $20 max per experiment
            enable_cost_alerts=True,
            enable_governance=True,
        )

        print("✅ GenOps W&B adapter created successfully")

        # Display governance configuration
        initial_metrics = adapter.get_metrics()
        print("\n🛡️ Governance Configuration:")
        print(f"   • Daily Budget Limit: ${initial_metrics['daily_budget_limit']:.2f}")
        print("   • Max Experiment Cost: $20.00")
        print(f"   • Current Usage: ${initial_metrics['daily_usage']:.2f}")
        print(f"   • Governance Policy: {initial_metrics['governance_policy']}")

        # Define hyperparameter sweep configuration
        print("\n🔬 Experiment Plan: Hyperparameter Sweep")
        sweep_config = {
            "learning_rate": [0.0001, 0.001, 0.01],
            "batch_size": [16, 32, 64],
            "model_size": ["small", "medium"],
            "optimizer": ["adam", "adamw"],
            "epochs": [5],  # Short runs for demo
        }

        print(f"   • Parameters to test: {list(sweep_config.keys())}")
        print(
            f"   • Total combinations: {np.prod([len(v) for v in sweep_config.values()])}"
        )
        print("   • Estimated time: 3-5 minutes")
        print("   • Estimated cost: $5-15")

        # Run hyperparameter sweep with governance
        print("\n🚀 Starting hyperparameter sweep with governance tracking...")
        sweep_results = run_hyperparameter_sweep(adapter, sweep_config)

        if not sweep_results:
            print("❌ No successful runs completed")
            return False

        # Analyze results
        analysis = analyze_sweep_results(sweep_results)

        print("\n📈 Sweep Results Analysis:")
        print(f"   • Successful runs: {analysis['total_runs']}")
        print(f"   • Total cost: ${analysis['statistics']['total_sweep_cost']:.3f}")
        print(f"   • Average accuracy: {analysis['statistics']['accuracy_mean']:.3f}")
        print(
            f"   • Cost range: ${min([r['total_cost'] for r in sweep_results]):.3f} - ${max([r['total_cost'] for r in sweep_results]):.3f}"
        )

        print("\n🏆 Best Results:")
        best = analysis["best_accuracy_run"]
        efficient = analysis["most_efficient_run"]
        cheap = analysis["cheapest_run"]

        print(f"   📊 Best Accuracy: {best['final_val_accuracy']:.3f}")
        print(f"      Config: {best['config']}")
        print(f"      Cost: ${best['total_cost']:.3f}")
        print(f"      URL: {best['wandb_url']}")

        print(
            f"   💰 Most Cost-Efficient: {efficient['cost_efficiency']:.1f} accuracy/dollar"
        )
        print(f"      Config: {efficient['config']}")
        print(
            f"      Accuracy: {efficient['final_val_accuracy']:.3f}, Cost: ${efficient['total_cost']:.3f}"
        )

        print(f"   💸 Cheapest: ${cheap['total_cost']:.3f}")
        print(f"      Config: {cheap['config']}")
        print(f"      Accuracy: {cheap['final_val_accuracy']:.3f}")

        # Display recommendations
        if analysis["recommendations"]:
            print("\n💡 Optimization Recommendations:")
            for i, rec in enumerate(analysis["recommendations"], 1):
                print(f"   {i}. {rec}")

        # Show governance impact
        final_metrics = adapter.get_metrics()
        print("\n🛡️ Governance Impact:")
        print(f"   • Total Usage: ${final_metrics['daily_usage']:.3f}")
        print(f"   • Budget Remaining: ${final_metrics['budget_remaining']:.3f}")
        print(f"   • Experiments Tracked: {final_metrics['operation_count']}")

        # Create experiment comparison summary
        print("\n📊 Creating experiment comparison...")

        # Initialize a summary run
        summary_run = wandb.init(
            project="genops-experiment-summary",
            name="sweep_analysis",
            tags=["summary", "analysis", "genops"],
        )

        # Log summary metrics
        summary_run.log(
            {
                "sweep_total_runs": analysis["total_runs"],
                "sweep_total_cost": analysis["statistics"]["total_sweep_cost"],
                "sweep_best_accuracy": analysis["best_accuracy_run"][
                    "final_val_accuracy"
                ],
                "sweep_best_efficiency": analysis["most_efficient_run"][
                    "cost_efficiency"
                ],
                "sweep_cheapest_cost": analysis["cheapest_run"]["total_cost"],
                "governance_budget_used": final_metrics["daily_usage"],
                "governance_budget_remaining": final_metrics["budget_remaining"],
            }
        )

        # Create comparison table
        comparison_data = []
        for result in sweep_results[:5]:  # Top 5 results
            comparison_data.append(
                [
                    result["run_id"],
                    result["config"]["learning_rate"],
                    result["config"]["batch_size"],
                    result["config"]["model_size"],
                    result["config"]["optimizer"],
                    f"{result['final_val_accuracy']:.3f}",
                    f"${result['total_cost']:.3f}",
                    f"{result['cost_efficiency']:.1f}",
                ]
            )

        comparison_table = wandb.Table(
            columns=[
                "Run ID",
                "LR",
                "Batch Size",
                "Model Size",
                "Optimizer",
                "Val Acc",
                "Cost",
                "Efficiency",
            ],
            data=comparison_data,
        )

        summary_run.log({"experiment_comparison": comparison_table})
        summary_run.finish()

        print(f"   ✅ Summary logged to W&B: {summary_run.url}")

        print("\n🎉 Experiment management with governance completed successfully!")

        print("\n📚 What you learned:")
        print("   ✅ How to run hyperparameter sweeps with unified governance")
        print("   ✅ Multi-run campaign tracking with cost aggregation")
        print("   ✅ Cost-aware experiment analysis and optimization")
        print("   ✅ Policy compliance for long-running experiment workflows")
        print("   ✅ Advanced cost attribution across experiment phases")

        print("\n🚀 Next Steps:")
        print("   • Learn cost optimization: python cost_optimization.py")
        print("   • Explore advanced features: python advanced_features.py")
        print("   • Deploy in production: python production_patterns.py")

        print("\n💡 Key Insights from this Sweep:")
        if analysis["recommendations"]:
            for rec in analysis["recommendations"][:2]:
                print(f"   • {rec}")
        print(
            f"   • Total experimental cost was ${analysis['statistics']['total_sweep_cost']:.2f}"
        )
        print(
            f"   • Best config achieved {analysis['best_accuracy_run']['final_val_accuracy']:.1%} accuracy"
        )
        print(f"   • Most efficient config: {analysis['most_efficient_run']['config']}")

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
