#!/usr/bin/env python3
"""
W&B Basic Tracking with GenOps Governance

This example demonstrates basic experiment tracking with Weights & Biases enhanced
with GenOps governance, cost intelligence, and team attribution.

Features demonstrated:
- Simple experiment tracking with W&B and GenOps
- Automatic cost attribution and team tracking
- Basic metrics logging with governance attributes
- Cost calculation and budget monitoring
- Team and project attribution for ML experiments

Usage:
    python basic_tracking.py

Prerequisites:
    pip install genops[wandb]
    export WANDB_API_KEY="your-wandb-api-key"
    export GENOPS_TEAM="your-team"      # Optional but recommended
    export GENOPS_PROJECT="your-project" # Optional but recommended

This example runs a simple ML training simulation with W&B tracking and
GenOps governance to demonstrate the basic integration patterns.
"""

import os
import random
import time
from datetime import datetime

import numpy as np


def simulate_realistic_ml_training(model_config):
    """
    Simulate a realistic ML training process with proper convergence curves.
    
    This simulates training a neural network for image classification with
    realistic training dynamics including:
    - Learning rate decay
    - Validation metrics
    - Early stopping potential
    - Realistic convergence behavior
    """
    print("🧠 Simulating realistic neural network training...")
    print(f"   • Model: {model_config.get('model_type', 'neural_network')}")
    print("   • Dataset: CIFAR-10 (simulated)")
    print(f"   • Optimizer: {model_config.get('optimizer', 'adam')}")
    print(f"   • Initial LR: {model_config.get('learning_rate', 0.001)}")

    # Model complexity affects convergence
    model_complexity = {
        'simple_cnn': {'base_acc': 0.75, 'convergence_rate': 0.8, 'cost_per_epoch': 0.08},
        'resnet18': {'base_acc': 0.85, 'convergence_rate': 0.6, 'cost_per_epoch': 0.12},
        'resnet50': {'base_acc': 0.88, 'convergence_rate': 0.4, 'cost_per_epoch': 0.18},
        'neural_network': {'base_acc': 0.80, 'convergence_rate': 0.7, 'cost_per_epoch': 0.10}
    }

    model_type = model_config.get('model_type', 'neural_network')
    model_props = model_complexity.get(model_type, model_complexity['neural_network'])

    # Training parameters
    epochs = model_config.get('epochs', 10)
    initial_lr = model_config.get('learning_rate', 0.001)
    batch_size = model_config.get('batch_size', 32)

    # Simulate dataset splits
    train_samples = 45000  # CIFAR-10 training set
    val_samples = 5000     # CIFAR-10 validation set
    steps_per_epoch = train_samples // batch_size

    print(f"   • Training samples: {train_samples:,}")
    print(f"   • Validation samples: {val_samples:,}")
    print(f"   • Steps per epoch: {steps_per_epoch}")
    print()

    # Initialize metrics tracking
    best_val_acc = 0.0
    patience_counter = 0
    patience_limit = 3

    for epoch in range(epochs):
        print(f"   📈 Epoch {epoch + 1}/{epochs}")

        # Learning rate decay
        current_lr = initial_lr * (0.95 ** epoch)  # 5% decay per epoch

        # Simulate realistic training progression
        progress = (epoch + 1) / epochs
        base_accuracy = model_props['base_acc']
        convergence_rate = model_props['convergence_rate']

        # Training accuracy (usually higher than validation)
        train_acc_gain = (base_accuracy * 0.25) * (1 - np.exp(-3 * progress * convergence_rate))
        train_accuracy = base_accuracy + train_acc_gain + random.uniform(-0.015, 0.015)

        # Validation accuracy (more conservative, the real metric)
        val_acc_gain = (base_accuracy * 0.2) * (1 - np.exp(-2.5 * progress * convergence_rate))
        val_accuracy = base_accuracy + val_acc_gain + random.uniform(-0.025, 0.025)

        # Ensure validation is typically lower than training (overfitting simulation)
        if val_accuracy > train_accuracy:
            val_accuracy = train_accuracy - random.uniform(0.01, 0.03)

        # Calculate losses (inversely related to accuracy)
        train_loss = max(0.02, 2.5 * (1 - train_accuracy) + random.uniform(-0.1, 0.1))
        val_loss = max(0.02, 2.5 * (1 - val_accuracy) + random.uniform(-0.05, 0.15))

        # Clamp to realistic ranges
        train_accuracy = max(0.1, min(0.99, train_accuracy))
        val_accuracy = max(0.1, min(0.99, val_accuracy))

        # Calculate epoch cost based on model complexity and batch size
        base_cost = model_props['cost_per_epoch']
        batch_factor = batch_size / 32  # Cost scales with batch size
        epoch_cost = base_cost * batch_factor + random.uniform(-0.01, 0.01)
        epoch_cost = max(0.02, epoch_cost)  # Minimum cost

        # Simulate GPU utilization and memory usage
        gpu_util = random.uniform(85, 98)  # High GPU utilization during training
        gpu_memory = random.uniform(6.2, 7.8)  # GB memory usage

        # Early stopping logic
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            improved = True
        else:
            patience_counter += 1
            improved = False

        # Detailed progress output
        print(f"      Train: acc={train_accuracy:.4f}, loss={train_loss:.4f}")
        print(f"      Val:   acc={val_accuracy:.4f}, loss={val_loss:.4f}")
        print(f"      LR: {current_lr:.6f}, Cost: ${epoch_cost:.3f}")
        print(f"      GPU: {gpu_util:.1f}% util, {gpu_memory:.1f}GB mem")
        if improved:
            print("      ✨ New best validation accuracy!")
        print()

        # Yield comprehensive metrics
        yield {
            'epoch': epoch,
            'train_accuracy': train_accuracy,
            'train_loss': train_loss,
            'val_accuracy': val_accuracy,
            'val_loss': val_loss,
            'learning_rate': current_lr,
            'epoch_cost': epoch_cost,
            'gpu_utilization': gpu_util,
            'gpu_memory_gb': gpu_memory,
            'best_val_acc': best_val_acc,
            'patience': patience_counter,
            'steps_per_epoch': steps_per_epoch,
            'improved': improved
        }

        # Early stopping check
        if patience_counter >= patience_limit and epoch >= 5:  # Allow at least 5 epochs
            print(f"      🛑 Early stopping triggered (patience={patience_limit})")
            break

        # Simulate training time
        time.sleep(0.15)  # Slightly longer for realism

    return {
        'best_val_accuracy': best_val_acc,
        'final_train_accuracy': train_accuracy,
        'total_epochs_run': epoch + 1,
        'early_stopped': patience_counter >= patience_limit,
        'model_complexity': model_props
    }


def main():
    """Main function demonstrating basic W&B tracking with GenOps governance."""
    print("🔬 W&B Basic Tracking with GenOps Governance")
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Check prerequisites
    api_key = os.getenv('WANDB_API_KEY')
    if not api_key:
        print("❌ WANDB_API_KEY environment variable not set")
        print("💡 Get your API key from https://wandb.ai/settings")
        print("   export WANDB_API_KEY='your-api-key'")
        return False

    team = os.getenv('GENOPS_TEAM', 'demo-team')
    project = os.getenv('GENOPS_PROJECT', 'basic-tracking-demo')

    print("📋 Configuration:")
    print(f"   • Team: {team}")
    print(f"   • Project: {project}")
    print(f"   • API Key: {'✅ Set' if api_key else '❌ Not set'}")
    print()

    try:
        # Import required modules
        import wandb

        from genops.providers.wandb import instrument_wandb

        print("✅ Successfully imported W&B and GenOps modules")

        # Create GenOps W&B adapter
        print("\n🔧 Creating GenOps W&B adapter...")
        adapter = instrument_wandb(
            wandb_api_key=api_key,
            team=team,
            project=project,
            daily_budget_limit=5.0,  # $5 daily budget for demo
            max_experiment_cost=2.0,  # $2 max per experiment
            enable_cost_alerts=True,
            enable_governance=True
        )

        print("✅ GenOps W&B adapter created successfully")

        # Display initial metrics
        initial_metrics = adapter.get_metrics()
        print("\n📊 Initial Governance Metrics:")
        print(f"   • Daily Budget: ${initial_metrics['daily_budget_limit']:.2f}")
        print(f"   • Budget Remaining: ${initial_metrics['budget_remaining']:.2f}")
        print(f"   • Governance Policy: {initial_metrics['governance_policy']}")
        print(f"   • Team: {initial_metrics['team']}")
        print(f"   • Project: {initial_metrics['project']}")

        # Start W&B experiment with governance
        print("\n🚀 Starting W&B experiment with GenOps tracking...")

        experiment_config = {
            'model_type': 'neural_network',
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 10
        }

        # Track experiment lifecycle with governance
        with adapter.track_experiment_lifecycle(
            experiment_name="basic-ml-training",
            experiment_type="training",
            max_cost=2.0
        ) as experiment:

            # Initialize W&B run
            run = wandb.init(
                project=f"genops-{project}",
                name="basic-tracking-demo",
                config=experiment_config,
                tags=["demo", "basic", "genops"]
            )

            print(f"   • W&B Run ID: {run.id}")
            print(f"   • W&B Project: {run.project}")
            print(f"   • Experiment ID: {experiment.run_id}")

            # Log initial configuration
            run.log({
                'genops_team': team,
                'genops_project': project,
                'genops_experiment_cost': 0.0
            })

            # Run realistic training simulation with comprehensive metrics
            print("\n🏃 Running realistic training simulation...")
            total_cost = 0.0
            training_metrics = []

            for metrics in simulate_realistic_ml_training(experiment_config):
                # Log comprehensive metrics to W&B
                wandb_metrics = {
                    'epoch': metrics['epoch'],
                    'train_accuracy': metrics['train_accuracy'],
                    'train_loss': metrics['train_loss'],
                    'val_accuracy': metrics['val_accuracy'],
                    'val_loss': metrics['val_loss'],
                    'learning_rate': metrics['learning_rate'],
                    'gpu_utilization': metrics['gpu_utilization'],
                    'gpu_memory_gb': metrics['gpu_memory_gb'],
                    'best_val_acc': metrics['best_val_acc'],
                    'patience': metrics['patience'],
                    'steps_per_epoch': metrics['steps_per_epoch']
                }

                run.log(wandb_metrics)

                # Track epoch cost
                epoch_cost = metrics['epoch_cost']
                total_cost += epoch_cost
                training_metrics.append(metrics)

                # Update experiment cost in governance
                experiment.estimated_cost += epoch_cost

                # Check for governance violations
                if experiment.estimated_cost > 1.8:  # Approaching limit
                    print(f"   ⚠️ Approaching cost limit: ${experiment.estimated_cost:.3f}")

                # Check if training was stopped early
                if metrics['epoch'] >= 4 and metrics['patience'] >= 3:  # Early stopping triggered
                    print("   🛑 Training stopped early due to lack of improvement")
                    break

            # Calculate final experiment metrics from realistic training
            if training_metrics:
                final_metrics = training_metrics[-1]  # Last epoch metrics
                final_accuracy = final_metrics['val_accuracy']
                final_loss = final_metrics['val_loss']
                best_accuracy = final_metrics['best_val_acc']
                total_epochs = final_metrics['epoch'] + 1
            else:
                # Fallback if no training occurred
                final_accuracy = 0.5
                final_loss = 1.0
                best_accuracy = 0.5
                total_epochs = 0

            # Log final comprehensive metrics
            run.log({
                'final_val_accuracy': final_accuracy,
                'final_val_loss': final_loss,
                'best_val_accuracy': best_accuracy,
                'total_epochs': total_epochs,
                'total_cost': total_cost,
                'cost_per_epoch': total_cost / max(total_epochs, 1),
                'cost_efficiency': best_accuracy / max(total_cost, 0.01),
                'early_stopped': total_epochs < experiment_config['epochs']
            })

            # Create a simple artifact for demonstration
            artifact = wandb.Artifact("model-weights", type="model")

            # Simulate saving model weights with realistic training results
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(f"Model weights for validation accuracy: {final_accuracy:.4f}\n")
                f.write(f"Best validation accuracy achieved: {best_accuracy:.4f}\n")
                f.write(f"Training completed after {total_epochs} epochs\n")
                f.write(f"Training cost: ${total_cost:.3f}\n")
                f.write(f"Cost efficiency: {best_accuracy / max(total_cost, 0.01):.2f} accuracy/dollar\n")
                model_file = f.name

            artifact.add_file(model_file)

            # Log artifact with governance
            adapter.log_governed_artifact(
                artifact=artifact,
                cost_estimate=0.01,  # $0.01 for artifact storage
                governance_metadata={
                    'final_val_accuracy': final_accuracy,
                    'best_val_accuracy': best_accuracy,
                    'total_epochs_trained': total_epochs,
                    'training_cost': total_cost,
                    'cost_efficiency': best_accuracy / max(total_cost, 0.01),
                    'early_stopped': total_epochs < experiment_config['epochs']
                }
            )

            print("\n💾 Logged model artifact with governance metadata")

            # Update final experiment cost
            experiment.estimated_cost = total_cost + 0.01  # Include artifact cost

            # Finish W&B run
            run.finish()

            print("\n✅ Realistic ML experiment completed successfully!")
            print(f"   • Final Cost: ${experiment.estimated_cost:.3f}")
            print(f"   • Best Validation Accuracy: {best_accuracy:.4f}")
            print(f"   • Final Validation Accuracy: {final_accuracy:.4f}")
            print(f"   • Epochs Completed: {total_epochs}/{experiment_config['epochs']}")
            if total_epochs < experiment_config['epochs']:
                print("   • Early Stopped: Yes (patience limit reached)")
            print(f"   • Cost Efficiency: {best_accuracy / max(total_cost, 0.01):.2f} accuracy/dollar")

        # Display final governance metrics
        final_metrics = adapter.get_metrics()
        print("\n📊 Final Governance Metrics:")
        print(f"   • Daily Usage: ${final_metrics['daily_usage']:.3f}")
        print(f"   • Budget Remaining: ${final_metrics['budget_remaining']:.3f}")
        print(f"   • Total Operations: {final_metrics['operation_count']}")
        print(f"   • Active Experiments: {final_metrics['active_experiments']}")

        # Get experiment cost summary
        experiment_summary = adapter.get_experiment_cost_summary(experiment.run_id)
        if experiment_summary:
            print("\n💰 Experiment Cost Breakdown:")
            print(f"   • Total Cost: ${experiment_summary.total_cost:.3f}")
            print(f"   • Compute Cost: ${experiment_summary.compute_cost:.3f}")
            print(f"   • Storage Cost: ${experiment_summary.storage_cost:.3f}")
            print(f"   • Duration: {experiment_summary.experiment_duration / 60:.1f} minutes")
            print(f"   • Efficiency: ${experiment_summary.resource_efficiency:.3f}/hour")

        print("\n🎉 Basic tracking with governance completed successfully!")
        print("\n📚 What happened:")
        print("   ✅ Created GenOps W&B adapter with governance policies")
        print("   ✅ Tracked ML experiment with automatic cost attribution")
        print("   ✅ Logged metrics, artifacts, and governance metadata")
        print("   ✅ Monitored budget limits and governance compliance")
        print("   ✅ Generated cost breakdown and efficiency analysis")

        print("\n🚀 Next Steps:")
        print("   • Try auto-instrumentation: python auto_instrumentation.py")
        print("   • Explore experiment management: python experiment_management.py")
        print("   • Learn cost optimization: python cost_optimization.py")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Install required packages: pip install genops[wandb]")
        return False

    except Exception as e:
        print(f"❌ Error during execution: {e}")
        print("💡 Check your configuration and try running setup_validation.py first")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
