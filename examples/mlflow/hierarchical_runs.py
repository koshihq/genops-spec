"""MLflow Hierarchical Runs Example with GenOps Governance.

Demonstrates:
- Parent-child run relationships
- Nested run tracking
- Cost aggregation across run hierarchy
- Hyperparameter tuning with nested runs
- Multi-level governance attribution
"""

import os
import sys
from pathlib import Path

import numpy as np

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import mlflow
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from genops.providers.mlflow import instrument_mlflow


def hyperparameter_search(adapter, X_train, X_test, y_train, y_test):
    """Perform hyperparameter search with nested runs."""
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5, 10],
    }

    best_accuracy = 0
    best_params = None
    best_run_id = None

    # Parent run for hyperparameter search
    with adapter.track_mlflow_run(
        experiment_name="hierarchical-demo", run_name="hyperparameter-search"
    ) as parent_run:
        print(f"Parent Run: {parent_run.info.run_id}")
        print()

        mlflow.log_param("search_type", "grid_search")
        mlflow.log_param(
            "total_combinations",
            len(param_grid["n_estimators"])
            * len(param_grid["max_depth"])
            * len(param_grid["min_samples_split"]),
        )

        # Iterate through parameter combinations
        run_count = 0
        for n_est in param_grid["n_estimators"]:
            for max_d in param_grid["max_depth"]:
                for min_split in param_grid["min_samples_split"]:
                    run_count += 1

                    # Child run for each parameter combination
                    with adapter.track_mlflow_run(
                        experiment_name="hierarchical-demo",
                        run_name=f"config-{run_count}",
                        parent_run_id=parent_run.info.run_id,
                    ) as child_run:
                        # Train model
                        model = RandomForestClassifier(
                            n_estimators=n_est,
                            max_depth=max_d,
                            min_samples_split=min_split,
                            random_state=42,
                        )
                        model.fit(X_train, y_train)

                        # Evaluate
                        y_pred = model.predict(X_test)
                        accuracy = accuracy_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred, average="weighted")

                        # Log parameters
                        mlflow.log_param("n_estimators", n_est)
                        mlflow.log_param("max_depth", max_d)
                        mlflow.log_param("min_samples_split", min_split)

                        # Log metrics
                        mlflow.log_metric("accuracy", accuracy)
                        mlflow.log_metric("f1_score", f1)

                        print(
                            f"  Run {run_count}: n_est={n_est}, "
                            f"max_depth={max_d}, min_split={min_split} "
                            f"→ Accuracy: {accuracy:.4f}"
                        )

                        # Track best
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_params = {
                                "n_estimators": n_est,
                                "max_depth": max_d,
                                "min_samples_split": min_split,
                            }
                            best_run_id = child_run.info.run_id

        # Log best results to parent
        mlflow.log_metric("best_accuracy", best_accuracy)
        mlflow.log_param("best_n_estimators", best_params["n_estimators"])
        mlflow.log_param("best_max_depth", best_params["max_depth"])
        mlflow.log_param("best_min_samples_split", best_params["min_samples_split"])
        mlflow.set_tag("best_child_run_id", best_run_id)

        print()
        print(f"✓ Search complete: {run_count} configurations tested")
        print(f"✓ Best accuracy: {best_accuracy:.4f}")
        print(f"✓ Best params: {best_params}")
        print()

    return best_params, best_accuracy, best_run_id


def cross_validation_runs(adapter, X, y):
    """Perform cross-validation with nested runs."""
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    with adapter.track_mlflow_run(
        experiment_name="hierarchical-demo", run_name="cross-validation"
    ) as parent_run:
        print(f"Parent Run: {parent_run.info.run_id}")
        print()

        mlflow.log_param("cv_strategy", "k-fold")
        mlflow.log_param("n_splits", 5)

        fold_accuracies = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            # Child run for each fold
            with adapter.track_mlflow_run(
                experiment_name="hierarchical-demo",
                run_name=f"fold-{fold}",
                parent_run_id=parent_run.info.run_id,
            ):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Train model
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

                # Evaluate
                y_pred = model.predict(X_val)
                accuracy = accuracy_score(y_val, y_pred)
                fold_accuracies.append(accuracy)

                # Log metrics
                mlflow.log_param("fold_number", fold)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("train_samples", len(train_idx))
                mlflow.log_metric("val_samples", len(val_idx))

                print(f"  Fold {fold}: Accuracy = {accuracy:.4f}")

        # Log aggregate metrics to parent
        mean_accuracy = np.mean(fold_accuracies)
        std_accuracy = np.std(fold_accuracies)

        mlflow.log_metric("mean_accuracy", mean_accuracy)
        mlflow.log_metric("std_accuracy", std_accuracy)
        mlflow.log_metric("min_accuracy", min(fold_accuracies))
        mlflow.log_metric("max_accuracy", max(fold_accuracies))

        print()
        print("✓ Cross-validation complete")
        print(f"✓ Mean accuracy: {mean_accuracy:.4f} (+/- {std_accuracy:.4f})")
        print()

    return mean_accuracy, std_accuracy


def ensemble_training(adapter, X_train, X_test, y_train, y_test):
    """Train ensemble with multiple models as child runs."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    models = {
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=50, random_state=42
        ),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "svm": SVC(kernel="rbf", random_state=42),
    }

    with adapter.track_mlflow_run(
        experiment_name="hierarchical-demo", run_name="ensemble-training"
    ) as parent_run:
        print(f"Parent Run: {parent_run.info.run_id}")
        print()

        mlflow.log_param("ensemble_size", len(models))
        mlflow.log_param("models", list(models.keys()))

        model_predictions = []
        model_accuracies = []

        for model_name, model in models.items():
            # Child run for each model
            with adapter.track_mlflow_run(
                experiment_name="hierarchical-demo",
                run_name=f"model-{model_name}",
                parent_run_id=parent_run.info.run_id,
            ):
                print(f"  Training {model_name}...")

                # Train
                model.fit(X_train, y_train)

                # Evaluate
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                # Store for ensemble
                model_predictions.append(y_pred)
                model_accuracies.append(accuracy)

                # Log
                mlflow.log_param("model_type", model_name)
                mlflow.log_metric("accuracy", accuracy)

                print(f"    ✓ Accuracy: {accuracy:.4f}")

        # Ensemble prediction (majority voting)
        print()
        print("  Computing ensemble prediction...")
        ensemble_pred = np.round(np.mean(model_predictions, axis=0))
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)

        # Log ensemble results to parent
        mlflow.log_metric("ensemble_accuracy", ensemble_accuracy)
        mlflow.log_metric("best_individual_accuracy", max(model_accuracies))
        mlflow.log_metric("mean_individual_accuracy", np.mean(model_accuracies))

        print(f"  ✓ Ensemble accuracy: {ensemble_accuracy:.4f}")
        print(f"  ✓ Best individual: {max(model_accuracies):.4f}")
        print()

    return ensemble_accuracy


def main():
    """Hierarchical runs workflow with governance."""
    print("=" * 70)
    print("MLflow Hierarchical Runs Example - GenOps Governance")
    print("=" * 70)
    print()

    # Configuration
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:///tmp/mlruns")
    team = os.getenv("GENOPS_TEAM", "ml-team")
    project = os.getenv("GENOPS_PROJECT", "hierarchical-demo")

    print("Configuration:")
    print(f"  Tracking URI: {tracking_uri}")
    print(f"  Team: {team}")
    print(f"  Project: {project}")
    print()

    # Create adapter
    adapter = instrument_mlflow(
        tracking_uri=tracking_uri, team=team, project=project, environment="development"
    )

    print("✓ MLflow adapter created")
    print()

    # Generate dataset
    print("Generating synthetic classification dataset...")
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print()

    # ========================================================================
    # Example 1: Hyperparameter Search with Nested Runs
    # ========================================================================
    print("Example 1: Hyperparameter Search")
    print("-" * 70)
    hyperparameter_search(adapter, X_train, X_test, y_train, y_test)

    # ========================================================================
    # Example 2: Cross-Validation with Nested Runs
    # ========================================================================
    print("Example 2: Cross-Validation")
    print("-" * 70)
    cross_validation_runs(adapter, X, y)

    # ========================================================================
    # Example 3: Ensemble Training with Nested Runs
    # ========================================================================
    print("Example 3: Ensemble Training")
    print("-" * 70)
    ensemble_training(adapter, X_train, X_test, y_train, y_test)

    # ========================================================================
    # Cost Summary
    # ========================================================================
    print("=" * 70)
    print("Cost Summary")
    print("=" * 70)

    metrics = adapter.get_metrics()
    print("\nGovernance Metrics:")
    print(f"  Daily Usage: ${metrics['daily_usage']:.6f}")
    print(f"  Operations Tracked: {metrics['operation_count']}")
    print(f"  Runs: {metrics.get('run_count', 'N/A')}")
    print()

    print("Hierarchical Run Benefits:")
    print("  ✓ Parent runs aggregate child costs")
    print("  ✓ Clear organization of related experiments")
    print("  ✓ Easy comparison of nested configurations")
    print("  ✓ Governance attributes inherited by children")
    print("  ✓ Complete cost attribution across hierarchy")
    print()

    print("Cost Attribution:")
    print("  • All child runs attributed to same team/project")
    print("  • Parent run costs include all children")
    print("  • Easy aggregation for budget tracking")
    print("  • Complete audit trail maintained")
    print()

    # ========================================================================
    # Cleanup Instructions
    # ========================================================================
    print("=" * 70)
    print("Example Complete!")
    print("=" * 70)
    print()
    print("View your hierarchical runs:")
    print(f"  1. Start MLflow UI: mlflow ui --backend-store-uri {tracking_uri}")
    print("  2. Open browser: http://localhost:5000")
    print("  3. Navigate to experiment 'hierarchical-demo'")
    print("  4. Expand parent runs to see nested children")
    print()
    print("Governance features enabled:")
    print(f"  ✓ All runs attributed to team '{team}'")
    print(f"  ✓ All runs attributed to project '{project}'")
    print("  ✓ Cost aggregation across run hierarchy")
    print("  ✓ Parent-child relationships preserved")
    print("  ✓ Complete governance telemetry")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExample interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError running example: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
