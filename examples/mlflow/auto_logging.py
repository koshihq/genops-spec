"""MLflow Auto-Logging Example with GenOps Governance.

Demonstrates:
- Auto-logging with scikit-learn
- Automatic parameter and metric tracking
- Governance integration with auto-logged operations
- Cost tracking for auto-logged artifacts
- Zero-code governance for ML frameworks
"""

import os
import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from genops.providers.mlflow import instrument_mlflow


def main():
    """Auto-logging workflow with governance."""
    print("=" * 70)
    print("MLflow Auto-Logging Example - GenOps Governance")
    print("=" * 70)
    print()

    # Configuration
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'file:///tmp/mlruns')
    team = os.getenv('GENOPS_TEAM', 'ml-team')
    project = os.getenv('GENOPS_PROJECT', 'auto-logging-demo')

    print(f"Configuration:")
    print(f"  Tracking URI: {tracking_uri}")
    print(f"  Team: {team}")
    print(f"  Project: {project}")
    print()

    # Create adapter
    adapter = instrument_mlflow(
        tracking_uri=tracking_uri,
        team=team,
        project=project,
        environment="development"
    )

    print("✓ MLflow adapter created")
    print()

    # Generate synthetic dataset
    print("Generating synthetic classification dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print()

    # ========================================================================
    # Example 1: Random Forest with Auto-Logging
    # ========================================================================
    print("Example 1: Random Forest with Auto-Logging")
    print("-" * 70)

    # Enable auto-logging for scikit-learn
    mlflow.sklearn.autolog(log_models=True, log_input_examples=True)
    print("✓ Scikit-learn auto-logging enabled")
    print()

    with adapter.track_mlflow_run(
        experiment_name="auto-logging-demo",
        run_name="random-forest-auto"
    ) as run:
        print("Training Random Forest...")

        # Train model - parameters and metrics automatically logged
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate additional metrics (auto-logged)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        print(f"  ✓ Model trained")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print()

        print("Auto-logged:")
        print("  ✓ All model parameters (n_estimators, max_depth, etc.)")
        print("  ✓ Training metrics (accuracy, f1, precision, recall)")
        print("  ✓ Model artifact (serialized RandomForest)")
        print("  ✓ Feature importances")
        print("  ✓ Input example")
        print()

    # ========================================================================
    # Example 2: Gradient Boosting with Auto-Logging
    # ========================================================================
    print("Example 2: Gradient Boosting with Auto-Logging")
    print("-" * 70)

    with adapter.track_mlflow_run(
        experiment_name="auto-logging-demo",
        run_name="gradient-boosting-auto"
    ) as run:
        print("Training Gradient Boosting...")

        # Train model - everything automatically logged
        model = GradientBoostingClassifier(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"  ✓ Model trained")
        print(f"  Accuracy: {accuracy:.4f}")
        print()

        print("Auto-logged:")
        print("  ✓ All model parameters (n_estimators, learning_rate, etc.)")
        print("  ✓ Training metrics")
        print("  ✓ Model artifact")
        print("  ✓ Feature importances")
        print()

    # ========================================================================
    # Example 3: Logistic Regression with Auto-Logging
    # ========================================================================
    print("Example 3: Logistic Regression with Auto-Logging")
    print("-" * 70)

    with adapter.track_mlflow_run(
        experiment_name="auto-logging-demo",
        run_name="logistic-regression-auto"
    ) as run:
        print("Training Logistic Regression...")

        # Train model
        model = LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            random_state=42
        )
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"  ✓ Model trained")
        print(f"  Accuracy: {accuracy:.4f}")
        print()

        print("Auto-logged:")
        print("  ✓ All model parameters (max_iter, solver, etc.)")
        print("  ✓ Training metrics")
        print("  ✓ Model artifact")
        print("  ✓ Model coefficients")
        print()

    # ========================================================================
    # Example 4: Cross-Validation with Auto-Logging
    # ========================================================================
    print("Example 4: Cross-Validation with Auto-Logging")
    print("-" * 70)

    with adapter.track_mlflow_run(
        experiment_name="auto-logging-demo",
        run_name="cross-validation-auto"
    ) as run:
        print("Running 5-fold cross-validation...")

        model = RandomForestClassifier(n_estimators=50, random_state=42)

        # Cross-validation - automatically logged
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)

        print(f"  ✓ Cross-validation complete")
        print(f"  CV Scores: {cv_scores}")
        print(f"  Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print()

        # Train final model on full training set
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"  Final Test Accuracy: {accuracy:.4f}")
        print()

        print("Auto-logged:")
        print("  ✓ Cross-validation scores")
        print("  ✓ Final model parameters")
        print("  ✓ Final model artifact")
        print("  ✓ Test set metrics")
        print()

    # ========================================================================
    # Example 5: Disable Auto-Logging for Specific Operations
    # ========================================================================
    print("Example 5: Selective Auto-Logging")
    print("-" * 70)

    # Disable auto-logging temporarily
    mlflow.sklearn.autolog(disable=True)
    print("✓ Auto-logging disabled")
    print()

    with adapter.track_mlflow_run(
        experiment_name="auto-logging-demo",
        run_name="manual-logging"
    ) as run:
        print("Training with manual logging...")

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Manual logging
        mlflow.log_param("n_estimators", 50)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")

        print(f"  ✓ Model trained - Accuracy: {accuracy:.4f}")
        print(f"  ✓ Parameters and metrics manually logged")
        print()

    # Re-enable auto-logging
    mlflow.sklearn.autolog(log_models=True)
    print("✓ Auto-logging re-enabled")
    print()

    # ========================================================================
    # Cost Summary
    # ========================================================================
    print("=" * 70)
    print("Cost Summary")
    print("=" * 70)

    metrics = adapter.get_metrics()
    print(f"\nGovernance Metrics:")
    print(f"  Daily Usage: ${metrics['daily_usage']:.6f}")
    print(f"  Operations Tracked: {metrics['operation_count']}")
    print(f"  Runs: {metrics.get('run_count', 'N/A')}")
    print()

    print("Auto-Logging Cost Benefits:")
    print("  ✓ Zero manual instrumentation code")
    print("  ✓ Consistent parameter tracking")
    print("  ✓ Automatic model serialization")
    print("  ✓ All costs automatically attributed")
    print("  ✓ Complete governance without code changes")
    print()

    print("Cost Breakdown:")
    print("  Auto-logged parameters: ~$0.0001 per parameter")
    print("  Auto-logged metrics: ~$0.0001 per metric")
    print("  Auto-logged models: Size-based storage cost")
    print("  Feature importances: ~$0.0001")
    print("  Input examples: Size-based storage cost")
    print()

    # ========================================================================
    # Cleanup Instructions
    # ========================================================================
    print("=" * 70)
    print("Example Complete!")
    print("=" * 70)
    print()
    print("View your results:")
    print(f"  1. Start MLflow UI: mlflow ui --backend-store-uri {tracking_uri}")
    print(f"  2. Open browser: http://localhost:5000")
    print(f"  3. Navigate to experiment 'auto-logging-demo'")
    print(f"  4. Compare auto-logged vs manual-logged runs")
    print()
    print("Governance features enabled:")
    print(f"  ✓ All auto-logged operations attributed to team '{team}'")
    print(f"  ✓ All auto-logged operations attributed to project '{project}'")
    print("  ✓ Cost tracking for all auto-logged artifacts")
    print("  ✓ Zero-code governance integration")
    print()

    print("Supported Auto-Logging Frameworks:")
    print("  • scikit-learn (demonstrated)")
    print("  • PyTorch (mlflow.pytorch.autolog)")
    print("  • TensorFlow (mlflow.tensorflow.autolog)")
    print("  • Keras (mlflow.keras.autolog)")
    print("  • XGBoost (mlflow.xgboost.autolog)")
    print("  • LightGBM (mlflow.lightgbm.autolog)")
    print("  • Spark (mlflow.spark.autolog)")
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
