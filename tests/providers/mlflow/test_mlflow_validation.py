"""Tests for MLflow validation."""

import os
from io import StringIO
from unittest.mock import MagicMock, patch

from src.genops.providers.mlflow.validation import (
    ValidationIssue,
    ValidationResult,
    print_validation_result,
    validate_setup,
)

# ============================================================================
# ValidationIssue Tests (2 tests)
# ============================================================================


def test_validation_issue_creation():
    """Test ValidationIssue dataclass creation."""
    issue = ValidationIssue(
        severity="error",
        component="dependencies",
        message="MLflow not installed",
        suggested_fix="pip install mlflow",
        documentation_link="https://mlflow.org",
    )

    assert issue.severity == "error"
    assert issue.component == "dependencies"
    assert issue.message == "MLflow not installed"
    assert issue.suggested_fix == "pip install mlflow"
    assert issue.documentation_link == "https://mlflow.org"


def test_validation_issue_minimal():
    """Test ValidationIssue with minimal fields."""
    issue = ValidationIssue(
        severity="warning", component="configuration", message="Config warning"
    )

    assert issue.severity == "warning"
    assert issue.component == "configuration"
    assert issue.message == "Config warning"
    assert issue.suggested_fix is None
    assert issue.documentation_link is None


# ============================================================================
# ValidationResult Tests (3 tests)
# ============================================================================


def test_validation_result_creation():
    """Test ValidationResult dataclass creation."""
    result = ValidationResult()

    assert result.passed is False
    assert len(result.issues) == 0
    assert len(result.configuration) == 0
    assert len(result.dependencies) == 0
    assert len(result.connectivity) == 0


def test_validation_result_has_errors():
    """Test ValidationResult.has_errors() method."""
    result = ValidationResult()

    assert result.has_errors() is False

    result.add_issue(
        ValidationIssue(severity="warning", component="test", message="Warning")
    )

    assert result.has_errors() is False

    result.add_issue(
        ValidationIssue(severity="error", component="test", message="Error")
    )

    assert result.has_errors() is True


def test_validation_result_get_issues_by_severity():
    """Test ValidationResult.get_issues_by_severity() method."""
    result = ValidationResult()

    result.add_issue(
        ValidationIssue(severity="error", component="test", message="Error 1")
    )
    result.add_issue(
        ValidationIssue(severity="warning", component="test", message="Warning 1")
    )
    result.add_issue(
        ValidationIssue(severity="error", component="test", message="Error 2")
    )
    result.add_issue(
        ValidationIssue(severity="info", component="test", message="Info 1")
    )

    errors = result.get_issues_by_severity("error")
    warnings = result.get_issues_by_severity("warning")
    infos = result.get_issues_by_severity("info")

    assert len(errors) == 2
    assert len(warnings) == 1
    assert len(infos) == 1


# ============================================================================
# Validation Function Tests (5 tests)
# ============================================================================


def test_validate_setup_all_passed():
    """Test validate_setup when all checks pass."""
    with patch("src.genops.providers.mlflow.validation.mlflow"):
        with patch("src.genops.providers.mlflow.validation.opentelemetry"):
            with patch("src.genops.providers.mlflow.validation.genops"):
                with patch("src.genops.providers.mlflow.validation.MlflowClient"):
                    result = validate_setup(
                        tracking_uri="http://localhost:5000",
                        check_connectivity=False,  # Skip connectivity
                        check_governance=False,  # Skip governance
                    )

                    assert result.passed is True
                    assert result.dependencies["mlflow"] is True
                    assert result.dependencies["opentelemetry"] is True
                    assert result.dependencies["genops"] is True


def test_validate_setup_mlflow_missing():
    """Test validate_setup when MLflow is missing."""
    with patch(
        "src.genops.providers.mlflow.validation.mlflow", side_effect=ImportError
    ):
        result = validate_setup(check_connectivity=False, check_governance=False)

        assert result.passed is False
        assert result.dependencies["mlflow"] is False

        errors = result.get_issues_by_severity("error")
        assert any("MLflow not installed" in e.message for e in errors)


def test_validate_setup_configuration():
    """Test validate_setup configuration checks."""
    with patch("src.genops.providers.mlflow.validation.mlflow"):
        with patch("src.genops.providers.mlflow.validation.opentelemetry"):
            with patch("src.genops.providers.mlflow.validation.genops"):
                with patch.dict(
                    os.environ,
                    {
                        "MLFLOW_TRACKING_URI": "http://test-server:5000",
                        "GENOPS_TEAM": "test-team",
                        "GENOPS_PROJECT": "test-project",
                    },
                    clear=True,
                ):
                    result = validate_setup(
                        check_connectivity=False, check_governance=False
                    )

                    assert (
                        result.configuration["tracking_uri"]
                        == "http://test-server:5000"
                    )
                    assert result.configuration["genops_team"] == "test-team"
                    assert result.configuration["genops_project"] == "test-project"


def test_validate_setup_with_connectivity():
    """Test validate_setup with connectivity checks."""
    with patch("src.genops.providers.mlflow.validation.mlflow"):
        with patch("src.genops.providers.mlflow.validation.opentelemetry"):
            with patch("src.genops.providers.mlflow.validation.genops"):
                with patch(
                    "src.genops.providers.mlflow.validation.MlflowClient"
                ) as mock_client:
                    # Mock successful connectivity
                    mock_instance = MagicMock()
                    mock_instance.search_experiments.return_value = []
                    mock_instance.search_registered_models.return_value = []
                    mock_client.return_value = mock_instance

                    result = validate_setup(
                        tracking_uri="http://localhost:5000",
                        check_connectivity=True,
                        check_governance=False,
                    )

                    assert result.connectivity["tracking_server"] is True


def test_validate_setup_connectivity_failure():
    """Test validate_setup when connectivity fails."""
    with patch("src.genops.providers.mlflow.validation.mlflow"):
        with patch("src.genops.providers.mlflow.validation.opentelemetry"):
            with patch("src.genops.providers.mlflow.validation.genops"):
                with patch(
                    "src.genops.providers.mlflow.validation.MlflowClient"
                ) as mock_client:
                    # Mock connection failure
                    mock_instance = MagicMock()
                    mock_instance.search_experiments.side_effect = Exception(
                        "Connection refused"
                    )
                    mock_client.return_value = mock_instance

                    result = validate_setup(
                        tracking_uri="http://localhost:5000",
                        check_connectivity=True,
                        check_governance=False,
                    )

                    assert result.passed is False
                    errors = result.get_issues_by_severity("error")
                    assert any("Cannot connect" in e.message for e in errors)


# ============================================================================
# Print Validation Tests (3 tests)
# ============================================================================


def test_print_validation_result_success():
    """Test print_validation_result for successful validation."""
    result = ValidationResult(passed=True)
    result.dependencies = {"mlflow": True, "opentelemetry": True, "genops": True}
    result.configuration = {"tracking_uri": "http://localhost:5000"}

    # Capture output
    import sys

    captured_output = StringIO()
    sys.stdout = captured_output

    try:
        print_validation_result(result)
        output = captured_output.getvalue()

        assert "PASSED" in output
        assert "✅" in output
        assert "SUCCESS" in output
    finally:
        sys.stdout = sys.__stdout__


def test_print_validation_result_with_errors():
    """Test print_validation_result with errors."""
    result = ValidationResult(passed=False)
    result.dependencies = {"mlflow": False, "opentelemetry": True, "genops": True}
    result.add_issue(
        ValidationIssue(
            severity="error",
            component="dependencies",
            message="MLflow not installed",
            suggested_fix="pip install mlflow",
        )
    )

    import sys

    captured_output = StringIO()
    sys.stdout = captured_output

    try:
        print_validation_result(result)
        output = captured_output.getvalue()

        assert "FAILED" in output
        assert "❌" in output
        assert "MLflow not installed" in output
        assert "pip install mlflow" in output
    finally:
        sys.stdout = sys.__stdout__


def test_print_validation_result_with_warnings():
    """Test print_validation_result with warnings."""
    result = ValidationResult(passed=True)
    result.dependencies = {"mlflow": True, "opentelemetry": True, "genops": True}
    result.add_issue(
        ValidationIssue(
            severity="warning",
            component="configuration",
            message="Governance attributes not set",
            suggested_fix="Set GENOPS_TEAM and GENOPS_PROJECT",
        )
    )

    import sys

    captured_output = StringIO()
    sys.stdout = captured_output

    try:
        print_validation_result(result)
        output = captured_output.getvalue()

        assert "WARNING" in output or "⚠️" in output
        assert "Governance attributes not set" in output
    finally:
        sys.stdout = sys.__stdout__


# ============================================================================
# Integration Test (1 test)
# ============================================================================


def test_validate_setup_full_integration():
    """Test validate_setup full integration with all checks."""
    # This test simulates a complete validation scenario
    with patch("src.genops.providers.mlflow.validation.mlflow") as mock_mlflow:
        mock_mlflow.__version__ = "2.9.0"

        with patch("src.genops.providers.mlflow.validation.opentelemetry") as mock_otel:
            mock_otel.version.__version__ = "1.20.0"

            with patch("src.genops.providers.mlflow.validation.genops") as mock_genops:
                mock_genops.__version__ = "0.1.0"

                with patch(
                    "src.genops.providers.mlflow.validation.MlflowClient"
                ) as mock_client:
                    # Mock successful connectivity
                    mock_instance = MagicMock()
                    mock_instance.search_experiments.return_value = []
                    mock_instance.search_registered_models.return_value = []
                    mock_client.return_value = mock_instance

                    with patch(
                        "src.genops.providers.mlflow.validation.GenOpsTelemetry"
                    ):
                        with patch("src.genops.providers.mlflow.validation.trace"):
                            with patch.dict(
                                os.environ,
                                {
                                    "GENOPS_TEAM": "integration-team",
                                    "GENOPS_PROJECT": "integration-project",
                                },
                            ):
                                result = validate_setup(
                                    tracking_uri="http://localhost:5000",
                                    check_connectivity=True,
                                    check_governance=True,
                                )

                                # Check all validations passed
                                assert result.dependencies["mlflow"] is True
                                assert result.dependencies["opentelemetry"] is True
                                assert result.dependencies["genops"] is True

                                # Check configuration
                                assert "mlflow_version" in result.configuration
                                assert "opentelemetry_version" in result.configuration
                                assert "genops_version" in result.configuration
                                assert (
                                    result.configuration["tracking_uri"]
                                    == "http://localhost:5000"
                                )
                                assert (
                                    result.configuration["genops_team"]
                                    == "integration-team"
                                )
                                assert (
                                    result.configuration["genops_project"]
                                    == "integration-project"
                                )

                                # Check connectivity
                                if result.connectivity:
                                    assert (
                                        result.connectivity.get("tracking_server")
                                        is True
                                    )
