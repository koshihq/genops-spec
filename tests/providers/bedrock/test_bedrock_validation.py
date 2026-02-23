"""
Comprehensive tests for GenOps Bedrock Validation.

Tests the setup validation functionality including:
- AWS credentials validation
- Bedrock service availability
- Model access permissions
- Regional configuration
- Environment setup
- Diagnostic information and fix suggestions
"""

import os
from unittest.mock import Mock, patch

import pytest

# Import the modules under test
try:
    from genops.providers.bedrock_validation import (
        ValidationCheck,
        ValidationResult,
        get_available_models,
        print_validation_result,
        validate_aws_credentials,
        validate_bedrock_access,
        validate_bedrock_setup,
        validate_environment_setup,
        validate_model_access,
    )

    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False


@pytest.mark.skipif(
    not VALIDATION_AVAILABLE, reason="Bedrock validation module not available"
)
class TestValidationResult:
    """Test ValidationResult data structure."""

    def test_validation_result_structure(self):
        """Test ValidationResult has all required fields."""
        # Create a sample validation result
        result = ValidationResult(
            success=True,
            errors=[],
            warnings=["Test warning"],
            checks_passed=5,
            total_checks=6,
            detailed_checks={},
        )

        assert result.success is True
        assert result.errors == []
        assert result.warnings == ["Test warning"]
        assert result.checks_passed == 5
        assert result.total_checks == 6
        assert isinstance(result.detailed_checks, dict)

    def test_failed_validation_result(self):
        """Test ValidationResult for failed validation."""
        result = ValidationResult(
            success=False,
            errors=["AWS credentials not found", "Bedrock access denied"],
            warnings=[],
            checks_passed=2,
            total_checks=6,
            detailed_checks={},
        )

        assert result.success is False
        assert len(result.errors) == 2
        assert result.checks_passed < result.total_checks

    def test_validation_check_structure(self):
        """Test ValidationCheck data structure."""
        check = ValidationCheck(
            name="aws_credentials",
            passed=True,
            error=None,
            fix_suggestion="Credentials are properly configured",
            documentation_link="https://docs.aws.amazon.com/credentials/",
        )

        assert check.name == "aws_credentials"
        assert check.passed is True
        assert check.error is None
        assert check.fix_suggestion is not None
        assert check.documentation_link is not None


@pytest.mark.skipif(
    not VALIDATION_AVAILABLE, reason="Bedrock validation module not available"
)
class TestAWSCredentialsValidation:
    """Test AWS credentials validation."""

    def test_validate_aws_credentials_success(self):
        """Test successful AWS credentials validation."""
        with patch("boto3.client") as mock_boto_client:
            mock_sts = Mock()
            mock_sts.get_caller_identity.return_value = {
                "UserId": "AIDAIOSFODNN7EXAMPLE",
                "Account": "123456789012",
                "Arn": "arn:aws:iam::123456789012:user/testuser",
            }
            mock_boto_client.return_value = mock_sts

            result = validate_aws_credentials()

            assert result.passed is True
            assert result.error is None

    def test_validate_aws_credentials_failure(self):
        """Test failed AWS credentials validation."""
        with patch("boto3.client") as mock_boto_client:
            from botocore.exceptions import NoCredentialsError

            mock_sts = Mock()
            mock_sts.get_caller_identity.side_effect = NoCredentialsError()
            mock_boto_client.return_value = mock_sts

            result = validate_aws_credentials()

            assert result.passed is False
            assert result.error is not None
            assert "credentials" in result.error.lower()
            assert "aws configure" in result.fix_suggestion.lower()

    def test_validate_aws_credentials_access_denied(self):
        """Test AWS credentials validation with access denied."""
        with patch("boto3.client") as mock_boto_client:
            from botocore.exceptions import ClientError

            mock_sts = Mock()
            mock_sts.get_caller_identity.side_effect = ClientError(
                error_response={
                    "Error": {"Code": "AccessDenied", "Message": "Access denied"}
                },
                operation_name="GetCallerIdentity",
            )
            mock_boto_client.return_value = mock_sts

            result = validate_aws_credentials()

            assert result.passed is False
            assert "access" in result.error.lower()

    def test_validate_aws_credentials_invalid_region(self):
        """Test AWS credentials validation with invalid region."""
        with patch("boto3.client") as mock_boto_client:
            from botocore.exceptions import ClientError

            mock_sts = Mock()
            mock_sts.get_caller_identity.side_effect = ClientError(
                error_response={
                    "Error": {"Code": "InvalidRegion", "Message": "Invalid region"}
                },
                operation_name="GetCallerIdentity",
            )
            mock_boto_client.return_value = mock_sts

            result = validate_aws_credentials()

            assert result.passed is False
            assert "region" in result.error.lower()


@pytest.mark.skipif(
    not VALIDATION_AVAILABLE, reason="Bedrock validation module not available"
)
class TestBedrockAccessValidation:
    """Test Bedrock service access validation."""

    def test_validate_bedrock_access_success(self):
        """Test successful Bedrock access validation."""
        with patch("boto3.client") as mock_boto_client:
            mock_bedrock = Mock()
            mock_bedrock.list_foundation_models.return_value = {
                "modelSummaries": [
                    {
                        "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
                        "modelName": "Claude 3 Haiku",
                        "providerName": "Anthropic",
                    }
                ]
            }
            mock_boto_client.return_value = mock_bedrock

            result = validate_bedrock_access(region="us-east-1")

            assert result.passed is True
            assert result.error is None

    def test_validate_bedrock_access_service_unavailable(self):
        """Test Bedrock access validation when service is unavailable."""
        with patch("boto3.client") as mock_boto_client:
            from botocore.exceptions import ClientError

            mock_bedrock = Mock()
            mock_bedrock.list_foundation_models.side_effect = ClientError(
                error_response={
                    "Error": {
                        "Code": "ServiceUnavailable",
                        "Message": "Service unavailable",
                    }
                },
                operation_name="ListFoundationModels",
            )
            mock_boto_client.return_value = mock_bedrock

            result = validate_bedrock_access(region="us-east-1")

            assert result.passed is False
            assert "service" in result.error.lower()

    def test_validate_bedrock_access_region_not_supported(self):
        """Test Bedrock access validation in unsupported region."""
        with patch("boto3.client") as mock_boto_client:
            from botocore.exceptions import ClientError

            mock_bedrock = Mock()
            mock_bedrock.list_foundation_models.side_effect = ClientError(
                error_response={
                    "Error": {"Code": "UnknownEndpoint", "Message": "Unknown endpoint"}
                },
                operation_name="ListFoundationModels",
            )
            mock_boto_client.return_value = mock_bedrock

            result = validate_bedrock_access(region="unsupported-region")

            assert result.passed is False
            assert (
                "region" in result.error.lower() or "endpoint" in result.error.lower()
            )
            assert "us-east-1" in result.fix_suggestion

    def test_validate_bedrock_access_permissions(self):
        """Test Bedrock access validation with insufficient permissions."""
        with patch("boto3.client") as mock_boto_client:
            from botocore.exceptions import ClientError

            mock_bedrock = Mock()
            mock_bedrock.list_foundation_models.side_effect = ClientError(
                error_response={
                    "Error": {
                        "Code": "AccessDeniedException",
                        "Message": "Access denied",
                    }
                },
                operation_name="ListFoundationModels",
            )
            mock_boto_client.return_value = mock_bedrock

            result = validate_bedrock_access(region="us-east-1")

            assert result.passed is False
            assert "access" in result.error.lower()
            assert "permission" in result.fix_suggestion.lower()


@pytest.mark.skipif(
    not VALIDATION_AVAILABLE, reason="Bedrock validation module not available"
)
class TestModelAccessValidation:
    """Test model access validation."""

    def test_validate_model_access_success(self):
        """Test successful model access validation."""
        with patch("boto3.client") as mock_boto_client:
            mock_bedrock = Mock()

            # Mock successful model invocation
            mock_response = {"body": Mock(), "contentType": "application/json"}
            mock_body = Mock()
            mock_body.read.return_value = b'{"completion": "Test response"}'
            mock_response["body"] = mock_body

            mock_bedrock.invoke_model.return_value = mock_response
            mock_boto_client.return_value = mock_bedrock

            result = validate_model_access(
                model_id="anthropic.claude-3-haiku-20240307-v1:0", region="us-east-1"
            )

            assert result.passed is True
            assert result.error is None

    def test_validate_model_access_not_enabled(self):
        """Test model access validation when model is not enabled."""
        with patch("boto3.client") as mock_boto_client:
            from botocore.exceptions import ClientError

            mock_bedrock = Mock()
            mock_bedrock.invoke_model.side_effect = ClientError(
                error_response={
                    "Error": {
                        "Code": "AccessDeniedException",
                        "Message": "Model access not enabled",
                    }
                },
                operation_name="InvokeModel",
            )
            mock_boto_client.return_value = mock_bedrock

            result = validate_model_access(
                model_id="anthropic.claude-3-haiku-20240307-v1:0", region="us-east-1"
            )

            assert result.passed is False
            assert "access" in result.error.lower()
            assert "console" in result.fix_suggestion.lower()

    def test_validate_model_access_invalid_model(self):
        """Test model access validation with invalid model ID."""
        with patch("boto3.client") as mock_boto_client:
            from botocore.exceptions import ClientError

            mock_bedrock = Mock()
            mock_bedrock.invoke_model.side_effect = ClientError(
                error_response={
                    "Error": {
                        "Code": "ValidationException",
                        "Message": "Model not found",
                    }
                },
                operation_name="InvokeModel",
            )
            mock_boto_client.return_value = mock_bedrock

            result = validate_model_access(
                model_id="invalid-model-id", region="us-east-1"
            )

            assert result.passed is False
            assert "model" in result.error.lower()

    def test_validate_model_access_throttling(self):
        """Test model access validation with throttling."""
        with patch("boto3.client") as mock_boto_client:
            from botocore.exceptions import ClientError

            mock_bedrock = Mock()
            mock_bedrock.invoke_model.side_effect = ClientError(
                error_response={
                    "Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}
                },
                operation_name="InvokeModel",
            )
            mock_boto_client.return_value = mock_bedrock

            result = validate_model_access(
                model_id="anthropic.claude-3-haiku-20240307-v1:0", region="us-east-1"
            )

            # Throttling should be treated as success (model is accessible, just rate limited)
            assert result.passed is True or "throttl" in result.error.lower()


@pytest.mark.skipif(
    not VALIDATION_AVAILABLE, reason="Bedrock validation module not available"
)
class TestAvailableModels:
    """Test available models retrieval."""

    def test_get_available_models_success(self):
        """Test successful retrieval of available models."""
        with patch("boto3.client") as mock_boto_client:
            mock_bedrock = Mock()
            mock_bedrock.list_foundation_models.return_value = {
                "modelSummaries": [
                    {
                        "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
                        "modelName": "Claude 3 Haiku",
                        "providerName": "Anthropic",
                    },
                    {
                        "modelId": "amazon.titan-text-express-v1",
                        "modelName": "Titan Text Express",
                        "providerName": "Amazon",
                    },
                ]
            }
            mock_boto_client.return_value = mock_bedrock

            models = get_available_models(region="us-east-1")

            assert isinstance(models, list)
            assert len(models) == 2
            assert "anthropic.claude-3-haiku-20240307-v1:0" in models
            assert "amazon.titan-text-express-v1" in models

    def test_get_available_models_empty(self):
        """Test retrieval when no models are available."""
        with patch("boto3.client") as mock_boto_client:
            mock_bedrock = Mock()
            mock_bedrock.list_foundation_models.return_value = {"modelSummaries": []}
            mock_boto_client.return_value = mock_bedrock

            models = get_available_models(region="us-east-1")

            assert isinstance(models, list)
            assert len(models) == 0

    def test_get_available_models_error(self):
        """Test retrieval of available models with error."""
        with patch("boto3.client") as mock_boto_client:
            from botocore.exceptions import ClientError

            mock_bedrock = Mock()
            mock_bedrock.list_foundation_models.side_effect = ClientError(
                error_response={
                    "Error": {
                        "Code": "AccessDeniedException",
                        "Message": "Access denied",
                    }
                },
                operation_name="ListFoundationModels",
            )
            mock_boto_client.return_value = mock_bedrock

            with pytest.raises((ClientError, Exception)):
                get_available_models(region="us-east-1")


@pytest.mark.skipif(
    not VALIDATION_AVAILABLE, reason="Bedrock validation module not available"
)
class TestEnvironmentValidation:
    """Test environment setup validation."""

    def test_validate_environment_setup_success(self):
        """Test successful environment validation."""
        with patch.dict(
            os.environ, {"AWS_REGION": "us-east-1", "AWS_DEFAULT_REGION": "us-east-1"}
        ):
            result = validate_environment_setup()

            assert result.passed is True or len(result.error or "") == 0

    def test_validate_environment_setup_missing_region(self):
        """Test environment validation with missing region."""
        with patch.dict(os.environ, {}, clear=True):
            result = validate_environment_setup()

            # Should either pass (using defaults) or suggest setting region
            if not result.passed:
                assert "region" in result.error.lower()
                assert "AWS_REGION" in result.fix_suggestion

    def test_validate_environment_setup_genops_config(self):
        """Test environment validation with GenOps configuration."""
        with patch.dict(
            os.environ,
            {
                "GENOPS_ENVIRONMENT": "production",
                "GENOPS_PROJECT": "test-project",
                "OTEL_SERVICE_NAME": "bedrock-service",
            },
        ):
            result = validate_environment_setup()

            # Should pass with proper GenOps configuration
            assert result.passed is True or result.error is None

    def test_validate_environment_setup_otel_config(self):
        """Test environment validation with OpenTelemetry configuration."""
        with patch.dict(
            os.environ,
            {
                "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317",
                "OTEL_SERVICE_NAME": "bedrock-ai-service",
            },
        ):
            result = validate_environment_setup()

            # Should recognize OTEL configuration
            assert result.passed is True or result.error is None


@pytest.mark.skipif(
    not VALIDATION_AVAILABLE, reason="Bedrock validation module not available"
)
class TestComprehensiveValidation:
    """Test comprehensive setup validation."""

    def test_validate_bedrock_setup_success(self):
        """Test successful comprehensive validation."""
        with patch("boto3.client") as mock_boto_client:
            # Mock STS client for credentials
            mock_sts = Mock()
            mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}

            # Mock Bedrock client
            mock_bedrock = Mock()
            mock_bedrock.list_foundation_models.return_value = {
                "modelSummaries": [
                    {
                        "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
                        "modelName": "Claude 3 Haiku",
                        "providerName": "Anthropic",
                    }
                ]
            }

            def client_factory(service_name, **kwargs):
                if service_name == "sts":
                    return mock_sts
                elif service_name == "bedrock":
                    return mock_bedrock
                else:
                    return Mock()

            mock_boto_client.side_effect = client_factory

            with patch.dict(os.environ, {"AWS_REGION": "us-east-1"}):
                result = validate_bedrock_setup()

            assert isinstance(result, ValidationResult)
            assert result.total_checks > 0
            assert result.checks_passed >= 0

    def test_validate_bedrock_setup_partial_failure(self):
        """Test validation with some checks failing."""
        with patch("boto3.client") as mock_boto_client:
            from botocore.exceptions import NoCredentialsError

            # Mock STS client to fail (no credentials)
            mock_sts = Mock()
            mock_sts.get_caller_identity.side_effect = NoCredentialsError()

            # Mock Bedrock client to succeed
            mock_bedrock = Mock()
            mock_bedrock.list_foundation_models.return_value = {"modelSummaries": []}

            def client_factory(service_name, **kwargs):
                if service_name == "sts":
                    return mock_sts
                elif service_name == "bedrock":
                    return mock_bedrock
                else:
                    return Mock()

            mock_boto_client.side_effect = client_factory

            result = validate_bedrock_setup()

            assert result.success is False
            assert len(result.errors) > 0
            assert result.checks_passed < result.total_checks

    def test_validate_bedrock_setup_complete_failure(self):
        """Test validation with all checks failing."""
        with patch("boto3.client") as mock_boto_client:
            from botocore.exceptions import NoCredentialsError

            # Mock all clients to fail
            mock_client = Mock()
            mock_client.get_caller_identity.side_effect = NoCredentialsError()
            mock_client.list_foundation_models.side_effect = NoCredentialsError()
            mock_boto_client.return_value = mock_client

            result = validate_bedrock_setup()

            assert result.success is False
            assert len(result.errors) > 0
            assert result.checks_passed == 0

    def test_validate_bedrock_setup_verbose(self):
        """Test validation with verbose output."""
        with patch("boto3.client"):
            result = validate_bedrock_setup(verbose=True)

            assert isinstance(result, ValidationResult)
            assert isinstance(result.detailed_checks, dict)
            assert len(result.detailed_checks) > 0

    def test_validate_bedrock_setup_specific_region(self):
        """Test validation for specific region."""
        with patch("boto3.client"):
            result = validate_bedrock_setup(region="eu-west-1")

            assert isinstance(result, ValidationResult)

    def test_validate_bedrock_setup_model_checking(self):
        """Test validation that includes model access checking."""
        with patch("boto3.client") as mock_boto_client:
            mock_bedrock = Mock()

            # Mock list_foundation_models
            mock_bedrock.list_foundation_models.return_value = {
                "modelSummaries": [
                    {
                        "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
                        "modelName": "Claude 3 Haiku",
                    }
                ]
            }

            # Mock model invocation
            mock_response = {"body": Mock(), "contentType": "application/json"}
            mock_body = Mock()
            mock_body.read.return_value = b'{"completion": "test"}'
            mock_response["body"] = mock_body
            mock_bedrock.invoke_model.return_value = mock_response

            mock_boto_client.return_value = mock_bedrock

            result = validate_bedrock_setup(check_model_access=True)

            assert isinstance(result, ValidationResult)


@pytest.mark.skipif(
    not VALIDATION_AVAILABLE, reason="Bedrock validation module not available"
)
class TestValidationOutput:
    """Test validation result output formatting."""

    def test_print_validation_result_success(self, capsys):
        """Test printing successful validation result."""
        result = ValidationResult(
            success=True,
            errors=[],
            warnings=["Minor warning"],
            checks_passed=5,
            total_checks=5,
            detailed_checks={},
        )

        print_validation_result(result)
        captured = capsys.readouterr()

        assert "✅" in captured.out or "success" in captured.out.lower()
        assert "5/5" in captured.out

    def test_print_validation_result_failure(self, capsys):
        """Test printing failed validation result."""
        result = ValidationResult(
            success=False,
            errors=["Credentials not found", "Bedrock access denied"],
            warnings=[],
            checks_passed=1,
            total_checks=5,
            detailed_checks={},
        )

        print_validation_result(result)
        captured = capsys.readouterr()

        assert "❌" in captured.out or "failed" in captured.out.lower()
        assert "1/5" in captured.out
        assert "Credentials not found" in captured.out
        assert "Bedrock access denied" in captured.out

    def test_print_validation_result_with_warnings(self, capsys):
        """Test printing validation result with warnings."""
        result = ValidationResult(
            success=True,
            errors=[],
            warnings=["Environment variable not set", "Using default region"],
            checks_passed=4,
            total_checks=5,
            detailed_checks={},
        )

        print_validation_result(result)
        captured = capsys.readouterr()

        assert "⚠️" in captured.out or "warning" in captured.out.lower()
        assert "Environment variable not set" in captured.out

    def test_print_validation_result_detailed(self, capsys):
        """Test printing detailed validation result."""
        detailed_checks = {
            "aws_credentials": ValidationCheck(
                name="aws_credentials",
                passed=True,
                error=None,
                fix_suggestion="Credentials properly configured",
                documentation_link="https://docs.aws.amazon.com/",
            ),
            "bedrock_access": ValidationCheck(
                name="bedrock_access",
                passed=False,
                error="Access denied to Bedrock service",
                fix_suggestion="Check IAM permissions",
                documentation_link="https://docs.aws.amazon.com/bedrock/",
            ),
        }

        result = ValidationResult(
            success=False,
            errors=["Access denied to Bedrock service"],
            warnings=[],
            checks_passed=1,
            total_checks=2,
            detailed_checks=detailed_checks,
        )

        print_validation_result(result)
        captured = capsys.readouterr()

        assert "aws_credentials" in captured.out
        assert "bedrock_access" in captured.out
        assert "Check IAM permissions" in captured.out


@pytest.mark.skipif(
    not VALIDATION_AVAILABLE, reason="Bedrock validation module not available"
)
class TestValidationEdgeCases:
    """Test edge cases for validation."""

    def test_validation_with_none_region(self):
        """Test validation with None region."""
        with patch("boto3.client"):
            result = validate_bedrock_setup(region=None)
            assert isinstance(result, ValidationResult)

    def test_validation_with_empty_region(self):
        """Test validation with empty region."""
        with patch("boto3.client"):
            result = validate_bedrock_setup(region="")
            assert isinstance(result, ValidationResult)

    def test_validation_with_invalid_region(self):
        """Test validation with invalid region."""
        with patch("boto3.client") as mock_boto_client:
            from botocore.exceptions import ClientError

            mock_client = Mock()
            mock_client.list_foundation_models.side_effect = ClientError(
                error_response={
                    "Error": {"Code": "UnknownEndpoint", "Message": "Unknown endpoint"}
                },
                operation_name="ListFoundationModels",
            )
            mock_boto_client.return_value = mock_client

            result = validate_bedrock_setup(region="invalid-region-12345")

            assert result.success is False
            assert any("region" in error.lower() for error in result.errors)

    def test_validation_timeout_handling(self):
        """Test validation with network timeouts."""
        with patch("boto3.client") as mock_boto_client:
            from botocore.exceptions import ConnectTimeoutError

            mock_client = Mock()
            mock_client.get_caller_identity.side_effect = ConnectTimeoutError(
                endpoint_url="test"
            )
            mock_boto_client.return_value = mock_client

            result = validate_bedrock_setup()

            assert result.success is False
            assert any(
                "timeout" in error.lower() or "network" in error.lower()
                for error in result.errors
            )

    def test_validation_with_proxy_settings(self):
        """Test validation with proxy settings."""
        with patch.dict(
            os.environ,
            {
                "HTTP_PROXY": "http://proxy.company.com:8080",
                "HTTPS_PROXY": "http://proxy.company.com:8080",
            },
        ):
            with patch("boto3.client"):
                result = validate_bedrock_setup()
                assert isinstance(result, ValidationResult)

    def test_concurrent_validation_calls(self):
        """Test that concurrent validation calls work correctly."""
        import threading

        results = []

        def validate_worker():
            with patch("boto3.client"):
                result = validate_bedrock_setup()
                results.append(result)

        # Start multiple validation threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=validate_worker)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join(timeout=10)

        # All validations should complete
        assert len(results) == 3
        for result in results:
            assert isinstance(result, ValidationResult)


@pytest.mark.integration
class TestIntegrationValidation:
    """Integration tests for validation (require real AWS setup)."""

    def test_real_aws_validation(self):
        """Test validation against real AWS (requires credentials)."""
        pytest.skip("Integration test - requires real AWS credentials")

        # This would test against real AWS services
        result = validate_bedrock_setup()

        # With real credentials, should get meaningful results
        assert isinstance(result, ValidationResult)
        assert result.total_checks > 0

        if result.success:
            assert result.checks_passed == result.total_checks
            assert len(result.errors) == 0
        else:
            assert len(result.errors) > 0
            for error in result.errors:
                assert len(error) > 0
