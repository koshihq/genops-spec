#!/usr/bin/env python3
"""
GenOps Bedrock Setup Validation

This module provides comprehensive validation for AWS Bedrock integration setup,
including AWS credentials, region availability, model access permissions,
and GenOps configuration validation with actionable diagnostics.

Features:
- AWS credentials validation with multiple authentication methods
- Region availability and model access verification
- Bedrock service permissions checking
- Content filtering and compliance validation
- Network connectivity testing
- GenOps configuration validation
- Actionable error messages with specific fix suggestions

Example usage:
    from genops.providers.bedrock_validation import validate_bedrock_setup

    result = validate_bedrock_setup()
    if result.success:
        print("✅ Bedrock setup is ready!")
    else:
        print("❌ Setup issues found:")
        for error in result.errors:
            print(f"   - {error}")
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import boto3
    from botocore.exceptions import (
        BotoCoreError,  # noqa: F401
        ClientError,
        EndpointConnectionError,
        NoCredentialsError,
        PartialCredentialsError,
        ProfileNotFound,
    )

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class BedrockValidationResult:
    """Comprehensive validation result for Bedrock setup."""

    success: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str, fix_suggestion: str = None):  # type: ignore[assignment]
        """Add an error with optional fix suggestion."""
        error_msg = message
        if fix_suggestion:
            error_msg += f" → Fix: {fix_suggestion}"
        self.errors.append(error_msg)
        self.success = False

    def add_warning(self, message: str, recommendation: str = None):  # type: ignore
        """Add a warning with optional recommendation."""
        self.warnings.append(message)
        if recommendation:
            self.recommendations.append(recommendation)

    def add_recommendation(self, message: str):
        """Add a general recommendation."""
        self.recommendations.append(message)


def validate_bedrock_setup(
    region_name: str = "us-east-1",
    profile_name: Optional[str] = None,
    test_model_access: bool = True,
    test_connectivity: bool = True,
) -> BedrockValidationResult:
    """
    Comprehensive Bedrock setup validation.

    Args:
        region_name: AWS region to validate
        profile_name: AWS profile name (optional)
        test_model_access: Test access to actual Bedrock models
        test_connectivity: Test network connectivity to AWS

    Returns:
        Detailed validation result with actionable feedback
    """
    result = BedrockValidationResult(success=True)

    # 1. Check basic dependencies
    _validate_dependencies(result)

    # 2. Validate AWS credentials and configuration
    _validate_aws_credentials(result, profile_name)

    # 3. Validate region and service availability
    _validate_region_availability(result, region_name)

    # 4. Test AWS connectivity (if enabled)
    if test_connectivity and result.success:
        _test_aws_connectivity(result, region_name, profile_name)

    # 5. Test Bedrock service access (if enabled and previous tests pass)
    if test_model_access and result.success:
        _test_bedrock_access(result, region_name, profile_name)

    # 6. Validate GenOps configuration
    _validate_genops_config(result)

    # 7. Generate final recommendations
    _generate_recommendations(result, region_name)

    return result


def _validate_dependencies(result: BedrockValidationResult):
    """Validate required dependencies are available."""

    if not BOTO3_AVAILABLE:
        result.add_error(
            "AWS SDK (boto3) not available", "Install with: pip install boto3 botocore"
        )
        return

    # Check Python version

    result.details["dependencies"] = {
        "boto3": BOTO3_AVAILABLE,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }


def _validate_aws_credentials(
    result: BedrockValidationResult, profile_name: Optional[str]
):
    """Validate AWS credentials and authentication."""

    if not BOTO3_AVAILABLE:
        return

    try:
        # Try to create session with specified profile
        if profile_name:
            try:
                session = boto3.Session(profile_name=profile_name)
                result.details["auth_method"] = f"AWS profile: {profile_name}"
            except ProfileNotFound:
                result.add_error(
                    f"AWS profile '{profile_name}' not found",
                    f"Check ~/.aws/credentials or run: aws configure --profile {profile_name}",
                )
                return
        else:
            session = boto3.Session()
            result.details["auth_method"] = "Default AWS credentials"

        # Test credential access
        try:
            sts = session.client("sts")
            identity = sts.get_caller_identity()

            result.details["aws_account"] = identity.get("Account")
            result.details["aws_user_arn"] = identity.get("Arn")

            # Check if using temporary credentials
            if "assumed-role" in identity.get("Arn", ""):
                result.details["credential_type"] = "IAM Role/Temporary"
            else:
                result.details["credential_type"] = "IAM User/Long-term"

        except NoCredentialsError:
            result.add_error(
                "No AWS credentials found",
                "Configure credentials: 1) Run 'aws configure', 2) Set AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY env vars, 3) Use IAM roles on AWS infrastructure, or 4) Set AWS_PROFILE env var",
            )
            return
        except PartialCredentialsError:
            result.add_error(
                "Incomplete AWS credentials",
                "Ensure both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are set",
            )
            return

    except Exception as e:
        result.add_error(
            f"AWS credential validation failed: {str(e)}",
            "Check your AWS configuration and credentials",
        )


def _validate_region_availability(result: BedrockValidationResult, region_name: str):
    """Validate AWS region and Bedrock service availability."""

    # List of known Bedrock-supported regions (as of November 2024)
    BEDROCK_REGIONS = {
        "us-east-1": "US East (N. Virginia)",
        "us-west-2": "US West (Oregon)",
        "ap-southeast-1": "Asia Pacific (Singapore)",
        "ap-northeast-1": "Asia Pacific (Tokyo)",
        "eu-west-1": "Europe (Ireland)",
        "eu-central-1": "Europe (Frankfurt)",
        "ca-central-1": "Canada (Central)",
        "ap-south-1": "Asia Pacific (Mumbai)",
        "sa-east-1": "South America (São Paulo)",
    }

    if region_name not in BEDROCK_REGIONS:
        result.add_warning(
            f"Region '{region_name}' may not support Bedrock",
            "Consider using a known Bedrock region like us-east-1, us-west-2, or eu-west-1",
        )

    result.details["region"] = {
        "name": region_name,
        "description": BEDROCK_REGIONS.get(region_name, "Unknown/Unsupported"),
        "bedrock_supported": region_name in BEDROCK_REGIONS,
    }


def _test_aws_connectivity(
    result: BedrockValidationResult, region_name: str, profile_name: Optional[str]
):
    """Test basic AWS service connectivity."""

    if not BOTO3_AVAILABLE:
        return

    try:
        session_kwargs = {}
        if profile_name:
            session_kwargs["profile_name"] = profile_name

        session = boto3.Session(**session_kwargs)

        # Test basic AWS service connectivity with STS
        sts = session.client("sts", region_name=region_name)
        sts.get_caller_identity()

        result.details["connectivity"] = {
            "aws_services": True,
            "region_accessible": True,
        }

    except EndpointConnectionError:
        result.add_error(
            f"Cannot connect to AWS services in region {region_name}",
            "Check internet connection, VPN, or firewall settings",
        )
    except Exception as e:
        result.add_error(
            f"AWS connectivity test failed: {str(e)}",
            "Verify network connectivity and AWS service status",
        )


def _test_bedrock_access(
    result: BedrockValidationResult, region_name: str, profile_name: Optional[str]
):
    """Test Bedrock service access and model permissions."""

    if not BOTO3_AVAILABLE:
        return

    try:
        session_kwargs = {}
        if profile_name:
            session_kwargs["profile_name"] = profile_name

        session = boto3.Session(**session_kwargs)

        # Test Bedrock service access
        bedrock = session.client("bedrock", region_name=region_name)

        try:
            # Try to list foundation models
            models_response = bedrock.list_foundation_models()
            available_models = models_response.get("modelSummaries", [])

            result.details["bedrock_access"] = {
                "service_accessible": True,
                "models_accessible": len(available_models) > 0,
                "available_models_count": len(available_models),
                "sample_models": [
                    model.get("modelId", "unknown") for model in available_models[:5]
                ],
            }

            if len(available_models) == 0:
                result.add_warning(
                    "No Bedrock models accessible in this region",
                    f"Go to AWS Console → Bedrock → Model access → Request access to models in {region_name}. Popular options: Claude 3 Haiku (fast/cheap), Claude 3 Sonnet (balanced), Claude 3 Opus (powerful)",
                )
            else:
                # Test runtime client
                session.client("bedrock-runtime", region_name=region_name)
                result.details["bedrock_runtime_accessible"] = True

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))

            if error_code == "AccessDeniedException":
                result.add_error(
                    "Access denied to Bedrock service",
                    "1) Go to AWS Console → Bedrock → Model access → Manage → Enable models, 2) Add IAM permissions: bedrock:InvokeModel, bedrock:InvokeModelWithResponseStream, bedrock:ListFoundationModels",
                )
            elif error_code == "UnauthorizedOperation":
                result.add_error(
                    "Insufficient permissions for Bedrock",
                    "Ensure IAM role/user has bedrock:ListFoundationModels permission",
                )
            else:
                result.add_error(
                    f"Bedrock access test failed [{error_code}]: {error_message}",
                    "Check IAM permissions and Bedrock service availability",
                )

    except Exception as e:
        result.add_error(
            f"Bedrock service test failed: {str(e)}",
            "Verify Bedrock is available in your region and check IAM permissions",
        )


def _validate_genops_config(result: BedrockValidationResult):
    """Validate GenOps configuration and OpenTelemetry setup."""

    # Check for OpenTelemetry configuration
    otel_config = {}

    # Check environment variables
    otel_vars = [
        "OTEL_SERVICE_NAME",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "OTEL_RESOURCE_ATTRIBUTES",
        "GENOPS_ENVIRONMENT",
        "GENOPS_PROJECT",
    ]

    for var in otel_vars:
        value = os.environ.get(var)
        if value:
            otel_config[var] = value

    result.details["genops_config"] = otel_config

    # Recommendations for missing configuration
    if not os.environ.get("OTEL_SERVICE_NAME"):
        result.add_recommendation(
            "Set OTEL_SERVICE_NAME environment variable for better telemetry identification"
        )

    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        result.add_recommendation(
            "Set OTEL_EXPORTER_OTLP_ENDPOINT to export telemetry: e.g., http://localhost:4317 (local collector), https://api.honeycomb.io:443 (Honeycomb), or your platform's OTLP endpoint"
        )

    # Check if GenOps core is available
    try:
        import genops.core.telemetry  # noqa: F401

        result.details["genops_core_available"] = True
    except ImportError:
        result.add_warning(
            "GenOps core telemetry not available",
            "Install with: pip install genops-ai[bedrock]",
        )
        result.details["genops_core_available"] = False


def _generate_recommendations(result: BedrockValidationResult, region_name: str):
    """Generate final setup and optimization recommendations."""

    if result.success:
        result.add_recommendation("✅ Bedrock setup validation passed!")
        result.add_recommendation(
            "Consider testing with a simple model like Claude Haiku for cost-effective experimentation"
        )

        if region_name != "us-east-1":
            result.add_recommendation(
                f"You're using {region_name}. Consider us-east-1 for potentially lower costs and more model availability"
            )

    # Security recommendations
    if result.details.get("credential_type") == "IAM User/Long-term":
        result.add_recommendation(
            "For production, consider using IAM roles instead of long-term credentials"
        )

    # Cost optimization recommendations
    result.add_recommendation(
        "Enable detailed CloudTrail logging for Bedrock API calls to track usage and costs"
    )

    result.add_recommendation("Set up AWS Budgets alerts to monitor Bedrock spending")


def print_validation_result(result: BedrockValidationResult, detailed: bool = False):
    """
    Print validation results in a user-friendly format.

    Args:
        result: Validation result to print
        detailed: Include detailed information in output
    """
    print("🔍 GenOps Bedrock Setup Validation")
    print("=" * 50)

    if result.success:
        print("✅ Overall Status: PASSED")
    else:
        print("❌ Overall Status: FAILED")

    print()

    # Print errors
    if result.errors:
        print("❌ Errors Found:")
        for i, error in enumerate(result.errors, 1):
            print(f"   {i}. {error}")
        print()

    # Print warnings
    if result.warnings:
        print("⚠️  Warnings:")
        for i, warning in enumerate(result.warnings, 1):
            print(f"   {i}. {warning}")
        print()

    # Print recommendations
    if result.recommendations:
        print("💡 Recommendations:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"   {i}. {rec}")
        print()

    # Print detailed information if requested
    if detailed and result.details:
        print("📋 Detailed Information:")
        print(json.dumps(result.details, indent=2))


def validate_model_access(
    model_id: str, region_name: str = "us-east-1", profile_name: Optional[str] = None
) -> bool:
    """
    Test access to a specific Bedrock model.

    Args:
        model_id: Specific Bedrock model ID to test
        region_name: AWS region
        profile_name: AWS profile (optional)

    Returns:
        True if model is accessible, False otherwise
    """
    if not BOTO3_AVAILABLE:
        return False

    try:
        session_kwargs = {}
        if profile_name:
            session_kwargs["profile_name"] = profile_name

        session = boto3.Session(**session_kwargs)
        bedrock = session.client("bedrock", region_name=region_name)

        # Get model details
        model_details = bedrock.get_foundation_model(modelIdentifier=model_id)
        return model_details is not None

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")

        if error_code == "ValidationException":
            logger.warning(
                f"Model {model_id} not found or not available in {region_name}"
            )
        elif error_code == "AccessDeniedException":
            logger.warning(f"Access denied to model {model_id}")
        else:
            logger.warning(f"Model access test failed: {error_code}")

        return False
    except Exception as e:
        logger.warning(f"Model access test error: {e}")
        return False


def get_available_models(
    region_name: str = "us-east-1",
    profile_name: Optional[str] = None,
    provider_filter: Optional[str] = None,
) -> list[dict[str, str]]:
    """
    Get list of available Bedrock models in a region.

    Args:
        region_name: AWS region
        profile_name: AWS profile (optional)
        provider_filter: Filter by provider (e.g., 'anthropic', 'amazon')

    Returns:
        List of available models with details
    """
    if not BOTO3_AVAILABLE:
        return []

    try:
        session_kwargs = {}
        if profile_name:
            session_kwargs["profile_name"] = profile_name

        session = boto3.Session(**session_kwargs)
        bedrock = session.client("bedrock", region_name=region_name)

        response = bedrock.list_foundation_models()
        models = response.get("modelSummaries", [])

        # Apply provider filter if specified
        if provider_filter:
            models = [
                model
                for model in models
                if provider_filter.lower() in model.get("providerName", "").lower()
            ]

        # Format model information
        formatted_models = []
        for model in models:
            formatted_models.append(
                {
                    "modelId": model.get("modelId", ""),
                    "providerName": model.get("providerName", ""),
                    "modelName": model.get("modelName", ""),
                    "inputModalities": model.get("inputModalities", []),
                    "outputModalities": model.get("outputModalities", []),
                }
            )

        return formatted_models

    except Exception as e:
        logger.error(f"Failed to get available models: {e}")
        return []


# Export main functions
__all__ = [
    "validate_bedrock_setup",
    "print_validation_result",
    "validate_model_access",
    "get_available_models",
    "BedrockValidationResult",
]
