"""
Validation utilities for GenOps Elasticsearch integration.

Provides comprehensive setup validation with actionable error messages
and recommendations to ensure smooth developer onboarding.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

from .client import (
    ELASTICSEARCH_AVAILABLE,
    ElasticAPIClient,
    ElasticAuthenticationError,
    ElasticConnectionError,
)

logger = logging.getLogger(__name__)


@dataclass
class ElasticValidationResult:
    """
    Results from Elasticsearch setup validation.

    Provides structured feedback with errors, warnings, and recommendations
    to help developers quickly resolve configuration issues.
    """

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    # Connection details
    connectivity: bool = False
    cluster_version: Optional[str] = None
    cluster_name: Optional[str] = None
    index_write_permission: bool = False
    ilm_supported: bool = False

    def add_error(self, message: str):
        """Add an error message."""
        self.errors.append(message)
        self.valid = False

    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)

    def add_recommendation(self, message: str):
        """Add a recommendation."""
        self.recommendations.append(message)


def validate_setup(
    elastic_url: Optional[str] = None,
    cloud_id: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    api_key: Optional[str] = None,
    api_id: Optional[str] = None,
    verify_certs: bool = True,
    test_index_write: bool = True,
) -> ElasticValidationResult:
    """
    Comprehensive validation of Elasticsearch setup.

    Checks:
    1. Environment variables
    2. URL format and accessibility
    3. Authentication configuration
    4. Cluster connectivity
    5. Version compatibility (ES 8.x or 9.x)
    6. Index write permissions
    7. ILM support

    Args:
        elastic_url: Elasticsearch cluster URL (or use ELASTIC_URL env var)
        cloud_id: Elastic Cloud deployment ID (or use ELASTIC_CLOUD_ID env var)
        username: Basic auth username (or use ELASTIC_USERNAME env var)
        password: Basic auth password (or use ELASTIC_PASSWORD env var)
        api_key: API key (or use ELASTIC_API_KEY env var)
        api_id: API key ID (or use ELASTIC_API_ID env var)
        verify_certs: Verify SSL certificates
        test_index_write: Test index write permission (creates temporary index)

    Returns:
        ElasticValidationResult with detailed feedback
    """
    result = ElasticValidationResult(valid=True)

    # Check if elasticsearch package is available
    if not ELASTICSEARCH_AVAILABLE:
        result.add_error(
            "elasticsearch package not installed. "
            "Install with: pip install 'genops-ai[elastic]' or pip install elasticsearch>=8.0.0"
        )
        return result

    # Environment variable fallbacks
    elastic_url = elastic_url or os.getenv("ELASTIC_URL")
    cloud_id = cloud_id or os.getenv("ELASTIC_CLOUD_ID")
    username = username or os.getenv("ELASTIC_USERNAME")
    password = password or os.getenv("ELASTIC_PASSWORD")
    api_key = api_key or os.getenv("ELASTIC_API_KEY")
    api_id = api_id or os.getenv("ELASTIC_API_ID")

    # 1. Validate connection configuration
    if not elastic_url and not cloud_id:
        result.add_error(
            "No Elasticsearch connection configured. "
            "Set ELASTIC_URL or ELASTIC_CLOUD_ID environment variable."
        )
        result.add_recommendation(
            "For local development: export ELASTIC_URL=http://localhost:9200"
        )
        result.add_recommendation(
            "For Elastic Cloud: export ELASTIC_CLOUD_ID=<your-cloud-id>"
        )
        return result

    # 2. Validate URL format
    if elastic_url:
        validation = _validate_url(elastic_url)
        if not validation["valid"]:
            result.add_error(f"Invalid Elasticsearch URL: {validation['error']}")
            result.add_recommendation(
                "URL should be in format: http://localhost:9200 or https://es.example.com:9200"
            )
        elif validation.get("insecure"):
            result.add_warning(
                "Using HTTP (not HTTPS) connection. This is insecure for production."
            )
            result.add_recommendation(
                "Use HTTPS in production: https://your-cluster:9200"
            )

    # 3. Validate authentication
    auth_validation = _validate_authentication(
        username, password, api_key, api_id, cloud_id
    )
    if not auth_validation["valid"]:
        result.add_error(auth_validation["error"])
        for rec in auth_validation.get("recommendations", []):
            result.add_recommendation(rec)
    elif auth_validation.get("warnings"):
        for warning in auth_validation["warnings"]:
            result.add_warning(warning)

    # If basic validation failed, return early
    if not result.valid:
        return result

    # 4. Test connectivity and permissions
    try:
        client = ElasticAPIClient(
            elastic_url=elastic_url,
            cloud_id=cloud_id,
            username=username,
            password=password,
            api_key=api_key,
            api_id=api_id,
            verify_certs=verify_certs,
        )

        # Health check
        try:
            result.connectivity = client.health_check()
            if not result.connectivity:
                result.add_error("Elasticsearch cluster is unhealthy (status: red)")
                result.add_recommendation("Check cluster health: GET /_cluster/health")
        except ElasticAuthenticationError as e:
            result.add_error(f"Authentication failed: {e}")
            result.add_recommendation(
                "Verify credentials with: curl -u username:password <elastic_url>"
            )
            return result
        except ElasticConnectionError as e:
            result.add_error(f"Connection failed: {e}")
            result.add_recommendation("Verify Elasticsearch is running and accessible")
            return result

        # Get cluster info
        try:
            info = client.get_cluster_info()
            result.cluster_name = info.get("cluster_name")
            result.cluster_version = client.get_version()

            # Validate version (ES 8.x or 9.x)
            if result.cluster_version:
                major_version = int(result.cluster_version.split(".")[0])
                if major_version < 8:
                    result.add_warning(
                        f"Elasticsearch {result.cluster_version} detected. "
                        "GenOps recommends ES 8.x or newer for optimal compatibility."
                    )
                    result.add_recommendation(
                        "Consider upgrading to Elasticsearch 8.x or 9.x"
                    )
                elif major_version >= 8:
                    result.add_recommendation(
                        f"✓ Elasticsearch {result.cluster_version} is compatible"
                    )

        except Exception as e:
            result.add_warning(f"Could not retrieve cluster info: {e}")

        # Test index write permission
        if test_index_write:
            test_index = f"genops-validation-test-{int(os.urandom(4).hex(), 16)}"
            try:
                # Create test index
                client.create_index(test_index)
                result.index_write_permission = True

                # Clean up test index
                try:
                    client.client.indices.delete(index=test_index)
                except Exception:
                    pass  # Best effort cleanup

            except Exception as e:
                result.add_error(f"Index write permission test failed: {e}")
                result.add_recommendation(
                    "Ensure user has 'create_index' and 'write' permissions"
                )

        # Check ILM support
        try:
            client.client.ilm.get_lifecycle()
            result.ilm_supported = True
        except Exception:
            result.add_warning("ILM (Index Lifecycle Management) not available")
            result.add_recommendation(
                "ILM requires Elasticsearch 6.6+ with appropriate license"
            )

        # Close client
        client.close()

    except Exception as e:
        result.add_error(f"Validation failed: {e}")

    return result


def _validate_url(url: str) -> dict:
    """Validate Elasticsearch URL format."""
    try:
        parsed = urlparse(url)

        if not parsed.scheme:
            return {
                "valid": False,
                "error": "URL must include scheme (http:// or https://)",
            }

        if parsed.scheme not in ["http", "https"]:
            return {
                "valid": False,
                "error": f"Invalid scheme '{parsed.scheme}'. Use 'http' or 'https'",
            }

        if not parsed.netloc:
            return {"valid": False, "error": "URL must include hostname"}

        return {"valid": True, "insecure": (parsed.scheme == "http")}

    except Exception as e:
        return {"valid": False, "error": str(e)}


def _validate_authentication(
    username: Optional[str],
    password: Optional[str],
    api_key: Optional[str],
    api_id: Optional[str],
    cloud_id: Optional[str],
) -> dict:
    """Validate authentication configuration."""
    # Check for authentication credentials
    has_basic_auth = username and password
    has_api_key = api_key

    if not has_basic_auth and not has_api_key:
        # No authentication - only OK for local development
        if cloud_id:
            return {
                "valid": False,
                "error": "Elastic Cloud requires authentication (API key or basic auth)",
                "recommendations": [
                    "Set ELASTIC_API_KEY for API key authentication (recommended)",
                    "Or set ELASTIC_USERNAME and ELASTIC_PASSWORD for basic auth",
                ],
            }
        else:
            # Local development - authentication optional
            return {
                "valid": True,
                "warnings": [
                    "No authentication configured. This is only acceptable for local development."
                ],
                "recommendations": [
                    "Use API key authentication in production: export ELASTIC_API_KEY=<your-key>"
                ],
            }

    # Validate basic auth
    if has_basic_auth:
        if not username:
            return {
                "valid": False,
                "error": "ELASTIC_PASSWORD provided but ELASTIC_USERNAME is missing",
            }
        if not password:
            return {
                "valid": False,
                "error": "ELASTIC_USERNAME provided but ELASTIC_PASSWORD is missing",
            }

        return {
            "valid": True,
            "warnings": [
                "Using basic authentication. Consider using API key authentication for better security."
            ],
            "recommendations": [
                "API keys provide better security and granular permissions"
            ],
        }

    # API key authentication
    if has_api_key:
        return {
            "valid": True,
            "recommendations": ["✓ Using API key authentication (recommended)"],
        }

    return {"valid": True}


def print_validation_result(result: ElasticValidationResult):
    """
    Print validation result with user-friendly formatting.

    Args:
        result: ElasticValidationResult to display
    """
    print("\n" + "=" * 70)
    print("GenOps Elasticsearch Setup Validation")
    print("=" * 70)

    # Overall status
    if result.valid:
        print("\n✅ Validation PASSED")
    else:
        print("\n❌ Validation FAILED")

    # Connection info
    if result.cluster_name or result.cluster_version:
        print("\n📊 Cluster Information:")
        if result.cluster_name:
            print(f"   • Cluster Name: {result.cluster_name}")
        if result.cluster_version:
            print(f"   • Version: {result.cluster_version}")

    # Connectivity
    print(
        f"\n🔌 Connectivity: {'✅ Connected' if result.connectivity else '❌ Failed'}"
    )

    # Permissions
    if result.index_write_permission:
        print("🔑 Permissions: ✅ Write access verified")
    elif not result.valid:
        print("🔑 Permissions: ⚠️  Could not verify (connection failed)")

    # ILM
    if result.ilm_supported:
        print("⏱️  ILM Support: ✅ Available")
    elif result.connectivity:
        print("⏱️  ILM Support: ⚠️  Not available")

    # Errors
    if result.errors:
        print("\n❌ Errors:")
        for error in result.errors:
            print(f"   • {error}")

    # Warnings
    if result.warnings:
        print("\n⚠️  Warnings:")
        for warning in result.warnings:
            print(f"   • {warning}")

    # Recommendations
    if result.recommendations:
        print("\n💡 Recommendations:")
        for rec in result.recommendations:
            print(f"   • {rec}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    """
    Run validation from command line:
    python -m genops.providers.elastic.validation
    """
    result = validate_setup()
    print_validation_result(result)

    # Exit with appropriate code
    exit(0 if result.valid else 1)
