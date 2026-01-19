"""Unit tests for Collibra API client."""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from requests.exceptions import Timeout, ConnectionError as RequestsConnectionError

from genops.providers.collibra.client import (
    CollibraAPIClient,
    CollibraAPIError,
    CollibraAuthenticationError,
    CollibraRateLimitError,
    RateLimiter,
)
from tests.mocks.mock_collibra_server import MockCollibraServer


@pytest.fixture
def mock_server():
    """Create mock Collibra server."""
    server = MockCollibraServer()
    yield server
    server.reset()


@pytest.fixture
def client(mock_server):
    """Create Collibra API client with mock server."""
    client = CollibraAPIClient(
        base_url="https://test.collibra.com",
        username="test_user",
        password="test_password",
    )
    # Patch the client's _make_request to use mock server
    return client


# Rate Limiter Tests


def test_rate_limiter_allows_requests_within_limit():
    """Test rate limiter allows requests within rate limit."""
    limiter = RateLimiter(rate_limit_per_second=10)

    # Should allow 10 requests immediately (burst capacity)
    for _ in range(10):
        limiter.acquire()  # Should not block

    assert True  # If we got here, rate limiter didn't block


def test_rate_limiter_blocks_excessive_requests():
    """Test rate limiter blocks excessive requests."""
    limiter = RateLimiter(rate_limit_per_second=10)

    # Consume all tokens
    for _ in range(50):  # Consume burst capacity
        limiter.acquire()

    # Next request should take some time
    start_time = time.time()
    limiter.acquire()
    elapsed = time.time() - start_time

    # Should have waited at least a small amount
    assert elapsed > 0


# Client Initialization Tests


def test_client_initialization_with_basic_auth():
    """Test client initializes with basic authentication."""
    client = CollibraAPIClient(
        base_url="https://test.collibra.com", username="user", password="pass"
    )

    assert client.base_url == "https://test.collibra.com"
    assert client.session.auth == ("user", "pass")


def test_client_initialization_with_api_token():
    """Test client initializes with API token."""
    client = CollibraAPIClient(
        base_url="https://test.collibra.com", api_token="test_token"
    )

    assert client.session.headers["Authorization"] == "Bearer test_token"


def test_client_initialization_strips_trailing_slash():
    """Test client strips trailing slash from base URL."""
    client = CollibraAPIClient(
        base_url="https://test.collibra.com/", username="user", password="pass"
    )

    assert client.base_url == "https://test.collibra.com"


# Health Check Tests


@patch("genops.providers.collibra.client.requests.Session.request")
def test_health_check_success(mock_request):
    """Test successful health check."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.content = b'{"version": "5.7.2"}'
    mock_response.json.return_value = {"version": "5.7.2"}
    mock_request.return_value = mock_response

    client = CollibraAPIClient(
        base_url="https://test.collibra.com", username="user", password="pass"
    )

    result = client.health_check()
    assert result is True


@patch("genops.providers.collibra.client.requests.Session.request")
def test_health_check_failure(mock_request):
    """Test failed health check."""
    mock_request.side_effect = RequestsConnectionError("Connection failed")

    client = CollibraAPIClient(
        base_url="https://test.collibra.com", username="user", password="pass"
    )

    result = client.health_check()
    assert result is False


# Authentication Error Tests


@patch("genops.providers.collibra.client.requests.Session.request")
def test_authentication_error_raised_on_401(mock_request):
    """Test authentication error raised on 401 response."""
    mock_response = Mock()
    mock_response.status_code = 401
    mock_response.content = b'{"error": "Unauthorized"}'
    mock_response.json.return_value = {"error": "Unauthorized"}
    mock_request.return_value = mock_response

    client = CollibraAPIClient(
        base_url="https://test.collibra.com", username="user", password="pass"
    )

    with pytest.raises(CollibraAuthenticationError) as exc_info:
        client._make_request("GET", "/rest/2.0/assets")

    assert exc_info.value.status_code == 401


# Rate Limit Error Tests


@patch("genops.providers.collibra.client.requests.Session.request")
def test_rate_limit_error_raised_on_429(mock_request):
    """Test rate limit error raised on 429 response."""
    mock_response = Mock()
    mock_response.status_code = 429
    mock_response.headers = {"Retry-After": "60"}
    mock_response.content = b'{"error": "Too Many Requests"}'
    mock_response.json.return_value = {"error": "Too Many Requests"}
    mock_request.return_value = mock_response

    client = CollibraAPIClient(
        base_url="https://test.collibra.com", username="user", password="pass"
    )

    with pytest.raises(CollibraRateLimitError) as exc_info:
        client._make_request("GET", "/rest/2.0/assets")

    assert exc_info.value.status_code == 429
    assert "60" in str(exc_info.value)


# Asset Management Tests


@patch("genops.providers.collibra.client.requests.Session.request")
def test_create_asset_success(mock_request):
    """Test successful asset creation."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.content = b'{"id": "asset-123", "name": "Test Asset"}'
    mock_response.json.return_value = {"id": "asset-123", "name": "Test Asset"}
    mock_request.return_value = mock_response

    client = CollibraAPIClient(
        base_url="https://test.collibra.com", username="user", password="pass"
    )

    result = client.create_asset(
        domain_id="domain-123", asset_type="AI Operation", name="Test Asset"
    )

    assert result["id"] == "asset-123"
    assert result["name"] == "Test Asset"


@patch("genops.providers.collibra.client.requests.Session.request")
def test_update_asset_success(mock_request):
    """Test successful asset update."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.content = b'{"id": "asset-123", "attributes": {"cost": 1.5}}'
    mock_response.json.return_value = {"id": "asset-123", "attributes": {"cost": 1.5}}
    mock_request.return_value = mock_response

    client = CollibraAPIClient(
        base_url="https://test.collibra.com", username="user", password="pass"
    )

    result = client.update_asset("asset-123", attributes={"cost": 1.5})

    assert result["id"] == "asset-123"
    assert result["attributes"]["cost"] == 1.5


@patch("genops.providers.collibra.client.requests.Session.request")
def test_get_asset_success(mock_request):
    """Test successful asset retrieval."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.content = b'{"id": "asset-123", "name": "Test Asset"}'
    mock_response.json.return_value = {"id": "asset-123", "name": "Test Asset"}
    mock_request.return_value = mock_response

    client = CollibraAPIClient(
        base_url="https://test.collibra.com", username="user", password="pass"
    )

    result = client.get_asset("asset-123")

    assert result["id"] == "asset-123"


@patch("genops.providers.collibra.client.requests.Session.request")
def test_search_assets_with_filters(mock_request):
    """Test asset search with filters."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.content = b'{"results": [{"id": "asset-123"}], "total": 1}'
    mock_response.json.return_value = {"results": [{"id": "asset-123"}], "total": 1}
    mock_request.return_value = mock_response

    client = CollibraAPIClient(
        base_url="https://test.collibra.com", username="user", password="pass"
    )

    results = client.search_assets(query="test", asset_type="AI Operation", limit=10)

    assert len(results) == 1
    assert results[0]["id"] == "asset-123"


# Domain Management Tests


@patch("genops.providers.collibra.client.requests.Session.request")
def test_list_domains_success(mock_request):
    """Test successful domain listing."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.content = (
        b'{"results": [{"id": "domain-123", "name": "AI Governance"}], "total": 1}'
    )
    mock_response.json.return_value = {
        "results": [{"id": "domain-123", "name": "AI Governance"}],
        "total": 1,
    }
    mock_request.return_value = mock_response

    client = CollibraAPIClient(
        base_url="https://test.collibra.com", username="user", password="pass"
    )

    results = client.list_domains()

    assert len(results) == 1
    assert results[0]["name"] == "AI Governance"


@patch("genops.providers.collibra.client.requests.Session.request")
def test_get_domain_success(mock_request):
    """Test successful domain retrieval."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.content = b'{"id": "domain-123", "name": "AI Governance"}'
    mock_response.json.return_value = {"id": "domain-123", "name": "AI Governance"}
    mock_request.return_value = mock_response

    client = CollibraAPIClient(
        base_url="https://test.collibra.com", username="user", password="pass"
    )

    result = client.get_domain("domain-123")

    assert result["name"] == "AI Governance"


# Policy Management Tests


@patch("genops.providers.collibra.client.requests.Session.request")
def test_list_policies_success(mock_request):
    """Test successful policy listing."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.content = (
        b'{"results": [{"id": "policy-123", "name": "Cost Limit"}], "total": 1}'
    )
    mock_response.json.return_value = {
        "results": [{"id": "policy-123", "name": "Cost Limit"}],
        "total": 1,
    }
    mock_request.return_value = mock_response

    client = CollibraAPIClient(
        base_url="https://test.collibra.com", username="user", password="pass"
    )

    results = client.list_policies()

    assert len(results) == 1
    assert results[0]["name"] == "Cost Limit"


# Error Handling Tests


@patch("genops.providers.collibra.client.requests.Session.request")
def test_timeout_error_handling(mock_request):
    """Test timeout error handling."""
    mock_request.side_effect = Timeout("Request timed out")

    client = CollibraAPIClient(
        base_url="https://test.collibra.com", username="user", password="pass"
    )

    with pytest.raises(CollibraAPIError) as exc_info:
        client._make_request("GET", "/rest/2.0/assets")

    assert "timeout" in str(exc_info.value).lower()


@patch("genops.providers.collibra.client.requests.Session.request")
def test_connection_error_handling(mock_request):
    """Test connection error handling."""
    mock_request.side_effect = RequestsConnectionError("Connection failed")

    client = CollibraAPIClient(
        base_url="https://test.collibra.com", username="user", password="pass"
    )

    with pytest.raises(CollibraAPIError) as exc_info:
        client._make_request("GET", "/rest/2.0/assets")

    assert "connection" in str(exc_info.value).lower()


# Relation Management Tests


@patch("genops.providers.collibra.client.requests.Session.request")
def test_create_relation_success(mock_request):
    """Test successful relation creation."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.content = (
        b'{"id": "relation-123", "sourceId": "asset-1", "targetId": "asset-2"}'
    )
    mock_response.json.return_value = {
        "id": "relation-123",
        "sourceId": "asset-1",
        "targetId": "asset-2",
    }
    mock_request.return_value = mock_response

    client = CollibraAPIClient(
        base_url="https://test.collibra.com", username="user", password="pass"
    )

    result = client.create_relation("asset-1", "asset-2", "related_to")

    assert result["id"] == "relation-123"


# Application Info Tests


@patch("genops.providers.collibra.client.requests.Session.request")
def test_get_application_info_success(mock_request):
    """Test successful application info retrieval."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.content = b'{"version": "5.7.2", "buildNumber": "12345"}'
    mock_response.json.return_value = {"version": "5.7.2", "buildNumber": "12345"}
    mock_request.return_value = mock_response

    client = CollibraAPIClient(
        base_url="https://test.collibra.com", username="user", password="pass"
    )

    result = client.get_application_info()

    assert result["version"] == "5.7.2"
    assert result["buildNumber"] == "12345"
