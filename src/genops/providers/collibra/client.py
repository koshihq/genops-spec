"""Collibra REST API client for GenOps integration."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


@dataclass
class CollibraAsset:
    """Collibra asset structure."""

    asset_id: Optional[str] = None
    domain_id: str = ""
    asset_type: str = ""
    name: str = ""
    display_name: Optional[str] = None
    attributes: Dict[str, Any] = None
    status: Optional[str] = None

    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}


@dataclass
class CollibraPolicy:
    """Collibra policy structure."""

    policy_id: str
    name: str
    description: str = ""
    enabled: bool = True
    enforcement_level: str = "block"
    conditions: Dict[str, Any] = None
    asset_types: List[str] = None
    tags: List[str] = None

    def __post_init__(self):
        if self.conditions is None:
            self.conditions = {}
        if self.asset_types is None:
            self.asset_types = []
        if self.tags is None:
            self.tags = []


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, rate_limit_per_second: int = 10):
        self.rate_limit = rate_limit_per_second
        self.tokens = rate_limit_per_second
        self.last_update = time.time()
        self.max_tokens = rate_limit_per_second * 5  # Burst capacity

    def acquire(self) -> None:
        """Acquire a token, blocking if necessary."""
        while True:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(
                self.max_tokens, self.tokens + elapsed * self.rate_limit
            )
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return

            # Wait until next token available
            sleep_time = (1 - self.tokens) / self.rate_limit
            time.sleep(sleep_time)


class CollibraAPIError(Exception):
    """Base exception for Collibra API errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response: Optional[Dict] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(message)


class CollibraAuthenticationError(CollibraAPIError):
    """Authentication failed."""

    pass


class CollibraRateLimitError(CollibraAPIError):
    """Rate limit exceeded."""

    pass


class CollibraAPIClient:
    """REST API client for Collibra Data Governance Center."""

    def __init__(
        self,
        base_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_token: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        rate_limit_per_second: int = 10,
        verify_ssl: bool = True,
    ):
        """
        Initialize Collibra API client.

        Args:
            base_url: Collibra instance URL (e.g., https://company.collibra.com)
            username: Basic auth username
            password: Basic auth password
            api_token: API token (alternative to username/password)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            rate_limit_per_second: API rate limit (requests per second)
            verify_ssl: Verify SSL certificates
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.rate_limiter = RateLimiter(rate_limit_per_second)

        # Configure session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,  # 1s, 2s, 4s, 8s, 16s
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Configure authentication
        if api_token:
            self.session.headers["Authorization"] = f"Bearer {api_token}"
        elif username and password:
            self.session.auth = (username, password)
        else:
            logger.warning(
                "No authentication credentials provided. "
                "API calls may fail if authentication is required."
            )

        # Default headers
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "GenOps-Collibra-Integration/1.0",
            }
        )

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict:
        """
        Make HTTP request to Collibra API with rate limiting and error handling.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            data: Request body data
            params: Query parameters

        Returns:
            Response JSON data

        Raises:
            CollibraAuthenticationError: Authentication failed
            CollibraRateLimitError: Rate limit exceeded
            CollibraAPIError: Other API errors
        """
        # Apply rate limiting
        self.rate_limiter.acquire()

        url = urljoin(self.base_url, endpoint)

        try:
            response = self.session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                timeout=self.timeout,
                verify=self.verify_ssl,
            )

            # Handle authentication errors
            if response.status_code == 401:
                raise CollibraAuthenticationError(
                    "Authentication failed. Check credentials.",
                    status_code=401,
                    response=response.json() if response.content else None,
                )

            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", 60))
                raise CollibraRateLimitError(
                    f"Rate limit exceeded. Retry after {retry_after} seconds.",
                    status_code=429,
                    response={"retry_after": retry_after},
                )

            # Raise for other error status codes
            response.raise_for_status()

            # Return JSON response or empty dict
            return response.json() if response.content else {}

        except requests.exceptions.Timeout as e:
            raise CollibraAPIError(
                f"Request timeout after {self.timeout}s: {str(e)}"
            )
        except requests.exceptions.ConnectionError as e:
            raise CollibraAPIError(f"Connection error: {str(e)}")
        except requests.exceptions.RequestException as e:
            raise CollibraAPIError(f"Request failed: {str(e)}")

    def health_check(self) -> bool:
        """
        Check API health and connectivity.

        Returns:
            True if API is healthy and accessible
        """
        try:
            # Try to get application info (lightweight endpoint)
            response = self._make_request("GET", "/rest/2.0/application/info")
            return response is not None
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    # Asset Management

    def create_asset(
        self,
        domain_id: str,
        asset_type: str,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
        display_name: Optional[str] = None,
    ) -> Dict:
        """
        Create a new asset in Collibra.

        Args:
            domain_id: Domain UUID where asset will be created
            asset_type: Asset type name or UUID
            name: Asset name
            attributes: Asset attributes
            display_name: Display name (optional)

        Returns:
            Created asset data
        """
        data = {
            "domainId": domain_id,
            "typeId": asset_type,  # Can be name or UUID
            "name": name,
        }

        if display_name:
            data["displayName"] = display_name

        if attributes:
            data["attributes"] = attributes

        return self._make_request("POST", "/rest/2.0/assets", data=data)

    def update_asset(self, asset_id: str, attributes: Dict[str, Any]) -> Dict:
        """
        Update an existing asset.

        Args:
            asset_id: Asset UUID
            attributes: Attributes to update

        Returns:
            Updated asset data
        """
        data = {"attributes": attributes}
        return self._make_request(
            "PATCH", f"/rest/2.0/assets/{asset_id}", data=data
        )

    def get_asset(self, asset_id: str) -> Dict:
        """
        Get asset by ID.

        Args:
            asset_id: Asset UUID

        Returns:
            Asset data
        """
        return self._make_request("GET", f"/rest/2.0/assets/{asset_id}")

    def search_assets(
        self,
        query: Optional[str] = None,
        asset_type: Optional[str] = None,
        domain_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict]:
        """
        Search for assets.

        Args:
            query: Search query string
            asset_type: Filter by asset type
            domain_id: Filter by domain
            limit: Maximum results to return
            offset: Pagination offset

        Returns:
            List of matching assets
        """
        params = {"limit": limit, "offset": offset}

        if query:
            params["name"] = query
        if asset_type:
            params["typeId"] = asset_type
        if domain_id:
            params["domainId"] = domain_id

        response = self._make_request("GET", "/rest/2.0/assets", params=params)
        return response.get("results", [])

    # Policy Management (simulated - Collibra may use different API)

    def list_policies(self, domain_id: Optional[str] = None) -> List[Dict]:
        """
        List governance policies.

        Note: This is a simplified implementation. Actual Collibra policy API
        may differ based on version and configuration.

        Args:
            domain_id: Filter by domain

        Returns:
            List of policies
        """
        params = {}
        if domain_id:
            params["domainId"] = domain_id

        # Collibra may use data quality rules, business rules, or custom policies
        # This endpoint is simplified for the integration
        try:
            response = self._make_request(
                "GET", "/rest/2.0/dataQualityRules", params=params
            )
            return response.get("results", [])
        except CollibraAPIError:
            logger.warning(
                "Policy listing not available. Check Collibra API version."
            )
            return []

    def get_policy(self, policy_id: str) -> Dict:
        """
        Get policy by ID.

        Args:
            policy_id: Policy UUID

        Returns:
            Policy data
        """
        return self._make_request("GET", f"/rest/2.0/dataQualityRules/{policy_id}")

    # Domain Management

    def get_domain(self, domain_id: str) -> Dict:
        """
        Get domain by ID.

        Args:
            domain_id: Domain UUID

        Returns:
            Domain data
        """
        return self._make_request("GET", f"/rest/2.0/domains/{domain_id}")

    def list_domains(self, community_id: Optional[str] = None) -> List[Dict]:
        """
        List domains.

        Args:
            community_id: Filter by community

        Returns:
            List of domains
        """
        params = {}
        if community_id:
            params["communityId"] = community_id

        response = self._make_request("GET", "/rest/2.0/domains", params=params)
        return response.get("results", [])

    # Relationship Management

    def create_relation(
        self, source_id: str, target_id: str, relation_type: str
    ) -> Dict:
        """
        Create a relationship between assets.

        Args:
            source_id: Source asset UUID
            target_id: Target asset UUID
            relation_type: Relation type name or UUID

        Returns:
            Created relation data
        """
        data = {
            "sourceId": source_id,
            "targetId": target_id,
            "typeId": relation_type,
        }
        return self._make_request("POST", "/rest/2.0/relations", data=data)

    def get_application_info(self) -> Dict:
        """
        Get Collibra application information.

        Returns:
            Application info including version
        """
        return self._make_request("GET", "/rest/2.0/application/info")
