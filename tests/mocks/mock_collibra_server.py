"""Mock Collibra API server for testing."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MockAPICall:
    """Record of an API call made to the mock server."""

    method: str
    endpoint: str
    data: Optional[Dict] = None
    params: Optional[Dict] = None
    timestamp: float = field(default_factory=time.time)


class MockCollibraServer:
    """Mock Collibra API server for testing."""

    def __init__(self, api_version: str = "5.7.2"):
        """
        Initialize mock server.

        Args:
            api_version: Simulated Collibra API version
        """
        self.api_version = api_version
        self.assets: Dict[str, Dict] = {}
        self.policies: Dict[str, Dict] = {}
        self.domains: Dict[str, Dict] = {}
        self.relations: Dict[str, Dict] = {}
        self.api_calls: List[MockAPICall] = []

        # Authentication state
        self.valid_username = "test_user"
        self.valid_password = "test_password"
        self.valid_token = "test_api_token"
        self.require_auth = True

        # Rate limiting
        self.rate_limit_enabled = False
        self.rate_limit_threshold = 100
        self.request_count = 0

        # Initialize with default domain
        self._create_default_domain()

    def _create_default_domain(self):
        """Create default test domain."""
        domain_id = str(uuid.uuid4())
        self.domains[domain_id] = {
            "id": domain_id,
            "name": "AI Governance",
            "type": "Domain",
            "description": "Domain for AI governance assets",
            "communityId": str(uuid.uuid4()),
        }

    def _check_auth(
        self, username: Optional[str], password: Optional[str], token: Optional[str]
    ) -> bool:
        """
        Check if authentication is valid.

        Args:
            username: Basic auth username
            password: Basic auth password
            token: API token

        Returns:
            True if authenticated
        """
        if not self.require_auth:
            return True

        if token:
            return token == self.valid_token

        if username and password:
            return (
                username == self.valid_username and password == self.valid_password
            )

        return False

    def _record_api_call(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ):
        """Record API call for inspection."""
        self.api_calls.append(
            MockAPICall(method=method, endpoint=endpoint, data=data, params=params)
        )
        self.request_count += 1

    def get_api_call_count(self, endpoint: Optional[str] = None) -> int:
        """
        Get count of API calls.

        Args:
            endpoint: Filter by endpoint (optional)

        Returns:
            Number of API calls
        """
        if endpoint:
            return sum(1 for call in self.api_calls if endpoint in call.endpoint)
        return len(self.api_calls)

    def reset(self):
        """Reset mock server state."""
        self.assets.clear()
        self.policies.clear()
        self.domains.clear()
        self.relations.clear()
        self.api_calls.clear()
        self.request_count = 0
        self._create_default_domain()

    # API Endpoint Handlers

    def handle_health_check(self) -> Dict:
        """Handle health check endpoint."""
        self._record_api_call("GET", "/rest/2.0/application/info")
        return {
            "version": self.api_version,
            "buildNumber": "12345",
            "environment": "test",
        }

    def handle_create_asset(
        self,
        domain_id: str,
        asset_type: str,
        name: str,
        attributes: Optional[Dict] = None,
        display_name: Optional[str] = None,
    ) -> Dict:
        """
        Handle asset creation.

        Args:
            domain_id: Domain UUID
            asset_type: Asset type
            name: Asset name
            attributes: Asset attributes
            display_name: Display name

        Returns:
            Created asset
        """
        asset_id = str(uuid.uuid4())

        asset = {
            "id": asset_id,
            "domainId": domain_id,
            "typeId": asset_type,
            "name": name,
            "displayName": display_name or name,
            "attributes": attributes or {},
            "createdOn": int(time.time() * 1000),
            "lastModifiedOn": int(time.time() * 1000),
            "status": "active",
        }

        self.assets[asset_id] = asset
        self._record_api_call(
            "POST",
            "/rest/2.0/assets",
            data={
                "domainId": domain_id,
                "typeId": asset_type,
                "name": name,
                "attributes": attributes,
            },
        )

        return asset

    def handle_update_asset(self, asset_id: str, attributes: Dict) -> Dict:
        """
        Handle asset update.

        Args:
            asset_id: Asset UUID
            attributes: Attributes to update

        Returns:
            Updated asset

        Raises:
            KeyError: If asset not found
        """
        if asset_id not in self.assets:
            raise KeyError(f"Asset not found: {asset_id}")

        asset = self.assets[asset_id]
        asset["attributes"].update(attributes)
        asset["lastModifiedOn"] = int(time.time() * 1000)

        self._record_api_call(
            "PATCH", f"/rest/2.0/assets/{asset_id}", data={"attributes": attributes}
        )

        return asset

    def handle_get_asset(self, asset_id: str) -> Dict:
        """
        Handle get asset.

        Args:
            asset_id: Asset UUID

        Returns:
            Asset data

        Raises:
            KeyError: If asset not found
        """
        if asset_id not in self.assets:
            raise KeyError(f"Asset not found: {asset_id}")

        self._record_api_call("GET", f"/rest/2.0/assets/{asset_id}")
        return self.assets[asset_id]

    def handle_search_assets(
        self,
        query: Optional[str] = None,
        asset_type: Optional[str] = None,
        domain_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict:
        """
        Handle asset search.

        Args:
            query: Search query
            asset_type: Filter by asset type
            domain_id: Filter by domain
            limit: Result limit
            offset: Pagination offset

        Returns:
            Search results
        """
        results = list(self.assets.values())

        # Apply filters
        if query:
            results = [a for a in results if query.lower() in a["name"].lower()]
        if asset_type:
            results = [a for a in results if a["typeId"] == asset_type]
        if domain_id:
            results = [a for a in results if a["domainId"] == domain_id]

        # Apply pagination
        total = len(results)
        results = results[offset : offset + limit]

        self._record_api_call(
            "GET",
            "/rest/2.0/assets",
            params={
                "name": query,
                "typeId": asset_type,
                "domainId": domain_id,
                "limit": limit,
                "offset": offset,
            },
        )

        return {"results": results, "total": total, "offset": offset, "limit": limit}

    def handle_list_policies(self, domain_id: Optional[str] = None) -> Dict:
        """
        Handle list policies.

        Args:
            domain_id: Filter by domain

        Returns:
            Policy list
        """
        results = list(self.policies.values())

        if domain_id:
            results = [p for p in results if p.get("domainId") == domain_id]

        self._record_api_call(
            "GET", "/rest/2.0/dataQualityRules", params={"domainId": domain_id}
        )

        return {"results": results, "total": len(results)}

    def handle_get_policy(self, policy_id: str) -> Dict:
        """
        Handle get policy.

        Args:
            policy_id: Policy UUID

        Returns:
            Policy data

        Raises:
            KeyError: If policy not found
        """
        if policy_id not in self.policies:
            raise KeyError(f"Policy not found: {policy_id}")

        self._record_api_call("GET", f"/rest/2.0/dataQualityRules/{policy_id}")
        return self.policies[policy_id]

    def handle_list_domains(self, community_id: Optional[str] = None) -> Dict:
        """
        Handle list domains.

        Args:
            community_id: Filter by community

        Returns:
            Domain list
        """
        results = list(self.domains.values())

        if community_id:
            results = [d for d in results if d.get("communityId") == community_id]

        self._record_api_call(
            "GET", "/rest/2.0/domains", params={"communityId": community_id}
        )

        return {"results": results, "total": len(results)}

    def handle_get_domain(self, domain_id: str) -> Dict:
        """
        Handle get domain.

        Args:
            domain_id: Domain UUID

        Returns:
            Domain data

        Raises:
            KeyError: If domain not found
        """
        if domain_id not in self.domains:
            raise KeyError(f"Domain not found: {domain_id}")

        self._record_api_call("GET", f"/rest/2.0/domains/{domain_id}")
        return self.domains[domain_id]

    def handle_create_relation(
        self, source_id: str, target_id: str, relation_type: str
    ) -> Dict:
        """
        Handle create relation.

        Args:
            source_id: Source asset UUID
            target_id: Target asset UUID
            relation_type: Relation type

        Returns:
            Created relation
        """
        relation_id = str(uuid.uuid4())

        relation = {
            "id": relation_id,
            "sourceId": source_id,
            "targetId": target_id,
            "typeId": relation_type,
            "createdOn": int(time.time() * 1000),
        }

        self.relations[relation_id] = relation
        self._record_api_call(
            "POST",
            "/rest/2.0/relations",
            data={"sourceId": source_id, "targetId": target_id, "typeId": relation_type},
        )

        return relation

    # Test Utilities

    def inject_policy(self, policy: Dict):
        """
        Inject a policy into the mock server for testing.

        Args:
            policy: Policy data
        """
        policy_id = policy.get("id") or str(uuid.uuid4())
        policy["id"] = policy_id
        self.policies[policy_id] = policy

    def inject_domain(self, domain: Dict):
        """
        Inject a domain into the mock server for testing.

        Args:
            domain: Domain data
        """
        domain_id = domain.get("id") or str(uuid.uuid4())
        domain["id"] = domain_id
        self.domains[domain_id] = domain

    def get_default_domain_id(self) -> str:
        """
        Get ID of the default test domain.

        Returns:
            Domain UUID
        """
        return list(self.domains.keys())[0]

    def set_rate_limit_enabled(self, enabled: bool, threshold: int = 100):
        """
        Enable/disable rate limiting for testing.

        Args:
            enabled: Enable rate limiting
            threshold: Request threshold before rate limiting
        """
        self.rate_limit_enabled = enabled
        self.rate_limit_threshold = threshold

    def should_rate_limit(self) -> bool:
        """
        Check if request should be rate limited.

        Returns:
            True if rate limit should be applied
        """
        return self.rate_limit_enabled and self.request_count >= self.rate_limit_threshold
