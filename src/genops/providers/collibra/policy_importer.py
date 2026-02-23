"""Policy importer for syncing Collibra policies to GenOps PolicyEngine."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from genops.core.policy import PolicyConfig, PolicyResult, register_policy
from genops.providers.collibra.client import CollibraAPIClient, CollibraAPIError

logger = logging.getLogger(__name__)


@dataclass
class PolicySyncStats:
    """Statistics for policy synchronization."""

    policies_imported: int = 0
    policies_updated: int = 0
    policies_failed: int = 0
    last_sync_time: float | None = None
    errors: list[str] = field(default_factory=list)

    def record_import(self, count: int = 1):
        """Record successful policy import."""
        self.policies_imported += count
        self.last_sync_time = time.time()

    def record_update(self, count: int = 1):
        """Record policy update."""
        self.policies_updated += count
        self.last_sync_time = time.time()

    def record_failure(self, error: str):
        """Record policy import failure."""
        self.policies_failed += 1
        self.errors.append(error)


class PolicyImporter:
    """
    Import and sync policies from Collibra to GenOps PolicyEngine.

    Supports:
    - One-time policy import from Collibra
    - Periodic background sync
    - Policy translation from Collibra to GenOps format
    - Custom policy transformation callbacks
    """

    # Mapping from Collibra policy types to GenOps policy names
    POLICY_TYPE_MAPPING = {
        "AI Cost Limit": "cost_limit",
        "AI Rate Limit": "rate_limit",
        "Content Filter": "content_filter",
        "Team Access Control": "team_access",
        "Budget Constraint": "budget_limit",
        "Model Governance": "model_governance",
    }

    # Mapping from Collibra enforcement levels to GenOps PolicyResult
    ENFORCEMENT_MAPPING = {
        "block": PolicyResult.BLOCKED,
        "blocked": PolicyResult.BLOCKED,
        "enforce": PolicyResult.BLOCKED,
        "warn": PolicyResult.WARNING,
        "warning": PolicyResult.WARNING,
        "alert": PolicyResult.WARNING,
        "rate_limit": PolicyResult.RATE_LIMITED,
        "throttle": PolicyResult.RATE_LIMITED,
        "allow": PolicyResult.ALLOWED,
        "allowed": PolicyResult.ALLOWED,
    }

    def __init__(
        self,
        client: CollibraAPIClient,
        domain_id: str | None = None,
        sync_interval_minutes: int = 5,
        enable_background_sync: bool = False,
        policy_transformer: Callable[[dict], PolicyConfig | None] | None = None,
    ):
        """
        Initialize policy importer.

        Args:
            client: Collibra API client
            domain_id: Collibra domain ID to import policies from (optional)
            sync_interval_minutes: Background sync interval
            enable_background_sync: Enable periodic background sync
            policy_transformer: Custom policy transformation function
        """
        self.client = client
        self.domain_id = domain_id
        self.sync_interval_minutes = sync_interval_minutes
        self.policy_transformer = policy_transformer

        # Statistics
        self.stats = PolicySyncStats()

        # Imported policy tracking
        self.imported_policies: dict[str, PolicyConfig] = {}

        # Background sync thread
        self.background_sync_enabled = enable_background_sync
        self.background_thread: threading.Thread | None = None
        self.shutdown_event = threading.Event()

        if self.background_sync_enabled:
            self._start_background_sync()

    def fetch_policies(self, domain_id: str | None = None) -> list[dict[str, Any]]:
        """
        Fetch policies from Collibra.

        Args:
            domain_id: Optional domain ID to filter policies

        Returns:
            List of Collibra policy dictionaries
        """
        try:
            # Use provided domain_id or instance default
            target_domain = domain_id or self.domain_id

            # Fetch assets with policy-related types
            # In a real implementation, this would use Collibra's policy API
            # For now, we simulate by fetching assets of type "Policy"
            policies = []

            # Fetch all domains if no specific domain provided
            if target_domain:
                domain_policies = self._fetch_domain_policies(target_domain)
                policies.extend(domain_policies)
            else:
                # Fetch from all domains
                domains = self.client.list_domains()
                for domain in domains:
                    domain_policies = self._fetch_domain_policies(domain["id"])
                    policies.extend(domain_policies)

            logger.info(f"Fetched {len(policies)} policies from Collibra")
            return policies

        except CollibraAPIError as e:
            logger.error(f"Failed to fetch policies from Collibra: {e}")
            self.stats.record_failure(str(e))
            return []

    def _fetch_domain_policies(self, domain_id: str) -> list[dict[str, Any]]:
        """
        Fetch policies from a specific Collibra domain.

        Args:
            domain_id: Collibra domain ID

        Returns:
            List of policy dictionaries
        """
        try:
            # Search for assets with policy-related types
            # Note: Collibra's actual policy API may differ; this is a simplified version
            assets = self.client.list_assets(domain_id=domain_id)

            # Filter for policy assets
            policy_assets = [
                asset
                for asset in assets
                if asset.get("typeId") in self.POLICY_TYPE_MAPPING.keys()
            ]

            return policy_assets

        except CollibraAPIError as e:
            logger.warning(f"Failed to fetch policies from domain {domain_id}: {e}")
            return []

    def translate_policy(self, collibra_policy: dict[str, Any]) -> PolicyConfig | None:
        """
        Translate Collibra policy to GenOps PolicyConfig.

        Args:
            collibra_policy: Collibra policy dictionary

        Returns:
            GenOps PolicyConfig or None if translation fails
        """
        try:
            # Use custom transformer if provided
            if self.policy_transformer:
                return self.policy_transformer(collibra_policy)

            # Default translation
            policy_type = collibra_policy.get("typeId", "")
            policy_name_base = self.POLICY_TYPE_MAPPING.get(
                policy_type, "custom_policy"
            )

            # Create unique policy name
            policy_id = collibra_policy.get("id", "unknown")
            policy_name = f"{policy_name_base}_{policy_id}"

            # Extract policy attributes
            attributes = collibra_policy.get("attributes", {})

            # Map enforcement level
            enforcement_str = attributes.get("enforcement_level", "block").lower()
            enforcement_level = self.ENFORCEMENT_MAPPING.get(
                enforcement_str, PolicyResult.BLOCKED
            )

            # Extract enabled status
            enabled = attributes.get("enabled", True)
            if isinstance(enabled, str):
                enabled = enabled.lower() in ["true", "yes", "enabled", "1"]

            # Extract description
            description = (
                collibra_policy.get("name", "")
                + " - "
                + attributes.get("description", "Imported from Collibra")
            )

            # Extract conditions based on policy type
            conditions = self._extract_policy_conditions(policy_type, attributes)

            # Create PolicyConfig
            policy_config = PolicyConfig(
                name=policy_name,
                description=description,
                enabled=enabled,
                enforcement_level=enforcement_level,
                conditions=conditions,
            )

            return policy_config

        except Exception as e:
            logger.error(f"Failed to translate policy {collibra_policy.get('id')}: {e}")
            self.stats.record_failure(str(e))
            return None

    def _extract_policy_conditions(
        self, policy_type: str, attributes: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Extract policy conditions from Collibra attributes.

        Args:
            policy_type: Collibra policy type
            attributes: Policy attributes

        Returns:
            Conditions dictionary for GenOps PolicyConfig
        """
        conditions = {}

        # Cost limit policy
        if policy_type == "AI Cost Limit":
            if "max_cost" in attributes:
                conditions["max_cost"] = float(attributes["max_cost"])
            elif "cost_limit" in attributes:
                conditions["max_cost"] = float(attributes["cost_limit"])

        # Rate limit policy
        elif policy_type == "AI Rate Limit":
            if "max_requests" in attributes:
                conditions["max_requests"] = int(attributes["max_requests"])
            elif "max_requests_per_minute" in attributes:
                conditions["max_requests_per_minute"] = int(
                    attributes["max_requests_per_minute"]
                )
            elif "rate_limit" in attributes:
                conditions["max_requests_per_minute"] = int(attributes["rate_limit"])

        # Content filter policy
        elif policy_type == "Content Filter":
            if "blocked_patterns" in attributes:
                patterns = attributes["blocked_patterns"]
                if isinstance(patterns, str):
                    patterns = [p.strip() for p in patterns.split(",")]
                conditions["blocked_patterns"] = patterns

        # Team access policy
        elif policy_type == "Team Access Control":
            if "allowed_teams" in attributes:
                teams = attributes["allowed_teams"]
                if isinstance(teams, str):
                    teams = [t.strip() for t in teams.split(",")]
                conditions["allowed_teams"] = teams

        # Budget constraint policy
        elif policy_type == "Budget Constraint":
            if "daily_budget" in attributes:
                conditions["daily_budget"] = float(attributes["daily_budget"])
            if "monthly_budget" in attributes:
                conditions["monthly_budget"] = float(attributes["monthly_budget"])

        # Model governance policy
        elif policy_type == "Model Governance":
            if "allowed_models" in attributes:
                models = attributes["allowed_models"]
                if isinstance(models, str):
                    models = [m.strip() for m in models.split(",")]
                conditions["allowed_models"] = models
            if "blocked_models" in attributes:
                models = attributes["blocked_models"]
                if isinstance(models, str):
                    models = [m.strip() for m in models.split(",")]
                conditions["blocked_models"] = models

        # Generic conditions - pass through any unrecognized attributes
        for key, value in attributes.items():
            if key not in [
                "enforcement_level",
                "enabled",
                "description",
                "name",
            ]:
                if key not in conditions:
                    conditions[key] = value

        return conditions

    def import_policies(
        self, domain_id: str | None = None, register: bool = True
    ) -> list[PolicyConfig]:
        """
        Import policies from Collibra and optionally register with GenOps.

        Args:
            domain_id: Optional domain ID to import from
            register: Whether to register policies with GenOps PolicyEngine

        Returns:
            List of imported PolicyConfig objects
        """
        logger.info("Starting policy import from Collibra...")

        # Fetch policies
        collibra_policies = self.fetch_policies(domain_id)

        # Translate policies
        imported_policies = []
        for collibra_policy in collibra_policies:
            policy_config = self.translate_policy(collibra_policy)
            if policy_config:
                imported_policies.append(policy_config)

                # Register with GenOps if requested
                if register:
                    try:
                        register_policy(
                            name=policy_config.name,
                            description=policy_config.description,
                            enabled=policy_config.enabled,
                            enforcement_level=policy_config.enforcement_level,
                            **policy_config.conditions,
                        )
                        self.imported_policies[policy_config.name] = policy_config
                        self.stats.record_import()
                        logger.debug(f"Registered policy: {policy_config.name}")
                    except Exception as e:
                        logger.error(
                            f"Failed to register policy {policy_config.name}: {e}"
                        )
                        self.stats.record_failure(str(e))

        logger.info(
            f"Policy import complete: {len(imported_policies)} policies imported"
        )
        return imported_policies

    def sync_policies(self, domain_id: str | None = None) -> dict[str, Any]:
        """
        Synchronize policies from Collibra (import new, update existing).

        Args:
            domain_id: Optional domain ID to sync from

        Returns:
            Sync statistics dictionary
        """
        logger.info("Starting policy synchronization...")

        # Import policies (this will register/update them)
        imported = self.import_policies(domain_id=domain_id, register=True)

        sync_result = {
            "imported": len(imported),
            "updated": self.stats.policies_updated,
            "failed": self.stats.policies_failed,
            "timestamp": time.time(),
        }

        logger.info(
            f"Policy sync complete: {sync_result['imported']} imported, "
            f"{sync_result['failed']} failed"
        )

        return sync_result

    def _start_background_sync(self):
        """Start background thread for periodic policy synchronization."""
        if self.background_thread is not None:
            logger.warning("Background sync thread already running")
            return

        self.shutdown_event.clear()
        self.background_thread = threading.Thread(
            target=self._background_sync_loop, daemon=True, name="CollibraPolicySync"
        )
        self.background_thread.start()
        logger.info(
            f"Started background policy sync thread "
            f"(interval: {self.sync_interval_minutes} minutes)"
        )

    def _background_sync_loop(self):
        """Background thread loop for periodic policy synchronization."""
        while not self.shutdown_event.is_set():
            # Wait for interval or shutdown signal
            interval_seconds = self.sync_interval_minutes * 60
            if self.shutdown_event.wait(timeout=interval_seconds):
                break  # Shutdown requested

            # Sync policies
            try:
                logger.debug("Background policy sync triggered")
                self.sync_policies()
            except Exception as e:
                logger.error(f"Error in background policy sync: {e}")
                self.stats.record_failure(str(e))

        logger.info("Background policy sync thread stopped")

    def shutdown(self, timeout: float = 5.0) -> bool:
        """
        Shutdown policy importer and stop background sync.

        Args:
            timeout: Maximum time to wait for shutdown

        Returns:
            True if shutdown completed successfully
        """
        logger.info("Shutting down policy importer...")

        # Signal background thread to stop
        self.shutdown_event.set()

        # Wait for background thread
        if self.background_thread and self.background_thread.is_alive():
            self.background_thread.join(timeout=timeout)

        return True

    def get_stats(self) -> PolicySyncStats:
        """
        Get policy synchronization statistics.

        Returns:
            Policy sync statistics
        """
        return self.stats

    def get_imported_policies(self) -> dict[str, PolicyConfig]:
        """
        Get all imported policies.

        Returns:
            Dictionary of policy name to PolicyConfig
        """
        return self.imported_policies.copy()
