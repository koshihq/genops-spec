"""
Elasticsearch API Client for GenOps AI governance telemetry.

This module provides a high-level wrapper around the official elasticsearch-py client,
handling authentication, bulk indexing, index management, and ILM policies.
"""

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.exceptions import (
        ConnectionError as ESConnectionError,
        AuthenticationException,
        TransportError,
    )
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False
    Elasticsearch = None
    ESConnectionError = Exception
    AuthenticationException = Exception
    TransportError = Exception

logger = logging.getLogger(__name__)


class ElasticAPIError(Exception):
    """Base exception for Elasticsearch API errors."""
    pass


class ElasticAuthenticationError(ElasticAPIError):
    """Raised when authentication fails."""
    pass


class ElasticConnectionError(ElasticAPIError):
    """Raised when connection to Elasticsearch fails."""
    pass


class ElasticIndexError(ElasticAPIError):
    """Raised when index operations fail."""
    pass


@dataclass
class ElasticDocument:
    """
    Represents a GenOps governance telemetry document for Elasticsearch.

    This structure aligns with GenOps governance semantic conventions
    while providing Elasticsearch-specific fields for optimal indexing.
    """
    # Core telemetry fields
    timestamp: str
    trace_id: str
    span_id: str
    operation_name: str
    operation_type: str  # "ai_operation", "cost", "policy", "budget"

    # Governance attributes (standard GenOps fields)
    team: Optional[str] = None
    project: Optional[str] = None
    environment: str = "production"
    customer_id: Optional[str] = None
    cost_center: Optional[str] = None
    feature: Optional[str] = None

    # Cost telemetry fields
    cost_total: Optional[float] = None
    cost_input: Optional[float] = None
    cost_output: Optional[float] = None
    cost_provider: Optional[str] = None
    cost_model: Optional[str] = None
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    tokens_total: Optional[int] = None

    # Policy telemetry fields
    policy_name: Optional[str] = None
    policy_result: Optional[str] = None  # "allowed", "blocked", "warning"
    policy_reason: Optional[str] = None

    # Budget telemetry fields
    budget_id: Optional[str] = None
    budget_limit: Optional[float] = None
    budget_consumed: Optional[float] = None
    budget_remaining: Optional[float] = None

    # Performance fields
    duration_ms: Optional[float] = None
    status: Optional[str] = None  # "success", "error", "timeout"

    # Additional attributes (flexible for custom telemetry)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Elasticsearch indexing."""
        doc = asdict(self)
        # Flatten attributes into main document
        if doc.get("attributes"):
            attrs = doc.pop("attributes")
            for key, value in attrs.items():
                # Prefix custom attributes to avoid conflicts
                if key not in doc:
                    doc[f"custom.{key}"] = value
        # Remove None values for cleaner indexing
        return {k: v for k, v in doc.items() if v is not None}


class ElasticAPIClient:
    """
    High-level Elasticsearch API client for GenOps governance telemetry.

    Handles:
    - Multiple authentication methods (Basic, API Key, Cloud ID)
    - Bulk document indexing with error handling
    - Index template management
    - ILM policy configuration
    - Cluster health checks and version detection
    """

    def __init__(
        self,
        elastic_url: Optional[str] = None,
        cloud_id: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_key: Optional[str] = None,
        api_id: Optional[str] = None,
        verify_certs: bool = True,
        ca_certs: Optional[str] = None,
        timeout: int = 30,
    ):
        """
        Initialize Elasticsearch client with flexible authentication.

        Args:
            elastic_url: Elasticsearch cluster URL (e.g., "http://localhost:9200")
            cloud_id: Elastic Cloud deployment ID (alternative to elastic_url)
            username: Basic auth username
            password: Basic auth password
            api_key: API key for authentication (recommended for production)
            api_id: API key ID (optional, used with api_key)
            verify_certs: Verify SSL certificates (disable for self-signed certs)
            ca_certs: Path to CA certificate bundle
            timeout: Request timeout in seconds
        """
        if not ELASTICSEARCH_AVAILABLE:
            raise ImportError(
                "elasticsearch package is required for Elastic integration. "
                "Install it with: pip install 'genops-ai[elastic]' or pip install elasticsearch>=8.0.0"
            )

        self.elastic_url = elastic_url
        self.cloud_id = cloud_id
        self.timeout = timeout

        # Determine authentication method and create client
        self.client = self._create_client(
            elastic_url=elastic_url,
            cloud_id=cloud_id,
            username=username,
            password=password,
            api_key=api_key,
            api_id=api_id,
            verify_certs=verify_certs,
            ca_certs=ca_certs,
            timeout=timeout,
        )

        # Cache cluster info
        self._cluster_info: Optional[Dict[str, Any]] = None
        self._cluster_version: Optional[str] = None

    def _create_client(
        self,
        elastic_url: Optional[str],
        cloud_id: Optional[str],
        username: Optional[str],
        password: Optional[str],
        api_key: Optional[str],
        api_id: Optional[str],
        verify_certs: bool,
        ca_certs: Optional[str],
        timeout: int,
    ) -> Elasticsearch:
        """Create Elasticsearch client with appropriate authentication."""
        client_args: Dict[str, Any] = {
            "request_timeout": timeout,
            "verify_certs": verify_certs,
        }

        if ca_certs:
            client_args["ca_certs"] = ca_certs

        # Priority: Cloud ID > elastic_url
        if cloud_id:
            client_args["cloud_id"] = cloud_id
        elif elastic_url:
            client_args["hosts"] = [elastic_url]
        else:
            raise ElasticConnectionError(
                "Either elastic_url or cloud_id must be provided"
            )

        # Authentication: API Key > Basic Auth
        if api_key:
            if api_id:
                client_args["api_key"] = (api_id, api_key)
            else:
                client_args["api_key"] = api_key
        elif username and password:
            client_args["basic_auth"] = (username, password)
        else:
            logger.warning(
                "No authentication credentials provided. "
                "This may work for local development but will fail in production."
            )

        try:
            return Elasticsearch(**client_args)
        except Exception as e:
            raise ElasticConnectionError(f"Failed to create Elasticsearch client: {e}")

    def health_check(self) -> bool:
        """
        Verify connection to Elasticsearch cluster.

        Returns:
            True if cluster is reachable and healthy, False otherwise
        """
        try:
            health = self.client.cluster.health()
            return health.get("status") in ["green", "yellow"]
        except AuthenticationException as e:
            raise ElasticAuthenticationError(f"Authentication failed: {e}")
        except ESConnectionError as e:
            raise ElasticConnectionError(f"Connection failed: {e}")
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_cluster_info(self) -> Dict[str, Any]:
        """
        Get Elasticsearch cluster information including version.

        Returns:
            Dictionary with cluster_name, version, and other metadata
        """
        if self._cluster_info is None:
            try:
                self._cluster_info = self.client.info()
                self._cluster_version = self._cluster_info.get("version", {}).get("number")
            except Exception as e:
                raise ElasticAPIError(f"Failed to get cluster info: {e}")

        return self._cluster_info

    def get_version(self) -> str:
        """Get Elasticsearch cluster version."""
        if self._cluster_version is None:
            self.get_cluster_info()
        return self._cluster_version or "unknown"

    def index_document(
        self,
        index: str,
        document: Union[ElasticDocument, Dict[str, Any]],
        doc_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Index a single document.

        Args:
            index: Target index name
            document: Document to index (ElasticDocument or dict)
            doc_id: Optional document ID (auto-generated if not provided)

        Returns:
            Elasticsearch response with _id, _index, result, etc.
        """
        if isinstance(document, ElasticDocument):
            doc_dict = document.to_dict()
        else:
            doc_dict = document

        try:
            response = self.client.index(
                index=index,
                document=doc_dict,
                id=doc_id,
            )
            return response
        except Exception as e:
            raise ElasticIndexError(f"Failed to index document: {e}")

    def bulk_index(
        self,
        index: str,
        documents: List[Union[ElasticDocument, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        Bulk index multiple documents for optimal performance.

        Args:
            index: Target index name
            documents: List of documents to index

        Returns:
            Dictionary with success count, errors, and timing info
        """
        if not documents:
            return {"success": 0, "errors": [], "took_ms": 0}

        # Prepare bulk request body
        bulk_body = []
        for doc in documents:
            # Index action
            bulk_body.append({"index": {"_index": index}})
            # Document
            if isinstance(doc, ElasticDocument):
                bulk_body.append(doc.to_dict())
            else:
                bulk_body.append(doc)

        try:
            response = self.client.bulk(operations=bulk_body)

            # Parse response
            items = response.get("items", [])
            errors = []
            success_count = 0

            for item in items:
                index_result = item.get("index", {})
                if index_result.get("status") in [200, 201]:
                    success_count += 1
                else:
                    errors.append({
                        "status": index_result.get("status"),
                        "error": index_result.get("error"),
                    })

            return {
                "success": success_count,
                "errors": errors,
                "took_ms": response.get("took", 0),
                "total": len(documents),
            }

        except Exception as e:
            raise ElasticIndexError(f"Bulk indexing failed: {e}")

    def create_index_template(
        self,
        template_name: str,
        index_pattern: str,
        mappings: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create an index template for consistent field mappings.

        Args:
            template_name: Template name
            index_pattern: Index pattern (e.g., "genops-ai-*")
            mappings: Field mappings
            settings: Index settings (shards, replicas, etc.)

        Returns:
            Elasticsearch response
        """
        template_body: Dict[str, Any] = {
            "index_patterns": [index_pattern],
        }

        if mappings:
            template_body["template"] = {"mappings": mappings}

        if settings:
            if "template" not in template_body:
                template_body["template"] = {}
            template_body["template"]["settings"] = settings

        try:
            response = self.client.indices.put_index_template(
                name=template_name,
                body=template_body,
            )
            return response
        except Exception as e:
            raise ElasticIndexError(f"Failed to create index template: {e}")

    def create_ilm_policy(
        self,
        policy_name: str,
        retention_days: int = 90,
        rollover_size: str = "50gb",
        rollover_age: str = "30d",
    ) -> Dict[str, Any]:
        """
        Create an Index Lifecycle Management (ILM) policy.

        Args:
            policy_name: ILM policy name
            retention_days: Number of days to retain data before deletion
            rollover_size: Rollover when index reaches this size
            rollover_age: Rollover when index reaches this age

        Returns:
            Elasticsearch response
        """
        policy_body = {
            "policy": {
                "phases": {
                    "hot": {
                        "actions": {
                            "rollover": {
                                "max_size": rollover_size,
                                "max_age": rollover_age,
                            }
                        }
                    },
                    "delete": {
                        "min_age": f"{retention_days}d",
                        "actions": {
                            "delete": {}
                        }
                    }
                }
            }
        }

        try:
            response = self.client.ilm.put_lifecycle(
                name=policy_name,
                body=policy_body,
            )
            return response
        except Exception as e:
            logger.warning(f"Failed to create ILM policy (may not be supported): {e}")
            return {"acknowledged": False, "error": str(e)}

    def index_exists(self, index: str) -> bool:
        """Check if an index exists."""
        try:
            return self.client.indices.exists(index=index)
        except Exception as e:
            logger.error(f"Failed to check index existence: {e}")
            return False

    def create_index(
        self,
        index: str,
        mappings: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create an index with optional mappings and settings.

        Args:
            index: Index name
            mappings: Field mappings
            settings: Index settings

        Returns:
            Elasticsearch response
        """
        body: Dict[str, Any] = {}
        if mappings:
            body["mappings"] = mappings
        if settings:
            body["settings"] = settings

        try:
            response = self.client.indices.create(index=index, body=body if body else None)
            return response
        except Exception as e:
            raise ElasticIndexError(f"Failed to create index: {e}")

    def close(self):
        """Close the Elasticsearch client connection."""
        try:
            self.client.close()
        except Exception as e:
            logger.error(f"Error closing Elasticsearch client: {e}")
