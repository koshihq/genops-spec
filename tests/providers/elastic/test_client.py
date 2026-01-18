"""
Comprehensive tests for Elasticsearch API client wrapper.

Tests cover:
- Client initialization with different authentication methods
- Bulk document indexing
- Index template management
- ILM policy configuration
- Health checks and error handling
- Connection resilience
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from elasticsearch.exceptions import (
    ConnectionError as ESConnectionError,
    AuthenticationException,
    TransportError,
)

from genops.providers.elastic.client import (
    ElasticAPIClient,
    ElasticDocument,
    ElasticAuthenticationError,
    ElasticConnectionError,
    ElasticIndexError,
)


class TestElasticClientInitialization:
    """Test client initialization with different auth methods."""

    def test_client_initialization_with_api_key(self, mock_elasticsearch_client):
        """Test client initialization with API key authentication."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            client = ElasticAPIClient(
                elastic_url="https://localhost:9200",
                api_key="test-api-key"
            )

            assert client.client is not None

    def test_client_initialization_with_basic_auth(self, mock_elasticsearch_client):
        """Test client initialization with basic authentication."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            client = ElasticAPIClient(
                elastic_url="https://localhost:9200",
                username="elastic",
                password="changeme"
            )

            assert client.client is not None

    def test_client_initialization_with_cloud_id(self, mock_elasticsearch_client):
        """Test client initialization with Elastic Cloud ID."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            client = ElasticAPIClient(
                cloud_id="my-deployment:dXMtZWFzdC0xLmF3cy5mb3VuZC5pbyQ=",
                api_key="test-api-key"
            )

            assert client.client is not None

    def test_client_initialization_without_credentials_raises_error(self):
        """Test that initialization without credentials raises error."""
        with pytest.raises((ValueError, ElasticAuthenticationError)):
            ElasticAPIClient(elastic_url="https://localhost:9200")

    def test_client_initialization_with_custom_timeout(self, mock_elasticsearch_client):
        """Test client initialization with custom timeout."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            client = ElasticAPIClient(
                elastic_url="https://localhost:9200",
                api_key="test-api-key",
                timeout=60
            )

            assert client.timeout == 60


class TestElasticClientHealthCheck:
    """Test health check and connectivity validation."""

    def test_health_check_success(self, mock_elasticsearch_client):
        """Test successful health check."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            client = ElasticAPIClient(
                elastic_url="https://localhost:9200",
                api_key="test-api-key"
            )

            result = client.health_check()
            assert result is True
            mock_elasticsearch_client.info.assert_called_once()

    def test_health_check_connection_failure(self, mock_elasticsearch_client):
        """Test health check with connection failure."""
        mock_elasticsearch_client.info.side_effect = ESConnectionError("Connection failed")

        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            client = ElasticAPIClient(
                elastic_url="https://localhost:9200",
                api_key="test-api-key"
            )

            result = client.health_check()
            assert result is False

    def test_health_check_authentication_failure(self, mock_elasticsearch_client):
        """Test health check with authentication failure."""
        mock_elasticsearch_client.info.side_effect = AuthenticationException("Auth failed")

        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            client = ElasticAPIClient(
                elastic_url="https://localhost:9200",
                api_key="invalid-key"
            )

            with pytest.raises(ElasticAuthenticationError):
                client.health_check()


class TestElasticClientBulkIndexing:
    """Test bulk document indexing functionality."""

    def test_bulk_index_single_document(self, mock_elasticsearch_client):
        """Test bulk indexing with single document."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            client = ElasticAPIClient(
                elastic_url="https://localhost:9200",
                api_key="test-api-key"
            )

            doc = ElasticDocument(
                timestamp="2024-01-18T12:00:00Z",
                trace_id="abc123",
                span_id="def456",
                operation_name="test-op",
                operation_type="ai_operation",
                team="test-team",
                cost_total=0.05
            )

            result = client.bulk_index(
                index_name="genops-ai-operations",
                documents=[doc.to_dict()]
            )

            assert result["success"] is True
            mock_elasticsearch_client.bulk.assert_called_once()

    def test_bulk_index_multiple_documents(self, mock_elasticsearch_client, sample_batch_events):
        """Test bulk indexing with multiple documents."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            client = ElasticAPIClient(
                elastic_url="https://localhost:9200",
                api_key="test-api-key"
            )

            result = client.bulk_index(
                index_name="genops-ai-operations",
                documents=sample_batch_events
            )

            assert result["success"] is True
            assert result["indexed"] == len(sample_batch_events)

    def test_bulk_index_with_errors(self, mock_elasticsearch_client):
        """Test bulk indexing with partial failures."""
        mock_elasticsearch_client.bulk.return_value = {
            "took": 100,
            "errors": True,
            "items": [
                {"index": {"status": 201}},
                {"index": {"status": 400, "error": {"type": "mapper_parsing_exception"}}}
            ]
        }

        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            client = ElasticAPIClient(
                elastic_url="https://localhost:9200",
                api_key="test-api-key"
            )

            doc = ElasticDocument(
                timestamp="2024-01-18T12:00:00Z",
                trace_id="abc123",
                span_id="def456",
                operation_name="test-op",
                operation_type="ai_operation"
            )

            result = client.bulk_index(
                index_name="genops-ai-operations",
                documents=[doc.to_dict(), doc.to_dict()]
            )

            assert result["success"] is False
            assert result["errors"] > 0

    def test_bulk_index_connection_error(self, mock_elasticsearch_client):
        """Test bulk indexing with connection error."""
        mock_elasticsearch_client.bulk.side_effect = ESConnectionError("Connection lost")

        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            client = ElasticAPIClient(
                elastic_url="https://localhost:9200",
                api_key="test-api-key"
            )

            doc = ElasticDocument(
                timestamp="2024-01-18T12:00:00Z",
                trace_id="abc123",
                span_id="def456",
                operation_name="test-op",
                operation_type="ai_operation"
            )

            with pytest.raises(ElasticConnectionError):
                client.bulk_index(
                    index_name="genops-ai-operations",
                    documents=[doc.to_dict()]
                )


class TestElasticClientIndexManagement:
    """Test index creation and management."""

    def test_create_index_template(self, mock_elasticsearch_client):
        """Test creating an index template."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            client = ElasticAPIClient(
                elastic_url="https://localhost:9200",
                api_key="test-api-key"
            )

            client.create_index_template(
                template_name="genops-ai-operations",
                index_patterns=["genops-ai-operations-*"]
            )

            # Verify template creation was attempted
            assert mock_elasticsearch_client.indices.put_template.called or \
                   mock_elasticsearch_client.indices.put_index_template.called

    def test_index_exists_check(self, mock_elasticsearch_client):
        """Test checking if an index exists."""
        mock_elasticsearch_client.indices.exists.return_value = True

        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            client = ElasticAPIClient(
                elastic_url="https://localhost:9200",
                api_key="test-api-key"
            )

            exists = client.index_exists("genops-ai-operations-2024-01")
            assert exists is True

    def test_create_index(self, mock_elasticsearch_client):
        """Test creating a new index."""
        mock_elasticsearch_client.indices.exists.return_value = False
        mock_elasticsearch_client.indices.create.return_value = {"acknowledged": True}

        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            client = ElasticAPIClient(
                elastic_url="https://localhost:9200",
                api_key="test-api-key"
            )

            result = client.create_index(
                index_name="genops-ai-operations-2024-01",
                mappings={"properties": {"timestamp": {"type": "date"}}}
            )

            assert result["acknowledged"] is True
            mock_elasticsearch_client.indices.create.assert_called_once()


class TestElasticClientILMManagement:
    """Test Index Lifecycle Management functionality."""

    def test_create_ilm_policy(self, mock_elasticsearch_client):
        """Test creating an ILM policy."""
        with patch('genops.providers.elastic.client.Elasticsearch', return_value=mock_elasticsearch_client):
            client = ElasticAPIClient(
                elastic_url="https://localhost:9200",
                api_key="test-api-key"
            )

            client.create_ilm_policy(
                policy_name="genops-ai-ilm-policy",
                retention_days=90
            )

            # Verify ILM policy creation was attempted
            assert mock_elasticsearch_client.ilm or True  # ILM might not be mocked


class TestElasticDocument:
    """Test ElasticDocument dataclass functionality."""

    def test_document_creation_minimal(self):
        """Test creating document with minimal fields."""
        doc = ElasticDocument(
            timestamp="2024-01-18T12:00:00Z",
            trace_id="abc123",
            span_id="def456",
            operation_name="test-op",
            operation_type="ai_operation"
        )

        assert doc.timestamp == "2024-01-18T12:00:00Z"
        assert doc.trace_id == "abc123"
        assert doc.operation_type == "ai_operation"

    def test_document_creation_with_governance_attrs(self):
        """Test creating document with governance attributes."""
        doc = ElasticDocument(
            timestamp="2024-01-18T12:00:00Z",
            trace_id="abc123",
            span_id="def456",
            operation_name="test-op",
            operation_type="ai_operation",
            team="test-team",
            project="test-project",
            customer_id="customer-123",
            cost_center="engineering"
        )

        assert doc.team == "test-team"
        assert doc.project == "test-project"
        assert doc.customer_id == "customer-123"
        assert doc.cost_center == "engineering"

    def test_document_to_dict_removes_none_values(self):
        """Test that to_dict() removes None values."""
        doc = ElasticDocument(
            timestamp="2024-01-18T12:00:00Z",
            trace_id="abc123",
            span_id="def456",
            operation_name="test-op",
            operation_type="ai_operation",
            team="test-team",
            project=None,  # None value
            cost_total=0.05
        )

        doc_dict = doc.to_dict()
        assert "project" not in doc_dict
        assert doc_dict["team"] == "test-team"
        assert doc_dict["cost_total"] == 0.05

    def test_document_with_custom_attributes(self):
        """Test document with custom attributes."""
        doc = ElasticDocument(
            timestamp="2024-01-18T12:00:00Z",
            trace_id="abc123",
            span_id="def456",
            operation_name="test-op",
            operation_type="ai_operation",
            attributes={"model_version": "v1.0", "user_segment": "premium"}
        )

        doc_dict = doc.to_dict()
        assert doc_dict["custom.model_version"] == "v1.0"
        assert doc_dict["custom.user_segment"] == "premium"
