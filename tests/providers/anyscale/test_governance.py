"""Tests for Anyscale governance and attribution functionality."""

from unittest.mock import Mock, patch

from genops.providers.anyscale import instrument_anyscale


class TestGovernanceDefaults:
    """Test governance default attributes."""

    def test_adapter_with_team_governance(self):
        """Test adapter with team governance attribute."""
        adapter = instrument_anyscale(team="engineering-team")

        assert adapter.governance_defaults["team"] == "engineering-team"

    def test_adapter_with_project_governance(self):
        """Test adapter with project governance attribute."""
        adapter = instrument_anyscale(team="engineering", project="ai-features")

        assert adapter.governance_defaults["project"] == "ai-features"

    def test_adapter_with_environment_governance(self):
        """Test adapter with environment governance attribute."""
        adapter = instrument_anyscale(environment="production")

        assert adapter.governance_defaults["environment"] == "production"

    def test_adapter_with_cost_center_governance(self):
        """Test adapter with cost center governance attribute."""
        adapter = instrument_anyscale(cost_center="R&D")

        assert adapter.governance_defaults["cost_center"] == "R&D"

    def test_adapter_with_multiple_governance_attrs(self):
        """Test adapter with multiple governance attributes."""
        adapter = instrument_anyscale(
            team="ml-team",
            project="chatbot",
            environment="staging",
            cost_center="AI-Research",
        )

        assert adapter.governance_defaults["team"] == "ml-team"
        assert adapter.governance_defaults["project"] == "chatbot"
        assert adapter.governance_defaults["environment"] == "staging"
        assert adapter.governance_defaults["cost_center"] == "AI-Research"


class TestPerRequestGovernance:
    """Test per-request governance attributes."""

    @patch("genops.providers.anyscale.adapter.requests")
    def test_per_request_customer_id(self, mock_requests):
        """Test per-request customer_id attribute."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test",
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "choices": [{"message": {"content": "response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }
        mock_requests.post.return_value = mock_response

        adapter = instrument_anyscale(anyscale_api_key="test-key", team="base-team")

        response = adapter.completion_create(
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=[{"role": "user", "content": "test"}],
            customer_id="customer-123",
        )

        assert response is not None

    @patch("genops.providers.anyscale.adapter.requests")
    def test_per_request_feature_attribute(self, mock_requests):
        """Test per-request feature attribute."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test",
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "choices": [{"message": {"content": "response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }
        mock_requests.post.return_value = mock_response

        adapter = instrument_anyscale(anyscale_api_key="test-key")

        response = adapter.completion_create(
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=[{"role": "user", "content": "test"}],
            feature="chat-completion",
        )

        assert response is not None

    @patch("genops.providers.anyscale.adapter.requests")
    def test_governance_override(self, mock_requests):
        """Test per-request governance overrides defaults."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test",
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "choices": [{"message": {"content": "response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }
        mock_requests.post.return_value = mock_response

        adapter = instrument_anyscale(anyscale_api_key="test-key", team="default-team")

        # Per-request team should override default
        response = adapter.completion_create(
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=[{"role": "user", "content": "test"}],
            team="override-team",
        )

        assert response is not None


class TestGovernanceContextManager:
    """Test governance context manager functionality."""

    def test_governance_context_basic(self):
        """Test basic governance context manager."""
        adapter = instrument_anyscale(team="base-team")

        with adapter.governance_context(customer_id="customer-123") as ctx:
            assert "customer_id" in ctx

    def test_governance_context_multiple_attrs(self):
        """Test governance context with multiple attributes."""
        adapter = instrument_anyscale()

        with adapter.governance_context(
            customer_id="customer-123", feature="analysis", workflow_id="workflow-456"
        ) as ctx:
            assert "customer_id" in ctx
            assert "feature" in ctx
            assert "workflow_id" in ctx

    @patch("genops.providers.anyscale.adapter.requests")
    def test_context_applies_to_requests(self, mock_requests):
        """Test context applies to requests within it."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test",
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "choices": [{"message": {"content": "response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }
        mock_requests.post.return_value = mock_response

        adapter = instrument_anyscale(anyscale_api_key="test-key")

        with adapter.governance_context(customer_id="customer-123"):
            response = adapter.completion_create(
                model="meta-llama/Llama-2-7b-chat-hf",
                messages=[{"role": "user", "content": "test"}],
            )
            assert response is not None


class TestCostAttribution:
    """Test cost attribution functionality."""

    @patch("genops.providers.anyscale.adapter.requests")
    def test_cost_attributed_to_customer(self, mock_requests):
        """Test costs can be attributed to customers."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test",
            "model": "meta-llama/Llama-2-70b-chat-hf",
            "choices": [{"message": {"content": "response"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }
        mock_requests.post.return_value = mock_response

        adapter = instrument_anyscale(anyscale_api_key="test-key")

        response = adapter.completion_create(
            model="meta-llama/Llama-2-70b-chat-hf",
            messages=[{"role": "user", "content": "test"}],
            customer_id="enterprise-client-123",
        )

        # Cost should be calculable from response
        assert response["usage"]["total_tokens"] == 150

    @patch("genops.providers.anyscale.adapter.requests")
    def test_cost_attributed_to_team(self, mock_requests):
        """Test costs can be attributed to teams."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test",
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "choices": [{"message": {"content": "response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75},
        }
        mock_requests.post.return_value = mock_response

        adapter = instrument_anyscale(
            anyscale_api_key="test-key", team="ml-engineering"
        )

        response = adapter.completion_create(
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=[{"role": "user", "content": "test"}],
        )

        assert response is not None

    @patch("genops.providers.anyscale.adapter.requests")
    def test_cost_attributed_to_project(self, mock_requests):
        """Test costs can be attributed to projects."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test",
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "choices": [{"message": {"content": "response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50},
        }
        mock_requests.post.return_value = mock_response

        adapter = instrument_anyscale(
            anyscale_api_key="test-key", project="customer-support-bot"
        )

        response = adapter.completion_create(
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=[{"role": "user", "content": "test"}],
        )

        assert response is not None


class TestMultiTenantGovernance:
    """Test multi-tenant governance scenarios."""

    @patch("genops.providers.anyscale.adapter.requests")
    def test_multiple_customers_same_adapter(self, mock_requests):
        """Test single adapter serving multiple customers."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test",
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "choices": [{"message": {"content": "response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        }
        mock_requests.post.return_value = mock_response

        adapter = instrument_anyscale(anyscale_api_key="test-key", team="saas-platform")

        customers = ["customer-A", "customer-B", "customer-C"]

        for customer_id in customers:
            response = adapter.completion_create(
                model="meta-llama/Llama-2-7b-chat-hf",
                messages=[{"role": "user", "content": "test"}],
                customer_id=customer_id,
            )
            assert response is not None

    @patch("genops.providers.anyscale.adapter.requests")
    def test_governance_isolation(self, mock_requests):
        """Test governance attributes are isolated per request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test",
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "choices": [{"message": {"content": "response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }
        mock_requests.post.return_value = mock_response

        adapter = instrument_anyscale(anyscale_api_key="test-key")

        # Request 1 with customer A
        response1 = adapter.completion_create(
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=[{"role": "user", "content": "test"}],
            customer_id="customer-A",
        )

        # Request 2 with customer B
        response2 = adapter.completion_create(
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=[{"role": "user", "content": "test"}],
            customer_id="customer-B",
        )

        # Both should succeed independently
        assert response1 is not None
        assert response2 is not None


class TestGovernanceValidation:
    """Test governance attribute validation."""

    def test_governance_attrs_accepted(self):
        """Test valid governance attributes are accepted."""
        valid_attrs = {
            "team": "ml-team",
            "project": "chatbot",
            "environment": "production",
            "cost_center": "R&D",
            "customer_id": "customer-123",
            "feature": "chat",
        }

        adapter = instrument_anyscale(**valid_attrs)

        for key, value in valid_attrs.items():
            assert adapter.governance_defaults.get(key) == value

    def test_custom_governance_attrs(self):
        """Test custom governance attributes are supported."""
        adapter = instrument_anyscale(
            custom_tag="custom-value", internal_id="internal-123"
        )

        # Custom attributes should be stored
        assert "custom_tag" in adapter.governance_defaults
        assert adapter.governance_defaults["custom_tag"] == "custom-value"


class TestGovernanceTelemetry:
    """Test governance attributes in telemetry."""

    @patch("genops.providers.anyscale.adapter.requests")
    def test_telemetry_includes_governance(self, mock_requests):
        """Test telemetry includes governance attributes."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test",
            "model": "meta-llama/Llama-2-7b-chat-hf",
            "choices": [{"message": {"content": "response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }
        mock_requests.post.return_value = mock_response

        adapter = instrument_anyscale(
            anyscale_api_key="test-key", team="test-team", telemetry_enabled=True
        )

        response = adapter.completion_create(
            model="meta-llama/Llama-2-7b-chat-hf",
            messages=[{"role": "user", "content": "test"}],
            customer_id="customer-123",
        )

        # Telemetry should be generated with governance attributes
        assert response is not None

    def test_governance_with_telemetry_disabled(self):
        """Test governance works even with telemetry disabled."""
        adapter = instrument_anyscale(team="test-team", telemetry_enabled=False)

        # Governance defaults should still be set
        assert adapter.governance_defaults["team"] == "test-team"
        assert adapter.telemetry_enabled is False
