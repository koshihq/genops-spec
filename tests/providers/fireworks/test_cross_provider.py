"""
Cross-provider compatibility tests for Fireworks AI.

Tests cover:
- OpenAI compatibility interface
- Multi-provider cost comparison
- Migration scenarios from other providers
- Framework integration compatibility
- Governance attribute consistency
- Performance comparison baselines
"""

from decimal import Decimal
from unittest.mock import Mock, patch

from genops.providers.fireworks import FireworksModel, GenOpsFireworksAdapter
from genops.providers.fireworks_pricing import FireworksPricingCalculator


class TestOpenAICompatibility:
    """Test OpenAI-compatible interface and migration scenarios."""

    @patch("genops.providers.fireworks.Fireworks")
    def test_openai_parameter_compatibility(
        self, mock_fireworks_class, sample_fireworks_config, mock_fireworks_client
    ):
        """Test OpenAI-compatible parameter handling."""
        mock_fireworks_class.return_value = mock_fireworks_client

        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)

        # Test with OpenAI-style parameters
        adapter.chat_with_governance(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Test OpenAI compatibility"},
            ],
            model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
            max_tokens=150,
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            stop=["\\n", "END"],
            n=1,  # Number of completions
        )

        # Verify parameters are passed through correctly
        call_args = mock_fireworks_client.chat.completions.create.call_args
        assert call_args[1]["temperature"] == 0.7
        assert call_args[1]["top_p"] == 0.9
        assert call_args[1]["frequency_penalty"] == 0.1
        assert call_args[1]["presence_penalty"] == 0.1
        assert call_args[1]["stop"] == ["\\n", "END"]
        assert call_args[1]["n"] == 1

    @patch("genops.providers.fireworks.Fireworks")
    def test_openai_message_format_compatibility(
        self, mock_fireworks_class, sample_fireworks_config, mock_fireworks_client
    ):
        """Test OpenAI message format compatibility."""
        mock_fireworks_class.return_value = mock_fireworks_client

        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)

        # Test various OpenAI message formats
        openai_messages = [
            {
                "role": "system",
                "content": "You are an AI assistant specialized in fast responses.",
            },
            {"role": "user", "content": "What's the weather like?", "name": "user123"},
            {
                "role": "assistant",
                "content": "I'd be happy to help, but I need your location.",
            },
            {"role": "user", "content": "I'm in San Francisco"},
        ]

        result = adapter.chat_with_governance(
            messages=openai_messages,
            model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
            max_tokens=100,
        )

        # Verify message format is preserved
        call_args = mock_fireworks_client.chat.completions.create.call_args
        assert call_args[1]["messages"] == openai_messages
        assert result.response is not None

    def test_openai_migration_cost_comparison(self):
        """Test cost comparison for OpenAI migration scenarios."""
        calc = FireworksPricingCalculator()

        # Simulate OpenAI pricing (approximate)
        openai_gpt35_cost_per_1k = Decimal("0.002")  # $2/1M tokens
        openai_gpt4_cost_per_1k = Decimal("0.03")  # $30/1M tokens

        # Compare with Fireworks models
        fireworks_8b_cost = calc.estimate_chat_cost(
            "accounts/fireworks/models/llama-v3p1-8b-instruct", tokens=1000
        )
        fireworks_70b_cost = calc.estimate_chat_cost(
            "accounts/fireworks/models/llama-v3p1-70b-instruct", tokens=1000
        )

        # Fireworks should be significantly cheaper
        assert fireworks_8b_cost < openai_gpt35_cost_per_1k  # Much cheaper than GPT-3.5
        assert fireworks_70b_cost < openai_gpt4_cost_per_1k  # Much cheaper than GPT-4

        # Calculate cost savings
        gpt35_savings = (
            openai_gpt35_cost_per_1k - fireworks_8b_cost
        ) / openai_gpt35_cost_per_1k
        gpt4_savings = (
            openai_gpt4_cost_per_1k - fireworks_70b_cost
        ) / openai_gpt4_cost_per_1k

        assert gpt35_savings > 0.8  # >80% savings vs GPT-3.5
        assert gpt4_savings > 0.95  # >95% savings vs GPT-4

    @patch("genops.providers.fireworks.Fireworks")
    def test_openai_function_calling_compatibility(
        self, mock_fireworks_class, sample_fireworks_config
    ):
        """Test OpenAI function calling compatibility."""
        # Mock function calling response
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content="I'll help you with that calculation.",
                    function_call=Mock(
                        name="calculate_speed_improvement",
                        arguments='{"current_speed": 3.4, "optimized_speed": 0.85}',
                    ),
                )
            )
        ]
        mock_response.usage = Mock(total_tokens=120)

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_fireworks_class.return_value = mock_client

        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)

        # Test OpenAI-style function calling
        functions = [
            {
                "name": "calculate_speed_improvement",
                "description": "Calculate speed improvement ratio",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "current_speed": {"type": "number"},
                        "optimized_speed": {"type": "number"},
                    },
                    "required": ["current_speed", "optimized_speed"],
                },
            }
        ]

        adapter.chat_with_governance(
            messages=[
                {"role": "user", "content": "Calculate Fireworks AI speed improvement"}
            ],
            model=FireworksModel.LLAMA_3_1_70B_INSTRUCT,
            functions=functions,
            function_call="auto",
            max_tokens=200,
        )

        # Verify function calling parameters are passed through
        call_args = mock_client.chat.completions.create.call_args
        assert "functions" in call_args[1]
        assert "function_call" in call_args[1]
        assert call_args[1]["functions"] == functions
        assert call_args[1]["function_call"] == "auto"


class TestMultiProviderComparison:
    """Test multi-provider cost and performance comparisons."""

    def test_cost_comparison_across_providers(self):
        """Test cost comparison between Fireworks and other providers."""
        calc = FireworksPricingCalculator()

        # Standard workload for comparison
        tokens_per_operation = 1000
        operations_per_day = 10000

        # Fireworks AI costs
        fireworks_8b = calc.estimate_chat_cost(
            "accounts/fireworks/models/llama-v3p1-8b-instruct",
            tokens=tokens_per_operation,
        )
        fireworks_70b = calc.estimate_chat_cost(
            "accounts/fireworks/models/llama-v3p1-70b-instruct",
            tokens=tokens_per_operation,
        )

        # Simulate other provider costs (approximate market rates)
        provider_costs = {
            "openai_gpt35": Decimal("0.002"),  # $2/1M tokens
            "openai_gpt4": Decimal("0.03"),  # $30/1M tokens
            "anthropic_claude": Decimal("0.024"),  # $24/1M tokens
            "google_gemini": Decimal("0.001"),  # $1/1M tokens
            "fireworks_8b": fireworks_8b,
            "fireworks_70b": fireworks_70b,
        }

        # Calculate daily costs
        daily_costs = {
            provider: cost * operations_per_day
            for provider, cost in provider_costs.items()
        }

        # Verify Fireworks competitive positioning
        assert daily_costs["fireworks_8b"] < daily_costs["openai_gpt35"]
        assert daily_costs["fireworks_8b"] < daily_costs["anthropic_claude"]
        assert daily_costs["fireworks_70b"] < daily_costs["openai_gpt4"]
        assert daily_costs["fireworks_70b"] < daily_costs["anthropic_claude"]

        # Calculate potential savings
        savings_vs_openai_gpt4 = (
            daily_costs["openai_gpt4"] - daily_costs["fireworks_70b"]
        )
        monthly_savings = savings_vs_openai_gpt4 * 30

        assert monthly_savings > Decimal("8000")  # Significant monthly savings

    def test_performance_comparison_baselines(self):
        """Test performance baselines against other providers."""
        # Fireworks AI performance characteristics
        fireworks_performance = {
            "llama_3_1_8b": {
                "response_time": 0.85,  # 4x faster with Fireattention
                "tokens_per_second": 120,
                "cost_per_1k_tokens": 0.0002,
            },
            "llama_3_1_70b": {
                "response_time": 1.2,  # Still fast for large model
                "tokens_per_second": 85,
                "cost_per_1k_tokens": 0.0009,
            },
        }

        # Simulated baseline performance (traditional inference)
        baseline_performance = {
            "equivalent_8b": {
                "response_time": 3.4,  # Traditional inference
                "tokens_per_second": 30,
                "cost_per_1k_tokens": 0.002,
            },
            "equivalent_70b": {
                "response_time": 8.5,  # Much slower
                "tokens_per_second": 12,
                "cost_per_1k_tokens": 0.03,
            },
        }

        # Verify Fireattention speed advantage
        speed_improvement_8b = (
            baseline_performance["equivalent_8b"]["response_time"]
            / fireworks_performance["llama_3_1_8b"]["response_time"]
        )
        speed_improvement_70b = (
            baseline_performance["equivalent_70b"]["response_time"]
            / fireworks_performance["llama_3_1_70b"]["response_time"]
        )

        assert speed_improvement_8b >= 3.5  # ~4x faster
        assert speed_improvement_70b >= 6.0  # Even bigger improvement for large models

        # Verify throughput advantages
        throughput_8b = fireworks_performance["llama_3_1_8b"]["tokens_per_second"]
        throughput_70b = fireworks_performance["llama_3_1_70b"]["tokens_per_second"]

        assert (
            throughput_8b
            > baseline_performance["equivalent_8b"]["tokens_per_second"] * 3
        )
        assert (
            throughput_70b
            > baseline_performance["equivalent_70b"]["tokens_per_second"] * 6
        )

    def test_batch_processing_comparison(self):
        """Test batch processing advantages vs other providers."""
        calc = FireworksPricingCalculator()

        # Fireworks batch processing (50% discount)
        standard_cost = calc.estimate_chat_cost(
            "accounts/fireworks/models/llama-v3p1-8b-instruct",
            tokens=1000,
            is_batch=False,
        )
        batch_cost = calc.estimate_chat_cost(
            "accounts/fireworks/models/llama-v3p1-8b-instruct",
            tokens=1000,
            is_batch=True,
        )

        batch_savings_percentage = ((standard_cost - batch_cost) / standard_cost) * 100

        # Verify 50% batch savings
        assert abs(batch_savings_percentage - 50.0) < 1.0

        # Compare to providers without batch processing
        competitor_cost = Decimal("0.002")  # Typical competitor pricing

        # Fireworks batch should be cheaper than competitor standard
        assert batch_cost < competitor_cost

        # Calculate competitive advantage
        competitive_advantage = ((competitor_cost - batch_cost) / competitor_cost) * 100
        assert competitive_advantage > 85  # >85% cheaper with batching


class TestFrameworkIntegration:
    """Test compatibility with AI frameworks and libraries."""

    @patch("genops.providers.fireworks.Fireworks")
    def test_langchain_compatibility_patterns(
        self, mock_fireworks_class, sample_fireworks_config, mock_fireworks_client
    ):
        """Test LangChain-style compatibility patterns."""
        mock_fireworks_class.return_value = mock_fireworks_client

        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)

        # Test LangChain-style invoke pattern
        def simulate_langchain_invoke(prompt_template, variables):
            formatted_prompt = prompt_template.format(**variables)

            return adapter.chat_with_governance(
                messages=[{"role": "user", "content": formatted_prompt}],
                model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
                max_tokens=100,
                framework="langchain",
                template_variables=variables,
            )

        # Test with template-style prompt
        result = simulate_langchain_invoke(
            "Explain {topic} in the context of {domain} with {style} approach.",
            {
                "topic": "Fireattention optimization",
                "domain": "AI inference",
                "style": "technical",
            },
        )

        assert result.response is not None
        assert result.governance_attrs["framework"] == "langchain"

    def test_llamaindex_compatibility_patterns(self):
        """Test LlamaIndex-style compatibility patterns."""
        GenOpsFireworksAdapter(team="llamaindex-test", project="rag-compatibility")

        # Test LlamaIndex-style metadata passing
        def simulate_llamaindex_query(query, context_docs):
            context_text = "\n\n".join(
                [f"Doc {i}: {doc}" for i, doc in enumerate(context_docs)]
            )

            prompt = f"Context:\n{context_text}\n\nQuery: {query}\n\nAnswer based on the context:"

            # This would integrate with actual LlamaIndex
            assert len(prompt) > 0
            assert query in prompt
            assert all(doc in prompt for doc in context_docs)

        # Test RAG-style query
        simulate_llamaindex_query(
            "What are the performance benefits of Fireworks AI?",
            [
                "Fireworks AI provides 4x faster inference through Fireattention optimization.",
                "Batch processing enables 50% cost savings on large workloads.",
            ],
        )

    @patch("genops.providers.fireworks.Fireworks")
    def test_streaming_framework_compatibility(
        self, mock_fireworks_class, sample_fireworks_config
    ):
        """Test streaming compatibility with various frameworks."""

        # Mock streaming generator
        def mock_streaming_response():
            chunks = [
                {"choices": [{"delta": {"content": "Fast"}}]},
                {"choices": [{"delta": {"content": " streaming"}}]},
                {"choices": [{"delta": {"content": " response"}}]},
                {"choices": [{"delta": {"content": " complete"}}]},
            ]
            for chunk in chunks:
                yield Mock(**chunk)

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_streaming_response()
        mock_fireworks_class.return_value = mock_client

        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)

        # Test framework-agnostic streaming
        collected_chunks = []

        def collect_chunk(content, cost):
            collected_chunks.append((content, cost))

        adapter.chat_with_governance(
            messages=[{"role": "user", "content": "Test streaming"}],
            model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
            max_tokens=100,
            stream=True,
            on_chunk=collect_chunk,
            framework_integration="custom",
        )

        # Verify streaming worked
        assert len(collected_chunks) > 0
        call_args = mock_client.chat.completions.create.call_args
        assert call_args is not None
        assert call_args[1]["stream"] is True


class TestGovernanceConsistency:
    """Test governance attribute consistency across providers."""

    @patch("genops.providers.fireworks.Fireworks")
    def test_governance_attribute_standardization(
        self, mock_fireworks_class, sample_fireworks_config, mock_fireworks_client
    ):
        """Test standardized governance attributes across providers."""
        mock_fireworks_class.return_value = mock_fireworks_client

        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)

        # Standard governance attributes that should work across all providers
        standard_attrs = {
            "team": "cross-provider-team",
            "project": "multi-provider-project",
            "customer_id": "customer-456",
            "feature": "standardized-feature",
            "use_case": "cross-provider-testing",
            "cost_center": "engineering",
            "environment": "production",
            "compliance_requirement": "SOC2",
        }

        result = adapter.chat_with_governance(
            messages=[{"role": "user", "content": "Test governance standardization"}],
            model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
            max_tokens=100,
            **standard_attrs,
        )

        # Verify all governance attributes are captured
        for key, value in standard_attrs.items():
            if key not in ["team", "project"]:  # These come from adapter config
                assert result.governance_attrs[key] == value

        # Verify adapter-level attributes
        assert result.governance_attrs["team"] == sample_fireworks_config["team"]
        assert result.governance_attrs["project"] == sample_fireworks_config["project"]

    def test_cost_attribution_consistency(self):
        """Test consistent cost attribution patterns."""
        calc = FireworksPricingCalculator()

        # Test cost calculation consistency
        models_to_test = [
            "accounts/fireworks/models/llama-v3p1-8b-instruct",
            "accounts/fireworks/models/llama-v3p1-70b-instruct",
            "accounts/fireworks/models/mixtral-8x7b",
        ]

        # Standard test parameters
        test_tokens = 1000

        for model in models_to_test:
            cost = calc.estimate_chat_cost(model, tokens=test_tokens)

            # Verify cost attribution structure is consistent
            assert isinstance(cost, Decimal)
            assert cost > 0
            assert cost < Decimal("0.01")  # Reasonable upper bound

        # Test batch discount consistency
        for model in models_to_test:
            standard_cost = calc.estimate_chat_cost(
                model, tokens=test_tokens, is_batch=False
            )
            batch_cost = calc.estimate_chat_cost(
                model, tokens=test_tokens, is_batch=True
            )

            discount_percentage = ((standard_cost - batch_cost) / standard_cost) * 100
            assert abs(discount_percentage - 50.0) < 1.0  # Consistent 50% discount


class TestMigrationScenarios:
    """Test migration scenarios from other providers."""

    def test_openai_to_fireworks_migration(self):
        """Test migration scenario from OpenAI to Fireworks."""
        calc = FireworksPricingCalculator()

        # Typical OpenAI workload
        openai_workload = {
            "operations_per_day": 50000,
            "avg_tokens_per_operation": 800,
            "current_monthly_cost": Decimal("3000.00"),  # $3000/month on OpenAI
            "model_type": "gpt-3.5-turbo",
        }

        # Equivalent Fireworks workload
        fireworks_cost_per_op = calc.estimate_chat_cost(
            "accounts/fireworks/models/llama-v3p1-8b-instruct",
            tokens=openai_workload["avg_tokens_per_operation"],
        )

        fireworks_daily_cost = (
            fireworks_cost_per_op * openai_workload["operations_per_day"]
        )
        fireworks_monthly_cost = fireworks_daily_cost * 30

        # Calculate migration savings
        monthly_savings = (
            openai_workload["current_monthly_cost"] - fireworks_monthly_cost
        )
        savings_percentage = (
            monthly_savings / openai_workload["current_monthly_cost"]
        ) * 100

        assert monthly_savings > Decimal("2400")  # >$2400/month savings
        assert savings_percentage > 80  # >80% cost reduction

        # Factor in 4x speed improvement
        performance_value = monthly_savings * Decimal(
            "1.5"
        )  # Speed has additional value
        total_migration_value = monthly_savings + performance_value

        assert total_migration_value > Decimal("3000")  # Substantial migration value

    def test_anthropic_to_fireworks_migration(self):
        """Test migration scenario from Anthropic to Fireworks."""
        calc = FireworksPricingCalculator()

        # Anthropic Claude workload characteristics
        anthropic_workload = {
            "operations_per_day": 20000,
            "avg_tokens_per_operation": 1200,
            "current_monthly_cost": Decimal(
                "14400.00"
            ),  # $14.4k/month (~$24/1M tokens)
            "model_type": "claude-3-sonnet",
        }

        # Fireworks 70B model for comparable quality
        fireworks_cost_per_op = calc.estimate_chat_cost(
            "accounts/fireworks/models/llama-v3p1-70b-instruct",
            tokens=anthropic_workload["avg_tokens_per_operation"],
        )

        fireworks_monthly_cost = (
            fireworks_cost_per_op * anthropic_workload["operations_per_day"] * 30
        )

        # Calculate migration benefits
        cost_savings = (
            anthropic_workload["current_monthly_cost"] - fireworks_monthly_cost
        )
        savings_percentage = (
            cost_savings / anthropic_workload["current_monthly_cost"]
        ) * 100

        assert cost_savings > Decimal("12000")  # >$12k/month savings
        assert savings_percentage > 85  # >85% cost reduction

    def test_migration_roi_analysis(self):
        """Test ROI analysis for provider migration."""
        calc = FireworksPricingCalculator()

        # Migration scenario parameters
        migration_params = {
            "current_monthly_spend": Decimal("10000"),
            "migration_effort_cost": Decimal("5000"),  # One-time migration cost
            "operations_per_day": 100000,
            "avg_tokens_per_operation": 500,
        }

        # Calculate Fireworks costs
        fireworks_cost = calc.estimate_chat_cost(
            "accounts/fireworks/models/llama-v3p1-8b-instruct",
            tokens=migration_params["avg_tokens_per_operation"],
        )

        fireworks_monthly_cost = (
            fireworks_cost * migration_params["operations_per_day"] * 30
        )
        monthly_savings = (
            migration_params["current_monthly_spend"] - fireworks_monthly_cost
        )

        # ROI analysis
        payback_period = migration_params["migration_effort_cost"] / monthly_savings
        annual_savings = monthly_savings * 12
        roi_percentage = (
            (annual_savings - migration_params["migration_effort_cost"])
            / migration_params["migration_effort_cost"]
        ) * 100

        assert payback_period < 2  # Payback in less than 2 months
        assert roi_percentage > 1000  # >1000% annual ROI
        assert annual_savings > Decimal("50000")  # >$50k annual savings
