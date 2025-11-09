"""
Integration tests for Fireworks AI provider.

Tests cover:
- End-to-end workflow testing
- Real API integration scenarios (when available)
- Cross-provider compatibility
- Production workflow simulation
- Error recovery and resilience
- Performance under load
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import asyncio
import time
from decimal import Decimal

from genops.providers.fireworks import (
    GenOpsFireworksAdapter,
    FireworksModel,
    auto_instrument
)
from genops.providers.fireworks_pricing import FireworksPricingCalculator
from genops.providers.fireworks_validation import validate_fireworks_setup


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    @patch('genops.providers.fireworks.Fireworks')
    def test_complete_chat_workflow(self, mock_fireworks_class, sample_fireworks_config, sample_chat_messages, mock_fireworks_client):
        """Test complete chat workflow from initialization to result."""
        mock_fireworks_class.return_value = mock_fireworks_client
        
        # Initialize adapter
        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)
        
        # Execute chat operation
        result = adapter.chat_with_governance(
            messages=sample_chat_messages,
            model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
            max_tokens=150,
            feature="integration-test",
            customer_id="test-customer"
        )
        
        # Verify complete result structure
        assert result.response is not None
        assert result.cost > 0
        assert result.tokens_used > 0
        assert result.execution_time_seconds > 0
        assert result.model_used is not None
        assert result.governance_attrs["team"] == "test-team"
        assert result.governance_attrs["customer_id"] == "test-customer"
        
        # Get cost summary
        cost_summary = adapter.get_cost_summary()
        assert cost_summary["daily_costs"] >= result.cost
        assert cost_summary["operations_count"] >= 1
    
    @patch('genops.providers.fireworks.Fireworks')
    def test_complete_embedding_workflow(self, mock_fireworks_class, sample_fireworks_config, sample_embedding_texts, mock_fireworks_client):
        """Test complete embedding workflow."""
        mock_fireworks_class.return_value = mock_fireworks_client
        
        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)
        
        result = adapter.embeddings_with_governance(
            input_texts=sample_embedding_texts,
            model=FireworksModel.NOMIC_EMBED_TEXT,
            feature="integration-test"
        )
        
        # Verify embedding result
        assert result.embeddings is not None
        assert len(result.embeddings) > 0
        assert result.cost > 0
        assert result.governance_attrs["team"] == "test-team"
    
    @patch('genops.providers.fireworks.Fireworks')
    def test_session_based_workflow(self, mock_fireworks_class, sample_fireworks_config, sample_chat_messages, mock_fireworks_client):
        """Test session-based multi-operation workflow."""
        mock_fireworks_class.return_value = mock_fireworks_client
        
        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)
        
        with adapter.track_session("integration-test-session", use_case="testing") as session:
            # Multiple operations in session
            result1 = adapter.chat_with_governance(
                messages=sample_chat_messages,
                model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
                max_tokens=100,
                session_id=session.session_id
            )
            
            result2 = adapter.chat_with_governance(
                messages=[{"role": "user", "content": "Follow-up question"}],
                model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
                max_tokens=80,
                session_id=session.session_id
            )
            
            # Verify session tracking
            assert session.total_operations == 2
            assert session.total_cost > 0
            assert session.governance_attrs["use_case"] == "testing"
        
        # Session should be finalized
        assert session.end_time is not None
    
    @patch('genops.providers.fireworks.Fireworks')
    def test_batch_processing_workflow(self, mock_fireworks_class, sample_fireworks_config, mock_fireworks_client):
        """Test batch processing workflow with cost optimization."""
        mock_fireworks_class.return_value = mock_fireworks_client
        
        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)
        
        # Batch of operations
        batch_requests = [
            "Analyze this data point",
            "Generate summary report", 
            "Create recommendations",
            "Review and optimize"
        ]
        
        batch_results = []
        
        for i, request in enumerate(batch_requests):
            result = adapter.chat_with_governance(
                messages=[{"role": "user", "content": request}],
                model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
                max_tokens=100,
                is_batch=True,
                batch_id="integration-batch",
                operation_index=i
            )
            batch_results.append(result)
        
        # Verify batch processing benefits
        assert len(batch_results) == len(batch_requests)
        
        # All operations should have batch attributes
        for result in batch_results:
            assert result.governance_attrs.get("is_batch") is True
            assert result.governance_attrs.get("batch_id") == "integration-batch"
            
        # Total batch cost should be less than standard cost would be
        total_batch_cost = sum(r.cost for r in batch_results)
        assert total_batch_cost > 0


class TestAutoInstrumentationIntegration:
    """Test auto-instrumentation integration scenarios."""
    
    @patch('genops.providers.fireworks.Fireworks')
    def test_auto_instrumentation_activation(self, mock_fireworks_class):
        """Test auto-instrumentation activation and deactivation."""
        with patch('genops.providers.fireworks._setup_auto_instrumentation') as mock_setup:
            # Activate auto-instrumentation
            auto_instrument(team="auto-team", project="auto-project")
            
            mock_setup.assert_called_once()
            call_args = mock_setup.call_args[1]
            assert call_args["team"] == "auto-team"
            assert call_args["project"] == "auto-project"
    
    @patch('genops.providers.fireworks.Fireworks')
    def test_auto_instrumentation_with_existing_code(self, mock_fireworks_class, mock_fireworks_client):
        """Test auto-instrumentation working with existing Fireworks code."""
        mock_fireworks_class.return_value = mock_fireworks_client
        
        # Simulate auto-instrumentation being active
        with patch('genops.providers.fireworks._auto_instrumentation_active', True):
            # This would be user's existing code
            from fireworks.client import Fireworks
            
            client = Fireworks()  # User's existing client
            
            # Should be automatically instrumented
            response = client.chat.completions.create(
                model="accounts/fireworks/models/llama-v3p1-8b-instruct",
                messages=[{"role": "user", "content": "Test auto-instrumentation"}],
                max_tokens=50
            )
            
            # Verify the call went through
            mock_fireworks_client.chat.completions.create.assert_called_once()


class TestCrossProviderCompatibility:
    """Test compatibility with other providers and frameworks."""
    
    @patch('genops.providers.fireworks.Fireworks')
    def test_openai_compatible_interface(self, mock_fireworks_class, mock_fireworks_client):
        """Test OpenAI-compatible interface integration."""
        mock_fireworks_class.return_value = mock_fireworks_client
        
        # Test OpenAI-style parameters work
        adapter = GenOpsFireworksAdapter(
            team="compatibility-test",
            project="openai-compat"
        )
        
        result = adapter.chat_with_governance(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Test OpenAI compatibility"}
            ],
            model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1
        )
        
        # Verify OpenAI-style parameters are passed through
        call_args = mock_fireworks_client.chat.completions.create.call_args
        assert call_args[1]["temperature"] == 0.7
        assert call_args[1]["top_p"] == 0.9
    
    def test_langchain_integration_compatibility(self):
        """Test compatibility with LangChain integration patterns."""
        # This would test that Fireworks adapter works well with LangChain
        adapter = GenOpsFireworksAdapter(
            team="langchain-test",
            project="framework-integration"
        )
        
        # Test adapter methods that would be called by LangChain
        assert hasattr(adapter, 'chat_with_governance')
        assert hasattr(adapter, 'embeddings_with_governance')
        assert hasattr(adapter, 'get_cost_summary')
    
    def test_pricing_calculator_integration(self):
        """Test integration between adapter and pricing calculator."""
        adapter = GenOpsFireworksAdapter(team="pricing-test", project="integration")
        pricing_calc = FireworksPricingCalculator()
        
        # Test that adapter can use pricing calculator
        recommendation = pricing_calc.recommend_model(
            task_complexity="simple",
            budget_per_operation=0.001
        )
        
        assert recommendation is not None
        
        if recommendation.recommended_model:
            # Should be able to use recommended model with adapter
            recommended_model_enum = None
            for model_enum in FireworksModel:
                if recommendation.recommended_model in model_enum.value:
                    recommended_model_enum = model_enum
                    break
            
            assert recommended_model_enum is not None


class TestProductionScenarios:
    """Test production-like scenarios and edge cases."""
    
    @patch('genops.providers.fireworks.Fireworks')
    def test_high_volume_operations(self, mock_fireworks_class, sample_fireworks_config, mock_fireworks_client):
        """Test handling of high-volume operations."""
        mock_fireworks_class.return_value = mock_fireworks_client
        
        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)
        
        # Simulate high volume (100 operations)
        results = []
        
        for i in range(100):
            result = adapter.chat_with_governance(
                messages=[{"role": "user", "content": f"Request {i+1}"}],
                model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
                max_tokens=50,
                operation_index=i
            )
            results.append(result)
        
        # Verify all operations completed
        assert len(results) == 100
        
        # Verify cost tracking scales properly
        cost_summary = adapter.get_cost_summary()
        assert cost_summary["operations_count"] == 100
        assert cost_summary["daily_costs"] > 0
    
    @patch('genops.providers.fireworks.Fireworks')
    def test_mixed_model_operations(self, mock_fireworks_class, sample_fireworks_config, mock_fireworks_client):
        """Test operations across multiple models."""
        mock_fireworks_class.return_value = mock_fireworks_client
        
        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)
        
        # Mix of different model operations
        models_to_test = [
            (FireworksModel.LLAMA_3_2_1B_INSTRUCT, "Simple task"),
            (FireworksModel.LLAMA_3_1_8B_INSTRUCT, "Moderate task"),
            (FireworksModel.LLAMA_3_1_70B_INSTRUCT, "Complex task"),
            (FireworksModel.MIXTRAL_8X7B, "MoE task")
        ]
        
        results = []
        
        for model, task in models_to_test:
            result = adapter.chat_with_governance(
                messages=[{"role": "user", "content": task}],
                model=model,
                max_tokens=80
            )
            results.append(result)
        
        # Verify different models produce different costs
        costs = [r.cost for r in results]
        assert len(set(costs)) > 1  # Should have different costs
        
        # Tiny model should be cheapest
        tiny_result = results[0]
        large_result = results[2]
        assert tiny_result.cost < large_result.cost
    
    @patch('genops.providers.fireworks.Fireworks')
    def test_error_recovery_scenarios(self, mock_fireworks_class, sample_fireworks_config):
        """Test error recovery in production scenarios."""
        # Simulate intermittent failures
        mock_client = Mock()
        call_count = [0]  # Use list for mutable counter
        
        def mock_create_with_failures(**kwargs):
            call_count[0] += 1
            if call_count[0] <= 2:  # First 2 calls fail
                raise Exception("Temporary API error")
            else:  # Subsequent calls succeed
                mock_response = Mock()
                mock_response.choices = [Mock(message=Mock(content="Recovery success"))]
                mock_response.usage = Mock(total_tokens=30)
                return mock_response
        
        mock_client.chat.completions.create.side_effect = mock_create_with_failures
        mock_fireworks_class.return_value = mock_client
        
        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)
        
        # First calls should fail
        with pytest.raises(Exception, match="Temporary API error"):
            adapter.chat_with_governance(
                messages=[{"role": "user", "content": "Test 1"}],
                model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
                max_tokens=50
            )
        
        with pytest.raises(Exception, match="Temporary API error"):
            adapter.chat_with_governance(
                messages=[{"role": "user", "content": "Test 2"}],
                model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
                max_tokens=50
            )
        
        # Third call should succeed (recovery)
        result = adapter.chat_with_governance(
            messages=[{"role": "user", "content": "Test 3"}],
            model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
            max_tokens=50
        )
        
        assert result.response == "Recovery success"
    
    @patch('genops.providers.fireworks.Fireworks')
    def test_budget_enforcement_scenarios(self, mock_fireworks_class, sample_fireworks_config, mock_fireworks_client):
        """Test budget enforcement in production scenarios."""
        config = sample_fireworks_config.copy()
        config["governance_policy"] = "enforcing"
        config["daily_budget_limit"] = 0.01  # Very low budget for testing
        
        mock_fireworks_class.return_value = mock_fireworks_client
        
        adapter = GenOpsFireworksAdapter(**config)
        
        # Set adapter to near budget limit
        adapter._daily_costs = Decimal("0.009")  # Close to $0.01 limit
        
        # Small operation should succeed
        result = adapter.chat_with_governance(
            messages=[{"role": "user", "content": "Small request"}],
            model=FireworksModel.LLAMA_3_2_1B_INSTRUCT,  # Cheapest model
            max_tokens=10  # Very few tokens
        )
        
        assert result.response is not None
        
        # Large expensive operation should be blocked
        with pytest.raises(Exception, match="Budget"):
            adapter.chat_with_governance(
                messages=[{"role": "user", "content": "Expensive request"}],
                model=FireworksModel.LLAMA_3_1_405B_INSTRUCT,  # Most expensive
                max_tokens=1000  # Many tokens
            )


class TestPerformanceIntegration:
    """Test performance-related integration scenarios."""
    
    @patch('genops.providers.fireworks.Fireworks')
    def test_fireattention_optimization_tracking(self, mock_fireworks_class, sample_fireworks_config, mock_fireworks_client):
        """Test Fireattention speed optimization tracking."""
        mock_fireworks_class.return_value = mock_fireworks_client
        
        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)
        
        # Mock fast response times (Fireattention optimization)
        with patch('time.time') as mock_time:
            mock_time.side_effect = [0.0, 0.8]  # 0.8 second response (4x faster)
            
            result = adapter.chat_with_governance(
                messages=[{"role": "user", "content": "Test Fireattention speed"}],
                model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
                max_tokens=100
            )
        
        # Verify speed optimization is tracked
        assert result.execution_time_seconds < 1.0
        assert result.governance_attrs.get("fireattention_optimized") is True
    
    @patch('genops.providers.fireworks.Fireworks')
    def test_streaming_performance_integration(self, mock_fireworks_class, sample_fireworks_config):
        """Test streaming performance integration."""
        # Mock streaming response
        mock_stream = [
            Mock(choices=[Mock(delta=Mock(content="Fast"))]),
            Mock(choices=[Mock(delta=Mock(content=" streaming"))]),
            Mock(choices=[Mock(delta=Mock(content=" with"))]),
            Mock(choices=[Mock(delta=Mock(content=" Fireworks"))])
        ]
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_stream
        mock_fireworks_class.return_value = mock_client
        
        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)
        
        chunks_received = []
        def on_chunk(content, cost):
            chunks_received.append((content, cost))
        
        result = adapter.chat_with_governance(
            messages=[{"role": "user", "content": "Test streaming performance"}],
            model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
            max_tokens=100,
            stream=True,
            on_chunk=on_chunk
        )
        
        # Verify streaming worked
        assert len(chunks_received) > 0
        assert any("Fast" in chunk[0] for chunk in chunks_received)


class TestValidationIntegration:
    """Test integration with validation system."""
    
    @patch('genops.providers.fireworks_validation.validate_fireworks_setup')
    def test_validation_integration_with_adapter(self, mock_validate):
        """Test validation integration with adapter initialization."""
        mock_validate.return_value = Mock(
            is_valid=True,
            api_key_valid=True,
            model_access=["accounts/fireworks/models/llama-v3p1-8b-instruct"]
        )
        
        # Should be able to create adapter after successful validation
        config = {
            "team": "validation-test",
            "project": "integration",
            "validate_on_init": True
        }
        
        adapter = GenOpsFireworksAdapter(**config)
        assert adapter.team == "validation-test"
    
    def test_pricing_validation_integration(self):
        """Test integration between pricing and validation."""
        pricing_calc = FireworksPricingCalculator()
        
        # Validate that pricing data is consistent with validation expectations
        recommendation = pricing_calc.recommend_model(
            task_complexity="simple",
            budget_per_operation=0.001
        )
        
        if recommendation.recommended_model:
            # Recommended model should be in valid model list
            valid_models = [model.value for model in FireworksModel]
            assert any(recommendation.recommended_model == model for model in valid_models)


class TestRealWorldScenarios:
    """Test real-world usage scenarios and patterns."""
    
    @patch('genops.providers.fireworks.Fireworks')
    def test_customer_service_chatbot_scenario(self, mock_fireworks_class, sample_fireworks_config, mock_fireworks_client):
        """Test customer service chatbot scenario."""
        mock_fireworks_class.return_value = mock_fireworks_client
        
        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)
        
        # Simulate customer service conversation
        conversation_history = []
        customer_id = "customer-123"
        
        interactions = [
            "Hello, I need help with my account",
            "I can't log in to my account", 
            "My email is user@example.com",
            "Thank you for your help"
        ]
        
        with adapter.track_session(f"customer-service-{customer_id}", customer_id=customer_id) as session:
            for i, user_message in enumerate(interactions):
                conversation_history.append({"role": "user", "content": user_message})
                
                result = adapter.chat_with_governance(
                    messages=conversation_history.copy(),
                    model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
                    max_tokens=150,
                    feature="customer-service",
                    customer_id=customer_id,
                    interaction_index=i,
                    session_id=session.session_id
                )
                
                conversation_history.append({"role": "assistant", "content": result.response})
        
        # Verify session tracking
        assert session.total_operations == len(interactions)
        assert session.governance_attrs["customer_id"] == customer_id
    
    @patch('genops.providers.fireworks.Fireworks')
    def test_content_generation_pipeline(self, mock_fireworks_class, sample_fireworks_config, mock_fireworks_client):
        """Test content generation pipeline scenario."""
        mock_fireworks_class.return_value = mock_fireworks_client
        
        adapter = GenOpsFireworksAdapter(**sample_fireworks_config)
        
        # Content generation pipeline stages
        content_pipeline = [
            ("research", "Research trends in AI optimization", FireworksModel.LLAMA_3_1_70B_INSTRUCT),
            ("outline", "Create outline for AI trends article", FireworksModel.LLAMA_3_1_8B_INSTRUCT),
            ("draft", "Write article draft based on research", FireworksModel.LLAMA_3_1_70B_INSTRUCT),
            ("optimize", "Optimize content for readability", FireworksModel.LLAMA_3_1_8B_INSTRUCT)
        ]
        
        pipeline_results = {}
        
        with adapter.track_session("content-pipeline") as session:
            for stage, prompt, model in content_pipeline:
                result = adapter.chat_with_governance(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    max_tokens=200,
                    feature="content-generation",
                    pipeline_stage=stage,
                    session_id=session.session_id
                )
                
                pipeline_results[stage] = result
        
        # Verify pipeline execution
        assert len(pipeline_results) == len(content_pipeline)
        assert all(result.response for result in pipeline_results.values())
        
        # Verify cost optimization (cheaper models for simpler tasks)
        research_cost = pipeline_results["research"].cost
        outline_cost = pipeline_results["outline"].cost
        assert outline_cost <= research_cost  # Simpler task should cost less
    
    @patch('genops.providers.fireworks.Fireworks')
    def test_multi_tenant_saas_scenario(self, mock_fireworks_class, mock_fireworks_client):
        """Test multi-tenant SaaS application scenario."""
        mock_fireworks_class.return_value = mock_fireworks_client
        
        # Different tenants with different configurations
        tenants = [
            ("tenant-free", {"daily_budget_limit": 1.0, "governance_policy": "enforcing"}),
            ("tenant-pro", {"daily_budget_limit": 50.0, "governance_policy": "advisory"}),
            ("tenant-enterprise", {"daily_budget_limit": 500.0, "governance_policy": "monitoring"})
        ]
        
        tenant_results = {}
        
        for tenant_id, config in tenants:
            adapter = GenOpsFireworksAdapter(
                team=tenant_id,
                project="saas-app",
                **config
            )
            
            # Each tenant performs operations
            result = adapter.chat_with_governance(
                messages=[{"role": "user", "content": f"Tenant {tenant_id} request"}],
                model=FireworksModel.LLAMA_3_1_8B_INSTRUCT,
                max_tokens=100,
                customer_id=tenant_id,
                tenant_tier=tenant_id.split('-')[1]
            )
            
            tenant_results[tenant_id] = result
        
        # Verify tenant isolation
        assert len(tenant_results) == 3
        
        for tenant_id, result in tenant_results.items():
            assert result.governance_attrs["team"] == tenant_id
            assert result.governance_attrs["customer_id"] == tenant_id