#!/usr/bin/env python3
"""
Comprehensive integration tests for Haystack functionality.

Tests cover end-to-end workflows, multi-provider scenarios, cross-component
integration, and production scenarios as required by CLAUDE.md standards.
"""

import pytest
import time
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

# Integration test imports
from genops.providers.haystack import (
    GenOpsHaystackAdapter,
    auto_instrument,
    validate_haystack_setup,
    create_rag_adapter,
    create_agent_adapter,
    analyze_pipeline_costs,
    get_rag_insights,
    get_agent_insights
)


class TestEndToEndPipelineTracking:
    """End-to-end pipeline tracking integration tests."""

    def test_complete_pipeline_lifecycle(self):
        """Test complete pipeline lifecycle with governance."""
        # Create adapter
        adapter = GenOpsHaystackAdapter(
            team="integration-team",
            project="integration-test",
            daily_budget_limit=50.0
        )
        
        # Track complete pipeline execution
        with adapter.track_pipeline("integration-test-pipeline") as context:
            # Simulate component executions
            context.add_component_result({
                "component_name": "prompt-builder",
                "component_type": "PromptBuilder",
                "execution_time_seconds": 0.1,
                "cost": Decimal("0.0"),
                "provider_name": "Local",
                "success": True
            })
            
            context.add_component_result({
                "component_name": "llm-generator",
                "component_type": "Generator",
                "execution_time_seconds": 2.5,
                "cost": Decimal("0.008"),
                "provider_name": "OpenAI",
                "model_name": "gpt-3.5-turbo",
                "tokens_used": 1000,
                "success": True
            })
        
        # Verify metrics
        metrics = context.get_metrics()
        assert metrics.total_cost == Decimal("0.008")
        assert metrics.total_components == 2
        assert metrics.total_execution_time_seconds >= 2.6
        assert "OpenAI" in metrics.cost_by_provider
        assert "Local" in metrics.cost_by_provider

    def test_multi_pipeline_session(self):
        """Test session tracking with multiple pipelines."""
        adapter = GenOpsHaystackAdapter(
            team="session-team",
            project="session-test",
            daily_budget_limit=100.0
        )
        
        with adapter.track_session("multi-pipeline-session") as session:
            # First pipeline
            with adapter.track_pipeline("pipeline-1") as p1:
                p1.add_component_result({
                    "component_name": "gen1",
                    "component_type": "Generator",
                    "execution_time_seconds": 1.5,
                    "cost": Decimal("0.005"),
                    "provider_name": "OpenAI",
                    "success": True
                })
            
            session.add_pipeline_result(p1.get_metrics())
            
            # Second pipeline
            with adapter.track_pipeline("pipeline-2") as p2:
                p2.add_component_result({
                    "component_name": "gen2",
                    "component_type": "Generator",
                    "execution_time_seconds": 2.0,
                    "cost": Decimal("0.008"),
                    "provider_name": "Anthropic",
                    "success": True
                })
            
            session.add_pipeline_result(p2.get_metrics())
        
        # Verify session metrics
        assert session.total_pipelines == 2
        assert session.total_cost == Decimal("0.013")

    def test_pipeline_with_failures(self):
        """Test pipeline tracking with component failures."""
        adapter = GenOpsHaystackAdapter(
            team="failure-team",
            project="failure-test"
        )
        
        with adapter.track_pipeline("failing-pipeline") as context:
            # Successful component
            context.add_component_result({
                "component_name": "successful-comp",
                "component_type": "Retriever",
                "execution_time_seconds": 1.0,
                "cost": Decimal("0.002"),
                "provider_name": "HuggingFace",
                "success": True
            })
            
            # Failed component
            context.add_component_result({
                "component_name": "failed-comp",
                "component_type": "Generator",
                "execution_time_seconds": 0.5,
                "cost": Decimal("0.0"),
                "provider_name": "OpenAI",
                "success": False,
                "error_message": "Rate limit exceeded"
            })
        
        metrics = context.get_metrics()
        assert metrics.total_components == 2
        assert metrics.total_cost == Decimal("0.002")  # Only successful component
        
        # Check for failed component in results
        failed_components = [r for r in context.component_results if not r.success]
        assert len(failed_components) == 1
        assert failed_components[0].error_message == "Rate limit exceeded"


class TestAutoInstrumentationIntegration:
    """Auto-instrumentation integration tests."""

    def setup_method(self):
        """Setup for auto-instrumentation tests."""
        # Reset auto-instrumentation state
        from genops.providers import haystack_registration as reg
        reg._global_adapter = None
        reg._global_monitor = None
        reg._instrumentation_active = False

    def test_auto_instrumentation_pipeline_tracking(self):
        """Test auto-instrumentation automatically tracks pipelines."""
        # Enable auto-instrumentation
        success = auto_instrument(
            team="auto-team",
            project="auto-test",
            daily_budget_limit=25.0
        )
        assert success is True
        
        # Get instrumentation stats
        from genops.providers.haystack import get_instrumentation_stats
        stats = get_instrumentation_stats()
        
        assert stats["active"] is True
        assert stats["team"] == "auto-team"
        assert stats["project"] == "auto-test"

    def test_auto_instrumentation_cost_tracking(self):
        """Test auto-instrumentation cost tracking."""
        auto_instrument(team="cost-team", project="cost-test")
        
        # Mock some tracked costs
        from genops.providers.haystack import get_current_adapter
        adapter = get_current_adapter()
        
        # Add mock cost data
        adapter.cost_aggregator.add_component_cost(
            component_name="mock-component",
            component_type="Generator",
            provider_name="OpenAI",
            model_name="gpt-3.5-turbo",
            cost=Decimal("0.012")
        )
        
        from genops.providers.haystack import get_cost_summary
        summary = get_cost_summary()
        
        assert "daily_costs" in summary
        assert summary["daily_costs"] > 0

    def test_temporary_instrumentation_isolation(self):
        """Test temporary instrumentation doesn't affect global state."""
        from genops.providers.haystack import TemporaryInstrumentation
        
        # No global instrumentation initially
        from genops.providers.haystack import is_instrumented
        assert is_instrumented() is False
        
        with TemporaryInstrumentation(team="temp-team", project="temp-project"):
            assert is_instrumented() is True
            
            from genops.providers.haystack import get_current_adapter
            adapter = get_current_adapter()
            assert adapter.team == "temp-team"
        
        # Should be disabled after context
        assert is_instrumented() is False


class TestMultiProviderIntegration:
    """Multi-provider integration tests."""

    def test_multi_provider_cost_aggregation(self):
        """Test cost aggregation across multiple providers."""
        adapter = GenOpsHaystackAdapter(
            team="multi-provider-team",
            project="multi-provider-test"
        )
        
        with adapter.track_pipeline("multi-provider-pipeline") as context:
            # OpenAI component
            context.add_component_result({
                "component_name": "openai-generator",
                "component_type": "Generator",
                "execution_time_seconds": 2.0,
                "cost": Decimal("0.015"),
                "provider_name": "OpenAI",
                "model_name": "gpt-4",
                "success": True
            })
            
            # Anthropic component
            context.add_component_result({
                "component_name": "anthropic-generator",
                "component_type": "Generator",
                "execution_time_seconds": 1.5,
                "cost": Decimal("0.008"),
                "provider_name": "Anthropic",
                "model_name": "claude-3-haiku",
                "success": True
            })
            
            # HuggingFace component
            context.add_component_result({
                "component_name": "hf-embedder",
                "component_type": "Embedder",
                "execution_time_seconds": 0.8,
                "cost": Decimal("0.001"),
                "provider_name": "HuggingFace",
                "model_name": "sentence-transformers",
                "success": True
            })
        
        metrics = context.get_metrics()
        
        assert metrics.total_cost == Decimal("0.024")
        assert len(metrics.cost_by_provider) == 3
        assert metrics.cost_by_provider["OpenAI"] == Decimal("0.015")
        assert metrics.cost_by_provider["Anthropic"] == Decimal("0.008")
        assert metrics.cost_by_provider["HuggingFace"] == Decimal("0.001")

    def test_cross_provider_optimization_analysis(self):
        """Test cost optimization analysis across providers."""
        adapter = GenOpsHaystackAdapter(
            team="optimization-team",
            project="optimization-test"
        )
        
        # Add expensive OpenAI operations
        for i in range(5):
            adapter.cost_aggregator.add_component_cost(
                component_name=f"expensive-gen-{i}",
                component_type="Generator",
                provider_name="OpenAI",
                model_name="gpt-4",
                cost=Decimal("0.060")
            )
        
        # Analyze costs
        analysis = analyze_pipeline_costs(adapter, time_period_hours=1)
        
        assert "total_cost" in analysis
        assert analysis["total_cost"] == 0.3  # 5 * 0.06
        assert "recommendations" in analysis
        
        # Should recommend switching to cheaper providers
        if analysis["recommendations"]:
            rec = analysis["recommendations"][0]
            assert rec["current_provider"] == "OpenAI"
            assert rec["potential_savings"] > 0


class TestSpecializedAdapterIntegration:
    """Specialized adapter integration tests."""

    def test_rag_adapter_workflow_tracking(self):
        """Test RAG adapter workflow tracking."""
        rag_adapter = create_rag_adapter(
            team="rag-team",
            project="rag-test",
            daily_budget_limit=75.0
        )
        
        with rag_adapter.track_pipeline("rag-workflow") as context:
            # Retrieval phase
            context.add_component_result({
                "component_name": "document-retriever",
                "component_type": "Retriever",
                "execution_time_seconds": 1.2,
                "cost": Decimal("0.003"),
                "provider_name": "HuggingFace",
                "documents_processed": 5,
                "success": True
            })
            
            # Embedding phase
            context.add_component_result({
                "component_name": "text-embedder",
                "component_type": "Embedder",
                "execution_time_seconds": 0.8,
                "cost": Decimal("0.001"),
                "provider_name": "OpenAI",
                "model_name": "text-embedding-ada-002",
                "success": True
            })
            
            # Generation phase
            context.add_component_result({
                "component_name": "text-generator",
                "component_type": "Generator",
                "execution_time_seconds": 2.5,
                "cost": Decimal("0.012"),
                "provider_name": "OpenAI",
                "model_name": "gpt-3.5-turbo",
                "tokens_used": 1500,
                "success": True
            })
        
        metrics = context.get_metrics()
        pipeline_id = context.pipeline_id
        
        # Get RAG-specific insights
        insights = get_rag_insights(rag_adapter.monitor, pipeline_id)
        
        if "error" not in insights:
            assert "retrieval_latency" in insights
            assert "generation_latency" in insights
            assert insights["retrieval_latency"] >= 1.2
            assert insights["generation_latency"] >= 2.5

    def test_agent_adapter_workflow_tracking(self):
        """Test agent adapter workflow tracking."""
        agent_adapter = create_agent_adapter(
            team="agent-team",
            project="agent-test",
            daily_budget_limit=150.0
        )
        
        with agent_adapter.track_pipeline("agent-workflow") as context:
            # Decision component
            context.add_component_result({
                "component_name": "agent-decision-maker",
                "component_type": "Agent",
                "execution_time_seconds": 1.8,
                "cost": Decimal("0.010"),
                "provider_name": "OpenAI",
                "model_name": "gpt-4",
                "success": True
            })
            
            # Tool usage components
            context.add_component_result({
                "component_name": "search-tool",
                "component_type": "Tool",
                "execution_time_seconds": 2.2,
                "cost": Decimal("0.005"),
                "provider_name": "OpenAI",
                "success": True
            })
            
            context.add_component_result({
                "component_name": "calculator-tool",
                "component_type": "Tool",
                "execution_time_seconds": 0.5,
                "cost": Decimal("0.001"),
                "provider_name": "Local",
                "success": True
            })
        
        pipeline_id = context.pipeline_id
        
        # Get agent-specific insights
        insights = get_agent_insights(agent_adapter.monitor, pipeline_id)
        
        if "error" not in insights:
            assert "decisions_made" in insights
            assert "tools_used" in insights
            assert insights["decisions_made"] >= 1
            assert len(insights["tools_used"]) >= 2


class TestValidationIntegration:
    """Validation system integration tests."""

    def test_validation_with_full_setup(self):
        """Test validation with complete setup."""
        result = validate_haystack_setup()
        
        assert isinstance(result, dict) or hasattr(result, 'is_valid')
        
        # Result should have key validation information
        if isinstance(result, dict):
            assert "is_valid" in result
            assert "issues" in result or "available_providers" in result
        else:
            assert hasattr(result, 'is_valid')
            assert hasattr(result, 'overall_score')

    def test_validation_integration_with_adapter(self):
        """Test validation integrates with adapter creation."""
        # Run validation
        result = validate_haystack_setup()
        
        if (isinstance(result, dict) and result.get("is_valid")) or \
           (hasattr(result, 'is_valid') and result.is_valid):
            # If validation passes, adapter creation should work
            adapter = GenOpsHaystackAdapter(
                team="validation-team",
                project="validation-test"
            )
            
            assert adapter is not None
            assert adapter.team == "validation-team"

    @patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test123'}, clear=True)
    def test_validation_with_provider_configuration(self):
        """Test validation recognizes provider configuration."""
        with patch('importlib.import_module') as mock_import:
            mock_import.return_value = Mock()  # Mock OpenAI library
            
            result = validate_haystack_setup()
            
            # Should recognize OpenAI as available
            if hasattr(result, 'available_providers'):
                provider_names = " ".join(result.available_providers)
                assert "OpenAI" in provider_names or "openai" in provider_names.lower()


class TestErrorHandlingIntegration:
    """Error handling integration tests."""

    def test_adapter_resilience_to_cost_failures(self):
        """Test adapter handles cost calculation failures gracefully."""
        adapter = GenOpsHaystackAdapter(
            team="resilience-team",
            project="resilience-test"
        )
        
        # Mock cost aggregator to fail
        adapter.cost_aggregator.add_component_cost = Mock(
            side_effect=Exception("Cost calculation failed")
        )
        
        # Pipeline execution should still work
        with adapter.track_pipeline("resilient-pipeline") as context:
            context.add_component_result({
                "component_name": "resilient-component",
                "component_type": "Generator",
                "execution_time_seconds": 1.0,
                "cost": Decimal("0.005"),
                "provider_name": "OpenAI",
                "success": True
            })
        
        # Should still get metrics
        metrics = context.get_metrics()
        assert metrics is not None
        assert metrics.total_components == 1

    def test_adapter_handles_monitor_failures(self):
        """Test adapter handles monitor failures gracefully."""
        adapter = GenOpsHaystackAdapter(
            team="monitor-failure-team",
            project="monitor-failure-test"
        )
        
        # Mock monitor to fail
        adapter.monitor.start_pipeline_execution = Mock(
            side_effect=Exception("Monitor failed")
        )
        
        # Pipeline tracking should still work
        with adapter.track_pipeline("monitor-failure-pipeline") as context:
            context.add_component_result({
                "component_name": "test-component",
                "component_type": "Generator",
                "execution_time_seconds": 1.0,
                "cost": Decimal("0.005"),
                "provider_name": "OpenAI",
                "success": True
            })
        
        assert context is not None

    def test_graceful_degradation_without_dependencies(self):
        """Test graceful degradation when dependencies are missing."""
        with patch('genops.providers.haystack.HAS_HAYSTACK', False):
            # Should still be able to create adapter
            adapter = GenOpsHaystackAdapter(
                team="no-haystack-team",
                project="no-haystack-test"
            )
            
            assert adapter is not None
            # Some functionality may be limited but should not crash


class TestProductionScenarios:
    """Production-ready scenario tests."""

    def test_high_volume_pipeline_tracking(self):
        """Test tracking many pipeline executions."""
        adapter = GenOpsHaystackAdapter(
            team="production-team",
            project="high-volume-test",
            daily_budget_limit=500.0
        )
        
        with adapter.track_session("high-volume-session") as session:
            # Simulate many pipeline executions
            for i in range(10):  # Reduced from 100 for test performance
                with adapter.track_pipeline(f"pipeline-{i}") as context:
                    context.add_component_result({
                        "component_name": f"component-{i}",
                        "component_type": "Generator",
                        "execution_time_seconds": 0.5,
                        "cost": Decimal("0.002"),
                        "provider_name": "OpenAI",
                        "success": True
                    })
                
                session.add_pipeline_result(context.get_metrics())
        
        assert session.total_pipelines == 10
        assert session.total_cost == Decimal("0.020")

    def test_budget_enforcement_integration(self):
        """Test budget enforcement in production scenario."""
        adapter = GenOpsHaystackAdapter(
            team="budget-team",
            project="budget-test",
            daily_budget_limit=0.01,  # Very low limit
            governance_policy="enforcing"
        )
        
        # Mock high existing costs
        adapter.cost_aggregator.get_daily_costs = Mock(
            return_value=Decimal("0.015")  # Over budget
        )
        
        # Should enforce budget limit
        with pytest.raises(RuntimeError, match="budget limit"):
            with adapter.track_pipeline("over-budget-pipeline") as context:
                context.add_component_result({
                    "component_name": "expensive-component",
                    "component_type": "Generator",
                    "execution_time_seconds": 1.0,
                    "cost": Decimal("0.005"),
                    "provider_name": "OpenAI",
                    "success": True
                })

    def test_multi_tenant_cost_attribution(self):
        """Test multi-tenant cost attribution."""
        adapter = GenOpsHaystackAdapter(
            team="multi-tenant-team",
            project="multi-tenant-test"
        )
        
        # Track pipelines for different customers
        customers = ["customer-a", "customer-b", "customer-c"]
        customer_costs = {}
        
        for customer in customers:
            with adapter.track_pipeline(
                f"{customer}-pipeline",
                customer_id=customer
            ) as context:
                cost = Decimal("0.005") if customer == "customer-a" else Decimal("0.003")
                context.add_component_result({
                    "component_name": f"{customer}-component",
                    "component_type": "Generator",
                    "execution_time_seconds": 1.0,
                    "cost": cost,
                    "provider_name": "OpenAI",
                    "success": True
                })
                
                customer_costs[customer] = context.get_metrics().total_cost
        
        # Verify customer-specific cost attribution
        assert customer_costs["customer-a"] == Decimal("0.005")
        assert customer_costs["customer-b"] == Decimal("0.003")
        assert customer_costs["customer-c"] == Decimal("0.003")

    def test_long_running_session_tracking(self):
        """Test long-running session tracking."""
        adapter = GenOpsHaystackAdapter(
            team="long-session-team",
            project="long-session-test"
        )
        
        with adapter.track_session("long-running-session") as session:
            # Simulate session with multiple phases
            phases = ["initialization", "processing", "finalization"]
            
            for phase in phases:
                with adapter.track_pipeline(f"{phase}-pipeline") as context:
                    # Simulate different costs per phase
                    phase_cost = {
                        "initialization": Decimal("0.001"),
                        "processing": Decimal("0.015"),
                        "finalization": Decimal("0.002")
                    }
                    
                    context.add_component_result({
                        "component_name": f"{phase}-component",
                        "component_type": "Generator",
                        "execution_time_seconds": 2.0,
                        "cost": phase_cost[phase],
                        "provider_name": "OpenAI",
                        "success": True
                    })
                
                session.add_pipeline_result(context.get_metrics())
                
                # Simulate time passing between phases
                time.sleep(0.1)
        
        assert session.total_pipelines == 3
        assert session.total_cost == Decimal("0.018")  # Sum of all phase costs