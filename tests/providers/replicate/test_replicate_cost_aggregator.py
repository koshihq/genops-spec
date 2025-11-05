#!/usr/bin/env python3
"""
Test Suite for ReplicateCostAggregator and Cost Context Management

Unit tests covering advanced cost aggregation functionality including:
- Multi-model cost tracking and aggregation
- Context manager lifecycle and cleanup
- Budget monitoring and alerting
- Cost optimization recommendations
- Performance metrics and efficiency calculations
- Governance attribute propagation

Target: ~24 tests covering cost aggregation scenarios
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager
from typing import Dict, Any

from src.genops.providers.replicate_cost_aggregator import (
    ReplicateCostAggregator,
    create_replicate_cost_context,
    ReplicateOperation,
    ReplicateCostSummary,
    BudgetAlert
)

class TestReplicateOperation:
    """Test ReplicateOperation data structure."""
    
    def test_operation_initialization_basic(self):
        """Test basic operation initialization."""
        operation = ReplicateOperation(
            operation_id="op-123",
            model="meta/llama-2-7b-chat",
            category="text",
            cost_usd=0.001234,
            timestamp=1000.0
        )
        
        assert operation.operation_id == "op-123"
        assert operation.model == "meta/llama-2-7b-chat"
        assert operation.category == "text"
        assert operation.cost_usd == 0.001234
        assert operation.timestamp == 1000.0
        assert operation.governance_attributes == {}
    
    def test_operation_initialization_with_governance(self):
        """Test operation initialization with governance attributes."""
        governance_attrs = {
            "team": "engineering",
            "project": "cost-tracking",
            "customer_id": "customer-123"
        }
        
        operation = ReplicateOperation(
            operation_id="op-456",
            model="black-forest-labs/flux-schnell",
            category="image",
            cost_usd=0.003,
            timestamp=2000.0,
            input_tokens=None,
            output_tokens=None,
            output_units=1,
            latency_ms=3000.0,
            hardware_type="gpu",
            governance_attributes=governance_attrs
        )
        
        assert operation.output_units == 1
        assert operation.latency_ms == 3000.0
        assert operation.hardware_type == "gpu"
        assert operation.governance_attributes == governance_attrs

class TestReplicateCostSummary:
    """Test ReplicateCostSummary data structure and calculations."""
    
    def test_cost_summary_initialization(self):
        """Test cost summary initialization with defaults."""
        summary = ReplicateCostSummary(
            total_cost=1.234,
            operation_count=5
        )
        
        assert summary.total_cost == 1.234
        assert summary.operation_count == 5
        assert summary.cost_by_model == {}
        assert summary.cost_by_category == {}
        assert summary.unique_models == set()
        assert summary.optimization_recommendations == []
    
    def test_cost_summary_with_complete_data(self):
        """Test cost summary with complete data."""
        summary = ReplicateCostSummary(
            total_cost=2.5,
            operation_count=10,
            cost_by_model={"model1": 1.5, "model2": 1.0},
            cost_by_category={"text": 2.0, "image": 0.5},
            unique_models={"model1", "model2"},
            unique_categories={"text", "image"},
            total_tokens=5000,
            total_output_units=25,
            total_time_ms=30000.0
        )
        
        # Should automatically calculate most/cheapest expensive models
        assert summary.most_expensive_model == "model1"
        assert summary.cheapest_model == "model2"

class TestBudgetAlert:
    """Test BudgetAlert data structure."""
    
    def test_budget_alert_creation(self):
        """Test budget alert creation with all fields."""
        alert = BudgetAlert(
            alert_type="warning",
            current_cost=7.5,
            budget_limit=10.0,
            percentage_used=75.0,
            remaining_budget=2.5,
            projected_cost=9.0,
            recommendation="Monitor remaining operations"
        )
        
        assert alert.alert_type == "warning"
        assert alert.current_cost == 7.5
        assert alert.budget_limit == 10.0
        assert alert.percentage_used == 75.0
        assert alert.remaining_budget == 2.5
        assert alert.projected_cost == 9.0
        assert alert.recommendation == "Monitor remaining operations"

class TestReplicateCostAggregatorInitialization:
    """Test cost aggregator initialization and configuration."""
    
    def test_aggregator_basic_initialization(self):
        """Test basic aggregator initialization."""
        aggregator = ReplicateCostAggregator("test-context")
        
        assert aggregator.context_name == "test-context"
        assert aggregator.context_id is not None
        assert aggregator.budget_limit is None
        assert aggregator.enable_alerts is True
        assert aggregator.optimization_threshold == 0.10
        assert len(aggregator.operations) == 0
        assert aggregator.total_cost == 0.0
    
    def test_aggregator_with_budget_limit(self):
        """Test aggregator initialization with budget limit."""
        aggregator = ReplicateCostAggregator(
            context_name="budget-context",
            budget_limit=50.0,
            enable_alerts=True
        )
        
        assert aggregator.budget_limit == 50.0
        assert aggregator.enable_alerts is True
    
    def test_aggregator_with_pricing_calculator(self):
        """Test aggregator initialization with pricing calculator."""
        with patch('src.genops.providers.replicate_cost_aggregator.ReplicatePricingCalculator') as mock_calc:
            mock_instance = Mock()
            mock_calc.return_value = mock_instance
            
            aggregator = ReplicateCostAggregator("test-context")
            
            assert aggregator._pricing_calculator is mock_instance
    
    def test_aggregator_without_pricing_calculator(self):
        """Test aggregator graceful handling when pricing calculator unavailable."""
        with patch('src.genops.providers.replicate_cost_aggregator.ReplicatePricingCalculator', side_effect=ImportError):
            aggregator = ReplicateCostAggregator("test-context")
            
            assert aggregator._pricing_calculator is None

class TestOperationTracking:
    """Test operation tracking and aggregation."""
    
    @pytest.fixture
    def aggregator(self):
        return ReplicateCostAggregator("test-operations")
    
    def test_add_operation_basic(self, aggregator):
        """Test adding basic operation."""
        operation_id = aggregator.add_operation(
            model="meta/llama-2-7b-chat",
            category="text",
            cost_usd=0.001234,
            input_tokens=100,
            output_tokens=150
        )
        
        assert operation_id is not None
        assert len(aggregator.operations) == 1
        assert aggregator.total_cost == 0.001234
        assert aggregator._cost_by_model["meta/llama-2-7b-chat"] == 0.001234
        assert aggregator._cost_by_category["text"] == 0.001234
        assert aggregator._operation_count_by_model["meta/llama-2-7b-chat"] == 1
    
    def test_add_operation_with_governance(self, aggregator):
        """Test adding operation with governance attributes."""
        operation_id = aggregator.add_operation(
            model="black-forest-labs/flux-schnell",
            category="image",
            cost_usd=0.003,
            output_units=1,
            latency_ms=3000.0,
            team="design-team",
            project="marketing-campaign",
            customer_id="client-456"
        )
        
        operation = aggregator.operations[0]
        assert operation.governance_attributes["team"] == "design-team"
        assert operation.governance_attributes["project"] == "marketing-campaign"
        assert operation.governance_attributes["customer_id"] == "client-456"
    
    def test_add_multiple_operations(self, aggregator):
        """Test adding multiple operations and aggregation."""
        # Add text operation
        aggregator.add_operation(
            model="meta/llama-2-7b-chat",
            category="text",
            cost_usd=0.001,
            team="engineering"
        )
        
        # Add image operation
        aggregator.add_operation(
            model="black-forest-labs/flux-schnell", 
            category="image",
            cost_usd=0.003,
            team="design"
        )
        
        # Add another text operation with same model
        aggregator.add_operation(
            model="meta/llama-2-7b-chat",
            category="text", 
            cost_usd=0.0015,
            team="engineering"
        )
        
        assert len(aggregator.operations) == 3
        assert aggregator.total_cost == 0.0055  # 0.001 + 0.003 + 0.0015
        assert aggregator._cost_by_model["meta/llama-2-7b-chat"] == 0.0025  # 0.001 + 0.0015
        assert aggregator._cost_by_category["text"] == 0.0025
        assert aggregator._cost_by_category["image"] == 0.003
        assert aggregator._operation_count_by_model["meta/llama-2-7b-chat"] == 2

class TestBudgetMonitoring:
    """Test budget monitoring and alert generation."""
    
    def test_budget_alerts_disabled(self):
        """Test aggregator with budget alerts disabled."""
        aggregator = ReplicateCostAggregator(
            "no-alerts-context",
            budget_limit=10.0,
            enable_alerts=False
        )
        
        # Add expensive operation
        aggregator.add_operation(
            model="expensive-model",
            category="video",
            cost_usd=8.0
        )
        
        # Should not generate alerts
        assert len(aggregator.alerts) == 0
    
    def test_budget_warning_alert(self):
        """Test budget warning alert at 75% threshold."""
        aggregator = ReplicateCostAggregator(
            "warning-context",
            budget_limit=10.0,
            enable_alerts=True
        )
        
        # Add operation that reaches 75% of budget
        aggregator.add_operation(
            model="test-model",
            category="text",
            cost_usd=7.5
        )
        
        assert len(aggregator.alerts) == 1
        alert = aggregator.alerts[0]
        assert alert.alert_type == "warning"
        assert alert.current_cost == 7.5
        assert alert.budget_limit == 10.0
        assert alert.percentage_used == 75.0
        assert alert.remaining_budget == 2.5
    
    def test_budget_critical_alert(self):
        """Test budget critical alert at 90% threshold."""
        aggregator = ReplicateCostAggregator(
            "critical-context",
            budget_limit=10.0,
            enable_alerts=True
        )
        
        # Add operation that reaches 90% of budget
        aggregator.add_operation(
            model="test-model",
            category="text",
            cost_usd=9.0
        )
        
        assert len(aggregator.alerts) == 1
        alert = aggregator.alerts[0]
        assert alert.alert_type == "critical"
        assert alert.percentage_used == 90.0
        assert "approaching budget limit" in alert.recommendation.lower()
    
    def test_budget_exceeded_alert(self):
        """Test budget exceeded alert."""
        aggregator = ReplicateCostAggregator(
            "exceeded-context",
            budget_limit=10.0,
            enable_alerts=True
        )
        
        # Add operation that exceeds budget
        aggregator.add_operation(
            model="expensive-model", 
            category="video",
            cost_usd=12.0
        )
        
        assert len(aggregator.alerts) == 1
        alert = aggregator.alerts[0]
        assert alert.alert_type == "exceeded"
        assert alert.current_cost == 12.0
        assert alert.remaining_budget == -2.0
        assert "stop operations immediately" in alert.recommendation.lower()
    
    def test_budget_alert_updates(self):
        """Test that budget alerts are updated with new operations."""
        aggregator = ReplicateCostAggregator(
            "update-context",
            budget_limit=10.0,
            enable_alerts=True
        )
        
        # First operation - under threshold
        aggregator.add_operation(
            model="test-model",
            category="text",
            cost_usd=5.0
        )
        assert len(aggregator.alerts) == 0
        
        # Second operation - triggers warning
        aggregator.add_operation(
            model="test-model",
            category="text", 
            cost_usd=3.0
        )
        assert len(aggregator.alerts) == 1
        assert aggregator.alerts[0].alert_type == "warning"
        
        # Third operation - escalates to critical
        aggregator.add_operation(
            model="test-model",
            category="text",
            cost_usd=2.0
        )
        assert len(aggregator.alerts) == 1  # Should replace previous alert
        assert aggregator.alerts[0].alert_type == "critical"

class TestCostSummaryGeneration:
    """Test cost summary generation and metrics calculation."""
    
    @pytest.fixture
    def populated_aggregator(self):
        """Create aggregator with multiple operations for testing."""
        aggregator = ReplicateCostAggregator("test-summary")
        
        # Add text operations
        aggregator.add_operation(
            model="meta/llama-2-7b-chat",
            category="text",
            cost_usd=0.001,
            input_tokens=100,
            output_tokens=150,
            latency_ms=1500.0,
            team="engineering"
        )
        
        aggregator.add_operation(
            model="meta/llama-2-13b-chat",
            category="text", 
            cost_usd=0.002,
            input_tokens=200,
            output_tokens=250,
            latency_ms=2000.0,
            team="research"
        )
        
        # Add image operations
        aggregator.add_operation(
            model="black-forest-labs/flux-schnell",
            category="image",
            cost_usd=0.003,
            output_units=1,
            latency_ms=3000.0,
            team="design"
        )
        
        return aggregator
    
    def test_get_current_summary_basic(self, populated_aggregator):
        """Test basic cost summary generation."""
        summary = populated_aggregator.get_current_summary()
        
        assert isinstance(summary, ReplicateCostSummary)
        assert summary.total_cost == 0.006  # 0.001 + 0.002 + 0.003
        assert summary.operation_count == 3
        
        # Check cost breakdowns
        assert summary.cost_by_model["meta/llama-2-7b-chat"] == 0.001
        assert summary.cost_by_model["meta/llama-2-13b-chat"] == 0.002
        assert summary.cost_by_model["black-forest-labs/flux-schnell"] == 0.003
        
        assert summary.cost_by_category["text"] == 0.003  # 0.001 + 0.002
        assert summary.cost_by_category["image"] == 0.003
        
        # Check unique collections
        assert len(summary.unique_models) == 3
        assert len(summary.unique_categories) == 2
        assert "text" in summary.unique_categories
        assert "image" in summary.unique_categories
    
    def test_get_current_summary_aggregated_metrics(self, populated_aggregator):
        """Test aggregated metrics in summary."""
        summary = populated_aggregator.get_current_summary()
        
        # Total tokens: (100+150) + (200+250) = 700
        assert summary.total_tokens == 700
        
        # Total output units: 1 (from image)
        assert summary.total_output_units == 1
        
        # Total time: 1500 + 2000 + 3000 = 6500ms
        assert summary.total_time_ms == 6500.0
        
        # Most/cheapest expensive models
        assert summary.most_expensive_model == "black-forest-labs/flux-schnell"
        assert summary.cheapest_model == "meta/llama-2-7b-chat"
    
    def test_get_current_summary_empty_aggregator(self):
        """Test summary generation for empty aggregator."""
        aggregator = ReplicateCostAggregator("empty-context")
        summary = aggregator.get_current_summary()
        
        assert summary.total_cost == 0.0
        assert summary.operation_count == 0
        assert len(summary.unique_models) == 0
        assert len(summary.unique_categories) == 0
    
    def test_get_current_summary_with_budget(self):
        """Test summary generation with budget information."""
        aggregator = ReplicateCostAggregator(
            "budget-summary-context",
            budget_limit=10.0,
            enable_alerts=True
        )
        
        aggregator.add_operation(
            model="test-model",
            category="text", 
            cost_usd=8.0
        )
        
        summary = aggregator.get_current_summary()
        
        assert summary.budget_status is not None
        budget_info = summary.budget_status
        assert budget_info["budget_limit"] == 10.0
        assert budget_info["percentage_used"] == 80.0
        assert budget_info["remaining_budget"] == 2.0
        assert len(budget_info["alerts"]) == 1  # Should have warning alert

class TestEfficiencyMetrics:
    """Test efficiency metrics calculation."""
    
    def test_calculate_efficiency_metrics_text_models(self):
        """Test efficiency metrics for text models."""
        aggregator = ReplicateCostAggregator("efficiency-text")
        
        # Add text operations with token data
        aggregator.add_operation(
            model="meta/llama-2-7b-chat",
            category="text",
            cost_usd=0.001,
            input_tokens=500,
            output_tokens=300,  # Total: 800 tokens
            latency_ms=2000.0
        )
        
        aggregator.add_operation(
            model="meta/llama-2-7b-chat", 
            category="text",
            cost_usd=0.002,
            input_tokens=1000,
            output_tokens=500,  # Total: 1500 tokens 
            latency_ms=3000.0
        )
        
        metrics = aggregator._calculate_efficiency_metrics()
        
        # Should calculate cost per 1K tokens
        total_tokens = 800 + 1500  # 2300 tokens
        total_cost = 0.001 + 0.002  # 0.003
        expected_cost_per_1k = (total_cost / total_tokens) * 1000
        
        assert "cost_per_1k_tokens" in metrics
        assert abs(metrics["cost_per_1k_tokens"] - expected_cost_per_1k) < 0.0001
        
        # Should calculate average cost per text operation
        assert "avg_cost_per_text_operation" in metrics
        assert metrics["avg_cost_per_text_operation"] == 0.0015  # (0.001 + 0.002) / 2
    
    def test_calculate_efficiency_metrics_image_models(self):
        """Test efficiency metrics for image models."""
        aggregator = ReplicateCostAggregator("efficiency-image")
        
        # Add image operations
        aggregator.add_operation(
            model="black-forest-labs/flux-schnell",
            category="image",
            cost_usd=0.006,  # 2 images * $0.003
            output_units=2,
            latency_ms=4000.0
        )
        
        aggregator.add_operation(
            model="black-forest-labs/flux-pro",
            category="image",
            cost_usd=0.04,  # 1 image * $0.04
            output_units=1,
            latency_ms=5000.0
        )
        
        metrics = aggregator._calculate_efficiency_metrics()
        
        # Should calculate cost per image
        total_images = 2 + 1  # 3 images
        total_cost = 0.006 + 0.04  # 0.046
        expected_cost_per_image = total_cost / total_images
        
        assert "cost_per_image" in metrics
        assert abs(metrics["cost_per_image"] - expected_cost_per_image) < 0.0001
    
    def test_calculate_efficiency_metrics_with_latency(self):
        """Test efficiency metrics including latency calculations."""
        aggregator = ReplicateCostAggregator("efficiency-latency")
        
        aggregator.add_operation(
            model="test-model",
            category="text",
            cost_usd=0.005,
            latency_ms=10000.0  # 10 seconds
        )
        
        metrics = aggregator._calculate_efficiency_metrics()
        
        # Should calculate cost per second
        assert "cost_per_second" in metrics
        assert metrics["cost_per_second"] == 0.0005  # $0.005 / 10 seconds
    
    def test_calculate_efficiency_metrics_empty(self):
        """Test efficiency metrics with no operations."""
        aggregator = ReplicateCostAggregator("empty-efficiency")
        
        metrics = aggregator._calculate_efficiency_metrics()
        
        assert isinstance(metrics, dict)
        # Should not crash and return empty metrics

class TestOptimizationRecommendations:
    """Test optimization recommendation generation."""
    
    def test_generate_recommendations_model_distribution(self):
        """Test recommendations based on model cost distribution."""
        aggregator = ReplicateCostAggregator("recommendations-distribution")
        
        # Add operations where one model dominates cost
        aggregator.add_operation(
            model="expensive-model",
            category="text",
            cost_usd=0.008  # 80% of total cost
        )
        
        aggregator.add_operation(
            model="cheap-model", 
            category="text",
            cost_usd=0.002  # 20% of total cost
        )
        
        recommendations = aggregator._generate_optimization_recommendations()
        
        # Should recommend considering alternatives for expensive model
        assert len(recommendations) > 0
        assert any("expensive-model" in rec for rec in recommendations)
        assert any("consider alternatives" in rec.lower() for rec in recommendations)
    
    def test_generate_recommendations_high_token_usage(self):
        """Test recommendations for high token usage."""
        aggregator = ReplicateCostAggregator("recommendations-tokens")
        
        # Add multiple text operations with high token counts
        for i in range(3):
            aggregator.add_operation(
                model="meta/llama-2-70b-chat",
                category="text",
                cost_usd=0.005,
                input_tokens=1500,  # High token count
                output_tokens=1000
            )
        
        recommendations = aggregator._generate_optimization_recommendations()
        
        # Should recommend token optimization
        assert any("large prompts" in rec.lower() for rec in recommendations)
    
    def test_generate_recommendations_many_images(self):
        """Test recommendations for many image generations."""
        aggregator = ReplicateCostAggregator("recommendations-images")
        
        # Add many image operations
        for i in range(8):  # More than 5 images
            aggregator.add_operation(
                model="black-forest-labs/flux-schnell",
                category="image",
                cost_usd=0.003
            )
        
        recommendations = aggregator._generate_optimization_recommendations()
        
        # Should recommend batch processing
        assert any("batch" in rec.lower() for rec in recommendations)
    
    def test_generate_recommendations_budget_limit(self):
        """Test recommendations when approaching budget limit."""
        aggregator = ReplicateCostAggregator(
            "recommendations-budget",
            budget_limit=10.0
        )
        
        # Add operation that uses 85% of budget
        aggregator.add_operation(
            model="expensive-model",
            category="video",
            cost_usd=8.5
        )
        
        recommendations = aggregator._generate_optimization_recommendations()
        
        # Should recommend budget caution
        assert any("budget limit" in rec.lower() for rec in recommendations)
    
    def test_generate_recommendations_with_pricing_calculator(self):
        """Test recommendations when pricing calculator provides alternatives."""
        with patch('src.genops.providers.replicate_cost_aggregator.ReplicatePricingCalculator') as mock_calc:
            mock_instance = Mock()
            mock_instance.get_model_alternatives.return_value = [
                ("cheaper-model", 0.6, "40% cost savings")
            ]
            mock_calc.return_value = mock_instance
            
            aggregator = ReplicateCostAggregator("recommendations-alternatives")
            aggregator.add_operation(
                model="expensive-model",
                category="text",
                cost_usd=0.01
            )
            
            recommendations = aggregator._generate_optimization_recommendations()
            
            # Should recommend cheaper alternative
            assert any("cheaper-model" in rec for rec in recommendations)
            assert any("40% cost savings" in rec for rec in recommendations)

class TestModelPerformanceAnalysis:
    """Test individual model performance analysis."""
    
    @pytest.fixture
    def aggregator_with_model_data(self):
        """Create aggregator with multiple operations for same model."""
        aggregator = ReplicateCostAggregator("model-performance")
        
        # Add multiple operations for same model
        aggregator.add_operation(
            model="meta/llama-2-7b-chat",
            category="text",
            cost_usd=0.001,
            latency_ms=1500.0
        )
        
        aggregator.add_operation(
            model="meta/llama-2-7b-chat",
            category="text",
            cost_usd=0.0015,
            latency_ms=2000.0
        )
        
        aggregator.add_operation(
            model="different-model",
            category="text",
            cost_usd=0.005,
            latency_ms=3000.0
        )
        
        return aggregator
    
    def test_get_model_performance_existing_model(self, aggregator_with_model_data):
        """Test getting performance data for existing model."""
        performance = aggregator_with_model_data.get_model_performance("meta/llama-2-7b-chat")
        
        assert performance is not None
        assert performance["model"] == "meta/llama-2-7b-chat"
        assert performance["operation_count"] == 2
        assert performance["total_cost"] == 0.0025  # 0.001 + 0.0015
        assert performance["average_cost"] == 0.00125  # 0.0025 / 2
        assert performance["average_latency_ms"] == 1750.0  # (1500 + 2000) / 2
        
        # Cost percentage relative to total
        total_cost = 0.0025 + 0.005  # 0.0075
        expected_percentage = (0.0025 / total_cost) * 100
        assert abs(performance["cost_percentage"] - expected_percentage) < 0.01
    
    def test_get_model_performance_nonexistent_model(self, aggregator_with_model_data):
        """Test getting performance data for non-existent model."""
        performance = aggregator_with_model_data.get_model_performance("nonexistent-model")
        
        assert performance is None

class TestExportFunctionality:
    """Test cost context export functionality."""
    
    @pytest.fixture
    def populated_aggregator(self):
        """Create aggregator with operations for export testing."""
        aggregator = ReplicateCostAggregator(
            "export-context",
            budget_limit=5.0
        )
        
        aggregator.add_operation(
            model="test-model-1",
            category="text",
            cost_usd=0.001,
            team="team-1"
        )
        
        aggregator.add_operation(
            model="test-model-2",
            category="image",
            cost_usd=0.003,
            team="team-2"
        )
        
        return aggregator
    
    def test_export_summary_structure(self, populated_aggregator):
        """Test export summary structure and completeness."""
        export_data = populated_aggregator.export_summary()
        
        # Check top-level structure
        assert "context_info" in export_data
        assert "cost_summary" in export_data
        assert "operations" in export_data
        assert "model_performance" in export_data
        
        # Check context info
        context_info = export_data["context_info"]
        assert context_info["name"] == "export-context"
        assert context_info["id"] == populated_aggregator.context_id
        assert context_info["budget_limit"] == 5.0
        assert "start_time" in context_info
        assert "duration_seconds" in context_info
        
        # Check operations export
        operations = export_data["operations"]
        assert len(operations) == 2
        assert all(isinstance(op, dict) for op in operations)
        
        # Check model performance export
        model_perf = export_data["model_performance"]
        assert "test-model-1" in model_perf
        assert "test-model-2" in model_perf

class TestCreateReplicateCostContext:
    """Test create_replicate_cost_context context manager."""
    
    @patch('src.genops.providers.replicate_cost_aggregator.tracer')
    def test_cost_context_basic_usage(self, mock_tracer):
        """Test basic cost context manager usage."""
        mock_span = Mock()
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)
        
        with create_replicate_cost_context("test-context") as context:
            assert isinstance(context, ReplicateCostAggregator)
            assert context.context_name == "test-context"
            assert context.budget_limit is None
        
        # Should have created span
        mock_tracer.start_as_current_span.assert_called_once()
    
    @patch('src.genops.providers.replicate_cost_aggregator.tracer')
    def test_cost_context_with_budget(self, mock_tracer):
        """Test cost context manager with budget limit."""
        mock_span = Mock()
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)
        
        with create_replicate_cost_context("budget-context", budget_limit=25.0) as context:
            assert context.budget_limit == 25.0
            
            # Add operation to test functionality within context
            context.add_operation(
                model="test-model",
                category="text",
                cost_usd=0.01
            )
            
            summary = context.get_current_summary()
            assert summary.total_cost == 0.01
    
    @patch('src.genops.providers.replicate_cost_aggregator.tracer')
    @patch('src.genops.providers.replicate_cost_aggregator.logger')
    def test_cost_context_success_logging(self, mock_logger, mock_tracer):
        """Test successful completion logging."""
        mock_span = Mock()
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)
        
        with create_replicate_cost_context("success-context") as context:
            context.add_operation(
                model="test-model",
                category="text", 
                cost_usd=0.005
            )
        
        # Should log successful completion
        mock_logger.info.assert_called_once()
        log_message = mock_logger.info.call_args[0][0]
        assert "success-context" in log_message
        assert "completed" in log_message
    
    @patch('src.genops.providers.replicate_cost_aggregator.tracer')
    @patch('src.genops.providers.replicate_cost_aggregator.logger')
    def test_cost_context_exception_handling(self, mock_logger, mock_tracer):
        """Test exception handling in cost context."""
        mock_span = Mock()
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)
        
        with pytest.raises(ValueError):
            with create_replicate_cost_context("error-context") as context:
                context.add_operation(
                    model="test-model",
                    category="text",
                    cost_usd=0.001
                )
                raise ValueError("Test exception")
        
        # Should record exception and log error
        mock_span.record_exception.assert_called_once()
        mock_logger.error.assert_called_once()

class TestContextManagerIntegration:
    """Test integration between context manager and aggregator."""
    
    @patch('src.genops.providers.replicate_cost_aggregator.tracer')
    def test_context_telemetry_attributes(self, mock_tracer):
        """Test that context manager sets proper telemetry attributes."""
        mock_span = Mock()
        mock_tracer.start_as_current_span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)
        
        with create_replicate_cost_context("telemetry-test", budget_limit=15.0) as context:
            context.add_operation("test-model", "text", 0.01)
        
        # Check that span was called with correct attributes
        call_args = mock_tracer.start_as_current_span.call_args
        span_name = call_args[0][0]
        attributes = call_args[1]["attributes"]
        
        assert span_name == "replicate.cost_context"
        assert attributes["genops.context_name"] == "telemetry-test"
        assert attributes["genops.budget_limit"] == 15.0
        
        # Check final attributes were set
        mock_span.set_attributes.assert_called_once()
        final_attrs = mock_span.set_attributes.call_args[0][0]
        assert "genops.total_cost" in final_attrs
        assert "genops.operation_count" in final_attrs
        assert "genops.success" in final_attrs