"""
Tests for PromptLayer cost tracking and attribution functionality.

Tests cost calculation accuracy, attribution mechanisms,
budget enforcement, and financial reporting features.
"""

import pytest
from unittest.mock import Mock, patch
from decimal import Decimal
import time

try:
    from genops.providers.promptlayer import (
        GenOpsPromptLayerAdapter,
        EnhancedPromptLayerSpan,
        GovernancePolicy
    )
    PROMPTLAYER_AVAILABLE = True
except ImportError:
    PROMPTLAYER_AVAILABLE = False


@pytest.mark.skipif(not PROMPTLAYER_AVAILABLE, reason="PromptLayer provider not available")
class TestCostCalculation:
    """Test cost calculation accuracy and mechanisms."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('genops.providers.promptlayer.PromptLayer'):
            self.adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-cost-test',
                team='cost-team',
                daily_budget_limit=10.0
            )
    
    def test_gpt4_cost_estimation(self):
        """Test GPT-4 cost estimation accuracy."""
        span = EnhancedPromptLayerSpan(
            operation_type='cost_test',
            operation_name='gpt4_test'
        )
        
        # Test GPT-4 pricing calculation
        span.update_token_usage(1000, 500, 'gpt-4')
        
        # GPT-4 approximate pricing: $0.03/1k input, $0.06/1k output
        expected_cost = (1000/1000 * 0.03) + (500/1000 * 0.06)
        expected_cost = 0.03 + 0.03  # $0.06 total
        
        assert abs(span.estimated_cost - expected_cost) < 0.001
        assert span.model == 'gpt-4'
        assert span.total_tokens == 1500
    
    def test_gpt35_cost_estimation(self):
        """Test GPT-3.5 cost estimation accuracy."""
        span = EnhancedPromptLayerSpan(
            operation_type='cost_test',
            operation_name='gpt35_test'
        )
        
        # Test GPT-3.5 pricing calculation
        span.update_token_usage(2000, 1000, 'gpt-3.5-turbo')
        
        # GPT-3.5 approximate pricing: $0.0015/1k input, $0.002/1k output
        expected_cost = (2000/1000 * 0.0015) + (1000/1000 * 0.002)
        expected_cost = 0.003 + 0.002  # $0.005 total
        
        assert abs(span.estimated_cost - expected_cost) < 0.001
        assert span.model == 'gpt-3.5-turbo'
    
    def test_manual_cost_override(self):
        """Test manual cost setting overrides token-based calculation."""
        span = EnhancedPromptLayerSpan(
            operation_type='cost_test',
            operation_name='manual_override'
        )
        
        # First set token-based cost
        span.update_token_usage(1000, 500, 'gpt-3.5-turbo')
        token_based_cost = span.estimated_cost
        
        # Then override with manual cost
        manual_cost = 0.025
        span.update_cost(manual_cost)
        
        assert span.estimated_cost == manual_cost
        assert span.estimated_cost != token_based_cost
    
    def test_cost_accumulation_precision(self):
        """Test cost accumulation maintains precision."""
        costs = [0.001, 0.0023, 0.00045, 0.00167]
        total_expected = sum(costs)
        
        accumulated_cost = 0.0
        for cost in costs:
            accumulated_cost += cost
        
        # Test precision is maintained
        assert abs(accumulated_cost - total_expected) < 1e-10
        
        # Test in adapter context
        for i, cost in enumerate(costs):
            with self.adapter.track_prompt_operation(f'precision_test_{i}') as span:
                span.update_cost(cost)
        
        assert abs(self.adapter.daily_usage - total_expected) < 1e-6


@pytest.mark.skipif(not PROMPTLAYER_AVAILABLE, reason="PromptLayer provider not available")
class TestCostAttribution:
    """Test cost attribution to teams, projects, customers."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('genops.providers.promptlayer.PromptLayer'):
            self.adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-attribution-test',
                team='attribution-team',
                project='attribution-project',
                daily_budget_limit=20.0
            )
    
    def test_team_cost_attribution(self):
        """Test costs are properly attributed to teams."""
        operations = [
            {'team': 'team-a', 'cost': 0.15},
            {'team': 'team-b', 'cost': 0.25},
            {'team': 'team-a', 'cost': 0.08},
            {'team': 'team-c', 'cost': 0.12}
        ]
        
        team_costs = {}
        for i, op in enumerate(operations):
            with self.adapter.track_prompt_operation(
                f'team_attribution_{i}',
                tags={'team_override': op['team']}
            ) as span:
                span.team = op['team']  # Override team
                span.update_cost(op['cost'])
                
                if op['team'] not in team_costs:
                    team_costs[op['team']] = 0.0
                team_costs[op['team']] += op['cost']
        
        # Verify team attribution
        expected_teams = {
            'team-a': 0.15 + 0.08,  # 0.23
            'team-b': 0.25,
            'team-c': 0.12
        }
        
        for team, expected_cost in expected_teams.items():
            assert abs(team_costs[team] - expected_cost) < 0.001
    
    def test_customer_cost_attribution(self):
        """Test costs are properly attributed to customers."""
        customer_operations = [
            {'customer_id': 'customer_001', 'cost': 0.45},
            {'customer_id': 'customer_002', 'cost': 0.32},
            {'customer_id': 'customer_001', 'cost': 0.18},
            {'customer_id': 'customer_003', 'cost': 0.67}
        ]
        
        customer_costs = {}
        for i, op in enumerate(customer_operations):
            with self.adapter.track_prompt_operation(
                f'customer_attribution_{i}',
                customer_id=op['customer_id']
            ) as span:
                span.update_cost(op['cost'])
                
                customer = op['customer_id']
                if customer not in customer_costs:
                    customer_costs[customer] = 0.0
                customer_costs[customer] += op['cost']
        
        # Verify customer attribution
        expected_customers = {
            'customer_001': 0.45 + 0.18,  # 0.63
            'customer_002': 0.32,
            'customer_003': 0.67
        }
        
        for customer, expected_cost in expected_customers.items():
            assert abs(customer_costs[customer] - expected_cost) < 0.001
    
    def test_cost_center_attribution(self):
        """Test costs are properly attributed to cost centers."""
        with self.adapter.track_prompt_operation(
            'cost_center_test',
            cost_center='rd-department'
        ) as span:
            span.update_cost(0.125)
            assert span.cost_center == 'rd-department'
            
            metrics = span.get_metrics()
            assert metrics['cost_center'] == 'rd-department'
    
    def test_multi_dimensional_attribution(self):
        """Test attribution across multiple dimensions."""
        with self.adapter.track_prompt_operation(
            'multi_dim_test',
            customer_id='enterprise_client',
            cost_center='sales-engineering'
        ) as span:
            span.update_cost(0.075)
            
            metrics = span.get_metrics()
            assert metrics['team'] == 'attribution-team'
            assert metrics['project'] == 'attribution-project'
            assert metrics['customer_id'] == 'enterprise_client'
            assert metrics['cost_center'] == 'sales-engineering'
            assert metrics['estimated_cost'] == 0.075


@pytest.mark.skipif(not PROMPTLAYER_AVAILABLE, reason="PromptLayer provider not available")
class TestBudgetEnforcement:
    """Test budget limits and enforcement mechanisms."""
    
    def test_daily_budget_advisory_mode(self):
        """Test daily budget in advisory mode."""
        with patch('genops.providers.promptlayer.PromptLayer'):
            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-budget-advisory',
                governance_policy=GovernancePolicy.ADVISORY,
                daily_budget_limit=0.10
            )
        
        # Use most of budget
        with adapter.track_prompt_operation('budget_test_1') as span:
            span.update_cost(0.08)
        
        # Exceed budget - should log violation but not fail
        with adapter.track_prompt_operation('budget_test_2') as span:
            span.update_cost(0.05)  # Total: 0.13, exceeds 0.10
            assert len(span.policy_violations) > 0
            assert any('budget' in v.lower() for v in span.policy_violations)
    
    def test_daily_budget_enforced_mode(self):
        """Test daily budget in enforced mode."""
        with patch('genops.providers.promptlayer.PromptLayer'):
            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-budget-enforced',
                governance_policy=GovernancePolicy.ENFORCED,
                daily_budget_limit=0.10
            )
        
        # Use most of budget
        with adapter.track_prompt_operation('budget_enforced_1') as span:
            span.update_cost(0.08)
        
        # Attempt to exceed budget - should raise exception
        adapter.daily_usage = 0.11  # Simulate exceeded budget
        
        with pytest.raises(ValueError, match="Daily budget limit"):
            with adapter.track_prompt_operation('budget_enforced_2') as span:
                pass
    
    def test_operation_cost_limit(self):
        """Test per-operation cost limits."""
        with patch('genops.providers.promptlayer.PromptLayer'):
            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-operation-limit',
                max_operation_cost=0.05
            )
        
        # Within limit
        with adapter.track_prompt_operation('op_limit_1', max_cost=0.05) as span:
            span.update_cost(0.03)
            assert len([v for v in span.policy_violations if 'operation cost' in v.lower()]) == 0
        
        # Exceed limit
        with adapter.track_prompt_operation('op_limit_2', max_cost=0.05) as span:
            span.update_cost(0.08)
            assert len([v for v in span.policy_violations if 'operation cost' in v.lower()]) > 0
    
    def test_budget_remaining_calculation(self):
        """Test budget remaining calculations."""
        with patch('genops.providers.promptlayer.PromptLayer'):
            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-budget-calc',
                daily_budget_limit=5.00
            )
        
        # Initial state
        metrics = adapter.get_metrics()
        assert metrics['budget_remaining'] == 5.00
        
        # After spending
        with adapter.track_prompt_operation('budget_calc_1') as span:
            span.update_cost(1.25)
        
        metrics = adapter.get_metrics()
        assert abs(metrics['budget_remaining'] - 3.75) < 0.001
        
        # After more spending
        with adapter.track_prompt_operation('budget_calc_2') as span:
            span.update_cost(2.10)
        
        metrics = adapter.get_metrics()
        assert abs(metrics['budget_remaining'] - 1.65) < 0.001


@pytest.mark.skipif(not PROMPTLAYER_AVAILABLE, reason="PromptLayer provider not available")
class TestCostOptimization:
    """Test cost optimization features and recommendations."""
    
    def test_cost_per_quality_calculation(self):
        """Test cost per quality point calculations."""
        span = EnhancedPromptLayerSpan(
            operation_type='optimization_test',
            operation_name='cost_per_quality'
        )
        
        span.update_cost(0.035)
        span.add_attributes({'quality_score': 0.87})
        
        # Calculate cost per quality point
        cost_per_quality = span.estimated_cost / span.metadata['quality_score']
        expected_cpq = 0.035 / 0.87
        
        assert abs(cost_per_quality - expected_cpq) < 0.001
    
    def test_model_cost_comparison(self):
        """Test cost comparison between different models."""
        models_costs = [
            {'model': 'gpt-3.5-turbo', 'tokens_in': 1000, 'tokens_out': 500},
            {'model': 'gpt-4', 'tokens_in': 1000, 'tokens_out': 500}
        ]
        
        costs = {}
        for model_config in models_costs:
            span = EnhancedPromptLayerSpan(
                operation_type='model_comparison',
                operation_name=f'test_{model_config["model"]}'
            )
            
            span.update_token_usage(
                model_config['tokens_in'],
                model_config['tokens_out'],
                model_config['model']
            )
            
            costs[model_config['model']] = span.estimated_cost
        
        # GPT-4 should be more expensive than GPT-3.5
        assert costs['gpt-4'] > costs['gpt-3.5-turbo']
        
        # Verify reasonable cost differences
        cost_ratio = costs['gpt-4'] / costs['gpt-3.5-turbo']
        assert cost_ratio > 10  # GPT-4 should be significantly more expensive
    
    def test_batch_cost_efficiency(self):
        """Test cost efficiency of batch operations."""
        with patch('genops.providers.promptlayer.PromptLayer'):
            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-batch-efficiency'
            )
        
        # Single operation costs
        single_costs = []
        for i in range(5):
            with adapter.track_prompt_operation(f'single_op_{i}') as span:
                span.update_cost(0.008)  # Fixed overhead per operation
                single_costs.append(span.estimated_cost)
        
        total_single_cost = sum(single_costs)
        
        # Batch operation cost
        with adapter.track_prompt_operation('batch_op') as batch_span:
            # Batch operations typically have lower per-item costs
            batch_span.update_cost(0.025)  # Lower total cost for 5 items
        
        batch_cost = batch_span.estimated_cost
        
        # Verify batch is more efficient
        assert batch_cost < total_single_cost
        
        efficiency_ratio = batch_cost / total_single_cost
        assert efficiency_ratio < 0.8  # At least 20% more efficient


@pytest.mark.skipif(not PROMPTLAYER_AVAILABLE, reason="PromptLayer provider not available")  
class TestFinancialReporting:
    """Test financial reporting and analytics features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('genops.providers.promptlayer.PromptLayer'):
            self.adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-reporting-test',
                team='reporting-team',
                project='financial-analytics',
                daily_budget_limit=25.0
            )
    
    def test_cost_aggregation_by_time_period(self):
        """Test cost aggregation over time periods."""
        operations_by_hour = {
            '2024-01-01T10': [0.12, 0.08, 0.15],
            '2024-01-01T11': [0.22, 0.18],
            '2024-01-01T12': [0.35, 0.09, 0.11, 0.07]
        }
        
        hourly_totals = {}
        for hour, costs in operations_by_hour.items():
            hourly_totals[hour] = sum(costs)
            
            for i, cost in enumerate(costs):
                with self.adapter.track_prompt_operation(
                    f'{hour}_op_{i}',
                    tags={'hour': hour}
                ) as span:
                    span.update_cost(cost)
                    span.add_attributes({'reporting_hour': hour})
        
        # Verify hourly aggregation
        expected_totals = {
            '2024-01-01T10': 0.35,
            '2024-01-01T11': 0.40,
            '2024-01-01T12': 0.62
        }
        
        for hour, expected_total in expected_totals.items():
            assert abs(hourly_totals[hour] - expected_total) < 0.001
    
    def test_cost_breakdown_by_operation_type(self):
        """Test cost breakdown by operation types."""
        operations = [
            {'type': 'classification', 'cost': 0.05},
            {'type': 'generation', 'cost': 0.12},
            {'type': 'classification', 'cost': 0.03}, 
            {'type': 'summarization', 'cost': 0.08},
            {'type': 'generation', 'cost': 0.15},
            {'type': 'generation', 'cost': 0.09}
        ]
        
        type_costs = {}
        for i, op in enumerate(operations):
            with self.adapter.track_prompt_operation(
                f'type_breakdown_{i}',
                operation_type=op['type']
            ) as span:
                span.update_cost(op['cost'])
                
                if op['type'] not in type_costs:
                    type_costs[op['type']] = 0.0
                type_costs[op['type']] += op['cost']
        
        # Verify cost breakdown
        expected_breakdown = {
            'classification': 0.05 + 0.03,  # 0.08
            'generation': 0.12 + 0.15 + 0.09,  # 0.36
            'summarization': 0.08
        }
        
        for op_type, expected_cost in expected_breakdown.items():
            assert abs(type_costs[op_type] - expected_cost) < 0.001
    
    def test_roi_calculation_metrics(self):
        """Test ROI calculation for operations."""
        operations_with_value = [
            {'cost': 0.08, 'business_value': 2.50},  # High ROI
            {'cost': 0.15, 'business_value': 1.20},  # Lower ROI
            {'cost': 0.12, 'business_value': 3.60},  # Highest ROI
        ]
        
        roi_metrics = []
        for i, op in enumerate(operations_with_value):
            with self.adapter.track_prompt_operation(f'roi_test_{i}') as span:
                span.update_cost(op['cost'])
                span.add_attributes({
                    'business_value': op['business_value'],
                    'roi_ratio': op['business_value'] / op['cost']
                })
                
                roi_metrics.append({
                    'operation_id': span.operation_id,
                    'cost': op['cost'],
                    'value': op['business_value'],
                    'roi': op['business_value'] / op['cost']
                })
        
        # Verify ROI calculations
        expected_rois = [
            2.50 / 0.08,  # 31.25
            1.20 / 0.15,  # 8.0
            3.60 / 0.12   # 30.0
        ]
        
        for i, expected_roi in enumerate(expected_rois):
            assert abs(roi_metrics[i]['roi'] - expected_roi) < 0.001
        
        # Find highest ROI operation
        best_roi_op = max(roi_metrics, key=lambda x: x['roi'])
        assert abs(best_roi_op['roi'] - 31.25) < 0.001
    
    def test_budget_utilization_reporting(self):
        """Test budget utilization reporting metrics."""
        budget_limit = 10.0
        with patch('genops.providers.promptlayer.PromptLayer'):
            adapter = GenOpsPromptLayerAdapter(
                promptlayer_api_key='pl-budget-utilization',
                daily_budget_limit=budget_limit
            )
        
        # Execute operations to use budget
        spending_pattern = [1.2, 2.3, 1.8, 1.5, 0.9]  # Total: 7.7
        
        for i, cost in enumerate(spending_pattern):
            with adapter.track_prompt_operation(f'budget_util_{i}') as span:
                span.update_cost(cost)
        
        # Calculate utilization metrics
        metrics = adapter.get_metrics()
        total_spent = sum(spending_pattern)
        utilization_rate = total_spent / budget_limit
        
        assert abs(metrics['daily_usage'] - total_spent) < 0.001
        assert abs(metrics['budget_remaining'] - (budget_limit - total_spent)) < 0.001
        
        # Verify utilization percentage
        expected_utilization = 7.7 / 10.0  # 77%
        actual_utilization = metrics['daily_usage'] / budget_limit
        assert abs(actual_utilization - expected_utilization) < 0.001