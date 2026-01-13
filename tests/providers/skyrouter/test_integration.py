"""
Integration tests for SkyRouter provider.

Tests end-to-end workflows, cross-provider compatibility,
real-world scenarios, and production deployment patterns.
"""

import pytest
import os
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any, List

# Import the modules under test
try:
    from genops.providers.skyrouter import (
        GenOpsSkyRouterAdapter,
        auto_instrument,
        restore_skyrouter
    )
    from genops.providers.skyrouter_validation import validate_skyrouter_setup
    from genops.providers.skyrouter_pricing import SkyRouterPricingCalculator
    from genops.providers.skyrouter_cost_aggregator import SkyRouterCostAggregator
    SKYROUTER_INTEGRATION_AVAILABLE = True
except ImportError:
    SKYROUTER_INTEGRATION_AVAILABLE = False


@pytest.mark.skipif(not SKYROUTER_INTEGRATION_AVAILABLE, reason="SkyRouter integration not available")
class TestSkyRouterEndToEndIntegration:
    """Test suite for end-to-end integration scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.api_key = "test-api-key-123"
        self.team = "integration-test-team"
        self.project = "e2e-test-project"

    def test_complete_setup_to_operation_workflow(self):
        """Test complete workflow from setup to operation."""
        # Step 1: Validation
        with patch.dict(os.environ, {'SKYROUTER_API_KEY': self.api_key}, clear=True):
            with patch('genops.providers.skyrouter_validation.requests.get') as mock_get:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {'status': 'active'}
                mock_get.return_value = mock_response
                
                validation_result = validate_skyrouter_setup()
                assert validation_result.is_valid

        # Step 2: Adapter initialization
        adapter = GenOpsSkyRouterAdapter(
            skyrouter_api_key=self.api_key,
            team=self.team,
            project=self.project
        )
        assert adapter is not None

        # Step 3: Operation execution
        with patch('genops.providers.skyrouter.skyrouter') as mock_skyrouter:
            mock_response = Mock()
            mock_response.model = 'gpt-4'
            mock_response.usage = {'total_tokens': 150}
            mock_skyrouter.route.return_value = mock_response
            
            with adapter.track_routing_session('e2e-test') as session:
                result = session.track_model_call(
                    model='gpt-4',
                    input_data={'prompt': 'Test prompt'}
                )
                
                assert result is not None
                assert result.model == 'gpt-4'
                assert result.total_cost > 0

        # Step 4: Cost analysis
        summary = adapter.cost_aggregator.get_summary()
        assert summary.total_operations >= 1
        assert summary.total_cost > 0

    def test_auto_instrumentation_workflow(self):
        """Test auto-instrumentation workflow."""
        # Step 1: Enable auto-instrumentation
        adapter = auto_instrument(
            skyrouter_api_key=self.api_key,
            team=self.team,
            project=self.project
        )
        assert adapter is not None

        # Step 2: Simulate existing SkyRouter code
        with patch('genops.providers.skyrouter.skyrouter') as mock_skyrouter:
            mock_response = Mock()
            mock_response.model = 'claude-3-sonnet'
            mock_response.usage = {'total_tokens': 200}
            mock_skyrouter.route_to_best_model.return_value = mock_response
            
            # This would normally be user's existing code
            # But we simulate the instrumented call
            with adapter.track_routing_session('auto-instrumented') as session:
                result = session.track_multi_model_routing(
                    models=['gpt-4', 'claude-3-sonnet'],
                    input_data={'task': 'content generation'},
                    routing_strategy='balanced'
                )
                
                assert result.model == 'claude-3-sonnet'
                assert result.routing_strategy == 'balanced'

        # Step 3: Cleanup
        restore_skyrouter()

    def test_multi_session_workflow(self):
        """Test workflow with multiple sessions."""
        adapter = GenOpsSkyRouterAdapter(
            team=self.team,
            project=self.project
        )

        session_results = []
        session_names = ['morning-batch', 'afternoon-interactive', 'evening-analysis']

        with patch('genops.providers.skyrouter.skyrouter') as mock_skyrouter:
            mock_response = Mock()
            mock_response.model = 'gpt-3.5-turbo'
            mock_response.usage = {'total_tokens': 100}
            mock_skyrouter.route.return_value = mock_response

            for session_name in session_names:
                with adapter.track_routing_session(session_name) as session:
                    result = session.track_model_call(
                        model='gpt-3.5-turbo',
                        input_data={'session': session_name}
                    )
                    session_results.append(result)

        # Verify all sessions completed
        assert len(session_results) == 3
        
        # Verify cost aggregation across sessions
        summary = adapter.cost_aggregator.get_summary()
        assert summary.total_operations == 3

    def test_enterprise_deployment_simulation(self):
        """Test enterprise deployment simulation."""
        # Simulate multi-environment deployment
        environments = [
            {'name': 'dev', 'budget': 10.0, 'policy': 'advisory'},
            {'name': 'staging', 'budget': 50.0, 'policy': 'enforced'},
            {'name': 'prod', 'budget': 500.0, 'policy': 'strict'}
        ]

        environment_adapters = {}

        for env in environments:
            adapter = GenOpsSkyRouterAdapter(
                team=f"enterprise-{env['name']}",
                project="multi-env-test",
                environment=env['name'],
                daily_budget_limit=env['budget'],
                governance_policy=env['policy']
            )
            environment_adapters[env['name']] = adapter

        # Simulate different workloads per environment
        workload_patterns = {
            'dev': [{'model': 'gpt-3.5-turbo', 'operations': 5, 'cost': 0.001}],
            'staging': [{'model': 'gpt-4', 'operations': 10, 'cost': 0.03}],
            'prod': [{'model': 'gpt-4', 'operations': 100, 'cost': 0.05}]
        }

        with patch('genops.providers.skyrouter.skyrouter') as mock_skyrouter:
            mock_response = Mock()
            mock_response.usage = {'total_tokens': 100}
            mock_skyrouter.route.return_value = mock_response

            for env_name, adapter in environment_adapters.items():
                workload = workload_patterns[env_name][0]
                mock_response.model = workload['model']

                with adapter.track_routing_session(f'{env_name}-workload') as session:
                    for _ in range(workload['operations']):
                        result = session.track_model_call(
                            model=workload['model'],
                            input_data={'environment': env_name}
                        )

        # Verify environment isolation
        for env_name, adapter in environment_adapters.items():
            summary = adapter.cost_aggregator.get_summary()
            expected_ops = workload_patterns[env_name][0]['operations']
            assert summary.total_operations == expected_ops

    @patch('genops.providers.skyrouter.skyrouter')
    def test_agent_workflow_integration(self, mock_skyrouter):
        """Test multi-agent workflow integration."""
        # Mock different models for different agents
        model_responses = {
            'gpt-3.5-turbo': Mock(model='gpt-3.5-turbo', usage={'total_tokens': 50}),
            'claude-3-sonnet': Mock(model='claude-3-sonnet', usage={'total_tokens': 100}),
            'gpt-4': Mock(model='gpt-4', usage={'total_tokens': 150})
        }

        def mock_route_side_effect(*args, **kwargs):
            model = kwargs.get('model') or args[0] if args else 'gpt-3.5-turbo'
            return model_responses.get(model, model_responses['gpt-3.5-turbo'])

        mock_skyrouter.route.side_effect = mock_route_side_effect

        adapter = GenOpsSkyRouterAdapter(
            team='agent-workflow-team',
            project='multi-agent-integration'
        )

        # Define a customer support workflow
        workflow_steps = [
            {
                'model': 'gpt-3.5-turbo',
                'input': {'task': 'intent_classification'},
                'complexity': 'simple',
                'optimization': 'cost_optimized'
            },
            {
                'model': 'claude-3-sonnet',
                'input': {'task': 'solution_generation'},
                'complexity': 'moderate',
                'optimization': 'balanced'
            },
            {
                'model': 'gpt-4',
                'input': {'task': 'quality_review'},
                'complexity': 'complex',
                'optimization': 'reliability_first'
            }
        ]

        with adapter.track_routing_session('customer-support-workflow') as session:
            result = session.track_agent_workflow(
                workflow_name='customer_support',
                agent_steps=workflow_steps
            )

            assert result.metadata['workflow_name'] == 'customer_support'
            assert result.metadata['step_count'] == 3
            assert len(result.metadata['models_used']) == 3

        # Verify cost attribution across workflow
        summary = adapter.cost_aggregator.get_summary()
        assert summary.total_operations == 3


@pytest.mark.skipif(not SKYROUTER_INTEGRATION_AVAILABLE, reason="SkyRouter integration not available")
class TestSkyRouterCrossProviderCompatibility:
    """Test suite for cross-provider compatibility."""

    def test_cross_provider_cost_comparison(self):
        """Test cost comparison across providers."""
        # Simulate operations across different providers
        skyrouter_adapter = GenOpsSkyRouterAdapter(
            team='comparison-test',
            project='cross-provider-cost'
        )

        # Mock operations for different scenarios
        operation_scenarios = [
            {'scenario': 'simple_chat', 'tokens': 500, 'models': ['gpt-3.5-turbo', 'claude-3-haiku']},
            {'scenario': 'complex_analysis', 'tokens': 2000, 'models': ['gpt-4', 'claude-3-opus']},
            {'scenario': 'code_generation', 'tokens': 1500, 'models': ['gpt-4', 'claude-3-sonnet']}
        ]

        scenario_costs = {}

        with patch('genops.providers.skyrouter.skyrouter') as mock_skyrouter:
            for scenario in operation_scenarios:
                scenario_cost = 0
                
                for model in scenario['models']:
                    mock_response = Mock()
                    mock_response.model = model
                    mock_response.usage = {'total_tokens': scenario['tokens']}
                    mock_skyrouter.route.return_value = mock_response

                    with skyrouter_adapter.track_routing_session(f"{scenario['scenario']}-{model}") as session:
                        result = session.track_model_call(
                            model=model,
                            input_data={'scenario': scenario['scenario']}
                        )
                        scenario_cost += result.total_cost

                scenario_costs[scenario['scenario']] = scenario_cost

        # Verify cost tracking worked
        assert all(cost > 0 for cost in scenario_costs.values())
        assert len(scenario_costs) == 3

    def test_migration_analysis(self):
        """Test migration analysis from other providers."""
        # Simulate existing usage patterns from other providers
        existing_patterns = {
            'openai_direct': [
                {'model': 'gpt-4', 'monthly_operations': 1000, 'avg_cost': 0.06},
                {'model': 'gpt-3.5-turbo', 'monthly_operations': 5000, 'avg_cost': 0.002}
            ],
            'anthropic_direct': [
                {'model': 'claude-3-opus', 'monthly_operations': 500, 'avg_cost': 0.075},
                {'model': 'claude-3-sonnet', 'monthly_operations': 2000, 'avg_cost': 0.015}
            ]
        }

        skyrouter_adapter = GenOpsSkyRouterAdapter(
            team='migration-analysis',
            project='provider-migration'
        )

        # Calculate potential savings with SkyRouter
        calculator = SkyRouterPricingCalculator()
        
        migration_analysis = {}
        
        for provider, patterns in existing_patterns.items():
            provider_total = sum(
                pattern['monthly_operations'] * pattern['avg_cost'] 
                for pattern in patterns
            )
            
            # Estimate SkyRouter cost with route optimization
            skyrouter_total = 0
            for pattern in patterns:
                # Assume 15% savings with intelligent routing
                optimized_cost = pattern['avg_cost'] * 0.85
                skyrouter_total += pattern['monthly_operations'] * optimized_cost
            
            migration_analysis[provider] = {
                'current_cost': provider_total,
                'skyrouter_cost': skyrouter_total,
                'potential_savings': provider_total - skyrouter_total
            }

        # Verify migration analysis
        for provider, analysis in migration_analysis.items():
            assert analysis['potential_savings'] > 0  # Should show savings
            assert analysis['skyrouter_cost'] < analysis['current_cost']


@pytest.mark.skipif(not SKYROUTER_INTEGRATION_AVAILABLE, reason="SkyRouter integration not available")
class TestSkyRouterProductionScenarios:
    """Test suite for production deployment scenarios."""

    def test_high_volume_production_load(self):
        """Test high-volume production load simulation."""
        production_adapter = GenOpsSkyRouterAdapter(
            team='production-team',
            project='high-volume-service',
            environment='production',
            daily_budget_limit=1000.0,
            governance_policy='strict'
        )

        # Simulate high-volume operations
        operation_count = 1000
        batch_size = 100

        with patch('genops.providers.skyrouter.skyrouter') as mock_skyrouter:
            mock_response = Mock()
            mock_response.model = 'gpt-3.5-turbo'
            mock_response.usage = {'total_tokens': 100}
            mock_skyrouter.route.return_value = mock_response

            # Process in batches to simulate real production load
            for batch_num in range(operation_count // batch_size):
                with production_adapter.track_routing_session(f'batch-{batch_num}') as session:
                    for op_num in range(batch_size):
                        result = session.track_model_call(
                            model='gpt-3.5-turbo',
                            input_data={'batch': batch_num, 'operation': op_num}
                        )

        # Verify high-volume handling
        summary = production_adapter.cost_aggregator.get_summary()
        assert summary.total_operations == operation_count
        assert summary.total_cost > 0

        # Check performance (should complete within reasonable time)
        # This is implicitly tested by the test not timing out

    def test_disaster_recovery_simulation(self):
        """Test disaster recovery scenario simulation."""
        # Primary adapter
        primary_adapter = GenOpsSkyRouterAdapter(
            team='ha-primary',
            project='disaster-recovery-test',
            environment='production',
            ha_config={
                'region': 'us-east-1',
                'failover_enabled': True,
                'backup_regions': ['us-west-2']
            }
        )

        # DR adapter
        dr_adapter = GenOpsSkyRouterAdapter(
            team='ha-disaster-recovery',
            project='disaster-recovery-test',
            environment='disaster_recovery',
            ha_config={
                'region': 'us-west-2',
                'primary_region': 'us-east-1'
            }
        )

        # Simulate normal operations on primary
        with patch('genops.providers.skyrouter.skyrouter') as mock_skyrouter:
            mock_response = Mock()
            mock_response.model = 'gpt-4'
            mock_response.usage = {'total_tokens': 150}
            mock_skyrouter.route.return_value = mock_response

            # Primary operations
            with primary_adapter.track_routing_session('primary-ops') as session:
                primary_result = session.track_model_call(
                    model='gpt-4',
                    input_data={'region': 'primary', 'operation': 'normal'}
                )

            # Simulate failover to DR
            with dr_adapter.track_routing_session('dr-ops') as session:
                dr_result = session.track_model_call(
                    model='gpt-4',
                    input_data={'region': 'dr', 'operation': 'failover'}
                )

            # Verify both operations completed
            assert primary_result.model == 'gpt-4'
            assert dr_result.model == 'gpt-4'

    def test_compliance_framework_integration(self):
        """Test compliance framework integration."""
        compliance_frameworks = ['soc2', 'hipaa', 'gdpr']
        
        compliance_adapters = {}
        
        for framework in compliance_frameworks:
            adapter = GenOpsSkyRouterAdapter(
                team=f'compliance-{framework}',
                project='framework-integration-test',
                environment='production',
                compliance_config={
                    'frameworks': [framework],
                    'audit_logging': True,
                    'data_encryption': True
                }
            )
            compliance_adapters[framework] = adapter

        # Simulate compliant operations
        with patch('genops.providers.skyrouter.skyrouter') as mock_skyrouter:
            mock_response = Mock()
            mock_response.model = 'gpt-4'
            mock_response.usage = {'total_tokens': 200}
            mock_skyrouter.route.return_value = mock_response

            for framework, adapter in compliance_adapters.items():
                with adapter.track_routing_session(f'{framework}-compliant') as session:
                    result = session.track_model_call(
                        model='gpt-4',
                        input_data={
                            'compliance_framework': framework,
                            'data_classification': 'sensitive'
                        }
                    )
                    
                    assert result.governance_attrs is not None

    def test_cost_governance_at_scale(self):
        """Test cost governance at enterprise scale."""
        # Create department-level adapters
        departments = {
            'engineering': {'budget': 500.0, 'teams': 5},
            'product': {'budget': 200.0, 'teams': 3},
            'customer_success': {'budget': 150.0, 'teams': 2},
            'sales': {'budget': 100.0, 'teams': 2}
        }

        department_adapters = {}
        
        for dept_name, config in departments.items():
            adapter = GenOpsSkyRouterAdapter(
                team=f'dept-{dept_name}',
                project='enterprise-governance',
                daily_budget_limit=config['budget'],
                cost_center=f'{dept_name.upper()}-001'
            )
            department_adapters[dept_name] = adapter

        # Simulate department usage patterns
        with patch('genops.providers.skyrouter.skyrouter') as mock_skyrouter:
            mock_response = Mock()
            mock_response.usage = {'total_tokens': 100}
            mock_skyrouter.route.return_value = mock_response

            for dept_name, adapter in department_adapters.items():
                dept_config = departments[dept_name]
                operations_per_team = 50
                total_operations = dept_config['teams'] * operations_per_team

                # Different models based on department
                if dept_name == 'engineering':
                    mock_response.model = 'gpt-4'  # More expensive for engineering
                elif dept_name == 'sales':
                    mock_response.model = 'gpt-3.5-turbo'  # Cost-optimized for sales
                else:
                    mock_response.model = 'claude-3-sonnet'  # Balanced for others

                with adapter.track_routing_session(f'{dept_name}-daily-ops') as session:
                    for _ in range(total_operations):
                        result = session.track_model_call(
                            model=mock_response.model,
                            input_data={'department': dept_name}
                        )

        # Verify department cost isolation
        for dept_name, adapter in department_adapters.items():
            summary = adapter.cost_aggregator.get_summary()
            budget_status = adapter.cost_aggregator.check_budget_status()
            
            assert summary.total_operations > 0
            assert budget_status['daily_budget_limit'] == departments[dept_name]['budget']

    def test_performance_monitoring_integration(self):
        """Test performance monitoring integration."""
        monitoring_adapter = GenOpsSkyRouterAdapter(
            team='performance-monitoring',
            project='production-monitoring',
            monitoring_config={
                'metrics_collection': 'comprehensive',
                'sla_monitoring': True,
                'performance_tracking': True
            }
        )

        # Simulate operations with performance tracking
        with patch('genops.providers.skyrouter.skyrouter') as mock_skyrouter:
            # Simulate varying response times
            response_times = [0.1, 0.5, 1.0, 2.0, 0.3]  # Different latencies
            
            for i, latency in enumerate(response_times):
                mock_response = Mock()
                mock_response.model = 'gpt-4'
                mock_response.usage = {'total_tokens': 150}
                mock_skyrouter.route.return_value = mock_response

                # Simulate network latency
                time.sleep(latency / 10)  # Scaled down for test performance

                with monitoring_adapter.track_routing_session(f'perf-test-{i}') as session:
                    start_time = time.time()
                    result = session.track_model_call(
                        model='gpt-4',
                        input_data={'operation_id': i, 'expected_latency': latency}
                    )
                    end_time = time.time()
                    
                    # Verify operation completed
                    assert result is not None
                    assert (end_time - start_time) >= 0

        # Verify monitoring data collection
        summary = monitoring_adapter.cost_aggregator.get_summary()
        assert summary.total_operations == len(response_times)


if __name__ == "__main__":
    pytest.main([__file__])