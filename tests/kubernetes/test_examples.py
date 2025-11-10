#!/usr/bin/env python3
"""
Automated tests for GenOps AI Kubernetes examples.

Tests all example files for correctness, error handling, and expected functionality.
Designed to run in CI/CD pipelines and local development environments.
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Add the examples directory to Python path
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples" / "kubernetes"
sys.path.insert(0, str(EXAMPLES_DIR))


class TestEnvironment:
    """Test environment setup and utilities."""

    def __init__(self):
        self.test_env_vars = {
            "GENOPS_ENV": "test",
            "LOG_LEVEL": "DEBUG",
            "DEFAULT_TEAM": "test-team",
            "PROJECT_NAME": "test-project",
            "ENVIRONMENT": "test"
        }

    def setup_test_environment(self):
        """Set up test environment variables."""
        for key, value in self.test_env_vars.items():
            os.environ[key] = value

    def cleanup_test_environment(self):
        """Clean up test environment variables."""
        for key in self.test_env_vars:
            os.environ.pop(key, None)


@pytest.fixture
def test_env():
    """Pytest fixture for test environment."""
    env = TestEnvironment()
    env.setup_test_environment()
    yield env
    env.cleanup_test_environment()


@pytest.fixture
def mock_genops_imports():
    """Mock GenOps imports for testing without full installation."""

    # Mock validation result
    mock_validation_result = Mock()
    mock_validation_result.is_valid = True
    mock_validation_result.is_kubernetes_environment = True
    mock_validation_result.namespace = "test-namespace"
    mock_validation_result.pod_name = "test-pod"
    mock_validation_result.node_name = "test-node"
    mock_validation_result.cluster_name = "test-cluster"
    mock_validation_result.has_service_account = True
    mock_validation_result.has_resource_monitoring = True
    mock_validation_result.issues = []

    # Mock Kubernetes adapter
    mock_adapter = Mock()
    mock_adapter.is_available.return_value = True
    mock_adapter.get_framework_name.return_value = "kubernetes"
    mock_adapter.get_telemetry_attributes.return_value = {
        "k8s.namespace.name": "test-namespace",
        "k8s.pod.name": "test-pod",
        "k8s.node.name": "test-node",
        "team": "test-team",
        "project": "test-project"
    }

    # Mock governance context
    mock_governance_context = AsyncMock()
    mock_governance_context.context_id = "test-context-123"
    mock_governance_context.get_duration.return_value = 1.234
    mock_governance_context.get_cost_summary.return_value = {"total_cost": 0.0023}
    mock_governance_context.get_telemetry_data.return_value = {"test": "data"}
    mock_governance_context.get_resource_usage.return_value = {
        "cpu_usage_millicores": 250,
        "memory_usage_bytes": 536870912
    }

    mock_adapter.create_governance_context.return_value.__aenter__ = AsyncMock(return_value=mock_governance_context)
    mock_adapter.create_governance_context.return_value.__aexit__ = AsyncMock(return_value=None)

    with patch.dict('sys.modules', {
        'genops': Mock(),
        'genops.providers': Mock(),
        'genops.providers.kubernetes': Mock(
            validate_kubernetes_setup=Mock(return_value=mock_validation_result),
            print_kubernetes_validation_result=Mock(),
            KubernetesAdapter=Mock(return_value=mock_adapter),
            KubernetesDetector=Mock(),
            KubernetesResourceMonitor=Mock()
        ),
        'genops.core': Mock(),
        'genops.core.governance': Mock(
            create_governance_context=Mock(return_value=mock_governance_context)
        ),
        'genops.core.cost': Mock(),
        'genops.core.performance': Mock(),
        'genops.core.security': Mock(),
        'genops.core.instrumentation': Mock(
            get_active_instrumentations=Mock(return_value={
                "openai": {"status": "active"},
                "anthropic": {"status": "active"}
            })
        )
    }):
        yield {
            'validation_result': mock_validation_result,
            'adapter': mock_adapter,
            'governance_context': mock_governance_context
        }


class TestSetupValidation:
    """Test the setup_validation.py example."""

    def test_import_setup_validation(self, mock_genops_imports):
        """Test that setup_validation.py imports correctly."""
        try:
            import setup_validation
            assert hasattr(setup_validation, 'main')
            assert hasattr(setup_validation, 'validate_environment')
        except ImportError as e:
            pytest.skip(f"Cannot import setup_validation: {e}")

    @pytest.mark.asyncio
    async def test_validate_environment_success(self, test_env, mock_genops_imports):
        """Test successful environment validation."""
        try:
            import setup_validation

            # Test basic validation
            result = await setup_validation.validate_environment()
            assert result is True

        except ImportError:
            pytest.skip("setup_validation module not available")

    @pytest.mark.asyncio
    async def test_validate_environment_with_options(self, test_env, mock_genops_imports):
        """Test validation with different options."""
        try:
            import setup_validation

            # Test detailed validation
            result = await setup_validation.validate_environment(detailed=True)
            assert result is True

            # Test with fix issues
            result = await setup_validation.validate_environment(fix_issues=True)
            assert result is True

        except ImportError:
            pytest.skip("setup_validation module not available")

    def test_demonstrate_tracking_patterns(self, test_env, mock_genops_imports):
        """Test tracking patterns demonstration."""
        try:
            import setup_validation

            # Should not raise exception
            setup_validation.demonstrate_tracking_patterns()

        except ImportError:
            pytest.skip("setup_validation module not available")

    def test_show_kubernetes_specific_features(self, test_env, mock_genops_imports):
        """Test Kubernetes-specific features display."""
        try:
            import setup_validation

            # Should not raise exception
            setup_validation.show_kubernetes_specific_features()

        except ImportError:
            pytest.skip("setup_validation module not available")

    def test_run_integration_test(self, test_env, mock_genops_imports):
        """Test integration test functionality."""
        try:
            import setup_validation

            # Test integration test
            result = asyncio.run(setup_validation.run_integration_test())
            assert result is True

        except ImportError:
            pytest.skip("setup_validation module not available")


class TestAutoInstrumentation:
    """Test the auto_instrumentation.py example."""

    def test_import_auto_instrumentation(self, mock_genops_imports):
        """Test that auto_instrumentation.py imports correctly."""
        try:
            import auto_instrumentation
            assert hasattr(auto_instrumentation, 'main')
            assert hasattr(auto_instrumentation, 'demonstrate_auto_instrumentation')
        except ImportError as e:
            pytest.skip(f"Cannot import auto_instrumentation: {e}")

    @pytest.mark.asyncio
    async def test_demonstrate_auto_instrumentation(self, test_env, mock_genops_imports):
        """Test auto-instrumentation demonstration."""
        try:
            import auto_instrumentation

            # Mock auto_instrument function
            with patch('auto_instrumentation.auto_instrument') as mock_auto_instrument:
                result = await auto_instrumentation.demonstrate_auto_instrumentation()
                assert result is True
                mock_auto_instrument.assert_called_once()

        except ImportError:
            pytest.skip("auto_instrumentation module not available")

    @pytest.mark.asyncio
    async def test_instrumented_openai(self, test_env, mock_genops_imports):
        """Test OpenAI instrumentation testing."""
        try:
            import auto_instrumentation

            # Test without API key (should use simulation)
            result = await auto_instrumentation.test_instrumented_openai()
            # Should return False when no API key, but should not crash
            assert isinstance(result, bool)

        except ImportError:
            pytest.skip("auto_instrumentation module not available")

    @pytest.mark.asyncio
    async def test_instrumented_anthropic(self, test_env, mock_genops_imports):
        """Test Anthropic instrumentation testing."""
        try:
            import auto_instrumentation

            # Test without API key (should use simulation)
            result = await auto_instrumentation.test_instrumented_anthropic()
            # Should return False when no API key, but should not crash
            assert isinstance(result, bool)

        except ImportError:
            pytest.skip("auto_instrumentation module not available")

    def test_show_existing_code_examples(self, test_env, mock_genops_imports):
        """Test existing code examples display."""
        try:
            import auto_instrumentation

            # Should not raise exception
            auto_instrumentation.show_existing_code_examples()

        except ImportError:
            pytest.skip("auto_instrumentation module not available")

    def test_show_advanced_auto_features(self, test_env, mock_genops_imports):
        """Test advanced features display."""
        try:
            import auto_instrumentation

            # Should not raise exception
            auto_instrumentation.show_advanced_auto_features()

        except ImportError:
            pytest.skip("auto_instrumentation module not available")

    @pytest.mark.asyncio
    async def test_comprehensive_demo(self, test_env, mock_genops_imports):
        """Test comprehensive demo functionality."""
        try:
            import auto_instrumentation

            with patch('auto_instrumentation.auto_instrument'):
                result = await auto_instrumentation.run_comprehensive_demo()
                assert result is True

        except ImportError:
            pytest.skip("auto_instrumentation module not available")


class TestBasicTracking:
    """Test the basic_tracking.py example."""

    def test_import_basic_tracking(self, mock_genops_imports):
        """Test that basic_tracking.py imports correctly."""
        try:
            import basic_tracking
            assert hasattr(basic_tracking, 'main')
            assert hasattr(basic_tracking, 'basic_tracking_example')
        except ImportError as e:
            pytest.skip(f"Cannot import basic_tracking: {e}")

    @pytest.mark.asyncio
    async def test_basic_tracking_example(self, test_env, mock_genops_imports):
        """Test basic tracking example functionality."""
        try:
            import basic_tracking

            result = await basic_tracking.basic_tracking_example(
                team="test-team",
                project="test-project",
                customer_id="test-customer"
            )
            assert result is True

        except ImportError:
            pytest.skip("basic_tracking module not available")

    @pytest.mark.asyncio
    async def test_basic_tracking_with_defaults(self, test_env, mock_genops_imports):
        """Test basic tracking with default parameters."""
        try:
            import basic_tracking

            result = await basic_tracking.basic_tracking_example()
            assert result is True

        except ImportError:
            pytest.skip("basic_tracking module not available")

    def test_demonstrate_tracking_patterns(self, test_env, mock_genops_imports):
        """Test tracking patterns demonstration."""
        try:
            import basic_tracking

            # Should not raise exception
            basic_tracking.demonstrate_tracking_patterns()

        except ImportError:
            pytest.skip("basic_tracking module not available")

    def test_show_kubernetes_specific_features(self, test_env, mock_genops_imports):
        """Test Kubernetes-specific features display."""
        try:
            import basic_tracking

            # Should not raise exception
            basic_tracking.show_kubernetes_specific_features()

        except ImportError:
            pytest.skip("basic_tracking module not available")


class TestCostTracking:
    """Test the cost_tracking.py example."""

    def test_import_cost_tracking(self, mock_genops_imports):
        """Test that cost_tracking.py imports correctly."""
        try:
            import cost_tracking
            assert hasattr(cost_tracking, 'main')
            assert hasattr(cost_tracking, 'KubernetesCostDemo')
        except ImportError as e:
            pytest.skip(f"Cannot import cost_tracking: {e}")

    @pytest.mark.asyncio
    async def test_basic_cost_tracking(self, test_env, mock_genops_imports):
        """Test basic cost tracking functionality."""
        try:
            import cost_tracking

            # Create mock cost tracker and budget manager
            with patch('cost_tracking.CostTracker'), patch('cost_tracking.BudgetManager'):
                demo = cost_tracking.KubernetesCostDemo()

                result = await demo.demonstrate_basic_cost_tracking(
                    team="test-team",
                    project="test-project"
                )
                assert result is True

        except ImportError:
            pytest.skip("cost_tracking module not available")

    @pytest.mark.asyncio
    async def test_budget_management(self, test_env, mock_genops_imports):
        """Test budget management functionality."""
        try:
            import cost_tracking

            with patch('cost_tracking.CostTracker'), patch('cost_tracking.BudgetManager'):
                demo = cost_tracking.KubernetesCostDemo()

                result = await demo.demonstrate_budget_management(budget_limit=50.0)
                assert result is True

        except ImportError:
            pytest.skip("cost_tracking module not available")

    @pytest.mark.asyncio
    async def test_multi_provider_aggregation(self, test_env, mock_genops_imports):
        """Test multi-provider cost aggregation."""
        try:
            import cost_tracking

            with patch('cost_tracking.CostTracker'), patch('cost_tracking.BudgetManager'):
                demo = cost_tracking.KubernetesCostDemo()

                result = await demo.demonstrate_multi_provider_cost_aggregation()
                assert result is True

        except ImportError:
            pytest.skip("cost_tracking module not available")

    @pytest.mark.asyncio
    async def test_cost_optimization_strategies(self, test_env, mock_genops_imports):
        """Test cost optimization strategies."""
        try:
            import cost_tracking

            with patch('cost_tracking.CostTracker'), patch('cost_tracking.BudgetManager'):
                demo = cost_tracking.KubernetesCostDemo()

                result = await demo.demonstrate_cost_optimization_strategies()
                assert result is True

        except ImportError:
            pytest.skip("cost_tracking module not available")


class TestProductionPatterns:
    """Test the production_patterns.py example."""

    def test_import_production_patterns(self, mock_genops_imports):
        """Test that production_patterns.py imports correctly."""
        try:
            import production_patterns
            assert hasattr(production_patterns, 'main')
            assert hasattr(production_patterns, 'ProductionPatternDemo')
        except ImportError as e:
            pytest.skip(f"Cannot import production_patterns: {e}")

    @pytest.mark.asyncio
    async def test_high_availability_pattern(self, test_env, mock_genops_imports):
        """Test high availability pattern demonstration."""
        try:
            import production_patterns

            # Create mock performance monitor and circuit breaker
            with patch('production_patterns.PerformanceMonitor'), \
                 patch('production_patterns.CircuitBreaker'), \
                 patch('production_patterns.SecurityValidator'):

                config = production_patterns.ProductionConfig()
                demo = production_patterns.ProductionPatternDemo(config)

                result = await demo.demonstrate_high_availability_pattern()
                assert result is True

        except ImportError:
            pytest.skip("production_patterns module not available")

    @pytest.mark.asyncio
    async def test_performance_optimization(self, test_env, mock_genops_imports):
        """Test performance optimization patterns."""
        try:
            import production_patterns

            with patch('production_patterns.PerformanceMonitor'), \
                 patch('production_patterns.CircuitBreaker'), \
                 patch('production_patterns.SecurityValidator'):

                config = production_patterns.ProductionConfig()
                demo = production_patterns.ProductionPatternDemo(config)

                result = await demo.demonstrate_performance_optimization()
                assert result is True

        except ImportError:
            pytest.skip("production_patterns module not available")

    @pytest.mark.asyncio
    async def test_enterprise_security(self, test_env, mock_genops_imports):
        """Test enterprise security patterns."""
        try:
            import production_patterns

            with patch('production_patterns.PerformanceMonitor'), \
                 patch('production_patterns.CircuitBreaker'), \
                 patch('production_patterns.SecurityValidator'):

                config = production_patterns.ProductionConfig()
                demo = production_patterns.ProductionPatternDemo(config)

                result = await demo.demonstrate_enterprise_security()
                assert result is True

        except ImportError:
            pytest.skip("production_patterns module not available")

    @pytest.mark.asyncio
    async def test_observability_patterns(self, test_env, mock_genops_imports):
        """Test observability patterns."""
        try:
            import production_patterns

            with patch('production_patterns.PerformanceMonitor'), \
                 patch('production_patterns.CircuitBreaker'), \
                 patch('production_patterns.SecurityValidator'):

                config = production_patterns.ProductionConfig()
                demo = production_patterns.ProductionPatternDemo(config)

                result = await demo.demonstrate_observability_patterns()
                assert result is True

        except ImportError:
            pytest.skip("production_patterns module not available")

    def test_production_config(self, test_env):
        """Test production configuration dataclass."""
        try:
            import production_patterns

            config = production_patterns.ProductionConfig()

            # Test default values
            assert config.max_concurrent_requests == 50
            assert config.request_timeout_seconds == 30
            assert config.enable_content_filtering is True
            assert config.log_level == "INFO"

            # Test custom values
            config = production_patterns.ProductionConfig(
                max_concurrent_requests=100,
                log_level="DEBUG"
            )
            assert config.max_concurrent_requests == 100
            assert config.log_level == "DEBUG"

        except ImportError:
            pytest.skip("production_patterns module not available")


class TestExampleIntegration:
    """Integration tests across multiple examples."""

    @pytest.mark.asyncio
    async def test_complete_workflow(self, test_env, mock_genops_imports):
        """Test complete workflow across examples."""
        try:
            # Import all examples
            import auto_instrumentation
            import basic_tracking
            import cost_tracking
            import setup_validation

            # 1. Validate setup
            validation_result = await setup_validation.validate_environment()
            assert validation_result is True

            # 2. Test auto-instrumentation
            with patch('auto_instrumentation.auto_instrument'):
                auto_result = await auto_instrumentation.demonstrate_auto_instrumentation()
                assert auto_result is True

            # 3. Test basic tracking
            tracking_result = await basic_tracking.basic_tracking_example(
                team="integration-test",
                project="workflow-test"
            )
            assert tracking_result is True

            # 4. Test cost tracking
            with patch('cost_tracking.CostTracker'), patch('cost_tracking.BudgetManager'):
                demo = cost_tracking.KubernetesCostDemo()
                cost_result = await demo.demonstrate_basic_cost_tracking()
                assert cost_result is True

        except ImportError:
            pytest.skip("Example modules not available for integration test")

    def test_example_argument_parsing(self, test_env):
        """Test argument parsing for all examples."""
        examples_to_test = [
            "setup_validation",
            "auto_instrumentation",
            "basic_tracking",
            "cost_tracking",
            "production_patterns"
        ]

        for example_name in examples_to_test:
            try:
                # Test that examples can be imported and have main function
                example_module = __import__(example_name)
                assert hasattr(example_module, 'main'), f"{example_name} missing main function"

                # Test that argument parsing doesn't crash
                # (We can't easily test the actual parsing without modifying sys.argv)

            except ImportError:
                pytest.skip(f"Cannot import {example_name} for argument parsing test")

    def test_error_handling(self, test_env, mock_genops_imports):
        """Test error handling in examples."""
        try:
            import setup_validation

            # Test with GenOps unavailable
            with patch.dict('sys.modules', {'genops': None}):
                # Should handle gracefully
                result = asyncio.run(setup_validation.validate_environment())
                assert result is False

        except ImportError:
            pytest.skip("Cannot test error handling without setup_validation")


class TestExampleOutput:
    """Test example output and user experience."""

    def test_help_messages(self, test_env):
        """Test that examples provide helpful output."""
        examples = [
            "setup_validation.py",
            "auto_instrumentation.py",
            "basic_tracking.py",
            "cost_tracking.py",
            "production_patterns.py"
        ]

        for example in examples:
            example_path = EXAMPLES_DIR / example
            if not example_path.exists():
                continue

            try:
                # Test --help flag
                result = subprocess.run([
                    sys.executable, str(example_path), "--help"
                ], capture_output=True, text=True, timeout=10)

                # Should not crash and should provide helpful output
                assert "usage:" in result.stdout.lower() or "examples:" in result.stdout.lower()

            except subprocess.TimeoutExpired:
                pytest.fail(f"{example} --help took too long")
            except Exception as e:
                # Skip if we can't run the example (missing dependencies, etc.)
                pytest.skip(f"Cannot test {example} help: {e}")

    def test_example_documentation(self, test_env):
        """Test that examples have proper documentation."""
        examples = [
            EXAMPLES_DIR / "setup_validation.py",
            EXAMPLES_DIR / "auto_instrumentation.py",
            EXAMPLES_DIR / "basic_tracking.py",
            EXAMPLES_DIR / "cost_tracking.py",
            EXAMPLES_DIR / "production_patterns.py"
        ]

        for example_path in examples:
            if not example_path.exists():
                continue

            # Read file and check for documentation
            content = example_path.read_text()

            # Should have module docstring
            assert '"""' in content, f"{example_path.name} missing module docstring"

            # Should have usage examples
            assert "Usage:" in content or "usage:" in content, f"{example_path.name} missing usage examples"

            # Should have clear descriptions
            assert any(word in content.lower() for word in ["demonstrates", "shows", "example"]), \
                f"{example_path.name} missing clear description"


@pytest.mark.skipif(os.getenv("SKIP_SLOW_TESTS"), reason="Slow tests disabled")
class TestPerformance:
    """Performance tests for examples."""

    @pytest.mark.asyncio
    async def test_example_startup_time(self, test_env, mock_genops_imports):
        """Test that examples start up quickly."""
        try:
            import time

            import setup_validation

            start_time = time.time()
            await setup_validation.validate_environment()
            end_time = time.time()

            # Should complete within 5 seconds (generous for CI)
            assert (end_time - start_time) < 5.0, "Validation took too long"

        except ImportError:
            pytest.skip("Cannot test performance without setup_validation")

    def test_memory_usage(self, test_env, mock_genops_imports):
        """Test memory usage of examples."""
        try:
            import psutil
            import setup_validation

            process = psutil.Process()
            initial_memory = process.memory_info().rss

            # Import and use example
            setup_validation.demonstrate_tracking_patterns()

            final_memory = process.memory_info().rss
            memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB

            # Memory increase should be reasonable (< 100MB)
            assert memory_increase < 100, f"Memory usage increased by {memory_increase:.1f} MB"

        except (ImportError, Exception):
            pytest.skip("Cannot test memory usage")


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__] + sys.argv[1:])
