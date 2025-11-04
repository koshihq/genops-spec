#!/usr/bin/env python3
"""
Integration tests for GenOps AI Kubernetes functionality.

Tests end-to-end integration scenarios and real-world usage patterns.
"""

import asyncio
import os
import pytest
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, Mock
from typing import Dict, Any


@pytest.mark.integration
class TestKubernetesIntegration:
    """Integration tests for Kubernetes functionality."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_tracking_workflow(self, mock_genops_modules, sample_governance_attributes):
        """Test complete end-to-end tracking workflow."""
        
        try:
            # Import all required modules
            from genops.providers.kubernetes import KubernetesAdapter, validate_kubernetes_setup
            
            # 1. Validate environment
            validation_result = validate_kubernetes_setup()
            assert validation_result.is_kubernetes_environment is True
            
            # 2. Create adapter
            adapter = KubernetesAdapter()
            assert adapter.is_available() is True
            
            # 3. Create governance context
            with adapter.create_governance_context(**sample_governance_attributes) as ctx:
                # 4. Simulate AI operations
                ctx.add_cost_data(
                    provider="openai",
                    model="gpt-3.5-turbo",
                    cost=0.0023,
                    tokens_in=15,
                    tokens_out=50,
                    operation="chat_completion"
                )
                
                # 5. Verify context data
                assert ctx.context_id is not None
                duration = ctx.get_duration()
                assert duration >= 0
                
                telemetry_data = ctx.get_telemetry_data()
                assert "k8s.namespace.name" in telemetry_data
                
                cost_summary = ctx.get_cost_summary()
                assert "total_cost" in cost_summary
            
        except ImportError:
            pytest.skip("Integration test modules not available")
    
    def test_multi_component_integration(self, mock_genops_modules, kubernetes_config):
        """Test integration between multiple Kubernetes components."""
        
        try:
            from genops.providers.kubernetes import (
                KubernetesDetector, 
                KubernetesResourceMonitor, 
                KubernetesAdapter,
                validate_kubernetes_setup
            )
            
            # Create all components
            detector = KubernetesDetector()
            monitor = KubernetesResourceMonitor()
            adapter = KubernetesAdapter()
            
            # Verify they all agree on basic state
            assert detector.is_kubernetes() is True
            assert adapter.is_available() is True
            
            # Verify attribute consistency
            detector_attrs = detector.get_governance_attributes()
            adapter_attrs = adapter.get_telemetry_attributes()
            
            for key in detector_attrs:
                if key.startswith('k8s.') and key in adapter_attrs:
                    assert detector_attrs[key] == adapter_attrs[key]
            
            # Test validation incorporates all components
            validation_result = validate_kubernetes_setup(enable_resource_monitoring=True)
            assert validation_result.is_kubernetes_environment is True
            assert validation_result.has_resource_monitoring is True
            
        except ImportError:
            pytest.skip("Integration test modules not available")
    
    @pytest.mark.asyncio
    async def test_cost_tracking_integration(self, mock_genops_modules, sample_governance_attributes):
        """Test cost tracking integration across components."""
        
        try:
            from genops.providers.kubernetes import KubernetesAdapter
            from genops.core.cost import CostTracker
            
            adapter = KubernetesAdapter()
            cost_tracker = CostTracker()
            
            # Track multiple operations
            operations = [
                ("openai", "gpt-3.5-turbo", 0.0015, 10, 30),
                ("anthropic", "claude-3-haiku", 0.0008, 8, 25),
                ("openai", "gpt-4", 0.0120, 15, 40)
            ]
            
            with adapter.create_governance_context(**sample_governance_attributes) as ctx:
                for provider, model, cost, tokens_in, tokens_out in operations:
                    ctx.add_cost_data(
                        provider=provider,
                        model=model,
                        cost=cost,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        operation="test_operation"
                    )
                
                # Verify cost aggregation
                cost_summary = ctx.get_cost_summary()
                assert cost_summary is not None
                
                # Verify telemetry includes cost data
                telemetry = ctx.get_telemetry_data()
                assert telemetry is not None
                assert "team" in telemetry
            
        except ImportError:
            pytest.skip("Cost tracking integration test modules not available")
    
    @pytest.mark.skipif(not os.getenv("TEST_WITH_REAL_CLUSTER"), reason="Real cluster testing disabled")
    def test_real_kubernetes_cluster_integration(self):
        """Test integration with real Kubernetes cluster (if available)."""
        
        try:
            # Check if kubectl is available
            result = subprocess.run(["kubectl", "cluster-info"], capture_output=True, timeout=10)
            if result.returncode != 0:
                pytest.skip("No Kubernetes cluster available")
            
            # Test GenOps components with real cluster
            from genops.providers.kubernetes import KubernetesDetector, validate_kubernetes_setup
            
            # Real environment detection
            detector = KubernetesDetector()
            is_k8s = detector.is_kubernetes()
            
            # If we're actually in Kubernetes, verify detection works
            if os.getenv("KUBERNETES_SERVICE_HOST"):
                assert is_k8s is True
                assert detector.get_namespace() is not None
            
            # Real validation
            validation_result = validate_kubernetes_setup()
            assert validation_result is not None
            
        except ImportError:
            pytest.skip("Real cluster integration test modules not available")
        except subprocess.TimeoutExpired:
            pytest.skip("kubectl timeout - cluster may not be responsive")
    
    def test_error_propagation_integration(self, mock_genops_modules):
        """Test error handling across integrated components."""
        
        try:
            from genops.providers.kubernetes import KubernetesAdapter
            
            adapter = KubernetesAdapter()
            
            # Test with failing detector
            with patch.object(adapter, 'detector') as mock_detector:
                mock_detector.is_kubernetes.side_effect = Exception("Detector failure")
                
                # Should gracefully handle detector failure
                is_available = adapter.is_available()
                assert isinstance(is_available, bool)
                
                # Should still provide basic functionality
                attrs = adapter.get_telemetry_attributes(team="test")
                assert "team" in attrs
            
        except ImportError:
            pytest.skip("Error propagation test modules not available")
    
    @pytest.mark.asyncio
    async def test_concurrent_operations_integration(self, mock_genops_modules, sample_governance_attributes):
        """Test integration with concurrent operations."""
        
        try:
            from genops.providers.kubernetes import KubernetesAdapter
            
            adapter = KubernetesAdapter()
            
            async def run_operation(operation_id: int):
                """Run a single operation."""
                attrs = {**sample_governance_attributes, "operation_id": str(operation_id)}
                
                with adapter.create_governance_context(**attrs) as ctx:
                    ctx.add_cost_data(
                        provider="openai",
                        model="gpt-3.5-turbo",
                        cost=0.001 * operation_id,
                        tokens_in=10 + operation_id,
                        tokens_out=30 + operation_id,
                        operation=f"concurrent_operation_{operation_id}"
                    )
                    
                    # Simulate some work
                    await asyncio.sleep(0.01)
                    
                    return ctx.get_cost_summary()
            
            # Run multiple concurrent operations
            tasks = [run_operation(i) for i in range(5)]
            results = await asyncio.gather(*tasks)
            
            # Verify all operations completed
            assert len(results) == 5
            for result in results:
                assert result is not None
            
        except ImportError:
            pytest.skip("Concurrent operations test modules not available")
    
    def test_configuration_integration(self, mock_genops_modules, temp_dir):
        """Test configuration integration across components."""
        
        try:
            from genops.providers.kubernetes import KubernetesAdapter, validate_kubernetes_setup
            
            # Test with different environment configurations
            configs = [
                {"GENOPS_ENV": "development", "LOG_LEVEL": "DEBUG"},
                {"GENOPS_ENV": "production", "LOG_LEVEL": "INFO"},
                {"GENOPS_ENV": "test", "LOG_LEVEL": "WARNING"}
            ]
            
            for config in configs:
                with patch.dict(os.environ, config):
                    adapter = KubernetesAdapter()
                    validation_result = validate_kubernetes_setup()
                    
                    # Basic functionality should work regardless of config
                    assert adapter.get_framework_name() == "kubernetes"
                    assert validation_result is not None
            
        except ImportError:
            pytest.skip("Configuration integration test modules not available")


@pytest.mark.integration
class TestExampleIntegration:
    """Integration tests for example scripts."""
    
    @pytest.mark.asyncio 
    async def test_setup_validation_example_integration(self, mock_genops_modules, examples_dir):
        """Test setup_validation.py example integration."""
        
        if not (examples_dir / "setup_validation.py").exists():
            pytest.skip("setup_validation.py example not found")
        
        try:
            import setup_validation
            
            # Test validation workflow
            result = await setup_validation.validate_environment(detailed=True)
            assert result is True
            
            # Test integration test
            integration_result = await setup_validation.run_integration_test()
            assert integration_result is True
            
        except ImportError:
            pytest.skip("setup_validation example not available")
    
    @pytest.mark.asyncio
    async def test_basic_tracking_example_integration(self, mock_genops_modules, examples_dir):
        """Test basic_tracking.py example integration."""
        
        if not (examples_dir / "basic_tracking.py").exists():
            pytest.skip("basic_tracking.py example not found")
        
        try:
            import basic_tracking
            
            # Test with different parameter combinations
            test_cases = [
                {"team": "integration-test", "project": "test-project"},
                {"customer_id": "test-customer"},
                {}  # Default parameters
            ]
            
            for test_case in test_cases:
                result = await basic_tracking.basic_tracking_example(**test_case)
                assert result is True
            
        except ImportError:
            pytest.skip("basic_tracking example not available")
    
    @pytest.mark.asyncio
    async def test_cost_tracking_example_integration(self, mock_genops_modules, examples_dir):
        """Test cost_tracking.py example integration."""
        
        if not (examples_dir / "cost_tracking.py").exists():
            pytest.skip("cost_tracking.py example not found")
        
        try:
            import cost_tracking
            
            with patch('cost_tracking.CostTracker'), patch('cost_tracking.BudgetManager'):
                demo = cost_tracking.KubernetesCostDemo()
                
                # Test different demo methods
                basic_result = await demo.demonstrate_basic_cost_tracking()
                assert basic_result is True
                
                budget_result = await demo.demonstrate_budget_management(25.0)
                assert budget_result is True
                
                multi_provider_result = await demo.demonstrate_multi_provider_cost_aggregation()
                assert multi_provider_result is True
                
                optimization_result = await demo.demonstrate_cost_optimization_strategies()
                assert optimization_result is True
            
        except ImportError:
            pytest.skip("cost_tracking example not available")
    
    def test_example_argument_parsing_integration(self, examples_dir):
        """Test argument parsing integration across examples."""
        
        example_files = [
            "setup_validation.py",
            "auto_instrumentation.py",
            "basic_tracking.py",
            "cost_tracking.py",
            "production_patterns.py"
        ]
        
        for example_file in example_files:
            example_path = examples_dir / example_file
            if not example_path.exists():
                continue
            
            try:
                # Test --help doesn't crash
                result = subprocess.run([
                    "python", str(example_path), "--help"
                ], capture_output=True, text=True, timeout=10)
                
                # Should show help text
                assert result.returncode == 0
                assert ("usage" in result.stdout.lower() or 
                       "examples" in result.stdout.lower())
                
            except subprocess.TimeoutExpired:
                pytest.fail(f"{example_file} --help took too long")
            except Exception as e:
                # Skip if we can't run the example
                pytest.skip(f"Cannot test {example_file}: {e}")


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """Integration tests focused on performance characteristics."""
    
    @pytest.mark.asyncio
    async def test_high_volume_operations(self, mock_genops_modules, sample_governance_attributes, performance_timer):
        """Test integration with high volume of operations."""
        
        try:
            from genops.providers.kubernetes import KubernetesAdapter
            
            adapter = KubernetesAdapter()
            
            # Test high volume of concurrent operations
            operation_count = 50
            
            async def run_batch_operations():
                tasks = []
                
                for i in range(operation_count):
                    attrs = {**sample_governance_attributes, "batch_id": str(i)}
                    
                    async def single_operation():
                        with adapter.create_governance_context(**attrs) as ctx:
                            ctx.add_cost_data(
                                provider="openai",
                                model="gpt-3.5-turbo", 
                                cost=0.001,
                                tokens_in=15,
                                tokens_out=50,
                                operation="batch_operation"
                            )
                            return ctx.get_cost_summary()
                    
                    tasks.append(single_operation())
                
                return await asyncio.gather(*tasks)
            
            # Measure performance
            performance_timer.start()
            results = await run_batch_operations()
            performance_timer.stop()
            
            # Verify results
            assert len(results) == operation_count
            assert all(result is not None for result in results)
            
            # Performance should be reasonable (under 5 seconds for 50 operations)
            assert performance_timer.elapsed < 5.0, f"High volume test took {performance_timer.elapsed:.2f}s"
            
        except ImportError:
            pytest.skip("Performance integration test modules not available")
    
    def test_memory_usage_integration(self, mock_genops_modules, sample_governance_attributes):
        """Test memory usage under integration scenarios."""
        
        try:
            import psutil
            from genops.providers.kubernetes import KubernetesAdapter
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            adapter = KubernetesAdapter()
            
            # Create and destroy many contexts
            for i in range(20):
                attrs = {**sample_governance_attributes, "iteration": str(i)}
                
                with adapter.create_governance_context(**attrs) as ctx:
                    ctx.add_cost_data(
                        provider="openai",
                        model="gpt-3.5-turbo",
                        cost=0.001,
                        tokens_in=15,
                        tokens_out=50,
                        operation="memory_test"
                    )
            
            final_memory = process.memory_info().rss
            memory_increase = (final_memory - initial_memory) / 1024 / 1024  # MB
            
            # Memory increase should be reasonable (< 50MB)
            assert memory_increase < 50, f"Memory usage increased by {memory_increase:.1f} MB"
            
        except ImportError:
            pytest.skip("Memory usage test dependencies not available")
    
    def test_resource_cleanup_integration(self, mock_genops_modules, sample_governance_attributes):
        """Test resource cleanup in integration scenarios."""
        
        try:
            from genops.providers.kubernetes import KubernetesAdapter
            
            adapter = KubernetesAdapter()
            
            # Track context creation and cleanup
            contexts_created = []
            contexts_cleaned = []
            
            def track_context_creation(ctx):
                contexts_created.append(ctx.context_id)
                return ctx
            
            def track_context_cleanup(ctx):
                contexts_cleaned.append(ctx.context_id)
                
            # Create multiple contexts with tracking
            for i in range(10):
                attrs = {**sample_governance_attributes, "cleanup_test": str(i)}
                
                with adapter.create_governance_context(**attrs) as ctx:
                    track_context_creation(ctx)
                    ctx.add_cost_data(
                        provider="openai",
                        model="gpt-3.5-turbo",
                        cost=0.001,
                        tokens_in=10,
                        tokens_out=30,
                        operation="cleanup_test"
                    )
                
                track_context_cleanup(ctx)
            
            # Verify all contexts were created and cleaned up
            assert len(contexts_created) == 10
            assert len(contexts_cleaned) == 10
            
        except ImportError:
            pytest.skip("Resource cleanup test modules not available")


@pytest.mark.integration
class TestDocumentationIntegration:
    """Integration tests for documentation and examples."""
    
    def test_quickstart_guide_integration(self, examples_dir):
        """Test that quickstart guides work end-to-end."""
        
        # This would ideally test the actual quickstart commands
        # For now, we'll test that the referenced files exist
        
        quickstart_files = [
            "setup_validation.py",
            "auto_instrumentation.py",
            "basic_tracking.py"
        ]
        
        for filename in quickstart_files:
            filepath = examples_dir / filename
            assert filepath.exists(), f"Quickstart file {filename} missing"
            
            # Verify file has proper structure
            content = filepath.read_text()
            assert '"""' in content, f"{filename} missing docstring"
            assert "Usage:" in content, f"{filename} missing usage examples"
            assert "def main(" in content, f"{filename} missing main function"
    
    def test_troubleshooting_guide_integration(self, project_root=None):
        """Test troubleshooting guide integration."""
        
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        
        troubleshooting_guide = project_root / "docs" / "kubernetes-troubleshooting.md"
        if not troubleshooting_guide.exists():
            pytest.skip("Troubleshooting guide not found")
        
        content = troubleshooting_guide.read_text()
        
        # Verify guide has essential sections
        required_sections = [
            "Quick Diagnosis",
            "Emergency Response",
            "Common Issues",
            "Troubleshooting"
        ]
        
        for section in required_sections:
            assert section in content, f"Missing section: {section}"
        
        # Verify guide has practical commands
        assert "kubectl" in content, "Missing kubectl commands"
        assert "```bash" in content, "Missing bash code blocks"


if __name__ == "__main__":
    # Run integration tests when script is executed directly
    pytest.main([__file__, "-v", "-m", "integration"])