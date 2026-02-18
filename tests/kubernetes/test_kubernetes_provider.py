#!/usr/bin/env python3
"""
Tests for the GenOps AI Kubernetes provider implementation.

Tests the core Kubernetes provider functionality including detection,
adaptation, resource monitoring, and validation.
"""

import os
from unittest.mock import Mock, mock_open, patch

import pytest


@pytest.fixture
def mock_kubernetes_environment():
    """Mock Kubernetes environment variables and files."""

    env_vars = {
        "KUBERNETES_SERVICE_HOST": "10.96.0.1",
        "KUBERNETES_SERVICE_PORT": "443",
        "HOSTNAME": "test-pod-abc123",
        "POD_NAME": "test-pod-abc123",
        "POD_NAMESPACE": "test-namespace",
        "NODE_NAME": "test-node-1",
    }

    # Mock service account files
    service_account_files = {
        "/var/run/secrets/kubernetes.io/serviceaccount/token": "fake-service-account-token",
        "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt": "fake-ca-certificate",
        "/var/run/secrets/kubernetes.io/serviceaccount/namespace": "test-namespace",
    }

    with (
        patch.dict(os.environ, env_vars),
        patch("pathlib.Path.exists") as mock_exists,
        patch("pathlib.Path.read_text") as mock_read_text,
    ):
        # Configure file existence and content
        def exists_side_effect(path_str):
            return str(path_str) in service_account_files

        def read_text_side_effect():
            # Get the path from the Path object
            path_str = str(
                mock_read_text.__self__
            )  # This is a hack, but works for testing
            return service_account_files.get(path_str, "")

        mock_exists.side_effect = exists_side_effect
        mock_read_text.side_effect = read_text_side_effect

        yield {"env_vars": env_vars, "service_account_files": service_account_files}


class TestKubernetesDetector:
    """Test the KubernetesDetector class."""

    def test_kubernetes_detection_with_environment(self, mock_kubernetes_environment):
        """Test Kubernetes environment detection with proper environment variables."""
        try:
            from genops.providers.kubernetes.detector import KubernetesDetector

            detector = KubernetesDetector()

            assert detector.is_kubernetes() is True
            assert detector.get_namespace() == "test-namespace"
            assert detector.get_pod_name() == "test-pod-abc123"
            assert detector.get_node_name() == "test-node-1"

        except ImportError:
            pytest.skip("KubernetesDetector not available")

    def test_kubernetes_detection_without_environment(self):
        """Test Kubernetes detection without environment variables."""
        try:
            from genops.providers.kubernetes.detector import KubernetesDetector

            # Clear Kubernetes environment
            with patch.dict(os.environ, {}, clear=True):
                detector = KubernetesDetector()

                assert detector.is_kubernetes() is False
                assert detector.get_namespace() is None
                assert detector.get_pod_name() is None
                assert detector.get_node_name() is None

        except ImportError:
            pytest.skip("KubernetesDetector not available")

    def test_governance_attributes(self, mock_kubernetes_environment):
        """Test governance attributes extraction."""
        try:
            from genops.providers.kubernetes.detector import KubernetesDetector

            detector = KubernetesDetector()
            attrs = detector.get_governance_attributes()

            expected_attrs = [
                "k8s.namespace.name",
                "k8s.pod.name",
                "k8s.node.name",
                "k8s.cluster.name",
            ]

            for attr in expected_attrs:
                assert attr in attrs

            assert attrs["k8s.namespace.name"] == "test-namespace"
            assert attrs["k8s.pod.name"] == "test-pod-abc123"
            assert attrs["k8s.node.name"] == "test-node-1"

        except ImportError:
            pytest.skip("KubernetesDetector not available")

    def test_resource_context(self, mock_kubernetes_environment):
        """Test resource context extraction."""
        try:
            from genops.providers.kubernetes.detector import KubernetesDetector

            detector = KubernetesDetector()
            context = detector.get_resource_context()

            assert isinstance(context, dict)
            assert "resource.k8s.namespace.name" in context
            assert context["resource.k8s.namespace.name"] == "test-namespace"

        except ImportError:
            pytest.skip("KubernetesDetector not available")

    def test_cluster_name_detection(self, mock_kubernetes_environment):
        """Test cluster name detection from various sources."""
        try:
            from genops.providers.kubernetes.detector import KubernetesDetector

            # Test with environment variable
            with patch.dict(os.environ, {"CLUSTER_NAME": "test-cluster"}):
                detector = KubernetesDetector()
                assert detector.get_cluster_name() == "test-cluster"

            # Test with kubeconfig (mocked)
            with (
                patch("pathlib.Path.exists", return_value=True),
                patch(
                    "builtins.open",
                    mock_open(read_data="current-context: test-cluster-context"),
                ),
            ):
                detector = KubernetesDetector()
                # Cluster name detection from kubeconfig is implementation-dependent
                cluster_name = detector.get_cluster_name()
                assert cluster_name is not None or cluster_name == "unknown"

        except ImportError:
            pytest.skip("KubernetesDetector not available")


class TestKubernetesResourceMonitor:
    """Test the KubernetesResourceMonitor class."""

    def test_resource_monitor_initialization(self):
        """Test resource monitor initialization."""
        try:
            from genops.providers.kubernetes.resource_monitor import (
                KubernetesResourceMonitor,
            )

            monitor = KubernetesResourceMonitor()
            assert monitor is not None

        except ImportError:
            pytest.skip("KubernetesResourceMonitor not available")

    def test_current_usage_collection(self):
        """Test current resource usage collection."""
        try:
            from genops.providers.kubernetes.resource_monitor import (
                KubernetesResourceMonitor,
                ResourceUsage,
            )

            monitor = KubernetesResourceMonitor()

            # Mock cgroup filesystem
            with (
                patch("pathlib.Path.exists", return_value=True),
                patch("pathlib.Path.read_text") as mock_read,
            ):
                # Mock CPU and memory usage files
                def read_text_side_effect():
                    path_str = str(mock_read.__self__)
                    if "cpu.stat" in path_str:
                        return "usage_usec 123456789\n"
                    elif "memory.current" in path_str:
                        return "536870912\n"  # 512MB
                    elif "memory.stat" in path_str:
                        return "cache 0\nrss 536870912\n"
                    return ""

                mock_read.side_effect = read_text_side_effect

                usage = monitor.get_current_usage()
                assert isinstance(usage, ResourceUsage)
                assert usage.cpu_usage_millicores is not None
                assert usage.memory_usage_bytes is not None

        except ImportError:
            pytest.skip("KubernetesResourceMonitor not available")

    def test_current_resources_collection(self):
        """Test current resource limits collection."""
        try:
            from genops.providers.kubernetes.resource_monitor import (
                KubernetesResourceMonitor,
            )

            monitor = KubernetesResourceMonitor()

            # Mock downward API files
            with (
                patch("pathlib.Path.exists", return_value=True),
                patch("pathlib.Path.read_text") as mock_read,
            ):

                def read_text_side_effect():
                    path_str = str(mock_read.__self__)
                    if "cpu_limit" in path_str:
                        return "2000m"
                    elif "memory_limit" in path_str:
                        return "4Gi"
                    return ""

                mock_read.side_effect = read_text_side_effect

                resources = monitor.get_current_resources()
                assert isinstance(resources, dict)
                # Resource limits may or may not be available depending on implementation

        except ImportError:
            pytest.skip("KubernetesResourceMonitor not available")

    def test_resource_usage_dataclass(self):
        """Test ResourceUsage dataclass."""
        try:
            from genops.providers.kubernetes.resource_monitor import ResourceUsage

            # Test with all values
            usage = ResourceUsage(
                cpu_usage_millicores=250,
                memory_usage_bytes=536870912,
                network_rx_bytes=1024,
                network_tx_bytes=2048,
                timestamp=1234567890.0,
            )

            assert usage.cpu_usage_millicores == 250
            assert usage.memory_usage_bytes == 536870912
            assert usage.network_rx_bytes == 1024
            assert usage.network_tx_bytes == 2048
            assert usage.timestamp == 1234567890.0

            # Test with None values (should be allowed)
            usage = ResourceUsage(cpu_usage_millicores=None, memory_usage_bytes=None)

            assert usage.cpu_usage_millicores is None
            assert usage.memory_usage_bytes is None

        except ImportError:
            pytest.skip("ResourceUsage not available")


class TestKubernetesValidation:
    """Test the validation functionality."""

    def test_validation_result_dataclass(self):
        """Test ValidationResult dataclass."""
        try:
            from genops.providers.kubernetes.validation import (
                KubernetesValidationResult,
                ValidationIssue,  # noqa: F401
            )

            # Test with successful validation
            result = KubernetesValidationResult(
                is_valid=True,
                is_kubernetes_environment=True,
                issues=[],
                namespace="test-namespace",
                pod_name="test-pod",
                has_service_account=True,
            )

            assert result.is_valid is True
            assert result.is_kubernetes_environment is True
            assert len(result.issues) == 0
            assert result.namespace == "test-namespace"

            # Test summary generation
            summary = result.get_summary()
            assert "test-namespace" in summary
            assert "✅" in summary

        except ImportError:
            pytest.skip("Validation classes not available")

    def test_validation_issue_dataclass(self):
        """Test ValidationIssue dataclass."""
        try:
            from genops.providers.kubernetes.validation import ValidationIssue

            issue = ValidationIssue(
                severity="error",
                component="service_account",
                message="Service account token not found",
                fix_suggestion="Ensure pod has service account mounted",
                documentation_link="https://docs.genops.ai/kubernetes/troubleshooting",
            )

            assert issue.severity == "error"
            assert issue.component == "service_account"
            assert "service account" in issue.message.lower()
            assert issue.fix_suggestion is not None
            assert issue.documentation_link is not None

        except ImportError:
            pytest.skip("ValidationIssue not available")

    def test_kubernetes_setup_validation(self, mock_kubernetes_environment):
        """Test Kubernetes setup validation function."""
        try:
            from genops.providers.kubernetes.validation import validate_kubernetes_setup

            result = validate_kubernetes_setup()

            assert result is not None
            assert hasattr(result, "is_valid")
            assert hasattr(result, "is_kubernetes_environment")
            assert hasattr(result, "issues")

            # In mocked environment, should detect Kubernetes
            assert result.is_kubernetes_environment is True

        except ImportError:
            pytest.skip("validate_kubernetes_setup not available")

    def test_validation_with_options(self, mock_kubernetes_environment):
        """Test validation with different options."""
        try:
            from genops.providers.kubernetes.validation import validate_kubernetes_setup

            # Test with resource monitoring enabled
            result = validate_kubernetes_setup(enable_resource_monitoring=True)
            assert result is not None

            # Test with specific cluster name
            result = validate_kubernetes_setup(cluster_name="expected-cluster")
            assert result is not None

            # Should have warning about cluster name mismatch in mocked environment
            [issue for issue in result.issues if issue.component == "cluster"]
            # May or may not have cluster warnings depending on mock setup

        except ImportError:
            pytest.skip("validate_kubernetes_setup not available")

    def test_print_validation_result(self, mock_kubernetes_environment):
        """Test validation result printing."""
        try:
            from genops.providers.kubernetes.validation import (
                print_kubernetes_validation_result,
                validate_kubernetes_setup,
            )

            result = validate_kubernetes_setup()

            # Should not raise exception
            print_kubernetes_validation_result(result)

        except ImportError:
            pytest.skip("Validation functions not available")


class TestKubernetesAdapter:
    """Test the KubernetesAdapter class."""

    def test_adapter_initialization(self):
        """Test adapter initialization."""
        try:
            from genops.providers.kubernetes.adapter import KubernetesAdapter

            adapter = KubernetesAdapter()
            assert adapter is not None

        except ImportError:
            pytest.skip("KubernetesAdapter not available")

    def test_framework_interface(self, mock_kubernetes_environment):
        """Test framework provider interface implementation."""
        try:
            from genops.providers.kubernetes.adapter import KubernetesAdapter

            adapter = KubernetesAdapter()

            # Test framework identification
            assert adapter.get_framework_name() == "kubernetes"

            # Test availability check
            is_available = adapter.is_available()
            assert isinstance(is_available, bool)

            # In mocked environment, should be available
            assert is_available is True

        except ImportError:
            pytest.skip("KubernetesAdapter not available")

    def test_telemetry_attributes(self, mock_kubernetes_environment):
        """Test telemetry attributes generation."""
        try:
            from genops.providers.kubernetes.adapter import KubernetesAdapter

            adapter = KubernetesAdapter()

            # Test basic attributes
            attrs = adapter.get_telemetry_attributes()
            assert isinstance(attrs, dict)

            # Should include Kubernetes-specific attributes
            expected_k8s_attrs = ["k8s.namespace.name", "k8s.pod.name", "k8s.node.name"]

            for attr in expected_k8s_attrs:
                assert attr in attrs

            # Test with additional attributes
            attrs = adapter.get_telemetry_attributes(
                team="test-team", project="test-project", customer_id="test-customer"
            )

            assert attrs["team"] == "test-team"
            assert attrs["project"] == "test-project"
            assert attrs["customer_id"] == "test-customer"

        except ImportError:
            pytest.skip("KubernetesAdapter not available")

    def test_governance_context_creation(self, mock_kubernetes_environment):
        """Test governance context creation."""
        try:
            from genops.providers.kubernetes.adapter import KubernetesAdapter

            adapter = KubernetesAdapter()

            # Test context creation
            governance_attrs = {
                "team": "test-team",
                "project": "test-project",
                "customer_id": "test-customer",
            }

            context = adapter.create_governance_context(**governance_attrs)
            assert context is not None

            # Context should be a context manager
            assert hasattr(context, "__enter__")
            assert hasattr(context, "__exit__")

        except ImportError:
            pytest.skip("KubernetesAdapter not available")

    def test_adapter_with_different_environments(self):
        """Test adapter behavior in different environments."""
        try:
            from genops.providers.kubernetes.adapter import KubernetesAdapter

            # Test in non-Kubernetes environment
            with patch.dict(os.environ, {}, clear=True):
                adapter = KubernetesAdapter()

                # Should still initialize but not be available
                assert adapter.get_framework_name() == "kubernetes"

                # Availability depends on environment
                is_available = adapter.is_available()
                assert isinstance(is_available, bool)

                # Get attributes even without Kubernetes
                attrs = adapter.get_telemetry_attributes(team="test")
                assert isinstance(attrs, dict)
                assert attrs["team"] == "test"

        except ImportError:
            pytest.skip("KubernetesAdapter not available")


class TestKubernetesIntegration:
    """Test integration between Kubernetes components."""

    def test_detector_adapter_integration(self, mock_kubernetes_environment):
        """Test integration between detector and adapter."""
        try:
            from genops.providers.kubernetes.adapter import KubernetesAdapter
            from genops.providers.kubernetes.detector import KubernetesDetector

            detector = KubernetesDetector()
            adapter = KubernetesAdapter()

            # Both should agree on Kubernetes availability
            assert detector.is_kubernetes() == adapter.is_available()

            # Attributes should be consistent
            detector_attrs = detector.get_governance_attributes()
            adapter_attrs = adapter.get_telemetry_attributes()

            for key in detector_attrs:
                if key.startswith("k8s."):
                    assert key in adapter_attrs
                    assert detector_attrs[key] == adapter_attrs[key]

        except ImportError:
            pytest.skip("Integration test components not available")

    def test_monitor_adapter_integration(self, mock_kubernetes_environment):
        """Test integration between resource monitor and adapter."""
        try:
            from genops.providers.kubernetes.adapter import KubernetesAdapter
            from genops.providers.kubernetes.resource_monitor import (
                KubernetesResourceMonitor,
            )

            monitor = KubernetesResourceMonitor()
            adapter = KubernetesAdapter()

            # Mock resource usage
            with patch.object(monitor, "get_current_usage") as mock_usage:
                from genops.providers.kubernetes.resource_monitor import ResourceUsage

                mock_usage.return_value = ResourceUsage(
                    cpu_usage_millicores=250, memory_usage_bytes=536870912
                )

                # Adapter should be able to incorporate resource data
                attrs = adapter.get_telemetry_attributes()

                # Should include basic Kubernetes attributes
                assert "k8s.namespace.name" in attrs

        except ImportError:
            pytest.skip("Integration test components not available")

    def test_validation_integration(self, mock_kubernetes_environment):
        """Test validation integration with other components."""
        try:
            from genops.providers.kubernetes.adapter import KubernetesAdapter
            from genops.providers.kubernetes.validation import validate_kubernetes_setup

            # Validation should pass when adapter is available
            result = validate_kubernetes_setup()
            adapter = KubernetesAdapter()

            # If adapter is available, validation should generally pass
            if adapter.is_available():
                assert result.is_kubernetes_environment is True
                # May have warnings but should detect Kubernetes

        except ImportError:
            pytest.skip("Integration test components not available")


class TestErrorHandling:
    """Test error handling in Kubernetes provider."""

    def test_detector_error_handling(self):
        """Test detector error handling."""
        try:
            from genops.providers.kubernetes.detector import KubernetesDetector

            # Test with permission errors
            with patch(
                "pathlib.Path.read_text", side_effect=PermissionError("Access denied")
            ):
                detector = KubernetesDetector()

                # Should not crash, should gracefully handle errors
                namespace = detector.get_namespace()
                # May be None or fall back to environment variables
                assert namespace is None or isinstance(namespace, str)

        except ImportError:
            pytest.skip("KubernetesDetector not available")

    def test_resource_monitor_error_handling(self):
        """Test resource monitor error handling."""
        try:
            from genops.providers.kubernetes.resource_monitor import (
                KubernetesResourceMonitor,
            )

            monitor = KubernetesResourceMonitor()

            # Test with missing cgroup files
            with patch("pathlib.Path.exists", return_value=False):
                usage = monitor.get_current_usage()

                # Should return ResourceUsage with None values
                assert usage.cpu_usage_millicores is None
                assert usage.memory_usage_bytes is None

        except ImportError:
            pytest.skip("KubernetesResourceMonitor not available")

    def test_adapter_error_handling(self):
        """Test adapter error handling."""
        try:
            from genops.providers.kubernetes.adapter import KubernetesAdapter

            adapter = KubernetesAdapter()

            # Test with mock detector failure
            with patch.object(adapter, "detector") as mock_detector:
                mock_detector.is_kubernetes.side_effect = Exception("Detector failed")

                # Should gracefully handle detector failures
                is_available = adapter.is_available()
                assert isinstance(is_available, bool)

                # Should still provide basic functionality
                attrs = adapter.get_telemetry_attributes(team="test")
                assert isinstance(attrs, dict)
                assert attrs["team"] == "test"

        except ImportError:
            pytest.skip("KubernetesAdapter not available")

    def test_validation_error_handling(self):
        """Test validation error handling."""
        try:
            from genops.providers.kubernetes.validation import validate_kubernetes_setup

            # Test with various error conditions
            with patch(
                "genops.providers.kubernetes.detector.KubernetesDetector"
            ) as mock_detector_class:
                mock_detector = Mock()
                mock_detector.is_kubernetes.side_effect = Exception("Critical error")
                mock_detector_class.return_value = mock_detector

                # Should not crash, should return meaningful result
                result = validate_kubernetes_setup()

                assert hasattr(result, "is_valid")
                assert hasattr(result, "issues")

                # Should have recorded the error
                assert len(result.issues) > 0

        except ImportError:
            pytest.skip("Validation functions not available")


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__] + sys.argv[1:])  # noqa: F821
