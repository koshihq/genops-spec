"""
Pytest configuration and fixtures for Kubernetes tests.

This file provides common fixtures and configuration for all Kubernetes tests.
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any, Generator


def pytest_configure(config):
    """Configure pytest for Kubernetes tests."""
    
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "kubernetes: marks tests that require Kubernetes"
    )
    
    # Set test environment
    os.environ["GENOPS_ENV"] = "test"
    os.environ["LOG_LEVEL"] = "DEBUG"


def pytest_collection_modifyitems(config, items):
    """Modify test items during collection."""
    
    # Skip tests that require Kubernetes if not available
    if not _is_kubernetes_available():
        skip_k8s = pytest.mark.skip(reason="Kubernetes not available")
        for item in items:
            if "kubernetes" in item.keywords:
                item.add_marker(skip_k8s)
    
    # Mark slow tests
    for item in items:
        if "slow" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.slow)


def _is_kubernetes_available() -> bool:
    """Check if Kubernetes is available for testing."""
    try:
        import subprocess
        result = subprocess.run(
            ["kubectl", "cluster-info"], 
            capture_output=True, 
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


@pytest.fixture(scope="session")
def kubernetes_available():
    """Session-scoped fixture to check Kubernetes availability."""
    return _is_kubernetes_available()


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_env_vars():
    """Fixture to mock environment variables."""
    original_env = os.environ.copy()
    
    def _set_env_vars(env_vars: Dict[str, str]):
        os.environ.update(env_vars)
        return env_vars
    
    def _reset_env():
        os.environ.clear()
        os.environ.update(original_env)
    
    yield _set_env_vars
    _reset_env()


@pytest.fixture
def mock_kubernetes_files(temp_dir):
    """Mock Kubernetes-related files."""
    
    # Create mock service account directory
    service_account_dir = temp_dir / "var" / "run" / "secrets" / "kubernetes.io" / "serviceaccount"
    service_account_dir.mkdir(parents=True)
    
    # Create mock service account files
    (service_account_dir / "token").write_text("fake-service-account-token")
    (service_account_dir / "ca.crt").write_text("fake-ca-certificate")
    (service_account_dir / "namespace").write_text("test-namespace")
    
    # Create mock cgroup directory
    cgroup_dir = temp_dir / "sys" / "fs" / "cgroup"
    cgroup_dir.mkdir(parents=True)
    
    # Create mock cgroup files
    (cgroup_dir / "cpu.stat").write_text("usage_usec 123456789\n")
    (cgroup_dir / "memory.current").write_text("536870912\n")
    (cgroup_dir / "memory.stat").write_text("cache 0\nrss 536870912\n")
    
    return {
        "service_account_dir": service_account_dir,
        "cgroup_dir": cgroup_dir
    }


@pytest.fixture
def mock_genops_modules():
    """Mock GenOps modules for testing without installation."""
    
    # Create comprehensive mocks
    mock_modules = {}
    
    # Mock validation result
    mock_validation_result = Mock()
    mock_validation_result.is_valid = True
    mock_validation_result.is_kubernetes_environment = True
    mock_validation_result.namespace = "test-namespace"
    mock_validation_result.pod_name = "test-pod-abc123"
    mock_validation_result.node_name = "test-node-1"
    mock_validation_result.cluster_name = "test-cluster"
    mock_validation_result.has_service_account = True
    mock_validation_result.has_resource_monitoring = True
    mock_validation_result.issues = []
    mock_validation_result.get_summary.return_value = "✅ Kubernetes environment valid (namespace: test-namespace)"
    
    # Mock validation issue
    mock_validation_issue = Mock()
    mock_validation_issue.severity = "warning"
    mock_validation_issue.component = "test"
    mock_validation_issue.message = "Test warning"
    mock_validation_issue.fix_suggestion = "Test fix"
    
    # Mock resource usage
    mock_resource_usage = Mock()
    mock_resource_usage.cpu_usage_millicores = 250
    mock_resource_usage.memory_usage_bytes = 536870912
    mock_resource_usage.network_rx_bytes = 1024
    mock_resource_usage.network_tx_bytes = 2048
    mock_resource_usage.timestamp = 1234567890.0
    
    # Mock Kubernetes detector
    mock_detector = Mock()
    mock_detector.is_kubernetes.return_value = True
    mock_detector.get_namespace.return_value = "test-namespace"
    mock_detector.get_pod_name.return_value = "test-pod-abc123"
    mock_detector.get_node_name.return_value = "test-node-1"
    mock_detector.get_cluster_name.return_value = "test-cluster"
    mock_detector.get_governance_attributes.return_value = {
        "k8s.namespace.name": "test-namespace",
        "k8s.pod.name": "test-pod-abc123",
        "k8s.node.name": "test-node-1",
        "k8s.cluster.name": "test-cluster"
    }
    mock_detector.get_resource_context.return_value = {
        "resource.k8s.namespace.name": "test-namespace",
        "resource.k8s.pod.name": "test-pod-abc123"
    }
    
    # Mock resource monitor
    mock_resource_monitor = Mock()
    mock_resource_monitor.get_current_usage.return_value = mock_resource_usage
    mock_resource_monitor.get_current_resources.return_value = {
        "cpu_limit": "2000m",
        "memory_limit": "4Gi"
    }
    
    # Mock Kubernetes adapter
    mock_adapter = Mock()
    mock_adapter.is_available.return_value = True
    mock_adapter.get_framework_name.return_value = "kubernetes"
    mock_adapter.get_telemetry_attributes.return_value = {
        "k8s.namespace.name": "test-namespace",
        "k8s.pod.name": "test-pod-abc123",
        "k8s.node.name": "test-node-1",
        "team": "test-team",
        "project": "test-project"
    }
    
    # Mock governance context
    mock_governance_context = Mock()
    mock_governance_context.context_id = "test-context-123"
    mock_governance_context.get_duration.return_value = 1.234
    mock_governance_context.get_cost_summary.return_value = {"total_cost": 0.0023}
    mock_governance_context.get_telemetry_data.return_value = {
        "k8s.namespace.name": "test-namespace",
        "team": "test-team"
    }
    mock_governance_context.get_resource_usage.return_value = {
        "cpu_usage_millicores": 250,
        "memory_usage_bytes": 536870912
    }
    mock_governance_context.add_cost_data = Mock()
    mock_governance_context.add_metadata = Mock()
    
    # Mock context manager
    mock_context_manager = Mock()
    mock_context_manager.__enter__ = Mock(return_value=mock_governance_context)
    mock_context_manager.__exit__ = Mock(return_value=None)
    mock_adapter.create_governance_context.return_value = mock_context_manager
    
    # Mock cost tracker and budget manager
    mock_cost_tracker = Mock()
    mock_budget_manager = Mock()
    
    # Mock performance monitor, circuit breaker, security validator
    mock_performance_monitor = Mock()
    mock_circuit_breaker = Mock()
    mock_circuit_breaker.record_success = Mock()
    mock_circuit_breaker.record_failure = Mock()
    mock_circuit_breaker.get_state.return_value = "CLOSED"
    mock_security_validator = Mock()
    
    # Mock auto_instrument function
    mock_auto_instrument = Mock()
    
    # Mock active instrumentations
    mock_get_active_instrumentations = Mock(return_value={
        "openai": {"status": "active"},
        "anthropic": {"status": "active"},
        "langchain": {"status": "active"}
    })
    
    # Create module mocks
    mock_modules = {
        'genops': Mock(auto_instrument=mock_auto_instrument),
        'genops.providers': Mock(),
        'genops.providers.kubernetes': Mock(
            KubernetesDetector=Mock(return_value=mock_detector),
            KubernetesResourceMonitor=Mock(return_value=mock_resource_monitor),
            KubernetesAdapter=Mock(return_value=mock_adapter),
            validate_kubernetes_setup=Mock(return_value=mock_validation_result),
            print_kubernetes_validation_result=Mock()
        ),
        'genops.providers.kubernetes.detector': Mock(
            KubernetesDetector=Mock(return_value=mock_detector)
        ),
        'genops.providers.kubernetes.resource_monitor': Mock(
            KubernetesResourceMonitor=Mock(return_value=mock_resource_monitor),
            ResourceUsage=Mock(return_value=mock_resource_usage)
        ),
        'genops.providers.kubernetes.adapter': Mock(
            KubernetesAdapter=Mock(return_value=mock_adapter)
        ),
        'genops.providers.kubernetes.validation': Mock(
            validate_kubernetes_setup=Mock(return_value=mock_validation_result),
            print_kubernetes_validation_result=Mock(),
            KubernetesValidationResult=Mock(return_value=mock_validation_result),
            ValidationIssue=Mock(return_value=mock_validation_issue)
        ),
        'genops.core': Mock(),
        'genops.core.governance': Mock(
            create_governance_context=Mock(return_value=mock_governance_context)
        ),
        'genops.core.cost': Mock(
            CostTracker=Mock(return_value=mock_cost_tracker),
            BudgetManager=Mock(return_value=mock_budget_manager)
        ),
        'genops.core.performance': Mock(
            PerformanceMonitor=Mock(return_value=mock_performance_monitor),
            CircuitBreaker=Mock(return_value=mock_circuit_breaker)
        ),
        'genops.core.security': Mock(
            SecurityValidator=Mock(return_value=mock_security_validator),
            ContentFilter=Mock()
        ),
        'genops.core.instrumentation': Mock(
            get_active_instrumentations=mock_get_active_instrumentations
        )
    }
    
    with patch.dict('sys.modules', mock_modules):
        yield {
            'validation_result': mock_validation_result,
            'detector': mock_detector,
            'resource_monitor': mock_resource_monitor,
            'adapter': mock_adapter,
            'governance_context': mock_governance_context,
            'cost_tracker': mock_cost_tracker,
            'budget_manager': mock_budget_manager,
            'performance_monitor': mock_performance_monitor,
            'circuit_breaker': mock_circuit_breaker,
            'security_validator': mock_security_validator
        }


@pytest.fixture
def mock_ai_providers():
    """Mock AI provider SDKs for testing."""
    
    # Mock OpenAI
    mock_openai_client = Mock()
    mock_openai_response = Mock()
    mock_openai_response.choices = [Mock()]
    mock_openai_response.choices[0].message.content = "Hello from mocked OpenAI!"
    mock_openai_client.chat.completions.create.return_value = mock_openai_response
    
    # Mock Anthropic
    mock_anthropic_client = Mock()
    mock_anthropic_response = Mock()
    mock_anthropic_response.content = [Mock()]
    mock_anthropic_response.content[0].text = "Hello from mocked Anthropic!"
    mock_anthropic_client.messages.create.return_value = mock_anthropic_response
    
    mocks = {
        'openai': Mock(
            AsyncOpenAI=Mock(return_value=mock_openai_client),
            OpenAI=Mock(return_value=mock_openai_client)
        ),
        'anthropic': Mock(
            AsyncAnthropic=Mock(return_value=mock_anthropic_client),
            Anthropic=Mock(return_value=mock_anthropic_client)
        )
    }
    
    with patch.dict('sys.modules', mocks):
        yield {
            'openai_client': mock_openai_client,
            'anthropic_client': mock_anthropic_client
        }


@pytest.fixture
def capture_output():
    """Capture stdout and stderr for testing output."""
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
        yield {
            'stdout': stdout_capture,
            'stderr': stderr_capture
        }


@pytest.fixture
def examples_dir():
    """Path to the examples directory."""
    return Path(__file__).parent.parent.parent / "examples" / "kubernetes"


@pytest.fixture(autouse=True)
def add_examples_to_path(examples_dir):
    """Automatically add examples directory to Python path."""
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    yield
    if str(examples_dir) in sys.path:
        sys.path.remove(str(examples_dir))


@pytest.fixture
def kubernetes_config():
    """Mock Kubernetes configuration for testing."""
    return {
        "namespace": "test-namespace",
        "pod_name": "test-pod-abc123",
        "node_name": "test-node-1", 
        "cluster_name": "test-cluster",
        "service_account": True,
        "resource_monitoring": True
    }


@pytest.fixture
def sample_governance_attributes():
    """Sample governance attributes for testing."""
    return {
        "team": "test-team",
        "project": "test-project",
        "customer_id": "test-customer-123",
        "environment": "test",
        "cost_center": "engineering",
        "feature": "kubernetes-testing"
    }


# Performance test utilities
@pytest.fixture
def performance_timer():
    """Timer fixture for performance tests."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.time()
        
        def stop(self):
            self.end_time = time.time()
        
        @property
        def elapsed(self):
            if self.start_time is None or self.end_time is None:
                return None
            return self.end_time - self.start_time
    
    return Timer()


# Skip markers for different test types
skip_without_genops = pytest.mark.skipif(
    not _check_genops_available(),
    reason="GenOps not installed"
)

skip_without_kubernetes = pytest.mark.skipif(
    not _is_kubernetes_available(),
    reason="Kubernetes not available"
)

skip_slow_tests = pytest.mark.skipif(
    os.getenv("SKIP_SLOW_TESTS", "false").lower() == "true",
    reason="Slow tests disabled"
)


def _check_genops_available() -> bool:
    """Check if GenOps is available for import."""
    try:
        import genops
        return True
    except ImportError:
        return False


# Test data
SAMPLE_TELEMETRY_ATTRIBUTES = {
    "k8s.namespace.name": "test-namespace",
    "k8s.pod.name": "test-pod-abc123",
    "k8s.node.name": "test-node-1",
    "k8s.cluster.name": "test-cluster",
    "team": "test-team",
    "project": "test-project",
    "customer_id": "test-customer",
    "environment": "test"
}

SAMPLE_COST_DATA = {
    "provider": "openai",
    "model": "gpt-3.5-turbo",
    "cost": 0.0023,
    "tokens_in": 15,
    "tokens_out": 50,
    "operation": "chat_completion"
}

SAMPLE_RESOURCE_USAGE = {
    "cpu_usage_millicores": 250,
    "memory_usage_bytes": 536870912,  # 512MB
    "network_rx_bytes": 1024,
    "network_tx_bytes": 2048
}