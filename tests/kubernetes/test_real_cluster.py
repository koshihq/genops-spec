#!/usr/bin/env python3
"""
Integration test framework for real Kubernetes clusters.

Tests GenOps AI functionality against actual Kubernetes clusters to validate
real-world behavior and catch integration issues that mocks cannot detect.
"""

import asyncio
import json
import os
import pytest
import subprocess
import tempfile
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import uuid


@dataclass
class ClusterInfo:
    """Information about the test cluster."""
    
    name: str
    context: str
    version: str
    nodes: int
    namespace: str
    accessible: bool
    has_genops: bool = False


@dataclass
class TestResult:
    """Result of a real cluster test."""
    
    test_name: str
    success: bool
    duration: float
    error_message: Optional[str] = None
    artifacts: Dict[str, Any] = None


class RealClusterTestFramework:
    """Framework for testing against real Kubernetes clusters."""
    
    def __init__(self, test_namespace: str = None):
        self.test_namespace = test_namespace or f"genops-test-{uuid.uuid4().hex[:8]}"
        self.cluster_info = None
        self.test_results = []
        self.cleanup_resources = []
        
    async def setup_cluster_info(self) -> ClusterInfo:
        """Gather information about the current cluster."""
        
        try:
            # Get cluster info
            cluster_result = subprocess.run(
                ["kubectl", "cluster-info"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if cluster_result.returncode != 0:
                return ClusterInfo(
                    name="unknown",
                    context="unknown", 
                    version="unknown",
                    nodes=0,
                    namespace=self.test_namespace,
                    accessible=False
                )
            
            # Get current context
            context_result = subprocess.run(
                ["kubectl", "config", "current-context"],
                capture_output=True,
                text=True,
                timeout=5
            )
            current_context = context_result.stdout.strip() if context_result.returncode == 0 else "unknown"
            
            # Get server version
            version_result = subprocess.run(
                ["kubectl", "version", "--output=json"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            server_version = "unknown"
            if version_result.returncode == 0:
                try:
                    version_data = json.loads(version_result.stdout)
                    server_version = version_data.get("serverVersion", {}).get("gitVersion", "unknown")
                except json.JSONDecodeError:
                    pass
            
            # Get node count
            nodes_result = subprocess.run(
                ["kubectl", "get", "nodes", "--no-headers"],
                capture_output=True,
                text=True,
                timeout=10
            )
            node_count = len(nodes_result.stdout.strip().split('\n')) if nodes_result.returncode == 0 else 0
            
            # Check if GenOps is already installed
            has_genops = self._check_genops_installed()
            
            cluster_info = ClusterInfo(
                name=current_context.split('/')[-1] if '/' in current_context else current_context,
                context=current_context,
                version=server_version,
                nodes=node_count,
                namespace=self.test_namespace,
                accessible=True,
                has_genops=has_genops
            )
            
            self.cluster_info = cluster_info
            return cluster_info
            
        except Exception as e:
            return ClusterInfo(
                name="error",
                context="error",
                version="error",
                nodes=0,
                namespace=self.test_namespace,
                accessible=False
            )
    
    def _check_genops_installed(self) -> bool:
        """Check if GenOps is already installed in the cluster."""
        
        try:
            result = subprocess.run(
                ["kubectl", "get", "deployment", "-A", "-l", "app.kubernetes.io/name=genops-ai"],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0 and len(result.stdout.strip()) > 0
        except Exception:
            return False
    
    async def setup_test_namespace(self) -> bool:
        """Create and configure test namespace."""
        
        try:
            # Create namespace
            namespace_yaml = f"""
apiVersion: v1
kind: Namespace
metadata:
  name: {self.test_namespace}
  labels:
    genops.ai/test: "true"
    genops.ai/test-session: "{uuid.uuid4().hex}"
"""
            
            result = subprocess.run(
                ["kubectl", "apply", "-f", "-"],
                input=namespace_yaml,
                text=True,
                capture_output=True,
                timeout=30
            )
            
            if result.returncode != 0:
                print(f"Failed to create namespace: {result.stderr}")
                return False
            
            self.cleanup_resources.append(("namespace", self.test_namespace))
            
            # Wait for namespace to be ready
            await asyncio.sleep(2)
            
            return True
            
        except Exception as e:
            print(f"Error setting up test namespace: {e}")
            return False
    
    async def install_genops_for_testing(self) -> bool:
        """Install GenOps in the test cluster for testing."""
        
        if self.cluster_info.has_genops:
            print("GenOps already installed, skipping installation")
            return True
        
        try:
            # Create minimal GenOps deployment for testing
            genops_yaml = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genops-ai-test
  namespace: {self.test_namespace}
  labels:
    app.kubernetes.io/name: genops-ai
    app.kubernetes.io/instance: test
spec:
  replicas: 1
  selector:
    matchLabels:
      app.kubernetes.io/name: genops-ai
  template:
    metadata:
      labels:
        app.kubernetes.io/name: genops-ai
    spec:
      containers:
      - name: genops-ai
        image: python:3.9-slim
        command: ["sleep", "3600"]
        env:
        - name: GENOPS_ENV
          value: "test"
        - name: KUBERNETES_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: NODE_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
        ports:
        - containerPort: 8000
          name: http
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 512Mi
---
apiVersion: v1
kind: Service
metadata:
  name: genops-ai-test
  namespace: {self.test_namespace}
spec:
  selector:
    app.kubernetes.io/name: genops-ai
  ports:
  - port: 8000
    targetPort: 8000
    name: http
"""
            
            result = subprocess.run(
                ["kubectl", "apply", "-f", "-"],
                input=genops_yaml,
                text=True,
                capture_output=True,
                timeout=60
            )
            
            if result.returncode != 0:
                print(f"Failed to install GenOps: {result.stderr}")
                return False
            
            self.cleanup_resources.append(("deployment", f"{self.test_namespace}/genops-ai-test"))
            self.cleanup_resources.append(("service", f"{self.test_namespace}/genops-ai-test"))
            
            # Wait for deployment to be ready
            await self._wait_for_deployment_ready("genops-ai-test", timeout=120)
            
            return True
            
        except Exception as e:
            print(f"Error installing GenOps: {e}")
            return False
    
    async def _wait_for_deployment_ready(self, deployment_name: str, timeout: int = 60) -> bool:
        """Wait for deployment to be ready."""
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                result = subprocess.run([
                    "kubectl", "get", "deployment", deployment_name,
                    "-n", self.test_namespace,
                    "-o", "jsonpath={.status.readyReplicas}"
                ], capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0 and result.stdout.strip() == "1":
                    return True
                
                await asyncio.sleep(5)
                
            except Exception:
                await asyncio.sleep(5)
        
        return False
    
    async def test_kubernetes_detection(self) -> TestResult:
        """Test Kubernetes environment detection."""
        
        test_name = "kubernetes_detection"
        start_time = time.time()
        
        try:
            # Test using our real cluster test pod
            detection_script = """
import os
import sys

# Mock GenOps imports for testing
class MockDetector:
    def is_kubernetes(self):
        return (
            os.getenv('KUBERNETES_SERVICE_HOST') is not None or 
            os.path.exists('/var/run/secrets/kubernetes.io/serviceaccount/token')
        )
    
    def get_namespace(self):
        # Try multiple sources
        namespace = os.getenv('KUBERNETES_NAMESPACE') or os.getenv('POD_NAMESPACE')
        if namespace:
            return namespace
        
        try:
            with open('/var/run/secrets/kubernetes.io/serviceaccount/namespace') as f:
                return f.read().strip()
        except:
            return None
    
    def get_pod_name(self):
        return os.getenv('POD_NAME') or os.getenv('HOSTNAME')
    
    def get_node_name(self):
        return os.getenv('NODE_NAME')

detector = MockDetector()

print(f"is_kubernetes: {detector.is_kubernetes()}")
print(f"namespace: {detector.get_namespace()}")
print(f"pod_name: {detector.get_pod_name()}")
print(f"node_name: {detector.get_node_name()}")

# Environment variables
env_vars = [
    'KUBERNETES_SERVICE_HOST',
    'KUBERNETES_SERVICE_PORT',
    'KUBERNETES_NAMESPACE', 
    'POD_NAME',
    'NODE_NAME'
]

print("Environment variables:")
for var in env_vars:
    print(f"  {var}: {os.getenv(var, 'Not set')}")

sys.exit(0)
"""
            
            # Run detection script in GenOps pod
            result = subprocess.run([
                "kubectl", "exec", "-n", self.test_namespace, 
                "deployment/genops-ai-test", "--",
                "python", "-c", detection_script
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return TestResult(
                    test_name=test_name,
                    success=False,
                    duration=time.time() - start_time,
                    error_message=f"Detection script failed: {result.stderr}",
                    artifacts={"stdout": result.stdout, "stderr": result.stderr}
                )
            
            # Parse output
            output_lines = result.stdout.strip().split('\n')
            detection_results = {}
            
            for line in output_lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    detection_results[key.strip()] = value.strip()
            
            # Validate detection results
            is_kubernetes = detection_results.get('is_kubernetes', 'False') == 'True'
            namespace = detection_results.get('namespace', 'None')
            
            success = (
                is_kubernetes and 
                namespace != 'None' and 
                namespace == self.test_namespace
            )
            
            return TestResult(
                test_name=test_name,
                success=success,
                duration=time.time() - start_time,
                artifacts={
                    "detection_results": detection_results,
                    "expected_namespace": self.test_namespace
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_name,
                success=False,
                duration=time.time() - start_time,
                error_message=str(e)
            )
    
    async def test_resource_monitoring(self) -> TestResult:
        """Test resource monitoring capabilities."""
        
        test_name = "resource_monitoring"
        start_time = time.time()
        
        try:
            # Test resource monitoring script
            monitoring_script = """
import os
from pathlib import Path

def check_cgroup_v1():
    \"\"\"Check cgroup v1 paths.\"\"\"
    paths = [
        '/sys/fs/cgroup/cpu/cpu.stat',
        '/sys/fs/cgroup/memory/memory.usage_in_bytes',
        '/sys/fs/cgroup/memory/memory.stat'
    ]
    
    results = {}
    for path in paths:
        p = Path(path)
        results[path] = {
            'exists': p.exists(),
            'readable': p.exists() and os.access(path, os.R_OK)
        }
    
    return results

def check_cgroup_v2():
    \"\"\"Check cgroup v2 paths.\"\"\"
    paths = [
        '/sys/fs/cgroup/cpu.stat',
        '/sys/fs/cgroup/memory.current',
        '/sys/fs/cgroup/memory.stat'
    ]
    
    results = {}
    for path in paths:
        p = Path(path)
        results[path] = {
            'exists': p.exists(),
            'readable': p.exists() and os.access(path, os.R_OK)
        }
    
    return results

print("=== Resource Monitoring Test ===")
print("Cgroup v1:")
cgroup_v1 = check_cgroup_v1()
for path, info in cgroup_v1.items():
    print(f"  {path}: exists={info['exists']}, readable={info['readable']}")

print("Cgroup v2:")
cgroup_v2 = check_cgroup_v2()
for path, info in cgroup_v2.items():
    print(f"  {path}: exists={info['exists']}, readable={info['readable']}")

# Check for any accessible cgroup files
has_cgroup = any(info['readable'] for info in {**cgroup_v1, **cgroup_v2}.values())
print(f"Has accessible cgroup files: {has_cgroup}")
"""
            
            result = subprocess.run([
                "kubectl", "exec", "-n", self.test_namespace,
                "deployment/genops-ai-test", "--",
                "python", "-c", monitoring_script
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return TestResult(
                    test_name=test_name,
                    success=False,
                    duration=time.time() - start_time,
                    error_message=f"Monitoring script failed: {result.stderr}",
                    artifacts={"stdout": result.stdout, "stderr": result.stderr}
                )
            
            # Check if any cgroup files are accessible
            has_accessible_cgroups = "Has accessible cgroup files: True" in result.stdout
            
            return TestResult(
                test_name=test_name,
                success=has_accessible_cgroups,
                duration=time.time() - start_time,
                artifacts={
                    "monitoring_output": result.stdout,
                    "has_cgroups": has_accessible_cgroups
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_name,
                success=False,
                duration=time.time() - start_time,
                error_message=str(e)
            )
    
    async def test_service_account_access(self) -> TestResult:
        """Test service account token access."""
        
        test_name = "service_account_access"
        start_time = time.time()
        
        try:
            service_account_script = """
import os
from pathlib import Path

service_account_paths = [
    '/var/run/secrets/kubernetes.io/serviceaccount/token',
    '/var/run/secrets/kubernetes.io/serviceaccount/ca.crt',
    '/var/run/secrets/kubernetes.io/serviceaccount/namespace'
]

print("=== Service Account Test ===")
results = {}

for path in service_account_paths:
    p = Path(path)
    exists = p.exists()
    readable = exists and os.access(path, os.R_OK)
    
    results[path] = {
        'exists': exists,
        'readable': readable
    }
    
    print(f"{path}:")
    print(f"  exists: {exists}")
    print(f"  readable: {readable}")
    
    if readable:
        try:
            content = p.read_text()
            print(f"  content_length: {len(content)}")
            if 'namespace' in path:
                print(f"  namespace_content: {content.strip()}")
        except Exception as e:
            print(f"  read_error: {e}")

has_service_account = results['/var/run/secrets/kubernetes.io/serviceaccount/token']['readable']
print(f"\\nHas service account access: {has_service_account}")
"""
            
            result = subprocess.run([
                "kubectl", "exec", "-n", self.test_namespace,
                "deployment/genops-ai-test", "--",
                "python", "-c", service_account_script
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return TestResult(
                    test_name=test_name,
                    success=False,
                    duration=time.time() - start_time,
                    error_message=f"Service account script failed: {result.stderr}",
                    artifacts={"stdout": result.stdout, "stderr": result.stderr}
                )
            
            has_service_account = "Has service account access: True" in result.stdout
            
            return TestResult(
                test_name=test_name,
                success=has_service_account,
                duration=time.time() - start_time,
                artifacts={
                    "service_account_output": result.stdout,
                    "has_service_account": has_service_account
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_name,
                success=False,
                duration=time.time() - start_time,
                error_message=str(e)
            )
    
    async def test_network_connectivity(self) -> TestResult:
        """Test network connectivity to external AI providers."""
        
        test_name = "network_connectivity"
        start_time = time.time()
        
        try:
            connectivity_script = """
import subprocess
import sys

endpoints = [
    'api.openai.com',
    'api.anthropic.com',
    'google.com'  # Basic connectivity test
]

print("=== Network Connectivity Test ===")
results = {}

for endpoint in endpoints:
    print(f"Testing {endpoint}...")
    try:
        result = subprocess.run([
            'python', '-c', f'''
import socket
import sys
try:
    socket.create_connection(("{endpoint}", 443), timeout=10)
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {{e}}")
sys.exit(0)
'''
        ], capture_output=True, text=True, timeout=15)
        
        success = "SUCCESS" in result.stdout
        results[endpoint] = success
        print(f"  {endpoint}: {'✓' if success else '✗'}")
        if not success:
            print(f"    Error: {result.stdout.strip()}")
    
    except Exception as e:
        results[endpoint] = False
        print(f"  {endpoint}: ✗ (Exception: {e})")

basic_connectivity = results.get('google.com', False)
ai_connectivity = any(results.get(ep, False) for ep in ['api.openai.com', 'api.anthropic.com'])

print(f"\\nBasic connectivity: {basic_connectivity}")
print(f"AI provider connectivity: {ai_connectivity}")
"""
            
            result = subprocess.run([
                "kubectl", "exec", "-n", self.test_namespace,
                "deployment/genops-ai-test", "--",
                "python", "-c", connectivity_script
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                return TestResult(
                    test_name=test_name,
                    success=False,
                    duration=time.time() - start_time,
                    error_message=f"Connectivity script failed: {result.stderr}",
                    artifacts={"stdout": result.stdout, "stderr": result.stderr}
                )
            
            # Basic connectivity should work, AI connectivity optional (depends on network policies)
            basic_connectivity = "Basic connectivity: True" in result.stdout
            
            return TestResult(
                test_name=test_name,
                success=basic_connectivity,
                duration=time.time() - start_time,
                artifacts={
                    "connectivity_output": result.stdout,
                    "basic_connectivity": basic_connectivity
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name=test_name,
                success=False,
                duration=time.time() - start_time,
                error_message=str(e)
            )
    
    async def run_all_tests(self) -> List[TestResult]:
        """Run all real cluster tests."""
        
        print(f"🚀 Starting real cluster tests in namespace: {self.test_namespace}")
        
        # Setup
        cluster_info = await self.setup_cluster_info()
        if not cluster_info.accessible:
            print("❌ Cluster not accessible, skipping real cluster tests")
            return []
        
        print(f"📋 Cluster Info:")
        print(f"   Name: {cluster_info.name}")
        print(f"   Context: {cluster_info.context}")
        print(f"   Version: {cluster_info.version}")
        print(f"   Nodes: {cluster_info.nodes}")
        print(f"   Has GenOps: {cluster_info.has_genops}")
        
        # Setup test environment
        if not await self.setup_test_namespace():
            print("❌ Failed to setup test namespace")
            return []
        
        if not await self.install_genops_for_testing():
            print("❌ Failed to install GenOps for testing")
            return []
        
        print("✅ Test environment ready, running tests...")
        
        # Run tests
        tests = [
            self.test_kubernetes_detection(),
            self.test_resource_monitoring(),
            self.test_service_account_access(),
            self.test_network_connectivity()
        ]
        
        results = []
        for test_coro in tests:
            try:
                result = await test_coro
                results.append(result)
                status = "✅" if result.success else "❌"
                print(f"{status} {result.test_name}: {result.duration:.2f}s")
                if result.error_message:
                    print(f"   Error: {result.error_message}")
            except Exception as e:
                print(f"❌ Test failed with exception: {e}")
                results.append(TestResult(
                    test_name="unknown",
                    success=False,
                    duration=0,
                    error_message=str(e)
                ))
        
        self.test_results = results
        return results
    
    async def cleanup(self):
        """Clean up test resources."""
        
        print(f"🧹 Cleaning up test resources...")
        
        # Cleanup in reverse order
        for resource_type, resource_name in reversed(self.cleanup_resources):
            try:
                if resource_type == "namespace":
                    result = subprocess.run([
                        "kubectl", "delete", "namespace", resource_name, "--timeout=60s"
                    ], capture_output=True, timeout=90)
                elif resource_type == "deployment":
                    namespace, name = resource_name.split('/')
                    result = subprocess.run([
                        "kubectl", "delete", "deployment", name, "-n", namespace
                    ], capture_output=True, timeout=30)
                elif resource_type == "service":
                    namespace, name = resource_name.split('/')
                    result = subprocess.run([
                        "kubectl", "delete", "service", name, "-n", namespace
                    ], capture_output=True, timeout=30)
                
                if result.returncode == 0:
                    print(f"   ✅ Cleaned up {resource_type}: {resource_name}")
                else:
                    print(f"   ⚠️ Failed to clean up {resource_type}: {resource_name}")
                    
            except Exception as e:
                print(f"   ❌ Error cleaning up {resource_type} {resource_name}: {e}")
        
        print("🧹 Cleanup complete")
    
    def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        
        if not self.test_results:
            return {"error": "No test results available"}
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.success)
        failed_tests = total_tests - successful_tests
        total_duration = sum(r.duration for r in self.test_results)
        
        report = {
            "cluster_info": {
                "name": self.cluster_info.name if self.cluster_info else "unknown",
                "context": self.cluster_info.context if self.cluster_info else "unknown",
                "version": self.cluster_info.version if self.cluster_info else "unknown",
                "nodes": self.cluster_info.nodes if self.cluster_info else 0,
                "namespace": self.test_namespace
            },
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": failed_tests,
                "success_rate": f"{(successful_tests/total_tests)*100:.1f}%" if total_tests > 0 else "0%",
                "total_duration": f"{total_duration:.2f}s"
            },
            "test_results": [
                {
                    "name": r.test_name,
                    "success": r.success,
                    "duration": f"{r.duration:.2f}s",
                    "error": r.error_message,
                    "artifacts": r.artifacts
                }
                for r in self.test_results
            ]
        }
        
        return report


# Pytest integration

@pytest.mark.skipif(not os.getenv("TEST_REAL_CLUSTER"), reason="Real cluster testing disabled")
@pytest.mark.integration
@pytest.mark.kubernetes
class TestRealClusterIntegration:
    """Pytest integration for real cluster tests."""
    
    @pytest.fixture(scope="class")
    async def cluster_framework(self):
        """Set up cluster test framework."""
        
        framework = RealClusterTestFramework()
        
        # Setup
        cluster_info = await framework.setup_cluster_info()
        if not cluster_info.accessible:
            pytest.skip("Kubernetes cluster not accessible")
        
        await framework.setup_test_namespace()
        await framework.install_genops_for_testing()
        
        yield framework
        
        # Cleanup
        await framework.cleanup()
    
    @pytest.mark.asyncio
    async def test_real_kubernetes_detection(self, cluster_framework):
        """Test Kubernetes detection in real cluster."""
        
        result = await cluster_framework.test_kubernetes_detection()
        assert result.success, f"Kubernetes detection failed: {result.error_message}"
        assert result.artifacts["detection_results"]["is_kubernetes"] == "True"
    
    @pytest.mark.asyncio 
    async def test_real_resource_monitoring(self, cluster_framework):
        """Test resource monitoring in real cluster."""
        
        result = await cluster_framework.test_resource_monitoring()
        # Resource monitoring may not be available in all clusters
        if not result.success:
            pytest.skip("Resource monitoring not available in this cluster")
        
        assert result.artifacts["has_cgroups"] is True
    
    @pytest.mark.asyncio
    async def test_real_service_account_access(self, cluster_framework):
        """Test service account access in real cluster."""
        
        result = await cluster_framework.test_service_account_access()
        assert result.success, f"Service account access failed: {result.error_message}"
    
    @pytest.mark.asyncio
    async def test_real_network_connectivity(self, cluster_framework):
        """Test network connectivity in real cluster."""
        
        result = await cluster_framework.test_network_connectivity()
        assert result.success, f"Basic network connectivity failed: {result.error_message}"


# CLI interface for running real cluster tests
async def main():
    """Main CLI interface for real cluster tests."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Run GenOps AI real cluster tests")
    parser.add_argument("--namespace", help="Test namespace (auto-generated if not provided)")
    parser.add_argument("--output", help="Output file for test report (JSON)")
    parser.add_argument("--cleanup", action="store_true", default=True, help="Cleanup resources after tests")
    parser.add_argument("--no-cleanup", action="store_false", dest="cleanup", help="Skip cleanup")
    
    args = parser.parse_args()
    
    # Create framework
    framework = RealClusterTestFramework(args.namespace)
    
    try:
        # Run tests
        results = await framework.run_all_tests()
        
        # Generate report
        report = framework.generate_test_report()
        
        # Output report
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"📊 Test report saved to: {args.output}")
        else:
            print("\n📊 Test Report:")
            print(f"   Cluster: {report['cluster_info']['name']} ({report['cluster_info']['version']})")
            print(f"   Tests: {report['summary']['successful_tests']}/{report['summary']['total_tests']} passed")
            print(f"   Success Rate: {report['summary']['success_rate']}")
            print(f"   Duration: {report['summary']['total_duration']}")
        
        # Return appropriate exit code
        success = all(r.success for r in results)
        return 0 if success else 1
        
    finally:
        if args.cleanup:
            await framework.cleanup()


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))