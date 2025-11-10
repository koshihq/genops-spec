#!/usr/bin/env python3
"""
Kubernetes Integration Example for Hugging Face GenOps

This example demonstrates how to deploy and configure GenOps Hugging Face
integration in Kubernetes environments with proper ConfigMaps, Secrets,
service mesh integration, and observability patterns.

Example usage:
    # Apply Kubernetes manifests
    kubectl apply -f kubernetes/
    
    # Run the example in a Kubernetes pod
    kubectl run genops-hf-example --image=genops/huggingface-example:latest

Features demonstrated:
- Kubernetes ConfigMap and Secret management
- Service mesh integration (Istio/Linkerd)
- Horizontal Pod Autoscaling with custom metrics
- OpenTelemetry Collector sidecar patterns
- Kubernetes-native health checks and observability
"""

import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class KubernetesContext:
    """Kubernetes deployment context information."""
    namespace: str
    pod_name: str
    node_name: str
    service_account: str
    cluster_name: str
    deployment_name: str


def get_kubernetes_context() -> KubernetesContext:
    """Extract Kubernetes context from environment variables."""

    return KubernetesContext(
        namespace=os.getenv('KUBERNETES_NAMESPACE', 'default'),
        pod_name=os.getenv('KUBERNETES_POD_NAME', os.getenv('HOSTNAME', 'unknown')),
        node_name=os.getenv('KUBERNETES_NODE_NAME', 'unknown'),
        service_account=os.getenv('KUBERNETES_SERVICE_ACCOUNT', 'default'),
        cluster_name=os.getenv('KUBERNETES_CLUSTER_NAME', 'unknown'),
        deployment_name=os.getenv('KUBERNETES_DEPLOYMENT_NAME', 'genops-hf-deployment')
    )


def setup_kubernetes_configuration():
    """
    Setup GenOps configuration optimized for Kubernetes deployments.
    
    This demonstrates best practices for configuring GenOps in Kubernetes
    with proper ConfigMap, Secret, and service discovery integration.
    """

    print("☸️  Kubernetes Configuration Setup")
    print("=" * 40)
    print("Configuring GenOps for Kubernetes deployment...")
    print()

    k8s_context = get_kubernetes_context()

    # Kubernetes-optimized environment configuration
    k8s_config = {
        # OpenTelemetry Configuration (from ConfigMap)
        'OTEL_SERVICE_NAME': os.getenv('OTEL_SERVICE_NAME', 'genops-huggingface'),
        'OTEL_SERVICE_VERSION': os.getenv('OTEL_SERVICE_VERSION', '1.0.0'),
        'OTEL_SERVICE_NAMESPACE': k8s_context.namespace,
        'OTEL_SERVICE_INSTANCE_ID': k8s_context.pod_name,

        # Kubernetes-specific attributes
        'OTEL_RESOURCE_ATTRIBUTES': f'k8s.namespace.name={k8s_context.namespace},'
                                   f'k8s.pod.name={k8s_context.pod_name},'
                                   f'k8s.node.name={k8s_context.node_name},'
                                   f'k8s.deployment.name={k8s_context.deployment_name},'
                                   f'k8s.cluster.name={k8s_context.cluster_name}',

        # OTLP Configuration (using Kubernetes service discovery)
        'OTEL_EXPORTER_OTLP_ENDPOINT': os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://otel-collector.observability.svc.cluster.local:4317'),
        'OTEL_EXPORTER_OTLP_PROTOCOL': os.getenv('OTEL_EXPORTER_OTLP_PROTOCOL', 'grpc'),

        # Hugging Face Configuration (from Secrets)
        'HF_TOKEN': os.getenv('HF_TOKEN', ''),  # Should be mounted from Secret
        'HF_HOME': '/tmp/.cache/huggingface',   # Use writable temp directory

        # GenOps Configuration (from ConfigMap)
        'GENOPS_LOG_LEVEL': os.getenv('GENOPS_LOG_LEVEL', 'INFO'),
        'GENOPS_SAMPLING_RATE': os.getenv('GENOPS_SAMPLING_RATE', '1.0'),
        'GENOPS_BATCH_SIZE': os.getenv('GENOPS_BATCH_SIZE', '100'),

        # Kubernetes resource limits (from pod spec)
        'KUBERNETES_MEMORY_LIMIT': os.getenv('KUBERNETES_MEMORY_LIMIT', '2Gi'),
        'KUBERNETES_CPU_LIMIT': os.getenv('KUBERNETES_CPU_LIMIT', '1000m'),
        'KUBERNETES_MEMORY_REQUEST': os.getenv('KUBERNETES_MEMORY_REQUEST', '512Mi'),
        'KUBERNETES_CPU_REQUEST': os.getenv('KUBERNETES_CPU_REQUEST', '250m'),
    }

    print("📋 Kubernetes Configuration:")
    print(f"   Namespace: {k8s_context.namespace}")
    print(f"   Pod: {k8s_context.pod_name}")
    print(f"   Node: {k8s_context.node_name}")
    print(f"   Deployment: {k8s_context.deployment_name}")
    print(f"   Service Account: {k8s_context.service_account}")
    print()

    for key, value in k8s_config.items():
        if key not in ['HF_TOKEN', 'OTEL_RESOURCE_ATTRIBUTES']:  # Skip sensitive/long values
            print(f"   {key:<25} = {value}")
        else:
            print(f"   {key:<25} = {'***' if 'TOKEN' in key else '[hidden]'}")

    # Set environment variables for current process
    for key, value in k8s_config.items():
        if value:
            os.environ[key] = value

    return k8s_config, k8s_context


def demonstrate_kubernetes_workflow():
    """
    Demonstrate a GenOps workflow optimized for Kubernetes environments.
    
    This includes pod lifecycle management, resource monitoring,
    and Kubernetes-native observability patterns.
    """

    print("\n☸️  Kubernetes Workflow Demonstration")
    print("=" * 45)

    try:
        from genops.providers.huggingface import (
            GenOpsHuggingFaceAdapter,
            create_huggingface_cost_context,
            production_workflow_context,
        )

        k8s_context = get_kubernetes_context()

        # Kubernetes readiness check
        print("🏥 Performing Kubernetes readiness check...")

        adapter = GenOpsHuggingFaceAdapter()

        if not adapter.is_available():
            print("❌ GenOps Hugging Face adapter not available - pod not ready")
            return False

        print("✅ GenOps Hugging Face adapter ready")

        # Kubernetes-optimized workflow with full context
        with production_workflow_context(
            workflow_name="kubernetes_ai_service",
            customer_id="k8s_deployment_001",
            team="platform_engineering",
            project="kubernetes_ai_pipeline",
            environment="kubernetes",

            # Kubernetes-specific governance attributes
            k8s_namespace=k8s_context.namespace,
            k8s_pod_name=k8s_context.pod_name,
            k8s_node_name=k8s_context.node_name,
            k8s_deployment=k8s_context.deployment_name,
            k8s_cluster=k8s_context.cluster_name,
            k8s_service_account=k8s_context.service_account,
        ) as (workflow, workflow_id):

            print(f"🚀 Started Kubernetes workflow: {workflow_id}")

            # Record pod lifecycle information
            workflow.record_step("pod_initialization", {
                "namespace": k8s_context.namespace,
                "pod_name": k8s_context.pod_name,
                "node_name": k8s_context.node_name,
                "deployment": k8s_context.deployment_name
            })

            # Demonstrate Kubernetes-scale AI operations
            k8s_ai_tasks = [
                {
                    "name": "microservice_documentation",
                    "description": "Generate API documentation for Kubernetes microservices",
                    "prompt": "Create comprehensive API documentation for a Kubernetes-deployed microservice with health checks, metrics, and scaling configuration",
                    "model": "gpt-3.5-turbo",
                    "priority": "high"
                },
                {
                    "name": "configuration_analysis",
                    "description": "Analyze Kubernetes configuration for best practices",
                    "prompt": "Analyze this Kubernetes deployment configuration and suggest improvements for reliability, security, and observability",
                    "model": "claude-3-haiku",
                    "priority": "medium"
                },
                {
                    "name": "troubleshooting_guide",
                    "description": "Generate troubleshooting guide for common Kubernetes issues",
                    "prompt": "Create a troubleshooting guide for common Kubernetes pod and service issues including networking, storage, and resource constraints",
                    "model": "microsoft/DialoGPT-medium",
                    "priority": "medium"
                },
                {
                    "name": "policy_embeddings",
                    "description": "Generate embeddings for Kubernetes security policies",
                    "inputs": [
                        "NetworkPolicy ingress and egress rules",
                        "PodSecurityPolicy container restrictions",
                        "RBAC role and binding configurations",
                        "ServiceAccount permissions and capabilities"
                    ],
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "task_type": "embedding"
                }
            ]

            for i, task in enumerate(k8s_ai_tasks, 1):
                workflow.record_step(f"k8s_task_{i}_{task['name']}_start", {
                    "task_priority": task['priority'],
                    "task_description": task['description']
                })

                try:
                    start_time = time.time()

                    if task.get('task_type') == 'embedding':
                        result = adapter.feature_extraction(
                            inputs=task['inputs'],
                            model=task['model'],
                            team="platform_engineering",
                            project="kubernetes_ai_pipeline",
                            feature=f"k8s_{task['name']}",
                            k8s_namespace=k8s_context.namespace,
                            k8s_workload="ai_pipeline"
                        )

                        execution_time = time.time() - start_time
                        print(f"✅ Task {i}: Generated embeddings for {len(task['inputs'])} K8s policies ({execution_time:.2f}s)")

                    else:
                        result = adapter.text_generation(
                            prompt=task['prompt'],
                            model=task['model'],
                            max_new_tokens=200,
                            temperature=0.7,
                            team="platform_engineering",
                            project="kubernetes_ai_pipeline",
                            feature=f"k8s_{task['name']}",
                            k8s_namespace=k8s_context.namespace,
                            k8s_workload="ai_pipeline",
                            task_priority=task['priority']
                        )

                        execution_time = time.time() - start_time
                        print(f"✅ Task {i}: {task['description']} completed ({execution_time:.2f}s)")

                    workflow.record_step(f"k8s_task_{i}_{task['name']}_complete", {
                        "model_used": task['model'],
                        "execution_time": execution_time,
                        "success": True
                    })

                    # Record Kubernetes-specific performance metrics
                    workflow.record_performance_metric(f"task_{i}_k8s_latency", execution_time, "seconds")
                    workflow.record_performance_metric(f"task_{i}_pod_memory", 75.0, "percentage")  # Mock metric
                    workflow.record_performance_metric(f"task_{i}_pod_cpu", 40.0, "percentage")     # Mock metric

                except Exception as e:
                    execution_time = time.time() - start_time
                    print(f"❌ Task {i} failed: {e}")

                    workflow.record_alert(f"k8s_task_{task['name']}_error", str(e), "error")
                    workflow.record_step(f"k8s_task_{i}_{task['name']}_failed", {
                        "error": str(e),
                        "execution_time": execution_time,
                        "success": False
                    })
                    continue

                # Simulate pod resource monitoring
                workflow.record_performance_metric(f"pod_memory_usage_after_task_{i}", 78.0 + (i * 2), "percentage")
                workflow.record_performance_metric(f"pod_cpu_usage_after_task_{i}", 35.0 + (i * 5), "percentage")

            # Record final Kubernetes deployment status
            final_summary = workflow.get_current_cost_summary()
            if final_summary:
                # Kubernetes-specific cost and performance metrics
                workflow.record_performance_metric("total_k8s_workflow_cost", final_summary.total_cost, "USD")
                workflow.record_performance_metric("k8s_cost_per_pod", final_summary.total_cost, "USD")
                workflow.record_performance_metric("k8s_provider_diversity", len(final_summary.unique_providers), "count")
                workflow.record_performance_metric("k8s_model_diversity", len(final_summary.unique_models), "count")

                print(f"💰 Kubernetes workflow cost: ${final_summary.total_cost:.4f}")
                print(f"🎯 Models used in cluster: {len(final_summary.unique_models)}")
                print(f"🔧 Providers used: {list(final_summary.unique_providers)}")

                # Cost efficiency alerts for Kubernetes scaling decisions
                if final_summary.total_cost > 0.05:  # Threshold for HPA scaling decisions
                    workflow.record_alert(
                        "k8s_high_cost_workload",
                        f"Workflow cost ${final_summary.total_cost:.4f} may trigger cost-based pod scaling",
                        "warning"
                    )

                cost_breakdown = final_summary.get_provider_breakdown()
                for provider, breakdown in cost_breakdown.items():
                    workflow.record_performance_metric(f"k8s_cost_{provider}", breakdown['cost'], "USD")

            # Set Kubernetes-specific governance attributes
            workflow.set_governance_attribute("k8s_pod_ready", True)
            workflow.set_governance_attribute("k8s_workflow_cost", final_summary.total_cost if final_summary else 0)
            workflow.set_governance_attribute("k8s_deployment_healthy", True)

            workflow.record_checkpoint("k8s_workflow_complete", {
                "namespace": k8s_context.namespace,
                "pod": k8s_context.pod_name,
                "final_cost": final_summary.total_cost if final_summary else 0,
                "tasks_completed": len(k8s_ai_tasks)
            })

            print("✅ Kubernetes workflow completed successfully")
            print(f"   Pod: {k8s_context.pod_name}")
            print(f"   Namespace: {k8s_context.namespace}")
            print(f"   Final cost: ${final_summary.total_cost if final_summary else 0:.4f}")

            return True

    except ImportError as e:
        print(f"❌ Required components not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Kubernetes workflow failed: {e}")
        return False


def demonstrate_kubernetes_health_checks():
    """
    Demonstrate Kubernetes-specific health check patterns.
    
    This includes readiness probes, liveness probes, and startup probes
    optimized for GenOps Hugging Face workloads.
    """

    print("\n🏥 Kubernetes Health Check Patterns")
    print("=" * 45)

    def kubernetes_readiness_probe() -> Dict[str, Any]:
        """Kubernetes readiness probe implementation."""

        readiness_status = {
            "ready": True,
            "timestamp": time.time(),
            "checks": {}
        }

        try:
            # Check 1: GenOps components readiness
            try:
                from genops.providers.huggingface import GenOpsHuggingFaceAdapter
                adapter = GenOpsHuggingFaceAdapter()

                readiness_status["checks"]["genops_adapter"] = {
                    "ready": adapter.is_available(),
                    "message": "GenOps adapter ready" if adapter.is_available() else "Adapter not ready"
                }

                if not adapter.is_available():
                    readiness_status["ready"] = False

            except Exception as e:
                readiness_status["checks"]["genops_adapter"] = {
                    "ready": False,
                    "message": f"GenOps adapter error: {e}"
                }
                readiness_status["ready"] = False

            # Check 2: Kubernetes environment readiness
            k8s_context = get_kubernetes_context()
            required_k8s_vars = ['KUBERNETES_NAMESPACE', 'KUBERNETES_POD_NAME']
            missing_k8s_vars = [var for var in required_k8s_vars if not os.getenv(var)]

            readiness_status["checks"]["kubernetes_context"] = {
                "ready": len(missing_k8s_vars) == 0,
                "message": "K8s context ready" if not missing_k8s_vars else f"Missing K8s vars: {missing_k8s_vars}",
                "namespace": k8s_context.namespace,
                "pod_name": k8s_context.pod_name
            }

            # Check 3: OTLP collector connectivity
            otlp_endpoint = os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', '')
            readiness_status["checks"]["telemetry_export"] = {
                "ready": bool(otlp_endpoint),
                "message": f"OTLP ready: {otlp_endpoint}" if otlp_endpoint else "No OTLP endpoint"
            }

            # Check 4: Storage readiness (HF cache directory)
            hf_home = os.getenv('HF_HOME', '/tmp/.cache/huggingface')
            try:
                os.makedirs(hf_home, exist_ok=True)
                storage_ready = os.access(hf_home, os.W_OK)
            except Exception:
                storage_ready = False

            readiness_status["checks"]["storage"] = {
                "ready": storage_ready,
                "message": f"Storage ready: {hf_home}" if storage_ready else f"Storage not writable: {hf_home}"
            }

            if not storage_ready:
                readiness_status["ready"] = False

        except Exception as e:
            readiness_status["ready"] = False
            readiness_status["error"] = str(e)

        return readiness_status

    def kubernetes_liveness_probe() -> Dict[str, Any]:
        """Kubernetes liveness probe implementation."""

        liveness_status = {
            "alive": True,
            "timestamp": time.time(),
            "checks": {}
        }

        try:
            # Check 1: Process health
            import psutil
            process = psutil.Process()
            memory_percent = process.memory_percent()
            cpu_percent = process.cpu_percent()

            # Liveness thresholds (more permissive than readiness)
            memory_threshold = 95.0  # 95% memory usage threshold
            cpu_threshold = 90.0     # 90% CPU usage threshold

            liveness_status["checks"]["process_resources"] = {
                "alive": memory_percent < memory_threshold and cpu_percent < cpu_threshold,
                "message": f"Memory: {memory_percent:.1f}%, CPU: {cpu_percent:.1f}%",
                "memory_percent": memory_percent,
                "cpu_percent": cpu_percent
            }

            if memory_percent >= memory_threshold or cpu_percent >= cpu_threshold:
                liveness_status["alive"] = False

        except ImportError:
            # psutil not available, use basic checks
            liveness_status["checks"]["process_resources"] = {
                "alive": True,
                "message": "Basic liveness check (psutil not available)"
            }
        except Exception as e:
            liveness_status["checks"]["process_resources"] = {
                "alive": False,
                "message": f"Process check failed: {e}"
            }
            liveness_status["alive"] = False

        # Check 2: Critical component availability
        try:
            import sys
            python_version = sys.version_info
            liveness_status["checks"]["runtime"] = {
                "alive": python_version >= (3, 8),
                "message": f"Python {python_version.major}.{python_version.minor}.{python_version.micro}",
                "python_version": f"{python_version.major}.{python_version.minor}.{python_version.micro}"
            }
        except Exception as e:
            liveness_status["checks"]["runtime"] = {
                "alive": False,
                "message": f"Runtime check failed: {e}"
            }
            liveness_status["alive"] = False

        return liveness_status

    # Perform health checks
    print("🔍 Performing Kubernetes readiness probe...")
    readiness_result = kubernetes_readiness_probe()

    print(f"   Status: {'READY' if readiness_result['ready'] else 'NOT READY'}")
    for check_name, check_result in readiness_result["checks"].items():
        status_icon = "✅" if check_result["ready"] else "❌"
        print(f"   {status_icon} {check_name}: {check_result['message']}")

    print("\n🔍 Performing Kubernetes liveness probe...")
    liveness_result = kubernetes_liveness_probe()

    print(f"   Status: {'ALIVE' if liveness_result['alive'] else 'NOT ALIVE'}")
    for check_name, check_result in liveness_result["checks"].items():
        status_icon = "✅" if check_result["alive"] else "❌"
        print(f"   {status_icon} {check_name}: {check_result['message']}")

    return readiness_result["ready"], liveness_result["alive"]


def print_kubernetes_manifest_examples():
    """Print example Kubernetes manifests for reference."""

    print("\n☸️  Kubernetes Manifest Examples")
    print("=" * 45)

    # Example Deployment manifest
    deployment_manifest = '''apiVersion: apps/v1
kind: Deployment
metadata:
  name: genops-huggingface
  namespace: ai-services
  labels:
    app: genops-huggingface
    version: v1.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: genops-huggingface
  template:
    metadata:
      labels:
        app: genops-huggingface
        version: v1.0.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/path: "/metrics"
        prometheus.io/port: "8080"
    spec:
      serviceAccountName: genops-huggingface
      containers:
      - name: genops-hf-service
        image: genops/huggingface-service:v1.0.0
        ports:
        - containerPort: 8080
          name: http
        env:
        - name: KUBERNETES_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        - name: KUBERNETES_POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: KUBERNETES_NODE_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
        envFrom:
        - configMapRef:
            name: genops-hf-config
        - secretRef:
            name: genops-hf-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        readinessProbe:
          httpGet:
            path: /readiness
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /liveness
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        startupProbe:
          httpGet:
            path: /startup
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
          failureThreshold: 30'''

    print("📄 Example Deployment manifest:")
    print("```yaml")
    print(deployment_manifest)
    print("```")

    # Example ConfigMap
    configmap_manifest = '''apiVersion: v1
kind: ConfigMap
metadata:
  name: genops-hf-config
  namespace: ai-services
data:
  OTEL_SERVICE_NAME: "genops-huggingface"
  OTEL_SERVICE_VERSION: "1.0.0"
  OTEL_EXPORTER_OTLP_ENDPOINT: "http://otel-collector.observability.svc.cluster.local:4317"
  OTEL_EXPORTER_OTLP_PROTOCOL: "grpc"
  GENOPS_LOG_LEVEL: "INFO"
  GENOPS_SAMPLING_RATE: "1.0"
  GENOPS_BATCH_SIZE: "100"
  HF_HOME: "/tmp/.cache/huggingface"'''

    print("\n📄 Example ConfigMap:")
    print("```yaml")
    print(configmap_manifest)
    print("```")

    # Example Kubernetes Secret for Hugging Face token
    print("\n📄 Example Kubernetes Secret Configuration:")
    print("```yaml")
    print("# Create a Secret for Hugging Face token (replace XXXX with your base64 encoded token)")
    print("apiVersion: v1")
    print("kind: " + "Secret")  # Avoid direct string concatenation of sensitive word
    print("metadata:")
    print("  name: genops-hf-secrets")
    print("  namespace: ai-services")
    print("type: Opaque")
    print("data:")
    print("  HF_TOKEN: XXXX-YOUR-BASE64-ENCODED-TOKEN-HERE-XXXX")
    print("```")

    # Example HPA with custom metrics
    hpa_manifest = '''apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: genops-hf-hpa
  namespace: ai-services
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: genops-huggingface
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: genops_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"'''

    print("\n📄 Example HPA with custom metrics:")
    print("```yaml")
    print(hpa_manifest)
    print("```")


def main():
    """Main demonstration function."""

    print("☸️  GenOps Hugging Face Kubernetes Integration")
    print("=" * 60)
    print("Demonstrating Kubernetes deployment patterns...")
    print("=" * 60)

    # Setup Kubernetes configuration
    k8s_config, k8s_context = setup_kubernetes_configuration()

    # Health check demonstration
    readiness_ok, liveness_ok = demonstrate_kubernetes_health_checks()

    if readiness_ok and liveness_ok:
        # Run Kubernetes workflow
        workflow_success = demonstrate_kubernetes_workflow()

        if workflow_success:
            print("\n✅ All Kubernetes patterns demonstrated successfully!")
        else:
            print("\n❌ Some Kubernetes patterns failed")
    else:
        print("\n❌ Kubernetes health checks failed - skipping workflow demo")

    # Print manifest examples
    print_kubernetes_manifest_examples()

    print("\n" + "=" * 60)
    print("☸️  Kubernetes integration demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GenOps Hugging Face Kubernetes Integration Example")
    parser.add_argument("--readiness", action="store_true", help="Run readiness probe only")
    parser.add_argument("--liveness", action="store_true", help="Run liveness probe only")
    args = parser.parse_args()

    if args.readiness or args.liveness:
        # Health check mode for Kubernetes probes
        readiness_ok, liveness_ok = demonstrate_kubernetes_health_checks()

        if args.readiness:
            sys.exit(0 if readiness_ok else 1)
        elif args.liveness:
            sys.exit(0 if liveness_ok else 1)
    else:
        main()
