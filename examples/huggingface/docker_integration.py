#!/usr/bin/env python3
"""
Docker Integration Example for Hugging Face GenOps

This example demonstrates how to configure GenOps Hugging Face integration
in containerized environments with proper configuration management and
telemetry export patterns.

Example usage:
    # Build and run the Docker container
    docker build -t genops-hf-example .
    docker run --env-file .env genops-hf-example

Features demonstrated:
- Container-optimized configuration
- Environment variable management
- OTLP endpoint configuration for containerized telemetry
- Health check patterns for GenOps services
- Multi-stage Docker builds for production
"""

import logging
import os
import sys
import time
from typing import Any, Dict

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_container_configuration():
    """
    Setup GenOps configuration optimized for container environments.
    
    This demonstrates best practices for configuring GenOps in Docker containers
    with proper environment variable handling and telemetry endpoints.
    """

    print("🐳 Docker Container Configuration")
    print("=" * 40)
    print("Setting up GenOps for containerized deployment...")
    print()

    # Container-optimized environment variables
    container_config = {
        # OpenTelemetry Configuration
        'OTEL_SERVICE_NAME': os.getenv('OTEL_SERVICE_NAME', 'genops-huggingface-service'),
        'OTEL_SERVICE_VERSION': os.getenv('OTEL_SERVICE_VERSION', '1.0.0'),
        'OTEL_ENVIRONMENT': os.getenv('OTEL_ENVIRONMENT', 'docker'),

        # OTLP Exporter Configuration (for containerized collectors)
        'OTEL_EXPORTER_OTLP_ENDPOINT': os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', 'http://otel-collector:4317'),
        'OTEL_EXPORTER_OTLP_PROTOCOL': os.getenv('OTEL_EXPORTER_OTLP_PROTOCOL', 'grpc'),
        'OTEL_EXPORTER_OTLP_TIMEOUT': os.getenv('OTEL_EXPORTER_OTLP_TIMEOUT', '10'),

        # Hugging Face Configuration
        'HF_TOKEN': os.getenv('HF_TOKEN', ''),
        'HF_HOME': os.getenv('HF_HOME', '/app/.cache/huggingface'),

        # GenOps Configuration
        'GENOPS_LOG_LEVEL': os.getenv('GENOPS_LOG_LEVEL', 'INFO'),
        'GENOPS_SAMPLING_RATE': os.getenv('GENOPS_SAMPLING_RATE', '1.0'),
        'GENOPS_EXPORT_TIMEOUT': os.getenv('GENOPS_EXPORT_TIMEOUT', '5'),

        # Container-specific settings
        'CONTAINER_MEMORY_LIMIT': os.getenv('CONTAINER_MEMORY_LIMIT', '2Gi'),
        'CONTAINER_CPU_LIMIT': os.getenv('CONTAINER_CPU_LIMIT', '1000m'),
    }

    print("📋 Container Configuration:")
    for key, value in container_config.items():
        # Mask sensitive values
        display_value = value if not any(secret in key.lower() for secret in ['token', 'key', 'secret']) else '***'
        print(f"   {key:<30} = {display_value}")

    # Set environment variables for current process
    for key, value in container_config.items():
        if value:
            os.environ[key] = value

    return container_config


def demonstrate_containerized_workflow():
    """
    Demonstrate a typical GenOps workflow optimized for container environments.
    
    This includes health checks, resource monitoring, and graceful shutdown patterns.
    """

    print("\n🔄 Containerized Workflow Demonstration")
    print("=" * 45)

    try:
        from genops.providers.huggingface import (
            GenOpsHuggingFaceAdapter,
            create_huggingface_cost_context,
            production_workflow_context,
        )

        # Container health check
        print("🏥 Performing container health check...")

        adapter = GenOpsHuggingFaceAdapter()

        # Verify adapter is available (health check pattern)
        if not adapter.is_available():
            print("❌ GenOps Hugging Face adapter not available - container unhealthy")
            return False

        print("✅ GenOps Hugging Face adapter healthy")

        # Container-optimized workflow
        with production_workflow_context(
            workflow_name="containerized_ai_service",
            customer_id="docker_deployment_001",
            team="container_ops",
            project="containerized_ai_pipeline",
            environment="docker",
            service_name=os.getenv('OTEL_SERVICE_NAME', 'genops-hf-service'),
            container_id=os.getenv('HOSTNAME', 'unknown'),
            deployment_version=os.getenv('OTEL_SERVICE_VERSION', '1.0.0')
        ) as (workflow, workflow_id):

            print(f"🚀 Started containerized workflow: {workflow_id}")

            # Record container resource information
            workflow.record_step("container_resource_check", {
                "memory_limit": os.getenv('CONTAINER_MEMORY_LIMIT', 'unknown'),
                "cpu_limit": os.getenv('CONTAINER_CPU_LIMIT', 'unknown'),
                "hostname": os.getenv('HOSTNAME', 'unknown')
            })

            # Demonstrate typical container AI operations
            tasks = [
                {
                    "name": "content_generation",
                    "prompt": "Generate API documentation for a containerized microservice",
                    "model": "gpt-3.5-turbo",
                    "feature": "documentation_generation"
                },
                {
                    "name": "content_classification",
                    "prompt": "Classify: 'Container orchestration with Kubernetes'",
                    "model": "microsoft/DialoGPT-medium",
                    "feature": "content_classification"
                },
                {
                    "name": "embedding_generation",
                    "inputs": ["microservice architecture", "container deployment", "kubernetes orchestration"],
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "feature": "semantic_search"
                }
            ]

            for i, task in enumerate(tasks, 1):
                workflow.record_step(f"task_{i}_{task['name']}_start")

                try:
                    if task['name'] == 'embedding_generation':
                        result = adapter.feature_extraction(
                            inputs=task['inputs'],
                            model=task['model'],
                            team="container_ops",
                            project="containerized_ai_pipeline",
                            feature=task['feature'],
                            container_task=True
                        )
                        print(f"✅ Task {i}: Generated embeddings for {len(task['inputs'])} items")

                    else:
                        result = adapter.text_generation(
                            prompt=task['prompt'],
                            model=task['model'],
                            max_new_tokens=150,
                            team="container_ops",
                            project="containerized_ai_pipeline",
                            feature=task['feature'],
                            container_task=True
                        )
                        print(f"✅ Task {i}: {task['name']} completed")

                    workflow.record_step(f"task_{i}_{task['name']}_complete", {
                        "model_used": task['model'],
                        "success": True
                    })

                except Exception as e:
                    print(f"❌ Task {i} failed: {e}")
                    workflow.record_alert(f"task_{task['name']}_error", str(e), "error")
                    workflow.record_step(f"task_{i}_{task['name']}_failed", {
                        "error": str(e),
                        "success": False
                    })
                    continue

                # Container resource check between tasks
                workflow.record_performance_metric(f"task_{i}_memory_usage", 85.0, "percentage")
                workflow.record_performance_metric(f"task_{i}_cpu_usage", 45.0, "percentage")

            # Final container status
            final_summary = workflow.get_current_cost_summary()
            if final_summary:
                workflow.record_performance_metric("total_container_cost", final_summary.total_cost, "USD")
                workflow.record_performance_metric("container_efficiency_score",
                                                  max(0, 100 - (final_summary.total_cost * 100)), "score")

                print(f"💰 Container workflow cost: ${final_summary.total_cost:.4f}")
                print(f"🎯 Models used: {len(final_summary.unique_models)}")
                print(f"🔧 Providers: {list(final_summary.unique_providers)}")

            print("✅ Containerized workflow completed successfully")
            return True

    except ImportError as e:
        print(f"❌ Required components not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Containerized workflow failed: {e}")
        return False


def demonstrate_health_check_endpoint():
    """
    Demonstrate container health check endpoint implementation.
    
    This pattern is essential for Kubernetes readiness/liveness probes.
    """

    print("\n🏥 Container Health Check Implementation")
    print("=" * 45)

    def health_check() -> Dict[str, Any]:
        """Comprehensive health check for container readiness."""

        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "checks": {}
        }

        try:
            # Check 1: GenOps components availability
            try:
                from genops.providers.huggingface import GenOpsHuggingFaceAdapter
                adapter = GenOpsHuggingFaceAdapter()

                health_status["checks"]["genops_adapter"] = {
                    "status": "pass" if adapter.is_available() else "fail",
                    "message": "GenOps Hugging Face adapter available" if adapter.is_available() else "Adapter not available"
                }
            except Exception as e:
                health_status["checks"]["genops_adapter"] = {
                    "status": "fail",
                    "message": f"GenOps adapter error: {e}"
                }
                health_status["status"] = "unhealthy"

            # Check 2: Environment configuration
            required_vars = ['OTEL_SERVICE_NAME', 'OTEL_EXPORTER_OTLP_ENDPOINT']
            missing_vars = [var for var in required_vars if not os.getenv(var)]

            health_status["checks"]["environment"] = {
                "status": "pass" if not missing_vars else "warn",
                "message": "All required environment variables set" if not missing_vars else f"Missing: {missing_vars}"
            }

            # Check 3: Telemetry export readiness
            otlp_endpoint = os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', '')
            health_status["checks"]["telemetry"] = {
                "status": "pass" if otlp_endpoint else "warn",
                "message": f"OTLP endpoint configured: {otlp_endpoint}" if otlp_endpoint else "No OTLP endpoint configured"
            }

            # Check 4: Resource availability (mock)
            memory_usage = 75.0  # Mock memory usage percentage
            cpu_usage = 50.0     # Mock CPU usage percentage

            resource_status = "pass"
            if memory_usage > 90 or cpu_usage > 80:
                resource_status = "warn"
            if memory_usage > 95 or cpu_usage > 90:
                resource_status = "fail"
                health_status["status"] = "unhealthy"

            health_status["checks"]["resources"] = {
                "status": resource_status,
                "message": f"Memory: {memory_usage}%, CPU: {cpu_usage}%",
                "memory_usage": memory_usage,
                "cpu_usage": cpu_usage
            }

        except Exception as e:
            health_status["status"] = "unhealthy"
            health_status["error"] = str(e)

        return health_status

    # Perform health check
    health_result = health_check()

    print(f"🏥 Health Check Result: {health_result['status'].upper()}")
    for check_name, check_result in health_result["checks"].items():
        status_icon = "✅" if check_result["status"] == "pass" else "⚠️" if check_result["status"] == "warn" else "❌"
        print(f"   {status_icon} {check_name}: {check_result['message']}")

    return health_result["status"] == "healthy"


def print_docker_configuration_examples():
    """Print example Docker configurations for reference."""

    print("\n🐳 Docker Configuration Examples")
    print("=" * 40)

    # Example Dockerfile
    dockerfile_content = '''# Multi-stage Dockerfile for GenOps Hugging Face service
FROM python:3.11-slim AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash genops

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim AS production

# Copy user from builder stage
COPY --from=builder /etc/passwd /etc/group /etc/
COPY --from=builder --chown=genops:genops /home/genops /home/genops

# Install Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set up application
WORKDIR /app
COPY --chown=genops:genops . .

# Configure environment
ENV PYTHONPATH=/app/src
ENV HF_HOME=/app/.cache/huggingface
ENV GENOPS_LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python docker_integration.py --health-check || exit 1

# Switch to non-root user
USER genops

# Default command
CMD ["python", "docker_integration.py"]'''

    print("📄 Example Dockerfile:")
    print("```dockerfile")
    print(dockerfile_content)
    print("```")

    # Example docker-compose.yml
    docker_compose_content = '''version: '3.8'

services:
  genops-hf-service:
    build: .
    environment:
      - OTEL_SERVICE_NAME=genops-huggingface-service
      - OTEL_SERVICE_VERSION=1.0.0
      - OTEL_ENVIRONMENT=docker-compose
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
      - HF_TOKEN=${HF_TOKEN}
      - GENOPS_LOG_LEVEL=INFO
    depends_on:
      - otel-collector
    networks:
      - genops-network
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1'
        reservations:
          memory: 512M
          cpus: '0.5'

  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    command: ["--config=/etc/otel-collector-config.yaml"]
    volumes:
      - ./otel-collector-config.yaml:/etc/otel-collector-config.yaml
    ports:
      - "4317:4317"
      - "4318:4318"
    networks:
      - genops-network

networks:
  genops-network:
    driver: bridge'''

    print("\n📄 Example docker-compose.yml:")
    print("```yaml")
    print(docker_compose_content)
    print("```")

    # Example environment file
    env_file_content = '''# GenOps Hugging Face Docker Environment Configuration

# Service Configuration
OTEL_SERVICE_NAME=genops-huggingface-service
OTEL_SERVICE_VERSION=1.0.0
OTEL_ENVIRONMENT=production

# OpenTelemetry Configuration
OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
OTEL_EXPORTER_OTLP_PROTOCOL=grpc
OTEL_EXPORTER_OTLP_TIMEOUT=10

# Hugging Face Configuration
HF_TOKEN=your_hf_token_here
HF_HOME=/app/.cache/huggingface

# GenOps Configuration
GENOPS_LOG_LEVEL=INFO
GENOPS_SAMPLING_RATE=1.0
GENOPS_EXPORT_TIMEOUT=5

# Container Resource Limits
CONTAINER_MEMORY_LIMIT=2Gi
CONTAINER_CPU_LIMIT=1000m'''

    print("\n📄 Example .env file:")
    print("```bash")
    print(env_file_content)
    print("```")


def main():
    """Main demonstration function."""

    print("🐳 GenOps Hugging Face Docker Integration")
    print("=" * 50)
    print("Demonstrating containerized deployment patterns...")
    print("=" * 50)

    # Setup container configuration
    container_config = setup_container_configuration()

    # Health check demonstration
    health_ok = demonstrate_health_check_endpoint()

    if health_ok:
        # Run containerized workflow
        workflow_success = demonstrate_containerized_workflow()

        if workflow_success:
            print("\n✅ All containerized patterns demonstrated successfully!")
        else:
            print("\n❌ Some containerized patterns failed")
    else:
        print("\n❌ Container health check failed - skipping workflow demo")

    # Print configuration examples
    print_docker_configuration_examples()

    print("\n" + "=" * 50)
    print("🐳 Docker integration demonstration complete!")
    print("=" * 50)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GenOps Hugging Face Docker Integration Example")
    parser.add_argument("--health-check", action="store_true", help="Run health check only")
    args = parser.parse_args()

    if args.health_check:
        # Health check mode for Docker HEALTHCHECK
        is_healthy = demonstrate_health_check_endpoint()
        sys.exit(0 if is_healthy else 1)
    else:
        main()
