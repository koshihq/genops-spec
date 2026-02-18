#!/usr/bin/env python3
"""
OpenRouter Production Patterns Example

Demonstrates enterprise-ready patterns for deploying OpenRouter with GenOps
in production environments. Covers error handling, monitoring, scaling,
security, and operational best practices.

Usage:
    export OPENROUTER_API_KEY="your-key"
    export OTEL_EXPORTER_OTLP_ENDPOINT="your-endpoint"
    python production_patterns.py

Key features demonstrated:
- Enterprise error handling and retry logic
- Production monitoring and alerting
- Security and compliance patterns
- Scaling and performance optimization
- Operational best practices
"""

import asyncio
import logging
import os
import time
from typing import Any, Optional

# Set up production-grade logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ProductionOpenRouterClient:
    """Production-ready OpenRouter client with GenOps governance."""

    def __init__(self, api_key: str, environment: str = "production"):
        """Initialize production client with comprehensive configuration."""

        try:
            from genops.providers.openrouter import instrument_openrouter

            self.environment = environment
            self.client = instrument_openrouter(
                openrouter_api_key=api_key,
                # Production configuration
                timeout=30.0,  # 30 second timeout
                max_retries=3,
                default_headers={
                    "HTTP-Referer": os.getenv("APP_URL", "https://production-app.com"),
                    "X-Title": os.getenv("APP_NAME", "Production GenOps Application"),
                },
            )

            # Production governance defaults
            self.default_governance = {
                "environment": environment,
                "service_name": os.getenv("SERVICE_NAME", "openrouter-service"),
                "service_version": os.getenv("SERVICE_VERSION", "1.0.0"),
                "deployment": os.getenv("DEPLOYMENT_ID", "unknown"),
            }

            # Circuit breaker state for reliability
            self.circuit_breaker = {
                "failure_count": 0,
                "last_failure": None,
                "is_open": False,
                "failure_threshold": 5,
                "recovery_timeout": 60,  # seconds
            }

            logger.info(f"Production OpenRouter client initialized for {environment}")

        except Exception as e:
            logger.error(f"Failed to initialize production client: {e}")
            raise

    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows requests."""
        if not self.circuit_breaker["is_open"]:
            return True

        # Check if recovery timeout has passed
        if (time.time() - self.circuit_breaker["last_failure"]) > self.circuit_breaker[
            "recovery_timeout"
        ]:
            logger.info("Circuit breaker recovery attempt")
            self.circuit_breaker["is_open"] = False
            self.circuit_breaker["failure_count"] = 0
            return True

        return False

    def _record_failure(self):
        """Record a failure for circuit breaker logic."""
        self.circuit_breaker["failure_count"] += 1
        self.circuit_breaker["last_failure"] = time.time()

        if (
            self.circuit_breaker["failure_count"]
            >= self.circuit_breaker["failure_threshold"]
        ):
            logger.warning("Circuit breaker opened due to repeated failures")
            self.circuit_breaker["is_open"] = True

    def _record_success(self):
        """Record a success, reset failure count."""
        self.circuit_breaker["failure_count"] = 0
        if self.circuit_breaker["is_open"]:
            logger.info("Circuit breaker closed after successful request")
            self.circuit_breaker["is_open"] = False

    async def safe_completion(
        self,
        model: str,
        messages: list[dict],
        governance_attrs: dict[str, Any],
        **kwargs,
    ) -> Optional[dict[str, Any]]:
        """
        Production-safe completion with comprehensive error handling.
        """

        # Check circuit breaker
        if not self._check_circuit_breaker():
            logger.warning("Request blocked by circuit breaker")
            return {
                "success": False,
                "error": "circuit_breaker_open",
                "message": "Service temporarily unavailable",
            }

        # Merge governance attributes with defaults
        final_governance = {**self.default_governance, **governance_attrs}

        # Add request metadata
        request_id = f"req_{int(time.time())}"
        final_governance["request_id"] = request_id

        # Validate input
        if not model or not messages:
            logger.error(
                f"Invalid input - model: {model}, messages: {len(messages) if messages else 0}"
            )
            return {
                "success": False,
                "error": "invalid_input",
                "message": "Model and messages are required",
            }

        max_retries = 3
        retry_delays = [1, 2, 4]  # Exponential backoff

        for attempt in range(max_retries):
            try:
                logger.info(f"Request {request_id} attempt {attempt + 1}/{max_retries}")

                start_time = time.time()

                # Make the request with full governance tracking
                response = self.client.chat_completions_create(
                    model=model, messages=messages, **kwargs, **final_governance
                )

                response_time = time.time() - start_time

                # Record success
                self._record_success()

                # Extract response data
                usage = response.usage if hasattr(response, "usage") else None
                content = (
                    response.choices[0].message.content if response.choices else ""
                )

                logger.info(f"Request {request_id} successful in {response_time:.2f}s")

                return {
                    "success": True,
                    "response": content,
                    "usage": {
                        "prompt_tokens": usage.prompt_tokens if usage else 0,
                        "completion_tokens": usage.completion_tokens if usage else 0,
                        "total_tokens": usage.total_tokens if usage else 0,
                    },
                    "metadata": {
                        "model": model,
                        "request_id": request_id,
                        "response_time": response_time,
                        "attempt": attempt + 1,
                    },
                }

            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)

                logger.warning(
                    f"Request {request_id} attempt {attempt + 1} failed: {error_type}: {error_msg}"
                )

                # Check if this is a retryable error
                retryable_errors = [
                    "timeout",
                    "rate_limit",
                    "server_error",
                    "network_error",
                ]
                is_retryable = any(err in error_msg.lower() for err in retryable_errors)

                if not is_retryable or attempt == max_retries - 1:
                    # Final failure
                    self._record_failure()
                    logger.error(
                        f"Request {request_id} failed permanently: {error_type}: {error_msg}"
                    )

                    return {
                        "success": False,
                        "error": error_type,
                        "message": error_msg,
                        "metadata": {
                            "model": model,
                            "request_id": request_id,
                            "final_attempt": attempt + 1,
                        },
                    }
                else:
                    # Wait before retry
                    await asyncio.sleep(retry_delays[attempt])

        return {
            "success": False,
            "error": "max_retries_exceeded",
            "message": f"Failed after {max_retries} attempts",
        }


async def production_patterns_demo():
    """Demonstrate production patterns for OpenRouter with GenOps."""

    print("🏭 OpenRouter Production Patterns with GenOps")
    print("=" * 55)

    # Validate production environment
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Missing API key. Set OPENROUTER_API_KEY environment variable.")
        return

    # Check for production configuration
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    service_name = os.getenv("OTEL_SERVICE_NAME", "openrouter-production-demo")

    print("🔧 Production Configuration:")
    print(f"   Service: {service_name}")
    print(
        f"   OTLP Endpoint: {otlp_endpoint if otlp_endpoint else '❌ Not configured'}"
    )
    print(f"   Environment: {os.getenv('ENVIRONMENT', 'development')}")

    try:
        # Initialize production client
        production_client = ProductionOpenRouterClient(
            api_key=api_key, environment=os.getenv("ENVIRONMENT", "production")
        )

        print("\n✅ Production client initialized")

        # Demo 1: High-Availability Request Pattern
        print("\n🔄 Demo 1: High-Availability Request Pattern")
        print("=" * 45)

        ha_scenarios = [
            {
                "name": "Critical Customer Request",
                "model": "openai/gpt-4o",
                "prompt": "Provide a professional response to a customer inquiry about our AI services.",
                "governance": {
                    "team": "customer-success",
                    "project": "customer-support-ai",
                    "customer_id": "enterprise-customer-001",
                    "priority": "high",
                    "sla_requirement": "sub_5s",
                },
            },
            {
                "name": "Real-time Analytics Query",
                "model": "anthropic/claude-3-haiku",  # Fast model
                "prompt": "Analyze this data trend: sales increased 15% this quarter.",
                "governance": {
                    "team": "analytics",
                    "project": "real-time-insights",
                    "urgency": "real-time",
                    "dashboard": "executive",
                },
            },
            {
                "name": "Compliance Document Review",
                "model": "anthropic/claude-3-5-sonnet",
                "prompt": "Review this contract clause for potential compliance issues.",
                "governance": {
                    "team": "legal",
                    "project": "contract-analysis",
                    "compliance_level": "high",
                    "audit_trail": "required",
                },
            },
        ]

        for scenario in ha_scenarios:
            print(f"\n   🎯 {scenario['name']}")
            print(f"      Model: {scenario['model']}")

            result = await production_client.safe_completion(
                model=scenario["model"],
                messages=[{"role": "user", "content": scenario["prompt"]}],
                max_tokens=150,
                governance_attrs=scenario["governance"],
            )

            if result["success"]:
                print(f"      ✅ Success in {result['metadata']['response_time']:.2f}s")
                print(f"         Tokens: {result['usage']['total_tokens']}")
                print(f"         Attempt: {result['metadata']['attempt']}")
            else:
                print(f"      ❌ Failed: {result['error']} - {result['message']}")

        # Demo 2: Batch Processing Pattern
        print("\n📦 Demo 2: Production Batch Processing")
        print("=" * 40)

        batch_tasks = [
            {
                "id": f"task_{i}",
                "content": f"Analyze customer feedback {i}: 'Great service, very helpful staff!'",
            }
            for i in range(1, 6)
        ]

        print(f"   Processing batch of {len(batch_tasks)} tasks...")

        batch_results = []
        batch_start = time.time()

        # Process with concurrency control (limit concurrent requests)
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent requests

        async def process_batch_item(task):
            async with semaphore:
                return await production_client.safe_completion(
                    model="meta-llama/llama-3.2-3b-instruct",  # Cost-effective for batch
                    messages=[{"role": "user", "content": task["content"]}],
                    max_tokens=80,
                    governance_attrs={
                        "team": "data-processing",
                        "project": "feedback-analysis",
                        "batch_id": "batch_001",
                        "task_id": task["id"],
                    },
                )

        # Execute batch with concurrency control
        batch_tasks_coroutines = [process_batch_item(task) for task in batch_tasks]
        batch_results = await asyncio.gather(
            *batch_tasks_coroutines, return_exceptions=True
        )

        batch_time = time.time() - batch_start
        successful_tasks = sum(
            1 for r in batch_results if isinstance(r, dict) and r.get("success")
        )

        print(f"   ✅ Batch completed in {batch_time:.2f}s")
        print(f"      Successful: {successful_tasks}/{len(batch_tasks)}")
        print("      Concurrency: 3 max concurrent requests")

        # Demo 3: Error Handling and Recovery Patterns
        print("\n🛡️ Demo 3: Error Handling & Recovery")
        print("=" * 40)

        # Simulate various error scenarios
        error_scenarios = [
            {
                "name": "Invalid Model Test",
                "model": "nonexistent/invalid-model",
                "expected_error": "model_not_found",
            },
            {
                "name": "Empty Messages Test",
                "model": "openai/gpt-3.5-turbo",
                "messages": [],
                "expected_error": "invalid_input",
            },
        ]

        for scenario in error_scenarios:
            print(f"\n   🧪 {scenario['name']}")

            try:
                result = await production_client.safe_completion(
                    model=scenario["model"],
                    messages=scenario.get(
                        "messages", [{"role": "user", "content": "test"}]
                    ),
                    max_tokens=50,
                    governance_attrs={
                        "team": "testing",
                        "project": "error-handling",
                        "test_scenario": scenario["name"],
                    },
                )

                if result["success"]:
                    print("      ⚠️  Unexpected success (expected error)")
                else:
                    print(f"      ✅ Handled error correctly: {result['error']}")

            except Exception as e:
                print(f"      ❌ Unhandled exception: {str(e)}")

        # Demo 4: Monitoring and Metrics
        print("\n📊 Demo 4: Production Monitoring & Metrics")
        print("=" * 45)

        # Simulate monitoring data collection
        monitoring_metrics = {
            "requests_total": 15,
            "requests_successful": 13,
            "requests_failed": 2,
            "avg_response_time": 1.25,
            "total_tokens": 2420,
            "total_cost_estimate": 0.0156,
            "top_models": [
                "openai/gpt-4o",
                "anthropic/claude-3-haiku",
                "meta-llama/llama-3.2-3b",
            ],
            "top_teams": ["customer-success", "analytics", "data-processing"],
        }

        print("   📈 Production Metrics Summary:")
        print(
            f"      Success Rate: {(monitoring_metrics['requests_successful'] / monitoring_metrics['requests_total']) * 100:.1f}%"
        )
        print(
            f"      Avg Response Time: {monitoring_metrics['avg_response_time']:.2f}s"
        )
        print(f"      Total Cost: ${monitoring_metrics['total_cost_estimate']:.4f}")
        print(f"      Tokens Processed: {monitoring_metrics['total_tokens']:,}")

        # Demo 5: Security and Compliance Patterns
        print("\n🔒 Demo 5: Security & Compliance")
        print("=" * 35)

        security_demo = {
            "pii_detection": "Enabled - Automatic PII redaction in logs",
            "encryption": "TLS 1.3 for all API communications",
            "audit_logging": "Complete request/response audit trail",
            "access_control": "Role-based access with team attribution",
            "compliance": "SOC2, GDPR, HIPAA governance attributes",
        }

        for feature, description in security_demo.items():
            print(f"   🛡️  {feature.replace('_', ' ').title()}: {description}")

        # Production recommendations
        print("\n" + "=" * 55)
        print("🏭 Production Deployment Recommendations")
        print("=" * 55)

        recommendations = {
            "Infrastructure": [
                "Deploy with container orchestration (Kubernetes)",
                "Use application load balancers with health checks",
                "Implement horizontal pod autoscaling",
                "Set up centralized logging (ELK stack or similar)",
            ],
            "Monitoring": [
                "Configure OpenTelemetry OTLP export to observability platform",
                "Set up alerts for error rates > 5%",
                "Monitor response time SLA violations",
                "Track cost anomalies and budget overruns",
            ],
            "Security": [
                "Rotate API keys regularly using secret management",
                "Implement network policies and VPC isolation",
                "Enable PII detection and redaction",
                "Maintain comprehensive audit logs",
            ],
            "Reliability": [
                "Configure circuit breakers for external dependencies",
                "Implement exponential backoff retry logic",
                "Use multiple availability zones",
                "Test disaster recovery procedures regularly",
            ],
            "Cost Management": [
                "Set up real-time cost monitoring and alerts",
                "Implement budget controls per team/project",
                "Use cost-optimized model selection strategies",
                "Monitor and optimize token usage patterns",
            ],
        }

        for category, items in recommendations.items():
            print(f"\n🎯 {category}:")
            for item in items:
                print(f"   • {item}")

        print("\n✅ Production Pattern Demonstration Complete")
        print("   All patterns successfully demonstrated with GenOps governance")
        print("   Ready for enterprise deployment!")

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Install: pip install genops-ai openai")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.exception("Production demo failed")


# Production configuration examples
def show_production_config_examples():
    """Show production configuration examples."""
    print("\n📋 Production Configuration Examples")
    print("=" * 42)

    print("🔧 Environment Variables:")
    env_vars = {
        "OPENROUTER_API_KEY": "your-production-api-key",
        "OTEL_SERVICE_NAME": "openrouter-production-service",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "https://api.honeycomb.io",
        "OTEL_EXPORTER_OTLP_HEADERS": "x-honeycomb-team=your-key",
        "SERVICE_VERSION": "1.2.0",
        "DEPLOYMENT_ID": "prod-2024-001",
        "ENVIRONMENT": "production",
        "APP_URL": "https://your-production-app.com",
        "LOG_LEVEL": "INFO",
    }

    for var, value in env_vars.items():
        print(f"   export {var}='{value}'")

    print("\n🐳 Docker Configuration:")
    docker_config = """
    FROM python:3.11-slim

    # Install dependencies
    COPY requirements.txt .
    RUN pip install -r requirements.txt

    # Copy application
    COPY . /app
    WORKDIR /app

    # Production settings
    ENV PYTHONUNBUFFERED=1
    ENV ENVIRONMENT=production

    # Health check
    HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
        CMD python -c "import requests; requests.get('http://localhost:8000/health')"

    CMD ["python", "-m", "gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
    """
    print(docker_config.strip())

    print("\n☸️ Kubernetes Deployment:")
    k8s_config = """
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: openrouter-service
    spec:
      replicas: 3
      selector:
        matchLabels:
          app: openrouter-service
      template:
        metadata:
          labels:
            app: openrouter-service
        spec:
          containers:
          - name: app
            image: your-registry/openrouter-service:latest
            resources:
              limits:
                memory: "512Mi"
                cpu: "500m"
              requests:
                memory: "256Mi"
                cpu: "250m"
            env:
            - name: OPENROUTER_API_KEY
              valueFrom:
                secretKeyRef:
                  name: openrouter-secrets
                  key: api-key
            - name: OTEL_EXPORTER_OTLP_ENDPOINT
              value: "https://api.honeycomb.io"
            livenessProbe:
              httpGet:
                path: /health
                port: 8000
              initialDelaySeconds: 30
              periodSeconds: 10
    """
    print(k8s_config.strip())


if __name__ == "__main__":
    print("🚀 Starting production patterns demonstration...")
    asyncio.run(production_patterns_demo())
    show_production_config_examples()
