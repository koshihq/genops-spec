#!/usr/bin/env python3
"""
Together AI Production Patterns with GenOps

Demonstrates enterprise-ready patterns for production Together AI deployments
with comprehensive governance, error handling, and operational best practices.

Usage:
    python production_patterns.py

Features:
    - Enterprise governance patterns with multi-tenant support
    - Circuit breaker patterns for resilient operations
    - Advanced error handling and retry strategies
    - Performance monitoring and optimization
    - Cost optimization with budget enforcement
    - Audit trails and compliance logging
"""

import logging
import sys
import time
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Optional

try:
    from genops.core.exceptions import (
        GenOpsBudgetExceededError,
        GenOpsConfigurationError,  # noqa: F401
    )
    from genops.providers.together import GenOpsTogetherAdapter, TogetherModel
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install: pip install genops-ai[together]")
    print("Then run: python setup_validation.py")
    sys.exit(1)

# Configure logging for production patterns
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""

    failure_threshold: int = 5
    recovery_timeout: int = 60
    success_threshold: int = 3


class CircuitBreakerState:
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


class ProductionCircuitBreaker:
    """Circuit breaker for resilient Together AI operations."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.state = CircuitBreakerState.CLOSED

    def call(self, func, *args, **kwargs):
        """Execute function through circuit breaker."""
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.config.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def _on_success(self):
        """Handle successful operation."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
        else:
            self.failure_count = 0

    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN


class EnterpriseTogetherService:
    """Enterprise-grade Together AI service with production patterns."""

    def __init__(
        self,
        adapter: GenOpsTogetherAdapter,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    ):
        self.adapter = adapter
        self.circuit_breaker = ProductionCircuitBreaker(
            circuit_breaker_config or CircuitBreakerConfig()
        )
        self.operation_count = 0
        self.error_count = 0

    def chat_with_resilience(
        self,
        messages: list[dict[str, Any]],
        model: str,
        max_retries: int = 3,
        fallback_model: Optional[str] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Chat with resilience patterns: retries, fallbacks, circuit breaker."""

        def _execute_chat():
            return self.adapter.chat_with_governance(
                messages=messages, model=model, **kwargs
            )

        # Try primary model with circuit breaker
        for attempt in range(max_retries):
            try:
                result = self.circuit_breaker.call(_execute_chat)
                self.operation_count += 1
                return {
                    "result": result,
                    "model_used": result.model_used,
                    "attempt": attempt + 1,
                    "fallback_used": False,
                    "circuit_breaker_state": self.circuit_breaker.state,
                }

            except Exception as e:
                logger.warning(f"Primary model attempt {attempt + 1} failed: {e}")
                self.error_count += 1

                if attempt < max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff

        # Try fallback model if primary fails
        if fallback_model:
            try:
                logger.info(f"Attempting fallback to {fallback_model}")
                result = self.adapter.chat_with_governance(
                    messages=messages,
                    model=fallback_model,
                    fallback_operation=True,
                    original_model=model,
                    **kwargs,
                )

                self.operation_count += 1
                return {
                    "result": result,
                    "model_used": result.model_used,
                    "attempt": max_retries + 1,
                    "fallback_used": True,
                    "circuit_breaker_state": self.circuit_breaker.state,
                }

            except Exception as e:
                logger.error(f"Fallback model also failed: {e}")

        raise Exception(f"All attempts failed for model {model}")


def demonstrate_multi_tenant_governance():
    """Demonstrate multi-tenant governance patterns."""
    print("🏢 Multi-Tenant Governance Patterns")
    print("=" * 50)

    # Create adapters for different tenants/customers
    tenants = [
        {
            "name": "acme-corp",
            "tier": "enterprise",
            "daily_budget": 500.0,
            "governance_policy": "strict",
        },
        {
            "name": "startup-inc",
            "tier": "standard",
            "daily_budget": 50.0,
            "governance_policy": "enforced",
        },
        {
            "name": "freelancer",
            "tier": "basic",
            "daily_budget": 10.0,
            "governance_policy": "advisory",
        },
    ]

    tenant_adapters = {}
    tenant_results = {}

    print("🏗️ Setting up multi-tenant environment...")

    for tenant in tenants:
        print(f"   Setting up {tenant['name']} ({tenant['tier']} tier)")

        tenant_adapters[tenant["name"]] = GenOpsTogetherAdapter(
            team="multi-tenant-demo",
            project=f"tenant-{tenant['name']}",
            environment="production",
            customer_id=tenant["name"],
            cost_center=f"tenant-{tenant['tier']}",
            daily_budget_limit=tenant["daily_budget"],
            governance_policy=tenant["governance_policy"],
            enable_cost_alerts=True,
            tags={"tenant_tier": tenant["tier"], "tenant_name": tenant["name"]},
        )

    # Simulate operations for each tenant
    test_query = "Explain the benefits of AI automation for business processes."

    print("\n🎯 Processing query for all tenants:")
    print(f"   Query: {test_query[:60]}...")

    for tenant_name, adapter in tenant_adapters.items():
        tenant_info = next(t for t in tenants if t["name"] == tenant_name)

        print(f"\n👤 {tenant_name} ({tenant_info['tier']} tier):")

        try:
            with adapter.track_session(f"{tenant_name}-operations") as session:
                # Select model based on tenant tier
                if tenant_info["tier"] == "enterprise":
                    model = TogetherModel.LLAMA_3_1_70B_INSTRUCT
                    max_tokens = 300
                elif tenant_info["tier"] == "standard":
                    model = TogetherModel.LLAMA_3_1_8B_INSTRUCT
                    max_tokens = 200
                else:  # basic
                    model = TogetherModel.LLAMA_3_1_8B_INSTRUCT
                    max_tokens = 150

                result = adapter.chat_with_governance(
                    messages=[{"role": "user", "content": test_query}],
                    model=model,
                    max_tokens=max_tokens,
                    session_id=session.session_id,
                    tenant_tier=tenant_info["tier"],
                    business_unit="ai-automation",
                )

                cost_summary = adapter.get_cost_summary()

                tenant_results[tenant_name] = {
                    "cost": float(result.cost),
                    "tokens": result.tokens_used,
                    "model": result.model_used,
                    "budget_utilization": cost_summary["daily_budget_utilization"],
                    "governance_policy": cost_summary["governance_policy"],
                }

                print(f"   ✅ Model: {result.model_used}")
                print(f"   💰 Cost: ${result.cost:.6f}")
                print(
                    f"   📊 Budget used: {cost_summary['daily_budget_utilization']:.1f}%"
                )
                print(f"   🛡️ Governance: {cost_summary['governance_policy']}")

        except GenOpsBudgetExceededError as e:
            print(f"   ❌ Budget exceeded: {e}")
            tenant_results[tenant_name] = {"error": "budget_exceeded"}
        except Exception as e:
            print(f"   ❌ Operation failed: {e}")
            tenant_results[tenant_name] = {"error": str(e)}

    # Multi-tenant summary
    successful_tenants = {k: v for k, v in tenant_results.items() if "error" not in v}

    if successful_tenants:
        print("\n📊 Multi-Tenant Summary:")
        total_cost = sum(t["cost"] for t in successful_tenants.values())
        avg_utilization = sum(
            t["budget_utilization"] for t in successful_tenants.values()
        ) / len(successful_tenants)

        print(f"   Successful operations: {len(successful_tenants)}/{len(tenants)}")
        print(f"   Total cost across tenants: ${total_cost:.6f}")
        print(f"   Average budget utilization: {avg_utilization:.1f}%")
        print(
            f"   Models used: {len({t['model'] for t in successful_tenants.values()})}"
        )


def demonstrate_circuit_breaker_pattern():
    """Demonstrate circuit breaker pattern for resilient operations."""
    print("\n⚡ Circuit Breaker & Resilience Patterns")
    print("=" * 50)

    adapter = GenOpsTogetherAdapter(
        team="resilience-demo",
        project="circuit-breaker",
        environment="production",
        daily_budget_limit=20.0,
        governance_policy="advisory",
    )

    # Configure circuit breaker with tight thresholds for demo
    circuit_config = CircuitBreakerConfig(
        failure_threshold=3, recovery_timeout=30, success_threshold=2
    )

    service = EnterpriseTogetherService(adapter, circuit_config)

    print("🔧 Testing circuit breaker with simulated failures...")

    test_scenarios = [
        {
            "name": "Normal Operations",
            "model": TogetherModel.LLAMA_3_1_8B_INSTRUCT,
            "should_fail": False,
            "iterations": 3,
        },
        {
            "name": "Simulated Failures",
            "model": "invalid-model-name",  # This will fail
            "should_fail": True,
            "iterations": 4,  # Trigger circuit breaker
            "fallback_model": TogetherModel.LLAMA_3_1_8B_INSTRUCT,
        },
        {
            "name": "Recovery Testing",
            "model": TogetherModel.LLAMA_3_1_8B_INSTRUCT,
            "should_fail": False,
            "iterations": 3,
        },
    ]

    for scenario in test_scenarios:
        print(f"\n🎯 {scenario['name']}:")

        for i in range(scenario["iterations"]):
            try:
                result = service.chat_with_resilience(
                    messages=[{"role": "user", "content": f"Test message {i + 1}"}],
                    model=scenario["model"],
                    max_retries=2,
                    fallback_model=scenario.get("fallback_model"),
                    max_tokens=50,
                    scenario=scenario["name"],
                    iteration=i + 1,
                )

                print(
                    f"   ✅ Operation {i + 1}: {result['model_used']} "
                    f"(attempt {result['attempt']}, "
                    f"fallback: {result['fallback_used']}, "
                    f"circuit: {result['circuit_breaker_state']})"
                )

            except Exception as e:
                print(f"   ❌ Operation {i + 1} failed: {str(e)[:60]}...")

    print("\n📊 Circuit Breaker Stats:")
    print(f"   Total operations attempted: {service.operation_count}")
    print(f"   Total errors: {service.error_count}")
    print(
        f"   Success rate: {((service.operation_count - service.error_count) / max(service.operation_count, 1)) * 100:.1f}%"
    )
    print(f"   Final circuit state: {service.circuit_breaker.state}")


def demonstrate_cost_governance_enforcement():
    """Demonstrate strict cost governance and budget enforcement."""
    print("\n💸 Cost Governance & Budget Enforcement")
    print("=" * 50)

    # Create adapter with very strict budget for demo
    strict_adapter = GenOpsTogetherAdapter(
        team="cost-governance",
        project="budget-enforcement",
        environment="production",
        daily_budget_limit=0.01,  # Very low budget for demo
        governance_policy="strict",  # Strict enforcement
        enable_cost_alerts=True,
    )

    print(
        f"💰 Testing strict budget enforcement (${strict_adapter.daily_budget_limit} daily limit)"
    )

    # Try operations that would exceed budget
    operations = [
        {"query": "Short answer please", "max_tokens": 20},
        {"query": "Another brief response", "max_tokens": 20},
        {"query": "One more quick query", "max_tokens": 20},
        {"query": "This should trigger budget limit", "max_tokens": 50},
    ]

    successful_ops = 0
    total_cost = Decimal("0")

    for i, op in enumerate(operations, 1):
        print(f"\n🎯 Operation {i}: {op['query']}")

        try:
            result = strict_adapter.chat_with_governance(
                messages=[{"role": "user", "content": op["query"]}],
                model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,  # Cheapest model
                max_tokens=op["max_tokens"],
                temperature=0.5,
                operation_index=i,
                budget_test=True,
            )

            successful_ops += 1
            total_cost += result.cost

            cost_summary = strict_adapter.get_cost_summary()

            print(f"   ✅ Success: ${result.cost:.6f}")
            print(
                f"   📊 Budget utilization: {cost_summary['daily_budget_utilization']:.1f}%"
            )

            if cost_summary["daily_budget_utilization"] > 75:
                print("   ⚠️ Approaching budget limit!")

        except GenOpsBudgetExceededError as e:
            print(f"   ❌ Budget exceeded: {e}")
            break
        except Exception as e:
            print(f"   ❌ Operation failed: {e}")

    print("\n📊 Budget Enforcement Results:")
    print(f"   Operations completed: {successful_ops}/{len(operations)}")
    print(f"   Total cost: ${total_cost:.6f}")
    print(f"   Budget limit: ${strict_adapter.daily_budget_limit:.6f}")
    print(
        f"   Budget protection: {'✅ Effective' if successful_ops < len(operations) else '❌ Not triggered'}"
    )


@contextmanager
def production_monitoring_context(
    operation_name: str, adapter: GenOpsTogetherAdapter
) -> Generator[dict[str, Any], None, None]:
    """Production monitoring context manager."""
    monitoring_data = {
        "operation_name": operation_name,
        "start_time": time.time(),
        "operation_id": str(uuid.uuid4()),
        "errors": [],
        "metrics": {},
    }

    logger.info(
        f"Starting operation: {operation_name} ({monitoring_data['operation_id']})"
    )

    try:
        yield monitoring_data

        # Log successful completion
        duration = time.time() - monitoring_data["start_time"]
        logger.info(f"Operation completed: {operation_name} in {duration:.2f}s")

        monitoring_data["metrics"]["duration"] = duration
        monitoring_data["metrics"]["success"] = True

    except Exception as e:
        # Log errors with full context
        duration = time.time() - monitoring_data["start_time"]
        monitoring_data["errors"].append(str(e))
        monitoring_data["metrics"]["duration"] = duration
        monitoring_data["metrics"]["success"] = False

        logger.error(f"Operation failed: {operation_name} after {duration:.2f}s - {e}")
        raise

    finally:
        # Always log final metrics
        cost_summary = adapter.get_cost_summary()
        monitoring_data["metrics"]["total_cost"] = cost_summary["daily_costs"]
        monitoring_data["metrics"]["budget_utilization"] = cost_summary[
            "daily_budget_utilization"
        ]

        logger.info(f"Operation metrics: {monitoring_data['metrics']}")


def demonstrate_production_monitoring():
    """Demonstrate production monitoring and observability patterns."""
    print("\n📊 Production Monitoring & Observability")
    print("=" * 50)

    adapter = GenOpsTogetherAdapter(
        team="production-monitoring",
        project="observability-demo",
        environment="production",
        daily_budget_limit=25.0,
        tags={
            "monitoring_enabled": "true",
            "environment": "production",
            "service": "ai-assistant",
        },
    )

    monitoring_tasks = [
        {
            "name": "customer_query_processing",
            "query": "How can AI improve customer service efficiency?",
            "expected_duration": 2.0,
            "criticality": "high",
        },
        {
            "name": "content_generation",
            "query": "Generate a product description for an AI-powered chatbot platform.",
            "expected_duration": 3.0,
            "criticality": "medium",
        },
        {
            "name": "data_analysis_request",
            "query": "Analyze the trends in AI adoption across different industries.",
            "expected_duration": 4.0,
            "criticality": "low",
        },
    ]

    print("📈 Testing production monitoring patterns...")

    operation_results = []

    for task in monitoring_tasks:
        print(f"\n🎯 {task['name']} (criticality: {task['criticality']})")

        with production_monitoring_context(task["name"], adapter) as monitor:
            try:
                result = adapter.chat_with_governance(
                    messages=[{"role": "user", "content": task["query"]}],
                    model=TogetherModel.LLAMA_3_1_8B_INSTRUCT,
                    max_tokens=200,
                    temperature=0.7,
                    operation_name=task["name"],
                    criticality=task["criticality"],
                    expected_duration=task["expected_duration"],
                )

                monitor["metrics"]["tokens_used"] = result.tokens_used
                monitor["metrics"]["cost"] = float(result.cost)
                monitor["metrics"]["model_used"] = result.model_used

                operation_results.append(
                    {
                        "name": task["name"],
                        "success": True,
                        "duration": monitor["metrics"]["duration"],
                        "cost": monitor["metrics"]["cost"],
                        "criticality": task["criticality"],
                    }
                )

                print(f"   ✅ Completed in {monitor['metrics']['duration']:.2f}s")
                print(f"   💰 Cost: ${monitor['metrics']['cost']:.6f}")
                print(f"   📏 Tokens: {monitor['metrics']['tokens_used']}")

                # Performance analysis
                if monitor["metrics"]["duration"] > task["expected_duration"]:
                    print(
                        f"   ⚠️ Slower than expected ({task['expected_duration']:.1f}s)"
                    )
                else:
                    print("   ⚡ Within performance target")

            except Exception as e:
                operation_results.append(
                    {
                        "name": task["name"],
                        "success": False,
                        "error": str(e),
                        "criticality": task["criticality"],
                    }
                )
                print(f"   ❌ Failed: {e}")

    # Production monitoring summary
    print("\n📊 Production Monitoring Summary:")
    successful_ops = [op for op in operation_results if op["success"]]
    failed_ops = [op for op in operation_results if not op["success"]]

    if successful_ops:
        avg_duration = sum(op["duration"] for op in successful_ops) / len(
            successful_ops
        )
        total_cost = sum(op["cost"] for op in successful_ops)

        print(
            f"   ✅ Successful operations: {len(successful_ops)}/{len(operation_results)}"
        )
        print(f"   ⏱️ Average duration: {avg_duration:.2f}s")
        print(f"   💰 Total cost: ${total_cost:.6f}")

        # Criticality analysis
        high_crit_success = len(
            [op for op in successful_ops if op["criticality"] == "high"]
        )
        print(f"   🔥 High criticality success rate: {high_crit_success}/1")

    if failed_ops:
        print(f"   ❌ Failed operations: {len(failed_ops)}")
        for failed_op in failed_ops:
            print(f"      • {failed_op['name']} ({failed_op['criticality']})")


def main():
    """Run all production pattern demonstrations."""
    print("🏭 Together AI Production Patterns with GenOps")
    print("=" * 60)

    try:
        # Run all production pattern demonstrations
        demonstrate_multi_tenant_governance()
        demonstrate_circuit_breaker_pattern()
        demonstrate_cost_governance_enforcement()
        demonstrate_production_monitoring()

        # Final production summary
        print("\n" + "=" * 60)
        print("🎯 Production Patterns Summary")
        print("=" * 60)

        print("✅ Enterprise patterns demonstrated:")
        print("   • Multi-tenant governance with tier-based resource allocation")
        print("   • Circuit breaker patterns for resilient operations")
        print("   • Strict cost governance with budget enforcement")
        print("   • Production monitoring with observability integration")
        print("   • Error handling and retry strategies")
        print("   • Performance monitoring and SLA tracking")

        print("\n🏗️ Production Readiness Checklist:")
        print("   ✅ Multi-tenant isolation and governance")
        print("   ✅ Circuit breakers for external service calls")
        print("   ✅ Budget enforcement and cost controls")
        print("   ✅ Comprehensive error handling")
        print("   ✅ Monitoring and alerting")
        print("   ✅ Audit trails and compliance logging")
        print("   ✅ Performance optimization patterns")

        print("\n🚀 Deployment Considerations:")
        print("   • Set appropriate budget limits per tenant/environment")
        print("   • Configure circuit breaker thresholds based on SLAs")
        print("   • Implement proper logging and monitoring")
        print("   • Set up cost alerts and governance policies")
        print("   • Plan for model fallback strategies")
        print("   • Test resilience patterns under load")

        return 0

    except Exception as e:
        print(f"❌ Production patterns demo failed: {e}")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
        sys.exit(1)
