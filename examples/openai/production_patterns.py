#!/usr/bin/env python3
"""
OpenAI Production Patterns Example

This example demonstrates enterprise-ready patterns for deploying OpenAI applications
with GenOps governance telemetry in production environments.

What you'll learn:
- Context manager patterns for complex workflows
- Policy enforcement and governance automation
- Error handling and resilience patterns
- Performance optimization and scaling
- Enterprise monitoring and alerting

Usage:
    python production_patterns.py

Prerequisites:
    pip install genops-ai[openai]
    export OPENAI_API_KEY="your_api_key_here"
"""

import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class WorkflowResult:
    """Result from a production workflow with full telemetry."""
    workflow_id: str
    success: bool
    total_cost: float
    operations_count: int
    duration: float
    error: Optional[str] = None
    metadata: dict[str, Any] = None

@contextmanager
def production_workflow_context(workflow_name: str, customer_id: str, **kwargs):
    """Production-ready context manager for complex AI workflows."""
    from genops import track

    workflow_id = f"{workflow_name}_{customer_id}_{int(time.time())}"
    start_time = time.time()

    print(f"🚀 Starting workflow: {workflow_name}")
    print(f"   Workflow ID: {workflow_id}")
    print(f"   Customer: {customer_id}")

    with track(workflow_name,
               workflow_id=workflow_id,
               customer_id=customer_id,
               **kwargs) as span:
        try:
            yield span, workflow_id

            duration = time.time() - start_time
            span.set_attribute("workflow_success", True)
            span.set_attribute("workflow_duration", duration)

            print(f"✅ Workflow completed: {workflow_name}")
            print(f"   Duration: {duration:.2f} seconds")

        except Exception as e:
            duration = time.time() - start_time
            span.set_attribute("workflow_success", False)
            span.set_attribute("workflow_error", str(e))
            span.set_attribute("workflow_duration", duration)

            print(f"❌ Workflow failed: {workflow_name}")
            print(f"   Error: {e}")
            print(f"   Duration: {duration:.2f} seconds")
            raise

def customer_support_workflow():
    """Enterprise customer support workflow with full governance."""
    print("🎧 Enterprise Customer Support Workflow")
    print("-" * 50)

    try:
        from genops.providers.openai import instrument_openai

        client = instrument_openai()

        # Simulate customer support ticket
        customer_ticket = {
            "ticket_id": "SUP-2024-001",
            "customer_id": "enterprise-customer-001",
            "priority": "high",
            "category": "billing",
            "description": "I was charged twice for my subscription this month and need a refund processed urgently.",
            "customer_tier": "enterprise"
        }

        with production_workflow_context(
            "customer_support_resolution",
            customer_ticket["customer_id"],
            team="customer-support",
            project="automated-support",
            environment="production",
            ticket_id=customer_ticket["ticket_id"],
            priority=customer_ticket["priority"]
        ) as (span, workflow_id):

            total_cost = 0
            operations = []

            # Step 1: Ticket classification and routing
            print("📋 Step 1: Ticket Classification")
            classification_response = client.chat_completions_create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Classify support tickets by category, urgency, and required department."},
                    {"role": "user", "content": f"Classify this ticket: {customer_ticket['description']}"}
                ],
                max_tokens=200,

                # Step-specific governance
                team="customer-support",
                project="automated-support",
                customer_id=customer_ticket["customer_id"],
                workflow_id=workflow_id,
                step="classification",
                ticket_id=customer_ticket["ticket_id"]
            )

            classification = classification_response.choices[0].message.content
            classification_cost = (classification_response.usage.prompt_tokens / 1000) * 0.0015 + (classification_response.usage.completion_tokens / 1000) * 0.002
            total_cost += classification_cost
            operations.append(("Classification", classification_cost))

            print(f"   Result: {classification[:100]}...")
            print(f"   Cost: ${classification_cost:.4f}")

            # Step 2: Generate initial response
            print("\n💬 Step 2: Response Generation")
            response_generation = client.chat_completions_create(
                model="gpt-4",  # Higher quality for customer-facing content
                messages=[
                    {"role": "system", "content": f"You are a professional customer support representative. Generate a helpful response for this {customer_ticket['priority']} priority {customer_ticket['category']} issue."},
                    {"role": "user", "content": customer_ticket["description"]}
                ],
                max_tokens=400,
                temperature=0.3,  # Lower temperature for professional tone

                # Enhanced governance for customer-facing content
                team="customer-support",
                project="automated-support",
                customer_id=customer_ticket["customer_id"],
                workflow_id=workflow_id,
                step="response_generation",
                ticket_id=customer_ticket["ticket_id"],
                customer_tier=customer_ticket["customer_tier"],
                content_type="customer_facing"
            )

            response_content = response_generation.choices[0].message.content
            response_cost = (response_generation.usage.prompt_tokens / 1000) * 0.03 + (response_generation.usage.completion_tokens / 1000) * 0.06
            total_cost += response_cost
            operations.append(("Response Generation", response_cost))

            print(f"   Response: {response_content[:150]}...")
            print(f"   Cost: ${response_cost:.4f}")

            # Step 3: Quality assurance check
            print("\n🔍 Step 3: Quality Assurance")
            qa_check = client.chat_completions_create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Review customer support responses for tone, accuracy, and completeness. Rate from 1-10 and suggest improvements."},
                    {"role": "user", "content": f"Review this response to a {customer_ticket['category']} issue: {response_content}"}
                ],
                max_tokens=200,

                # QA governance tracking
                team="customer-support",
                project="automated-support",
                customer_id=customer_ticket["customer_id"],
                workflow_id=workflow_id,
                step="quality_assurance",
                ticket_id=customer_ticket["ticket_id"],
                qa_check=True
            )

            qa_result = qa_check.choices[0].message.content
            qa_cost = (qa_check.usage.prompt_tokens / 1000) * 0.0015 + (qa_check.usage.completion_tokens / 1000) * 0.002
            total_cost += qa_cost
            operations.append(("Quality Assurance", qa_cost))

            print(f"   QA Result: {qa_result[:100]}...")
            print(f"   Cost: ${qa_cost:.4f}")

            # Set workflow-level metrics
            span.set_attribute("total_operations", len(operations))
            span.set_attribute("total_cost", total_cost)
            span.set_attribute("ticket_category", customer_ticket["category"])
            span.set_attribute("customer_tier", customer_ticket["customer_tier"])

            print("\n📊 Workflow Summary:")
            print(f"   • Total operations: {len(operations)}")
            print(f"   • Total cost: ${total_cost:.4f}")
            print(f"   • Average cost per operation: ${total_cost / len(operations):.4f}")

            for operation, cost in operations:
                print(f"   • {operation}: ${cost:.4f}")

        return True

    except Exception as e:
        print(f"❌ Customer support workflow error: {e}")
        return False

def content_pipeline_with_policy_enforcement():
    """Content generation pipeline with policy enforcement."""
    print("\n\n📝 Content Pipeline with Policy Enforcement")
    print("-" * 50)

    try:
        from genops.providers.openai import instrument_openai

        client = instrument_openai()

        # Content generation requests
        content_requests = [
            {
                "type": "blog_post",
                "topic": "Benefits of renewable energy for businesses",
                "target_audience": "business_executives",
                "word_count": 800
            },
            {
                "type": "social_media",
                "topic": "New product launch announcement",
                "target_audience": "general_public",
                "word_count": 100
            },
            {
                "type": "technical_documentation",
                "topic": "API integration best practices",
                "target_audience": "developers",
                "word_count": 1200
            }
        ]

        with production_workflow_context(
            "content_generation_pipeline",
            "content-team-001",
            team="content-marketing",
            project="automated-content",
            environment="production"
        ) as (span, workflow_id):

            total_content_cost = 0
            generated_content = []

            for i, request in enumerate(content_requests, 1):
                print(f"\n🎯 Content Request {i}: {request['type']} - {request['topic'][:40]}...")

                # Policy enforcement check
                policy_check = enforce_content_policy(client, request, workflow_id)
                if not policy_check["approved"]:
                    print(f"   ❌ Policy violation: {policy_check['reason']}")
                    continue

                # Content generation
                content_result = generate_content_with_governance(
                    client, request, workflow_id, i
                )

                if content_result:
                    generated_content.append(content_result)
                    total_content_cost += content_result["cost"]

                    print(f"   ✅ Generated: {len(content_result['content'])} chars")
                    print(f"   💰 Cost: ${content_result['cost']:.4f}")

            # Pipeline summary
            span.set_attribute("content_requests", len(content_requests))
            span.set_attribute("content_generated", len(generated_content))
            span.set_attribute("pipeline_cost", total_content_cost)

            print("\n📊 Content Pipeline Results:")
            print(f"   • Requests processed: {len(content_requests)}")
            print(f"   • Content pieces generated: {len(generated_content)}")
            print(f"   • Total pipeline cost: ${total_content_cost:.4f}")
            print(f"   • Average cost per piece: ${total_content_cost / max(len(generated_content), 1):.4f}")

        return True

    except Exception as e:
        print(f"❌ Content pipeline error: {e}")
        return False

def enforce_content_policy(client, request: dict, workflow_id: str) -> dict:
    """Enforce content policy using AI-powered policy checking."""
    # Simple policy enforcement example
    restricted_topics = ["controversial", "political", "medical advice"]

    topic_lower = request["topic"].lower()
    for restricted in restricted_topics:
        if restricted in topic_lower:
            return {
                "approved": False,
                "reason": f"Topic contains restricted content: {restricted}"
            }

    # In production, you might use a dedicated policy model here
    return {"approved": True, "reason": "Content approved"}

def generate_content_with_governance(client, request: dict, workflow_id: str, request_index: int) -> Optional[dict]:
    """Generate content with full governance tracking."""
    try:
        # Select model based on content complexity
        model_selection = {
            "social_media": "gpt-3.5-turbo",     # Simple, fast
            "blog_post": "gpt-4",                # Higher quality
            "technical_documentation": "gpt-4"   # Complex, accurate
        }

        model = model_selection.get(request["type"], "gpt-3.5-turbo")

        response = client.chat_completions_create(
            model=model,
            messages=[
                {"role": "system", "content": f"You are a professional {request['type']} writer. Create high-quality content for {request['target_audience']}."},
                {"role": "user", "content": f"Write a {request['word_count']}-word {request['type']} about: {request['topic']}"}
            ],
            max_tokens=min(request["word_count"] * 2, 2000),  # Rough token estimate
            temperature=0.7,

            # Detailed content governance
            team="content-marketing",
            project="automated-content",
            workflow_id=workflow_id,
            content_type=request["type"],
            target_audience=request["target_audience"],
            word_count=request["word_count"],
            request_index=request_index,
            model_selection_reason="complexity_based"
        )

        content = response.choices[0].message.content

        # Calculate cost based on actual model used
        if model == "gpt-4":
            cost = (response.usage.prompt_tokens / 1000) * 0.03 + (response.usage.completion_tokens / 1000) * 0.06
        else:
            cost = (response.usage.prompt_tokens / 1000) * 0.0015 + (response.usage.completion_tokens / 1000) * 0.002

        return {
            "content": content,
            "cost": cost,
            "tokens": response.usage.total_tokens,
            "model": model,
            "type": request["type"]
        }

    except Exception as e:
        print(f"   ❌ Content generation failed: {e}")
        return None

def resilience_and_error_handling():
    """Demonstrate production-grade error handling and resilience patterns."""
    print("\n\n🛡️  Resilience and Error Handling Patterns")
    print("-" * 50)

    try:
        from genops.providers.openai import instrument_openai

        client = instrument_openai()

        # Test scenarios including potential failure cases
        test_scenarios = [
            {
                "name": "Normal Operation",
                "model": "gpt-3.5-turbo",
                "prompt": "What is machine learning?",
                "should_fail": False
            },
            {
                "name": "Rate Limit Simulation",
                "model": "gpt-3.5-turbo",
                "prompt": "Explain artificial intelligence",
                "should_fail": False  # We'll simulate this
            },
            {
                "name": "Invalid Model Test",
                "model": "nonexistent-model",
                "prompt": "This should fail",
                "should_fail": True
            }
        ]

        with production_workflow_context(
            "resilience_testing",
            "resilience-demo",
            team="reliability-team",
            project="error-handling"
        ) as (span, workflow_id):

            results = []

            for scenario in test_scenarios:
                print(f"\n🧪 Testing: {scenario['name']}")

                try:
                    # Implement retry logic with exponential backoff
                    max_retries = 3
                    retry_delay = 1

                    for attempt in range(max_retries):
                        try:
                            start_time = time.time()

                            response = client.chat_completions_create(
                                model=scenario["model"],
                                messages=[{"role": "user", "content": scenario["prompt"]}],
                                max_tokens=100,

                                # Error handling governance
                                team="reliability-team",
                                project="error-handling",
                                workflow_id=workflow_id,
                                test_scenario=scenario["name"],
                                attempt_number=attempt + 1,
                                max_retries=max_retries
                            )

                            duration = time.time() - start_time

                            results.append({
                                "scenario": scenario["name"],
                                "success": True,
                                "attempt": attempt + 1,
                                "duration": duration,
                                "tokens": response.usage.total_tokens
                            })

                            print(f"   ✅ Success on attempt {attempt + 1}")
                            print(f"   📊 Duration: {duration:.2f}s, Tokens: {response.usage.total_tokens}")
                            break

                        except Exception as e:
                            if attempt < max_retries - 1:
                                print(f"   ⚠️  Attempt {attempt + 1} failed: {e}")
                                print(f"   🔄 Retrying in {retry_delay}s...")
                                time.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                            else:
                                # Final failure
                                results.append({
                                    "scenario": scenario["name"],
                                    "success": False,
                                    "error": str(e),
                                    "attempts": max_retries
                                })
                                print(f"   ❌ Failed after {max_retries} attempts: {e}")

                except Exception as e:
                    results.append({
                        "scenario": scenario["name"],
                        "success": False,
                        "error": str(e),
                        "attempts": 1
                    })
                    print(f"   ❌ Immediate failure: {e}")

            # Analyze results
            successful_tests = sum(1 for r in results if r["success"])
            total_tests = len(results)

            span.set_attribute("total_tests", total_tests)
            span.set_attribute("successful_tests", successful_tests)
            span.set_attribute("success_rate", successful_tests / total_tests if total_tests > 0 else 0)

            print("\n📊 Resilience Test Results:")
            print(f"   • Total tests: {total_tests}")
            print(f"   • Successful: {successful_tests}")
            print(f"   • Success rate: {successful_tests / total_tests * 100:.1f}%")

            print("\n💡 Production Resilience Patterns:")
            print("   • Retry logic with exponential backoff")
            print("   • Detailed error categorization and logging")
            print("   • Circuit breaker patterns for cascading failures")
            print("   • Graceful degradation when AI services are unavailable")

        return True

    except Exception as e:
        print(f"❌ Resilience testing error: {e}")
        return False

def performance_monitoring_and_alerting():
    """Demonstrate performance monitoring and alerting patterns."""
    print("\n\n📈 Performance Monitoring and Alerting")
    print("-" * 50)

    try:
        from genops.providers.openai import instrument_openai

        client = instrument_openai()

        # Performance test scenarios
        performance_thresholds = {
            "response_time_ms": 5000,    # 5 seconds max
            "cost_per_request": 0.01,    # $0.01 max per request
            "tokens_per_request": 1000,  # 1000 tokens max
            "success_rate": 0.95         # 95% success rate min
        }

        print("📊 Performance Thresholds:")
        for metric, threshold in performance_thresholds.items():
            print(f"   • {metric}: {threshold}")

        with production_workflow_context(
            "performance_monitoring",
            "monitoring-demo",
            team="sre-team",
            project="performance-optimization",
            environment="production"
        ) as (span, workflow_id):

            performance_metrics = {
                "total_requests": 0,
                "successful_requests": 0,
                "total_response_time": 0,
                "total_cost": 0,
                "total_tokens": 0,
                "alerts": []
            }

            # Simulate multiple requests for performance analysis
            test_requests = [
                "Explain quantum computing briefly",
                "What are the benefits of cloud computing?",
                "How does machine learning work?",
                "Describe the future of artificial intelligence"
            ]

            for i, request in enumerate(test_requests, 1):
                print(f"\n📡 Request {i}: {request[:30]}...")

                try:
                    start_time = time.time()

                    response = client.chat_completions_create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": request}],
                        max_tokens=200,

                        # Performance monitoring governance
                        team="sre-team",
                        project="performance-optimization",
                        workflow_id=workflow_id,
                        request_id=f"perf-test-{i}",
                        performance_monitoring=True
                    )

                    response_time = (time.time() - start_time) * 1000  # Convert to ms
                    tokens = response.usage.total_tokens
                    cost = (response.usage.prompt_tokens / 1000) * 0.0015 + (response.usage.completion_tokens / 1000) * 0.002

                    # Update metrics
                    performance_metrics["total_requests"] += 1
                    performance_metrics["successful_requests"] += 1
                    performance_metrics["total_response_time"] += response_time
                    performance_metrics["total_cost"] += cost
                    performance_metrics["total_tokens"] += tokens

                    # Check thresholds and generate alerts
                    alerts = check_performance_thresholds(response_time, cost, tokens, performance_thresholds)
                    if alerts:
                        performance_metrics["alerts"].extend(alerts)
                        for alert in alerts:
                            print(f"   🚨 ALERT: {alert}")

                    print(f"   ✅ Response time: {response_time:.0f}ms")
                    print(f"   💰 Cost: ${cost:.4f}")
                    print(f"   📊 Tokens: {tokens}")

                except Exception as e:
                    performance_metrics["total_requests"] += 1
                    performance_metrics["alerts"].append(f"Request failed: {e}")
                    print(f"   ❌ Request failed: {e}")

            # Calculate final metrics
            avg_response_time = performance_metrics["total_response_time"] / max(performance_metrics["successful_requests"], 1)
            success_rate = performance_metrics["successful_requests"] / performance_metrics["total_requests"] if performance_metrics["total_requests"] > 0 else 0
            avg_cost = performance_metrics["total_cost"] / max(performance_metrics["successful_requests"], 1)
            avg_tokens = performance_metrics["total_tokens"] / max(performance_metrics["successful_requests"], 1)

            # Set performance metrics in span
            span.set_attribute("avg_response_time_ms", avg_response_time)
            span.set_attribute("success_rate", success_rate)
            span.set_attribute("avg_cost_per_request", avg_cost)
            span.set_attribute("avg_tokens_per_request", avg_tokens)
            span.set_attribute("total_alerts", len(performance_metrics["alerts"]))

            print("\n📈 Performance Summary:")
            print(f"   • Average response time: {avg_response_time:.0f}ms")
            print(f"   • Success rate: {success_rate * 100:.1f}%")
            print(f"   • Average cost per request: ${avg_cost:.4f}")
            print(f"   • Average tokens per request: {avg_tokens:.0f}")
            print(f"   • Total alerts: {len(performance_metrics['alerts'])}")

            if performance_metrics["alerts"]:
                print("\n🚨 Performance Alerts:")
                for alert in performance_metrics["alerts"]:
                    print(f"   • {alert}")

        return True

    except Exception as e:
        print(f"❌ Performance monitoring error: {e}")
        return False

def check_performance_thresholds(response_time: float, cost: float, tokens: int, thresholds: dict) -> list[str]:
    """Check performance metrics against thresholds and generate alerts."""
    alerts = []

    if response_time > thresholds["response_time_ms"]:
        alerts.append(f"High response time: {response_time:.0f}ms > {thresholds['response_time_ms']}ms")

    if cost > thresholds["cost_per_request"]:
        alerts.append(f"High cost: ${cost:.4f} > ${thresholds['cost_per_request']}")

    if tokens > thresholds["tokens_per_request"]:
        alerts.append(f"High token usage: {tokens} > {thresholds['tokens_per_request']}")

    return alerts

def main():
    """Run production patterns demonstrations."""
    print("🏭 OpenAI Production Patterns with GenOps")
    print("=" * 60)

    # Check prerequisites
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set")
        print("💡 Fix: export OPENAI_API_KEY='your_api_key_here'")
        return False

    success = True

    # Run production pattern examples
    success &= customer_support_workflow()
    success &= content_pipeline_with_policy_enforcement()
    success &= resilience_and_error_handling()
    success &= performance_monitoring_and_alerting()

    # Summary
    print("\n" + "=" * 60)
    if success:
        print("🎉 Production patterns demonstration completed!")

        print("\n🏭 Production Patterns Covered:")
        print("   ✅ Complex workflow orchestration with context managers")
        print("   ✅ Policy enforcement and governance automation")
        print("   ✅ Resilience patterns with retry logic and error handling")
        print("   ✅ Performance monitoring and alerting systems")

        print("\n💼 Enterprise Benefits:")
        print("   • Complete audit trail and cost attribution")
        print("   • Automated governance and compliance enforcement")
        print("   • Proactive performance monitoring and alerting")
        print("   • Resilient systems with graceful failure handling")
        print("   • Scalable patterns for high-volume production workloads")

        print("\n🚀 Deployment Recommendations:")
        print("   • Implement circuit breaker patterns for external API calls")
        print("   • Set up comprehensive monitoring dashboards")
        print("   • Configure automated alerting for cost and performance thresholds")
        print("   • Establish backup provider strategies for critical workflows")
        print("   • Regular performance testing and capacity planning")

        return True
    else:
        print("❌ Production patterns demonstration encountered issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
