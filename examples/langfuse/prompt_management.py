#!/usr/bin/env python3
"""
Langfuse Prompt Management with GenOps Cost Intelligence Example

This example demonstrates advanced prompt management workflows with Langfuse
enhanced by GenOps cost optimization and governance. Perfect for teams that
need systematic prompt engineering with cost attribution and performance tracking.

Usage:
    python prompt_management.py

Prerequisites:
    pip install genops[langfuse]
    export LANGFUSE_PUBLIC_KEY="pk-lf-your-public-key"
    export LANGFUSE_SECRET_KEY="sk-lf-your-secret-key"
    export OPENAI_API_KEY="your-openai-api-key"  # Or another provider
"""

import os
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


@dataclass
class PromptVersion:
    """Represents a prompt version with cost and performance metrics."""

    version_id: str
    prompt_template: str
    version_number: str
    tags: list[str]
    cost_per_execution: float = 0.0
    avg_latency_ms: float = 0.0
    quality_score: float = 0.0
    execution_count: int = 0
    total_cost: float = 0.0
    governance_metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.governance_metadata is None:
            self.governance_metadata = {}


class PromptManager:
    """Advanced prompt management with GenOps cost intelligence."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.prompt_registry = {}
        self.version_history = {}
        self.performance_metrics = {}

    def register_prompt(
        self,
        prompt_id: str,
        prompt_template: str,
        version: str = "1.0.0",
        tags: Optional[list[str]] = None,
        **governance_attrs,
    ) -> PromptVersion:
        """Register a new prompt version with governance tracking."""

        version_id = f"{prompt_id}_{version}_{str(uuid.uuid4())[:8]}"

        prompt_version = PromptVersion(
            version_id=version_id,
            prompt_template=prompt_template,
            version_number=version,
            tags=tags or [],
            governance_metadata=governance_attrs,
        )

        if prompt_id not in self.prompt_registry:
            self.prompt_registry[prompt_id] = []
            self.version_history[prompt_id] = []

        self.prompt_registry[prompt_id].append(prompt_version)
        self.version_history[prompt_id].append(version_id)

        print(f"📝 Registered prompt '{prompt_id}' version {version}")
        print(f"   🆔 Version ID: {version_id}")
        print(f"   🏷️  Tags: {', '.join(tags) if tags else 'None'}")
        if governance_attrs:
            print(
                f"   🛡️  Governance: {', '.join(f'{k}={v}' for k, v in governance_attrs.items())}"
            )

        return prompt_version

    def execute_prompt_version(
        self,
        prompt_id: str,
        version_id: str,
        variables: dict[str, Any],
        model: str = "gpt-3.5-turbo",
        max_cost: float = 0.10,
        **governance_attrs,
    ) -> dict[str, Any]:
        """Execute a specific prompt version with cost tracking."""

        # Find the prompt version
        prompt_version = None
        for version in self.prompt_registry.get(prompt_id, []):
            if version.version_id == version_id:
                prompt_version = version
                break

        if not prompt_version:
            raise ValueError(f"Prompt version {version_id} not found for {prompt_id}")

        # Format the prompt with variables
        try:
            formatted_prompt = prompt_version.prompt_template.format(**variables)
        except KeyError as e:
            raise ValueError(f"Missing variable {e} for prompt template") from e

        # Merge governance attributes
        merged_governance = {
            **prompt_version.governance_metadata,
            **governance_attrs,
            "prompt_id": prompt_id,
            "prompt_version": version_id,
        }

        with self.adapter.trace_with_governance(
            name=f"prompt_execution_{prompt_id}", **merged_governance
        ):
            # Execute the prompt with cost tracking
            response = self.adapter.generation_with_cost_tracking(
                prompt=formatted_prompt,
                model=model,
                max_cost=max_cost,
                operation=f"prompt_{prompt_id}_execution",
                **merged_governance,
            )

            # Update prompt version metrics
            prompt_version.execution_count += 1
            prompt_version.total_cost += response.usage.cost
            prompt_version.cost_per_execution = (
                prompt_version.total_cost / prompt_version.execution_count
            )

            # Update latency (running average)
            if prompt_version.avg_latency_ms == 0:
                prompt_version.avg_latency_ms = response.usage.latency_ms
            else:
                prompt_version.avg_latency_ms = (
                    prompt_version.avg_latency_ms * (prompt_version.execution_count - 1)
                    + response.usage.latency_ms
                ) / prompt_version.execution_count

            return {
                "response": response,
                "prompt_version": prompt_version,
                "formatted_prompt": formatted_prompt,
                "variables_used": variables,
                "execution_metrics": {
                    "cost": response.usage.cost,
                    "latency_ms": response.usage.latency_ms,
                    "tokens": response.usage.total_tokens,
                },
            }

    def compare_prompt_versions(
        self,
        prompt_id: str,
        test_variables: list[dict[str, Any]],
        models: Optional[list[str]] = None,
        **governance_attrs,
    ) -> dict[str, Any]:
        """Compare all versions of a prompt with governance tracking."""

        if prompt_id not in self.prompt_registry:
            raise ValueError(f"Prompt {prompt_id} not found in registry")

        if models is None:
            models = ["gpt-3.5-turbo"]

        prompt_versions = self.prompt_registry[prompt_id]
        comparison_results = {}

        print(f"🔬 Comparing {len(prompt_versions)} versions of '{prompt_id}'")
        print(f"   🧪 Test cases: {len(test_variables)}")
        print(f"   🤖 Models: {', '.join(models)}")

        for version in prompt_versions:
            version_results = []

            print(
                f"\n📊 Testing version {version.version_number} ({version.version_id[:12]}...)"
            )

            for i, variables in enumerate(test_variables, 1):
                for model in models:
                    print(f"   🧪 Test case {i}/{len(test_variables)} on {model}")

                    try:
                        result = self.execute_prompt_version(
                            prompt_id=prompt_id,
                            version_id=version.version_id,
                            variables=variables,
                            model=model,
                            max_cost=0.15,
                            comparison_test=True,
                            test_case=i,
                            **governance_attrs,
                        )

                        version_results.append(
                            {
                                "test_case": i,
                                "model": model,
                                "variables": variables,
                                "response": result["response"].content,
                                "cost": result["execution_metrics"]["cost"],
                                "latency_ms": result["execution_metrics"]["latency_ms"],
                                "tokens": result["execution_metrics"]["tokens"],
                            }
                        )

                    except Exception as e:
                        print(f"     ❌ Failed: {e}")
                        version_results.append(
                            {
                                "test_case": i,
                                "model": model,
                                "variables": variables,
                                "error": str(e),
                                "cost": 0.0,
                                "latency_ms": 0.0,
                                "tokens": 0,
                            }
                        )

            # Calculate version summary
            successful_results = [r for r in version_results if "error" not in r]
            if successful_results:
                avg_cost = sum(r["cost"] for r in successful_results) / len(
                    successful_results
                )
                avg_latency = sum(r["latency_ms"] for r in successful_results) / len(
                    successful_results
                )
                avg_tokens = sum(r["tokens"] for r in successful_results) / len(
                    successful_results
                )
                success_rate = len(successful_results) / len(version_results)
            else:
                avg_cost = avg_latency = avg_tokens = success_rate = 0.0

            comparison_results[version.version_id] = {
                "version": version,
                "results": version_results,
                "summary": {
                    "success_rate": success_rate,
                    "avg_cost": avg_cost,
                    "avg_latency_ms": avg_latency,
                    "avg_tokens": avg_tokens,
                    "total_executions": len(version_results),
                },
            }

        return comparison_results

    def optimize_prompt_cost(
        self, prompt_id: str, target_cost_reduction: float = 0.2, **governance_attrs
    ) -> list[dict[str, Any]]:
        """Generate cost optimization suggestions for a prompt."""

        if prompt_id not in self.prompt_registry:
            raise ValueError(f"Prompt {prompt_id} not found")

        prompt_versions = self.prompt_registry[prompt_id]

        print(f"💡 Analyzing cost optimization opportunities for '{prompt_id}'")
        print(f"   🎯 Target cost reduction: {target_cost_reduction:.1%}")

        optimizations = []

        for version in prompt_versions:
            if version.execution_count == 0:
                continue

            current_cost = version.cost_per_execution
            target_cost = current_cost * (1 - target_cost_reduction)

            # Analyze prompt characteristics
            len(version.prompt_template)
            word_count = len(version.prompt_template.split())

            suggestions = []

            # Length-based optimizations
            if word_count > 100:
                suggestions.append(
                    {
                        "type": "prompt_length_reduction",
                        "description": "Reduce prompt length by removing redundant instructions",
                        "estimated_cost_savings": current_cost * 0.15,
                        "implementation": "Simplify instructions and remove examples",
                    }
                )

            # Model selection optimizations
            if current_cost > 0.01:  # High cost threshold
                suggestions.append(
                    {
                        "type": "model_optimization",
                        "description": "Consider using a more cost-effective model",
                        "estimated_cost_savings": current_cost * 0.4,
                        "implementation": "Test with gpt-3.5-turbo instead of gpt-4",
                    }
                )

            # Template optimization
            if "{" in version.prompt_template and "}" in version.prompt_template:
                suggestions.append(
                    {
                        "type": "template_optimization",
                        "description": "Optimize variable placement to reduce token usage",
                        "estimated_cost_savings": current_cost * 0.1,
                        "implementation": "Move variables to end of prompt template",
                    }
                )

            # Caching opportunities
            if version.execution_count > 10:
                suggestions.append(
                    {
                        "type": "response_caching",
                        "description": "Implement caching for repeated similar requests",
                        "estimated_cost_savings": current_cost * 0.3,
                        "implementation": "Cache responses based on variable patterns",
                    }
                )

            optimization_data = {
                "version_id": version.version_id,
                "version_number": version.version_number,
                "current_cost_per_execution": current_cost,
                "target_cost": target_cost,
                "execution_count": version.execution_count,
                "suggestions": suggestions,
                "total_potential_savings": sum(
                    s["estimated_cost_savings"] for s in suggestions
                ),
            }

            optimizations.append(optimization_data)

        return optimizations


def demonstrate_prompt_registration():
    """Demonstrate prompt registration and versioning."""
    print("📝 Prompt Registration and Version Management")
    print("=" * 44)

    try:
        from genops.providers.langfuse import instrument_langfuse

        # Initialize adapter for prompt management
        adapter = instrument_langfuse(
            team="prompt-engineering-team",
            project="prompt-optimization",
            environment="development",
            budget_limits={"daily": 3.0},
        )

        print("✅ GenOps Langfuse adapter initialized for prompt management")
        print(f"   🏷️  Team: {adapter.team}")
        print(f"   📊 Project: {adapter.project}")

        # Initialize prompt manager
        manager = PromptManager(adapter)

        # Register different versions of a customer support prompt
        customer_support_prompts = [
            {
                "version": "1.0.0",
                "template": "You are a helpful customer support agent. Please assist the customer with their question: {customer_question}. Provide a clear and professional response.",
                "tags": ["customer-support", "basic", "professional"],
            },
            {
                "version": "1.1.0",
                "template": "You are an expert customer support specialist. Customer question: {customer_question}. Provide a detailed, empathetic response with specific solutions.",
                "tags": ["customer-support", "detailed", "empathetic"],
            },
            {
                "version": "2.0.0",
                "template": "As a senior customer success manager, help resolve this customer inquiry: {customer_question}. Include troubleshooting steps if applicable and offer additional resources.",
                "tags": ["customer-support", "senior-level", "comprehensive"],
            },
        ]

        registered_versions = []

        for prompt_config in customer_support_prompts:
            version = manager.register_prompt(
                prompt_id="customer_support_assistant",
                prompt_template=prompt_config["template"],
                version=prompt_config["version"],
                tags=prompt_config["tags"],
                customer_id="prompt-mgmt-customer",
                cost_center="customer-success",
            )
            registered_versions.append(version)

        print(f"\n✅ Registered {len(registered_versions)} prompt versions")
        print("   📋 Prompt ID: customer_support_assistant")
        print(f"   🔢 Versions: {[v.version_number for v in registered_versions]}")

        return manager, registered_versions

    except Exception as e:
        print(f"❌ Prompt registration failed: {e}")
        return None, None


def demonstrate_prompt_execution():
    """Demonstrate prompt execution with variable substitution."""
    print("\n🚀 Prompt Execution with Cost Tracking")
    print("=" * 38)

    manager, versions = demonstrate_prompt_registration()
    if not manager or not versions:
        return None

    try:
        # Test scenarios for customer support
        test_scenarios = [
            {
                "scenario": "Password Reset",
                "variables": {
                    "customer_question": "I forgot my password and can't log into my account. How do I reset it?"
                },
                "expected_topics": ["password", "reset", "account", "login"],
            },
            {
                "scenario": "Billing Inquiry",
                "variables": {
                    "customer_question": "Why was I charged twice for my subscription this month?"
                },
                "expected_topics": ["billing", "charge", "subscription", "duplicate"],
            },
            {
                "scenario": "Feature Request",
                "variables": {
                    "customer_question": "Can you add dark mode to the mobile app?"
                },
                "expected_topics": ["feature", "dark mode", "mobile", "app"],
            },
        ]

        execution_results = []

        for scenario in test_scenarios:
            print(f"\n🧪 Testing scenario: {scenario['scenario']}")
            print("-" * 30)

            # Test with the latest version (2.0.0)
            latest_version = max(versions, key=lambda v: v.version_number)

            result = manager.execute_prompt_version(
                prompt_id="customer_support_assistant",
                version_id=latest_version.version_id,
                variables=scenario["variables"],
                model="gpt-3.5-turbo",
                max_cost=0.08,
                customer_id="execution-test-customer",
                cost_center="customer-success",
                scenario=scenario["scenario"],
            )

            execution_results.append(result)

            print(f"   📝 Response: {result['response'].content[:100]}...")
            print(f"   💰 Cost: ${result['execution_metrics']['cost']:.6f}")
            print(f"   ⏱️  Latency: {result['execution_metrics']['latency_ms']:.0f}ms")
            print(f"   🎯 Tokens: {result['execution_metrics']['tokens']}")
            print(f"   📊 Version: {result['prompt_version'].version_number}")

        print(f"\n✅ Executed {len(execution_results)} prompt scenarios")

        # Show updated version metrics
        print("\n📊 Version Performance Summary:")
        print(f"   Version: {latest_version.version_number}")
        print(f"   Executions: {latest_version.execution_count}")
        print(f"   Avg Cost: ${latest_version.cost_per_execution:.6f}")
        print(f"   Avg Latency: {latest_version.avg_latency_ms:.0f}ms")
        print(f"   Total Cost: ${latest_version.total_cost:.6f}")

        return execution_results

    except Exception as e:
        print(f"❌ Prompt execution failed: {e}")
        return None


def demonstrate_version_comparison():
    """Demonstrate A/B testing of prompt versions."""
    print("\n🔬 A/B Testing: Comparing Prompt Versions")
    print("=" * 40)

    manager, versions = demonstrate_prompt_registration()
    if not manager:
        return None

    try:
        # Test cases for comparison
        test_cases = [
            {
                "customer_question": "My order hasn't arrived yet and it's been 5 days. What should I do?"
            },
            {
                "customer_question": "I want to cancel my subscription but I can't find the option in settings."
            },
            {
                "customer_question": "The app keeps crashing when I try to upload photos. Can you help?"
            },
        ]

        print(f"🧪 Running A/B test with {len(test_cases)} test cases")
        print("📊 Comparing cost, performance, and response quality across versions")

        comparison_results = manager.compare_prompt_versions(
            prompt_id="customer_support_assistant",
            test_variables=test_cases,
            models=["gpt-3.5-turbo"],
            customer_id="ab-test-customer",
            cost_center="product-optimization",
            ab_test=True,
        )

        print("\n📈 A/B Test Results Summary:")
        print("=" * 28)

        # Sort versions by performance
        version_summaries = []
        for version_id, data in comparison_results.items():
            summary = data["summary"]
            version = data["version"]

            version_summaries.append(
                {
                    "version": version.version_number,
                    "version_id": version_id,
                    "success_rate": summary["success_rate"],
                    "avg_cost": summary["avg_cost"],
                    "avg_latency": summary["avg_latency_ms"],
                    "avg_tokens": summary["avg_tokens"],
                }
            )

        # Display comparison table
        print("Version | Success Rate | Avg Cost    | Avg Latency | Avg Tokens")
        print("-" * 65)

        for summary in sorted(version_summaries, key=lambda x: x["version"]):
            print(
                f"{summary['version']:<7} | {summary['success_rate']:>11.1%} | ${summary['avg_cost']:>10.6f} | {summary['avg_latency']:>10.0f}ms | {summary['avg_tokens']:>9.0f}"
            )

        # Identify best performing version
        best_cost = min(version_summaries, key=lambda x: x["avg_cost"])
        best_speed = min(version_summaries, key=lambda x: x["avg_latency"])
        best_success = max(version_summaries, key=lambda x: x["success_rate"])

        print("\n🏆 Performance Winners:")
        print(
            f"   💰 Most Cost Effective: Version {best_cost['version']} (${best_cost['avg_cost']:.6f})"
        )
        print(
            f"   ⚡ Fastest: Version {best_speed['version']} ({best_speed['avg_latency']:.0f}ms)"
        )
        print(
            f"   ✅ Most Reliable: Version {best_success['version']} ({best_success['success_rate']:.1%})"
        )

        return comparison_results

    except Exception as e:
        print(f"❌ Version comparison failed: {e}")
        return None


def demonstrate_cost_optimization():
    """Demonstrate cost optimization analysis and recommendations."""
    print("\n💡 Cost Optimization Analysis")
    print("=" * 29)

    manager, versions = demonstrate_prompt_registration()
    if not manager:
        return None

    try:
        # First, execute some prompts to generate cost data
        print("📊 Generating cost data for optimization analysis...")

        sample_variables = [
            {"customer_question": "How do I update my payment method?"},
            {"customer_question": "What are your business hours?"},
            {
                "customer_question": "I need help with setting up two-factor authentication."
            },
            {"customer_question": "Can I get a refund for my last purchase?"},
            {
                "customer_question": "The website is loading slowly for me. Any suggestions?"
            },
        ]

        # Execute each version a few times to build cost history
        for version in versions[:2]:  # Test first two versions
            for variables in sample_variables[:3]:  # Test with 3 scenarios each
                try:
                    manager.execute_prompt_version(
                        prompt_id="customer_support_assistant",
                        version_id=version.version_id,
                        variables=variables,
                        model="gpt-3.5-turbo",
                        max_cost=0.05,
                        customer_id="cost-analysis-customer",
                        cost_center="optimization",
                    )
                except Exception as e:
                    print(f"   ⚠️  Execution failed: {e}")

        # Run cost optimization analysis
        optimizations = manager.optimize_prompt_cost(
            prompt_id="customer_support_assistant",
            target_cost_reduction=0.25,  # 25% cost reduction target
            customer_id="cost-optimization-customer",
            cost_center="cost-management",
        )

        print("\n💰 Cost Optimization Recommendations:")
        print("=" * 37)

        for _i, opt in enumerate(optimizations, 1):
            if opt["current_cost_per_execution"] == 0:
                continue

            print(f"\n📊 Version {opt['version_number']} Analysis:")
            print(
                f"   💰 Current cost per execution: ${opt['current_cost_per_execution']:.6f}"
            )
            print(f"   🎯 Target cost: ${opt['target_cost']:.6f}")
            print(f"   📈 Total executions: {opt['execution_count']}")
            print(
                f"   💡 Potential total savings: ${opt['total_potential_savings']:.6f}"
            )

            print("\n   🔧 Optimization Suggestions:")
            for j, suggestion in enumerate(opt["suggestions"], 1):
                print(f"   {j}. {suggestion['type'].replace('_', ' ').title()}")
                print(f"      📝 {suggestion['description']}")
                print(
                    f"      💰 Est. savings: ${suggestion['estimated_cost_savings']:.6f}"
                )
                print(f"      🛠️  Implementation: {suggestion['implementation']}")
                print()

        # Calculate overall optimization potential
        total_current_cost = sum(
            opt["current_cost_per_execution"] * opt["execution_count"]
            for opt in optimizations
            if opt["execution_count"] > 0
        )
        total_potential_savings = sum(
            opt["total_potential_savings"]
            for opt in optimizations
            if opt["execution_count"] > 0
        )

        if total_current_cost > 0:
            savings_percentage = (total_potential_savings / total_current_cost) * 100
            print("🎯 Overall Optimization Potential:")
            print(f"   📊 Total historical cost: ${total_current_cost:.6f}")
            print(f"   💰 Potential savings: ${total_potential_savings:.6f}")
            print(f"   📈 Percentage savings: {savings_percentage:.1f}%")

        return optimizations

    except Exception as e:
        print(f"❌ Cost optimization analysis failed: {e}")
        return None


def demonstrate_prompt_governance():
    """Demonstrate governance features for prompt management."""
    print("\n🛡️ Prompt Management Governance Features")
    print("=" * 41)

    try:
        from genops.providers.langfuse import instrument_langfuse

        # Initialize adapter with governance policies
        adapter = instrument_langfuse(
            team="governance-prompt-team",
            project="enterprise-prompt-management",
            environment="production",
            budget_limits={"daily": 5.0, "monthly": 100.0},
        )

        manager = PromptManager(adapter)

        print("🏛️  Enterprise Governance Features:")
        governance_features = [
            "💰 Cost attribution per prompt version and execution",
            "🏷️  Team and project tracking for all prompt operations",
            "📊 Customer-specific prompt performance analytics",
            "🛡️  Budget enforcement with automatic cost controls",
            "📈 Compliance reporting for prompt usage across teams",
            "🔍 Audit trails for all prompt modifications and executions",
            "⚠️  Policy violation detection and alerting",
        ]

        for feature in governance_features:
            print(f"   {feature}")

        # Demonstrate governance attributes in prompt registration
        manager.register_prompt(
            prompt_id="enterprise_email_assistant",
            prompt_template="As a professional business email assistant, help draft an email about: {email_topic}. Ensure the tone is {tone} and include {key_points}.",
            version="1.0.0",
            tags=["business", "email", "professional"],
            customer_id="enterprise-123",
            cost_center="business-communications",
            compliance_level="high",
            data_classification="internal",
            approval_required=False,
        )

        print("\n📋 Registered Enterprise Prompt:")
        print("   🆔 ID: enterprise_email_assistant")
        print("   📊 Governance attributes: customer_id, cost_center, compliance_level")
        print("   🛡️  Data classification: internal")
        print("   ✅ Approval required: No")

        # Show governance summary
        cost_summary = adapter.get_cost_summary("daily")
        print("\n📊 Current Governance Summary:")
        print(f"   💰 Daily cost: ${cost_summary['total_cost']:.6f}")
        print(f"   📈 Operations: {cost_summary['operation_count']}")
        print(f"   🏷️  Team: {cost_summary['governance']['team']}")
        print(f"   📊 Project: {cost_summary['governance']['project']}")
        print(f"   💡 Budget remaining: ${cost_summary['budget_remaining']:.6f}")

        return True

    except Exception as e:
        print(f"❌ Governance demonstration failed: {e}")
        return False


def show_next_steps():
    """Show next steps for advanced prompt management."""
    print("\n🚀 Advanced Prompt Management & Next Steps")
    print("=" * 42)

    advanced_features = [
        (
            "🔄 Automated A/B Testing",
            "Continuous prompt optimization with statistical significance",
            "Set up automated version comparison workflows",
        ),
        (
            "📊 Performance Dashboards",
            "Real-time prompt performance monitoring",
            "Integrate with existing observability platforms",
        ),
        (
            "🎯 Personalized Prompts",
            "Dynamic prompt generation based on user context",
            "Implement context-aware prompt selection",
        ),
        (
            "🔍 Prompt Analytics",
            "Deep analysis of prompt performance patterns",
            "Advanced analytics with business intelligence",
        ),
        (
            "🏭 Enterprise Deployment",
            "Scale prompt management across organization",
            "python production_patterns.py",
        ),
    ]

    for title, description, next_step in advanced_features:
        print(f"   {title}")
        print(f"     Purpose: {description}")
        print(f"     Next Step: {next_step}")
        print()

    print("📚 Resources for Advanced Prompt Management:")
    print("   • Advanced Observability: python advanced_observability.py")
    print("   • Production Patterns: python production_patterns.py")
    print("   • Comprehensive Guide: docs/integrations/langfuse.md")
    print("   • Prompt Engineering Best Practices: docs/prompt-engineering.md")


def main():
    """Main function to run the prompt management example."""
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check prerequisites
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        print("❌ Missing LANGFUSE_PUBLIC_KEY environment variable")
        print("💡 Get your keys at: https://cloud.langfuse.com/")
        return False

    if not os.getenv("LANGFUSE_SECRET_KEY"):
        print("❌ Missing LANGFUSE_SECRET_KEY environment variable")
        print("💡 Get your keys at: https://cloud.langfuse.com/")
        return False

    if not any([os.getenv("OPENAI_API_KEY"), os.getenv("ANTHROPIC_API_KEY")]):
        print("❌ No AI provider API keys found")
        print("💡 Set at least one:")
        print("   export OPENAI_API_KEY='your_openai_key'")
        print("   export ANTHROPIC_API_KEY='your_anthropic_key'")
        return False

    # Run prompt management demonstrations
    success = True

    # Prompt registration and versioning
    manager, versions = demonstrate_prompt_registration()
    success &= manager is not None and versions is not None

    # Prompt execution with cost tracking
    execution_results = demonstrate_prompt_execution()
    success &= execution_results is not None

    # A/B testing of versions
    comparison_results = demonstrate_version_comparison()
    success &= comparison_results is not None

    # Cost optimization analysis
    optimization_results = demonstrate_cost_optimization()
    success &= optimization_results is not None

    # Governance features
    governance_success = demonstrate_prompt_governance()
    success &= governance_success

    if success:
        show_next_steps()
        print("\n" + "📝" * 20)
        print("Prompt Management + GenOps Cost Intelligence complete!")
        print("Advanced prompt engineering with governance and optimization!")
        print("Enterprise-ready prompt management with cost attribution!")
        print("📝" * 20)
        return True
    else:
        print("\n❌ Some demonstrations failed. Check the errors above.")
        return False


if __name__ == "__main__":
    """Main entry point."""
    success = main()
    sys.exit(0 if success else 1)
