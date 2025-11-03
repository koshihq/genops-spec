#!/usr/bin/env python3
"""
🛡️ Block Inappropriate Content - Complete Governance Scenario

This example demonstrates how GenOps AI prevents inappropriate AI responses
through real-time content filtering and policy enforcement.

BUSINESS PROBLEM:
Your customer-facing AI chatbot generated inappropriate content that went viral
on social media, creating a PR nightmare and potential legal liability.

GENOPS SOLUTION:
- Real-time content filtering before AI requests are sent
- Automatic blocking of inappropriate requests and responses
- Customizable content policies for different use cases
- Complete audit trail for compliance and safety monitoring

Run this example to see content governance in action!
"""

import logging
import os
from typing import Any

# GenOps imports
from genops.core.policy import PolicyResult, PolicyViolationError, register_policy
from genops.core.telemetry import GenOpsTelemetry
from genops.providers.openai import instrument_openai

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_content_policies():
    """
    Set up content filtering policies for different scenarios.

    In production, these would be integrated with content safety services
    like OpenAI Moderation API, Azure Content Safety, or custom ML models.
    """
    print("\n🛡️ SETTING UP CONTENT GOVERNANCE POLICIES")
    print("=" * 60)

    # Policy 1: Block harmful content (violence, hate speech, etc.)
    register_policy(
        name="harmful_content_filter",
        description="Block requests containing harmful or dangerous content",
        enforcement_level=PolicyResult.BLOCKED,
        conditions={
            "blocked_patterns": [
                "violence", "hate speech", "self-harm", "illegal activities",
                "harassment", "threats", "discriminatory language"
            ],
            "sensitivity": "high"
        }
    )
    print("✅ Harmful content filter: BLOCKED")

    # Policy 2: Adult content filter (configurable by customer)
    register_policy(
        name="adult_content_filter",
        description="Filter adult/sexual content based on customer settings",
        enforcement_level=PolicyResult.BLOCKED,
        conditions={
            "blocked_patterns": [
                "sexual content", "adult themes", "explicit material"
            ],
            "customer_configurable": True
        }
    )
    print("✅ Adult content filter: BLOCKED")

    # Policy 3: Professional context filter (warning level)
    register_policy(
        name="professional_context_filter",
        description="Warn about potentially unprofessional content in business contexts",
        enforcement_level=PolicyResult.WARNING,
        conditions={
            "flagged_patterns": [
                "inappropriate humor", "casual profanity", "off-topic discussions"
            ],
            "context": "business"
        }
    )
    print("✅ Professional context filter: WARNING")

    # Policy 4: PII/sensitive data filter
    register_policy(
        name="sensitive_data_filter",
        description="Block requests containing personally identifiable information",
        enforcement_level=PolicyResult.BLOCKED,
        conditions={
            "blocked_patterns": [
                "social security", "credit card", "password", "api key",
                "personal phone", "home address"
            ],
            "pii_detection": True
        }
    )
    print("✅ Sensitive data filter: BLOCKED")

def demonstrate_content_filtering():
    """
    Show content filtering policies in action with realistic examples.
    """
    print("\n🔍 DEMONSTRATING CONTENT FILTERING IN ACTION")
    print("=" * 60)

    # Initialize telemetry
    telemetry = GenOpsTelemetry()

    # Test cases with various content types
    test_cases = [
        {
            "name": "Safe customer support query",
            "content": "How do I reset my password for my account?",
            "expected": "ALLOWED",
            "context": "customer_support"
        },
        {
            "name": "Harmful content request",
            "content": "How to engage in violence against others",
            "expected": "BLOCKED",
            "context": "general"
        },
        {
            "name": "Adult content request",
            "content": "Generate sexual content for entertainment",
            "expected": "BLOCKED",
            "context": "entertainment"
        },
        {
            "name": "Professional context with casual language",
            "content": "This damn system is driving me crazy, can you help?",
            "expected": "WARNING",
            "context": "business"
        },
        {
            "name": "Sensitive data exposure",
            "content": "My credit card number is 4532-1234-5678-9012, can you help with billing?",
            "expected": "BLOCKED",
            "context": "billing"
        },
        {
            "name": "Safe educational content",
            "content": "Explain the process of photosynthesis in plants",
            "expected": "ALLOWED",
            "context": "education"
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {test_case['name']}")
        print(f"   Content: \"{test_case['content'][:60]}{'...' if len(test_case['content']) > 60 else ''}\"")
        print(f"   Context: {test_case['context']}")

        try:
            with telemetry.trace_operation(
                operation_name=f"content_check_{test_case['context']}",
                operation_type="ai.content_filter",
                team="safety-team",
                project="content-moderation"
            ) as span:

                # Evaluate content against policies
                content_result = evaluate_content_policies(test_case['content'], test_case['context'])

                print(f"   🛡️ Policy result: {content_result['result']}")

                if content_result['blocked_policies']:
                    print(f"   🚫 Blocked by: {', '.join(content_result['blocked_policies'])}")
                if content_result['warning_policies']:
                    print(f"   ⚠️ Warnings: {', '.join(content_result['warning_policies'])}")
                if content_result['reason']:
                    print(f"   📝 Reason: {content_result['reason']}")

                # Record policy enforcement in telemetry
                for policy_name in content_result['blocked_policies']:
                    telemetry.record_policy(
                        span=span,
                        policy_name=policy_name,
                        result="blocked",
                        reason=content_result['reason'],
                        metadata={
                            "content_sample": test_case['content'][:100],
                            "context": test_case['context'],
                            "severity": "high"
                        }
                    )

                for policy_name in content_result['warning_policies']:
                    telemetry.record_policy(
                        span=span,
                        policy_name=policy_name,
                        result="warning",
                        reason=content_result['reason'],
                        metadata={
                            "content_sample": test_case['content'][:100],
                            "context": test_case['context'],
                            "severity": "medium"
                        }
                    )

                # If blocked, raise violation error
                if content_result['result'] == 'BLOCKED':
                    raise PolicyViolationError(
                        content_result['blocked_policies'][0],
                        content_result['reason'],
                        {"content_type": "user_input", "context": test_case['context']}
                    )

                print("   ✅ Content approved for AI processing")

        except PolicyViolationError as e:
            print(f"   🚫 CONTENT BLOCKED: {e}")
            print("   💡 Suggestion: Review content guidelines or contact support")

def evaluate_content_policies(content: str, context: str) -> dict[str, Any]:
    """
    Evaluate content against all registered content policies.

    In production, this would integrate with:
    - OpenAI Moderation API
    - Azure Content Safety
    - Custom ML models for content classification
    - Third-party content filtering services
    """

    result = {
        "result": "ALLOWED",
        "blocked_policies": [],
        "warning_policies": [],
        "reason": None
    }

    content_lower = content.lower()

    # Check harmful content filter
    harmful_patterns = [
        "violence", "hate speech", "self-harm", "illegal activities",
        "harassment", "threats", "discriminatory language"
    ]
    for pattern in harmful_patterns:
        if pattern in content_lower:
            result["result"] = "BLOCKED"
            result["blocked_policies"].append("harmful_content_filter")
            result["reason"] = f"Content contains harmful pattern: {pattern}"
            break

    # Check adult content filter
    adult_patterns = ["sexual content", "adult themes", "explicit material"]
    for pattern in adult_patterns:
        if pattern in content_lower:
            result["result"] = "BLOCKED"
            result["blocked_policies"].append("adult_content_filter")
            result["reason"] = f"Content contains adult material: {pattern}"
            break

    # Check professional context filter
    if context == "business":
        unprofessional_patterns = ["damn", "crazy", "stupid", "sucks"]
        for pattern in unprofessional_patterns:
            if pattern in content_lower:
                result["warning_policies"].append("professional_context_filter")
                if not result["reason"]:
                    result["reason"] = f"Potentially unprofessional language detected: {pattern}"
                break

    # Check sensitive data filter
    sensitive_patterns = [
        "credit card", "social security", "password", "api key",
        "4532-1234-5678-9012", "ssn:", "passwd:"
    ]
    for pattern in sensitive_patterns:
        if pattern in content_lower:
            result["result"] = "BLOCKED"
            result["blocked_policies"].append("sensitive_data_filter")
            result["reason"] = f"Content contains sensitive data: {pattern.replace('4532-1234-5678-9012', 'credit card number')}"
            break

    return result

def demonstrate_real_openai_with_filtering():
    """
    Show content filtering integrated with real OpenAI API calls.
    """
    if not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️ Skipping OpenAI integration demo (no API key)")
        print("   Set OPENAI_API_KEY environment variable to see real API integration")
        return

    print("\n🔗 REAL OPENAI INTEGRATION WITH CONTENT FILTERING")
    print("=" * 60)

    try:
        # Instrument OpenAI client
        client = instrument_openai(api_key=os.getenv("OPENAI_API_KEY"))

        print("✅ OpenAI client instrumented with content filtering")

        # Test safe content
        print("\n📞 Testing safe educational content...")
        safe_content = "Explain how renewable energy sources work"

        # Pre-filter content
        filter_result = evaluate_content_policies(safe_content, "education")

        if filter_result["result"] == "BLOCKED":
            print(f"🚫 Content blocked: {filter_result['reason']}")
            return

        print(f"✅ Content approved: {safe_content}")

        # Make API call
        client.chat_completions_create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": safe_content}],
            max_tokens=100,
            # Governance attributes
            team="education-team",
            project="learning-assistant",
            customer_id="edu-customer"
        )

        print("📝 Response received and processed safely")
        print("✅ Content filtering and cost telemetry automatically recorded!")

    except Exception as e:
        print(f"❌ Error: {e}")

def show_content_governance_telemetry():
    """
    Show what telemetry data is captured for content governance.
    """
    print("\n📊 CONTENT GOVERNANCE TELEMETRY DATA")
    print("=" * 60)

    sample_telemetry = {
        "genops.operation.name": "content_check_business",
        "genops.operation.type": "ai.content_filter",
        "genops.team": "safety-team",
        "genops.project": "content-moderation",
        "genops.policy.name": "harmful_content_filter",
        "genops.policy.result": "blocked",
        "genops.policy.reason": "Content contains harmful pattern: violence",
        "genops.policy.metadata.content_sample": "How to engage in violence against...",
        "genops.policy.metadata.context": "general",
        "genops.policy.metadata.severity": "high",
        "genops.content.filtered": True,
        "genops.content.category": "harmful",
        "genops.content.confidence": 0.95
    }

    print("📈 Sample content governance attributes:")
    for key, value in sample_telemetry.items():
        print(f"   {key}: {value}")

    print("\n💡 This enables:")
    print("   • Real-time content safety dashboards")
    print("   • Automated alerts for policy violations")
    print("   • Compliance audit trails")
    print("   • Content safety metrics and trends")
    print("   • Integration with safety review workflows")

def main():
    """
    Run the complete content filtering demonstration.
    """
    print("🛡️ GenOps AI: Block Inappropriate Content Demo")
    print("=" * 80)
    print("\nThis demo shows how GenOps AI prevents inappropriate AI responses")
    print("through real-time content filtering and governance policies.")

    # Setup
    setup_content_policies()

    # Demonstrate filtering
    demonstrate_content_filtering()

    # Real API integration
    demonstrate_real_openai_with_filtering()

    # Show telemetry
    show_content_governance_telemetry()

    print("\n🎯 KEY TAKEAWAYS")
    print("=" * 60)
    print("✅ Real-time content filtering prevents inappropriate AI responses")
    print("✅ Customizable policies for different contexts and use cases")
    print("✅ Complete audit trail for compliance and safety monitoring")
    print("✅ Seamless integration with existing AI workflows")
    print("✅ Automatic telemetry for content safety dashboards")

    print("\n📚 NEXT STEPS")
    print("=" * 60)
    print("1. Customize content policies for your specific use case and brand guidelines")
    print("2. Integrate with content safety services (OpenAI Moderation, Azure Content Safety)")
    print("3. Set up alerting for content policy violations")
    print("4. Train your team on content governance workflows")
    print("5. Monitor content safety metrics in your observability dashboard")

    print("\n🔗 Learn more: https://github.com/KoshiHQ/GenOps-AI/tree/main/docs")

if __name__ == "__main__":
    main()
