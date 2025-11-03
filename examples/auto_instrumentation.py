"""Auto-instrumentation examples for GenOps AI."""

import logging

# Configure logging to see GenOps initialization messages
logging.basicConfig(level=logging.INFO)

def example_simple_init():
    """Example 1: Simple auto-instrumentation setup."""
    print("=" * 60)
    print("Example 1: Simple Auto-Instrumentation")
    print("=" * 60)

    import genops

    # One-line initialization - auto-detects and instruments everything
    genops.init()

    # Show status
    status = genops.status()
    print("✓ Initialization complete")
    print(f"  Instrumented providers: {', '.join(status['instrumented_providers'])}")
    print(f"  Available providers: {status['available_providers']}")

    # Now any AI provider calls will be automatically tracked
    # (This would work if OpenAI/Anthropic were installed)
    print("\n💡 Any AI provider calls are now automatically tracked with governance telemetry!")


def example_advanced_init():
    """Example 2: Advanced configuration with specific settings."""
    print("\n" + "=" * 60)
    print("Example 2: Advanced Auto-Instrumentation Configuration")
    print("=" * 60)

    import genops

    # Advanced initialization with custom configuration
    genops.init(
        service_name="my-ai-service",
        service_version="1.0.0",
        environment="development",
        exporter_type="console",  # or "otlp" for production
        # For OTLP: otlp_endpoint="https://api.honeycomb.io",
        # For OTLP: otlp_headers={"x-honeycomb-team": "your-api-key"},
        default_team="ai-team",
        default_project="chatbot-service",
        default_environment="dev"
    )

    print("✓ Advanced initialization complete with custom settings")

    # Show the configured default attributes
    defaults = genops.get_default_attributes()
    print(f"  Default governance attributes: {defaults}")


def example_manual_with_defaults():
    """Example 3: Using manual instrumentation with auto-instrumentation defaults."""
    print("\n" + "=" * 60)
    print("Example 3: Manual Instrumentation with Auto-Init Defaults")
    print("=" * 60)

    import genops

    # Initialize with defaults
    genops.init(
        default_team="platform-team",
        default_project="ai-platform",
        exporter_type="console"
    )

    # Get the default attributes for manual instrumentation
    defaults = genops.get_default_attributes()
    print(f"Auto-configured defaults: {defaults}")

    # Use manual instrumentation that inherits defaults
    @genops.track_usage(
        operation_name="sentiment_analysis",
        # team and project are inherited from init()
        feature="content-moderation"
    )
    def analyze_sentiment(text: str) -> dict:
        """Analyze text sentiment (mock implementation)."""
        # This would call an actual AI service
        return {
            "sentiment": "positive" if "good" in text.lower() else "neutral",
            "confidence": 0.85
        }

    # Use the instrumented function
    result = analyze_sentiment("This is a good example")
    print(f"✓ Manual instrumentation completed: {result}")


def example_provider_specific():
    """Example 4: Provider-specific instrumentation."""
    print("\n" + "=" * 60)
    print("Example 4: Provider-Specific Instrumentation")
    print("=" * 60)

    import genops

    # Initialize with specific providers only
    genops.init(
        providers=["openai"],  # Only instrument OpenAI, not Anthropic
        service_name="openai-only-service"
    )

    status = genops.status()
    print("✓ Provider-specific initialization")
    print("  Requested providers: ['openai']")
    print(f"  Actually instrumented: {status['instrumented_providers']}")


def example_with_policies():
    """Example 5: Auto-instrumentation with governance policies."""
    print("\n" + "=" * 60)
    print("Example 5: Auto-Instrumentation with Governance Policies")
    print("=" * 60)

    import genops
    from genops.core.policy import PolicyResult, register_policy

    # Initialize GenOps
    genops.init(
        service_name="governed-ai-service",
        default_team="ai-governance",
        exporter_type="console"
    )

    # Register governance policies
    register_policy(
        name="cost_control",
        description="Prevent expensive operations",
        enforcement_level=PolicyResult.WARNING,
        max_cost=1.00
    )

    register_policy(
        name="content_safety",
        description="Filter unsafe content",
        enforcement_level=PolicyResult.BLOCKED,
        blocked_patterns=["violence", "explicit"]
    )

    print("✓ Auto-instrumentation + governance policies configured")
    print("  All AI provider calls will be automatically tracked AND governed")

    # Example of using policy enforcement with auto-instrumentation
    @genops.enforce_policy(["cost_control"])
    def expensive_ai_operation(prompt: str) -> str:
        """AI operation with cost governance."""
        # This would call an actual AI service
        # Cost tracking happens automatically via auto-instrumentation
        return f"AI response to: {prompt[:50]}..."

    try:
        result = expensive_ai_operation("Generate a comprehensive report")
        print(f"✓ Policy-governed operation: {result}")
    except Exception as e:
        print(f"⚠️ Policy violation: {e}")


def example_uninstrumentation():
    """Example 6: Removing instrumentation."""
    print("\n" + "=" * 60)
    print("Example 6: Removing Auto-Instrumentation")
    print("=" * 60)

    import genops

    # Check current status
    status_before = genops.status()
    print(f"Before uninstrumentation: {status_before['initialized']}")

    # Remove all instrumentation
    genops.uninstrument()

    # Check status after
    status_after = genops.status()
    print(f"After uninstrumentation: {status_after['initialized']}")
    print("✓ All GenOps instrumentation removed")


def main():
    """Run all auto-instrumentation examples."""
    print("🚀 GenOps AI Auto-Instrumentation Examples")
    print("This demonstrates the OpenLLMetry-inspired auto-instrumentation system")

    # Run examples
    example_simple_init()
    example_advanced_init()
    example_manual_with_defaults()
    example_provider_specific()
    example_with_policies()
    example_uninstrumentation()

    print("\n" + "=" * 60)
    print("🎉 All Examples Complete!")
    print("=" * 60)
    print("Key Benefits of Auto-Instrumentation:")
    print("• One-line setup: genops.init()")
    print("• Automatic provider detection")
    print("• Zero-code governance telemetry")
    print("• Compatible with existing AI code")
    print("• Configurable defaults and policies")
    print("\nNext Steps:")
    print("1. Install AI providers: pip install openai anthropic")
    print("2. Add genops.init() to your app startup")
    print("3. Your existing AI code gets automatic governance!")


if __name__ == "__main__":
    main()
