"""
Example: Basic LangChain Chain Tracking
Demonstrates: Simple chain execution with governance telemetry
Use case: Getting started with GenOps LangChain integration
"""

import logging
import os

# Core LangChain imports
try:
    from langchain.chains import LLMChain
    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate
except ImportError:
    print("❌ LangChain not installed. Run: pip install langchain")
    exit(1)

# GenOps imports
try:
    from genops.core.telemetry import GenOpsTelemetry
    from genops.providers.langchain import instrument_langchain
except ImportError:
    print("❌ GenOps not installed. Run: pip install genops-ai[langchain]")
    exit(1)

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_environment() -> bool:
    """Verify required environment variables are set."""
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"❌ Missing required environment variables: {missing_vars}")
        print("Set them with: export OPENAI_API_KEY=your_key_here")
        return False

    print("✅ Environment variables configured")
    return True


def create_simple_chain() -> LLMChain:
    """Create a simple LangChain for demonstration."""
    # Simple prompt template
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Write a brief, informative summary about {topic} in 2-3 sentences.",
    )

    # Initialize OpenAI LLM
    llm = OpenAI(
        temperature=0.7,
        max_tokens=150,
        model_name="gpt-3.5-turbo-instruct",  # Cost-effective for examples
    )

    # Create chain
    chain = LLMChain(llm=llm, prompt=prompt)

    return chain


def basic_chain_tracking_example():
    """Demonstrate basic chain tracking with GenOps."""
    print("🔄 Running Basic Chain Tracking Example")
    print("=" * 50)

    # Initialize GenOps LangChain adapter
    print("1. Initializing GenOps LangChain adapter...")
    adapter = instrument_langchain()

    # Create a simple chain
    print("2. Creating LangChain chain...")
    chain = create_simple_chain()

    # Track chain execution with governance attributes
    print("3. Executing chain with GenOps tracking...")

    try:
        result = adapter.instrument_chain_run(
            chain=chain,
            # Chain input
            topic="artificial intelligence",
            # Governance attributes for cost attribution and compliance
            team="examples-team",
            project="basic-chain-demo",
            environment="development",
            customer_id="example_customer_001",
            # Optional: Chain execution parameters
            # These get passed to the chain's run() method
            verbose=True,
        )

        print("✅ Chain execution successful!")
        print(f"📄 Result: {result}")

        return result

    except Exception as e:
        print(f"❌ Chain execution failed: {e}")
        logger.exception("Chain execution error")
        raise


def verify_telemetry():
    """Verify that telemetry is being captured correctly."""
    print("\n🔍 Verifying Telemetry Setup")
    print("=" * 50)

    # Check if OpenTelemetry is configured
    try:
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)

        with tracer.start_as_current_span("telemetry_test") as span:
            span.set_attribute("test.genops.verification", "success")
            print("✅ OpenTelemetry is working")

    except Exception as e:
        print(f"⚠️  OpenTelemetry issue: {e}")

    # Check GenOps telemetry
    try:
        GenOpsTelemetry()
        print("✅ GenOps telemetry initialized")

        # Verify OTLP endpoint
        otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        if otlp_endpoint:
            print(f"📡 OTLP endpoint configured: {otlp_endpoint}")
        else:
            print("⚠️  No OTLP endpoint configured (telemetry will be logged only)")

    except Exception as e:
        print(f"⚠️  GenOps telemetry issue: {e}")

    print("\n💡 To view telemetry in observability platform:")
    print("   1. Ensure OTEL_EXPORTER_OTLP_ENDPOINT is set")
    print(
        "   2. Run observability stack: docker-compose -f docker-compose.observability.yml up"
    )
    print("   3. Visit Grafana at http://localhost:3000")


def demonstrate_cost_information():
    """Show how to access cost information from chain execution."""
    print("\n💰 Cost Information Demo")
    print("=" * 50)

    from genops.providers.langchain import create_chain_cost_context

    # Use cost context to track costs explicitly
    chain = create_simple_chain()

    with create_chain_cost_context("basic_demo_chain") as cost_context:
        # Run the chain within cost tracking context
        result = chain.run(topic="machine learning")
        print(f"Chain result: {result}")

        # Get cost summary
        current_summary = cost_context.get_current_summary()
        if current_summary and current_summary.llm_calls:
            print(f"💰 Total cost so far: ${current_summary.total_cost:.4f}")
            print(
                f"🔢 Total tokens: {current_summary.total_tokens_input + current_summary.total_tokens_output}"
            )
            print(f"🏢 Providers used: {list(current_summary.unique_providers)}")

    # Get final summary after context closes
    final_summary = cost_context.get_final_summary()
    if final_summary:
        print(f"✅ Final cost: ${final_summary.total_cost:.4f}")
        print(f"⏱️  Total time: {final_summary.total_time:.2f}s")


def main():
    """Main example function."""
    print("🚀 GenOps LangChain Basic Chain Tracking Example")
    print("=" * 60)

    # Check environment
    if not setup_environment():
        return

    try:
        # Run basic tracking example
        basic_chain_tracking_example()

        # Verify telemetry setup
        verify_telemetry()

        # Demonstrate cost tracking
        demonstrate_cost_information()

        print("\n🎉 Example completed successfully!")
        print("Next steps:")
        print("   - Check your observability platform for telemetry data")
        print("   - Try the multi_provider_costs.py example for advanced cost tracking")
        print("   - Explore rag_pipeline_monitoring.py for RAG applications")

    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("Troubleshooting:")
        print("   - Verify your OpenAI API key is set correctly")
        print("   - Check that genops-ai[langchain] is installed")
        print("   - Review the error details above")


if __name__ == "__main__":
    main()
