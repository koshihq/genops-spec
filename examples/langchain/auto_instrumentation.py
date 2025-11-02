"""
Example: Auto-Instrumentation with LangChain
Demonstrates: Zero-code setup with automatic telemetry capture
Use case: Minimal setup for existing LangChain applications
"""

import logging
import os

# Core LangChain imports
try:
    from langchain.chains import LLMChain, SimpleSequentialChain
    from langchain.chains.summarize import load_summarize_chain
    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate
    from langchain.schema import Document
    from langchain.text_splitter import CharacterTextSplitter
except ImportError:
    print("❌ LangChain not installed. Run: pip install langchain")
    exit(1)

# GenOps imports - AUTO INSTRUMENTATION
try:
    # This is the key: auto-instrument enables zero-code telemetry
    from genops import auto_instrument
    from genops.core.context import set_governance_context
    from genops.providers.langchain import get_cost_aggregator
except ImportError:
    print("❌ GenOps not installed. Run: pip install genops-ai[langchain]")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_environment() -> bool:
    """Verify environment is configured for auto-instrumentation."""
    required_vars = ["OPENAI_API_KEY"]
    optional_vars = {
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317",
        "OTEL_SERVICE_NAME": "langchain-auto-instrumentation-example",
        "GENOPS_ENVIRONMENT": "development"
    }

    missing_required = [var for var in required_vars if not os.getenv(var)]
    if missing_required:
        print(f"❌ Missing required variables: {missing_required}")
        return False

    # Set optional environment variables if not present
    for var, default in optional_vars.items():
        if not os.getenv(var):
            os.environ[var] = default
            print(f"🔧 Set {var}={default}")

    print("✅ Environment configured for auto-instrumentation")
    return True


def enable_auto_instrumentation():
    """Enable GenOps auto-instrumentation for all supported frameworks."""
    print("🔄 Enabling auto-instrumentation...")

    # This single call enables automatic telemetry for:
    # - LangChain chains, agents, and tools
    # - OpenAI API calls
    # - Anthropic API calls
    # - Vector store operations
    # - And more!
    auto_instrument()

    print("✅ Auto-instrumentation enabled!")
    print("   All LangChain operations will now automatically capture:")
    print("   - Chain execution telemetry")
    print("   - Cost tracking across providers")
    print("   - Performance metrics")
    print("   - Error tracking")


def set_governance_attributes():
    """Set global governance attributes for cost attribution."""
    print("🏢 Setting governance context...")

    # Set governance context that will apply to all operations
    set_governance_context({
        "team": "ai-automation",
        "project": "auto-instrumentation-demo",
        "environment": "development",
        "customer_id": "internal_testing",
        "deployment": "local",
        "cost_center": "engineering"
    })

    print("✅ Governance context set - all operations will be attributed properly")


def create_sequential_chain() -> SimpleSequentialChain:
    """Create a multi-step chain that will demonstrate auto-instrumentation."""

    # Step 1: Generate a topic outline
    outline_prompt = PromptTemplate(
        input_variables=["topic"],
        template="""Create a detailed outline for an article about {topic}.

        The outline should have:
        - An engaging introduction
        - 3-4 main sections with subsections
        - A conclusion

        Outline:"""
    )

    outline_chain = LLMChain(
        llm=OpenAI(temperature=0.7, max_tokens=300),
        prompt=outline_prompt,
        output_key="outline"
    )

    # Step 2: Write the article from the outline
    article_prompt = PromptTemplate(
        input_variables=["outline"],
        template="""Based on this outline:

        {outline}

        Write a comprehensive, well-structured article. Make it informative and engaging.

        Article:"""
    )

    article_chain = LLMChain(
        llm=OpenAI(temperature=0.6, max_tokens=800),
        prompt=article_prompt,
        output_key="article"
    )

    # Create sequential chain
    sequential_chain = SimpleSequentialChain(
        chains=[outline_chain, article_chain],
        verbose=True  # This will show the intermediate steps
    )

    return sequential_chain


def demonstrate_automatic_tracking():
    """Show how auto-instrumentation works with regular LangChain code."""
    print("\n📝 Running Sequential Chain with Auto-Instrumentation")
    print("=" * 60)

    # Create and run chain - NO GENOPS CODE NEEDED!
    # Auto-instrumentation captures everything automatically
    chain = create_sequential_chain()

    try:
        # This is just normal LangChain code - telemetry happens automatically
        result = chain.run("artificial intelligence in healthcare")

        print("✅ Sequential chain completed successfully!")
        print(f"📄 Final result length: {len(result)} characters")
        print(f"📄 Result preview: {result[:200]}...")

        return result

    except Exception as e:
        print(f"❌ Chain execution failed: {e}")
        logger.exception("Sequential chain error")
        raise


def demonstrate_cost_visibility():
    """Show how to access cost information with auto-instrumentation."""
    print("\n💰 Accessing Cost Information")
    print("=" * 60)

    # Get the global cost aggregator
    cost_aggregator = get_cost_aggregator()

    # Check active chains
    active_chains = cost_aggregator.get_active_chains()
    print(f"📊 Active chains being tracked: {len(active_chains)}")

    # Run a simple chain to generate some cost data
    simple_chain = LLMChain(
        llm=OpenAI(temperature=0.5, max_tokens=100),
        prompt=PromptTemplate(
            input_variables=["question"],
            template="Provide a concise answer to: {question}"
        )
    )

    # This will automatically be tracked due to auto-instrumentation
    result = simple_chain.run("What are the benefits of AI governance?")
    print(f"🤖 Chain result: {result}")

    # Check updated active chains
    updated_chains = cost_aggregator.get_active_chains()
    print(f"📊 Updated active chains: {len(updated_chains)}")


def demonstrate_different_chain_types():
    """Show auto-instrumentation working with different LangChain components."""
    print("\n🔗 Testing Different Chain Types")
    print("=" * 60)

    # 1. Simple LLMChain
    print("1. Testing LLMChain...")
    llm_chain = LLMChain(
        llm=OpenAI(temperature=0.3, max_tokens=50),
        prompt=PromptTemplate(
            input_variables=["task"],
            template="Complete this task briefly: {task}"
        )
    )
    result1 = llm_chain.run("Explain quantum computing")
    print(f"   ✅ LLMChain result: {result1[:100]}...")

    # 2. Summarization chain
    print("2. Testing Summarization chain...")

    # Create some sample documents
    sample_text = """
    Artificial Intelligence (AI) has become increasingly important in modern healthcare systems.
    It offers numerous benefits including improved diagnostic accuracy, personalized treatment plans,
    and more efficient administrative processes. However, AI implementation also presents challenges
    such as data privacy concerns, the need for regulatory compliance, and ensuring equitable access
    to AI-enhanced healthcare services across different populations.
    """

    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(sample_text)]

    summarize_chain = load_summarize_chain(
        OpenAI(temperature=0.2, max_tokens=100),
        chain_type="map_reduce"
    )

    summary = summarize_chain.run(docs)
    print(f"   ✅ Summarization result: {summary}")

    print("✅ All chain types automatically instrumented!")


def check_telemetry_export():
    """Verify that telemetry is being exported correctly."""
    print("\n📡 Checking Telemetry Export")
    print("=" * 60)

    # Check OpenTelemetry configuration
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    service_name = os.getenv("OTEL_SERVICE_NAME")

    print(f"📡 OTLP Endpoint: {otlp_endpoint}")
    print(f"🏷️  Service Name: {service_name}")

    # Verify OpenTelemetry is working
    try:
        from opentelemetry import trace
        tracer = trace.get_tracer(__name__)

        with tracer.start_as_current_span("auto_instrumentation_test") as span:
            span.set_attribute("genops.test.auto_instrumentation", True)
            span.set_attribute("genops.example.name", "auto_instrumentation")
            print("✅ OpenTelemetry span created successfully")

    except Exception as e:
        print(f"⚠️  OpenTelemetry issue: {e}")

    print("\n💡 Telemetry Data Locations:")
    print(f"   - OTLP Exporter: {otlp_endpoint}")
    if "localhost" in str(otlp_endpoint):
        print("   - Grafana Dashboard: http://localhost:3000")
        print("   - Jaeger Traces: http://localhost:16686")
    print("   - Console logs: Check your application logs")


def main():
    """Main example demonstrating auto-instrumentation."""
    print("🚀 GenOps LangChain Auto-Instrumentation Example")
    print("=" * 70)

    # Setup environment
    if not setup_environment():
        return

    try:
        # Enable auto-instrumentation (this is the key step!)
        enable_auto_instrumentation()

        # Set governance context for cost attribution
        set_governance_attributes()

        # Now run normal LangChain code - telemetry is automatic!
        demonstrate_automatic_tracking()

        # Show cost visibility
        demonstrate_cost_visibility()

        # Test different chain types
        demonstrate_different_chain_types()

        # Verify telemetry export
        check_telemetry_export()

        print("\n🎉 Auto-instrumentation example completed!")
        print("\n🔑 Key Takeaways:")
        print("   ✅ Single auto_instrument() call enables telemetry for everything")
        print("   ✅ No changes needed to existing LangChain code")
        print("   ✅ Automatic cost tracking across all LLM providers")
        print("   ✅ Governance attributes applied to all operations")
        print("   ✅ Performance and error tracking included")

        print("\n📊 What was automatically captured:")
        print("   - Chain execution times and token usage")
        print("   - Cost attribution by team, project, and customer")
        print("   - LLM provider usage (OpenAI, Anthropic, etc.)")
        print("   - Error tracking and exception handling")
        print("   - Performance metrics for each operation")

        print("\n🎯 Next Steps:")
        print("   - Check your observability dashboard for telemetry data")
        print("   - Try multi_provider_costs.py for advanced cost scenarios")
        print("   - Explore rag_pipeline_monitoring.py for RAG applications")

    except Exception as e:
        print(f"\n❌ Auto-instrumentation example failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("   - Ensure OpenAI API key is set: export OPENAI_API_KEY=your_key")
        print("   - Verify GenOps installation: pip install genops-ai[langchain]")
        print("   - Check OpenTelemetry configuration")
        logger.exception("Example execution error")


if __name__ == "__main__":
    main()
