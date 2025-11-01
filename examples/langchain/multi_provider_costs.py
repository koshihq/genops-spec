"""
Example: Multi-Provider Cost Tracking
Demonstrates: Cost aggregation across multiple LLM providers (OpenAI, Anthropic, Cohere)
Use case: Applications using multiple LLM providers for different tasks
"""

import os
import logging
from typing import Dict, List, Tuple
from decimal import Decimal

# Core LangChain imports
try:
    from langchain.chains import LLMChain
    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate
    from langchain.schema import BaseLanguageModel
except ImportError:
    print("❌ LangChain not installed. Run: pip install langchain")
    exit(1)

# Try to import additional providers
try:
    from langchain.llms import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("⚠️  Anthropic not available. Install with: pip install anthropic")

try:
    from langchain.llms import Cohere  
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False
    print("⚠️  Cohere not available. Install with: pip install cohere")

# GenOps imports
try:
    from genops.providers.langchain import (
        instrument_langchain,
        create_chain_cost_context,
        get_cost_aggregator
    )
    from genops.core.telemetry import GenOpsTelemetry
except ImportError:
    print("❌ GenOps not installed. Run: pip install genops-ai[langchain]")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_provider_availability() -> Dict[str, bool]:
    """Check which LLM providers are available and configured."""
    providers = {
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "anthropic": ANTHROPIC_AVAILABLE and bool(os.getenv("ANTHROPIC_API_KEY")),
        "cohere": COHERE_AVAILABLE and bool(os.getenv("COHERE_API_KEY"))
    }
    
    print("🔍 Provider Availability Check:")
    for provider, available in providers.items():
        status = "✅" if available else "❌"
        print(f"   {status} {provider.title()}")
    
    available_count = sum(providers.values())
    if available_count < 2:
        print("\n⚠️  This example works best with multiple providers.")
        print("   Set additional API keys to see full multi-provider cost tracking:")
        print("   export ANTHROPIC_API_KEY=your_key")
        print("   export COHERE_API_KEY=your_key")
    
    return providers


def create_provider_chains(available_providers: Dict[str, bool]) -> Dict[str, LLMChain]:
    """Create chains for each available provider with different use cases."""
    chains = {}
    
    # OpenAI - Good for general text generation
    if available_providers["openai"]:
        openai_llm = OpenAI(
            temperature=0.7,
            max_tokens=200,
            model_name="gpt-3.5-turbo-instruct"
        )
        
        openai_chain = LLMChain(
            llm=openai_llm,
            prompt=PromptTemplate(
                input_variables=["task", "context"],
                template="Task: {task}\nContext: {context}\n\nResponse:"
            ),
            output_key="openai_response"
        )
        chains["openai"] = openai_chain
        print("✅ Created OpenAI chain (general text generation)")
    
    # Anthropic - Good for analysis and reasoning
    if available_providers["anthropic"]:
        anthropic_llm = Anthropic(
            temperature=0.3,
            max_tokens_to_sample=300,
            model="claude-instant-1"
        )
        
        anthropic_chain = LLMChain(
            llm=anthropic_llm,
            prompt=PromptTemplate(
                input_variables=["content"],
                template="Analyze the following content and provide insights:\n\n{content}\n\nAnalysis:"
            ),
            output_key="anthropic_analysis"
        )
        chains["anthropic"] = anthropic_chain
        print("✅ Created Anthropic chain (analysis and reasoning)")
    
    # Cohere - Good for summarization
    if available_providers["cohere"]:
        cohere_llm = Cohere(
            temperature=0.1,
            max_tokens=150,
            model="command"
        )
        
        cohere_chain = LLMChain(
            llm=cohere_llm,
            prompt=PromptTemplate(
                input_variables=["text"],
                template="Summarize the following text concisely:\n\n{text}\n\nSummary:"
            ),
            output_key="cohere_summary"
        )
        chains["cohere"] = cohere_chain
        print("✅ Created Cohere chain (summarization)")
    
    return chains


def demonstrate_individual_provider_costs(chains: Dict[str, LLMChain]):
    """Show cost tracking for individual providers."""
    print("\n💰 Individual Provider Cost Tracking")
    print("=" * 50)
    
    adapter = instrument_langchain()
    individual_costs = {}
    
    sample_content = """
    Artificial Intelligence is transforming healthcare by enabling more accurate diagnostics,
    personalized treatment plans, and efficient administrative processes. However, the implementation
    of AI in healthcare also raises important questions about data privacy, regulatory compliance,
    and ensuring equitable access to AI-enhanced healthcare services.
    """
    
    # Test each provider individually
    for provider_name, chain in chains.items():
        print(f"\n🔄 Testing {provider_name.title()} provider...")
        
        try:
            if provider_name == "openai":
                result = adapter.instrument_chain_run(
                    chain,
                    task="Generate a creative title",
                    context=sample_content,
                    team="content-team",
                    project="multi-provider-demo",
                    provider_test=provider_name
                )
            elif provider_name == "anthropic":
                result = adapter.instrument_chain_run(
                    chain,
                    content=sample_content,
                    team="analysis-team", 
                    project="multi-provider-demo",
                    provider_test=provider_name
                )
            elif provider_name == "cohere":
                result = adapter.instrument_chain_run(
                    chain,
                    text=sample_content,
                    team="summarization-team",
                    project="multi-provider-demo",
                    provider_test=provider_name
                )
            
            print(f"   ✅ {provider_name.title()} result: {result[:100]}...")
            individual_costs[provider_name] = "tracked"
            
        except Exception as e:
            print(f"   ❌ {provider_name.title()} failed: {e}")
            individual_costs[provider_name] = "failed"
    
    return individual_costs


def demonstrate_multi_provider_workflow(chains: Dict[str, LLMChain]):
    """Demonstrate a workflow using multiple providers with cost aggregation."""
    print("\n🔗 Multi-Provider Workflow with Cost Aggregation")
    print("=" * 60)
    
    # Use context manager to aggregate costs across providers
    with create_chain_cost_context("multi_provider_workflow") as cost_context:
        workflow_results = {}
        
        print("🔄 Executing multi-step workflow across providers...")
        
        sample_document = """
        The future of artificial intelligence in business operations looks promising.
        Companies are increasingly adopting AI solutions for automation, decision-making,
        and customer service. Key trends include natural language processing for customer
        interactions, machine learning for predictive analytics, and computer vision for
        quality control in manufacturing. However, successful AI adoption requires
        careful planning, appropriate infrastructure, and ongoing maintenance.
        """
        
        # Step 1: Use OpenAI for initial processing (if available)
        if "openai" in chains:
            print("   Step 1: OpenAI - Initial content generation...")
            try:
                openai_result = chains["openai"].run(
                    task="Create 3 discussion questions",
                    context=sample_document
                )
                workflow_results["questions"] = openai_result
                print(f"   ✅ Generated questions: {openai_result[:100]}...")
            except Exception as e:
                print(f"   ❌ OpenAI step failed: {e}")
        
        # Step 2: Use Anthropic for analysis (if available)  
        if "anthropic" in chains:
            print("   Step 2: Anthropic - Content analysis...")
            try:
                anthropic_result = chains["anthropic"].run(content=sample_document)
                workflow_results["analysis"] = anthropic_result
                print(f"   ✅ Analysis complete: {anthropic_result[:100]}...")
            except Exception as e:
                print(f"   ❌ Anthropic step failed: {e}")
        
        # Step 3: Use Cohere for summarization (if available)
        if "cohere" in chains:
            print("   Step 3: Cohere - Content summarization...")
            try:
                cohere_result = chains["cohere"].run(text=sample_document)
                workflow_results["summary"] = cohere_result
                print(f"   ✅ Summary complete: {cohere_result[:100]}...")
            except Exception as e:
                print(f"   ❌ Cohere step failed: {e}")
        
        # Record additional custom cost (e.g., processing time, storage)
        cost_context.record_generation_cost(0.001)  # $0.001 for processing
        
        print(f"\n📊 Workflow completed with {len(workflow_results)} successful steps")
    
    # Access final cost summary
    final_summary = cost_context.get_final_summary()
    if final_summary:
        print_cost_summary(final_summary, "Multi-Provider Workflow")
    
    return workflow_results


def demonstrate_customer_cost_attribution(chains: Dict[str, LLMChain]):
    """Show how to track costs per customer across multiple providers."""
    print("\n👥 Customer Cost Attribution Demo")
    print("=" * 50)
    
    customers = ["customer_001", "customer_002", "customer_003"]
    customer_costs = {}
    
    for customer_id in customers:
        print(f"\n🔄 Processing requests for {customer_id}...")
        
        with create_chain_cost_context(f"customer_{customer_id}") as cost_context:
            
            # Each customer gets processed by available providers
            for provider_name, chain in chains.items():
                try:
                    if provider_name == "openai":
                        result = chain.run(
                            task=f"Create a personalized greeting",
                            context=f"Customer {customer_id} preferences"
                        )
                    elif provider_name == "anthropic":
                        result = chain.run(
                            content=f"Customer {customer_id} behavior analysis request"
                        )
                    elif provider_name == "cohere":
                        result = chain.run(
                            text=f"Customer {customer_id} interaction history summary"
                        )
                    
                    print(f"   ✅ {provider_name.title()}: {result[:50]}...")
                    
                except Exception as e:
                    print(f"   ❌ {provider_name.title()} failed: {e}")
        
        # Store customer cost summary
        final_summary = cost_context.get_final_summary()
        if final_summary:
            customer_costs[customer_id] = {
                "total_cost": final_summary.total_cost,
                "providers": list(final_summary.unique_providers),
                "models": list(final_summary.unique_models),
                "tokens": final_summary.total_tokens_input + final_summary.total_tokens_output
            }
    
    # Print customer cost breakdown
    print("\n💳 Customer Cost Breakdown:")
    total_all_customers = 0
    for customer_id, costs in customer_costs.items():
        print(f"   {customer_id}:")
        print(f"     💰 Cost: ${costs['total_cost']:.4f}")
        print(f"     🏢 Providers: {costs['providers']}")
        print(f"     🤖 Models: {costs['models']}")
        print(f"     🔢 Tokens: {costs['tokens']}")
        total_all_customers += costs['total_cost']
    
    print(f"\n💰 Total across all customers: ${total_all_customers:.4f}")
    
    return customer_costs


def print_cost_summary(summary, workflow_name: str):
    """Helper function to print cost summary in a readable format."""
    print(f"\n📊 {workflow_name} - Cost Summary")
    print("-" * 40)
    print(f"💰 Total Cost: ${summary.total_cost:.4f}")
    print(f"💵 Currency: {summary.currency}")
    print(f"⏱️  Total Time: {summary.total_time:.2f}s")
    
    if summary.llm_calls:
        print(f"🔗 LLM Calls: {len(summary.llm_calls)}")
        print(f"🏢 Providers: {list(summary.unique_providers)}")
        print(f"🤖 Models: {list(summary.unique_models)}")
        print(f"📝 Input Tokens: {summary.total_tokens_input:,}")
        print(f"📄 Output Tokens: {summary.total_tokens_output:,}")
        
        print("\n💰 Cost Breakdown by Provider:")
        for provider, cost in summary.cost_by_provider.items():
            print(f"   {provider}: ${cost:.4f}")
        
        print("\n🤖 Cost Breakdown by Model:")
        for model, cost in summary.cost_by_model.items():
            print(f"   {model}: ${cost:.4f}")
    
    if summary.generation_cost > 0:
        print(f"⚙️  Processing Cost: ${summary.generation_cost:.4f}")


def generate_cost_report(individual_costs, workflow_results, customer_costs):
    """Generate a comprehensive cost report."""
    print("\n📈 Multi-Provider Cost Report")
    print("=" * 60)
    
    print("✅ Successfully demonstrated:")
    print("   - Individual provider cost tracking")
    print("   - Multi-provider workflow cost aggregation") 
    print("   - Customer-specific cost attribution")
    print("   - Real-time cost breakdowns by provider and model")
    
    print("\n🎯 Key Benefits of Multi-Provider Cost Tracking:")
    print("   ✅ Unified cost visibility across all LLM providers")
    print("   ✅ Automatic cost attribution by team, project, customer")
    print("   ✅ Real-time cost aggregation within operations")
    print("   ✅ Detailed breakdowns for billing and budgeting")
    print("   ✅ No code changes needed for existing LangChain applications")
    
    total_customers_processed = len(customer_costs)
    total_providers_used = len([p for p, status in individual_costs.items() if status == "tracked"])
    
    print(f"\n📊 This Demo Statistics:")
    print(f"   🏢 Providers Used: {total_providers_used}")
    print(f"   👥 Customers Processed: {total_customers_processed}")
    print(f"   🔗 Workflow Steps: {len(workflow_results)}")


def main():
    """Main example demonstrating multi-provider cost tracking."""
    print("🚀 GenOps LangChain Multi-Provider Cost Tracking")
    print("=" * 70)
    
    try:
        # Check provider availability
        available_providers = check_provider_availability()
        
        if not any(available_providers.values()):
            print("❌ No LLM providers configured. Please set API keys.")
            return
        
        # Create provider chains
        chains = create_provider_chains(available_providers)
        
        if not chains:
            print("❌ No chains created. Check provider configuration.")
            return
        
        # Demonstrate individual provider costs
        individual_costs = demonstrate_individual_provider_costs(chains)
        
        # Demonstrate multi-provider workflow
        workflow_results = demonstrate_multi_provider_workflow(chains)
        
        # Demonstrate customer cost attribution
        customer_costs = demonstrate_customer_cost_attribution(chains)
        
        # Generate final report
        generate_cost_report(individual_costs, workflow_results, customer_costs)
        
        print("\n🎉 Multi-provider cost tracking demo completed!")
        print("\n🎯 Next Steps:")
        print("   - Set up more provider API keys to see full multi-provider benefits")
        print("   - Check your observability dashboard for detailed cost telemetry")
        print("   - Try rag_pipeline_monitoring.py for RAG-specific cost tracking")
        print("   - Explore budget_monitoring.py for cost alerting and limits")
        
    except Exception as e:
        print(f"\n❌ Multi-provider demo failed: {e}")
        print("\n🔧 Troubleshooting:")
        print("   - Ensure at least one LLM provider API key is set")
        print("   - Verify GenOps installation: pip install genops-ai[langchain]") 
        print("   - Check provider-specific dependencies")
        logger.exception("Multi-provider demo error")


if __name__ == "__main__":
    main()