"""Basic usage examples for GenOps AI."""

import os

from genops import enforce_policy, track, track_usage
from genops.core.policy import PolicyResult, register_policy
from genops.core.tracker import track_cost, track_evaluation


# Example 1: Function decorator for tracking
@track_usage(
    operation_name="analyze_sentiment",
    team="nlp-team",
    project="customer-feedback",
    feature="sentiment-analysis"
)
def analyze_sentiment(text: str) -> dict:
    """Analyze sentiment of text (mock implementation)."""
    # Simulate AI processing
    sentiment_score = 0.75

    # Manually record cost for this operation
    track_cost(
        cost=0.002,
        provider="openai",
        model="text-davinci-003",
        tokens_input=len(text.split()) * 1.3,
        tokens_output=10
    )

    # Record evaluation metrics
    track_evaluation(
        evaluation_name="confidence_score",
        score=sentiment_score,
        threshold=0.7,
        passed=sentiment_score > 0.7
    )

    return {
        "sentiment": "positive" if sentiment_score > 0.5 else "negative",
        "confidence": sentiment_score
    }


# Example 2: Context manager for block-level tracking
def process_documents(documents: list) -> list:
    """Process multiple documents with governance tracking."""
    results = []

    with track(
        operation_name="document_processing_batch",
        team="content-team",
        project="document-analyzer",
        customer="enterprise-client-123"
    ) as span:
        # Add custom attributes
        span.set_attribute("batch_size", len(documents))

        total_cost = 0
        for _i, doc in enumerate(documents):
            doc_result = process_single_document(doc)
            results.append(doc_result)
            total_cost += 0.005  # Mock cost per document

        # Record batch cost
        track_cost(
            cost=total_cost,
            provider="anthropic",
            model="claude-3-sonnet",
            batch_size=len(documents)
        )

        # Record batch evaluation
        track_evaluation(
            evaluation_name="batch_success_rate",
            score=len(results) / len(documents),
            threshold=0.95,
            passed=len(results) == len(documents)
        )

    return results


def process_single_document(document: str) -> dict:
    """Process a single document (mock implementation)."""
    return {
        "processed": True,
        "word_count": len(document.split()),
        "summary": document[:100] + "..." if len(document) > 100 else document
    }


# Example 3: Policy enforcement
def setup_governance_policies():
    """Set up governance policies for AI operations."""

    # Register cost limit policy
    register_policy(
        name="cost_limit",
        description="Limit per-operation costs to prevent runaway spending",
        enforcement_level=PolicyResult.BLOCKED,
        max_cost=1.00  # $1 per operation
    )

    # Register content filtering policy
    register_policy(
        name="content_filter",
        description="Block operations with inappropriate content",
        enforcement_level=PolicyResult.BLOCKED,
        blocked_patterns=["violence", "hate", "explicit"]
    )

    # Register team access policy
    register_policy(
        name="team_access",
        description="Restrict model access to authorized teams",
        enforcement_level=PolicyResult.WARNING,
        allowed_teams=["nlp-team", "content-team", "research-team"]
    )


@enforce_policy(["cost_limit", "content_filter"])
@track_usage(
    operation_name="generate_content",
    team="content-team",
    project="blog-generator"
)
def generate_content(prompt: str) -> str:
    """Generate content with policy enforcement."""
    # This function will be checked against policies before execution

    # Simulate content generation cost
    estimated_cost = len(prompt) * 0.0001  # Mock cost calculation

    track_cost(
        cost=estimated_cost,
        provider="openai",
        model="gpt-4",
        tokens_input=len(prompt.split()) * 1.3,
        tokens_output=100
    )

    # Mock generated content
    return f"Generated content based on: {prompt[:50]}..."


# Example 4: Provider instrumentation
def example_with_openai():
    """Example using OpenAI with GenOps instrumentation."""
    try:
        from genops.providers import instrument_openai

        # Option 1: Instrument existing client
        # openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # genops_client = instrument_openai(openai_client)

        # Option 2: Create instrumented client directly
        genops_client = instrument_openai(api_key=os.getenv("OPENAI_API_KEY"))

        # Use normally - telemetry is automatic
        with track(
            operation_name="openai_chat_completion",
            team="ai-team",
            project="chatbot",
            customer="demo-user"
        ):
            response = genops_client.chat_completions_create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": "Hello! How are you?"}
                ],
                max_tokens=100
            )

            return response.choices[0].message.content

    except ImportError:
        print("OpenAI not available. Install with: pip install openai")
        return None


def example_with_anthropic():
    """Example using Anthropic with GenOps instrumentation."""
    try:
        from genops.providers import instrument_anthropic

        # Create instrumented client
        genops_client = instrument_anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        # Use normally - telemetry is automatic
        with track(
            operation_name="anthropic_message",
            team="ai-team",
            project="assistant",
            customer="demo-user"
        ):
            response = genops_client.messages_create(
                model="claude-3-sonnet",
                messages=[
                    {"role": "user", "content": "Hello! How are you?"}
                ],
                max_tokens=100
            )

            return response.content[0].text

    except ImportError:
        print("Anthropic not available. Install with: pip install anthropic")
        return None


def main():
    """Run examples."""
    print("GenOps AI Basic Usage Examples")
    print("=" * 40)

    # Set up policies
    print("1. Setting up governance policies...")
    setup_governance_policies()

    # Example 1: Function decorator
    print("\n2. Function decorator example...")
    result1 = analyze_sentiment("This is a great product! I love it.")
    print(f"Sentiment analysis result: {result1}")

    # Example 2: Context manager
    print("\n3. Context manager example...")
    docs = [
        "Document 1 content here",
        "Document 2 with different content",
        "Document 3 with more text"
    ]
    result2 = process_documents(docs)
    print(f"Processed {len(result2)} documents")

    # Example 3: Policy enforcement
    print("\n4. Policy enforcement example...")
    try:
        result3 = generate_content("Write a blog post about AI governance")
        print(f"Generated content: {result3}")
    except Exception as e:
        print(f"Policy violation: {e}")

    # Example 4: Provider instrumentation
    print("\n5. Provider instrumentation examples...")

    if os.getenv("OPENAI_API_KEY"):
        openai_result = example_with_openai()
        if openai_result:
            print(f"OpenAI result: {openai_result}")
    else:
        print("OPENAI_API_KEY not set, skipping OpenAI example")

    if os.getenv("ANTHROPIC_API_KEY"):
        anthropic_result = example_with_anthropic()
        if anthropic_result:
            print(f"Anthropic result: {anthropic_result}")
    else:
        print("ANTHROPIC_API_KEY not set, skipping Anthropic example")

    print("\n✓ Examples completed!")
    print("Check your OpenTelemetry collector/exporter for telemetry data.")


if __name__ == "__main__":
    main()
