#!/usr/bin/env python3
"""
Basic Anyscale Completion Example with GenOps Tracking

This example demonstrates:
- Setting up GenOps Anyscale adapter
- Making a basic chat completion request
- Tracking costs and token usage
- Adding governance attributes

Prerequisites:
- export ANYSCALE_API_KEY='your-api-key-here'
- pip install genops-ai
"""

import os

from genops.providers.anyscale import calculate_completion_cost, instrument_anyscale


def main():
    # Check API key
    if not os.getenv("ANYSCALE_API_KEY"):
        print("❌ ERROR: ANYSCALE_API_KEY environment variable not set")
        print("Fix: export ANYSCALE_API_KEY='your-api-key-here'")
        print("Get your key from: https://console.anyscale.com/credentials")
        return

    print("=" * 70)
    print("GenOps Anyscale - Basic Completion Example")
    print("=" * 70 + "\n")

    # Initialize GenOps Anyscale adapter with governance defaults
    adapter = instrument_anyscale(
        team="examples-team", project="basic-completion", environment="development"
    )

    print("✅ GenOps Anyscale adapter initialized\n")

    # Example 1: Simple completion
    print("📝 Example 1: Simple Chat Completion")
    print("-" * 70)

    response = adapter.completion_create(
        model="meta-llama/Llama-2-70b-chat-hf",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        temperature=0.7,
        max_tokens=100,
    )

    # Extract response
    message = response["choices"][0]["message"]["content"]
    usage = response["usage"]

    print("Model: meta-llama/Llama-2-70b-chat-hf")
    print(f"Response: {message}\n")

    # Show token usage
    print("📊 Token Usage:")
    print(f"   Input tokens: {usage['prompt_tokens']}")
    print(f"   Output tokens: {usage['completion_tokens']}")
    print(f"   Total tokens: {usage['total_tokens']}\n")

    # Calculate and show cost
    cost = calculate_completion_cost(
        model="meta-llama/Llama-2-70b-chat-hf",
        input_tokens=usage["prompt_tokens"],
        output_tokens=usage["completion_tokens"],
    )
    print(f"💰 Cost: ${cost:.6f} (at $1/M token rate)\n")

    # Example 2: Completion with customer attribution
    print("📝 Example 2: Completion with Customer Attribution")
    print("-" * 70)

    response2 = adapter.completion_create(
        model="meta-llama/Llama-2-70b-chat-hf",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum computing in one sentence."},
        ],
        temperature=0.7,
        max_tokens=100,
        customer_id="customer-123",  # Governance attribute
        feature="chat-completion",  # Governance attribute
    )

    message2 = response2["choices"][0]["message"]["content"]
    usage2 = response2["usage"]
    cost2 = calculate_completion_cost(
        model="meta-llama/Llama-2-70b-chat-hf",
        input_tokens=usage2["prompt_tokens"],
        output_tokens=usage2["completion_tokens"],
    )

    print("Customer: customer-123")
    print("Feature: chat-completion")
    print(f"Response: {message2}\n")
    print(f"💰 Cost: ${cost2:.6f}\n")

    # Example 3: Using smaller model for cost optimization
    print("📝 Example 3: Cost Optimization with Smaller Model")
    print("-" * 70)

    response3 = adapter.completion_create(
        model="meta-llama/Llama-2-7b-chat-hf",  # Smaller model
        messages=[{"role": "user", "content": "What is 2+2?"}],
        temperature=0.7,
        max_tokens=50,
    )

    message3 = response3["choices"][0]["message"]["content"]
    usage3 = response3["usage"]
    cost3 = calculate_completion_cost(
        model="meta-llama/Llama-2-7b-chat-hf",
        input_tokens=usage3["prompt_tokens"],
        output_tokens=usage3["completion_tokens"],
    )

    print("Model: meta-llama/Llama-2-7b-chat-hf (smaller, cheaper)")
    print(f"Response: {message3}\n")
    print(f"💰 Cost: ${cost3:.6f} (vs ${cost:.6f} for 70B model)")

    savings_pct = ((cost - cost3) / cost) * 100 if cost > 0 else 0
    print(f"💡 Savings: {savings_pct:.1f}% by using smaller model\n")

    # Summary
    print("=" * 70)
    print("✅ Examples completed successfully!")
    print("=" * 70)
    print("\n🎯 Key Takeaways:")
    print("   ✅ GenOps automatically tracks token usage and costs")
    print(
        "   ✅ Governance attributes (team, customer, feature) enable cost attribution"
    )
    print("   ✅ Model selection significantly impacts costs (70B vs 7B)")
    print("   ✅ All requests generate OpenTelemetry traces for observability\n")

    print("📚 Next Steps:")
    print("   - Try different models: Mistral, CodeLlama, etc.")
    print("   - Add more governance attributes for fine-grained tracking")
    print("   - Integrate with your observability stack (Datadog, Honeycomb, etc.)")
    print("   - See docs/anyscale-quickstart.md for more examples\n")


if __name__ == "__main__":
    main()
