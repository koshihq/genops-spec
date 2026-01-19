#!/usr/bin/env python3
"""
Multi-Model Cost Comparison - 15 Minute Tutorial

Learn how to optimize costs by comparing models.
Demonstrates:
- Cost comparison across Llama-2 70B, 13B, and 7B models
- Performance vs cost trade-offs
- Automatic cost calculation
- Model selection guidance

Prerequisites:
- export ANYSCALE_API_KEY='your-api-key'
- pip install genops-ai
"""

import os
from genops.providers.anyscale import (
    instrument_anyscale,
    calculate_completion_cost,
    get_model_pricing
)

# Check API key
if not os.getenv("ANYSCALE_API_KEY"):
    print("❌ ERROR: ANYSCALE_API_KEY not set")
    exit(1)

print("=" * 70)
print("GenOps Anyscale - Multi-Model Cost Comparison")
print("=" * 70 + "\n")

# Create adapter
adapter = instrument_anyscale(
    team="cost-optimization",
    project="model-comparison"
)

# Test prompt
test_prompt = """
Analyze this business scenario and provide recommendations:
A startup is deciding between building in-house ML infrastructure
or using managed services. What factors should they consider?
"""

# Models to compare
models = [
    ("meta-llama/Llama-2-70b-chat-hf", "Llama-2 70B (Most Capable)"),
    ("meta-llama/Llama-2-13b-chat-hf", "Llama-2 13B (Balanced)"),
    ("meta-llama/Llama-2-7b-chat-hf", "Llama-2 7B (Most Efficient)"),
]

print("Testing the same prompt across three models...\n")

results = []

for model_id, model_name in models:
    print(f"📊 Testing: {model_name}")
    print("-" * 70)

    # Get pricing info
    pricing = get_model_pricing(model_id)
    print(f"Pricing: ${pricing.input_cost_per_million}/M input, "
          f"${pricing.output_cost_per_million}/M output")

    # Make request
    response = adapter.completion_create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are a helpful business consultant."},
            {"role": "user", "content": test_prompt}
        ],
        temperature=0.7,
        max_tokens=200
    )

    # Extract results
    content = response['choices'][0]['message']['content']
    usage = response['usage']

    # Calculate cost
    cost = calculate_completion_cost(
        model=model_id,
        input_tokens=usage['prompt_tokens'],
        output_tokens=usage['completion_tokens']
    )

    print(f"Response length: {len(content)} characters")
    print(f"Tokens used: {usage['total_tokens']} "
          f"({usage['prompt_tokens']} in, {usage['completion_tokens']} out)")
    print(f"Cost: ${cost:.6f}\n")

    results.append({
        'name': model_name,
        'model_id': model_id,
        'content': content,
        'tokens': usage['total_tokens'],
        'cost': cost
    })

# Compare results
print("=" * 70)
print("COST COMPARISON SUMMARY")
print("=" * 70 + "\n")

# Sort by cost (descending)
results.sort(key=lambda x: x['cost'], reverse=True)

most_expensive = results[0]
cheapest = results[-1]

for i, result in enumerate(results, 1):
    savings_vs_expensive = (
        (most_expensive['cost'] - result['cost']) / most_expensive['cost'] * 100
        if result != most_expensive else 0
    )

    print(f"{i}. {result['name']}")
    print(f"   Cost: ${result['cost']:.6f}")
    if savings_vs_expensive > 0:
        print(f"   Savings: {savings_vs_expensive:.1f}% cheaper than {most_expensive['name']}")
    print()

# Calculate total savings
print("💡 INSIGHTS:")
print(f"   • Most expensive: {most_expensive['name']} (${most_expensive['cost']:.6f})")
print(f"   • Most efficient: {cheapest['name']} (${cheapest['cost']:.6f})")

savings_amount = most_expensive['cost'] - cheapest['cost']
savings_pct = (savings_amount / most_expensive['cost']) * 100

print(f"   • Potential savings: {savings_pct:.1f}% (${savings_amount:.6f} per request)")
print()

# Extrapolate to scale
print("📈 AT SCALE:")
requests_per_month = [1000, 10000, 100000]
for req_count in requests_per_month:
    expensive_monthly = most_expensive['cost'] * req_count
    cheap_monthly = cheapest['cost'] * req_count
    monthly_savings = expensive_monthly - cheap_monthly

    print(f"   {req_count:,} requests/month:")
    print(f"      {most_expensive['name']}: ${expensive_monthly:.2f}")
    print(f"      {cheapest['name']}: ${cheap_monthly:.2f}")
    print(f"      💰 Monthly savings: ${monthly_savings:.2f}")

print()
print("=" * 70)
print("✅ Cost comparison complete!")
print("=" * 70)

print("\n🎯 RECOMMENDATIONS:")
print("   • Use Llama-2-7B for: simple tasks, classification, routing")
print("   • Use Llama-2-13B for: balanced performance, most general use cases")
print("   • Use Llama-2-70B for: complex reasoning, critical analysis, highest quality")
print()
print("📚 Next Steps:")
print("   • Try embeddings_workflow.py for RAG pipelines")
print("   • See production_deployment.py for high-volume patterns")
