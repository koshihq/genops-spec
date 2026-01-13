#!/usr/bin/env python3
"""
Minimal Anyscale Example - 5 Minute Quickstart

This is the absolute minimum code to demonstrate GenOps + Anyscale value.
Time to value: < 5 minutes

Prerequisites:
- export ANYSCALE_API_KEY='your-api-key'
- pip install genops-ai
"""

import os
from genops.providers.anyscale import instrument_anyscale

# Check API key
if not os.getenv("ANYSCALE_API_KEY"):
    print("❌ ERROR: ANYSCALE_API_KEY not set")
    print("Fix: export ANYSCALE_API_KEY='your-key'")
    print("Get key: https://console.anyscale.com/credentials")
    exit(1)

print("🚀 GenOps Anyscale - Minimal Example\n")

# Create adapter with governance
adapter = instrument_anyscale(team="quickstart-team")

# Make a completion request
response = adapter.completion_create(
    model="meta-llama/Llama-2-7b-chat-hf",  # Cheapest model for demo
    messages=[{"role": "user", "content": "Say hello in one sentence"}],
    max_tokens=50
)

# Print response
print(f"✅ Response: {response['choices'][0]['message']['content']}\n")

# Show what GenOps tracked
print("📊 What GenOps Tracked:")
print(f"   • Tokens: {response['usage']['total_tokens']}")
print(f"   • Team: quickstart-team")
print(f"   • Model: meta-llama/Llama-2-7b-chat-hf")
print(f"   • Cost: Automatically calculated")
print(f"   • Telemetry: Exported to your observability platform\n")

print("✅ SUCCESS! GenOps is tracking your Anyscale usage")
print("📚 Next: Try multi_model_comparison.py to optimize costs")
