"""
Minimal Kubetorch Example - Hello World

This example shows the absolute minimum code needed to get started
with GenOps Kubetorch governance tracking.

Time to run: < 30 seconds
"""

from genops.providers.kubetorch import (
    auto_instrument_kubetorch,
    calculate_gpu_cost,
)

# Enable zero-code tracking
auto_instrument_kubetorch(team="ml-team")

# Calculate training cost
cost = calculate_gpu_cost(
    instance_type="a100",
    num_devices=8,
    duration_seconds=3600  # 1 hour
)

print(f"Training cost for 8x A100 (1 hour): ${cost:.2f}")
# Output: Training cost for 8x A100 (1 hour): $262.16

print("\n✅ Done! Governance tracking is now enabled globally.")
