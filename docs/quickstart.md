# Quick Start Guide

Get GenOps AI up and running with governance telemetry in under 5 minutes! This guide will have you tracking AI costs, enforcing policies, and collecting governance data with minimal setup.

## 🎯 **What You'll Learn**

By the end of this guide, you'll have:
- ✅ GenOps AI installed and configured
- ✅ Auto-instrumentation tracking your AI operations
- ✅ Cost attribution flowing to your observability stack  
- ✅ A governance policy enforcing spend limits
- ✅ Rich telemetry data for FinOps and compliance

**Time required:** 5 minutes  
**Prerequisites:** Python 3.8+ and an AI application using OpenAI or Anthropic

---

## 🚀 **Step 1: Installation**

Install GenOps AI with your preferred AI providers:

```bash
# Basic installation
pip install genops

# With provider support  
pip install "genops[openai,anthropic]"

# With all providers and development tools
pip install "genops[all,dev]"
```

Verify installation:
```bash
genops version
# Output: GenOps AI v0.1.0 - OpenTelemetry-native governance for AI
```

---

## ⚡ **Step 2: One-Line Auto-Instrumentation**

The fastest way to get started is with auto-instrumentation. Add this **one line** to your application startup:

```python
import genops

# Auto-instrument all AI providers with governance
genops.init(
    service_name="my-ai-app",
    environment="production",
    exporter_type="console",  # Use "otlp" for production
    
    # Default governance attributes
    default_team="engineering",
    default_project="customer-support"
)

print("🎉 GenOps AI initialized! Your AI calls are now governed.")
```

**That's it!** Your existing AI code now automatically emits governance telemetry.

---

## 🔍 **Step 3: Test with Your AI Code**

Your existing AI code now gets automatic governance tracking:

```python
# Your existing OpenAI code - now with automatic governance!
import openai

client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful customer support assistant."},
        {"role": "user", "content": "How do I reset my password?"}
    ]
)

print(f"AI Response: {response.choices[0].message.content}")

# ✅ This call was automatically tracked with:
# - Cost calculation ($0.0015 for this example)
# - Token usage (input: 25, output: 50)
# - Model and provider information
# - Team and project attribution
# - Timestamp and request metadata
```

Check what's being tracked:
```python
status = genops.status()
print(f"Instrumented providers: {status['instrumented_providers']}")
print(f"Default attributes: {status['default_attributes']}")

# Output:
# Instrumented providers: ['openai']
# Default attributes: {'team': 'engineering', 'project': 'customer-support'}
```

---

## 📊 **Step 4: Add Rich Governance Context**

Enhance your governance data with specific context:

```python
import genops

# Manual instrumentation with rich context
@genops.track_usage(
    operation_name="customer_password_reset",
    feature="password_management",
    customer_id="cust_12345",
    customer_tier="premium"
)
def handle_password_reset(customer_email: str):
    """Handle customer password reset with full governance tracking."""
    
    # Your AI logic here
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Generate a helpful password reset email."},
            {"role": "user", "content": f"Customer {customer_email} needs password reset help"}
        ]
    )
    
    return response.choices[0].message.content

# Call with governance tracking
reset_email = handle_password_reset("customer@company.com")

# ✅ This creates rich telemetry with:
# - Cost attribution to specific customer and feature
# - Team and project from auto-instrumentation defaults
# - Customer tier for advanced analytics
# - Operation-specific metadata
```

---

## 🛡️ **Step 5: Add Governance Policies**

Enforce spend limits and content policies:

```python
from genops import register_policy, PolicyResult, enforce_policy

# Register governance policies
register_policy(
    name="cost_control",
    description="Prevent expensive AI operations",
    enforcement_level=PolicyResult.BLOCKED,
    max_cost=5.00  # Block operations over $5
)

register_policy(
    name="content_safety", 
    description="Filter unsafe content",
    enforcement_level=PolicyResult.WARNING,
    blocked_patterns=["password", "sensitive"]  # Warn on sensitive content
)

# Apply policies to operations
@enforce_policy(["cost_control", "content_safety"])
@genops.track_usage(operation_name="content_generation")
def generate_marketing_content(prompt: str):
    """Generate marketing content with governance policies."""
    
    return client.chat.completions.create(
        model="gpt-4",  # More expensive model, policy will check cost
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000
    )

# Test policy enforcement
try:
    content = generate_marketing_content("Write a blog post about our product")
    print(f"✅ Content generated: {content.choices[0].message.content[:100]}...")
except genops.PolicyViolationError as e:
    print(f"❌ Policy violation: {e.policy_name} - {e.reason}")
```

---

## 🔧 **Step 6: Connect to Your Observability Stack**

For production, send telemetry to your existing observability platform:

### **Option A: OTLP (Recommended)**
```python
genops.init(
    service_name="customer-support-ai",
    environment="production", 
    exporter_type="otlp",
    otlp_endpoint="https://api.honeycomb.io",  # Your OTLP endpoint
    otlp_headers={
        "x-honeycomb-team": "your-api-key",
        "x-honeycomb-dataset": "ai-governance"
    },
    default_team="customer-success",
    default_project="support-chatbot"
)
```

### **Option B: Platform-Specific**
```python
# Datadog
genops.init(
    exporter_type="otlp",
    otlp_endpoint="https://trace.agent.datadoghq.com",
    otlp_headers={"DD-API-KEY": "your-datadog-key"}
)

# New Relic  
genops.init(
    exporter_type="otlp",
    otlp_endpoint="https://otlp.nr-data.net:4317",
    otlp_headers={"api-key": "your-newrelic-key"}
)
```

---

## 📈 **Step 7: View Your Governance Data**

Your telemetry data is now flowing to your observability platform with attributes like:

```json
{
  "genops.operation.name": "customer_password_reset",
  "genops.operation.type": "ai.inference",
  "genops.provider": "openai",
  "genops.model": "gpt-3.5-turbo",
  "genops.cost.total": 0.0015,
  "genops.cost.currency": "USD",
  "genops.tokens.input": 25,
  "genops.tokens.output": 50,
  "genops.tokens.total": 75,
  "genops.team": "engineering", 
  "genops.project": "customer-support",
  "genops.feature": "password_management",
  "genops.customer_id": "cust_12345",
  "genops.customer_tier": "premium",
  "genops.policy.name": "cost_control",
  "genops.policy.result": "allowed"
}
```

**Create dashboards** in your platform to track:
- 💰 **Cost per customer, team, or feature**
- 📊 **Token usage and model efficiency** 
- 🛡️ **Policy violations and governance events**
- 📈 **AI spend trends and budget utilization**

---

## 🎉 **What's Next?**

Congratulations! You now have AI governance telemetry flowing. Here are some next steps:

### **Immediate Actions**
- 📊 **Set up dashboards** in your observability platform
- 🔔 **Create alerts** for cost thresholds and policy violations
- 📋 **Share governance data** with your FinOps and compliance teams

### **Advanced Features**
- 🏗️ **Add more providers** - AWS Bedrock, Google Gemini (coming soon)
- 🔗 **Integrate frameworks** - LangChain, LlamaIndex (coming soon)
- 🏢 **Enterprise deployment** - Connect to [Koshi control plane](https://getkoshi.ai)

### **Community & Learning**
- 📖 **Read the full documentation** - [GitHub Docs](https://github.com/KoshiHQ/GenOps-AI/tree/main/docs)
- 💬 **Join discussions** - [GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)
- 🤝 **Contribute** - [Contributing Guide](../CONTRIBUTING.md)

---

## 📋 **Complete Example**

Here's a complete working example combining all the concepts:

```python
#!/usr/bin/env python3
"""Complete GenOps AI governance example."""

import genops
from genops import register_policy, PolicyResult, enforce_policy
import openai

# 1. Initialize GenOps with your observability stack
genops.init(
    service_name="customer-support-ai",
    environment="production",
    exporter_type="console",  # Change to "otlp" for production
    # otlp_endpoint="https://api.honeycomb.io",
    # otlp_headers={"x-honeycomb-team": "your-api-key"},
    
    # Default governance context
    default_team="customer-success", 
    default_project="support-chatbot"
)

# 2. Register governance policies
register_policy(
    name="support_cost_limit",
    description="Control support AI costs",
    enforcement_level=PolicyResult.BLOCKED,
    max_cost=2.00
)

register_policy(
    name="response_quality",
    description="Ensure quality responses",
    enforcement_level=PolicyResult.WARNING,
    min_confidence=0.8
)

# 3. Create governed AI operations
@enforce_policy(["support_cost_limit", "response_quality"])
@genops.track_usage(
    operation_name="customer_support_response",
    feature="chat_support"
)
def generate_support_response(customer_query: str, customer_id: str, tier: str = "standard"):
    """Generate customer support response with full governance."""
    
    # Select model based on customer tier
    model = "gpt-4" if tier == "premium" else "gpt-3.5-turbo"
    
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful customer support assistant."},
            {"role": "user", "content": customer_query}
        ],
        temperature=0.3,
        # Governance context
        customer_id=customer_id,
        customer_tier=tier
    )
    
    return response.choices[0].message.content

# 4. Use with governance tracking
if __name__ == "__main__":
    try:
        # Generate support response
        response = generate_support_response(
            customer_query="How do I upgrade my account?",
            customer_id="cust_67890",
            tier="premium"
        )
        
        print(f"✅ Support Response: {response}")
        
        # Check governance status
        status = genops.status()
        print(f"\n📊 Governance Status:")
        print(f"   Instrumented: {status['initialized']}")
        print(f"   Providers: {status['instrumented_providers']}")
        print(f"   Defaults: {status['default_attributes']}")
        
    except genops.PolicyViolationError as e:
        print(f"❌ Policy Violation: {e.policy_name}")
        print(f"   Reason: {e.reason}")
        print(f"   Metadata: {e.metadata}")
    
    except Exception as e:
        print(f"💥 Error: {e}")
    
    finally:
        # Clean up
        genops.uninstrument()
        print("\n🧹 GenOps instrumentation removed.")
```

**Run this example:**
```bash
python complete_example.py
```

You'll see governance telemetry flowing with cost attribution, policy enforcement, and rich metadata for FinOps and compliance teams.

---

## 🆘 **Troubleshooting**

### **Common Issues**

**Q: "No telemetry data appearing"**
- Check that `genops.init()` was called before AI operations
- Verify your OTLP endpoint and headers are correct
- Try `exporter_type="console"` to see local output

**Q: "Provider not instrumented"**  
- Install provider packages: `pip install openai anthropic`
- Check `genops.status()['available_providers']` for availability

**Q: "Policy violations not working"**
- Ensure policies are registered before applying `@enforce_policy`
- Check policy conditions match your use case
- Use `PolicyResult.WARNING` for testing

**Q: "High telemetry overhead"**
- Use sampling in production: `genops.init(sampling_rate=0.1)`
- Check [Performance Guide](advanced/performance.md) for optimization

### **Getting Help**

- 📖 **Documentation** - [GitHub Docs](https://github.com/KoshiHQ/GenOps-AI/tree/main/docs) 
- 💬 **Community** - [GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)
- 🐛 **Issues** - [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)

---

**🎊 Congratulations!** You now have production-ready AI governance with GenOps AI. Your AI operations are tracked, governed, and ready for enterprise accountability.

**Next:** Check out our [governance patterns](governance/) and [integration guides](integrations/) to unlock the full power of AI telemetry! 🚀