<p align="center">
  <img width="500" src="./assets/brand/genops-logo-optimized.jpg" alt="GenOps AI - OpenTelemetry-native governance for AI systems" style="max-width: 100%;">
</p>

> 🚧 **Preview Release** - Complete AI governance platform with attribution, validation & compliance features. [Community contributions welcome!](CONTRIBUTING.md)

<div align="center">
  <h3>OpenTelemetry-native governance for AI systems</h3>
  <p><em>Turn AI telemetry into actionable accountability</em></p>
  
  [![GitHub stars](https://img.shields.io/github/stars/KoshiHQ/GenOps-AI?style=social)](https://github.com/KoshiHQ/GenOps-AI/stargazers)
  [![CI Status](https://img.shields.io/github/actions/workflow/status/KoshiHQ/GenOps-AI/ci.yml?branch=main)](https://github.com/KoshiHQ/GenOps-AI/actions)  
  [![PyPI version](https://badge.fury.io/py/genops.svg)](https://badge.fury.io/py/genops)
  [![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
  [![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
  [![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
  [![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-native-purple.svg)](https://opentelemetry.io/)
</div>

---

## 🎯 **What is GenOps AI?**

GenOps AI is an **open-source governance framework** that brings cost attribution, policy enforcement, and compliance automation to AI systems using **OpenTelemetry standards**.

While [OpenLLMetry](https://github.com/traceloop/openllmetry) tells you *what* your AI is doing (prompts, completions, tokens), **GenOps AI tells you *why and how* — with governance telemetry** that enables:

- 💰 **Cost Attribution** across teams, projects, features, and customers
- 🛡️ **Policy Enforcement** with configurable limits and content filtering  
- 📊 **Budget Tracking** with automated alerts and spend controls
- 🔍 **Compliance Automation** with evaluation metrics and audit trails
- 📈 **Observability Integration** with your existing monitoring stack

**Built on OpenTelemetry standards, works alongside OpenLLMetry and other observability tools.**

---

## 🚨 **The Problem: AI Costs Are Out of Control**

**Real scenarios happening right now:**
- 💸 **$50,000 surprise bills** - One customer's chat feature cost 10x more than expected
- 🔍 **No visibility** - "Which team/feature is burning through our AI budget?"  
- 🚫 **No guardrails** - Production systems calling GPT-4 when GPT-3.5 would work
- 📊 **CFO questions** - "How much is AI costing us per customer?"
- ⚖️ **Compliance gaps** - No audit trail for AI decisions in regulated industries

**Without governance, AI is just expensive magic.** ✨💸

### 👥 **Who This Is For**

- **DevOps Teams**: "I need AI costs in my existing dashboards"
- **FinOps Teams**: "I need per-customer AI cost allocation"  
- **Platform Teams**: "I need budget controls before we go bankrupt"
- **Compliance Teams**: "I need audit trails for AI decisions"
- **CTOs**: "I need to understand what we're actually paying for"

---

## ✨ **Key Features**

### 🚀 **Provider Instrumentation** (Production-Ready)
```python
from genops.providers.openai import instrument_openai

# Instrument OpenAI with automatic governance tracking
client = instrument_openai(api_key="your-openai-key")

# All calls now include cost, token, and governance telemetry
response = client.chat_completions_create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}],
    # Governance attributes
    team="support-team",
    project="ai-assistant", 
    customer_id="enterprise-123"
)
# ✅ Cost, tokens, policies automatically tracked and exported via OpenTelemetry
```

### 🎛️ **Manual Telemetry Tracking** 
```python
from genops.core.telemetry import GenOpsTelemetry

telemetry = GenOpsTelemetry()

# Track any operation with governance context
with telemetry.trace_operation(
    operation_name="customer_support",
    team="support-team",
    project="ai-chatbot",
    customer_id="customer_123"
) as span:
    # Your AI processing logic
    ai_response = call_your_ai_model(message)
    
    # Record governance telemetry
    telemetry.record_cost(span, cost=0.05, provider="openai", model="gpt-3.5-turbo")
    telemetry.record_evaluation(span, metric_name="quality", score=0.92)

# Governance data automatically flows to your observability stack via OpenTelemetry
```

### 🛡️ **Policy Enforcement**
```python
from genops.core.policy import register_policy, PolicyResult, _policy_engine

# Register governance policies  
register_policy(
    name="cost_limit",
    enforcement_level=PolicyResult.BLOCKED,
    conditions={"max_cost": 5.00}
)

# Evaluate policies before operations
def safe_ai_operation(prompt: str, estimated_cost: float):
    # Check policy before operation
    result = _policy_engine.evaluate_policy("cost_limit", {"cost": estimated_cost})
    
    if result.result == PolicyResult.BLOCKED:
        raise Exception(f"Policy violation: {result.reason}")
    
    return call_ai_model(prompt)  # Proceeds if policy allows
```

### 📊 **Rich Governance Telemetry**
```python
from genops.core.telemetry import GenOpsTelemetry

telemetry = GenOpsTelemetry()

with telemetry.trace_operation(operation_name="document_analysis") as span:
    # AI processing...
    ai_result = process_document()
    
    # Record comprehensive governance signals
    telemetry.record_cost(span, cost=2.50, currency="USD", provider="openai")
    telemetry.record_policy(span, policy_name="content_safety", result="allowed") 
    telemetry.record_evaluation(span, metric_name="quality_score", score=0.92)
    telemetry.record_budget(span, budget_name="monthly_ai_spend", allocated=1000, consumed=150)
```

---

## 🚀 **Quick Start**

### Installation

```bash
pip install genops

# With AI provider support
pip install "genops[openai,anthropic]"  # For OpenAI + Anthropic
pip install "genops[all]"               # All providers
```

### ⚡ **30-Second Test**

Verify your installation works:

```bash
# Test the CLI
genops --version

# Quick Python test
python -c "import genops; print('✅ GenOps AI installed successfully!')"
```

### 5-Minute Governance Setup

```python
from genops.providers.openai import instrument_openai
import genops

# 1. Set default attribution (once at app startup)
genops.set_default_attributes(
    team="platform-engineering",
    project="ai-services", 
    environment="production"
)

# 2. Instrument your AI providers  
client = instrument_openai(api_key="your-openai-key")

# 3. Use normally - defaults inherited automatically
response = client.chat_completions_create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}],
    # Only specify what's unique to this operation
    customer_id="enterprise-123",
    feature="chat-assistant"
    # team, project, environment automatically included!
)

# 4. OpenTelemetry exports complete attribution data
# ✅ Cost, tokens, team, customer, feature → Your observability platform
```

**That's it!** Full governance telemetry with intelligent attribution inheritance.

### 🎯 **Real-World Governance Scenarios**

See complete end-to-end examples that solve real business problems:

```bash
# 🚨 Prevent AI budget overruns with automatic enforcement
python examples/governance_scenarios/budget_enforcement.py

# 🛡️ Block inappropriate content with real-time filtering  
python examples/governance_scenarios/content_filtering.py

# 📊 Track AI costs per customer for usage-based billing
python examples/governance_scenarios/customer_attribution.py
```

Each scenario shows working code with realistic business problems and governance solutions.

---

## 📖 **Core Concepts**

### **Governance Semantics**
GenOps extends OpenTelemetry with standardized governance attributes:

- **`genops.cost.*`** - Cost attribution and financial tracking
- **`genops.policy.*`** - Policy enforcement results and violations  
- **`genops.eval.*`** - Quality, safety, and performance evaluations
- **`genops.budget.*`** - Spend tracking and limit management

### **Provider Adapters** 
Pre-built integrations with accurate cost models:

- ✅ **OpenAI** (GPT-3.5, GPT-4, GPT-4-turbo) with per-token pricing
- ✅ **Anthropic** (Claude-3 Sonnet, Opus, Haiku) with accurate costs
- 🚧 **AWS Bedrock** (coming soon)
- 🚧 **Google Gemini** (coming soon)
- 🚧 **LangChain** (coming soon) 
- 🚧 **LlamaIndex** (coming soon)

### **Observability Stack Integration**
Works with your existing tools:

- 📊 **Datadog, Honeycomb, New Relic** - OTLP export
- 📈 **Grafana Tempo, Jaeger** - Distributed tracing
- 🔍 **Elasticsearch, Splunk** - Log aggregation
- ☁️ **AWS X-Ray, Google Cloud Trace** - Cloud-native tracing

---

## 🏗️ **Architecture**

```mermaid
graph TB
    A[Your AI Application] --> B[GenOps AI SDK]
    B --> C[OpenTelemetry]
    C --> D[Your Observability Stack]
    
    B --> E[Provider Adapters]
    E --> F[OpenAI/Anthropic/Bedrock...]
    
    D --> G[Dashboards & Alerts]
    D --> H[Cost Attribution]  
    D --> I[Policy Automation]
    D --> J[Enterprise Dashboards]
    
    style B fill:#e1f5fe
    style D fill:#f3e5f5
    style J fill:#fff3e0
```

**GenOps AI sits alongside OpenLLMetry** in your telemetry stack, adding the governance layer that turns observability data into business accountability.

---

## 🎭 **Usage Examples**

### **Multi-Provider Cost Attribution**
```python
import genops

# Initialize with default governance context
genops.init(default_team="ai-research", default_project="multimodal")

# Use different providers - all automatically tracked
import openai
import anthropic

# OpenAI for quick tasks
openai_response = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Summarize this"}],
    # Inherits team/project, adds specific context
    customer_id="enterprise_123", 
    feature="document_summary"
)

# Anthropic for complex reasoning
anthropic_client = anthropic.Anthropic()
claude_response = anthropic_client.messages.create(
    model="claude-3-opus-20240229", 
    max_tokens=2048,
    messages=[{"role": "user", "content": "Analyze this data"}],
    # Different feature, same customer
    customer_id="enterprise_123",
    feature="data_analysis"  
)

# All operations tagged with cost, provider, customer, feature
# Perfect for FinOps dashboards and customer billing
```

### **Policy-Driven Governance**
```python
import genops
from genops import register_policy, PolicyResult

# Set up governance policies
register_policy("cost_limit", max_cost=10.0, enforcement_level=PolicyResult.BLOCKED)
register_policy("content_safety", blocked_patterns=["violence"], enforcement_level=PolicyResult.WARNING)
register_policy("team_budget", max_monthly_spend=5000, enforcement_level=PolicyResult.RATE_LIMITED)

# Apply policies to operations
def generate_content(prompt: str, customer_tier: str):
    from genops.core.policy import _policy_engine, PolicyViolationError
    
    if customer_tier == "enterprise":
        model = "gpt-4"
        estimated_cost = 0.12  # Higher cost estimate
    else:
        model = "gpt-3.5-turbo"
        estimated_cost = 0.03
    
    # Check policies before operation
    context = {"cost": estimated_cost, "content": prompt}
    cost_result = _policy_engine.evaluate_policy("cost_limit", context)
    
    if cost_result.result == PolicyResult.BLOCKED:
        raise PolicyViolationError("cost_limit", cost_result.reason)
        
    return call_ai_model(model, prompt)

# Policy evaluation with error handling
try:
    result = generate_content("Write a story", "enterprise") 
    # ✅ Allowed: cost under $10, content safe
except PolicyViolationError as e:
    # ❌ Blocked: policy violation with detailed context
    logger.warning(f"Policy {e.policy_name}: {e.reason}")
```

### **Custom Evaluations & Compliance**
```python
from genops.core.telemetry import GenOpsTelemetry

def moderate_content(text: str):
    telemetry = GenOpsTelemetry()
    
    with telemetry.trace_operation(operation_name="content_moderation") as span:
        # Your content moderation logic
        safety_score = run_safety_model(text)
        toxicity_score = check_toxicity(text)
        
        # Record compliance metrics
        telemetry.record_evaluation(span, metric_name="safety", score=safety_score, 
                                   threshold=0.8, passed=safety_score > 0.8)
        telemetry.record_evaluation(span, metric_name="toxicity", score=toxicity_score,
                                   threshold=0.2, passed=toxicity_score < 0.2) 
        
        # Policy decision
        if safety_score > 0.8 and toxicity_score < 0.2:
            telemetry.record_policy(span, policy_name="content_policy", result="approved")
            return {"approved": True, "reason": "Content meets safety standards"}
        else:
            telemetry.record_policy(span, policy_name="content_policy", result="rejected", 
                                   reason="Safety threshold not met")
            return {"approved": False, "reason": "Content violates policy"}

# Rich governance telemetry automatically exported for audit trails
```

### **Budget Tracking & Alerts**  
```python
from genops.core.telemetry import GenOpsTelemetry

def process_customer_requests(customer_id: str, requests: list):
    telemetry = GenOpsTelemetry()
    
    # Track budget utilization per customer
    with telemetry.trace_operation(f"customer_{customer_id}_processing") as span:
        total_cost = 0
        
        for request in requests:
            response = process_with_ai(request)
            request_cost = calculate_cost(response)
            total_cost += request_cost
            
        # Update customer budget tracking
        customer_budget = get_customer_budget(customer_id)
        remaining = customer_budget.limit - customer_budget.used - total_cost
        
        telemetry.record_budget(
            span=span,
            budget_name=f"customer_{customer_id}_monthly",
            allocated=customer_budget.limit,
            consumed=customer_budget.used + total_cost, 
            remaining=remaining
        )
        
        # Automatic alerts when budget utilization > 80%
        if remaining / customer_budget.limit < 0.2:
            telemetry.record_policy(span, policy_name="budget_warning", result="triggered", 
                                   reason=f"Customer {customer_id} at 80% budget utilization")
```

---

## 🏢 **Production Ready**

### **Compliance & Audit Trails**
GenOps AI automatically creates detailed audit logs for:
- **Cost attribution** with exact token counts and pricing models
- **Policy decisions** with enforcement context and reasoning
- **Data flow tracking** for privacy and compliance requirements  
- **Model usage patterns** for governance and risk management

### **Observability Integration**
Works with your existing tools and workflows:
- **Per-customer cost allocation** for accurate billing
- **Team and department spend tracking** for budget management
- **Feature-level cost analysis** for product decisions
- **Model efficiency metrics** for optimization opportunities
- **Real-time dashboards** using your current observability platform

---

## 🤝 **Community & Support**

### **Contributing**
We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup and testing guidelines
- Code standards and review process
- Community guidelines and code of conduct

### **Getting Help**
- 📖 **Documentation**: [GitHub Docs](https://github.com/KoshiHQ/GenOps-AI/tree/main/docs)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)
- 🐛 **Issues**: [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)

### **Roadmap**
See our [public roadmap](https://github.com/KoshiHQ/GenOps-AI/projects) for upcoming features:
- 🚧 AWS Bedrock and Google Gemini adapters
- 🚧 LangChain and LlamaIndex integrations  
- 🚧 OpenTelemetry Collector processors for real-time governance
- 🚧 Pre-built dashboards for major observability platforms

---

## 📄 **License**

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---

## 🌟 **Why GenOps AI?**

**Traditional AI monitoring tells you what happened. GenOps AI tells you what it cost, who did it, whether it should have been allowed, and how well it worked.**

- **For DevOps Teams**: Integrate AI governance into existing observability workflows
- **For FinOps Teams**: Get precise cost attribution and budget controls
- **For Compliance Teams**: Automated policy enforcement with audit trails
- **For Product Teams**: Feature-level AI cost analysis and optimization insights

**Open source, OpenTelemetry-native, and designed to work with your existing stack.**

---

## 🤝 **Community & Quick Wins**

**New to open source?** Start here:
- 🐛 [Good first issues](https://github.com/KoshiHQ/GenOps-AI/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) - Perfect for newcomers
- 📚 [Documentation improvements](https://github.com/KoshiHQ/GenOps-AI/issues?q=is%3Aissue+is%3Aopen+label%3Adocumentation) - Help others learn
- 🔧 [Help fix our CI tests!](https://github.com/KoshiHQ/GenOps-AI/issues?q=is%3Aissue+is%3Aopen+label%3Aci-fix) - Great for contributors who love debugging

**5-minute contributions welcome!** Every small improvement helps the community grow.

**Looking for bigger challenges?**
- 🏗️ [Provider integrations](https://github.com/KoshiHQ/GenOps-AI/issues?q=is%3Aissue+is%3Aopen+label%3Aprovider) - Add AWS Bedrock, Google Gemini support
- 📊 [Dashboard templates](https://github.com/KoshiHQ/GenOps-AI/issues?q=is%3Aissue+is%3Aopen+label%3Adashboard) - Pre-built observability dashboards
- 🤖 [AI governance patterns](https://github.com/KoshiHQ/GenOps-AI/issues?q=is%3Aissue+is%3Aopen+label%3Agovernance) - Real-world scenarios

---

## ⚠️ **Known Issues & Contributing**

This is a **preview release** with comprehensive features but some ongoing CI test issues:

### 🚧 Current Status
- ✅ **Core functionality working**: Security scans pass, package installation works
- ✅ **Comprehensive examples**: All governance scenarios and integrations functional
- ⚠️ **Some CI tests failing**: Integration tests and Python 3.11 compatibility
- 🤝 **Community help wanted**: [See open issues](https://github.com/KoshiHQ/GenOps-AI/issues) for contribution opportunities

### 🆘 Need Help?
- 💬 **Questions**: [GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)
- 🐛 **Issues**: [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- 🤝 **Contributing**: [Contributing Guide](CONTRIBUTING.md)

---

## ✨ Contributors

Thanks goes to these wonderful people who have contributed to GenOps AI:

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

---

## 🏷️ **Trademark & Brand Guidelines**

### **GenOps AI Trademark Usage**

The "GenOps AI" name and associated branding are trademarks used to identify this project and its official implementations.

**✅ Acceptable Use:**
- Referring to this project in documentation, blog posts, or presentations
- Building integrations or extensions that work with GenOps AI
- Using "Built with GenOps AI" or "Powered by GenOps AI" attributions
- Community projects that extend or integrate with GenOps AI functionality

**❌ Prohibited Use:**
- Using "GenOps" in the name of competing commercial AI governance products
- Creating confusion about official vs. community implementations  
- Using GenOps branding for unrelated products or services
- Implying official endorsement without permission

**📄 License Note:** The GenOps AI code is licensed under Apache 2.0, but trademark rights are separate from code rights. You're free to use, modify, and distribute the code under Apache 2.0, but please respect our trademark guidelines when naming your projects or products.

For questions about trademark usage, please open an issue or contact the maintainers.

---

## 📄 **Legal & Licensing**

- **Code License**: [Apache License 2.0](LICENSE) - Permissive open source license
- **Contributor Agreement**: All contributions require [DCO sign-off](CONTRIBUTING.md#developer-certificate-of-origin-dco)
- **Copyright**: Copyright © 2024 GenOps AI Contributors
- **Trademark**: "GenOps AI" and associated marks are trademarks of the project maintainers

---

<div align="center">
  <p><strong>Ready to bring governance to your AI systems?</strong></p>
  
  ```bash
  pip install genops
  ```
  
  <p>⭐ <strong>Star us on GitHub</strong> if you find GenOps AI useful!</p>
  
  [![GitHub stars](https://img.shields.io/github/stars/KoshiHQ/GenOps-AI?style=social)](https://github.com/KoshiHQ/GenOps-AI/stargazers)
</div>