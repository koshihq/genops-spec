<p align="center">
  <img width="500" src="./assets/brand/genops-logo-optimized.jpg" alt="GenOps: Open Runtime Governance for AI Systems" style="max-width: 100%;">
</p>

# 🧭 GenOps AI: Runtime Governance for AI Workloads

GenOps AI is the open-source standard that gives you **human-in-the-loop control** over AI systems in production, built on <a href="https://opentelemetry.io" target="_blank">OpenTelemetry</a>.

**Runtime governance for all your existing AI model providers, tools/systems, and frameworks**: standard telemetry + policy enforcement that enables you to control, govern, and intervene in AI operations as they happen in real time — not after the fact.

<div align="center">
  
  [![GitHub stars](https://img.shields.io/github/stars/KoshiHQ/GenOps-AI?style=social)](https://github.com/KoshiHQ/GenOps-AI/stargazers)
  [![CI Status](https://img.shields.io/github/actions/workflow/status/KoshiHQ/GenOps-AI/ci.yml?branch=main)](https://github.com/KoshiHQ/GenOps-AI/actions)  
  [![PyPI version](https://badge.fury.io/py/genops.svg)](https://badge.fury.io/py/genops)
  [![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
  [![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
  [![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
  [![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-native-purple.svg)](https://opentelemetry.io/)

</div>

---

## 🚨 The Problem: AI Systems Running Wild in Production

You've deployed AI systems that work great in development. But in production, you've lost **human control** over what's happening inside your AI stack...

Sound familiar?

- 🤖 **AI black boxes** — Your models are making decisions, but you can't intervene when things go wrong
- 💸 **Runaway costs** — AI spend spiraling with no way to enforce budgets or limits in real-time
- 🚫 **No guardrails** — AI systems operating without policy enforcement or human oversight
- ⚡ **After-the-fact alerts** — You find out about problems hours later, not when you can actually do something
- 🤷‍♂️ **Zero governance** — Great observability dashboards, but no way to actually *control* what's happening

The result: You have visibility into your AI systems but no ability to govern them when it matters.

**You need runtime control that lets humans stay in the loop of AI decision-making.**

## 👥 Who This Is For

**If you're building with AI, GenOps is for you:**

**🧑‍💻 Individual Developers**
- Keep human control over your AI experiments with runtime guardrails
- Enforce budget limits and policy constraints as your code runs
- Intervene in AI operations before they cause problems
- Share governance patterns with your team without enterprise overhead

**👨‍💼 Team Leads & Senior Engineers**
- Maintain oversight and control over your team's AI systems in production
- Enforce spending limits and usage policies across all AI operations
- Get real-time alerts with the power to take action immediately
- Demonstrate responsible AI governance to management

**🛠️ Platform Engineers**
- Build AI governance into your infrastructure with human-in-the-loop controls
- Support multiple teams with unified policy enforcement
- Use familiar OpenTelemetry patterns for runtime governance
- Scale from individual oversight to organization-wide AI control

**Start individual. Scale with your team. Grow into your organization.**

---

## 💡 The GenOps Solution

GenOps adds the **runtime governance layer** your AI stack is missing — enabling human control without replacing the tools you already love:

- **Real-time policy enforcement** across LLM providers, AI frameworks, routing tools, vector databases, and more
- **Budget guardrails** with automatic intervention when limits are exceeded
- **Human-in-the-loop controls** that let you govern AI operations as they happen
- **Runtime intervention** — standard OpenTelemetry telemetry + governance actions

Because GenOps uses standard OpenTelemetry, it integrates with whatever AI tools and observability platforms you're already using. Keep your existing tools, add the governance layer that gives you control.

---

## ⚙️ What GenOps Delivers

**🏛️ Runtime Governance & Control**
- Enforce policies and budgets across ALL your AI tools as operations happen
- Automatic intervention that spans LLM providers + AI frameworks + routing services + vector databases
- Human-in-the-loop controls with team-based access and approval workflows
- Works with any combination of AI tools you're using

**💰 Intelligent Budget Enforcement**
- Automatically halt spending when limits are exceeded across all providers
- Real-time cost controls regardless of which tools you use
- Per-project, per-team, per-customer budget enforcement across all providers
- Proactive budget governance with human approval gates

**📊 Governance Dashboards & Controls**  
- Ready-to-use policy controls integrated with your existing observability tools
- Cost controls, usage policies, and intervention capabilities
- Answers questions like "who can approve this AI operation?" and "should we stop this request?"
- Real-time governance actions with audit trails for compliance

**🔧 Zero-Friction Integration**
- 30-second setup with auto-instrumentation that detects your AI libraries
- Works with whatever AI tools you already use (LLM providers, AI frameworks, routing services, etc.)
- Standard OpenTelemetry output compatible with 15+ observability platforms
- No vendor lock-in or tool replacement required - enhances your existing stack

---

## 🔄 Governance vs. Observability: The Critical Difference

**Traditional AI Observability Tools:**
- 📊 Show you what happened *after* AI operations complete
- 📈 Great dashboards and analytics, but no way to intervene
- ⚠️ Alert you to problems when it's too late to prevent them
- 📋 Track costs and usage for reporting purposes

**GenOps Runtime Governance:**
- 🛑 **Intervenes in AI operations as they happen** with human-controlled policies
- ⚡ **Enforces budgets and constraints in real-time** before problems occur
- 🎯 **Enables human decision-making** at critical points in AI workflows
- 🔒 **Prevents** runaway costs, policy violations, and ungoverned AI behavior

**The result:** Move from "What happened?" to "What should happen?" — shifting from reactive monitoring to proactive human control of AI systems.

---

## 🚀 Defining the Open Standard for AI Governance

GenOps is pioneering the **next generation of AI control systems** by extending OpenTelemetry into the governance domain.

**Why this matters:**
- 📐 **Shaping the standard** — Building the open-source foundation that will define how AI governance works
- 🏗️ **Architecture for the future** — Human-centric AI governance patterns that scale from startups to enterprises  
- 🤝 **Community-driven** — Open development with transparent standards and contribution pathways
- 🌍 **Vendor-neutral** — No lock-in to any specific AI provider or observability platform

**Join us in defining how humans maintain control over AI systems in production.**

---

## 🤝 Works with Your Existing Stack

**Keep the tools you love, add the governance layer that gives you control:**

**Already using LLM providers directly?** GenOps adds runtime budget enforcement and policy controls without changing your code.

**Already using AI frameworks or routing tools?** GenOps enables human-in-the-loop governance across all your AI operations.

**Already using observability platforms?** GenOps extends your existing dashboards with governance controls and intervention capabilities.

**The result**: Runtime governance across all your AI tools without migration pain or vendor lock-in.

---

## 📦 Quick Start

### 1. Install the SDK
```bash
pip install genops
```

### 2. Initialize with governance policies
```python
from genops import GenOps

# Enable runtime governance with human controls
GenOps.init(
    daily_budget_limit=100.0,      # Automatically halt operations at $100/day
    require_approval_over=50.0,    # Require human approval for requests >$50
    enable_policy_enforcement=True  # Enable real-time policy checks
)

# Your existing AI code now has governance controls
import openai
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
# ↑ This request is now governed by your policies and budget limits
```

### 3. Immediate governance & control
GenOps automatically captures governance telemetry with control actions:

```json
{
  "trace_id": "abc123",
  "span_name": "openai.chat.completion",
  "attributes": {
    "genops.cost.total": 0.002,
    "genops.cost.currency": "USD",
    "genops.governance.policy_check": "passed",
    "genops.governance.budget_remaining": 97.50,
    "genops.governance.approval_required": false,
    "genops.governance.human_intervention": "none",
    "genops.provider": "openai",
    "genops.model": "gpt-4",
    "genops.team": "engineering",
    "genops.project": "chatbot"
  }
}
```

**Control your AI systems through your existing observability stack** - Datadog, Grafana, Honeycomb, or any OpenTelemetry-compatible platform.

---

## 💡 What You'll See in 5 Minutes

After the 3-step setup above, GenOps immediately provides real-time governance and human control:

### **Runtime Governance Dashboard**
```
🛡️ AI Governance Status (Real-Time)
┌─────────────────┬──────────┬─────────────┬─────────────┐
│ Team            │ Budget   │ Used        │ Status      │
├─────────────────┼──────────┼─────────────┼─────────────┤
│ engineering     │ $100.00  │ $76.60      │ ✅ Active   │
│ product         │ $75.00   │ $15.80      │ ✅ Active   │
│ marketing       │ $50.00   │ $48.20      │ ⚠️  Warning │
└─────────────────┴──────────┴─────────────┴─────────────┘
```

### **Human-in-the-Loop Controls**
```
🎯 Pending Governance Actions
→ High-cost request from 'product' team requires approval ($52.30)
→ Budget threshold exceeded for 'marketing' - operations paused
→ Policy violation detected in 'research' workflow - human review needed
```

### **Real-Time Intervention**
```
🚨 GOVERNANCE ACTION: Team 'marketing' budget limit reached
   ├── All AI operations automatically paused
   ├── Team lead notified for budget increase approval  
   ├── 3 pending requests waiting for human decision
   └── ✅ Human approved budget increase - operations resumed
```

**This works with your existing observability tools** - tracking data appears in Datadog traces, Grafana dashboards, or wherever you already monitor your applications.

---

---

## 🔧 How Teams Use GenOps Governance

**Individual Developer Control**
Start by adding governance policies to your personal AI projects. GenOps provides immediate budget enforcement and policy controls across your development work.

**Team Governance Pattern**  
Share governance policies across team members with human-in-the-loop approval workflows. Multiple developers operate under unified governance controls with shared oversight dashboards.

**Organization Control Scaling**
As governance requirements grow, teams scale from individual oversight to organization-wide AI control systems with enterprise policy automation, compliance workflows, and centralized human decision-making.

**Common Governance Progression:**
1. **Individual**: Implement GenOps governance policies for personal AI projects
2. **Team**: Standardize on GenOps control systems across team members  
3. **Organization**: Scale governance foundation to enterprise control platforms

**When you need organization-wide AI governance, the OpenTelemetry foundation scales to enterprise control systems.**

---

## 🔌 Integrations & Support

### 🧠 AI & LLM Ecosystem
- ✅ [OpenAI](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/openai) (<a href="https://openai.com/" target="_blank">↗</a>)
- ✅ [Anthropic](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/anthropic) (<a href="https://www.anthropic.com/" target="_blank">↗</a>)
- ✅ [Google Gemini](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/gemini) (<a href="https://deepmind.google/technologies/gemini/" target="_blank">↗</a>)
- ✅ [Hugging Face](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/huggingface) (<a href="https://huggingface.co/docs/inference-providers/index" target="_blank">↗</a>)
- ✅ [AWS Bedrock](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/bedrock) (<a href="https://aws.amazon.com/bedrock/" target="_blank">↗</a>)
- ✅ [OpenRouter](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/openrouter) (<a href="https://openrouter.ai/" target="_blank">↗</a>)
- ✅ [LiteLLM](https://github.com/KoshiHQ/GenOps-AI/tree/main/docs/litellm-quickstart.md) (<a href="https://docs.litellm.ai/" target="_blank">↗</a>)
- ✅ [Vercel AI SDK](https://github.com/KoshiHQ/GenOps-AI/tree/main/docs/vercel-ai-sdk-quickstart.md) (<a href="https://ai-sdk.dev/" target="_blank">↗</a>)
- ✅ [Helicone](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/helicone) (<a href="https://helicone.ai/" target="_blank">↗</a>)
- ✅ [LangChain](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/langchain) (<a href="https://python.langchain.com/" target="_blank">↗</a>)
- ✅ [LlamaIndex](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/llamaindex) (<a href="https://www.llamaindex.ai/" target="_blank">↗</a>)
- ✅ [Haystack AI](https://github.com/KoshiHQ/GenOps-AI/tree/main/docs/integrations/haystack.md) (<a href="https://haystack.deepset.ai/" target="_blank">↗</a>)
- ✅ [CrewAI](https://github.com/KoshiHQ/GenOps-AI/tree/main/docs/quickstart/crewai-quickstart.md) (<a href="https://www.crewai.com/" target="_blank">↗</a>)
- ✅ [Replicate](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/replicate) (<a href="https://replicate.com/" target="_blank">↗</a>)
- ✅ [Cohere](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/cohere) (<a href="https://cohere.com/" target="_blank">↗</a>)
- ✅ [Mistral](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/mistral) (<a href="https://mistral.ai/" target="_blank">↗</a>)
- ✅ [Ollama](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/ollama) (<a href="https://ollama.com/" target="_blank">↗</a>)
- ✅ [Perplexity AI](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/perplexity) (<a href="https://www.perplexity.ai/" target="_blank">↗</a>)
- ✅ [Together AI](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/together) (<a href="https://www.together.ai/" target="_blank">↗</a>)
- ✅ [Fireworks AI](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/fireworks) (<a href="https://fireworks.ai/" target="_blank">↗</a>)
- ✅ [Traceloop + OpenLLMetry](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/traceloop) (<a href="https://traceloop.com/" target="_blank">↗</a>)
- ✅ [PromptLayer](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/promptlayer) (<a href="https://promptlayer.com/" target="_blank">↗</a>)
- ✅ [Weights & Biases](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/wandb) (<a href="https://wandb.ai/" target="_blank">↗</a>)
- ✅ [Arize AI](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/arize) (<a href="https://arize.com/" target="_blank">↗</a>)
- ✅ [Raindrop AI](https://github.com/KoshiHQ/GenOps-AI/tree/main/docs/raindrop-quickstart.md) (<a href="https://www.raindrop.ai/" target="_blank">↗</a>)
- ✅ [PostHog](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/posthog) (<a href="https://posthog.com/" target="_blank">↗</a>)
- ✅ [Langfuse](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/langfuse) (<a href="https://langfuse.com/" target="_blank">↗</a>)
- ✅ [AutoGen](https://github.com/KoshiHQ/GenOps-AI/tree/main/docs/integrations/autogen.md) (<a href="https://github.com/microsoft/autogen" target="_blank">↗</a>)
- ✅ [Dust](https://github.com/KoshiHQ/GenOps-AI/tree/main/docs/dust-quickstart.md) (<a href="https://dust.tt/" target="_blank">↗</a>)
- ✅ [Flowise](https://github.com/KoshiHQ/GenOps-AI/tree/main/docs/flowise-quickstart.md) (<a href="https://flowiseai.com/" target="_blank">↗</a>)
- ✅ [Griptape](https://github.com/KoshiHQ/GenOps-AI/tree/main/docs/griptape-quickstart.md) (<a href="https://www.griptape.ai/" target="_blank">↗</a>)
- ✅ [SkyRouter](https://github.com/KoshiHQ/GenOps-AI/tree/main/docs/skyrouter-quickstart.md) (<a href="https://skyrouter.ai/" target="_blank">↗</a>)
- ✅ [Databricks Unity Catalog](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/databricks_unity_catalog) (<a href="https://docs.databricks.com/en/data-governance/unity-catalog/index.html" target="_blank">↗</a>)
- ☐ ElevenLabs (<a href="https://elevenlabs.io/" target="_blank">↗</a>)
- ☐ Deepgram (<a href="https://deepgram.com/" target="_blank">↗</a>)
- ☐ OpenAI Whisper (<a href="https://openai.com/research/whisper" target="_blank">↗</a>)
- ☐ Descript (<a href="https://www.descript.com/" target="_blank">↗</a>)
- ☐ AssemblyAI (<a href="https://www.assemblyai.com/" target="_blank">↗</a>)
- ☐ Twilio ConversationRelay (<a href="https://www.twilio.com/docs/voice/conversationrelay" target="_blank">↗</a>)

---

### 🏗️ Platform & Infrastructure
- ✅ [Kubernetes](https://github.com/KoshiHQ/GenOps-AI/tree/main/docs/kubernetes-getting-started.md) (<a href="https://kubernetes.io/" target="_blank">↗</a>)
- ✅ [OpenTelemetry Collector](https://github.com/KoshiHQ/GenOps-AI/tree/main/observability) (<a href="https://opentelemetry.io/docs/collector/" target="_blank">↗</a>)
- ✅ [Datadog](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/observability/datadog_integration.py) (<a href="https://www.datadoghq.com/" target="_blank">↗</a>)
- ✅ [Grafana](https://github.com/KoshiHQ/GenOps-AI/tree/main/observability/grafana) (<a href="https://grafana.com/" target="_blank">↗</a>)
- ✅ [Loki](https://github.com/KoshiHQ/GenOps-AI/tree/main/observability/loki-config.yaml) (<a href="https://grafana.com/oss/loki/" target="_blank">↗</a>)
- ✅ [Honeycomb](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/observability/honeycomb_integration.py) (<a href="https://www.honeycomb.io/" target="_blank">↗</a>)
- ✅ [Prometheus](https://github.com/KoshiHQ/GenOps-AI/tree/main/observability/prometheus.yml) (<a href="https://prometheus.io/" target="_blank">↗</a>)
- ✅ [Tempo](https://github.com/KoshiHQ/GenOps-AI/tree/main/observability/tempo-config.yaml) (<a href="https://grafana.com/oss/tempo/" target="_blank">↗</a>)
- ☐ Docker (<a href="https://www.docker.com/" target="_blank">↗</a>)
- ☐ AWS Lambda (<a href="https://aws.amazon.com/lambda/" target="_blank">↗</a>)
- ☐ Google Cloud Run (<a href="https://cloud.google.com/run" target="_blank">↗</a>)
- ☐ Azure Functions (<a href="https://azure.microsoft.com/en-us/products/functions/" target="_blank">↗</a>)
- ☐ New Relic (<a href="https://newrelic.com/" target="_blank">↗</a>)
- ☐ Jaeger (<a href="https://www.jaegertracing.io/" target="_blank">↗</a>)
- ☐ SigNoz (<a href="https://signoz.io/" target="_blank">↗</a>)
- ☐ OpenCost (<a href="https://www.opencost.io/" target="_blank">↗</a>)
- ☐ Finout (<a href="https://www.finout.io/" target="_blank">↗</a>)
- ☐ CloudZero (<a href="https://www.cloudzero.com/" target="_blank">↗</a>)
- ☐ AWS Cost Explorer (<a href="https://aws.amazon.com/aws-cost-management/" target="_blank">↗</a>)
- ☐ GCP Billing (<a href="https://cloud.google.com/billing/docs" target="_blank">↗</a>)
- ☐ Azure Cost Management (<a href="https://azure.microsoft.com/en-us/products/cost-management/" target="_blank">↗</a>)
- ☐ Segment (<a href="https://segment.com/" target="_blank">↗</a>)
- ☐ Amplitude (<a href="https://amplitude.com/" target="_blank">↗</a>)
- ☐ Mixpanel (<a href="https://mixpanel.com/" target="_blank">↗</a>)
- ☐ OPA (Open Policy Agent) (<a href="https://www.openpolicyagent.org/" target="_blank">↗</a>)
- ☐ Kyverno (<a href="https://kyverno.io/" target="_blank">↗</a>)
- ☐ Cloud Custodian (<a href="https://cloudcustodian.io/" target="_blank">↗</a>)
- ☐ HashiCorp Sentinel (<a href="https://www.hashicorp.com/sentinel" target="_blank">↗</a>)
- ☐ Datadog Cloud Security (<a href="https://www.datadoghq.com/product/cloud-security-management/" target="_blank">↗</a>)
- ☐ Azure Policy (<a href="https://azure.microsoft.com/en-us/products/policy/" target="_blank">↗</a>)
- ☐ AWS Config (<a href="https://aws.amazon.com/config/" target="_blank">↗</a>)
- ☐ BigQuery (<a href="https://cloud.google.com/bigquery" target="_blank">↗</a>)
- ☐ Snowflake (<a href="https://www.snowflake.com/" target="_blank">↗</a>)
- ☐ AWS S3 (<a href="https://aws.amazon.com/s3/" target="_blank">↗</a>)
- ☐ GCS (<a href="https://cloud.google.com/storage" target="_blank">↗</a>)
- ☐ Azure Blob (<a href="https://azure.microsoft.com/en-us/products/storage/blobs/" target="_blank">↗</a>)
- ☐ Splunk (<a href="https://www.splunk.com/" target="_blank">↗</a>)
- ☐ Elastic (<a href="https://www.elastic.co/" target="_blank">↗</a>)

---

## 🚀 Ready for Production

### **Team Collaboration**
Share insights and optimize together:
- **Cost transparency** — Everyone sees what AI requests actually cost
- **Performance comparison** — Compare models and prompts across the team
- **Debugging support** — Help teammates troubleshoot AI issues faster
- **Best practices sharing** — Learn what works from your team's real usage

### **Scales with Your Growth**
Built to grow from individual to organization:
- **Individual projects** — Track your personal AI usage and costs
- **Team visibility** — Share insights without enterprise overhead
- **Department adoption** — Proven patterns that work at scale
- **Organization readiness** — When you need more, we're ready to help

---

## 🤝 **Community & Support**

### **Contributing**
We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup and testing guidelines
- Code standards and review process
- Community guidelines and code of conduct

### **Getting Help**
- 📖 **Documentation**: [GitHub Docs](https://github.com/KoshiHQ/GenOps-AI/tree/main/docs)
- 📊 **Performance Guide**: [Performance Benchmarking](https://github.com/KoshiHQ/GenOps-AI/blob/main/docs/performance-benchmarking.md)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)
- 🐛 **Issues**: [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)

---

## 📄 **License**

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---

## 🌟 **Why GenOps Framework?**

**Because great AI tools shouldn't run without human oversight and control.**

- **vs Routing tools**: We don't replace routing — we add governance controls and human oversight to it
- **vs Monitoring platforms**: We don't replace monitoring — we add AI governance actions and policy enforcement to it  
- **vs Analytics dashboards**: We don't replace analytics — we add human-controlled budget enforcement to it
- **vs Build-it-yourself**: Standard OpenTelemetry approach with governance extensions instead of custom control scripts

**The only platform that adds cross-stack AI governance WITHOUT replacing your existing tools.**

*When you're ready to scale AI governance across larger teams, GenOps provides the control foundation for unified human oversight and policy enforcement platforms.*

---

## 🤝 **Community & Quick Wins**

**New to open source?** Start here:
- 🐛 [Good first issues](https://github.com/KoshiHQ/GenOps-AI/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) - Perfect for newcomers
- 📚 [Documentation improvements](https://github.com/KoshiHQ/GenOps-AI/issues?q=is%3Aissue+is%3Aopen+label%3Adocumentation) - Help others learn
- 🔧 [Help fix our CI tests!](https://github.com/KoshiHQ/GenOps-AI/issues?q=is%3Aissue+is%3Aopen+label%3Aci-fix) - Great for contributors who love debugging

**5-minute contributions welcome!** Every small improvement helps the community grow.

**Looking for bigger challenges?**
- 🏗️ [Provider integrations](https://github.com/KoshiHQ/GenOps-AI/issues?q=is%3Aissue+is%3Aopen+label%3Aprovider) - Add new provider integrations
- 📊 [Dashboard templates](https://github.com/KoshiHQ/GenOps-AI/issues?q=is%3Aissue+is%3Aopen+label%3Adashboard) - Pre-built observability dashboards
- 🤖 [Cross-stack tracking patterns](https://github.com/KoshiHQ/GenOps-AI/issues?q=is%3Aissue+is%3Aopen+label%3Agovernance) - Real-world cost tracking scenarios

---

## 🚀 **Project Status & Contributing**

GenOps is actively developed with comprehensive cross-stack AI tracking features ready for production use:

### ✅ **Current Status**
- ✅ **Core functionality**: Security scans pass, package installation works
- ✅ **Production examples**: All cost tracking scenarios and integrations functional
- ✅ **OpenTelemetry compliance**: Standard OTLP telemetry export working
- 🤝 **Community contributions welcome**: [See open issues](https://github.com/KoshiHQ/GenOps-AI/issues) for opportunities

### 🆘 **Need Help?**
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
  <p><strong>Ready to connect your AI tools without the custom scripts?</strong></p>
  
  ```bash
  pip install genops
  ```
  
  <p>⭐ <strong>Star us on GitHub</strong> if you find GenOps AI useful!</p>
  
  [![GitHub stars](https://img.shields.io/github/stars/KoshiHQ/GenOps-AI?style=social)](https://github.com/KoshiHQ/GenOps-AI/stargazers)
</div>