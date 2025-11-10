<p align="center">
  <img width="500" src="./assets/brand/genops-logo-optimized.jpg" alt="GenOps: Open Runtime Governance for AI Systems" style="max-width: 100%;">
</p>

# 🧭 GenOps: Connect Your AI Tools Without the DIY Scripting

GenOps is the open-source framework that connects all your existing AI tools and LLM workloads, built on [OpenTelemetry](https://opentelemetry.io) standards.

**Think of it as OpenTelemetry for AI**: standard telemetry that gives you cross-stack tracking of usage + costs across any combination of AI tools, providers, and observability platforms.

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

## 🚨 The Problem: Great AI Tools, BUT Lacking Cross-Stack Tracking

You're using the best AI tools — LLM providers, AI frameworks, routing services, vector databases. But when your manager asks where all the AI money is going across your entire stack...

Sound familiar?

- 🏗️ **Siloed tools** — LLM provider dashboards, framework logs, routing metrics, vector database stats, but no unified view
- 💸 **Scattered costs** — AI spend across multiple providers and services with no unified tracking
- 📊 **No team visibility** — Great individual tools, but no cross-stack tracking for your entire AI stack
- ⚖️ **Manual reporting** — Building custom scripts to answer "how much did we spend on what?"
- 🤷‍♂️ **DIY dashboards** — Each category of tool has its own metrics, but you're building glue code to connect them

The result: You have best-in-class AI tools but you're writing custom code to connect them.

**You need cross-stack tracking that works with the tools you already love.**

## 👥 Who This Is For

**If you're building with AI, GenOps is for you:**

**🧑‍💻 Individual Developers**
- Track your AI costs and usage across all your projects
- Compare model performance and costs to optimize your choices
- Debug AI requests with proper observability and tracing
- Share results with your team without enterprise overhead

**👨‍💼 Team Leads & Senior Engineers**
- Get visibility into your team's AI spend and usage patterns
- Help your team make better model choices based on real data
- Show management exactly where AI budget is going
- Become the AI expert your company relies on

**🛠️ Platform Engineers**
- Integrate AI governance into existing observability stack
- Support multiple teams with zero additional infrastructure
- Use familiar OpenTelemetry patterns and tools
- Scale from individual developers to organization-wide adoption

**Start individual. Scale with your team. Grow into your organization.**

---

## 💡 The GenOps Solution

GenOps adds the cross-stack tracking layer your AI stack is missing — without replacing the tools you already love:

- **Unified visibility** across LLM providers, AI frameworks, routing tools, vector databases, and more
- **Cost attribution** that spans your entire AI toolchain automatically
- **Team dashboards** with cost breakdowns and usage patterns across all your AI tools
- **Zero custom coding** — standard OpenTelemetry output works with your existing monitoring

Because GenOps uses standard OpenTelemetry, it works with whatever AI tools and observability platforms you're already using. Keep your existing tools, add the cross-stack tracking layer.

---

## ⚙️ What GenOps Delivers

**🏛️ Unified Cross-Stack Tracking**
- See costs and usage across ALL your AI tools in one place
- Automatic tracking that spans LLM providers + AI frameworks + routing services + vector databases
- Team breakdowns and project attribution without custom coding
- Works with any combination of AI tools you're using

**💰 Automatic Cost Tracking**
- Track spending across all providers and frameworks automatically
- See total AI costs regardless of which tools you use
- Per-project, per-team, per-customer attribution across all providers
- Budget monitoring and alerts that cover your entire AI stack

**📊 Team Dashboards & Reporting**  
- Ready-to-use attributions and tagging in your existing observability tools
- Cost breakdowns, usage patterns, and performance metrics
- Answers questions like "what did each team spend last month?"
- Export data for finance and management reports

**🔧 Zero-Friction Integration**
- 30-second setup with auto-instrumentation that detects your AI libraries
- Works with whatever AI tools you already use (LLM providers, AI frameworks, routing services, etc.)
- Standard OpenTelemetry output compatible with 15+ observability platforms
- No vendor lock-in or tool replacement required - enhances your existing stack

---

## 🤝 Works with Your Existing Stack

**Keep the tools you love, add the cross-stack tracking you need:**

**Already using LLM providers directly?** GenOps adds automatic cost tracking and team attribution without changing your code.

**Already using AI frameworks or routing tools?** GenOps connects all your AI tools into unified dashboards.

**Already using observability platforms?** GenOps emits standard OpenTelemetry data that works with your current dashboards.

**The result**: Cross-stack AI tracking across all your tools without migration pain or vendor lock-in.

---

## 📦 Quick Start

### 1. Install the SDK
```bash
pip install genops
```

### 2. Initialize in your app
```python
from genops import GenOps
GenOps.init()  # Auto-detects LLM providers, AI frameworks, routing services in your environment

# Your existing AI code works unchanged
import openai
client = openai.OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)
# ↑ This request is now automatically tracked with cost & usage telemetry
```

### 3. Immediate cost & usage visibility
GenOps automatically captures tracking telemetry:

```json
{
  "trace_id": "abc123",
  "span_name": "openai.chat.completion",
  "attributes": {
    "genops.cost.total": 0.002,
    "genops.cost.currency": "USD",
    "genops.provider": "openai",
    "genops.model": "gpt-4",
    "genops.tokens.input": 8,
    "genops.tokens.output": 12,
    "genops.team": "engineering",
    "genops.project": "chatbot"
  }
}
```

**View data in your existing observability stack** - Datadog, Grafana, Honeycomb, or any OpenTelemetry-compatible platform.

---

## 💡 What You'll See in 5 Minutes

After the 3-step setup above, GenOps immediately provides cross-stack cost and usage tracking:

### **Cost Attribution Dashboard**
```
📊 AI Costs by Team (Last 7 Days)
┌─────────────────┬──────────┬─────────────┐
│ Team            │ Cost     │ Requests    │
├─────────────────┼──────────┼─────────────┤
│ engineering     │ $23.40   │ 1,247       │
│ product         │ $15.80   │ 892         │
│ marketing       │ $8.20    │ 445         │
└─────────────────┴──────────┴─────────────┘
```

### **Cross-Provider Tracking**
```
🔄 Model Usage Across Your Stack
LLM Provider A: $18.30 (62% of total)
LLM Provider B: $12.80 (35% of total)  
Local Models: $0.00 (3% of total)
```

### **Smart Monitoring & Alerts**
```
⚠️  Budget Alert: Team 'engineering' approaching 80% of monthly AI budget
📋 Usage Alert: Unusual spike in LLM requests detected  
✅ Cost Optimization: Suggested model alternatives could save 30%
```

**This works with your existing observability tools** - tracking data appears in Datadog traces, Grafana dashboards, or wherever you already monitor your applications.

---

## 🌟 Featured Integration: Haystack AI

**Comprehensive RAG & Agent Workflow Governance** - Our most complete integration with enterprise-ready patterns.

```python
# Zero-code setup for existing Haystack pipelines
from genops.providers.haystack import auto_instrument
auto_instrument(team="ai-research", project="rag-system")

# Your existing code works unchanged - governance added automatically!
pipeline = Pipeline()
pipeline.add_component("retriever", BM25Retriever(...))
pipeline.add_component("llm", OpenAIGenerator(...))
result = pipeline.run({"query": "What is RAG?"})

# ✅ Automatic cost tracking, budget controls, performance monitoring
```

**What makes this special:**
- **🎯 Specialized patterns**: RAG workflows, agent systems, multi-provider optimization
- **📚 Complete documentation**: [2,900+ line integration guide](docs/integrations/haystack.md) with 7 production-ready examples
- **⚡ 5-minute setup**: From zero to full governance in under 5 minutes
- **🏗️ Production-ready**: Enterprise deployment patterns, monitoring, scaling strategies

**[→ Try the 5-minute Haystack quickstart](docs/integrations/haystack.md)** | **[📊 Browse 7 examples](examples/haystack/)**

---

## 🔧 How Teams Use GenOps Framework

**Individual Developer Pattern**
Start by instrumenting personal AI projects with GenOps telemetry. The framework provides immediate visibility into costs and usage patterns across your development work.

**Team Integration Pattern**  
Share governance data across team members using the same OpenTelemetry foundation. Multiple developers can contribute telemetry to shared observability dashboards.

**Organization Scaling Pattern**
As governance needs grow beyond what the framework can handle alone, teams typically need additional tooling for policy automation, compliance workflows, and enterprise controls.

**Common Adoption Progression:**
1. **Individual**: Implement GenOps instrumentation for personal projects
2. **Team**: Standardize on GenOps telemetry across team members  
3. **Organization**: Framework foundation ready for governance platform integration

**When you need more than instrumentation can provide, the OpenTelemetry foundation scales to enterprise governance platforms.**

---

## 🔌 Integrations & Support

### 🧠 AI & LLM Ecosystem
- ✅ [OpenRouter](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/openrouter) (<a href="https://openrouter.ai/" target="_blank">↗</a>)
- ✅ [OpenAI](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/openai) (<a href="https://openai.com/" target="_blank">↗</a>)
- ✅ [Anthropic](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/anthropic) (<a href="https://www.anthropic.com/" target="_blank">↗</a>)
- ✅ [Hugging Face](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/huggingface) (<a href="https://huggingface.co/docs/inference-providers/index" target="_blank">↗</a>)
- ✅ [AWS Bedrock](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/bedrock) (<a href="https://aws.amazon.com/bedrock/" target="_blank">↗</a>)
- ✅ [Google Gemini](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/gemini) (<a href="https://deepmind.google/technologies/gemini/" target="_blank">↗</a>)
- ✅ [Replicate](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/replicate) (<a href="https://replicate.com/" target="_blank">↗</a>)
- ✅ [LangChain](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/langchain) (<a href="https://python.langchain.com/" target="_blank">↗</a>)
- ✅ [LlamaIndex](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/llamaindex) (<a href="https://www.llamaindex.ai/" target="_blank">↗</a>)
- ✅ [Haystack AI](https://github.com/KoshiHQ/GenOps-AI/tree/main/docs/integrations/haystack.md) (<a href="https://haystack.deepset.ai/" target="_blank">↗</a>) - Complete RAG & agent workflow governance
- ✅ [Ollama](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/ollama) (<a href="https://ollama.com/" target="_blank">↗</a>)
- ✅ [Cohere](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/cohere) (<a href="https://cohere.com/" target="_blank">↗</a>)
- ✅ [Mistral](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/mistral) (<a href="https://mistral.ai/" target="_blank">↗</a>)
- ✅ [Helicone](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/helicone) (<a href="https://helicone.ai/" target="_blank">↗</a>)
- ✅ [Langfuse](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/langfuse) (<a href="https://langfuse.com/" target="_blank">↗</a>)
- ✅ [Traceloop + OpenLLMetry](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/traceloop) (<a href="https://traceloop.com/" target="_blank">↗</a>)
- ✅ [PromptLayer](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/promptlayer) (<a href="https://promptlayer.com/" target="_blank">↗</a>)
- ✅ [Weights & Biases](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/wandb) (<a href="https://wandb.ai/" target="_blank">↗</a>)
- ✅ [Arize AI](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/arize) (<a href="https://arize.com/" target="_blank">↗</a>)
- ✅ [PostHog](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/posthog) (<a href="https://posthog.com/" target="_blank">↗</a>)
- ✅ [Perplexity AI](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/perplexity) (<a href="https://www.perplexity.ai/" target="_blank">↗</a>)
- ✅ [Together AI](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/together) (<a href="https://www.together.ai/" target="_blank">↗</a>)
- ✅ [Fireworks AI](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/fireworks) (<a href="https://fireworks.ai/" target="_blank">↗</a>)
- ☐ CrewAI (<a href="https://www.crewai.com/" target="_blank">↗</a>)
- ☐ AutoGen (<a href="https://github.com/microsoft/autogen" target="_blank">↗</a>)
- ☐ Dust (<a href="https://dust.tt/" target="_blank">↗</a>)
- ☐ Flowise (<a href="https://flowiseai.com/" target="_blank">↗</a>)
- ☐ Griptape (<a href="https://www.griptape.ai/" target="_blank">↗</a>)
- ☐ SkyRouter (<a href="https://skyrouter.ai/" target="_blank">↗</a>)
- ☐ Databricks Unity Catalog (<a href="https://docs.databricks.com/en/data-governance/unity-catalog/index.html" target="_blank">↗</a>)
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

**Because great AI tools shouldn't require custom glue code to connect them.**

- **vs Routing tools**: We don't replace routing — we add cost tracking and observability to it
- **vs Monitoring platforms**: We don't replace monitoring — we add AI-specific metrics to it  
- **vs Analytics dashboards**: We don't replace analytics — we add unified AI cost data to it
- **vs Build-it-yourself**: Standard OpenTelemetry approach instead of custom integration scripts

**The only framework that adds cross-stack AI tracking WITHOUT replacing your existing tools.**

*When you're ready to scale AI operations across larger teams, the GenOps framework provides the telemetry foundation for unified cost management and reporting platforms.*

---

## 🤝 **Community & Quick Wins**

**New to open source?** Start here:
- 🐛 [Good first issues](https://github.com/KoshiHQ/GenOps-AI/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) - Perfect for newcomers
- 📚 [Documentation improvements](https://github.com/KoshiHQ/GenOps-AI/issues?q=is%3Aissue+is%3Aopen+label%3Adocumentation) - Help others learn
- 🔧 [Help fix our CI tests!](https://github.com/KoshiHQ/GenOps-AI/issues?q=is%3Aissue+is%3Aopen+label%3Aci-fix) - Great for contributors who love debugging

**5-minute contributions welcome!** Every small improvement helps the community grow.

**Looking for bigger challenges?**
- 🏗️ [Provider integrations](https://github.com/KoshiHQ/GenOps-AI/issues?q=is%3Aissue+is%3Aopen+label%3Aprovider) - Add Mistral, Replicate, LlamaIndex support
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