<p align="center">
  <img width="500" src="./assets/brand/genops-logo-optimized.jpg" alt="GenOps: Open Runtime Governance for AI Systems" style="max-width: 100%;">
</p>

# 🧭 GenOps: The Governance Backbone for Your AI Stack

GenOps is the open-source governance layer that works with your existing AI tools and LLM workloads, built on [OpenTelemetry](https://opentelemetry.io) standards.

**Think of it as the OpenTelemetry fabric for AI governance**: standard telemetry that enables unified governance across any combination of AI tools, providers, and observability platforms.

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

## 🚨 The Problem: Great AI Tools, Zero Governance

You're using the best AI tools — routing optimization, trace monitoring, analytics platforms. But when your manager asks about governance, compliance, or cost attribution across all these tools...

Sound familiar?

- 🏗️ **Siloed tools** — some track routing, some track traces, some track usage, but what tracks governance?
- 💸 **Scattered costs** — AI spend across multiple tools with no unified attribution
- 📊 **No team oversight** — Great individual tools, but no governance layer for your team
- ⚖️ **Compliance gaps** — No audit trail that spans your entire AI stack
- 🤷‍♂️ **Policy blind spots** — Each tool has policies, but no unified governance

The result: You have best-in-class AI tools but no governance backbone to tie them together.

**You need a governance layer that works with the tools you already love.**

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

GenOps adds the governance layer your AI stack is missing — without replacing the tools you already love:

- **Unified governance** across routing tools, trace monitoring, analytics platforms, and more
- **Cost attribution** that spans your entire AI toolchain
- **Policy compliance** with audit trails across all your tools
- **Team visibility** into AI usage across your complete stack

Because GenOps uses standard OpenTelemetry, it works with whatever AI tools and observability platforms you're already using. Keep your existing tools, add the governance backbone.

---

## ⚙️ What You Get Out of the Box

**🏛️ Unified Governance Layer**
- See governance across ALL your AI tools in one place
- Cost attribution that spans routing tools + monitoring systems + your direct API calls
- Policy compliance tracking across your complete AI stack
- Audit trails that work with any tool combination

**💰 Cross-Tool Cost Attribution**
- Track spending across routing tools, direct providers, and frameworks
- See total AI costs regardless of which tools you use
- Per-project, per-team, per-customer attribution across all providers
- Budget alerts that cover your entire AI stack

**📋 Policy & Compliance**  
- Consistent governance policies across all your AI tools
- Compliance audit trails that span your complete toolchain
- Data residency tracking across multiple providers and tools
- Risk management for your entire AI operations

**🔧 Zero-Friction Integration**
- 30-second setup with auto-instrumentation
- Works with whatever AI tools you already use
- Standard OpenTelemetry output for any observability platform
- No vendor lock-in or tool replacement required

---

## 🤝 Works with Your Existing Stack

**Keep the tools you love, add the governance you need:**

**Already using routing optimization tools?** GenOps adds cost attribution and policy tracking without changing your setup.

**Already using trace monitoring systems?** GenOps adds governance and compliance data to your existing observability.

**Already using Datadog/Grafana/Honeycomb?** GenOps emits standard OpenTelemetry data that works with your current dashboards.

**The result**: Unified governance across all your AI tools without migration pain or vendor lock-in.

---

## 📦 Quick Start

### 1. Install the SDK
```bash
pip install genops
```

### 2. Initialize in your app
```python
from genops import GenOps
GenOps.init()
```

### 3. Run your app
GenOps automatically collects runtime telemetry and governance signals.  
View data in your existing observability stack or policy engine.

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

## 🔌 Works with Your Existing AI Tools

### Governance Connectors

GenOps adds governance to the AI tools you already use — no need to switch or replace anything.

#### AI Tool Integrations

- ✅ [OpenRouter](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/openrouter) - Add cost attribution and policy tracking to your model routing
- 🚧 **Helicone** - Add governance layer to your LLM traces and observability  
- 🚧 **Traceloop** - Add compliance and audit trails to your LLM monitoring
- 🚧 **PostHog** - Add AI governance to your product analytics

#### Core Provider Support

- ✅ [OpenAI](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/openai) - Direct governance for OpenAI usage
- ✅ [Anthropic](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/anthropic) - Direct governance for Claude usage
- ✅ [LangChain](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/langchain) - Multi-provider governance for LangChain applications

#### Infrastructure Support

- ✅ [Kubernetes](https://github.com/KoshiHQ/GenOps-AI/tree/main/docs/kubernetes-getting-started.md) - Container orchestration governance
- 🚧 **Docker** - Container-level AI governance
- 🚧 **Serverless** - AWS Lambda, Google Cloud Run, Azure Functions

### Community Ecosystem

**Can't find your tool?** GenOps uses standard OpenTelemetry — if your tool supports OTel, it works with GenOps. Plus, we welcome community contributions for new integrations.

### Supported Destinations

GenOps exports standardized telemetry and governance events to your existing stack.

#### Observability & Monitoring

- ✅ [OpenTelemetry Collector](https://github.com/KoshiHQ/GenOps-AI/tree/main/observability) (<a href="https://opentelemetry.io/docs/collector/" target="_blank">↗</a>)
- ✅ [Datadog](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/observability/datadog_integration.py) (<a href="https://www.datadoghq.com/" target="_blank">↗</a>)
- ✅ [Grafana](https://github.com/KoshiHQ/GenOps-AI/tree/main/observability/grafana) (<a href="https://grafana.com/" target="_blank">↗</a>)
- ✅ [Loki](https://github.com/KoshiHQ/GenOps-AI/tree/main/observability/loki-config.yaml) (<a href="https://grafana.com/oss/loki/" target="_blank">↗</a>)
- ✅ [Honeycomb](https://github.com/KoshiHQ/GenOps-AI/tree/main/examples/observability/honeycomb_integration.py) (<a href="https://www.honeycomb.io/" target="_blank">↗</a>)
- ✅ [Prometheus](https://github.com/KoshiHQ/GenOps-AI/tree/main/observability/prometheus.yml) (<a href="https://prometheus.io/" target="_blank">↗</a>)
- ✅ [Tempo](https://github.com/KoshiHQ/GenOps-AI/tree/main/observability/tempo-config.yaml) (<a href="https://grafana.com/oss/tempo/" target="_blank">↗</a>)
- ☐ New Relic (<a href="https://newrelic.com/" target="_blank">↗</a>)
- ☐ Jaeger (<a href="https://www.jaegertracing.io/" target="_blank">↗</a>)
- ☐ SigNoz (<a href="https://signoz.io/" target="_blank">↗</a>)

#### Cost & FinOps Platforms

- ☐ OpenCost (<a href="https://www.opencost.io/" target="_blank">↗</a>)
- ☐ Finout (<a href="https://www.finout.io/" target="_blank">↗</a>)
- ☐ CloudZero (<a href="https://www.cloudzero.com/" target="_blank">↗</a>)
- ☐ AWS Cost Explorer (<a href="https://aws.amazon.com/aws-cost-management/" target="_blank">↗</a>)
- ☐ GCP Billing (<a href="https://cloud.google.com/billing/docs" target="_blank">↗</a>)
- ☐ Azure Cost Management (<a href="https://azure.microsoft.com/en-us/products/cost-management/" target="_blank">↗</a>)
- ☐ Cloudflare Workers AI Analytics (<a href="https://developers.cloudflare.com/workers-ai/" target="_blank">↗</a>)
- ☐ Traceloop (<a href="https://traceloop.com/" target="_blank">↗</a>)
- ☐ OpenLLMetry (<a href="https://github.com/traceloop/openllmetry" target="_blank">↗</a>)

### Policy & Compliance

- ☐ OPA (Open Policy Agent) (<a href="https://www.openpolicyagent.org/" target="_blank">↗</a>)
- ☐ Kyverno (<a href="https://kyverno.io/" target="_blank">↗</a>)
- ☐ Cloud Custodian (<a href="https://cloudcustodian.io/" target="_blank">↗</a>)
- ☐ HashiCorp Sentinel (<a href="https://www.hashicorp.com/sentinel" target="_blank">↗</a>)
- ☐ Rego-compatible policies

### Data & Security Pipelines

- ☐ BigQuery (<a href="https://cloud.google.com/bigquery" target="_blank">↗</a>)
- ☐ Snowflake (<a href="https://www.snowflake.com/" target="_blank">↗</a>)
- ☐ S3 (<a href="https://aws.amazon.com/s3/" target="_blank">↗</a>)
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

## 🌟 **Why GenOps Framework?**

**Because great AI tools deserve a governance backbone.**

- **vs Routing tools**: We don't replace routing — we add governance instrumentation to it
- **vs Trace monitoring**: We don't replace monitoring — we add compliance telemetry to it  
- **vs Analytics platforms**: We don't replace analytics — we add policy tracking to it
- **vs Build-it-yourself**: Standard OpenTelemetry approach for governance instrumentation

**The only framework that adds governance telemetry WITHOUT replacing your existing tools.**

*When you're ready to scale AI governance across larger organizations, the GenOps framework provides the instrumentation foundation for enterprise governance platforms.*

*When you're ready to scale AI governance across larger organizations, GenOps provides the foundation for enterprise control planes and governance workflows.*

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