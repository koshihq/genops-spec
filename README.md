<p align="center">
  <img width="500" src="./assets/brand/genops-logo-optimized.jpg" alt="GenOps: Open Runtime Governance for AI Systems" style="max-width: 100%;">
</p>

# GenOps: Runtime Authority Control for AI Systems

GenOps extends <a href="https://opentelemetry.io" target="_blank">OpenTelemetry</a> with **runtime authority control** for production AI workloads, the missing control plane that enables organizations to deploy autonomous AI systems with governance built into the infrastructure.

For comprehensive operational foundations and architectural context, see: 📄 [**GenOps Technical Whitepaper**](https://github.com/koshihq/genops-whitepaper/)
<br><br>

<div align="center">
  
  [![GitHub stars](https://img.shields.io/github/stars/koshihq/genops-spec?style=social)](https://github.com/koshihq/genops-spec/stargazers)
  [![CI Status](https://img.shields.io/github/actions/workflow/status/koshihq/genops-spec/ci.yml?branch=main)](https://github.com/koshihq/genops-spec/actions)  
  [![PyPI version](https://badge.fury.io/py/genops.svg)](https://badge.fury.io/py/genops)
  [![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
  [![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
  [![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
  [![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-native-purple.svg)](https://opentelemetry.io/)

</div>

---

## The Problem: The Missing Authority Control Plane

Modern AI infrastructure stacks have sophisticated layers for execution, orchestration, and observability. **But there's no native control plane for authority.**

Your AI systems operate with unlimited authority:

- **No IAM equivalent for AI actions** - AI agents make resource decisions without delegated authority constraints
- **No RBAC for autonomous systems** - AI operations bypass the role-based controls you use everywhere else
- **Observability ≠ Control** - Great dashboards showing what happened, no infrastructure to prevent what shouldn't happen
- **Governance gap** - Policies exist in documents, not in runtime enforcement systems
- **Cost attribution without prevention** - You can measure AI spend but can't constrain it at execution time

**Modern AI stacks need the same authority infrastructure that cloud systems have had for decades.**

## Who This Is For

**If you're building with AI, GenOps is for you:**

**Individual Developers**
- Add runtime budget constraints to AI experiments without changing existing code
- Prevent cost overruns with automatic intervention policies
- Track AI operations with standard observability tools
- Share governance patterns across projects with team-wide visibility

**Team Leads & Senior Engineers**
- Implement authority boundaries for autonomous AI systems
- Enforce budget allocation and usage policies at the infrastructure level
- Get real-time governance telemetry through existing observability platforms
- Enable confident AI adoption with built-in constraint enforcement

**Platform Engineers**
- Extend OpenTelemetry with AI governance semantics using familiar patterns
- Build authority control into infrastructure without application-specific changes
- Support multiple teams with unified policy enforcement across all AI providers
- Scale from individual budget controls to organization-wide authority management

**Start individual. Scale with your team. Grow into your organization.**

---

## The GenOps Solution: Infrastructure-Native Authority Control

GenOps fills the **missing authority control plane** in modern AI infrastructure, using the same patterns that enabled confident cloud adoption:

- **Authority boundaries** that constrain AI system decision-making at the infrastructure level
- **Runtime budget enforcement** with automatic intervention before overruns occur
- **Policy constraints** that prevent unauthorized operations across all AI providers and frameworks
- **Governance telemetry** that extends OpenTelemetry with cost, compliance, and policy semantics

**The FinOps model for AI systems**: Just as cloud infrastructure needed cost control to enable confident scaling, AI systems need authority control to enable confident autonomy.

Because GenOps extends standard OpenTelemetry, it enhances your existing observability stack without replacing tools or creating vendor lock-in.

---

## What GenOps Delivers

**Authority Control Infrastructure**
- Runtime policy enforcement across all AI providers, frameworks, and tools
- Automatic constraint application that prevents unauthorized operations before they execute
- Governance workflows with approval gates for high-impact AI operations
- Cross-provider authority management that works with any AI tool combination

**Budget Authority Enforcement**
- Prevent cost overruns with automatic intervention at configurable budget thresholds
- Real-time cost constraints that operate independently of AI provider billing
- Multi-dimensional budget enforcement: per-team, per-project, per-customer allocation
- Cost attribution and optimization intelligence across all AI operations

**Governance Observability**  
- Authority control telemetry integrated with existing observability platforms (Datadog, Grafana, Honeycomb)
- Cost, policy, and compliance metrics using standard OpenTelemetry semantic conventions
- Real-time governance dashboards that show authority boundaries and constraint violations
- Complete audit trails for compliance reporting and organizational accountability

**Infrastructure-Native Integration**
- Zero-code auto-instrumentation that extends existing AI applications with governance
- Standard OTLP telemetry output compatible with 15+ observability platforms
- Provider-agnostic approach that works across OpenAI, Anthropic, Bedrock, Gemini, and more
- No application changes required - governance becomes part of the infrastructure layer

---

## Authority Control vs. Observability: The Infrastructure Gap

**Traditional AI Observability Platforms:**
- Excellent at showing what happened *after* AI operations complete
- Comprehensive dashboards and cost attribution for analysis
- Alert on problems that have already occurred
- Track usage and performance for optimization

**GenOps Authority Control:**
- **Constrains AI operations before they execute** using infrastructure-level policies
- **Enforces budget and resource limits at runtime** preventing overruns
- **Enables governance boundaries** that autonomous AI systems cannot exceed
- **Prevents** unauthorized operations through authority constraint enforcement

**The critical difference:** Observability tools answer "What happened?" while GenOps answers "What is *allowed* to happen?" 

**Both are essential**: Observability for understanding, GenOps for control. They complement each other in production AI infrastructure.

---

## The Missing Standard: Authority Control for AI Infrastructure

GenOps extends OpenTelemetry semantic conventions into the governance domain, creating the open standard for AI authority control that the industry needs.

**Why this approach matters:**
- **Standards-based foundation** - Building on OpenTelemetry ensures interoperability and prevents vendor lock-in
- **Infrastructure-first architecture** - Authority control becomes part of the platform, not bolted onto applications
- **Community-driven development** - Open-source approach with transparent governance and contribution pathways
- **Provider-agnostic** - Works across all AI providers, frameworks, and observability platforms

**The inevitable evolution**: Just as FinOps emerged to control cloud costs, GenOps provides the authority control that AI systems require for confident autonomous operation.

---

## Works with Your Existing Stack

**Keep the tools you love, add the governance layer that gives you control:**

**Already using LLM providers directly?** GenOps adds runtime authority constraints without changing your application code.

**Already using AI frameworks or routing tools?** GenOps provides governance infrastructure that works across all your existing tools.

**Already using observability platforms?** GenOps extends your existing telemetry with governance semantics and control capabilities.

**The result**: Authority control infrastructure for your entire AI stack without migration complexity or vendor lock-in.

---

## Quick Start

### 1. Install the SDK
```bash
pip install genops
```

### 2. Initialize with governance policies
```python
from genops import GenOps

# Enable runtime authority control with budget constraints
GenOps.init(
    daily_budget_limit=100.0,      # Automatically halt operations at $100/day
    require_approval_over=50.0,    # Authority escalation for requests >$50
    enable_policy_enforcement=True  # Enable real-time constraint checking
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
    "genops.governance.authority_constraint": "budget_limit",
    "genops.provider": "openai",
    "genops.model": "gpt-4",
    "genops.team": "engineering",
    "genops.project": "chatbot"
  }
}
```

**Control your AI systems through your existing observability stack** - Datadog, Grafana, Honeycomb, or any OpenTelemetry-compatible platform.

---

## What You'll See in 5 Minutes

After the 3-step setup above, GenOps immediately provides runtime authority control and constraint enforcement:

### **Runtime Governance Dashboard**
```
AI Governance Status (Real-Time)
┌─────────────────┬──────────┬─────────────┬─────────────┐
│ Team            │ Budget   │ Used        │ Status      │
├─────────────────┼──────────┼─────────────┼─────────────┤
│ engineering     │ $100.00  │ $76.60      │ Active      │
│ product         │ $75.00   │ $15.80      │ Active      │
│ marketing       │ $50.00   │ $48.20      │ Warning     │
└─────────────────┴──────────┴─────────────┴─────────────┘
```

### **Authority Control Actions**
```
Active Governance Constraints
→ High-cost request from 'product' team requires authority escalation ($52.30)
→ Budget threshold exceeded for 'marketing' - operations automatically paused
→ Policy violation detected in 'research' workflow - constraint enforcement applied
```

### **Real-Time Intervention**
```
AUTHORITY CONTROL: Team 'marketing' budget constraint activated
   ├── All AI operations automatically paused by infrastructure
   ├── Budget authority escalation triggered
   ├── 3 pending requests held pending approval workflow
   └── Authority boundary updated - operations resumed
```

**This works with your existing observability tools** - tracking data appears in Datadog traces, Grafana dashboards, or wherever you already monitor your applications.

---

---

## Authority Control Architecture Patterns

**Individual Developer Authority**
Add budget and policy constraints to personal AI projects without changing application code. GenOps provides immediate authority boundary enforcement across all AI operations.

**Team Authority Coordination**  
Unified authority policies across team members with shared budget pools and escalation workflows. Multiple developers operate within consistent constraint frameworks with centralized governance telemetry.

**Organization Authority Infrastructure**
Scale authority control from individual constraints to organization-wide AI governance with enterprise policy frameworks, compliance automation, and centralized constraint management.

**Authority Control Evolution:**
1. **Individual**: Budget and policy constraints for personal AI operations
2. **Team**: Shared authority boundaries with unified constraint enforcement
3. **Organization**: Enterprise authority infrastructure with compliance integration

**The OpenTelemetry foundation ensures authority control scales from individual developers to enterprise infrastructure without platform lock-in.**

---

## Integrations & Support

### AI & LLM Ecosystem
- ✅ [OpenAI](https://github.com/koshihq/genops-spec/tree/main/examples/openai) (<a href="https://openai.com/" target="_blank">↗</a>)
- ✅ [Anthropic](https://github.com/koshihq/genops-spec/tree/main/examples/anthropic) (<a href="https://www.anthropic.com/" target="_blank">↗</a>)
- ✅ [Anyscale](docs/anyscale-quickstart.md) (<a href="https://www.anyscale.com/endpoints" target="_blank">↗</a>)
- ✅ [Google Gemini](https://github.com/koshihq/genops-spec/tree/main/examples/gemini) (<a href="https://deepmind.google/technologies/gemini/" target="_blank">↗</a>)
- ✅ [Hugging Face](https://github.com/koshihq/genops-spec/tree/main/examples/huggingface) (<a href="https://huggingface.co/docs/inference-providers/index" target="_blank">↗</a>)
- ✅ [AWS Bedrock](https://github.com/koshihq/genops-spec/tree/main/examples/bedrock) (<a href="https://aws.amazon.com/bedrock/" target="_blank">↗</a>)
- ✅ [Anyscale](https://github.com/koshihq/genops-spec/tree/main/docs/anyscale-quickstart.md) (<a href="https://www.anyscale.com/endpoints" target="_blank">↗</a>)
- ✅ [OpenRouter](https://github.com/koshihq/genops-spec/tree/main/examples/openrouter) (<a href="https://openrouter.ai/" target="_blank">↗</a>)
- ✅ [LiteLLM](https://github.com/koshihq/genops-spec/tree/main/docs/litellm-quickstart.md) (<a href="https://docs.litellm.ai/" target="_blank">↗</a>)
- ✅ [Vercel AI SDK](https://github.com/koshihq/genops-spec/tree/main/docs/vercel-ai-sdk-quickstart.md) (<a href="https://ai-sdk.dev/" target="_blank">↗</a>)
- ✅ [Helicone](https://github.com/koshihq/genops-spec/tree/main/examples/helicone) (<a href="https://helicone.ai/" target="_blank">↗</a>)
- ✅ [LangChain](https://github.com/koshihq/genops-spec/tree/main/examples/langchain) (<a href="https://python.langchain.com/" target="_blank">↗</a>)
- ✅ [LlamaIndex](https://github.com/koshihq/genops-spec/tree/main/examples/llamaindex) (<a href="https://www.llamaindex.ai/" target="_blank">↗</a>)
- ✅ [Haystack AI](https://github.com/koshihq/genops-spec/tree/main/docs/integrations/haystack.md) (<a href="https://haystack.deepset.ai/" target="_blank">↗</a>)
- ✅ [Kubetorch](https://github.com/koshihq/genops-spec/tree/main/docs/kubetorch-quickstart.md) (<a href="https://www.run.house/" target="_blank">↗</a>)
- ✅ [CrewAI](https://github.com/koshihq/genops-spec/tree/main/docs/quickstart/crewai-quickstart.md) (<a href="https://www.crewai.com/" target="_blank">↗</a>)
- ✅ [Replicate](https://github.com/koshihq/genops-spec/tree/main/examples/replicate) (<a href="https://replicate.com/" target="_blank">↗</a>)
- ✅ [Cohere](https://github.com/koshihq/genops-spec/tree/main/examples/cohere) (<a href="https://cohere.com/" target="_blank">↗</a>)
- ✅ [Mistral](https://github.com/koshihq/genops-spec/tree/main/examples/mistral) (<a href="https://mistral.ai/" target="_blank">↗</a>)
- ✅ [Ollama](https://github.com/koshihq/genops-spec/tree/main/examples/ollama) (<a href="https://ollama.com/" target="_blank">↗</a>)
- ✅ [Perplexity AI](https://github.com/koshihq/genops-spec/tree/main/examples/perplexity) (<a href="https://www.perplexity.ai/" target="_blank">↗</a>)
- ✅ [Together AI](https://github.com/koshihq/genops-spec/tree/main/examples/together) (<a href="https://www.together.ai/" target="_blank">↗</a>)
- ✅ [Fireworks AI](https://github.com/koshihq/genops-spec/tree/main/examples/fireworks) (<a href="https://fireworks.ai/" target="_blank">↗</a>)
- ✅ [Traceloop + OpenLLMetry](https://github.com/koshihq/genops-spec/tree/main/examples/traceloop) (<a href="https://traceloop.com/" target="_blank">↗</a>)
- ✅ [PromptLayer](https://github.com/koshihq/genops-spec/tree/main/examples/promptlayer) (<a href="https://promptlayer.com/" target="_blank">↗</a>)
- ✅ [Weights & Biases](https://github.com/koshihq/genops-spec/tree/main/examples/wandb) (<a href="https://wandb.ai/" target="_blank">↗</a>)
- ✅ [Arize AI](https://github.com/koshihq/genops-spec/tree/main/examples/arize) (<a href="https://arize.com/" target="_blank">↗</a>)
- ✅ [Raindrop AI](https://github.com/koshihq/genops-spec/tree/main/docs/raindrop-quickstart.md) (<a href="https://www.raindrop.ai/" target="_blank">↗</a>)
- ✅ [PostHog](https://github.com/koshihq/genops-spec/tree/main/examples/posthog) (<a href="https://posthog.com/" target="_blank">↗</a>)
- ✅ [Langfuse](https://github.com/koshihq/genops-spec/tree/main/examples/langfuse) (<a href="https://langfuse.com/" target="_blank">↗</a>)
- ✅ [AutoGen](https://github.com/koshihq/genops-spec/tree/main/docs/integrations/autogen.md) (<a href="https://github.com/microsoft/autogen" target="_blank">↗</a>)
- ✅ [Dust](https://github.com/koshihq/genops-spec/tree/main/docs/dust-quickstart.md) (<a href="https://dust.tt/" target="_blank">↗</a>)
- ✅ [Flowise](https://github.com/koshihq/genops-spec/tree/main/docs/flowise-quickstart.md) (<a href="https://flowiseai.com/" target="_blank">↗</a>)
- ✅ [Griptape](https://github.com/koshihq/genops-spec/tree/main/docs/griptape-quickstart.md) (<a href="https://www.griptape.ai/" target="_blank">↗</a>)
- ✅ [SkyRouter](https://github.com/koshihq/genops-spec/tree/main/docs/skyrouter-quickstart.md) (<a href="https://skyrouter.ai/" target="_blank">↗</a>)
- ✅ [Databricks Unity Catalog](https://github.com/koshihq/genops-spec/tree/main/examples/databricks_unity_catalog) (<a href="https://docs.databricks.com/en/data-governance/unity-catalog/index.html" target="_blank">↗</a>)
- ✅ [MLflow](https://github.com/koshihq/genops-spec/tree/main/docs/mlflow-quickstart.md) (<a href="https://mlflow.org/" target="_blank">↗</a>)
- ☐ ElevenLabs (<a href="https://elevenlabs.io/" target="_blank">↗</a>)
- ☐ Deepgram (<a href="https://deepgram.com/" target="_blank">↗</a>)
- ☐ OpenAI Whisper (<a href="https://openai.com/research/whisper" target="_blank">↗</a>)
- ☐ Descript (<a href="https://www.descript.com/" target="_blank">↗</a>)
- ☐ AssemblyAI (<a href="https://www.assemblyai.com/" target="_blank">↗</a>)
- ☐ Twilio ConversationRelay (<a href="https://www.twilio.com/docs/voice/conversationrelay" target="_blank">↗</a>)

---

### Platform & Infrastructure
- ✅ [Kubernetes](https://github.com/koshihq/genops-spec/tree/main/docs/kubernetes-getting-started.md) (<a href="https://kubernetes.io/" target="_blank">↗</a>)
- ✅ [OpenTelemetry Collector](https://github.com/koshihq/genops-spec/tree/main/docs/otel-collector-quickstart.md) (<a href="https://opentelemetry.io/docs/collector/" target="_blank">↗</a>)
- ✅ [Datadog](https://github.com/koshihq/genops-spec/tree/main/docs/datadog-quickstart.md) (<a href="https://www.datadoghq.com/" target="_blank">↗</a>)
- ✅ [Grafana](https://github.com/koshihq/genops-spec/tree/main/docs/grafana-quickstart.md) (<a href="https://grafana.com/" target="_blank">↗</a>)
- ✅ [Loki](https://github.com/koshihq/genops-spec/tree/main/docs/loki-quickstart.md) (<a href="https://grafana.com/oss/loki/" target="_blank">↗</a>)
- ✅ [Honeycomb](https://github.com/koshihq/genops-spec/tree/main/docs/honeycomb-quickstart.md) (<a href="https://www.honeycomb.io/" target="_blank">↗</a>)
- ✅ [Cribl](https://github.com/koshihq/genops-spec/tree/main/docs/integrations/cribl.md) (<a href="https://cribl.io" target="_blank">↗</a>)
- ✅ [Prometheus](https://github.com/koshihq/genops-spec/tree/main/docs/prometheus-quickstart.md) (<a href="https://prometheus.io/" target="_blank">↗</a>)
- ✅ [Tempo](https://github.com/koshihq/genops-spec/tree/main/docs/tempo-quickstart.md) (<a href="https://grafana.com/oss/tempo/" target="_blank">↗</a>)
- ✅ [Splunk](https://github.com/koshihq/genops-spec/tree/main/docs/splunk-quickstart.md) (<a href="https://www.splunk.com/" target="_blank">↗</a>)
- ✅ [Elastic](https://github.com/koshihq/genops-spec/tree/main/docs/quickstarts/elastic-quickstart.md) (<a href="https://www.elastic.co/" target="_blank">↗</a>)
- ✅ [Collibra](https://github.com/koshihq/genops-spec/tree/main/docs/quickstarts/collibra-quickstart.md) (<a href="https://www.collibra.com/" target="_blank">↗</a>)
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

---

## Ready for Production

### **Team Collaboration**
Share insights and optimize together:
- **Cost transparency** - Everyone sees what AI requests actually cost
- **Performance comparison** - Compare models and prompts across the team
- **Debugging support** - Help teammates troubleshoot AI issues faster
- **Best practices sharing** - Learn what works from your team's real usage

### **Scales with Your Growth**
Built to grow from individual to organization:
- **Individual projects** - Track your personal AI usage and costs
- **Team visibility** - Share insights without enterprise overhead
- **Department adoption** - Proven patterns that work at scale
- **Organization readiness** - When you need more, we're ready to help

---

## **Community & Support**

### **Contributing**
We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup and testing guidelines
- Code standards and review process
- Community guidelines and code of conduct

### **Getting Help**
- **Documentation**: [GitHub Docs](https://github.com/koshihq/genops-spec/tree/main/docs)
- **Performance Guide**: [Performance Benchmarking](https://github.com/koshihq/genops-spec/blob/main/docs/performance-benchmarking.md)
- **Discussions**: [GitHub Discussions](https://github.com/koshihq/genops-spec/discussions)
- **Issues**: [GitHub Issues](https://github.com/koshihq/genops-spec/issues)

---

## **License**

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---

## **Why GenOps Framework?**

**Because autonomous AI systems need the same authority infrastructure that cloud systems have.**

- **vs Routing tools**: We don't replace routing, we add authority constraints to the infrastructure layer
- **vs Monitoring platforms**: We don't replace monitoring, we add governance telemetry and constraint enforcement
- **vs Analytics dashboards**: We don't replace analytics, we add runtime budget and policy enforcement
- **vs Build-it-yourself**: Standard OpenTelemetry approach with governance extensions instead of custom constraint systems

**The only platform that adds authority control infrastructure WITHOUT replacing your existing AI tools.**

*When you're ready to scale authority control across larger teams, GenOps provides the governance infrastructure foundation for enterprise AI systems with confidence and constraints.*

---

## **Community & Quick Wins**

**New to open source?** Start here:
- [Good first issues](https://github.com/koshihq/genops-spec/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) - Perfect for newcomers
- [Documentation improvements](https://github.com/koshihq/genops-spec/issues?q=is%3Aissue+is%3Aopen+label%3Adocumentation) - Help others learn
- [Help fix our CI tests!](https://github.com/koshihq/genops-spec/issues?q=is%3Aissue+is%3Aopen+label%3Aci-fix) - Great for contributors who love debugging

**5-minute contributions welcome!** Every small improvement helps the community grow.

**Looking for bigger challenges?**
- [Provider integrations](https://github.com/koshihq/genops-spec/issues?q=is%3Aissue+is%3Aopen+label%3Aprovider) - Add new provider integrations
- [Dashboard templates](https://github.com/koshihq/genops-spec/issues?q=is%3Aissue+is%3Aopen+label%3Adashboard) - Pre-built observability dashboards
- [Cross-stack tracking patterns](https://github.com/koshihq/genops-spec/issues?q=is%3Aissue+is%3Aopen+label%3Agovernance) - Real-world cost tracking scenarios

---

## **Project Status & Contributing**

GenOps is actively developed with comprehensive cross-stack AI tracking features ready for production use:

### **Current Status**
- **Core functionality**: Security scans pass, package installation works
- **Production examples**: All cost tracking scenarios and integrations functional
- **OpenTelemetry compliance**: Standard OTLP telemetry export working
- **Community contributions welcome**: [See open issues](https://github.com/koshihq/genops-spec/issues) for opportunities

### **Need Help?**
- **Questions**: [GitHub Discussions](https://github.com/koshihq/genops-spec/discussions)
- **Issues**: [GitHub Issues](https://github.com/koshihq/genops-spec/issues)
- **Contributing**: [Contributing Guide](CONTRIBUTING.md)

---

## Contributors

Thanks goes to these wonderful people who have contributed to GenOps:

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

---

## **Trademark & Brand Guidelines**

### **GenOps Trademark Usage**

The "GenOps" name and associated branding are trademarks used to identify this project and its official implementations.

**✅ Acceptable Use:**
- Referring to this project in documentation, blog posts, or presentations
- Building integrations or extensions that work with GenOps
- Using "Built with GenOps" or "Powered by GenOps" attributions
- Community projects that extend or integrate with GenOps functionality

**❌ Prohibited Use:**
- Using "GenOps" in the name of competing commercial AI governance products
- Creating confusion about official vs. community implementations  
- Using GenOps branding for unrelated products or services
- Implying official endorsement without permission

**License Note:** The GenOps code is licensed under Apache 2.0, but trademark rights are separate from code rights. You're free to use, modify, and distribute the code under Apache 2.0, but please respect our trademark guidelines when naming your projects or products.

For questions about trademark usage, please open an issue or contact the maintainers.

---

## **Legal & Licensing**

- **Code License**: [Apache License 2.0](LICENSE) - Permissive open source license
- **Contributor Agreement**: All contributions require [DCO sign-off](CONTRIBUTING.md#developer-certificate-of-origin-dco)
- **Copyright**: Copyright © 2024 GenOps Contributors
- **Trademark**: "GenOps" and associated marks are trademarks of the project maintainers

---

<div align="center">
  <p><strong>Ready to connect your AI tools without the custom scripts?</strong></p>
  
  ```bash
  pip install genops
  ```
  
  <p><strong>Star us on GitHub</strong> if you find GenOps useful!</p>
  
  [![GitHub stars](https://img.shields.io/github/stars/koshihq/genops-spec?style=social)](https://github.com/koshihq/genops-spec/stargazers)
</div>