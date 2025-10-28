# 🛡️ GenOps AI Governance Scenarios

**Real-world examples showing how GenOps AI solves critical AI governance problems.**

Instead of abstract concepts, these scenarios demonstrate concrete business value through end-to-end governance automation.

---

## 🚨 **1. Prevent AI Budget Overruns** (`budget_enforcement.py`)

**The Problem**: Your OpenAI bill exploded from $500 to $5,000 because of runaway batch jobs.

**The Solution**: Automatic budget limits with real-time enforcement.

```bash
python examples/governance_scenarios/budget_enforcement.py
```

**What You'll See**:
- ✅ Operations within budget proceed normally
- 🚫 Expensive operations automatically blocked
- ⚠️ Budget warnings with alert notifications
- 📊 Complete cost telemetry for attribution

---

## 🛡️ **2. Block Inappropriate Content** (`content_filtering.py`)

**The Problem**: Your AI assistant generated inappropriate responses, creating PR nightmares.

**The Solution**: Real-time content policies with automatic blocking.

```bash
python examples/governance_scenarios/content_filtering.py
```

**What You'll See**:
- 🚫 Blocked inappropriate requests before they reach AI providers
- ✅ Safe content proceeding through normal workflows
- 📝 Complete audit trail of all content filtering decisions
- 🔧 Customizable content policies for different use cases

---

## 📊 **3. Cost Per Customer Attribution** (`customer_attribution.py`)

**The Problem**: Finance wants to know AI costs per customer but you can't track it.

**The Solution**: Automatic cost attribution with customer-level reporting.

```bash
python examples/governance_scenarios/customer_attribution.py
```

**What You'll See**:
- 💰 Real-time cost tracking per customer, team, and project
- 📈 Multi-tenant cost attribution across your application
- 🔄 Automatic chargeback calculations
- 📊 Dashboards showing cost trends by customer segment

---

## 🔍 **4. Compliance Audit Trail** (`compliance_audit.py`)

**The Problem**: Regulators want complete audit trails of AI decisions and evaluations.

**The Solution**: Automated compliance logging with evaluation metrics.

```bash
python examples/governance_scenarios/compliance_audit.py
```

**What You'll See**:
- 📋 Automatic evaluation scoring and thresholds
- 🔍 Complete audit trails for regulatory compliance
- ✅ Pass/fail tracking for AI quality metrics
- 📊 Compliance dashboards and reporting

---

## 🚀 **Quick Start**

### 1. Install Dependencies

```bash
# Install GenOps AI
pip install -e .

# For OpenAI examples
pip install openai

# For Anthropic examples  
pip install anthropic
```

### 2. Set API Keys (Optional)

```bash
# To see real API integration
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### 3. Run Any Scenario

```bash
# Budget enforcement
python examples/governance_scenarios/budget_enforcement.py

# Content filtering  
python examples/governance_scenarios/content_filtering.py

# Customer attribution
python examples/governance_scenarios/customer_attribution.py

# Compliance audit
python examples/governance_scenarios/compliance_audit.py
```

---

## 📊 **What Makes These Examples Special**

### ✅ **Complete End-to-End Scenarios**
Not just code snippets - full working examples with realistic business problems and solutions.

### ✅ **Real Business Value**
Each scenario solves a concrete problem that costs companies real money or creates real risk.

### ✅ **Production-Ready Patterns** 
Examples show proper error handling, logging, telemetry, and integration patterns.

### ✅ **OpenTelemetry Native**
All governance data exports to your existing observability platforms (Datadog, Honeycomb, etc.).

---

## 🎯 **Business Impact**

These scenarios demonstrate how GenOps AI delivers:

| Problem | Impact Without GenOps | Solution With GenOps |
|---------|----------------------|---------------------|
| **Budget Overruns** | $500 → $5,000 surprise bills | Automatic limits prevent overruns |
| **Inappropriate Content** | PR disasters, brand damage | Real-time content filtering |
| **Cost Attribution** | No visibility into customer costs | Detailed cost attribution by customer |
| **Compliance** | Manual audit processes | Automated compliance logging |

---

## 🔗 **Integration with Your Stack**

These examples work with your existing tools:

- **🔍 Observability**: Datadog, Honeycomb, New Relic, Grafana
- **🤖 AI Providers**: OpenAI, Anthropic, AWS Bedrock, Google Gemini  
- **📊 Dashboards**: Any OpenTelemetry-compatible platform
- **🚨 Alerting**: PagerDuty, Slack, email via your observability platform

---

## 📚 **Next Steps**

1. **Run the scenarios** to see GenOps AI governance in action
2. **Adapt the policies** to your specific business requirements  
3. **Set up OpenTelemetry** integration with your observability platform
4. **Configure alerting** for budget warnings and policy violations
5. **Scale across your organization** with custom governance policies

**Learn more**: [GitHub Docs](https://github.com/KoshiHQ/GenOps-AI/tree/main/docs)