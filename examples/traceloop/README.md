# Traceloop + OpenLLMetry LLM Observability + GenOps Governance Examples

**🎯 Add enterprise governance to your OpenLLMetry LLM observability in 5 minutes**

This directory contains comprehensive examples demonstrating how GenOps enhances OpenLLMetry with enterprise-grade governance, cost intelligence, and policy enforcement for production AI applications, with optional integration to the Traceloop commercial platform.

---

## 🤔 Why Do I Need This?

If you're building production LLM applications, you're likely facing these challenges:

❌ **Without GenOps Governance:**
- No visibility into LLM costs across teams and projects
- Manual budget tracking and cost attribution
- No policy enforcement or compliance validation
- Limited observability context for business decisions
- Difficult to optimize costs or prevent budget overruns

✅ **With GenOps + OpenLLMetry + Traceloop:**
- **Automatic cost attribution** to teams, projects, and customers
- **Real-time budget enforcement** with policy compliance
- **Enhanced observability** with business context in every trace
- **Cost optimization insights** and recommendations
- **Enterprise governance** for compliance and audit requirements
- **Optional commercial platform** for advanced insights and analytics

---

## 🧠 What is This Integration?

### 🔍 What is OpenLLMetry?
**OpenLLMetry is an open-source observability framework** that extends OpenTelemetry with LLM-specific instrumentation. It captures detailed traces of LLM operations and provides powerful analytics for optimization. **It's Apache 2.0 licensed and vendor-neutral.**

### 🏢 What is Traceloop?
**Traceloop is a commercial platform** built on OpenLLMetry that provides enterprise features like advanced insights, model experimentation, drift detection, and team collaboration tools. **You can use OpenLLMetry without Traceloop**, or upgrade to the commercial platform for additional capabilities.

### 💡 The Perfect Combination
**GenOps + OpenLLMetry + Traceloop** = Complete LLM observability with enterprise governance

- **🔍 Enhanced Observability**: Every OpenLLMetry trace includes governance context (team, project, customer)
- **💰 Cost Intelligence**: Precise cost tracking and attribution integrated with observability
- **🛡️ Governance Integration**: Policy compliance and budget enforcement within observability workflows
- **📊 Business Intelligence**: Cost optimization insights with team-based attribution
- **🎯 Evaluation Governance**: LLM evaluation tracking with cost and compliance oversight
- **🚀 Enterprise Readiness**: Production-grade governance for LLM observability at scale
- **🏭 Commercial Platform**: Optional upgrade to Traceloop for advanced insights and team features

---

## ⚡ Quick Value Assessment (2 minutes)

**Before diving in, let's see if this is right for your team:**

### ✅ Perfect For:
- **Engineering Teams** using or considering OpenLLMetry who need cost visibility and governance
- **FinOps Teams** requiring detailed LLM cost attribution and budget controls
- **Enterprise Organizations** needing compliance tracking and audit trails for AI operations
- **Multi-team Companies** where different teams use LLMs with shared budgets
- **Production AI Applications** requiring cost optimization and governance automation

### 🤔 Consider Alternatives If:
- You have simple, single-developer LLM projects with no cost concerns
- You only need basic cost tracking without detailed observability
- You don't plan to use OpenTelemetry-based observability practices

**📊 Team Size Guidelines:**
- **1-2 developers**: Start with Level 1 examples (basic governance with open-source OpenLLMetry)
- **3-10 developers**: Focus on Level 2 (advanced observability and evaluation)
- **10+ developers**: Implement Level 3 (enterprise governance and consider Traceloop platform)

---

## 🚀 Getting Started

### Phase 1: Before You Start (5 minutes)

**First, ensure you have the prerequisites:**

1. **Python Environment**
   ```bash
   python3 --version  # Ensure Python 3.8+
   ```

2. **AI Provider Account** (choose one)
   - [OpenAI Platform](https://platform.openai.com/api-keys) (recommended for getting started)
   - [Anthropic Console](https://console.anthropic.com/) (alternative option)
   - Any provider you're already using

3. **Optional: Traceloop Platform Account** (for commercial features)
   - Sign up at [app.traceloop.com](https://app.traceloop.com/)
   - Note your API key for commercial platform features

### Phase 2: Installation (1 minute)

```bash
# Install GenOps with Traceloop + OpenLLMetry integration
pip install genops[traceloop]

# This includes:
# - OpenLLMetry (open-source framework)
# - Traceloop SDK (commercial platform integration)
# - GenOps governance enhancements

# Verify installation
python -c "import genops, openllmetry; print('✅ Installation successful')"
```

**Quick Troubleshooting:**
- ❌ `ModuleNotFoundError: No module named 'genops'` → Run `pip install genops[traceloop]` again
- ❌ `ModuleNotFoundError: No module named 'openllmetry'` → Run `pip install openllmetry` directly

### Phase 3: Configuration (2 minutes)

Set up your environment variables:

```bash
# Required: At least one AI provider
export OPENAI_API_KEY="your-openai-api-key"             # If using OpenAI
export ANTHROPIC_API_KEY="your-anthropic-api-key"       # If using Anthropic

# Optional: Traceloop commercial platform (enterprise features)
export TRACELOOP_API_KEY="your-traceloop-api-key"       # From app.traceloop.com
export TRACELOOP_BASE_URL="https://app.traceloop.com"   # Default (optional)
```

**Quick Test:** Verify your setup works:
```bash
# Test OpenLLMetry availability
python -c "import openllmetry; print('✅ OpenLLMetry ready')"

# Test Traceloop SDK (optional)
python -c "from traceloop.sdk import Traceloop; print('✅ Traceloop SDK ready')"
```

### Phase 4: Validation (30 seconds)

**🎯 Run this first** to ensure everything is configured correctly:

```bash
python setup_validation.py
```

**Expected output:** ✅ **Overall Status: PASSED**

**If validation fails:** Check the error messages - they include specific fixes for common issues.

---

## 📚 Learning Path Guide

### 🎯 Your Learning Journey

**Total Time Investment:** 4-6 hours (spread across days/weeks)  
**Immediate Value:** Visible in first 5 minutes  
**Production Ready:** After Level 2 completion  

### Level 1: Getting Started (15 minutes total)
**Goal:** Understand the value and get immediate results  
**When to Use:** Perfect for initial evaluation and proof-of-concept

**Learning Outcomes:**
- ✅ See enhanced OpenLLMetry traces with governance attributes
- ✅ Understand automatic cost attribution and team tracking  
- ✅ Experience zero-code governance integration
- ✅ Get immediate cost visibility for your LLM operations

**Examples:**

**[setup_validation.py](setup_validation.py)** ⭐ *Start here* (30 seconds)
- Comprehensive setup validation with actionable diagnostics
- Verify API keys, connectivity, and basic functionality
- Test governance integration and performance baseline
- Understand open-source vs commercial platform options

**[basic_tracking.py](basic_tracking.py)** (5 minutes)
- Simple LLM operations with enhanced OpenLLMetry tracing
- See governance attributes integrated with observability
- Experience cost attribution and team tracking
- Minimal code changes for maximum governance enhancement

**[auto_instrumentation.py](auto_instrumentation.py)** (5 minutes)  
- Zero-code setup for existing OpenLLMetry applications
- Automatic governance enhancement with no code changes
- Perfect for teams already using OpenLLMetry instrumentation
- Drop-in governance integration that "just works"

**💡 Level 1 Success Criteria:**
- [ ] Validation script shows ✅ **Overall Status: PASSED**
- [ ] You can see cost attribution in OpenTelemetry traces
- [ ] Your existing OpenLLMetry code works with governance
- [ ] You understand the open-source foundation and commercial options

---

### Level 2: Advanced Observability (1 hour total)
**Goal:** Build production-ready evaluation and optimization workflows  
**When to Use:** When you need advanced LLM evaluation and prompt optimization

**Learning Outcomes:**
- ✅ Implement governance-aware LLM evaluation workflows
- ✅ Build cost-optimized prompt management systems
- ✅ Create A/B testing frameworks with governance attribution
- ✅ Establish evaluation pipelines with compliance tracking

**Examples:**

**[traceloop_platform.py](traceloop_platform.py)** (30 minutes)
- Commercial Traceloop platform integration with governance tracking
- Advanced insights and analytics with cost attribution
- Team collaboration features with budget enforcement
- Enterprise-grade observability with governance automation

**[advanced_observability.py](advanced_observability.py)** (30 minutes)
- Advanced OpenLLMetry patterns with cost optimization insights
- Multi-provider observability with unified governance
- Complex workflow tracing with detailed cost analysis
- Optimization recommendations based on usage patterns

**💡 Level 2 Success Criteria:**
- [ ] You can run cost-attributed LLM evaluations
- [ ] Your team can optimize operations based on cost/performance data
- [ ] You have advanced observability with governance tracking
- [ ] You understand commercial platform benefits and upgrade path

---

### Level 3: Enterprise Governance (4+ hours total)
**Goal:** Master production-grade governance for enterprise deployment  
**When to Use:** For production systems requiring enterprise governance and compliance

**Learning Outcomes:**
- ✅ Deploy advanced observability with hierarchical tracing
- ✅ Implement multi-provider governance with unified tracking
- ✅ Build high-availability systems with governance automation
- ✅ Create compliance monitoring and audit systems

**Examples:**

**[production_patterns.py](production_patterns.py)** (4 hours)
- Enterprise-ready deployment patterns and high-availability
- Governance automation with compliance monitoring
- Production monitoring with cost intelligence and alerts
- Disaster recovery and business continuity patterns

**💡 Level 3 Success Criteria:**
- [ ] You can deploy multi-region governance systems
- [ ] Your organization has automated compliance monitoring
- [ ] You have production-grade cost intelligence dashboards
- [ ] You understand enterprise governance patterns

---

## 🏃 Running Examples

### Option 1: Individual Examples (Recommended for Learning)

```bash
# 🎯 Level 1: Getting Started (15 minutes total)
python setup_validation.py      # ⭐ Always start here
python basic_tracking.py        # See governance in action  
python auto_instrumentation.py  # Zero-code integration

# 📊 Level 2: Advanced Observability (1 hour total)
python traceloop_platform.py    # Commercial platform features
python advanced_observability.py # Advanced patterns

# 🏭 Level 3: Enterprise Governance (4+ hours total)
python production_patterns.py   # Enterprise deployment
```

### Option 2: Complete Suite (For Comprehensive Evaluation)

```bash
# Run all examples with validation (~20 minutes active time)
./run_all_examples.sh
```

This script includes progress tracking, error handling, and comprehensive reporting.

---

## 🎯 Industry-Specific Use Cases

### 🏦 Financial Services
- **Compliance:** SOC2, PCI DSS audit trails for all LLM operations
- **Cost Control:** Department-level budget attribution and enforcement
- **Risk Management:** Policy compliance for customer data processing
- **Examples:** Start with `traceloop_platform.py` for compliance tracking

### 🏥 Healthcare
- **HIPAA Compliance:** Encrypted governance attributes and audit logs
- **Cost Attribution:** Patient care vs. research cost separation
- **Quality Assurance:** Evaluation workflows with governance oversight
- **Examples:** Focus on `production_patterns.py` for compliance automation

### 🏢 Enterprise SaaS
- **Customer Attribution:** Per-customer cost tracking and billing
- **Team Governance:** Department-level budget controls and reporting
- **Feature Development:** A/B testing with cost attribution
- **Examples:** `advanced_observability.py` for cost-optimized customer experiences

### 🎓 Research & Education
- **Grant Tracking:** Research project cost attribution and reporting
- **Collaboration:** Multi-team governance with shared resources
- **Evaluation:** Research quality metrics with cost tracking
- **Examples:** `basic_tracking.py` for simple project attribution

---

## 💰 ROI & Business Value

### Small Teams (1-5 developers)
**Investment:** ~2 hours setup  
**Savings:** 20-40% LLM cost reduction through optimization  
**Value:** Clear cost visibility and basic governance

### Growing Teams (5-20 developers)  
**Investment:** ~1 day implementation  
**Savings:** 30-50% cost reduction + 50% faster debugging  
**Value:** Team attribution, budget controls, evaluation workflows

### Enterprise (20+ developers)
**Investment:** ~1 week enterprise deployment  
**Savings:** 40-60% cost reduction + compliance automation  
**Value:** Full governance automation, audit trails, enterprise observability

---

## 🔧 Quick Troubleshooting

### Setup Issues
**❌ "Command not found: python"**
```bash
# On macOS/Linux, try python3
python3 setup_validation.py
```

**❌ "OpenLLMetry not found"**
```bash
# Install OpenLLMetry directly
pip install openllmetry
# Or reinstall with all dependencies
pip install genops[traceloop]
```

**❌ "No LLM provider API keys found"**
```bash
# Verify at least one provider is configured
echo $OPENAI_API_KEY       # Should be set if using OpenAI
echo $ANTHROPIC_API_KEY    # Should be set if using Anthropic
```

### Advanced Troubleshooting
**❌ Governance integration issues:**
```bash
# Enable detailed logging for diagnosis
export GENOPS_LOG_LEVEL=DEBUG
python basic_tracking.py
```

**❌ OpenLLMetry connectivity problems:**
```bash
# Test OpenLLMetry instrumentation
python -c "import openllmetry; openllmetry.instrument(); print('✅ Ready')"
```

---

## 🆘 Need Help?

### 📚 Documentation
- **[5-Minute Quickstart Guide](../../docs/traceloop-quickstart.md)** - Fastest way to get started
- **[Complete Integration Guide](../../docs/integrations/traceloop.md)** - Comprehensive reference
- **[CLAUDE.md](../../CLAUDE.md)** - Development standards and patterns

### 💬 Community Support  
- **[GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)** - Bug reports and feature requests
- **[GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)** - Community questions and sharing

### 🚀 Professional Services
For enterprise deployments, custom integrations, or professional services, contact our team for dedicated support.

---

## 🌟 What's Next?

### After Level 1 (Basic Understanding):
1. **Integrate with your application:** Use patterns from `basic_tracking.py`
2. **Set up team attribution:** Configure governance attributes for your teams
3. **Monitor cost trends:** Watch your observability dashboard for governance insights

### After Level 2 (Advanced Features):
1. **Evaluate Traceloop platform:** Consider commercial platform for advanced insights
2. **Optimize operations:** Use cost intelligence from `advanced_observability.py`
3. **Set up advanced monitoring:** Create governance-aware observability workflows

### After Level 3 (Enterprise Ready):
1. **Production deployment:** Follow `production_patterns.py` guidance
2. **Enterprise integration:** Connect to your existing observability stack
3. **Team training:** Share governance patterns across your organization

---

**🎉 Ready to enhance your OpenLLMetry observability with GenOps governance?**

**Start your journey:** `python setup_validation.py`