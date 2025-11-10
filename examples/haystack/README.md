# Haystack AI + GenOps Examples

**🎯 Complete learning path from 5-minute quickstart to enterprise production deployment**

Welcome to the comprehensive Haystack AI + GenOps integration examples! This directory contains 7 carefully crafted examples that take you from basic pipeline tracking to enterprise-grade governance patterns.

## 🚀 Quick Start

**⏱️ Just want to get started?** Jump to [`basic_pipeline_tracking.py`](basic_pipeline_tracking.py) and follow the 5-minute setup guide.

## 📚 Learning Path Overview

Our examples follow a proven **5-minute → 30-minute → 2-hour** progression designed to maximize your learning efficiency:

```
🎯 5-minute Value → 🏗️ 30-minute Deep Dive → 🚀 2-hour Production Mastery
```

### 🎯 **Phase 1: Quick Start (5 minutes)**
Perfect for getting immediate value and understanding core concepts.

| Example | Description | Time | Use Case |
|---------|-------------|------|----------|
| [`basic_pipeline_tracking.py`](basic_pipeline_tracking.py) | Essential pipeline governance patterns | 5 min | First-time setup, basic Q&A, cost tracking |

**What you'll learn:** Auto-instrumentation, cost tracking, governance attributes, budget awareness

---

### 🏗️ **Phase 2: Specialized Patterns (30 minutes each)**
Deep dive into specific AI workflow patterns with production-ready implementations.

| Example | Description | Time | Use Case |
|---------|-------------|------|----------|
| [`rag_workflow_governance.py`](rag_workflow_governance.py) | RAG pipeline specialization | 30 min | Document Q&A, knowledge bases, retrieval optimization |
| [`agent_workflow_tracking.py`](agent_workflow_tracking.py) | Agent system monitoring | 30 min | Multi-step agents, tool usage, decision tracking |
| [`multi_provider_cost_aggregation.py`](multi_provider_cost_aggregation.py) | Cross-provider optimization | 30 min | Cost analysis, provider selection, optimization |

**What you'll learn:** Specialized workflow patterns, advanced monitoring, cost optimization strategies

---

### 🚀 **Phase 3: Production Mastery (2+ hours each)**
Enterprise-grade patterns for production deployment and advanced governance.

| Example | Description | Time | Use Case |
|---------|-------------|------|----------|
| [`performance_optimization.py`](performance_optimization.py) | Advanced performance patterns | 2 hrs | Caching, parallel processing, load testing |
| [`enterprise_governance_patterns.py`](enterprise_governance_patterns.py) | Multi-tenant governance | 2 hrs | Compliance, audit trails, SLA enforcement |
| [`production_deployment_patterns.py`](production_deployment_patterns.py) | Production deployment | 2 hrs | Kubernetes, monitoring, high availability |

**What you'll learn:** Production deployment, enterprise governance, performance optimization, scalability

---

## 🛠️ Prerequisites & Setup

### **System Requirements**
- **Python**: 3.9+ (3.11+ recommended for best performance)
- **Memory**: 4GB+ RAM (8GB+ for production examples)
- **Storage**: 1GB free space for dependencies

### **Required Dependencies**
```bash
# Core dependencies - required for all examples
pip install genops-ai[haystack] haystack-ai

# AI Provider Dependencies - install for providers you'll use
pip install openai              # For OpenAI models (GPT-4, GPT-3.5, embeddings)
pip install anthropic           # For Claude models
pip install cohere-ai           # For Cohere models
pip install transformers        # For Hugging Face models
```

### **Environment Configuration**
Set up your API keys for the providers you plan to use:

```bash
# OpenAI (most examples)
export OPENAI_API_KEY="sk-your-openai-key-here"

# Anthropic (for provider comparison examples)
export ANTHROPIC_API_KEY="your-anthropic-key-here"

# Cohere (for multi-provider examples)
export COHERE_API_KEY="your-cohere-key-here"
```

### **Quick Validation**
Verify your setup before running examples:

```bash
# Interactive validation with guided setup (recommended)
../../validate

# Or use Python directly
python ../../scripts/validate_setup.py

# Expected: ✅ 95%+ validation score
```

**Alternatively, validate programmatically:**
```python
from genops.providers.haystack import validate_haystack_setup, print_validation_result

result = validate_haystack_setup()
print_validation_result(result)
```

---

## 🎓 Recommended Learning Sequence

### **For First-Time Users** (Total: ~1 hour)
1. **Start Here**: [`basic_pipeline_tracking.py`](basic_pipeline_tracking.py) (5 min)
2. **Choose Your Path**: Pick one specialized pattern based on your use case (30 min)
3. **Explore**: Browse other specialized examples as needed

### **For RAG Applications** (Total: ~1.5 hours)
1. [`basic_pipeline_tracking.py`](basic_pipeline_tracking.py) → Basic concepts
2. [`rag_workflow_governance.py`](rag_workflow_governance.py) → RAG specialization
3. [`performance_optimization.py`](performance_optimization.py) → Production optimization

### **For Agent Systems** (Total: ~1.5 hours)
1. [`basic_pipeline_tracking.py`](basic_pipeline_tracking.py) → Basic concepts
2. [`agent_workflow_tracking.py`](agent_workflow_tracking.py) → Agent patterns
3. [`enterprise_governance_patterns.py`](enterprise_governance_patterns.py) → Production governance

### **For Production Deployment** (Total: ~4+ hours)
1. [`basic_pipeline_tracking.py`](basic_pipeline_tracking.py) → Foundations
2. [`multi_provider_cost_aggregation.py`](multi_provider_cost_aggregation.py) → Cost optimization
3. [`performance_optimization.py`](performance_optimization.py) → Performance tuning
4. [`enterprise_governance_patterns.py`](enterprise_governance_patterns.py) → Governance
5. [`production_deployment_patterns.py`](production_deployment_patterns.py) → Deployment

---

## 🏃‍♂️ Quick Command Reference

### **Run Any Example**
```bash
# Navigate to examples directory
cd examples/haystack

# Run with Python (all examples are self-contained)
python basic_pipeline_tracking.py
python rag_workflow_governance.py
# ... etc
```

### **Validate Your Environment**
```bash
# Quick validation script
python -c "from genops.providers.haystack import validate_haystack_setup, print_validation_result; print_validation_result(validate_haystack_setup())"
```

### **Get Help**
```bash
# Any example with --help shows usage information
python basic_pipeline_tracking.py --help
```

---

## 📊 Example Complexity Matrix

| Example | Lines of Code | Concepts Covered | Production Ready | Time Investment |
|---------|---------------|------------------|------------------|-----------------|
| `basic_pipeline_tracking.py` | 433 | Core patterns | ✅ Basic | 5 minutes |
| `rag_workflow_governance.py` | 485 | RAG specialization | ✅ Production | 30 minutes |
| `agent_workflow_tracking.py` | 631 | Agent workflows | ✅ Production | 30 minutes |
| `multi_provider_cost_aggregation.py` | 725 | Cost optimization | ✅ Production | 30 minutes |
| `performance_optimization.py` | 999 | Advanced patterns | ✅ Enterprise | 2 hours |
| `enterprise_governance_patterns.py` | 681 | Compliance | ✅ Enterprise | 2 hours |
| `production_deployment_patterns.py` | 992 | Deployment | ✅ Enterprise | 2 hours |

**Total**: ~4,900 lines of production-ready code with comprehensive documentation

---

## 🎯 Choose Your Developer Persona

### **👨‍💻 Data Scientist**
**Goal**: Add governance to ML experiments and research workflows
- **Start**: [`basic_pipeline_tracking.py`](basic_pipeline_tracking.py)
- **Next**: [`rag_workflow_governance.py`](rag_workflow_governance.py) or [`agent_workflow_tracking.py`](agent_workflow_tracking.py)
- **Focus**: Cost tracking, experiment governance, budget controls

### **🏗️ ML Engineer**  
**Goal**: Build production-ready AI pipelines with comprehensive monitoring
- **Start**: [`basic_pipeline_tracking.py`](basic_pipeline_tracking.py)
- **Next**: [`performance_optimization.py`](performance_optimization.py)
- **Then**: [`production_deployment_patterns.py`](production_deployment_patterns.py)
- **Focus**: Performance, scalability, production patterns

### **🛡️ Platform Engineer**
**Goal**: Enterprise governance, compliance, and multi-tenant AI infrastructure
- **Start**: [`basic_pipeline_tracking.py`](basic_pipeline_tracking.py)
- **Next**: [`enterprise_governance_patterns.py`](enterprise_governance_patterns.py)
- **Then**: [`production_deployment_patterns.py`](production_deployment_patterns.py)
- **Focus**: Governance, compliance, multi-tenancy, security

### **💰 FinOps/Cost Optimizer**
**Goal**: AI cost management and optimization across teams and projects
- **Start**: [`basic_pipeline_tracking.py`](basic_pipeline_tracking.py)
- **Next**: [`multi_provider_cost_aggregation.py`](multi_provider_cost_aggregation.py)
- **Focus**: Cost tracking, budget management, optimization strategies

---

## 🚨 Common Issues & Quick Fixes

### **"ModuleNotFoundError: No module named 'haystack'"**
```bash
# Fix: Install Haystack
pip install haystack-ai

# Verify installation
python -c "import haystack; print(f'Haystack {haystack.__version__} installed')"
```

### **"ModuleNotFoundError: No module named 'genops'"**
```bash
# Fix: Install GenOps with Haystack support
pip install genops-ai[haystack]

# Or install separately
pip install genops-ai haystack-ai
```

### **"AuthenticationError" or API Key Issues**
```bash
# Fix: Set your API keys
export OPENAI_API_KEY="sk-your-key-here"

# Verify key is set
echo $OPENAI_API_KEY
```

### **"ValidationError" during setup**
```python
# Fix: Run comprehensive validation
from genops.providers.haystack import validate_haystack_setup, print_validation_result
result = validate_haystack_setup()
print_validation_result(result)
# Follow the specific fix suggestions provided
```

### **Examples Running Slowly**
- **Cause**: Network latency to AI providers
- **Fix**: Consider using faster models (gpt-3.5-turbo vs gpt-4) for testing
- **Production**: Use caching patterns from `performance_optimization.py`

---

## 🌟 What Makes These Examples Special

### **🏆 Production-Grade Quality**
- **Enterprise patterns**: Multi-tenant governance, compliance, audit trails
- **Error handling**: Comprehensive retry logic and failure recovery
- **Performance**: Caching, parallel processing, optimization strategies
- **Monitoring**: Complete observability with OpenTelemetry integration

### **📚 Educational Excellence**  
- **Progressive complexity**: Each example builds on previous knowledge
- **Clear documentation**: Every concept explained with practical examples
- **Real-world scenarios**: Not toy examples, but production-ready patterns
- **Best practices**: Following CLAUDE.md Developer Experience Standards

### **🔧 Developer-Friendly**
- **Zero-code auto-instrumentation**: Works with existing Haystack code
- **Comprehensive validation**: Proactive error detection and fixes
- **Rich output**: Beautiful console formatting with progress indicators
- **Extensible patterns**: Easy to adapt for your specific use cases

---

## 🤝 Need Help?

### **Documentation**
- **Integration Guide**: [`docs/integrations/haystack.md`](../../docs/integrations/haystack.md) - Complete reference
- **Quickstart**: [`docs/haystack-quickstart.md`](../../docs/haystack-quickstart.md) - 5-minute setup
- **API Reference**: Included in integration guide

### **Community**
- **Issues**: [Report bugs or request features](https://github.com/anthropics/claude-code/issues)
- **Discussions**: Share experiences and get help
- **Contributing**: PRs welcome! See CONTRIBUTING.md

### **Enterprise Support**
For production deployments, enterprise features, and professional support, see our enterprise offerings.

---

## 🚀 Ready to Begin?

**Start with**: [`basic_pipeline_tracking.py`](basic_pipeline_tracking.py)

**Time commitment**: 5 minutes to working pipeline with complete governance

**Result**: Cost-aware, budget-controlled Haystack pipeline with OpenTelemetry integration

```bash
python basic_pipeline_tracking.py
```

**Happy building with Haystack + GenOps!** 🎉