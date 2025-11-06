# Ollama GenOps Examples

**🎯 New here? [Skip to: Where do I start?](#where-do-i-start) | 📚 Need definitions? [Skip to: What do these terms mean?](#what-do-these-terms-mean)**

---

## 🌟 **Where do I start?**

**👋 First time with GenOps + Ollama? Answer one question:**

❓ **Do you have Ollama running locally with models you want to track costs for?**
- **✅ YES** → Jump to Phase 2: [`local_model_optimization.py`](#local_model_optimizationpy---phase-2) (15 min)
- **❌ NO** → Start with Phase 1: [`hello_ollama_minimal.py`](#hello_ollama_minimalpy---start-here---phase-1) (30 sec)

❓ **Are you a manager/non-technical person?**
- Read [\"What GenOps does\"](#what-genops-does) then watch your team run the examples

❓ **Are you deploying to production?**
- Start with [Phase 1](#phase-1-prove-it-works-30-seconds-) for concepts, then jump to [Phase 3](#phase-3-production-ready-1-2-hours-)

❓ **Having errors or issues?**
- Jump straight to [Quick fixes](#having-issues)

---

## 📖 **What do these terms mean?**

**New to Ollama/GenOps? Here are the key terms you'll see:**

**🧠 Essential Ollama Terms:**
- **Ollama**: Platform for running LLMs locally on your hardware (free, private, no API costs)
- **Local Models**: AI models that run on your computer/server (LLaMA, Mistral, CodeLLaMA, etc.)
- **Model Size**: How much disk/RAM a model needs (1B, 3B, 7B, 13B parameters)
- **Inference**: Running the model to generate text (costs electricity + compute time)
- **Quantization**: Making models smaller/faster (Q4, Q8 versions)

**📊 GenOps + Local Models Terms (the main concept):**
- **GenOps**: Cost tracking + governance for AI (now works with local models too!)
- **Infrastructure Costs**: What it costs to run local models (electricity, GPU time, server costs)
- **Resource Attribution**: Knowing which team/project used GPU/CPU time and how much
- **Cost Per Inference**: How much each AI request costs you in infrastructure
- **Hardware Optimization**: Making your local setup more cost-efficient

**That's it! You know enough to get started.**

---

## 🧭 **Your Learning Journey**

**This directory implements a 30 seconds → 30 minutes → 2 hours learning path:**

### 🎯 **Phase 1: Prove It Works (30 seconds)** ⚡
**Goal**: See GenOps tracking your local Ollama models - build confidence first

**What you'll learn**: GenOps automatically tracks local model costs (GPU time, electricity, infrastructure)  
**What you need**: Ollama running with at least one model downloaded  
**Success**: See \"✅ SUCCESS! GenOps is now tracking\" message

**Next**: Once you see it work → Phase 2 for optimization

---

### 🏗️ **Phase 2: Add Local Model Optimization (15-30 minutes)** 🚀  
**Goal**: Optimize local model costs and performance with data-driven recommendations

**What you'll learn**: Infrastructure cost analysis, model comparison, resource optimization  
**What you need**: Multiple models for comparison  
**Success**: See cost breakdowns and optimization recommendations for your hardware

**Next**: Once you understand local optimization → Phase 3 for production

---

### 🎓 **Phase 3: Production Ready (1-2 hours)** 🏛️
**Goal**: Deploy with enterprise patterns, monitoring, budget controls

**What you'll learn**: Production monitoring, load balancing, budget enforcement  
**What you need**: Production deployment experience  
**Success**: Running production Ollama with comprehensive governance

**Next**: You're now a GenOps + Ollama expert! 🎉

---

**Having Issues?** → [Quick fixes](#having-issues) | **Skip Ahead?** → [Examples](#examples-by-progressive-phase) | **Want Full Reference?** → [Complete Integration Guide](../../docs/integrations/ollama.md)

## 📋 Examples by Progressive Phase

### 🎯 **Phase 1: Prove It Works (30 seconds)**

#### [`hello_ollama_minimal.py`](hello_ollama_minimal.py) ⭐ **START HERE**
✅ **30-second confidence builder** - Just run it and see GenOps tracking your local models  
🎯 **What you'll accomplish**: Verify GenOps works with your Ollama setup and see cost tracking in action  
▶️ **Next step after success**: Move to [`local_model_optimization.py`](local_model_optimization.py) to optimize costs

**✅ Ready for Phase 2?** After running `hello_ollama_minimal.py` successfully, you should see:
- "✅ SUCCESS! GenOps is now tracking your Ollama usage" message
- Infrastructure cost calculations displayed
- Resource metrics (CPU/memory usage) shown
If you see these, you're ready for optimization!

### 🏗️ **Phase 2: Add Local Model Optimization (15-30 minutes)**

#### [`local_model_optimization.py`](local_model_optimization.py) ⭐ **For cost optimization**
✅ **Local model efficiency** - Compare models, optimize resources, reduce infrastructure costs (15-30 min)  
🎯 **What you'll learn**: Which models are most cost-efficient for your use cases and hardware  
▶️ **Ready for production?**: Move to Phase 3 production deployment

### 🎓 **Phase 3: Production Ready (1-2 hours)**

#### [`ollama_production_deployment.py`](ollama_production_deployment.py) ⭐ **For production**
✅ **Enterprise deployment** - Load balancing, monitoring, budget controls, Kubernetes patterns (45 min - 1 hour)  
🎯 **What you'll learn**: Production-ready local model deployment with comprehensive governance  
▶️ **You're now ready**: Deploy GenOps Ollama governance to production! 🎉

---

**🚀 That's it!** Three examples, three phases, complete GenOps + Ollama mastery.

## 💡 What You Get

**After completing all phases:**
- ✅ **Infrastructure Cost Tracking**: See exactly what local models cost (GPU time, electricity, compute)
- ✅ **Resource Optimization**: Get recommendations to reduce costs and improve performance
- ✅ **Team Attribution**: Know which teams use which models and how much infrastructure they consume
- ✅ **Hardware Intelligence**: Optimize your specific hardware setup (CPU, GPU, memory)
- ✅ **Zero Cloud Costs**: All tracking happens locally - no API fees, maximum privacy
- ✅ **Production Patterns**: Enterprise-ready deployment with monitoring and governance

---

## 🚀 Ready to Start?

**🎯 Choose Your Path (recommended order):**
1. **New to GenOps + Ollama?** → [`hello_ollama_minimal.py`](hello_ollama_minimal.py) *(Start here - 30 seconds)*
2. **Want cost optimization?** → [`local_model_optimization.py`](local_model_optimization.py) *(Optimize resources - 15-30 minutes)*
3. **Ready for production?** → [`ollama_production_deployment.py`](ollama_production_deployment.py) *(Enterprise patterns - 1 hour)*

**🔀 Or Jump to Specific Needs:**
- **Full documentation** → [Complete Ollama Integration Guide](../../docs/integrations/ollama.md)
- **5-minute setup** → [Ollama Quickstart Guide](../../docs/ollama-quickstart.md)

---

## 🛠️ Quick Setup

**💻 Hardware Requirements:**
- **Minimum**: 4GB RAM, 2GB disk space
- **Recommended**: 8GB+ RAM, 10GB+ disk space, GPU optional but helps performance
- **GPU Support**: NVIDIA GPUs with CUDA, Apple Silicon Macs, AMD GPUs (experimental)

```bash
# 1. Install Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Start Ollama server
ollama serve

# 3. Pull a model for testing
ollama pull llama3.2:1b  # Small, fast model (1.3GB, 4GB+ RAM recommended)
# OR
ollama pull llama3.2:3b  # More capable model (2.0GB, 8GB+ RAM recommended)
# OR
ollama pull llama3.2:11b # Large model (7.5GB, 16GB+ RAM recommended)

# 4. Install GenOps with Ollama support
pip install genops-ai[ollama]

# 5. Run first example
python hello_ollama_minimal.py
```

**✅ That's all you need to get started!**

---

## 🆘 Having Issues?

**🔧 Quick fixes for common problems:**

**Ollama Issues:**
- **\"Connection refused\"** → Start Ollama: `ollama serve`
- **\"No models found\"** → Pull a model: `ollama pull llama3.2:1b`
- **\"Model not found\"** → Check available: `ollama list`
- **\"Ollama not installed\"** → Install: `curl -fsSL https://ollama.ai/install.sh | sh`

**GenOps Issues:**
- **Import errors** → Install: `pip install genops-ai[ollama]`
- **\"No module named 'ollama'\"** → Install client: `pip install ollama`
- **Permission errors** → Run with appropriate permissions for GPU access

**Performance Issues:**
- **Slow inference** → Try smaller model: `ollama pull llama3.2:1b`
- **High memory usage** → Check system resources: `free -h` or `htop`
- **GPU not detected** → Check GPU availability and drivers:
  - **NVIDIA**: `nvidia-smi` (install CUDA drivers if missing)
  - **Apple Silicon**: Should work automatically on M1/M2/M3 Macs  
  - **AMD**: Install ROCm support (experimental, Linux only)
  - **No GPU**: Ollama runs on CPU - expect slower but functional performance

**Still stuck?** Run the diagnostic:
```python
from genops.providers.ollama.validation import validate_setup, print_validation_result
result = validate_setup()
print_validation_result(result, detailed=True)
```

---

## 🎯 What GenOps Does for Local Models

**For managers and non-technical folks:**

GenOps brings the same governance you'd have with cloud AI providers to your local models:

**💰 Infrastructure Cost Tracking**
- See exactly what each model costs to run (electricity, GPU time, server costs)
- Track costs by team, project, and customer for local model usage
- Get alerts when infrastructure usage approaches budget limits
- Compare local vs cloud costs to optimize your AI strategy

**📊 Resource Optimization**
- Monitor GPU, CPU, and memory usage across all models
- Get recommendations to reduce infrastructure costs
- Identify which models are most efficient for your use cases
- Optimize hardware utilization and scaling decisions

**🏛️ Enterprise Governance**
- Same team attribution and project tracking as cloud providers
- Compliance reporting and audit trails for local model usage
- Budget controls and cost enforcement for infrastructure resources
- Integrates with your existing monitoring and observability tools

**🔒 Privacy & Control**
- All tracking happens on your infrastructure - maximum privacy
- No data sent to external services - complete control
- Works offline - no internet dependency for tracking
- Your models, your data, your infrastructure, your governance

**Think of it as \"enterprise AI governance for local models\" - you get the same insights and controls you'd have with cloud providers, but everything stays on your hardware.**

---

**🎉 Ready to become a GenOps + Ollama expert?**

**📚 Complete Learning Path:**
1. **30 seconds**: [`python hello_ollama_minimal.py`](hello_ollama_minimal.py) - Prove it works
2. **15-30 minutes**: [`python local_model_optimization.py`](local_model_optimization.py) - Optimize costs  
3. **1 hour**: [`python ollama_production_deployment.py`](ollama_production_deployment.py) - Production deployment

**🚀 Quick Start**: `python hello_ollama_minimal.py`

## 📚 Documentation & Resources

**📖 Complete Guides:**
- **[5-Minute Quickstart](../../docs/ollama-quickstart.md)** - Get running in 5 minutes with copy-paste examples
- **[Complete Integration Guide](../../docs/integrations/ollama.md)** - Full API reference and advanced patterns
- **[Security Best Practices](../../docs/security-best-practices.md)** - Enterprise security guidance
- **[CI/CD Integration](../../docs/ci-cd-integration.md)** - Automated testing and deployment

**🤝 Community & Support:**
- **[GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)** - Questions, ideas, and community help
- **[GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)** - Bug reports and feature requests