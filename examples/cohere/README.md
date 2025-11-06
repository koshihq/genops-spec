# Cohere GenOps Examples

**🎯 New here? [Skip to: Where do I start?](#where-do-i-start) | 📚 Need definitions? [Skip to: What do these terms mean?](#what-do-these-terms-mean)**

---

## 🌟 **Where do I start?**

**👋 First time with GenOps + Cohere? Answer one question:**

❓ **Do you have a Cohere API key and want to see cost tracking immediately?**
- **✅ YES** → Jump to Phase 1: [`hello_cohere_minimal.py`](#hello_cohere_minimalpy---start-here---phase-1) (30 sec)
- **❌ NO** → Get your API key at [Cohere Dashboard](https://dashboard.cohere.ai/), then start Phase 1

❓ **Are you using multiple Cohere operations (chat, embed, rerank)?**
- **✅ YES** → Start with Phase 2: [`multi_operation_tracking.py`](#multi_operation_trackingpy---phase-2) (15 min)
- **❌ NO** → Start with Phase 1 to understand basics first

❓ **Are you a manager/non-technical person?**
- Read ["What GenOps does for Cohere"](#what-genops-does-for-cohere) then watch your team run the examples

❓ **Are you deploying to production?**
- Start with [Phase 1](#phase-1-prove-it-works-30-seconds-) for concepts, then jump to [Phase 3](#phase-3-production-ready-1-2-hours-)

❓ **Having errors or issues?**
- Jump straight to [Quick fixes](#having-issues)

---

## 📖 **What do these terms mean?**

**New to Cohere/GenOps? Here are the key terms you'll see:**

**🧠 Essential Cohere Terms:**
- **Cohere**: Enterprise AI platform for text generation, embedding, and search optimization
- **Command Models**: Text generation models (Command, Command-R, Command-R+)
- **Embed Models**: Text embedding models for semantic search and classification
- **Rerank Models**: Document reranking models for search relevance optimization
- **Multi-Modal**: Text + image embedding capabilities (Embed v4.0)
- **Token-based + Operation-based Pricing**: Mix of token costs and per-operation costs

**📊 GenOps + Cohere Terms (the main concept):**
- **GenOps**: Cost tracking + governance for AI (now works with all Cohere operations!)
- **Multi-Operation Tracking**: Track costs across generation, embedding, and reranking in unified workflows
- **Operation Attribution**: Knowing which team/project used which Cohere services and how much
- **Cost Per Operation Type**: Separate cost tracking for text generation, embeddings, and search
- **Enterprise Optimization**: Cost efficiency across Cohere's specialized AI operations

**That's it! You know enough to get started.**

---

## 🧭 **Your Learning Journey**

**This directory implements a 30 seconds → 30 minutes → 2 hours learning path:**

### 🎯 **Phase 1: Prove It Works (30 seconds)** ⚡
**Goal**: See GenOps tracking your Cohere operations - build confidence first

**What you'll learn**: GenOps automatically tracks all Cohere costs (generation, embedding, rerank)  
**What you need**: Cohere API key (free tier works)  
**Success**: See "✅ SUCCESS! GenOps is now tracking" message

**Next**: Once you see it work → Phase 2 for multi-operation tracking

---

### 🏗️ **Phase 2: Multi-Operation Tracking (15-30 minutes)** 🚀  
**Goal**: Track costs across Cohere's specialized operations (chat + embed + rerank)

**What you'll learn**: Unified cost tracking, operation-specific optimization, team attribution  
**What you need**: Basic understanding from Phase 1  
**Success**: See cost breakdowns across all operation types with optimization insights

**Next**: Once you understand multi-operation workflows → Phase 3 for production

---

### 🎓 **Phase 3: Production Ready (1-2 hours)** 🏛️
**Goal**: Deploy with enterprise patterns, cost optimization, and governance controls

**What you'll learn**: Production monitoring, budget controls, cost optimization strategies  
**What you need**: Production deployment experience  
**Success**: Running production Cohere with comprehensive cost governance

**Next**: You're now a GenOps + Cohere expert! 🎉

---

**Having Issues?** → [Quick fixes](#having-issues) | **Skip Ahead?** → [Examples](#examples-by-progressive-phase) | **Want Full Reference?** → [Complete Integration Guide](../../docs/integrations/cohere.md)

## 📋 Examples by Progressive Phase

### 🎯 **Phase 1: Prove It Works (30 seconds)**

#### [`hello_cohere_minimal.py`](hello_cohere_minimal.py) ⭐ **START HERE**
✅ **30-second confidence builder** - Just run it and see GenOps tracking your Cohere operations  
🎯 **What you'll accomplish**: Verify GenOps works with Cohere and see cost tracking in action  
▶️ **Next step after success**: Move to [`multi_operation_tracking.py`](multi_operation_tracking.py) for multi-operation workflows

**✅ Ready for Phase 2?** After running `hello_cohere_minimal.py` successfully, you should see:
- "✅ SUCCESS! GenOps is now tracking your Cohere usage" message
- Cost calculations displayed (input tokens, output tokens, total cost)
- Operation metrics (latency, tokens per second) shown
If you see these, you're ready for multi-operation tracking!

### 🏗️ **Phase 2: Multi-Operation Tracking (15-30 minutes)**

#### [`multi_operation_tracking.py`](multi_operation_tracking.py) ⭐ **For unified workflows**
✅ **Multi-operation cost tracking** - Track chat, embed, and rerank in unified workflows (15-30 min)  
🎯 **What you'll learn**: Cost attribution across all Cohere operations and optimization insights  
▶️ **Ready for production?**: Move to Phase 3 production deployment

#### [`cost_optimization.py`](cost_optimization.py) ⭐ **For cost efficiency**
✅ **Advanced cost optimization** - Compare models, optimize operation types, reduce costs (20-40 min)  
🎯 **What you'll learn**: Which models are most cost-efficient and when to use each operation type  
▶️ **Enterprise ready?**: Move to Phase 3 production patterns

### 🎓 **Phase 3: Production Ready (1-2 hours)**

#### [`auto_instrumentation.py`](auto_instrumentation.py) ⭐ **For zero-code integration**
✅ **Zero-code instrumentation** - Works with existing Cohere code unchanged (30-45 min)  
🎯 **What you'll learn**: How to add GenOps tracking without changing existing applications  
▶️ **Production deployment**: Ready for enterprise deployment patterns

#### [`enterprise_deployment.py`](enterprise_deployment.py) ⭐ **For production**
✅ **Enterprise deployment** - Cost controls, monitoring, governance patterns (45 min - 1 hour)  
🎯 **What you'll learn**: Production-ready Cohere deployment with comprehensive cost governance  
▶️ **You're now ready**: Deploy GenOps Cohere governance to production! 🎉

---

**🚀 That's it!** Four examples, three phases, complete GenOps + Cohere mastery.

## 💡 What You Get

**After completing all phases:**
- ✅ **Multi-Operation Cost Tracking**: See exactly what each Cohere operation costs (generation, embedding, rerank)
- ✅ **Unified Workflow Optimization**: Get recommendations across all operation types in complex workflows
- ✅ **Team Attribution**: Know which teams use which operations and how much they cost
- ✅ **Enterprise Intelligence**: Optimize your specific Cohere usage patterns and model selection
- ✅ **Production Governance**: Enterprise-ready deployment with monitoring and cost controls
- ✅ **Specialization Insights**: Understand when to use generation vs embedding vs reranking for cost efficiency

---

## 🚀 Ready to Start?

**🎯 Choose Your Path (recommended order):**
1. **New to GenOps + Cohere?** → [`hello_cohere_minimal.py`](hello_cohere_minimal.py) *(Start here - 30 seconds)*
2. **Want multi-operation tracking?** → [`multi_operation_tracking.py`](multi_operation_tracking.py) *(Unified workflows - 15-30 minutes)*
3. **Ready for production?** → [`enterprise_deployment.py`](enterprise_deployment.py) *(Enterprise patterns - 1 hour)*

**🔀 Or Jump to Specific Needs:**
- **Full documentation** → [Complete Cohere Integration Guide](../../docs/integrations/cohere.md)
- **5-minute setup** → [Cohere Quickstart Guide](../../docs/cohere-quickstart.md)

---

## 🛠️ Quick Setup

```bash
# 1. Get your Cohere API key from https://dashboard.cohere.ai/
export CO_API_KEY="your-cohere-api-key"

# 2. Install Cohere client (if not already installed)
pip install cohere

# 3. Install GenOps with Cohere support
pip install genops-ai

# 4. Run first example
python hello_cohere_minimal.py
```

**✅ That's all you need to get started!**

---

## 🆘 Having Issues?

**🔧 Quick fixes for common problems:**

**Cohere Issues:**
- **"Invalid API key"** → Check your key: `echo $CO_API_KEY`
- **"Unauthorized"** → Verify key format (should start with 'co_' or 'ck_')
- **"Model not found"** → Try basic model: `command-light`
- **"Rate limit exceeded"** → Wait or check your Cohere usage limits

**GenOps Issues:**
- **Import errors** → Install: `pip install genops-ai`
- **"No module named 'cohere'"** → Install client: `pip install cohere`
- **Cost calculation errors** → Check model name spelling and availability

**Performance Issues:**
- **Slow responses** → Try lighter model: `command-light` instead of `command-r-plus`
- **High costs** → Use model comparison examples to find optimal models
- **API timeouts** → Check network connection and Cohere service status

**Still stuck?** Run the diagnostic:
```python
from genops.providers.cohere_validation import validate_setup, print_validation_result
result = validate_setup()
print_validation_result(result, detailed=True)
```

---

## 🎯 What GenOps Does for Cohere

**For managers and non-technical folks:**

GenOps brings comprehensive governance to your Cohere AI operations:

**💰 Multi-Operation Cost Tracking**
- See exactly what each operation costs (text generation, embeddings, document reranking)
- Track costs by team, project, and customer across all Cohere services
- Get alerts when costs approach budget limits across operation types
- Compare costs between different Cohere models and operation strategies

**📊 Enterprise Optimization**
- Monitor usage patterns across generation, embedding, and search operations
- Get recommendations for when to use each operation type for cost efficiency
- Identify which teams are using which Cohere services and optimize accordingly
- Advanced insights for complex workflows that combine multiple operations

**🏛️ Production Governance**
- Same team attribution and project tracking across all Cohere operations
- Compliance reporting and audit trails for enterprise AI usage
- Budget controls and cost enforcement for all operation types
- Integrates with your existing monitoring and observability tools

**🎯 Cohere Specialization**
- Purpose-built for Cohere's unique multi-operation model (generation + embedding + rerank)
- Optimized for enterprise search, classification, and document analysis workflows
- Advanced cost attribution for complex AI pipelines using multiple Cohere services
- Specialized insights for hybrid workflows combining different operation types

**Think of it as "enterprise AI governance for Cohere's specialized operations" - you get unified cost tracking and optimization across all of Cohere's AI capabilities.**

---

**🎉 Ready to become a GenOps + Cohere expert?**

**📚 Complete Learning Path:**
1. **30 seconds**: [`python hello_cohere_minimal.py`](hello_cohere_minimal.py) - Prove it works
2. **15-30 minutes**: [`python multi_operation_tracking.py`](multi_operation_tracking.py) - Multi-operation workflows  
3. **1 hour**: [`python enterprise_deployment.py`](enterprise_deployment.py) - Production deployment

**🚀 Quick Start**: `python hello_cohere_minimal.py`

## 📚 Documentation & Resources

**📖 Complete Guides:**
- **[5-Minute Quickstart](../../docs/cohere-quickstart.md)** - Get running in 5 minutes with copy-paste examples
- **[Complete Integration Guide](../../docs/integrations/cohere.md)** - Full API reference and advanced patterns
- **[Security Best Practices](../../docs/security-best-practices.md)** - Enterprise security guidance
- **[CI/CD Integration](../../docs/ci-cd-integration.md)** - Automated testing and deployment

**🤝 Community & Support:**
- **[GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)** - Questions, ideas, and community help
- **[GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)** - Bug reports and feature requests