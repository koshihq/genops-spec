# Replicate GenOps Examples

**🎯 New here? [Skip to: Where do I start?](#where-do-i-start) | 📚 Need definitions? [Skip to: What do these terms mean?](#what-do-these-terms-mean)**

---

## 🌟 **Where do I start?**

**👋 First time with GenOps + Replicate? Answer one question:**

❓ **Do you have existing Replicate code that you want to add cost tracking to?**
- **✅ YES** → Jump to Phase 2: [`auto_instrumentation.py`](#auto_instrumentationpy---phase-2) (15 min)
- **❌ NO** → Start with Phase 1: [`hello_genops_minimal.py`](#hello_genops_minimalpy---start-here---phase-1) (30 sec)

❓ **Are you a manager/non-technical person?**
- Read ["What GenOps does"](#what-genops-does) then watch your team run the examples

❓ **Are you deploying to production?**
- Start with [Phase 1](#phase-1-prove-it-works-30-seconds-) for concepts, then jump to [Phase 3](#phase-3-production-ready-1-2-hours-)

❓ **Having errors or issues?**
- Jump straight to [Quick fixes](#having-issues)

---

## 📖 **What do these terms mean?**

**New to AI/GenOps? Here are the key terms you'll see:**

**🧠 Essential AI Terms:**
- **Replicate**: Platform for running AI models in the cloud (like AWS but for AI)
- **Model**: Different AI "brains" - text (Llama), image (FLUX), video (Veo), audio (Whisper)
- **Prompt**: The text you send to ask the AI something  
- **Token**: Unit of AI processing (roughly 4 characters of text)

**📊 GenOps Terms (the main concept):**
- **GenOps**: Cost tracking + team budgets for AI (like monitoring for websites, but for AI)
- **Instrumentation**: Adding tracking to your AI code (GenOps does this automatically)
- **Cost Attribution**: Knowing which team/project spent what on AI
- **Governance**: Rules and budgets to control AI spending

**That's it! You know enough to get started.**

---

## 🧭 **Your Learning Journey**

**This directory implements a 30 seconds → 30 minutes → 2 hours learning path:**

### 🎯 **Phase 1: Prove It Works (30 seconds)** ⚡
**Goal**: See GenOps tracking your Replicate calls - build confidence first

**What you'll learn**: GenOps automatically tracks AI costs across all model types  
**What you need**: API token from Replicate  
**Success**: See "✅ SUCCESS! GenOps is now tracking" message

**Next**: Once you see it work → Phase 2 for team tracking

---

### 🏗️ **Phase 2: Add Team Tracking (15-30 minutes)** 🚀  
**Goal**: Track which teams/projects spend what on AI across text, image, video models

**What you'll learn**: Cost attribution, governance attributes, multi-modal optimization  
**What you need**: Basic Python knowledge  
**Success**: See cost breakdowns by team/project across different model types

**Next**: Once you understand team tracking → Phase 3 for production

---

### 🎓 **Phase 3: Production Ready (1-2 hours)** 🏛️
**Goal**: Deploy with monitoring, optimization, and enterprise features

**What you'll learn**: Intelligent model routing, batch processing, advanced budget controls  
**What you need**: Production deployment experience  
**Success**: Running in production with cost optimization across all Replicate models

**Next**: You're now a GenOps + Replicate expert! 🎉

---

**Having Issues?** → [Quick fixes](#having-issues) | **Skip Ahead?** → [Examples](#examples-by-progressive-phase)

## 📋 Examples by Progressive Phase

### 🎯 **Phase 1: Prove It Works (30 seconds)**

#### [`hello_genops_minimal.py`](hello_genops_minimal.py) ⭐ **START HERE**
✅ **30-second confidence builder** - Just run it and see GenOps tracking your Replicate calls

### 🏗️ **Phase 2: Add Team Tracking (15-30 minutes)**

#### [`auto_instrumentation.py`](auto_instrumentation.py) ⭐ **For existing Replicate code**
✅ **Add GenOps to existing apps** - Zero code changes to your current Replicate calls (15 min)

#### [`basic_tracking.py`](basic_tracking.py) ⭐ **For new team projects**
✅ **Team cost attribution** - Track which teams spend what on AI across model types (10 min)

### 🎓 **Phase 3: Production Ready (1-2 hours)**

#### [`cost_optimization.py`](cost_optimization.py) ⭐ **For production deployment**
✅ **Advanced cost optimization** - Intelligent routing, batch processing, enterprise governance (45 min)

---

**🚀 That's it!** Three examples, three phases, complete GenOps + Replicate mastery.

## 💡 What You Get

**After completing all phases:**
- ✅ **Cost Tracking**: See exactly how much each AI call costs across all model types
- ✅ **Team Attribution**: Know which teams spend what on text, image, video, audio AI  
- ✅ **Budget Control**: Set limits and get alerts across your entire AI workflow
- ✅ **Zero Code Changes**: Works with your existing Replicate apps
- ✅ **Multi-Modal Intelligence**: Optimize across text, image, video, and audio models

---

## 🚀 Ready to Start?

**Just pick your situation:**
- **New to GenOps?** → [`hello_genops_minimal.py`](hello_genops_minimal.py)
- **Have existing Replicate code?** → [`auto_instrumentation.py`](auto_instrumentation.py) 
- **Setting up team tracking?** → [`basic_tracking.py`](basic_tracking.py)
- **Going to production?** → [`cost_optimization.py`](cost_optimization.py)

---

## 🛠️ Quick Setup

```bash
# 1. Install
pip install genops-ai[replicate]

# 2. Get API token from https://replicate.com/account/api-tokens
export REPLICATE_API_TOKEN="r8_your_token_here"

# 3. Run first example
python hello_genops_minimal.py
```

**✅ That's all you need to get started!**

---

## 🆘 Having Issues?

**🔧 Quick fixes for common problems:**
- **`ImportError: replicate`** → `pip install replicate`  
- **API token error** → Get free token at https://replicate.com/account/api-tokens
- **Model not found** → Try different model from https://replicate.com/explore
- **Still stuck?** → Check [`hello_genops_minimal.py`](hello_genops_minimal.py) - it has detailed error messages

---

**🎉 Ready to become a GenOps + Replicate expert? Start with the 30-second example!**

👉 [`python hello_genops_minimal.py`](hello_genops_minimal.py)