# AutoGen + GenOps Examples

Progressive examples for AutoGen conversation governance, from 3-minute quickstart to advanced enterprise patterns.

## 🚀 Quick Start

**New to AutoGen + GenOps?** Start here:

```bash
# 1. Install 
pip install genops[autogen]

# 2. Validate (30 seconds)
python examples/autogen/setup_validation.py

# 3. Try quickstart (3 minutes)
python examples/autogen/01_quickstart_demo.py
```

## 📚 Progressive Learning Path

### **Level 1: Getting Started (5 minutes)**

**[`setup_validation.py`](setup_validation.py)** - 30-second environment validation
- ✅ Check installations and API keys
- ✅ Validate GenOps integration
- ✅ Quick diagnostics and fixes

**[`01_quickstart_demo.py`](01_quickstart_demo.py)** - 3-minute value demonstration  
- ✅ One-line governance setup
- ✅ Zero code changes to existing AutoGen
- ✅ Immediate cost tracking

### **Level 2: Intermediate Tracking (15 minutes)**

**[`02_conversation_tracking.py`](02_conversation_tracking.py)** - Detailed conversation analysis
- ✅ Manual conversation tracking with context managers
- ✅ Real-time cost monitoring and budget alerts
- ✅ Conversation analytics and performance metrics
- ✅ Cost optimization insights

**[`basic_conversation_tracking.py`](basic_conversation_tracking.py)** - Comprehensive workflow
- ✅ Complete setup validation and instrumentation
- ✅ Step-by-step governance implementation
- ✅ Session analytics and recommendations

### **Level 3: Advanced Patterns (30 minutes)**

**[`03_group_chat_monitoring.py`](03_group_chat_monitoring.py)** - Multi-agent governance
- ✅ Group chat orchestration tracking  
- ✅ Role-based cost attribution
- ✅ Agent collaboration analytics
- ✅ Advanced multi-provider optimization

## 🎯 Choose Your Path

### **I want to get started in 3 minutes:**
```bash
python examples/autogen/01_quickstart_demo.py
```

### **I want to understand conversation tracking:**
```bash
python examples/autogen/02_conversation_tracking.py
```

### **I want to monitor group chats:**
```bash
python examples/autogen/03_group_chat_monitoring.py
```

### **I want to validate my setup:**
```bash
python examples/autogen/setup_validation.py --verbose
```

## 📋 Prerequisites

- Python 3.8+
- AutoGen: `pip install pyautogen`
- GenOps: `pip install genops`
- API Key: Set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`

## 🔧 Example Structure

Each example follows the same pattern:
- **Clear learning objectives** - What you'll learn
- **Time investment** - How long it takes
- **Runnable code** - Copy/paste and run immediately
- **Step-by-step explanation** - Understand what's happening
- **Next steps** - Where to go from here

## 💡 Pro Tips

- **Start with validation**: Always run `setup_validation.py` first
- **Use simulation mode**: Examples work without API keys (simulated data)
- **Check your budget**: Set `GENOPS_BUDGET_LIMIT` environment variable
- **Enable verbose mode**: Add `--verbose` flag for detailed output

## 🤝 Getting Help

- **Quick issues**: Run `python examples/autogen/setup_validation.py --verbose`
- **Documentation**: [AutoGen Quickstart Guide](../../docs/quickstart/autogen-quickstart.md)
- **Community**: [GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)
- **Bug reports**: [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)

## 🚀 What's Next?

After completing these examples:

1. **Read the comprehensive guide**: [`docs/integrations/autogen.md`](../../docs/integrations/autogen.md)
2. **Try your own AutoGen code** with the one-line setup
3. **Explore production patterns** in enterprise documentation
4. **Join the community** and share your experience

---

**Ready to add governance to your AutoGen applications?** Start with `01_quickstart_demo.py` and experience the power of zero-code instrumentation! 🎉