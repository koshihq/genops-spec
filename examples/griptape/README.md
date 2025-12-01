# Griptape + GenOps Examples

**🚀 Get GenOps governance for your Griptape AI applications in 5 minutes.**

> **New to Griptape?** It's a modular Python framework for AI agents and workflows with chain-of-thought reasoning, tools, and memory. Works with 20+ AI providers (OpenAI, Anthropic, Google, etc.). **GenOps adds cost tracking, team attribution, and governance** - with zero code changes!

## 🎯 Start Here (5 Minutes)

### 1. One-Command Setup
```bash
pip install genops griptape && export OPENAI_API_KEY="your-key" GENOPS_TEAM="your-team"
```

### 2. Copy-Paste Demo
```bash
# Download and run immediately (if using from GitHub)
curl -O https://raw.githubusercontent.com/KoshiHQ/GenOps-AI/main/examples/griptape/01_basic_agent.py
python 01_basic_agent.py

# Or if you have the repo locally:
python 01_basic_agent.py
```

### 3. See Immediate Results
```
✅ GenOps governance enabled
💰 Cost tracking: $0.000523 for Agent execution
📊 Team attribution: your-team
🔍 Request ID: griptape-agent-1700123456789
```

**🎉 Success!** You now have full GenOps governance for Griptape.

## 📚 Progressive Learning Path

### ⭐ **Beginner (5 minutes each)**
| Example | What You'll Learn | Time |
|---------|-------------------|------|
| **[01. Basic Agent](01_basic_agent.py)** | Core governance setup with single Agent | 5 min |
| **[02. Auto-Instrumentation](02_auto_instrumentation.py)** | Zero-code integration for existing apps | 5 min |

**Ready for more?** ⬇️

### ⭐⭐ **Intermediate (15 minutes each)**
| Example | What You'll Learn | Time |
|---------|-------------------|------|
| **[03. Pipeline Workflows](03_pipeline_workflows.py)** | Sequential task execution with cost tracking | 15 min |
| **[04. Multi-Provider Setup](04_multi_provider_setup.py)** | Unified governance across multiple LLM providers | 15 min |

### ⭐⭐⭐ **Advanced (Coming Soon!)**
| Example | What You'll Learn | Status |
|---------|-------------------|--------|
| **05. Memory Management** | Conversation and task memory governance | 🚧 Coming Soon |
| **06. RAG Engine Governance** | Retrieval-augmented generation cost tracking | 🚧 Coming Soon |
| **07. Multi-Tenant SaaS** | Customer isolation and per-tenant billing | 🚧 Coming Soon |
| **08. Enterprise Governance** | Complete enterprise deployment patterns | 🚧 Coming Soon |

**Want these examples?** [Star the repo](https://github.com/KoshiHQ/GenOps-AI) and [open an issue](https://github.com/KoshiHQ/GenOps-AI/issues) requesting the specific examples you need!

## 📖 Complete Documentation

**For comprehensive information:**
- 📚 **[Complete Integration Guide](../../docs/integrations/griptape.md)** - Production deployment, API reference, advanced patterns
- 🚀 **[5-Minute Quickstart](../../docs/griptape-quickstart.md)** - Get started immediately
- 🛠️ **[Setup Validation](setup_validation.py)** - Diagnostic tool for troubleshooting

## 🔧 Quick Troubleshooting

**"Griptape not found"**
```bash
pip install griptape
```

**"GenOps not installed"**
```bash
pip install genops
```

**"API key not found"**
```bash
export OPENAI_API_KEY="your-actual-key"
```

**"Still not working?"**
```bash
python setup_validation.py  # Comprehensive diagnostic
```

## Architecture Patterns

### Auto-Instrumentation Pattern (Recommended)
Use GenOps to automatically instrument all Griptape structures:
```python
from genops.providers.griptape import auto_instrument

auto_instrument(team="ai-team", project="agent-workflows")

# Your existing Griptape code works unchanged
from griptape.structures import Agent
agent = Agent(tasks=[PromptTask("Analyze data")])
result = agent.run("Input data")  # ✅ Now tracked
```

### Manual Instrumentation Pattern  
Controlled instrumentation for specific use cases:
```python
from genops.providers.griptape import instrument_griptape

griptape = instrument_griptape(team="ai-team", project="analysis")
agent = griptape.create_agent([PromptTask("Research task")])
result = agent.run("Research data")  # ✅ Tracked with control
```

### Context Manager Pattern
Fine-grained governance for individual operations:
```python
from genops.providers.griptape import GenOpsGriptapeAdapter

adapter = GenOpsGriptapeAdapter(team="ai-team", project="custom")
with adapter.track_agent("research-agent") as request:
    # Agent execution with detailed tracking
    result = agent.run("Complex analysis")
    print(f"Cost: ${request.total_cost:.6f}")
```

## Griptape Structure Support

### Agents
Single-task operations with LLM provider tracking:
```python
# Automatic governance for Agent execution
agent = Agent(tasks=[PromptTask("Single task analysis")])
result = agent.run("Data to analyze")
```

### Pipelines
Sequential task execution with cost aggregation:
```python
# Pipeline with automatic task-level governance
pipeline = Pipeline(tasks=[task1, task2, task3])
result = pipeline.run({"input": "data"})
```

### Workflows
Parallel task monitoring and attribution:
```python
# Workflow with concurrent task tracking
workflow = Workflow(tasks=[[task1, task2], [task3]])
result = workflow.run({"tasks": task_list})
```

### Engines
RAG, Extraction, Summary, Evaluation tracking:
```python
# Engine operations with governance
with adapter.track_engine("rag-engine", "rag") as request:
    response = rag_engine.process("Query about documents")
```

## Multi-Provider Support

GenOps automatically tracks costs across all Griptape-supported providers:

- **OpenAI**: GPT-3.5, GPT-4, GPT-4 Turbo with real-time cost calculation
- **Anthropic**: Claude-3 family with token-accurate attribution
- **Google**: Gemini Pro and Vision with usage tracking
- **Cohere**: Command and Embed models with cost estimation
- **Mistral**: All model variants with pricing integration
- **Local Models**: Ollama and others with zero-cost tracking

## Memory System Integration

### Conversation Memory
```python
# Agent with conversation memory governance
agent = Agent(
    memory=ConversationMemory(),
    tasks=[PromptTask("Continue our conversation")]
)
# Memory operations automatically tracked
```

### Task Memory
```python
# Pipeline with task memory cost attribution
pipeline = Pipeline(
    memory=TaskMemory(),
    tasks=[analysis_task, report_task]
)
# Memory storage and retrieval governance included
```

## Cost Tracking Features

- **Real-Time Monitoring**: Live cost updates during structure execution
- **Multi-Provider Attribution**: Unified costs across all providers
- **Team and Project Tracking**: Per-team, per-project, per-customer breakdown
- **Budget Controls**: Automatic budget enforcement and alerting
- **Usage Analytics**: Detailed patterns and optimization insights
- **Memory Cost Tracking**: Conversation and task memory governance

## Production Deployment

### Docker Integration
```dockerfile
FROM python:3.11-slim
RUN pip install genops griptape
COPY . .
ENV GENOPS_TEAM=production
CMD ["python", "griptape_app.py"]
```

### Kubernetes Patterns
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: griptape-ai-app
spec:
  template:
    spec:
      containers:
      - name: app
        env:
        - name: GENOPS_TEAM
          value: "production-team"
        - name: OTEL_EXPORTER_OTLP_ENDPOINT
          value: "http://jaeger:14268/api/traces"
```

## Troubleshooting

### Common Issues

**1. Griptape Not Found**
```bash
# Install Griptape framework
pip install griptape
```

**2. Import Errors**
```bash
# Check Griptape installation
python -c "import griptape; print('✅ Griptape available')"
```

**3. Auto-Instrumentation Not Working**
```bash
# Verify instrumentation status
python -c "from genops.providers.griptape.registration import is_instrumented; print(f'Instrumented: {is_instrumented()}')"
```

**4. Missing API Keys**
```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

### Validation Tools

Run comprehensive setup validation:
```python
from genops.providers.griptape.registration import validate_griptape_setup
result = validate_griptape_setup()
if result['issues']:
    print("Issues found:", result['issues'])
    print("Recommendations:", result['recommendations'])
```

Quick health check:
```python
from genops.providers.griptape.registration import is_instrumented
if is_instrumented():
    print("✅ Ready to go!")
else:
    print("❌ Setup issues detected")
```

## Integration Modes

### 1. Auto-Instrumentation Mode (Recommended)
- **Best for**: Existing applications, zero code changes
- **Setup**: Single `auto_instrument()` call
- **Pros**: No code changes, automatic detection
- **Cons**: Global instrumentation effects

### 2. Manual Instrumentation Mode
- **Best for**: Controlled governance, specific structures
- **Setup**: Use `instrument_griptape()` wrapper
- **Pros**: Fine-grained control, isolated scope
- **Cons**: Requires code changes to use wrapper

### 3. Context Manager Mode
- **Best for**: Custom governance, detailed tracking
- **Setup**: Direct adapter usage with context managers
- **Pros**: Maximum control, custom attribution
- **Cons**: More verbose, manual tracking required

## Performance Considerations

- **Telemetry Overhead**: <3ms per structure execution
- **Memory Usage**: ~15MB for adapter with full monitoring
- **Network**: OTLP export in configurable batches
- **Sampling**: Configurable for high-volume applications

## 🤝 Support & Next Steps

### **Need Help?**
- 🚀 **[5-Minute Quickstart](../../docs/griptape-quickstart.md)** - Start here if you're new
- 📚 **[Complete Integration Guide](../../docs/integrations/griptape.md)** - Comprehensive documentation
- 🔧 **[Setup Validation](setup_validation.py)** - Run diagnostic checks
- 🐛 **[GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)** - Report bugs and request features
- 💬 **[Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)** - Community help and tips

### **Ready for Production?**
- 🐳 **Docker & Kubernetes**: See [integration guide](../../docs/integrations/griptape.md#production-deployment)
- 🏢 **Enterprise Deployment**: Full governance patterns and scaling
- 📊 **Monitoring Setup**: Grafana, Datadog, Honeycomb integration
- 🛡️ **Security & Compliance**: Enterprise governance templates

---

**⏰ Total Setup Time**: 5 minutes | **✨ Result**: Full GenOps governance for Griptape