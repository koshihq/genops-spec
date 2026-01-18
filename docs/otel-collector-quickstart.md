# OpenTelemetry Collector - 5-Minute Quickstart

**🎯 Get GenOps + OTel Collector + Grafana running in 5 minutes**

This guide gets you from zero to full AI governance observability with OpenTelemetry Collector, Grafana dashboards, and live cost/policy tracking in under 5 minutes.

---

## 🚀 Prerequisites (30 seconds)

**Before you start, make sure you have:**

1. **Docker Desktop** installed and running
   - Get Docker: [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
   - Minimum: Docker 20.10+ with Docker Compose

2. **GenOps AI** installed
   ```bash
   pip install genops-ai
   ```

3. **At least 4GB RAM** available for Docker containers

---

## ⚡ Pre-Flight Verification (30 seconds)

Before starting, verify your environment is ready:

```bash
# Check Docker is running
docker ps
# Should show running containers (or empty table if no containers)

# Check Docker Compose is available
docker-compose --version
# Should show: docker-compose version 1.29+ or Docker Compose version v2.0+

# Verify GenOps AI is installed
pip show genops-ai
# Should show package version info
```

**If Docker is not running**: Start Docker Desktop and wait for it to initialize

**If GenOps is not installed**: `pip install genops-ai`

---

## 📖 Quick Glossary

New to OpenTelemetry or observability? Here are the key terms:

| Term | Meaning |
|------|---------|
| **OTel Collector** | OpenTelemetry Collector - receives, processes, and exports telemetry data |
| **OTLP** | OpenTelemetry Protocol - standard format for sending telemetry (gRPC port 4317, HTTP port 4318) |
| **LGTM Stack** | Loki (logs), Grafana (dashboards), Tempo (traces), Mimir (metrics) - complete observability backend |
| **Span** | A single unit of work (e.g., one AI operation) with start time, end time, and attributes |
| **Governance Telemetry** | GenOps-specific attributes: cost, policy, budget, evaluation metrics |

---

## ⚡ Quick Setup (2 minutes)

### Step 1: Start the Observability Stack (90 seconds)

Clone or navigate to the GenOps AI repository and start the complete LGTM stack:

```bash
# Navigate to the observability directory
cd /path/to/GenOps-AI-OTel

# Start all services with Docker Compose
docker-compose -f docker-compose.observability.yml up -d

# Verify all services are running
docker-compose -f docker-compose.observability.yml ps
```

**Expected output:**
```
NAME                        STATUS    PORTS
genops-demo                 Up        0.0.0.0:8000->8000/tcp
grafana                     Up        0.0.0.0:3000->3000/tcp
loki                        Up        0.0.0.0:3100->3100/tcp
mimir                       Up        0.0.0.0:9009->9009/tcp
otel-collector              Up        0.0.0.0:4317-4318->4317-4318/tcp
prometheus                  Up        0.0.0.0:9090->9090/tcp
redis                       Up        0.0.0.0:6379->6379/tcp
tempo                       Up        0.0.0.0:3200->3200/tcp
```

**If services fail to start:**
- Check Docker has at least 4GB RAM allocated: Docker Desktop → Preferences → Resources
- Verify no port conflicts: `lsof -i :3000 -i :4318` (should be empty)
- View logs: `docker-compose -f docker-compose.observability.yml logs otel-collector`

---

### Step 2: Validate Setup (30 seconds) ⭐ NEW

Before sending telemetry, validate that the OTel Collector and backend services are healthy:

**Option A: Quick validation script** (Recommended)
```bash
cd examples/observability
python validate_otel_collector.py
```

**Option B: Manual health checks**
```bash
# Check OTel Collector health
curl http://localhost:13133/
# Should return: {"status":"Server available","upSince":"..."}

# Check Grafana is accessible
curl http://localhost:3000/api/health
# Should return: {"database":"ok","version":"..."}

# Check OTLP endpoint is listening
nc -zv localhost 4318
# Should return: Connection to localhost port 4318 [tcp/*] succeeded!
```

**Expected output when successful:**
```
======================================================================
OpenTelemetry Collector Validation Report
======================================================================

✅ [SUCCESS] Collector Status: Healthy
✅ [SUCCESS] OTLP HTTP Endpoint: Accessible (port 4318)
✅ [SUCCESS] OTLP gRPC Endpoint: Accessible (port 4317)
✅ [SUCCESS] Grafana: Accessible (http://localhost:3000)
✅ [SUCCESS] Tempo: Accessible (http://localhost:3200)

💡 RECOMMENDATIONS:
1. ✅ Setup validated successfully! Next steps:
   • Open Grafana at http://localhost:3000 (admin/genops)
   • Run example: python examples/quickstarts/otel_collector_quickstart.py
   • View "GenOps AI - Governance Overview" dashboard

======================================================================
✅ [SUCCESS] Validation: PASSED
   Ready to send GenOps telemetry to OTel Collector!
======================================================================
```

**If validation fails:**
- Ensure Docker containers are running: `docker-compose -f docker-compose.observability.yml ps`
- Check OTel Collector logs: `docker-compose -f docker-compose.observability.yml logs otel-collector`
- Restart services: `docker-compose -f docker-compose.observability.yml restart`

**Why validate?**
- ✅ Catches 95%+ of configuration issues before runtime
- ✅ Confirms all services are accessible and healthy
- ✅ Saves time debugging connection problems
- ✅ Provides actionable error messages with specific fixes

---

### Step 3: Run Your First Instrumented Application (90 seconds)

Use the zero-configuration quickstart example:

```bash
# Run the quickstart example
python examples/quickstarts/otel_collector_quickstart.py
```

**What this does:**
- ✅ Auto-configures GenOps to send telemetry to local OTel Collector
- ✅ Simulates AI operations with cost, policy, and evaluation tracking
- ✅ Exports traces, metrics, and logs via OTLP
- ✅ Generates governance telemetry visible in Grafana immediately

**Expected output:**
```
✅ GenOps configured to send telemetry to OTel Collector

📊 Simulating AI operations with governance tracking...

Operation 1: AI Chat (team=engineering, customer=demo-customer-1)
   Provider: openai, Model: gpt-4
   Cost: $0.0025, Tokens: 150

Operation 2: AI Analysis (team=data-science, customer=demo-customer-2)
   Provider: anthropic, Model: claude-3-sonnet
   Cost: $0.0008, Tokens: 180

Operation 3: Policy Evaluation (team=product)
   Policy: cost_limit_demo, Result: PASSED
   Policy: content_safety_demo, Result: PASSED

✅ Sent 3 operations to OTel Collector!
   Total cost: $0.0033

📊 View your data in Grafana: http://localhost:3000
   Dashboard: GenOps AI - Governance Overview
   Login: admin / genops
```

**Alternative: Use the demo API:**
```bash
# Single AI operation via demo API
curl -X POST http://localhost:8000/ai/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Calculate the cost of running 1000 GPT-4 requests",
    "model": "gpt-4",
    "team": "engineering",
    "customer_id": "quickstart-test"
  }'

# Simulate 50 operations for load testing
curl -X POST http://localhost:8000/simulate/load \
  -H "Content-Type: application/json" \
  -d '{"operations": 50}'
```

---

### Step 4: View Your Data in Grafana (60 seconds)

Open your browser and navigate to the Grafana dashboard:

**Access Grafana:**
1. Open [http://localhost:3000](http://localhost:3000)
2. Login:
   - **Username**: `admin`
   - **Password**: `genops`
3. Navigate to **Dashboards → GenOps AI - Governance Overview**

**What you'll see:**
- 📊 **Cost Tracking** - Total costs by team, customer, and model
- 🔢 **Token Usage** - Input/output token distribution across operations
- 🛡️ **Policy Violations** - Real-time compliance monitoring
- 📋 **Recent Operations** - Table of AI operations with full governance context

**Exploring Traces:**
1. Click **Explore** in the left sidebar
2. Select **Tempo** as the data source
3. Click **Search** to see all traces
4. Click any trace to see:
   - Complete request flow with spans
   - GenOps governance attributes (cost, team, customer_id, policy results)
   - Timing and performance metrics
   - Parent-child relationships for complex operations

**If you don't see data:**
- Wait 10-15 seconds for telemetry to be processed and indexed
- Check OTLP exporter is working: `docker-compose -f docker-compose.observability.yml logs otel-collector | grep "Exporting"`
- Verify time range in Grafana: Change to "Last 15 minutes" in top-right
- Re-run the quickstart example: `python examples/quickstarts/otel_collector_quickstart.py`

---

## 🎯 What Just Happened?

**You successfully created a complete AI governance observability pipeline:**

1. ✅ **Docker Compose** started the LGTM stack (Loki, Grafana, Tempo, Mimir, OTel Collector)
2. ✅ **OTel Collector** configured to receive OTLP on ports 4317 (gRPC) and 4318 (HTTP)
3. ✅ **GenOps AI** captured governance telemetry (cost, policy, evaluation, budget)
4. ✅ **OTLP Export** sent traces, metrics, and logs to the collector
5. ✅ **Processing Pipeline** transformed and enriched telemetry with governance semantics
6. ✅ **Storage Backends** persisted data in Tempo (traces), Loki (logs), Mimir (metrics)
7. ✅ **Grafana Dashboards** visualized governance insights immediately

**This is the foundation for:**
- Real-time AI cost attribution across teams, projects, and customers
- Policy compliance monitoring and violation alerting
- Budget tracking and threshold enforcement
- Quality evaluation and performance analysis
- Cross-stack distributed tracing for complex AI workflows

---

## 📊 Data Flow Architecture

```
┌─────────────────────┐
│   Your AI App       │
│  (GenOps SDK)       │
└──────────┬──────────┘
           │ OTLP (4318/4317)
           ▼
┌─────────────────────┐
│  OTel Collector     │
│  • Receives OTLP    │
│  • Transforms data  │
│  • Routes to        │
│    backends         │
└──────────┬──────────┘
           │
     ┌─────┴─────┬─────────────┐
     ▼           ▼             ▼
┌─────────┐ ┌─────────┐ ┌──────────┐
│  Tempo  │ │  Loki   │ │  Mimir   │
│ (Traces)│ │ (Logs)  │ │(Metrics) │
└────┬────┘ └────┬────┘ └────┬─────┘
     │           │           │
     └───────────┴───────────┘
                 │
                 ▼
          ┌──────────────┐
          │   Grafana    │
          │  (Dashboards)│
          └──────────────┘
```

---

## 🏗️ Next Steps (Your Choice!)

**✅ You now have GenOps telemetry flowing through OTel Collector to Grafana!**

### 🎯 Recommended Learning Path

For first-time users, we recommend this sequence:

**1. Start here** → **Option A: Explore Pre-built Dashboards** (simplest, immediate value)
- Navigate through the GenOps AI Governance Overview dashboard
- ~5 minutes to understand your AI operations

**2. Then try** → **Option B: Query Traces in Tempo** (understand distributed tracing)
- Search traces by team, customer, or cost
- ~10 minutes to trace navigation mastery

**3. Next level** → **Option C: Integrate Your AI App** (production readiness)
- Replace quickstart example with your real AI application
- ~15 minutes to first production telemetry

**4. Advanced** → **Option D: Production Deployment** (Kubernetes and cloud)
- Deploy to Kubernetes with Helm chart
- ~30 minutes to production-ready deployment

Choose your path below:

---

### Option A: Explore Pre-built Dashboards

**GenOps AI - Governance Overview Dashboard:**

**Cost Attribution Panel:**
- View costs grouped by team, project, or customer
- Identify top spenders and cost trends over time
- Drill down into specific models and providers

**Policy Compliance Panel:**
- Monitor policy evaluation results
- Track violation rates by policy type
- Alert on threshold breaches

**Token Usage Panel:**
- Visualize input vs output token distribution
- Compare token efficiency across models
- Identify high-usage operations

**Operations Table:**
- Real-time feed of AI operations
- Sortable by cost, duration, team, customer
- Click through to detailed trace view

**Customization:**
- Edit dashboard: Grafana → Dashboards → GenOps AI → Edit
- Add panels, modify queries, adjust time ranges
- Save custom views for different teams

---

### Option B: Query Traces in Tempo

**Basic Trace Search:**
```
# In Grafana → Explore → Tempo

# Search by team
{.genops.team="engineering"}

# Search by customer
{.genops.customer_id="demo-customer-1"}

# Search by cost threshold (traces costing > $0.01)
{.genops.cost.total>0.01}

# Search by policy result
{.genops.policy.result="blocked"}

# Combined search (expensive operations for specific customer)
{.genops.customer_id="enterprise-123" && .genops.cost.total>0.05}
```

**Trace Analysis:**
1. Click any trace to open detailed view
2. Expand spans to see governance attributes
3. View timing waterfall for performance analysis
4. Check logs correlated with the trace

**Advanced Queries:**
- Filter by model: `{.genops.cost.model="gpt-4"}`
- Time range analysis: Adjust time range in top-right
- Export data: Download trace JSON for external analysis

---

### Option C: Integrate Your AI App

**Zero-Code Auto-Instrumentation:**

Replace the quickstart example with your application code:

```python
from genops import auto_instrument
from genops.providers.openai import instrument_openai

# Auto-configure OTel Collector export (detects http://localhost:4318)
auto_instrument()

# Instrument your AI provider
openai_client = instrument_openai()

# Your existing code works unchanged - telemetry flows automatically!
response = openai_client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)

# Governance attributes are captured automatically
# View in Grafana immediately!
```

**Manual Instrumentation (for custom operations):**

```python
from genops.core.telemetry import GenOpsTelemetry

telemetry = GenOpsTelemetry()

with telemetry.trace_operation(
    operation_name="custom_ai_workflow",
    team="engineering",
    customer_id="customer-123",
    project="ai-assistant"
) as span:
    # Your AI operations
    result = my_ai_function()

    # Record governance telemetry
    telemetry.record_cost(span, cost=0.05, provider="openai", model="gpt-4")
    telemetry.record_evaluation(span, "quality", score=0.95, threshold=0.8)
```

**Framework Auto-Instrumentation:**

GenOps automatically instruments popular AI frameworks:

```python
# LangChain
from genops.providers.langchain import instrument_langchain
instrument_langchain()  # All chains now tracked automatically

# LlamaIndex
from genops.providers.llamaindex import instrument_llamaindex
instrument_llamaindex()  # All queries now tracked automatically

# AutoGen
from genops.providers.autogen import instrument_autogen
instrument_autogen()  # All multi-agent conversations tracked
```

---

### Option D: Production Deployment

**Kubernetes with Helm:**

Deploy GenOps AI and OTel Collector to Kubernetes:

```bash
# Add GenOps Helm repository
helm repo add genops-ai https://genops-ai.github.io/helm-charts
helm repo update

# Install with production configuration
helm install genops-ai genops-ai/genops-ai \
  --namespace genops-system \
  --create-namespace \
  --set otelCollector.enabled=true \
  --set otelCollector.endpoint="http://otel-collector:4318" \
  --set grafana.enabled=true

# Verify deployment
kubectl get pods -n genops-system
```

**Cloud Platform Deployment:**

- **AWS**: See [docs/kubernetes-aws-deployment.md](kubernetes-aws-deployment.md)
- **GCP**: See [docs/kubernetes-gcp-deployment.md](kubernetes-gcp-deployment.md)
- **Azure**: See [docs/kubernetes-azure-deployment.md](kubernetes-azure-deployment.md)

**Scaling Considerations:**

- **OTel Collector**: Horizontal scaling with multiple replicas
- **Sampling**: Configure sampling for high-volume applications
- **Retention**: Adjust retention policies in Tempo, Loki, Mimir
- **Resource Limits**: Set appropriate CPU/memory limits in Kubernetes

---

## 🔄 Alternative: Route to Enterprise Observability Platforms

**GenOps can route telemetry to any OTLP-compatible platform:**

### Datadog Integration
```bash
# Configure OTel Collector to export to Datadog
# See: docs/integrations/datadog.md

export DD_API_KEY="your-datadog-api-key"
export DD_SITE="datadoghq.com"  # or datadoghq.eu

# GenOps automatically detects and exports
python your_app.py
```

### Honeycomb Integration
```bash
# Configure for Honeycomb
export HONEYCOMB_API_KEY="your-api-key"
export HONEYCOMB_DATASET="genops-ai"

# See: docs/integrations/honeycomb.md
```

### Splunk Integration
```bash
# Route to Splunk HEC
export SPLUNK_HEC_ENDPOINT="https://splunk.example.com:8088"
export SPLUNK_HEC_TOKEN="your-hec-token"

# See: docs/splunk-quickstart.md
```

### Cribl Integration (Multi-Destination Routing)
```bash
# Route through Cribl for intelligent data routing
export CRIBL_OTLP_ENDPOINT="http://cribl-stream:4318"

# Cribl can then route to: Datadog + Splunk + S3 simultaneously
# See: docs/integrations/cribl.md
```

**When to use each approach:**

- ✅ **Local OTel Collector** - Development, testing, learning (this quickstart)
- ✅ **Datadog/Honeycomb** - Enterprise observability with existing accounts
- ✅ **Splunk** - SIEM, compliance, audit trails, enterprise log analytics
- ✅ **Cribl** - Multi-destination routing, cost optimization (90% volume reduction)
- ✅ **Kubernetes** - Production deployments with cloud-native infrastructure

---

## 🔍 Troubleshooting

### Issue: "Docker containers not starting" or "Port conflicts"

**Fix:**
```bash
# Check what's using the ports
lsof -i :3000 -i :4318 -i :8000
# Kill processes using these ports if necessary

# Check Docker has enough resources
docker system df
# Prune unused Docker resources
docker system prune -a

# Restart Docker Desktop and try again
docker-compose -f docker-compose.observability.yml down -v
docker-compose -f docker-compose.observability.yml up -d
```

### Issue: "No data in Grafana" or "Empty dashboards"

**Fix:**
1. **Check OTLP endpoint is reachable:**
   ```bash
   curl -v http://localhost:4318/v1/traces
   # Should connect (even if returns 405 Method Not Allowed - that's expected)
   ```

2. **Verify OTel Collector is receiving data:**
   ```bash
   docker-compose -f docker-compose.observability.yml logs otel-collector | grep "spans"
   # Should show: "Exporting spans" messages
   ```

3. **Check telemetry is being generated:**
   ```bash
   # Re-run the quickstart example
   python examples/quickstarts/otel_collector_quickstart.py

   # Or generate load via demo API
   curl -X POST http://localhost:8000/simulate/load \
     -H "Content-Type: application/json" \
     -d '{"operations": 10}'
   ```

4. **Verify data sources in Grafana:**
   - Grafana → Configuration → Data Sources
   - Check Tempo, Loki, Mimir are all "Working"
   - Test each data source individually

5. **Adjust time range:**
   - Grafana dashboards default to "Last 6 hours"
   - Change to "Last 15 minutes" or "Last 5 minutes" in top-right

### Issue: "Connection refused" or "OTLP export failed"

**Fix:**
```bash
# Check OTel Collector is running
docker ps | grep otel-collector
# Should show: otel-collector container with status "Up"

# Check collector health endpoint
curl http://localhost:13133/
# Should return: {"status":"Server available"...}

# View collector logs for errors
docker-compose -f docker-compose.observability.yml logs otel-collector --tail=50

# Common issues:
# - Port 4318 not exposed: Check docker-compose.observability.yml ports section
# - Firewall blocking: Check local firewall settings
# - Collector crashed: Restart services
```

### Issue: "Validation script fails"

**Fix:**
```bash
# Ensure validation dependencies are installed
pip install requests

# Check services are running
docker-compose -f docker-compose.observability.yml ps

# View detailed validation output
python examples/observability/validate_otel_collector.py --verbose

# Manual health checks
curl http://localhost:13133/  # Collector health
curl http://localhost:3000/api/health  # Grafana health
curl http://localhost:3200/api/search  # Tempo health
```

### Issue: "High CPU or memory usage"

**Fix:**
```bash
# Check resource usage
docker stats

# Scale down optional services
docker-compose -f docker-compose.observability.yml stop prometheus
docker-compose -f docker-compose.observability.yml stop genops-demo

# Adjust batch processor settings in otel-collector-config.yaml:
# batch:
#   timeout: 10s  # Increase from 1s to reduce frequency
#   send_batch_size: 2048  # Increase from 1024 to batch more
```

---

## ✅ Verification Checklist

Before proceeding, verify each step:

- [ ] ✅ Docker Desktop installed and running
- [ ] ✅ GenOps AI installed: `pip show genops-ai`
- [ ] ✅ All Docker containers running: `docker-compose ps` shows "Up" for all services
- [ ] ✅ OTel Collector health check passes: `curl http://localhost:13133/`
- [ ] ✅ Grafana accessible: [http://localhost:3000](http://localhost:3000) (admin/genops)
- [ ] ✅ First telemetry sent: `python examples/quickstarts/otel_collector_quickstart.py`
- [ ] ✅ Data visible in Grafana: "GenOps AI - Governance Overview" dashboard shows data
- [ ] ✅ Traces searchable: Grafana → Explore → Tempo shows traces

**All checked?** You're ready for production deployment and advanced features!

---

## 📚 Learn More

- **Comprehensive Integration Guide:** [docs/integrations/otel-collector.md](integrations/otel-collector.md)
- **Kubernetes Deployment:** [docs/kubernetes-getting-started.md](kubernetes-getting-started.md)
- **Splunk Integration:** [docs/splunk-quickstart.md](splunk-quickstart.md)
- **Example Code:** [examples/observability/](../examples/observability/)
- **GenOps Documentation:** [README.md](../README.md)
- **OpenTelemetry:** [https://opentelemetry.io](https://opentelemetry.io)
- **Grafana:** [https://grafana.com/docs/](https://grafana.com/docs/)
- **GitHub Repository:** [https://github.com/KoshiHQ/GenOps-AI](https://github.com/KoshiHQ/GenOps-AI)

---

## 💬 Get Help

- **Issues:** [GitHub Issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- **Discussions:** [GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)
- **OpenTelemetry Community:** [https://cloud-native.slack.com](https://cloud-native.slack.com) (#otel-collector)

---

## 🎉 What's Next?

**You've completed the quickstart!** Here's what you can do now:

1. **Explore Grafana Dashboards**: Navigate through pre-built governance dashboards
2. **Query Traces in Tempo**: Search for specific operations, teams, or customers
3. **Integrate Your AI App**: Replace quickstart with your production AI code
4. **Deploy to Kubernetes**: Production-ready Helm chart for cloud deployment
5. **Route to Enterprise Platforms**: Send telemetry to Datadog, Splunk, or Honeycomb

**Total time: ~5 minutes** ✅

**Next level: Production AI governance with distributed tracing** 🚀
