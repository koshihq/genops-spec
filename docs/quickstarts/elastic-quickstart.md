# Elastic (Elasticsearch) - 5-Minute Quickstart

**Time to First Value: ≤ 5 minutes**

Get your AI governance telemetry flowing into Elasticsearch with zero-code setup. This quickstart demonstrates the fastest path from installation to seeing your first cost and policy data in Kibana.

## What You'll Accomplish

In just 5 minutes, you'll:

1. ✅ Connect GenOps to your Elasticsearch cluster
2. ✅ Track AI operations with zero code changes
3. ✅ Export cost and governance telemetry automatically
4. ✅ Query your data in Kibana with pre-built KQL examples

## Prerequisites

- **Elasticsearch 8.x or 9.x** (local or Elastic Cloud)
- **Kibana** (optional, for visualization)
- **Python 3.8+** with GenOps AI installed

**Don't have Elasticsearch?** Quick local setup:

```bash
# Using Docker (fastest way)
docker run -d --name elasticsearch \
  -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.12.0

# Verify it's running
curl http://localhost:9200
```

---

## Step 1: Set Environment Variables (1 minute)

Choose your connection method and set the corresponding environment variables.

### Option A: Elastic Cloud (Recommended for Production)

```bash
# Get your Cloud ID from: https://cloud.elastic.co/deployments
export ELASTIC_CLOUD_ID="your-deployment:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJGFiYzEyMw=="
export ELASTIC_API_KEY="your-api-key"  # Create in Kibana: Stack Management > API Keys
```

### Option B: Self-Hosted Elasticsearch

```bash
# For local development
export ELASTIC_URL="http://localhost:9200"

# For production (with authentication)
export ELASTIC_URL="https://es.yourcompany.com:9200"
export ELASTIC_API_KEY="your-api-key"
```

### Option C: Basic Authentication (Development Only)

```bash
export ELASTIC_URL="http://localhost:9200"
export ELASTIC_USERNAME="elastic"
export ELASTIC_PASSWORD="your-password"
```

**Verify your configuration:**

```bash
python -m genops.providers.elastic.validation
```

Expected output:
```
======================================================================
GenOps Elasticsearch Setup Validation
======================================================================

✅ Validation PASSED

📊 Cluster Information:
   • Cluster Name: elasticsearch
   • Version: 8.12.0

🔌 Connectivity: ✅ Connected
🔑 Permissions: ✅ Write access verified
⏱️  ILM Support: ✅ Available
```

---

## Step 2: Auto-Instrument (1 minute)

**Zero-code setup** - just call `auto_instrument()` and you're done!

```python
from genops.providers.elastic import auto_instrument

# Auto-detect connection from environment variables
adapter = auto_instrument(
    team="ml-platform",           # For cost attribution
    project="recommendations",    # For project tracking
    environment="development"     # development/staging/production
)
```

That's it! Your application is now exporting telemetry to Elasticsearch.

---

## Step 3: Track AI Operations (2 minutes)

Use the context manager to track any AI operation:

```python
# Track an AI operation
with adapter.track_ai_operation("gpt4-completion", customer_id="acme-corp") as span:

    # Your AI code here
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "Explain quantum computing"}]
    )

    # Record cost telemetry
    adapter.record_cost(
        span,
        cost=0.05,                # Total cost in USD
        provider="openai",
        model="gpt-4",
        tokens_input=50,
        tokens_output=150
    )

    # Record policy enforcement (optional)
    adapter.record_policy(
        span,
        policy_name="budget-constraint",
        result="allowed",         # "allowed", "blocked", or "warning"
        reason="Within monthly budget"
    )

print("✅ Operation tracked and exported to Elasticsearch!")
```

**For batch operations,** data is buffered and flushed automatically every 60 seconds or when 100 operations accumulate (configurable).

**Want realtime export?** Change the export mode:

```python
adapter = auto_instrument(
    team="ml-platform",
    export_mode="realtime"  # Export immediately after each operation
)
```

---

## Step 4: View in Kibana (1 minute)

### Configure Index Pattern

1. Open Kibana: `http://localhost:5601`
2. Navigate to: **Management → Stack Management → Index Patterns**
3. Create pattern: `genops-ai-*`
4. Select time field: `timestamp`
5. Click **Create index pattern**

### Query Your Data

Navigate to **Discover** and try these KQL queries:

**All AI operations for your team:**
```kql
genops.team: "ml-platform"
```

**Cost attribution by customer:**
```kql
genops.cost.total > 1.0 AND genops.customer_id: "acme-corp"
```

**Policy violations:**
```kql
genops.policy.result: "blocked"
```

**Operations by model:**
```kql
genops.cost.model: "gpt-4" OR genops.cost.model: "claude-3-sonnet"
```

**High-cost operations (> $1):**
```kql
genops.cost.total > 1.0
```

---

## What Just Happened?

Let's break down the telemetry flow:

```
Your Application
       ↓
   adapter.track_ai_operation()
       ↓
   Record cost/policy data
       ↓
   EventExporter (BATCH mode)
       ↓
   Buffer 100 operations OR wait 60 seconds
       ↓
   Bulk export via _bulk API
       ↓
   Elasticsearch Index: genops-ai-ml-platform-2025.01.18
       ↓
   Query in Kibana with KQL
```

**Index naming pattern:** `{prefix}-{namespace}-{date}`
- **Prefix:** `genops-ai` (configurable)
- **Namespace:** `ml-platform` (your team name)
- **Date:** `2025.01.18` (today's date)

**Automatic features enabled:**
- ✅ **Time-based indices:** Daily rollover for efficient querying
- ✅ **ILM (Index Lifecycle Management):** 90-day retention policy
- ✅ **Bulk indexing:** Optimized performance (100 ops/batch)
- ✅ **Background flush:** Automatic periodic export

---

## Complete Example (Copy-Paste Ready)

```python
import os
from genops.providers.elastic import auto_instrument

# Set connection (if not already in environment)
os.environ["ELASTIC_URL"] = "http://localhost:9200"

# Auto-instrument
adapter = auto_instrument(
    team="ml-platform",
    project="recommendations",
    environment="development"
)

# Track AI operations
with adapter.track_ai_operation("gpt4-completion", customer_id="acme-corp") as span:
    # Simulate AI call
    cost = 0.05

    # Record telemetry
    adapter.record_cost(
        span,
        cost=cost,
        provider="openai",
        model="gpt-4",
        tokens_input=50,
        tokens_output=150
    )

    adapter.record_policy(
        span,
        policy_name="budget-constraint",
        result="allowed"
    )

# Force flush (optional - automatic in batch mode)
adapter.flush()

print(f"✅ Telemetry exported to Elasticsearch!")
print(f"   • Query in Kibana: genops.team: \"ml-platform\"")

# Graceful shutdown (recommended)
adapter.shutdown()
```

**Run it:**

```bash
python your_script.py
```

**Expected output:**
```
✅ Telemetry exported to Elasticsearch!
   • Query in Kibana: genops.team: "ml-platform"
```

---

## Next Steps

You've successfully set up Elasticsearch integration in under 5 minutes! 🎉

### Recommended Next Steps:

1. **[Full Integration Guide](../integrations/elastic.md)** - Learn about advanced features:
   - Multi-provider cost tracking
   - Budget management
   - Policy enforcement
   - High-throughput optimization
   - Production deployment patterns

2. **[Example Integration](../../examples/observability/elastic_integration.py)** - Complete working example with:
   - OpenAI, Anthropic, and Bedrock integration
   - Kibana dashboard creation
   - Advanced KQL queries

3. **[Kibana Dashboards](../../observability/elastic/dashboards/)** - Pre-built dashboards for:
   - AI Operations Overview
   - Cost Attribution by Team/Project/Model
   - Governance & Compliance Tracking

### Production Checklist:

- [ ] Use `ELASTIC_API_KEY` instead of basic auth
- [ ] Enable HTTPS (`https://` URLs)
- [ ] Configure ILM retention policy for your needs
- [ ] Set `environment="production"` in production
- [ ] Add custom governance attributes (cost_center, feature, etc.)
- [ ] Monitor export performance with `adapter.get_metrics()`

---

## Common Questions

### Q1: Should I use batch, realtime, or hybrid mode?

**Decision Tree:**
- **Batch mode** (recommended for production): Best for most use cases. Exports in bulk every 60 seconds or when 100 operations accumulate, whichever comes first. Minimal network overhead and excellent performance.
- **Realtime mode**: Use for development/debugging when you need immediate visibility into each operation. Higher network overhead.
- **Hybrid mode**: Critical events (errors, policy violations) export immediately; normal operations batch. Best for monitoring production issues while maintaining efficiency.

**Quick rule:** Start with batch mode. Switch to realtime only for debugging.

### Q2: Why isn't my data appearing in Kibana?

**Most common causes:**
1. **Batch mode buffering**: Batch mode waits 60 seconds or 100 operations before exporting. Force flush with `adapter.exporter.flush()` for immediate export.
2. **Time range mismatch**: Check Kibana's time picker (top-right) includes your operation timestamps.
3. **Index pattern not refreshed**: Go to "Management → Index Patterns → genops-ai-* → Refresh fields"
4. **No data exported yet**: Run `adapter.get_metrics()` to check if operations were recorded and exported.

### Q3: How do I track custom business metrics?

**Simple approach** - use custom attributes:
```python
with adapter.track_ai_operation(
    "customer-support-query",
    customer_tier="premium",           # Custom attribute
    support_category="billing",        # Custom attribute
    ticket_id="TICKET-12345"          # Custom attribute
) as span:
    # Your AI operation
    adapter.record_cost(span, cost=0.05, provider="openai", model="gpt-4")
```

All custom attributes are automatically indexed in Elasticsearch and searchable in Kibana.

### Q4: What happens if Elasticsearch is unavailable?

**Graceful degradation:**
- Operations continue normally in your application
- Export failures are logged (check `adapter.get_metrics()` for failure counts)
- Batch mode: Events remain in memory buffer (up to `batch_size` operations)
- Events older than buffer are dropped (telemetry is non-blocking by design)

**Production recommendation:** Set up monitoring alerts on `export_failure_rate` metric.

### Q5: How much does Elasticsearch storage cost?

**Storage calculation:**
- ~500 bytes per AI operation (varies by attributes)
- 1 million operations/day ≈ 500 MB/day ≈ 15 GB/month
- With 90-day retention: ~45 GB total

**Cost optimization:**
1. Use ILM to automatically delete old data (default: 90 days)
2. Configure hot/warm/cold data tiers for older indices
3. Adjust retention based on compliance requirements (shorter = less storage)

**Example:** 100K operations/day × 90 days ≈ 4.5 GB (very manageable)

### Q6: Can I use this with Elastic Cloud?

**Yes!** Elastic Cloud is fully supported:

```python
from genops.providers.elastic import instrument_elastic

adapter = instrument_elastic(
    cloud_id="deployment-name:dXMtZWFzdC0xLmF3cy5mb3VuZC5pbyQ...",
    api_key="your-elastic-cloud-api-key",
    # ... rest of config
)
```

Create API keys in Kibana: **Management → Security → API Keys**

### Q7: How do I track multiple AI providers simultaneously?

**Pattern for multi-provider tracking:**
```python
adapter = instrument_elastic(...)

# Track OpenAI call
with adapter.track_ai_operation("openai-call") as span:
    adapter.record_cost(span, cost=0.05, provider="openai", model="gpt-4")

# Track Anthropic call in same session
with adapter.track_ai_operation("anthropic-call") as span:
    adapter.record_cost(span, cost=0.03, provider="anthropic", model="claude-3-sonnet")

# Costs automatically aggregated by provider in Elasticsearch
```

Query in Kibana: `genops.cost.provider: "openai" OR genops.cost.provider: "anthropic"`

### Q8: Can I customize the index names?

**Yes!** Use `index_prefix` and `namespace`:

```python
adapter = instrument_elastic(
    index_prefix="mycompany-ai",        # Custom prefix
    namespace="ml-platform",            # Your team/namespace
    # Creates indices like: mycompany-ai-ml-platform-2025.01.18
)
```

**Multi-tenant indexing:** Use different namespaces for different teams:
- Team A: `namespace="team-a"` → `genops-ai-team-a-*`
- Team B: `namespace="team-b"` → `genops-ai-team-b-*`

### Q9: What permissions does the API key need?

**Minimum required permissions:**
- `create_index` - Create daily indices automatically
- `write` - Index telemetry documents
- `manage_ilm` - Configure Index Lifecycle Management (optional)

**Create restricted API key in Kibana:**
1. Go to **Management → Security → API Keys**
2. Click **Create API Key**
3. Set privileges:
   ```json
   {
     "indices": [
       {
         "names": ["genops-ai-*"],
         "privileges": ["create_index", "write", "manage_ilm"]
       }
     ]
   }
   ```

### Q10: How do I migrate from another observability tool?

**Migration strategy:**
1. **Run in parallel** - Keep existing tool running while testing Elastic integration
2. **Use hybrid export** - Export to both systems during transition period
3. **Validate data** - Compare dashboards and metrics between systems
4. **Cutover gradually** - Move team by team or environment by environment

**Dual export example:**
```python
# Existing tool (Datadog, Honeycomb, etc.)
import existing_tool

# Add Elastic integration
from genops.providers.elastic import instrument_elastic
elastic_adapter = instrument_elastic(...)

# Track to both
def track_operation(name):
    existing_tool.track(name)  # Keep existing
    with elastic_adapter.track_ai_operation(name) as span:
        # New telemetry
        pass
```

Once validated, remove existing tool integration.

---

## Troubleshooting

### Connection Failed

**Problem:** `ElasticConnectionError: Connection failed`

**Solution:**
1. Verify Elasticsearch is running: `curl http://localhost:9200`
2. Check environment variables: `echo $ELASTIC_URL`
3. Run validation: `python -m genops.providers.elastic.validation`

### Authentication Failed

**Problem:** `ElasticAuthenticationError: Authentication failed`

**Solution:**
1. Verify credentials are correct
2. For API key: Check it hasn't expired in Kibana
3. For basic auth: Ensure user has `create_index` and `write` permissions
4. Test manually: `curl -H "Authorization: ApiKey YOUR_KEY" http://localhost:9200`

### No Data in Kibana

**Problem:** Index pattern created but no documents visible

**Solution:**
1. Check index exists: `curl http://localhost:9200/_cat/indices/genops-ai-*?v`
2. Force flush: `adapter.flush()` (batch mode buffers for 60s)
3. Verify time range in Kibana (top-right corner)
4. Check for errors: `adapter.get_metrics()`

### elasticsearch Package Not Installed

**Problem:** `ImportError: elasticsearch package is required`

**Solution:**
```bash
pip install 'genops-ai[elastic]'
# Or directly:
pip install elasticsearch>=8.0.0
```

### ILM Not Supported

**Problem:** Warning about ILM not available

**Solution:** ILM requires:
- Elasticsearch 6.6+ (you're using 8.x or 9.x, so this is OK)
- Appropriate Elasticsearch license (Basic license includes ILM)
- If using OSS version, ILM may not be available (non-critical warning)

---

## Support & Resources

- **Issues:** Report at [github.com/KoshiHQ/GenOps-AI/issues](https://github.com/KoshiHQ/GenOps-AI/issues)
- **Documentation:** Full integration guide at [docs/integrations/elastic.md](../integrations/elastic.md)
- **Examples:** Working code at [examples/observability/elastic_integration.py](../../examples/observability/elastic_integration.py)
- **Elasticsearch Docs:** [elastic.co/guide/en/elasticsearch/reference/current](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)

---

**You're all set!** Start tracking AI governance telemetry in Elasticsearch. 🚀
