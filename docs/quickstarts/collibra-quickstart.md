# Collibra 5-Minute Quickstart

Get started with GenOps + Collibra integration in under 5 minutes. This guide shows you how to automatically export AI governance telemetry to Collibra with zero code changes to your existing applications.

## What You'll Accomplish

- Set up Collibra integration with one line of code
- Export AI operation telemetry to Collibra automatically
- Track costs, policies, and governance metadata
- View results in your Collibra instance

**Time Required:** 5 minutes

---

## Prerequisites

Before starting, ensure you have:

1. **Collibra Instance Access**
   - Collibra Data Governance Center URL
   - Valid credentials (username/password or API token)
   - At least one domain created in Collibra

2. **GenOps Installed**
   ```bash
   pip install genops
   ```

3. **Python Environment**
   - Python 3.8 or higher
   - Basic familiarity with Python

---

## Step 1: Set Environment Variables (1 minute)

Configure your Collibra credentials as environment variables:

### Option A: Basic Authentication
```bash
export COLLIBRA_URL="https://your-instance.collibra.com"
export COLLIBRA_USERNAME="your-username"
export COLLIBRA_PASSWORD="your-password"
```

### Option B: API Token Authentication
```bash
export COLLIBRA_URL="https://your-instance.collibra.com"
export COLLIBRA_API_TOKEN="your-api-token"
```

### Optional: Set Governance Attributes
```bash
export GENOPS_TEAM="your-team-name"
export GENOPS_PROJECT="your-project-name"
```

**Verify Configuration:**
```bash
echo $COLLIBRA_URL
# Should output: https://your-instance.collibra.com
```

### Verify Setup (Optional)

Before proceeding, you can validate your configuration:

```bash
python -m genops.providers.collibra.validation
```

**Successful Validation Output:**
```
Collibra Integration Validation Report
============================================================

[SUCCESS] Connection Status: Connected
[SUCCESS] API Version: 7.0
[SUCCESS] Available Domains: 2 domains accessible
   - AI Governance (id: domain-abc123)
   - Data Quality (id: domain-def456)
[SUCCESS] Policy Access: 3 policies available

============================================================
[SUCCESS] Validation: PASSED
   Ready to integrate GenOps with Collibra!
============================================================
```

**Failed Validation Output:**
```
Collibra Integration Validation Report
============================================================

[ERROR] Connection Status: Not Connected

[ERROR] Errors:
  - Authentication failed. Check credentials:
    1. Verify username/password or API token
    2. Check if account has access to Collibra
    3. Verify credentials haven't expired

============================================================
[ERROR] Validation: FAILED
   Fix the errors above before proceeding.
============================================================
```

If validation passes, continue to Step 2. If it fails, check your environment variables.

---

## Step 2: Auto-Instrument Your Application (1 minute)

Add **one line** to initialize the Collibra integration. After this, you'll wrap your AI operations (shown in Step 3) to enable automatic tracking.

**The "One Line":**
```python
from genops.providers.collibra import auto_instrument

adapter = auto_instrument()  # ← This is the "one line" that enables integration
```

That's it! The integration is now active. Next, we'll show how to track operations.

---

## Step 3: Run Your First Operation (2 minutes)

Track an AI operation with automatic Collibra export:

```python
from genops.providers.collibra import auto_instrument

# Initialize adapter
adapter = auto_instrument(
    team="data-science",
    project="llm-experiment"
)

# Track AI operation (automatically exported to Collibra)
with adapter.track_ai_operation("gpt-4-completion") as span:
    # Your AI operation here
    # For example: result = openai.chat.completions.create(...)

    # Record cost information
    adapter.record_cost(
        span,
        cost=0.02,
        provider="openai",
        model="gpt-4",
        tokens_input=150,
        tokens_output=200
    )

# View metrics
metrics = adapter.get_metrics()
print(f"Operations tracked: {metrics['operation_count']}")
print(f"Total cost: ${metrics['total_cost']:.2f}")
print(f"Assets exported: {metrics['assets_exported']}")

# Flush remaining data to Collibra
adapter.flush()
adapter.shutdown()
```

**Run the script:**
```bash
python your_script.py
```

**Expected Output:**
```
Operations tracked: 1
Total cost: $0.02
Assets exported: 1
```

---

## Step 4: View Results in Collibra (1 minute)

1. **Log into your Collibra instance:**
   - Navigate to your Collibra URL
   - Sign in with your credentials

2. **Navigate to your AI Governance domain:**
   - Open the domain picker
   - Select your designated AI governance domain

3. **View exported assets:**
   - Search for "AI Operation Cost" asset type
   - Find your recent operation
   - View governance metadata: team, project, cost, tokens

4. **Explore the data:**
   - Cost information from your AI operations
   - Governance attributes (team, project, environment)
   - Token usage metrics
   - Timestamp and operation details

---

## What Just Happened?

Here's what GenOps + Collibra just did for you:

1. **Auto-Instrumentation**: `auto_instrument()` configured the integration using your environment variables
2. **Automatic Export**: Every operation tracked with `track_ai_operation()` was automatically exported to Collibra
3. **Cost Attribution**: Cost and token data were captured and sent to Collibra for governance
4. **Zero Code Changes**: Your existing AI code works unchanged - just wrap it in the context manager

### Key Benefits

- **100x Fewer API Calls**: Batch mode groups exports for efficiency
- **Transparent Governance**: All operations automatically tracked
- **Cost Visibility**: Real-time cost attribution to teams and projects
- **Audit Trail**: Complete operation history in Collibra

---

## Complete Example: Cost Limit Policy

Let's walk through a complete workflow from creating a policy in Collibra to enforcing it in your application.

### Scenario

You want to prevent any single AI operation from costing more than $5.00.

### Step 1: Create Policy in Collibra (2 minutes)

1. Log into your Collibra instance
2. Navigate to your "AI Governance" domain
3. Click "+ Asset" → Select "AI Cost Limit" asset type
4. Fill in the fields:
   - **Name**: "Production Cost Limit"
   - **Enforcement Level**: "block"
   - **Enabled**: true
   - **Max Cost**: 5.0
   - **Description**: "Prevent expensive operations in production"
5. Click "Save"

### Step 2: Import Policy in GenOps (< 1 minute)

```python
from genops.providers.collibra import auto_instrument

# Enable policy sync when creating adapter
adapter = auto_instrument(
    team="ml-platform",
    project="cost-governance",
    enable_policy_sync=True  # ← Enables policy import
)

# Wait 5 minutes for automatic sync, OR force immediate sync:
adapter.sync_policies()

print(f"Policies loaded: {len(adapter.get_imported_policies())}")
# Output: Policies loaded: 1
```

### Step 3: Policy Enforces Automatically (< 1 second)

Now when you run AI operations, the policy checks costs:

```python
from genops.core.policy import PolicyViolationError

# Operation 1: Low cost - ALLOWED
try:
    with adapter.track_ai_operation("small-task") as span:
        adapter.record_cost(span, cost=2.50, provider="openai", model="gpt-3.5-turbo")
    print("[OK] Small task completed: $2.50")
except PolicyViolationError:
    print("[BLOCKED] Operation exceeded cost limit")

# Output: [OK] Small task completed: $2.50


# Operation 2: High cost - BLOCKED
try:
    with adapter.track_ai_operation("expensive-task") as span:
        adapter.record_cost(span, cost=8.00, provider="openai", model="gpt-4")
    print("[OK] Expensive task completed: $8.00")
except PolicyViolationError as e:
    print(f"[BLOCKED] {e.policy_name}: {e.message}")

# Output: [BLOCKED] cost_limit_production: Cost $8.00 exceeds maximum $5.00
```

### Step 4: View Results in Collibra (1 minute)

1. Return to your Collibra domain
2. You'll see new "AI Operation Cost" assets:
   - **small-task**: Cost $2.50, Policy Result: ALLOWED
   - **expensive-task**: Cost $8.00, Policy Result: BLOCKED

The complete governance cycle: Create policy → Import → Enforce → Export results.

For more policy types and advanced scenarios, see the [Policy Mapping Guide](../policies/collibra-policy-mapping.md).

---

## Next Steps

### Explore More Features

1. **Manual Instrumentation** - Learn advanced configuration options
   ```bash
   python examples/collibra/02_basic_export.py
   ```
   [View Example](../../examples/collibra/02_basic_export.py)

2. **Policy Import** - Import and enforce Collibra policies at runtime
   ```bash
   python examples/collibra/03_policy_import.py
   ```
   [View Example](../../examples/collibra/03_policy_import.py)

3. **Read Full Integration Guide** - Comprehensive documentation
   - [Collibra Integration Guide](../integrations/collibra.md)
   - [Policy Mapping Reference](../policies/collibra-policy-mapping.md)

### Common Configurations

**Enable Real-Time Export:**
```python
adapter = auto_instrument(export_mode="realtime")
```

**Set Daily Budget Limit:**
```python
adapter = auto_instrument(daily_budget_limit=100.0)
```

**Enable Policy Sync:**
```python
adapter = auto_instrument(enable_policy_sync=True)
```

### Understanding Policy Sync Timing

**Important**: Policy changes in Collibra take up to **5 minutes** to apply in GenOps due to the background sync interval.

**Timeline**:
1. You create a policy in Collibra (t=0)
2. GenOps syncs policies every 5 minutes (t=5 min)
3. New policy is now enforced on operations (t=5+ min)

**For Immediate Policy Updates**:
```python
# Manual sync - updates policies immediately
adapter.sync_policies()
```

This is especially useful during development when you're testing policy configurations.

---

## Troubleshooting

### Issue: "COLLIBRA_URL not set"
**Solution:**
```bash
export COLLIBRA_URL="https://your-instance.collibra.com"
```

### Issue: "Authentication failed"
**Solutions:**
1. Verify credentials are correct
2. Check if account has Collibra access
3. Ensure credentials haven't expired
4. Try API token instead of username/password

### Issue: "No domains found"
**Solution:**
- Create at least one domain in Collibra
- Or specify explicit `domain_id` parameter:
  ```python
  adapter = auto_instrument(domain_id="your-domain-id")
  ```

### Issue: "Connection timeout"
**Solutions:**
1. Verify Collibra URL is accessible
2. Check network connectivity
3. Confirm firewall allows HTTPS traffic
4. Try increasing timeout:
   ```python
   from genops.providers.collibra import GenOpsCollibraAdapter
   adapter = GenOpsCollibraAdapter(timeout=60)
   ```

### Run Validation

Test your setup with the built-in validation utility:

```bash
python -m genops.providers.collibra.validation
```

This will check:
- Environment variables
- URL format
- Authentication
- Connectivity
- Domain access

---

## Additional Resources

- **Full Documentation**: [Collibra Integration Guide](../integrations/collibra.md)
- **Policy Reference**: [Collibra Policy Mapping](../policies/collibra-policy-mapping.md)
- **Example Code**: [Collibra Examples](../../examples/collibra/)
- **API Reference**: [GenOpsCollibraAdapter API](../integrations/collibra.md#api-reference)

---

## Support

Need help? Here are your options:

1. **Documentation**: Read the [full integration guide](../integrations/collibra.md)
2. **Examples**: Explore [working examples](../../examples/collibra/)
3. **Issues**: Report issues at [GitHub Issues](https://github.com/anthropics/claude-code/issues)
4. **Community**: Ask questions in the GenOps community

---

**Congratulations!** You've successfully integrated GenOps with Collibra. Your AI operations now have transparent governance and cost attribution.
