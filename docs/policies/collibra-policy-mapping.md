# Collibra Policy Mapping Reference

Complete reference for creating and mapping governance policies between Collibra Data Governance Center and GenOps PolicyEngine.

## Table of Contents

1. [Overview](#overview)
2. [Policy Architecture](#policy-architecture)
3. [Supported Policy Types](#supported-policy-types)
4. [Creating Policies in Collibra](#creating-policies-in-collibra)
5. [Policy Translation](#policy-translation)
6. [Enforcement Levels](#enforcement-levels)
7. [Policy Examples](#policy-examples)
8. [Troubleshooting](#troubleshooting)

---

## Overview

### What is Policy Mapping?

Policy mapping is the process of translating governance policies defined in Collibra into runtime enforcement rules in GenOps. This enables:

- **Centralized Policy Management**: Define policies once in Collibra
- **Runtime Enforcement**: Policies automatically enforced on AI operations
- **Audit Trail**: Policy evaluations tracked and reported back to Collibra
- **Compliance**: Maintain governance standards across all AI systems

### Policy Flow

```
┌──────────────────┐
│ Define Policy    │  1. Create policy in Collibra UI
│  in Collibra     │     Select type, set conditions
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Automatic Import │  2. GenOps imports policy
│  (Every 5 min)   │     Translates to PolicyConfig
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Runtime Check    │  3. Policy evaluated before operation
│  (Pre-execution) │     Block/warn/allow based on rules
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Export Result    │  4. Evaluation result sent to Collibra
│  to Collibra     │     Complete audit trail
└──────────────────┘
```

---

## Policy Architecture

### Collibra Policy Components

Each policy in Collibra has these components:

| Component | Purpose | Example |
|-----------|---------|---------|
| **Asset Type** | Categorizes policy | "AI Cost Limit" |
| **Name** | Human-readable identifier | "Production Cost Limit" |
| **Description** | Policy purpose and scope | "Max $10 per operation" |
| **Enforcement Level** | Action to take on violation | "block" |
| **Enabled** | Whether policy is active | true |
| **Conditions** | Policy-specific rules | max_cost: 10.0 |

### GenOps PolicyConfig

Policies are translated to GenOps `PolicyConfig` format:

```python
from genops.core.policy import PolicyConfig, PolicyResult

policy = PolicyConfig(
    name="cost_limit_policy-001",           # Unique name
    description="Max $10 per operation",     # Human description
    enabled=True,                            # Active status
    enforcement_level=PolicyResult.BLOCKED,  # Enforcement action
    conditions={                             # Policy-specific rules
        "max_cost": 10.0
    }
)
```

---

## Supported Policy Types

GenOps supports 6 policy types for AI governance:

### Summary Table

| # | Collibra Asset Type | GenOps Policy Name | Purpose |
|---|--------------------|--------------------|---------|
| 1 | AI Cost Limit | `cost_limit` | Enforce max cost per operation |
| 2 | AI Rate Limit | `rate_limit` | Throttle request rate |
| 3 | Content Filter | `content_filter` | Block sensitive content patterns |
| 4 | Team Access Control | `team_access` | Restrict operations to specific teams |
| 5 | Budget Constraint | `budget_limit` | Enforce daily/monthly budgets |
| 6 | Model Governance | `model_governance` | Control allowed/blocked models |

---

## Policy Type 1: AI Cost Limit

### Purpose

Prevent operations exceeding specified cost thresholds.

### Collibra Configuration

**Asset Type:** `AI Cost Limit`

**Required Attributes:**

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `enforcement_level` | string | Action on violation | "block" |
| `enabled` | boolean | Policy active status | true |
| `max_cost` | float | Maximum cost (USD) | 10.0 |

**Optional Attributes:**

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `description` | string | Policy description | "Production cost limit" |
| `cost_limit` | float | Alternative to max_cost | 10.0 |

### GenOps Translation

```python
PolicyConfig(
    name="cost_limit_policy-001",
    description="AI Cost Limit - Production cost limit",
    enabled=True,
    enforcement_level=PolicyResult.BLOCKED,
    conditions={
        "max_cost": 10.0
    }
)
```

### Runtime Behavior

```python
# Policy: max_cost = 10.0

# Operation 1: Cost $5 → ALLOWED
with adapter.track_ai_operation("low-cost-op") as span:
    adapter.record_cost(span, cost=5.0)  # Proceeds

# Operation 2: Cost $15 → BLOCKED
with adapter.track_ai_operation("high-cost-op") as span:
    adapter.record_cost(span, cost=15.0)  # Raises PolicyViolationError
```

### Creation in Collibra UI

1. Navigate to **Assets > Create Asset**
2. Select **Asset Type**: "AI Cost Limit"
3. Set **Name**: "Production Cost Limit"
4. Add **Attributes**:
   - `enforcement_level`: "block"
   - `enabled`: "true"
   - `max_cost`: "10.0"
5. Set **Description**: "Maximum $10 per AI operation in production"
6. Assign to **Domain**: Your AI Governance domain
7. **Save** asset

---

## Policy Type 2: AI Rate Limit

### Purpose

Control request rate to prevent resource abuse and manage costs.

### Collibra Configuration

**Asset Type:** `AI Rate Limit`

**Required Attributes:**

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `enforcement_level` | string | Action on violation | "rate_limit" |
| `enabled` | boolean | Policy active status | true |
| `max_requests_per_minute` | integer | Request limit | 100 |

**Optional Attributes:**

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `max_requests` | integer | Alternative attribute | 100 |
| `rate_limit` | integer | Alternative attribute | 100 |

### GenOps Translation

```python
PolicyConfig(
    name="rate_limit_policy-002",
    description="AI Rate Limit - 100 req/min",
    enabled=True,
    enforcement_level=PolicyResult.RATE_LIMITED,
    conditions={
        "max_requests_per_minute": 100
    }
)
```

### Runtime Behavior

- Token bucket algorithm for smooth rate limiting
- Requests exceeding limit are delayed (not blocked)
- Per-team or per-project limits supported

### Creation in Collibra UI

1. Create asset with **Asset Type**: "AI Rate Limit"
2. Set **Name**: "Team Rate Limit"
3. Add **Attributes**:
   - `enforcement_level`: "rate_limit"
   - `enabled`: "true"
   - `max_requests_per_minute`: "100"
4. **Save** asset

---

## Policy Type 3: Content Filter

### Purpose

Block operations containing sensitive or prohibited content patterns.

### Collibra Configuration

**Asset Type:** `Content Filter`

**Required Attributes:**

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `enforcement_level` | string | Action on violation | "block" |
| `enabled` | boolean | Policy active status | true |
| `blocked_patterns` | string | Comma-separated patterns | "confidential,secret,private" |

### GenOps Translation

```python
PolicyConfig(
    name="content_filter_policy-003",
    description="Content Filter - Block sensitive terms",
    enabled=True,
    enforcement_level=PolicyResult.BLOCKED,
    conditions={
        "blocked_patterns": ["confidential", "secret", "private"]
    }
)
```

### Runtime Behavior

- Case-insensitive pattern matching
- Content checked before operation execution
- Operation blocked if any pattern matches

### Creation in Collibra UI

1. Create asset with **Asset Type**: "Content Filter"
2. Set **Name**: "Sensitive Content Filter"
3. Add **Attributes**:
   - `enforcement_level`: "block"
   - `enabled`: "true"
   - `blocked_patterns`: "confidential,secret,private,ssn,credit card"
4. **Save** asset

---

## Policy Type 4: Team Access Control

### Purpose

Restrict AI operations to authorized teams only.

### Collibra Configuration

**Asset Type:** `Team Access Control`

**Required Attributes:**

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `enforcement_level` | string | Action on violation | "block" |
| `enabled` | boolean | Policy active status | true |
| `allowed_teams` | string | Comma-separated teams | "ml-platform,data-science" |

### GenOps Translation

```python
PolicyConfig(
    name="team_access_policy-004",
    description="Team Access Control - Authorized teams only",
    enabled=True,
    enforcement_level=PolicyResult.BLOCKED,
    conditions={
        "allowed_teams": ["ml-platform", "data-science"]
    }
)
```

### Runtime Behavior

- Team attribute checked against allowed list
- Unauthorized teams blocked from execution
- Empty allowed_teams list allows all teams

### Creation in Collibra UI

1. Create asset with **Asset Type**: "Team Access Control"
2. Set **Name**: "Production Access Control"
3. Add **Attributes**:
   - `enforcement_level`: "block"
   - `enabled`: "true"
   - `allowed_teams`: "ml-platform,data-science,ai-research"
4. **Save** asset

---

## Policy Type 5: Budget Constraint

### Purpose

Enforce spending limits over daily or monthly periods.

### Collibra Configuration

**Asset Type:** `Budget Constraint`

**Required Attributes:**

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `enforcement_level` | string | Action on violation | "block" |
| `enabled` | boolean | Policy active status | true |

**Optional Attributes (at least one required):**

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `daily_budget` | float | Daily spend limit (USD) | 100.0 |
| `monthly_budget` | float | Monthly spend limit (USD) | 3000.0 |

### GenOps Translation

```python
PolicyConfig(
    name="budget_limit_policy-005",
    description="Budget Constraint - Daily and monthly limits",
    enabled=True,
    enforcement_level=PolicyResult.BLOCKED,
    conditions={
        "daily_budget": 100.0,
        "monthly_budget": 3000.0
    }
)
```

### Runtime Behavior

- Running total tracked per team/project
- Operations blocked when budget exhausted
- Budget resets at midnight (daily) or month boundary (monthly)
- Warning threshold at 90% of budget

### Creation in Collibra UI

1. Create asset with **Asset Type**: "Budget Constraint"
2. Set **Name**: "Team Monthly Budget"
3. Add **Attributes**:
   - `enforcement_level`: "block"
   - `enabled`: "true"
   - `daily_budget`: "100.0"
   - `monthly_budget`: "3000.0"
4. **Save** asset

---

## Policy Type 6: Model Governance

### Purpose

Control which AI models can be used in operations.

### Collibra Configuration

**Asset Type:** `Model Governance`

**Required Attributes:**

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `enforcement_level` | string | Action on violation | "block" |
| `enabled` | boolean | Policy active status | true |

**Optional Attributes (at least one required):**

| Attribute | Type | Description | Example |
|-----------|------|-------------|---------|
| `allowed_models` | string | Comma-separated allowed models | "gpt-4,claude-3" |
| `blocked_models` | string | Comma-separated blocked models | "gpt-3.5-turbo" |

### GenOps Translation

```python
PolicyConfig(
    name="model_governance_policy-006",
    description="Model Governance - Allowed models only",
    enabled=True,
    enforcement_level=PolicyResult.BLOCKED,
    conditions={
        "allowed_models": ["gpt-4", "claude-3"],
        "blocked_models": ["gpt-3.5-turbo"]
    }
)
```

### Runtime Behavior

- Model name checked against allowed/blocked lists
- Blocked models prevented from execution
- Allowed list takes precedence over blocked list
- Empty allowed_models means all models allowed (except blocked)

### Creation in Collibra UI

1. Create asset with **Asset Type**: "Model Governance"
2. Set **Name**: "Production Model Policy"
3. Add **Attributes**:
   - `enforcement_level`: "block"
   - `enabled`: "true"
   - `allowed_models`: "gpt-4,claude-3-opus,claude-3-sonnet"
   - `blocked_models`: "gpt-3.5-turbo,text-davinci-002"
4. **Save** asset

---

## Enforcement Levels

### Supported Enforcement Actions

| Enforcement Level | Collibra Value | GenOps Result | Behavior |
|-------------------|----------------|---------------|----------|
| **Block** | "block", "blocked", "enforce" | `PolicyResult.BLOCKED` | Operation prevented, exception raised |
| **Warning** | "warn", "warning", "alert" | `PolicyResult.WARNING` | Warning logged, operation continues |
| **Rate Limit** | "rate_limit", "throttle" | `PolicyResult.RATE_LIMITED` | Operation delayed/queued |
| **Allow** | "allow", "allowed" | `PolicyResult.ALLOWED` | Operation permitted |

### Enforcement Level Mapping

```python
ENFORCEMENT_MAPPING = {
    "block": PolicyResult.BLOCKED,
    "blocked": PolicyResult.BLOCKED,
    "enforce": PolicyResult.BLOCKED,
    "warn": PolicyResult.WARNING,
    "warning": PolicyResult.WARNING,
    "alert": PolicyResult.WARNING,
    "rate_limit": PolicyResult.RATE_LIMITED,
    "throttle": PolicyResult.RATE_LIMITED,
    "allow": PolicyResult.ALLOWED,
    "allowed": PolicyResult.ALLOWED,
}
```

### Example: Different Enforcement Levels

**Block Enforcement:**
```python
# Policy: enforcement_level = "block"
# Result: PolicyViolationError raised
try:
    with adapter.track_ai_operation("blocked-op") as span:
        pass  # Never executes
except PolicyViolationError as e:
    print(f"Blocked: {e}")
```

**Warning Enforcement:**
```python
# Policy: enforcement_level = "warn"
# Result: Warning logged, operation continues
with adapter.track_ai_operation("warned-op") as span:
    pass  # Executes with warning log
```

**Rate Limit Enforcement:**
```python
# Policy: enforcement_level = "rate_limit"
# Result: Operation delayed until rate limit allows
with adapter.track_ai_operation("throttled-op") as span:
    pass  # Delayed but eventually executes
```

---

## Policy Examples

### Example 1: Production Cost Control

**Scenario:** Prevent expensive operations in production

**Collibra Policy:**
- Asset Type: AI Cost Limit
- Name: "Production Cost Gate"
- enforcement_level: "block"
- max_cost: 5.0
- enabled: true

**Effect:**
- All operations >$5 blocked in production
- Developers use cheaper models or batch operations
- Cost overruns prevented at infrastructure level

### Example 2: Team-Based Rate Limiting

**Scenario:** Different rate limits per team

**Collibra Policies:**

Policy 1 (Data Science Team):
- Asset Type: AI Rate Limit
- Name: "Data Science Rate Limit"
- enforcement_level: "rate_limit"
- max_requests_per_minute: 200
- team: "data-science"

Policy 2 (Marketing Team):
- Asset Type: AI Rate Limit
- Name: "Marketing Rate Limit"
- enforcement_level: "rate_limit"
- max_requests_per_minute: 50
- team: "marketing"

### Example 3: Multi-Policy Governance

**Scenario:** Comprehensive governance with multiple policies

**Collibra Policies:**

1. **Cost Limit** (max_cost: 10.0)
2. **Rate Limit** (max_requests: 100/min)
3. **Content Filter** (blocked_patterns: "confidential,secret")
4. **Model Governance** (allowed_models: "gpt-4,claude-3")

**Effect:**
- All policies evaluated before operation
- Operation must satisfy ALL policies to proceed
- First violation blocks operation

---

## Creating Policies in Collibra

### Step-by-Step Process

#### Step 1: Navigate to Asset Creation

1. Log into Collibra
2. Navigate to **Assets** section
3. Click **Create Asset** button

#### Step 2: Select Asset Type

1. In the **Asset Type** dropdown, select one of:
   - AI Cost Limit
   - AI Rate Limit
   - Content Filter
   - Team Access Control
   - Budget Constraint
   - Model Governance

2. Click **Next** or **Continue**

#### Step 3: Set Basic Information

1. **Name**: Enter descriptive policy name
2. **Domain**: Select your AI Governance domain
3. **Description**: Explain policy purpose and scope
4. **Status**: Set to "Approved" or "Active"

#### Step 4: Configure Attributes

1. Click **Add Attribute** for each required attribute
2. Set attribute values according to policy type (see sections above)
3. Required for all policies:
   - `enforcement_level`: "block", "warn", or "rate_limit"
   - `enabled`: "true" or "false"
4. Add policy-specific attributes

#### Step 5: Set Relationships (Optional)

1. Link to related assets:
   - Domains (scope)
   - Teams (ownership)
   - Data elements (what's governed)

#### Step 6: Save and Activate

1. Review all settings
2. Click **Save** or **Create**
3. Verify policy appears in asset list
4. Policy will be imported by GenOps within 5 minutes

### Verification

After creating policy, verify it's imported:

```python
from genops.providers.collibra import GenOpsCollibraAdapter

adapter = GenOpsCollibraAdapter(enable_policy_sync=True)

# Wait for background sync (or manually sync)
result = adapter.sync_policies()
print(f"Policies imported: {result['imported']}")

# View imported policies
if adapter.policy_importer:
    policies = adapter.policy_importer.get_imported_policies()
    for name, config in policies.items():
        print(f"  - {name}: {config.description}")
```

---

## Policy Translation

### How Policy Translation Works

When GenOps imports policies from Collibra, it automatically converts them into `PolicyConfig` objects that the GenOps PolicyEngine understands.

**Important**: You **never write this PolicyConfig code manually**. GenOps generates it automatically during policy import. The examples below show you what GenOps creates internally so you understand the mapping.

#### Your Role vs. GenOps Role

| What You Do | What GenOps Does |
|-------------|------------------|
| Create policy asset in Collibra UI | Import policy asset from Collibra |
| Set policy attributes (max_cost, etc.) | Convert to PolicyConfig object |
| Enable `enable_policy_sync=True` | Register policy with PolicyEngine |
| Write your AI operation code | Check operation against policies at runtime |

**Where You Interact With Policies**:
```python
# You don't write PolicyConfig - you just handle the result:
try:
    with adapter.track_ai_operation("my-operation") as span:
        result = my_ai_function()
        adapter.record_cost(span, cost=15.0)  # Might violate policy
except PolicyViolationError as e:
    print(f"Operation blocked by policy: {e.policy_name}")
    print(f"Reason: {e.message}")
    # Handle the violation - retry with different approach, log, alert, etc.
```

Now let's see what GenOps creates internally for each policy type.

### Translation Process

1. **Fetch**: GenOps retrieves policy assets from Collibra via REST API
2. **Filter**: Only assets matching supported types are processed
3. **Translate**: Collibra attributes mapped to GenOps PolicyConfig
4. **Register**: Policies registered with GenOps PolicyEngine
5. **Enforce**: Policies applied to subsequent AI operations

### Attribute Mapping

| Collibra Attribute | GenOps PolicyConfig Field |
|-------------------|---------------------------|
| Asset Type ID | Policy type inference |
| Asset Name | Included in policy name |
| `enforcement_level` | `enforcement_level` |
| `enabled` | `enabled` |
| `description` | `description` |
| Policy-specific attributes | `conditions` dictionary |

### Custom Policy Translation

For custom policy types, provide a transformer:

```python
from genops.providers.collibra import PolicyImporter

def custom_transformer(collibra_policy):
    """Custom policy transformation logic."""
    return PolicyConfig(
        name=f"custom_{collibra_policy['id']}",
        description=collibra_policy.get('description', ''),
        enabled=collibra_policy.get('attributes', {}).get('enabled', True),
        enforcement_level=PolicyResult.WARNING,
        conditions=collibra_policy.get('attributes', {})
    )

importer = PolicyImporter(
    client=client,
    policy_transformer=custom_transformer
)
```

---

## Troubleshooting

### Issue: Policies Not Imported

**Symptoms:**
- `sync_policies()` returns 0 imported
- No policies visible in GenOps

**Diagnosis:**
```python
result = adapter.sync_policies()
print(f"Imported: {result['imported']}")
print(f"Failed: {result['failed']}")

if adapter.policy_importer:
    stats = adapter.policy_importer.get_stats()
    print(f"Errors: {stats.errors}")
```

**Solutions:**
1. Verify policies exist in Collibra
2. Check asset types match supported types
3. Ensure `enable_policy_sync=True` in adapter
4. Verify account has policy read permission
5. Check domain contains policy assets

### Issue: Policy Not Enforced

**Symptoms:**
- Policy imported but not blocking operations

**Solutions:**
1. Verify policy is enabled:
   ```python
   policies = adapter.policy_importer.get_imported_policies()
   print(policies[policy_name].enabled)
   ```

2. Check enforcement level:
   ```python
   print(policies[policy_name].enforcement_level)
   # Should be PolicyResult.BLOCKED for blocking
   ```

3. Verify operation attributes match policy conditions
4. Check policy conditions are valid

### Issue: Incorrect Policy Behavior

**Symptoms:**
- Policy behaves differently than expected

**Solutions:**
1. Review policy conditions in Collibra
2. Check attribute names match exactly
3. Verify data types (string, float, integer)
4. Test policy evaluation:
   ```python
   from genops.core.policy import check_policy

   result = check_policy("policy_name", {"cost": 5.0})
   print(f"Result: {result.result.value}")
   print(f"Reason: {result.reason}")
   ```

### Issue: Policy Sync Slow

**Symptoms:**
- Policy changes take long to reflect

**Solutions:**
1. Reduce sync interval:
   ```python
   adapter = GenOpsCollibraAdapter(
       enable_policy_sync=True,
       policy_sync_interval_minutes=1  # Sync every minute
   )
   ```

2. Manual sync after policy changes:
   ```python
   adapter.sync_policies()
   ```

---

## Additional Resources

- **Integration Guide**: [Collibra Integration](../integrations/collibra.md)
- **Quickstart**: [5-Minute Quickstart](../quickstarts/collibra-quickstart.md)
- **Examples**: [Policy Import Example](../../examples/collibra/03_policy_import.py)
- **GenOps Policy Engine**: [Policy Documentation](../core/policy.md)

---

**Last Updated:** 2025-01-12
**Version:** 1.0.0
