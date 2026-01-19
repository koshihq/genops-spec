# Collibra Integration Guide

Complete guide for integrating GenOps AI with Collibra Data Governance Center for bidirectional AI governance.

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Installation & Setup](#installation--setup)
4. [Authentication](#authentication)
5. [Quick Start](#quick-start)
6. [How It Works](#how-it-works)
7. [Export Configuration](#export-configuration)
8. [Policy Import & Enforcement](#policy-import--enforcement)
9. [Configuration Reference](#configuration-reference)
10. [Governance Attributes](#governance-attributes)
11. [Policy Types](#policy-types)
12. [Error Handling Best Practices](#error-handling-best-practices)
13. [Advanced Patterns](#advanced-patterns)
14. [Troubleshooting](#troubleshooting)
15. [API Reference](#api-reference)
16. [Performance Considerations](#performance-considerations)

---

## Overview

### What is the Collibra Integration?

The GenOps Collibra integration provides **bidirectional governance** for AI systems:

- **Export TO Collibra**: GenOps automatically exports AI operation telemetry (cost, policy, evaluation, budget) to Collibra as governance assets
- **Import FROM Collibra**: Collibra governance policies are imported and enforced at runtime on AI operations

### Value Proposition

**For Data Governance Teams:**
- Centralized AI governance in your existing Collibra instance
- Audit trail of all AI operations with cost attribution
- Policy-based control over AI resource usage
- Compliance tracking and reporting

**For AI/ML Teams:**
- Transparent cost tracking across teams and projects
- Automated budget enforcement
- Policy-guided AI operations
- Zero-code governance integration

**For FinOps Practitioners:**
- Real-time AI cost attribution
- Budget constraints at the infrastructure level
- Multi-provider cost aggregation
- Chargeback and showback capabilities

### Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│   AI Application│         │  GenOps Provider │         │    Collibra     │
│                 │         │     (Client)     │         │   Governance    │
│  - OpenAI       │────────▶│                  │────────▶│    Platform     │
│  - Anthropic    │ telemetry│  - Cost Tracking│  HTTPS  │                 │
│  - Bedrock      │         │  - Policy Check  │         │  - Assets       │
│  - Gemini       │         │  - Batch Export  │         │  - Policies     │
│                 │◀────────│  - Policy Import │◀────────│  - Metadata     │
│                 │ policies │                  │         │                 │
└─────────────────┘         └──────────────────┘         └─────────────────┘
```

### Key Features

- **Zero-Code Auto-Instrumentation**: `auto_instrument()` enables integration with one line
- **Batch Export**: Reduces API calls by 100x through intelligent batching
- **Real-Time Export**: Critical events (policy violations, high costs) exported immediately
- **Hybrid Mode**: Automatic mode selection based on event criticality
- **Policy Enforcement**: 6 policy types imported from Collibra and enforced at runtime
- **Background Sync**: Automatic policy updates from Collibra
- **Budget Constraints**: Daily/monthly budget limits with automated enforcement
- **Multi-Provider**: Works with OpenAI, Anthropic, Bedrock, Gemini, and all GenOps providers

---

## Core Concepts

Before diving into the integration, let's clarify key Collibra concepts:

### Collibra Domain

A **Domain** is a logical container in Collibra that groups related assets. Think of it as a workspace or folder. For AI governance, you'll typically have a dedicated "AI Governance" domain that contains all your AI operation data, policies, and compliance records.

### Assets and Asset Types

An **Asset** is any data object in Collibra - similar to a database record. Each asset has an **Asset Type** that defines its structure. GenOps uses these asset types:

- **AI Operation Cost**: Records cost data from AI operations
- **AI Policy Evaluation**: Records policy check results
- **AI Budget Tracking**: Tracks budget consumption

### Policies

In Collibra, **Policies** are stored as assets with specific attributes. GenOps imports these policy assets and converts them into runtime enforcement rules in the PolicyEngine.

**Key Distinction**:
- Collibra stores **policy definitions** (what the rules are)
- GenOps enforces **policy decisions** (blocking, warning, or allowing operations)

### Enforcement Levels

Every policy has an enforcement level that determines what happens when a rule is violated:

- **BLOCKED**: Operation is prevented from executing
- **WARNING**: Operation proceeds but logs a warning
- **RATE_LIMITED**: Operation is throttled/delayed
- **ALLOWED**: Operation proceeds normally

### Data Flow: Bidirectional Sync

```
Collibra Domain                    GenOps Application
     ↓                                      ↑
1. Policies (definitions)  →  2. Import & Convert  →  3. Runtime Enforcement
     ↑                                      ↓
6. View Results           ←   5. Export Results  ←  4. Operation Telemetry
```

Now that you understand these concepts, let's proceed with installation.

> **See Also**:
> - [How It Works](#how-it-works) - Detailed workflow explanation
> - [Policy Translation](../policies/collibra-policy-mapping.md) - Policy mapping details
> - [5-Minute Quickstart](../quickstarts/collibra-quickstart.md) - Quick setup guide

---

## Installation & Setup

### Prerequisites

1. **Collibra Instance**
   - Collibra Data Governance Center (version 2023.x or later)
   - Admin or user account with appropriate permissions
   - At least one domain created for AI governance assets

2. **GenOps Installation**
   ```bash
   pip install genops
   ```

3. **Python Environment**
   - Python 3.8 or higher
   - Network access to your Collibra instance

### Verify Installation

```python
from genops.providers.collibra import auto_instrument, validate_setup

# Check if Collibra provider is available
result = validate_setup(
    collibra_url="https://your-instance.collibra.com",
    username="your-username",
    password="your-password"
)

if result.valid:
    print("Collibra integration ready!")
else:
    print(f"Setup issues: {result.errors}")
```

---

## Authentication

Collibra integration supports two authentication methods:

### Method 1: Basic Authentication (Username/Password)

**Environment Variables:**
```bash
export COLLIBRA_URL="https://your-instance.collibra.com"
export COLLIBRA_USERNAME="your-username"
export COLLIBRA_PASSWORD="your-password"
```

**Direct Configuration:**
```python
from genops.providers.collibra import GenOpsCollibraAdapter

adapter = GenOpsCollibraAdapter(
    collibra_url="https://your-instance.collibra.com",
    username="your-username",
    password="your-password"
)
```

### Method 2: API Token Authentication (Recommended)

**Environment Variables:**
```bash
export COLLIBRA_URL="https://your-instance.collibra.com"
export COLLIBRA_API_TOKEN="your-api-token"
```

**Direct Configuration:**
```python
adapter = GenOpsCollibraAdapter(
    collibra_url="https://your-instance.collibra.com",
    api_token="your-api-token"
)
```

### Authentication Method Comparison

| Feature | Basic Auth (Username/Password) | API Token (Recommended) |
|---------|-------------------------------|------------------------|
| Security | Less secure (credential exposure) | More secure (scoped tokens) |
| Rotation | Requires password change | Easy token regeneration |
| Expiration | Depends on password policy | Explicit expiration dates |
| Audit Trail | User-level logging | Token-specific logging |
| Best For | Development/testing | Production environments |
| Setup Complexity | Simple | Requires token generation |

**Recommendation**: Use API tokens for production environments and basic auth only for development.

### Authentication Best Practices

1. **Use API Tokens**: More secure than username/password
2. **Rotate Credentials**: Regular credential rotation for security
3. **Environment Variables**: Never hardcode credentials in source code
4. **Least Privilege**: Use accounts with minimum required permissions
5. **Token Expiry**: Monitor and renew API tokens before expiration

### Required Permissions

Your Collibra account needs these permissions:

| Permission | Purpose |
|------------|---------|
| Asset Read | View existing assets and domains |
| Asset Create | Export new AI operation assets |
| Asset Update | Update existing asset metadata |
| Policy Read | Import governance policies |
| Domain Read | List available domains |

---

## Quick Start

### 5-Minute Integration

The fastest way to get started:

```python
from genops.providers.collibra import auto_instrument

# One-line setup
adapter = auto_instrument()

# Track AI operations (automatically exported to Collibra)
with adapter.track_ai_operation("gpt-4-completion") as span:
    # Your AI operation
    result = openai.chat.completions.create(...)

    # Record cost
    adapter.record_cost(span, cost=0.05, provider="openai", model="gpt-4")

# Flush and cleanup
adapter.flush()
adapter.shutdown()
```

See the [5-minute quickstart guide](../quickstarts/collibra-quickstart.md) for step-by-step instructions.

### Manual Instrumentation

For more control over configuration:

```python
from genops.providers.collibra import GenOpsCollibraAdapter

# Configure adapter explicitly
adapter = GenOpsCollibraAdapter(
    collibra_url="https://your-instance.collibra.com",
    username="user@company.com",
    password="password",
    team="ml-platform",
    project="ai-governance-demo",
    environment="production",
    export_mode="batch",
    batch_size=100,
    batch_interval_seconds=60,
    daily_budget_limit=100.0,
    enable_cost_tracking=True
)

# Use the adapter
with adapter.track_ai_operation("batch-processing") as span:
    for item in batch:
        process_with_ai(item)
    adapter.record_cost(span, cost=5.50, provider="anthropic")

adapter.shutdown()
```

---

## How It Works

Let's walk through what happens when you use the Collibra integration:

### Setup Phase

1. **Configuration**: You set environment variables (COLLIBRA_URL, credentials)
2. **Initialization**: `auto_instrument()` or `GenOpsCollibraAdapter()` creates the adapter
3. **Validation**: Adapter validates your Collibra connection and credentials
4. **Domain Selection**: Adapter automatically finds or uses your specified domain

### Operation Phase

1. **Track Operation**: You wrap your AI code in `track_ai_operation()`
2. **Record Data**: As your AI runs, you call `record_cost()`, `record_policy()`, etc.
3. **Buffer or Export**: Based on export mode:
   - **Batch**: Data buffered until threshold or interval
   - **Real-time**: Data exported immediately
   - **Hybrid**: Critical events go immediately, others batch
4. **Create Assets**: Adapter creates corresponding assets in your Collibra domain

### Policy Sync Phase (If Enabled)

1. **Background Sync**: Every 5 minutes, adapter fetches policy assets from Collibra
2. **Translation**: Collibra policy assets → GenOps PolicyConfig objects
3. **Registration**: Policies registered with GenOps PolicyEngine
4. **Enforcement**: PolicyEngine checks operations against active policies

### Key Timing Points

- **Policy sync**: 5-minute interval (or manual with `adapter.sync_policies()`)
- **Batch export**: Default every 60 seconds or 100 operations
- **Real-time export**: Immediate (< 1 second)

This understanding will help you make better configuration decisions.

---

## Export Configuration

### Export Modes

GenOps supports three export modes for different use cases:

#### 1. Batch Mode (Default - Recommended)

Accumulates operations and exports in batches for efficiency.

```python
adapter = GenOpsCollibraAdapter(
    export_mode="batch",
    batch_size=100,              # Export after 100 operations
    batch_interval_seconds=60    # Or every 60 seconds
)
```

**Benefits:**
- 100x fewer API calls
- Lower latency on operations
- Better throughput for high-volume applications

**Use When:**
- Normal operations (not critical events)
- High-volume AI applications
- Cost optimization is priority

#### 2. Real-Time Mode

Exports each operation immediately after completion.

```python
adapter = GenOpsCollibraAdapter(
    export_mode="realtime"
)
```

**Benefits:**
- Immediate visibility in Collibra
- No data loss risk
- Real-time dashboards

**Use When:**
- Critical operations requiring immediate tracking
- Low-volume applications
- Real-time monitoring is essential

#### 3. Hybrid Mode (Intelligent)

Automatically selects mode based on event criticality.

```python
adapter = GenOpsCollibraAdapter(
    export_mode="hybrid"
)
```

**Critical Events (Real-Time):**
- Policy violations (blocked/rate-limited)
- High-cost operations (>$10)
- Budget exceeded

**Regular Events (Batch):**
- Normal cost operations
- Standard policy evaluations
- Routine AI operations

**Use When:**
- Mixed workload with varying criticality
- Want automatic optimization
- Need both efficiency and responsiveness

### Export Behavior Comparison

| Feature | Batch | Real-Time | Hybrid |
|---------|-------|-----------|--------|
| API Calls | ~100x fewer | Most | Optimized |
| Latency | Low | Medium | Low |
| Visibility | Delayed | Immediate | Mixed |
| Best For | High volume | Critical ops | Mixed workload |
| Cost | Lowest | Highest | Medium |

### Choosing an Export Mode

Use this decision matrix to select the right export mode:

| Scenario | Recommended Mode | Why |
|----------|------------------|-----|
| High-volume production (>1000 ops/day) | **Batch** | 100x fewer API calls, lower costs, better performance |
| Critical operations requiring immediate visibility | **Real-time** | See results in Collibra within seconds |
| Mixed workload (routine + critical ops) | **Hybrid** | Best of both - automatic intelligent routing |
| Development/testing | **Real-time** | Easier debugging with immediate feedback |
| Budget-constrained environments | **Batch** | Minimizes Collibra API usage costs |
| Compliance-first (must log immediately) | **Real-time** or **Hybrid** | Meets audit requirements |

**Quick Decision Tree**:
```
Do you have >500 operations per day?
├─ YES → Do you need sub-second visibility?
│   ├─ YES → Use Hybrid mode
│   └─ NO → Use Batch mode
└─ NO → Use Real-time mode (simplest)
```

**Example Configuration**:
```python
# High-volume production
adapter = GenOpsCollibraAdapter(export_mode="batch", batch_size=100, batch_interval_seconds=60)

# Critical operations
adapter = GenOpsCollibraAdapter(export_mode="realtime")

# Mixed workload (recommended)
adapter = GenOpsCollibraAdapter(export_mode="hybrid")  # Intelligently routes based on criticality
```

### Manual Flush

Force immediate export of buffered operations:

```python
# Flush pending operations
count = adapter.flush()
print(f"Exported {count} operations to Collibra")
```

**When to Use:**
- End of batch processing
- Before application shutdown
- At periodic checkpoints
- After critical operations

---

## Policy Import & Enforcement

### Overview

The Collibra integration supports **bidirectional policy sync**:

1. Policies defined in Collibra
2. Automatically imported to GenOps
3. Enforced at runtime on AI operations
4. Results exported back to Collibra

### Enable Policy Sync

```python
adapter = GenOpsCollibraAdapter(
    enable_policy_sync=True,
    policy_sync_interval_minutes=5  # Sync every 5 minutes
)
```

### Policy Workflow

```
┌────────────────┐
│ Create Policy  │  1. Define policy in Collibra UI
│   in Collibra  │     (e.g., "AI Cost Limit: $10")
└────────┬───────┘
         │
         ▼
┌────────────────┐
│ Policy Import  │  2. GenOps imports policy
│   (Automatic)  │     Translates to GenOps format
└────────┬───────┘
         │
         ▼
┌────────────────┐
│   Runtime      │  3. Policy enforced on operations
│  Enforcement   │     Block if violates policy
└────────┬───────┘
         │
         ▼
┌────────────────┐
│ Export Result  │  4. Enforcement result to Collibra
│  to Collibra   │     (allowed/blocked/warning)
└────────────────┘
```

### Supported Policy Types

See [Policy Types](#policy-types) section for complete reference.

### Manual Policy Sync

Trigger policy sync on demand:

```python
# Manual sync
result = adapter.sync_policies()

print(f"Policies imported: {result['imported']}")
print(f"Policies updated: {result['updated']}")
print(f"Failed: {result['failed']}")
```

### Policy Statistics

View policy sync statistics:

```python
if adapter.policy_importer:
    stats = adapter.policy_importer.get_stats()

    print(f"Total imported: {stats.policies_imported}")
    print(f"Total updated: {stats.policies_updated}")
    print(f"Failed: {stats.policies_failed}")
    print(f"Last sync: {stats.last_sync_time}")
```

---

## Configuration Reference

### Complete Configuration Options

```python
GenOpsCollibraAdapter(
    # Authentication
    collibra_url: str = None,                    # Collibra instance URL
    username: str = None,                        # Basic auth username
    password: str = None,                        # Basic auth password
    api_token: str = None,                       # API token (alternative to username/password)

    # Domain Configuration
    domain_id: str = None,                       # Target Collibra domain (auto-detected if omitted)

    # Governance Attributes
    team: str = None,                            # Team name for cost attribution
    project: str = None,                         # Project name
    environment: str = "development",            # Environment (development/staging/production)

    # Export Configuration
    export_mode: str = "batch",                  # Export mode: batch, realtime, hybrid
    batch_size: int = 100,                       # Max operations before auto-flush
    batch_interval_seconds: int = 60,            # Auto-flush interval (seconds)

    # Policy Configuration
    enable_policy_sync: bool = False,            # Enable policy import from Collibra
    policy_sync_interval_minutes: int = 5,       # Policy sync frequency (minutes)

    # Cost Tracking
    enable_cost_tracking: bool = True,           # Enable automatic cost tracking
    daily_budget_limit: float = None,            # Daily budget limit (USD)
    enable_cost_alerts: bool = False,            # Enable budget alerts

    # System Configuration
    auto_validate: bool = True,                  # Validate setup on initialization
    timeout: int = 30,                           # API request timeout (seconds)
    max_retries: int = 3,                        # Max retry attempts
    rate_limit_per_second: int = 10              # API rate limit
)
```

### Environment Variable Mapping

All configuration options can be set via environment variables:

| Environment Variable | Config Parameter | Default |
|---------------------|------------------|---------|
| `COLLIBRA_URL` | `collibra_url` | None (required) |
| `COLLIBRA_USERNAME` | `username` | None |
| `COLLIBRA_PASSWORD` | `password` | None |
| `COLLIBRA_API_TOKEN` | `api_token` | None |
| `GENOPS_TEAM` | `team` | None |
| `GENOPS_PROJECT` | `project` | None |
| `GENOPS_ENVIRONMENT` | `environment` | "development" |

### Configuration Precedence

1. **Direct parameters** (highest priority)
2. **Environment variables**
3. **Default values** (lowest priority)

Example:
```python
# Environment: GENOPS_TEAM="env-team"
# Direct parameter overrides environment variable

adapter = GenOpsCollibraAdapter(
    team="direct-team"  # This takes precedence
)
# Result: team = "direct-team"
```

---

## Governance Attributes

### Standard Attributes

GenOps supports 6 standard governance attributes for cost attribution and access control:

| Attribute | Type | Purpose | Example |
|-----------|------|---------|---------|
| `team` | str | Team/department attribution | "ml-platform" |
| `project` | str | Project-level tracking | "chatbot-v2" |
| `customer_id` | str | Customer billing attribution | "enterprise-123" |
| `environment` | str | Environment segregation | "production" |
| `cost_center` | str | Financial reporting | "engineering" |
| `feature` | str | Feature-level tracking | "chat-completion" |

### Setting Governance Attributes

#### Global (Adapter-Level)

```python
adapter = GenOpsCollibraAdapter(
    team="ml-platform",
    project="ai-governance",
    environment="production"
)

# All operations inherit these attributes
```

#### Per-Operation (Override)

```python
with adapter.track_ai_operation(
    "customer-specific-task",
    customer_id="customer-456",        # Override
    feature="premium-feature"          # Additional attribute
) as span:
    # Operation tracked with these specific attributes
    pass
```

### Attribute Inheritance

```python
# Adapter configured with team="ml-platform"
adapter = GenOpsCollibraAdapter(team="ml-platform")

# Operation 1: Inherits team
with adapter.track_ai_operation("op1") as span:
    pass  # team="ml-platform"

# Operation 2: Overrides team
with adapter.track_ai_operation("op2", team="data-science") as span:
    pass  # team="data-science"
```

### Custom Attributes

Add custom attributes for domain-specific tracking:

```python
with adapter.track_ai_operation(
    "custom-operation",
    model_version="v2.1",           # Custom attribute
    data_source="production-db",    # Custom attribute
    priority="high"                 # Custom attribute
) as span:
    pass
```

### Attribute Mapping to Collibra

GenOps attributes are automatically mapped to Collibra asset attributes:

| GenOps Attribute | Collibra Asset Attribute |
|------------------|--------------------------|
| `genops.team` | `team` |
| `genops.project` | `project` |
| `genops.customer_id` | `customer_identifier` |
| `genops.environment` | `environment` |
| `genops.cost_center` | `cost_center` |
| `genops.cost.total` | `cost_amount` |
| `genops.cost.provider` | `ai_provider` |
| `genops.cost.model` | `model_name` |

---

## Policy Types

The Collibra integration supports 6 policy types for AI governance:

### Choosing the Right Policy Type

| Your Goal | Policy Type | Example Use Case |
|-----------|-------------|------------------|
| Limit cost per operation | **AI Cost Limit** | "No single LLM call over $10" |
| Control team spending | **Budget Constraint** | "Team Alpha: $1000/month limit" |
| Prevent API rate limit hits | **AI Rate Limit** | "Max 100 requests/minute" |
| Restrict team access | **Team Access Control** | "Only ML team can use GPT-4" |
| Control which models are used | **Model Governance** | "Block GPT-3.5, allow GPT-4" |
| Filter sensitive content | **Content Filter** | "Block queries with 'confidential'" |

**Common Combinations**:
```python
# Cost governance stack
policies = [
    "AI Cost Limit" (per-operation: $5),
    "Budget Constraint" (team daily: $500)
]

# Compliance stack
policies = [
    "Team Access Control" (allowed teams),
    "Model Governance" (approved models),
    "Content Filter" (blocked patterns)
]
```

---

### 1. AI Cost Limit

Enforce maximum cost per operation.

**Collibra Asset Type:** `AI Cost Limit`

**GenOps Policy:** `cost_limit`

**Configuration:**
```python
# In Collibra, create policy with:
# - enforcement_level: "block"
# - max_cost: 10.0
```

**Enforcement:**
- Operations exceeding cost limit are blocked
- Cost calculated before operation execution
- Actual cost tracked after completion

**Example:**
```python
# Policy imported from Collibra: max_cost = 10.0
# This operation would be blocked if estimated cost > $10
with adapter.track_ai_operation("expensive-operation") as span:
    # Policy check happens here
    adapter.record_cost(span, cost=5.0)  # Allowed
```

### 2. AI Rate Limit

Throttle request rate to prevent abuse.

**Collibra Asset Type:** `AI Rate Limit`

**GenOps Policy:** `rate_limit`

**Configuration:**
```python
# In Collibra, create policy with:
# - enforcement_level: "rate_limit"
# - max_requests_per_minute: 100
```

**Enforcement:**
- Requests exceeding limit are rate-limited
- Token bucket algorithm for smooth rate limiting
- Per-team or per-project limits

### 3. Content Filter

Block operations containing specific patterns.

**Collibra Asset Type:** `Content Filter`

**GenOps Policy:** `content_filter`

**Configuration:**
```python
# In Collibra, create policy with:
# - enforcement_level: "block"
# - blocked_patterns: "confidential,secret,private"
```

**Enforcement:**
- Content checked against blocked patterns
- Case-insensitive matching
- Operations blocked if match found

### 4. Team Access Control

Restrict operations to specific teams.

**Collibra Asset Type:** `Team Access Control`

**GenOps Policy:** `team_access`

**Configuration:**
```python
# In Collibra, create policy with:
# - enforcement_level: "block"
# - allowed_teams: "ml-platform,data-science"
```

**Enforcement:**
- Only specified teams can execute operations
- Team attribute checked against allowed list
- Unauthorized teams blocked

### 5. Budget Constraint

Enforce daily/monthly budget limits.

**Collibra Asset Type:** `Budget Constraint`

**GenOps Policy:** `budget_limit`

**Configuration:**
```python
# In Collibra, create policy with:
# - enforcement_level: "block"
# - daily_budget: 100.0
# - monthly_budget: 3000.0
```

**Enforcement:**
- Running budget tracked per team/project
- Operations blocked when budget exhausted
- Budget resets daily/monthly

### 6. Model Governance

Restrict allowed/blocked AI models.

**Collibra Asset Type:** `Model Governance`

**GenOps Policy:** `model_governance`

**Configuration:**
```python
# In Collibra, create policy with:
# - enforcement_level: "block"
# - allowed_models: "gpt-4,claude-3"
# - blocked_models: "gpt-3.5-turbo"
```

**Enforcement:**
- Model checked against allowed/blocked lists
- Blocked models prevented from execution
- Allowed list takes precedence

### Enforcement Levels

All policies support multiple enforcement levels:

| Enforcement Level | Behavior |
|-------------------|----------|
| `block` / `blocked` | Operation prevented, exception raised |
| `warn` / `warning` | Warning logged, operation continues |
| `rate_limit` / `throttle` | Operation delayed/queued |
| `allow` / `allowed` | Operation permitted |

### Policy Creation in Collibra

See [Policy Mapping Documentation](../policies/collibra-policy-mapping.md) for detailed instructions on creating policies in Collibra UI.

---

## Error Handling Best Practices

### Handling Policy Violations

```python
from genops.core.policy import PolicyViolationError

def run_ai_operation_with_fallback():
    """Example: Try expensive model, fall back to cheaper if policy blocks."""

    # First attempt: GPT-4 (expensive)
    try:
        with adapter.track_ai_operation("smart-completion") as span:
            result = call_gpt4()
            adapter.record_cost(span, cost=0.50, model="gpt-4")
            return result
    except PolicyViolationError as e:
        if "cost" in e.policy_name.lower():
            print(f"Cost policy blocked GPT-4: {e.message}")
            # Fall back to cheaper model
            with adapter.track_ai_operation("budget-completion") as span:
                result = call_gpt3_5_turbo()
                adapter.record_cost(span, cost=0.05, model="gpt-3.5-turbo")
                return result
        else:
            # Other policy violation - don't retry
            raise

# Use in your code
try:
    result = run_ai_operation_with_fallback()
except PolicyViolationError as e:
    # All fallbacks exhausted
    logger.error(f"Operation blocked by policy: {e.policy_name}")
    send_alert(f"AI operation requires manual approval: {e.message}")
```

### Handling Connection Errors

```python
from genops.providers.collibra.client import CollibraAPIError, CollibraAuthenticationError

try:
    adapter = auto_instrument()
except CollibraAuthenticationError:
    print("Authentication failed - check credentials")
    # Fall back to logging-only mode
    adapter = None
except CollibraAPIError as e:
    print(f"Collibra connection failed: {e.message}")
    # Continue without Collibra integration
    adapter = None

# Graceful degradation
if adapter:
    with adapter.track_ai_operation("my-op") as span:
        result = my_ai_function()
else:
    # Just run without tracking
    result = my_ai_function()
```

### Recommended Logging

```python
import logging

logger = logging.getLogger("genops.collibra")
logger.setLevel(logging.INFO)

# Log policy decisions
def log_policy_result(operation_name, cost, policy_result):
    logger.info(
        f"Operation: {operation_name}, Cost: ${cost:.2f}, "
        f"Policy: {policy_result}, Team: {adapter.team}"
    )

# Use in your code
with adapter.track_ai_operation("analysis") as span:
    result = analyze_data()
    adapter.record_cost(span, cost=2.50)
    log_policy_result("analysis", 2.50, "ALLOWED")
```

> **See Also**:
> - [Troubleshooting](#troubleshooting) - Common issues and solutions
> - [Policy Types](#policy-types) - Understanding policy enforcement
> - [Export Configuration](#export-configuration) - Export mode selection

---

## Advanced Patterns

### Multi-Provider Cost Aggregation

Track costs across multiple AI providers:

```python
adapter = GenOpsCollibraAdapter(team="ml-platform")

# OpenAI operation
with adapter.track_ai_operation("gpt-completion") as span:
    result = openai.chat.completions.create(...)
    adapter.record_cost(span, cost=0.05, provider="openai", model="gpt-4")

# Anthropic operation
with adapter.track_ai_operation("claude-completion") as span:
    result = anthropic.messages.create(...)
    adapter.record_cost(span, cost=0.03, provider="anthropic", model="claude-3")

# Bedrock operation
with adapter.track_ai_operation("bedrock-completion") as span:
    result = bedrock.invoke_model(...)
    adapter.record_cost(span, cost=0.02, provider="bedrock", model="titan")

# View aggregated metrics
metrics = adapter.get_metrics()
print(f"Total cost across all providers: ${metrics['total_cost']:.2f}")
```

### Customer Attribution

Track costs per customer for billing:

```python
def process_customer_request(customer_id: str, request: str):
    with adapter.track_ai_operation(
        "customer-request",
        customer_id=customer_id,
        request_type="chat"
    ) as span:
        # AI processing
        result = ai_provider.process(request)

        # Cost attributed to customer
        adapter.record_cost(
            span,
            cost=calculate_cost(result),
            provider="openai"
        )

        return result

# Query Collibra for customer-specific costs
# (via Collibra UI or API)
```

### Budget-Constrained Operations

Enforce budget limits at runtime:

```python
adapter = GenOpsCollibraAdapter(
    team="ml-platform",
    daily_budget_limit=100.0,
    enable_cost_alerts=True
)

try:
    with adapter.track_ai_operation("operation") as span:
        # Operation checked against budget
        result = expensive_ai_operation()
        adapter.record_cost(span, cost=50.0)
except BudgetExceededError:
    # Handle budget exceeded
    logger.warning("Daily budget exceeded, using fallback")
    result = fallback_operation()
```

### Batch Processing with Progress Tracking

Track batch operations efficiently:

```python
adapter = GenOpsCollibraAdapter(export_mode="batch", batch_size=50)

for batch_id, items in enumerate(batches):
    with adapter.track_ai_operation(
        f"batch-{batch_id}",
        batch_size=len(items),
        batch_id=batch_id
    ) as span:
        total_cost = 0

        for item in items:
            result = process_item(item)
            total_cost += item_cost(result)

        adapter.record_cost(span, cost=total_cost)

    # Periodic flush
    if batch_id % 10 == 0:
        adapter.flush()

# Final flush
adapter.shutdown()
```

### Policy Enforcement with Fallbacks

Graceful handling of policy violations:

```python
from genops.core.policy import PolicyViolationError

def ai_operation_with_fallback(prompt: str):
    try:
        with adapter.track_ai_operation("primary-model") as span:
            result = expensive_model(prompt)
            adapter.record_cost(span, cost=5.0, model="gpt-4")
            return result
    except PolicyViolationError as e:
        # Policy blocked expensive model, use cheaper alternative
        logger.warning(f"Policy blocked: {e}, using fallback")

        with adapter.track_ai_operation("fallback-model") as span:
            result = cheap_model(prompt)
            adapter.record_cost(span, cost=0.1, model="gpt-3.5")
            return result
```

### Environment-Specific Configuration

Different configurations per environment:

```python
import os

environment = os.getenv("ENVIRONMENT", "development")

config = {
    "development": {
        "export_mode": "realtime",
        "enable_policy_sync": False,
        "daily_budget_limit": 10.0
    },
    "staging": {
        "export_mode": "hybrid",
        "enable_policy_sync": True,
        "daily_budget_limit": 100.0
    },
    "production": {
        "export_mode": "batch",
        "batch_size": 500,
        "enable_policy_sync": True,
        "daily_budget_limit": 1000.0
    }
}

adapter = GenOpsCollibraAdapter(
    environment=environment,
    **config[environment]
)
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue: Authentication Failed (401)

**Symptoms:**
```
CollibraAuthenticationError: Authentication failed (401)
```

**Solutions:**
1. Verify credentials are correct
2. Check if account has Collibra access
3. Ensure password hasn't expired
4. Try API token instead of username/password
5. Verify account isn't locked

**Test Authentication:**
```bash
python -m genops.providers.collibra.validation
```

#### Issue: Connection Timeout

**Symptoms:**
```
requests.exceptions.Timeout: Connection timed out
```

**Solutions:**
1. Verify Collibra URL is accessible
2. Check network connectivity
3. Confirm firewall allows HTTPS traffic
4. Increase timeout setting:
   ```python
   adapter = GenOpsCollibraAdapter(timeout=60)
   ```

#### Issue: No Domains Found

**Symptoms:**
```
Warning: No Collibra domains found
```

**Solutions:**
1. Create at least one domain in Collibra UI
2. Or specify explicit domain_id:
   ```python
   adapter = GenOpsCollibraAdapter(domain_id="your-domain-id")
   ```
3. Verify account has permission to view domains

#### Issue: Rate Limit Exceeded (429)

**Symptoms:**
```
CollibraRateLimitError: Rate limit exceeded (429)
```

**Solutions:**
1. Switch to batch mode (if using realtime):
   ```python
   adapter = GenOpsCollibraAdapter(export_mode="batch")
   ```
2. Reduce rate limit:
   ```python
   adapter = GenOpsCollibraAdapter(rate_limit_per_second=5)
   ```
3. Contact Collibra admin to increase rate limits

#### Issue: Policy Import Fails

**Symptoms:**
```
Failed to import policies: No policies found
```

**Solutions:**
1. Verify policies exist in Collibra
2. Check policy types match supported types
3. Ensure account has policy read permission
4. Verify domain contains policies:
   ```python
   result = adapter.client.list_policies()
   print(f"Found {len(result)} policies")
   ```

#### Issue: High Memory Usage

**Symptoms:**
- Memory usage grows over time
- Out of memory errors

**Solutions:**
1. Reduce batch size:
   ```python
   adapter = GenOpsCollibraAdapter(batch_size=50)
   ```
2. Enable more frequent flushing:
   ```python
   adapter = GenOpsCollibraAdapter(batch_interval_seconds=30)
   ```
3. Call `flush()` periodically in long-running processes

#### Issue: Metrics Not Appearing in Collibra

**Symptoms:**
- Operations tracked but not visible in Collibra

**Solutions:**
1. Call `flush()` to export pending data:
   ```python
   adapter.flush()
   ```
2. Check export stats:
   ```python
   stats = adapter.get_export_summary()
   print(f"Exported: {stats['assets_created']}")
   print(f"Failed: {stats['assets_failed']}")
   ```
3. Verify domain_id is correct
4. Check network connectivity to Collibra

### Diagnostic Tools

#### Validation Utility

Run comprehensive validation:

```bash
python -m genops.providers.collibra.validation
```

#### Export Statistics

View detailed export metrics:

```python
summary = adapter.get_export_summary()
print(f"Assets created: {summary['assets_created']}")
print(f"Assets failed: {summary['assets_failed']}")
print(f"Batches sent: {summary['batches_sent']}")
print(f"Avg export time: {summary['average_export_time_ms']:.1f}ms")
```

#### Policy Sync Statistics

Check policy import status:

```python
if adapter.policy_importer:
    stats = adapter.policy_importer.get_stats()
    print(f"Policies imported: {stats.policies_imported}")
    print(f"Failed: {stats.policies_failed}")
    print(f"Errors: {stats.errors}")
```

#### Debug Logging

Enable debug logging for troubleshooting:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("genops.providers.collibra")
logger.setLevel(logging.DEBUG)
```

---

## API Reference

### GenOpsCollibraAdapter

Main adapter class for Collibra integration.

#### Constructor

```python
GenOpsCollibraAdapter(
    collibra_url: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    api_token: Optional[str] = None,
    domain_id: Optional[str] = None,
    team: Optional[str] = None,
    project: Optional[str] = None,
    environment: str = "development",
    export_mode: str = "batch",
    batch_size: int = 100,
    batch_interval_seconds: int = 60,
    enable_policy_sync: bool = False,
    policy_sync_interval_minutes: int = 5,
    enable_cost_tracking: bool = True,
    daily_budget_limit: Optional[float] = None,
    enable_cost_alerts: bool = False,
    auto_validate: bool = True
)
```

#### Methods

**track_ai_operation(operation_name, operation_type="ai.inference", **governance_attrs)**

Context manager for tracking AI operations.

```python
with adapter.track_ai_operation("operation-name", team="ml-platform") as span:
    # Your AI operation
    pass
```

**Returns:** OpenTelemetry span

**record_cost(span, cost, provider="", model="", tokens_input=None, tokens_output=None, **metadata)**

Record cost telemetry on a span.

```python
adapter.record_cost(
    span,
    cost=0.05,
    provider="openai",
    model="gpt-4",
    tokens_input=150,
    tokens_output=200
)
```

**record_policy(span, policy_name, policy_result, policy_reason=None)**

Record policy enforcement telemetry.

```python
adapter.record_policy(
    span,
    policy_name="cost_limit",
    policy_result="allowed",
    policy_reason="Within budget"
)
```

**sync_policies() → Dict[str, Any]**

Manually sync policies from Collibra.

```python
result = adapter.sync_policies()
# Returns: {"imported": 5, "updated": 2, "failed": 0}
```

**flush() → int**

Flush pending exports to Collibra.

```python
count = adapter.flush()
# Returns: number of assets exported
```

**shutdown(timeout=5.0)**

Shutdown adapter and flush remaining data.

```python
adapter.shutdown(timeout=10.0)
```

**get_metrics() → Dict[str, Any]**

Get adapter metrics.

```python
metrics = adapter.get_metrics()
# Returns: {
#     "operation_count": 10,
#     "total_cost": 5.50,
#     "daily_budget_limit": 100.0,
#     "budget_remaining": 94.50,
#     "assets_exported": 10,
#     "assets_failed": 0,
#     "buffer_size": 0
# }
```

**get_export_summary() → Dict[str, Any]**

Get export statistics.

```python
summary = adapter.get_export_summary()
# Returns: {
#     "assets_created": 10,
#     "assets_failed": 0,
#     "batches_sent": 1,
#     "total_cost": 5.50,
#     "average_export_time_ms": 150.5
# }
```

### Module Functions

**auto_instrument(collibra_url=None, team=None, project=None, environment="development", **kwargs) → GenOpsCollibraAdapter**

Zero-code auto-instrumentation.

```python
from genops.providers.collibra import auto_instrument

adapter = auto_instrument(team="ml-platform", project="ai-demo")
```

**validate_setup(collibra_url=None, username=None, password=None, api_token=None, check_connectivity=True) → CollibraValidationResult**

Validate Collibra setup.

```python
from genops.providers.collibra import validate_setup

result = validate_setup()
if not result.valid:
    print(f"Errors: {result.errors}")
```

**print_validation_result(result: CollibraValidationResult)**

Print validation results in user-friendly format.

```python
from genops.providers.collibra import validate_setup, print_validation_result

result = validate_setup()
print_validation_result(result)
```

---

## Performance Considerations

### Batch Mode Optimization

**Recommendation:** Use batch mode for high-volume applications (>100 ops/min)

```python
adapter = GenOpsCollibraAdapter(
    export_mode="batch",
    batch_size=100,
    batch_interval_seconds=60
)
```

**Performance Impact:**
- Reduces API calls by 100x
- Lower latency per operation (no blocking on export)
- Higher throughput (operations don't wait for API)

**Trade-offs:**
- Delayed visibility in Collibra (up to batch_interval_seconds)
- Risk of data loss if application crashes before flush

### Real-Time Mode Considerations

**Use Cases:**
- Low-volume applications (<10 ops/min)
- Critical operations requiring immediate tracking
- Real-time dashboards and alerts

**Performance Impact:**
- Higher latency per operation (waits for API)
- More API calls (one per operation)
- Lower throughput

### Hybrid Mode Balance

**Recommendation:** Use hybrid mode for mixed workloads

```python
adapter = GenOpsCollibraAdapter(export_mode="hybrid")
```

**Behavior:**
- Critical events: Real-time export
- Regular operations: Batch export
- Automatic optimization

### Memory Usage

**Batch Buffer Size:**
- Default batch_size: 100 operations
- Memory per operation: ~1-2 KB
- Total buffer memory: ~100-200 KB

**High-Volume Optimization:**
```python
# For >1000 ops/min, use larger batch size
adapter = GenOpsCollibraAdapter(
    batch_size=500,
    batch_interval_seconds=30
)
```

### Network Latency

**Typical Latencies:**
- Batch export: 200-500ms per batch (100 operations)
- Real-time export: 100-200ms per operation
- Policy sync: 500-1000ms (periodic)

**Optimization:**
- Use batch mode to amortize network latency
- Configure longer batch intervals for lower priority data
- Use hybrid mode to prioritize critical events

### Rate Limiting

**Default Rate Limit:** 10 requests/second

```python
adapter = GenOpsCollibraAdapter(rate_limit_per_second=10)
```

**Considerations:**
- Collibra instance may have server-side limits
- Adjust based on your Collibra plan and usage
- Monitor for 429 (Rate Limit Exceeded) errors

### Scalability Guidelines

| Operations/Minute | Recommended Config |
|-------------------|-------------------|
| <10 | Real-time mode |
| 10-100 | Hybrid mode |
| 100-1000 | Batch mode (batch_size=100) |
| 1000+ | Batch mode (batch_size=500) |

---

## Additional Resources

- **Quickstart Guide**: [5-Minute Quickstart](../quickstarts/collibra-quickstart.md)
- **Policy Reference**: [Collibra Policy Mapping](../policies/collibra-policy-mapping.md)
- **Examples**: [Collibra Examples Directory](../../examples/collibra/)
- **GenOps Documentation**: [Main Documentation](../README.md)
- **Collibra Documentation**: [Collibra Help Center](https://productresources.collibra.com/)

---

## Support and Community

- **GitHub Issues**: [Report Issues](https://github.com/anthropics/claude-code/issues)
- **Documentation**: [GenOps Documentation](../README.md)
- **Community**: Join the GenOps community discussions

---

**Last Updated:** 2025-01-12
**Version:** 1.0.0
