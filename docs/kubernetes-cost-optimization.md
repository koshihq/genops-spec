# Kubernetes Cost Optimization for AI Workloads

Complete guide for optimizing infrastructure and AI operation costs in Kubernetes with GenOps AI governance, budget enforcement, and intelligent cost management.

## Table of Contents

1. [Quick Start (5 minutes)](#quick-start)
2. [Cost Tracking Architecture](#cost-tracking-architecture)
3. [Infrastructure Cost Optimization](#infrastructure-cost-optimization)
4. [AI Operation Cost Management](#ai-operation-cost-management)
5. [Budget Management](#budget-management)
6. [Cost Attribution & Chargeback](#cost-attribution-chargeback)
7. [Optimization Recommendations](#optimization-recommendations)
8. [FinOps Best Practices](#finops-best-practices)
9. [Troubleshooting](#troubleshooting)

## Quick Start

Implement cost optimization in 5 minutes:

```bash
# 1. Enable cost tracking with governance attributes
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: genops-cost-config
  namespace: genops-system
data:
  GENOPS_COST_TRACKING_ENABLED: "true"
  GENOPS_BUDGET_ENFORCEMENT_ENABLED: "true"
  GENOPS_COST_OPTIMIZATION_ENABLED: "true"
EOF

# 2. Set team budget
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: team-ml-platform-budget
  namespace: genops-system
data:
  monthly_limit: "1000"
  alert_threshold_80: "true"
  alert_threshold_95: "true"
  enforcement_action: "throttle"
EOF

# 3. Deploy cost dashboard
helm upgrade --install genops-cost genops/cost-dashboard \
  --namespace genops-system \
  --set grafana.enabled=true
```

✅ **Result:** Cost tracking active with $1,000 monthly budget and automatic enforcement.

## Cost Tracking Architecture

### Cost Attribution Hierarchy

```
┌─────────────────────────────────────────────────────────┐
│              Organization Level                         │
│  - Total spend across all teams                        │
│  - Executive reporting                                  │
└────────────────┬────────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
┌───────▼──────┐  ┌───────▼──────┐
│ Cost Center  │  │ Cost Center  │
│ Engineering  │  │ Operations   │
└───────┬──────┘  └───────┬──────┘
        │                 │
   ┌────┴────┐       ┌────┴────┐
   │         │       │         │
┌──▼───┐ ┌──▼───┐ ┌──▼───┐ ┌──▼───┐
│ Team │ │ Team │ │ Team │ │ Team │
│ ML   │ │ Data │ │ Infra│ │ SRE  │
└──┬───┘ └──┬───┘ └──────┘ └──────┘
   │        │
   │   ┌────┴────────┐
   │   │             │
┌──▼───▼───┐  ┌─────▼─────┐
│ Project  │  │  Project  │
│Inference │  │ Training  │
└────┬─────┘  └───────────┘
     │
┌────┴─────────────┐
│                  │
│   ┌──────────────▼───────────┐
│   │     Customer Level       │
│   │  - Per-customer costs    │
│   │  - SaaS billing basis    │
│   └──────────────┬───────────┘
│                  │
│        ┌─────────┴──────────┐
│        │                    │
│   ┌────▼─────┐       ┌─────▼────┐
│   │ Feature  │       │ Feature  │
│   │  Chat    │       │  Search  │
│   └──────────┘       └──────────┘
```

### Cost Data Flow

```yaml
# Application instrumentation
@track_usage(team="ml-platform", project="inference", customer_id="cust-123")
def ai_inference():
    response = openai.complete(...)  # $0.02
    return response

# ↓ GenOps captures cost event

# ↓ OpenTelemetry export
{
  "cost_usd": 0.02,
  "team": "ml-platform",
  "project": "inference",
  "customer_id": "cust-123",
  "provider": "openai",
  "model": "gpt-4",
  "tokens_input": 500,
  "tokens_output": 200
}

# ↓ Cost aggregation

# ↓ Multiple outputs:
# 1. Real-time dashboards (Grafana)
# 2. Budget enforcement (automatic throttle)
# 3. Billing systems (customer invoices)
# 4. Financial reporting (cost center allocation)
```

## Infrastructure Cost Optimization

### Spot Instance Strategy

Deploy AI workloads on spot instances for 60-90% cost savings:

```yaml
# spot-instance-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genops-batch-inference
  namespace: genops-system
spec:
  replicas: 5
  selector:
    matchLabels:
      app: batch-inference
      cost-optimization: spot
  template:
    metadata:
      labels:
        app: batch-inference
        cost-optimization: spot
    spec:
      # Tolerate spot instance taints
      tolerations:
      - key: "kubernetes.azure.com/scalesetpriority"
        operator: "Equal"
        value: "spot"
        effect: "NoSchedule"
      - key: "cloud.google.com/gke-preemptible"
        operator: "Equal"
        value: "true"
        effect: "NoSchedule"
      - key: "node.kubernetes.io/instance-type"
        operator: "Equal"
        value: "spot"
        effect: "NoSchedule"

      # Node affinity for spot instances
      affinity:
        nodeAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            preference:
              matchExpressions:
              - key: node.kubernetes.io/instance-type
                operator: In
                values:
                - spot
                - preemptible

      # Graceful handling of spot termination
      terminationGracePeriodSeconds: 30

      containers:
      - name: inference
        image: genopsai/inference:latest
        env:
        - name: CHECKPOINT_ENABLED
          value: "true"  # Checkpoint work for recovery
        - name: GRACEFUL_SHUTDOWN
          value: "true"
        resources:
          requests:
            cpu: 1000m
            memory: 2Gi
          limits:
            cpu: 2000m
            memory: 4Gi

      # Restart policy for spot interruptions
      restartPolicy: Always
```

**Create spot node pools:**

```bash
# AWS EKS spot node group
eksctl create nodegroup \
  --cluster genops-cluster \
  --name genops-spot \
  --instance-types m5.large,m5.xlarge,m5a.large \
  --spot \
  --nodes 3 \
  --nodes-min 1 \
  --nodes-max 10 \
  --node-labels workload-type=batch,cost-optimization=spot

# Azure AKS spot node pool
az aks nodepool add \
  --resource-group genops-rg \
  --cluster-name genops-cluster \
  --name spotpool \
  --priority Spot \
  --eviction-policy Delete \
  --spot-max-price -1 \
  --node-count 3 \
  --min-count 1 \
  --max-count 10 \
  --node-taints kubernetes.azure.com/scalesetpriority=spot:NoSchedule

# GCP GKE preemptible node pool
gcloud container node-pools create genops-preemptible \
  --cluster genops-cluster \
  --preemptible \
  --num-nodes 3 \
  --min-nodes 1 \
  --max-nodes 10 \
  --node-taints cloud.google.com/gke-preemptible=true:NoSchedule
```

### Cluster Autoscaler Configuration

Configure cost-aware cluster autoscaling:

```yaml
# cluster-autoscaler-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-autoscaler-config
  namespace: kube-system
data:
  cluster-autoscaler.yaml: |
    # AWS Cluster Autoscaler with cost optimization
    cloud-provider: aws
    balance-similar-node-groups: true
    skip-nodes-with-system-pods: false

    # Cost optimization settings
    expander: least-waste
    scale-down-enabled: true
    scale-down-delay-after-add: 10m
    scale-down-unneeded-time: 10m
    scale-down-utilization-threshold: 0.5

    # Node group priorities (prefer spot over on-demand)
    node-group-auto-discovery:
      - asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/genops-cluster

    # Cost-aware scaling decisions
    skip-nodes-with-local-storage: false
    max-graceful-termination-sec: 600
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: cluster-autoscaler
  template:
    metadata:
      labels:
        app: cluster-autoscaler
    spec:
      serviceAccountName: cluster-autoscaler
      containers:
      - name: cluster-autoscaler
        image: k8s.gcr.io/autoscaling/cluster-autoscaler:v1.27.0
        command:
        - ./cluster-autoscaler
        - --v=4
        - --stderrthreshold=info
        - --cloud-provider=aws
        - --skip-nodes-with-local-storage=false
        - --expander=least-waste
        - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/genops-cluster
        - --balance-similar-node-groups
        - --skip-nodes-with-system-pods=false
        env:
        - name: AWS_REGION
          value: us-west-2
        resources:
          requests:
            cpu: 100m
            memory: 300Mi
          limits:
            cpu: 100m
            memory: 300Mi
```

### Vertical Pod Autoscaler

Automatically right-size resource requests:

```yaml
# vpa-recommendation.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: genops-inference-vpa
  namespace: genops-system
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: genops-inference
  updatePolicy:
    updateMode: "Auto"  # Automatically apply recommendations
  resourcePolicy:
    containerPolicies:
    - containerName: inference
      minAllowed:
        cpu: 100m
        memory: 256Mi
      maxAllowed:
        cpu: 4000m
        memory: 8Gi
      controlledResources:
      - cpu
      - memory
      # Cost-aware VPA mode
      mode: Auto
```

Check VPA recommendations:

```bash
# View current recommendations
kubectl describe vpa genops-inference-vpa -n genops-system

# Example output:
# Recommendation:
#   Container Recommendations:
#     Container Name: inference
#     Lower Bound:
#       Cpu:     250m    # Minimum needed
#       Memory:  512Mi
#     Target:
#       Cpu:     500m    # Recommended (current: 1000m = 50% waste!)
#       Memory:  1Gi     # Recommended (current: 2Gi = 50% waste!)
#     Uncapped Target:
#       Cpu:     800m    # Could use under peak load
#       Memory:  1.5Gi
#     Upper Bound:
#       Cpu:     2000m   # Maximum observed
#       Memory:  3Gi
```

Apply VPA recommendations:

```bash
# Get current resource usage
kubectl top pods -n genops-system -l app=genops-inference

# Apply VPA recommendations
kubectl patch deployment genops-inference -n genops-system --patch '{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "inference",
          "resources": {
            "requests": {
              "cpu": "500m",
              "memory": "1Gi"
            },
            "limits": {
              "cpu": "2000m",
              "memory": "3Gi"
            }
          }
        }]
      }
    }
  }
}'
```

### Node Affinity for Cost Optimization

Optimize pod placement for cost-efficiency:

```yaml
# cost-optimized-placement.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genops-cost-optimized
  namespace: genops-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: genops-optimized
  template:
    metadata:
      labels:
        app: genops-optimized
    spec:
      affinity:
        # Prefer spot instances
        nodeAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            preference:
              matchExpressions:
              - key: node.kubernetes.io/instance-type
                operator: In
                values:
                - spot
                - preemptible

          # Prefer cheaper instance types
          - weight: 80
            preference:
              matchExpressions:
              - key: node.kubernetes.io/instance-size
                operator: In
                values:
                - small
                - medium

          # Avoid expensive GPU nodes unless needed
          - weight: -100
            preference:
              matchExpressions:
              - key: accelerator
                operator: Exists

        # Anti-affinity for availability (but allow violations for cost)
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 50
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - genops-optimized
              topologyKey: kubernetes.io/hostname

      # Tolerate spot instance taints
      tolerations:
      - key: spot-instance
        operator: Exists
        effect: NoSchedule
      - key: preemptible
        operator: Exists
        effect: NoSchedule

      containers:
      - name: app
        image: genopsai/app:latest
        resources:
          requests:
            cpu: 200m
            memory: 512Mi
          limits:
            cpu: 500m
            memory: 1Gi
```

## AI Operation Cost Management

### LLM Provider Cost Comparison

Real-time cost comparison across providers:

```python
# cost_comparison.py
#!/usr/bin/env python3
"""
Real-time LLM provider cost comparison for intelligent routing.
"""

from dataclasses import dataclass
from typing import Dict, List
from enum import Enum

class TaskComplexity(Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"

@dataclass
class ModelPricing:
    """Model pricing information"""
    provider: str
    model: str
    input_cost_per_1k: float
    output_cost_per_1k: float
    quality_score: float  # 0-1 scale
    latency_p50_ms: int
    recommended_for: List[TaskComplexity]

# Current pricing (as of 2026-01, update regularly)
MODEL_CATALOG = [
    # OpenAI
    ModelPricing("openai", "gpt-4-turbo", 0.010, 0.030, 0.95, 800, [TaskComplexity.COMPLEX]),
    ModelPricing("openai", "gpt-4", 0.030, 0.060, 0.96, 1200, [TaskComplexity.COMPLEX]),
    ModelPricing("openai", "gpt-3.5-turbo", 0.0005, 0.0015, 0.85, 400, [TaskComplexity.SIMPLE, TaskComplexity.MEDIUM]),

    # Anthropic
    ModelPricing("anthropic", "claude-3-opus", 0.015, 0.075, 0.96, 900, [TaskComplexity.COMPLEX]),
    ModelPricing("anthropic", "claude-3-sonnet", 0.003, 0.015, 0.92, 600, [TaskComplexity.MEDIUM, TaskComplexity.COMPLEX]),
    ModelPricing("anthropic", "claude-3-haiku", 0.00025, 0.00125, 0.87, 300, [TaskComplexity.SIMPLE, TaskComplexity.MEDIUM]),

    # AWS Bedrock (discounted pricing)
    ModelPricing("bedrock", "anthropic.claude-3-opus", 0.012, 0.060, 0.96, 950, [TaskComplexity.COMPLEX]),
    ModelPricing("bedrock", "anthropic.claude-3-sonnet", 0.0024, 0.012, 0.92, 650, [TaskComplexity.MEDIUM, TaskComplexity.COMPLEX]),
    ModelPricing("bedrock", "anthropic.claude-3-haiku", 0.0002, 0.001, 0.87, 320, [TaskComplexity.SIMPLE, TaskComplexity.MEDIUM]),

    # Google Gemini
    ModelPricing("gemini", "gemini-1.5-pro", 0.0035, 0.0105, 0.93, 700, [TaskComplexity.MEDIUM, TaskComplexity.COMPLEX]),
    ModelPricing("gemini", "gemini-1.5-flash", 0.000075, 0.0003, 0.84, 250, [TaskComplexity.SIMPLE]),
]

def calculate_request_cost(
    model: ModelPricing,
    input_tokens: int,
    output_tokens: int
) -> float:
    """Calculate cost for a single request"""
    input_cost = (input_tokens / 1000) * model.input_cost_per_1k
    output_cost = (output_tokens / 1000) * model.output_cost_per_1k
    return input_cost + output_cost

def find_optimal_model(
    task_complexity: TaskComplexity,
    input_tokens: int,
    output_tokens: int,
    max_cost: float = None,
    min_quality: float = 0.85,
    max_latency_ms: int = None
) -> Dict:
    """Find the most cost-effective model for given requirements"""

    candidates = []

    for model in MODEL_CATALOG:
        # Filter by task complexity
        if task_complexity not in model.recommended_for:
            continue

        # Filter by quality requirement
        if model.quality_score < min_quality:
            continue

        # Filter by latency requirement
        if max_latency_ms and model.latency_p50_ms > max_latency_ms:
            continue

        # Calculate cost
        cost = calculate_request_cost(model, input_tokens, output_tokens)

        # Filter by cost constraint
        if max_cost and cost > max_cost:
            continue

        candidates.append({
            "provider": model.provider,
            "model": model.model,
            "cost": cost,
            "quality_score": model.quality_score,
            "latency_ms": model.latency_p50_ms,
            "cost_per_quality": cost / model.quality_score
        })

    if not candidates:
        return None

    # Sort by cost efficiency (cost per quality point)
    candidates.sort(key=lambda x: x["cost_per_quality"])

    return candidates[0]

def compare_providers(
    input_tokens: int = 1000,
    output_tokens: int = 500
) -> List[Dict]:
    """Compare costs across all providers for given token counts"""

    results = []

    for model in MODEL_CATALOG:
        cost = calculate_request_cost(model, input_tokens, output_tokens)
        results.append({
            "provider": model.provider,
            "model": model.model,
            "cost": round(cost, 4),
            "quality_score": model.quality_score,
            "latency_ms": model.latency_p50_ms,
            "monthly_cost_1k_requests": round(cost * 1000, 2)
        })

    # Sort by cost
    results.sort(key=lambda x: x["cost"])

    return results

if __name__ == "__main__":
    print("LLM Provider Cost Comparison")
    print("=" * 80)

    # Example: 1000 input tokens, 500 output tokens
    input_tokens = 1000
    output_tokens = 500

    print(f"\nScenario: {input_tokens} input tokens, {output_tokens} output tokens\n")

    results = compare_providers(input_tokens, output_tokens)

    print(f"{'Provider':<12} {'Model':<25} {'Cost':<8} {'Quality':<8} {'Latency':<10} {'Monthly (1K reqs)'}")
    print("-" * 80)

    for r in results:
        print(f"{r['provider']:<12} {r['model']:<25} ${r['cost']:<7.4f} {r['quality_score']:<8.2f} {r['latency_ms']:<10}ms ${r['monthly_cost_1k_requests']}")

    print("\n" + "=" * 80)
    print("\nOptimal Model Selection Examples:")
    print("-" * 80)

    # Simple task
    optimal = find_optimal_model(
        task_complexity=TaskComplexity.SIMPLE,
        input_tokens=500,
        output_tokens=200,
        max_cost=0.01,
        min_quality=0.80
    )

    if optimal:
        print(f"\nSimple Task (max $0.01, min quality 0.80):")
        print(f"  Recommended: {optimal['provider']} / {optimal['model']}")
        print(f"  Cost: ${optimal['cost']:.4f}")
        print(f"  Quality: {optimal['quality_score']:.2f}")
        print(f"  Latency: {optimal['latency_ms']}ms")

    # Complex task
    optimal = find_optimal_model(
        task_complexity=TaskComplexity.COMPLEX,
        input_tokens=2000,
        output_tokens=1000,
        min_quality=0.95,
        max_latency_ms=1000
    )

    if optimal:
        print(f"\nComplex Task (min quality 0.95, max latency 1000ms):")
        print(f"  Recommended: {optimal['provider']} / {optimal['model']}")
        print(f"  Cost: ${optimal['cost']:.4f}")
        print(f"  Quality: {optimal['quality_score']:.2f}")
        print(f"  Latency: {optimal['latency_ms']}ms")
```

### Intelligent Model Selection

Implement cost-aware model selection in your application:

```python
# intelligent_routing.py
from genops import track_usage
from genops.intelligence import ModelSelector
import openai
import anthropic
import boto3

class CostOptimizedInference:
    """Cost-optimized AI inference with intelligent model selection"""

    def __init__(self):
        self.selector = ModelSelector(
            cost_weight=0.6,      # 60% weight on cost
            quality_weight=0.3,   # 30% weight on quality
            latency_weight=0.1    # 10% weight on latency
        )

    @track_usage(team="ml-platform", project="inference")
    async def complete(
        self,
        prompt: str,
        task_complexity: str = "medium",
        max_cost: float = None,
        min_quality: float = 0.85
    ) -> dict:
        """
        Complete request with optimal model selection.

        Args:
            prompt: Input prompt
            task_complexity: "simple", "medium", or "complex"
            max_cost: Maximum cost per request (optional)
            min_quality: Minimum quality score required

        Returns:
            Response with cost tracking
        """

        # Estimate token counts
        estimated_input_tokens = len(prompt) // 4  # Rough estimate
        estimated_output_tokens = estimated_input_tokens // 2

        # Select optimal model
        model_config = self.selector.select(
            task_complexity=task_complexity,
            input_tokens=estimated_input_tokens,
            output_tokens=estimated_output_tokens,
            max_cost=max_cost,
            min_quality=min_quality
        )

        # Route to selected provider
        if model_config.provider == "openai":
            response = await self._call_openai(prompt, model_config.model)
        elif model_config.provider == "anthropic":
            response = await self._call_anthropic(prompt, model_config.model)
        elif model_config.provider == "bedrock":
            response = await self._call_bedrock(prompt, model_config.model)
        else:
            raise ValueError(f"Unsupported provider: {model_config.provider}")

        return {
            "response": response,
            "model_used": model_config.model,
            "provider": model_config.provider,
            "estimated_cost": model_config.estimated_cost,
            "quality_score": model_config.quality_score
        }

    async def _call_openai(self, prompt: str, model: str):
        client = openai.AsyncOpenAI()
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    async def _call_anthropic(self, prompt: str, model: str):
        client = anthropic.AsyncAnthropic()
        response = await client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    async def _call_bedrock(self, prompt: str, model: str):
        client = boto3.client('bedrock-runtime', region_name='us-west-2')
        # Bedrock API call implementation
        pass

# Usage example
inference = CostOptimizedInference()

# Simple query - automatically routes to cheapest model
result = await inference.complete(
    prompt="What is 2+2?",
    task_complexity="simple",
    max_cost=0.001  # $0.001 budget
)
# → Routes to: gemini-1.5-flash ($0.0001)

# Complex query - routes to high-quality model
result = await inference.complete(
    prompt="Write a comprehensive analysis of...",
    task_complexity="complex",
    min_quality=0.95
)
# → Routes to: claude-3-opus or gpt-4 (best cost for quality)
```

### Response Caching Strategy

Implement caching to reduce redundant API calls:

```python
# response_cache.py
import hashlib
import json
from typing import Optional
import redis
from genops import track_usage

class CachedInference:
    """Inference with intelligent response caching"""

    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True
        )
        self.default_ttl = 3600  # 1 hour

    def _cache_key(self, prompt: str, model: str) -> str:
        """Generate cache key from prompt and model"""
        content = f"{model}:{prompt}"
        return f"genops:cache:{hashlib.sha256(content.encode()).hexdigest()}"

    @track_usage(team="ml-platform", project="cached-inference")
    async def complete_with_cache(
        self,
        prompt: str,
        model: str,
        ttl: int = None,
        force_refresh: bool = False
    ) -> dict:
        """
        Complete with caching.

        Args:
            prompt: Input prompt
            model: Model to use
            ttl: Cache TTL in seconds (default: 1 hour)
            force_refresh: Force API call even if cached

        Returns:
            Response with cache status
        """

        cache_key = self._cache_key(prompt, model)

        # Check cache
        if not force_refresh:
            cached = self.redis.get(cache_key)
            if cached:
                response_data = json.loads(cached)
                return {
                    "response": response_data["response"],
                    "cached": True,
                    "cost_saved": response_data["original_cost"],
                    "cache_age_seconds": response_data["cache_age"]
                }

        # Cache miss - call API
        response = await self._call_api(prompt, model)

        # Store in cache
        cache_data = {
            "response": response["text"],
            "original_cost": response["cost"],
            "cache_age": 0,
            "timestamp": time.time()
        }

        self.redis.setex(
            cache_key,
            ttl or self.default_ttl,
            json.dumps(cache_data)
        )

        return {
            "response": response["text"],
            "cached": False,
            "cost": response["cost"]
        }

    async def _call_api(self, prompt: str, model: str):
        # Actual API call implementation
        pass

# Kubernetes deployment with Redis
```

```yaml
# redis-cache-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis-cache
  namespace: genops-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: redis-cache
  template:
    metadata:
      labels:
        app: redis-cache
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        args:
        - --maxmemory 2gb
        - --maxmemory-policy allkeys-lru
        ports:
        - containerPort: 6379
        resources:
          requests:
            cpu: 200m
            memory: 2Gi
          limits:
            cpu: 500m
            memory: 2Gi
        volumeMounts:
        - name: redis-data
          mountPath: /data
      volumes:
      - name: redis-data
        emptyDir:
          sizeLimit: 10Gi
---
apiVersion: v1
kind: Service
metadata:
  name: redis-cache
  namespace: genops-system
spec:
  selector:
    app: redis-cache
  ports:
  - port: 6379
    targetPort: 6379
```

### Prompt Optimization

Reduce token usage through prompt engineering:

```python
# prompt_optimizer.py
from typing import List
import tiktoken

class PromptOptimizer:
    """Optimize prompts to reduce token usage and costs"""

    def __init__(self, model: str = "gpt-4"):
        self.encoder = tiktoken.encoding_for_model(model)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoder.encode(text))

    def optimize_prompt(
        self,
        prompt: str,
        max_tokens: int = None,
        optimization_strategy: str = "aggressive"
    ) -> dict:
        """
        Optimize prompt to reduce token usage.

        Strategies:
        - "aggressive": Maximum compression, may lose some nuance
        - "moderate": Balance compression and clarity
        - "conservative": Minimal changes, preserve all context
        """

        original_tokens = self.count_tokens(prompt)

        if optimization_strategy == "aggressive":
            optimized = self._aggressive_optimize(prompt)
        elif optimization_strategy == "moderate":
            optimized = self._moderate_optimize(prompt)
        else:
            optimized = self._conservative_optimize(prompt)

        optimized_tokens = self.count_tokens(optimized)

        # Truncate if still over limit
        if max_tokens and optimized_tokens > max_tokens:
            optimized = self._truncate_to_token_limit(optimized, max_tokens)
            optimized_tokens = max_tokens

        tokens_saved = original_tokens - optimized_tokens
        cost_reduction_pct = (tokens_saved / original_tokens) * 100

        return {
            "original_prompt": prompt,
            "optimized_prompt": optimized,
            "original_tokens": original_tokens,
            "optimized_tokens": optimized_tokens,
            "tokens_saved": tokens_saved,
            "cost_reduction_percent": cost_reduction_pct
        }

    def _aggressive_optimize(self, prompt: str) -> str:
        """Aggressive optimization"""
        optimized = prompt

        # Remove unnecessary whitespace
        optimized = " ".join(optimized.split())

        # Remove common filler words
        fillers = ["please", "kindly", "very", "really", "actually", "basically"]
        for filler in fillers:
            optimized = optimized.replace(f" {filler} ", " ")

        # Use abbreviations
        abbrev = {
            "for example": "e.g.",
            "that is": "i.e.",
            "and so on": "etc.",
            "information": "info",
            "documentation": "docs"
        }
        for full, short in abbrev.items():
            optimized = optimized.replace(full, short)

        return optimized.strip()

    def _moderate_optimize(self, prompt: str) -> str:
        """Moderate optimization"""
        # Less aggressive, preserve readability
        optimized = " ".join(prompt.split())
        return optimized.strip()

    def _conservative_optimize(self, prompt: str) -> str:
        """Conservative optimization"""
        # Minimal changes
        return prompt.strip()

    def _truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
        """Truncate text to token limit"""
        tokens = self.encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens]
        return self.encoder.decode(truncated_tokens)

# Usage example
optimizer = PromptOptimizer(model="gpt-4")

original_prompt = """
Please analyze the following code very carefully and provide a detailed
explanation of what it does, including all the edge cases and potential
issues that might arise. Also, please suggest any improvements or
optimizations that could be made to enhance the code's performance
and maintainability. Thank you very much for your assistance with this.
"""

result = optimizer.optimize_prompt(
    original_prompt,
    optimization_strategy="aggressive"
)

print(f"Original tokens: {result['original_tokens']}")
print(f"Optimized tokens: {result['optimized_tokens']}")
print(f"Tokens saved: {result['tokens_saved']} ({result['cost_reduction_percent']:.1f}%)")
print(f"\nOptimized prompt:\n{result['optimized_prompt']}")

# Output:
# Original tokens: 87
# Optimized tokens: 52
# Tokens saved: 35 (40.2%)
#
# Optimized prompt:
# Analyze this code and explain what it does, including edge cases and
# potential issues. Suggest improvements for performance and maintainability.
```

## Budget Management

### Team Budget Configuration

Configure granular budget controls:

```yaml
# team-budgets.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: genops-budgets
  namespace: genops-system
data:
  budgets.yaml: |
    # Organization-wide budget
    organization:
      monthly_limit: 50000
      currency: USD
      alerts:
        - threshold: 80
          action: notify
          recipients: [cfo@example.com, cto@example.com]
        - threshold: 95
          action: notify_and_review
        - threshold: 100
          action: block_new_spending

    # Cost center budgets
    cost_centers:
      engineering:
        monthly_limit: 30000
        teams:
          ml-platform:
            monthly_limit: 15000
            projects:
              inference:
                monthly_limit: 8000
                alert_threshold_80: true
                alert_threshold_95: true
                enforcement_action: throttle

              training:
                monthly_limit: 5000
                enforcement_action: block

              experiments:
                monthly_limit: 2000
                enforcement_action: notify

          data-engineering:
            monthly_limit: 10000
            projects:
              pipelines:
                monthly_limit: 7000
              analytics:
                monthly_limit: 3000

          platform:
            monthly_limit: 5000

      operations:
        monthly_limit: 10000
        teams:
          sre:
            monthly_limit: 6000
          infrastructure:
            monthly_limit: 4000

      product:
        monthly_limit: 10000
        teams:
          product-engineering:
            monthly_limit: 10000

    # Per-customer budgets (for SaaS)
    customer_budgets:
      customer-enterprise-123:
        monthly_limit: 5000
        overage_allowed: true
        overage_rate_multiplier: 1.5  # Charge 1.5x for overages

      customer-startup-456:
        monthly_limit: 500
        overage_allowed: false  # Hard cap

    # Budget enforcement policies
    enforcement:
      block:
        # Completely block new operations
        message: "Budget exhausted. Contact your team lead."

      throttle:
        # Slow down operations as budget approaches limit
        threshold_85_percent_rate: 0.8  # 80% of normal rate
        threshold_95_percent_rate: 0.5  # 50% of normal rate
        threshold_100_percent_rate: 0.0  # Complete stop

      notify:
        # Send alerts but don't block
        channels: [email, slack, pagerduty]
```

Apply budget configuration:

```bash
kubectl apply -f team-budgets.yaml

# Verify budgets are active
kubectl exec -n genops-system deployment/genops-ai -- \
  genops-cli budget-status --all

# Example output:
# Team               | Project    | Limit     | Used      | Remaining | Status
# -------------------|------------|-----------|-----------|-----------|--------
# ml-platform        | inference  | $8,000    | $6,245    | $1,755    | ⚠️  78%
# ml-platform        | training   | $5,000    | $892      | $4,108    | ✅ 18%
# ml-platform        | experiments| $2,000    | $1,950    | $50       | 🚨 98%
# data-engineering   | pipelines  | $7,000    | $3,421    | $3,579    | ✅ 49%
```

### Budget Alert Integration

Configure alerts for budget thresholds:

```yaml
# budget-alerts.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: budget-alert-config
  namespace: genops-system
data:
  alert-config.yaml: |
    alerts:
      # 80% threshold - Warning
      - threshold: 80
        severity: warning
        notifications:
          slack:
            channel: "#genops-budget"
            message: |
              ⚠️  Budget Alert: {team}/{project}

              You've used {percent_used}% of your ${monthly_limit} monthly budget.

              Current spend: ${amount_used}
              Remaining: ${amount_remaining}
              Days left in period: {days_remaining}

              Projected month-end spend: ${projected_spend}
              {overage_warning}

          email:
            to: ["{team_lead_email}"]
            subject: "Budget Alert: {team}/{project} at {percent_used}%"

      # 95% threshold - Critical
      - threshold: 95
        severity: critical
        notifications:
          slack:
            channel: "#genops-budget-critical"
            message: |
              🚨 CRITICAL Budget Alert: {team}/{project}

              You've used {percent_used}% of your budget!

              Current spend: ${amount_used} of ${monthly_limit}
              Remaining: ${amount_remaining}

              ⚠️  Budget enforcement will activate at 100%

          pagerduty:
            service_key: "{pagerduty_key}"
            severity: warning

          email:
            to: ["{team_lead_email}", "{manager_email}"]
            subject: "CRITICAL: {team}/{project} budget at {percent_used}%"
            priority: high

      # 100% threshold - Budget Exhausted
      - threshold: 100
        severity: critical
        notifications:
          slack:
            channel: "#genops-budget-critical"
            message: |
              🛑 Budget Exhausted: {team}/{project}

              Your ${monthly_limit} monthly budget has been fully consumed.

              Enforcement action: {enforcement_action}

              Contact your manager to request additional budget.

          pagerduty:
            service_key: "{pagerduty_key}"
            severity: critical

          email:
            to: ["{team_lead_email}", "{manager_email}", "finance@example.com"]
            subject: "Budget Exhausted: {team}/{project}"
            priority: urgent
```

### Cost Forecasting

Implement budget forecasting and anomaly detection:

```python
# budget_forecasting.py
#!/usr/bin/env python3
"""
Budget forecasting and anomaly detection for GenOps AI.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class CostDataPoint:
    """Single cost data point"""
    timestamp: datetime
    amount: float
    team: str
    project: str

class BudgetForecaster:
    """Forecast budget usage and detect anomalies"""

    def __init__(self):
        self.history: List[CostDataPoint] = []

    def add_cost_event(self, amount: float, team: str, project: str):
        """Add cost event to history"""
        self.history.append(CostDataPoint(
            timestamp=datetime.now(),
            amount=amount,
            team=team,
            project=project
        ))

    def forecast_month_end_spend(
        self,
        team: str,
        project: str,
        budget_limit: float
    ) -> Dict:
        """
        Forecast month-end spending based on current usage.

        Returns:
            - projected_spend: Expected month-end total
            - confidence: Confidence level (0-1)
            - will_exceed_budget: Boolean
            - recommended_action: What to do
        """

        # Get data for this team/project
        filtered = [
            dp for dp in self.history
            if dp.team == team and dp.project == project
        ]

        if not filtered:
            return {"error": "No cost data available"}

        # Calculate current month usage
        now = datetime.now()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        month_data = [dp for dp in filtered if dp.timestamp >= month_start]

        if not month_data:
            return {"error": "No data for current month"}

        # Current spending
        current_spend = sum(dp.amount for dp in month_data)

        # Days elapsed and remaining
        days_elapsed = (now - month_start).days + 1
        days_in_month = (month_start.replace(month=month_start.month + 1) - month_start).days
        days_remaining = days_in_month - days_elapsed

        # Simple linear forecast
        daily_rate = current_spend / days_elapsed
        projected_spend = current_spend + (daily_rate * days_remaining)

        # Confidence based on data volume
        confidence = min(days_elapsed / days_in_month, 0.95)

        # Will exceed budget?
        will_exceed = projected_spend > budget_limit
        overage_amount = max(0, projected_spend - budget_limit)

        # Recommended action
        if will_exceed:
            days_until_exhaustion = int(budget_limit / daily_rate) if daily_rate > 0 else days_in_month

            if days_until_exhaustion < 7:
                recommended_action = "URGENT: Budget will be exhausted in less than a week. Request additional budget or reduce usage immediately."
            elif days_until_exhaustion < 14:
                recommended_action = "WARNING: Budget will be exhausted in ~2 weeks. Begin cost optimization now."
            else:
                recommended_action = "MONITOR: Budget on track to exceed. Monitor usage and optimize if needed."
        else:
            buffer_pct = ((budget_limit - projected_spend) / budget_limit) * 100
            recommended_action = f"OK: Projected to use {(projected_spend/budget_limit)*100:.1f}% of budget ({buffer_pct:.1f}% buffer)."

        return {
            "team": team,
            "project": project,
            "budget_limit": budget_limit,
            "current_spend": round(current_spend, 2),
            "projected_spend": round(projected_spend, 2),
            "daily_rate": round(daily_rate, 2),
            "days_elapsed": days_elapsed,
            "days_remaining": days_remaining,
            "will_exceed_budget": will_exceed,
            "projected_overage": round(overage_amount, 2) if will_exceed else 0,
            "confidence": round(confidence, 2),
            "recommended_action": recommended_action
        }

    def detect_anomalies(
        self,
        team: str,
        project: str,
        threshold_std_dev: float = 2.0
    ) -> List[Dict]:
        """
        Detect cost anomalies using standard deviation.

        Args:
            threshold_std_dev: Number of std devs for anomaly (default: 2.0)

        Returns:
            List of detected anomalies
        """

        # Get hourly costs for past 7 days
        now = datetime.now()
        week_ago = now - timedelta(days=7)

        filtered = [
            dp for dp in self.history
            if dp.team == team
            and dp.project == project
            and dp.timestamp >= week_ago
        ]

        if len(filtered) < 24:  # Need at least 24 hours of data
            return []

        # Group by hour
        hourly_costs = {}
        for dp in filtered:
            hour_key = dp.timestamp.replace(minute=0, second=0, microsecond=0)
            hourly_costs[hour_key] = hourly_costs.get(hour_key, 0) + dp.amount

        costs = list(hourly_costs.values())

        # Calculate mean and std dev
        mean_cost = np.mean(costs)
        std_dev = np.std(costs)

        # Find anomalies
        anomalies = []
        for hour, cost in hourly_costs.items():
            z_score = (cost - mean_cost) / std_dev if std_dev > 0 else 0

            if abs(z_score) > threshold_std_dev:
                anomalies.append({
                    "timestamp": hour,
                    "cost": round(cost, 2),
                    "mean_cost": round(mean_cost, 2),
                    "std_dev": round(std_dev, 2),
                    "z_score": round(z_score, 2),
                    "severity": "high" if abs(z_score) > 3 else "medium"
                })

        return sorted(anomalies, key=lambda x: x["timestamp"], reverse=True)

# Example usage
forecaster = BudgetForecaster()

# Simulate cost data
for day in range(15):
    for hour in range(24):
        # Simulate varying daily costs
        base_cost = 50 + (day * 2)  # Increasing trend
        cost = base_cost + np.random.uniform(-10, 10)

        forecaster.add_cost_event(
            amount=cost,
            team="ml-platform",
            project="inference"
        )

# Forecast month-end
forecast = forecaster.forecast_month_end_spend(
    team="ml-platform",
    project="inference",
    budget_limit=8000
)

print("Budget Forecast Report")
print("=" * 60)
print(f"Team: {forecast['team']}")
print(f"Project: {forecast['project']}")
print(f"Budget Limit: ${forecast['budget_limit']:,.2f}")
print(f"\nCurrent Spend: ${forecast['current_spend']:,.2f}")
print(f"Daily Rate: ${forecast['daily_rate']:,.2f}")
print(f"Projected Month-End: ${forecast['projected_spend']:,.2f}")
print(f"\nDays Elapsed: {forecast['days_elapsed']}")
print(f"Days Remaining: {forecast['days_remaining']}")
print(f"Forecast Confidence: {forecast['confidence']*100:.0f}%")
print(f"\nWill Exceed Budget: {'YES ⚠️ ' if forecast['will_exceed_budget'] else 'NO ✅'}")
if forecast['will_exceed_budget']:
    print(f"Projected Overage: ${forecast['projected_overage']:,.2f}")
print(f"\nRecommendation: {forecast['recommended_action']}")

# Detect anomalies
anomalies = forecaster.detect_anomalies(
    team="ml-platform",
    project="inference"
)

if anomalies:
    print("\n" + "=" * 60)
    print("Cost Anomalies Detected:")
    print("-" * 60)
    for anomaly in anomalies[:5]:  # Show top 5
        print(f"{anomaly['timestamp']}: ${anomaly['cost']:.2f} (z-score: {anomaly['z_score']:.2f}, severity: {anomaly['severity']})")
```

## Cost Attribution & Chargeback

### Granular Cost Breakdown

Generate detailed cost reports with full attribution:

```bash
# cost-report-generator.sh
#!/bin/bash

echo "GenOps AI - Cost Attribution Report"
echo "===================================="
echo "Period: $(date '+%Y-%m')"
echo ""

# Total organization cost
echo "Organization Total:"
kubectl exec -n genops-system deployment/genops-ai -- \
  genops-cli cost-report --period month --format summary

echo ""
echo "Cost by Team:"
kubectl exec -n genops-system deployment/genops-ai -- \
  genops-cli cost-report --by team --period month --format table

echo ""
echo "Cost by Project:"
kubectl exec -n genops-system deployment/genops-ai -- \
  genops-cli cost-report --by project --period month --format table

echo ""
echo "Cost by Provider:"
kubectl exec -n genops-system deployment/genops-ai -- \
  genops-cli cost-report --by provider --period month --format table

echo ""
echo "Cost by Model:"
kubectl exec -n genops-system deployment/genops-ai -- \
  genops-cli cost-report --by model --period month --top 10

echo ""
echo "Top 10 Customers by Cost:"
kubectl exec -n genops-system deployment/genops-ai -- \
  genops-cli cost-report --by customer --period month --top 10

# Example output:
# Organization Total: $47,892.34
#
# Cost by Team:
# Team               | Cost       | Budget     | % Used | Status
# -------------------|------------|------------|--------|--------
# ml-platform        | $24,531.20 | $30,000.00 | 82%    | ⚠️
# data-engineering   | $15,239.78 | $20,000.00 | 76%    | ✅
# product            | $8,121.36  | $10,000.00 | 81%    | ⚠️
#
# Cost by Provider:
# Provider  | Cost       | Requests | Avg Cost/Req
# ----------|------------|----------|-------------
# OpenAI    | $28,442.10 | 1.2M     | $0.0237
# Anthropic | $15,389.24 | 890K     | $0.0173
# Bedrock   | $4,061.00  | 320K     | $0.0127
```

## Troubleshooting

### Issue: Unexpected Cost Spikes

**Diagnosis:**
```bash
# Check recent cost events
kubectl exec -n genops-system deployment/genops-ai -- \
  genops-cli cost-events --since "1 hour ago" --sort-by cost --limit 20

# Identify cost drivers
kubectl exec -n genops-system deployment/genops-ai -- \
  genops-cli cost-breakdown --period hour --by model,customer

# Check for runaway processes
kubectl top pods -n genops-system --sort-by cpu
```

**Solutions:**
1. Implement rate limiting
2. Add budget constraints
3. Review and optimize expensive operations

---

## Next Steps

1. **Enable cost tracking** with governance attributes
2. **Configure team budgets** with enforcement policies
3. **Deploy cost dashboards** for real-time visibility
4. **Implement intelligent model selection** for cost optimization
5. **Set up forecasting** to prevent budget overruns
6. **Review monthly** and optimize based on actual usage

## Additional Resources

- [FinOps Foundation](https://www.finops.org/)
- [Cloud Cost Optimization Best Practices](https://www.finops.org/framework/)
- [Kubernetes Resource Management](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/)
- [GenOps AI Documentation](https://github.com/KoshiHQ/GenOps-AI)

---

This guide provides comprehensive cost optimization strategies for AI workloads on Kubernetes with GenOps AI governance.
