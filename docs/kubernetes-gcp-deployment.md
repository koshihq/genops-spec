# GenOps AI on Google Kubernetes Engine (GKE)

Complete deployment guide for GenOps AI on Google Kubernetes Engine with native GCP integrations, cost optimization, and enterprise security.

## Table of Contents

1. [Quick Start (5 minutes)](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Prerequisites](#prerequisites)
4. [GKE Cluster Setup](#gke-cluster-setup)
5. [GenOps Deployment](#genops-deployment)
6. [GCP Service Integrations](#gcp-service-integrations)
7. [Cost Management](#cost-management)
8. [Security & Compliance](#security-compliance)
9. [Monitoring & Observability](#monitoring-observability)
10. [Production Optimizations](#production-optimizations)
11. [Troubleshooting](#troubleshooting)

## Quick Start

Deploy GenOps AI on GKE in 5 minutes with basic configuration:

```bash
# 1. Create GKE cluster (if needed)
gcloud container clusters create genops-cluster \
  --zone us-central1-a \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10 \
  --enable-autorepair \
  --enable-autoupgrade

# 2. Get cluster credentials
gcloud container clusters get-credentials genops-cluster --zone us-central1-a

# 3. Install GenOps with GCP optimizations
helm repo add genops https://charts.genops.ai
helm install genops-ai genops/genops-ai \
  --set cloud.provider=gcp \
  --set gcp.project=$(gcloud config get-value project) \
  --set gcp.zone=us-central1-a \
  --set observability.backend=stackdriver

# 4. Verify deployment
kubectl get pods -n genops-system
```

✅ **Result:** GenOps AI running on GKE with Cloud Monitoring integration and GCP cost optimization enabled.

## Architecture Overview

### GenOps on GKE Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Google Cloud VPC                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                  GKE Control Plane                      │ │
│  │              (Fully Managed by Google)                 │ │
│  └─────────────────────────────────────────────────────────┘ │
│                            │                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                   GKE Worker Nodes                     │ │
│  │  ┌─────────────────┐  ┌─────────────────┐             │ │
│  │  │   GenOps Core   │  │  GenOps Proxy   │             │ │
│  │  │   - Policies    │  │  - Cost Tracking│             │ │
│  │  │   - Budget Mgmt │  │  - Rate Limiting│             │ │
│  │  │   - Evaluation  │  │  - Load Balance │             │ │
│  │  └─────────────────┘  └─────────────────┘             │ │
│  │           │                     │                      │ │
│  │  ┌─────────────────┐  ┌─────────────────┐             │ │
│  │  │  AI Workloads   │  │  OpenTelemetry  │             │ │
│  │  │  - LangChain    │  │  - Jaeger       │             │ │
│  │  │  - Custom Apps  │  │  - Prometheus   │             │ │
│  │  │  - Jupyter      │  │  - Grafana      │             │ │
│  │  └─────────────────┘  └─────────────────┘             │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
    ┌────────────────────────────────────────────────────────┐
    │               Google Cloud Services Integration        │
    │                                                        │
    │  Cloud Monitoring    BigQuery         IAM & Security   │
    │  Cloud Trace         Vertex AI        Secret Manager   │
    │  Cloud Logging       Cloud Storage    Cloud KMS        │
    │  Cloud Billing API   Pub/Sub          Firewall Rules   │
    └────────────────────────────────────────────────────────┘
```

### Key Components

- **GKE Autopilot/Standard**: Fully managed Kubernetes control plane
- **Node Auto Provisioning**: Automatic node pool creation and optimization
- **GenOps Workloads**: Governance services with GCP-native integrations
- **Google Cloud Load Balancer**: High-performance load balancing
- **VPC-native Networking**: Pod-level IP addresses with advanced security
- **Persistent Disk CSI**: High-performance persistent storage

## Prerequisites

### Required GCP APIs

Enable the necessary Google Cloud APIs:

```bash
# Enable required APIs
gcloud services enable \
  container.googleapis.com \
  cloudbilling.googleapis.com \
  monitoring.googleapis.com \
  logging.googleapis.com \
  cloudtrace.googleapis.com \
  storage-component.googleapis.com \
  secretmanager.googleapis.com \
  aiplatform.googleapis.com \
  bigquery.googleapis.com \
  pubsub.googleapis.com

# Verify API enablement
gcloud services list --enabled
```

### Required IAM Permissions

Create an IAM policy for GKE and GenOps operations:

```bash
# Create custom role for GenOps
cat > genops-gke-role.yaml << 'EOF'
title: GenOps GKE Role
description: Custom role for GenOps AI on GKE operations
stage: GA
includedPermissions:
  # GKE permissions
  - container.clusters.create
  - container.clusters.delete
  - container.clusters.get
  - container.clusters.list
  - container.clusters.update
  - container.operations.get
  - container.operations.list
  
  # Cost and billing permissions
  - cloudbilling.budgets.get
  - cloudbilling.budgets.list
  - billing.accounts.get
  - billing.resourceCosts.get
  
  # Monitoring and logging
  - monitoring.dashboards.create
  - monitoring.dashboards.update
  - monitoring.metricDescriptors.create
  - monitoring.timeSeries.create
  - logging.logEntries.create
  - logging.sinks.create
  
  # Vertex AI permissions
  - aiplatform.endpoints.predict
  - aiplatform.models.predict
  - aiplatform.endpoints.get
  
  # Storage and secrets
  - storage.objects.create
  - storage.objects.get
  - storage.objects.delete
  - secretmanager.versions.access
  - secretmanager.secrets.get
  
  # BigQuery for analytics
  - bigquery.jobs.create
  - bigquery.tables.create
  - bigquery.tables.get
  - bigquery.datasets.get
EOF

# Create the custom role
gcloud iam roles create genops.gkeOperator \
  --project=$(gcloud config get-value project) \
  --file=genops-gke-role.yaml

# Create service account
gcloud iam service-accounts create genops-gke-sa \
  --description="GenOps AI GKE Service Account" \
  --display-name="GenOps GKE SA"

# Assign roles
gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
  --member="serviceAccount:genops-gke-sa@$(gcloud config get-value project).iam.gserviceaccount.com" \
  --role="projects/$(gcloud config get-value project)/roles/genops.gkeOperator"

gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
  --member="serviceAccount:genops-gke-sa@$(gcloud config get-value project).iam.gserviceaccount.com" \
  --role="roles/container.admin"
```

### Required Tools

```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize gcloud
gcloud init
gcloud auth application-default login

# Install kubectl
gcloud components install kubectl

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify installations
gcloud version
kubectl version --client
helm version
```

## GKE Cluster Setup

### Production-Ready GKE Cluster

Create a production-ready GKE cluster with optimal configuration:

```bash
# Set project variables
export PROJECT_ID=$(gcloud config get-value project)
export CLUSTER_NAME=genops-production
export REGION=us-central1
export ZONE=us-central1-a

# Create production cluster with Autopilot (recommended for simplicity)
gcloud container clusters create-auto $CLUSTER_NAME \
  --region=$REGION \
  --project=$PROJECT_ID \
  --enable-network-policy \
  --enable-vertical-pod-autoscaling \
  --enable-shielded-nodes \
  --labels=environment=production,project=genops-ai,cost-center=engineering

# Alternative: Create standard cluster with custom node pools
cat > genops-gke-cluster.yaml << 'EOF'
# Standard GKE cluster configuration
gcloud container clusters create genops-production \
  --zone us-central1-a \
  --project $(gcloud config get-value project) \
  --machine-type e2-standard-4 \
  --num-nodes 3 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 10 \
  --enable-autorepair \
  --enable-autoupgrade \
  --enable-network-policy \
  --enable-ip-alias \
  --network default \
  --subnetwork default \
  --enable-shielded-nodes \
  --shielded-secure-boot \
  --shielded-integrity-monitoring \
  --disk-type pd-ssd \
  --disk-size 100GB \
  --image-type COS_CONTAINERD \
  --enable-cloud-logging \
  --enable-cloud-monitoring \
  --labels environment=production,project=genops-ai \
  --node-labels environment=production,cost-optimization=enabled
EOF

# Get cluster credentials
gcloud container clusters get-credentials $CLUSTER_NAME --region=$REGION

# Verify cluster is running
kubectl cluster-info
kubectl get nodes
```

### Add Cost-Optimized Node Pools

```bash
# Create preemptible node pool for cost savings
gcloud container node-pools create cost-optimized \
  --cluster=$CLUSTER_NAME \
  --zone=$ZONE \
  --machine-type e2-standard-2 \
  --num-nodes 0 \
  --enable-autoscaling \
  --min-nodes 0 \
  --max-nodes 10 \
  --preemptible \
  --node-labels=cost-optimization=enabled,workload-type=batch \
  --node-taints=preemptible=true:NoSchedule

# Create GPU node pool for AI workloads (optional)
gcloud container node-pools create gpu-workers \
  --cluster=$CLUSTER_NAME \
  --zone=$ZONE \
  --machine-type n1-standard-4 \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --num-nodes 0 \
  --enable-autoscaling \
  --min-nodes 0 \
  --max-nodes 3 \
  --node-labels=workload-type=gpu \
  --node-taints=nvidia.com/gpu=present:NoSchedule

# Install GPU device plugin
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

## GenOps Deployment

### Prepare GenOps Configuration

Create GCP-optimized GenOps configuration:

```bash
# Create GenOps namespace
kubectl create namespace genops-system

# Create GCP-specific configuration
cat > genops-gcp-values.yaml << 'EOF'
# GenOps AI Helm Chart Values for GCP GKE

# Global configuration
global:
  environment: production
  cloud:
    provider: gcp
    project: PROJECT_ID_PLACEHOLDER
    region: us-central1
    zone: us-central1-a
  governance:
    team: platform-engineering
    project: genops-deployment
    cost_center: engineering

# Core GenOps services
genops:
  image:
    repository: genopsai/genops
    tag: "1.0.0"
    pullPolicy: IfNotPresent
  
  replicas: 3
  
  resources:
    requests:
      cpu: 200m
      memory: 512Mi
    limits:
      cpu: 500m
      memory: 1Gi
  
  # GCP-specific configuration
  gcp:
    project: PROJECT_ID_PLACEHOLDER
    region: us-central1
    zone: us-central1-a
    enableCostOptimization: true
    enableVertexAI: true
    enableCloudTracing: true
    
    # Cost management
    billing:
      enabled: true
      budgetAlerts: true
      bigQueryExport: true
      
    # Storage configuration
    storage:
      bucket: genops-governance-data
      region: us-central1
      
    # Secret Manager integration
    secretManager:
      projectId: PROJECT_ID_PLACEHOLDER
      secretName: genops-ai-keys

# Proxy service for AI workloads
proxy:
  enabled: true
  replicas: 2
  
  service:
    type: LoadBalancer
    annotations:
      cloud.google.com/load-balancer-type: Internal
      networking.gke.io/load-balancer-type: Internal
  
  # Rate limiting and cost controls
  rateLimit:
    enabled: true
    requestsPerMinute: 1000
    costPerHour: 100
    
  # Multi-provider support
  providers:
    openai:
      enabled: true
      secretKey: openai-api-key
    anthropic:
      enabled: true
      secretKey: anthropic-api-key
    vertexai:
      enabled: true
      project: PROJECT_ID_PLACEHOLDER
      region: us-central1

# Observability stack
observability:
  # Cloud Monitoring integration
  stackdriver:
    enabled: true
    project: PROJECT_ID_PLACEHOLDER
    
  # Cloud Trace for distributed tracing
  cloudTrace:
    enabled: true
    sampling: 0.1
    
  # Prometheus for metrics
  prometheus:
    enabled: true
    retention: 30d
    storage:
      class: ssd
      size: 100Gi
      
  # Grafana for dashboards
  grafana:
    enabled: true
    adminPassword: "change-me-in-production"
    dashboards:
      gcp: true
      cost: true
      performance: true

# Storage configuration
storage:
  class: ssd
  size: 50Gi
  
# Security configuration
security:
  podSecurityPolicy: true
  networkPolicies: true
  workloadIdentity: true
  
  # RBAC
  rbac:
    enabled: true

# Auto-scaling configuration
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPU: 70
  targetMemory: 80

# Cost optimization
costOptimization:
  enabled: true
  preemptibleNodes: true
  nodeAffinity: true
  resourceOptimization: true
  
  # Scheduled scaling for cost savings
  schedule:
    enabled: true
    # Scale down during non-business hours
    scaleDown:
      schedule: "0 18 * * *"
      replicas: 1
    scaleUp:
      schedule: "0 8 * * *" 
      replicas: 3
EOF

# Replace project ID placeholder
sed -i "s/PROJECT_ID_PLACEHOLDER/$(gcloud config get-value project)/g" genops-gcp-values.yaml
```

### Deploy GenOps with Helm

```bash
# Add GenOps Helm repository
helm repo add genops https://charts.genops.ai
helm repo update

# Install GenOps AI
helm install genops-ai genops/genops-ai \
  --namespace genops-system \
  --values genops-gcp-values.yaml \
  --wait \
  --timeout 10m

# Verify deployment
kubectl get pods -n genops-system
kubectl get services -n genops-system

# Check logs
kubectl logs -n genops-system deployment/genops-ai --tail=100
```

### Configure Workload Identity

Set up Workload Identity for secure GCP API access:

```bash
# Enable Workload Identity on cluster (if not already enabled)
gcloud container clusters update $CLUSTER_NAME \
  --zone=$ZONE \
  --workload-pool=$(gcloud config get-value project).svc.id.goog

# Create Kubernetes service account
kubectl create serviceaccount genops-ksa \
  --namespace genops-system

# Create Google service account
gcloud iam service-accounts create genops-gsa \
  --project=$(gcloud config get-value project)

# Bind service accounts
gcloud iam service-accounts add-iam-policy-binding \
  genops-gsa@$(gcloud config get-value project).iam.gserviceaccount.com \
  --role roles/iam.workloadIdentityUser \
  --member "serviceAccount:$(gcloud config get-value project).svc.id.goog[genops-system/genops-ksa]"

# Annotate Kubernetes service account
kubectl annotate serviceaccount genops-ksa \
  --namespace genops-system \
  iam.gke.io/gcp-service-account=genops-gsa@$(gcloud config get-value project).iam.gserviceaccount.com

# Update deployment to use service account
kubectl patch deployment genops-ai \
  --namespace genops-system \
  --patch '{"spec":{"template":{"spec":{"serviceAccountName":"genops-ksa"}}}}'
```

## GCP Service Integrations

### Vertex AI Integration

Configure GenOps to work with Vertex AI:

```bash
# Create Vertex AI-specific configuration
cat > vertex-ai-integration.yaml << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: vertex-ai-config
  namespace: genops-system
data:
  config.yaml: |
    vertex_ai:
      project_id: PROJECT_ID_PLACEHOLDER
      region: us-central1
      models:
        - name: text-bison@001
          cost_per_1k_input: 0.0005
          cost_per_1k_output: 0.0005
        - name: chat-bison@001
          cost_per_1k_input: 0.0005
          cost_per_1k_output: 0.0005
        - name: code-bison@001
          cost_per_1k_input: 0.0005
          cost_per_1k_output: 0.0005
        - name: codechat-bison@001
          cost_per_1k_input: 0.0005
          cost_per_1k_output: 0.0005
      governance:
        enable_cost_tracking: true
        enable_content_filtering: true
        enable_budget_limits: true
        enable_usage_quotas: true
EOF

# Replace project ID and apply
sed -i "s/PROJECT_ID_PLACEHOLDER/$(gcloud config get-value project)/g" vertex-ai-integration.yaml
kubectl apply -f vertex-ai-integration.yaml
```

### Cloud Monitoring Integration

Configure comprehensive Cloud Monitoring integration:

```bash
# Install Google Cloud Monitoring operator
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/prometheus-engine/main/manifests/operator.yaml

# Create custom metrics for GenOps
cat > genops-gcp-metrics.yaml << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: genops-metrics-config
  namespace: genops-system
data:
  metrics.yaml: |
    custom_metrics:
      - name: genops_ai_requests_total
        type: counter
        description: Total AI API requests processed
        labels:
          - provider
          - model
          - team
          - project
      
      - name: genops_cost_per_hour
        type: gauge
        description: Cost per hour by team/project
        labels:
          - team
          - project
          - cost_center
      
      - name: genops_policy_violations_total
        type: counter
        description: Total policy violations
        labels:
          - policy_type
          - severity
      
      - name: genops_budget_utilization
        type: gauge
        description: Budget utilization percentage
        labels:
          - budget_name
          - team
    
    export_settings:
      stackdriver:
        enabled: true
        project: PROJECT_ID_PLACEHOLDER
        interval: 60s
        prefix: custom.googleapis.com/genops/
EOF

sed -i "s/PROJECT_ID_PLACEHOLDER/$(gcloud config get-value project)/g" genops-gcp-metrics.yaml
kubectl apply -f genops-gcp-metrics.yaml
```

### BigQuery Integration for Analytics

Set up BigQuery for cost analytics and reporting:

```bash
# Create BigQuery dataset
bq mk --dataset \
  --description "GenOps AI cost and usage analytics" \
  --location=US \
  $(gcloud config get-value project):genops_analytics

# Create cost tracking table
bq mk --table \
  $(gcloud config get-value project):genops_analytics.cost_tracking \
  timestamp:TIMESTAMP,team:STRING,project:STRING,provider:STRING,model:STRING,cost:FLOAT,tokens_in:INTEGER,tokens_out:INTEGER,operation:STRING

# Create usage analytics table
bq mk --table \
  $(gcloud config get-value project):genops_analytics.usage_analytics \
  timestamp:TIMESTAMP,user_id:STRING,team:STRING,project:STRING,request_type:STRING,response_time:FLOAT,success:BOOLEAN

# Create budget tracking table
bq mk --table \
  $(gcloud config get-value project):genops_analytics.budget_tracking \
  timestamp:TIMESTAMP,budget_name:STRING,allocated:FLOAT,used:FLOAT,remaining:FLOAT,utilization_percent:FLOAT

# Configure GenOps to export to BigQuery
cat > bigquery-export-config.yaml << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: bigquery-export-config
  namespace: genops-system
data:
  export.yaml: |
    bigquery:
      enabled: true
      project_id: PROJECT_ID_PLACEHOLDER
      dataset: genops_analytics
      tables:
        cost_tracking: cost_tracking
        usage_analytics: usage_analytics  
        budget_tracking: budget_tracking
      batch_size: 100
      flush_interval: 60s
EOF

sed -i "s/PROJECT_ID_PLACEHOLDER/$(gcloud config get-value project)/g" bigquery-export-config.yaml
kubectl apply -f bigquery-export-config.yaml
```

### Cloud Billing API Integration

Configure automatic cost tracking and budgets:

```bash
# Create billing configuration
cat > billing-integration.yaml << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: billing-config
  namespace: genops-system
data:
  billing.yaml: |
    billing:
      enabled: true
      project_id: PROJECT_ID_PLACEHOLDER
      billing_account_id: BILLING_ACCOUNT_ID_PLACEHOLDER
      
      budgets:
        - name: genops-monthly-budget
          amount: 1000
          currency: USD
          time_unit: MONTHLY
          filters:
            projects:
              - PROJECT_ID_PLACEHOLDER
            labels:
              - key: "project"
                value: "genops-ai"
          
          alerts:
            - threshold: 0.8
              type: ACTUAL
              emails:
                - platform-team@company.com
            - threshold: 1.0
              type: FORECASTED
              emails:
                - platform-team@company.com
      
      cost_optimization:
        preemptible_percentage: 50
        auto_scaling: true
        scheduled_scaling: true
EOF

# Get billing account ID
BILLING_ACCOUNT=$(gcloud billing accounts list --format="value(name)" | head -n1)
sed -i "s/PROJECT_ID_PLACEHOLDER/$(gcloud config get-value project)/g" billing-integration.yaml
sed -i "s/BILLING_ACCOUNT_ID_PLACEHOLDER/$BILLING_ACCOUNT/g" billing-integration.yaml

kubectl apply -f billing-integration.yaml
```

## Cost Management

### Node Pool Optimization

Create cost-optimized node pools:

```bash
# Create spot/preemptible node pool for batch workloads
gcloud container node-pools create spot-pool \
  --cluster=$CLUSTER_NAME \
  --zone=$ZONE \
  --machine-type e2-standard-2 \
  --spot \
  --num-nodes 0 \
  --enable-autoscaling \
  --min-nodes 0 \
  --max-nodes 20 \
  --node-labels=cost-optimization=spot,workload-type=batch \
  --node-taints=spot=true:NoSchedule

# Create mixed node pool for optimal cost/performance
gcloud container node-pools create mixed-pool \
  --cluster=$CLUSTER_NAME \
  --zone=$ZONE \
  --machine-type e2-standard-4 \
  --num-nodes 1 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 8 \
  --enable-autorepair \
  --enable-autoupgrade \
  --node-labels=cost-optimization=mixed,workload-type=general
```

### Cluster Autoscaler Configuration

Configure intelligent cluster autoscaling:

```bash
# Configure cluster autoscaler
kubectl apply -f - << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: cluster-autoscaler-status
  namespace: kube-system
  labels:
    k8s-addon: cluster-autoscaler.addons.k8s.io
    k8s-app: cluster-autoscaler
data:
  nodes.max: "50"
  nodes.min: "3"
  scale-down-delay-after-add: "10m"
  scale-down-unneeded-time: "10m"
  skip-nodes-with-local-storage: "false"
  skip-nodes-with-system-pods: "false"
  balance-similar-node-groups: "true"
  expander: "least-waste"
EOF

# Enable cluster autoscaler on existing node pools
for pool in default-pool cost-optimized spot-pool mixed-pool; do
  gcloud container clusters update $CLUSTER_NAME \
    --zone=$ZONE \
    --enable-autoscaling \
    --node-pool=$pool \
    --min-nodes=0 \
    --max-nodes=10 || true
done
```

### Cost Monitoring and Alerting

Set up comprehensive cost monitoring:

```bash
# Create cost monitoring namespace
kubectl create namespace cost-monitoring

# Deploy GCP cost exporter
cat > gcp-cost-exporter.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gcp-cost-exporter
  namespace: cost-monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: gcp-cost-exporter
  template:
    metadata:
      labels:
        app: gcp-cost-exporter
    spec:
      serviceAccountName: genops-ksa
      containers:
      - name: gcp-cost-exporter
        image: genopsai/gcp-cost-exporter:latest
        env:
        - name: GOOGLE_CLOUD_PROJECT
          value: PROJECT_ID_PLACEHOLDER
        - name: CLUSTER_NAME
          value: genops-production
        - name: CLUSTER_ZONE
          value: us-central1-a
        ports:
        - containerPort: 9090
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 200m
            memory: 256Mi
---
apiVersion: v1
kind: Service
metadata:
  name: gcp-cost-exporter
  namespace: cost-monitoring
  labels:
    app: gcp-cost-exporter
spec:
  ports:
  - port: 9090
    targetPort: 9090
  selector:
    app: gcp-cost-exporter
EOF

sed -i "s/PROJECT_ID_PLACEHOLDER/$(gcloud config get-value project)/g" gcp-cost-exporter.yaml
kubectl apply -f gcp-cost-exporter.yaml
```

### Budget Alerts and Controls

Create automated budget management:

```bash
# Create budget alert function
cat > budget-alert-function.js << 'EOF'
const { BigQuery } = require('@google-cloud/bigquery');
const { PubSub } = require('@google-cloud/pubsub');

exports.budgetAlert = async (data, context) => {
  const message = Buffer.from(data.data, 'base64').toString();
  const budgetNotification = JSON.parse(message);
  
  console.log('Budget notification:', budgetNotification);
  
  const costAmount = budgetNotification.costAmount;
  const budgetAmount = budgetNotification.budgetAmount;
  const budgetDisplayName = budgetNotification.budgetDisplayName;
  
  // Calculate utilization percentage
  const utilization = (costAmount / budgetAmount) * 100;
  
  if (utilization > 80) {
    // Scale down non-critical workloads
    console.log(`High budget utilization: ${utilization}%. Scaling down...`);
    
    // Publish scaling message
    const pubsub = new PubSub();
    await pubsub.topic('genops-scaling').publish(Buffer.from(JSON.stringify({
      action: 'scale-down',
      reason: 'budget-limit',
      utilization: utilization
    })));
  }
  
  // Log to BigQuery for analytics
  const bigquery = new BigQuery();
  const dataset = bigquery.dataset('genops_analytics');
  const table = dataset.table('budget_tracking');
  
  await table.insert([{
    timestamp: new Date(),
    budget_name: budgetDisplayName,
    allocated: budgetAmount,
    used: costAmount,
    remaining: budgetAmount - costAmount,
    utilization_percent: utilization
  }]);
};
EOF

# Deploy Cloud Function (requires gcloud functions)
gcloud functions deploy budget-alert \
  --runtime nodejs18 \
  --trigger-topic budget-notifications \
  --source . \
  --entry-point budgetAlert \
  --memory 256MB
```

## Security & Compliance

### Workload Identity and IAM

Configure secure workload identity:

```bash
# Create IAM policy for GenOps workloads
cat > genops-workload-policy.json << 'EOF'
{
  "bindings": [
    {
      "role": "roles/monitoring.metricWriter",
      "members": [
        "serviceAccount:genops-gsa@PROJECT_ID.iam.gserviceaccount.com"
      ]
    },
    {
      "role": "roles/logging.logWriter", 
      "members": [
        "serviceAccount:genops-gsa@PROJECT_ID.iam.gserviceaccount.com"
      ]
    },
    {
      "role": "roles/cloudtrace.agent",
      "members": [
        "serviceAccount:genops-gsa@PROJECT_ID.iam.gserviceaccount.com"
      ]
    },
    {
      "role": "roles/aiplatform.user",
      "members": [
        "serviceAccount:genops-gsa@PROJECT_ID.iam.gserviceaccount.com"
      ]
    },
    {
      "role": "roles/bigquery.dataEditor",
      "members": [
        "serviceAccount:genops-gsa@PROJECT_ID.iam.gserviceaccount.com"
      ]
    }
  ]
}
EOF

# Apply IAM policy
sed -i "s/PROJECT_ID/$(gcloud config get-value project)/g" genops-workload-policy.json
gcloud projects set-iam-policy $(gcloud config get-value project) genops-workload-policy.json
```

### Network Security Policies

Configure VPC-native networking security:

```bash
# Create network policies for GenOps namespace
cat > genops-network-policies.yaml << 'EOF'
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: genops-default-deny
  namespace: genops-system
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: genops-allow-internal
  namespace: genops-system
spec:
  podSelector:
    matchLabels:
      app: genops-ai
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: genops-system
    - podSelector: {}
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: genops-system
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS to GCP APIs
    - protocol: TCP
      port: 53   # DNS
    - protocol: UDP
      port: 53   # DNS
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: genops-allow-external-ai-apis
  namespace: genops-system
spec:
  podSelector:
    matchLabels:
      component: proxy
  policyTypes:
  - Egress
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443
EOF

kubectl apply -f genops-network-policies.yaml
```

### Pod Security Standards

Implement pod security standards:

```bash
# Apply pod security standards to namespace
kubectl label namespace genops-system \
  pod-security.kubernetes.io/enforce=restricted \
  pod-security.kubernetes.io/audit=restricted \
  pod-security.kubernetes.io/warn=restricted

# Create security context constraints
cat > genops-security-context.yaml << 'EOF'
apiVersion: v1
kind: SecurityContext
metadata:
  name: genops-security-context
spec:
  runAsNonRoot: true
  runAsUser: 10001
  runAsGroup: 10001
  fsGroup: 10001
  seccompProfile:
    type: RuntimeDefault
  capabilities:
    drop:
      - ALL
    add:
      - NET_BIND_SERVICE
---
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: genops-ai-pdb
  namespace: genops-system
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: genops-ai
EOF

kubectl apply -f genops-security-context.yaml
```

### Binary Authorization

Configure container image security:

```bash
# Enable Binary Authorization
gcloud container binauthz policy import policy.yaml

# Create security policy
cat > binauthz-policy.yaml << 'EOF'
admissionWhitelistPatterns:
- namePattern: gcr.io/PROJECT_ID/*
- namePattern: genopsai/*
defaultAdmissionRule:
  requireAttestationsBy:
  - projects/PROJECT_ID/attestors/prod-attestor
  evaluationMode: REQUIRE_ATTESTATION
  enforcementMode: ENFORCED_BLOCK_AND_AUDIT_LOG
name: projects/PROJECT_ID/policy
EOF

sed -i "s/PROJECT_ID/$(gcloud config get-value project)/g" binauthz-policy.yaml
gcloud container binauthz policy import binauthz-policy.yaml
```

## Monitoring & Observability

### Comprehensive Monitoring Stack

Deploy full observability stack for GenOps:

```bash
# Install Prometheus and Grafana using Google Cloud Marketplace or custom deployment
kubectl create namespace monitoring

# Install Prometheus operator
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --set grafana.adminPassword=admin \
  --set prometheus.prometheusSpec.retention=30d \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=100Gi

# Install Jaeger for distributed tracing
helm repo add jaegertracing https://jaegertracing.github.io/helm-charts  
helm install jaeger jaegertracing/jaeger \
  --namespace monitoring \
  --set provisionDataStore.cassandra=false \
  --set storage.type=memory

# Configure Google Cloud Monitoring integration
cat > cloud-monitoring-config.yaml << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-gcp-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      external_labels:
        project_id: 'PROJECT_ID'
        cluster: 'genops-production'
        location: 'us-central1-a'
    
    remote_write:
    - url: 'https://monitoring.googleapis.com:443/v1/projects/PROJECT_ID/location/global/prometheus/api/v1/write'
      queue_config:
        capacity: 2500
        max_shards: 200
        min_shards: 1
        max_samples_per_send: 500
        batch_send_deadline: 5s
        min_backoff: 30ms
        max_backoff: 100ms
EOF

sed -i "s/PROJECT_ID/$(gcloud config get-value project)/g" cloud-monitoring-config.yaml
kubectl apply -f cloud-monitoring-config.yaml
```

### Custom Dashboards

Create GenOps-specific monitoring dashboards:

```bash
# Create GenOps dashboard for Grafana
cat > genops-gcp-dashboard.json << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "GenOps AI - Google Cloud Platform",
    "tags": ["genops", "ai", "cost", "governance", "gcp"],
    "timezone": "browser",
    "panels": [
      {
        "title": "AI API Requests by Provider",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(genops_ai_requests_total[5m])) by (provider)",
            "legendFormat": "{{provider}}"
          }
        ],
        "yAxes": [{
          "label": "Requests/sec"
        }]
      },
      {
        "title": "Cost per Hour by Team",
        "type": "graph", 
        "targets": [
          {
            "expr": "sum(genops_cost_per_hour) by (team, project)",
            "legendFormat": "{{team}}/{{project}}"
          }
        ],
        "yAxes": [{
          "label": "USD per hour"
        }]
      },
      {
        "title": "GKE Node Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "100 - (avg(irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
            "legendFormat": "CPU Utilization %"
          },
          {
            "expr": "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100",
            "legendFormat": "Memory Utilization %"
          }
        ]
      },
      {
        "title": "Budget Utilization",
        "type": "singlestat",
        "targets": [
          {
            "expr": "(sum(genops_budget_used) / sum(genops_budget_limit)) * 100",
            "legendFormat": "Budget Used %"
          }
        ],
        "thresholds": "80,95",
        "colorBackground": true
      },
      {
        "title": "Vertex AI Model Performance",
        "type": "table",
        "targets": [
          {
            "expr": "avg_over_time(genops_vertex_ai_latency[1h]) by (model)",
            "format": "table"
          }
        ]
      },
      {
        "title": "Policy Violations",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(genops_policy_violations_total[5m])) by (policy_type)",
            "legendFormat": "{{policy_type}}"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
EOF

kubectl create configmap genops-gcp-dashboard \
  --from-file=dashboard.json=genops-gcp-dashboard.json \
  --namespace monitoring
```

### Cloud Trace Integration

Configure distributed tracing with Cloud Trace:

```bash
# Configure Cloud Trace integration
kubectl apply -f - << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: cloud-trace-config
  namespace: genops-system
data:
  trace.yaml: |
    cloud_trace:
      enabled: true
      project_id: PROJECT_ID
      sampling_rate: 0.1
      
    jaeger:
      enabled: true
      endpoint: http://jaeger-collector.monitoring:14268/api/traces
      
    opentelemetry:
      enabled: true
      exporters:
        - google_cloud_trace
        - jaeger
      
      resource:
        attributes:
          service.name: genops-ai
          service.version: 1.0.0
          cloud.provider: gcp
          cloud.platform: gcp_kubernetes_engine
          k8s.cluster.name: genops-production
EOF

sed -i "s/PROJECT_ID/$(gcloud config get-value project)/g" /tmp/cloud-trace-config.yaml
kubectl apply -f /tmp/cloud-trace-config.yaml
```

## Production Optimizations

### High Availability Configuration

Configure GenOps for high availability:

```bash
# Configure multi-zone deployment
kubectl patch deployment genops-ai \
  -n genops-system \
  -p='{"spec":{"replicas":3,"template":{"spec":{"affinity":{"podAntiAffinity":{"requiredDuringSchedulingIgnoredDuringExecution":[{"labelSelector":{"matchExpressions":[{"key":"app","operator":"In","values":["genops-ai"]}]},"topologyKey":"failure-domain.beta.kubernetes.io/zone"}]}}}}}}'

# Create pod disruption budget
kubectl apply -f - << 'EOF'
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: genops-ai-pdb
  namespace: genops-system
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: genops-ai
EOF
```

### Auto-scaling Configuration

Configure horizontal and vertical pod autoscaling:

```bash
# Horizontal Pod Autoscaler
kubectl apply -f - << 'EOF'
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: genops-ai-hpa
  namespace: genops-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: genops-ai
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: External
    external:
      metric:
        name: custom.googleapis.com|genops|ai_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
    scaleUp:
      stabilizationWindowSeconds: 60
EOF

# Vertical Pod Autoscaler
kubectl apply -f - << 'EOF'
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: genops-ai-vpa
  namespace: genops-system
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: genops-ai
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: genops-ai
      maxAllowed:
        cpu: 2
        memory: 4Gi
      minAllowed:
        cpu: 100m
        memory: 256Mi
EOF
```

### Backup and Disaster Recovery

Implement backup and disaster recovery:

```bash
# Create backup configuration for persistent data
cat > backup-config.yaml << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: backup-config
  namespace: genops-system
data:
  backup.yaml: |
    backup:
      enabled: true
      schedule: "0 2 * * *"  # Daily at 2 AM
      retention_days: 30
      
      destinations:
        - type: gcs
          bucket: genops-backup-bucket
          path: /kubernetes-backups/
        - type: bigquery
          dataset: genops_backup
          
      components:
        - persistent_volumes
        - secrets
        - configmaps
        - custom_resources
        
      notifications:
        - type: email
          recipients:
            - platform-team@company.com
        - type: slack
          webhook: https://hooks.slack.com/services/...
EOF

kubectl apply -f backup-config.yaml

# Create backup service account and permissions
gcloud iam service-accounts create genops-backup-sa \
  --description="GenOps backup service account"

gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
  --member="serviceAccount:genops-backup-sa@$(gcloud config get-value project).iam.gserviceaccount.com" \
  --role="roles/storage.admin"

# Create backup CronJob
kubectl apply -f - << 'EOF'
apiVersion: batch/v1
kind: CronJob
metadata:
  name: genops-backup
  namespace: genops-system
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          serviceAccountName: genops-backup-sa
          containers:
          - name: backup
            image: genopsai/backup-tool:latest
            env:
            - name: GOOGLE_CLOUD_PROJECT
              value: PROJECT_ID
            - name: BACKUP_BUCKET
              value: genops-backup-bucket
            command:
            - /bin/sh
            - -c
            - |
              echo "Starting backup at $(date)"
              kubectl get all -n genops-system -o yaml > /tmp/genops-backup.yaml
              gsutil cp /tmp/genops-backup.yaml gs://genops-backup-bucket/$(date +%Y-%m-%d)/
              echo "Backup completed at $(date)"
          restartPolicy: OnFailure
EOF

sed -i "s/PROJECT_ID/$(gcloud config get-value project)/g" /tmp/backup-cronjob.yaml
kubectl apply -f /tmp/backup-cronjob.yaml
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: Pods Stuck in Pending State

**Diagnosis:**
```bash
kubectl describe pod <pod-name> -n genops-system
kubectl get events -n genops-system --sort-by=.metadata.creationTimestamp
gcloud container clusters describe $CLUSTER_NAME --zone=$ZONE
```

**Solutions:**
1. **Insufficient Resources:**
   ```bash
   # Check node capacity
   kubectl top nodes
   kubectl describe nodes
   
   # Add more nodes or create new node pool
   gcloud container node-pools create additional-pool \
     --cluster=$CLUSTER_NAME \
     --zone=$ZONE \
     --num-nodes=3 \
     --machine-type=e2-standard-4
   ```

2. **Node Pool Constraints:**
   ```bash
   # Check node pool status
   gcloud container node-pools list --cluster=$CLUSTER_NAME --zone=$ZONE
   
   # Enable autoscaling if needed
   gcloud container clusters update $CLUSTER_NAME \
     --zone=$ZONE \
     --enable-autoscaling \
     --min-nodes=1 \
     --max-nodes=10
   ```

#### Issue: High GCP Costs

**Diagnosis:**
```bash
# Check current billing
gcloud billing budgets list --billing-account=$BILLING_ACCOUNT

# Analyze resource usage
kubectl top nodes
kubectl top pods -n genops-system

# Check GKE cluster costs
gcloud container clusters describe $CLUSTER_NAME --zone=$ZONE --format="value(currentNodeCount,currentMasterVersion)"
```

**Solutions:**
1. **Enable Preemptible Nodes:**
   ```bash
   # Create preemptible node pool
   gcloud container node-pools create preemptible-pool \
     --cluster=$CLUSTER_NAME \
     --zone=$ZONE \
     --preemptible \
     --num-nodes=2 \
     --machine-type=e2-standard-2
   
   # Migrate workloads to preemptible nodes
   kubectl patch deployment genops-ai -n genops-system --patch '{"spec":{"template":{"spec":{"tolerations":[{"key":"cloud.google.com/gke-preemptible","operator":"Equal","value":"true","effect":"NoSchedule"}]}}}}'
   ```

2. **Optimize Resource Requests:**
   ```bash
   # Check current resource usage
   kubectl describe deployment genops-ai -n genops-system
   
   # Update resource requests
   kubectl patch deployment genops-ai -n genops-system --patch '{"spec":{"template":{"spec":{"containers":[{"name":"genops-ai","resources":{"requests":{"cpu":"100m","memory":"256Mi"},"limits":{"cpu":"300m","memory":"512Mi"}}}]}}}}'
   ```

#### Issue: Vertex AI Connection Problems

**Diagnosis:**
```bash
# Check service account permissions
gcloud projects get-iam-policy $(gcloud config get-value project)

# Test Vertex AI connectivity
kubectl exec -n genops-system deployment/genops-ai -- curl -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  https://us-central1-aiplatform.googleapis.com/v1/projects/$(gcloud config get-value project)/locations/us-central1/endpoints

# Check logs
kubectl logs -n genops-system deployment/genops-ai | grep -i vertex
```

**Solutions:**
1. **Fix Service Account Permissions:**
   ```bash
   # Add required roles
   gcloud projects add-iam-policy-binding $(gcloud config get-value project) \
     --member="serviceAccount:genops-gsa@$(gcloud config get-value project).iam.gserviceaccount.com" \
     --role="roles/aiplatform.user"
   
   # Restart deployment
   kubectl rollout restart deployment/genops-ai -n genops-system
   ```

2. **Update Workload Identity:**
   ```bash
   # Re-configure workload identity binding
   gcloud iam service-accounts add-iam-policy-binding \
     genops-gsa@$(gcloud config get-value project).iam.gserviceaccount.com \
     --role roles/iam.workloadIdentityUser \
     --member "serviceAccount:$(gcloud config get-value project).svc.id.goog[genops-system/genops-ksa]"
   ```

### Health Checks and Validation

```bash
# Comprehensive health check script
cat > health-check-gcp.sh << 'EOF'
#!/bin/bash
echo "🔍 GenOps GKE Health Check"
echo "=========================="

# Check cluster health
echo "📋 Cluster Status:"
gcloud container clusters describe $CLUSTER_NAME --zone=$ZONE --format="value(status)"
kubectl cluster-info
kubectl get nodes

# Check GenOps deployment
echo -e "\n🚀 GenOps Deployment:"
kubectl get pods -n genops-system
kubectl get services -n genops-system

# Check resource usage
echo -e "\n📊 Resource Usage:"
kubectl top nodes
kubectl top pods -n genops-system

# Check workload identity
echo -e "\n🔐 Workload Identity:"
kubectl get sa genops-ksa -n genops-system -o yaml | grep -i annotation

# Check GCP API connectivity
echo -e "\n☁️ GCP Integration:"
kubectl exec -n genops-system deployment/genops-ai -- curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  https://monitoring.googleapis.com/v3/projects/$(gcloud config get-value project)/metricDescriptors

# Check cost tracking
echo -e "\n💰 Cost Tracking:"
bq query --use_legacy_sql=false --format=prettyjson \
  "SELECT COUNT(*) as records FROM \`$(gcloud config get-value project).genops_analytics.cost_tracking\` WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)"

echo -e "\n✅ Health check complete"
EOF

chmod +x health-check-gcp.sh
./health-check-gcp.sh
```

### Performance Optimization

```bash
# Enable GKE performance monitoring
gcloud container clusters update $CLUSTER_NAME \
  --zone=$ZONE \
  --enable-network-policy \
  --logging=SYSTEM,WORKLOAD \
  --monitoring=SYSTEM,WORKLOAD

# Configure performance settings
kubectl apply -f - << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: genops-performance-config
  namespace: genops-system
data:
  performance.yaml: |
    gcp:
      optimization:
        connection_pooling: true
        request_batching: true
        cache_enabled: true
        cache_ttl: 300s
        
      vertex_ai:
        request_timeout: 30s
        retry_attempts: 3
        connection_pool_size: 50
        
      monitoring:
        sample_rate: 0.1
        metrics_interval: 30s
        
      networking:
        keep_alive: true
        max_idle_connections: 100
        idle_timeout: 90s
EOF

# Apply performance settings
kubectl rollout restart deployment/genops-ai -n genops-system
```

---

## Next Steps

1. **Set up advanced monitoring** with custom Cloud Monitoring dashboards
2. **Configure GitOps workflow** with Cloud Build and Anthos Config Management  
3. **Enable multi-region deployment** for global availability
4. **Optimize costs** with committed use discounts and sustained use discounts
5. **Implement advanced security** with Binary Authorization and GKE Autopilot security features

## Additional Resources

- [GKE Best Practices Guide](https://cloud.google.com/kubernetes-engine/docs/best-practices)
- [GenOps AI Documentation](https://docs.genops.ai)
- [Kubernetes Cost Optimization on GCP](https://cloud.google.com/kubernetes-engine/docs/how-to/cost-optimization)
- [Google Cloud Cost Management](https://cloud.google.com/cost-management)

This guide provides a comprehensive foundation for deploying GenOps AI on Google Kubernetes Engine with production-ready configurations, cost optimization, and enterprise security.