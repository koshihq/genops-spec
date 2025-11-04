# GenOps AI on Azure Kubernetes Service (AKS)

Complete deployment guide for GenOps AI on Azure Kubernetes Service with native Azure integrations, cost optimization, and enterprise security.

## Table of Contents

1. [Quick Start (5 minutes)](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Prerequisites](#prerequisites)
4. [AKS Cluster Setup](#aks-cluster-setup)
5. [GenOps Deployment](#genops-deployment)
6. [Azure Service Integrations](#azure-service-integrations)
7. [Cost Management](#cost-management)
8. [Security & Compliance](#security-compliance)
9. [Monitoring & Observability](#monitoring-observability)
10. [Production Optimizations](#production-optimizations)
11. [Troubleshooting](#troubleshooting)

## Quick Start

Deploy GenOps AI on AKS in 5 minutes with basic configuration:

```bash
# 1. Create AKS cluster (if needed)
az aks create \
  --resource-group genops-rg \
  --name genops-cluster \
  --node-count 3 \
  --enable-addons monitoring \
  --enable-managed-identity \
  --generate-ssh-keys

# 2. Get cluster credentials
az aks get-credentials --resource-group genops-rg --name genops-cluster

# 3. Install GenOps with Azure optimizations
helm repo add genops https://charts.genops.ai
helm install genops-ai genops/genops-ai \
  --set cloud.provider=azure \
  --set azure.subscriptionId=$(az account show --query id -o tsv) \
  --set azure.resourceGroup=genops-rg \
  --set observability.backend=azuremonitor

# 4. Verify deployment
kubectl get pods -n genops-system
```

✅ **Result:** GenOps AI running on AKS with Azure Monitor integration and Azure cost optimization enabled.

## Architecture Overview

### GenOps on AKS Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Azure Virtual Network               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                  AKS Control Plane                      │ │
│  │              (Fully Managed by Azure)                  │ │
│  └─────────────────────────────────────────────────────────┘ │
│                            │                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                   AKS Worker Nodes                     │ │
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
    │               Azure Services Integration               │
    │                                                        │
    │  Azure Monitor     Cost Management  Azure AD & RBAC   │
    │  Application       Azure OpenAI     Key Vault         │
    │  Insights          Blob Storage     Azure Policy      │
    │  Log Analytics     Service Bus      Azure Firewall    │
    └────────────────────────────────────────────────────────┘
```

### Key Components

- **AKS Control Plane**: Fully managed Kubernetes API server and etcd
- **Virtual Machine Scale Sets**: Auto-scaling worker nodes with Spot VM support
- **GenOps Services**: Core governance services with Azure-native integrations
- **Azure Load Balancer**: Layer 4 and Layer 7 load balancing
- **Azure CNI**: Native VNet integration with subnet-level security
- **Azure Disks**: High-performance persistent storage for governance data

## Prerequisites

### Required Azure Permissions

Create an Azure service principal with required permissions:

```bash
# Create resource group
az group create --name genops-rg --location eastus

# Create service principal
az ad sp create-for-rbac \
  --name genops-sp \
  --role Contributor \
  --scopes /subscriptions/$(az account show --query id -o tsv)/resourceGroups/genops-rg

# Add additional permissions for cost management
az role assignment create \
  --assignee $(az ad sp show --id genops-sp --query appId -o tsv) \
  --role "Cost Management Reader" \
  --scope /subscriptions/$(az account show --query id -o tsv)

# Add Key Vault permissions
az role assignment create \
  --assignee $(az ad sp show --id genops-sp --query appId -o tsv) \
  --role "Key Vault Secrets User" \
  --scope /subscriptions/$(az account show --query id -o tsv)/resourceGroups/genops-rg
```

### Required Azure Resource Providers

Register necessary Azure resource providers:

```bash
# Register required resource providers
az provider register --namespace Microsoft.ContainerService
az provider register --namespace Microsoft.OperationalInsights
az provider register --namespace Microsoft.Insights
az provider register --namespace Microsoft.Storage
az provider register --namespace Microsoft.KeyVault
az provider register --namespace Microsoft.CognitiveServices
az provider register --namespace Microsoft.EventHub
az provider register --namespace Microsoft.ServiceBus
az provider register --namespace Microsoft.Network

# Check registration status
az provider show --namespace Microsoft.ContainerService --query registrationState
```

### Required Tools

```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login to Azure
az login
az account set --subscription "your-subscription-id"

# Install kubectl
az aks install-cli

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Verify installations
az version
kubectl version --client
helm version
```

## AKS Cluster Setup

### Production-Ready AKS Cluster

Create a production-ready AKS cluster with optimal configuration:

```bash
# Set variables
export SUBSCRIPTION_ID=$(az account show --query id -o tsv)
export RESOURCE_GROUP=genops-production-rg
export CLUSTER_NAME=genops-production
export LOCATION=eastus
export NODE_COUNT=3

# Create resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create Log Analytics workspace for monitoring
az monitor log-analytics workspace create \
  --resource-group $RESOURCE_GROUP \
  --workspace-name genops-analytics \
  --location $LOCATION

# Create AKS cluster with production configuration
az aks create \
  --resource-group $RESOURCE_GROUP \
  --name $CLUSTER_NAME \
  --location $LOCATION \
  --node-count $NODE_COUNT \
  --node-vm-size Standard_D4s_v3 \
  --enable-managed-identity \
  --enable-addons monitoring \
  --workspace-resource-id $(az monitor log-analytics workspace show --resource-group $RESOURCE_GROUP --workspace-name genops-analytics --query id -o tsv) \
  --network-plugin azure \
  --network-policy calico \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 10 \
  --enable-encryption-at-host \
  --generate-ssh-keys \
  --tags Environment=production Project=genops-ai CostCenter=engineering
```

### Add Cost-Optimized Node Pools

```bash
# Create Spot VM node pool for cost savings
az aks nodepool add \
  --resource-group $RESOURCE_GROUP \
  --cluster-name $CLUSTER_NAME \
  --name spotpool \
  --priority Spot \
  --eviction-policy Delete \
  --spot-max-price -1 \
  --node-count 0 \
  --min-count 0 \
  --max-count 10 \
  --node-vm-size Standard_D2s_v3 \
  --enable-cluster-autoscaler \
  --node-taints kubernetes.azure.com/scalesetpriority=spot:NoSchedule \
  --labels cost-optimization=enabled workload-type=batch

# Create GPU node pool for AI workloads (optional)
az aks nodepool add \
  --resource-group $RESOURCE_GROUP \
  --cluster-name $CLUSTER_NAME \
  --name gpupool \
  --node-count 0 \
  --min-count 0 \
  --max-count 3 \
  --node-vm-size Standard_NC6s_v3 \
  --enable-cluster-autoscaler \
  --node-taints sku=gpu:NoSchedule \
  --labels workload-type=gpu accelerator=nvidia

# Get cluster credentials
az aks get-credentials --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME

# Verify cluster is running
kubectl cluster-info
kubectl get nodes
```

### Configure Azure Container Registry

```bash
# Create Azure Container Registry
az acr create \
  --resource-group $RESOURCE_GROUP \
  --name genopsacr$(date +%s) \
  --sku Premium \
  --admin-enabled true

# Attach ACR to AKS cluster
az aks update \
  --resource-group $RESOURCE_GROUP \
  --name $CLUSTER_NAME \
  --attach-acr genopsacr$(date +%s)
```

## GenOps Deployment

### Prepare GenOps Configuration

Create Azure-optimized GenOps configuration:

```bash
# Create GenOps namespace
kubectl create namespace genops-system

# Create Azure Key Vault for secrets
az keyvault create \
  --resource-group $RESOURCE_GROUP \
  --name genops-keyvault-$(date +%s) \
  --location $LOCATION \
  --enable-rbac-authorization true

# Store API keys in Key Vault
az keyvault secret set \
  --vault-name genops-keyvault-$(date +%s) \
  --name openai-api-key \
  --value "your-openai-key"

az keyvault secret set \
  --vault-name genops-keyvault-$(date +%s) \
  --name anthropic-api-key \
  --value "your-anthropic-key"

# Create Azure-specific configuration
cat > genops-azure-values.yaml << 'EOF'
# GenOps AI Helm Chart Values for Azure AKS

# Global configuration
global:
  environment: production
  cloud:
    provider: azure
    subscriptionId: SUBSCRIPTION_ID_PLACEHOLDER
    resourceGroup: genops-production-rg
    location: eastus
  governance:
    team: platform-engineering
    project: genops-deployment
    cost_center: engineering

# Core GenOps services
genops:
  image:
    repository: genopsacr.azurecr.io/genops
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
  
  # Azure-specific configuration
  azure:
    subscriptionId: SUBSCRIPTION_ID_PLACEHOLDER
    resourceGroup: genops-production-rg
    location: eastus
    enableCostOptimization: true
    enableAzureOpenAI: true
    enableApplicationInsights: true
    
    # Cost management
    costManagement:
      enabled: true
      budgetAlerts: true
      
    # Storage configuration
    storage:
      accountName: genopsstorage
      containerName: governance-data
      
    # Key Vault integration
    keyVault:
      name: KEYVAULT_NAME_PLACEHOLDER
      resourceGroup: genops-production-rg

# Proxy service for AI workloads
proxy:
  enabled: true
  replicas: 2
  
  service:
    type: LoadBalancer
    annotations:
      service.beta.kubernetes.io/azure-load-balancer-internal: "true"
      service.beta.kubernetes.io/azure-load-balancer-resource-group: genops-production-rg
  
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
    azureopenai:
      enabled: true
      endpoint: https://genops-openai.openai.azure.com/
      apiVersion: "2023-12-01-preview"

# Observability stack
observability:
  # Azure Monitor integration
  azureMonitor:
    enabled: true
    workspaceId: WORKSPACE_ID_PLACEHOLDER
    
  # Application Insights for tracing
  applicationInsights:
    enabled: true
    instrumentationKey: APPINSIGHTS_KEY_PLACEHOLDER
    
  # Prometheus for metrics
  prometheus:
    enabled: true
    retention: 30d
    storage:
      class: managed-premium
      size: 100Gi
      
  # Grafana for dashboards
  grafana:
    enabled: true
    adminPassword: "change-me-in-production"
    dashboards:
      azure: true
      cost: true
      performance: true

# Storage configuration
storage:
  class: managed-premium
  size: 50Gi
  
# Security configuration
security:
  podSecurityPolicy: true
  networkPolicies: true
  aadIntegration: true
  
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
  spotInstances: true
  nodeAffinity: true
  resourceOptimization: true
  
  # Scheduled scaling for cost savings
  schedule:
    enabled: true
    scaleDown:
      schedule: "0 18 * * *"
      replicas: 1
    scaleUp:
      schedule: "0 8 * * *" 
      replicas: 3
EOF

# Replace placeholders
sed -i "s/SUBSCRIPTION_ID_PLACEHOLDER/$SUBSCRIPTION_ID/g" genops-azure-values.yaml

# Get workspace ID and App Insights key
WORKSPACE_ID=$(az monitor log-analytics workspace show --resource-group $RESOURCE_GROUP --workspace-name genops-analytics --query customerId -o tsv)
sed -i "s/WORKSPACE_ID_PLACEHOLDER/$WORKSPACE_ID/g" genops-azure-values.yaml

KEYVAULT_NAME=$(az keyvault list --resource-group $RESOURCE_GROUP --query '[0].name' -o tsv)
sed -i "s/KEYVAULT_NAME_PLACEHOLDER/$KEYVAULT_NAME/g" genops-azure-values.yaml
```

### Deploy GenOps with Helm

```bash
# Add GenOps Helm repository
helm repo add genops https://charts.genops.ai
helm repo update

# Install GenOps AI
helm install genops-ai genops/genops-ai \
  --namespace genops-system \
  --values genops-azure-values.yaml \
  --wait \
  --timeout 10m

# Verify deployment
kubectl get pods -n genops-system
kubectl get services -n genops-system

# Check logs
kubectl logs -n genops-system deployment/genops-ai --tail=100
```

### Configure Azure AD Integration

Set up Azure AD integration for secure authentication:

```bash
# Enable Azure AD integration on AKS
az aks update \
  --resource-group $RESOURCE_GROUP \
  --name $CLUSTER_NAME \
  --enable-aad \
  --aad-admin-group-object-ids $(az ad group show --group "AKS Admins" --query objectId -o tsv) \
  --enable-azure-rbac

# Create Azure AD application for GenOps
az ad app create \
  --display-name "GenOps AI Application" \
  --reply-urls "https://genops.company.com/auth/callback" \
  --required-resource-accesses '[{"resourceAppId":"00000003-0000-0000-c000-000000000000","resourceAccess":[{"id":"e1fe6dd8-ba31-4d61-89e7-88639da4683d","type":"Scope"}]}]'

# Create service principal
az ad sp create --id $(az ad app show --id "GenOps AI Application" --query appId -o tsv)
```

## Azure Service Integrations

### Azure OpenAI Integration

Configure GenOps to work with Azure OpenAI:

```bash
# Create Azure OpenAI service
az cognitiveservices account create \
  --resource-group $RESOURCE_GROUP \
  --name genops-openai \
  --location eastus \
  --kind OpenAI \
  --sku S0 \
  --custom-domain genops-openai

# Deploy models
az cognitiveservices account deployment create \
  --resource-group $RESOURCE_GROUP \
  --name genops-openai \
  --deployment-name gpt-35-turbo \
  --model-name gpt-35-turbo \
  --model-version "0613" \
  --model-format OpenAI \
  --scale-type Standard \
  --capacity 120

az cognitiveservices account deployment create \
  --resource-group $RESOURCE_GROUP \
  --name genops-openai \
  --deployment-name gpt-4 \
  --model-name gpt-4 \
  --model-version "0613" \
  --model-format OpenAI \
  --scale-type Standard \
  --capacity 10

# Create Azure OpenAI configuration
cat > azure-openai-integration.yaml << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: azure-openai-config
  namespace: genops-system
data:
  config.yaml: |
    azure_openai:
      endpoint: https://genops-openai.openai.azure.com/
      api_version: "2023-12-01-preview"
      models:
        - name: gpt-35-turbo
          deployment_name: gpt-35-turbo
          cost_per_1k_input: 0.0015
          cost_per_1k_output: 0.002
        - name: gpt-4
          deployment_name: gpt-4
          cost_per_1k_input: 0.03
          cost_per_1k_output: 0.06
      governance:
        enable_cost_tracking: true
        enable_content_filtering: true
        enable_budget_limits: true
        enable_usage_quotas: true
EOF

kubectl apply -f azure-openai-integration.yaml
```

### Azure Monitor Integration

Configure comprehensive Azure Monitor integration:

```bash
# Install Azure Monitor for containers
kubectl apply -f https://raw.githubusercontent.com/Microsoft/OMS-docker/ci_feature_prod/Kubernetes/container-azm-ms-agentconfig.yaml

# Create custom metrics for GenOps
cat > genops-azure-metrics.yaml << 'EOF'
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
        dimensions:
          - provider
          - model
          - team
          - project
      
      - name: genops_cost_per_hour
        type: gauge
        description: Cost per hour by team/project
        dimensions:
          - team
          - project
          - cost_center
      
      - name: genops_policy_violations_total
        type: counter
        description: Total policy violations
        dimensions:
          - policy_type
          - severity
      
      - name: genops_budget_utilization
        type: gauge
        description: Budget utilization percentage
        dimensions:
          - budget_name
          - team
    
    export_settings:
      azure_monitor:
        enabled: true
        workspace_id: WORKSPACE_ID_PLACEHOLDER
        interval: 60s
        namespace: GenOps
EOF

sed -i "s/WORKSPACE_ID_PLACEHOLDER/$WORKSPACE_ID/g" genops-azure-metrics.yaml
kubectl apply -f genops-azure-metrics.yaml
```

### Azure Storage Integration

Set up Azure Blob Storage for governance data:

```bash
# Create storage account
az storage account create \
  --resource-group $RESOURCE_GROUP \
  --name genopsstorage$(date +%s) \
  --location $LOCATION \
  --sku Standard_LRS \
  --kind StorageV2 \
  --access-tier Hot

# Create container for governance data
az storage container create \
  --account-name genopsstorage$(date +%s) \
  --name governance-data \
  --public-access off

# Configure GenOps to use Azure Storage
cat > azure-storage-config.yaml << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: azure-storage-config
  namespace: genops-system
data:
  storage.yaml: |
    azure_storage:
      enabled: true
      account_name: STORAGE_ACCOUNT_PLACEHOLDER
      container_name: governance-data
      
    backup:
      enabled: true
      schedule: "0 2 * * *"
      retention_days: 30
      
    export:
      cost_data: true
      policy_logs: true
      audit_trails: true
      performance_metrics: true
EOF

STORAGE_ACCOUNT=$(az storage account list --resource-group $RESOURCE_GROUP --query '[0].name' -o tsv)
sed -i "s/STORAGE_ACCOUNT_PLACEHOLDER/$STORAGE_ACCOUNT/g" azure-storage-config.yaml
kubectl apply -f azure-storage-config.yaml
```

### Azure Cost Management Integration

Set up automated cost tracking and budgets:

```bash
# Create cost management budget
az consumption budget create \
  --budget-name "GenOps-AKS-Monthly" \
  --amount 1000 \
  --time-grain Monthly \
  --time-period start-date=$(date -d "first day of this month" +%Y-%m-01) \
  --category Cost \
  --filter resourceGroupName=$RESOURCE_GROUP \
  --notifications '[{
    "enabled": true,
    "operator": "GreaterThanOrEqualTo",
    "threshold": 80,
    "contactEmails": ["platform-team@company.com"],
    "contactGroups": [],
    "contactRoles": ["Owner"]
  }]'

# Configure cost export
az costmanagement export create \
  --name genops-cost-export \
  --type Usage \
  --scope /subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP \
  --storage-account-id $(az storage account show --name $STORAGE_ACCOUNT --resource-group $RESOURCE_GROUP --query id -o tsv) \
  --storage-container governance-data \
  --directory-path cost-exports \
  --time-frame MonthToDate \
  --recurrence Daily
```

## Cost Management

### Node Pool Optimization

Create cost-optimized node pools:

```bash
# Create additional Spot VM node pool with different VM sizes
az aks nodepool add \
  --resource-group $RESOURCE_GROUP \
  --cluster-name $CLUSTER_NAME \
  --name spotpool2 \
  --priority Spot \
  --eviction-policy Delete \
  --spot-max-price 0.5 \
  --node-count 0 \
  --min-count 0 \
  --max-count 15 \
  --node-vm-size Standard_B2s \
  --enable-cluster-autoscaler \
  --node-taints kubernetes.azure.com/scalesetpriority=spot:NoSchedule \
  --labels cost-optimization=enabled workload-type=burstable

# Create mixed mode node pool
az aks nodepool add \
  --resource-group $RESOURCE_GROUP \
  --cluster-name $CLUSTER_NAME \
  --name mixedpool \
  --node-count 2 \
  --min-count 1 \
  --max-count 8 \
  --node-vm-size Standard_D2s_v3 \
  --enable-cluster-autoscaler \
  --labels cost-optimization=mixed workload-type=general
```

### Cluster Autoscaler Configuration

Configure intelligent cluster autoscaling:

```bash
# Update cluster autoscaler settings
kubectl create configmap cluster-autoscaler-status \
  --from-literal=nodes.max=50 \
  --from-literal=nodes.min=3 \
  --from-literal=scale-down-enabled=true \
  --from-literal=scale-down-delay-after-add=10m \
  --from-literal=scale-down-unneeded-time=10m \
  --from-literal=skip-nodes-with-local-storage=false \
  --from-literal=skip-nodes-with-system-pods=false \
  --from-literal=balance-similar-node-groups=true \
  --from-literal=expander=least-waste \
  --namespace kube-system

# Apply autoscaler configuration
kubectl patch deployment cluster-autoscaler \
  --namespace kube-system \
  --patch='{"spec":{"template":{"spec":{"containers":[{"name":"cluster-autoscaler","command":["./cluster-autoscaler","--v=4","--stderrthreshold=info","--cloud-provider=azure","--skip-nodes-with-local-storage=false","--expander=least-waste","--node-group-auto-discovery=asg:tag=k8s-io-cluster-autoscaler-enabled","--balance-similar-node-groups","--skip-nodes-with-system-pods=false","--scale-down-enabled=true","--scale-down-delay-after-add=10m","--scale-down-unneeded-time=10m"]}]}}}}'
```

### Cost Monitoring Dashboard

Create comprehensive cost monitoring:

```bash
# Create cost monitoring namespace
kubectl create namespace cost-monitoring

# Deploy Azure cost exporter
cat > azure-cost-exporter.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: azure-cost-exporter
  namespace: cost-monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: azure-cost-exporter
  template:
    metadata:
      labels:
        app: azure-cost-exporter
    spec:
      containers:
      - name: azure-cost-exporter
        image: genopsai/azure-cost-exporter:latest
        env:
        - name: AZURE_SUBSCRIPTION_ID
          value: SUBSCRIPTION_ID_PLACEHOLDER
        - name: AZURE_RESOURCE_GROUP
          value: genops-production-rg
        - name: CLUSTER_NAME
          value: genops-production
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
  name: azure-cost-exporter
  namespace: cost-monitoring
  labels:
    app: azure-cost-exporter
spec:
  ports:
  - port: 9090
    targetPort: 9090
  selector:
    app: azure-cost-exporter
EOF

sed -i "s/SUBSCRIPTION_ID_PLACEHOLDER/$SUBSCRIPTION_ID/g" azure-cost-exporter.yaml
kubectl apply -f azure-cost-exporter.yaml
```

### Automated Cost Controls

Create automated cost management:

```bash
# Create Azure Function for cost control
cat > cost-control-function.cs << 'EOF'
using System;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.Management.ContainerService;
using Microsoft.Azure.Management.CostManagement;
using Microsoft.Extensions.Logging;

public static class CostControlFunction
{
    [FunctionName("CostControl")]
    public static async Task Run(
        [TimerTrigger("0 */15 * * * *")] TimerInfo myTimer,
        ILogger log)
    {
        log.LogInformation($"Cost control function executed at: {DateTime.Now}");
        
        var costManagementClient = new CostManagementClient(credentials);
        var containerServiceClient = new ContainerServiceClient(credentials);
        
        // Get current costs
        var costData = await GetCurrentCosts(costManagementClient);
        var budgetUtilization = costData.CurrentSpend / costData.BudgetLimit;
        
        if (budgetUtilization > 0.8)
        {
            log.LogWarning($"Budget utilization: {budgetUtilization:P}. Scaling down non-critical workloads.");
            
            // Scale down spot instance node pools
            await ScaleNodePool(containerServiceClient, "spotpool", 0);
            await ScaleNodePool(containerServiceClient, "spotpool2", 0);
            
            // Send alert
            await SendCostAlert(budgetUtilization);
        }
        else if (budgetUtilization < 0.5)
        {
            // Scale up if under-utilized and demand exists
            await OptimizeNodePools(containerServiceClient);
        }
    }
}
EOF

# Deploy function (requires Azure Functions Core Tools)
# func azure functionapp publish genops-cost-control
```

## Security & Compliance

### Azure AD Integration and RBAC

Configure comprehensive Azure AD integration:

```bash
# Create Azure AD groups for RBAC
az ad group create \
  --display-name "GenOps-Admins" \
  --mail-nickname genops-admins

az ad group create \
  --display-name "GenOps-Users" \
  --mail-nickname genops-users

az ad group create \
  --display-name "GenOps-Viewers" \
  --mail-nickname genops-viewers

# Create Kubernetes RBAC configuration
cat > genops-rbac.yaml << 'EOF'
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: genops-admin
rules:
- apiGroups: ["*"]
  resources: ["*"]
  verbs: ["*"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: genops-user
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "secrets"]
  verbs: ["get", "list", "create", "update", "patch"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "create", "update", "patch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: genops-viewer
rules:
- apiGroups: ["*"]
  resources: ["*"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: genops-admins
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: genops-admin
subjects:
- kind: Group
  name: "GENOPS_ADMINS_GROUP_ID"
  apiGroup: rbac.authorization.k8s.io
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: genops-users
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: genops-user
subjects:
- kind: Group
  name: "GENOPS_USERS_GROUP_ID"
  apiGroup: rbac.authorization.k8s.io
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: genops-viewers
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: genops-viewer
subjects:
- kind: Group
  name: "GENOPS_VIEWERS_GROUP_ID"
  apiGroup: rbac.authorization.k8s.io
EOF

# Get group IDs and apply RBAC
ADMINS_GROUP_ID=$(az ad group show --group "GenOps-Admins" --query objectId -o tsv)
USERS_GROUP_ID=$(az ad group show --group "GenOps-Users" --query objectId -o tsv)
VIEWERS_GROUP_ID=$(az ad group show --group "GenOps-Viewers" --query objectId -o tsv)

sed -i "s/GENOPS_ADMINS_GROUP_ID/$ADMINS_GROUP_ID/g" genops-rbac.yaml
sed -i "s/GENOPS_USERS_GROUP_ID/$USERS_GROUP_ID/g" genops-rbac.yaml
sed -i "s/GENOPS_VIEWERS_GROUP_ID/$VIEWERS_GROUP_ID/g" genops-rbac.yaml

kubectl apply -f genops-rbac.yaml
```

### Network Security Policies

Configure comprehensive network security:

```bash
# Create network security policies
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
      port: 443  # HTTPS to Azure APIs
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

### Pod Security Standards and Azure Policy

Implement pod security standards:

```bash
# Apply pod security standards
kubectl label namespace genops-system \
  pod-security.kubernetes.io/enforce=restricted \
  pod-security.kubernetes.io/audit=restricted \
  pod-security.kubernetes.io/warn=restricted

# Create Azure Policy for AKS governance
az policy definition create \
  --name "GenOps-Pod-Security-Policy" \
  --description "Enforce security policies for GenOps pods" \
  --rules '{
    "if": {
      "allOf": [
        {
          "field": "type",
          "equals": "Microsoft.ContainerService/managedClusters/pods"
        },
        {
          "field": "Microsoft.ContainerService/managedClusters/pods/namespace",
          "equals": "genops-system"
        }
      ]
    },
    "then": {
      "effect": "audit"
    }
  }' \
  --params '{
    "allowedImages": {
      "type": "Array",
      "defaultValue": ["genopsai/*", "mcr.microsoft.com/*"]
    }
  }'

# Assign policy to resource group
az policy assignment create \
  --name "genops-security-policy" \
  --policy "GenOps-Pod-Security-Policy" \
  --scope /subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP
```

### Key Vault Integration

Configure secure secrets management:

```bash
# Install Azure Key Vault CSI driver
kubectl apply -f https://raw.githubusercontent.com/Azure/secrets-store-csi-driver-provider-azure/master/deployment/provider-azure-installer.yaml

# Create SecretProviderClass for GenOps secrets
cat > genops-secret-provider.yaml << 'EOF'
apiVersion: secrets-store.csi.x-k8s.io/v1
kind: SecretProviderClass
metadata:
  name: genops-secrets
  namespace: genops-system
spec:
  provider: azure
  parameters:
    usePodIdentity: "false"
    useVMManagedIdentity: "true"
    userAssignedIdentityID: ""
    keyvaultName: KEYVAULT_NAME_PLACEHOLDER
    objects:  |
      array:
        - |
          objectName: openai-api-key
          objectType: secret
          objectVersion: ""
        - |
          objectName: anthropic-api-key
          objectType: secret
          objectVersion: ""
    tenantId: TENANT_ID_PLACEHOLDER
EOF

TENANT_ID=$(az account show --query tenantId -o tsv)
sed -i "s/KEYVAULT_NAME_PLACEHOLDER/$KEYVAULT_NAME/g" genops-secret-provider.yaml
sed -i "s/TENANT_ID_PLACEHOLDER/$TENANT_ID/g" genops-secret-provider.yaml

kubectl apply -f genops-secret-provider.yaml

# Update GenOps deployment to use Key Vault secrets
kubectl patch deployment genops-ai \
  --namespace genops-system \
  --patch='{"spec":{"template":{"spec":{"volumes":[{"name":"secrets-store-inline","csi":{"driver":"secrets-store.csi.k8s.io","readOnly":true,"volumeAttributes":{"secretProviderClass":"genops-secrets"}}}],"containers":[{"name":"genops-ai","volumeMounts":[{"name":"secrets-store-inline","mountPath":"/mnt/secrets-store","readOnly":true}]}]}}}}'
```

## Monitoring & Observability

### Comprehensive Monitoring Stack

Deploy full observability stack for GenOps:

```bash
# Create monitoring namespace
kubectl create namespace monitoring

# Install Prometheus and Grafana
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --set grafana.adminPassword=admin \
  --set prometheus.prometheusSpec.retention=30d \
  --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=100Gi

# Configure Azure Monitor integration
kubectl apply -f - << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: azure-monitor-config
  namespace: monitoring
data:
  config.yaml: |
    azure_monitor:
      workspace_id: WORKSPACE_ID_PLACEHOLDER
      workspace_key: WORKSPACE_KEY_PLACEHOLDER
      
    log_analytics:
      enabled: true
      custom_logs: true
      
    application_insights:
      enabled: true
      instrumentation_key: APPINSIGHTS_KEY_PLACEHOLDER
EOF
```

### Custom Dashboards

Create GenOps-specific Azure dashboards:

```bash
# Create Azure dashboard definition
cat > genops-azure-dashboard.json << 'EOF'
{
  "lenses": {
    "0": {
      "order": 0,
      "parts": {
        "0": {
          "position": {"x": 0, "y": 0, "rowSpan": 4, "colSpan": 6},
          "metadata": {
            "inputs": [{
              "name": "chartType",
              "value": "Line"
            }, {
              "name": "metrics",
              "value": [{
                "resourceMetadata": {
                  "id": "/subscriptions/SUBSCRIPTION_ID/resourceGroups/RESOURCE_GROUP/providers/Microsoft.ContainerService/managedClusters/CLUSTER_NAME"
                },
                "name": "kube_pod_status_ready",
                "aggregationType": {
                  "displayName": "Average"
                },
                "namespace": "insights.container/pods",
                "metricVisualization": {
                  "displayName": "Ready Pods"
                }
              }]
            }],
            "type": "Extension/HubsExtension/PartType/MonitorChartPart"
          }
        },
        "1": {
          "position": {"x": 6, "y": 0, "rowSpan": 4, "colSpan": 6},
          "metadata": {
            "inputs": [{
              "name": "chartType",
              "value": "Line"
            }, {
              "name": "metrics",
              "value": [{
                "resourceMetadata": {
                  "id": "/subscriptions/SUBSCRIPTION_ID/resourceGroups/RESOURCE_GROUP"
                },
                "name": "genops_cost_per_hour",
                "aggregationType": {
                  "displayName": "Sum"
                },
                "namespace": "GenOps/Custom",
                "metricVisualization": {
                  "displayName": "Cost per Hour"
                }
              }]
            }],
            "type": "Extension/HubsExtension/PartType/MonitorChartPart"
          }
        }
      }
    }
  },
  "metadata": {
    "model": {
      "timeRange": {
        "value": {
          "relative": {
            "duration": 24,
            "timeUnit": 1
          }
        },
        "type": "MsPortalFx.Composition.Configuration.ValueTypes.TimeRange"
      }
    }
  },
  "name": "GenOps AI - Azure Dashboard",
  "type": "Microsoft.Portal/dashboards",
  "location": "INSERT_LOCATION",
  "tags": {
    "hidden-title": "GenOps AI - Azure Dashboard"
  }
}
EOF

# Create the dashboard
az portal dashboard create \
  --resource-group $RESOURCE_GROUP \
  --name "genops-dashboard" \
  --input-path genops-azure-dashboard.json
```

### Application Insights Integration

Configure distributed tracing with Application Insights:

```bash
# Create Application Insights instance
az monitor app-insights component create \
  --resource-group $RESOURCE_GROUP \
  --app genops-insights \
  --location $LOCATION \
  --kind web

# Get instrumentation key
APPINSIGHTS_KEY=$(az monitor app-insights component show \
  --resource-group $RESOURCE_GROUP \
  --app genops-insights \
  --query instrumentationKey -o tsv)

# Configure Application Insights integration
kubectl apply -f - << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: application-insights-config
  namespace: genops-system
data:
  appinsights.yaml: |
    application_insights:
      enabled: true
      instrumentation_key: APPINSIGHTS_KEY_PLACEHOLDER
      
    telemetry:
      sampling_rate: 0.1
      auto_collect:
        requests: true
        dependencies: true
        exceptions: true
        performance_counters: true
        
    custom_events:
      ai_requests: true
      cost_tracking: true
      policy_violations: true
      budget_alerts: true
EOF

sed -i "s/APPINSIGHTS_KEY_PLACEHOLDER/$APPINSIGHTS_KEY/g" /tmp/appinsights-config.yaml
kubectl apply -f /tmp/appinsights-config.yaml
```

## Production Optimizations

### High Availability Configuration

Configure GenOps for high availability:

```bash
# Configure multi-zone deployment
kubectl patch deployment genops-ai \
  -n genops-system \
  -p='{"spec":{"replicas":3,"template":{"spec":{"affinity":{"podAntiAffinity":{"requiredDuringSchedulingIgnoredDuringExecution":[{"labelSelector":{"matchExpressions":[{"key":"app","operator":"In","values":["genops-ai"]}]},"topologyKey":"topology.kubernetes.io/zone"}]}}}}}}'

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
# Horizontal Pod Autoscaler with custom metrics
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
        name: azure_monitor_genops_requests_per_second
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
# Install Velero with Azure plugin
helm repo add vmware-tanzu https://vmware-tanzu.github.io/helm-charts/
helm install velero vmware-tanzu/velero \
  --namespace velero \
  --create-namespace \
  --set configuration.provider=azure \
  --set configuration.backupStorageLocation.name=azure \
  --set configuration.backupStorageLocation.bucket=$STORAGE_ACCOUNT \
  --set configuration.backupStorageLocation.config.resourceGroup=$RESOURCE_GROUP \
  --set configuration.backupStorageLocation.config.storageAccount=$STORAGE_ACCOUNT \
  --set snapshotsEnabled=true \
  --set configuration.volumeSnapshotLocation.name=azure \
  --set configuration.volumeSnapshotLocation.config.resourceGroup=$RESOURCE_GROUP

# Create backup schedule
kubectl apply -f - << 'EOF'
apiVersion: velero.io/v1
kind: Schedule
metadata:
  name: genops-daily-backup
  namespace: velero
spec:
  schedule: "0 2 * * *"
  template:
    includedNamespaces:
    - genops-system
    - monitoring
    storageLocation: azure
    volumeSnapshotLocations:
    - azure
    ttl: 720h0m0s
EOF
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: Pods Stuck in Pending State

**Diagnosis:**
```bash
kubectl describe pod <pod-name> -n genops-system
kubectl get events -n genops-system --sort-by=.metadata.creationTimestamp
az aks show --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME
```

**Solutions:**
1. **Node Pool Capacity:**
   ```bash
   # Check node pool status
   az aks nodepool list --resource-group $RESOURCE_GROUP --cluster-name $CLUSTER_NAME
   
   # Scale up node pool
   az aks nodepool scale \
     --resource-group $RESOURCE_GROUP \
     --cluster-name $CLUSTER_NAME \
     --name nodepool1 \
     --node-count 5
   ```

2. **Spot VM Evictions:**
   ```bash
   # Check Spot VM events
   kubectl get events --field-selector reason=Evicted
   
   # Add regular node pool as fallback
   az aks nodepool add \
     --resource-group $RESOURCE_GROUP \
     --cluster-name $CLUSTER_NAME \
     --name regularpool \
     --node-count 2 \
     --node-vm-size Standard_D2s_v3
   ```

#### Issue: High Azure Costs

**Diagnosis:**
```bash
# Check current costs
az consumption usage list \
  --start-date $(date -d '30 days ago' +%Y-%m-%d) \
  --end-date $(date +%Y-%m-%d) \
  --output table

# Check AKS cluster costs
az aks show --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME --query agentPoolProfiles
```

**Solutions:**
1. **Enable Spot VMs:**
   ```bash
   # Migrate workloads to spot nodes
   kubectl patch deployment genops-ai -n genops-system --patch '{"spec":{"template":{"spec":{"tolerations":[{"key":"kubernetes.azure.com/scalesetpriority","operator":"Equal","value":"spot","effect":"NoSchedule"}],"nodeSelector":{"kubernetes.azure.com/scalesetpriority":"spot"}}}}}'
   ```

2. **Right-size Resources:**
   ```bash
   # Check resource usage
   kubectl top pods -n genops-system
   
   # Update resource limits
   kubectl patch deployment genops-ai -n genops-system --patch '{"spec":{"template":{"spec":{"containers":[{"name":"genops-ai","resources":{"requests":{"cpu":"100m","memory":"256Mi"},"limits":{"cpu":"300m","memory":"512Mi"}}}]}}}}'
   ```

#### Issue: Azure OpenAI Connection Problems

**Diagnosis:**
```bash
# Test Azure OpenAI connectivity
az cognitiveservices account show \
  --resource-group $RESOURCE_GROUP \
  --name genops-openai

# Check deployment status
az cognitiveservices account deployment list \
  --resource-group $RESOURCE_GROUP \
  --name genops-openai

# Check logs
kubectl logs -n genops-system deployment/genops-ai | grep -i openai
```

**Solutions:**
1. **Fix Authentication:**
   ```bash
   # Update managed identity permissions
   az role assignment create \
     --assignee $(az aks show --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME --query identityProfile.kubeletidentity.clientId -o tsv) \
     --role "Cognitive Services User" \
     --scope $(az cognitiveservices account show --resource-group $RESOURCE_GROUP --name genops-openai --query id -o tsv)
   ```

2. **Update Configuration:**
   ```bash
   # Update endpoint configuration
   kubectl patch configmap azure-openai-config -n genops-system --patch '{"data":{"config.yaml":"azure_openai:\n  endpoint: https://genops-openai.openai.azure.com/\n  api_version: \"2023-12-01-preview\""}}'
   
   # Restart deployment
   kubectl rollout restart deployment/genops-ai -n genops-system
   ```

### Health Checks and Validation

```bash
# Comprehensive health check script
cat > health-check-azure.sh << 'EOF'
#!/bin/bash
echo "🔍 GenOps AKS Health Check"
echo "=========================="

# Check cluster health
echo "📋 Cluster Status:"
az aks show --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME --query provisioningState -o tsv
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

# Check Azure integrations
echo -e "\n☁️ Azure Integration:"
az cognitiveservices account show --resource-group $RESOURCE_GROUP --name genops-openai --query provisioningState -o tsv
az storage account show --resource-group $RESOURCE_GROUP --name $STORAGE_ACCOUNT --query provisioningState -o tsv

# Check cost tracking
echo -e "\n💰 Cost Tracking:"
az consumption usage list --output table --max-items 5

echo -e "\n✅ Health check complete"
EOF

chmod +x health-check-azure.sh
./health-check-azure.sh
```

### Performance Optimization

```bash
# Enable Azure performance features
az aks update \
  --resource-group $RESOURCE_GROUP \
  --name $CLUSTER_NAME \
  --enable-pod-identity \
  --enable-secret-rotation

# Configure performance settings
kubectl apply -f - << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: genops-performance-config
  namespace: genops-system
data:
  performance.yaml: |
    azure:
      optimization:
        connection_pooling: true
        request_batching: true
        cache_enabled: true
        cache_ttl: 300s
        
      azure_openai:
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

1. **Set up advanced monitoring** with custom Azure Monitor workbooks
2. **Configure GitOps workflow** with Azure DevOps and ArgoCD  
3. **Enable multi-region deployment** for global availability
4. **Optimize costs** with Azure Reserved Instances and Spot VMs
5. **Implement advanced security** with Azure Policy and Azure Security Center

## Additional Resources

- [AKS Best Practices Guide](https://docs.microsoft.com/en-us/azure/aks/best-practices)
- [GenOps AI Documentation](https://docs.genops.ai)
- [Azure Kubernetes Service Cost Optimization](https://docs.microsoft.com/en-us/azure/aks/concepts-sustainable-software-engineering)
- [Azure Cost Management](https://docs.microsoft.com/en-us/azure/cost-management-billing/)

This guide provides a comprehensive foundation for deploying GenOps AI on Azure Kubernetes Service with production-ready configurations, cost optimization, and enterprise security.