# Multi-Cloud Kubernetes Deployment for GenOps AI

Complete guide for deploying GenOps AI across multiple cloud providers with unified governance tracking, cost optimization, and operational excellence.

## Table of Contents

1. [Quick Start (5 minutes)](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Prerequisites](#prerequisites)
4. [Federation Setup](#federation-setup)
5. [Deployment Strategies](#deployment-strategies)
6. [Network Connectivity](#network-connectivity)
7. [Cost Optimization](#cost-optimization)
8. [Governance & Compliance](#governance-compliance)
9. [Migration Scenarios](#migration-scenarios)
10. [Operational Excellence](#operational-excellence)
11. [Troubleshooting](#troubleshooting)

## Quick Start

Deploy GenOps AI across AWS and Azure in 5 minutes:

```bash
# 1. Install multi-cloud CLI tools
./scripts/install-multicloud-tools.sh

# 2. Deploy to primary cloud (AWS)
kubectl config use-context aws-primary
helm install genops-ai genops/genops-ai \
  --set cloud.provider=aws \
  --set multicloud.enabled=true \
  --set multicloud.federation=true

# 3. Deploy to secondary cloud (Azure)
kubectl config use-context azure-secondary
helm install genops-ai genops/genops-ai \
  --set cloud.provider=azure \
  --set multicloud.enabled=true \
  --set multicloud.primary=false \
  --set multicloud.primaryEndpoint=https://genops.aws.example.com
```

✅ **Result:** GenOps AI running on both AWS and Azure with unified governance tracking.

## Architecture Overview

### Multi-Cloud Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                   Multi-Cloud Control Plane                     │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │         KubeFed / Multi-Cluster Service Mesh               │ │
│  │         - Unified Service Discovery                        │ │
│  │         - Cross-Cloud Load Balancing                       │ │
│  │         - Global Traffic Management                        │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼────────┐    ┌───────▼────────┐    ┌───────▼────────┐
│   AWS EKS      │    │   Azure AKS    │    │   GCP GKE      │
│                │    │                │    │                │
│ ┌────────────┐ │    │ ┌────────────┐ │    │ ┌────────────┐ │
│ │ GenOps AI  │ │    │ │ GenOps AI  │ │    │ │ GenOps AI  │ │
│ │  - Core    │ │    │ │  - Core    │ │    │ │  - Core    │ │
│ │  - Proxy   │ │    │ │  - Proxy   │ │    │ │  - Proxy   │ │
│ └────────────┘ │    │ └────────────┘ │    │ └────────────┘ │
│                │    │                │    │                │
│ ┌────────────┐ │    │ ┌────────────┐ │    │ ┌────────────┐ │
│ │ AI Workload│ │    │ │ AI Workload│ │    │ │ AI Workload│ │
│ │ - OpenAI   │ │    │ │ - Azure AI │ │    │ │ - Vertex AI│ │
│ │ - Bedrock  │ │    │ │ - OpenAI   │ │    │ │ - OpenAI   │ │
│ └────────────┘ │    │ └────────────┘ │    │ └────────────┘ │
└────────────────┘    └────────────────┘    └────────────────┘
        │                     │                     │
┌───────▼──────────────────────▼──────────────────────▼───────┐
│            Unified Observability & Governance               │
│                                                              │
│  - Cross-Cloud Cost Aggregation                            │
│  - Unified Policy Enforcement                              │
│  - Global Budget Management                                │
│  - Multi-Provider Telemetry                                │
└──────────────────────────────────────────────────────────────┘
```

### Key Components

- **Federation Layer**: KubeFed or multi-cluster service mesh for unified control
- **Cloud-Specific Clusters**: Native Kubernetes services (EKS, AKS, GKE)
- **GenOps Control Plane**: Centralized governance and cost tracking
- **Cross-Cloud Networking**: VPN mesh, VPC peering, or transit gateways
- **Unified Observability**: Centralized metrics, logs, and traces

## Prerequisites

### Required Cloud Access

**AWS Account:**
```bash
# Configure AWS credentials
aws configure set aws_access_key_id YOUR_AWS_KEY
aws configure set aws_secret_access_key YOUR_AWS_SECRET
aws configure set default.region us-west-2

# Verify access
aws sts get-caller-identity
aws eks list-clusters
```

**Azure Account:**
```bash
# Login to Azure
az login
az account set --subscription YOUR_SUBSCRIPTION_ID

# Verify access
az account show
az aks list
```

**GCP Account:**
```bash
# Authenticate with GCP
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Verify access
gcloud config list
gcloud container clusters list
```

### Required Tools

```bash
# Install multi-cloud CLI tools
cat > install-multicloud-tools.sh << 'EOF'
#!/bin/bash

echo "Installing multi-cloud Kubernetes tools..."

# kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# AWS eksctl
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# GCP gcloud
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

# kubefed (optional for federation)
curl -LO https://github.com/kubernetes-sigs/kubefed/releases/download/v0.10.0/kubefedctl-0.10.0-linux-amd64.tgz
tar -xzf kubefedctl-0.10.0-linux-amd64.tgz
sudo mv kubefedctl /usr/local/bin/

# kubectx/kubens for context switching
sudo git clone https://github.com/ahmetb/kubectx /opt/kubectx
sudo ln -s /opt/kubectx/kubectx /usr/local/bin/kubectx
sudo ln -s /opt/kubectx/kubens /usr/local/bin/kubens

echo "✅ Multi-cloud tools installed successfully"
EOF

chmod +x install-multicloud-tools.sh
./install-multicloud-tools.sh
```

### Network Requirements

**Cross-Cloud Connectivity Options:**
1. **VPN Mesh**: IPsec tunnels between clouds (low cost, moderate performance)
2. **Cloud Provider Peering**: AWS PrivateLink, Azure Private Link, GCP Private Service Connect
3. **Transit Gateway**: Centralized hub-and-spoke networking (AWS Transit Gateway, Azure Virtual WAN)
4. **SD-WAN**: Software-defined wide area network for complex topologies

## Federation Setup

### KubeFed Multi-Cluster Federation

Deploy Kubernetes Federation for unified multi-cloud management:

```bash
# Create host cluster context (where KubeFed will run)
kubectl config use-context aws-primary

# Install KubeFed
helm repo add kubefed-charts https://raw.githubusercontent.com/kubernetes-sigs/kubefed/master/charts
helm install kubefed kubefed-charts/kubefed \
  --namespace kube-federation-system \
  --create-namespace

# Wait for KubeFed to be ready
kubectl wait --for=condition=Ready pods --all -n kube-federation-system --timeout=300s

# Join AWS cluster
kubefedctl join aws-cluster \
  --cluster-context aws-primary \
  --host-cluster-context aws-primary \
  --v=2

# Join Azure cluster
kubefedctl join azure-cluster \
  --cluster-context azure-secondary \
  --host-cluster-context aws-primary \
  --v=2

# Join GCP cluster (optional)
kubefedctl join gcp-cluster \
  --cluster-context gcp-tertiary \
  --host-cluster-context aws-primary \
  --v=2

# Verify federation
kubectl -n kube-federation-system get kubefedclusters
```

### Federated GenOps Deployment

Create federated resources for multi-cloud GenOps deployment:

```yaml
# federated-genops-namespace.yaml
apiVersion: types.kubefed.io/v1beta1
kind: FederatedNamespace
metadata:
  name: genops-system
  namespace: genops-system
spec:
  placement:
    clusters:
    - name: aws-cluster
    - name: azure-cluster
    - name: gcp-cluster
---
# federated-genops-deployment.yaml
apiVersion: types.kubefed.io/v1beta1
kind: FederatedDeployment
metadata:
  name: genops-ai
  namespace: genops-system
spec:
  template:
    metadata:
      labels:
        app: genops-ai
        multicloud: "true"
    spec:
      replicas: 3
      selector:
        matchLabels:
          app: genops-ai
      template:
        metadata:
          labels:
            app: genops-ai
        spec:
          containers:
          - name: genops-ai
            image: genopsai/genops:1.0.0
            ports:
            - containerPort: 8080
            env:
            - name: GENOPS_CLOUD_PROVIDER
              value: "multicloud"
            - name: GENOPS_FEDERATION_ENABLED
              value: "true"
            resources:
              requests:
                cpu: 200m
                memory: 512Mi
              limits:
                cpu: 500m
                memory: 1Gi
  placement:
    clusters:
    - name: aws-cluster
    - name: azure-cluster
    - name: gcp-cluster
  overrides:
  - clusterName: aws-cluster
    clusterOverrides:
    - path: "/spec/template/spec/containers/0/env/-"
      value:
        name: CLOUD_PROVIDER_REGION
        value: us-west-2
  - clusterName: azure-cluster
    clusterOverrides:
    - path: "/spec/template/spec/containers/0/env/-"
      value:
        name: CLOUD_PROVIDER_REGION
        value: eastus
  - clusterName: gcp-cluster
    clusterOverrides:
    - path: "/spec/template/spec/containers/0/env/-"
      value:
        name: CLOUD_PROVIDER_REGION
        value: us-central1
```

Apply federated resources:

```bash
kubectl apply -f federated-genops-namespace.yaml
kubectl apply -f federated-genops-deployment.yaml

# Verify deployment across all clusters
for cluster in aws-cluster azure-cluster gcp-cluster; do
  echo "Checking $cluster:"
  kubectl --context $cluster get pods -n genops-system
done
```

## Deployment Strategies

### 1. Active-Active Multi-Cloud

Deploy GenOps AI across multiple clouds with load balancing:

```yaml
# active-active-genops.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: genops-multicloud-config
  namespace: genops-system
data:
  config.yaml: |
    multicloud:
      enabled: true
      strategy: active-active

      # Cloud provider configurations
      providers:
        aws:
          enabled: true
          region: us-west-2
          endpoint: https://genops-aws.example.com
          weight: 40  # 40% of traffic

        azure:
          enabled: true
          region: eastus
          endpoint: https://genops-azure.example.com
          weight: 40  # 40% of traffic

        gcp:
          enabled: true
          region: us-central1
          endpoint: https://genops-gcp.example.com
          weight: 20  # 20% of traffic

      # Load balancing configuration
      loadBalancing:
        algorithm: least-cost  # Options: round-robin, least-cost, geo-proximity
        healthCheck:
          enabled: true
          interval: 30s
          timeout: 5s

      # Failover configuration
      failover:
        enabled: true
        automaticFailover: true
        healthThreshold: 3  # Failed health checks before failover
```

### 2. Primary-Backup Configuration

Configure primary cloud with automatic failover to backup:

```yaml
# primary-backup-genops.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: genops-failover-config
  namespace: genops-system
data:
  config.yaml: |
    multicloud:
      enabled: true
      strategy: primary-backup

      primary:
        provider: aws
        region: us-west-2
        endpoint: https://genops-aws.example.com

      backup:
        provider: azure
        region: eastus
        endpoint: https://genops-azure.example.com

      failover:
        enabled: true
        automatic: true
        healthCheck:
          interval: 30s
          failureThreshold: 3
          successThreshold: 2
        switchback:
          automatic: false  # Manual switchback to primary
          cooldown: 300s
```

### 3. Geographic Load Distribution

Route traffic based on user geography for optimal latency:

```bash
# Create global load balancer with geo-routing
cat > geo-routing-policy.yaml << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: genops-geo-routing
  namespace: genops-system
data:
  routing.yaml: |
    geoRouting:
      enabled: true

      regions:
        # North America -> AWS US West
        - name: north-america
          countries: [US, CA, MX]
          targetCloud: aws
          targetRegion: us-west-2
          endpoint: https://genops-aws.example.com

        # Europe -> Azure West Europe
        - name: europe
          countries: [GB, FR, DE, IT, ES, NL, BE, SE, NO, DK]
          targetCloud: azure
          targetRegion: westeurope
          endpoint: https://genops-azure.example.com

        # Asia Pacific -> GCP Asia Southeast
        - name: asia-pacific
          countries: [JP, CN, KR, SG, AU, IN]
          targetCloud: gcp
          targetRegion: asia-southeast1
          endpoint: https://genops-gcp.example.com

      # Fallback for unmatched regions
      default:
        targetCloud: aws
        targetRegion: us-west-2
        endpoint: https://genops-aws.example.com
EOF

kubectl apply -f geo-routing-policy.yaml
```

### 4. Cost-Optimized Workload Placement

Automatically place workloads on the lowest-cost cloud provider:

```python
# cost-optimizer.py
#!/usr/bin/env python3
"""
Multi-cloud cost optimization for GenOps AI workloads.
Analyzes costs across providers and recommends optimal placement.
"""

import json
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class CloudCost:
    """Cloud provider cost information"""
    provider: str
    region: str
    compute_cost_per_hour: float
    storage_cost_per_gb: float
    network_egress_cost_per_gb: float
    ai_api_cost_multiplier: float  # 1.0 = baseline

# Define current pricing (update regularly)
CLOUD_COSTS = [
    CloudCost("aws", "us-west-2", 0.192, 0.023, 0.09, 1.0),
    CloudCost("azure", "eastus", 0.198, 0.025, 0.087, 1.05),
    CloudCost("gcp", "us-central1", 0.189, 0.020, 0.12, 0.95),
]

def calculate_workload_cost(
    cloud_cost: CloudCost,
    compute_hours: float,
    storage_gb: float,
    egress_gb: float,
    ai_requests: int
) -> float:
    """Calculate total cost for workload on specific cloud"""
    compute = compute_hours * cloud_cost.compute_cost_per_hour
    storage = storage_gb * cloud_cost.storage_cost_per_gb
    network = egress_gb * cloud_cost.network_egress_cost_per_gb
    ai_baseline = ai_requests * 0.002  # $0.002 per request baseline
    ai_cost = ai_baseline * cloud_cost.ai_api_cost_multiplier

    return compute + storage + network + ai_cost

def recommend_cloud_placement(
    compute_hours: float = 730,  # 1 month
    storage_gb: float = 100,
    egress_gb: float = 500,
    ai_requests: int = 100000
) -> Dict:
    """Recommend optimal cloud placement based on cost"""

    costs = []
    for cloud in CLOUD_COSTS:
        total_cost = calculate_workload_cost(
            cloud, compute_hours, storage_gb, egress_gb, ai_requests
        )
        costs.append({
            "provider": cloud.provider,
            "region": cloud.region,
            "monthly_cost": round(total_cost, 2),
            "breakdown": {
                "compute": round(compute_hours * cloud.compute_cost_per_hour, 2),
                "storage": round(storage_gb * cloud.storage_cost_per_gb, 2),
                "network": round(egress_gb * cloud.network_egress_cost_per_gb, 2),
                "ai_api": round(ai_requests * 0.002 * cloud.ai_api_cost_multiplier, 2)
            }
        })

    # Sort by cost
    costs.sort(key=lambda x: x["monthly_cost"])

    # Calculate savings
    cheapest = costs[0]["monthly_cost"]
    for cost in costs[1:]:
        cost["savings_vs_cheapest"] = round(cost["monthly_cost"] - cheapest, 2)
        cost["savings_percent"] = round(
            ((cost["monthly_cost"] - cheapest) / cost["monthly_cost"]) * 100, 2
        )

    return {
        "recommended": costs[0],
        "all_options": costs,
        "parameters": {
            "compute_hours": compute_hours,
            "storage_gb": storage_gb,
            "egress_gb": egress_gb,
            "ai_requests": ai_requests
        }
    }

if __name__ == "__main__":
    # Example usage
    result = recommend_cloud_placement(
        compute_hours=730,  # 1 month
        storage_gb=100,
        egress_gb=500,
        ai_requests=100000
    )

    print("Multi-Cloud Cost Optimization Analysis")
    print("=" * 50)
    print(f"\nRecommended Provider: {result['recommended']['provider']}")
    print(f"Region: {result['recommended']['region']}")
    print(f"Monthly Cost: ${result['recommended']['monthly_cost']}")
    print("\nCost Breakdown:")
    for key, value in result['recommended']['breakdown'].items():
        print(f"  {key}: ${value}")

    print("\n\nAll Options:")
    print("-" * 50)
    for option in result['all_options']:
        print(f"\n{option['provider']} ({option['region']})")
        print(f"  Monthly Cost: ${option['monthly_cost']}")
        if 'savings_vs_cheapest' in option:
            print(f"  Extra Cost: ${option['savings_vs_cheapest']} (+{option['savings_percent']}%)")
```

## Network Connectivity

### VPN Mesh Configuration

Create VPN connections between cloud providers:

**AWS to Azure VPN:**
```bash
# Create AWS Customer Gateway for Azure
aws ec2 create-customer-gateway \
  --type ipsec.1 \
  --public-ip <AZURE_VPN_GATEWAY_IP> \
  --bgp-asn 65000 \
  --tag-specifications 'ResourceType=customer-gateway,Tags=[{Key=Name,Value=azure-vpn}]'

# Create Virtual Private Gateway
aws ec2 create-vpn-gateway \
  --type ipsec.1 \
  --amazon-side-asn 64512 \
  --tag-specifications 'ResourceType=vpn-gateway,Tags=[{Key=Name,Value=multicloud-vgw}]'

# Create VPN Connection
aws ec2 create-vpn-connection \
  --type ipsec.1 \
  --customer-gateway-id <CUSTOMER_GATEWAY_ID> \
  --vpn-gateway-id <VPN_GATEWAY_ID> \
  --options TunnelOptions=[{TunnelInsideCidr=169.254.21.0/30,PreSharedKey=YOUR_PRESHARED_KEY}]

# Download configuration
aws ec2 describe-vpn-connections \
  --vpn-connection-ids <VPN_CONNECTION_ID> \
  --query 'VpnConnections[0].CustomerGatewayConfiguration' \
  --output text > aws-azure-vpn-config.xml
```

**Azure VPN Gateway:**
```bash
# Create Virtual Network Gateway
az network vnet-gateway create \
  --name azure-vpn-gateway \
  --resource-group genops-rg \
  --vnet genops-vnet \
  --gateway-type Vpn \
  --vpn-type RouteBased \
  --sku VpnGw1 \
  --public-ip-address azure-vpn-ip

# Create Local Network Gateway (represents AWS)
az network local-gateway create \
  --name aws-local-gateway \
  --resource-group genops-rg \
  --gateway-ip-address <AWS_VPN_ENDPOINT_IP> \
  --local-address-prefixes 10.0.0.0/16  # AWS VPC CIDR

# Create VPN Connection
az network vpn-connection create \
  --name azure-to-aws \
  --resource-group genops-rg \
  --vnet-gateway1 azure-vpn-gateway \
  --local-gateway2 aws-local-gateway \
  --shared-key YOUR_PRESHARED_KEY \
  --connection-type IPsec
```

### Cross-Cloud Service Discovery

Configure DNS and service discovery across clouds:

```yaml
# multicloud-service-discovery.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: genops-service-discovery
  namespace: genops-system
data:
  coredns-custom: |
    # Custom CoreDNS configuration for multi-cloud
    genops.aws.internal:53 {
        errors
        cache 30
        forward . 10.0.0.2  # AWS VPC DNS
    }

    genops.azure.internal:53 {
        errors
        cache 30
        forward . 10.1.0.2  # Azure Virtual Network DNS
    }

    genops.gcp.internal:53 {
        errors
        cache 30
        forward . 10.2.0.2  # GCP VPC DNS
    }

    # Multi-cloud service resolution
    genops.multicloud:53 {
        errors
        cache 30
        template IN A {
            match "^genops-ai\.genops\.multicloud\.$"
            answer "{{ .Name }} 60 IN A 10.0.1.100"  # AWS endpoint
            answer "{{ .Name }} 60 IN A 10.1.1.100"  # Azure endpoint
            answer "{{ .Name }} 60 IN A 10.2.1.100"  # GCP endpoint
            fallthrough
        }
    }
---
apiVersion: v1
kind: Service
metadata:
  name: genops-multicloud-dns
  namespace: kube-system
spec:
  selector:
    k8s-app: kube-dns
  ports:
  - name: dns
    port: 53
    protocol: UDP
  - name: dns-tcp
    port: 53
    protocol: TCP
```

### Global Load Balancing

Implement global load balancing with health checks:

```yaml
# global-load-balancer.yaml
apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: genops-multicloud-gateway
  namespace: genops-system
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 443
      name: https
      protocol: HTTPS
    tls:
      mode: SIMPLE
      credentialName: genops-tls-cert
    hosts:
    - "*.genops.example.com"
---
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: genops-multicloud-routing
  namespace: genops-system
spec:
  hosts:
  - "genops.example.com"
  gateways:
  - genops-multicloud-gateway
  http:
  - match:
    - headers:
        x-cloud-preference:
          exact: aws
    route:
    - destination:
        host: genops-ai.aws.svc.cluster.local
        port:
          number: 8080
  - match:
    - headers:
        x-cloud-preference:
          exact: azure
    route:
    - destination:
        host: genops-ai.azure.svc.cluster.local
        port:
          number: 8080
  - match:
    - headers:
        x-cloud-preference:
          exact: gcp
    route:
    - destination:
        host: genops-ai.gcp.svc.cluster.local
        port:
          number: 8080
  # Default: weighted distribution
  - route:
    - destination:
        host: genops-ai.aws.svc.cluster.local
        port:
          number: 8080
      weight: 40
    - destination:
        host: genops-ai.azure.svc.cluster.local
        port:
          number: 8080
      weight: 40
    - destination:
        host: genops-ai.gcp.svc.cluster.local
        port:
          number: 8080
      weight: 20
    retries:
      attempts: 3
      perTryTimeout: 2s
      retryOn: 5xx,connect-failure,refused-stream
```

## Cost Optimization

### Instance Type Selection Matrix

Optimal instance types across cloud providers:

```yaml
# cost-optimized-instance-matrix.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: instance-cost-matrix
  namespace: genops-system
data:
  matrix.yaml: |
    # Instance type recommendations by workload
    workload_types:
      # General purpose AI workloads
      general:
        aws:
          instance: m5.large
          vcpu: 2
          memory_gb: 8
          cost_per_hour: 0.096

        azure:
          instance: Standard_D2s_v3
          vcpu: 2
          memory_gb: 8
          cost_per_hour: 0.096

        gcp:
          instance: n1-standard-2
          vcpu: 2
          memory_gb: 7.5
          cost_per_hour: 0.095

      # Compute-intensive workloads
      compute:
        aws:
          instance: c5.xlarge
          vcpu: 4
          memory_gb: 8
          cost_per_hour: 0.17

        azure:
          instance: Standard_F4s_v2
          vcpu: 4
          memory_gb: 8
          cost_per_hour: 0.169

        gcp:
          instance: n1-highcpu-4
          vcpu: 4
          memory_gb: 3.6
          cost_per_hour: 0.142

      # Memory-intensive workloads
      memory:
        aws:
          instance: r5.large
          vcpu: 2
          memory_gb: 16
          cost_per_hour: 0.126

        azure:
          instance: Standard_E2s_v3
          vcpu: 2
          memory_gb: 16
          cost_per_hour: 0.126

        gcp:
          instance: n1-highmem-2
          vcpu: 2
          memory_gb: 13
          cost_per_hour: 0.118
```

### Spot/Preemptible Instance Strategy

Configure spot instances across clouds for 60-90% cost savings:

```yaml
# spot-instance-nodepool.yaml
# AWS Spot instances via eksctl
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig
metadata:
  name: genops-multicloud
  region: us-west-2
managedNodeGroups:
  - name: genops-spot
    instanceTypes: ["m5.large", "m5.xlarge", "c5.large"]
    spot: true
    minSize: 1
    maxSize: 10
    desiredCapacity: 3
    labels:
      workload-type: batch
      cost-optimization: spot
    taints:
      - key: spot-instance
        value: "true"
        effect: NoSchedule
---
# Azure spot VMs via AKS
apiVersion: v1
kind: ConfigMap
metadata:
  name: azure-spot-config
data:
  nodepool.json: |
    {
      "name": "genopsspot",
      "count": 3,
      "vmSize": "Standard_D2s_v3",
      "type": "VirtualMachineScaleSets",
      "mode": "User",
      "scaleSetPriority": "Spot",
      "scaleSetEvictionPolicy": "Delete",
      "spotMaxPrice": -1,
      "nodeTaints": ["kubernetes.azure.com/scalesetpriority=spot:NoSchedule"]
    }
---
# GCP preemptible VMs via GKE
apiVersion: v1
kind: ConfigMap
metadata:
  name: gcp-preemptible-config
data:
  nodepool.yaml: |
    name: genops-preemptible
    initialNodeCount: 3
    config:
      machineType: n1-standard-2
      preemptible: true
      taints:
      - key: cloud.google.com/gke-preemptible
        value: "true"
        effect: NoSchedule
```

**Deploy workloads on spot instances:**
```yaml
# spot-workload-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genops-batch-processor
  namespace: genops-system
spec:
  replicas: 5
  selector:
    matchLabels:
      app: genops-batch
  template:
    metadata:
      labels:
        app: genops-batch
    spec:
      # Tolerate spot instance taints
      tolerations:
      - key: spot-instance
        operator: Equal
        value: "true"
        effect: NoSchedule
      - key: kubernetes.azure.com/scalesetpriority
        operator: Equal
        value: spot
        effect: NoSchedule
      - key: cloud.google.com/gke-preemptible
        operator: Equal
        value: "true"
        effect: NoSchedule

      # Node affinity for spot instances
      affinity:
        nodeAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            preference:
              matchExpressions:
              - key: workload-type
                operator: In
                values:
                - batch
                - spot

      containers:
      - name: batch-processor
        image: genopsai/batch-processor:latest
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 1000m
            memory: 2Gi
```

### Cross-Cloud Data Transfer Cost Management

Minimize data transfer costs between clouds:

```yaml
# data-transfer-optimization.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: data-transfer-policy
  namespace: genops-system
data:
  policy.yaml: |
    # Data transfer cost optimization
    dataTansfer:
      # Prefer intra-region transfers
      regionAffinity: true

      # Cache frequently accessed data locally
      caching:
        enabled: true
        ttl: 3600  # 1 hour
        maxSize: 10GB

      # Compress data before transfer
      compression:
        enabled: true
        algorithm: gzip
        minSize: 1MB  # Only compress files > 1MB

      # Batch transfers to reduce overhead
      batching:
        enabled: true
        batchSize: 100
        maxWait: 60s

      # Monitor and alert on expensive transfers
      monitoring:
        enabled: true
        costThreshold: 10  # Alert if transfer cost > $10/day

      # Data transfer routes (ordered by cost)
      routes:
        # Intra-cloud (cheapest)
        - source: aws-us-west-2
          destination: aws-us-west-2
          cost_per_gb: 0.00

        # Same cloud, different region
        - source: aws-us-west-2
          destination: aws-eu-west-1
          cost_per_gb: 0.02

        # Cross-cloud via VPN (medium cost)
        - source: aws-us-west-2
          destination: azure-eastus
          cost_per_gb: 0.05
          method: vpn

        # Cross-cloud via internet (highest cost)
        - source: aws-us-west-2
          destination: gcp-us-central1
          cost_per_gb: 0.12
          method: internet
```

## Governance & Compliance

### Unified Policy Enforcement

Enforce policies consistently across all cloud providers:

```yaml
# multicloud-policy-enforcement.yaml
apiVersion: constraints.gatekeeper.sh/v1beta1
kind: K8sRequiredLabels
metadata:
  name: multicloud-governance-labels
spec:
  match:
    kinds:
    - apiGroups: ["*"]
      kinds: ["Pod", "Deployment", "Service"]
    namespaces:
    - genops-system
  parameters:
    labels:
      # Required governance labels
      - key: "genops.ai/team"
        allowedRegex: "^[a-z0-9-]+$"
      - key: "genops.ai/project"
        allowedRegex: "^[a-z0-9-]+$"
      - key: "genops.ai/environment"
        allowedRegex: "^(dev|staging|prod)$"
      - key: "genops.ai/cost-center"
        allowedRegex: "^[a-z0-9-]+$"
      # Cloud-specific labels
      - key: "genops.ai/cloud-provider"
        allowedRegex: "^(aws|azure|gcp)$"
      - key: "genops.ai/region"
        allowedRegex: "^[a-z0-9-]+$"
---
# Budget constraint across clouds
apiVersion: v1
kind: ConfigMap
metadata:
  name: multicloud-budget-policy
  namespace: genops-system
data:
  budget.yaml: |
    budgets:
      # Global budget across all clouds
      global:
        monthly_limit: 10000  # $10,000/month total
        currency: USD
        alerts:
          - threshold: 80
            action: notify
            recipients: [platform-team@example.com]
          - threshold: 95
            action: throttle
          - threshold: 100
            action: block

      # Per-cloud budgets
      aws:
        monthly_limit: 4000  # $4,000/month
        alerts:
          - threshold: 90
            action: notify

      azure:
        monthly_limit: 4000  # $4,000/month
        alerts:
          - threshold: 90
            action: notify

      gcp:
        monthly_limit: 2000  # $2,000/month
        alerts:
          - threshold: 90
            action: notify

      # Per-team budgets (apply across all clouds)
      by_team:
        ai-research:
          monthly_limit: 3000
        product-engineering:
          monthly_limit: 5000
        customer-success:
          monthly_limit: 2000
```

### Cross-Cloud Audit Logging

Centralize audit logs from all cloud providers:

```yaml
# centralized-audit-logging.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: audit-aggregation-config
  namespace: genops-system
data:
  fluentd.conf: |
    # AWS CloudTrail logs
    <source>
      @type cloudwatch_logs
      region us-west-2
      log_group_name /aws/eks/genops-cluster/audit
      use_aws_timestamp true
      tag aws.audit
    </source>

    # Azure Activity logs
    <source>
      @type azure_loganalytics
      workspace_id YOUR_WORKSPACE_ID
      shared_key YOUR_SHARED_KEY
      tag azure.audit
    </source>

    # GCP Cloud Audit logs
    <source>
      @type google_cloud_logging
      project_id YOUR_PROJECT_ID
      filter 'resource.type="k8s_cluster" AND logName="projects/YOUR_PROJECT_ID/logs/cloudaudit.googleapis.com%2Factivity"'
      tag gcp.audit
    </source>

    # Enrich with governance metadata
    <filter *.audit>
      @type record_transformer
      enable_ruby true
      <record>
        cloud_provider ${tag_parts[0]}
        cluster_name ${record["cluster_name"]}
        governance_team ${record["labels"]["genops.ai/team"] || "unknown"}
        governance_project ${record["labels"]["genops.ai/project"] || "unknown"}
        governance_environment ${record["labels"]["genops.ai/environment"] || "unknown"}
      </record>
    </filter>

    # Send to centralized SIEM
    <match *.audit>
      @type forward
      <server>
        name splunk-hec
        host splunk.example.com
        port 8088
      </server>
      <buffer>
        @type file
        path /var/log/fluentd-buffers/audit
        flush_interval 10s
      </buffer>
    </match>
---
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: audit-log-collector
  namespace: genops-system
spec:
  selector:
    matchLabels:
      app: audit-collector
  template:
    metadata:
      labels:
        app: audit-collector
    spec:
      serviceAccountName: audit-collector
      containers:
      - name: fluentd
        image: fluent/fluentd-kubernetes-daemonset:v1-debian-forward
        env:
        - name: CLOUD_PROVIDER
          valueFrom:
            fieldRef:
              fieldPath: metadata.labels['cloud-provider']
        volumeMounts:
        - name: config
          mountPath: /fluentd/etc/fluent.conf
          subPath: fluentd.conf
      volumes:
      - name: config
        configMap:
          name: audit-aggregation-config
```

### Data Residency & Sovereignty

Ensure compliance with regional data requirements:

```yaml
# data-residency-policy.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: data-residency-rules
  namespace: genops-system
data:
  residency.yaml: |
    # Data residency rules by region/country
    rules:
      # GDPR (European Union)
      - name: gdpr-compliance
        regions: [eu-west-1, eu-central-1, westeurope, northeurope]
        requirements:
          data_residency: EU
          encryption_required: true
          data_sovereignty: true
          allowed_clouds: [aws-eu, azure-eu, gcp-eu]
          cross_border_transfer: false
        enforcement: strict

      # CCPA (California)
      - name: ccpa-compliance
        regions: [us-west-1, us-west-2, westus, westus2]
        requirements:
          data_residency: US
          encryption_required: true
          data_deletion_support: true
          allowed_clouds: [aws-us, azure-us, gcp-us]
        enforcement: moderate

      # China Data Laws
      - name: china-compliance
        regions: [cn-north-1, chinaeast, chinanorth]
        requirements:
          data_residency: CN
          local_cloud_only: true
          government_access: required
          allowed_clouds: [aws-china, azure-china]
          cross_border_transfer: false
        enforcement: strict

      # Default (permissive)
      - name: default
        regions: ["*"]
        requirements:
          encryption_required: true
          allowed_clouds: [aws, azure, gcp]
        enforcement: moderate
```

## Migration Scenarios

### Workload Migration Between Clouds

Migrate GenOps AI workloads from AWS to Azure:

```bash
# migration-aws-to-azure.sh
#!/bin/bash

echo "🔄 Migrating GenOps AI from AWS to Azure"
echo "========================================"

# Step 1: Backup AWS deployment
echo "📦 Backing up AWS deployment..."
kubectl config use-context aws-cluster
kubectl get all -n genops-system -o yaml > genops-aws-backup.yaml
velero backup create genops-aws-migration \
  --include-namespaces genops-system \
  --wait

# Step 2: Export configuration and data
echo "📤 Exporting configuration..."
kubectl get configmap -n genops-system -o yaml > genops-configmaps.yaml
kubectl get secret -n genops-system -o yaml > genops-secrets.yaml

# Step 3: Prepare Azure cluster
echo "🎯 Preparing Azure cluster..."
kubectl config use-context azure-cluster

# Create namespace
kubectl create namespace genops-system

# Step 4: Migrate secrets (re-encrypt for Azure)
echo "🔐 Migrating secrets..."
kubectl apply -f genops-secrets.yaml -n genops-system

# Step 5: Migrate configuration
echo "⚙️  Migrating configuration..."
kubectl apply -f genops-configmaps.yaml -n genops-system

# Step 6: Deploy GenOps on Azure
echo "🚀 Deploying GenOps on Azure..."
helm install genops-ai genops/genops-ai \
  --namespace genops-system \
  --set cloud.provider=azure \
  --set cloud.region=eastus \
  --set migration.source=aws \
  --set migration.dataImport=true \
  --wait

# Step 7: Verify deployment
echo "✅ Verifying Azure deployment..."
kubectl wait --for=condition=Ready pods --all -n genops-system --timeout=300s
kubectl get pods -n genops-system

# Step 8: Migrate data
echo "📊 Migrating governance data..."
kubectl exec -n genops-system deployment/genops-ai -- \
  genops-cli migrate \
  --source s3://genops-aws-bucket \
  --destination azureblob://genopsazurestorage \
  --verify

# Step 9: Test Azure deployment
echo "🧪 Testing Azure deployment..."
kubectl port-forward -n genops-system svc/genops-ai 8080:8080 &
PF_PID=$!
sleep 5

curl -f http://localhost:8080/health || {
  echo "❌ Health check failed"
  kill $PF_PID
  exit 1
}

kill $PF_PID

# Step 10: Update DNS for cutover
echo "🌐 Updating DNS..."
echo "Manual step: Update DNS to point to Azure endpoint"
echo "Azure endpoint: $(kubectl get svc -n genops-system genops-ai -o jsonpath='{.status.loadBalancer.ingress[0].ip}')"

echo "✅ Migration complete!"
echo "Next steps:"
echo "1. Monitor Azure deployment for 24-48 hours"
echo "2. Update DNS to Azure endpoint"
echo "3. After validation, decommission AWS resources"
```

### Cost Comparison Analysis

Compare costs between cloud providers for informed migration:

```python
# cloud-cost-comparison.py
#!/usr/bin/env python3
"""
Comprehensive cost comparison for multi-cloud migration decisions.
"""

from dataclasses import dataclass
from typing import Dict, List
import json

@dataclass
class WorkloadProfile:
    """Define workload characteristics"""
    name: str
    compute_hours_monthly: float
    memory_gb: int
    storage_gb: int
    network_egress_gb: float
    ai_api_calls: int
    high_availability: bool

@dataclass
class CloudPricing:
    """Cloud provider pricing"""
    provider: str
    region: str
    compute_cost_per_hour: float
    memory_cost_per_gb_hour: float
    storage_cost_per_gb_month: float
    network_egress_cost_per_gb: float
    ai_api_cost_per_1k: float
    load_balancer_cost_per_hour: float

# Define pricing for major clouds (example rates)
PRICING = {
    "aws": CloudPricing(
        provider="AWS",
        region="us-west-2",
        compute_cost_per_hour=0.096,
        memory_cost_per_gb_hour=0.012,
        storage_cost_per_gb_month=0.10,
        network_egress_cost_per_gb=0.09,
        ai_api_cost_per_1k=2.00,
        load_balancer_cost_per_hour=0.0225
    ),
    "azure": CloudPricing(
        provider="Azure",
        region="eastus",
        compute_cost_per_hour=0.096,
        memory_cost_per_gb_hour=0.012,
        storage_cost_per_gb_month=0.0184,
        network_egress_cost_per_gb=0.087,
        ai_api_cost_per_1k=2.10,
        load_balancer_cost_per_hour=0.025
    ),
    "gcp": CloudPricing(
        provider="GCP",
        region="us-central1",
        compute_cost_per_hour=0.0475,
        memory_cost_per_gb_hour=0.00637,
        storage_cost_per_gb_month=0.020,
        network_egress_cost_per_gb=0.12,
        ai_api_cost_per_1k=1.90,
        load_balancer_cost_per_hour=0.025
    )
}

def calculate_monthly_cost(workload: WorkloadProfile, pricing: CloudPricing) -> Dict:
    """Calculate monthly cost for workload on specific cloud"""

    # Compute cost
    compute = workload.compute_hours_monthly * pricing.compute_cost_per_hour

    # Memory cost
    memory = workload.compute_hours_monthly * workload.memory_gb * pricing.memory_cost_per_gb_hour

    # Storage cost
    storage = workload.storage_gb * pricing.storage_cost_per_gb_month

    # Network egress
    network = workload.network_egress_gb * pricing.network_egress_cost_per_gb

    # AI API calls
    ai_api = (workload.ai_api_calls / 1000) * pricing.ai_api_cost_per_1k

    # High availability (load balancer, multi-AZ)
    ha_cost = 0
    if workload.high_availability:
        ha_cost = (30 * 24 * pricing.load_balancer_cost_per_hour) + (compute * 0.1)

    total = compute + memory + storage + network + ai_api + ha_cost

    return {
        "provider": pricing.provider,
        "region": pricing.region,
        "breakdown": {
            "compute": round(compute, 2),
            "memory": round(memory, 2),
            "storage": round(storage, 2),
            "network": round(network, 2),
            "ai_api": round(ai_api, 2),
            "high_availability": round(ha_cost, 2)
        },
        "total_monthly": round(total, 2),
        "total_annual": round(total * 12, 2)
    }

def compare_clouds(workload: WorkloadProfile) -> Dict:
    """Compare costs across all clouds"""

    results = []
    for provider, pricing in PRICING.items():
        cost = calculate_monthly_cost(workload, pricing)
        results.append(cost)

    # Sort by cost
    results.sort(key=lambda x: x["total_monthly"])

    # Calculate savings potential
    cheapest = results[0]["total_monthly"]
    for i, result in enumerate(results):
        if i > 0:
            savings = result["total_monthly"] - cheapest
            savings_percent = (savings / result["total_monthly"]) * 100
            result["potential_savings"] = {
                "amount": round(savings, 2),
                "percent": round(savings_percent, 2),
                "annual": round(savings * 12, 2)
            }

    return {
        "workload": workload.name,
        "recommended_provider": results[0]["provider"],
        "results": results
    }

# Example workload profiles
WORKLOADS = [
    WorkloadProfile(
        name="Production AI Service",
        compute_hours_monthly=730,  # 24/7
        memory_gb=16,
        storage_gb=500,
        network_egress_gb=1000,
        ai_api_calls=500000,
        high_availability=True
    ),
    WorkloadProfile(
        name="Development Environment",
        compute_hours_monthly=176,  # 8 hours/day, 22 days
        memory_gb=8,
        storage_gb=100,
        network_egress_gb=50,
        ai_api_calls=10000,
        high_availability=False
    ),
    WorkloadProfile(
        name="Batch Processing",
        compute_hours_monthly=200,
        memory_gb=32,
        storage_gb=1000,
        network_egress_gb=500,
        ai_api_calls=1000000,
        high_availability=False
    )
]

if __name__ == "__main__":
    print("Multi-Cloud Cost Comparison Analysis")
    print("=" * 70)

    for workload in WORKLOADS:
        print(f"\n\nWorkload: {workload.name}")
        print("-" * 70)

        comparison = compare_clouds(workload)

        print(f"\n✅ Recommended Provider: {comparison['recommended_provider']}")

        for result in comparison["results"]:
            print(f"\n{result['provider']} ({result['region']}):")
            print(f"  Monthly Cost: ${result['total_monthly']}")
            print(f"  Annual Cost: ${result['total_annual']}")
            print(f"  Breakdown:")
            for category, cost in result['breakdown'].items():
                print(f"    {category}: ${cost}")

            if "potential_savings" in result:
                savings = result["potential_savings"]
                print(f"  💰 Potential Savings: ${savings['amount']}/month (${savings['annual']}/year, {savings['percent']}%)")
```

## Operational Excellence

### Unified Monitoring Across Clouds

Deploy centralized monitoring for all cloud environments:

```yaml
# multicloud-prometheus.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-multicloud-config
  namespace: monitoring
data:
  prometheus.yml: |
    global:
      scrape_interval: 30s
      evaluation_interval: 30s
      external_labels:
        cluster_type: multicloud

    # Scrape configs for each cloud
    scrape_configs:
      # AWS EKS cluster
      - job_name: 'genops-aws'
        kubernetes_sd_configs:
        - role: pod
          api_server: https://aws-cluster-api.example.com
          tls_config:
            ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
          bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
        relabel_configs:
        - source_labels: [__meta_kubernetes_namespace]
          regex: genops-system
          action: keep
        - source_labels: [__meta_kubernetes_pod_label_app]
          target_label: app
        - replacement: aws
          target_label: cloud_provider
        - replacement: us-west-2
          target_label: region

      # Azure AKS cluster
      - job_name: 'genops-azure'
        kubernetes_sd_configs:
        - role: pod
          api_server: https://azure-cluster-api.example.com
          tls_config:
            ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
          bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
        relabel_configs:
        - source_labels: [__meta_kubernetes_namespace]
          regex: genops-system
          action: keep
        - source_labels: [__meta_kubernetes_pod_label_app]
          target_label: app
        - replacement: azure
          target_label: cloud_provider
        - replacement: eastus
          target_label: region

      # GCP GKE cluster
      - job_name: 'genops-gcp'
        kubernetes_sd_configs:
        - role: pod
          api_server: https://gcp-cluster-api.example.com
          tls_config:
            ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
          bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
        relabel_configs:
        - source_labels: [__meta_kubernetes_namespace]
          regex: genops-system
          action: keep
        - source_labels: [__meta_kubernetes_pod_label_app]
          target_label: app
        - replacement: gcp
          target_label: cloud_provider
        - replacement: us-central1
          target_label: region

    # Alerting rules
    rule_files:
      - '/etc/prometheus/rules/*.yml'
---
# Multi-cloud alerting rules
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-multicloud-rules
  namespace: monitoring
data:
  multicloud-alerts.yml: |
    groups:
    - name: multicloud-genops
      interval: 30s
      rules:
      # Alert if any cloud provider is down
      - alert: CloudProviderDown
        expr: up{job=~"genops-(aws|azure|gcp)"} == 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Cloud provider {{ $labels.cloud_provider }} is down"
          description: "GenOps on {{ $labels.cloud_provider }} ({{ $labels.region }}) has been down for more than 5 minutes"

      # Alert on cost anomalies
      - alert: CrossCloudCostAnomaly
        expr: |
          (
            sum by (cloud_provider) (rate(genops_cost_total[1h]))
            >
            sum by (cloud_provider) (rate(genops_cost_total[1h] offset 24h)) * 1.5
          )
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "Unusual cost increase on {{ $labels.cloud_provider }}"
          description: "Cost on {{ $labels.cloud_provider }} has increased by 50% compared to yesterday"

      # Alert if traffic is not balanced
      - alert: UnbalancedMultiCloudTraffic
        expr: |
          (
            max by (cloud_provider) (rate(genops_requests_total[5m]))
            >
            min by (cloud_provider) (rate(genops_requests_total[5m])) * 3
          )
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Traffic imbalance across clouds"
          description: "One cloud is receiving 3x more traffic than another"
```

### Cross-Cloud CI/CD Pipeline

Implement unified CI/CD across multiple clouds:

```yaml
# .github/workflows/multicloud-deploy.yml
name: Multi-Cloud Deployment

on:
  push:
    branches: [main]
  workflow_dispatch:
    inputs:
      target_clouds:
        description: 'Target clouds (comma-separated: aws,azure,gcp)'
        required: true
        default: 'aws,azure,gcp'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build GenOps image
        run: |
          docker build -t genopsai/genops:${{ github.sha }} .
          docker tag genopsai/genops:${{ github.sha }} genopsai/genops:latest

      - name: Push to registries
        run: |
          # AWS ECR
          aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin ${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.us-west-2.amazonaws.com
          docker tag genopsai/genops:${{ github.sha }} ${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.us-west-2.amazonaws.com/genops:${{ github.sha }}
          docker push ${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.us-west-2.amazonaws.com/genops:${{ github.sha }}

          # Azure ACR
          az acr login --name ${{ secrets.AZURE_REGISTRY_NAME }}
          docker tag genopsai/genops:${{ github.sha }} ${{ secrets.AZURE_REGISTRY_NAME }}.azurecr.io/genops:${{ github.sha }}
          docker push ${{ secrets.AZURE_REGISTRY_NAME }}.azurecr.io/genops:${{ github.sha }}

          # GCP GCR
          gcloud auth configure-docker
          docker tag genopsai/genops:${{ github.sha }} gcr.io/${{ secrets.GCP_PROJECT_ID }}/genops:${{ github.sha }}
          docker push gcr.io/${{ secrets.GCP_PROJECT_ID }}/genops:${{ github.sha }}

  deploy-aws:
    needs: build
    runs-on: ubuntu-latest
    if: contains(github.event.inputs.target_clouds, 'aws')
    steps:
      - name: Configure AWS
        uses: aws-actions/configure-aws-credentials@v2
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-west-2

      - name: Update kubeconfig
        run: aws eks update-kubeconfig --name genops-cluster --region us-west-2

      - name: Deploy to AWS
        run: |
          helm upgrade --install genops-ai genops/genops-ai \
            --namespace genops-system \
            --set image.repository=${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.us-west-2.amazonaws.com/genops \
            --set image.tag=${{ github.sha }} \
            --set cloud.provider=aws \
            --wait

  deploy-azure:
    needs: build
    runs-on: ubuntu-latest
    if: contains(github.event.inputs.target_clouds, 'azure')
    steps:
      - name: Azure Login
        uses: azure/login@v1
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}

      - name: Set AKS context
        run: az aks get-credentials --resource-group genops-rg --name genops-cluster

      - name: Deploy to Azure
        run: |
          helm upgrade --install genops-ai genops/genops-ai \
            --namespace genops-system \
            --set image.repository=${{ secrets.AZURE_REGISTRY_NAME }}.azurecr.io/genops \
            --set image.tag=${{ github.sha }} \
            --set cloud.provider=azure \
            --wait

  deploy-gcp:
    needs: build
    runs-on: ubuntu-latest
    if: contains(github.event.inputs.target_clouds, 'gcp')
    steps:
      - name: GCP Authentication
        uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_CREDENTIALS }}

      - name: Set GKE context
        run: gcloud container clusters get-credentials genops-cluster --region us-central1

      - name: Deploy to GCP
        run: |
          helm upgrade --install genops-ai genops/genops-ai \
            --namespace genops-system \
            --set image.repository=gcr.io/${{ secrets.GCP_PROJECT_ID }}/genops \
            --set image.tag=${{ github.sha }} \
            --set cloud.provider=gcp \
            --wait

  verify:
    needs: [deploy-aws, deploy-azure, deploy-gcp]
    runs-on: ubuntu-latest
    steps:
      - name: Verify deployments
        run: |
          for context in aws-cluster azure-cluster gcp-cluster; do
            echo "Verifying $context..."
            kubectl --context $context get pods -n genops-system
            kubectl --context $context rollout status deployment/genops-ai -n genops-system
          done
```

## Troubleshooting

### Common Multi-Cloud Issues

#### Issue: Cross-Cloud Network Connectivity Failures

**Diagnosis:**
```bash
# Test connectivity between clouds
kubectl exec -n genops-system deployment/genops-ai -- \
  curl -v https://genops-azure.example.com/health

# Check VPN status (AWS)
aws ec2 describe-vpn-connections \
  --vpn-connection-ids <VPN_ID> \
  --query 'VpnConnections[0].VgwTelemetry'

# Check VPN status (Azure)
az network vpn-connection show \
  --name azure-to-aws \
  --resource-group genops-rg \
  --query connectionStatus
```

**Solutions:**

1. **Verify VPN tunnels are up:**
   ```bash
   # AWS: Check tunnel status
   aws ec2 describe-vpn-connections \
     --vpn-connection-ids <VPN_ID> \
     --query 'VpnConnections[0].VgwTelemetry[*].[OutsideIpAddress,Status]' \
     --output table

   # Azure: Verify connection
   az network vpn-connection show \
     --name azure-to-aws \
     --resource-group genops-rg
   ```

2. **Check route tables:**
   ```bash
   # AWS: Verify routes to Azure CIDR
   aws ec2 describe-route-tables \
     --filters "Name=vpc-id,Values=<VPC_ID>" \
     --query 'RouteTables[*].Routes[?DestinationCidrBlock==`10.1.0.0/16`]'

   # Azure: Check effective routes
   az network nic show-effective-route-table \
     --name genops-nic \
     --resource-group genops-rg
   ```

3. **Verify security groups/NSGs:**
   ```bash
   # AWS: Check security group rules
   aws ec2 describe-security-groups \
     --group-ids <SG_ID> \
     --query 'SecurityGroups[0].IpPermissions'

   # Azure: Check NSG rules
   az network nsg rule list \
     --nsg-name genops-nsg \
     --resource-group genops-rg \
     --output table
   ```

#### Issue: Inconsistent Policy Enforcement Across Clouds

**Diagnosis:**
```bash
# Check policy status on each cloud
for context in aws-cluster azure-cluster gcp-cluster; do
  echo "Checking policies on $context:"
  kubectl --context $context get constraints -A
done

# Check for policy violations
kubectl get constraints -A -o json | \
  jq '.items[] | select(.status.totalViolations > 0) | {name: .metadata.name, violations: .status.totalViolations}'
```

**Solutions:**

1. **Sync policies across clusters:**
   ```bash
   # Export policies from primary cluster
   kubectl --context aws-cluster get constraints -A -o yaml > policies.yaml

   # Apply to other clusters
   kubectl --context azure-cluster apply -f policies.yaml
   kubectl --context gcp-cluster apply -f policies.yaml
   ```

2. **Use federation for policy distribution:**
   ```yaml
   # federated-constraint.yaml
   apiVersion: types.kubefed.io/v1beta1
   kind: FederatedConstraint
   metadata:
     name: governance-labels-required
   spec:
     template:
       # Policy definition
     placement:
       clusters:
       - name: aws-cluster
       - name: azure-cluster
       - name: gcp-cluster
   ```

#### Issue: Cost Tracking Discrepancies

**Diagnosis:**
```bash
# Compare costs across clouds
kubectl exec -n genops-system deployment/genops-ai -- \
  genops-cli cost-report --cloud all --period last-7-days --format json > cost-report.json

# Analyze discrepancies
python3 << 'EOF'
import json
with open('cost-report.json') as f:
    data = json.load(f)

for cloud in ['aws', 'azure', 'gcp']:
    print(f"{cloud}: ${data[cloud]['total']}")
    if data[cloud]['tracking_errors'] > 0:
        print(f"  ⚠️  {data[cloud]['tracking_errors']} tracking errors")
EOF
```

**Solutions:**

1. **Verify cost tracking configuration:**
   ```bash
   kubectl get configmap genops-cost-config -n genops-system -o yaml
   ```

2. **Re-sync cost data:**
   ```bash
   kubectl exec -n genops-system deployment/genops-ai -- \
     genops-cli cost-sync --cloud all --force
   ```

3. **Enable detailed cost logging:**
   ```bash
   kubectl patch deployment genops-ai -n genops-system \
     --patch '{"spec":{"template":{"spec":{"containers":[{"name":"genops-ai","env":[{"name":"COST_TRACKING_DEBUG","value":"true"}]}]}}}}'
   ```

#### Issue: Federation Control Plane Failures

**Diagnosis:**
```bash
# Check KubeFed status
kubectl get kubefedclusters -n kube-federation-system
kubectl describe kubefedclusters -n kube-federation-system

# Check federation controller logs
kubectl logs -n kube-federation-system deployment/kubefed-controller-manager --tail=100
```

**Solutions:**

1. **Rejoin failed clusters:**
   ```bash
   # Remove and rejoin cluster
   kubefedctl unjoin azure-cluster --host-cluster-context aws-primary
   kubefedctl join azure-cluster \
     --cluster-context azure-secondary \
     --host-cluster-context aws-primary
   ```

2. **Verify cluster connectivity:**
   ```bash
   # Test connectivity to member clusters
   kubectl --context aws-primary cluster-info
   kubectl --context azure-secondary cluster-info
   ```

3. **Reset federation resources:**
   ```bash
   # Delete and recreate federated resources
   kubectl delete federateddeployment genops-ai -n genops-system
   kubectl apply -f federated-genops-deployment.yaml
   ```

### Health Check Script

```bash
# multicloud-health-check.sh
#!/bin/bash

echo "🌐 Multi-Cloud GenOps Health Check"
echo "===================================="

CLUSTERS=("aws-cluster" "azure-cluster" "gcp-cluster")
PASSED=0
FAILED=0

for cluster in "${CLUSTERS[@]}"; do
  echo -e "\n📋 Checking $cluster..."

  # Check cluster connectivity
  if ! kubectl --context $cluster cluster-info &> /dev/null; then
    echo "  ❌ Cannot connect to $cluster"
    ((FAILED++))
    continue
  fi
  echo "  ✅ Cluster connectivity OK"

  # Check GenOps pods
  PODS=$(kubectl --context $cluster get pods -n genops-system --field-selector=status.phase=Running --no-headers | wc -l)
  if [ "$PODS" -lt 1 ]; then
    echo "  ❌ No running GenOps pods"
    ((FAILED++))
  else
    echo "  ✅ $PODS GenOps pods running"
    ((PASSED++))
  fi

  # Check service endpoints
  ENDPOINT=$(kubectl --context $cluster get svc genops-ai -n genops-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
  if [ -z "$ENDPOINT" ]; then
    ENDPOINT=$(kubectl --context $cluster get svc genops-ai -n genops-system -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null)
  fi

  if [ -n "$ENDPOINT" ]; then
    echo "  ✅ Service endpoint: $ENDPOINT"
  else
    echo "  ⚠️  No external endpoint found"
  fi
done

echo -e "\n===================================="
echo "Summary: $PASSED passed, $FAILED failed"

if [ $FAILED -gt 0 ]; then
  exit 1
fi
```

---

## Next Steps

1. **Start with dual-cloud setup** - Deploy to primary and secondary clouds first
2. **Implement cost optimization** - Use spot instances and workload placement strategies
3. **Set up unified monitoring** - Deploy centralized Prometheus and Grafana
4. **Configure failover** - Test automatic failover between clouds
5. **Optimize network costs** - Implement cross-cloud data transfer optimization
6. **Expand to third cloud** - Add GCP for tri-cloud deployment

## Additional Resources

- [AWS EKS Documentation](https://docs.aws.amazon.com/eks/)
- [Azure AKS Documentation](https://docs.microsoft.com/en-us/azure/aks/)
- [GCP GKE Documentation](https://cloud.google.com/kubernetes-engine/docs)
- [KubeFed Multi-Cluster](https://github.com/kubernetes-sigs/kubefed)
- [Istio Multi-Cluster](https://istio.io/latest/docs/setup/install/multicluster/)
- [GenOps AI Documentation](https://github.com/KoshiHQ/GenOps-AI)

---

This guide provides a comprehensive foundation for deploying GenOps AI across multiple cloud providers with unified governance, cost optimization, and operational excellence.
