# GenOps AI on Amazon EKS

Complete deployment guide for GenOps AI on Amazon Elastic Kubernetes Service (EKS) with native AWS integrations, cost optimization, and enterprise security.

## Table of Contents

1. [Quick Start (5 minutes)](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Prerequisites](#prerequisites)
4. [EKS Cluster Setup](#eks-cluster-setup)
5. [GenOps Deployment](#genops-deployment)
6. [AWS Service Integrations](#aws-service-integrations)
7. [Cost Management](#cost-management)
8. [Security & Compliance](#security-compliance)
9. [Monitoring & Observability](#monitoring-observability)
10. [Production Optimizations](#production-optimizations)
11. [Troubleshooting](#troubleshooting)

## Quick Start

Deploy GenOps AI on EKS in 5 minutes with basic configuration:

```bash
# 1. Create EKS cluster (if needed)
eksctl create cluster --name genops-cluster --version 1.28 --region us-west-2 --nodegroup-name standard-workers --node-type m5.large --nodes 3

# 2. Install GenOps with AWS optimizations
helm repo add genops https://charts.genops.ai
helm install genops-ai genops/genops-ai \
  --set cloud.provider=aws \
  --set aws.region=us-west-2 \
  --set aws.enableCostOptimization=true \
  --set observability.backend=cloudwatch

# 3. Verify deployment
kubectl get pods -n genops-system
```

✅ **Result:** GenOps AI running on EKS with CloudWatch integration and AWS cost optimization enabled.

## Architecture Overview

### GenOps on EKS Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Amazon VPC                           │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    EKS Control Plane                    │ │
│  │                 (Managed by AWS)                       │ │
│  └─────────────────────────────────────────────────────────┘ │
│                            │                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                   EKS Worker Nodes                     │ │
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
    │                AWS Services Integration                │
    │                                                        │
    │  CloudWatch    Cost Explorer    IAM Roles & Policies   │
    │  X-Ray         Bedrock         Secrets Manager        │
    │  Parameter     S3 Storage      CloudFormation        │
    │  Store                                                │
    └────────────────────────────────────────────────────────┘
```

### Key Components

- **EKS Control Plane**: Managed Kubernetes API server and etcd
- **Managed Node Groups**: Auto-scaling worker nodes with Spot instance support
- **GenOps Pods**: Core governance and proxy services
- **AWS Load Balancer Controller**: Intelligent traffic routing
- **AWS CNI**: Native VPC networking with security groups
- **Amazon EBS CSI Driver**: Persistent storage for governance data

## Prerequisites

### Required AWS Permissions

Create an IAM policy for EKS and GenOps operations:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "eks:*",
                "ec2:*",
                "iam:CreateServiceLinkedRole",
                "iam:CreateRole",
                "iam:AttachRolePolicy",
                "logs:*",
                "cloudwatch:*",
                "ce:*",
                "bedrock:*",
                "s3:*",
                "ssm:GetParameter",
                "ssm:PutParameter",
                "secretsmanager:*"
            ],
            "Resource": "*"
        }
    ]
}
```

### Required Tools

```bash
# Install required CLI tools
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

### AWS Account Setup

```bash
# Configure AWS credentials
aws configure set aws_access_key_id YOUR_ACCESS_KEY
aws configure set aws_secret_access_key YOUR_SECRET_KEY
aws configure set default.region us-west-2

# Verify access
aws sts get-caller-identity
aws eks list-clusters
```

## EKS Cluster Setup

### Production-Ready EKS Cluster

Create a production-ready EKS cluster with optimal configuration:

```bash
# Create cluster configuration file
cat > genops-eks-cluster.yaml << 'EOF'
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: genops-production
  region: us-west-2
  version: "1.28"
  tags:
    Environment: production
    Project: genops-ai
    CostCenter: engineering

# VPC and networking
vpc:
  enableDnsHostnames: true
  enableDnsSupport: true
  subnets:
    private:
      us-west-2a: { cidr: 192.168.0.0/19 }
      us-west-2b: { cidr: 192.168.32.0/19 }
      us-west-2c: { cidr: 192.168.64.0/19 }
    public:
      us-west-2a: { cidr: 192.168.96.0/24 }
      us-west-2b: { cidr: 192.168.97.0/24 }
      us-west-2c: { cidr: 192.168.98.0/24 }

# Control plane configuration
controlPlane:
  logging:
    enable: ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  tags:
    Environment: production

# Node groups
managedNodeGroups:
  # General purpose nodes
  - name: general-purpose
    instanceType: m5.large
    desiredCapacity: 3
    minSize: 2
    maxSize: 10
    availabilityZones: ["us-west-2a", "us-west-2b", "us-west-2c"]
    volumeSize: 100
    ssh:
      publicKeyName: your-key-pair
    labels:
      role: general
      cost-optimization: "true"
    tags:
      NodeGroup: general-purpose
      
  # Cost-optimized spot instances for batch workloads
  - name: spot-workers
    instanceTypes: ["m5.large", "m5.xlarge", "c5.large", "c5.xlarge"]
    spot: true
    desiredCapacity: 2
    minSize: 0
    maxSize: 20
    availabilityZones: ["us-west-2a", "us-west-2b", "us-west-2c"]
    labels:
      role: batch
      cost-optimization: "true"
      workload-type: spot
    taints:
      - key: spot-instance
        value: "true"
        effect: NoSchedule
    tags:
      NodeGroup: spot-workers

# Add-ons
addons:
  - name: vpc-cni
    version: v1.15.1-eksbuild.1
  - name: coredns
    version: v1.10.1-eksbuild.4
  - name: kube-proxy 
    version: v1.28.2-eksbuild.2
  - name: aws-ebs-csi-driver
    version: v1.24.1-eksbuild.1

# IAM service accounts
iam:
  withOIDC: true
  serviceAccounts:
    - metadata:
        name: aws-load-balancer-controller
        namespace: kube-system
      wellKnownPolicies:
        awsLoadBalancerController: true
    - metadata:
        name: ebs-csi-controller-sa
        namespace: kube-system
      wellKnownPolicies:
        ebsCSIController: true
    - metadata:
        name: genops-service-account
        namespace: genops-system
      attachPolicyARNs:
        - arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy
        - arn:aws:iam::aws:policy/AWSXRayDaemonWriteAccess
      attachPolicy:
        Version: "2012-10-17"
        Statement:
          - Effect: Allow
            Action:
              - "bedrock:*"
              - "ce:*"
              - "s3:GetObject"
              - "s3:PutObject"
              - "ssm:GetParameter"
              - "secretsmanager:GetSecretValue"
            Resource: "*"

# CloudWatch logging
cloudWatch:
  clusterLogging:
    enable: ["api", "audit", "authenticator", "controllerManager", "scheduler"]
EOF

# Create the cluster
eksctl create cluster -f genops-eks-cluster.yaml
```

### Verify EKS Setup

```bash
# Verify cluster is running
kubectl get nodes
kubectl get pods --all-namespaces

# Check cluster info
kubectl cluster-info

# Test connectivity
kubectl get svc
```

## GenOps Deployment

### Prepare GenOps Configuration

Create AWS-optimized GenOps configuration:

```bash
# Create GenOps namespace
kubectl create namespace genops-system

# Create AWS-specific configuration
cat > genops-aws-values.yaml << 'EOF'
# GenOps AI Helm Chart Values for AWS EKS

# Global configuration
global:
  environment: production
  cloud:
    provider: aws
    region: us-west-2
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
  
  # AWS-specific configuration
  aws:
    region: us-west-2
    enableCostOptimization: true
    enableBedrock: true
    enableXRayTracing: true
    
    # Cost management
    costExplorer:
      enabled: true
      budgetAlerts: true
      
    # Storage configuration
    s3:
      bucket: genops-governance-data
      region: us-west-2
      
    # Parameter Store for configuration
    parameterStore:
      prefix: /genops/production/
      
    # Secrets Manager integration
    secretsManager:
      secretName: genops-ai-keys

# Proxy service for AI workloads
proxy:
  enabled: true
  replicas: 2
  
  service:
    type: LoadBalancer
    annotations:
      service.beta.kubernetes.io/aws-load-balancer-type: nlb
      service.beta.kubernetes.io/aws-load-balancer-scheme: internal
      service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled: "true"
  
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
    bedrock:
      enabled: true
      region: us-west-2

# Observability stack
observability:
  # CloudWatch integration
  cloudwatch:
    enabled: true
    region: us-west-2
    namespace: GenOps/Production
    
  # X-Ray distributed tracing
  xray:
    enabled: true
    sampling: 0.1
    
  # Prometheus for metrics
  prometheus:
    enabled: true
    retention: 30d
    storage:
      class: gp3
      size: 100Gi
      
  # Grafana for dashboards
  grafana:
    enabled: true
    adminPassword: "change-me-in-production"
    dashboards:
      aws: true
      cost: true
      performance: true

# Storage configuration
storage:
  class: gp3
  size: 50Gi
  
# Security configuration
security:
  podSecurityPolicy: true
  networkPolicies: true
  
  # RBAC
  rbac:
    enabled: true
    
  # Service mesh (optional)
  istio:
    enabled: false

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
    # Scale down during non-business hours
    scaleDown:
      schedule: "0 18 * * *"
      replicas: 1
    scaleUp:
      schedule: "0 8 * * *" 
      replicas: 3
EOF
```

### Deploy GenOps with Helm

```bash
# Add GenOps Helm repository
helm repo add genops https://charts.genops.ai
helm repo update

# Install GenOps AI
helm install genops-ai genops/genops-ai \
  --namespace genops-system \
  --values genops-aws-values.yaml \
  --wait \
  --timeout 10m

# Verify deployment
kubectl get pods -n genops-system
kubectl get services -n genops-system

# Check logs
kubectl logs -n genops-system deployment/genops-ai --tail=100
```

### Post-Deployment Configuration

```bash
# Create AWS-specific secrets
kubectl create secret generic genops-ai-keys \
  --namespace genops-system \
  --from-literal=openai-api-key="your-openai-key" \
  --from-literal=anthropic-api-key="your-anthropic-key"

# Configure AWS Parameter Store
aws ssm put-parameter \
  --name "/genops/production/budget-limit" \
  --value "1000" \
  --type "String" \
  --description "Monthly budget limit in USD"

aws ssm put-parameter \
  --name "/genops/production/cost-center" \
  --value "engineering" \
  --type "String" \
  --description "Default cost center for attribution"
```

## AWS Service Integrations

### Amazon Bedrock Integration

Configure GenOps to work with Amazon Bedrock:

```bash
# Create Bedrock-specific configuration
cat > bedrock-integration.yaml << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: bedrock-config
  namespace: genops-system
data:
  config.yaml: |
    bedrock:
      region: us-west-2
      models:
        - name: anthropic.claude-v2
          cost_per_1k_input: 0.008
          cost_per_1k_output: 0.024
        - name: anthropic.claude-instant-v1
          cost_per_1k_input: 0.0008 
          cost_per_1k_output: 0.0024
        - name: ai21.j2-ultra-v1
          cost_per_1k_input: 0.0188
          cost_per_1k_output: 0.0188
        - name: cohere.command-text-v14
          cost_per_1k_input: 0.0015
          cost_per_1k_output: 0.002
      governance:
        enable_cost_tracking: true
        enable_content_filtering: true
        enable_budget_limits: true
EOF

kubectl apply -f bedrock-integration.yaml
```

### CloudWatch Integration

Configure comprehensive CloudWatch integration:

```bash
# Install CloudWatch Container Insights
curl -O https://raw.githubusercontent.com/aws-samples/amazon-cloudwatch-container-insights/latest/k8s-deployment-manifest-templates/deployment-mode/daemonset/container-insights-monitoring/quickstart/cwagent-fluentd-quickstart.yaml

# Update with cluster name and region
sed -i.bak -e "s/{{cluster_name}}/genops-production/" -e "s/{{region_name}}/us-west-2/" cwagent-fluentd-quickstart.yaml

kubectl apply -f cwagent-fluentd-quickstart.yaml

# Create custom CloudWatch dashboard
cat > genops-dashboard.json << 'EOF'
{
  "widgets": [
    {
      "type": "metric",
      "properties": {
        "metrics": [
          ["AWS/EKS", "cluster_failed_request_count", "ClusterName", "genops-production"],
          [".", "cluster_request_count", ".", "."]
        ],
        "region": "us-west-2",
        "title": "EKS API Server Metrics",
        "period": 300
      }
    },
    {
      "type": "metric", 
      "properties": {
        "metrics": [
          ["GenOps/Production", "AIRequestCount"],
          [".", "CostPerHour"],
          [".", "ActiveUsers"]
        ],
        "region": "us-west-2",
        "title": "GenOps Usage Metrics",
        "period": 300
      }
    }
  ]
}
EOF

# Create dashboard
aws cloudwatch put-dashboard \
  --dashboard-name "GenOps-EKS-Production" \
  --dashboard-body file://genops-dashboard.json
```

### Cost Explorer Integration

Set up automated cost tracking and budgets:

```bash
# Create cost budget
cat > genops-budget.json << 'EOF'
{
  "Budget": {
    "BudgetName": "GenOps-EKS-Monthly",
    "BudgetLimit": {
      "Amount": "1000",
      "Unit": "USD"
    },
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST",
    "CostFilters": {
      "TagKey": [
        "Project"
      ],
      "TagValue": [
        "genops-ai"
      ]
    }
  },
  "NotificationsWithSubscribers": [
    {
      "Notification": {
        "NotificationType": "ACTUAL",
        "ComparisonOperator": "GREATER_THAN",
        "Threshold": 80
      },
      "Subscribers": [
        {
          "SubscriptionType": "EMAIL",
          "Address": "platform-team@company.com"
        }
      ]
    }
  ]
}
EOF

# Create the budget
aws budgets create-budget --account-id $(aws sts get-caller-identity --query Account --output text) --budget file://genops-budget.json
```

## Cost Management

### Instance Right-Sizing

Optimize EKS node groups for cost efficiency:

```bash
# Create mixed instance node group
cat > cost-optimized-nodegroup.yaml << 'EOF'
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: genops-production
  region: us-west-2

managedNodeGroups:
  - name: cost-optimized
    instanceTypes: ["m5.large", "m5.xlarge", "c5.large", "c5.xlarge", "t3.large"]
    spot: true
    desiredCapacity: 3
    minSize: 1
    maxSize: 20
    
    # Mixed instance policy for cost optimization
    mixedInstancesPolicy:
      instanceTypes: ["m5.large", "m5.xlarge", "c5.large", "c5.xlarge"]
      onDemandBaseCapacity: 1
      onDemandPercentageAboveBaseCapacity: 25
      spotInstancePools: 4
      
    labels:
      cost-optimization: enabled
      workload-type: mixed
      
    tags:
      CostOptimization: enabled
      AutoScaling: enabled
EOF

# Update node group
eksctl create nodegroup -f cost-optimized-nodegroup.yaml
```

### Cluster Autoscaler

Deploy cluster autoscaler for cost optimization:

```bash
# Install cluster autoscaler
curl -O https://raw.githubusercontent.com/kubernetes/autoscaler/master/cluster-autoscaler/cloudprovider/aws/examples/cluster-autoscaler-autodiscover.yaml

# Update with cluster name
sed -i.bak -e "s/<YOUR CLUSTER NAME>/genops-production/" cluster-autoscaler-autodiscover.yaml

# Add cost optimization annotations
kubectl annotate deployment cluster-autoscaler \
  cluster-autoscaler.kubernetes.io/safe-to-evict="false" \
  -n kube-system

kubectl apply -f cluster-autoscaler-autodiscover.yaml

# Configure cost-aware scaling
kubectl patch deployment cluster-autoscaler \
  -n kube-system \
  -p='{"spec":{"template":{"spec":{"containers":[{"name":"cluster-autoscaler","command":["./cluster-autoscaler","--v=4","--stderrthreshold=info","--cloud-provider=aws","--skip-nodes-with-local-storage=false","--expander=least-waste","--node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/genops-production","--balance-similar-node-groups","--skip-nodes-with-system-pods=false","--scale-down-enabled=true","--scale-down-delay-after-add=10m","--scale-down-unneeded-time=10m"]}]}}}}'
```

### Cost Monitoring Dashboard

Create a comprehensive cost monitoring setup:

```bash
# Create cost monitoring namespace
kubectl create namespace cost-monitoring

# Deploy cost monitoring stack
cat > cost-monitoring.yaml << 'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: aws-cost-exporter
  namespace: cost-monitoring
spec:
  replicas: 1
  selector:
    matchLabels:
      app: aws-cost-exporter
  template:
    metadata:
      labels:
        app: aws-cost-exporter
    spec:
      serviceAccountName: aws-cost-exporter
      containers:
      - name: aws-cost-exporter
        image: genopsai/aws-cost-exporter:latest
        env:
        - name: AWS_REGION
          value: us-west-2
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
  name: aws-cost-exporter
  namespace: cost-monitoring
  labels:
    app: aws-cost-exporter
spec:
  ports:
  - port: 9090
    targetPort: 9090
  selector:
    app: aws-cost-exporter
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: aws-cost-exporter
  namespace: cost-monitoring
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::ACCOUNT_ID:role/genops-cost-exporter-role
EOF

kubectl apply -f cost-monitoring.yaml
```

## Security & Compliance

### IAM Roles and Policies

Create least-privilege IAM configuration:

```bash
# Create GenOps service role
cat > genops-service-role-policy.json << 'EOF'
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream",
                "bedrock:ListFoundationModels",
                "bedrock:GetFoundationModel"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "ce:GetCostAndUsage",
                "ce:GetDimensionValues",
                "ce:GetReservationCoverage",
                "ce:GetReservationPurchaseRecommendation",
                "ce:GetUsageReport"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject"
            ],
            "Resource": "arn:aws:s3:::genops-governance-data/*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "ssm:GetParameter",
                "ssm:PutParameter",
                "ssm:GetParameters",
                "ssm:GetParametersByPath"
            ],
            "Resource": "arn:aws:ssm:us-west-2:*:parameter/genops/production/*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "secretsmanager:GetSecretValue"
            ],
            "Resource": "arn:aws:secretsmanager:us-west-2:*:secret:genops-ai-keys*"
        }
    ]
}
EOF

# Create the role
aws iam create-role \
  --role-name genops-service-role \
  --assume-role-policy-document file://eks-service-account-trust-policy.json

aws iam put-role-policy \
  --role-name genops-service-role \
  --policy-name genops-service-policy \
  --policy-document file://genops-service-role-policy.json
```

### Network Security

Configure VPC security and network policies:

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
      port: 443  # HTTPS to AWS APIs
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
# Create pod security policy
cat > genops-pod-security.yaml << 'EOF'
apiVersion: v1
kind: Namespace
metadata:
  name: genops-system
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
---
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
EOF

kubectl apply -f genops-pod-security.yaml
```

## Monitoring & Observability

### Comprehensive Monitoring Stack

Deploy full observability stack for GenOps:

```bash
# Create monitoring namespace
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
  --set storage.type=elasticsearch \
  --set elasticsearch.deploy=true

# Configure GenOps metrics endpoint
kubectl apply -f - << 'EOF'
apiVersion: v1
kind: Service
metadata:
  name: genops-metrics
  namespace: genops-system
  labels:
    app: genops-ai
spec:
  ports:
  - name: metrics
    port: 8080
    targetPort: 8080
  selector:
    app: genops-ai
EOF
```

### Custom Dashboards

Create GenOps-specific Grafana dashboards:

```bash
# Create GenOps dashboard configmap
cat > genops-grafana-dashboard.json << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "GenOps AI - AWS EKS",
    "tags": ["genops", "ai", "cost", "governance"],
    "timezone": "browser",
    "panels": [
      {
        "title": "AI API Requests",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(genops_ai_requests_total[5m])) by (provider)",
            "legendFormat": "{{provider}}"
          }
        ]
      },
      {
        "title": "Cost per Hour",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(genops_cost_per_hour) by (team, project)",
            "legendFormat": "{{team}}/{{project}}"
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
      },
      {
        "title": "Budget Utilization",
        "type": "singlestat",
        "targets": [
          {
            "expr": "(sum(genops_budget_used) / sum(genops_budget_limit)) * 100",
            "legendFormat": "Budget Used %"
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

kubectl create configmap genops-dashboard \
  --from-file=dashboard.json=genops-grafana-dashboard.json \
  --namespace monitoring
```

### AWS X-Ray Integration

Configure distributed tracing with X-Ray:

```bash
# Deploy X-Ray daemon
kubectl apply -f - << 'EOF'
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: xray-daemon
  namespace: genops-system
spec:
  selector:
    matchLabels:
      app: xray-daemon
  template:
    metadata:
      labels:
        app: xray-daemon
    spec:
      serviceAccountName: xray-daemon
      containers:
      - name: xray-daemon
        image: amazon/aws-xray-daemon:latest
        command:
          - /usr/bin/xray
          - -b
          - 0.0.0.0:2000
          - -o
        ports:
        - name: xray-ingest
          containerPort: 2000
          protocol: UDP
        - name: xray-tcp
          containerPort: 2000
          protocol: TCP
        resources:
          requests:
            cpu: 100m
            memory: 256Mi
          limits:
            cpu: 200m
            memory: 512Mi
        env:
        - name: AWS_REGION
          value: us-west-2
---
apiVersion: v1
kind: Service
metadata:
  name: xray-daemon
  namespace: genops-system
spec:
  selector:
    app: xray-daemon
  ports:
  - name: xray-ingest
    port: 2000
    protocol: UDP
  - name: xray-tcp
    port: 2000
    protocol: TCP
EOF
```

## Production Optimizations

### High Availability Configuration

Configure GenOps for high availability:

```bash
# Update GenOps deployment for HA
kubectl patch deployment genops-ai \
  -n genops-system \
  -p='{"spec":{"replicas":3,"template":{"spec":{"affinity":{"podAntiAffinity":{"requiredDuringSchedulingIgnoredDuringExecution":[{"labelSelector":{"matchExpressions":[{"key":"app","operator":"In","values":["genops-ai"]}]},"topologyKey":"kubernetes.io/hostname"}]}}}}}}'

# Configure pod disruption budget
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
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 20
        periodSeconds: 60
EOF

# Vertical Pod Autoscaler (optional)
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
# Install Velero for backup
helm repo add vmware-tanzu https://vmware-tanzu.github.io/helm-charts/
helm install velero vmware-tanzu/velero \
  --namespace velero \
  --create-namespace \
  --set-file credentials.secretContents.cloud=aws-credentials \
  --set configuration.provider=aws \
  --set configuration.backupStorageLocation.name=aws \
  --set configuration.backupStorageLocation.bucket=genops-backup-bucket \
  --set configuration.backupStorageLocation.config.region=us-west-2 \
  --set snapshotsEnabled=true \
  --set configuration.volumeSnapshotLocation.name=aws \
  --set configuration.volumeSnapshotLocation.config.region=us-west-2

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
    storageLocation: aws
    volumeSnapshotLocations:
    - aws
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
```

**Solutions:**
1. **Resource Constraints:**
   ```bash
   # Check resource availability
   kubectl top nodes
   kubectl describe nodes
   
   # Increase node capacity
   eksctl scale nodegroup --cluster=genops-production --name=general-purpose --nodes=5
   ```

2. **Pod Security Policy:**
   ```bash
   # Check security context
   kubectl get pod <pod-name> -o yaml | grep -A 10 securityContext
   
   # Update security context if needed
   kubectl patch deployment genops-ai -n genops-system --patch '{"spec":{"template":{"spec":{"securityContext":{"runAsUser":10001}}}}}'
   ```

#### Issue: High Cost Alerts

**Diagnosis:**
```bash
# Check current costs
aws ce get-cost-and-usage \
  --time-period Start=2024-01-01,End=2024-01-31 \
  --granularity MONTHLY \
  --metrics BlendedCost \
  --group-by Type=DIMENSION,Key=SERVICE

# Check GenOps metrics
kubectl logs -n genops-system deployment/genops-ai | grep -i cost
```

**Solutions:**
1. **Enable Spot Instances:**
   ```bash
   # Scale up spot instance node group
   eksctl scale nodegroup --cluster=genops-production --name=spot-workers --nodes=3
   
   # Migrate workloads to spot nodes
   kubectl patch deployment genops-ai -n genops-system --patch '{"spec":{"template":{"spec":{"tolerations":[{"key":"spot-instance","operator":"Equal","value":"true","effect":"NoSchedule"}]}}}}'
   ```

2. **Right-size Resources:**
   ```bash
   # Check current resource usage
   kubectl top pods -n genops-system
   
   # Update resource requests/limits
   kubectl patch deployment genops-ai -n genops-system --patch '{"spec":{"template":{"spec":{"containers":[{"name":"genops-ai","resources":{"requests":{"cpu":"100m","memory":"256Mi"},"limits":{"cpu":"300m","memory":"512Mi"}}}]}}}}'
   ```

#### Issue: API Rate Limiting

**Diagnosis:**
```bash
# Check rate limiting logs
kubectl logs -n genops-system deployment/genops-proxy | grep -i "rate limit"

# Check current request rates
kubectl exec -n genops-system deployment/genops-ai -- curl localhost:8080/metrics | grep genops_requests_per_second
```

**Solutions:**
1. **Increase Rate Limits:**
   ```bash
   # Update rate limit configuration
   kubectl patch configmap genops-config -n genops-system --patch '{"data":{"rate_limit":"2000"}}'
   
   # Restart deployment
   kubectl rollout restart deployment/genops-ai -n genops-system
   ```

2. **Scale Proxy Tier:**
   ```bash
   # Scale proxy deployment
   kubectl scale deployment genops-proxy --replicas=5 -n genops-system
   ```

### Health Checks and Validation

```bash
# Comprehensive health check script
cat > health-check.sh << 'EOF'
#!/bin/bash
echo "🔍 GenOps EKS Health Check"
echo "=========================="

# Check cluster health
echo "📋 Cluster Status:"
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

# Check logs for errors
echo -e "\n🔍 Recent Errors:"
kubectl logs -n genops-system deployment/genops-ai --tail=20 | grep -i error || echo "No errors found"

# Check AWS integration
echo -e "\n☁️ AWS Integration:"
aws eks describe-cluster --name genops-production --query 'cluster.status'
aws ce get-cost-and-usage --time-period Start=$(date -d '7 days ago' +%Y-%m-%d),End=$(date +%Y-%m-%d) --granularity DAILY --metrics BlendedCost --output table

echo -e "\n✅ Health check complete"
EOF

chmod +x health-check.sh
./health-check.sh
```

### Performance Optimization

```bash
# Enable performance monitoring
kubectl apply -f - << 'EOF'
apiVersion: v1
kind: ConfigMap
metadata:
  name: genops-performance-config
  namespace: genops-system
data:
  performance.yaml: |
    monitoring:
      enabled: true
      sample_rate: 0.1
      metrics_interval: 30s
    
    optimization:
      connection_pooling: true
      request_batching: true
      cache_enabled: true
      cache_ttl: 300s
    
    aws:
      request_timeout: 30s
      retry_attempts: 3
      connection_pool_size: 50
EOF

# Apply performance settings
kubectl rollout restart deployment/genops-ai -n genops-system
```

---

## Next Steps

1. **Set up monitoring alerts** for cost thresholds and performance metrics
2. **Implement GitOps workflow** with ArgoCD for automated deployments  
3. **Configure multi-region setup** for disaster recovery
4. **Optimize costs** with Reserved Instances and Savings Plans
5. **Enable advanced security** with GuardDuty and Security Hub

## Additional Resources

- [AWS EKS Best Practices Guide](https://aws.github.io/aws-eks-best-practices/)
- [GenOps AI Documentation](https://docs.genops.ai)
- [Kubernetes Cost Optimization](https://kubernetes.io/docs/concepts/cluster-administration/manage-deployment/#cost-optimization)
- [AWS Cost Management](https://aws.amazon.com/aws-cost-management/)

This guide provides a comprehensive foundation for deploying GenOps AI on Amazon EKS with production-ready configurations, cost optimization, and enterprise security.