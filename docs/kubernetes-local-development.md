# GenOps AI Local Development Setup

Complete guide for setting up GenOps AI in local Kubernetes environments. Perfect for development, testing, and learning before deploying to production clusters.

## 🎯 Overview

This guide covers:
- **kind (Kubernetes in Docker)** - Recommended for development
- **minikube** - Great for testing different Kubernetes versions  
- **Docker Desktop** - Built-in Kubernetes for Mac/Windows users
- **Development workflows** - Hot reloading, debugging, testing

## 🚀 Quick Start - kind (Recommended)

### Prerequisites
- Docker installed and running
- `kubectl` CLI tool
- 8GB+ RAM available for Docker

### 1. Install kind

**macOS:**
```bash
# Using Homebrew
brew install kind

# Or using go
go install sigs.k8s.io/kind@v0.20.0
```

**Linux:**
```bash
# Download binary
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind
```

**Windows:**
```powershell
# Using Chocolatey
choco install kind

# Or download from GitHub releases
```

### 2. Create GenOps Development Cluster

```bash
# Create kind config for GenOps development
cat <<EOF > genops-kind-config.yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: genops-dev
nodes:
- role: control-plane
  kubeadmConfigPatches:
  - |
    kind: InitConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        node-labels: "genops.ai/node-type=development"
  extraPortMappings:
  - containerPort: 30080
    hostPort: 8080
    protocol: TCP
  - containerPort: 30090
    hostPort: 9090
    protocol: TCP
- role: worker
- role: worker
EOF

# Create cluster
kind create cluster --config genops-kind-config.yaml

# Verify cluster
kubectl cluster-info --context kind-genops-dev
kubectl get nodes
```

### 3. Install GenOps AI

```bash
# Add Helm repository
helm repo add genops https://charts.genops.ai
helm repo update

# Install with development settings
helm install genops genops/genops-ai \
  --namespace genops-system \
  --create-namespace \
  --set global.environment=development \
  --set deployment.replicaCount=1 \
  --set resources.requests.cpu=100m \
  --set resources.requests.memory=256Mi \
  --set resources.limits.cpu=500m \
  --set resources.limits.memory=512Mi \
  --set service.type=NodePort \
  --set service.nodePort=30080
```

### 4. Verify Installation

```bash
# Wait for GenOps to be ready
kubectl wait --for=condition=available --timeout=300s deployment/genops-ai -n genops-system

# Test local access
curl http://localhost:8080/health
# Should return: {"status": "healthy", "kubernetes": true}

# Check logs
kubectl logs -n genops-system deployment/genops-ai
```

### ✅ Quick Start Complete!

GenOps AI is now running locally at `http://localhost:8080`

---

## 🏗️ Alternative Local Setups

### Option B: minikube

**Great for:** Testing different Kubernetes versions, resource constraints

```bash
# Install minikube (macOS)
brew install minikube

# Start with sufficient resources
minikube start \
  --cpus=4 \
  --memory=8192 \
  --kubernetes-version=v1.28.0 \
  --profile=genops-dev

# Enable addons
minikube addons enable ingress
minikube addons enable metrics-server

# Install GenOps
kubectl config use-context genops-dev
helm install genops genops/genops-ai \
  --namespace genops-system \
  --create-namespace \
  --set global.environment=development

# Access GenOps
minikube service genops-ai -n genops-system --url
```

### Option C: Docker Desktop Kubernetes

**Great for:** Windows/Mac users who already have Docker Desktop

```bash
# Enable Kubernetes in Docker Desktop settings
# Then deploy GenOps with LoadBalancer service

kubectl config use-context docker-desktop

helm install genops genops/genops-ai \
  --namespace genops-system \
  --create-namespace \
  --set global.environment=development \
  --set service.type=LoadBalancer

# Wait for external IP (localhost)
kubectl get services -n genops-system -w
```

---

## 🛠️ Development Workflow

### Hot Development Setup

**1. Clone GenOps AI Repository**
```bash
git clone https://github.com/KoshiHQ/GenOps-AI.git
cd GenOps-AI
```

**2. Set up Local Python Environment**
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install in development mode
pip install -e .
pip install -e ".[dev]"  # Includes testing dependencies
```

**3. Configure Environment Variables**
```bash
# Create local environment file
cat > .env.local <<EOF
# API Keys (add your own)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Development settings
GENOPS_ENV=development
LOG_LEVEL=DEBUG
KUBERNETES_SERVICE_HOST=localhost
KUBERNETES_SERVICE_PORT=6443

# OpenTelemetry (optional)
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=genops-ai-dev

# Team attribution for testing
DEFAULT_TEAM=dev-team
PROJECT_NAME=local-development
EOF

# Load environment
source .env.local
```

### Development Container Setup

**Option 1: Build and Deploy Custom Image**
```bash
# Build development image
docker build -t genops-ai:dev .

# Load image into kind
kind load docker-image genops-ai:dev --name genops-dev

# Deploy with custom image
helm upgrade genops genops/genops-ai \
  --namespace genops-system \
  --set image.repository=genops-ai \
  --set image.tag=dev \
  --set image.pullPolicy=IfNotPresent
```

**Option 2: Live Development with Skaffold**
```bash
# Install Skaffold
brew install skaffold  # macOS
# or download from https://skaffold.dev/docs/install/

# Create skaffold config
cat > skaffold.yaml <<EOF
apiVersion: skaffold/v2beta29
kind: Config
metadata:
  name: genops-dev
build:
  artifacts:
  - image: genops-ai-dev
    context: .
    docker:
      dockerfile: Dockerfile.dev
deploy:
  helm:
    releases:
    - name: genops
      chartPath: charts/genops-ai
      namespace: genops-system
      setValues:
        image.repository: genops-ai-dev
        global.environment: development
EOF

# Start live development
skaffold dev --port-forward
```

### Testing Your Changes

**1. Unit Tests**
```bash
# Run unit tests
pytest tests/unit/

# Run with coverage
pytest tests/unit/ --cov=src/genops --cov-report=html
```

**2. Integration Tests with Local Cluster**
```bash
# Run integration tests against local cluster
export KUBECONFIG=~/.kube/config
export GENOPS_TEST_NAMESPACE=genops-system
export GENOPS_TEST_ENDPOINT=http://localhost:8080

pytest tests/integration/
```

**3. Example Validation**
```bash
# Test examples against local cluster
cd examples/kubernetes/

# Run validation
python setup_validation.py --detailed

# Test auto-instrumentation  
python auto_instrumentation.py --demo-only

# Test cost tracking
python cost_tracking.py --budget 10.00
```

---

## 🔧 Development Tools Integration

### VS Code Setup

**1. Install Extensions**
```bash
code --install-extension ms-python.python
code --install-extension ms-kubernetes-tools.vscode-kubernetes-tools
code --install-extension ms-vscode.vscode-json
```

**2. Create VS Code Configuration**
```bash
mkdir -p .vscode

# Launch configuration
cat > .vscode/launch.json <<EOF
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug GenOps AI",
            "type": "python",
            "request": "launch",
            "module": "genops.main",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src",
                "LOG_LEVEL": "DEBUG",
                "GENOPS_ENV": "development"
            },
            "console": "integratedTerminal"
        },
        {
            "name": "Debug Tests",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": [
                "tests/",
                "-v",
                "--tb=short"
            ],
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            },
            "console": "integratedTerminal"
        }
    ]
}
EOF

# Tasks configuration
cat > .vscode/tasks.json <<EOF
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Deploy to kind",
            "type": "shell",
            "command": "docker build -t genops-ai:dev . && kind load docker-image genops-ai:dev --name genops-dev && helm upgrade genops charts/genops-ai --set image.tag=dev",
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always"
            }
        },
        {
            "label": "Port Forward GenOps",
            "type": "shell",
            "command": "kubectl port-forward -n genops-system service/genops-ai 8080:8000",
            "isBackground": true,
            "group": "test"
        }
    ]
}
EOF
```

### PyCharm Setup

**1. Configure Run Configuration**
- Go to Run → Edit Configurations
- Add new Python configuration:
  - Name: "GenOps AI Debug"
  - Script path: `src/genops/main.py`
  - Environment variables: `LOG_LEVEL=DEBUG;GENOPS_ENV=development`
  - Working directory: project root

### Git Hooks for Development

```bash
# Install pre-commit hooks
pip install pre-commit
cat > .pre-commit-config.yaml <<EOF
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
    -   id: check-yaml
    -   id: check-added-large-files
-   repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
    -   id: black
        language_version: python3
-   repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.0.254
    hooks:
    -   id: ruff
        args: [--fix]
-   repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.1
    hooks:
    -   id: mypy
        additional_dependencies: [types-requests]
EOF

pre-commit install
```

---

## 🧪 Advanced Development Scenarios

### Multi-Provider Development

**Set up multiple AI providers for testing:**
```bash
# Create development secrets
kubectl create secret generic genops-secrets \
  --namespace genops-system \
  --from-literal=openai-api-key="$OPENAI_API_KEY" \
  --from-literal=anthropic-api-key="$ANTHROPIC_API_KEY" \
  --from-literal=azure-api-key="$AZURE_OPENAI_API_KEY" \
  --dry-run=client -o yaml | kubectl apply -f -

# Enable all providers
helm upgrade genops genops/genops-ai \
  --namespace genops-system \
  --set providers.openai.enabled=true \
  --set providers.anthropic.enabled=true \
  --set providers.azure.enabled=true
```

### Monitoring Stack Development

**Deploy full monitoring stack locally:**
```bash
# Install Prometheus stack
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set grafana.service.type=NodePort \
  --set grafana.service.nodePort=30090 \
  --set prometheus.service.type=NodePort

# Access Grafana at http://localhost:9090
# Default credentials: admin / prom-operator
```

### Database Development

**Set up local database for development:**
```bash
# Deploy PostgreSQL for testing
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install postgres bitnami/postgresql \
  --namespace genops-system \
  --set auth.postgresPassword=devpassword \
  --set primary.service.type=NodePort \
  --set primary.service.nodePorts.postgresql=30432

# Update GenOps to use local database
helm upgrade genops genops/genops-ai \
  --namespace genops-system \
  --set database.enabled=true \
  --set database.host=postgres-postgresql \
  --set database.password=devpassword
```

---

## 📊 Performance Testing

### Local Load Testing

```bash
# Install hey for load testing
go install github.com/rakyll/hey@latest

# Test GenOps performance
hey -z 60s -c 10 \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"test"}],"max_tokens":1}' \
  http://localhost:8080/chat/completions
```

### Resource Usage Monitoring

```bash
# Monitor resource usage in real-time
watch kubectl top pods -n genops-system

# Get detailed resource metrics
kubectl describe pod -n genops-system -l app.kubernetes.io/name=genops-ai
```

---

## 🧹 Cleanup and Reset

### Clean Development Environment

```bash
# Delete GenOps deployment
helm uninstall genops -n genops-system

# Delete namespace
kubectl delete namespace genops-system

# Reset kind cluster
kind delete cluster --name genops-dev

# Clean Docker images
docker image prune -f
docker volume prune -f
```

### Reset to Clean State

```bash
# Complete cleanup script
cat > cleanup-dev-env.sh <<'EOF'
#!/bin/bash
echo "🧹 Cleaning GenOps development environment..."

# Stop any running port-forwards
pkill -f "kubectl port-forward"

# Delete Helm releases
helm uninstall genops -n genops-system 2>/dev/null || true
helm uninstall prometheus -n monitoring 2>/dev/null || true

# Delete namespaces
kubectl delete namespace genops-system monitoring --ignore-not-found

# Delete kind cluster
kind delete cluster --name genops-dev 2>/dev/null || true

# Clean Docker
docker system prune -f

echo "✅ Development environment cleaned!"
EOF

chmod +x cleanup-dev-env.sh
./cleanup-dev-env.sh
```

---

## 🚀 Next Steps

### Ready for Production?
- **[Production Deployment Guide](kubernetes-getting-started.md#-phase-3-production-mastery-2-hours)**
- **[Security Hardening](kubernetes-security.md)**
- **[Multi-Cloud Deployment](kubernetes-multi-cloud.md)**

### Advanced Development
- **[Custom Provider Development](provider-development.md)**
- **[Operator Development](operator-development.md)**
- **[Contributing Guide](../CONTRIBUTING.md)**

---

## 📚 Troubleshooting Local Development

### Common Issues

**kind cluster won't start:**
```bash
# Check Docker resources
docker system df
docker system prune  # if needed

# Restart Docker Desktop
# Then recreate cluster
```

**GenOps pods stuck in Pending:**
```bash
# Check node resources
kubectl describe nodes

# Check resource requests
kubectl describe pod -n genops-system -l app.kubernetes.io/name=genops-ai
```

**Port forwarding not working:**
```bash
# Kill existing port forwards
pkill -f "kubectl port-forward"

# Restart with verbose output
kubectl port-forward -n genops-system service/genops-ai 8080:8000 --v=6
```

**Hot reloading not working with Skaffold:**
```bash
# Check Skaffold logs
skaffold dev --verbosity=debug

# Verify Docker daemon connection
docker ps
```

### Getting Help

- **[Troubleshooting Guide](kubernetes-troubleshooting.md)**: Comprehensive issue resolution
- **[GitHub Discussions](https://github.com/KoshiHQ/GenOps-AI/discussions)**: Development questions
- **[Discord #development](https://discord.gg/genops-ai)**: Real-time help

---

**🎉 Happy Developing!** You now have a complete local GenOps AI development environment. Start building, testing, and contributing!