# CI/CD Integration for GenOps AI on Kubernetes

Complete guide for implementing production-grade CI/CD pipelines, GitOps workflows, and automated deployment strategies for GenOps AI with governance validation.

## Table of Contents

1. [Quick Start (5 minutes)](#quick-start)
2. [GitOps Fundamentals](#gitops-fundamentals)
3. [CI/CD Pipeline Patterns](#cicd-pipeline-patterns)
4. [Automated Testing](#automated-testing)
5. [Deployment Strategies](#deployment-strategies)
6. [Helm Chart Management](#helm-chart-management)
7. [Security & Compliance](#security-compliance)
8. [Troubleshooting](#troubleshooting)

## Quick Start

Deploy GenOps AI with GitOps in 5 minutes:

```bash
# 1. Install ArgoCD
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# 2. Create GenOps Application
kubectl apply -f - <<EOF
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: genops-ai
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/KoshiHQ/GenOps-AI
    targetRevision: main
    path: k8s/base
  destination:
    server: https://kubernetes.default.svc
    namespace: genops-system
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
EOF

# 3. Access ArgoCD UI
kubectl port-forward svc/argocd-server -n argocd 8080:443
```

✅ **Result:** ArgoCD continuously deploys GenOps AI from Git repository.

## GitOps Fundamentals

### ArgoCD Installation and Configuration

Deploy ArgoCD for GitOps continuous delivery:

```bash
# Install ArgoCD
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Wait for ArgoCD to be ready
kubectl wait --for=condition=Ready pods --all -n argocd --timeout=300s

# Get admin password
ARGO_PASSWORD=$(kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d)
echo "ArgoCD Admin Password: $ARGO_PASSWORD"

# Access ArgoCD UI
kubectl port-forward svc/argocd-server -n argocd 8080:443
```

**Configure ArgoCD for GenOps:**

```yaml
# argocd-genops-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: argocd-cm
  namespace: argocd
data:
  # Enable resource tracking
  application.resourceTrackingMethod: annotation

  # Configure repositories
  repositories: |
    - url: https://github.com/KoshiHQ/GenOps-AI
      name: genops-main
      type: git
    - url: https://ghcr.io/koshihq/helm-charts
      name: genops-helm
      type: helm

  # Configure resource exclusions
  resource.exclusions: |
    - apiGroups:
      - cilium.io
      kinds:
      - CiliumIdentity
      clusters:
      - "*"

  # Configure sync options
  resource.customizations: |
    batch/Job:
      health.lua: |
        hs = {}
        if obj.status ~= nil then
          if obj.status.succeeded ~= nil and obj.status.succeeded > 0 then
            hs.status = "Healthy"
            hs.message = "Job completed successfully"
            return hs
          end
        end
        hs.status = "Progressing"
        hs.message = "Job in progress"
        return hs
```

Apply ArgoCD configuration:

```bash
kubectl apply -f argocd-genops-config.yaml

# Restart ArgoCD server to apply config
kubectl rollout restart deployment/argocd-server -n argocd
```

### FluxCD Continuous Delivery

Alternative GitOps with FluxCD:

```bash
# Install Flux CLI
curl -s https://fluxcd.io/install.sh | sudo bash

# Bootstrap Flux with GitHub
flux bootstrap github \
  --owner=YOUR_GITHUB_ORG \
  --repository=genops-infrastructure \
  --branch=main \
  --path=./clusters/production \
  --personal

# Verify Flux installation
flux check

# Create GitRepository source
flux create source git genops-ai \
  --url=https://github.com/KoshiHQ/GenOps-AI \
  --branch=main \
  --interval=1m \
  --export > genops-gitrepo.yaml

kubectl apply -f genops-gitrepo.yaml
```

**Create Flux Kustomization:**

```yaml
# genops-kustomization.yaml
apiVersion: kustomize.toolkit.fluxcd.io/v1
kind: Kustomization
metadata:
  name: genops-ai
  namespace: flux-system
spec:
  interval: 5m
  path: ./k8s/overlays/production
  prune: true
  sourceRef:
    kind: GitRepository
    name: genops-ai
  healthChecks:
  - apiVersion: apps/v1
    kind: Deployment
    name: genops-ai
    namespace: genops-system
  timeout: 5m
  wait: true
```

Apply Flux Kustomization:

```bash
kubectl apply -f genops-kustomization.yaml

# Watch Flux reconciliation
flux get kustomizations --watch
```

### Kustomize Overlay Management

Structure for multi-environment deployments:

```bash
# Create Kustomize directory structure
mkdir -p k8s/{base,overlays/{dev,staging,production}}

# Base configuration
cat > k8s/base/kustomization.yaml <<'EOF'
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
- deployment.yaml
- service.yaml
- configmap.yaml

commonLabels:
  app: genops-ai
  managed-by: kustomize
EOF

# Base deployment
cat > k8s/base/deployment.yaml <<'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genops-ai
spec:
  replicas: 1  # Override in overlays
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
        image: genopsai/genops:latest
        ports:
        - containerPort: 8080
        env:
        - name: GENOPS_ENVIRONMENT
          value: "base"
        resources:
          requests:
            cpu: 100m
            memory: 256Mi
          limits:
            cpu: 500m
            memory: 512Mi
EOF

# Production overlay
cat > k8s/overlays/production/kustomization.yaml <<'EOF'
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: genops-system

bases:
- ../../base

patchesStrategicMerge:
- deployment-patch.yaml

configMapGenerator:
- name: genops-config
  behavior: merge
  literals:
  - GENOPS_ENVIRONMENT=production
  - GENOPS_LOG_LEVEL=info
  - OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector.monitoring:4318

replicas:
- name: genops-ai
  count: 3

images:
- name: genopsai/genops
  newTag: v1.0.0
EOF

# Production deployment patch
cat > k8s/overlays/production/deployment-patch.yaml <<'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genops-ai
spec:
  template:
    spec:
      containers:
      - name: genops-ai
        resources:
          requests:
            cpu: 1000m
            memory: 2Gi
          limits:
            cpu: 2000m
            memory: 4Gi
        env:
        - name: GENOPS_TEAM
          value: "platform-engineering"
        - name: GENOPS_COST_CENTER
          value: "engineering"
EOF
```

Build and preview Kustomize overlays:

```bash
# Build dev overlay
kustomize build k8s/overlays/dev

# Build production overlay
kustomize build k8s/overlays/production

# Apply production with kubectl
kubectl apply -k k8s/overlays/production

# Verify deployment
kubectl get all -n genops-system
```

### Git Repository Structure

Best practices for GitOps repositories:

```bash
genops-infrastructure/
├── README.md
├── .github/
│   └── workflows/
│       ├── validate.yml          # Validate manifests
│       └── sync.yml              # Trigger ArgoCD sync
│
├── apps/
│   ├── base/                     # Base application configs
│   │   └── genops-ai/
│   │       ├── kustomization.yaml
│   │       ├── deployment.yaml
│   │       ├── service.yaml
│   │       └── configmap.yaml
│   │
│   └── overlays/                 # Environment-specific overlays
│       ├── dev/
│       │   ├── kustomization.yaml
│       │   └── patches/
│       ├── staging/
│       │   ├── kustomization.yaml
│       │   └── patches/
│       └── production/
│           ├── kustomization.yaml
│           └── patches/
│
├── infrastructure/               # Infrastructure components
│   ├── argocd/
│   │   ├── applications/         # ArgoCD Applications
│   │   ├── projects/             # ArgoCD Projects
│   │   └── repositories/         # Repository credentials
│   ├── monitoring/
│   │   ├── prometheus/
│   │   └── grafana/
│   └── networking/
│       ├── ingress/
│       └── cert-manager/
│
├── helm-charts/                  # Custom Helm charts
│   └── genops-ai/
│       ├── Chart.yaml
│       ├── values.yaml
│       ├── templates/
│       └── values/
│           ├── dev-values.yaml
│           ├── staging-values.yaml
│           └── prod-values.yaml
│
└── scripts/
    ├── validate-manifests.sh
    ├── promote-to-staging.sh
    └── promote-to-production.sh
```

## CI/CD Pipeline Patterns

### GitHub Actions Complete Workflow

Production-ready GitHub Actions pipeline:

```yaml
# .github/workflows/ci-cd.yml
name: GenOps AI - CI/CD Pipeline

on:
  push:
    branches: [main, develop, 'feature/*']
  pull_request:
    branches: [main, develop]
  release:
    types: [published]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  # Job 1: Code quality and testing
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov ruff mypy

    - name: Lint with Ruff
      run: ruff check src/

    - name: Type check with mypy
      run: mypy src/

    - name: Run unit tests
      run: pytest tests/unit --cov=src --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage.xml

  # Job 2: Security scanning
  security:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'

    - name: Upload Trivy results to GitHub Security
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'

  # Job 3: Build and push Docker image
  build:
    needs: [test, security]
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    outputs:
      image-tag: ${{ steps.meta.outputs.tags }}

    steps:
    - uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=sha,prefix={{branch}}-

    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
        build-args: |
          VERSION=${{ github.sha }}
          BUILD_DATE=${{ github.event.head_commit.timestamp }}

    - name: Scan Docker image with Trivy
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: ${{ steps.meta.outputs.tags }}
        format: 'sarif'
        output: 'trivy-image-results.sarif'

  # Job 4: Deploy to Development
  deploy-dev:
    needs: build
    if: github.ref == 'refs/heads/develop'
    runs-on: ubuntu-latest
    environment:
      name: development
      url: https://genops-dev.example.com

    steps:
    - uses: actions/checkout@v4

    - name: Install kubectl
      uses: azure/setup-kubectl@v3

    - name: Configure kubectl
      run: |
        mkdir -p ~/.kube
        echo "${{ secrets.KUBE_CONFIG_DEV }}" | base64 -d > ~/.kube/config

    - name: Install Helm
      uses: azure/setup-helm@v3

    - name: Deploy to Development
      run: |
        helm upgrade --install genops-ai ./helm-charts/genops-ai \
          --namespace genops-dev \
          --create-namespace \
          --values helm-charts/genops-ai/values/dev-values.yaml \
          --set image.tag=${{ github.sha }} \
          --set deployment.timestamp=$(date +%s) \
          --wait \
          --timeout 10m

    - name: Verify deployment
      run: |
        kubectl rollout status deployment/genops-ai -n genops-dev --timeout=5m
        kubectl get pods -n genops-dev -l app=genops-ai

    - name: Run smoke tests
      run: |
        ENDPOINT=$(kubectl get svc genops-ai -n genops-dev -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
        curl -f http://$ENDPOINT:8080/health || exit 1

  # Job 5: Deploy to Staging
  deploy-staging:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment:
      name: staging
      url: https://genops-staging.example.com

    steps:
    - uses: actions/checkout@v4

    - name: Configure kubectl
      run: |
        mkdir -p ~/.kube
        echo "${{ secrets.KUBE_CONFIG_STAGING }}" | base64 -d > ~/.kube/config

    - name: Deploy to Staging
      run: |
        helm upgrade --install genops-ai ./helm-charts/genops-ai \
          --namespace genops-staging \
          --create-namespace \
          --values helm-charts/genops-ai/values/staging-values.yaml \
          --set image.tag=${{ github.sha }} \
          --wait

    - name: Run integration tests
      run: |
        kubectl run integration-test \
          --namespace genops-staging \
          --image=genopsai/integration-tests:latest \
          --restart=Never \
          --rm -i -- \
          --target http://genops-ai.genops-staging.svc.cluster.local:8080

  # Job 6: Deploy to Production (manual approval required)
  deploy-production:
    needs: [deploy-staging]
    if: github.event_name == 'release'
    runs-on: ubuntu-latest
    environment:
      name: production
      url: https://genops.example.com

    steps:
    - uses: actions/checkout@v4

    - name: Configure kubectl
      run: |
        mkdir -p ~/.kube
        echo "${{ secrets.KUBE_CONFIG_PROD }}" | base64 -d > ~/.kube/config

    - name: Deploy to Production with Blue-Green
      run: |
        # Deploy green environment
        helm upgrade --install genops-ai-green ./helm-charts/genops-ai \
          --namespace genops \
          --values helm-charts/genops-ai/values/prod-values.yaml \
          --set image.tag=${{ github.sha }} \
          --set service.selector.version=green \
          --wait

        # Wait for health checks
        kubectl wait --for=condition=Ready pods -l app=genops-ai,version=green -n genops --timeout=5m

        # Run smoke tests on green
        kubectl run prod-smoke-test --rm -i --restart=Never \
          --image=curlimages/curl:latest -- \
          curl -f http://genops-ai-green.genops.svc.cluster.local:8080/health

        # Switch traffic to green
        kubectl patch service genops-ai -n genops \
          --patch '{"spec":{"selector":{"version":"green"}}}'

        # Clean up blue environment after 5 minutes
        sleep 300
        helm uninstall genops-ai-blue -n genops || true

    - name: Notify deployment
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        text: 'GenOps AI deployed to production'
        webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### GitLab CI Pipeline

Complete GitLab CI/CD configuration:

```yaml
# .gitlab-ci.yml
stages:
- test
- build
- deploy-dev
- deploy-staging
- deploy-production

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"
  IMAGE_TAG: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA

# Test stage
test:unit:
  stage: test
  image: python:3.11
  script:
  - pip install -r requirements.txt pytest pytest-cov
  - pytest tests/unit --cov=src --cov-report=term --cov-report=xml
  coverage: '/(?i)total.*? (100(?:\.0+)?\%|[1-9]?\d(?:\.\d+)?\%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

test:lint:
  stage: test
  image: python:3.11
  script:
  - pip install ruff mypy
  - ruff check src/
  - mypy src/

test:security:
  stage: test
  image: aquasec/trivy:latest
  script:
  - trivy fs --exit-code 1 --severity HIGH,CRITICAL .

# Build stage
build:
  stage: build
  image: docker:latest
  services:
  - docker:dind
  before_script:
  - echo "$CI_REGISTRY_PASSWORD" | docker login -u "$CI_REGISTRY_USER" --password-stdin $CI_REGISTRY
  script:
  - docker build -t $IMAGE_TAG .
  - docker push $IMAGE_TAG
  - docker tag $IMAGE_TAG $CI_REGISTRY_IMAGE:latest
  - docker push $CI_REGISTRY_IMAGE:latest
  only:
  - main
  - develop

# Deploy to Development
deploy:dev:
  stage: deploy-dev
  image: alpine/helm:latest
  before_script:
  - kubectl config use-context genops/dev-cluster
  script:
  - |
    helm upgrade --install genops-ai ./helm-charts/genops-ai \
      --namespace genops-dev \
      --create-namespace \
      --values helm-charts/genops-ai/values/dev-values.yaml \
      --set image.tag=$CI_COMMIT_SHA \
      --wait
  - kubectl rollout status deployment/genops-ai -n genops-dev
  environment:
    name: development
    url: https://genops-dev.example.com
  only:
  - develop

# Deploy to Staging
deploy:staging:
  stage: deploy-staging
  image: alpine/helm:latest
  before_script:
  - kubectl config use-context genops/staging-cluster
  script:
  - |
    helm upgrade --install genops-ai ./helm-charts/genops-ai \
      --namespace genops-staging \
      --create-namespace \
      --values helm-charts/genops-ai/values/staging-values.yaml \
      --set image.tag=$CI_COMMIT_SHA \
      --wait
  - kubectl rollout status deployment/genops-ai -n genops-staging
  environment:
    name: staging
    url: https://genops-staging.example.com
  only:
  - main

# Deploy to Production (manual)
deploy:production:
  stage: deploy-production
  image: alpine/helm:latest
  before_script:
  - kubectl config use-context genops/prod-cluster
  script:
  - |
    helm upgrade --install genops-ai ./helm-charts/genops-ai \
      --namespace genops \
      --values helm-charts/genops-ai/values/prod-values.yaml \
      --set image.tag=$CI_COMMIT_SHA \
      --wait
  - kubectl rollout status deployment/genops-ai -n genops
  environment:
    name: production
    url: https://genops.example.com
  when: manual
  only:
  - main
```

### Jenkins Declarative Pipeline

```groovy
// Jenkinsfile
pipeline {
    agent any

    environment {
        DOCKER_REGISTRY = 'ghcr.io'
        IMAGE_NAME = 'koshihq/genops-ai'
        KUBECONFIG = credentials('kubernetes-config')
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Test') {
            parallel {
                stage('Unit Tests') {
                    steps {
                        sh '''
                            python -m venv venv
                            . venv/bin/activate
                            pip install -r requirements.txt pytest
                            pytest tests/unit
                        '''
                    }
                }

                stage('Lint') {
                    steps {
                        sh '''
                            . venv/bin/activate
                            pip install ruff
                            ruff check src/
                        '''
                    }
                }

                stage('Security Scan') {
                    steps {
                        sh '''
                            trivy fs --exit-code 1 --severity HIGH,CRITICAL .
                        '''
                    }
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    dockerImage = docker.build("${DOCKER_REGISTRY}/${IMAGE_NAME}:${env.BUILD_NUMBER}")
                }
            }
        }

        stage('Push Docker Image') {
            steps {
                script {
                    docker.withRegistry("https://${DOCKER_REGISTRY}", 'docker-registry-credentials') {
                        dockerImage.push("${env.BUILD_NUMBER}")
                        dockerImage.push("latest")
                    }
                }
            }
        }

        stage('Deploy to Development') {
            when {
                branch 'develop'
            }
            steps {
                sh '''
                    helm upgrade --install genops-ai ./helm-charts/genops-ai \
                      --namespace genops-dev \
                      --create-namespace \
                      --values helm-charts/genops-ai/values/dev-values.yaml \
                      --set image.tag=${BUILD_NUMBER} \
                      --wait
                '''
            }
        }

        stage('Deploy to Staging') {
            when {
                branch 'main'
            }
            steps {
                sh '''
                    helm upgrade --install genops-ai ./helm-charts/genops-ai \
                      --namespace genops-staging \
                      --create-namespace \
                      --values helm-charts/genops-ai/values/staging-values.yaml \
                      --set image.tag=${BUILD_NUMBER} \
                      --wait
                '''
            }
        }

        stage('Deploy to Production') {
            when {
                branch 'main'
            }
            steps {
                input message: 'Deploy to Production?', ok: 'Deploy'

                sh '''
                    helm upgrade --install genops-ai ./helm-charts/genops-ai \
                      --namespace genops \
                      --values helm-charts/genops-ai/values/prod-values.yaml \
                      --set image.tag=${BUILD_NUMBER} \
                      --wait

                    kubectl rollout status deployment/genops-ai -n genops --timeout=10m
                '''
            }
        }

        stage('Smoke Test') {
            when {
                branch 'main'
            }
            steps {
                sh '''
                    ENDPOINT=$(kubectl get svc genops-ai -n genops -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
                    curl -f http://$ENDPOINT:8080/health || exit 1
                '''
            }
        }
    }

    post {
        always {
            cleanWs()
        }
        success {
            slackSend(
                color: 'good',
                message: "Pipeline succeeded: ${env.JOB_NAME} ${env.BUILD_NUMBER}"
            )
        }
        failure {
            slackSend(
                color: 'danger',
                message: "Pipeline failed: ${env.JOB_NAME} ${env.BUILD_NUMBER}"
            )
        }
    }
}
```

### Governance Validation in CI

Validate budget and policy constraints before deployment:

```yaml
# .github/workflows/governance-validation.yml
name: Governance Validation

on:
  pull_request:
    branches: [main]

jobs:
  validate-budget:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Check Budget Impact
      run: |
        # Estimate deployment cost
        ESTIMATED_COST=$(python scripts/estimate-deployment-cost.py \
          --environment production \
          --replicas 3 \
          --instance-type m5.large)

        echo "Estimated monthly cost: \$$ESTIMATED_COST"

        # Check against budget
        BUDGET_LIMIT=10000
        if (( $(echo "$ESTIMATED_COST > $BUDGET_LIMIT" | bc -l) )); then
          echo "::error::Estimated cost \$$ESTIMATED_COST exceeds budget limit \$$BUDGET_LIMIT"
          exit 1
        fi

    - name: Validate Cost Attribution
      run: |
        # Ensure all resources have cost labels
        for file in k8s/**/*.yaml; do
          if ! grep -q "genops.ai/team" "$file"; then
            echo "::error::Missing team label in $file"
            exit 1
          fi
          if ! grep -q "genops.ai/cost-center" "$file"; then
            echo "::error::Missing cost-center label in $file"
            exit 1
          fi
        done

  validate-policy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4

    - name: Install Conftest
      run: |
        wget https://github.com/open-policy-agent/conftest/releases/download/v0.45.0/conftest_0.45.0_Linux_x86_64.tar.gz
        tar xzf conftest_0.45.0_Linux_x86_64.tar.gz
        sudo mv conftest /usr/local/bin/

    - name: Validate Kubernetes Manifests
      run: |
        conftest test k8s/**/*.yaml -p policies/

    - name: Check Resource Limits
      run: |
        # Ensure all containers have resource limits
        for file in k8s/**/*.yaml; do
          if grep -q "kind: Deployment" "$file"; then
            if ! grep -A 20 "containers:" "$file" | grep -q "limits:"; then
              echo "::error::Missing resource limits in $file"
              exit 1
            fi
          fi
        done

    - name: Validate Security Context
      run: |
        # Ensure pods run as non-root
        for file in k8s/**/*.yaml; do
          if grep -q "kind: Deployment" "$file"; then
            if ! grep -A 30 "containers:" "$file" | grep -q "runAsNonRoot: true"; then
              echo "::warning::Pod should run as non-root in $file"
            fi
          fi
        done
```

## Automated Testing

### Unit Testing in CI

```yaml
# pytest configuration
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    --cov=src
    --cov-report=term-missing
    --cov-report=html
    --cov-fail-under=80
    -v
```

```python
# tests/unit/test_genops_core.py
import pytest
from genops import track_usage
from genops.core import CostTracker

def test_cost_tracking():
    """Test basic cost tracking functionality"""
    tracker = CostTracker()

    @track_usage(team="test-team", project="test-project")
    def mock_operation():
        return "result"

    result = mock_operation()

    assert result == "result"
    assert tracker.get_cost("test-team", "test-project") >= 0

def test_budget_enforcement():
    """Test budget enforcement"""
    from genops.budget import BudgetEnforcer

    enforcer = BudgetEnforcer(limit=100.0)

    # Should allow under budget
    assert enforcer.check_budget(50.0) == True

    # Should block over budget
    with pytest.raises(BudgetExhausted):
        enforcer.check_budget(200.0)
```

### Integration Testing

```bash
# scripts/integration-test.sh
#!/bin/bash

set -e

echo "Creating test cluster..."
kind create cluster --name genops-test

echo "Installing GenOps AI..."
helm install genops-ai ./helm-charts/genops-ai \
  --namespace genops-test \
  --create-namespace \
  --values helm-charts/genops-ai/values/test-values.yaml \
  --wait

echo "Waiting for pods to be ready..."
kubectl wait --for=condition=Ready pods --all -n genops-test --timeout=300s

echo "Running integration tests..."
kubectl run integration-test \
  --namespace genops-test \
  --image=genopsai/integration-tests:latest \
  --restart=Never \
  --rm -i -- \
  --target http://genops-ai.genops-test.svc.cluster.local:8080 \
  --test-suite integration

echo "Cleaning up..."
kind delete cluster --name genops-test

echo "✅ Integration tests passed"
```

### Helm Chart Validation

```bash
# Validate Helm chart
helm lint helm-charts/genops-ai

# Dry run
helm install genops-ai ./helm-charts/genops-ai \
  --namespace genops \
  --values helm-charts/genops-ai/values/prod-values.yaml \
  --dry-run \
  --debug

# Template validation
helm template genops-ai ./helm-charts/genops-ai \
  --values helm-charts/genops-ai/values/prod-values.yaml \
  | kubectl --dry-run=client -f -
```

### Policy Testing with Conftest

```rego
# policies/resource-limits.rego
package main

deny[msg] {
  input.kind == "Deployment"
  not input.spec.template.spec.containers[_].resources.limits
  msg = "Containers must have resource limits defined"
}

deny[msg] {
  input.kind == "Deployment"
  container := input.spec.template.spec.containers[_]
  not container.resources.limits.memory
  msg = sprintf("Container '%s' must have memory limit", [container.name])
}

deny[msg] {
  input.kind == "Deployment"
  container := input.spec.template.spec.containers[_]
  not container.resources.limits.cpu
  msg = sprintf("Container '%s' must have CPU limit", [container.name])
}
```

```bash
# Test policies
conftest test k8s/base/deployment.yaml -p policies/
```

## Deployment Strategies

### Blue-Green Deployment with Argo Rollouts

```yaml
# blue-green-rollout.yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: genops-ai
  namespace: genops-system
spec:
  replicas: 3
  revisionHistoryLimit: 2
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
        image: genopsai/genops:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            cpu: 500m
            memory: 1Gi
          limits:
            cpu: 1000m
            memory: 2Gi

  strategy:
    blueGreen:
      activeService: genops-ai
      previewService: genops-ai-preview
      autoPromotionEnabled: false
      scaleDownDelaySeconds: 300
      prePromotionAnalysis:
        templates:
        - templateName: success-rate
        - templateName: response-time
      postPromotionAnalysis:
        templates:
        - templateName: error-rate
---
apiVersion: v1
kind: Service
metadata:
  name: genops-ai
  namespace: genops-system
spec:
  selector:
    app: genops-ai
  ports:
  - port: 8080
    targetPort: 8080
---
apiVersion: v1
kind: Service
metadata:
  name: genops-ai-preview
  namespace: genops-system
spec:
  selector:
    app: genops-ai
  ports:
  - port: 8080
    targetPort: 8080
---
# Analysis template for success rate
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: success-rate
  namespace: genops-system
spec:
  metrics:
  - name: success-rate
    initialDelay: 30s
    interval: 1m
    successCondition: result >= 0.95
    failureLimit: 3
    provider:
      prometheus:
        address: http://prometheus.monitoring:9090
        query: |
          sum(rate(http_requests_total{status!~"5.."}[5m]))
          /
          sum(rate(http_requests_total[5m]))
```

Deploy Argo Rollouts:

```bash
# Install Argo Rollouts
kubectl create namespace argo-rollouts
kubectl apply -n argo-rollouts -f https://github.com/argoproj/argo-rollouts/releases/latest/download/install.yaml

# Install kubectl plugin
curl -LO https://github.com/argoproj/argo-rollouts/releases/latest/download/kubectl-argo-rollouts-linux-amd64
chmod +x kubectl-argo-rollouts-linux-amd64
sudo mv kubectl-argo-rollouts-linux-amd64 /usr/local/bin/kubectl-argo-rollouts

# Deploy blue-green rollout
kubectl apply -f blue-green-rollout.yaml

# Trigger rollout
kubectl argo rollouts set image genops-ai genops-ai=genopsai/genops:v2.0.0 -n genops-system

# Promote after validation
kubectl argo rollouts promote genops-ai -n genops-system

# Monitor rollout
kubectl argo rollouts get rollout genops-ai -n genops-system --watch
```

### Canary Deployment with Flagger

```yaml
# canary-deployment.yaml
apiVersion: flagger.app/v1beta1
kind: Canary
metadata:
  name: genops-ai
  namespace: genops-system
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: genops-ai
  progressDeadlineSeconds: 600
  service:
    port: 8080
    targetPort: 8080
  analysis:
    interval: 1m
    threshold: 5
    maxWeight: 50
    stepWeight: 10
    metrics:
    - name: request-success-rate
      thresholdRange:
        min: 99
      interval: 1m
    - name: request-duration
      thresholdRange:
        max: 500
      interval: 1m
    webhooks:
    - name: load-test
      url: http://flagger-loadtester.genops-system/
      timeout: 5s
      metadata:
        cmd: "hey -z 1m -q 10 -c 2 http://genops-ai-canary.genops-system:8080/health"
```

Install Flagger:

```bash
# Install Flagger
helm repo add flagger https://flagger.app
helm upgrade -i flagger flagger/flagger \
  --namespace flagger-system \
  --create-namespace \
  --set prometheus.install=true \
  --set meshProvider=kubernetes

# Install load tester
kubectl apply -k github.com/fluxcd/flagger//kustomize/tester?ref=main

# Deploy canary
kubectl apply -f canary-deployment.yaml

# Watch canary analysis
kubectl get canary -n genops-system --watch
```

## Helm Chart Management

### Custom Helm Chart Structure

```bash
helm-charts/genops-ai/
├── Chart.yaml
├── values.yaml
├── values/
│   ├── dev-values.yaml
│   ├── staging-values.yaml
│   └── prod-values.yaml
├── templates/
│   ├── NOTES.txt
│   ├── _helpers.tpl
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   ├── hpa.yaml
│   ├── ingress.yaml
│   ├── serviceaccount.yaml
│   └── tests/
│       └── test-connection.yaml
└── README.md
```

**Chart.yaml:**

```yaml
apiVersion: v2
name: genops-ai
description: GenOps AI governance telemetry for Kubernetes
type: application
version: 1.0.0
appVersion: "1.0.0"
keywords:
- genops
- ai
- governance
- opentelemetry
home: https://github.com/KoshiHQ/GenOps-AI
sources:
- https://github.com/KoshiHQ/GenOps-AI
maintainers:
- name: GenOps Team
  email: team@koshi.tech
dependencies:
- name: prometheus
  version: "15.x.x"
  repository: https://prometheus-community.github.io/helm-charts
  condition: prometheus.enabled
```

**values.yaml:**

```yaml
# Default values for genops-ai
replicaCount: 1

image:
  repository: genopsai/genops
  pullPolicy: IfNotPresent
  tag: "latest"

imagePullSecrets: []
nameOverride: ""
fullnameOverride: ""

serviceAccount:
  create: true
  annotations: {}
  name: ""

podAnnotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "8080"
  prometheus.io/path: "/metrics"

podSecurityContext:
  runAsNonRoot: true
  runAsUser: 1000
  fsGroup: 1000

securityContext:
  capabilities:
    drop:
    - ALL
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false

service:
  type: ClusterIP
  port: 8080

ingress:
  enabled: false
  className: ""
  annotations: {}
  hosts:
  - host: genops.example.com
    paths:
    - path: /
      pathType: Prefix
  tls: []

resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 500m
    memory: 1Gi

autoscaling:
  enabled: false
  minReplicas: 1
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80
  targetMemoryUtilizationPercentage: 80

genops:
  config:
    team: "platform-engineering"
    environment: "production"
    costCenter: "engineering"
    exporterEndpoint: "http://otel-collector:4318"
    logLevel: "info"

prometheus:
  enabled: false
```

### Helm Hooks

```yaml
# templates/pre-install-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: "{{ .Release.Name }}-pre-install"
  labels:
    {{- include "genops-ai.labels" . | nindent 4 }}
  annotations:
    "helm.sh/hook": pre-install
    "helm.sh/hook-weight": "-5"
    "helm.sh/hook-delete-policy": hook-succeeded
spec:
  template:
    metadata:
      name: "{{ .Release.Name }}-pre-install"
    spec:
      restartPolicy: Never
      containers:
      - name: pre-install-validation
        image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
        command: ['sh', '-c', 'echo Pre-install validation; exit 0']
```

## Security & Compliance

### Secret Management with Sealed Secrets

```bash
# Install Sealed Secrets
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.24.0/controller.yaml

# Install kubeseal CLI
wget https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.24.0/kubeseal-0.24.0-linux-amd64.tar.gz
tar -xvzf kubeseal-0.24.0-linux-amd64.tar.gz kubeseal
sudo install -m 755 kubeseal /usr/local/bin/kubeseal

# Create sealed secret
echo -n 'my-secret-api-key' | kubectl create secret generic genops-api-key \
  --dry-run=client \
  --from-file=api-key=/dev/stdin \
  -o yaml | \
kubeseal -o yaml > sealed-secret.yaml

# Apply sealed secret (safe to commit)
kubectl apply -f sealed-secret.yaml
```

### Container Image Scanning with Trivy

```bash
# Scan image
trivy image genopsai/genops:latest

# Scan with CI exit code
trivy image --exit-code 1 --severity HIGH,CRITICAL genopsai/genops:latest

# Generate SARIF report for GitHub
trivy image --format sarif --output trivy-results.sarif genopsai/genops:latest
```

### Image Signing with Cosign

```bash
# Install Cosign
wget https://github.com/sigstore/cosign/releases/download/v2.2.0/cosign-linux-amd64
chmod +x cosign-linux-amd64
sudo mv cosign-linux-amd64 /usr/local/bin/cosign

# Generate key pair
cosign generate-key-pair

# Sign image
cosign sign --key cosign.key genopsai/genops:v1.0.0

# Verify image
cosign verify --key cosign.pub genopsai/genops:v1.0.0
```

## Troubleshooting

### Common CI/CD Issues

#### Issue: Helm Upgrade Fails

**Diagnosis:**
```bash
# Check Helm release status
helm list -n genops-system

# Get release history
helm history genops-ai -n genops-system

# Check deployment status
kubectl rollout status deployment/genops-ai -n genops-system
```

**Solutions:**

1. **Rollback to previous version:**
   ```bash
   helm rollback genops-ai -n genops-system
   ```

2. **Force upgrade:**
   ```bash
   helm upgrade --install genops-ai ./helm-charts/genops-ai \
     --namespace genops-system \
     --force \
     --wait
   ```

#### Issue: ArgoCD Out of Sync

**Diagnosis:**
```bash
# Check application status
kubectl get application genops-ai -n argocd -o yaml

# View sync status
argocd app get genops-ai

# Check diff
argocd app diff genops-ai
```

**Solutions:**

1. **Sync application:**
   ```bash
   argocd app sync genops-ai
   ```

2. **Force sync:**
   ```bash
   argocd app sync genops-ai --force
   ```

---

## Next Steps

1. **Choose GitOps tool** (ArgoCD or FluxCD)
2. **Set up CI/CD pipeline** (GitHub Actions, GitLab CI, or Jenkins)
3. **Implement deployment strategy** (Blue-Green or Canary)
4. **Configure automated testing** in pipeline
5. **Set up security scanning** and image signing
6. **Deploy to production** with confidence

## Additional Resources

- [ArgoCD Documentation](https://argo-cd.readthedocs.io/)
- [FluxCD Documentation](https://fluxcd.io/docs/)
- [Helm Documentation](https://helm.sh/docs/)
- [Argo Rollouts](https://argoproj.github.io/argo-rollouts/)
- [Flagger](https://docs.flagger.app/)
- [GenOps AI Documentation](https://github.com/KoshiHQ/GenOps-AI)

---

This guide provides comprehensive CI/CD patterns for deploying GenOps AI on Kubernetes with production-grade automation and governance.
