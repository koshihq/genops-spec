# CI/CD Integration Guide for GenOps OpenRouter

This guide provides comprehensive CI/CD integration patterns for deploying GenOps OpenRouter services across different platforms and environments.

## Table of Contents

- [GitHub Actions](#github-actions)
- [GitLab CI/CD](#gitlab-cicd)
- [Jenkins Pipeline](#jenkins-pipeline)
- [Azure DevOps](#azure-devops)
- [AWS CodePipeline](#aws-codepipeline)
- [Testing Strategies](#testing-strategies)
- [Security Scanning](#security-scanning)
- [Deployment Strategies](#deployment-strategies)

## GitHub Actions

### Complete Workflow

Create `.github/workflows/openrouter-service.yml`:

```yaml
name: GenOps OpenRouter Service CI/CD

on:
  push:
    branches: [ main, develop ]
    paths: [ 'src/**', 'tests/**', 'examples/openrouter/**', '.github/workflows/**' ]
  pull_request:
    branches: [ main ]
    paths: [ 'src/**', 'tests/**', 'examples/openrouter/**' ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}/openrouter-service

jobs:
  test:
    name: Test Suite
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
        pip install -e .
    
    - name: Lint with ruff
      run: |
        ruff check src/ tests/
        ruff format --check src/ tests/
    
    - name: Type check with mypy
      run: |
        mypy src/genops/providers/openrouter.py
        mypy src/genops/providers/openrouter_pricing.py
        mypy src/genops/providers/openrouter_validation.py
    
    - name: Test with pytest
      env:
        OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY_TEST }}
      run: |
        pytest tests/providers/test_openrouter.py -v --cov=src/genops/providers --cov-report=xml --cov-report=term-missing
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: openrouter
        name: openrouter-coverage
        fail_ci_if_error: true

  integration-test:
    name: Integration Tests
    runs-on: ubuntu-latest
    needs: test
    if: github.event_name == 'push'
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -e .
    
    - name: Run OpenRouter validation
      env:
        OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY_TEST }}
        OTEL_EXPORTER_OTLP_ENDPOINT: ${{ secrets.HONEYCOMB_ENDPOINT }}
        OTEL_EXPORTER_OTLP_HEADERS: ${{ secrets.HONEYCOMB_HEADERS }}
      run: |
        python -c "
        from genops.providers.openrouter import validate_setup, print_validation_result
        result = validate_setup()
        print_validation_result(result)
        exit(0 if result.is_valid else 1)
        "
    
    - name: Test basic functionality
      env:
        OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY_TEST }}
      run: |
        python examples/openrouter/basic_tracking.py
    
    - name: Test cost tracking
      env:
        OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY_TEST }}
      run: |
        python examples/openrouter/cost_tracking.py

  security-scan:
    name: Security Scanning
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Run Bandit security scan
      run: |
        pip install bandit
        bandit -r src/genops/providers/openrouter*.py -f json -o bandit-report.json
    
    - name: Run Safety dependency check
      run: |
        pip install safety
        safety check --json --output safety-report.json || true
    
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          bandit-report.json
          safety-report.json

  build-image:
    name: Build Docker Image
    runs-on: ubuntu-latest
    needs: [test, integration-test, security-scan]
    if: github.event_name == 'push'
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
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
          type=sha,prefix={{branch}}-
          type=raw,value=latest,enable={{is_default_branch}}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: examples/openrouter/docker
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }}
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'

  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: build-image
    if: github.ref == 'refs/heads/develop'
    environment: staging
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Configure kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'v1.28.0'
    
    - name: Set up Kubernetes config
      run: |
        mkdir -p $HOME/.kube
        echo "${{ secrets.KUBE_CONFIG_STAGING }}" | base64 -d > $HOME/.kube/config
        chmod 600 $HOME/.kube/config
    
    - name: Deploy to staging
      run: |
        # Update image tag in deployment
        sed -i "s|image: .*|image: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:develop-${{ github.sha }}|" examples/openrouter/k8s/deployment.yaml
        
        # Apply manifests
        kubectl apply -f examples/openrouter/k8s/ -n genops-openrouter-staging
        
        # Wait for rollout
        kubectl rollout status deployment/openrouter-service -n genops-openrouter-staging --timeout=300s
    
    - name: Run smoke tests
      run: |
        # Get service endpoint
        SERVICE_URL=$(kubectl get service openrouter-service-internal -n genops-openrouter-staging -o jsonpath='{.spec.clusterIP}')
        
        # Health check
        kubectl run test-pod --image=curlimages/curl --rm -i --restart=Never -- \
          curl -f http://$SERVICE_URL:8000/health
        
        # API test
        kubectl run test-pod --image=curlimages/curl --rm -i --restart=Never -- \
          curl -X POST http://$SERVICE_URL:8000/chat/completions \
          -H "Content-Type: application/json" \
          -d '{"model": "meta-llama/llama-3.2-1b-instruct", "messages": [{"role": "user", "content": "test"}], "max_tokens": 5}'

  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: build-image
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Configure kubectl
      uses: azure/setup-kubectl@v3
      with:
        version: 'v1.28.0'
    
    - name: Set up Kubernetes config
      run: |
        mkdir -p $HOME/.kube
        echo "${{ secrets.KUBE_CONFIG_PRODUCTION }}" | base64 -d > $HOME/.kube/config
        chmod 600 $HOME/.kube/config
    
    - name: Deploy to production (Blue-Green)
      run: |
        # Update image tag
        sed -i "s|image: .*|image: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:main-${{ github.sha }}|" examples/openrouter/k8s/deployment.yaml
        
        # Deploy to blue environment
        kubectl apply -f examples/openrouter/k8s/ -n genops-openrouter-blue
        kubectl rollout status deployment/openrouter-service -n genops-openrouter-blue --timeout=300s
        
        # Run production smoke tests
        SERVICE_URL=$(kubectl get service openrouter-service-internal -n genops-openrouter-blue -o jsonpath='{.spec.clusterIP}')
        kubectl run prod-test --image=curlimages/curl --rm -i --restart=Never -- \
          curl -f http://$SERVICE_URL:8000/health
        
        # Switch traffic (update ingress)
        kubectl patch ingress openrouter-service-ingress -n genops-openrouter \
          -p '{"spec":{"rules":[{"host":"api.openrouter.your-domain.com","http":{"paths":[{"path":"/","pathType":"Prefix","backend":{"service":{"name":"openrouter-service-internal","port":{"number":8000}}}}]}}]}}'
        
        # Clean up old green deployment after successful switch
        kubectl delete namespace genops-openrouter-green --ignore-not-found=true
        kubectl create namespace genops-openrouter-green || true
    
    - name: Notify deployment
      uses: 8398a7/action-slack@v3
      with:
        status: ${{ job.status }}
        channel: '#deployments'
        webhook_url: ${{ secrets.SLACK_WEBHOOK }}
        message: |
          🚀 GenOps OpenRouter deployed to production
          Image: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:main-${{ github.sha }}
          Commit: ${{ github.sha }}

  performance-test:
    name: Performance Testing
    runs-on: ubuntu-latest
    needs: deploy-staging
    if: github.ref == 'refs/heads/develop'
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Install k6
      run: |
        sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
        echo "deb https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
        sudo apt-get update
        sudo apt-get install k6
    
    - name: Run load tests
      env:
        STAGING_URL: ${{ secrets.STAGING_SERVICE_URL }}
        OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY_TEST }}
      run: |
        cat > load-test.js << 'EOF'
        import http from 'k6/http';
        import { check, sleep } from 'k6';

        export let options = {
          stages: [
            { duration: '2m', target: 10 },
            { duration: '5m', target: 50 },
            { duration: '2m', target: 0 },
          ],
        };

        export default function() {
          let payload = JSON.stringify({
            model: 'meta-llama/llama-3.2-1b-instruct',
            messages: [{ role: 'user', content: 'Load test message' }],
            max_tokens: 10,
            team: 'load-testing',
            project: 'ci-cd-pipeline'
          });

          let params = {
            headers: { 'Content-Type': 'application/json' },
          };

          let response = http.post(`${__ENV.STAGING_URL}/chat/completions`, payload, params);
          
          check(response, {
            'status is 200': (r) => r.status === 200,
            'response time < 5000ms': (r) => r.timings.duration < 5000,
          });

          sleep(1);
        }
        EOF
        
        k6 run load-test.js
```

### Repository Secrets Setup

Configure these secrets in your GitHub repository:

```bash
# API Keys
OPENROUTER_API_KEY_TEST=your-test-api-key

# Observability
HONEYCOMB_ENDPOINT=https://api.honeycomb.io
HONEYCOMB_HEADERS=x-honeycomb-team=your-key

# Kubernetes
KUBE_CONFIG_STAGING=base64-encoded-kubeconfig
KUBE_CONFIG_PRODUCTION=base64-encoded-kubeconfig

# Notifications
SLACK_WEBHOOK=your-slack-webhook-url

# Staging Environment
STAGING_SERVICE_URL=https://staging-api.your-domain.com
```

## GitLab CI/CD

Create `.gitlab-ci.yml`:

```yaml
variables:
  DOCKER_REGISTRY: registry.gitlab.com
  IMAGE_NAME: $CI_PROJECT_PATH/openrouter-service
  KUBERNETES_NAMESPACE: genops-openrouter

stages:
  - test
  - security
  - build
  - deploy-staging
  - performance
  - deploy-production

# Test stage
test:
  stage: test
  image: python:3.11
  services:
    - docker:dind
  variables:
    PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"
  cache:
    paths:
      - .cache/pip/
      - venv/
  before_script:
    - python -m venv venv
    - source venv/bin/activate
    - pip install --upgrade pip
    - pip install -r requirements.txt -r requirements-dev.txt
    - pip install -e .
  script:
    - ruff check src/ tests/
    - ruff format --check src/ tests/
    - mypy src/genops/providers/openrouter*.py
    - pytest tests/providers/test_openrouter.py -v --cov=src/genops/providers --cov-report=xml
  coverage: '/TOTAL.*\s+(\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml
    paths:
      - coverage.xml
    expire_in: 1 week
  parallel:
    matrix:
      - PYTHON_VERSION: ["3.8", "3.9", "3.10", "3.11", "3.12"]

# Security scanning
security_scan:
  stage: security
  image: python:3.11
  script:
    - pip install bandit safety
    - bandit -r src/genops/providers/openrouter*.py -f json -o bandit-report.json
    - safety check --json --output safety-report.json || true
  artifacts:
    paths:
      - bandit-report.json
      - safety-report.json
    expire_in: 1 week

# Docker build
build_image:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  variables:
    DOCKER_TLS_CERTDIR: "/certs"
  before_script:
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
  script:
    - docker build -t $DOCKER_REGISTRY/$IMAGE_NAME:$CI_COMMIT_SHA examples/openrouter/docker/
    - docker build -t $DOCKER_REGISTRY/$IMAGE_NAME:$CI_COMMIT_REF_SLUG examples/openrouter/docker/
    - docker push $DOCKER_REGISTRY/$IMAGE_NAME:$CI_COMMIT_SHA
    - docker push $DOCKER_REGISTRY/$IMAGE_NAME:$CI_COMMIT_REF_SLUG
  only:
    - main
    - develop

# Deploy to staging
deploy_staging:
  stage: deploy-staging
  image: bitnami/kubectl:latest
  environment:
    name: staging
    url: https://staging-api.your-domain.com
  before_script:
    - mkdir -p $HOME/.kube
    - echo "$KUBE_CONFIG_STAGING" | base64 -d > $HOME/.kube/config
    - chmod 600 $HOME/.kube/config
  script:
    - sed -i "s|image: .*|image: $DOCKER_REGISTRY/$IMAGE_NAME:$CI_COMMIT_SHA|" examples/openrouter/k8s/deployment.yaml
    - kubectl apply -f examples/openrouter/k8s/ -n genops-openrouter-staging
    - kubectl rollout status deployment/openrouter-service -n genops-openrouter-staging --timeout=300s
    # Smoke test
    - SERVICE_URL=$(kubectl get service openrouter-service-internal -n genops-openrouter-staging -o jsonpath='{.spec.clusterIP}')
    - kubectl run test-pod --image=curlimages/curl --rm -i --restart=Never -- curl -f http://$SERVICE_URL:8000/health
  only:
    - develop

# Performance testing
performance_test:
  stage: performance
  image: grafana/k6:latest
  needs:
    - deploy_staging
  script:
    - |
      cat > load-test.js << 'EOF'
      import http from 'k6/http';
      import { check, sleep } from 'k6';

      export let options = {
        stages: [
          { duration: '1m', target: 10 },
          { duration: '3m', target: 30 },
          { duration: '1m', target: 0 },
        ],
      };

      export default function() {
        let payload = JSON.stringify({
          model: 'meta-llama/llama-3.2-1b-instruct',
          messages: [{ role: 'user', content: 'Performance test' }],
          max_tokens: 10,
          team: 'performance-testing'
        });

        let response = http.post(`${__ENV.STAGING_URL}/chat/completions`, payload, {
          headers: { 'Content-Type': 'application/json' },
        });
        
        check(response, {
          'status is 200': (r) => r.status === 200,
          'response time < 3000ms': (r) => r.timings.duration < 3000,
        });

        sleep(1);
      }
      EOF
    - k6 run load-test.js
  variables:
    STAGING_URL: https://staging-api.your-domain.com
  only:
    - develop

# Deploy to production
deploy_production:
  stage: deploy-production
  image: bitnami/kubectl:latest
  environment:
    name: production
    url: https://api.your-domain.com
  before_script:
    - mkdir -p $HOME/.kube
    - echo "$KUBE_CONFIG_PRODUCTION" | base64 -d > $HOME/.kube/config
    - chmod 600 $HOME/.kube/config
  script:
    - sed -i "s|image: .*|image: $DOCKER_REGISTRY/$IMAGE_NAME:$CI_COMMIT_SHA|" examples/openrouter/k8s/deployment.yaml
    - kubectl apply -f examples/openrouter/k8s/ -n genops-openrouter
    - kubectl rollout status deployment/openrouter-service -n genops-openrouter --timeout=600s
  after_script:
    # Send notification
    - 'curl -X POST -H "Content-type: application/json" --data "{\"text\":\"🚀 GenOps OpenRouter deployed to production: $CI_COMMIT_SHA\"}" $SLACK_WEBHOOK'
  only:
    - main
  when: manual
  allow_failure: false
```

## Jenkins Pipeline

Create `Jenkinsfile`:

```groovy
pipeline {
    agent any
    
    environment {
        DOCKER_REGISTRY = 'your-registry.com'
        IMAGE_NAME = 'genops/openrouter-service'
        KUBECONFIG_STAGING = credentials('kubeconfig-staging')
        KUBECONFIG_PRODUCTION = credentials('kubeconfig-production')
        OPENROUTER_API_KEY = credentials('openrouter-api-key-test')
        SLACK_WEBHOOK = credentials('slack-webhook')
    }
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Test') {
            parallel {
                stage('Python 3.8') {
                    agent {
                        docker {
                            image 'python:3.8'
                            args '-v /var/run/docker.sock:/var/run/docker.sock'
                        }
                    }
                    steps {
                        sh '''
                            python -m venv venv
                            . venv/bin/activate
                            pip install --upgrade pip
                            pip install -r requirements.txt -r requirements-dev.txt
                            pip install -e .
                            pytest tests/providers/test_openrouter.py -v --cov=src/genops/providers --cov-report=xml
                        '''
                    }
                    post {
                        always {
                            publishCoverage adapters: [
                                coberturaAdapter('coverage.xml')
                            ], sourceFileResolver: sourceFiles('STORE_LAST_BUILD')
                        }
                    }
                }
                
                stage('Python 3.11') {
                    agent {
                        docker {
                            image 'python:3.11'
                            args '-v /var/run/docker.sock:/var/run/docker.sock'
                        }
                    }
                    steps {
                        sh '''
                            python -m venv venv
                            . venv/bin/activate
                            pip install --upgrade pip
                            pip install -r requirements.txt -r requirements-dev.txt
                            pip install -e .
                            ruff check src/ tests/
                            ruff format --check src/ tests/
                            mypy src/genops/providers/openrouter*.py
                            pytest tests/providers/test_openrouter.py -v
                        '''
                    }
                }
            }
        }
        
        stage('Security Scan') {
            steps {
                sh '''
                    python -m venv venv
                    . venv/bin/activate
                    pip install bandit safety
                    bandit -r src/genops/providers/openrouter*.py -f json -o bandit-report.json
                    safety check --json --output safety-report.json || true
                '''
            }
            post {
                always {
                    archiveArtifacts artifacts: '*-report.json', fingerprint: true
                }
            }
        }
        
        stage('Integration Test') {
            when {
                anyOf {
                    branch 'main'
                    branch 'develop'
                }
            }
            steps {
                sh '''
                    python -m venv venv
                    . venv/bin/activate
                    pip install -r requirements.txt
                    pip install -e .
                    python -c "
                    from genops.providers.openrouter import validate_setup, print_validation_result
                    result = validate_setup()
                    print_validation_result(result)
                    exit(0 if result.is_valid else 1)
                    "
                '''
            }
        }
        
        stage('Build Image') {
            when {
                anyOf {
                    branch 'main'
                    branch 'develop'
                }
            }
            steps {
                script {
                    def image = docker.build("${DOCKER_REGISTRY}/${IMAGE_NAME}:${env.BUILD_NUMBER}", "examples/openrouter/docker/")
                    docker.withRegistry("https://${DOCKER_REGISTRY}", 'docker-registry-credentials') {
                        image.push()
                        image.push("${env.BRANCH_NAME}-${env.BUILD_NUMBER}")
                        if (env.BRANCH_NAME == 'main') {
                            image.push('latest')
                        }
                    }
                }
            }
        }
        
        stage('Deploy Staging') {
            when {
                branch 'develop'
            }
            steps {
                script {
                    kubernetesDeploy(
                        configs: 'examples/openrouter/k8s/*.yaml',
                        kubeconfigId: 'kubeconfig-staging',
                        namespace: 'genops-openrouter-staging'
                    )
                }
                sh '''
                    kubectl rollout status deployment/openrouter-service -n genops-openrouter-staging --timeout=300s
                    
                    # Smoke test
                    SERVICE_URL=$(kubectl get service openrouter-service-internal -n genops-openrouter-staging -o jsonpath='{.spec.clusterIP}')
                    kubectl run test-pod --image=curlimages/curl --rm -i --restart=Never -- \\
                        curl -f http://$SERVICE_URL:8000/health
                '''
            }
        }
        
        stage('Performance Test') {
            when {
                branch 'develop'
            }
            steps {
                sh '''
                    docker run --rm -i grafana/k6:latest run - <<EOF
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  stages: [
    { duration: '1m', target: 10 },
    { duration: '2m', target: 20 },
    { duration: '1m', target: 0 },
  ],
};

export default function() {
  let payload = JSON.stringify({
    model: 'meta-llama/llama-3.2-1b-instruct',
    messages: [{ role: 'user', content: 'Performance test' }],
    max_tokens: 10,
    team: 'jenkins-testing'
  });

  let response = http.post('https://staging-api.your-domain.com/chat/completions', payload, {
    headers: { 'Content-Type': 'application/json' },
  });
  
  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 5000ms': (r) => r.timings.duration < 5000,
  });

  sleep(1);
}
EOF
                '''
            }
        }
        
        stage('Deploy Production') {
            when {
                branch 'main'
            }
            steps {
                input message: 'Deploy to production?', ok: 'Deploy'
                script {
                    kubernetesDeploy(
                        configs: 'examples/openrouter/k8s/*.yaml',
                        kubeconfigId: 'kubeconfig-production',
                        namespace: 'genops-openrouter'
                    )
                }
                sh '''
                    kubectl rollout status deployment/openrouter-service -n genops-openrouter --timeout=600s
                    
                    # Production smoke test
                    SERVICE_URL=$(kubectl get service openrouter-service-internal -n genops-openrouter -o jsonpath='{.spec.clusterIP}')
                    kubectl run prod-test --image=curlimages/curl --rm -i --restart=Never -- \\
                        curl -f http://$SERVICE_URL:8000/health
                '''
            }
            post {
                success {
                    sh '''
                        curl -X POST -H "Content-type: application/json" \\
                            --data "{\\"text\\":\\"🚀 GenOps OpenRouter deployed to production: ${BUILD_NUMBER}\\"}" \\
                            ${SLACK_WEBHOOK}
                    '''
                }
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
        failure {
            sh '''
                curl -X POST -H "Content-type: application/json" \\
                    --data "{\\"text\\":\\"❌ GenOps OpenRouter deployment failed: ${BUILD_NUMBER}\\"}" \\
                    ${SLACK_WEBHOOK}
            '''
        }
    }
}
```

## Testing Strategies

### Unit Testing Strategy

```python
# tests/ci_cd/test_openrouter_integration.py
"""CI/CD-specific integration tests for OpenRouter."""

import pytest
import os
from unittest.mock import patch, MagicMock

class TestCICDIntegration:
    """Test CI/CD-specific scenarios."""
    
    def test_environment_validation(self):
        """Test that CI/CD environment variables are properly validated."""
        required_vars = [
            "OPENROUTER_API_KEY",
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "OTEL_SERVICE_NAME"
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            pytest.skip(f"Missing required environment variables: {missing_vars}")
    
    def test_deployment_readiness(self):
        """Test that the service is ready for deployment."""
        from genops.providers.openrouter import validate_setup
        
        result = validate_setup()
        
        # In CI/CD, we should have minimal warnings
        assert result.is_valid, "Service must be valid for deployment"
        assert result.summary["error_count"] == 0, "No errors allowed for deployment"
        
        # Allow some warnings in CI/CD (missing optional configs)
        assert result.summary["warning_count"] <= 5, "Too many warnings for deployment"
    
    @pytest.mark.integration
    def test_api_connectivity(self):
        """Test API connectivity in CI/CD environment."""
        from genops.providers.openrouter import instrument_openrouter
        
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            pytest.skip("No API key available for integration test")
        
        client = instrument_openrouter(openrouter_api_key=api_key)
        
        # Test minimal request to verify connectivity
        response = client.chat_completions_create(
            model="meta-llama/llama-3.2-1b-instruct",
            messages=[{"role": "user", "content": "CI/CD test"}],
            max_tokens=5,
            team="ci-cd",
            project="integration-test"
        )
        
        assert response is not None
        assert hasattr(response, 'choices')
        assert len(response.choices) > 0
```

### Load Testing Script

```javascript
// ci-cd/load-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

const errorRate = new Rate('errors');

export let options = {
  stages: [
    { duration: '30s', target: 5 },   // Ramp up
    { duration: '2m', target: 20 },   // Stay at 20 users
    { duration: '30s', target: 50 },  // Spike test
    { duration: '1m', target: 20 },   // Back to normal
    { duration: '30s', target: 0 },   // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<5000'], // 95% of requests under 5s
    http_req_failed: ['rate<0.1'],     // Error rate under 10%
    errors: ['rate<0.1'],              // Custom error rate
  },
};

export default function() {
  const payload = JSON.stringify({
    model: 'meta-llama/llama-3.2-1b-instruct',
    messages: [
      { 
        role: 'user', 
        content: `Load test message ${Math.random()}` 
      }
    ],
    max_tokens: 10,
    team: 'load-testing',
    project: 'ci-cd-pipeline',
    environment: 'staging'
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
  };

  const response = http.post(`${__ENV.BASE_URL}/chat/completions`, payload, params);
  
  const result = check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 5000ms': (r) => r.timings.duration < 5000,
    'response has content': (r) => r.body && r.body.length > 0,
    'valid json response': (r) => {
      try {
        JSON.parse(r.body);
        return true;
      } catch {
        return false;
      }
    },
  });

  errorRate.add(!result);
  
  sleep(1);
}

export function handleSummary(data) {
  return {
    'load-test-results.json': JSON.stringify(data, null, 2),
  };
}
```

## Security Scanning

### Bandit Configuration

Create `.bandit`:

```yaml
tests: ['B101', 'B601']
skips: ['B101', 'B601']

exclude_dirs:
  - '/tests/'
  - '/venv/'
  - '/.venv/'

# Exclude test files from certain checks
exclude: |
  */test_*.py,
  */tests/*
```

### Safety Configuration

Create `.safety-policy.json`:

```json
{
  "security": {
    "ignore": [],
    "continue-on-error": false
  },
  "alert": {
    "ignore": {
      "vulnerability": [],
      "cve": [],
      "id": []
    }
  },
  "report": {
    "only-affected": true,
    "output": {
      "format": "json",
      "file": "safety-report.json"
    }
  }
}
```

## Deployment Strategies

### Blue-Green Deployment Script

```bash
#!/bin/bash
# ci-cd/deploy-blue-green.sh

set -e

NAMESPACE=${NAMESPACE:-genops-openrouter}
IMAGE_TAG=${IMAGE_TAG:-latest}
TIMEOUT=${TIMEOUT:-300}

echo "🔄 Starting Blue-Green deployment for OpenRouter service"

# Determine current environment
CURRENT_ENV=$(kubectl get ingress openrouter-service-ingress -n $NAMESPACE -o jsonpath='{.spec.rules[0].http.paths[0].backend.service.name}' | grep -o 'blue\|green' || echo 'green')
NEW_ENV=$([ "$CURRENT_ENV" = "blue" ] && echo "green" || echo "blue")

echo "📊 Current environment: $CURRENT_ENV"
echo "🎯 Target environment: $NEW_ENV"

# Deploy to new environment
echo "🚀 Deploying to $NEW_ENV environment..."
sed "s/openrouter-service/openrouter-service-$NEW_ENV/g" examples/openrouter/k8s/deployment.yaml | \
sed "s/image: .*/image: $IMAGE_TAG/" | \
kubectl apply -f - -n $NAMESPACE

# Wait for deployment
echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/openrouter-service-$NEW_ENV -n $NAMESPACE --timeout=${TIMEOUT}s

# Health check
echo "🔍 Running health checks..."
SERVICE_URL=$(kubectl get service openrouter-service-$NEW_ENV -n $NAMESPACE -o jsonpath='{.spec.clusterIP}')
kubectl run health-check-$NEW_ENV --image=curlimages/curl --rm -i --restart=Never -- \
  curl -f http://$SERVICE_URL:8000/health

# Switch traffic
echo "🔄 Switching traffic to $NEW_ENV..."
kubectl patch ingress openrouter-service-ingress -n $NAMESPACE \
  -p "{\"spec\":{\"rules\":[{\"host\":\"api.your-domain.com\",\"http\":{\"paths\":[{\"path\":\"/\",\"pathType\":\"Prefix\",\"backend\":{\"service\":{\"name\":\"openrouter-service-$NEW_ENV\",\"port\":{\"number\":8000}}}}]}}]}}"

# Verify switch
echo "✅ Verifying traffic switch..."
sleep 10
kubectl run traffic-check --image=curlimages/curl --rm -i --restart=Never -- \
  curl -f https://api.your-domain.com/health

# Clean up old environment
echo "🧹 Cleaning up $CURRENT_ENV environment..."
kubectl delete deployment openrouter-service-$CURRENT_ENV -n $NAMESPACE --ignore-not-found=true
kubectl delete service openrouter-service-$CURRENT_ENV -n $NAMESPACE --ignore-not-found=true

echo "🎉 Blue-Green deployment completed successfully!"
echo "   Active environment: $NEW_ENV"
echo "   Image: $IMAGE_TAG"
```

### Canary Deployment Script

```bash
#!/bin/bash
# ci-cd/deploy-canary.sh

set -e

NAMESPACE=${NAMESPACE:-genops-openrouter}
IMAGE_TAG=${IMAGE_TAG:-latest}
CANARY_PERCENTAGE=${CANARY_PERCENTAGE:-10}

echo "🐤 Starting Canary deployment for OpenRouter service"
echo "   Target percentage: ${CANARY_PERCENTAGE}%"

# Deploy canary version
echo "🚀 Deploying canary version..."
sed 's/openrouter-service/openrouter-service-canary/g' examples/openrouter/k8s/deployment.yaml | \
sed "s/image: .*/image: $IMAGE_TAG/" | \
sed "s/replicas: [0-9]*/replicas: 1/" | \
kubectl apply -f - -n $NAMESPACE

# Wait for canary deployment
kubectl rollout status deployment/openrouter-service-canary -n $NAMESPACE --timeout=300s

# Configure traffic split (using Istio VirtualService)
cat <<EOF | kubectl apply -f -
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: openrouter-service-vs
  namespace: $NAMESPACE
spec:
  hosts:
  - api.your-domain.com
  http:
  - match:
    - headers:
        canary:
          exact: "true"
    route:
    - destination:
        host: openrouter-service-canary
        port:
          number: 8000
  - route:
    - destination:
        host: openrouter-service
        port:
          number: 8000
      weight: $((100 - CANARY_PERCENTAGE))
    - destination:
        host: openrouter-service-canary
        port:
          number: 8000
      weight: $CANARY_PERCENTAGE
EOF

echo "✅ Canary deployment completed!"
echo "   Canary traffic: ${CANARY_PERCENTAGE}%"
echo "   Stable traffic: $((100 - CANARY_PERCENTAGE))%"

# Monitor metrics for 5 minutes
echo "📊 Monitoring canary metrics..."
sleep 300

# Promote or rollback based on metrics
ERROR_RATE=$(kubectl exec -n monitoring deployment/prometheus -- \
  promtool query instant 'rate(http_requests_total{job="openrouter-service-canary",status=~"5.."}[5m]) / rate(http_requests_total{job="openrouter-service-canary"}[5m])' | \
  grep -o '[0-9.]*' | head -1)

if (( $(echo "$ERROR_RATE < 0.05" | bc -l) )); then
    echo "✅ Canary metrics look good, promoting to full deployment"
    # Promote canary (implementation depends on your setup)
else
    echo "❌ Canary metrics show issues, rolling back"
    kubectl delete deployment openrouter-service-canary -n $NAMESPACE
fi
```

## Best Practices

### Environment Management

1. **Environment Parity**
   - Identical configurations across environments
   - Same image tags and versions
   - Consistent resource limits

2. **Secret Management**
   - Use CI/CD platform secret management
   - Rotate secrets regularly
   - Separate secrets per environment

3. **Configuration Management**
   - Environment-specific ConfigMaps
   - Feature flags for environment differences
   - Validation in CI/CD pipeline

### Monitoring and Alerting

1. **Deployment Monitoring**
   - Health check endpoints
   - Readiness probes
   - Resource utilization

2. **Application Metrics**
   - Request rates and latencies
   - Error rates by endpoint
   - Cost tracking per deployment

3. **Alert Integration**
   - Slack/Teams notifications
   - PagerDuty for critical issues
   - Email summaries

### Rollback Strategies

1. **Automatic Rollback**
   - Health check failures
   - High error rates
   - Performance degradation

2. **Manual Rollback**
   - Business logic issues
   - Data consistency problems
   - Customer impact

3. **Database Considerations**
   - Migration compatibility
   - Backup before deployment
   - Rollback procedures

---

This CI/CD integration guide provides comprehensive automation patterns for deploying GenOps OpenRouter services with enterprise-grade reliability, security, and observability.