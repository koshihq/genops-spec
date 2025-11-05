#!/usr/bin/env python3
"""
AWS ECS + Bedrock Integration Example

This example demonstrates container deployment patterns for AWS Bedrock
with GenOps governance, optimized for ECS Fargate and EC2 deployments.

Features demonstrated:
- Docker containerization with GenOps Bedrock integration
- ECS task definitions and service configurations
- Container-optimized telemetry and logging
- Auto-scaling policies based on AI workload metrics
- Health check patterns for containerized AI services
- Multi-container architectures with sidecar patterns

Example usage:
    python ecs_integration.py

Note: This example shows ECS deployment patterns. For actual deployment,
build Docker images and deploy using AWS CLI, CDK, or Terraform.
"""

import sys
import os
import json

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

def create_dockerfile():
    """Create optimized Dockerfile for GenOps Bedrock applications."""
    
    print("🐳 Docker Container Configuration")
    print("=" * 38)
    print("Optimized containerization for GenOps Bedrock applications:")
    print()
    
    dockerfile_content = '''
# Multi-stage build for optimized production image
FROM python:3.9-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    ca-certificates \\
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN groupadd -r genops && useradd -r -g genops genops

# Set up application directory
WORKDIR /app
COPY --chown=genops:genops . .

# Set environment variables for GenOps
ENV GENOPS_ENVIRONMENT=production
ENV GENOPS_PROJECT=bedrock-ecs-integration
ENV OTEL_SERVICE_NAME=bedrock-ecs-service
ENV OTEL_RESOURCE_ATTRIBUTES="service.name=bedrock-ecs,deployment.environment=production"

# Configure Python for containers
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8080/health || exit 1

# Switch to non-root user
USER genops

# Expose application port
EXPOSE 8080

# Start application with GenOps instrumentation
CMD ["python", "app.py"]
'''
    
    requirements_content = '''
# Core dependencies
genops-ai[bedrock]==1.0.0
boto3>=1.29.0
botocore>=1.32.0

# Web framework
flask>=2.3.0
gunicorn>=21.2.0

# Observability
opentelemetry-instrumentation-flask
opentelemetry-exporter-otlp
opentelemetry-instrumentation-boto3sqs
opentelemetry-instrumentation-requests

# Monitoring and health checks
prometheus-client>=0.18.0
psutil>=5.9.0
'''
    
    print("📄 Dockerfile Features:")
    print("   ✅ Multi-stage build for optimized image size")
    print("   ✅ Non-root user for security")
    print("   ✅ Health check endpoint")
    print("   ✅ GenOps environment configuration")
    print("   ✅ OpenTelemetry instrumentation")
    print("   ✅ Production-ready Python settings")
    print()
    
    return dockerfile_content, requirements_content


def create_flask_application():
    """Create Flask application with GenOps Bedrock integration."""
    
    print("🌐 Flask Application with GenOps")
    print("=" * 38)
    print("Production-ready web service for AI processing:")
    print()
    
    app_code = '''
import os
import json
import logging
from flask import Flask, request, jsonify
import boto3
from datetime import datetime

# GenOps Bedrock integration
from genops.providers.bedrock import GenOpsBedrockAdapter, instrument_bedrock
from genops.providers.bedrock_workflow import production_workflow_context, ComplianceLevel
from genops.providers.bedrock_cost_aggregator import create_bedrock_cost_context

# OpenTelemetry instrumentation
from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.boto3sqs import Boto3SQSInstrumentor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Enable GenOps auto-instrumentation
instrument_bedrock()

# Enable OpenTelemetry instrumentation
FlaskInstrumentor().instrument_app(app)
Boto3SQSInstrumentor().instrument()

# Initialize GenOps Bedrock adapter
bedrock_adapter = GenOpsBedrockAdapter(
    region_name=os.environ.get('AWS_REGION', 'us-east-1'),
    default_model="anthropic.claude-3-haiku-20240307-v1:0"
)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for ECS health checks."""
    try:
        # Verify Bedrock connectivity
        if bedrock_adapter.is_available():
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.utcnow().isoformat(),
                'service': 'bedrock-ecs-service',
                'version': '1.0.0',
                'bedrock_available': True
            }), 200
        else:
            return jsonify({
                'status': 'unhealthy',
                'error': 'Bedrock not available'
            }), 503
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503

@app.route('/analyze', methods=['POST'])
def analyze_document():
    """Document analysis endpoint with GenOps governance."""
    try:
        data = request.get_json()
        
        document_text = data.get('document_text', '')
        customer_id = data.get('customer_id', 'unknown')
        analysis_type = data.get('analysis_type', 'general')
        
        if not document_text:
            return jsonify({'error': 'document_text is required'}), 400
        
        # Create production workflow context
        with production_workflow_context(
            workflow_name="ecs_document_analysis",
            customer_id=customer_id,
            team="ecs-ai-service",
            project="containerized-ai-processing",
            environment="production",
            compliance_level=ComplianceLevel.SOC2,
            budget_limit=1.00,  # $1.00 per request
            region=os.environ.get('AWS_REGION', 'us-east-1')
        ) as (workflow, workflow_id):
            
            # Step 1: Document classification
            workflow.record_step("classification", {
                'analysis_type': analysis_type,
                'document_length': len(document_text),
                'container_id': os.environ.get('HOSTNAME', 'unknown')
            })
            
            classification = bedrock_adapter.text_generation(
                prompt=f"Classify this document for {analysis_type} analysis: {document_text[:500]}",
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                max_tokens=100,
                temperature=0.1,
                team="ecs-ai-service",
                customer_id=customer_id,
                feature=f"classification_{analysis_type}"
            )
            
            # Step 2: Content analysis
            workflow.record_step("analysis", {
                'classification': classification.content[:100]
            })
            
            # Choose model based on classification for cost optimization
            if 'financial' in classification.content.lower():
                model = "anthropic.claude-3-sonnet-20240229-v1:0"  # Higher accuracy
            else:
                model = "anthropic.claude-3-haiku-20240307-v1:0"   # Cost-effective
            
            analysis = bedrock_adapter.text_generation(
                prompt=f"Analyze this {classification.content} document: {document_text}",
                model_id=model,
                max_tokens=300,
                temperature=0.3,
                team="ecs-ai-service",
                customer_id=customer_id,
                feature=f"analysis_{analysis_type}"
            )
            
            # Record container-specific metrics
            workflow.record_performance_metric("container_id", 
                                             os.environ.get('HOSTNAME', 'unknown'), "string")
            workflow.record_performance_metric("ecs_task_arn", 
                                             os.environ.get('ECS_TASK_ARN', 'unknown'), "string")
            
            # Get final cost summary
            final_cost = workflow.get_current_cost_summary()
            
            return jsonify({
                'workflow_id': workflow_id,
                'classification': classification.content.strip(),
                'analysis': analysis.content.strip(),
                'cost': final_cost.total_cost,
                'performance': {
                    'total_latency_ms': final_cost.total_latency_ms,
                    'models_used': list(final_cost.unique_models),
                    'total_operations': final_cost.total_operations
                },
                'container_info': {
                    'hostname': os.environ.get('HOSTNAME', 'unknown'),
                    'task_arn': os.environ.get('ECS_TASK_ARN', 'unknown')
                },
                'governance': {
                    'customer_id': customer_id,
                    'team': 'ecs-ai-service',
                    'compliance': 'SOC2'
                }
            }), 200
            
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return jsonify({
            'error': str(e),
            'error_type': type(e).__name__
        }), 500

@app.route('/batch', methods=['POST'])
def batch_processing():
    """Batch processing endpoint optimized for ECS scaling."""
    try:
        data = request.get_json()
        documents = data.get('documents', [])
        customer_id = data.get('customer_id', 'unknown')
        
        if not documents:
            return jsonify({'error': 'documents list is required'}), 400
        
        # Use cost context for batch tracking
        with create_bedrock_cost_context(f"ecs_batch_{customer_id}") as cost_context:
            
            results = []
            
            for i, doc in enumerate(documents):
                try:
                    # Process each document
                    result = bedrock_adapter.text_generation(
                        prompt=f"Summarize document {i+1}: {doc.get('text', '')[:500]}",
                        model_id="amazon.titan-text-express-v1",  # Cost-effective for batch
                        max_tokens=150,
                        temperature=0.2,
                        team="ecs-batch-service",
                        customer_id=customer_id,
                        feature="batch_processing"
                    )
                    
                    results.append({
                        'document_id': doc.get('id', f'doc_{i+1}'),
                        'summary': result.content.strip(),
                        'cost': result.cost_usd,
                        'latency_ms': result.latency_ms
                    })
                    
                except Exception as e:
                    results.append({
                        'document_id': doc.get('id', f'doc_{i+1}'),
                        'error': str(e)
                    })
            
            # Get batch summary
            batch_summary = cost_context.get_current_summary()
            
            return jsonify({
                'batch_id': cost_context.context_id,
                'documents_processed': len(results),
                'results': results,
                'batch_cost': batch_summary.total_cost,
                'average_cost_per_doc': batch_summary.get_average_cost_per_operation(),
                'total_latency_ms': batch_summary.total_latency_ms,
                'container_info': {
                    'hostname': os.environ.get('HOSTNAME', 'unknown')
                }
            }), 200
            
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return jsonify({
            'error': str(e),
            'error_type': type(e).__name__
        }), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint for monitoring."""
    # In production, this would return Prometheus-formatted metrics
    return jsonify({
        'service_name': 'bedrock-ecs-service',
        'requests_total': 'Counter metric',
        'request_duration_seconds': 'Histogram metric',
        'bedrock_operations_total': 'Counter metric',
        'bedrock_cost_total': 'Counter metric',
        'container_memory_usage': 'Gauge metric'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
'''
    
    print("🌐 Flask Application Features:")
    print("   ✅ RESTful API for document analysis")
    print("   ✅ Health check endpoint for ECS")
    print("   ✅ Batch processing with cost optimization")
    print("   ✅ Production workflow context")
    print("   ✅ Container-specific metrics")
    print("   ✅ Prometheus metrics endpoint")
    print("   ✅ Comprehensive error handling")
    print()
    
    return app_code


def create_ecs_task_definition():
    """Create ECS task definition with GenOps configuration."""
    
    print("📋 ECS Task Definition")
    print("=" * 25)
    print("Container orchestration with GenOps governance:")
    print()
    
    task_definition = {
        "family": "genops-bedrock-service",
        "networkMode": "awsvpc",
        "requiresCompatibilities": ["FARGATE"],
        "cpu": "1024",  # 1 vCPU
        "memory": "2048",  # 2GB RAM
        "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
        "taskRoleArn": "arn:aws:iam::ACCOUNT:role/genops-bedrock-task-role",
        
        "containerDefinitions": [
            {
                "name": "genops-bedrock-app",
                "image": "YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/genops-bedrock:latest",
                "portMappings": [
                    {
                        "containerPort": 8080,
                        "protocol": "tcp"
                    }
                ],
                "essential": True,
                "environment": [
                    {"name": "AWS_REGION", "value": "us-east-1"},
                    {"name": "GENOPS_ENVIRONMENT", "value": "production"},
                    {"name": "GENOPS_PROJECT", "value": "bedrock-ecs-service"},
                    {"name": "OTEL_SERVICE_NAME", "value": "bedrock-ecs"},
                    {"name": "OTEL_RESOURCE_ATTRIBUTES", "value": "service.name=bedrock-ecs,deployment.environment=production"},
                    {"name": "GENOPS_SAMPLING_RATE", "value": "1.0"},
                    {"name": "GENOPS_ASYNC_EXPORT", "value": "true"},
                    {"name": "GENOPS_CIRCUIT_BREAKER", "value": "true"}
                ],
                "secrets": [
                    {
                        "name": "OTEL_EXPORTER_OTLP_ENDPOINT",
                        "valueFrom": "arn:aws:secretsmanager:us-east-1:ACCOUNT:secret:genops/otel-endpoint"
                    }
                ],
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": "/ecs/genops-bedrock-service",
                        "awslogs-region": "us-east-1",
                        "awslogs-stream-prefix": "ecs"
                    }
                },
                "healthCheck": {
                    "command": ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"],
                    "interval": 30,
                    "timeout": 5,
                    "retries": 3,
                    "startPeriod": 60
                },
                "ulimits": [
                    {
                        "name": "nofile",
                        "softLimit": 65536,
                        "hardLimit": 65536
                    }
                ]
            },
            {
                "name": "otel-collector-sidecar",
                "image": "otel/opentelemetry-collector-contrib:latest",
                "essential": False,
                "portMappings": [
                    {"containerPort": 4317, "protocol": "tcp"},
                    {"containerPort": 8889, "protocol": "tcp"}
                ],
                "environment": [
                    {"name": "OTEL_CONFIG_FILE", "value": "/etc/otel-collector-config.yml"}
                ],
                "mountPoints": [
                    {
                        "sourceVolume": "otel-config",
                        "containerPath": "/etc/otel-collector-config.yml",
                        "readOnly": True
                    }
                ],
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": "/ecs/genops-bedrock-service",
                        "awslogs-region": "us-east-1",
                        "awslogs-stream-prefix": "otel-sidecar"
                    }
                }
            }
        ],
        
        "volumes": [
            {
                "name": "otel-config",
                "host": {
                    "sourcePath": "/opt/otel-collector-config.yml"
                }
            }
        ]
    }
    
    # IAM policy for task role
    task_role_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                    "bedrock:ListFoundationModels"
                ],
                "Resource": "*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "cloudtrail:PutEvents",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents"
                ],
                "Resource": "*"
            },
            {
                "Effect": "Allow",
                "Action": [
                    "secretsmanager:GetSecretValue"
                ],
                "Resource": "arn:aws:secretsmanager:us-east-1:ACCOUNT:secret:genops/*"
            }
        ]
    }
    
    print("🏗️ Task Definition Features:")
    print("   ✅ Fargate-compatible configuration")
    print("   ✅ Multi-container setup with OpenTelemetry sidecar")
    print("   ✅ Health checks for service availability")
    print("   ✅ Comprehensive environment configuration")
    print("   ✅ Secrets management integration")
    print("   ✅ CloudWatch logging configuration")
    print("   ✅ Proper IAM permissions for Bedrock")
    print()
    
    return task_definition, task_role_policy


def create_ecs_service_configuration():
    """Create ECS service with auto-scaling configuration."""
    
    print("🎛️ ECS Service Configuration")
    print("=" * 32)
    print("Auto-scaling service with load balancing:")
    print()
    
    service_definition = {
        "serviceName": "genops-bedrock-service",
        "cluster": "genops-production-cluster",
        "taskDefinition": "genops-bedrock-service:1",
        "desiredCount": 2,
        "launchType": "FARGATE",
        "platformVersion": "LATEST",
        
        "networkConfiguration": {
            "awsvpcConfiguration": {
                "subnets": [
                    "subnet-12345678",  # Private subnet 1
                    "subnet-87654321"   # Private subnet 2
                ],
                "securityGroups": [
                    "sg-genops-bedrock"
                ],
                "assignPublicIp": "DISABLED"
            }
        },
        
        "loadBalancers": [
            {
                "targetGroupArn": "arn:aws:elasticloadbalancing:us-east-1:ACCOUNT:targetgroup/genops-bedrock-tg/12345",
                "containerName": "genops-bedrock-app",
                "containerPort": 8080
            }
        ],
        
        "deploymentConfiguration": {
            "maximumPercent": 200,
            "minimumHealthyPercent": 100,
            "deploymentCircuitBreaker": {
                "enable": True,
                "rollback": True
            }
        },
        
        "healthCheckGracePeriodSeconds": 120,
        "enableExecuteCommand": True,  # For debugging
        
        "tags": [
            {"key": "Project", "value": "GenOps-AI"},
            {"key": "Service", "value": "Bedrock-Integration"},
            {"key": "Environment", "value": "Production"},
            {"key": "CostCenter", "value": "AI-Platform"}
        ]
    }
    
    # Auto-scaling configuration
    autoscaling_config = {
        "service_name": "genops-bedrock-service",
        "cluster": "genops-production-cluster",
        "min_capacity": 2,
        "max_capacity": 20,
        "target_cpu_utilization": 70,
        "target_memory_utilization": 80,
        "scale_out_cooldown": 300,  # 5 minutes
        "scale_in_cooldown": 300,
        
        "custom_metrics": [
            {
                "metric_name": "bedrock_requests_per_minute",
                "target_value": 100,
                "scale_out_threshold": 120,
                "scale_in_threshold": 80
            },
            {
                "metric_name": "bedrock_average_cost_per_request",
                "target_value": 0.01,
                "alert_threshold": 0.02  # Alert if cost too high
            }
        ]
    }
    
    print("⚖️ Service Configuration Features:")
    print("   ✅ High availability with multiple AZs")
    print("   ✅ Application Load Balancer integration")
    print("   ✅ Rolling deployment with circuit breaker")
    print("   ✅ Auto-scaling based on CPU, memory, and custom metrics")
    print("   ✅ Cost center tagging for billing")
    print("   ✅ ECS Exec enabled for debugging")
    print()
    
    print("📈 Auto-scaling Triggers:")
    print("   🎯 CPU utilization > 70%")
    print("   🎯 Memory utilization > 80%") 
    print("   🎯 Bedrock requests > 100/minute")
    print("   🚨 Cost per request > $0.02 (alert)")
    print()
    
    return service_definition, autoscaling_config


def create_monitoring_configuration():
    """Create CloudWatch monitoring and alerting configuration."""
    
    print("📊 CloudWatch Monitoring Setup")
    print("=" * 35)
    print("Comprehensive monitoring for ECS Bedrock service:")
    print()
    
    # CloudWatch dashboards
    dashboard_config = {
        "dashboard_name": "GenOps-Bedrock-ECS-Dashboard",
        "widgets": [
            {
                "type": "metric",
                "properties": {
                    "metrics": [
                        ["AWS/ECS", "CPUUtilization", "ServiceName", "genops-bedrock-service"],
                        ["AWS/ECS", "MemoryUtilization", "ServiceName", "genops-bedrock-service"],
                        ["AWS/ApplicationELB", "RequestCount", "TargetGroup", "genops-bedrock-tg"],
                        ["AWS/ApplicationELB", "ResponseTime", "TargetGroup", "genops-bedrock-tg"]
                    ],
                    "period": 300,
                    "stat": "Average",
                    "region": "us-east-1",
                    "title": "ECS Service Metrics"
                }
            },
            {
                "type": "metric",
                "properties": {
                    "metrics": [
                        ["GenOps/Bedrock", "OperationCount", "Service", "bedrock-ecs"],
                        ["GenOps/Bedrock", "TotalCost", "Service", "bedrock-ecs"],
                        ["GenOps/Bedrock", "AverageLatency", "Service", "bedrock-ecs"],
                        ["GenOps/Bedrock", "ErrorRate", "Service", "bedrock-ecs"]
                    ],
                    "period": 300,
                    "stat": "Sum",
                    "region": "us-east-1",
                    "title": "GenOps Bedrock Metrics"
                }
            }
        ]
    }
    
    # CloudWatch alarms
    alarms_config = [
        {
            "alarm_name": "GenOps-Bedrock-ECS-HighCPU",
            "description": "CPU utilization too high",
            "metric_name": "CPUUtilization",
            "namespace": "AWS/ECS",
            "threshold": 80,
            "comparison_operator": "GreaterThanThreshold",
            "evaluation_periods": 2,
            "actions": ["arn:aws:sns:us-east-1:ACCOUNT:genops-alerts"]
        },
        {
            "alarm_name": "GenOps-Bedrock-HighCostPerRequest",
            "description": "Cost per request exceeds budget",
            "metric_name": "CostPerRequest",
            "namespace": "GenOps/Bedrock",
            "threshold": 0.02,
            "comparison_operator": "GreaterThanThreshold",
            "evaluation_periods": 1,
            "actions": ["arn:aws:sns:us-east-1:ACCOUNT:genops-cost-alerts"]
        },
        {
            "alarm_name": "GenOps-Bedrock-ECS-HealthCheckFail",
            "description": "Health check failures detected",
            "metric_name": "UnHealthyHostCount",
            "namespace": "AWS/ApplicationELB",
            "threshold": 0,
            "comparison_operator": "GreaterThanThreshold",
            "evaluation_periods": 2,
            "actions": ["arn:aws:sns:us-east-1:ACCOUNT:genops-urgent-alerts"]
        }
    ]
    
    print("📈 Dashboard Widgets:")
    print("   📊 ECS service metrics (CPU, Memory, Requests)")
    print("   💰 GenOps cost and performance metrics")
    print("   🎯 Custom AI workload metrics")
    print("   ⚡ Real-time latency and throughput")
    print()
    
    print("🚨 Alert Conditions:")
    for alarm in alarms_config:
        print(f"   🔔 {alarm['alarm_name']}: {alarm['description']}")
    
    print()
    
    return dashboard_config, alarms_config


def main():
    """Main demonstration function."""
    
    print("🐳 Welcome to GenOps Bedrock ECS Integration!")
    print()
    print("This example demonstrates container deployment patterns")
    print("for AWS Bedrock with GenOps governance and auto-scaling.")
    print()
    
    demos = [
        ("Docker Configuration", create_dockerfile),
        ("Flask Application", create_flask_application),
        ("ECS Task Definition", create_ecs_task_definition),
        ("ECS Service Config", create_ecs_service_configuration),
        ("CloudWatch Monitoring", create_monitoring_configuration)
    ]
    
    results = {}
    
    for demo_name, demo_func in demos:
        print(f"🚀 {demo_name}")
        print("=" * (len(demo_name) + 3))
        
        try:
            result = demo_func()
            results[demo_name] = result
            print(f"✅ {demo_name} completed successfully\n")
        except Exception as e:
            print(f"❌ {demo_name} failed: {e}\n")
    
    # Summary
    print("🎉 ECS Integration Demo Summary")
    print("=" * 38)
    
    print("🏆 Container AI Features Demonstrated:")
    print("   🐳 Docker containerization with GenOps optimization")
    print("   📋 ECS Fargate deployment with auto-scaling")
    print("   🌐 Load balanced web service with health checks")
    print("   📊 CloudWatch monitoring and alerting")
    print("   💰 Cost-aware auto-scaling policies")
    print("   🛡️ Production-ready security and IAM")
    print()
    
    print("🚀 Deployment Instructions:")
    print("   1. Build Docker image: docker build -t genops-bedrock .")
    print("   2. Push to ECR: docker tag & docker push")
    print("   3. Register ECS task definition")
    print("   4. Create ECS service with load balancer")
    print("   5. Set up CloudWatch dashboards and alarms")
    print("   6. Configure auto-scaling policies")
    print()
    
    print("🎯 Next Steps:")
    print("   → ML pipelines: python sagemaker_integration.py")
    print("   → Set up CI/CD pipeline for container deployment")
    print("   → Implement blue-green deployments")
    print("   → Configure VPC endpoints for private networking")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)