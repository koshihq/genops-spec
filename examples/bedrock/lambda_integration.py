#!/usr/bin/env python3
"""
AWS Lambda + Bedrock Integration Example

This example demonstrates serverless deployment patterns for AWS Bedrock
with GenOps governance, optimized for Lambda cold starts and cost efficiency.

Features demonstrated:
- Lambda-optimized GenOps setup with minimal cold start overhead
- Event-driven AI processing with automatic scaling
- Cost-efficient serverless architectures
- Lambda-specific performance tuning and monitoring
- API Gateway integration patterns
- Step Functions workflow orchestration

Example usage:
    python lambda_integration.py

Note: This example shows Lambda deployment patterns. For actual Lambda deployment,
package the functions and deploy using AWS SAM, Serverless Framework, or CDK.
"""

import json
import os
import sys

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def create_lambda_handler_example():
    """Create example Lambda handler with GenOps Bedrock integration."""

    print("⚡ AWS Lambda Handler with GenOps Bedrock")
    print("=" * 45)
    print("Serverless AI processing with automatic governance and cost tracking:")
    print()

    # Lambda handler code example
    lambda_handler_code = '''
import json
import os
from typing import Dict, Any

# GenOps Bedrock integration for Lambda
from genops.providers.bedrock import GenOpsBedrockAdapter, instrument_bedrock
from genops.providers.bedrock_workflow import production_workflow_context, ComplianceLevel

# Enable auto-instrumentation for optimal Lambda performance
instrument_bedrock()

# Initialize adapter outside handler for connection reuse
adapter = GenOpsBedrockAdapter(
    region_name=os.environ.get('AWS_REGION', 'us-east-1'),
    default_model="anthropic.claude-3-haiku-20240307-v1:0"  # Fast model for Lambda
)

def lambda_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Lambda handler for AI-powered document analysis.

    Optimized for serverless with GenOps governance and cost tracking.
    """

    try:
        # Extract request data
        document_text = event.get('document_text', '')
        analysis_type = event.get('analysis_type', 'general')
        customer_id = event.get('customer_id', 'unknown')

        # Create serverless workflow context
        with production_workflow_context(
            workflow_name="lambda_document_analysis",
            customer_id=customer_id,
            team="serverless-ai",
            project="document-processing-api",
            environment="production",
            compliance_level=ComplianceLevel.SOC2,
            budget_limit=0.50,  # $0.50 per Lambda invocation
            region=os.environ.get('AWS_REGION', 'us-east-1')
        ) as (workflow, workflow_id):

            # Step 1: Document classification
            workflow.record_step("classification", {
                "analysis_type": analysis_type,
                "document_length": len(document_text)
            })

            classification_prompt = f"Classify this document as: {analysis_type}. Text: {document_text[:500]}"
            classification = adapter.text_generation(
                prompt=classification_prompt,
                model_id="anthropic.claude-3-haiku-20240307-v1:0",  # Fast for Lambda
                max_tokens=50,
                temperature=0.1,
                team="serverless-ai",
                customer_id=customer_id,
                feature=f"lambda_classification_{analysis_type}"
            )

            # Step 2: Content extraction based on classification
            workflow.record_step("extraction", {
                "classification_result": classification.content[:100]
            })

            extraction_prompt = f"Extract key information from this {classification.content} document: {document_text}"
            extraction = adapter.text_generation(
                prompt=extraction_prompt,
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                max_tokens=200,
                temperature=0.2,
                team="serverless-ai",
                customer_id=customer_id,
                feature=f"lambda_extraction_{analysis_type}"
            )

            # Record performance metrics
            workflow.record_performance_metric("lambda_execution_time",
                                             context.get_remaining_time_in_millis(), "milliseconds")
            workflow.record_performance_metric("document_chars_processed",
                                             len(document_text), "characters")

            # Get final cost summary
            final_cost = workflow.get_current_cost_summary()

            # Return results with governance data
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'workflow_id': workflow_id,
                    'classification': classification.content.strip(),
                    'extraction': extraction.content.strip(),
                    'cost': final_cost.total_cost,
                    'performance': {
                        'total_latency_ms': final_cost.total_latency_ms,
                        'models_used': list(final_cost.unique_models),
                        'total_operations': final_cost.total_operations
                    },
                    'governance': {
                        'customer_id': customer_id,
                        'team': 'serverless-ai',
                        'compliance_level': 'SOC2'
                    }
                }),
                'headers': {
                    'Content-Type': 'application/json',
                    'X-GenOps-Workflow-Id': workflow_id,
                    'X-GenOps-Cost': str(final_cost.total_cost)
                }
            }

    except Exception as e:
        # Error handling with GenOps context
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'error_type': type(e).__name__,
                'message': 'AI processing failed - check logs for details'
            })
        }
'''

    print("📄 Lambda Handler Code Generated:")
    print("   ✅ GenOps auto-instrumentation enabled")
    print("   ✅ Connection reuse for performance")
    print("   ✅ Production workflow context")
    print("   ✅ SOC2 compliance tracking")
    print("   ✅ Comprehensive error handling")
    print("   ✅ Cost and performance metrics")
    print()

    return lambda_handler_code


def create_sam_template():
    """Create AWS SAM template for deployment."""

    print("📦 AWS SAM Deployment Template")
    print("=" * 35)

    sam_template = {
        "AWSTemplateFormatVersion": "2010-09-09",
        "Transform": "AWS::Serverless-2016-10-31",
        "Description": "GenOps Bedrock Lambda Integration",
        "Globals": {
            "Function": {
                "Runtime": "python3.9",
                "Timeout": 300,
                "MemorySize": 1024,
                "Environment": {
                    "Variables": {
                        "GENOPS_ENVIRONMENT": "production",
                        "GENOPS_PROJECT": "bedrock-lambda-integration",
                        "OTEL_SERVICE_NAME": "bedrock-lambda-ai",
                        "GENOPS_SAMPLING_RATE": "1.0",
                        "GENOPS_ASYNC_EXPORT": "true",
                    }
                },
            }
        },
        "Resources": {
            "DocumentAnalysisFunction": {
                "Type": "AWS::Serverless::Function",
                "Properties": {
                    "CodeUri": "src/",
                    "Handler": "lambda_handler.lambda_handler",
                    "Description": "AI-powered document analysis with GenOps governance",
                    "Environment": {
                        "Variables": {
                            "GENOPS_DEFAULT_TEAM": "serverless-ai",
                            "GENOPS_DEFAULT_PROJECT": "document-processing",
                        }
                    },
                    "Events": {
                        "DocumentAnalysisApi": {
                            "Type": "Api",
                            "Properties": {"Path": "/analyze", "Method": "post"},
                        }
                    },
                    "Policies": [
                        "AWSLambdaBasicExecutionRole",
                        {
                            "Version": "2012-10-17",
                            "Statement": [
                                {
                                    "Effect": "Allow",
                                    "Action": [
                                        "bedrock:InvokeModel",
                                        "bedrock:InvokeModelWithResponseStream",
                                        "bedrock:ListFoundationModels",
                                    ],
                                    "Resource": "*",
                                },
                                {
                                    "Effect": "Allow",
                                    "Action": [
                                        "cloudtrail:PutEvents",
                                        "logs:CreateLogGroup",
                                        "logs:CreateLogStream",
                                        "logs:PutLogEvents",
                                    ],
                                    "Resource": "*",
                                },
                            ],
                        },
                    ],
                },
            },
            "BatchProcessingFunction": {
                "Type": "AWS::Serverless::Function",
                "Properties": {
                    "CodeUri": "src/",
                    "Handler": "batch_processor.lambda_handler",
                    "Timeout": 900,  # 15 minutes for batch processing
                    "MemorySize": 2048,
                    "Description": "Batch AI processing with cost optimization",
                    "Environment": {
                        "Variables": {
                            "GENOPS_BATCH_SIZE": "50",
                            "GENOPS_CIRCUIT_BREAKER": "true",
                        }
                    },
                    "Events": {
                        "S3TriggerEvent": {
                            "Type": "S3",
                            "Properties": {
                                "Bucket": {"Ref": "DocumentsBucket"},
                                "Events": "s3:ObjectCreated:*",
                                "Filter": {
                                    "S3Key": {
                                        "Rules": [
                                            {"Name": "prefix", "Value": "documents/"}
                                        ]
                                    }
                                },
                            },
                        }
                    },
                },
            },
            "DocumentsBucket": {
                "Type": "AWS::S3::Bucket",
                "Properties": {
                    "BucketName": {
                        "Fn::Sub": "genops-bedrock-documents-${AWS::AccountId}"
                    },
                    "NotificationConfiguration": {
                        "LambdaConfigurations": [
                            {
                                "Event": "s3:ObjectCreated:*",
                                "Function": {
                                    "Fn::GetAtt": ["BatchProcessingFunction", "Arn"]
                                },
                            }
                        ]
                    },
                },
            },
        },
        "Outputs": {
            "DocumentAnalysisApi": {
                "Description": "API Gateway endpoint URL for document analysis",
                "Value": {
                    "Fn::Sub": "https://${ServerlessRestApi}.execute-api.${AWS::Region}.amazonaws.com/Prod/analyze/"
                },
            },
            "DocumentsBucketName": {
                "Description": "S3 bucket for document uploads",
                "Value": {"Ref": "DocumentsBucket"},
            },
        },
    }

    print("🏗️ SAM Template Features:")
    print("   ✅ Two Lambda functions (API + Batch processing)")
    print("   ✅ API Gateway integration")
    print("   ✅ S3 trigger for batch processing")
    print("   ✅ Proper IAM permissions for Bedrock")
    print("   ✅ GenOps environment configuration")
    print("   ✅ CloudTrail integration")
    print()

    return sam_template


def create_step_functions_integration():
    """Create Step Functions workflow with GenOps Bedrock."""

    print("🔄 AWS Step Functions Integration")
    print("=" * 38)
    print("Complex AI workflow orchestration with state management:")
    print()

    step_functions_code = '''
import json
import boto3
from genops.providers.bedrock_workflow import production_workflow_context, ComplianceLevel
from genops.providers.bedrock import GenOpsBedrockAdapter

def document_classification_handler(event, context):
    """Step 1: Document classification"""

    document_text = event['document_text']
    workflow_id = event['workflow_id']
    customer_id = event['customer_id']

    adapter = GenOpsBedrockAdapter()

    classification = adapter.text_generation(
        prompt=f"Classify document type and sensitivity: {document_text[:500]}",
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        temperature=0.1,
        team="step-functions-ai",
        customer_id=customer_id,
        feature="document_classification"
    )

    return {
        'workflow_id': workflow_id,
        'document_text': document_text,
        'classification': classification.content.strip(),
        'classification_cost': classification.cost_usd,
        'customer_id': customer_id,
        'next_step': 'content_extraction'
    }

def content_extraction_handler(event, context):
    """Step 2: Content extraction based on classification"""

    classification = event['classification']
    document_text = event['document_text']
    customer_id = event['customer_id']

    adapter = GenOpsBedrockAdapter()

    # Choose model based on document classification
    if 'financial' in classification.lower() or 'legal' in classification.lower():
        model = "anthropic.claude-3-sonnet-20240229-v1:0"  # Higher accuracy for sensitive docs
    else:
        model = "anthropic.claude-3-haiku-20240307-v1:0"   # Cost-effective for general docs

    extraction = adapter.text_generation(
        prompt=f"Extract structured information from this {classification} document: {document_text}",
        model_id=model,
        max_tokens=300,
        temperature=0.2,
        team="step-functions-ai",
        customer_id=customer_id,
        feature=f"content_extraction_{classification.lower()}"
    )

    return {
        'workflow_id': event['workflow_id'],
        'classification': classification,
        'extraction': extraction.content.strip(),
        'extraction_cost': extraction.cost_usd,
        'model_used': model,
        'customer_id': customer_id,
        'total_cost': event.get('classification_cost', 0) + extraction.cost_usd
    }

def compliance_validation_handler(event, context):
    """Step 3: SOC2 compliance validation"""

    classification = event['classification']
    extraction = event['extraction']
    customer_id = event['customer_id']

    adapter = GenOpsBedrockAdapter()

    compliance_check = adapter.text_generation(
        prompt=f"Validate SOC2 compliance for {classification} document with extracted data: {extraction}",
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        max_tokens=100,
        temperature=0.1,
        team="step-functions-ai",
        customer_id=customer_id,
        feature="compliance_validation"
    )

    # Determine if workflow should continue based on compliance
    compliance_passed = "compliant" in compliance_check.content.lower()

    return {
        'workflow_id': event['workflow_id'],
        'classification': classification,
        'extraction': extraction,
        'compliance_status': compliance_check.content.strip(),
        'compliance_passed': compliance_passed,
        'customer_id': customer_id,
        'total_cost': event['total_cost'] + compliance_check.cost_usd,
        'workflow_complete': True
    }
'''

    step_functions_definition = {
        "Comment": "GenOps Bedrock Document Processing Workflow",
        "StartAt": "DocumentClassification",
        "States": {
            "DocumentClassification": {
                "Type": "Task",
                "Resource": "arn:aws:lambda:us-east-1:ACCOUNT:function:DocumentClassificationFunction",
                "Next": "ContentExtraction",
                "Retry": [
                    {
                        "ErrorEquals": [
                            "Lambda.ServiceException",
                            "Lambda.AWSLambdaException",
                        ],
                        "IntervalSeconds": 2,
                        "MaxAttempts": 3,
                        "BackoffRate": 2.0,
                    }
                ],
                "Catch": [{"ErrorEquals": ["States.ALL"], "Next": "WorkflowFailed"}],
            },
            "ContentExtraction": {
                "Type": "Task",
                "Resource": "arn:aws:lambda:us-east-1:ACCOUNT:function:ContentExtractionFunction",
                "Next": "ComplianceValidation",
                "Retry": [
                    {
                        "ErrorEquals": [
                            "Lambda.ServiceException",
                            "Lambda.AWSLambdaException",
                        ],
                        "IntervalSeconds": 2,
                        "MaxAttempts": 3,
                        "BackoffRate": 2.0,
                    }
                ],
            },
            "ComplianceValidation": {
                "Type": "Task",
                "Resource": "arn:aws:lambda:us-east-1:ACCOUNT:function:ComplianceValidationFunction",
                "Next": "ComplianceCheck",
            },
            "ComplianceCheck": {
                "Type": "Choice",
                "Choices": [
                    {
                        "Variable": "$.compliance_passed",
                        "BooleanEquals": True,
                        "Next": "WorkflowSuccess",
                    }
                ],
                "Default": "ComplianceFailure",
            },
            "WorkflowSuccess": {"Type": "Succeed"},
            "ComplianceFailure": {
                "Type": "Fail",
                "Error": "ComplianceValidationFailed",
                "Cause": "Document failed SOC2 compliance validation",
            },
            "WorkflowFailed": {
                "Type": "Fail",
                "Error": "WorkflowExecutionFailed",
                "Cause": "Workflow execution encountered an error",
            },
        },
    }

    print("🔄 Step Functions Workflow Features:")
    print("   ✅ Multi-step AI processing pipeline")
    print("   ✅ Intelligent model selection based on content")
    print("   ✅ SOC2 compliance validation with branching logic")
    print("   ✅ Error handling and retry policies")
    print("   ✅ Cost tracking across workflow steps")
    print("   ✅ Conditional workflow paths based on AI analysis")
    print()

    return step_functions_code, step_functions_definition


def demonstrate_api_gateway_integration():
    """Demonstrate API Gateway patterns with GenOps Bedrock."""

    print("🌐 API Gateway Integration Patterns")
    print("=" * 40)
    print("RESTful API for AI services with comprehensive governance:")
    print()

    # API Gateway configuration
    api_patterns = [
        {
            "endpoint": "POST /analyze/document",
            "description": "Single document analysis",
            "lambda": "DocumentAnalysisFunction",
            "features": ["Real-time processing", "SOC2 compliance", "Cost tracking"],
        },
        {
            "endpoint": "POST /analyze/batch",
            "description": "Batch document processing",
            "lambda": "BatchProcessingFunction",
            "features": ["Async processing", "Cost optimization", "Progress tracking"],
        },
        {
            "endpoint": "GET /workflows/{workflow_id}",
            "description": "Workflow status and costs",
            "lambda": "WorkflowStatusFunction",
            "features": ["Real-time status", "Cost breakdown", "Performance metrics"],
        },
        {
            "endpoint": "GET /analytics/costs",
            "description": "Cost analytics and optimization",
            "lambda": "CostAnalyticsFunction",
            "features": ["Cost trends", "Model recommendations", "Budget alerts"],
        },
    ]

    print("🔗 API Endpoints:")
    for pattern in api_patterns:
        print(f"   📍 {pattern['endpoint']}")
        print(f"      {pattern['description']}")
        print(f"      Lambda: {pattern['lambda']}")
        for feature in pattern["features"]:
            print(f"      ✅ {feature}")
        print()

    # Request/Response examples
    print("📨 Example Request/Response:")

    example_request = {
        "document_text": "QUARTERLY FINANCIAL REPORT Q3 2024...",
        "analysis_type": "financial",
        "customer_id": "enterprise-client-123",
        "options": {
            "compliance_level": "SOC2",
            "budget_limit": 0.50,
            "priority": "high",
        },
    }

    example_response = {
        "workflow_id": "wf_bedrock_20241104_001",
        "classification": "Financial quarterly report",
        "extraction": {
            "revenue": "$2.3B",
            "net_income": "$450M",
            "growth_rate": "15% YoY",
        },
        "cost": 0.023,
        "performance": {
            "total_latency_ms": 2150,
            "models_used": ["claude-3-haiku", "claude-3-sonnet"],
            "total_operations": 2,
        },
        "compliance": {"soc2_validated": True, "audit_trail_id": "audit_20241104_001"},
    }

    print(f"📤 Request: {json.dumps(example_request, indent=2)}")
    print()
    print(f"📥 Response: {json.dumps(example_response, indent=2)}")
    print()


def demonstrate_cost_optimization_patterns():
    """Demonstrate Lambda-specific cost optimization patterns."""

    print("💰 Lambda Cost Optimization Patterns")
    print("=" * 42)
    print("Serverless-specific strategies for minimizing costs:")
    print()

    optimization_strategies = [
        {
            "strategy": "Cold Start Optimization",
            "description": "Minimize initialization overhead",
            "techniques": [
                "Connection pooling outside handler",
                "Lazy loading of GenOps components",
                "Provisioned concurrency for critical functions",
                "Smaller deployment packages",
            ],
        },
        {
            "strategy": "Model Selection Based on Lambda Timeout",
            "description": "Choose models based on available execution time",
            "techniques": [
                "Fast models (Claude Haiku) for short timeouts",
                "Premium models (Claude Sonnet) for longer processing",
                "Dynamic model selection based on remaining time",
                "Timeout-aware batch processing",
            ],
        },
        {
            "strategy": "Memory and CPU Optimization",
            "description": "Balance memory allocation with cost",
            "techniques": [
                "1024MB for standard AI processing",
                "2048MB+ for batch operations",
                "CPU-intensive models need higher memory",
                "Monitor actual memory usage",
            ],
        },
        {
            "strategy": "Request Batching",
            "description": "Process multiple requests per invocation",
            "techniques": [
                "SQS trigger with batch size optimization",
                "S3 event batching for file processing",
                "API Gateway request aggregation",
                "Cost amortization across batch items",
            ],
        },
    ]

    for strategy in optimization_strategies:
        print(f"🎯 {strategy['strategy']}:")
        print(f"   {strategy['description']}")
        for technique in strategy["techniques"]:
            print(f"   ✅ {technique}")
        print()

    # Cost comparison example
    print("📊 Lambda Cost Scenarios:")

    cost_scenarios = [
        {
            "scenario": "Single document (128MB)",
            "cost": "$0.0001",
            "ai_cost": "$0.002",
            "total": "$0.0021",
        },
        {
            "scenario": "Single document (1024MB)",
            "cost": "$0.0005",
            "ai_cost": "$0.002",
            "total": "$0.0025",
        },
        {
            "scenario": "Batch 10 docs (2048MB)",
            "cost": "$0.0020",
            "ai_cost": "$0.015",
            "total": "$0.017",
        },
        {
            "scenario": "Complex analysis (1024MB)",
            "cost": "$0.0008",
            "ai_cost": "$0.008",
            "total": "$0.0088",
        },
    ]

    for scenario in cost_scenarios:
        print(f"   💳 {scenario['scenario']}: {scenario['total']} total")
        print(f"      (Lambda: {scenario['cost']}, AI: {scenario['ai_cost']})")

    print()


def main():
    """Main demonstration function."""

    print("⚡ Welcome to GenOps Bedrock Lambda Integration!")
    print()
    print("This example demonstrates serverless deployment patterns")
    print("for AWS Bedrock with GenOps governance and cost optimization.")
    print()

    demos = [
        ("Lambda Handler Example", create_lambda_handler_example),
        ("SAM Deployment Template", create_sam_template),
        ("Step Functions Integration", create_step_functions_integration),
        ("API Gateway Patterns", demonstrate_api_gateway_integration),
        ("Cost Optimization", demonstrate_cost_optimization_patterns),
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
    print("🎉 Lambda Integration Demo Summary")
    print("=" * 40)

    print("🏆 Serverless AI Features Demonstrated:")
    print("   ⚡ Lambda-optimized GenOps integration")
    print("   🔄 Step Functions workflow orchestration")
    print("   🌐 API Gateway RESTful endpoints")
    print("   📦 AWS SAM deployment templates")
    print("   💰 Serverless cost optimization strategies")
    print("   🛡️ Enterprise governance in serverless architecture")
    print()

    print("🚀 Deployment Instructions:")
    print("   1. Save the Lambda handler code to src/lambda_handler.py")
    print("   2. Create requirements.txt with: genops-ai[bedrock]")
    print("   3. Use the SAM template for deployment: sam deploy")
    print("   4. Configure API Gateway endpoints")
    print("   5. Set up monitoring with CloudWatch")
    print()

    print("🎯 Next Steps:")
    print("   → Container deployment: python ecs_integration.py")
    print("   → ML pipelines: python sagemaker_integration.py")
    print("   → Set up CloudWatch dashboards for serverless monitoring")
    print("   → Implement API throttling and rate limiting")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
