#!/usr/bin/env python3
"""
AWS SageMaker + Bedrock Integration Example

This example demonstrates ML pipeline integration patterns combining
AWS SageMaker and Bedrock with GenOps governance for end-to-end MLOps.

Features demonstrated:
- SageMaker pipeline integration with Bedrock foundation models
- Model training cost attribution alongside inference costs
- MLOps workflows with comprehensive governance
- Data science experiment tracking with GenOps
- Model versioning and A/B testing patterns
- SageMaker Inference Endpoints with Bedrock augmentation

Example usage:
    python sagemaker_integration.py

Note: This example shows SageMaker integration patterns. For actual deployment,
use SageMaker SDK and configure IAM roles for cross-service access.
"""

import os
import sys

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def create_sagemaker_training_integration():
    """Demonstrate SageMaker training with Bedrock augmentation."""

    print("🧠 SageMaker Training + Bedrock Integration")
    print("=" * 48)
    print("ML model training augmented with foundation model capabilities:")
    print()

    training_script = '''
import os
import json
import boto3
import argparse
import pandas as pd
from sagemaker.session import Session
from sagemaker.experiments import Run

# GenOps integration for comprehensive governance
from genops.providers.bedrock import GenOpsBedrockAdapter
from genops.providers.bedrock_workflow import production_workflow_context, ComplianceLevel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='/opt/ml/input/data/training')
    parser.add_argument('--model-dir', type=str, default='/opt/ml/model')
    parser.add_argument('--experiment-name', type=str, required=True)
    parser.add_argument('--customer-id', type=str, required=True)
    return parser.parse_args()

def augment_training_data_with_bedrock(data_df, customer_id, bedrock_adapter):
    """Augment training data using Bedrock foundation models."""

    augmented_samples = []

    for idx, row in data_df.iterrows():
        original_text = row['text']
        label = row['label']

        # Generate synthetic variations using Bedrock
        variations = bedrock_adapter.text_generation(
            prompt=f"Generate 3 paraphrased versions of this text while keeping the same meaning: {original_text}",
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            max_tokens=200,
            temperature=0.7,
            team="ml-training",
            customer_id=customer_id,
            feature="data_augmentation"
        )

        # Parse variations and add to dataset
        for i, variation in enumerate(variations.content.split('\\n')[:3]):
            if variation.strip():
                augmented_samples.append({
                    'text': variation.strip(),
                    'label': label,
                    'source': 'bedrock_augmented',
                    'original_idx': idx
                })

    return pd.DataFrame(augmented_samples)

def train_model_with_governance(args):
    """Train ML model with comprehensive GenOps governance."""

    with production_workflow_context(
        workflow_name="sagemaker_training_with_bedrock",
        customer_id=args.customer_id,
        team="ml-engineering",
        project="foundation-model-augmented-training",
        environment="training",
        compliance_level=ComplianceLevel.SOC2,
        cost_center="ML-Training-Infrastructure"
    ) as (workflow, workflow_id):

        # Initialize GenOps Bedrock adapter
        bedrock_adapter = GenOpsBedrockAdapter(
            region_name=os.environ.get('AWS_REGION', 'us-east-1')
        )

        # Step 1: Load and analyze training data
        workflow.record_step("data_loading", {
            'data_path': args.data_path,
            'experiment_name': args.experiment_name
        })

        print(f"Loading training data from {args.data_path}")
        train_df = pd.read_csv(os.path.join(args.data_path, 'train.csv'))

        # Analyze data characteristics using Bedrock
        data_analysis = bedrock_adapter.text_generation(
            prompt=f"Analyze this training dataset structure and suggest improvements: {train_df.head().to_string()}",
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            max_tokens=300,
            temperature=0.3,
            team="ml-engineering",
            customer_id=args.customer_id,
            feature="data_analysis"
        )

        workflow.record_performance_metric("original_samples", len(train_df), "count")

        # Step 2: Data augmentation using Bedrock
        workflow.record_step("data_augmentation", {
            'augmentation_model': "claude-3-haiku",
            'target_augmentation_ratio': 2.0
        })

        print("Augmenting training data using Bedrock...")
        augmented_df = augment_training_data_with_bedrock(
            train_df, args.customer_id, bedrock_adapter
        )

        # Combine original and augmented data
        combined_df = pd.concat([train_df, augmented_df], ignore_index=True)
        workflow.record_performance_metric("augmented_samples", len(augmented_df), "count")
        workflow.record_performance_metric("total_samples", len(combined_df), "count")

        # Step 3: Model training (simplified)
        workflow.record_step("model_training", {
            'model_type': 'custom_classifier',
            'training_samples': len(combined_df)
        })

        print("Training model with augmented dataset...")
        # Simulate training process
        training_metrics = {
            'accuracy': 0.94,
            'f1_score': 0.91,
            'training_time': 3600,  # 1 hour
            'epochs': 10
        }

        # Record training metrics
        for metric_name, value in training_metrics.items():
            workflow.record_performance_metric(f"training_{metric_name}", value,
                                             "percentage" if "accuracy" in metric_name or "f1" in metric_name else "seconds" if "time" in metric_name else "count")

        # Step 4: Model validation using Bedrock
        workflow.record_step("model_validation", {
            'validation_method': 'bedrock_assisted'
        })

        validation_analysis = bedrock_adapter.text_generation(
            prompt=f"Analyze these training results and suggest improvements: Accuracy: {training_metrics['accuracy']}, F1: {training_metrics['f1_score']}. Training samples: {len(combined_df)}",
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            max_tokens=250,
            temperature=0.2,
            team="ml-engineering",
            customer_id=args.customer_id,
            feature="model_validation"
        )

        # Save model and metadata
        model_metadata = {
            'workflow_id': workflow_id,
            'training_cost': workflow.get_current_cost_summary().total_cost,
            'original_samples': len(train_df),
            'augmented_samples': len(augmented_df),
            'training_metrics': training_metrics,
            'bedrock_analysis': data_analysis.content[:200],
            'validation_analysis': validation_analysis.content[:200]
        }

        # Save to model directory
        os.makedirs(args.model_dir, exist_ok=True)
        with open(os.path.join(args.model_dir, 'model_metadata.json'), 'w') as f:
            json.dump(model_metadata, f, indent=2)

        print(f"Model training completed. Workflow ID: {workflow_id}")
        print(f"Total training cost (including Bedrock): ${workflow.get_current_cost_summary().total_cost:.4f}")

if __name__ == "__main__":
    args = parse_args()
    train_model_with_governance(args)
'''

    print("🏋️ Training Script Features:")
    print("   ✅ Data augmentation using Bedrock foundation models")
    print("   ✅ Comprehensive training governance with cost tracking")
    print("   ✅ ML experiment tracking integration")
    print("   ✅ Model validation assisted by Bedrock analysis")
    print("   ✅ Training cost attribution alongside infrastructure costs")
    print("   ✅ SOC2 compliance for training workflows")
    print()

    return training_script


def create_sagemaker_inference_integration():
    """Demonstrate SageMaker inference endpoint with Bedrock augmentation."""

    print("🔮 SageMaker Inference + Bedrock Hybrid")
    print("=" * 42)
    print("Hybrid inference combining custom models with foundation models:")
    print()

    inference_code = '''
import json
import boto3
import numpy as np
from typing import Dict, Any, List

# GenOps integration for inference governance
from genops.providers.bedrock import GenOpsBedrockAdapter
from genops.providers.bedrock_cost_aggregator import create_bedrock_cost_context

class HybridInferenceHandler:
    """
    SageMaker inference handler that combines custom models with Bedrock.

    This enables sophisticated AI pipelines that leverage both custom trained
    models and foundation model capabilities with unified cost tracking.
    """

    def __init__(self):
        self.bedrock_adapter = GenOpsBedrockAdapter(
            region_name='us-east-1',
            default_model="anthropic.claude-3-haiku-20240307-v1:0"
        )
        self.custom_model = None  # Load your custom model here

    def model_fn(self, model_dir: str):
        """Load custom model for SageMaker inference."""
        # Load custom model
        print(f"Loading custom model from {model_dir}")

        # Load model metadata including GenOps workflow info
        with open(f"{model_dir}/model_metadata.json", 'r') as f:
            self.model_metadata = json.load(f)

        return self

    def predict_fn(self, input_data: Dict[str, Any], model) -> Dict[str, Any]:
        """Hybrid prediction using custom model + Bedrock."""

        text_inputs = input_data.get('texts', [])
        customer_id = input_data.get('customer_id', 'unknown')
        prediction_type = input_data.get('type', 'classification')

        # Use cost context for unified tracking
        with create_bedrock_cost_context(f"hybrid_inference_{customer_id}") as cost_context:

            results = []

            for text in text_inputs:
                try:
                    # Step 1: Custom model prediction
                    custom_prediction = self._custom_model_predict(text)

                    # Step 2: Bedrock augmentation based on confidence
                    if custom_prediction['confidence'] < 0.8:
                        # Low confidence - augment with Bedrock
                        bedrock_analysis = self.bedrock_adapter.text_generation(
                            prompt=f"Analyze and classify this text: {text}. Provide confidence reasoning.",
                            model_id="anthropic.claude-3-haiku-20240307-v1:0",
                            max_tokens=150,
                            temperature=0.2,
                            team="ml-inference",
                            customer_id=customer_id,
                            feature=f"confidence_boost_{prediction_type}"
                        )

                        # Combine predictions
                        final_prediction = self._combine_predictions(
                            custom_prediction,
                            bedrock_analysis.content,
                            bedrock_analysis.cost_usd
                        )
                    else:
                        # High confidence - use custom model only
                        final_prediction = custom_prediction
                        final_prediction['bedrock_used'] = False
                        final_prediction['bedrock_cost'] = 0.0

                    results.append(final_prediction)

                except Exception as e:
                    results.append({
                        'text': text,
                        'error': str(e),
                        'prediction': None
                    })

            # Get total cost summary
            cost_summary = cost_context.get_current_summary()

            return {
                'predictions': results,
                'cost_breakdown': {
                    'total_bedrock_cost': cost_summary.total_cost,
                    'sagemaker_inference_cost': len(text_inputs) * 0.001,  # Estimated
                    'total_cost': cost_summary.total_cost + (len(text_inputs) * 0.001)
                },
                'performance': {
                    'total_latency_ms': cost_summary.total_latency_ms,
                    'bedrock_operations': cost_summary.total_operations,
                    'texts_processed': len(text_inputs)
                },
                'governance': {
                    'customer_id': customer_id,
                    'team': 'ml-inference',
                    'workflow_context': cost_context.context_id
                }
            }

    def _custom_model_predict(self, text: str) -> Dict[str, Any]:
        """Simulate custom model prediction."""
        # Simulate model inference
        prediction = {
            'text': text,
            'predicted_class': 'positive' if len(text) % 2 == 0 else 'negative',
            'confidence': 0.75 + (len(text) % 10) * 0.02,  # Simulate varying confidence
            'model_version': self.model_metadata.get('workflow_id', 'unknown')
        }
        return prediction

    def _combine_predictions(self, custom_pred: Dict, bedrock_analysis: str, bedrock_cost: float) -> Dict[str, Any]:
        """Combine custom model and Bedrock predictions."""

        # Simple combination logic - in practice, this would be more sophisticated
        bedrock_confidence = 0.9 if 'confident' in bedrock_analysis.lower() else 0.7

        # Weighted average of confidences
        combined_confidence = (custom_pred['confidence'] * 0.6) + (bedrock_confidence * 0.4)

        return {
            'text': custom_pred['text'],
            'predicted_class': custom_pred['predicted_class'],
            'confidence': combined_confidence,
            'custom_model_confidence': custom_pred['confidence'],
            'bedrock_analysis': bedrock_analysis[:100],
            'bedrock_used': True,
            'bedrock_cost': bedrock_cost,
            'model_version': custom_pred['model_version']
        }

# Handler instance for SageMaker
handler = HybridInferenceHandler()

def model_fn(model_dir):
    return handler.model_fn(model_dir)

def predict_fn(input_data, model):
    return model.predict_fn(input_data, model)
'''

    print("🎯 Hybrid Inference Features:")
    print("   ✅ Custom model + Bedrock foundation model combination")
    print("   ✅ Confidence-based intelligent routing")
    print("   ✅ Unified cost tracking across both models")
    print("   ✅ Real-time performance monitoring")
    print("   ✅ Customer attribution for inference costs")
    print("   ✅ Governance context preservation")
    print()

    return inference_code


def create_sagemaker_pipeline_integration():
    """Create SageMaker Pipeline with Bedrock integration."""

    print("🔄 SageMaker Pipeline + Bedrock MLOps")
    print("=" * 42)
    print("End-to-end ML pipeline with foundation model integration:")
    print()

    pipeline_code = '''
import boto3
from sagemaker import get_execution_role
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.pipeline import Pipeline
from sagemaker.pipeline.steps import TrainingStep, ProcessingStep
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.parameters import ParameterString, ParameterFloat

# GenOps workflow integration
from genops.providers.bedrock_workflow import production_workflow_context, ComplianceLevel

def create_genops_ml_pipeline(
    customer_id: str,
    project_name: str,
    bedrock_augmentation: bool = True,
    budget_limit: float = 100.0
):
    """Create SageMaker Pipeline with GenOps governance."""

    # SageMaker session and role
    session = boto3.Session()
    role = get_execution_role()

    # Pipeline parameters
    customer_param = ParameterString(name="CustomerID", default_value=customer_id)
    budget_param = ParameterFloat(name="BudgetLimit", default_value=budget_limit)

    # Create GenOps governance context for the entire pipeline
    with production_workflow_context(
        workflow_name="sagemaker_ml_pipeline",
        customer_id=customer_id,
        team="ml-platform",
        project=project_name,
        environment="production",
        compliance_level=ComplianceLevel.SOC2,
        budget_limit=budget_limit,
        cost_center="ML-Platform-Engineering"
    ) as (workflow, workflow_id):

        # Step 1: Data preprocessing with Bedrock augmentation
        if bedrock_augmentation:
            preprocessing_step = ProcessingStep(
                name="BedrockDataAugmentation",
                processor=SKLearn(
                    framework_version="0.23-1",
                    instance_type="ml.m5.xlarge",
                    instance_count=1,
                    role=role,
                    entry_point="preprocess_with_bedrock.py",
                    source_dir="code",
                    env={
                        'GENOPS_WORKFLOW_ID': workflow_id,
                        'GENOPS_CUSTOMER_ID': customer_id,
                        'GENOPS_TEAM': 'ml-platform'
                    }
                ),
                inputs=[
                    ProcessingInput(
                        source=f"s3://ml-data-bucket/{customer_id}/raw/",
                        destination="/opt/ml/processing/input"
                    )
                ],
                outputs=[
                    ProcessingOutput(
                        output_name="augmented_data",
                        source="/opt/ml/processing/output",
                        destination=f"s3://ml-data-bucket/{customer_id}/processed/"
                    )
                ]
            )

        # Step 2: Model training with governance
        training_step = TrainingStep(
            name="ModelTrainingWithGovernance",
            estimator=SKLearn(
                framework_version="0.23-1",
                instance_type="ml.m5.2xlarge",
                instance_count=1,
                role=role,
                entry_point="train_with_governance.py",
                source_dir="code",
                hyperparameters={
                    'customer-id': customer_param,
                    'budget-limit': budget_param,
                    'workflow-id': workflow_id
                },
                env={
                    'GENOPS_WORKFLOW_ID': workflow_id,
                    'GENOPS_BUDGET_LIMIT': str(budget_limit)
                }
            ),
            inputs={
                "training": TrainingInput(
                    s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs["augmented_data"].S3Output.S3Uri
                    if bedrock_augmentation else f"s3://ml-data-bucket/{customer_id}/raw/"
                )
            }
        )

        # Step 3: Model evaluation with Bedrock analysis
        evaluation_step = ProcessingStep(
            name="BedrockModelEvaluation",
            processor=SKLearn(
                framework_version="0.23-1",
                instance_type="ml.m5.large",
                instance_count=1,
                role=role,
                entry_point="evaluate_with_bedrock.py",
                source_dir="code"
            ),
            inputs=[
                ProcessingInput(
                    source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                    destination="/opt/ml/processing/model"
                ),
                ProcessingInput(
                    source=f"s3://ml-data-bucket/{customer_id}/test/",
                    destination="/opt/ml/processing/test"
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation_report",
                    source="/opt/ml/processing/evaluation",
                    destination=f"s3://ml-results-bucket/{customer_id}/evaluation/"
                )
            ]
        )

        # Create pipeline
        pipeline_steps = []
        if bedrock_augmentation:
            pipeline_steps.append(preprocessing_step)
        pipeline_steps.extend([training_step, evaluation_step])

        pipeline = Pipeline(
            name=f"GenOps-ML-Pipeline-{customer_id}",
            parameters=[customer_param, budget_param],
            steps=pipeline_steps,
            pipeline_definition_config={
                "PipelineDefinitionConfig": {
                    "UseCompiledCode": False
                }
            }
        )

        # Record pipeline creation in workflow
        workflow.record_step("pipeline_creation", {
            'pipeline_name': pipeline.name,
            'steps_count': len(pipeline_steps),
            'bedrock_augmentation': bedrock_augmentation,
            'customer_id': customer_id
        })

        workflow.record_performance_metric("pipeline_steps", len(pipeline_steps), "count")

        return pipeline, workflow_id

def execute_pipeline_with_monitoring(pipeline, workflow_id: str, customer_id: str):
    """Execute pipeline with GenOps monitoring."""

    print(f"Executing ML pipeline with GenOps governance...")
    print(f"Pipeline: {pipeline.name}")
    print(f"Workflow ID: {workflow_id}")
    print(f"Customer: {customer_id}")

    # Execute pipeline
    execution = pipeline.start(
        execution_display_name=f"genops-execution-{int(time.time())}",
        parameters={
            'CustomerID': customer_id,
            'BudgetLimit': 100.0
        }
    )

    print(f"Pipeline execution started: {execution.arn}")

    # Monitor execution (simplified)
    execution_steps = [
        {'name': 'BedrockDataAugmentation', 'status': 'Completed', 'duration': 600},
        {'name': 'ModelTrainingWithGovernance', 'status': 'Completed', 'duration': 3600},
        {'name': 'BedrockModelEvaluation', 'status': 'Completed', 'duration': 300}
    ]

    total_duration = sum(step['duration'] for step in execution_steps)

    print(f"Pipeline completed in {total_duration} seconds")
    print("Step summary:")
    for step in execution_steps:
        print(f"  ✅ {step['name']}: {step['status']} ({step['duration']}s)")

    return execution

# Example usage
if __name__ == "__main__":
    pipeline, workflow_id = create_genops_ml_pipeline(
        customer_id="enterprise-ml-client",
        project_name="customer-sentiment-analysis",
        bedrock_augmentation=True,
        budget_limit=150.0
    )

    # Upsert pipeline
    pipeline.upsert(role_arn=get_execution_role())

    # Execute with monitoring
    execution = execute_pipeline_with_monitoring(pipeline, workflow_id, "enterprise-ml-client")
'''

    print("🏭 SageMaker Pipeline Features:")
    print("   ✅ End-to-end ML pipeline with Bedrock integration")
    print("   ✅ Data augmentation using foundation models")
    print("   ✅ Governance context throughout pipeline execution")
    print("   ✅ Cost attribution across training and inference")
    print("   ✅ Automated model evaluation with AI analysis")
    print("   ✅ Budget limits and cost monitoring")
    print()

    return pipeline_code


def create_model_monitoring_integration():
    """Create SageMaker Model Monitor integration with Bedrock analysis."""

    print("📊 Model Monitoring + Bedrock Analysis")
    print("=" * 43)
    print("Intelligent model monitoring with foundation model insights:")
    print()

    monitoring_code = '''
import json
import boto3
from datetime import datetime, timedelta
from sagemaker.model_monitor import ModelMonitor, DataCaptureConfig

# GenOps integration for monitoring governance
from genops.providers.bedrock import GenOpsBedrockAdapter
from genops.providers.bedrock_cost_aggregator import create_bedrock_cost_context

class GenOpsModelMonitor:
    """
    Enhanced SageMaker Model Monitor with Bedrock-powered analysis.

    Combines traditional model monitoring with AI-powered insights
    for comprehensive model governance and performance analysis.
    """

    def __init__(self, endpoint_name: str, customer_id: str):
        self.endpoint_name = endpoint_name
        self.customer_id = customer_id
        self.bedrock_adapter = GenOpsBedrockAdapter()

        # Initialize SageMaker Model Monitor
        self.monitor = ModelMonitor(
            role=get_execution_role(),
            instance_count=1,
            instance_type='ml.m5.xlarge',
            volume_size_in_gb=20,
            max_runtime_in_seconds=3600
        )

    def setup_data_capture(self, s3_capture_path: str):
        """Set up data capture for the endpoint."""

        data_capture_config = DataCaptureConfig(
            enable_capture=True,
            sampling_percentage=100,  # Capture all requests
            destination_s3_uri=s3_capture_path,
            capture_options=["REQUEST", "RESPONSE"],
            csv_content_types=["application/json"],
            json_content_types=["application/json"]
        )

        return data_capture_config

    def analyze_model_drift_with_bedrock(self, monitoring_results: dict) -> dict:
        """Analyze model drift using Bedrock for intelligent insights."""

        with create_bedrock_cost_context(f"model_monitoring_{self.customer_id}") as cost_context:

            # Prepare monitoring data for analysis
            drift_metrics = monitoring_results.get('drift_metrics', {})
            performance_metrics = monitoring_results.get('performance_metrics', {})

            analysis_prompt = f"""
            Analyze this model monitoring report and provide actionable insights:

            Model Endpoint: {self.endpoint_name}
            Customer: {self.customer_id}

            Drift Metrics:
            - Data drift score: {drift_metrics.get('data_drift_score', 'N/A')}
            - Feature drift: {drift_metrics.get('feature_drift', 'N/A')}
            - Prediction drift: {drift_metrics.get('prediction_drift', 'N/A')}

            Performance Metrics:
            - Accuracy: {performance_metrics.get('accuracy', 'N/A')}
            - Latency P95: {performance_metrics.get('latency_p95', 'N/A')}ms
            - Error rate: {performance_metrics.get('error_rate', 'N/A')}%

            Provide:
            1. Risk assessment (Low/Medium/High)
            2. Root cause analysis
            3. Specific recommendations for improvement
            4. Urgency level for action needed
            """

            # Get Bedrock analysis
            drift_analysis = self.bedrock_adapter.text_generation(
                prompt=analysis_prompt,
                model_id="anthropic.claude-3-sonnet-20240229-v1:0",  # Use powerful model for analysis
                max_tokens=400,
                temperature=0.2,  # Low temperature for consistent analysis
                team="ml-monitoring",
                customer_id=self.customer_id,
                feature="model_drift_analysis"
            )

            # Parse analysis for structured insights
            insights = {
                'bedrock_analysis': drift_analysis.content,
                'analysis_cost': drift_analysis.cost_usd,
                'risk_level': 'Medium',  # Would parse from analysis in practice
                'recommendations': [],   # Would extract from analysis
                'urgency': 'Monitor',   # Would determine from analysis
                'timestamp': datetime.utcnow().isoformat()
            }

            # Extract risk level and recommendations (simplified)
            if 'high risk' in drift_analysis.content.lower():
                insights['risk_level'] = 'High'
                insights['urgency'] = 'Immediate'
            elif 'low risk' in drift_analysis.content.lower():
                insights['risk_level'] = 'Low'
                insights['urgency'] = 'Routine'

            return insights

    def create_monitoring_schedule(self, baseline_s3_uri: str, output_s3_uri: str):
        """Create monitoring schedule with Bedrock analysis integration."""

        # Create monitoring schedule
        monitor_schedule_name = f"genops-monitor-{self.endpoint_name}-{int(time.time())}"

        self.monitor.create_monitoring_schedule(
            monitor_schedule_name=monitor_schedule_name,
            endpoint_input=self.endpoint_name,
            output_s3_uri=output_s3_uri,
            statistics=f"{baseline_s3_uri}/statistics.json",
            constraints=f"{baseline_s3_uri}/constraints.json",
            schedule_cron_expression="cron(0 */6 * * * ?)",  # Every 6 hours
            enable_cloudwatch_metrics=True
        )

        return monitor_schedule_name

    def process_monitoring_report(self, report_s3_path: str) -> dict:
        """Process monitoring report with Bedrock insights."""

        # Simulate loading monitoring report
        monitoring_results = {
            'drift_metrics': {
                'data_drift_score': 0.23,
                'feature_drift': {'feature_1': 0.15, 'feature_2': 0.31},
                'prediction_drift': 0.18
            },
            'performance_metrics': {
                'accuracy': 0.89,
                'latency_p95': 245,
                'error_rate': 2.3,
                'throughput_rps': 45
            }
        }

        # Get Bedrock analysis
        insights = self.analyze_model_drift_with_bedrock(monitoring_results)

        # Create comprehensive report
        comprehensive_report = {
            'endpoint_name': self.endpoint_name,
            'customer_id': self.customer_id,
            'monitoring_results': monitoring_results,
            'bedrock_insights': insights,
            'report_timestamp': datetime.utcnow().isoformat(),
            'next_actions': []
        }

        # Determine next actions based on risk level
        if insights['risk_level'] == 'High':
            comprehensive_report['next_actions'] = [
                'Immediate model retraining recommended',
                'Review data pipeline for quality issues',
                'Consider A/B test with updated model',
                'Alert ML engineering team'
            ]
        elif insights['risk_level'] == 'Medium':
            comprehensive_report['next_actions'] = [
                'Schedule model retraining within 1 week',
                'Monitor closely for trend changes',
                'Review feature engineering pipeline'
            ]
        else:
            comprehensive_report['next_actions'] = [
                'Continue routine monitoring',
                'Document performance trends'
            ]

        return comprehensive_report

# Example monitoring setup
def setup_genops_model_monitoring(endpoint_name: str, customer_id: str):
    """Set up comprehensive model monitoring with GenOps governance."""

    monitor = GenOpsModelMonitor(endpoint_name, customer_id)

    # S3 paths for monitoring
    s3_capture_path = f"s3://ml-monitoring-bucket/{customer_id}/data-capture/"
    s3_baseline_path = f"s3://ml-monitoring-bucket/{customer_id}/baseline/"
    s3_output_path = f"s3://ml-monitoring-bucket/{customer_id}/monitoring-output/"

    # Set up data capture
    data_capture_config = monitor.setup_data_capture(s3_capture_path)
    print(f"Data capture configured for endpoint: {endpoint_name}")

    # Create monitoring schedule
    schedule_name = monitor.create_monitoring_schedule(s3_baseline_path, s3_output_path)
    print(f"Monitoring schedule created: {schedule_name}")

    # Process initial report (would be automated)
    report = monitor.process_monitoring_report(s3_output_path)
    print(f"Initial monitoring report generated with {report['bedrock_insights']['risk_level']} risk level")

    return monitor, report

if __name__ == "__main__":
    # Example usage
    monitor, initial_report = setup_genops_model_monitoring(
        endpoint_name="genops-sentiment-endpoint",
        customer_id="enterprise-ml-client"
    )

    print("\\nMonitoring Summary:")
    print(f"Risk Level: {initial_report['bedrock_insights']['risk_level']}")
    print(f"Next Actions: {len(initial_report['next_actions'])} items")
    print(f"Analysis Cost: ${initial_report['bedrock_insights']['analysis_cost']:.4f}")
'''

    print("🔍 Model Monitoring Features:")
    print("   ✅ SageMaker Model Monitor with Bedrock analysis")
    print("   ✅ Intelligent drift detection with AI insights")
    print("   ✅ Automated risk assessment and recommendations")
    print("   ✅ Cost attribution for monitoring operations")
    print("   ✅ Comprehensive governance reporting")
    print("   ✅ Actionable alerts based on AI analysis")
    print()

    return monitoring_code


def main():
    """Main demonstration function."""

    print("🧠 Welcome to GenOps Bedrock SageMaker Integration!")
    print()
    print("This example demonstrates ML pipeline patterns combining")
    print("SageMaker and Bedrock with comprehensive MLOps governance.")
    print()

    demos = [
        ("Training Integration", create_sagemaker_training_integration),
        ("Inference Integration", create_sagemaker_inference_integration),
        ("Pipeline Integration", create_sagemaker_pipeline_integration),
        ("Model Monitoring", create_model_monitoring_integration),
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
    print("🎉 SageMaker Integration Demo Summary")
    print("=" * 42)

    print("🏆 MLOps Features Demonstrated:")
    print("   🧠 Training data augmentation with Bedrock foundation models")
    print("   🔮 Hybrid inference combining custom + foundation models")
    print("   🔄 End-to-end ML pipelines with governance integration")
    print("   📊 Intelligent model monitoring with AI-powered analysis")
    print("   💰 Unified cost tracking across training and inference")
    print("   🛡️ SOC2 compliance for ML workflows")
    print()

    print("🚀 MLOps Best Practices Demonstrated:")
    print("   ✅ Comprehensive experiment tracking with governance")
    print("   ✅ Cost attribution for training and inference")
    print("   ✅ Model versioning with GenOps workflow IDs")
    print("   ✅ Automated model evaluation with AI insights")
    print("   ✅ Production monitoring with drift detection")
    print("   ✅ Budget controls and cost optimization")
    print()

    print("🎯 Implementation Guide:")
    print("   1. Set up SageMaker execution roles with Bedrock permissions")
    print("   2. Configure S3 buckets for model artifacts and monitoring")
    print("   3. Deploy training scripts with GenOps integration")
    print("   4. Create inference endpoints with hybrid prediction")
    print("   5. Set up monitoring schedules with Bedrock analysis")
    print("   6. Configure CloudWatch dashboards for ML + AI metrics")
    print()

    print("💡 Next Steps:")
    print("   → Implement A/B testing frameworks with GenOps governance")
    print("   → Set up MLOps CI/CD pipelines with cost validation")
    print("   → Create model registry with governance metadata")
    print("   → Implement automated model retraining triggers")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
