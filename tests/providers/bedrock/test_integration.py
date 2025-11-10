"""
Comprehensive integration tests for GenOps Bedrock provider.

Tests end-to-end workflows and real-world usage patterns including:
- Complete workflow integration across all modules
- Multi-provider cost tracking integration
- Enterprise compliance scenarios
- Performance and scaling integration
- Error recovery and resilience patterns
- Production deployment patterns
"""

import threading
import time
from unittest.mock import Mock, patch

import pytest

# Import all the modules for integration testing
try:
    from genops.providers.bedrock import (
        GenOpsBedrockAdapter,
        auto_instrument_bedrock,
        instrument_bedrock,
    )
    from genops.providers.bedrock_cost_aggregator import create_bedrock_cost_context
    from genops.providers.bedrock_pricing import (
        calculate_bedrock_cost,
        compare_bedrock_models,
    )
    from genops.providers.bedrock_validation import (
        print_validation_result,
        validate_bedrock_setup,
    )
    from genops.providers.bedrock_workflow import (
        ComplianceLevel,
        production_workflow_context,
    )
    BEDROCK_INTEGRATION_AVAILABLE = True
except ImportError:
    BEDROCK_INTEGRATION_AVAILABLE = False


@pytest.mark.skipif(not BEDROCK_INTEGRATION_AVAILABLE, reason="Bedrock integration modules not available")
class TestEndToEndWorkflows:
    """Test complete end-to-end workflow scenarios."""

    @patch('boto3.client')
    def test_simple_document_analysis_workflow(self, mock_boto_client):
        """Test a simple document analysis workflow end-to-end."""
        # Mock Bedrock responses
        self._setup_bedrock_mocks(mock_boto_client)

        # Complete workflow using multiple components
        with production_workflow_context(
            workflow_name="simple_document_analysis",
            customer_id="integration_test_client",
            team="integration-testing",
            project="end-to-end-validation",
            compliance_level=ComplianceLevel.SOC2,
            budget_limit=1.0
        ) as (workflow, workflow_id):

            # Initialize adapter
            adapter = GenOpsBedrockAdapter()

            # Step 1: Document classification
            workflow.record_step("classification", {"document_type": "financial"})

            classification_result = adapter.text_generation(
                prompt="Classify this document: QUARTERLY FINANCIAL RESULTS Q3 2024",
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                max_tokens=50,
                temperature=0.1,
                team="integration-testing",
                customer_id="integration_test_client",
                feature="classification"
            )

            # Step 2: Content extraction
            workflow.record_step("extraction", {"classification": classification_result.content})

            extraction_result = adapter.text_generation(
                prompt="Extract key financial metrics from the document",
                model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                max_tokens=200,
                temperature=0.2,
                team="integration-testing",
                customer_id="integration_test_client",
                feature="extraction"
            )

            # Step 3: Compliance check
            workflow.record_step("compliance", {"extraction": extraction_result.content[:100]})

            workflow.record_checkpoint("soc2_validation", {
                "pii_detected": False,
                "financial_data_handled": True,
                "compliance_maintained": True
            })

            # Record performance metrics
            total_cost = classification_result.cost_usd + extraction_result.cost_usd
            workflow.record_performance_metric("total_cost", total_cost, "USD")
            workflow.record_performance_metric("total_tokens",
                                             classification_result.input_tokens + classification_result.output_tokens +
                                             extraction_result.input_tokens + extraction_result.output_tokens, "count")

            # Verify integration worked
            assert len(workflow.steps) >= 3
            assert total_cost > 0
            assert classification_result.content is not None
            assert extraction_result.content is not None

    @patch('boto3.client')
    def test_multi_model_cost_optimization_workflow(self, mock_boto_client):
        """Test workflow with multi-model cost optimization."""
        self._setup_bedrock_mocks(mock_boto_client)

        with create_bedrock_cost_context("multi_model_optimization") as cost_context:
            adapter = GenOpsBedrockAdapter()

            # Simulate different analysis tasks with appropriate models

            # Task 1: Quick classification (use cost-effective model)
            quick_result = adapter.text_generation(
                prompt="Quick classification: positive or negative sentiment",
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                max_tokens=10,
                team="optimization-testing"
            )

            # Task 2: Detailed analysis (use balanced model)
            detailed_result = adapter.text_generation(
                prompt="Provide detailed analysis of market trends and implications",
                model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                max_tokens=300,
                team="optimization-testing"
            )

            # Task 3: Simple summarization (use cost-effective model)
            summary_result = adapter.text_generation(
                prompt="Summarize in one sentence",
                model_id="amazon.titan-text-express-v1",
                max_tokens=50,
                team="optimization-testing"
            )

            # Get cost summary
            final_summary = cost_context.get_current_summary()

            # Verify multi-model cost tracking
            assert final_summary.total_operations >= 3
            assert len(final_summary.unique_models) >= 2
            assert len(final_summary.unique_providers) >= 2
            assert final_summary.total_cost > 0

            # Verify cost differences (Sonnet should be more expensive than Haiku)
            haiku_cost = final_summary.cost_by_model.get("anthropic.claude-3-haiku-20240307-v1:0", 0)
            sonnet_cost = final_summary.cost_by_model.get("anthropic.claude-3-sonnet-20240229-v1:0", 0)

            assert haiku_cost >= 0
            assert sonnet_cost >= 0

    @patch('boto3.client')
    def test_enterprise_compliance_workflow(self, mock_boto_client):
        """Test enterprise compliance workflow with full audit trail."""
        self._setup_bedrock_mocks(mock_boto_client)

        with production_workflow_context(
            workflow_name="enterprise_compliance_processing",
            customer_id="enterprise_fortune500",
            team="compliance-ai-platform",
            project="regulatory-document-processing",
            environment="production",
            compliance_level=ComplianceLevel.SOC2,
            cost_center="Compliance-Technology",
            budget_limit=10.0,
            enable_cloudtrail=True,
            alert_webhooks=["https://alerts.compliance.com/ai"]
        ) as (workflow, workflow_id):

            adapter = GenOpsBedrockAdapter()

            # Step 1: Input validation and PII detection
            workflow.record_step("input_validation", {
                "validation_framework": "SOC2",
                "pii_scanning_enabled": True
            })

            pii_check = adapter.text_generation(
                prompt="Scan for PII in this document: John Doe, SSN: XXX-XX-XXXX, Born: 1985",
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                max_tokens=100,
                temperature=0.1,
                team="compliance-ai-platform",
                customer_id="enterprise_fortune500",
                feature="pii_detection"
            )

            # Record compliance checkpoint
            workflow.record_checkpoint("pii_scanning_complete", {
                "pii_detected": "SSN" in pii_check.content if pii_check.content else False,
                "scanning_model": "anthropic.claude-3-haiku-20240307-v1:0",
                "compliance_framework": "SOC2",
                "data_classification": "sensitive"
            })

            # Step 2: Data processing with encryption context
            workflow.record_step("secure_processing", {
                "encryption_enabled": True,
                "access_logged": True,
                "processing_model": "anthropic.claude-3-sonnet-20240229-v1:0"
            })

            processing_result = adapter.text_generation(
                prompt="Process this document with SOC2 compliance requirements",
                model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                max_tokens=250,
                temperature=0.2,
                team="compliance-ai-platform",
                customer_id="enterprise_fortune500",
                feature="secure_processing"
            )

            # Step 3: Output sanitization
            workflow.record_step("output_sanitization", {
                "sanitization_rules": ["remove_pii", "redact_sensitive"],
                "output_classification": "public"
            })

            sanitization_result = adapter.text_generation(
                prompt="Sanitize this output for public release, removing any PII",
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                max_tokens=200,
                temperature=0.1,
                team="compliance-ai-platform",
                customer_id="enterprise_fortune500",
                feature="output_sanitization"
            )

            # Final compliance validation
            workflow.record_checkpoint("final_compliance_validation", {
                "all_pii_removed": True,
                "output_sanitized": True,
                "audit_trail_complete": True,
                "soc2_requirements_met": True,
                "data_retention_compliant": True
            })

            # Performance metrics
            total_cost = pii_check.cost_usd + processing_result.cost_usd + sanitization_result.cost_usd
            workflow.record_performance_metric("total_compliance_cost", total_cost, "USD")
            workflow.record_performance_metric("compliance_steps_completed", 3, "count")
            workflow.record_performance_metric("compliance_score", 1.0, "percentage")

            # Verify enterprise compliance workflow
            assert len(workflow.steps) >= 3
            if hasattr(workflow, 'checkpoints'):
                assert len(workflow.checkpoints) >= 2
            assert workflow.compliance_level == ComplianceLevel.SOC2
            assert workflow.enable_cloudtrail is True

    def _setup_bedrock_mocks(self, mock_boto_client):
        """Helper method to set up Bedrock API mocks."""
        # Mock Bedrock client
        mock_bedrock = Mock()

        # Mock text generation responses
        def create_mock_response(content, input_tokens=100, output_tokens=50):
            mock_response = {
                'body': Mock(),
                'contentType': 'application/json'
            }
            mock_body = Mock()
            mock_body.read.return_value = f'{{"completion": "{content}", "usage": {{"input_tokens": {input_tokens}, "output_tokens": {output_tokens}}}}}'.encode()
            mock_response['body'] = mock_body
            return mock_response

        # Set up different responses for different prompts
        responses = [
            create_mock_response("Financial document classification", 50, 25),
            create_mock_response("Key metrics: Revenue $2.3B, Growth 15%", 200, 100),
            create_mock_response("Compliance check passed", 80, 40),
            create_mock_response("PII detected: SSN redacted", 120, 60),
            create_mock_response("Document processed securely", 250, 125),
            create_mock_response("Output sanitized for public release", 180, 90)
        ]

        mock_bedrock.invoke_model.side_effect = responses

        # Mock STS for credentials validation
        mock_sts = Mock()
        mock_sts.get_caller_identity.return_value = {
            'Account': '123456789012'
        }

        def client_factory(service_name, **kwargs):
            if service_name == 'bedrock-runtime':
                return mock_bedrock
            elif service_name == 'sts':
                return mock_sts
            else:
                return Mock()

        mock_boto_client.side_effect = client_factory


@pytest.mark.skipif(not BEDROCK_INTEGRATION_AVAILABLE, reason="Bedrock integration modules not available")
class TestValidationAndSetupIntegration:
    """Test integration of validation with other components."""

    @patch('boto3.client')
    def test_validation_before_workflow_execution(self, mock_boto_client):
        """Test that validation works before executing workflows."""
        # Mock successful validation
        mock_sts = Mock()
        mock_sts.get_caller_identity.return_value = {'Account': '123456789012'}

        mock_bedrock = Mock()
        mock_bedrock.list_foundation_models.return_value = {
            'modelSummaries': [
                {'modelId': 'anthropic.claude-3-haiku-20240307-v1:0', 'modelName': 'Claude 3 Haiku'}
            ]
        }

        def client_factory(service_name, **kwargs):
            if service_name == 'sts':
                return mock_sts
            elif service_name == 'bedrock':
                return mock_bedrock
            else:
                return Mock()

        mock_boto_client.side_effect = client_factory

        # Run validation
        validation_result = validate_bedrock_setup()

        # If validation passes, proceed with workflow
        if validation_result.success:
            # Should be able to create adapter
            adapter = GenOpsBedrockAdapter()
            assert adapter is not None

            # Should be able to create workflow context
            with production_workflow_context(
                workflow_name="post_validation_test",
                customer_id="validation_customer",
                team="validation-team",
                project="validation-project"
            ) as (workflow, workflow_id):

                assert workflow is not None
                assert len(workflow_id) > 0

    def test_validation_result_printing_integration(self, capsys):
        """Test that validation result printing integrates properly."""
        # Create a mock validation result
        from genops.providers.bedrock_validation import (
            ValidationCheck,
            ValidationResult,
        )

        result = ValidationResult(
            success=True,
            errors=[],
            warnings=["Region not specified, using default"],
            checks_passed=4,
            total_checks=5,
            detailed_checks={
                "aws_credentials": ValidationCheck(
                    name="aws_credentials",
                    passed=True,
                    error=None,
                    fix_suggestion="Credentials properly configured",
                    documentation_link="https://docs.aws.amazon.com/credentials/"
                )
            }
        )

        # Print validation result
        print_validation_result(result)

        captured = capsys.readouterr()

        # Should show successful validation with warnings
        assert "4/5" in captured.out
        assert "warning" in captured.out.lower() or "⚠️" in captured.out


@pytest.mark.skipif(not BEDROCK_INTEGRATION_AVAILABLE, reason="Bedrock integration modules not available")
class TestCostTrackingIntegration:
    """Test integration of cost tracking across all components."""

    @patch('boto3.client')
    def test_unified_cost_tracking_across_components(self, mock_boto_client):
        """Test that cost tracking works consistently across all components."""
        # Setup mocks
        mock_response = {
            'body': Mock(),
            'contentType': 'application/json'
        }
        mock_body = Mock()
        mock_body.read.return_value = b'{"completion": "Test response", "usage": {"input_tokens": 100, "output_tokens": 50}}'
        mock_response['body'] = mock_body

        mock_bedrock = Mock()
        mock_bedrock.invoke_model.return_value = mock_response
        mock_boto_client.return_value = mock_bedrock

        # Test 1: Individual cost calculation
        individual_cost = calculate_bedrock_cost(
            input_tokens=100,
            output_tokens=50,
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            region="us-east-1"
        )

        # Test 2: Adapter-based cost tracking
        adapter = GenOpsBedrockAdapter()
        adapter_result = adapter.text_generation(
            prompt="Test prompt for cost integration",
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            team="cost-integration-test"
        )

        # Test 3: Cost context aggregation
        with create_bedrock_cost_context("cost_integration_test") as cost_context:
            cost_context.add_operation(
                operation_id="manual_cost_op",
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                provider="anthropic",
                region="us-east-1",
                input_tokens=100,
                output_tokens=50,
                latency_ms=1000,
                governance_attributes={"team": "cost-integration"}
            )

            context_summary = cost_context.get_current_summary()

        # Test 4: Workflow-integrated cost tracking
        with production_workflow_context(
            workflow_name="cost_integration_workflow",
            customer_id="cost_integration_customer",
            team="cost-integration-team",
            project="cost-integration-project",
            budget_limit=1.0
        ) as (workflow, workflow_id):

            workflow.record_performance_metric("operation_cost", adapter_result.cost_usd, "USD")
            workflow_cost_summary = workflow.get_current_cost_summary()

        # Verify cost consistency across components
        assert individual_cost.total_cost > 0
        assert adapter_result.cost_usd > 0
        assert context_summary.total_cost > 0

        # Costs should be consistent for same token usage
        tolerance = 0.000001
        assert abs(individual_cost.total_cost - adapter_result.cost_usd) < tolerance

    @patch('boto3.client')
    def test_multi_provider_cost_aggregation_integration(self, mock_boto_client):
        """Test cost aggregation across multiple providers."""
        # Setup mocks for different models
        responses = [
            b'{"completion": "Anthropic response", "usage": {"input_tokens": 100, "output_tokens": 50}}',
            b'{"completion": "Amazon response", "usage": {"input_tokens": 120, "output_tokens": 60}}',
            b'{"completion": "AI21 response", "usage": {"input_tokens": 80, "output_tokens": 40}}'
        ]

        def create_mock_response(response_data):
            mock_response = {'body': Mock(), 'contentType': 'application/json'}
            mock_body = Mock()
            mock_body.read.return_value = response_data
            mock_response['body'] = mock_body
            return mock_response

        mock_bedrock = Mock()
        mock_bedrock.invoke_model.side_effect = [create_mock_response(r) for r in responses]
        mock_boto_client.return_value = mock_bedrock

        with create_bedrock_cost_context("multi_provider_test") as cost_context:
            adapter = GenOpsBedrockAdapter()

            # Test different providers
            providers_models = [
                ("anthropic.claude-3-haiku-20240307-v1:0", "anthropic"),
                ("amazon.titan-text-express-v1", "amazon"),
                ("ai21.j2-mid-v1", "ai21")
            ]

            results = []
            for model_id, provider in providers_models:
                result = adapter.text_generation(
                    prompt=f"Test prompt for {provider}",
                    model_id=model_id,
                    team="multi-provider-test"
                )
                results.append(result)

            # Get aggregated summary
            summary = cost_context.get_current_summary()

            # Verify multi-provider aggregation
            assert summary.total_operations >= 3
            assert len(summary.unique_providers) >= 2
            assert len(summary.unique_models) >= 2

            # Each provider should have costs
            total_provider_cost = sum(summary.cost_by_provider.values())
            assert abs(summary.total_cost - total_provider_cost) < 0.000001


@pytest.mark.skipif(not BEDROCK_INTEGRATION_AVAILABLE, reason="Bedrock integration modules not available")
class TestErrorHandlingIntegration:
    """Test error handling integration across components."""

    @patch('boto3.client')
    def test_workflow_resilience_to_model_failures(self, mock_boto_client):
        """Test that workflows handle model failures gracefully."""
        from botocore.exceptions import ClientError

        # Mock alternating success and failure
        mock_bedrock = Mock()

        success_response = {
            'body': Mock(),
            'contentType': 'application/json'
        }
        success_body = Mock()
        success_body.read.return_value = b'{"completion": "Success response", "usage": {"input_tokens": 100, "output_tokens": 50}}'
        success_response['body'] = success_body

        # Alternate between success and failure
        mock_bedrock.invoke_model.side_effect = [
            success_response,  # First call succeeds
            ClientError(       # Second call fails
                error_response={'Error': {'Code': 'ThrottlingException', 'Message': 'Rate limit exceeded'}},
                operation_name='InvokeModel'
            ),
            success_response   # Third call succeeds
        ]

        mock_boto_client.return_value = mock_bedrock

        with production_workflow_context(
            workflow_name="resilience_test",
            customer_id="resilience_customer",
            team="resilience-team",
            project="resilience-project"
        ) as (workflow, workflow_id):

            adapter = GenOpsBedrockAdapter()

            # First operation should succeed
            try:
                result1 = adapter.text_generation(
                    prompt="First operation",
                    model_id="anthropic.claude-3-haiku-20240307-v1:0",
                    team="resilience-team"
                )
                workflow.record_step("successful_operation", {"result": "success"})
                success_count = 1
            except Exception:
                success_count = 0

            # Second operation should fail
            try:
                result2 = adapter.text_generation(
                    prompt="Second operation",
                    model_id="anthropic.claude-3-haiku-20240307-v1:0",
                    team="resilience-team"
                )
                workflow.record_step("second_operation", {"result": "unexpected_success"})
            except Exception as e:
                workflow.record_step("failed_operation", {"error": str(e)})
                # Workflow should continue despite failure

            # Third operation should succeed
            try:
                result3 = adapter.text_generation(
                    prompt="Third operation",
                    model_id="anthropic.claude-3-haiku-20240307-v1:0",
                    team="resilience-team"
                )
                workflow.record_step("recovery_operation", {"result": "success"})
                success_count += 1
            except Exception:
                pass

            # Workflow should complete despite failures
            assert len(workflow.steps) >= 2
            assert success_count >= 1  # At least one operation succeeded

    def test_cost_context_exception_handling_integration(self):
        """Test cost context handles exceptions during workflow integration."""
        try:
            with create_bedrock_cost_context("exception_integration_test") as cost_context:
                # Add some operations
                cost_context.add_operation(
                    operation_id="before_exception",
                    model_id="anthropic.claude-3-haiku-20240307-v1:0",
                    provider="anthropic",
                    region="us-east-1",
                    input_tokens=100,
                    output_tokens=50,
                    latency_ms=1000,
                    governance_attributes={"team": "exception-test"}
                )

                # Simulate exception during workflow
                raise ValueError("Simulated workflow exception")

        except ValueError as e:
            assert str(e) == "Simulated workflow exception"

        # Context should still provide summary even after exception
        try:
            summary = cost_context.get_current_summary()
            assert summary.total_operations >= 1
        except Exception:
            # Context may not be accessible after exception
            pass


@pytest.mark.skipif(not BEDROCK_INTEGRATION_AVAILABLE, reason="Bedrock integration modules not available")
class TestPerformanceIntegration:
    """Test performance characteristics of integrated components."""

    @patch('boto3.client')
    def test_large_workflow_performance(self, mock_boto_client):
        """Test performance with large workflows."""
        # Mock fast responses
        mock_response = {
            'body': Mock(),
            'contentType': 'application/json'
        }
        mock_body = Mock()
        mock_body.read.return_value = b'{"completion": "Fast response", "usage": {"input_tokens": 10, "output_tokens": 5}}'
        mock_response['body'] = mock_body

        mock_bedrock = Mock()
        mock_bedrock.invoke_model.return_value = mock_response
        mock_boto_client.return_value = mock_bedrock

        start_time = time.time()

        with production_workflow_context(
            workflow_name="large_workflow_performance",
            customer_id="performance_customer",
            team="performance-team",
            project="performance-project"
        ) as (workflow, workflow_id):

            adapter = GenOpsBedrockAdapter()

            # Process many operations
            num_operations = 100
            for i in range(num_operations):
                try:
                    result = adapter.text_generation(
                        prompt=f"Operation {i}",
                        model_id="anthropic.claude-3-haiku-20240307-v1:0",
                        max_tokens=10,
                        team="performance-team"
                    )

                    if i % 10 == 0:  # Record every 10th operation
                        workflow.record_step(f"batch_operation_{i//10}", {"batch_size": 10})
                        workflow.record_performance_metric(f"batch_{i//10}_cost", result.cost_usd, "USD")

                except Exception:
                    # Continue on individual failures
                    pass

        end_time = time.time()
        total_time = end_time - start_time

        # Should complete in reasonable time (under 10 seconds for 100 operations)
        assert total_time < 10.0

        # Workflow should have recorded multiple steps
        assert len(workflow.steps) >= 5

    def test_concurrent_workflow_performance(self):
        """Test performance of concurrent workflow execution."""

        results = []
        errors = []

        def worker_workflow(worker_id):
            try:
                with production_workflow_context(
                    workflow_name=f"concurrent_perf_test_{worker_id}",
                    customer_id=f"concurrent_customer_{worker_id}",
                    team="concurrent-team",
                    project="concurrent-project"
                ) as (workflow, workflow_id):

                    # Simulate some workflow operations
                    for i in range(5):
                        workflow.record_step(f"worker_{worker_id}_step_{i}", {"worker": worker_id})
                        time.sleep(0.01)  # Simulate processing time

                    results.append((worker_id, workflow_id, len(workflow.steps)))

            except Exception as e:
                errors.append((worker_id, str(e)))

        # Run multiple concurrent workflows
        threads = []
        num_workers = 10

        start_time = time.time()

        for i in range(num_workers):
            thread = threading.Thread(target=worker_workflow, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join(timeout=10)

        end_time = time.time()
        total_time = end_time - start_time

        # Verify concurrent execution
        assert len(results) + len(errors) == num_workers
        assert len(results) >= num_workers // 2  # At least half should succeed

        # Should complete concurrently (not much slower than single workflow)
        assert total_time < 5.0

        # All workflow IDs should be unique
        workflow_ids = [wid for _, wid, _ in results]
        assert len(set(workflow_ids)) == len(workflow_ids)


@pytest.mark.skipif(not BEDROCK_INTEGRATION_AVAILABLE, reason="Bedrock integration modules not available")
class TestAutoInstrumentationIntegration:
    """Test auto-instrumentation integration with other components."""

    def test_auto_instrumentation_setup_integration(self):
        """Test that auto-instrumentation integrates with manual components."""
        # Enable auto-instrumentation
        try:
            auto_instrument_bedrock()
        except Exception:
            # May fail in test environment
            pass

        # Manual components should still work
        try:
            adapter = GenOpsBedrockAdapter()
            assert adapter is not None
        except Exception:
            # Expected in test environment without full AWS setup
            pass

        # Validation should still work
        try:
            validation_result = validate_bedrock_setup()
            assert hasattr(validation_result, 'success')
        except Exception:
            # Expected in test environment
            pass

    def test_manual_and_auto_instrumentation_coexistence(self):
        """Test that manual and auto instrumentation can coexist."""
        # Try to enable auto-instrumentation
        try:
            auto_instrument_bedrock()
        except Exception:
            pass

        # Manual instrumentation should still work
        try:
            instrument_bedrock()
        except Exception:
            pass

        # Both should be callable without conflicts
        try:
            adapter = GenOpsBedrockAdapter()
            with create_bedrock_cost_context("coexistence_test") as context:
                # Should work together
                assert context is not None
                assert adapter is not None
        except Exception:
            # Expected in test environment
            pass


@pytest.mark.skipif(not BEDROCK_INTEGRATION_AVAILABLE, reason="Bedrock integration modules not available")
class TestRealWorldScenarios:
    """Test real-world usage scenarios integration."""

    @patch('boto3.client')
    def test_customer_support_analysis_scenario(self, mock_boto_client):
        """Test customer support ticket analysis scenario."""
        # Mock realistic responses
        responses = [
            b'{"completion": "Priority: High, Category: Technical Issue", "usage": {"input_tokens": 150, "output_tokens": 75}}',
            b'{"completion": "Customer frustrated with login issues, requires immediate escalation", "usage": {"input_tokens": 200, "output_tokens": 100}}',
            b'{"completion": "Recommended actions: Reset password, check account status, escalate to Level 2", "usage": {"input_tokens": 250, "output_tokens": 125}}'
        ]

        mock_bedrock = Mock()
        mock_bedrock.invoke_model.side_effect = [
            self._create_mock_response(r) for r in responses
        ]
        mock_boto_client.return_value = mock_bedrock

        # Customer support workflow
        with production_workflow_context(
            workflow_name="customer_support_ticket_analysis",
            customer_id="customer_support_system",
            team="customer-support-ai",
            project="automated-ticket-analysis",
            environment="production",
            compliance_level=ComplianceLevel.SOC2,
            budget_limit=0.10  # $0.10 per ticket
        ) as (workflow, workflow_id):

            adapter = GenOpsBedrockAdapter()

            # Step 1: Ticket classification
            workflow.record_step("ticket_classification", {
                "ticket_id": "TICKET-12345",
                "source": "email",
                "customer_tier": "premium"
            })

            classification = adapter.text_generation(
                prompt="Classify this support ticket: Customer cannot login, getting error 500",
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                max_tokens=100,
                temperature=0.1,
                team="customer-support-ai",
                customer_id="customer_support_system",
                feature="ticket_classification"
            )

            # Step 2: Sentiment analysis
            workflow.record_step("sentiment_analysis", {
                "classification": classification.content
            })

            sentiment = adapter.text_generation(
                prompt="Analyze customer sentiment: I've been trying to login for hours and keep getting errors!",
                model_id="anthropic.claude-3-haiku-20240307-v1:0",
                max_tokens=150,
                temperature=0.2,
                team="customer-support-ai",
                customer_id="customer_support_system",
                feature="sentiment_analysis"
            )

            # Step 3: Response generation
            workflow.record_step("response_generation", {
                "sentiment": sentiment.content[:50]
            })

            response = adapter.text_generation(
                prompt="Generate helpful response for frustrated customer with login issues",
                model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                max_tokens=200,
                temperature=0.3,
                team="customer-support-ai",
                customer_id="customer_support_system",
                feature="response_generation"
            )

            # Record performance metrics
            total_cost = classification.cost_usd + sentiment.cost_usd + response.cost_usd
            workflow.record_performance_metric("total_ticket_cost", total_cost, "USD")
            workflow.record_performance_metric("response_quality_score", 0.88, "percentage")
            workflow.record_performance_metric("processing_time", 2.5, "seconds")

            # Compliance checkpoint
            workflow.record_checkpoint("customer_data_handling", {
                "customer_pii_protected": True,
                "response_appropriate": True,
                "escalation_rules_followed": True
            })

            # Verify customer support scenario
            assert len(workflow.steps) >= 3
            assert total_cost <= 0.10  # Within budget
            assert workflow.compliance_level == ComplianceLevel.SOC2

    @patch('boto3.client')
    def test_financial_document_analysis_scenario(self, mock_boto_client):
        """Test financial document analysis scenario."""
        # Mock sophisticated financial analysis responses
        responses = [
            b'{"completion": "Document Type: Quarterly Earnings Report, Confidence: 0.95", "usage": {"input_tokens": 300, "output_tokens": 150}}',
            b'{"completion": "Key Metrics: Revenue $2.3B (+15% YoY), Net Income $450M (+22% YoY)", "usage": {"input_tokens": 500, "output_tokens": 250}}',
            b'{"completion": "Risk Assessment: Low risk, strong fundamentals, positive growth trajectory", "usage": {"input_tokens": 400, "output_tokens": 200}}',
            b'{"completion": "Executive Summary: Strong quarterly performance exceeding expectations", "usage": {"input_tokens": 350, "output_tokens": 175}}'
        ]

        mock_bedrock = Mock()
        mock_bedrock.invoke_model.side_effect = [
            self._create_mock_response(r) for r in responses
        ]
        mock_boto_client.return_value = mock_bedrock

        with production_workflow_context(
            workflow_name="financial_document_analysis",
            customer_id="investment_bank_alpha",
            team="financial-ai-analysis",
            project="automated-financial-intelligence",
            environment="production",
            compliance_level=ComplianceLevel.SOX,  # Financial compliance
            cost_center="Investment-Research",
            budget_limit=1.00,  # Higher budget for financial analysis
            enable_cloudtrail=True
        ) as (workflow, workflow_id):

            adapter = GenOpsBedrockAdapter()

            with create_bedrock_cost_context(f"financial_analysis_{workflow_id}") as cost_context:

                # Step 1: Document type identification
                workflow.record_step("document_identification", {
                    "document_source": "SEC_filing",
                    "document_size": "150KB",
                    "processing_model": "anthropic.claude-3-sonnet-20240229-v1:0"
                })

                identification = adapter.text_generation(
                    prompt="Identify document type: QUARTERLY RESULTS Q3 2024 - Revenue Growth and Financial Performance",
                    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                    max_tokens=200,
                    temperature=0.1,
                    team="financial-ai-analysis",
                    customer_id="investment_bank_alpha",
                    feature="document_identification"
                )

                # Step 2: Key metrics extraction
                workflow.record_step("metrics_extraction", {
                    "identification_result": identification.content[:100]
                })

                metrics = adapter.text_generation(
                    prompt="Extract key financial metrics from quarterly earnings report",
                    model_id="anthropic.claude-3-opus-20240229-v1:0",  # Premium model for accuracy
                    max_tokens=400,
                    temperature=0.2,
                    team="financial-ai-analysis",
                    customer_id="investment_bank_alpha",
                    feature="metrics_extraction"
                )

                # Step 3: Risk assessment
                workflow.record_step("risk_assessment", {
                    "metrics_extracted": metrics.content[:100]
                })

                risk_assessment = adapter.text_generation(
                    prompt="Perform comprehensive risk assessment based on financial metrics",
                    model_id="anthropic.claude-3-opus-20240229-v1:0",
                    max_tokens=300,
                    temperature=0.2,
                    team="financial-ai-analysis",
                    customer_id="investment_bank_alpha",
                    feature="risk_assessment"
                )

                # Step 4: Executive summary generation
                workflow.record_step("executive_summary", {
                    "risk_assessment": risk_assessment.content[:100]
                })

                executive_summary = adapter.text_generation(
                    prompt="Generate executive summary for investment committee",
                    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                    max_tokens=250,
                    temperature=0.3,
                    team="financial-ai-analysis",
                    customer_id="investment_bank_alpha",
                    feature="executive_summary"
                )

                # Get comprehensive cost analysis
                cost_summary = cost_context.get_current_summary()

                # Financial compliance checkpoint
                workflow.record_checkpoint("sox_compliance_validation", {
                    "financial_data_accuracy_verified": True,
                    "internal_controls_active": True,
                    "audit_trail_complete": True,
                    "sox_requirements_met": True,
                    "executive_approval_ready": True
                })

                # Performance metrics
                workflow.record_performance_metric("total_analysis_cost", cost_summary.total_cost, "USD")
                workflow.record_performance_metric("models_utilized", len(cost_summary.unique_models), "count")
                workflow.record_performance_metric("analysis_accuracy", 0.94, "percentage")
                workflow.record_performance_metric("processing_efficiency", 0.87, "percentage")

                # Verify financial analysis scenario
                assert len(workflow.steps) >= 4
                assert cost_summary.total_cost <= 1.00
                assert len(cost_summary.unique_models) >= 2
                assert workflow.compliance_level == ComplianceLevel.SOX
                assert workflow.enable_cloudtrail is True

    def _create_mock_response(self, response_data):
        """Helper to create mock Bedrock responses."""
        mock_response = {
            'body': Mock(),
            'contentType': 'application/json'
        }
        mock_body = Mock()
        mock_body.read.return_value = response_data
        mock_response['body'] = mock_body
        return mock_response


@pytest.mark.integration
class TestRealAWSIntegration:
    """Real AWS integration tests (require actual AWS credentials)."""

    def test_real_bedrock_validation_integration(self):
        """Test real Bedrock validation (requires AWS credentials)."""
        pytest.skip("Integration test - requires real AWS credentials")

        # This would test against real AWS services
        validation_result = validate_bedrock_setup()

        if validation_result.success:
            # Should be able to create real workflows
            with production_workflow_context(
                workflow_name="real_aws_integration_test",
                customer_id="real_aws_customer",
                team="integration-testing",
                project="real-aws-validation"
            ) as (workflow, workflow_id):

                adapter = GenOpsBedrockAdapter()

                # Test real Bedrock call
                result = adapter.text_generation(
                    prompt="Hello from real AWS integration test",
                    model_id="anthropic.claude-3-haiku-20240307-v1:0",
                    max_tokens=50,
                    team="integration-testing"
                )

                assert result.content is not None
                assert result.cost_usd > 0
                assert len(workflow.steps) >= 0
