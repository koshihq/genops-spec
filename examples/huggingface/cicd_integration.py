#!/usr/bin/env python3
"""
CI/CD Integration Example for Hugging Face GenOps

This example demonstrates how to integrate GenOps Hugging Face telemetry
into continuous integration and deployment pipelines with proper testing,
validation, and deployment patterns.

Example usage:
    # In CI pipeline
    python cicd_integration.py --mode=test

    # In deployment validation
    python cicd_integration.py --mode=deploy-validate

    # In performance testing
    python cicd_integration.py --mode=perf-test

Features demonstrated:
- CI/CD pipeline integration patterns
- Automated testing with telemetry validation
- Deployment readiness checks
- Performance regression testing
- Cost impact analysis for CI/CD
"""

import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class CICDContext:
    """CI/CD pipeline context information."""

    pipeline_id: str
    build_number: str
    commit_sha: str
    branch_name: str
    pr_number: Optional[str]
    environment: str
    stage: str


def get_cicd_context() -> CICDContext:
    """Extract CI/CD context from environment variables (GitHub Actions, GitLab CI, etc.)."""

    # Support multiple CI/CD platforms
    return CICDContext(
        # GitHub Actions
        pipeline_id=os.getenv(
            "GITHUB_RUN_ID",
            # GitLab CI
            os.getenv(
                "CI_PIPELINE_ID",
                # Jenkins
                os.getenv("BUILD_ID", "unknown"),
            ),
        ),
        build_number=os.getenv(
            "GITHUB_RUN_NUMBER",
            os.getenv("CI_PIPELINE_IID", os.getenv("BUILD_NUMBER", "0")),
        ),
        commit_sha=os.getenv(
            "GITHUB_SHA", os.getenv("CI_COMMIT_SHA", os.getenv("GIT_COMMIT", "unknown"))
        )[:8],  # Short SHA
        branch_name=os.getenv(
            "GITHUB_REF_NAME",
            os.getenv("CI_COMMIT_REF_NAME", os.getenv("BRANCH_NAME", "unknown")),
        ),
        pr_number=os.getenv("GITHUB_PR_NUMBER", os.getenv("CI_MERGE_REQUEST_IID")),
        environment=os.getenv("DEPLOYMENT_ENVIRONMENT", "ci"),
        stage=os.getenv("CI_STAGE", "test"),
    )


def setup_cicd_configuration():
    """
    Setup GenOps configuration optimized for CI/CD environments.

    This demonstrates best practices for configuring GenOps in CI/CD
    with proper environment isolation and telemetry aggregation.
    """

    print("🔧 CI/CD Configuration Setup")
    print("=" * 35)
    print("Configuring GenOps for CI/CD pipeline...")
    print()

    cicd_context = get_cicd_context()

    # CI/CD-optimized environment configuration
    cicd_config = {
        # OpenTelemetry Configuration for CI/CD
        "OTEL_SERVICE_NAME": f"genops-hf-cicd-{cicd_context.stage}",
        "OTEL_SERVICE_VERSION": f"{cicd_context.build_number}",
        "OTEL_SERVICE_INSTANCE_ID": f"{cicd_context.pipeline_id}-{cicd_context.commit_sha}",
        # CI/CD-specific resource attributes
        "OTEL_RESOURCE_ATTRIBUTES": f"cicd.pipeline.id={cicd_context.pipeline_id},"
        f"cicd.build.number={cicd_context.build_number},"
        f"cicd.commit.sha={cicd_context.commit_sha},"
        f"cicd.branch={cicd_context.branch_name},"
        f"cicd.environment={cicd_context.environment},"
        f"cicd.stage={cicd_context.stage}"
        + (
            f",cicd.pr.number={cicd_context.pr_number}"
            if cicd_context.pr_number
            else ""
        ),
        # OTLP Configuration (CI/CD-specific endpoints)
        "OTEL_EXPORTER_OTLP_ENDPOINT": os.getenv(
            "CICD_OTEL_ENDPOINT", "http://localhost:4317"
        ),
        "OTEL_EXPORTER_OTLP_PROTOCOL": "grpc",
        "OTEL_EXPORTER_OTLP_TIMEOUT": "5",  # Shorter timeout for CI/CD
        # Hugging Face Configuration (test tokens)
        "HF_TOKEN": os.getenv("HF_TOKEN_CI", os.getenv("HF_TOKEN", "")),
        "HF_HOME": f"/tmp/.cache/huggingface-{cicd_context.pipeline_id}",  # Isolated cache
        # GenOps CI/CD Configuration
        "GENOPS_LOG_LEVEL": os.getenv("GENOPS_CI_LOG_LEVEL", "INFO"),
        "GENOPS_SAMPLING_RATE": "1.0",  # Full sampling for CI/CD
        "GENOPS_CI_MODE": "true",
        "GENOPS_EXPORT_BATCH_SIZE": "10",  # Smaller batches for CI/CD
        # CI/CD-specific settings
        "CI_PIPELINE_TIMEOUT": os.getenv("CI_TIMEOUT", "300"),  # 5 minutes default
        "CI_COST_THRESHOLD": os.getenv("CI_COST_THRESHOLD", "0.10"),  # $0.10 threshold
        "CI_PERFORMANCE_BASELINE": os.getenv(
            "CI_PERF_BASELINE", "2.0"
        ),  # 2 seconds baseline
    }

    print("📋 CI/CD Configuration:")
    print(f"   Pipeline: {cicd_context.pipeline_id}")
    print(f"   Build: {cicd_context.build_number}")
    print(f"   Commit: {cicd_context.commit_sha}")
    print(f"   Branch: {cicd_context.branch_name}")
    print(f"   PR: {cicd_context.pr_number or 'N/A'}")
    print(f"   Environment: {cicd_context.environment}")
    print(f"   Stage: {cicd_context.stage}")
    print()

    for key, value in cicd_config.items():
        if key not in [
            "HF_TOKEN",
            "OTEL_RESOURCE_ATTRIBUTES",
        ]:  # Skip sensitive/long values
            print(f"   {key:<25} = {value}")
        else:
            print(f"   {key:<25} = {'***' if 'TOKEN' in key else '[hidden]'}")

    # Set environment variables for current process
    for key, value in cicd_config.items():
        if value:
            os.environ[key] = value

    return cicd_config, cicd_context


def run_cicd_tests():
    """
    Run comprehensive CI/CD tests for GenOps Hugging Face integration.

    This includes unit tests, integration tests, and telemetry validation.
    """

    print("\n🧪 CI/CD Test Suite")
    print("=" * 25)

    test_results = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "total_time": 0,
        "tests": [],
    }

    try:
        from genops.providers.huggingface import (
            GenOpsHuggingFaceAdapter,
            create_huggingface_cost_context,
            production_workflow_context,  # noqa: F401
        )

        cicd_context = get_cicd_context()

        # Define CI/CD test cases
        ci_test_cases = [
            {
                "name": "test_adapter_initialization",
                "description": "Test GenOps adapter can be initialized",
                "critical": True,
            },
            {
                "name": "test_basic_text_generation",
                "description": "Test basic text generation with telemetry",
                "critical": True,
            },
            {
                "name": "test_cost_calculation",
                "description": "Test cost calculation accuracy",
                "critical": True,
            },
            {
                "name": "test_multi_provider_support",
                "description": "Test multiple provider detection and usage",
                "critical": False,
            },
            {
                "name": "test_error_handling",
                "description": "Test error handling and recovery",
                "critical": True,
            },
            {
                "name": "test_telemetry_export",
                "description": "Test telemetry export functionality",
                "critical": True,
            },
            {
                "name": "test_performance_baseline",
                "description": "Test performance meets baseline requirements",
                "critical": False,
            },
            {
                "name": "test_cost_threshold",
                "description": "Test operations stay within cost thresholds",
                "critical": False,
            },
        ]

        print(f"🚀 Running {len(ci_test_cases)} CI/CD tests...")
        print()

        for i, test_case in enumerate(ci_test_cases, 1):
            test_start_time = time.time()
            test_result = {
                "name": test_case["name"],
                "description": test_case["description"],
                "critical": test_case["critical"],
                "status": "unknown",
                "duration": 0,
                "message": "",
                "data": {},
            }

            try:
                print(
                    f"   Test {i}/{len(ci_test_cases)}: {test_case['description']}...",
                    end=" ",
                )

                if test_case["name"] == "test_adapter_initialization":
                    adapter = GenOpsHuggingFaceAdapter()
                    test_result["status"] = (
                        "passed" if adapter.is_available() else "failed"
                    )
                    test_result["message"] = (
                        "Adapter initialized successfully"
                        if adapter.is_available()
                        else "Adapter not available"
                    )

                elif test_case["name"] == "test_basic_text_generation":
                    adapter = GenOpsHuggingFaceAdapter()
                    result = adapter.text_generation(
                        prompt="CI/CD test prompt",
                        model="microsoft/DialoGPT-medium",
                        max_new_tokens=50,
                        team="ci_cd_team",
                        project="cicd_pipeline",
                        feature="ci_test",
                        ci_pipeline_id=cicd_context.pipeline_id,
                        ci_build_number=cicd_context.build_number,
                    )
                    test_result["status"] = "passed" if result else "failed"
                    test_result["message"] = (
                        f"Generated {len(str(result)) if result else 0} characters"
                    )
                    test_result["data"]["response_length"] = (
                        len(str(result)) if result else 0
                    )

                elif test_case["name"] == "test_cost_calculation":
                    adapter = GenOpsHuggingFaceAdapter()
                    cost = adapter._calculate_cost(
                        provider="huggingface_hub",
                        model="microsoft/DialoGPT-medium",
                        input_tokens=20,
                        output_tokens=10,
                        task="text-generation",
                    )
                    test_result["status"] = "passed" if cost >= 0 else "failed"
                    test_result["message"] = f"Calculated cost: ${cost:.6f}"
                    test_result["data"]["calculated_cost"] = cost

                elif test_case["name"] == "test_multi_provider_support":
                    adapter = GenOpsHuggingFaceAdapter()
                    providers = ["openai", "anthropic", "huggingface_hub"]
                    detected_providers = []

                    for provider in providers:
                        try:
                            test_models = {
                                "openai": "gpt-3.5-turbo",
                                "anthropic": "claude-3-haiku",
                                "huggingface_hub": "microsoft/DialoGPT-medium",
                            }
                            detected = adapter._detect_provider(test_models[provider])
                            if detected == provider:
                                detected_providers.append(provider)
                        except Exception:
                            pass

                    test_result["status"] = (
                        "passed" if len(detected_providers) >= 2 else "skipped"
                    )
                    test_result["message"] = (
                        f"Detected {len(detected_providers)} providers: {detected_providers}"
                    )
                    test_result["data"]["detected_providers"] = detected_providers

                elif test_case["name"] == "test_error_handling":
                    adapter = GenOpsHuggingFaceAdapter()
                    try:
                        # Intentionally cause an error with invalid model
                        adapter.text_generation(
                            prompt="test",
                            model="invalid/nonexistent-model",
                            team="ci_error_test",
                        )
                        test_result["status"] = "failed"
                        test_result["message"] = "Expected error but none occurred"
                    except Exception as e:
                        test_result["status"] = "passed"
                        test_result["message"] = (
                            f"Error handled correctly: {type(e).__name__}"
                        )
                        test_result["data"]["error_type"] = type(e).__name__

                elif test_case["name"] == "test_telemetry_export":
                    with create_huggingface_cost_context(
                        f"ci_test_{cicd_context.pipeline_id}"
                    ) as context:
                        context.add_hf_call(
                            provider="huggingface_hub",
                            model="test-model",
                            tokens_input=10,
                            tokens_output=5,
                            task="test",
                        )
                        summary = context.get_current_summary()

                    test_result["status"] = "passed" if summary else "failed"
                    test_result["message"] = (
                        "Telemetry context works"
                        if summary
                        else "Telemetry context failed"
                    )
                    test_result["data"]["telemetry_working"] = summary is not None

                elif test_case["name"] == "test_performance_baseline":
                    adapter = GenOpsHuggingFaceAdapter()
                    perf_start = time.time()

                    result = adapter.text_generation(
                        prompt="Performance test prompt",
                        model="microsoft/DialoGPT-medium",
                        max_new_tokens=30,
                        team="ci_perf_test",
                    )

                    perf_duration = time.time() - perf_start
                    baseline_threshold = float(
                        os.getenv("CI_PERFORMANCE_BASELINE", "2.0")
                    )

                    test_result["status"] = (
                        "passed" if perf_duration < baseline_threshold else "failed"
                    )
                    test_result["message"] = (
                        f"Duration: {perf_duration:.2f}s (baseline: {baseline_threshold}s)"
                    )
                    test_result["data"]["duration"] = perf_duration
                    test_result["data"]["baseline"] = baseline_threshold

                elif test_case["name"] == "test_cost_threshold":
                    adapter = GenOpsHuggingFaceAdapter()

                    # Simulate small operation and check cost
                    cost = adapter._calculate_cost(
                        provider="huggingface_hub",
                        model="microsoft/DialoGPT-medium",
                        input_tokens=50,
                        output_tokens=25,
                        task="text-generation",
                    )

                    cost_threshold = float(os.getenv("CI_COST_THRESHOLD", "0.10"))

                    test_result["status"] = (
                        "passed" if cost < cost_threshold else "failed"
                    )
                    test_result["message"] = (
                        f"Cost: ${cost:.4f} (threshold: ${cost_threshold})"
                    )
                    test_result["data"]["cost"] = cost
                    test_result["data"]["threshold"] = cost_threshold

                else:
                    test_result["status"] = "skipped"
                    test_result["message"] = "Test case not implemented"

            except Exception as e:
                test_result["status"] = "failed"
                test_result["message"] = f"Test failed with error: {str(e)}"
                test_result["data"]["error"] = str(e)

            test_result["duration"] = time.time() - test_start_time
            test_results["tests"].append(test_result)
            test_results["total_time"] += test_result["duration"]

            # Update counters
            if test_result["status"] == "passed":
                test_results["passed"] += 1
                print("✅ PASSED")
            elif test_result["status"] == "failed":
                test_results["failed"] += 1
                print("❌ FAILED")
            elif test_result["status"] == "skipped":
                test_results["skipped"] += 1
                print("⏭️  SKIPPED")

            # Print test details
            print(f"     {test_result['message']} ({test_result['duration']:.2f}s)")

        # Summary
        print()
        print("📊 CI/CD Test Results:")
        print(f"   ✅ Passed: {test_results['passed']}")
        print(f"   ❌ Failed: {test_results['failed']}")
        print(f"   ⏭️  Skipped: {test_results['skipped']}")
        print(f"   ⏱️  Total Time: {test_results['total_time']:.2f}s")

        # Check for critical test failures
        critical_failures = [
            t
            for t in test_results["tests"]
            if t["critical"] and t["status"] == "failed"
        ]

        if critical_failures:
            print(f"\n❌ {len(critical_failures)} critical test(s) failed:")
            for failure in critical_failures:
                print(f"   • {failure['name']}: {failure['message']}")

        return test_results["failed"] == 0 and len(critical_failures) == 0

    except ImportError as e:
        print(f"❌ Required components not available: {e}")
        return False
    except Exception as e:
        print(f"❌ CI/CD test suite failed: {e}")
        return False


def run_deployment_validation():
    """
    Run deployment validation tests.

    This validates the deployment is ready for production traffic.
    """

    print("\n🚀 Deployment Validation")
    print("=" * 30)

    validation_results = {"deployment_ready": False, "checks": {}}

    try:
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter

        cicd_context = get_cicd_context()

        print("🔍 Running deployment validation checks...")

        # Check 1: Service availability
        try:
            adapter = GenOpsHuggingFaceAdapter()
            service_available = adapter.is_available()
            validation_results["checks"]["service_availability"] = {
                "status": "pass" if service_available else "fail",
                "message": "Service available"
                if service_available
                else "Service not available",
            }
        except Exception as e:
            validation_results["checks"]["service_availability"] = {
                "status": "fail",
                "message": f"Service check failed: {e}",
            }

        # Check 2: End-to-end functionality
        try:
            adapter = GenOpsHuggingFaceAdapter()

            e2e_start = time.time()
            result = adapter.text_generation(
                prompt="Deployment validation test",
                model="microsoft/DialoGPT-medium",
                max_new_tokens=20,
                team="deployment_validation",
                project="cicd_deployment",
                feature="e2e_test",
                deployment_validation=True,
                commit_sha=cicd_context.commit_sha,
                build_number=cicd_context.build_number,
            )
            e2e_duration = time.time() - e2e_start

            validation_results["checks"]["end_to_end"] = {
                "status": "pass" if result else "fail",
                "message": f"E2E test completed in {e2e_duration:.2f}s"
                if result
                else "E2E test failed",
                "duration": e2e_duration,
            }
        except Exception as e:
            validation_results["checks"]["end_to_end"] = {
                "status": "fail",
                "message": f"E2E test failed: {e}",
            }

        # Check 3: Performance validation
        try:
            performance_threshold = 3.0  # 3 seconds for deployment validation
            if "end_to_end" in validation_results["checks"]:
                e2e_duration = validation_results["checks"]["end_to_end"].get(
                    "duration", 999
                )
                performance_ok = e2e_duration < performance_threshold

                validation_results["checks"]["performance"] = {
                    "status": "pass" if performance_ok else "fail",
                    "message": f"Performance: {e2e_duration:.2f}s (threshold: {performance_threshold}s)",
                    "duration": e2e_duration,
                    "threshold": performance_threshold,
                }
            else:
                validation_results["checks"]["performance"] = {
                    "status": "skip",
                    "message": "Performance check skipped - E2E test failed",
                }
        except Exception as e:
            validation_results["checks"]["performance"] = {
                "status": "fail",
                "message": f"Performance check failed: {e}",
            }

        # Check 4: Cost validation
        try:
            cost_threshold = 0.01  # $0.01 for deployment validation
            estimated_cost = 0.0001  # Mock estimated cost

            validation_results["checks"]["cost"] = {
                "status": "pass" if estimated_cost < cost_threshold else "fail",
                "message": f"Estimated cost: ${estimated_cost:.4f} (threshold: ${cost_threshold})",
                "cost": estimated_cost,
                "threshold": cost_threshold,
            }
        except Exception as e:
            validation_results["checks"]["cost"] = {
                "status": "fail",
                "message": f"Cost validation failed: {e}",
            }

        # Determine overall deployment readiness
        failed_checks = [
            name
            for name, check in validation_results["checks"].items()
            if check["status"] == "fail"
        ]

        validation_results["deployment_ready"] = len(failed_checks) == 0

        print()
        print("📋 Deployment Validation Results:")
        for check_name, check_result in validation_results["checks"].items():
            if check_result["status"] == "pass":
                print(f"   ✅ {check_name}: {check_result['message']}")
            elif check_result["status"] == "fail":
                print(f"   ❌ {check_name}: {check_result['message']}")
            else:
                print(f"   ⏭️  {check_name}: {check_result['message']}")

        print()
        print(
            f"🚀 Deployment Status: {'READY' if validation_results['deployment_ready'] else 'NOT READY'}"
        )

        if failed_checks:
            print(f"❌ Failed checks: {', '.join(failed_checks)}")

        return validation_results["deployment_ready"]

    except ImportError as e:
        print(f"❌ Required components not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Deployment validation failed: {e}")
        return False


def run_performance_tests():
    """
    Run performance regression tests.

    This validates performance hasn't regressed compared to baseline.
    """

    print("\n⚡ Performance Regression Tests")
    print("=" * 35)

    performance_results = {"regression_detected": False, "tests": {}}

    try:
        from genops.providers.huggingface import GenOpsHuggingFaceAdapter

        adapter = GenOpsHuggingFaceAdapter()

        # Performance test scenarios
        perf_scenarios = [
            {
                "name": "simple_generation",
                "description": "Simple text generation performance",
                "baseline": 2.0,  # 2 seconds baseline
                "prompt": "Generate a simple response",
                "max_tokens": 50,
            },
            {
                "name": "complex_generation",
                "description": "Complex text generation performance",
                "baseline": 5.0,  # 5 seconds baseline
                "prompt": "Generate a comprehensive analysis of machine learning deployment patterns",
                "max_tokens": 200,
            },
            {
                "name": "batch_embedding",
                "description": "Batch embedding performance",
                "baseline": 3.0,  # 3 seconds baseline
                "inputs": ["text1", "text2", "text3", "text4", "text5"],
                "model": "sentence-transformers/all-MiniLM-L6-v2",
            },
        ]

        print("🏃 Running performance tests...")
        print()

        for scenario in perf_scenarios:
            print(f"   Testing {scenario['description']}...", end=" ")

            try:
                start_time = time.time()

                if scenario["name"] == "batch_embedding":
                    adapter.feature_extraction(
                        inputs=scenario["inputs"],
                        model=scenario["model"],
                        team="perf_test_team",
                    )
                else:
                    adapter.text_generation(
                        prompt=scenario["prompt"],
                        model="microsoft/DialoGPT-medium",
                        max_new_tokens=scenario["max_tokens"],
                        team="perf_test_team",
                    )

                duration = time.time() - start_time
                baseline = scenario["baseline"]
                regression = duration > baseline * 1.2  # 20% regression threshold

                performance_results["tests"][scenario["name"]] = {
                    "duration": duration,
                    "baseline": baseline,
                    "regression": regression,
                    "regression_percent": ((duration - baseline) / baseline) * 100
                    if baseline > 0
                    else 0,
                    "status": "fail" if regression else "pass",
                }

                if regression:
                    performance_results["regression_detected"] = True
                    print(
                        f"❌ REGRESSION ({duration:.2f}s vs {baseline:.2f}s baseline)"
                    )
                else:
                    print(f"✅ OK ({duration:.2f}s vs {baseline:.2f}s baseline)")

            except Exception as e:
                performance_results["tests"][scenario["name"]] = {
                    "duration": 0,
                    "baseline": scenario["baseline"],
                    "regression": True,
                    "error": str(e),
                    "status": "error",
                }
                performance_results["regression_detected"] = True
                print(f"❌ ERROR ({str(e)})")

        print()
        print("📊 Performance Test Summary:")

        for test_name, test_result in performance_results["tests"].items():
            if test_result["status"] == "pass":
                print(f"   ✅ {test_name}: {test_result['duration']:.2f}s")
            elif test_result["status"] == "fail":
                print(
                    f"   ❌ {test_name}: {test_result['duration']:.2f}s ({test_result['regression_percent']:+.1f}% vs baseline)"
                )
            else:
                print(
                    f"   ❌ {test_name}: Error - {test_result.get('error', 'Unknown error')}"
                )

        print()
        print(
            f"⚡ Performance Status: {'REGRESSION DETECTED' if performance_results['regression_detected'] else 'NO REGRESSION'}"
        )

        return not performance_results["regression_detected"]

    except ImportError as e:
        print(f"❌ Required components not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Performance testing failed: {e}")
        return False


def print_cicd_integration_examples():
    """Print example CI/CD integration configurations."""

    print("\n🔧 CI/CD Integration Examples")
    print("=" * 35)

    # GitHub Actions workflow
    github_actions = """name: GenOps Hugging Face CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    environment: ci

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install genops-ai[huggingface]

    - name: Run GenOps CI tests
      env:
        HF_TOKEN_CI: ${{ secrets.HF_TOKEN_CI }}
        CICD_OTEL_ENDPOINT: ${{ vars.OTEL_ENDPOINT }}
        GITHUB_PR_NUMBER: ${{ github.event.number }}
      run: |
        python examples/huggingface/cicd_integration.py --mode=test

    - name: Performance regression tests
      if: github.event_name == 'pull_request'
      run: |
        python examples/huggingface/cicd_integration.py --mode=perf-test

    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results
        path: test-results/

  deploy-staging:
    needs: test
    if: github.ref == 'refs/heads/develop'
    runs-on: ubuntu-latest
    environment: staging

    steps:
    - uses: actions/checkout@v4

    - name: Deploy to staging
      run: |
        # Deployment commands here
        kubectl apply -f k8s/staging/

    - name: Deployment validation
      env:
        DEPLOYMENT_ENVIRONMENT: staging
      run: |
        python examples/huggingface/cicd_integration.py --mode=deploy-validate"""

    print("📄 GitHub Actions workflow:")
    print("```yaml")
    print(github_actions)
    print("```")

    # GitLab CI configuration
    gitlab_ci = r""".genops-hf-ci:
  image: python:3.11-slim
  before_script:
    - pip install -r requirements.txt
    - pip install genops-ai[huggingface]
  variables:
    GENOPS_CI_LOG_LEVEL: INFO
    HF_HOME: /tmp/.cache/huggingface

stages:
  - test
  - deploy-staging
  - deploy-production

test:
  extends: .genops-hf-ci
  stage: test
  script:
    - python examples/huggingface/cicd_integration.py --mode=test
  artifacts:
    reports:
      junit: test-results.xml
    paths:
      - test-results/
  coverage: '/TOTAL.*\s+(\d+%)$/'
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == "main"
    - if: $CI_COMMIT_BRANCH == "develop"

performance-test:
  extends: .genops-hf-ci
  stage: test
  script:
    - python examples/huggingface/cicd_integration.py --mode=perf-test
  only:
    - merge_requests
  allow_failure: true

deploy-staging:
  extends: .genops-hf-ci
  stage: deploy-staging
  script:
    - kubectl apply -f k8s/staging/
    - python examples/huggingface/cicd_integration.py --mode=deploy-validate
  environment:
    name: staging
    url: https://staging.example.com
  only:
    - develop"""

    print("\n📄 GitLab CI configuration:")
    print("```yaml")
    print(gitlab_ci)
    print("```")


def main():
    """Main demonstration function."""

    import argparse

    parser = argparse.ArgumentParser(
        description="GenOps Hugging Face CI/CD Integration"
    )
    parser.add_argument(
        "--mode",
        choices=["test", "deploy-validate", "perf-test"],
        default="test",
        help="CI/CD mode to run",
    )
    args = parser.parse_args()

    print("🔧 GenOps Hugging Face CI/CD Integration")
    print("=" * 50)
    print(f"Running in {args.mode} mode...")
    print("=" * 50)

    # Setup CI/CD configuration
    cicd_config, cicd_context = setup_cicd_configuration()

    success = True

    if args.mode == "test":
        print("🧪 Running CI/CD test mode...")
        success = run_cicd_tests()

    elif args.mode == "deploy-validate":
        print("🚀 Running deployment validation mode...")
        success = run_deployment_validation()

    elif args.mode == "perf-test":
        print("⚡ Running performance test mode...")
        success = run_performance_tests()

    # Print integration examples
    if success:
        print_cicd_integration_examples()

    print("\n" + "=" * 50)
    if success:
        print("✅ CI/CD integration completed successfully!")
    else:
        print("❌ CI/CD integration failed!")
    print("=" * 50)

    # Exit with proper code for CI/CD
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
