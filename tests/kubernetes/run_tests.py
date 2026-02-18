#!/usr/bin/env python3
"""
Test runner for GenOps AI Kubernetes tests.

This script provides a convenient way to run Kubernetes-specific tests
with different configurations and environments.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


class KubernetesTestRunner:
    """Test runner for Kubernetes tests."""

    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.project_root = self.test_dir.parent.parent

    def run_tests(
        self,
        test_pattern: Optional[str] = None,
        verbose: bool = False,
        coverage: bool = False,
        slow_tests: bool = False,
        integration_tests: bool = False,
        parallel: bool = False,
        output_format: str = "auto",
    ) -> int:
        """Run Kubernetes tests with specified options."""

        cmd = ["python", "-m", "pytest"]

        # Test directory
        if test_pattern:
            cmd.append(f"{self.test_dir}/{test_pattern}")
        else:
            cmd.append(str(self.test_dir))

        # Verbosity
        if verbose:
            cmd.extend(["-v", "-s"])

        # Coverage
        if coverage:
            cmd.extend(
                [
                    "--cov=src/genops/providers/kubernetes",
                    "--cov-report=html",
                    "--cov-report=term-missing",
                    f"--cov-report=html:{self.project_root}/coverage_html",
                ]
            )

        # Test selection
        markers = []
        if not slow_tests:
            markers.append("not slow")
        if integration_tests:
            markers.append("integration")

        if markers:
            cmd.extend(["-m", " and ".join(markers)])

        # Parallel execution
        if parallel:
            try:
                import pytest_xdist  # noqa: F401

                cmd.extend(["-n", "auto"])
            except ImportError:
                print("Warning: pytest-xdist not installed, running sequentially")

        # Output format
        if output_format == "junit":
            cmd.extend(["--junit-xml", f"{self.project_root}/test-results.xml"])
        elif output_format == "json":
            cmd.extend(
                [
                    "--json-report",
                    "--json-report-file",
                    f"{self.project_root}/test-results.json",
                ]
            )

        # Environment setup
        env = os.environ.copy()
        env.update(
            {
                "PYTHONPATH": str(self.project_root / "src"),
                "GENOPS_ENV": "test",
                "LOG_LEVEL": "DEBUG",
            }
        )

        print(f"Running command: {' '.join(cmd)}")
        print(f"Working directory: {self.test_dir}")

        # Run tests
        try:
            result = subprocess.run(cmd, cwd=self.test_dir, env=env)
            return result.returncode
        except KeyboardInterrupt:
            print("\nTests interrupted by user")
            return 130
        except Exception as e:
            print(f"Error running tests: {e}")
            return 1

    def list_tests(self) -> None:
        """List available tests."""

        cmd = ["python", "-m", "pytest", "--collect-only", "-q", str(self.test_dir)]

        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.project_root / "src")

        try:
            subprocess.run(cmd, cwd=self.test_dir, env=env)
        except Exception as e:
            print(f"Error listing tests: {e}")

    def check_dependencies(self) -> bool:
        """Check if test dependencies are available."""

        print("🔍 Checking test dependencies...")

        # Check pytest
        try:
            import pytest

            print(f"✅ pytest: {pytest.__version__}")
        except ImportError:
            print("❌ pytest not installed")
            return False

        # Check optional dependencies
        optional_deps = [
            ("pytest-xdist", "parallel test execution"),
            ("pytest-cov", "coverage reporting"),
            ("pytest-json-report", "JSON test reports"),
            ("pytest-html", "HTML test reports"),
        ]

        for dep, description in optional_deps:
            try:
                __import__(dep.replace("-", "_"))
                print(f"✅ {dep}: available ({description})")
            except ImportError:
                print(f"⚠️ {dep}: not installed ({description})")

        # Check Kubernetes availability
        try:
            result = subprocess.run(
                ["kubectl", "cluster-info"], capture_output=True, timeout=5
            )
            if result.returncode == 0:
                print("✅ Kubernetes cluster: available")
            else:
                print(
                    "⚠️ Kubernetes cluster: not available (some tests will be skipped)"
                )
        except Exception:
            print("⚠️ kubectl not found or Kubernetes not available")

        # Check GenOps installation
        try:
            sys.path.insert(0, str(self.project_root / "src"))
            import genops  # noqa: F401

            print("✅ GenOps: available")
        except ImportError:
            print("⚠️ GenOps: not installed (using mocks)")

        print("\n🎯 Dependencies check complete")
        return True

    def run_specific_test_suite(self, suite: str) -> int:
        """Run a specific test suite."""

        suites = {
            "examples": "test_examples.py",
            "provider": "test_kubernetes_provider.py",
            "integration": "test_integration.py",
            "performance": "test_performance.py",
        }

        if suite not in suites:
            print(f"Unknown test suite: {suite}")
            print(f"Available suites: {', '.join(suites.keys())}")
            return 1

        return self.run_tests(test_pattern=suites[suite], verbose=True)

    def generate_test_report(self) -> None:
        """Generate comprehensive test report."""

        print("📊 Generating comprehensive test report...")

        # Run tests with coverage and reporting
        cmd = [
            "python",
            "-m",
            "pytest",
            str(self.test_dir),
            "--cov=src/genops/providers/kubernetes",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-report=xml",
            "--junit-xml=test-results.xml",
            "--html=test-report.html",
            "--self-contained-html",
            "-v",
        ]

        env = os.environ.copy()
        env.update({"PYTHONPATH": str(self.project_root / "src"), "GENOPS_ENV": "test"})

        try:
            subprocess.run(cmd, cwd=self.project_root, env=env)

            print("\n📋 Test Report Generated:")
            print(f"  HTML Report: {self.project_root}/test-report.html")
            print(f"  Coverage HTML: {self.project_root}/htmlcov/index.html")
            print(f"  JUnit XML: {self.project_root}/test-results.xml")
            print(f"  Coverage XML: {self.project_root}/coverage.xml")

        except Exception as e:
            print(f"Error generating report: {e}")


def main():
    """Main CLI interface."""

    parser = argparse.ArgumentParser(
        description="Run GenOps AI Kubernetes tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all tests
    python run_tests.py

    # Run with coverage
    python run_tests.py --coverage

    # Run specific test file
    python run_tests.py --pattern test_examples.py

    # Run integration tests only
    python run_tests.py --integration

    # Run specific test suite
    python run_tests.py --suite examples

    # Generate comprehensive report
    python run_tests.py --report

    # Check dependencies
    python run_tests.py --check-deps
        """,
    )

    parser.add_argument("--pattern", "-p", help="Test file pattern to run")

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    parser.add_argument(
        "--coverage", "-c", action="store_true", help="Generate coverage report"
    )

    parser.add_argument("--slow", action="store_true", help="Include slow tests")

    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests"
    )

    parser.add_argument(
        "--parallel", "-j", action="store_true", help="Run tests in parallel"
    )

    parser.add_argument(
        "--format",
        choices=["auto", "junit", "json"],
        default="auto",
        help="Output format",
    )

    parser.add_argument(
        "--suite",
        "-s",
        choices=["examples", "provider", "integration", "performance"],
        help="Run specific test suite",
    )

    parser.add_argument(
        "--list", "-l", action="store_true", help="List available tests"
    )

    parser.add_argument(
        "--check-deps", action="store_true", help="Check test dependencies"
    )

    parser.add_argument(
        "--report", "-r", action="store_true", help="Generate comprehensive test report"
    )

    args = parser.parse_args()

    runner = KubernetesTestRunner()

    # Handle special commands first
    if args.check_deps:
        if not runner.check_dependencies():
            return 1
        return 0

    if args.list:
        runner.list_tests()
        return 0

    if args.report:
        runner.generate_test_report()
        return 0

    if args.suite:
        return runner.run_specific_test_suite(args.suite)

    # Run tests with specified options
    return runner.run_tests(
        test_pattern=args.pattern,
        verbose=args.verbose,
        coverage=args.coverage,
        slow_tests=args.slow,
        integration_tests=args.integration,
        parallel=args.parallel,
        output_format=args.format,
    )


if __name__ == "__main__":
    sys.exit(main())
