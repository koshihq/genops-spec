#!/usr/bin/env python3
"""
Test runner for Together AI provider comprehensive test suite.

Runs all tests with proper configuration and generates test reports.
Supports different test categories and coverage reporting.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def run_command(command, description=""):
    """Run a command and return the result."""
    print(f"🔧 {description}")
    print(f"   Command: {' '.join(command)}")

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode == 0:
        print("   ✅ Success")
        if result.stdout.strip():
            print(f"   Output: {result.stdout.strip()}")
    else:
        print(f"   ❌ Failed (exit code: {result.returncode})")
        if result.stderr.strip():
            print(f"   Error: {result.stderr.strip()}")
        if result.stdout.strip():
            print(f"   Output: {result.stdout.strip()}")

    return result.returncode == 0, result.stdout, result.stderr


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run Together AI provider tests")
    parser.add_argument(
        "--category",
        choices=["unit", "integration", "performance", "cross_provider", "all"],
        default="all",
        help="Category of tests to run",
    )
    parser.add_argument(
        "--coverage", action="store_true", help="Generate coverage report"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--fast", action="store_true", help="Skip slow tests")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")

    args = parser.parse_args()

    print("🧪 Together AI Provider Test Suite")
    print("=" * 50)

    # Set up test directory
    test_dir = Path(__file__).parent
    os.chdir(test_dir)

    # Base pytest command
    pytest_cmd = ["python", "-m", "pytest"]

    # Add verbosity
    if args.verbose:
        pytest_cmd.extend(["-v", "-s"])
    else:
        pytest_cmd.append("-v")

    # Add parallel execution
    if args.parallel:
        pytest_cmd.extend(["-n", "auto"])

    # Add coverage
    if args.coverage:
        pytest_cmd.extend(
            [
                "--cov=src.genops.providers.together",
                "--cov=src.genops.providers.together_pricing",
                "--cov=src.genops.providers.together_validation",
                "--cov-report=html",
                "--cov-report=term-missing",
            ]
        )

    # Category-specific test selection
    if args.category == "unit":
        pytest_cmd.extend(["-m", "unit"])
        test_files = ["test_adapter.py", "test_pricing.py", "test_validation.py"]
    elif args.category == "integration":
        pytest_cmd.extend(["-m", "integration"])
        test_files = ["test_integration.py"]
    elif args.category == "performance":
        pytest_cmd.extend(["-m", "performance"])
        test_files = ["test_performance.py"]
    elif args.category == "cross_provider":
        pytest_cmd.extend(["-m", "cross_provider"])
        test_files = ["test_cross_provider.py"]
    else:  # all
        test_files = [
            "test_adapter.py",
            "test_pricing.py",
            "test_validation.py",
            "test_integration.py",
            "test_cross_provider.py",
            "test_performance.py",
        ]

    # Skip slow tests if requested
    if args.fast:
        pytest_cmd.extend(["-m", "not slow"])

    # Add test files
    existing_test_files = [f for f in test_files if os.path.exists(f)]
    pytest_cmd.extend(existing_test_files)

    print("📊 Test Configuration:")
    print(f"   Category: {args.category}")
    print(f"   Test files: {len(existing_test_files)}")
    print(f"   Coverage: {args.coverage}")
    print(f"   Parallel: {args.parallel}")
    print(f"   Fast mode: {args.fast}")
    print()

    # Check if tests exist
    if not existing_test_files:
        print("❌ No test files found!")
        return 1

    # Run the tests
    print("🚀 Running tests...")
    success, stdout, stderr = run_command(pytest_cmd, f"Running {args.category} tests")

    if success:
        print()
        print("🎉 Test Results Summary:")

        # Extract summary from pytest output
        lines = stdout.split("\n")
        summary_lines = []
        in_summary = False

        for line in lines:
            if "=" in line and (
                "passed" in line or "failed" in line or "error" in line
            ):
                in_summary = True
                summary_lines.append(line)
            elif in_summary and line.strip():
                summary_lines.append(line)
            elif in_summary and not line.strip():
                break

        for line in summary_lines:
            print(f"   {line}")

        # Coverage report location
        if args.coverage:
            coverage_html = test_dir / "htmlcov" / "index.html"
            if coverage_html.exists():
                print(f"   📊 Coverage report: {coverage_html}")

        print()
        print("✅ All tests completed successfully!")
        return 0

    else:
        print()
        print("❌ Tests failed!")
        print()
        print("Error details:")
        if stderr:
            print(stderr)
        if stdout:
            print(stdout)
        return 1


def run_specific_test(test_name):
    """Run a specific test by name."""
    test_dir = Path(__file__).parent
    os.chdir(test_dir)

    pytest_cmd = ["python", "-m", "pytest", "-v", "-s", "-k", test_name]

    print(f"🧪 Running specific test: {test_name}")
    success, stdout, stderr = run_command(pytest_cmd, f"Running test: {test_name}")

    if success:
        print("✅ Test passed!")
    else:
        print("❌ Test failed!")
        if stderr:
            print(f"Error: {stderr}")

    return success


def check_test_requirements():
    """Check if test requirements are met."""
    print("🔍 Checking test requirements...")

    required_packages = ["pytest", "pytest-cov", "pytest-xdist", "psutil"]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"   ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package} (missing)")

    if missing_packages:
        print()
        print("📦 Install missing packages:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False

    print("   ✅ All requirements met!")
    return True


if __name__ == "__main__":
    # Check requirements first
    if not check_test_requirements():
        print("\n❌ Requirements not met. Please install missing packages.")
        sys.exit(1)

    # Run main test suite
    exit_code = main()
    sys.exit(exit_code)
