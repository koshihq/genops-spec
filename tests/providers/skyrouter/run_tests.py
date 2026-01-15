#!/usr/bin/env python3
"""
SkyRouter Test Suite Runner

Comprehensive test runner for the SkyRouter provider integration.
Executes the complete test suite with detailed reporting and coverage analysis.

Usage:
    python run_tests.py [options]
    
Options:
    --verbose    Enable verbose output
    --coverage   Run with coverage analysis
    --integration Include integration tests
    --performance Include performance tests
    --enterprise Include enterprise tests
    --fast       Run only fast unit tests
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, capture_output=False):
    """Run a command and handle errors."""
    print(f"Running: {' '.join(cmd)}")
    try:
        if capture_output:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout
        else:
            subprocess.run(cmd, check=True)
            return None
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        if capture_output and e.stderr:
            print(f"Error: {e.stderr}")
        return None


def check_dependencies():
    """Check if required dependencies are installed."""
    print("Checking dependencies...")
    
    # Check if pytest is installed
    try:
        import pytest
        print(f"✅ pytest {pytest.__version__} found")
    except ImportError:
        print("❌ pytest not found. Install with: pip install pytest")
        return False
    
    # Check if coverage is available (optional)
    try:
        import coverage
        print(f"✅ coverage {coverage.__version__} found")
    except ImportError:
        print("⚠️  coverage not found. Install with: pip install coverage")
    
    # Check if SkyRouter provider is available
    try:
        from genops.providers.skyrouter import GenOpsSkyRouterAdapter
        print("✅ SkyRouter provider found")
    except ImportError:
        print("⚠️  SkyRouter provider not found. Some tests may be skipped.")
    
    return True


def run_tests(args):
    """Run the test suite with specified options."""
    if not check_dependencies():
        return False
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test directory
    test_dir = Path(__file__).parent
    cmd.append(str(test_dir))
    
    # Configure verbosity
    if args.verbose:
        cmd.extend(["-v", "-s"])
    else:
        cmd.append("-q")
    
    # Configure coverage
    if args.coverage:
        cmd.extend([
            "--cov=genops.providers.skyrouter",
            "--cov=genops.providers.skyrouter_pricing", 
            "--cov=genops.providers.skyrouter_validation",
            "--cov=genops.providers.skyrouter_cost_aggregator",
            "--cov-report=term-missing",
            "--cov-report=html:coverage_html"
        ])
    
    # Configure test selection
    markers = []
    
    if args.fast:
        # Only run fast unit tests (exclude slow integration/performance tests)
        markers.append("not integration and not performance")
    else:
        # Include/exclude specific test categories
        if not args.integration:
            markers.append("not integration")
        if not args.performance:
            markers.append("not performance")
        if not args.enterprise:
            markers.append("not enterprise")
    
    if markers:
        cmd.extend(["-m", " and ".join(markers)])
    
    # Add additional pytest options
    cmd.extend([
        "--tb=short",  # Shorter traceback format
        "--strict-markers",  # Enforce marker definitions
        "--disable-warnings"  # Reduce noise from warnings
    ])
    
    print("\n🧪 Running SkyRouter Test Suite")
    print("=" * 50)
    
    success = run_command(cmd) is not None
    
    if args.coverage and success:
        print("\n📊 Coverage report generated in coverage_html/")
    
    return success


def run_specific_test_file(test_file, verbose=False):
    """Run a specific test file."""
    cmd = ["python", "-m", "pytest", str(test_file)]
    if verbose:
        cmd.extend(["-v", "-s"])
    
    print(f"\n🧪 Running {test_file.name}")
    print("=" * 30)
    
    return run_command(cmd) is not None


def run_test_analysis():
    """Run test analysis and reporting."""
    print("\n📋 Test Suite Analysis")
    print("=" * 25)
    
    test_dir = Path(__file__).parent
    test_files = list(test_dir.glob("test_*.py"))
    
    print(f"📁 Test files found: {len(test_files)}")
    for test_file in test_files:
        print(f"   • {test_file.name}")
    
    # Count test functions
    total_tests = 0
    for test_file in test_files:
        try:
            with open(test_file, 'r') as f:
                content = f.read()
                test_count = content.count("def test_")
                total_tests += test_count
                print(f"   📊 {test_file.name}: {test_count} tests")
        except Exception as e:
            print(f"   ❌ Error reading {test_file.name}: {e}")
    
    print(f"\n📈 Total estimated tests: {total_tests}")
    
    # Check for required test coverage
    required_modules = [
        "test_skyrouter_adapter.py",
        "test_skyrouter_pricing.py", 
        "test_skyrouter_validation.py",
        "test_skyrouter_cost_aggregator.py",
        "test_integration.py"
    ]
    
    missing_modules = []
    for module in required_modules:
        if not (test_dir / module).exists():
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\n⚠️  Missing test modules: {missing_modules}")
    else:
        print("\n✅ All required test modules present")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="SkyRouter Test Suite Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --verbose          # Run with verbose output
  python run_tests.py --coverage         # Run with coverage analysis
  python run_tests.py --fast             # Run only fast unit tests
  python run_tests.py --integration      # Include integration tests
  python run_tests.py --analysis         # Show test suite analysis
        """
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Run with coverage analysis"
    )
    
    parser.add_argument(
        "--integration", "-i",
        action="store_true",
        help="Include integration tests"
    )
    
    parser.add_argument(
        "--performance", "-p",
        action="store_true",
        help="Include performance tests"
    )
    
    parser.add_argument(
        "--enterprise", "-e",
        action="store_true",
        help="Include enterprise feature tests"
    )
    
    parser.add_argument(
        "--fast", "-f",
        action="store_true",
        help="Run only fast unit tests"
    )
    
    parser.add_argument(
        "--analysis", "-a",
        action="store_true",
        help="Show test suite analysis"
    )
    
    parser.add_argument(
        "--file",
        help="Run specific test file"
    )
    
    args = parser.parse_args()
    
    # Handle specific operations
    if args.analysis:
        run_test_analysis()
        return
    
    if args.file:
        test_file = Path(args.file)
        if not test_file.exists():
            test_file = Path(__file__).parent / args.file
        
        if test_file.exists():
            success = run_specific_test_file(test_file, args.verbose)
        else:
            print(f"❌ Test file not found: {args.file}")
            success = False
    else:
        # Run main test suite
        success = run_tests(args)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()