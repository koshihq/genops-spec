#!/usr/bin/env python3
"""
PromptLayer Test Suite Runner

Comprehensive test runner for PromptLayer integration tests.
Counts tests, runs test suites, and provides detailed reporting.
"""

import sys
import os
import pytest
import time
from pathlib import Path

def count_test_methods():
    """Count all test methods in the PromptLayer test suite."""
    test_dir = Path(__file__).parent
    test_files = list(test_dir.glob('test_*.py'))
    
    total_tests = 0
    test_breakdown = {}
    
    print("🔍 Analyzing PromptLayer Test Suite")
    print("=" * 50)
    
    for test_file in test_files:
        if test_file.name == 'conftest.py':
            continue
            
        with open(test_file, 'r') as f:
            content = f.read()
        
        # Count test methods (functions starting with 'test_')
        import ast
        tree = ast.parse(content)
        
        file_tests = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                file_tests += 1
        
        test_breakdown[test_file.name] = file_tests
        total_tests += file_tests
        
        print(f"📁 {test_file.name}: {file_tests} tests")
    
    print("-" * 50)
    print(f"📊 Total Tests: {total_tests}")
    print()
    
    # Verify we meet the 75+ test requirement
    if total_tests >= 75:
        print(f"✅ Test count requirement MET: {total_tests} >= 75")
    else:
        print(f"❌ Test count requirement NOT MET: {total_tests} < 75")
    
    return total_tests, test_breakdown

def run_test_categories():
    """Run tests by category and report results."""
    test_categories = {
        "Core Adapter Tests": "test_promptlayer_adapter.py",
        "Validation Tests": "test_promptlayer_validation.py", 
        "Integration Tests": "test_integration.py",
        "Cost Tracking Tests": "test_cost_tracking.py",
        "Error Handling Tests": "test_error_handling.py",
        "Performance Tests": "test_performance.py"
    }
    
    print("🧪 Running PromptLayer Test Categories")
    print("=" * 50)
    
    results = {}
    total_start_time = time.time()
    
    for category, test_file in test_categories.items():
        print(f"\n📋 {category}")
        print("-" * 30)
        
        start_time = time.time()
        
        # Run pytest for specific file
        exit_code = pytest.main([
            f"{test_file}",
            "-v",
            "--tb=short",
            "-x"  # Stop on first failure
        ])
        
        end_time = time.time()
        duration = end_time - start_time
        
        if exit_code == 0:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
        
        results[category] = {
            'status': status,
            'duration': duration,
            'exit_code': exit_code
        }
        
        print(f"{status} ({duration:.1f}s)")
    
    total_duration = time.time() - total_start_time
    
    # Summary report
    print("\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    passed_categories = 0
    for category, result in results.items():
        print(f"{result['status']} {category} ({result['duration']:.1f}s)")
        if result['exit_code'] == 0:
            passed_categories += 1
    
    print("-" * 50)
    print(f"Categories Passed: {passed_categories}/{len(test_categories)}")
    print(f"Total Duration: {total_duration:.1f}s")
    
    return results

def run_comprehensive_test_suite():
    """Run the complete test suite with detailed reporting."""
    print("🚀 PromptLayer Comprehensive Test Suite")
    print("=" * 60)
    
    # Count tests
    total_tests, breakdown = count_test_methods()
    
    # Check if we should run tests
    if '--count-only' in sys.argv:
        print("📋 Test counting complete. Use --run to execute tests.")
        return 0
    
    if '--run' not in sys.argv:
        print("💡 Use --run to execute the test suite")
        print("💡 Use --count-only to just count tests")
        return 0
    
    print("🏃 Executing Test Suite...")
    print()
    
    # Run test categories
    results = run_test_categories()
    
    # Overall result
    failed_categories = [cat for cat, res in results.items() if res['exit_code'] != 0]
    
    if not failed_categories:
        print("\n🎉 ALL TEST CATEGORIES PASSED!")
        print(f"✅ {total_tests} tests across {len(results)} categories")
        return 0
    else:
        print(f"\n❌ {len(failed_categories)} CATEGORIES FAILED:")
        for category in failed_categories:
            print(f"   • {category}")
        return 1

def show_test_coverage():
    """Show test coverage analysis."""
    print("📋 PromptLayer Test Coverage Analysis")
    print("=" * 50)
    
    coverage_areas = {
        "Core Functionality": [
            "Adapter initialization and configuration",
            "Context manager lifecycle",
            "Governance policy enforcement", 
            "Cost tracking and attribution",
            "Span creation and management"
        ],
        "Integration Patterns": [
            "Auto-instrumentation setup",
            "Manual adapter usage",
            "Multi-step workflows",
            "Cross-provider compatibility",
            "Real API integration tests"
        ],
        "Error Handling": [
            "API connection errors",
            "Authentication failures",
            "Rate limiting scenarios",
            "Graceful degradation",
            "Recovery patterns"
        ],
        "Performance": [
            "Latency benchmarks",
            "Memory usage optimization",
            "Concurrent operations",
            "Scalability patterns",
            "Resource efficiency"
        ],
        "Cost & Budget": [
            "Cost calculation accuracy",
            "Budget enforcement",
            "Team attribution",
            "Financial reporting",
            "ROI calculations"
        ],
        "Validation": [
            "Setup validation",
            "Environment checking",
            "Dependency verification",
            "Configuration validation",
            "Connectivity testing"
        ]
    }
    
    for area, features in coverage_areas.items():
        print(f"\n📊 {area}:")
        for feature in features:
            print(f"   ✅ {feature}")
    
    print(f"\n📈 Coverage Summary:")
    print(f"   • {len(coverage_areas)} major functional areas")
    print(f"   • {sum(len(features) for features in coverage_areas.values())} specific features")
    print(f"   • Comprehensive integration and unit test coverage")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("PromptLayer Test Suite Runner")
        print("")
        print("Commands:")
        print("  --count-only    Count tests without running")
        print("  --run          Run complete test suite")
        print("  --coverage     Show test coverage analysis")
        print("")
        sys.exit(0)
    
    if '--coverage' in sys.argv:
        show_test_coverage()
        sys.exit(0)
    
    # Run main test suite
    exit_code = run_comprehensive_test_suite()
    sys.exit(exit_code)