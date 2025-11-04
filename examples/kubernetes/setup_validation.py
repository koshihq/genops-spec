#!/usr/bin/env python3
"""
✅ Kubernetes Setup Validation Example

Validates that your Kubernetes environment is properly configured for GenOps AI.
This script demonstrates the standard validation pattern and provides actionable fixes.

Usage:
    python setup_validation.py
    python setup_validation.py --detailed
    python setup_validation.py --fix-issues
"""

import argparse
import sys
from typing import List, Optional

# Import GenOps Kubernetes validation
try:
    from genops.providers.kubernetes import (
        validate_kubernetes_setup,
        print_kubernetes_validation_result
    )
    GENOPS_AVAILABLE = True
except ImportError:
    GENOPS_AVAILABLE = False
    print("⚠️  GenOps not installed. Install with: pip install genops")


def validate_environment(detailed: bool = False, fix_issues: bool = False) -> bool:
    """
    Validate Kubernetes environment for GenOps AI.
    
    Args:
        detailed: Show detailed validation information
        fix_issues: Attempt to fix common issues automatically
        
    Returns:
        True if validation passes, False otherwise
    """
    
    if not GENOPS_AVAILABLE:
        print("❌ GenOps AI not available")
        print("   Fix: pip install genops")
        return False
    
    print("🚢 Validating Kubernetes Environment for GenOps AI")
    print("=" * 60)
    
    try:
        # Run comprehensive validation
        result = validate_kubernetes_setup(
            enable_resource_monitoring=True,
            cluster_name=None  # Auto-detect
        )
        
        # Print results in user-friendly format
        print_kubernetes_validation_result(result)
        
        if detailed:
            print_detailed_validation_info(result)
        
        if fix_issues and not result.is_valid:
            attempt_common_fixes(result)
            
            # Re-validate after fixes
            print("\n🔄 Re-validating after fixes...")
            result = validate_kubernetes_setup()
            print_kubernetes_validation_result(result)
        
        return result.is_valid
        
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        return False


def print_detailed_validation_info(result) -> None:
    """Print detailed validation information."""
    
    print("\n🔍 DETAILED VALIDATION INFORMATION")
    print("=" * 60)
    
    if result.is_kubernetes_environment:
        print(f"📊 Environment Details:")
        print(f"   Namespace: {result.namespace or 'Not detected'}")
        print(f"   Pod Name: {result.pod_name or 'Not detected'}")
        print(f"   Node Name: {result.node_name or 'Not detected'}")
        print(f"   Cluster: {result.cluster_name or 'Not detected'}")
        
        print(f"\n⚙️ Capabilities:")
        print(f"   Service Account: {'✅' if result.has_service_account else '❌'}")
        print(f"   Resource Monitoring: {'✅' if result.has_resource_monitoring else '❌'}")
        
        if result.cpu_limit or result.memory_limit:
            print(f"\n💾 Resource Limits:")
            if result.cpu_limit:
                print(f"   CPU Limit: {result.cpu_limit}")
            if result.memory_limit:
                print(f"   Memory Limit: {result.memory_limit}")
    
    # Show environment variables
    print(f"\n🌍 Environment Variables:")
    env_vars = [
        "KUBERNETES_SERVICE_HOST",
        "KUBERNETES_SERVICE_PORT", 
        "HOSTNAME",
        "POD_NAME",
        "POD_NAMESPACE",
        "NODE_NAME"
    ]
    
    import os
    for var in env_vars:
        value = os.getenv(var, "Not set")
        print(f"   {var}: {value}")


def attempt_common_fixes(result) -> None:
    """Attempt to fix common validation issues."""
    
    print("\n🔧 ATTEMPTING COMMON FIXES")
    print("=" * 60)
    
    fixes_applied = []
    
    # Check for missing environment variables
    import os
    if not result.pod_name:
        if not os.getenv("HOSTNAME"):
            print("⚠️  Cannot fix missing pod name - requires Kubernetes downward API")
        else:
            print("✅ Pod name available via HOSTNAME")
            fixes_applied.append("Pod name detection")
    
    if not result.pod_namespace:
        if not os.getenv("POD_NAMESPACE"):
            print("⚠️  Cannot fix missing namespace - requires Kubernetes downward API")
            print("   Add to your deployment:")
            print("   env:")
            print("   - name: POD_NAMESPACE")
            print("     valueFrom:")
            print("       fieldRef:")
            print("         fieldPath: metadata.namespace")
        else:
            print("✅ Pod namespace available via environment")
            fixes_applied.append("Namespace detection")
    
    # Check service account
    if not result.has_service_account:
        print("⚠️  Cannot auto-fix service account - manual intervention required")
        print("   Ensure your pod has a service account with appropriate permissions")
    
    if fixes_applied:
        print(f"\n✅ Applied {len(fixes_applied)} fixes:")
        for fix in fixes_applied:
            print(f"   • {fix}")
    else:
        print("ℹ️  No automatic fixes available - manual intervention required")


def demonstrate_kubernetes_detection() -> None:
    """Demonstrate Kubernetes environment detection capabilities."""
    
    print("\n🔍 KUBERNETES DETECTION DEMONSTRATION")
    print("=" * 60)
    
    if not GENOPS_AVAILABLE:
        print("❌ GenOps not available for demonstration")
        return
    
    from genops.providers.kubernetes import KubernetesDetector
    
    detector = KubernetesDetector()
    
    print(f"Running in Kubernetes: {detector.is_kubernetes()}")
    print(f"Namespace: {detector.get_namespace() or 'Unknown'}")
    print(f"Pod Name: {detector.get_pod_name() or 'Unknown'}")
    print(f"Node Name: {detector.get_node_name() or 'Unknown'}")
    
    # Show governance attributes
    print(f"\n📊 Governance Attributes:")
    attrs = detector.get_governance_attributes()
    for key, value in attrs.items():
        print(f"   {key}: {value}")
    
    # Show resource context
    print(f"\n🎯 Resource Context:")
    resource_attrs = detector.get_resource_context()
    for key, value in resource_attrs.items():
        print(f"   {key}: {value}")


def run_integration_test() -> bool:
    """Run a basic integration test to verify everything works."""
    
    print("\n🧪 INTEGRATION TEST")
    print("=" * 60)
    
    if not GENOPS_AVAILABLE:
        print("❌ Cannot run integration test - GenOps not available")
        return False
    
    try:
        from genops.providers.kubernetes import KubernetesAdapter
        
        # Test adapter creation
        adapter = KubernetesAdapter()
        print("✅ Kubernetes adapter created successfully")
        
        # Test basic operations
        is_available = adapter.is_available()
        print(f"✅ Kubernetes environment available: {is_available}")
        
        framework_name = adapter.get_framework_name()
        print(f"✅ Framework name: {framework_name}")
        
        # Test telemetry attributes
        attrs = adapter.get_telemetry_attributes(test_attr="test_value")
        print(f"✅ Telemetry attributes collected: {len(attrs)} attributes")
        
        print("\n🎉 Integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    """Main validation script."""
    
    parser = argparse.ArgumentParser(
        description="Validate Kubernetes setup for GenOps AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python setup_validation.py                    # Basic validation
    python setup_validation.py --detailed         # Detailed validation
    python setup_validation.py --fix-issues       # Attempt to fix issues
    python setup_validation.py --demo            # Show detection capabilities
    python setup_validation.py --test            # Run integration test
        """
    )
    
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed validation information"
    )
    
    parser.add_argument(
        "--fix-issues", 
        action="store_true",
        help="Attempt to fix common issues automatically"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true", 
        help="Demonstrate Kubernetes detection capabilities"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run integration test"
    )
    
    args = parser.parse_args()
    
    success = True
    
    # Run validation by default or if explicitly requested
    if not args.demo and not args.test:
        success = validate_environment(
            detailed=args.detailed,
            fix_issues=args.fix_issues
        )
    
    # Run demo if requested
    if args.demo:
        demonstrate_kubernetes_detection()
    
    # Run integration test if requested
    if args.test:
        test_success = run_integration_test()
        success = success and test_success
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()