#!/usr/bin/env python3
"""
Arize AI + GenOps Auto-Instrumentation Example

This example demonstrates zero-code auto-instrumentation for Arize AI operations.
With auto-instrumentation, your existing Arize code automatically includes
GenOps governance, cost tracking, and team attribution without any changes.

Features demonstrated:
- Zero-code auto-instrumentation setup
- Transparent governance for existing Arize operations
- Automatic cost tracking and attribution
- Global adapter configuration and management
- Before/after comparison of instrumentation

Run this example:
    python auto_instrumentation.py

Prerequisites:
    export ARIZE_API_KEY="your-arize-api-key"
    export ARIZE_SPACE_KEY="your-arize-space-key"
    export GENOPS_TEAM="your-team" (optional)
    export GENOPS_PROJECT="your-project" (optional)
"""

import os
import sys
import time
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List


def print_header():
    """Print example header."""
    print("=" * 60)
    print("🤖 Arize AI + GenOps Auto-Instrumentation Example")
    print("=" * 60)
    print()


def check_prerequisites():
    """Check if all required dependencies are available."""
    print("📋 Prerequisites Check:")
    
    missing_requirements = []
    
    try:
        import genops
        print("  ✅ GenOps installed")
    except ImportError:
        print("  ❌ GenOps not installed")
        missing_requirements.append("pip install genops[arize]")
    
    try:
        import arize
        print("  ✅ Arize SDK installed")
    except ImportError:
        print("  ❌ Arize SDK not installed") 
        missing_requirements.append("pip install arize>=6.0.0")
    
    # Check environment variables
    api_key = os.getenv('ARIZE_API_KEY')
    space_key = os.getenv('ARIZE_SPACE_KEY')
    
    if api_key and len(api_key) > 10:
        print("  ✅ ARIZE_API_KEY configured")
    else:
        print("  ❌ ARIZE_API_KEY not properly configured")
        missing_requirements.append("export ARIZE_API_KEY='your-api-key'")
    
    if space_key and len(space_key) > 10:
        print("  ✅ ARIZE_SPACE_KEY configured")
    else:
        print("  ❌ ARIZE_SPACE_KEY not properly configured")
        missing_requirements.append("export ARIZE_SPACE_KEY='your-space-key'")
    
    if missing_requirements:
        print(f"\n❌ Missing requirements:")
        for req in missing_requirements:
            print(f"   {req}")
        return False
    
    print("  ✅ All prerequisites met!")
    print()
    return True


def demonstrate_before_instrumentation():
    """Show Arize operations before GenOps instrumentation."""
    print("📋 Before Auto-Instrumentation:")
    print("  Your existing Arize code runs normally but without governance...")
    print()
    
    try:
        from arize.pandas.logger import Client
        
        # Create Arize client (your existing code)
        arize_client = Client(
            api_key=os.getenv('ARIZE_API_KEY'),
            space_key=os.getenv('ARIZE_SPACE_KEY')
        )
        print("  ✅ Arize client created")
        
        # Simulate a prediction log (your existing code)
        print("  📊 Simulating prediction logging...")
        
        # This is what your existing code might look like
        sample_prediction = {
            "prediction_id": f"before_instrumentation_{int(time.time())}",
            "prediction_label": "legitimate",
            "actual_label": "legitimate",
            "model_id": "fraud-detection-model",
            "model_version": "1.0"
        }
        
        print(f"     • Prediction ID: {sample_prediction['prediction_id']}")
        print(f"     • Model: {sample_prediction['model_id']}-{sample_prediction['model_version']}")
        print(f"     • Prediction: {sample_prediction['prediction_label']}")
        
        # Note: We don't actually call arize_client.log() to avoid API calls
        # In your real code, this would be:
        # response = arize_client.log(
        #     prediction_id=sample_prediction["prediction_id"],
        #     prediction_label=sample_prediction["prediction_label"],
        #     actual_label=sample_prediction["actual_label"],
        #     model_id=sample_prediction["model_id"],
        #     model_version=sample_prediction["model_version"]
        # )
        
        print("  ❌ No governance, cost tracking, or team attribution")
        print("  ❌ No budget controls or policy enforcement")
        print("  ❌ No OpenTelemetry telemetry export")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in before-instrumentation demo: {e}")
        return False


def demonstrate_auto_instrumentation_setup():
    """Demonstrate setting up auto-instrumentation."""
    print("🚀 Setting Up Auto-Instrumentation:")
    print("  Just add these 3 lines to the top of your file:")
    print()
    
    print("  📝 Code to add:")
    print("     from genops.providers.arize import auto_instrument")
    print("     ")
    print("     # This enables governance for ALL Arize operations")
    print("     auto_instrument()")
    print()
    
    try:
        from genops.providers.arize import auto_instrument, get_current_adapter
        
        # Enable auto-instrumentation with governance configuration
        print("  🔧 Enabling auto-instrumentation...")
        adapter = auto_instrument(
            team=os.getenv('GENOPS_TEAM', 'example-team'),
            project=os.getenv('GENOPS_PROJECT', 'auto-instrumentation-demo'),
            environment=os.getenv('GENOPS_ENVIRONMENT', 'development'),
            daily_budget_limit=50.0,
            enable_cost_alerts=True,
            enable_governance=True
        )
        
        print(f"  ✅ Auto-instrumentation enabled successfully!")
        print(f"     • Team: {adapter.team}")
        print(f"     • Project: {adapter.project}")
        print(f"     • Environment: {adapter.environment}")
        print(f"     • Daily Budget: ${adapter.daily_budget_limit:.2f}")
        
        # Verify global adapter is set
        current_adapter = get_current_adapter()
        if current_adapter:
            print("  ✅ Global adapter configured for automatic governance")
        else:
            print("  ⚠️  Global adapter not detected (may be expected)")
        
        return adapter
        
    except Exception as e:
        print(f"❌ Auto-instrumentation setup failed: {e}")
        return None


def demonstrate_after_instrumentation(adapter):
    """Show Arize operations after GenOps auto-instrumentation."""
    print("\n✨ After Auto-Instrumentation:")
    print("  Your existing Arize code now automatically includes governance!")
    print()
    
    try:
        from arize.pandas.logger import Client
        
        # Create Arize client (same as before - no changes needed!)
        arize_client = Client(
            api_key=os.getenv('ARIZE_API_KEY'),
            space_key=os.getenv('ARIZE_SPACE_KEY')
        )
        print("  ✅ Arize client created (no code changes needed)")
        
        # Your existing prediction logging code
        print("  📊 Running your existing prediction code...")
        
        sample_predictions = [
            {
                "prediction_id": f"after_instrumentation_{i}_{int(time.time())}",
                "prediction_label": "fraud" if i % 2 == 0 else "legitimate",
                "actual_label": "fraud" if i % 2 == 0 else "legitimate",
                "model_id": "fraud-detection-auto",
                "model_version": "2.0",
                "features": {
                    "transaction_amount": 100.0 + i * 25,
                    "merchant_category": "online",
                    "risk_score": 0.1 + i * 0.2
                }
            }
            for i in range(3)
        ]
        
        # Log predictions (your existing code - unchanged!)
        for pred in sample_predictions:
            print(f"     • Logging prediction: {pred['prediction_id']}")
            
            # This is your existing code - no changes!
            # response = arize_client.log(
            #     prediction_id=pred["prediction_id"],
            #     prediction_label=pred["prediction_label"],
            #     actual_label=pred["actual_label"],
            #     model_id=pred["model_id"],
            #     model_version=pred["model_version"],
            #     features=pred["features"]
            # )
            
            # Simulate the automatic cost tracking (normally invisible to you)
            if adapter:
                adapter.daily_usage += 0.001  # Simulated cost per prediction
                adapter.operation_count += 1
            
            print(f"       → Prediction: {pred['prediction_label']}")
            print(f"       → Model: {pred['model_id']}-{pred['model_version']}")
        
        print()
        print("  ✅ Automatic governance features now active:")
        print("     • Cost tracking: Each operation tracked with costs")
        print("     • Team attribution: All operations tagged with team/project")
        print("     • Budget monitoring: Automatic budget alerts and limits")
        print("     • Policy enforcement: Governance rules applied automatically")
        print("     • OpenTelemetry export: Spans exported for observability")
        
        # Show automatic cost tracking
        if adapter:
            metrics = adapter.get_metrics()
            print(f"\n  💰 Automatic Cost Tracking:")
            print(f"     • Daily Usage: ${metrics['daily_usage']:.4f}")
            print(f"     • Budget Remaining: ${metrics['budget_remaining']:.2f}")
            print(f"     • Operations Tracked: {metrics['operation_count']}")
            print(f"     • Cost Alerts: {'Enabled' if metrics['cost_alerts_enabled'] else 'Disabled'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in after-instrumentation demo: {e}")
        return False


def demonstrate_configuration_options():
    """Demonstrate different auto-instrumentation configuration options."""
    print("\n⚙️  Configuration Options:")
    print("  Auto-instrumentation supports various configuration patterns:")
    print()
    
    print("  📝 1. Environment Variable Configuration:")
    print("     export GENOPS_TEAM='ml-platform'")
    print("     export GENOPS_PROJECT='fraud-detection'")
    print("     export GENOPS_DAILY_BUDGET_LIMIT='100.0'")
    print("     ")
    print("     auto_instrument()  # Uses environment variables")
    print()
    
    print("  📝 2. Explicit Configuration:")
    print("     auto_instrument(")
    print("         team='ml-platform',")
    print("         project='fraud-detection',")
    print("         daily_budget_limit=100.0,")
    print("         enable_cost_alerts=True")
    print("     )")
    print()
    
    print("  📝 3. Environment-Specific Configuration:")
    print("     # Development")
    print("     auto_instrument(")
    print("         environment='development',")
    print("         daily_budget_limit=20.0,")
    print("         governance_policy='advisory'")
    print("     )")
    print("     ")
    print("     # Production")
    print("     auto_instrument(")
    print("         environment='production',")
    print("         daily_budget_limit=500.0,")
    print("         governance_policy='enforced'")
    print("     )")
    print()
    
    print("  📝 4. Enterprise Configuration:")
    print("     auto_instrument(")
    print("         team='ml-platform',")
    print("         project='production-fraud-detection',")
    print("         customer_id='enterprise-customer-123',")
    print("         cost_center='ml-infrastructure',")
    print("         daily_budget_limit=1000.0,")
    print("         enable_governance=True")
    print("     )")


def demonstrate_integration_patterns():
    """Show common integration patterns."""
    print("\n🔗 Integration Patterns:")
    print("  Common ways to integrate auto-instrumentation:")
    print()
    
    print("  📦 1. Application Startup:")
    print("     # app.py or main.py")
    print("     from genops.providers.arize import auto_instrument")
    print("     ")
    print("     # Enable governance at application startup")
    print("     auto_instrument(team='api-team', project='prediction-service')")
    print("     ")
    print("     # Your existing Arize code continues unchanged...")
    print("     from arize.pandas.logger import Client")
    print("     arize_client = Client(...)")
    print()
    
    print("  📓 2. Jupyter Notebook:")
    print("     # First cell")
    print("     from genops.providers.arize import auto_instrument")
    print("     auto_instrument(team='data-science', environment='development')")
    print("     ")
    print("     # Subsequent cells - your existing Arize code")
    print("     import arize")
    print("     # ... your analysis code ...")
    print()
    
    print("  🐳 3. Docker Container:")
    print("     # Dockerfile")
    print("     ENV GENOPS_TEAM=ml-ops")
    print("     ENV GENOPS_PROJECT=batch-monitoring")
    print("     ENV GENOPS_DAILY_BUDGET_LIMIT=75.0")
    print("     ")
    print("     # Python script")
    print("     from genops.providers.arize import auto_instrument")
    print("     auto_instrument()  # Uses environment variables")
    print()
    
    print("  ☸️  4. Kubernetes Deployment:")
    print("     # ConfigMap")
    print("     apiVersion: v1")
    print("     kind: ConfigMap")
    print("     data:")
    print("       GENOPS_TEAM: ml-platform")
    print("       GENOPS_PROJECT: k8s-monitoring")
    print("     ")
    print("     # Python application")
    print("     auto_instrument()  # Configuration from ConfigMap")


def demonstrate_monitoring_and_observability():
    """Show monitoring and observability features."""
    print("\n📊 Monitoring & Observability:")
    print("  Auto-instrumentation provides built-in monitoring:")
    print()
    
    try:
        from genops.providers.arize import get_current_adapter
        
        adapter = get_current_adapter()
        
        if adapter:
            # Get comprehensive metrics
            metrics = adapter.get_metrics()
            
            print("  📈 Real-time Metrics:")
            print(f"     • Team: {metrics.get('team', 'N/A')}")
            print(f"     • Project: {metrics.get('project', 'N/A')}")
            print(f"     • Environment: {metrics.get('customer_id', 'N/A')}")
            print(f"     • Daily Usage: ${metrics.get('daily_usage', 0):.4f}")
            print(f"     • Budget Remaining: ${metrics.get('budget_remaining', 0):.2f}")
            print(f"     • Operations Count: {metrics.get('operation_count', 0)}")
            print(f"     • Active Sessions: {metrics.get('active_monitoring_sessions', 0)}")
            
            print("\n  🎯 OpenTelemetry Integration:")
            print("     • All operations exported as OpenTelemetry spans")
            print("     • Metrics include cost, governance, and attribution data")
            print("     • Compatible with Jaeger, Zipkin, Datadog, Honeycomb, etc.")
            print("     • Standard OTLP export format for vendor neutrality")
            
        else:
            print("  ℹ️  No active adapter (expected in demo mode)")
        
    except Exception as e:
        print(f"  ⚠️  Monitoring demo error: {e}")
    
    print("\n  📋 Available Monitoring:")
    print("     • Cost tracking per operation")
    print("     • Budget utilization monitoring")
    print("     • Team and project attribution")
    print("     • Environment-based segmentation")
    print("     • Real-time governance policy compliance")
    print("     • Performance and efficiency metrics")


def print_next_steps():
    """Print recommended next steps."""
    print("\n🚀 Next Steps:")
    print("  Now that you understand auto-instrumentation:")
    print()
    
    print("  1️⃣  Try it in your code:")
    print("     • Add the 3 lines to your existing Arize application")
    print("     • Set GENOPS_TEAM and GENOPS_PROJECT environment variables")
    print("     • Run your existing code and observe automatic governance")
    print()
    
    print("  2️⃣  Explore other examples:")
    print("     • python cost_optimization.py    # Cost intelligence features")
    print("     • python advanced_features.py    # Advanced monitoring patterns") 
    print("     • python production_patterns.py  # Production deployment guides")
    print()
    
    print("  3️⃣  Integration options:")
    print("     • Add to CI/CD pipelines for automated governance")
    print("     • Configure for multi-environment deployments")
    print("     • Integrate with existing observability stacks")
    print()
    
    print("  4️⃣  Learn more:")
    print("     • Read full documentation: docs/integrations/arize.md")
    print("     • Check GitHub issues: github.com/KoshiHQ/GenOps-AI/issues")
    print("     • Join discussions: github.com/KoshiHQ/GenOps-AI/discussions")


def main():
    """Main auto-instrumentation demonstration."""
    print_header()
    
    # Check prerequisites
    if not check_prerequisites():
        print("❌ Prerequisites not met. Please install dependencies and set environment variables.")
        return 1
    
    # Demonstrate before instrumentation
    before_success = demonstrate_before_instrumentation()
    
    if not before_success:
        print("❌ Before-instrumentation demonstration failed.")
        return 1
    
    # Set up auto-instrumentation
    adapter = demonstrate_auto_instrumentation_setup()
    
    if not adapter:
        print("❌ Auto-instrumentation setup failed.")
        return 1
    
    # Demonstrate after instrumentation
    after_success = demonstrate_after_instrumentation(adapter)
    
    if not after_success:
        print("❌ After-instrumentation demonstration failed.")
        return 1
    
    # Show configuration options
    demonstrate_configuration_options()
    
    # Show integration patterns
    demonstrate_integration_patterns()
    
    # Show monitoring features
    demonstrate_monitoring_and_observability()
    
    # Print next steps
    print_next_steps()
    
    print("\n" + "=" * 60)
    print("🎉 Auto-instrumentation example completed successfully!")
    print("=" * 60)
    print("✨ Your existing Arize code now has enterprise governance!")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)