#!/bin/bash
"""
Run All Databricks Unity Catalog Examples

This script runs all examples in the recommended order.
Great for testing the complete integration or learning all features.

Usage:
    ./run_all_examples.sh
    bash run_all_examples.sh
"""

set -e  # Exit on any error

echo "🚀 Running All Databricks Unity Catalog Examples"
echo "================================================"
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "❌ Python not found. Please install Python 3.9+ to run these examples."
        exit 1
    fi
    PYTHON_CMD="python"
else
    PYTHON_CMD="python3"
fi

echo "Using Python: $PYTHON_CMD"

# Function to run an example with error handling
run_example() {
    local example_name="$1"
    local description="$2"
    
    echo
    echo "🔄 Running: $example_name"
    echo "   $description"
    echo "   Command: $PYTHON_CMD $example_name"
    echo
    
    if $PYTHON_CMD "$example_name"; then
        echo "✅ $example_name completed successfully"
    else
        echo "❌ $example_name failed"
        echo "💡 Check the error messages above for troubleshooting guidance"
        echo "💡 You can run individual examples separately to debug issues"
        exit 1
    fi
}

# Check that we're in the right directory
if [[ ! -f "setup_validation.py" ]]; then
    echo "❌ Please run this script from the databricks_unity_catalog examples directory"
    echo "   Expected files: setup_validation.py, basic_tracking.py, auto_instrumentation.py"
    exit 1
fi

echo "📋 Running examples in recommended learning order..."

# Level 1: Getting Started (Required)
echo
echo "📚 LEVEL 1: Getting Started"
echo "=========================="

run_example "setup_validation.py" "Validate Databricks Unity Catalog setup (⭐ REQUIRED FIRST)"
run_example "basic_tracking.py" "Basic governance tracking with Unity Catalog"
run_example "auto_instrumentation.py" "Zero-code auto-instrumentation setup"

echo
echo "🎉 All examples completed successfully!"
echo
echo "📚 What you've learned:"
echo "   ✅ How to validate and set up Databricks Unity Catalog with GenOps"
echo "   ✅ How to track governance operations with cost attribution"
echo "   ✅ How to use auto-instrumentation for zero-code integration"
echo
echo "🎯 Next steps:"
echo "   • Review the output above to understand the governance data captured"
echo "   • Try running individual examples again to explore specific features"
echo "   • Read the README.md for more advanced examples and documentation"
echo "   • Apply these patterns to your own Databricks Unity Catalog applications"
echo
echo "📖 Additional resources:"
echo "   • README.md - Complete examples documentation"
echo "   • ../../docs/databricks-unity-catalog-quickstart.md - 5-minute quickstart guide"
echo "   • ../../docs/integrations/databricks-unity-catalog.md - Comprehensive integration guide"
echo
echo "✨ Happy data governing with GenOps + Databricks Unity Catalog!"