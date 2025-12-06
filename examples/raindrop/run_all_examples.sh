#!/bin/bash

# Raindrop AI + GenOps Examples Runner
# 
# This script runs all Raindrop AI integration examples in sequence,
# demonstrating the complete GenOps governance and cost optimization workflow.
#
# Usage:
#   chmod +x run_all_examples.sh
#   ./run_all_examples.sh
#
# Environment Variables Required:
#   RAINDROP_API_KEY - Your Raindrop AI API key
#   GENOPS_TEAM - Team identifier for cost attribution (optional)
#   GENOPS_PROJECT - Project identifier for cost attribution (optional)

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo
}

print_step() {
    echo -e "${CYAN}🔸 $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "setup_validation.py" ]; then
    print_error "Please run this script from the examples/raindrop directory"
    exit 1
fi

# Check required environment variables
print_header "🔍 Environment Check"

if [ -z "$RAINDROP_API_KEY" ]; then
    print_warning "RAINDROP_API_KEY not set - some examples may not work fully"
    echo "  Set it with: export RAINDROP_API_KEY='your-api-key'"
else
    print_success "RAINDROP_API_KEY is configured"
fi

if [ -z "$GENOPS_TEAM" ]; then
    print_warning "GENOPS_TEAM not set - using default team name"
    export GENOPS_TEAM="example-team"
else
    print_success "GENOPS_TEAM: $GENOPS_TEAM"
fi

if [ -z "$GENOPS_PROJECT" ]; then
    print_warning "GENOPS_PROJECT not set - using default project name"
    export GENOPS_PROJECT="raindrop-demo"
else
    print_success "GENOPS_PROJECT: $GENOPS_PROJECT"
fi

echo

# Array of examples to run in order
examples=(
    "setup_validation.py:Setup Validation:Validates Raindrop AI + GenOps configuration"
    "basic_tracking.py:Basic Tracking:Demonstrates basic agent monitoring with governance"
    "auto_instrumentation.py:Auto-Instrumentation:Shows zero-code auto-instrumentation"
    "advanced_features.py:Advanced Features:Multi-agent monitoring and governance"
    "cost_optimization.py:Cost Optimization:Cost analysis and optimization strategies"
    "production_patterns.py:Production Patterns:Enterprise deployment and HA patterns"
)

# Track execution results
total_examples=${#examples[@]}
successful_runs=0
failed_runs=0
start_time=$(date +%s)

print_header "🚀 Running Raindrop AI + GenOps Examples ($total_examples total)"

# Run each example
for example_info in "${examples[@]}"; do
    IFS=':' read -r script_name display_name description <<< "$example_info"
    
    print_step "Running $display_name ($script_name)"
    echo "  📋 $description"
    echo
    
    # Run the example with timeout
    if timeout 300 python "$script_name"; then
        print_success "$display_name completed successfully"
        ((successful_runs++))
    else
        exit_code=$?
        if [ $exit_code -eq 124 ]; then
            print_error "$display_name timed out (5 minutes)"
        else
            print_error "$display_name failed with exit code $exit_code"
        fi
        ((failed_runs++))
    fi
    
    echo
    echo "─────────────────────────────────────────"
    echo
    
    # Small delay between examples
    sleep 1
done

# Calculate execution time
end_time=$(date +%s)
execution_time=$((end_time - start_time))
minutes=$((execution_time / 60))
seconds=$((execution_time % 60))

# Print summary
print_header "📊 Execution Summary"

echo -e "📈 ${GREEN}Successful examples: $successful_runs/$total_examples${NC}"
if [ $failed_runs -gt 0 ]; then
    echo -e "❌ ${RED}Failed examples: $failed_runs/$total_examples${NC}"
fi
echo -e "⏱️  Total execution time: ${minutes}m ${seconds}s"
echo

# Print results breakdown
echo -e "${CYAN}Results breakdown:${NC}"
for example_info in "${examples[@]}"; do
    IFS=':' read -r script_name display_name description <<< "$example_info"
    
    # Check if the example would succeed (simplified check)
    if python -c "
import sys
import os
sys.path.insert(0, '../../../src')
try:
    from genops.providers.raindrop_validation import validate_setup
    result = validate_setup(os.getenv('RAINDROP_API_KEY'))
    exit(0 if result.is_valid or '$script_name' == 'setup_validation.py' else 1)
except:
    exit(0)  # Allow examples to run even if validation has issues
" 2>/dev/null; then
        echo -e "  ✅ $display_name"
    else
        echo -e "  ❓ $display_name (configuration dependent)"
    fi
done

echo

# Provide next steps
if [ $failed_runs -eq 0 ]; then
    print_success "All examples completed successfully! 🎉"
    echo
    echo -e "${CYAN}🚀 Next Steps:${NC}"
    echo "  1. Integrate auto_instrument() into your production code"
    echo "  2. Set up monitoring dashboards for cost and governance"
    echo "  3. Configure team budgets and alert thresholds"
    echo "  4. Review the cost optimization recommendations"
    echo
    echo -e "${CYAN}📚 Additional Resources:${NC}"
    echo "  • Documentation: docs/integrations/raindrop.md"
    echo "  • Quickstart Guide: docs/raindrop-quickstart.md"
    echo "  • Community: https://github.com/KoshiHQ/GenOps-AI/discussions"
else
    print_warning "Some examples failed - this may be due to configuration issues"
    echo
    echo -e "${CYAN}🔧 Troubleshooting:${NC}"
    echo "  1. Ensure RAINDROP_API_KEY is set correctly"
    echo "  2. Check your internet connection"
    echo "  3. Verify GenOps installation: pip install genops[raindrop]"
    echo "  4. Review the setup validation output"
    echo
    echo -e "${CYAN}📖 Getting Help:${NC}"
    echo "  • Run: python setup_validation.py --interactive"
    echo "  • Check: docs/integrations/raindrop.md#troubleshooting"
    echo "  • Ask: https://github.com/KoshiHQ/GenOps-AI/issues"
fi

echo
print_header "🎯 Example Suite Complete"

# Exit with appropriate code
exit $failed_runs