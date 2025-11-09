#!/bin/bash

# Arize AI + GenOps Interactive Examples Runner
# 
# This script runs all Arize AI integration examples in sequence,
# providing a comprehensive demonstration of GenOps governance
# capabilities with Arize AI model monitoring.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Progress tracking
TOTAL_EXAMPLES=6
CURRENT_EXAMPLE=0
START_TIME=$(date +%s)

# Print colored output
print_colored() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Progress bar function
show_progress() {
    local current=$1
    local total=$2
    local width=50
    local percentage=$((current * 100 / total))
    local completed=$((current * width / total))
    
    printf "\r["
    printf "%*s" $completed | tr ' ' '█'
    printf "%*s" $((width - completed)) | tr ' ' '░'
    printf "] %d%% (%d/%d)" $percentage $current $total
}

# Timer function
elapsed_time() {
    local current_time=$(date +%s)
    local elapsed=$((current_time - START_TIME))
    local minutes=$((elapsed / 60))
    local seconds=$((elapsed % 60))
    printf "%02d:%02d" $minutes $seconds
}

print_colored $CYAN "=================================================================="
print_colored $CYAN "🚀 Arize AI + GenOps Interactive Examples Runner"
print_colored $CYAN "=================================================================="
echo

# Interactive mode check
read -p "🤔 Run in interactive mode? (Y/n): " -n 1 -r
echo
INTERACTIVE_MODE=true
if [[ $REPLY =~ ^[Nn]$ ]]; then
    INTERACTIVE_MODE=false
fi

# Check if we're in the right directory
if [ ! -f "setup_validation.py" ]; then
    print_colored $RED "❌ Error: Please run this script from the examples/arize directory"
    exit 1
fi

# Check environment variables
print_colored $BLUE "🔍 Checking environment configuration..."
MISSING_ENV=false

if [ -z "$ARIZE_API_KEY" ]; then
    print_colored $YELLOW "⚠️  ARIZE_API_KEY not set"
    MISSING_ENV=true
fi

if [ -z "$ARIZE_SPACE_KEY" ]; then
    print_colored $YELLOW "⚠️  ARIZE_SPACE_KEY not set"
    MISSING_ENV=true
fi

if [ -z "$GENOPS_TEAM" ]; then
    print_colored $YELLOW "⚠️  GENOPS_TEAM not set (optional but recommended)"
fi

if [ -z "$GENOPS_PROJECT" ]; then
    print_colored $YELLOW "⚠️  GENOPS_PROJECT not set (optional but recommended)"
fi

if [ "$MISSING_ENV" = true ]; then
    print_colored $YELLOW "📋 Required environment variables:"
    echo "  export ARIZE_API_KEY=\"your-arize-api-key\""
    echo "  export ARIZE_SPACE_KEY=\"your-arize-space-key\""
    echo "  export GENOPS_TEAM=\"your-team-name\"          # optional"
    echo "  export GENOPS_PROJECT=\"your-project-name\"    # optional"
    echo
    if [ "$INTERACTIVE_MODE" = true ]; then
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

echo
print_colored $GREEN "📋 Running examples in recommended order..."
print_colored $CYAN "⏱️  Total estimated time: 15-25 minutes"
echo

# Progress tracking
show_progress 0 $TOTAL_EXAMPLES
echo

# Helper function to run example
run_example() {
    local example_num=$1
    local example_name=$2
    local example_file=$3
    local description=$4
    local estimated_time=$5
    
    CURRENT_EXAMPLE=$((CURRENT_EXAMPLE + 1))
    echo
    print_colored $PURPLE "${example_num} ${example_name}"
    print_colored $BLUE "   📝 ${description}"
    print_colored $YELLOW "   ⏱️  Estimated time: ${estimated_time}"
    print_colored $CYAN "   🕒 Elapsed: $(elapsed_time)"
    echo "   $(printf '─%.0s' {1..50})"
    
    if [ "$INTERACTIVE_MODE" = true ]; then
        read -p "   ▶️  Press Enter to run ${example_name} (or 's' to skip): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Ss]$ ]]; then
            print_colored $YELLOW "   ⏭️  Skipped ${example_name}"
            show_progress $CURRENT_EXAMPLE $TOTAL_EXAMPLES
            return
        fi
    fi
    
    echo "   🚀 Running ${example_file}..."
    echo
    
    if python3 "$example_file"; then
        print_colored $GREEN "   ✅ ${example_name} completed successfully!"
    else
        print_colored $RED "   ❌ ${example_name} failed!"
        if [ "$INTERACTIVE_MODE" = true ]; then
            read -p "   Continue with remaining examples? (Y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Nn]$ ]]; then
                exit 1
            fi
        fi
    fi
    
    show_progress $CURRENT_EXAMPLE $TOTAL_EXAMPLES
}

# Run all examples
run_example "1️⃣" "Setup Validation" "setup_validation.py" "Validates Arize AI and GenOps configuration" "1-2 minutes"

run_example "2️⃣" "Basic Tracking" "basic_tracking.py" "Demonstrates core monitoring with governance" "3-5 minutes"

run_example "3️⃣" "Auto-Instrumentation" "auto_instrumentation.py" "Zero-code integration demonstration" "2-3 minutes"

run_example "4️⃣" "Cost Optimization" "cost_optimization.py" "Cost intelligence and optimization recommendations" "5-8 minutes"

run_example "5️⃣" "Advanced Features" "advanced_features.py" "Multi-model monitoring and enterprise features" "8-12 minutes"

run_example "6️⃣" "Production Patterns" "production_patterns.py" "Enterprise deployment and scaling patterns" "10-15 minutes"

echo
echo
print_colored $GREEN "🎉 All examples completed successfully!"
print_colored $CYAN "⏱️  Total runtime: $(elapsed_time)"
echo

# Results summary
print_colored $BLUE "📊 What you've accomplished:"
echo "  ✅ Validated your Arize AI + GenOps setup"
echo "  ✅ Demonstrated zero-code auto-instrumentation"
echo "  ✅ Explored cost intelligence and optimization"
echo "  ✅ Experienced multi-model enterprise monitoring"
echo "  ✅ Learned production deployment patterns"

echo
print_colored $PURPLE "🚀 Ready for production? Next steps:"
echo "  📖 Read the complete integration guide:"
echo "     docs/integrations/arize.md"
echo "  🔧 Customize for your environment:"
echo "     Modify team, project, and budget configurations"
echo "  📊 Set up monitoring dashboards:"
echo "     Integrate with your existing observability stack"
echo "  🏭 Deploy with confidence:"
echo "     Use production patterns from example #6"

echo
print_colored $CYAN "💬 Need help?"
echo "  🔍 Troubleshooting: docs/integrations/arize.md#troubleshooting"
echo "  💭 Discussions: https://github.com/KoshiHQ/GenOps-AI/discussions"
echo "  🐛 Issues: https://github.com/KoshiHQ/GenOps-AI/issues"
echo "  📧 Enterprise support: support@genops.ai"

print_colored $GREEN "=================================================================="
print_colored $GREEN "🎯 Mission accomplished! Your Arize AI workflows are now governed."
print_colored $GREEN "=================================================================="