#!/bin/bash

# SkyRouter + GenOps Examples Runner
# 
# This script runs all SkyRouter examples in sequence with proper error handling
# and progress reporting. Perfect for validating your setup and exploring all
# multi-model routing capabilities.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Script metadata
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="$SCRIPT_DIR"
START_TIME=$(date +%s)

echo -e "${PURPLE}🚀 SkyRouter + GenOps Examples Runner${NC}"
echo -e "${PURPLE}===========================================${NC}"
echo ""

# Function to print colored output
print_header() {
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}$(printf '=%.0s' $(seq 1 ${#1}))${NC}"
}

print_step() {
    echo -e "${CYAN}📋 $1${NC}"
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

# Function to check prerequisites
check_prerequisites() {
    print_step "Checking prerequisites..."
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        print_error "Python is not installed or not in PATH"
        exit 1
    fi
    
    # Use python3 if available, otherwise python
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    else
        PYTHON_CMD="python"
    fi
    
    # Check Python version
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
    print_success "Found $PYTHON_VERSION"
    
    # Check if GenOps is installed
    if ! $PYTHON_CMD -c "import genops" 2>/dev/null; then
        print_error "GenOps is not installed"
        echo "Install with: pip install genops[skyrouter]"
        exit 1
    fi
    print_success "GenOps is installed"
    
    # Check environment variables
    if [[ -z "$SKYROUTER_API_KEY" ]]; then
        print_warning "SKYROUTER_API_KEY not set"
        echo "Set with: export SKYROUTER_API_KEY='your-api-key'"
        echo "Note: Examples will use mock data without a real API key"
    else
        print_success "SKYROUTER_API_KEY is configured"
    fi
    
    if [[ -z "$GENOPS_TEAM" ]]; then
        print_warning "GENOPS_TEAM not set, using default"
        export GENOPS_TEAM="examples-team"
    else
        print_success "GENOPS_TEAM: $GENOPS_TEAM"
    fi
    
    if [[ -z "$GENOPS_PROJECT" ]]; then
        print_warning "GENOPS_PROJECT not set, using default"
        export GENOPS_PROJECT="skyrouter-examples"
    else
        print_success "GENOPS_PROJECT: $GENOPS_PROJECT"
    fi
    
    echo ""
}

# Function to run a single example
run_example() {
    local example_file="$1"
    local example_name="$2"
    local description="$3"
    local time_estimate="$4"
    
    print_header "$example_name"
    echo "📝 Description: $description"
    echo "⏱️  Estimated time: $time_estimate"
    echo ""
    
    if [[ ! -f "$EXAMPLES_DIR/$example_file" ]]; then
        print_error "Example file $example_file not found"
        return 1
    fi
    
    print_step "Running $example_file..."
    echo ""
    
    # Run the example with timeout
    local start_time=$(date +%s)
    
    if timeout 300 $PYTHON_CMD "$EXAMPLES_DIR/$example_file"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        print_success "$example_name completed in ${duration}s"
        echo ""
        return 0
    else
        local exit_code=$?
        print_error "$example_name failed (exit code: $exit_code)"
        echo ""
        return $exit_code
    fi
}

# Function to show example menu
show_menu() {
    echo -e "${CYAN}📋 Available Examples:${NC}"
    echo "1. Setup Validation (2 min) - Validate configuration"
    echo "2. Basic Routing (5 min) - Multi-model routing fundamentals"
    echo "3. Auto-Instrumentation (3 min) - Zero-code integration"
    echo "4. Route Optimization (15 min) - Advanced optimization"
    echo "5. Agent Workflows (20 min) - Multi-agent patterns"
    echo "6. Enterprise Patterns (30 min) - Production deployment"
    echo "7. Run All Examples (75 min) - Complete walkthrough"
    echo "8. Exit"
    echo ""
    echo -n "Choose an option (1-8): "
}

# Function to run interactive menu
run_interactive() {
    while true; do
        show_menu
        read -r choice
        
        case $choice in
            1)
                run_example "setup_validation.py" "Setup Validation" "Validate SkyRouter + GenOps configuration" "2 minutes"
                ;;
            2)
                run_example "basic_routing.py" "Basic Routing" "Multi-model routing with governance" "5 minutes"
                ;;
            3)
                run_example "auto_instrumentation.py" "Auto-Instrumentation" "Zero-code integration" "3 minutes"
                ;;
            4)
                if [[ -f "$EXAMPLES_DIR/route_optimization.py" ]]; then
                    run_example "route_optimization.py" "Route Optimization" "Advanced routing optimization" "15 minutes"
                else
                    print_warning "route_optimization.py not yet available"
                fi
                ;;
            5)
                if [[ -f "$EXAMPLES_DIR/agent_workflows.py" ]]; then
                    run_example "agent_workflows.py" "Agent Workflows" "Multi-agent routing patterns" "20 minutes"
                else
                    print_warning "agent_workflows.py not yet available"
                fi
                ;;
            6)
                if [[ -f "$EXAMPLES_DIR/enterprise_patterns.py" ]]; then
                    run_example "enterprise_patterns.py" "Enterprise Patterns" "Production deployment patterns" "30 minutes"
                else
                    print_warning "enterprise_patterns.py not yet available"
                fi
                ;;
            7)
                run_all_examples
                ;;
            8)
                echo "👋 Goodbye!"
                exit 0
                ;;
            *)
                print_error "Invalid option. Please choose 1-8."
                ;;
        esac
        
        echo ""
        echo -n "Press Enter to continue..."
        read -r
        clear
    done
}

# Function to run all examples in sequence
run_all_examples() {
    print_header "Running All SkyRouter Examples"
    echo ""
    
    local examples=(
        "setup_validation.py|Setup Validation|Validate configuration|2 min"
        "basic_routing.py|Basic Routing|Multi-model routing fundamentals|5 min"
        "auto_instrumentation.py|Auto-Instrumentation|Zero-code integration|3 min"
    )
    
    # Optional examples (may not exist yet)
    local optional_examples=(
        "route_optimization.py|Route Optimization|Advanced optimization|15 min"
        "agent_workflows.py|Agent Workflows|Multi-agent patterns|20 min"
        "enterprise_patterns.py|Enterprise Patterns|Production deployment|30 min"
    )
    
    local total_examples=0
    local successful_examples=0
    local failed_examples=0
    
    # Run core examples
    for example_info in "${examples[@]}"; do
        IFS='|' read -r file name desc time <<< "$example_info"
        total_examples=$((total_examples + 1))
        
        if run_example "$file" "$name" "$desc" "$time"; then
            successful_examples=$((successful_examples + 1))
        else
            failed_examples=$((failed_examples + 1))
            print_warning "Continuing with next example..."
        fi
        echo ""
    done
    
    # Run optional examples if they exist
    for example_info in "${optional_examples[@]}"; do
        IFS='|' read -r file name desc time <<< "$example_info"
        
        if [[ -f "$EXAMPLES_DIR/$file" ]]; then
            total_examples=$((total_examples + 1))
            
            if run_example "$file" "$name" "$desc" "$time"; then
                successful_examples=$((successful_examples + 1))
            else
                failed_examples=$((failed_examples + 1))
                print_warning "Continuing with next example..."
            fi
            echo ""
        fi
    done
    
    # Show final summary
    local end_time=$(date +%s)
    local total_duration=$((end_time - START_TIME))
    local minutes=$((total_duration / 60))
    local seconds=$((total_duration % 60))
    
    print_header "Examples Summary"
    echo "📊 Total examples: $total_examples"
    echo "✅ Successful: $successful_examples"
    
    if [[ $failed_examples -gt 0 ]]; then
        echo "❌ Failed: $failed_examples"
    fi
    
    echo "⏱️  Total time: ${minutes}m ${seconds}s"
    echo ""
    
    if [[ $failed_examples -eq 0 ]]; then
        print_success "All examples completed successfully! 🎉"
        echo ""
        echo "🚀 Next Steps:"
        echo "• Review docs/skyrouter-quickstart.md for quick integration"
        echo "• Check docs/integrations/skyrouter.md for complete guide"
        echo "• Explore docs/skyrouter-performance-benchmarks.md for optimization"
        echo "• Join discussions at https://github.com/KoshiHQ/GenOps-AI/discussions"
    else
        print_warning "Some examples failed. Check the output above for details."
        echo ""
        echo "🔧 Troubleshooting:"
        echo "• Verify SKYROUTER_API_KEY is set correctly"
        echo "• Ensure internet connectivity for API calls"
        echo "• Check GenOps installation: pip install --upgrade genops[skyrouter]"
    fi
}

# Main execution
main() {
    # Check if running in CI or automated environment
    if [[ -n "$CI" ]] || [[ "$1" == "--non-interactive" ]] || [[ "$1" == "--all" ]]; then
        check_prerequisites
        run_all_examples
        exit $?
    fi
    
    # Show help if requested
    if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
        echo "SkyRouter + GenOps Examples Runner"
        echo ""
        echo "Usage:"
        echo "  $0                    # Interactive mode"
        echo "  $0 --all              # Run all examples non-interactively"
        echo "  $0 --non-interactive  # Run all examples without prompts"
        echo "  $0 --help             # Show this help"
        echo ""
        echo "Environment Variables:"
        echo "  SKYROUTER_API_KEY     # Your SkyRouter API key"
        echo "  GENOPS_TEAM           # Team name for cost attribution"
        echo "  GENOPS_PROJECT        # Project name for cost attribution"
        echo ""
        echo "Examples:"
        echo "  export SKYROUTER_API_KEY='your-key'"
        echo "  export GENOPS_TEAM='ai-platform'"
        echo "  export GENOPS_PROJECT='skyrouter-demo'"
        echo "  $0"
        exit 0
    fi
    
    # Run prerequisites check
    check_prerequisites
    
    # Check if user wants to run specific example
    if [[ -n "$1" ]] && [[ -f "$EXAMPLES_DIR/$1" ]]; then
        run_example "$1" "$(basename "$1" .py)" "Individual example run" "varies"
        exit $?
    fi
    
    # Clear screen and run interactive mode
    clear
    echo -e "${PURPLE}🔀 Welcome to SkyRouter + GenOps Examples!${NC}"
    echo ""
    echo "This interactive runner helps you explore multi-model routing"
    echo "capabilities with comprehensive governance across 150+ models."
    echo ""
    run_interactive
}

# Run main function with all arguments
main "$@"