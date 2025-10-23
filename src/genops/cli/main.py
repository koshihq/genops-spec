"""GenOps AI CLI main module."""

import argparse
import json
import logging
import sys
from typing import Optional

from genops import __version__
from genops.core.policy import register_policy, PolicyResult


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def cmd_version(args) -> int:
    """Print version information."""
    print(f"GenOps AI v{__version__}")
    print("OpenTelemetry-native governance for AI")
    return 0


def cmd_policy_register(args) -> int:
    """Register a new governance policy."""
    try:
        # Parse conditions from JSON if provided
        conditions = {}
        if args.conditions:
            conditions = json.loads(args.conditions)
        
        # Map enforcement level string to enum
        enforcement_mapping = {
            "allowed": PolicyResult.ALLOWED,
            "blocked": PolicyResult.BLOCKED,
            "warning": PolicyResult.WARNING,
            "rate_limited": PolicyResult.RATE_LIMITED,
        }
        enforcement_level = enforcement_mapping.get(args.enforcement, PolicyResult.BLOCKED)
        
        # Register the policy
        register_policy(
            name=args.name,
            description=args.description or "",
            enabled=args.enabled,
            enforcement_level=enforcement_level,
            **conditions
        )
        
        print(f"Policy '{args.name}' registered successfully")
        return 0
        
    except json.JSONDecodeError as e:
        print(f"Error parsing conditions JSON: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error registering policy: {e}", file=sys.stderr)
        return 1


def cmd_status(args) -> int:
    """Show GenOps AI status and configuration."""
    print("GenOps AI Status:")
    print(f"Version: {__version__}")
    print("OpenTelemetry Configuration:")
    
    # Check OpenTelemetry setup
    try:
        from opentelemetry import trace
        tracer = trace.get_tracer("genops-cli-test")
        print("✓ OpenTelemetry available")
        
        # Test span creation
        with tracer.start_as_current_span("test-span") as span:
            span.set_attribute("test", True)
        print("✓ Span creation working")
        
    except Exception as e:
        print(f"✗ OpenTelemetry issue: {e}")
    
    # Check provider availability
    print("\nProvider Support:")
    
    try:
        import openai
        print("✓ OpenAI available")
    except ImportError:
        print("✗ OpenAI not available (install with: pip install openai)")
    
    try:
        import anthropic
        print("✓ Anthropic available")
    except ImportError:
        print("✗ Anthropic not available (install with: pip install anthropic)")
    
    return 0


def cmd_demo(args) -> int:
    """Run a simple demo of GenOps AI functionality."""
    print("Running GenOps AI Demo...")
    
    try:
        from genops import track_usage, track
        from genops.core.policy import register_policy, PolicyResult
        
        # Register a demo policy
        register_policy(
            name="demo_cost_limit",
            description="Demo cost limit policy",
            enforcement_level=PolicyResult.WARNING,
            max_cost=1.00
        )
        
        print("✓ Registered demo policy")
        
        # Demo decorator usage
        @track_usage(
            operation_name="demo_operation",
            team="demo-team",
            project="genops-demo"
        )
        def demo_function():
            return "Hello from GenOps AI!"
        
        result = demo_function()
        print(f"✓ Demo function executed: {result}")
        
        # Demo context manager usage
        with track(
            operation_name="demo_context",
            team="demo-team",
            project="genops-demo"
        ) as span:
            span.set_attribute("demo.value", 42)
            print("✓ Context manager demo completed")
        
        print("\nDemo completed successfully!")
        print("Check your OpenTelemetry collector/exporter for the telemetry data.")
        
        return 0
        
    except Exception as e:
        print(f"Demo failed: {e}", file=sys.stderr)
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="genops",
        description="GenOps AI - OpenTelemetry-native governance for AI"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"GenOps AI v{__version__}"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Show version information")
    version_parser.set_defaults(func=cmd_version)
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show GenOps AI status")
    status_parser.set_defaults(func=cmd_status)
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run GenOps AI demo")
    demo_parser.set_defaults(func=cmd_demo)
    
    # Policy commands
    policy_parser = subparsers.add_parser("policy", help="Manage governance policies")
    policy_subparsers = policy_parser.add_subparsers(dest="policy_command")
    
    # Policy register command
    policy_register_parser = policy_subparsers.add_parser("register", help="Register a new policy")
    policy_register_parser.add_argument("name", help="Policy name")
    policy_register_parser.add_argument("--description", help="Policy description")
    policy_register_parser.add_argument(
        "--enforcement",
        choices=["allowed", "blocked", "warning", "rate_limited"],
        default="blocked",
        help="Enforcement level (default: blocked)"
    )
    policy_register_parser.add_argument(
        "--enabled",
        action="store_true",
        default=True,
        help="Enable policy (default: true)"
    )
    policy_register_parser.add_argument(
        "--conditions",
        help="Policy conditions as JSON string"
    )
    policy_register_parser.set_defaults(func=cmd_policy_register)
    
    return parser


def main() -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Handle no command case
    if not hasattr(args, 'func'):
        if args.command == "policy" and not hasattr(args, 'policy_command'):
            print("Error: policy command requires a subcommand", file=sys.stderr)
            return 1
        parser.print_help()
        return 0
    
    # Execute the command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        logging.exception("Unexpected error")
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())