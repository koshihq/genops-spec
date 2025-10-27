"""GenOps AI CLI main module."""

import argparse
import json
import logging
import sys

from genops import __version__
from genops.core.policy import PolicyResult, register_policy


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
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
        enforcement_level = enforcement_mapping.get(
            args.enforcement, PolicyResult.BLOCKED
        )

        # Register the policy
        register_policy(
            name=args.name,
            description=args.description or "",
            enabled=args.enabled,
            enforcement_level=enforcement_level,
            **conditions,
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

    # Check auto-instrumentation status
    from genops import status

    instrumentation_status = status()

    print(
        f"Auto-instrumentation: {'✓ Initialized' if instrumentation_status['initialized'] else '✗ Not initialized'}"
    )

    if instrumentation_status["initialized"]:
        print(
            f"Instrumented providers: {', '.join(instrumentation_status['instrumented_providers']) or 'None'}"
        )
        if instrumentation_status["default_attributes"]:
            print(f"Default attributes: {instrumentation_status['default_attributes']}")

    print("\nOpenTelemetry Configuration:")

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
    available_providers = instrumentation_status.get("available_providers", {})

    for provider, available in available_providers.items():
        status_icon = "✓" if available else "✗"
        status_text = (
            "available"
            if available
            else f"not available (install with: pip install {provider})"
        )
        print(f"{status_icon} {provider.title()}: {status_text}")

    return 0


def cmd_init(args) -> int:
    """Initialize GenOps AI auto-instrumentation."""
    print("Initializing GenOps AI auto-instrumentation...")

    try:
        from genops import init

        # Build initialization arguments
        init_kwargs = {}

        if args.service_name:
            init_kwargs["service_name"] = args.service_name
        if args.environment:
            init_kwargs["environment"] = args.environment
        if args.exporter_type:
            init_kwargs["exporter_type"] = args.exporter_type
        if args.otlp_endpoint:
            init_kwargs["otlp_endpoint"] = args.otlp_endpoint
        if args.team:
            init_kwargs["default_team"] = args.team
        if args.project:
            init_kwargs["default_project"] = args.project

        # Initialize
        instrumentor = init(**init_kwargs)

        # Show status
        status_info = instrumentor.status()
        print("✓ GenOps AI initialized successfully!")
        print(
            f"  Instrumented providers: {', '.join(status_info['instrumented_providers']) or 'None'}"
        )
        print(f"  Service name: {init_kwargs.get('service_name', 'genops-ai-app')}")

        if status_info["default_attributes"]:
            print(f"  Default attributes: {status_info['default_attributes']}")

        return 0

    except Exception as e:
        print(f"Initialization failed: {e}", file=sys.stderr)
        return 1


def cmd_demo(args) -> int:
    """Run a simple demo of GenOps AI functionality."""
    print("Running GenOps AI Demo...")

    try:
        from genops import track, track_usage
        from genops.core.policy import PolicyResult, register_policy

        # Register a demo policy
        register_policy(
            name="demo_cost_limit",
            description="Demo cost limit policy",
            enforcement_level=PolicyResult.WARNING,
            max_cost=1.00,
        )

        print("✓ Registered demo policy")

        # Demo decorator usage
        @track_usage(
            operation_name="demo_operation", team="demo-team", project="genops-demo"
        )
        def demo_function():
            return "Hello from GenOps AI!"

        result = demo_function()
        print(f"✓ Demo function executed: {result}")

        # Demo context manager usage
        with track(
            operation_name="demo_context", team="demo-team", project="genops-demo"
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
        prog="genops", description="GenOps AI - OpenTelemetry-native governance for AI"
    )

    parser.add_argument(
        "--version", action="version", version=f"GenOps AI v{__version__}"
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Version command
    version_parser = subparsers.add_parser("version", help="Show version information")
    version_parser.set_defaults(func=cmd_version)

    # Status command
    status_parser = subparsers.add_parser("status", help="Show GenOps AI status")
    status_parser.set_defaults(func=cmd_status)

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize auto-instrumentation")
    init_parser.add_argument("--service-name", help="Service name for telemetry")
    init_parser.add_argument("--environment", help="Environment (dev, staging, prod)")
    init_parser.add_argument(
        "--exporter-type",
        choices=["console", "otlp"],
        default="console",
        help="Telemetry exporter type",
    )
    init_parser.add_argument("--otlp-endpoint", help="OTLP endpoint URL")
    init_parser.add_argument("--team", help="Default team attribute")
    init_parser.add_argument("--project", help="Default project attribute")
    init_parser.set_defaults(func=cmd_init)

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run GenOps AI demo")
    demo_parser.set_defaults(func=cmd_demo)

    # Policy commands
    policy_parser = subparsers.add_parser("policy", help="Manage governance policies")
    policy_subparsers = policy_parser.add_subparsers(dest="policy_command")

    # Policy register command
    policy_register_parser = policy_subparsers.add_parser(
        "register", help="Register a new policy"
    )
    policy_register_parser.add_argument("name", help="Policy name")
    policy_register_parser.add_argument("--description", help="Policy description")
    policy_register_parser.add_argument(
        "--enforcement",
        choices=["allowed", "blocked", "warning", "rate_limited"],
        default="blocked",
        help="Enforcement level (default: blocked)",
    )
    policy_register_parser.add_argument(
        "--enabled",
        action="store_true",
        default=True,
        help="Enable policy (default: true)",
    )
    policy_register_parser.add_argument(
        "--conditions", help="Policy conditions as JSON string"
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
    if not hasattr(args, "func"):
        if args.command == "policy" and not hasattr(args, "policy_command"):
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
