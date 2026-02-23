"""Tests for GenOps AI CLI."""

from unittest.mock import MagicMock, patch

import pytest

from genops.cli.main import (
    cmd_demo,
    cmd_init,
    cmd_policy_register,
    cmd_status,
    cmd_version,
    create_parser,
    main,
)


class TestCLICommands:
    """Test individual CLI commands."""

    def test_cmd_version(self, capsys):
        """Test version command."""
        args = MagicMock()
        result = cmd_version(args)

        captured = capsys.readouterr()
        assert "GenOps AI v" in captured.out
        assert "OpenTelemetry-native governance" in captured.out
        assert result == 0

    def test_cmd_status_not_initialized(self, capsys):
        """Test status command when not initialized."""
        args = MagicMock()

        with patch("genops.status") as mock_status:
            mock_status.return_value = {
                "initialized": False,
                "instrumented_providers": [],
                "default_attributes": {},
                "available_providers": {"openai": False, "anthropic": False},
            }

            result = cmd_status(args)

            captured = capsys.readouterr()
            assert "GenOps AI Status:" in captured.out
            assert "✗ Not initialized" in captured.out
            assert "✗ Openai: not available" in captured.out
            assert result == 0

    def test_cmd_status_initialized(self, capsys):
        """Test status command when initialized."""
        args = MagicMock()

        with patch("genops.status") as mock_status:
            mock_status.return_value = {
                "initialized": True,
                "instrumented_providers": ["openai", "anthropic"],
                "default_attributes": {"team": "ai-team", "project": "chatbot"},
                "available_providers": {"openai": True, "anthropic": True},
            }

            result = cmd_status(args)

            captured = capsys.readouterr()
            assert "✓ Initialized" in captured.out
            assert "openai, anthropic" in captured.out
            assert "ai-team" in captured.out
            assert "✓ Openai: available" in captured.out
            assert result == 0

    def test_cmd_init_basic(self, capsys):
        """Test init command with basic parameters."""
        args = MagicMock()
        args.service_name = "test-service"
        args.environment = "testing"
        args.exporter_type = "console"
        args.otlp_endpoint = None
        args.team = "test-team"
        args.project = "test-project"

        mock_instrumentor = MagicMock()
        mock_instrumentor.status.return_value = {
            "instrumented_providers": ["openai"],
            "default_attributes": {"team": "test-team", "project": "test-project"},
        }

        with patch("genops.init") as mock_init:
            mock_init.return_value = mock_instrumentor

            result = cmd_init(args)

            captured = capsys.readouterr()
            assert "Initializing GenOps AI" in captured.out
            assert "✓ GenOps AI initialized successfully!" in captured.out
            assert "openai" in captured.out
            assert "test-service" in captured.out
            assert result == 0

            # Verify init was called with correct arguments
            mock_init.assert_called_once_with(
                service_name="test-service",
                environment="testing",
                exporter_type="console",
                default_team="test-team",
                default_project="test-project",
            )

    def test_cmd_init_failure(self, capsys):
        """Test init command when initialization fails."""
        args = MagicMock()
        args.service_name = "test-service"
        args.environment = None
        args.exporter_type = None
        args.otlp_endpoint = None
        args.team = None
        args.project = None

        with patch("genops.init") as mock_init:
            mock_init.side_effect = Exception("Initialization failed")

            result = cmd_init(args)

            captured = capsys.readouterr()
            assert "Initialization failed" in captured.err
            assert result == 1

    def test_cmd_demo_success(self, capsys):
        """Test demo command successful execution."""
        args = MagicMock()

        with patch("genops.track_usage") as _mock_track_usage:
            with patch("genops.track") as mock_track:
                with patch("genops.core.policy.register_policy"):
                    # Mock the context manager
                    mock_span = MagicMock()
                    mock_track.return_value.__enter__.return_value = mock_span
                    mock_track.return_value.__exit__.return_value = None

                    result = cmd_demo(args)

                    captured = capsys.readouterr()
                    assert "Running GenOps AI Demo..." in captured.out
                    assert "✓ Registered demo policy" in captured.out
                    assert "✓ Demo function executed" in captured.out
                    assert "Demo completed successfully!" in captured.out
                    assert result == 0

    def test_cmd_demo_failure(self, capsys):
        """Test demo command when execution fails."""
        args = MagicMock()

        with patch("genops.core.policy.register_policy") as mock_register:
            mock_register.side_effect = Exception("Demo failed")

            result = cmd_demo(args)

            captured = capsys.readouterr()
            assert "Demo failed" in captured.err
            assert result == 1

    def test_cmd_policy_register_success(self, capsys):
        """Test policy registration command."""
        args = MagicMock()
        args.name = "test_policy"
        args.description = "Test policy description"
        args.enforcement = "blocked"
        args.enabled = True
        args.conditions = '{"max_cost": 10.0}'

        with patch("genops.cli.main.register_policy") as mock_register:
            result = cmd_policy_register(args)

            captured = capsys.readouterr()
            assert "Policy 'test_policy' registered successfully" in captured.out
            assert result == 0

            # Verify register_policy was called correctly
            mock_register.assert_called_once()
            call_kwargs = mock_register.call_args[1]
            assert call_kwargs["name"] == "test_policy"
            assert call_kwargs["description"] == "Test policy description"
            assert call_kwargs["max_cost"] == 10.0

    def test_cmd_policy_register_invalid_json(self, capsys):
        """Test policy registration with invalid JSON conditions."""
        args = MagicMock()
        args.name = "test_policy"
        args.description = "Test policy"
        args.enforcement = "blocked"
        args.enabled = True
        args.conditions = '{"invalid": json}'

        result = cmd_policy_register(args)

        captured = capsys.readouterr()
        assert "Error parsing conditions JSON" in captured.err
        assert result == 1

    def test_cmd_policy_register_failure(self, capsys):
        """Test policy registration when register_policy fails."""
        args = MagicMock()
        args.name = "test_policy"
        args.description = "Test policy"
        args.enforcement = "blocked"
        args.enabled = True
        args.conditions = None

        with patch("genops.cli.main.register_policy") as mock_register:
            mock_register.side_effect = Exception("Registration failed")

            result = cmd_policy_register(args)

            captured = capsys.readouterr()
            assert "Error registering policy" in captured.err
            assert result == 1


class TestCLIParser:
    """Test CLI argument parser."""

    def test_create_parser(self):
        """Test parser creation and basic structure."""
        parser = create_parser()

        assert parser.prog == "genops"
        assert "GenOps AI" in parser.description

        # Test that all expected commands are present
        help_text = parser.format_help()
        assert "version" in help_text
        assert "status" in help_text
        assert "init" in help_text
        assert "demo" in help_text
        assert "policy" in help_text

    def test_parser_version_command(self):
        """Test version command parsing."""
        parser = create_parser()

        args = parser.parse_args(["version"])
        assert args.command == "version"
        assert hasattr(args, "func")

    def test_parser_status_command(self):
        """Test status command parsing."""
        parser = create_parser()

        args = parser.parse_args(["status"])
        assert args.command == "status"

    def test_parser_init_command_basic(self):
        """Test init command parsing with basic arguments."""
        parser = create_parser()

        args = parser.parse_args(["init"])
        assert args.command == "init"
        assert args.exporter_type == "console"  # default value

    def test_parser_init_command_full(self):
        """Test init command parsing with all arguments."""
        parser = create_parser()

        args = parser.parse_args(
            [
                "init",
                "--service-name",
                "my-service",
                "--environment",
                "production",
                "--exporter-type",
                "otlp",
                "--otlp-endpoint",
                "https://api.honeycomb.io",
                "--team",
                "ai-platform",
                "--project",
                "chatbot",
            ]
        )

        assert args.service_name == "my-service"
        assert args.environment == "production"
        assert args.exporter_type == "otlp"
        assert args.otlp_endpoint == "https://api.honeycomb.io"
        assert args.team == "ai-platform"
        assert args.project == "chatbot"

    def test_parser_policy_register_command(self):
        """Test policy register command parsing."""
        parser = create_parser()

        args = parser.parse_args(
            [
                "policy",
                "register",
                "cost_limit",
                "--description",
                "Cost limit policy",
                "--enforcement",
                "warning",
                "--conditions",
                '{"max_cost": 5.0}',
            ]
        )

        assert args.command == "policy"
        assert args.policy_command == "register"
        assert args.name == "cost_limit"
        assert args.description == "Cost limit policy"
        assert args.enforcement == "warning"
        assert args.conditions == '{"max_cost": 5.0}'

    def test_parser_verbose_flag(self):
        """Test verbose flag parsing."""
        parser = create_parser()

        args = parser.parse_args(["-v", "status"])
        assert args.verbose is True

        args = parser.parse_args(["--verbose", "status"])
        assert args.verbose is True

        args = parser.parse_args(["status"])
        assert args.verbose is False


class TestMainFunction:
    """Test main CLI entry point."""

    def test_main_no_args(self, capsys):
        """Test main function with no arguments shows help."""
        with patch("sys.argv", ["genops"]):
            result = main()

            captured = capsys.readouterr()
            assert "usage:" in captured.out
            assert result == 0

    def test_main_version_command(self, capsys):
        """Test main function with version command."""
        with patch("sys.argv", ["genops", "version"]):
            result = main()

            captured = capsys.readouterr()
            assert "GenOps AI v" in captured.out
            assert result == 0

    def test_main_invalid_command(self, capsys):
        """Test main function with invalid command."""
        with patch("sys.argv", ["genops", "invalid-command"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

            # argparse exits with code 2 for invalid arguments
            assert exc_info.value.code == 2

    def test_main_keyboard_interrupt(self, capsys):
        """Test main function handles keyboard interrupt."""

        def mock_cmd_that_raises_keyboard_interrupt(args):
            raise KeyboardInterrupt()

        with patch("sys.argv", ["genops", "status"]):
            with patch(
                "genops.cli.main.cmd_status", mock_cmd_that_raises_keyboard_interrupt
            ):
                result = main()

                captured = capsys.readouterr()
                assert "Interrupted by user" in captured.err
                assert result == 130

    def test_main_unexpected_exception(self, capsys):
        """Test main function handles unexpected exceptions."""

        def mock_cmd_that_raises_exception(args):
            raise Exception("Unexpected error")

        with patch("sys.argv", ["genops", "status"]):
            with patch("genops.cli.main.cmd_status", mock_cmd_that_raises_exception):
                result = main()

                captured = capsys.readouterr()
                assert "Error: Unexpected error" in captured.err
                assert result == 1

    def test_main_policy_no_subcommand(self, capsys):
        """Test main function with policy command but no subcommand shows help."""
        with patch("sys.argv", ["genops", "policy"]):
            result = main()

            captured = capsys.readouterr()
            assert "usage:" in captured.out
            assert result == 0

    def test_main_with_verbose_logging(self, capsys):
        """Test main function enables verbose logging."""
        with patch("sys.argv", ["genops", "-v", "version"]):
            with patch("genops.cli.main.setup_logging") as mock_setup_logging:
                result = main()

                # Verify verbose logging was enabled
                mock_setup_logging.assert_called_once_with(True)
                assert result == 0


class TestCLIIntegration:
    """Integration tests for CLI functionality."""

    def test_full_cli_workflow(self, capsys):
        """Test complete CLI workflow: init -> status -> demo."""
        mock_instrumentor = MagicMock()
        mock_instrumentor.status.return_value = {
            "initialized": True,
            "instrumented_providers": ["openai"],
            "default_attributes": {"team": "test-team"},
            "available_providers": {"openai": True, "anthropic": False},
        }

        with patch("genops.init") as mock_init:
            with patch("genops.status") as mock_status:
                with patch("genops.core.policy.register_policy"):
                    with patch("genops.track_usage") as _mock_track_usage:
                        with patch("genops.track") as mock_track:
                            mock_init.return_value = mock_instrumentor
                            mock_status.return_value = (
                                mock_instrumentor.status.return_value
                            )

                            # Test init command
                            with patch(
                                "sys.argv", ["genops", "init", "--team", "test-team"]
                            ):
                                result = main()
                                assert result == 0

                            # Test status command
                            with patch("sys.argv", ["genops", "status"]):
                                result = main()
                                assert result == 0

                            # Test demo command
                            mock_track.return_value.__enter__.return_value = MagicMock()
                            mock_track.return_value.__exit__.return_value = None

                            with patch("sys.argv", ["genops", "demo"]):
                                result = main()
                                assert result == 0
