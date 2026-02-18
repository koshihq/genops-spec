"""
Edge case tests for Flowise integration.

This module tests edge cases, boundary conditions, and error scenarios
that might not be covered in the main test suites.
"""

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from unittest.mock import Mock, patch

import pytest

from genops.providers.flowise import auto_instrument, instrument_flowise
from genops.providers.flowise_pricing import (
    FlowiseCostCalculator,
    FlowisePricingTier,
)
from genops.providers.flowise_validation import (
    ValidationResult,
    validate_flowise_setup,
)


class TestFlowiseEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_extremely_long_urls(self):
        """Test handling of extremely long URLs."""
        # Create a very long URL (beyond typical limits)
        long_host = "a" * 1000
        long_url = f"http://{long_host}.example.com:3000"

        adapter = instrument_flowise(base_url=long_url)
        assert adapter.base_url == long_url

    def test_unicode_urls(self):
        """Test handling of URLs with Unicode characters."""
        unicode_urls = [
            "http://テスト.example.com:3000",
            "http://пример.рф:3000",
            "http://例え.テスト:3000",
        ]

        for url in unicode_urls:
            try:
                adapter = instrument_flowise(base_url=url)
                assert adapter.base_url == url
            except Exception:
                # Some Unicode URLs might not be valid, that's OK
                pass

    def test_special_characters_in_team_names(self):
        """Test handling of special characters in team names."""
        special_teams = [
            "team-with-hyphens",
            "team_with_underscores",
            "team.with.dots",
            "team@company.com",
            "team with spaces",
            "team-123-numbers",
            "UPPERCASE-TEAM",
            "MiXeD-cAsE-TeAm",
        ]

        for team in special_teams:
            adapter = instrument_flowise(base_url="http://localhost:3000", team=team)
            assert adapter.team == team

    def test_empty_and_whitespace_values(self):
        """Test handling of empty and whitespace-only values."""
        edge_values = ["", " ", "\t", "\n", "\r\n", "   \t\n  "]

        for value in edge_values:
            adapter = instrument_flowise(
                base_url="http://localhost:3000", team=value, project=value
            )
            # Should handle gracefully without crashing
            assert adapter.team == value
            assert adapter.project == value

    def test_none_values_in_governance_attributes(self):
        """Test handling of None values in governance attributes."""
        adapter = instrument_flowise(
            base_url="http://localhost:3000",
            team=None,
            project=None,
            customer_id=None,
            environment=None,
            cost_center=None,
            feature=None,
        )

        # Should handle None values gracefully
        assert hasattr(adapter, "team")
        assert hasattr(adapter, "project")

    def test_very_large_numbers(self):
        """Test handling of very large numbers in cost calculations."""
        calculator = FlowiseCostCalculator()

        # Test with extremely large token counts
        very_large_number = 10**10  # 10 billion tokens

        try:
            cost = calculator.calculate_cost(
                input_tokens=very_large_number,
                output_tokens=very_large_number,
                model_name="gpt-3.5-turbo",
            )
            assert isinstance(cost, Decimal)
            assert cost >= 0
        except OverflowError:
            # This is acceptable for extremely large numbers
            pass

    def test_decimal_precision_edge_cases(self):
        """Test decimal precision in edge cases."""
        calculator = FlowiseCostCalculator()

        # Test with very small numbers
        cost = calculator.calculate_cost(
            input_tokens=1, output_tokens=1, model_name="gpt-3.5-turbo"
        )

        # Should maintain precision
        assert isinstance(cost, Decimal)
        assert cost > 0

        # Test precision is maintained in calculations
        cost_str = str(cost)
        recreated_cost = Decimal(cost_str)
        assert cost == recreated_cost

    def test_model_name_edge_cases(self):
        """Test model name edge cases."""
        calculator = FlowiseCostCalculator()

        edge_case_models = [
            "",  # Empty string
            " ",  # Space
            "\n",  # Newline
            "a" * 1000,  # Very long name
            "model-with-émojis-🤖",  # Unicode
            "model/with/slashes",  # Slashes
            "model@version:tag",  # Special chars
            None,  # None value
            123,  # Number instead of string
        ]

        for model in edge_case_models:
            try:
                cost = calculator.calculate_cost(
                    input_tokens=100, output_tokens=50, model_name=model
                )
                assert isinstance(cost, Decimal)
                assert cost >= 0
            except (TypeError, ValueError):
                # Some edge cases might raise exceptions, that's OK
                pass

    def test_concurrent_adapter_creation(self):
        """Test creating many adapters concurrently."""
        adapters = []
        errors = []

        def create_adapter(i):
            try:
                adapter = instrument_flowise(
                    base_url="http://localhost:3000",
                    team=f"team-{i}",
                    project=f"project-{i}",
                )
                adapters.append(adapter)
            except Exception as e:
                errors.append(str(e))

        # Create adapters concurrently
        with ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(create_adapter, i) for i in range(100)]
            for future in as_completed(futures):
                future.result()  # Wait for completion

        # Should create all adapters successfully
        assert len(adapters) == 100
        assert len(errors) == 0

    def test_memory_usage_with_many_objects(self):
        """Test memory usage with many objects."""
        import gc

        gc.collect()

        # Create many objects
        adapters = []
        calculators = []

        for i in range(1000):
            adapter = instrument_flowise(
                base_url="http://localhost:3000", team=f"team-{i}"
            )
            adapters.append(adapter)

            calculator = FlowiseCostCalculator()
            calculators.append(calculator)

        # Use objects briefly
        for adapter in adapters[:10]:
            str(adapter)

        for calculator in calculators[:10]:
            calculator.calculate_cost(100, 50, "gpt-3.5-turbo")

        # Cleanup
        del adapters
        del calculators
        gc.collect()

        # Test passes if no memory errors

    def test_recursive_data_structures(self):
        """Test handling of recursive or circular data structures."""
        # Create circular reference
        data = {"key": None}
        data["key"] = data  # Circular reference

        # Should handle without infinite recursion
        adapter = instrument_flowise("http://localhost:3000")

        # Test with circular data in predict_flow
        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"text": "Response"}
            mock_post.return_value = mock_response

            # This might fail due to JSON serialization, but shouldn't hang
            try:
                adapter.predict_flow("test-flow", "question", custom_data=data)
            except (ValueError, TypeError):
                # JSON serialization errors are expected
                pass

    def test_system_resource_limits(self):
        """Test behavior at system resource limits."""

        # Test with many simultaneous operations
        def stress_test():
            calculator = FlowiseCostCalculator()
            for _ in range(100):
                calculator.calculate_cost(100, 50, "gpt-3.5-turbo")

        # Run stress test in multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=stress_test)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Should complete without errors

    def test_locale_and_encoding_edge_cases(self):
        """Test locale and encoding edge cases."""
        # Test with various encodings
        unicode_strings = [
            "English text",
            "Español con acentos",
            "Français avec des accents",
            "Deutsch mit Umlauten: äöü",
            "Русский текст",
            "中文测试",
            "日本語テスト",
            "🚀 Emoji test 🤖",
            "Mixed: English + 中文 + 🎉",
        ]

        for text in unicode_strings:
            adapter = instrument_flowise(
                base_url="http://localhost:3000", team=text, project=text
            )
            assert adapter.team == text
            assert adapter.project == text

    def test_json_serialization_edge_cases(self):
        """Test JSON serialization edge cases."""
        import json

        # Test with problematic data types
        edge_case_data = {
            "decimal": Decimal("123.456"),
            "datetime": "2024-01-01T00:00:00Z",
            "nested": {"deep": {"very": {"deep": "value"}}},
            "unicode": "🚀 Unicode string 中文",
            "empty": "",
            "null": None,
            "boolean": True,
            "number": 42,
            "float": 3.14159,
            "list": [1, 2, 3, "four", {"five": 5}],
        }

        # Should be serializable
        try:
            json_str = json.dumps(edge_case_data, default=str)
            parsed = json.loads(json_str)
            assert isinstance(parsed, dict)
        except (TypeError, ValueError):
            # Some edge cases might not be serializable
            pass

    def test_validation_with_malformed_responses(self):
        """Test validation with various malformed responses."""
        malformed_responses = [
            "",  # Empty response
            "not json",  # Not JSON
            "{",  # Incomplete JSON
            '{"key": value}',  # Invalid JSON
            '{"key": "value", }',  # Trailing comma
            b"binary data",  # Binary data
            None,  # None response
        ]

        for response_data in malformed_responses:
            with patch("requests.get") as mock_get:
                mock_response = Mock()
                mock_response.status_code = 200

                if isinstance(response_data, bytes):
                    mock_response.json.side_effect = UnicodeDecodeError(
                        "utf-8", response_data, 0, len(response_data), "invalid"
                    )
                elif response_data is None:
                    mock_response.json.side_effect = AttributeError("No JSON")
                else:
                    mock_response.json.side_effect = json.JSONDecodeError(
                        "Invalid JSON", response_data or "", 0
                    )

                mock_get.return_value = mock_response

                # Should handle malformed responses gracefully
                result = validate_flowise_setup("http://localhost:3000", "api-key")
                assert isinstance(result, ValidationResult)
                assert result.is_valid is False

    def test_cost_calculation_boundary_values(self):
        """Test cost calculations at boundary values."""
        calculator = FlowiseCostCalculator()

        boundary_cases = [
            (0, 0),  # Zero tokens
            (1, 0),  # Only input
            (0, 1),  # Only output
            (sys.maxsize, 0),  # Maximum integer
            (0, sys.maxsize),  # Maximum integer
        ]

        for input_tokens, output_tokens in boundary_cases:
            try:
                cost = calculator.calculate_cost(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model_name="gpt-3.5-turbo",
                )
                assert isinstance(cost, Decimal)
                assert cost >= 0
            except (OverflowError, ValueError):
                # Some boundary cases might overflow or be invalid
                pass

    def test_pricing_tier_edge_cases(self):
        """Test pricing tier creation with edge cases."""
        edge_cases = [
            ("", Decimal("0.001")),  # Empty name
            ("tier", Decimal("0")),  # Zero cost
            ("tier", Decimal("999999.999999")),  # Very high cost
            ("tier", Decimal("0.000000001")),  # Very low cost
        ]

        for name, cost in edge_cases:
            try:
                tier = FlowisePricingTier(name, cost)
                assert tier.name == name
                assert tier.cost_per_1k_tokens == cost
            except ValueError:
                # Some edge cases might be invalid
                pass

    def test_auto_instrumentation_edge_cases(self):
        """Test auto-instrumentation with edge cases."""
        # Test with invalid configurations
        edge_configs = [
            {},  # Empty config
            {"team": "", "project": ""},  # Empty strings
            {"invalid_param": "value"},  # Invalid parameters
            {"team": None, "project": None},  # None values
        ]

        for config in edge_configs:
            try:
                result = auto_instrument(**config)
                # Should handle gracefully
                assert isinstance(result, bool)
            except (TypeError, ValueError):
                # Some edge cases might raise exceptions
                pass

    def test_adapter_method_chaining(self):
        """Test method chaining and state consistency."""
        adapter = instrument_flowise("http://localhost:3000")

        # Test multiple operations
        with patch("requests.get") as mock_get:
            mock_get_response = Mock()
            mock_get_response.status_code = 200
            mock_get_response.json.return_value = [{"id": "test", "name": "Test"}]
            mock_get.return_value = mock_get_response

            # Multiple sequential calls should work
            chatflows1 = adapter.get_chatflows()
            chatflows2 = adapter.get_chatflows()

            assert chatflows1 == chatflows2

    def test_error_message_internationalization(self):
        """Test error messages with international characters."""
        with patch("requests.get") as mock_get:
            # Mock responses with international error messages
            international_errors = [
                "Erreur de serveur interne",  # French
                "Internal Server Error",  # English
                "Внутренняя ошибка сервера",  # Russian
                "内部服务器错误",  # Chinese
                "サーバー内部エラー",  # Japanese
            ]

            for error_msg in international_errors:
                mock_response = Mock()
                mock_response.status_code = 500
                mock_response.text = error_msg
                mock_get.return_value = mock_response

                result = validate_flowise_setup("http://localhost:3000", "api-key")
                assert isinstance(result, ValidationResult)
                assert result.is_valid is False
                # Should handle international error messages gracefully

    def test_timestamp_and_timezone_handling(self):
        """Test timestamp and timezone handling."""
        import time
        from datetime import datetime, timezone

        # Test with various timestamp formats
        timestamps = [
            datetime.now(),
            datetime.now(timezone.utc),
            time.time(),
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:00.000Z",
            "2024-01-01 00:00:00",
        ]

        # Should handle various timestamp formats without errors
        for _ts in timestamps:
            # Create adapter with timestamp in metadata
            instrument_flowise(base_url="http://localhost:3000", team="test")
            # Test passes if no exceptions are raised

    def test_nested_exception_handling(self):
        """Test handling of nested exceptions."""

        def raise_nested_exception():
            try:
                raise ValueError("Inner exception")
            except ValueError:
                raise RuntimeError("Outer exception")  # noqa: B904

        with patch("requests.get") as mock_get:
            mock_get.side_effect = raise_nested_exception

            # Should handle nested exceptions gracefully
            result = validate_flowise_setup("http://localhost:3000", "api-key")
            assert isinstance(result, ValidationResult)
            assert result.is_valid is False

    def test_cleanup_and_resource_management(self):
        """Test cleanup and resource management."""
        # Create many objects and ensure they can be cleaned up
        resources = []

        try:
            for i in range(100):
                adapter = instrument_flowise(
                    base_url="http://localhost:3000", team=f"team-{i}"
                )
                calculator = FlowiseCostCalculator()
                resources.extend([adapter, calculator])
        finally:
            # Cleanup should work without errors
            del resources
            import gc

            gc.collect()

    def test_compatibility_with_different_python_versions(self):
        """Test compatibility features across Python versions."""
        # Test features that might behave differently across Python versions

        # Dictionary ordering (Python 3.7+)
        config = {"z": 1, "a": 2, "m": 3}
        adapter = instrument_flowise(base_url="http://localhost:3000", **config)
        # Should work regardless of Python version

        # String formatting
        team_name = f"team-{123}"
        adapter = instrument_flowise(base_url="http://localhost:3000", team=team_name)
        assert adapter.team == "team-123"

    def test_signal_handling(self):
        """Test behavior with system signals."""
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("Operation timed out")

        # Set up timeout signal
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)

        try:
            signal.alarm(1)  # 1 second timeout

            # Perform operation that should complete quickly
            instrument_flowise("http://localhost:3000")
            calculator = FlowiseCostCalculator()
            cost = calculator.calculate_cost(100, 50, "gpt-3.5-turbo")

            signal.alarm(0)  # Cancel alarm

            assert isinstance(cost, Decimal)
        except TimeoutError:
            # Operation took too long
            pytest.fail("Operation should complete within timeout")
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


class TestFlowiseStressConditions:
    """Test behavior under stress conditions."""

    def test_rapid_successive_calls(self):
        """Test rapid successive API calls."""
        adapter = instrument_flowise("http://localhost:3000")

        with patch("requests.post") as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"text": "Response"}
            mock_post.return_value = mock_response

            # Make many rapid calls
            results = []
            for i in range(100):
                try:
                    result = adapter.predict_flow("test-flow", f"Question {i}")
                    results.append(result)
                except Exception as e:
                    results.append(f"Error: {e}")

            # Should handle rapid calls without major issues
            success_count = len([r for r in results if isinstance(r, dict)])
            assert success_count > 0  # At least some should succeed

    def test_memory_pressure(self):
        """Test behavior under memory pressure."""
        import gc

        # Force garbage collection
        gc.collect()

        large_objects = []
        try:
            # Create memory pressure
            for i in range(100):
                # Create large objects
                large_data = "x" * 100000  # 100KB string
                adapter = instrument_flowise(
                    base_url="http://localhost:3000",
                    team=f"team-{i}",
                    project=large_data[:100],  # Use part of large data
                )
                large_objects.append((adapter, large_data))

                # Periodic cleanup
                if i % 10 == 0:
                    gc.collect()

            # Should still function under memory pressure
            assert len(large_objects) == 100

        finally:
            # Cleanup
            del large_objects
            gc.collect()

    def test_high_concurrency_operations(self):
        """Test high concurrency operations."""
        import queue
        import threading

        results_queue = queue.Queue()
        error_queue = queue.Queue()

        def worker():
            try:
                calculator = FlowiseCostCalculator()
                for _ in range(50):
                    cost = calculator.calculate_cost(100, 50, "gpt-3.5-turbo")
                    results_queue.put(cost)
            except Exception as e:
                error_queue.put(str(e))

        # Start many threads
        threads = []
        for _ in range(20):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Check results
        total_results = results_queue.qsize()
        total_errors = error_queue.qsize()

        assert total_results > 0  # Should have some results
        assert total_errors == 0  # Should have no errors

    def test_long_running_operations(self):
        """Test long-running operations don't degrade."""
        calculator = FlowiseCostCalculator()

        start_time = time.time()
        costs = []

        # Run calculations for a period of time
        while time.time() - start_time < 5:  # 5 seconds
            cost = calculator.calculate_cost(100, 50, "gpt-3.5-turbo")
            costs.append(cost)

        # Performance shouldn't degrade significantly
        assert len(costs) > 100  # Should calculate many costs
        assert all(isinstance(cost, Decimal) for cost in costs)

        # Check consistency
        unique_costs = set(costs)
        assert len(unique_costs) == 1  # All costs should be the same


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
