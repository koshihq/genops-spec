"""
Test suite for GenOps Gemini provider integration.

This package contains comprehensive tests for all Gemini components:

- test_gemini_adapter: Core adapter functionality tests
- test_gemini_pricing: Pricing and cost calculation tests  
- test_gemini_cost_aggregator: Cost aggregation and context manager tests
- test_gemini_validation: Setup validation and diagnostics tests
- test_gemini_integration: End-to-end integration tests

Test Coverage:
- Unit tests: ~35 tests per module (140+ total)
- Integration tests: ~17 tests (real workflows)
- Cost calculation tests: ~24 tests (pricing accuracy)
- Validation tests: ~15 tests (setup verification)
- Error handling tests: Comprehensive failure scenarios

Run all tests:
    pytest tests/providers/gemini/ -v

Run specific test module:
    pytest tests/providers/gemini/test_gemini_adapter.py -v
"""
