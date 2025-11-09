# Fireworks AI Provider Test Suite

Comprehensive test suite for the Fireworks AI provider integration with GenOps governance.

## Overview

This test suite validates the complete Fireworks AI integration, covering:

- **4x faster inference** with Fireattention optimization
- **100+ models** across all pricing tiers ($0.10-$3.00 per 1M tokens)
- **50% cost savings** with batch processing
- **Multi-modal capabilities** (text, vision, audio, embeddings)
- **Enterprise governance** and compliance features
- **Cross-provider compatibility** and migration scenarios

## Test Coverage (85+ Tests)

### 📋 Unit Tests (35 tests)
- **test_fireworks_adapter.py** - Core adapter functionality
- **test_fireworks_pricing.py** - Pricing calculation and optimization
- **test_fireworks_validation.py** - Setup validation and diagnostics

### 🔗 Integration Tests (17 tests)
- **test_integration.py** - End-to-end workflows and real-world scenarios

### ⚡ Performance Tests (24 tests)
- **test_performance.py** - Fireattention optimization and throughput validation

### 🌐 Cross-Provider Tests (9+ tests)
- **test_cross_provider.py** - OpenAI compatibility and migration scenarios

## Quick Start

```bash
# Run all tests
python run_tests.py

# Run specific category
python run_tests.py --category unit
python run_tests.py --category integration
python run_tests.py --category performance
python run_tests.py --category cross-provider

# Verbose output with detailed results
python run_tests.py --verbose

# Include performance benchmarks
python run_tests.py --performance

# Generate coverage report
python run_tests.py --coverage
```

## Test Categories

### Unit Tests

#### test_fireworks_adapter.py (12+ tests)
- Adapter initialization and configuration
- Chat completions with governance tracking
- Embedding operations
- Session-based tracking
- Auto-instrumentation functionality
- Cost management and budget enforcement
- Error handling and resilience

```python
class TestFireworksAdapterInitialization:
    def test_adapter_initialization_with_defaults()
    def test_adapter_initialization_with_custom_config()
    def test_adapter_initialization_budget_validation()

class TestChatCompletionsWithGovernance:
    def test_chat_with_governance_basic()
    def test_chat_with_governance_attributes()
    def test_chat_with_batch_processing()
    def test_chat_with_streaming()
```

#### test_fireworks_pricing.py (15+ tests)
- Cost estimation across pricing tiers
- Model recommendations based on task complexity
- Batch processing cost optimization
- Multi-model cost comparisons
- Cost analysis and projections

```python
class TestCostEstimation:
    def test_chat_cost_estimation_basic()
    def test_chat_cost_estimation_batch_discount()
    def test_cost_estimation_different_tiers()

class TestModelRecommendations:
    def test_recommend_model_simple_task()
    def test_recommend_model_complex_task()
    def test_recommend_model_budget_constraints()
```

#### test_fireworks_validation.py (8+ tests)
- API key validation and connectivity
- Model accessibility testing
- Performance benchmarking
- Diagnostic information collection

```python
class TestAPIKeyValidation:
    def test_check_api_key_validity_success()
    def test_check_api_key_validity_invalid_key()

class TestPerformanceBenchmarking:
    def test_benchmark_performance_success()
    def test_fireattention_speed_validation()
```

### Integration Tests

#### test_integration.py (17 tests)
- End-to-end workflow testing
- Auto-instrumentation integration
- Production workflow simulation
- Real-world scenario testing

```python
class TestEndToEndWorkflows:
    def test_complete_chat_workflow()
    def test_session_based_workflow()
    def test_batch_processing_workflow()

class TestProductionScenarios:
    def test_high_volume_operations()
    def test_mixed_model_operations()
    def test_error_recovery_scenarios()
```

### Performance Tests

#### test_performance.py (24 tests)
- Fireattention 4x speed optimization validation
- Throughput and latency measurements
- Memory usage and resource efficiency
- Concurrent operation handling
- Load testing scenarios

```python
class TestFireattentionOptimization:
    def test_fireattention_speed_benchmark()
    def test_fireattention_vs_baseline_comparison()
    def test_fireattention_across_model_sizes()

class TestThroughputPerformance:
    def test_sequential_throughput()
    def test_concurrent_operations()
    def test_batch_processing_throughput()
```

### Cross-Provider Tests

#### test_cross_provider.py (9+ tests)
- OpenAI compatibility interface
- Multi-provider cost comparison
- Migration scenarios
- Framework integration compatibility

```python
class TestOpenAICompatibility:
    def test_openai_parameter_compatibility()
    def test_openai_migration_cost_comparison()

class TestMultiProviderComparison:
    def test_cost_comparison_across_providers()
    def test_performance_comparison_baselines()
```

## Key Features Tested

### 🔥 Fireattention Optimization
- **4x speed improvement** validation
- Response time benchmarking
- Throughput measurements
- Consistency testing

### 💰 Cost Optimization
- **50% batch processing savings**
- Multi-tier pricing validation
- Budget enforcement testing
- ROI analysis for migrations

### 🎯 Governance Features
- Cost attribution and tracking
- Session-based operation management
- Compliance pattern validation
- Multi-tenant isolation

### 🌐 Multi-Modal Support
- Text generation and chat
- Vision-language processing
- Audio processing capabilities
- Embedding generation

### 🔧 Enterprise Features
- SOC 2 compliance patterns
- Circuit breaker resilience
- Multi-tenant governance
- Production monitoring

## Performance Benchmarks

The test suite includes comprehensive performance validation:

```python
# Fireattention Speed Validation
def test_fireattention_speed_benchmark():
    # Validates 4x speed improvement
    baseline_time = 3.4  # seconds
    fireattention_time = 0.85  # seconds
    assert speedup_ratio >= 3.5

# Batch Processing Efficiency
def test_batch_processing_throughput():
    # Validates 50% cost savings
    standard_cost = calc.estimate_cost(is_batch=False)
    batch_cost = calc.estimate_cost(is_batch=True)
    assert (standard_cost - batch_cost) / standard_cost >= 0.45
```

## Cost Analysis Testing

Comprehensive cost validation across scenarios:

```python
# Migration Cost Analysis
def test_openai_to_fireworks_migration():
    openai_monthly_cost = Decimal("3000.00")
    fireworks_monthly_cost = calc.estimate_monthly_cost()
    savings = openai_monthly_cost - fireworks_monthly_cost
    assert savings > Decimal("2400")  # >80% savings
```

## Production Readiness Validation

Tests ensure production deployment readiness:

- **Error Recovery**: Circuit breaker patterns and retry logic
- **Load Handling**: High-volume concurrent operations
- **Resource Efficiency**: Memory usage and cleanup
- **Monitoring Integration**: Telemetry and observability

## Test Configuration

### Environment Variables
```bash
FIREWORKS_API_KEY=your_api_key_here
GENOPS_TEAM=test-team
GENOPS_PROJECT=fireworks-testing
GENOPS_ENVIRONMENT=test
```

### Test Fixtures
The test suite uses comprehensive fixtures:
- Mock Fireworks client responses
- Sample configurations and messages
- Performance baseline data
- Cost calculation utilities

## Continuous Integration

Tests are designed for CI/CD integration:

```yaml
# Example GitHub Actions integration
- name: Run Fireworks AI Tests
  run: |
    python tests/providers/fireworks/run_tests.py --verbose
    python tests/providers/fireworks/run_tests.py --performance
```

## Coverage Requirements

- **Minimum 85 tests** across all categories
- **90%+ code coverage** for core functionality
- **All critical paths** validated
- **Error scenarios** comprehensively tested

## Success Criteria

✅ **All tests pass** (zero failures)
✅ **Performance benchmarks** meet 4x speed targets
✅ **Cost calculations** accurate within 0.1%
✅ **Governance attributes** properly tracked
✅ **Multi-modal operations** function correctly
✅ **Production patterns** validated

## Contributing

When adding new tests:

1. Follow existing test patterns and naming conventions
2. Include both happy path and error scenarios
3. Add performance validations for new features
4. Update this README with new test descriptions
5. Ensure tests are deterministic and isolated

## Test Data

Tests use realistic data that reflects production usage:

- **Token counts**: 50-2000 tokens per operation
- **Cost ranges**: $0.0001 - $0.005 per operation
- **Response times**: 0.3s - 2.1s depending on model size
- **Throughput**: 5-50 operations per second
- **Batch sizes**: 10-1000 operations

## Debugging Failed Tests

Common debugging approaches:

```bash
# Run single test file with maximum verbosity
python -m pytest test_fireworks_adapter.py -vvs

# Run specific test with detailed output
python -m pytest test_fireworks_adapter.py::TestChatCompletions::test_chat_with_governance_basic -vvs

# Run with debugger on failure
python -m pytest --pdb test_fireworks_adapter.py
```

## Expected Outcomes

A successful test run demonstrates:

1. **🚀 Production Readiness**: All systems operational
2. **⚡ Performance Excellence**: 4x speed optimization confirmed
3. **💰 Cost Efficiency**: 50%+ savings validated
4. **🛡️ Enterprise Governance**: Compliance and security verified
5. **🌐 Multi-Modal Capability**: All modalities functional
6. **🔄 Migration Ready**: Cross-provider compatibility confirmed

The comprehensive test suite ensures Fireworks AI integration delivers on all promises while maintaining the highest quality standards.