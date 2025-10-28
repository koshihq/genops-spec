# Contributing to GenOps AI

Thank you for your interest in contributing to GenOps AI! 🎉 

GenOps AI is building the future of **OpenTelemetry-native AI governance**, and we welcome contributions from developers, DevOps engineers, FinOps practitioners, and anyone passionate about bringing accountability to AI systems.

## 🌟 **Ways to Contribute**

### **Code Contributions**
- 🔌 **Provider Adapters**: Add support for new AI providers (AWS Bedrock, Google Gemini, etc.)
- 🏗️ **OpenTelemetry Processors**: Build OTEL Collector extensions for real-time governance
- 🧪 **Testing**: Improve test coverage and add integration tests
- 🐛 **Bug Fixes**: Fix issues and improve reliability
- ⚡ **Performance**: Optimize telemetry overhead and provider adapters

### **Documentation & Community**
- 📖 **Documentation**: Improve guides, tutorials, and API references
- 🎬 **Examples**: Create real-world usage examples and case studies  
- 🗣️ **Content**: Write blog posts, give talks, create videos
- 💬 **Support**: Help other users in GitHub Discussions and issues

### **Integration & Ecosystem**
- 📊 **Dashboards**: Create pre-built dashboards for observability platforms
- 🔗 **Framework Integrations**: Build LangChain, LlamaIndex, and other framework integrations
- 🏢 **Enterprise Features**: Contribute governance patterns and compliance automation

---

## 🚀 **Getting Started**

### **Development Setup**

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/GenOps-AI.git
   cd GenOps-AI
   ```

2. **Set Up Python Environment**
   ```bash
   # Python 3.9+ required
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install GenOps in development mode with all dependencies
   pip install -e ".[dev,all]"
   ```

3. **Verify Installation**
   ```bash
   # Run quick validation tests
   python test_quick.py
   
   # Run the full test suite
   python run_tests.py
   
   # Test CLI functionality
   genops version
   genops status
   ```

4. **Set Up Pre-commit Hooks** (Optional but Recommended)
   ```bash
   pip install pre-commit
   pre-commit install
   ```

### **Project Structure**

```
GenOps-AI/
├── src/genops/                 # Main package code
│   ├── core/                   # Core telemetry and policy engines
│   ├── providers/              # AI provider adapters (OpenAI, Anthropic, etc.)
│   ├── auto_instrumentation.py # OpenLLMetry-inspired auto-setup
│   └── cli/                    # Command-line interface
├── tests/                      # Comprehensive test suite (3,149+ lines)
│   ├── core/                   # Core functionality tests
│   ├── providers/              # Provider adapter tests
│   ├── integration/            # End-to-end workflow tests
│   └── utils/                  # Test utilities and mocks
├── examples/                   # Usage examples and demos
├── docs/                       # Documentation source
└── run_tests.py               # Comprehensive test runner
```

---

## 🧪 **Development Workflow**

### **Running Tests**

We maintain **high test coverage** with multiple test runners:

```bash
# Quick validation (30 seconds)
python test_quick.py

# Full test suite with coverage
python run_tests.py

# Run specific test categories
python -m pytest tests/core/           # Core functionality
python -m pytest tests/providers/     # Provider adapters  
python -m pytest tests/integration/   # Integration tests

# Run tests with coverage report
python -m pytest tests/ --cov=src/genops --cov-report=html
open htmlcov/index.html  # View coverage report
```

### **Code Quality**

We maintain high code quality standards:

```bash
# Format code
ruff format src/ tests/

# Lint code
ruff check src/ tests/

# Type checking
mypy src/genops/

# All quality checks (included in run_tests.py)
python run_tests.py
```

### **Testing Guidelines**

- **Write tests first** for new features (TDD approach)
- **Mock external services** - all tests should run without API keys
- **Test error scenarios** - include failure cases and edge conditions
- **Maintain test coverage** - aim for 80%+ coverage on new code
- **Use realistic data** - test with actual token counts and cost models

---

## 🎯 **Contribution Guidelines**

### **Pull Request Process**

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Make Your Changes**
   - Write clean, well-documented code
   - Follow existing code patterns and conventions
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Thoroughly**
   ```bash
   # Run full test suite
   python run_tests.py
   
   # Test with different Python versions if possible
   python3.8 run_tests.py
   python3.11 run_tests.py
   ```

4. **Commit with Clear Messages**
   ```bash
   git add .
   git commit -m "Add AWS Bedrock provider adapter with cost model
   
   - Implement BedrockAdapter with full Claude/Titan support
   - Add accurate cost calculations for all Bedrock models
   - Include comprehensive tests with mock responses
   - Update documentation with Bedrock examples
   
   Fixes #123"
   ```

5. **Submit Pull Request**
   - Provide a clear description of changes
   - Link to relevant issues
   - Include testing instructions
   - Add screenshots/examples for UI changes

### **Code Standards**

- **Python Style**: Follow PEP 8, use `ruff` for formatting/linting
- **Type Hints**: Use type annotations for all public APIs
- **Documentation**: Include docstrings for all public functions/classes
- **Error Handling**: Gracefully handle provider failures and missing dependencies
- **Backwards Compatibility**: Maintain compatibility with existing APIs

### **Commit Message Format**

Use clear, descriptive commit messages:

```
Add/Fix/Update: Brief description (50 chars max)

Detailed explanation of what was changed and why.
Include context about the problem being solved.

- Bullet points for specific changes
- Reference issues with "Fixes #123" or "Closes #456"  
- Include breaking changes with "BREAKING CHANGE:"
```

---

## 🔌 **Adding New Provider Adapters**

Provider adapters are one of our most valuable contributions! Here's how to add support for a new AI provider:

### **1. Create Provider Module**

```python
# src/genops/providers/newprovider.py
"""NewProvider adapter for GenOps AI governance."""

import logging
from typing import Any, Optional

from genops.core.telemetry import GenOpsTelemetry

logger = logging.getLogger(__name__)

try:
    from newprovider import NewProviderClient  # Provider's SDK
    HAS_NEWPROVIDER = True
except ImportError:
    HAS_NEWPROVIDER = False
    NewProviderClient = None
    logger.warning("NewProvider not installed. Install with: pip install newprovider")


class GenOpsNewProviderAdapter:
    """NewProvider adapter with automatic governance telemetry."""
    
    # Add pricing model (cost per 1K tokens)
    PRICING = {
        "model-1": {"input": 0.001, "output": 0.002},
        "model-2": {"input": 0.005, "output": 0.010},
    }
    
    def __init__(self, client: Optional[Any] = None, **client_kwargs):
        if not HAS_NEWPROVIDER:
            raise ImportError("NewProvider package not found. Install with: pip install newprovider")
        
        self.client = client or NewProviderClient(**client_kwargs)
        self.telemetry = GenOpsTelemetry()
    
    def generate_text(self, **kwargs) -> Any:
        """Generate text with governance tracking."""
        model = kwargs.get("model", "model-1")
        
        operation_name = "newprovider.generate"
        
        with self.telemetry.trace_operation(
            operation_name=operation_name,
            operation_type="ai.inference",
            provider="newprovider",
            model=model,
            **kwargs  # Include governance attributes
        ) as span:
            try:
                # Call provider API
                response = self.client.generate(**kwargs)
                
                # Extract token usage
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                
                # Calculate cost
                cost = self._calculate_cost(model, input_tokens, output_tokens)
                
                # Record governance telemetry
                span.set_attribute("genops.tokens.input", input_tokens)
                span.set_attribute("genops.tokens.output", output_tokens)
                span.set_attribute("genops.tokens.total", input_tokens + output_tokens)
                span.set_attribute("genops.cost.total", cost)
                span.set_attribute("genops.cost.currency", "USD")
                
                return response
                
            except Exception as e:
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise
    
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on model pricing."""
        pricing = self.PRICING.get(model, self.PRICING["model-1"])
        
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return round(input_cost + output_cost, 6)


# Auto-instrumentation support
def patch_newprovider():
    """Patch NewProvider for auto-instrumentation."""
    if not HAS_NEWPROVIDER:
        return False
    
    # Implement monkey-patching logic here
    # Follow patterns from openai.py and anthropic.py
    return True

def unpatch_newprovider():
    """Remove NewProvider patches."""
    # Implement cleanup logic
    pass
```

### **2. Add Tests**

```python
# tests/providers/test_newprovider.py
"""Tests for NewProvider adapter."""

import pytest
from unittest.mock import MagicMock
from tests.utils.mock_providers import MockNewProviderClient

from genops.providers.newprovider import GenOpsNewProviderAdapter


class TestGenOpsNewProviderAdapter:
    """Test NewProvider adapter with governance tracking."""
    
    def test_generate_text_basic(self, mock_span_recorder):
        """Test basic text generation with governance tracking."""
        mock_client = MockNewProviderClient()
        adapter = GenOpsNewProviderAdapter(client=mock_client)
        
        response = adapter.generate_text(
            model="model-1",
            prompt="Test prompt",
            max_tokens=100
        )
        
        # Verify governance telemetry
        spans = mock_span_recorder.get_finished_spans()
        assert len(spans) == 1
        
        span = spans[0]
        assert span.attributes["genops.provider"] == "newprovider"
        assert span.attributes["genops.model"] == "model-1"
        assert "genops.cost.total" in span.attributes
    
    # Add more comprehensive tests following existing patterns
```

### **3. Update Configuration**

```python
# src/genops/providers/__init__.py
"""Provider adapters for GenOps AI."""

from genops.providers.openai import GenOpsOpenAIAdapter
from genops.providers.anthropic import GenOpsAnthropicAdapter  
from genops.providers.newprovider import GenOpsNewProviderAdapter  # Add this

__all__ = [
    "GenOpsOpenAIAdapter",
    "GenOpsAnthropicAdapter", 
    "GenOpsNewProviderAdapter",  # Add this
]
```

```toml
# pyproject.toml - Add optional dependency
[project.optional-dependencies]
newprovider = ["newprovider>=1.0.0"]
```

### **4. Update Auto-Instrumentation**

```python
# src/genops/auto_instrumentation.py
def _setup_provider_registry(self):
    """Set up the registry of available provider patches."""
    # ... existing providers ...
    
    from genops.providers.newprovider import patch_newprovider, unpatch_newprovider
    
    self.provider_patches['newprovider'] = {
        'patch': patch_newprovider,
        'unpatch': unpatch_newprovider,
        'module': 'newprovider'
    }
```

---

## 📖 **Documentation Contributions**

### **API Documentation**

- Use **clear docstrings** with examples
- Include **type hints** for all parameters
- Document **exceptions** and error scenarios
- Provide **usage examples** for complex features

### **User Guides**

- Write **step-by-step tutorials** for common use cases
- Include **working code examples** that can be copy-pasted
- Cover **integration scenarios** with popular tools
- Address **troubleshooting** and common issues

### **Architecture Documentation**

- Create **ADRs (Architecture Decision Records)** for significant decisions
- Document **semantic conventions** for governance attributes
- Explain **design patterns** and best practices
- Provide **integration guides** for observability platforms

---

## 🏷️ **Issue Labels & Project Board**

We use GitHub labels to organize work:

- **`good first issue`** - Perfect for new contributors
- **`help wanted`** - Community contributions welcome  
- **`provider`** - New provider adapter needed
- **`documentation`** - Documentation improvements
- **`bug`** - Something isn't working
- **`enhancement`** - New feature or improvement
- **`testing`** - Test improvements needed

Check our [Project Board](https://github.com/KoshiHQ/GenOps-AI/projects) for current priorities.

---

## 🤝 **Community Guidelines**

### **Code of Conduct**

We are committed to providing a welcoming and inclusive environment. Please read our [Code of Conduct](CODE_OF_CONDUCT.md).

### **Communication**

- **GitHub Discussions** - For questions, ideas, and general discussion
- **GitHub Issues** - For bug reports and feature requests
- **Pull Requests** - For code contributions with discussion
- **Email** - [hello@genopsai.org](mailto:hello@genopsai.org) for private matters

### **Recognition**

We recognize contributors in several ways:

- **Contributors section** in README
- **Release notes** mention significant contributions
- **Community highlights** in project updates
- **Referral opportunities** for Koshi commercial platform

---

## 🎖️ **Maintainer Guidelines**

### **For Core Maintainers**

- **Review PRs promptly** - Aim for initial feedback within 48 hours
- **Maintain high standards** - Code quality, tests, and documentation
- **Be welcoming** - Help new contributors succeed
- **Communicate decisions** - Use ADRs for architectural changes
- **Coordinate releases** - Follow semantic versioning

### **Release Process**

1. **Update CHANGELOG.md** with all changes
2. **Bump version** in `src/genops/__init__.py` 
3. **Create release tag** following semver (v1.2.3)
4. **Publish to PyPI** via GitHub Actions
5. **Update documentation** if needed
6. **Announce release** in community channels

---

## 🚀 **What's Next?**

Ready to contribute? Here are some great places to start:

### **🔥 Urgent: CI Test Fixes (Great First Issues!)**
- Fix failing integration tests ([View CI Status](https://github.com/KoshiHQ/GenOps-AI/actions))
- Resolve Python 3.11 compatibility issues  
- Improve test stability and reliability
- Debug cancelled test scenarios

### **🌟 Other Good First Issues**
- Add cost models for existing providers
- Improve error messages and documentation  
- Add examples for specific use cases
- Write integration tests for edge cases

### **🔥 High Impact Contributions**
- **AWS Bedrock adapter** - High demand from enterprise users
- **Google Gemini adapter** - Growing market share
- **LangChain integration** - Popular framework integration
- **Grafana dashboard templates** - Pre-built observability dashboards

### **🏗️ Advanced Contributions** 
- **OpenTelemetry Collector processors** - Real-time governance
- **Async provider support** - High-throughput workloads
- **Multi-tenant governance** - SaaS deployment patterns
- **Advanced policy DSL** - Complex governance rules

---

## 📞 **Getting Help**

Stuck? We're here to help!

- 💬 **GitHub Discussions** - [Ask the community](https://github.com/KoshiHQ/GenOps-AI/discussions)
- 🐛 **Issues** - [Report bugs or request features](https://github.com/KoshiHQ/GenOps-AI/issues)
- 📧 **Email** - [hello@genopsai.org](mailto:hello@genopsai.org)
- 📖 **Documentation** - [docs.genopsai.org](https://docs.genopsai.org)

---

**Thank you for helping make AI governance accessible to everyone!** 🙏

Every contribution, no matter how small, helps build a more accountable AI ecosystem. Together, we're creating the standards and tools that will govern the next generation of AI systems.

*Happy coding!* 🚀