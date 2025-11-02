"""Provider adapters for GenOps AI governance."""

# Explicit imports to satisfy CodeQL security requirements
# Import with try/except for optional dependencies
try:
    from genops.providers.openai import (
        instrument_openai,
        patch_openai, 
        unpatch_openai,
    )
    _openai_available = True
except ImportError:
    # Create stub functions for unavailable providers
    def instrument_openai(*args, **kwargs):
        raise ImportError("OpenAI provider not available. Install with: pip install openai")
    
    def patch_openai(*args, **kwargs):
        raise ImportError("OpenAI provider not available. Install with: pip install openai")
    
    def unpatch_openai(*args, **kwargs):
        raise ImportError("OpenAI provider not available. Install with: pip install openai")
    
    _openai_available = False

try:
    from genops.providers.anthropic import (
        instrument_anthropic,
        patch_anthropic,
        unpatch_anthropic,
    )
    _anthropic_available = True
except ImportError:
    # Create stub functions for unavailable providers  
    def instrument_anthropic(*args, **kwargs):
        raise ImportError("Anthropic provider not available. Install with: pip install anthropic")
    
    def patch_anthropic(*args, **kwargs):
        raise ImportError("Anthropic provider not available. Install with: pip install anthropic")
    
    def unpatch_anthropic(*args, **kwargs):
        raise ImportError("Anthropic provider not available. Install with: pip install anthropic")
    
    _anthropic_available = False

try:
    from genops.providers.openrouter import (
        instrument_openrouter,
        patch_openrouter,
        unpatch_openrouter,
    )
    _openrouter_available = True
except ImportError:
    # Create stub functions for unavailable providers  
    def instrument_openrouter(*args, **kwargs):
        raise ImportError("OpenRouter provider not available. Install with: pip install openai")
    
    def patch_openrouter(*args, **kwargs):
        raise ImportError("OpenRouter provider not available. Install with: pip install openai")
    
    def unpatch_openrouter(*args, **kwargs):
        raise ImportError("OpenRouter provider not available. Install with: pip install openai")
    
    _openrouter_available = False

# Explicit __all__ definition with all available exports
__all__ = [
    "instrument_openai",
    "patch_openai", 
    "unpatch_openai",
    "instrument_anthropic",
    "patch_anthropic",
    "unpatch_anthropic",
    "instrument_openrouter",
    "patch_openrouter",
    "unpatch_openrouter",
]
