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

try:
    from genops.providers.bedrock import (
        instrument_bedrock,
        auto_instrument_bedrock,
        GenOpsBedrockAdapter,
        validate_setup as validate_bedrock_setup,
        print_validation_result as print_bedrock_validation_result,
    )
    _bedrock_available = True
except ImportError:
    # Create stub functions for unavailable providers
    def instrument_bedrock(*args, **kwargs):
        raise ImportError("Bedrock provider not available. Install with: pip install boto3")

    def auto_instrument_bedrock(*args, **kwargs):
        raise ImportError("Bedrock provider not available. Install with: pip install boto3")

    def GenOpsBedrockAdapter(*args, **kwargs):
        raise ImportError("Bedrock provider not available. Install with: pip install boto3")
    
    def validate_bedrock_setup(*args, **kwargs):
        raise ImportError("Bedrock provider not available. Install with: pip install boto3")
    
    def print_bedrock_validation_result(*args, **kwargs):
        raise ImportError("Bedrock provider not available. Install with: pip install boto3")

    _bedrock_available = False

try:
    from genops.providers.helicone import (
        instrument_helicone,
        GenOpsHeliconeAdapter,
        create_helicone_adapter,
    )
    from genops.providers.helicone_validation import (
        validate_setup as validate_helicone_setup,
        print_validation_result as print_helicone_validation_result,
    )
    _helicone_available = True
except ImportError:
    # Create stub functions for unavailable providers
    def instrument_helicone(*args, **kwargs):
        raise ImportError("Helicone provider not available. Install with: pip install 'genops[helicone]'")

    def GenOpsHeliconeAdapter(*args, **kwargs):
        raise ImportError("Helicone provider not available. Install with: pip install 'genops[helicone]'")
    
    def create_helicone_adapter(*args, **kwargs):
        raise ImportError("Helicone provider not available. Install with: pip install 'genops[helicone]'")
    
    def validate_helicone_setup(*args, **kwargs):
        raise ImportError("Helicone provider not available. Install with: pip install 'genops[helicone]'")
    
    def print_helicone_validation_result(*args, **kwargs):
        raise ImportError("Helicone provider not available. Install with: pip install 'genops[helicone]'")

    _helicone_available = False

try:
    from genops.providers.langfuse import (
        instrument_langfuse,
        GenOpsLangfuseAdapter,
        create_langfuse_adapter,
    )
    from genops.providers.langfuse_validation import (
        validate_setup as validate_langfuse_setup,
        print_validation_result as print_langfuse_validation_result,
    )
    _langfuse_available = True
except ImportError:
    # Create stub functions for unavailable providers
    def instrument_langfuse(*args, **kwargs):
        raise ImportError("Langfuse provider not available. Install with: pip install 'genops[langfuse]'")

    def GenOpsLangfuseAdapter(*args, **kwargs):
        raise ImportError("Langfuse provider not available. Install with: pip install 'genops[langfuse]'")
    
    def create_langfuse_adapter(*args, **kwargs):
        raise ImportError("Langfuse provider not available. Install with: pip install 'genops[langfuse]'")
    
    def validate_langfuse_setup(*args, **kwargs):
        raise ImportError("Langfuse provider not available. Install with: pip install 'genops[langfuse]'")
    
    def print_langfuse_validation_result(*args, **kwargs):
        raise ImportError("Langfuse provider not available. Install with: pip install 'genops[langfuse]'")

    _langfuse_available = False

try:
    from genops.providers.arize import (
        instrument_arize,
        auto_instrument as auto_instrument_arize,
        GenOpsArizeAdapter,
    )
    from genops.providers.arize_validation import (
        validate_setup as validate_arize_setup,
        print_validation_result as print_arize_validation_result,
    )
    _arize_available = True
except ImportError:
    # Create stub functions for unavailable providers
    def instrument_arize(*args, **kwargs):
        raise ImportError("Arize provider not available. Install with: pip install 'genops[arize]'")

    def auto_instrument_arize(*args, **kwargs):
        raise ImportError("Arize provider not available. Install with: pip install 'genops[arize]'")

    def GenOpsArizeAdapter(*args, **kwargs):
        raise ImportError("Arize provider not available. Install with: pip install 'genops[arize]'")
    
    def validate_arize_setup(*args, **kwargs):
        raise ImportError("Arize provider not available. Install with: pip install 'genops[arize]'")
    
    def print_arize_validation_result(*args, **kwargs):
        raise ImportError("Arize provider not available. Install with: pip install 'genops[arize]'")

    _arize_available = False

try:
    from genops.providers.dust import (
        instrument_dust,
        GenOpsDustAdapter,
        auto_instrument as auto_instrument_dust,
    )
    from genops.providers.dust_validation import (
        validate_setup as validate_dust_setup,
        print_validation_result as print_dust_validation_result,
    )
    _dust_available = True
except ImportError:
    # Create stub functions for unavailable providers
    def instrument_dust(*args, **kwargs):
        raise ImportError("Dust provider not available. Install with: pip install requests")

    def GenOpsDustAdapter(*args, **kwargs):
        raise ImportError("Dust provider not available. Install with: pip install requests")

    def auto_instrument_dust(*args, **kwargs):
        raise ImportError("Dust provider not available. Install with: pip install requests")
    
    def validate_dust_setup(*args, **kwargs):
        raise ImportError("Dust provider not available. Install with: pip install requests")
    
    def print_dust_validation_result(*args, **kwargs):
        raise ImportError("Dust provider not available. Install with: pip install requests")

    _dust_available = False

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
    "instrument_bedrock",
    "auto_instrument_bedrock",
    "GenOpsBedrockAdapter",
    "validate_bedrock_setup",
    "print_bedrock_validation_result",
    "instrument_helicone",
    "GenOpsHeliconeAdapter",
    "create_helicone_adapter",
    "validate_helicone_setup",
    "print_helicone_validation_result",
    "instrument_langfuse",
    "GenOpsLangfuseAdapter",
    "create_langfuse_adapter",
    "validate_langfuse_setup",
    "print_langfuse_validation_result",
    "instrument_arize",
    "auto_instrument_arize",
    "GenOpsArizeAdapter",
    "validate_arize_setup",
    "print_arize_validation_result",
    "instrument_dust",
    "GenOpsDustAdapter",
    "auto_instrument_dust",
    "validate_dust_setup",
    "print_dust_validation_result",
]
