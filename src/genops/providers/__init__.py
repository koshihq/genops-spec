"""Provider adapters for GenOps AI governance."""

# Lazy imports to avoid optional dependency errors during package installation
def __getattr__(name):
    """Lazy import to avoid optional dependency errors."""
    if name in ["instrument_openai", "patch_openai", "unpatch_openai"]:
        from genops.providers.openai import instrument_openai, patch_openai, unpatch_openai
        if name == "instrument_openai":
            return instrument_openai
        elif name == "patch_openai":
            return patch_openai
        elif name == "unpatch_openai":
            return unpatch_openai
    
    elif name in ["instrument_anthropic", "patch_anthropic", "unpatch_anthropic"]:
        from genops.providers.anthropic import instrument_anthropic, patch_anthropic, unpatch_anthropic
        if name == "instrument_anthropic":
            return instrument_anthropic
        elif name == "patch_anthropic":
            return patch_anthropic
        elif name == "unpatch_anthropic":
            return unpatch_anthropic
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "instrument_openai",
    "patch_openai", 
    "unpatch_openai",
    "instrument_anthropic",
    "patch_anthropic",
    "unpatch_anthropic",
]
