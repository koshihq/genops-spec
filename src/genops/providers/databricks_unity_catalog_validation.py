"""Main validation module for Databricks Unity Catalog provider."""

from .databricks_unity_catalog.validation import (
    ValidationIssue,
    ValidationResult,
    validate_setup,
    print_validation_result,
)

# Re-export validation functions for easy access
__all__ = [
    "ValidationIssue", 
    "ValidationResult",
    "validate_setup",
    "print_validation_result",
]


def validate_databricks_unity_catalog_setup(**kwargs):
    """
    Convenience function to validate Databricks Unity Catalog setup.
    
    This is the main entry point for validation checks.
    
    Args:
        **kwargs: Arguments passed to validate_setup()
        
    Returns:
        ValidationResult with detailed validation information
    """
    return validate_setup(**kwargs)