"""
GenOps Core Exceptions

Custom exception classes for GenOps AI governance operations.
"""
from typing import Optional


class GenOpsError(Exception):
    """Base exception class for GenOps operations."""
    pass


class GenOpsBudgetExceededError(GenOpsError):
    """Raised when operation would exceed budget limits."""

    def __init__(self, message: str, budget_limit: Optional[float] = None, current_cost: Optional[float] = None, operation_cost: Optional[float] = None) -> None:
        super().__init__(message)
        self.budget_limit = budget_limit
        self.current_cost = current_cost
        self.operation_cost = operation_cost


class GenOpsConfigurationError(GenOpsError):
    """Raised when configuration is invalid or missing."""
    pass


class GenOpsValidationError(GenOpsError):
    """Raised when validation fails."""
    pass


class GenOpsProviderError(GenOpsError):
    """Raised when provider operations fail."""
    pass


class GenOpsSessionError(GenOpsError):
    """Raised when session management fails."""
    pass
