"""
GenOps Core Exceptions

Custom exception classes for GenOps AI governance operations.
"""


class GenOpsError(Exception):
    """Base exception class for GenOps operations."""

    pass


class GenOpsBudgetExceededError(GenOpsError):
    """Raised when operation would exceed budget limits."""

    def __init__(
        self, message, budget_limit=None, current_cost=None, operation_cost=None
    ):
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
