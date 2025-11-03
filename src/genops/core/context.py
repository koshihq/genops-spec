"""Global attribution context management for GenOps AI."""

import logging
import threading
from contextvars import ContextVar
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Thread-local storage for default attributes
_default_attributes: dict[str, Any] = {}
_context_lock = threading.Lock()

# Context variables for async support
_context_attributes: ContextVar[dict[str, Any]] = ContextVar('genops_context', default=None)


def set_default_attributes(**attributes: Any) -> None:
    """
    Set global default attribution attributes for all GenOps operations.

    These attributes will be automatically included in all telemetry unless
    explicitly overridden at the operation level.

    Args:
        **attributes: Key-value pairs for default attribution

    Example:
        import genops

        # Set defaults for the entire application
        genops.set_default_attributes(
            team="platform-engineering",
            project="ai-services",
            environment="production",
            cost_center="engineering"
        )

        # All subsequent operations inherit these defaults
        client = instrument_openai(api_key="key")
        response = client.chat_completions_create(
            model="gpt-4",
            messages=[...],
            # Only need to specify operation-specific attributes
            customer_id="enterprise-123",
            feature="chat-assistant"
            # team, project, environment, cost_center automatically included
        )
    """
    global _default_attributes
    with _context_lock:
        _default_attributes.update(attributes)


def get_default_attributes() -> dict[str, Any]:
    """
    Get the current global default attributes.

    Returns:
        Dict containing all currently set default attributes
    """
    with _context_lock:
        return _default_attributes.copy()


def clear_default_attributes() -> None:
    """
    Clear all global default attributes.

    Useful for testing or when you need to reset attribution context.
    """
    global _default_attributes
    with _context_lock:
        _default_attributes.clear()


def update_default_attributes(**attributes: Any) -> None:
    """
    Update specific default attributes without clearing others.

    Args:
        **attributes: Key-value pairs to update

    Example:
        # Change environment from development to production
        genops.update_default_attributes(environment="production")
    """
    set_default_attributes(**attributes)


def set_context(**attributes: Any) -> None:
    """
    Set context-specific attributes (for async/request-scoped attribution).

    Unlike default attributes, context attributes are scoped to the current
    context (thread, async task, or request) and don't affect other contexts.

    Args:
        **attributes: Key-value pairs for context-specific attribution

    Example:
        # In a web request handler
        @app.route('/api/chat')
        def chat_endpoint():
            genops.set_context(
                user_id=request.user.id,
                customer_id=request.headers.get('X-Customer-ID'),
                request_id=request.id
            )

            # AI operations in this request automatically get these attributes
            response = ai_chat(request.json['message'])
            return response
    """
    current_context = _context_attributes.get() or {}
    updated_context = current_context.copy()
    updated_context.update(attributes)
    _context_attributes.set(updated_context)


def get_context() -> dict[str, Any]:
    """
    Get the current context-specific attributes.

    Returns:
        Dict containing all currently set context attributes
    """
    return _context_attributes.get() or {}


def clear_context() -> None:
    """
    Clear all context-specific attributes.

    Useful at the end of request processing or async task completion.
    """
    _context_attributes.set({})


def get_effective_attributes(**overrides: Any) -> dict[str, Any]:
    """
    Get the effective attributes for an operation, combining defaults,
    context, and operation-specific overrides.

    Priority order (highest to lowest):
    1. Operation-specific overrides
    2. Context attributes
    3. Default attributes

    Args:
        **overrides: Operation-specific attribute overrides

    Returns:
        Dict containing the final effective attributes
    """
    # Start with defaults
    effective = get_default_attributes()

    # Add context attributes (higher priority than defaults)
    effective.update(get_context())

    # Add operation-specific overrides (highest priority)
    effective.update(overrides)

    # Remove None values
    effective = {k: v for k, v in effective.items() if v is not None}

    # Validate attributes if validation is enabled
    try:
        from genops.core.validation import validate_tags
        validation_result = validate_tags(effective)

        # Log validation warnings and errors
        if validation_result.warnings:
            for warning in validation_result.warnings:
                logger.warning(f"Tag validation warning: {warning['message']}")

        if validation_result.violations:
            for violation in validation_result.violations:
                if violation.get('severity') == 'error':
                    logger.error(f"Tag validation error: {violation['message']}")
                elif violation.get('severity') == 'block':
                    from genops.core.validation import TagValidationError
                    raise TagValidationError(
                        f"Tag validation blocked operation: {violation['message']}",
                        violations=[violation],
                        warnings=validation_result.warnings
                    )

        return validation_result.cleaned_attributes

    except ImportError:
        # Validation not available, return without validation
        return effective
    except Exception as e:
        logger.error(f"Tag validation failed: {e}")
        return effective


# Convenience functions for common attribution patterns
def set_team_defaults(team: str, project: Optional[str] = None,
                      cost_center: Optional[str] = None, **kwargs: Any) -> None:
    """
    Set default attributes for a team.

    Args:
        team: Team name
        project: Optional project name
        cost_center: Optional cost center for financial attribution
        **kwargs: Additional team-specific attributes
    """
    attrs = {"team": team}
    if project:
        attrs["project"] = project
    if cost_center:
        attrs["cost_center"] = cost_center
    attrs.update(kwargs)
    set_default_attributes(**attrs)


def set_customer_context(customer_id: str, customer_name: Optional[str] = None,
                        tier: Optional[str] = None, **kwargs: Any) -> None:
    """
    Set customer context for the current operation scope.

    Args:
        customer_id: Customer identifier
        customer_name: Optional customer display name
        tier: Optional customer tier (enterprise, startup, etc.)
        **kwargs: Additional customer-specific attributes
    """
    attrs = {"customer_id": customer_id}
    if customer_name:
        attrs["customer"] = customer_name
    if tier:
        attrs["customer_tier"] = tier
    attrs.update(kwargs)
    set_context(**attrs)


def set_user_context(user_id: str, **kwargs: Any) -> None:
    """
    Set user context for individual user attribution.

    Args:
        user_id: User identifier
        **kwargs: Additional user-specific attributes
    """
    attrs = {"user_id": user_id}
    attrs.update(kwargs)
    set_context(**attrs)
