"""
Production-grade error handling for GenOps LlamaIndex integration.

Implements circuit breaker patterns, exponential backoff, and graceful degradation
following CLAUDE.md developer best practices.
"""

import logging
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failed, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: float = 60.0  # Seconds before attempting recovery
    success_threshold: int = 3  # Successes needed to close from half-open
    timeout: float = 30.0  # Request timeout in seconds


@dataclass
class RetryConfig:
    """Configuration for retry with exponential backoff."""

    max_retries: int = 3
    base_delay: float = 1.0  # Initial delay in seconds
    max_delay: float = 30.0  # Maximum delay in seconds
    backoff_multiplier: float = 2.0  # Exponential backoff multiplier
    jitter: bool = True  # Add random jitter to prevent thundering herd


@dataclass
class ErrorMetrics:
    """Metrics for error tracking and monitoring."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    circuit_breaker_opens: int = 0
    retry_attempts: int = 0
    timeout_errors: int = 0
    rate_limit_errors: int = 0
    last_error: Optional[str] = None
    last_error_time: Optional[float] = None

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def error_rate(self) -> float:
        """Calculate error rate percentage."""
        return 100.0 - self.success_rate


class CircuitBreaker:
    """Circuit breaker implementation for API calls."""

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.metrics = ErrorMetrics()

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        self.metrics.total_requests += 1

        if self._should_reject():
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is open. "
                f"Last failure: {self.metrics.last_error}"
            )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except Exception as e:
            self._on_failure(str(e))
            raise

    def _should_reject(self) -> bool:
        """Check if request should be rejected."""
        if self.state == CircuitState.CLOSED:
            return False

        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if (
                self.last_failure_time
                and time.time() - self.last_failure_time >= self.config.recovery_timeout
            ):
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker '{self.name}' entering half-open state")
                return False
            return True

        # HALF_OPEN state - allow limited requests
        return False

    def _on_success(self):
        """Handle successful request."""
        self.metrics.successful_requests += 1

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info(f"Circuit breaker '{self.name}' closed after recovery")

    def _on_failure(self, error_message: str):
        """Handle failed request."""
        self.metrics.failed_requests += 1
        self.metrics.last_error = error_message
        self.metrics.last_error_time = time.time()

        if "timeout" in error_message.lower():
            self.metrics.timeout_errors += 1
        elif "rate limit" in error_message.lower():
            self.metrics.rate_limit_errors += 1

        if self.state == CircuitState.HALF_OPEN:
            # Failed during recovery - go back to open
            self.state = CircuitState.OPEN
            self.last_failure_time = time.time()
            logger.warning(f"Circuit breaker '{self.name}' failed during recovery")
        else:
            self.failure_count += 1
            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
                self.last_failure_time = time.time()
                self.metrics.circuit_breaker_opens += 1
                logger.error(
                    f"Circuit breaker '{self.name}' opened after {self.failure_count} failures"
                )

    def get_status(self) -> dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "success_rate": self.metrics.success_rate,
                "error_rate": self.metrics.error_rate,
                "circuit_opens": self.metrics.circuit_breaker_opens,
                "timeout_errors": self.metrics.timeout_errors,
                "rate_limit_errors": self.metrics.rate_limit_errors,
            },
        }


class RetryHandler:
    """Exponential backoff retry handler."""

    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()

    def retry(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with exponential backoff retry."""
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                last_exception = e

                # Don't retry on certain errors
                if self._should_not_retry(e):
                    raise

                # Don't delay on last attempt
                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Retry attempt {attempt + 1} after {delay:.2f}s delay: {str(e)}"
                    )
                    time.sleep(delay)

        # All retries exhausted
        raise RetryExhaustedError(
            f"Failed after {self.config.max_retries + 1} attempts"
        ) from last_exception

    def _should_not_retry(self, exception: Exception) -> bool:
        """Check if exception should not be retried."""
        error_str = str(exception).lower()

        # Don't retry on authentication errors
        if "authentication" in error_str or "unauthorized" in error_str:
            return True

        # Don't retry on invalid request errors
        if "400" in error_str or "bad request" in error_str:
            return True

        # Don't retry on quota exceeded (different from rate limiting)
        if "quota exceeded" in error_str:
            return True

        return False

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        base_delay = self.config.base_delay * (self.config.backoff_multiplier**attempt)
        delay = min(base_delay, self.config.max_delay)

        if self.config.jitter:
            # Add random jitter (±25%)
            jitter = delay * 0.25 * (2 * random.random() - 1)
            delay += jitter

        return max(0.1, delay)  # Minimum 100ms delay


class ProviderHealthMonitor:
    """Monitor provider health and implement graceful degradation."""

    def __init__(self):
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self.retry_handler = RetryHandler()
        self.provider_priorities: dict[str, int] = {
            "openai": 1,
            "anthropic": 2,
            "google": 3,
            "cohere": 4,
        }

    def get_circuit_breaker(self, provider: str) -> CircuitBreaker:
        """Get or create circuit breaker for provider."""
        if provider not in self.circuit_breakers:
            self.circuit_breakers[provider] = CircuitBreaker(
                name=f"{provider}_circuit_breaker", config=CircuitBreakerConfig()
            )
        return self.circuit_breakers[provider]

    def call_with_protection(
        self, provider: str, func: Callable[..., T], *args, **kwargs
    ) -> T:
        """Call provider function with full error protection."""
        circuit_breaker = self.get_circuit_breaker(provider)

        def protected_call():
            return circuit_breaker.call(func, *args, **kwargs)

        return self.retry_handler.retry(protected_call)

    def get_healthy_providers(self) -> list[str]:
        """Get list of currently healthy providers."""
        healthy = []

        for provider in self.provider_priorities.keys():
            if provider in self.circuit_breakers:
                breaker = self.circuit_breakers[provider]
                if breaker.state != CircuitState.OPEN:
                    healthy.append(provider)
            else:
                # No circuit breaker yet means no failures
                healthy.append(provider)

        # Sort by priority
        healthy.sort(key=lambda p: self.provider_priorities.get(p, 999))
        return healthy

    def get_fallback_provider(self, failed_provider: str) -> Optional[str]:
        """Get next best provider when primary fails."""
        healthy_providers = self.get_healthy_providers()

        # Remove the failed provider
        if failed_provider in healthy_providers:
            healthy_providers.remove(failed_provider)

        return healthy_providers[0] if healthy_providers else None

    def get_system_health(self) -> dict[str, Any]:
        """Get overall system health status."""
        healthy_providers = self.get_healthy_providers()
        all_providers = list(self.provider_priorities.keys())

        provider_status = {}
        for provider in all_providers:
            if provider in self.circuit_breakers:
                provider_status[provider] = self.circuit_breakers[provider].get_status()
            else:
                provider_status[provider] = {"state": "healthy", "no_data": True}

        return {
            "healthy_providers": healthy_providers,
            "total_providers": len(all_providers),
            "health_percentage": len(healthy_providers) / len(all_providers) * 100,
            "provider_status": provider_status,
            "recommendations": self._get_health_recommendations(
                healthy_providers, all_providers
            ),
        }

    def _get_health_recommendations(
        self, healthy: list[str], all_providers: list[str]
    ) -> list[str]:
        """Generate health recommendations."""
        recommendations = []

        if len(healthy) == 0:
            recommendations.append(
                "CRITICAL: All providers unavailable - check network and API keys"
            )
        elif len(healthy) == 1:
            recommendations.append(
                f"WARNING: Only {healthy[0]} available - single point of failure"
            )
        elif len(healthy) < len(all_providers):
            failed = set(all_providers) - set(healthy)
            recommendations.append(
                f"INFO: {len(failed)} provider(s) degraded: {', '.join(failed)}"
            )

        return recommendations


# Custom exceptions
class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""

    pass


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""

    pass


class GracefulDegradationError(Exception):
    """Raised when graceful degradation is needed."""

    pass


# Decorators for easy use
def with_circuit_breaker(provider: str, config: Optional[CircuitBreakerConfig] = None):
    """Decorator to add circuit breaker protection to functions."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        circuit_breaker = CircuitBreaker(f"{provider}_{func.__name__}", config)

        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return circuit_breaker.call(func, *args, **kwargs)

        return wrapper

    return decorator


def with_retry(config: Optional[RetryConfig] = None):
    """Decorator to add retry logic to functions."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        retry_handler = RetryHandler(config)

        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            return retry_handler.retry(func, *args, **kwargs)

        return wrapper

    return decorator


@contextmanager
def graceful_degradation(
    primary_provider: str,
    fallback_providers: list[str],
    health_monitor: ProviderHealthMonitor,
):
    """Context manager for graceful degradation between providers."""
    providers_to_try = [primary_provider] + fallback_providers

    for provider in providers_to_try:
        try:
            yield provider
            break  # Success, no need to try other providers

        except Exception as e:
            logger.warning(f"Provider {provider} failed: {str(e)}")

            if provider == providers_to_try[-1]:
                # Last provider failed
                raise GracefulDegradationError(
                    f"All providers failed. Last error from {provider}: {str(e)}"
                ) from e

            # Try next provider
            continue


# Global health monitor instance
_global_health_monitor: Optional[ProviderHealthMonitor] = None


def get_health_monitor() -> ProviderHealthMonitor:
    """Get global health monitor instance."""
    global _global_health_monitor
    if _global_health_monitor is None:
        _global_health_monitor = ProviderHealthMonitor()
    return _global_health_monitor


# Export main classes and functions
__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "RetryHandler",
    "RetryConfig",
    "ProviderHealthMonitor",
    "ErrorMetrics",
    "CircuitBreakerOpenError",
    "RetryExhaustedError",
    "GracefulDegradationError",
    "with_circuit_breaker",
    "with_retry",
    "graceful_degradation",
    "get_health_monitor",
]
