"""Anyscale provider adapter for GenOps AI governance."""

from __future__ import annotations

import logging
import time
import uuid
import os
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from contextlib import contextmanager

from genops.providers.base import BaseFrameworkProvider
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# Check for required dependencies
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("requests not installed. Install with: pip install requests")

# Optional: OpenAI SDK for compatibility (Anyscale is OpenAI-compatible)
try:
    from openai import OpenAI
    HAS_OPENAI_SDK = True
except ImportError:
    HAS_OPENAI_SDK = False
    logger.info("OpenAI SDK not installed. Will use direct HTTP requests. Install with: pip install openai")


@dataclass
class AnyscaleOperation:
    """Represents a single Anyscale operation for tracking."""

    operation_id: str
    operation_type: str  # 'completion', 'chat', 'embedding'
    model: str
    start_time: float
    end_time: Optional[float] = None

    # Token usage
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None

    # Cost tracking
    cost: Optional[float] = None
    currency: str = "USD"

    # Performance metrics
    latency_ms: Optional[float] = None
    first_token_ms: Optional[float] = None

    # Governance attributes
    governance_attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Calculate operation duration in milliseconds."""
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000


@dataclass
class AnyscaleCostSummary:
    """Cost summary for Anyscale operations."""

    total_cost: float = 0.0
    currency: str = "USD"
    operations: List[AnyscaleOperation] = field(default_factory=list)
    cost_by_model: Dict[str, float] = field(default_factory=dict)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_operations: int = 0

    def add_operation(self, operation: AnyscaleOperation):
        """Add an operation to the summary."""
        self.operations.append(operation)
        self.total_operations += 1

        if operation.cost:
            self.total_cost += operation.cost
            if operation.model not in self.cost_by_model:
                self.cost_by_model[operation.model] = 0.0
            self.cost_by_model[operation.model] += operation.cost

        if operation.input_tokens:
            self.total_input_tokens += operation.input_tokens
        if operation.output_tokens:
            self.total_output_tokens += operation.output_tokens


class GenOpsAnyscaleAdapter(BaseFrameworkProvider):
    """
    GenOps adapter for Anyscale Endpoints with comprehensive governance.

    Provides cost tracking, telemetry, and policy enforcement for:
    - Chat completions with multiple model support
    - Embeddings generation
    - Multi-provider cost aggregation
    - Team-based attribution and governance
    """

    def __init__(
        self,
        anyscale_api_key: Optional[str] = None,
        anyscale_base_url: str = "https://api.endpoints.anyscale.com/v1",
        telemetry_enabled: bool = True,
        cost_tracking_enabled: bool = True,
        debug: bool = False,
        # Enterprise features
        enable_retry: bool = True,
        max_retries: int = 3,
        retry_backoff_factor: float = 1.0,
        enable_circuit_breaker: bool = False,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: int = 60,
        sampling_rate: float = 1.0,
        request_timeout: int = 60,
        **governance_defaults
    ):
        """
        Initialize GenOps Anyscale adapter.

        Args:
            anyscale_api_key: Anyscale API key (or set ANYSCALE_API_KEY env var)
            anyscale_base_url: Base URL for Anyscale Endpoints API
            telemetry_enabled: Enable OpenTelemetry export
            cost_tracking_enabled: Enable cost calculation and tracking
            debug: Enable debug logging

            Enterprise features:
            enable_retry: Enable automatic retry on transient failures (default: True)
            max_retries: Maximum retry attempts (default: 3)
            retry_backoff_factor: Exponential backoff multiplier (default: 1.0)
            enable_circuit_breaker: Enable circuit breaker pattern (default: False)
            circuit_breaker_threshold: Failures before opening circuit (default: 5)
            circuit_breaker_timeout: Circuit recovery timeout in seconds (default: 60)
            sampling_rate: Telemetry sampling rate 0.0-1.0 (default: 1.0 = 100%)
            request_timeout: Request timeout in seconds (default: 60)

            **governance_defaults: Default governance attributes (team, project, etc.)
        """
        super().__init__()

        # API configuration
        self.anyscale_api_key = anyscale_api_key or os.getenv("ANYSCALE_API_KEY")
        self.anyscale_base_url = anyscale_base_url.rstrip('/')
        self.telemetry_enabled = telemetry_enabled
        self.cost_tracking_enabled = cost_tracking_enabled
        self.debug = debug
        self.governance_defaults = governance_defaults

        # Validate API key
        if not self.anyscale_api_key:
            logger.warning(
                "ANYSCALE_API_KEY not set. Set via environment variable or constructor parameter. "
                "Some operations will fail without authentication."
            )

        # Initialize HTTP client or OpenAI SDK client
        if HAS_OPENAI_SDK and self.anyscale_api_key:
            self.client = OpenAI(
                api_key=self.anyscale_api_key,
                base_url=self.anyscale_base_url
            )
            self._use_sdk = True
            logger.debug("Initialized Anyscale adapter with OpenAI SDK")
        elif HAS_REQUESTS:
            self.client = None  # Will use requests directly
            self._use_sdk = False
            logger.debug("Initialized Anyscale adapter with HTTP requests")
        else:
            raise ImportError(
                "Neither OpenAI SDK nor requests library available. "
                "Install with: pip install openai OR pip install requests"
            )

        # Load pricing calculator
        from .pricing import AnyscalePricing
        self._pricing = AnyscalePricing()

        # Operation tracking
        self._current_operations: Dict[str, AnyscaleOperation] = {}

        # Enterprise features
        self.enable_retry = enable_retry
        self.max_retries = max_retries
        self.retry_backoff_factor = retry_backoff_factor
        self.enable_circuit_breaker = enable_circuit_breaker
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        self.sampling_rate = max(0.0, min(1.0, sampling_rate))  # Clamp to [0.0, 1.0]
        self.request_timeout = request_timeout

        # Circuit breaker state
        self._circuit_breaker_state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._circuit_breaker_failure_count = 0
        self._circuit_breaker_last_failure_time: Optional[float] = None

        logger.info(f"GenOps Anyscale adapter initialized (telemetry={'enabled' if telemetry_enabled else 'disabled'}, "
                   f"retry={'enabled' if enable_retry else 'disabled'}, "
                   f"circuit_breaker={'enabled' if enable_circuit_breaker else 'disabled'}, "
                   f"sampling={sampling_rate*100:.0f}%)")

    # Security methods for secret protection and input validation

    def _sanitize_error_message(self, error: Exception) -> str:
        """Remove sensitive information from error messages."""
        import re
        error_str = str(error)
        # Redact anything that looks like a bearer token
        error_str = re.sub(r'Bearer\s+\S+', 'Bearer [REDACTED]', error_str)
        # Redact API keys
        error_str = re.sub(r'api[_-]?key["\']?\s*[:=]\s*["\']?\S+', 'api_key=[REDACTED]', error_str, flags=re.IGNORECASE)
        return error_str

    def _sanitize_response_text(self, text: str, max_length: int = 200) -> str:
        """Sanitize API response text before logging."""
        import re
        if not text:
            return "No response text"
        truncated = text[:max_length]
        sanitized = re.sub(r'Bearer\s+\S+', 'Bearer [REDACTED]', truncated)
        sanitized = re.sub(r'"token":\s*"\S+"', '"token": "[REDACTED]"', sanitized)
        return sanitized

    def _build_headers(self) -> dict:
        """Build HTTP headers with secret protection."""
        auth_value = "Bearer " + self.anyscale_api_key
        return {
            "Authorization": auth_value,
            "Content-Type": "application/json"
        }

    def _validate_endpoint(self, endpoint: str) -> str:
        """Validate endpoint path to prevent injection."""
        if not endpoint.startswith('/'):
            endpoint = '/' + endpoint
        if '://' in endpoint:
            raise ValueError("Endpoint must not contain protocol")
        if '..' in endpoint:
            raise ValueError("Endpoint must not contain '..'")
        return endpoint

    def _validate_completion_response(self, response_data: dict) -> dict:
        """Validate completion response structure."""
        required_fields = ['choices', 'usage']
        for field in required_fields:
            if field not in response_data:
                raise ValueError(f"Invalid response: missing '{field}'")
        if not isinstance(response_data['choices'], list) or not response_data['choices']:
            raise ValueError("Invalid response: 'choices' must be non-empty list")
        usage = response_data.get('usage', {})
        for token_field in ['prompt_tokens', 'completion_tokens', 'total_tokens']:
            if token_field in usage:
                value = usage[token_field]
                if not isinstance(value, int) or value < 0 or value > 1000000:
                    raise ValueError(f"Invalid token count for {token_field}: {value}")
        return response_data

    def _validate_embeddings_response(self, response_data: dict) -> dict:
        """Validate embeddings response structure."""
        required_fields = ['data', 'usage']
        for field in required_fields:
            if field not in response_data:
                raise ValueError(f"Invalid response: missing '{field}'")
        if not isinstance(response_data['data'], list) or not response_data['data']:
            raise ValueError("Invalid response: 'data' must be non-empty list")
        return response_data

    # BaseFrameworkProvider abstract method implementations

    def setup_governance_attributes(self) -> None:
        """Setup Anyscale-specific governance attributes."""
        # Add any Anyscale-specific request attributes
        self.REQUEST_ATTRIBUTES = {
            'model', 'messages', 'temperature', 'max_tokens', 'top_p',
            'frequency_penalty', 'presence_penalty', 'stop', 'stream',
            'input', 'encoding_format', 'dimensions'  # For embeddings
        }

    def get_framework_name(self) -> str:
        """Return framework name."""
        return "anyscale"

    def get_framework_type(self) -> str:
        """Return framework type."""
        return self.FRAMEWORK_TYPE_INFERENCE

    def get_framework_version(self) -> str | None:
        """Return Anyscale SDK version if available."""
        try:
            if self._use_sdk:
                import openai
                return f"openai-{openai.__version__}"
            else:
                import requests
                return f"requests-{requests.__version__}"
        except Exception as e:
            logger.debug(f"Failed to get framework version: {e}")
            return None

    def is_framework_available(self) -> bool:
        """Check if Anyscale can be used."""
        return (HAS_OPENAI_SDK or HAS_REQUESTS) and bool(self.anyscale_api_key)

    def calculate_cost(self, operation_context: dict) -> float:
        """
        Calculate cost for Anyscale operation.

        Args:
            operation_context: Dict with keys:
                - model: Model name
                - input_tokens: Number of input tokens
                - output_tokens: Number of output tokens

        Returns:
            Estimated cost in USD
        """
        if not self.cost_tracking_enabled:
            return 0.0

        model = operation_context.get('model', '')
        input_tokens = operation_context.get('input_tokens', 0)
        output_tokens = operation_context.get('output_tokens', 0)

        try:
            cost = self._pricing.calculate_cost(model, input_tokens, output_tokens)
            return cost
        except Exception as e:
            logger.warning(f"Failed to calculate cost: {e}")
            return 0.0

    def get_operation_mappings(self) -> dict[str, str]:
        """Return mapping of operations to instrumentation methods."""
        return {
            'chat.completions.create': 'completion_create',
            'completions.create': 'completion_create',
            'embeddings.create': 'embeddings_create',
        }

    def _record_framework_metrics(self, span: Any, operation_type: str, context: dict) -> None:
        """Record Anyscale-specific metrics on span."""
        if not span:
            return

        try:
            # Record model info
            if 'model' in context:
                span.set_attribute("genops.anyscale.model", context['model'])

            # Record token usage
            if 'input_tokens' in context:
                span.set_attribute("genops.anyscale.tokens.input", context['input_tokens'])
            if 'output_tokens' in context:
                span.set_attribute("genops.anyscale.tokens.output", context['output_tokens'])
            if 'total_tokens' in context:
                span.set_attribute("genops.anyscale.tokens.total", context['total_tokens'])

            # Record cost
            if 'cost' in context:
                span.set_attribute("genops.anyscale.cost.total", context['cost'])
                span.set_attribute("genops.anyscale.cost.currency", "USD")

            # Record performance metrics
            if 'latency_ms' in context:
                span.set_attribute("genops.anyscale.performance.latency_ms", context['latency_ms'])

            # Record operation type
            span.set_attribute("genops.anyscale.operation.type", operation_type)

        except Exception as e:
            logger.debug(f"Failed to record framework metrics: {e}")

    def _apply_instrumentation(self, **config) -> None:
        """Apply instrumentation (called by base class)."""
        # Anyscale uses direct API calls, instrumentation happens at method level
        logger.debug("Anyscale instrumentation applied")

    def _remove_instrumentation(self) -> None:
        """Remove instrumentation (called by base class)."""
        logger.debug("Anyscale instrumentation removed")

    # Enterprise feature methods

    def _should_sample_request(self) -> bool:
        """Determine if request should generate telemetry based on sampling rate."""
        if self.sampling_rate >= 1.0:
            return True
        if self.sampling_rate <= 0.0:
            return False

        import random
        return random.random() < self.sampling_rate

    def _check_circuit_breaker(self) -> None:
        """Check circuit breaker state and raise exception if open."""
        if not self.enable_circuit_breaker:
            return

        if self._circuit_breaker_state == "OPEN":
            # Check if recovery timeout has passed
            if (self._circuit_breaker_last_failure_time and
                time.time() - self._circuit_breaker_last_failure_time > self.circuit_breaker_timeout):
                logger.info("Circuit breaker: Moving to HALF_OPEN state")
                self._circuit_breaker_state = "HALF_OPEN"
                self._circuit_breaker_failure_count = 0
            else:
                raise Exception(
                    f"Circuit breaker is OPEN - too many failures. "
                    f"Will retry after {self.circuit_breaker_timeout}s"
                )

    def _record_circuit_breaker_success(self) -> None:
        """Record successful request for circuit breaker."""
        if not self.enable_circuit_breaker:
            return

        if self._circuit_breaker_state == "HALF_OPEN":
            logger.info("Circuit breaker: Success in HALF_OPEN, moving to CLOSED")
            self._circuit_breaker_state = "CLOSED"
            self._circuit_breaker_failure_count = 0

    def _record_circuit_breaker_failure(self) -> None:
        """Record failed request for circuit breaker."""
        if not self.enable_circuit_breaker:
            return

        self._circuit_breaker_failure_count += 1
        self._circuit_breaker_last_failure_time = time.time()

        if self._circuit_breaker_failure_count >= self.circuit_breaker_threshold:
            logger.warning(
                f"Circuit breaker: Threshold reached ({self._circuit_breaker_failure_count} failures), "
                f"opening circuit"
            )
            self._circuit_breaker_state = "OPEN"

    def _make_request_with_retry(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        json: Dict[str, Any],
        timeout: int
    ) -> Any:
        """Make HTTP request with retry logic."""
        # Check circuit breaker before attempting request
        self._check_circuit_breaker()

        last_exception = None
        attempt = 0
        max_attempts = self.max_retries if self.enable_retry else 1

        while attempt < max_attempts:
            try:
                if attempt > 0:
                    # Calculate exponential backoff
                    wait_time = min(
                        self.retry_backoff_factor * (2 ** (attempt - 1)),
                        10  # Max 10 seconds
                    )
                    logger.debug(f"Retry attempt {attempt}/{max_attempts-1}, waiting {wait_time:.2f}s")
                    time.sleep(wait_time)

                # Make the request
                response = requests.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=json,
                    timeout=timeout
                )

                # Check for HTTP errors
                if response.status_code >= 500:
                    # Server error - retry
                    raise Exception(f"Server error: {response.status_code}")
                elif response.status_code == 429:
                    # Rate limit - retry with backoff
                    raise Exception("Rate limit exceeded")
                elif response.status_code >= 400:
                    # Client error - don't retry
                    response.raise_for_status()

                # Success
                self._record_circuit_breaker_success()
                return response

            except Exception as e:
                last_exception = e
                attempt += 1

                # Record failure for circuit breaker
                self._record_circuit_breaker_failure()

                if attempt >= max_attempts:
                    logger.error(f"Request failed after {max_attempts} attempts: {e}")
                    raise

                logger.warning(f"Request attempt {attempt} failed: {e}")

        # Should not reach here, but handle gracefully
        if last_exception:
            raise last_exception
        raise Exception("Request failed with unknown error")

    # Anyscale-specific API methods

    def completion_create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a chat completion with governance tracking.

        Args:
            model: Model name (e.g., "meta-llama/Llama-2-70b-chat-hf")
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters and governance attributes

        Returns:
            Completion response dict
        """
        # Extract governance attributes
        governance_attrs, request_attrs, api_kwargs = self._extract_attributes(kwargs)

        # Merge with defaults
        effective_governance = {**self.governance_defaults, **governance_attrs}

        # Create operation tracking
        operation_id = str(uuid.uuid4())
        operation = AnyscaleOperation(
            operation_id=operation_id,
            operation_type="chat.completion",
            model=model,
            start_time=time.time(),
            governance_attributes=effective_governance
        )

        # Build trace attributes
        trace_attrs = self._build_trace_attributes(
            operation_name="anyscale.completion.create",
            operation_type="ai.inference",
            governance_attrs=effective_governance,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Start OpenTelemetry span
        with tracer.start_as_current_span(
            "anyscale.completion.create",
            attributes=trace_attrs
        ) as span:
            try:
                # Make API call
                if self._use_sdk:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **api_kwargs
                    )
                    response_dict = self._parse_sdk_response(response)
                else:
                    response_dict = self._make_http_request(
                        endpoint="/chat/completions",
                        data={
                            "model": model,
                            "messages": messages,
                            "temperature": temperature,
                            **({"max_tokens": max_tokens} if max_tokens else {}),
                            **api_kwargs
                        }
                    )

                # Extract token usage
                usage = response_dict.get('usage', {})
                operation.input_tokens = usage.get('prompt_tokens', 0)
                operation.output_tokens = usage.get('completion_tokens', 0)
                operation.total_tokens = usage.get('total_tokens', 0)

                # Calculate cost
                if self.cost_tracking_enabled:
                    operation.cost = self.calculate_cost({
                        'model': model,
                        'input_tokens': operation.input_tokens,
                        'output_tokens': operation.output_tokens
                    })

                # Record metrics
                operation.end_time = time.time()
                operation.latency_ms = operation.duration_ms

                # Update span
                self._record_framework_metrics(span, "chat.completion", {
                    'model': model,
                    'input_tokens': operation.input_tokens,
                    'output_tokens': operation.output_tokens,
                    'total_tokens': operation.total_tokens,
                    'cost': operation.cost,
                    'latency_ms': operation.latency_ms
                })

                span.set_status(Status(StatusCode.OK))

                if self.debug:
                    logger.debug(
                        f"Completion created: model={model}, tokens={operation.total_tokens}, "
                        f"cost=${operation.cost:.6f}, latency={operation.latency_ms:.2f}ms"
                    )

                return response_dict

            except Exception as e:
                operation.end_time = time.time()
                sanitized_error = self._sanitize_error_message(e)
                span.set_status(Status(StatusCode.ERROR, sanitized_error))
                span.record_exception(e)
                logger.error(f"Completion failed: {sanitized_error}")
                raise

    def embeddings_create(
        self,
        model: str,
        input: Union[str, List[str]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create embeddings with governance tracking.

        Args:
            model: Embedding model name (e.g., "thenlper/gte-large")
            input: Text or list of texts to embed
            **kwargs: Additional parameters and governance attributes

        Returns:
            Embeddings response dict
        """
        # Extract governance attributes
        governance_attrs, request_attrs, api_kwargs = self._extract_attributes(kwargs)
        effective_governance = {**self.governance_defaults, **governance_attrs}

        # Create operation tracking
        operation_id = str(uuid.uuid4())
        operation = AnyscaleOperation(
            operation_id=operation_id,
            operation_type="embedding",
            model=model,
            start_time=time.time(),
            governance_attributes=effective_governance
        )

        # Build trace attributes
        trace_attrs = self._build_trace_attributes(
            operation_name="anyscale.embeddings.create",
            operation_type="ai.embedding",
            governance_attrs=effective_governance,
            model=model
        )

        # Start OpenTelemetry span
        with tracer.start_as_current_span(
            "anyscale.embeddings.create",
            attributes=trace_attrs
        ) as span:
            try:
                # Make API call
                if self._use_sdk:
                    response = self.client.embeddings.create(
                        model=model,
                        input=input,
                        **api_kwargs
                    )
                    response_dict = self._parse_sdk_response(response)
                else:
                    response_dict = self._make_http_request(
                        endpoint="/embeddings",
                        data={
                            "model": model,
                            "input": input,
                            **api_kwargs
                        }
                    )

                # Extract token usage
                usage = response_dict.get('usage', {})
                operation.input_tokens = usage.get('total_tokens', 0)
                operation.output_tokens = 0  # Embeddings don't have output tokens

                # Calculate cost
                if self.cost_tracking_enabled:
                    operation.cost = self.calculate_cost({
                        'model': model,
                        'input_tokens': operation.input_tokens,
                        'output_tokens': 0
                    })

                # Record metrics
                operation.end_time = time.time()
                operation.latency_ms = operation.duration_ms

                # Update span
                self._record_framework_metrics(span, "embedding", {
                    'model': model,
                    'input_tokens': operation.input_tokens,
                    'cost': operation.cost,
                    'latency_ms': operation.latency_ms
                })

                span.set_status(Status(StatusCode.OK))

                if self.debug:
                    logger.debug(
                        f"Embeddings created: model={model}, tokens={operation.input_tokens}, "
                        f"cost=${operation.cost:.6f}, latency={operation.latency_ms:.2f}ms"
                    )

                return response_dict

            except Exception as e:
                operation.end_time = time.time()
                sanitized_error = self._sanitize_error_message(e)
                span.set_status(Status(StatusCode.ERROR, sanitized_error))
                span.record_exception(e)
                logger.error(f"Embeddings failed: {sanitized_error}")
                raise

    def _make_http_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make direct HTTP request to Anyscale API."""
        if not HAS_REQUESTS:
            raise ImportError("requests library required for HTTP API calls")

        # Validate endpoint to prevent injection
        validated_endpoint = self._validate_endpoint(endpoint)
        url = f"{self.anyscale_base_url}{validated_endpoint}"

        # Use header builder to prevent secret exposure
        headers = self._build_headers()

        response = requests.post(url, json=data, headers=headers, timeout=60)
        response.raise_for_status()

        # Validate response structure based on endpoint type
        response_json = response.json()
        if '/completions' in endpoint:
            return self._validate_completion_response(response_json)
        elif '/embeddings' in endpoint:
            return self._validate_embeddings_response(response_json)

        return response_json

    def _parse_sdk_response(self, response: Any) -> Dict[str, Any]:
        """Parse OpenAI SDK response to dict."""
        if hasattr(response, 'model_dump'):
            return response.model_dump()
        elif hasattr(response, 'dict'):
            return response.dict()
        else:
            return dict(response)

    @contextmanager
    def governance_context(self, **attributes):
        """
        Context manager to set governance attributes for operations.

        Example:
            with adapter.governance_context(team="ml-team", customer_id="acme-corp"):
                response = adapter.completion_create(...)
        """
        old_defaults = self.governance_defaults.copy()
        self.governance_defaults.update(attributes)
        try:
            yield
        finally:
            self.governance_defaults = old_defaults


# Convenience factory function
def instrument_anyscale(
    anyscale_api_key: Optional[str] = None,
    **governance_defaults
) -> GenOpsAnyscaleAdapter:
    """
    Create and initialize GenOps Anyscale adapter.

    Args:
        anyscale_api_key: Anyscale API key (or set ANYSCALE_API_KEY env var)
        **governance_defaults: Default governance attributes

    Returns:
        Initialized GenOpsAnyscaleAdapter

    Example:
        adapter = instrument_anyscale(
            team="ml-research",
            project="chatbot",
            environment="production"
        )
    """
    return GenOpsAnyscaleAdapter(
        anyscale_api_key=anyscale_api_key,
        **governance_defaults
    )


# Export public API
__all__ = [
    'GenOpsAnyscaleAdapter',
    'AnyscaleOperation',
    'AnyscaleCostSummary',
    'instrument_anyscale',
]
