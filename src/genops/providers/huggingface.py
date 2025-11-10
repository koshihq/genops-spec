"""Hugging Face provider adapter for GenOps AI governance."""

from __future__ import annotations

import logging
import re
from typing import Any, Union

from genops.core.telemetry import GenOpsTelemetry

logger = logging.getLogger(__name__)

try:
    import huggingface_hub
    from huggingface_hub import InferenceClient

    HAS_HUGGINGFACE = True
except ImportError:
    HAS_HUGGINGFACE = False
    InferenceClient = None
    huggingface_hub = None
    logger.warning("Hugging Face Hub not installed. Install with: pip install huggingface_hub")


class GenOpsHuggingFaceAdapter:
    """Hugging Face adapter with automatic governance telemetry and multi-provider support."""

    # Supported AI tasks
    SUPPORTED_TASKS = {
        "text-generation",
        "chat-completion",
        "text-to-image",
        "feature-extraction",
        "speech-to-text",
        "image-classification",
        "image-to-text",
        "text-to-speech",
        "automatic-speech-recognition",
        "conversational",
        "fill-mask",
        "question-answering",
        "sentiment-analysis",
        "summarization",
        "translation",
        "zero-shot-classification"
    }

    # Provider detection patterns
    PROVIDER_PATTERNS = {
        "openai": r"(gpt-|dall-e|whisper|text-embedding)",
        "anthropic": r"claude-",
        "cohere": r"(command-|embed-)",
        "meta": r"(llama|meta-llama)",
        "mistral": r"mistral",
        "google": r"(gemma|flan-)",
        "huggingface_hub": r"^[^/]+/[^/]+$",  # org/model format indicates Hub model
    }

    def __init__(self, client: Any | None = None, **client_kwargs: Any):
        if not HAS_HUGGINGFACE:
            raise ImportError(
                "Hugging Face Hub package not found. Install with: pip install huggingface_hub"
            )

        self.client = client or InferenceClient(**client_kwargs)
        self.telemetry = GenOpsTelemetry()

        # Performance configuration
        import os
        self.sampling_rate = float(os.getenv('GENOPS_SAMPLING_RATE', '1.0'))
        self.async_export = os.getenv('GENOPS_ASYNC_EXPORT', 'true').lower() == 'true'
        self.batch_size = int(os.getenv('GENOPS_BATCH_SIZE', '100'))
        self.export_timeout = int(os.getenv('GENOPS_EXPORT_TIMEOUT', '5'))

        # Circuit breaker configuration
        self.circuit_breaker_enabled = os.getenv('GENOPS_CIRCUIT_BREAKER', 'true').lower() == 'true'
        self.circuit_breaker_threshold = int(os.getenv('GENOPS_CB_THRESHOLD', '5'))
        self.circuit_breaker_window = int(os.getenv('GENOPS_CB_WINDOW', '60'))

        # Circuit breaker state
        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_failure = 0
        self._circuit_breaker_open = False

        # Define governance and request attributes
        self.GOVERNANCE_ATTRIBUTES = {
            'team', 'project', 'feature', 'customer_id', 'customer',
            'environment', 'cost_center', 'user_id', 'experiment_id',
            'model_version', 'dataset_id'
        }

        self.REQUEST_ATTRIBUTES = {
            'temperature', 'max_tokens', 'max_new_tokens', 'top_p', 'top_k',
            'repetition_penalty', 'frequency_penalty', 'presence_penalty',
            'do_sample', 'seed', 'stop', 'stream', 'details'
        }

    def _extract_attributes(self, kwargs: dict) -> tuple[dict, dict, dict]:
        """Extract governance and request attributes from kwargs."""
        governance_attrs = {}
        request_attrs = {}
        api_kwargs = kwargs.copy()

        # Extract governance attributes
        for attr in self.GOVERNANCE_ATTRIBUTES:
            if attr in kwargs:
                governance_attrs[attr] = kwargs[attr]
                api_kwargs.pop(attr)

        # Extract request attributes
        for attr in self.REQUEST_ATTRIBUTES:
            if attr in kwargs:
                request_attrs[attr] = kwargs[attr]

        return governance_attrs, request_attrs, api_kwargs

    def _should_sample(self) -> bool:
        """Determine if this operation should be sampled based on sampling rate."""
        if self.sampling_rate >= 1.0:
            return True
        if self.sampling_rate <= 0.0:
            return False

        import random
        return random.random() < self.sampling_rate

    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker is open and should block operations."""
        if not self.circuit_breaker_enabled:
            return False

        import time
        current_time = time.time()

        # If circuit breaker is open, check if enough time has passed to retry
        if self._circuit_breaker_open:
            if current_time - self._circuit_breaker_last_failure > self.circuit_breaker_window:
                # Reset circuit breaker for retry
                self._circuit_breaker_open = False
                self._circuit_breaker_failures = 0
                logger.info("Circuit breaker reset - attempting retry")
                return False
            else:
                # Circuit breaker still open
                return True

        return False

    def _record_circuit_breaker_failure(self):
        """Record a circuit breaker failure."""
        if not self.circuit_breaker_enabled:
            return

        import time
        current_time = time.time()

        self._circuit_breaker_failures += 1
        self._circuit_breaker_last_failure = current_time

        if self._circuit_breaker_failures >= self.circuit_breaker_threshold:
            self._circuit_breaker_open = True
            logger.warning(f"Circuit breaker opened after {self._circuit_breaker_failures} failures")

    def _record_circuit_breaker_success(self):
        """Record a successful operation for circuit breaker."""
        if not self.circuit_breaker_enabled:
            return

        # Reset failure count on successful operation
        if self._circuit_breaker_failures > 0:
            logger.debug("Circuit breaker - resetting failure count after successful operation")
            self._circuit_breaker_failures = 0

    def _async_export_telemetry(self, span_data: dict, cost_data: dict = None):
        """Export telemetry data asynchronously if configured."""
        if not self.async_export:
            return

        import threading

        def export_worker():
            try:
                # This would integrate with actual async telemetry export
                # For now, just log that async export would occur
                logger.debug("Async telemetry export triggered")
                if cost_data:
                    logger.debug(f"Async cost data: {cost_data}")
            except Exception as e:
                logger.warning(f"Async telemetry export failed: {e}")

        thread = threading.Thread(target=export_worker, daemon=True)
        thread.start()

    def get_performance_config(self) -> dict:
        """Get current performance configuration."""
        return {
            "sampling_rate": self.sampling_rate,
            "async_export": self.async_export,
            "batch_size": self.batch_size,
            "export_timeout": self.export_timeout,
            "circuit_breaker_enabled": self.circuit_breaker_enabled,
            "circuit_breaker_threshold": self.circuit_breaker_threshold,
            "circuit_breaker_window": self.circuit_breaker_window,
            "circuit_breaker_open": self._circuit_breaker_open,
            "circuit_breaker_failures": self._circuit_breaker_failures
        }

    def _detect_provider(self, model: str) -> str:
        """Detect the underlying provider based on model name."""
        if not model:
            return "unknown"

        model_lower = model.lower()

        for provider, pattern in self.PROVIDER_PATTERNS.items():
            if re.search(pattern, model_lower):
                return provider

        # Default to huggingface_hub for unrecognized patterns
        return "huggingface_hub"

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation for cost calculation."""
        if not text:
            return 0
        # Approximation: 1 token ≈ 4 characters for English text
        return len(text) // 4

    def _calculate_cost(
        self,
        provider: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        task: str = "text-generation"
    ) -> float:
        """Calculate cost based on provider, model, and token usage."""
        try:
            from genops.providers.huggingface_pricing import calculate_huggingface_cost
            return calculate_huggingface_cost(
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                task=task
            )
        except ImportError:
            # Fallback to basic estimation
            return self._fallback_cost_estimation(input_tokens, output_tokens, provider)

    def _fallback_cost_estimation(self, input_tokens: int, output_tokens: int, provider: str) -> float:
        """Fallback cost estimation when pricing module unavailable."""
        # Very rough estimates based on typical provider pricing
        cost_per_1k_tokens = {
            "openai": {"input": 0.0015, "output": 0.002},  # GPT-3.5 Turbo rates
            "anthropic": {"input": 0.0008, "output": 0.0024},  # Claude Haiku rates
            "cohere": {"input": 0.001, "output": 0.002},
            "huggingface_hub": {"input": 0.0001, "output": 0.0002},  # Much cheaper
        }.get(provider, {"input": 0.0005, "output": 0.001})  # Default rates

        input_cost = (input_tokens / 1000) * cost_per_1k_tokens["input"]
        output_cost = (output_tokens / 1000) * cost_per_1k_tokens["output"]

        return input_cost + output_cost

    def text_generation(self, prompt: str, **kwargs) -> Any:
        """Generate text with governance tracking."""
        # Check circuit breaker first
        if self._check_circuit_breaker():
            raise Exception("Circuit breaker is open - operation blocked")

        # Check sampling decision
        if not self._should_sample():
            logger.debug("Operation skipped due to sampling configuration")
            # Still make the API call but skip telemetry
            return self.client.text_generation(prompt, **kwargs)

        governance_attrs, request_attrs, api_kwargs = self._extract_attributes(kwargs)

        model = api_kwargs.get("model") or getattr(self.client, "model", "unknown")
        provider = self._detect_provider(model)

        # Estimate input tokens
        input_tokens = self._estimate_tokens(prompt)

        operation_name = "huggingface.text_generation"

        with self.telemetry.trace_operation(
            operation_name=operation_name,
            operation_type="ai.inference",
            provider="huggingface",
            model=model,
            **governance_attrs
        ) as span:
            try:
                # Add request attributes to span
                for attr, value in request_attrs.items():
                    span.set_attribute(f"genops.request.{attr}", value)

                # Set provider and task attributes
                span.set_attribute("genops.provider.detected", provider)
                span.set_attribute("genops.task.type", "text-generation")
                span.set_attribute("genops.tokens.input", input_tokens)

                # Make the API call with circuit breaker monitoring
                try:
                    response = self.client.text_generation(prompt, **api_kwargs)
                    # Record successful operation for circuit breaker
                    self._record_circuit_breaker_success()
                except Exception as api_error:
                    # Record failure for circuit breaker
                    self._record_circuit_breaker_failure()
                    raise api_error

                # Estimate output tokens from response
                if hasattr(response, 'generated_text'):
                    output_text = response.generated_text
                elif isinstance(response, str):
                    output_text = response
                else:
                    output_text = str(response)

                output_tokens = self._estimate_tokens(output_text)
                span.set_attribute("genops.tokens.output", output_tokens)

                # Calculate and record cost
                cost = self._calculate_cost(
                    provider=provider,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    task="text-generation"
                )

                if cost > 0:
                    self.telemetry.record_cost(
                        span=span,
                        cost=cost,
                        currency="USD",
                        provider=provider,
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens
                    )

                # Trigger async telemetry export if configured
                span_data = {
                    "operation": operation_name,
                    "provider": provider,
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }
                cost_data = {
                    "cost": cost,
                    "currency": "USD",
                    "provider": provider
                } if cost > 0 else None
                self._async_export_telemetry(span_data, cost_data)

                return response

            except Exception as e:
                span.set_attribute("genops.error.message", str(e))
                span.set_attribute("genops.error.type", type(e).__name__)
                logger.error(f"Hugging Face text generation failed: {e}")

                # Record circuit breaker failure if it's an API error
                if "circuit breaker" not in str(e).lower():
                    self._record_circuit_breaker_failure()

                raise

    def chat_completion(self, messages: list, **kwargs) -> Any:
        """Create chat completion with governance tracking."""
        # Check circuit breaker first
        if self._check_circuit_breaker():
            raise Exception("Circuit breaker is open - operation blocked")

        # Check sampling decision
        if not self._should_sample():
            logger.debug("Operation skipped due to sampling configuration")
            # Still make the API call but skip telemetry
            return self.client.chat.completions.create(messages=messages, **kwargs)

        governance_attrs, request_attrs, api_kwargs = self._extract_attributes(kwargs)

        model = api_kwargs.get("model") or getattr(self.client, "model", "unknown")
        provider = self._detect_provider(model)

        # Estimate input tokens from messages
        input_text = " ".join([
            msg.get("content", "") for msg in messages
            if isinstance(msg, dict) and msg.get("content")
        ])
        input_tokens = self._estimate_tokens(input_text)

        operation_name = "huggingface.chat.completion"

        with self.telemetry.trace_operation(
            operation_name=operation_name,
            operation_type="ai.inference",
            provider="huggingface",
            model=model,
            **governance_attrs
        ) as span:
            try:
                # Add request attributes to span
                for attr, value in request_attrs.items():
                    span.set_attribute(f"genops.request.{attr}", value)

                # Set provider and task attributes
                span.set_attribute("genops.provider.detected", provider)
                span.set_attribute("genops.task.type", "chat-completion")
                span.set_attribute("genops.tokens.input", input_tokens)
                span.set_attribute("genops.messages.count", len(messages))

                # Make the API call with circuit breaker monitoring
                try:
                    response = self.client.chat.completions.create(
                        messages=messages,
                        **api_kwargs
                    )
                    # Record successful operation for circuit breaker
                    self._record_circuit_breaker_success()
                except Exception as api_error:
                    # Record failure for circuit breaker
                    self._record_circuit_breaker_failure()
                    raise api_error

                # Extract output tokens from response
                output_tokens = 0
                if hasattr(response, 'choices') and response.choices:
                    choice = response.choices[0]
                    if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                        output_text = choice.message.content
                        output_tokens = self._estimate_tokens(output_text)
                elif hasattr(response, 'generated_text'):
                    output_tokens = self._estimate_tokens(response.generated_text)

                span.set_attribute("genops.tokens.output", output_tokens)

                # Calculate and record cost
                cost = self._calculate_cost(
                    provider=provider,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    task="chat-completion"
                )

                if cost > 0:
                    self.telemetry.record_cost(
                        span=span,
                        cost=cost,
                        currency="USD",
                        provider=provider,
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens
                    )

                # Trigger async telemetry export if configured
                span_data = {
                    "operation": operation_name,
                    "provider": provider,
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens
                }
                cost_data = {
                    "cost": cost,
                    "currency": "USD",
                    "provider": provider
                } if cost > 0 else None
                self._async_export_telemetry(span_data, cost_data)

                return response

            except Exception as e:
                span.set_attribute("genops.error.message", str(e))
                span.set_attribute("genops.error.type", type(e).__name__)
                logger.error(f"Hugging Face chat completion failed: {e}")

                # Record circuit breaker failure if it's an API error
                if "circuit breaker" not in str(e).lower():
                    self._record_circuit_breaker_failure()

                raise

    def feature_extraction(self, inputs: Union[str, list], **kwargs) -> Any:
        """Extract features/embeddings with governance tracking."""
        # Check circuit breaker first
        if self._check_circuit_breaker():
            raise Exception("Circuit breaker is open - operation blocked")

        # Check sampling decision
        if not self._should_sample():
            logger.debug("Operation skipped due to sampling configuration")
            # Still make the API call but skip telemetry
            return self.client.feature_extraction(inputs, **kwargs)

        governance_attrs, request_attrs, api_kwargs = self._extract_attributes(kwargs)

        model = api_kwargs.get("model") or getattr(self.client, "model", "unknown")
        provider = self._detect_provider(model)

        # Estimate input tokens
        if isinstance(inputs, str):
            input_tokens = self._estimate_tokens(inputs)
        elif isinstance(inputs, list):
            total_text = " ".join(str(item) for item in inputs)
            input_tokens = self._estimate_tokens(total_text)
        else:
            input_tokens = 0

        operation_name = "huggingface.feature_extraction"

        with self.telemetry.trace_operation(
            operation_name=operation_name,
            operation_type="ai.inference",
            provider="huggingface",
            model=model,
            **governance_attrs
        ) as span:
            try:
                # Add request attributes to span
                for attr, value in request_attrs.items():
                    span.set_attribute(f"genops.request.{attr}", value)

                # Set provider and task attributes
                span.set_attribute("genops.provider.detected", provider)
                span.set_attribute("genops.task.type", "feature-extraction")
                span.set_attribute("genops.tokens.input", input_tokens)

                # Make the API call with circuit breaker monitoring
                try:
                    response = self.client.feature_extraction(inputs, **api_kwargs)
                    # Record successful operation for circuit breaker
                    self._record_circuit_breaker_success()
                except Exception as api_error:
                    # Record failure for circuit breaker
                    self._record_circuit_breaker_failure()
                    raise api_error

                # For embeddings, output "tokens" could be embedding dimensions
                if hasattr(response, 'shape') and len(response.shape) > 1:
                    embedding_dims = response.shape[-1]
                    span.set_attribute("genops.embedding.dimensions", embedding_dims)

                # Calculate cost (typically lower for embeddings)
                cost = self._calculate_cost(
                    provider=provider,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=0,  # Embeddings don't generate text tokens
                    task="feature-extraction"
                )

                if cost > 0:
                    self.telemetry.record_cost(
                        span=span,
                        cost=cost,
                        currency="USD",
                        provider=provider,
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=0
                    )

                # Trigger async telemetry export if configured
                span_data = {
                    "operation": operation_name,
                    "provider": provider,
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": 0
                }
                cost_data = {
                    "cost": cost,
                    "currency": "USD",
                    "provider": provider
                } if cost > 0 else None
                self._async_export_telemetry(span_data, cost_data)

                return response

            except Exception as e:
                span.set_attribute("genops.error.message", str(e))
                span.set_attribute("genops.error.type", type(e).__name__)
                logger.error(f"Hugging Face feature extraction failed: {e}")

                # Record circuit breaker failure if it's an API error
                if "circuit breaker" not in str(e).lower():
                    self._record_circuit_breaker_failure()

                raise

    def text_to_image(self, prompt: str, **kwargs) -> Any:
        """Generate images from text with governance tracking."""
        # Check circuit breaker first
        if self._check_circuit_breaker():
            raise Exception("Circuit breaker is open - operation blocked")

        # Check sampling decision
        if not self._should_sample():
            logger.debug("Operation skipped due to sampling configuration")
            # Still make the API call but skip telemetry
            return self.client.text_to_image(prompt, **kwargs)

        governance_attrs, request_attrs, api_kwargs = self._extract_attributes(kwargs)

        model = api_kwargs.get("model") or getattr(self.client, "model", "unknown")
        provider = self._detect_provider(model)

        input_tokens = self._estimate_tokens(prompt)

        operation_name = "huggingface.text_to_image"

        with self.telemetry.trace_operation(
            operation_name=operation_name,
            operation_type="ai.inference",
            provider="huggingface",
            model=model,
            **governance_attrs
        ) as span:
            try:
                # Add request attributes to span
                for attr, value in request_attrs.items():
                    span.set_attribute(f"genops.request.{attr}", value)

                # Set provider and task attributes
                span.set_attribute("genops.provider.detected", provider)
                span.set_attribute("genops.task.type", "text-to-image")
                span.set_attribute("genops.tokens.input", input_tokens)

                # Make the API call with circuit breaker monitoring
                try:
                    response = self.client.text_to_image(prompt, **api_kwargs)
                    # Record successful operation for circuit breaker
                    self._record_circuit_breaker_success()
                except Exception as api_error:
                    # Record failure for circuit breaker
                    self._record_circuit_breaker_failure()
                    raise api_error

                # For images, we track generation count instead of output tokens
                image_count = 1
                if hasattr(response, '__len__'):
                    image_count = len(response)
                span.set_attribute("genops.images.generated", image_count)

                # Calculate cost for image generation
                cost = self._calculate_cost(
                    provider=provider,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=0,
                    task="text-to-image"
                )

                if cost > 0:
                    self.telemetry.record_cost(
                        span=span,
                        cost=cost,
                        currency="USD",
                        provider=provider,
                        model=model,
                        input_tokens=input_tokens,
                        output_tokens=0,
                        images_generated=image_count
                    )

                # Trigger async telemetry export if configured
                span_data = {
                    "operation": operation_name,
                    "provider": provider,
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": 0,
                    "images_generated": image_count
                }
                cost_data = {
                    "cost": cost,
                    "currency": "USD",
                    "provider": provider
                } if cost > 0 else None
                self._async_export_telemetry(span_data, cost_data)

                return response

            except Exception as e:
                span.set_attribute("genops.error.message", str(e))
                span.set_attribute("genops.error.type", type(e).__name__)
                logger.error(f"Hugging Face text-to-image failed: {e}")

                # Record circuit breaker failure if it's an API error
                if "circuit breaker" not in str(e).lower():
                    self._record_circuit_breaker_failure()

                raise

    def get_supported_tasks(self) -> list[str]:
        """Return list of supported AI tasks."""
        return sorted(self.SUPPORTED_TASKS)

    def detect_provider_for_model(self, model: str) -> str:
        """Public method to detect provider for a given model."""
        return self._detect_provider(model)

    def is_available(self) -> bool:
        """Check if Hugging Face Hub is available."""
        return HAS_HUGGINGFACE and self.client is not None


# Auto-instrumentation functions for zero-code setup
def instrument_huggingface(**config):
    """Auto-instrument Hugging Face InferenceClient with GenOps telemetry."""
    if not HAS_HUGGINGFACE:
        logger.warning("Hugging Face Hub not available for instrumentation")
        return False

    try:
        # Store original methods
        original_text_generation = InferenceClient.text_generation
        original_chat_completions_create = None
        original_feature_extraction = InferenceClient.feature_extraction
        original_text_to_image = InferenceClient.text_to_image

        # Try to get chat completions method (may not exist in all versions)
        if hasattr(InferenceClient, 'chat') and hasattr(InferenceClient.chat, 'completions'):
            original_chat_completions_create = InferenceClient.chat.completions.create

        def wrapped_text_generation(self, *args, **kwargs):
            adapter = GenOpsHuggingFaceAdapter(client=self)
            if args:
                return adapter.text_generation(args[0], **kwargs)
            return adapter.text_generation("", **kwargs)

        def wrapped_chat_completions_create(self, *args, **kwargs):
            adapter = GenOpsHuggingFaceAdapter(client=self._client if hasattr(self, '_client') else None)
            messages = kwargs.get('messages', args[0] if args else [])
            return adapter.chat_completion(messages, **kwargs)

        def wrapped_feature_extraction(self, *args, **kwargs):
            adapter = GenOpsHuggingFaceAdapter(client=self)
            inputs = args[0] if args else kwargs.get('inputs', "")
            return adapter.feature_extraction(inputs, **kwargs)

        def wrapped_text_to_image(self, *args, **kwargs):
            adapter = GenOpsHuggingFaceAdapter(client=self)
            prompt = args[0] if args else kwargs.get('prompt', "")
            return adapter.text_to_image(prompt, **kwargs)

        # Apply instrumentation
        InferenceClient.text_generation = wrapped_text_generation
        if original_chat_completions_create:
            InferenceClient.chat.completions.create = wrapped_chat_completions_create
        InferenceClient.feature_extraction = wrapped_feature_extraction
        InferenceClient.text_to_image = wrapped_text_to_image

        # Store original methods for potential restoration
        InferenceClient._genops_original_text_generation = original_text_generation
        InferenceClient._genops_original_chat_completions_create = original_chat_completions_create
        InferenceClient._genops_original_feature_extraction = original_feature_extraction
        InferenceClient._genops_original_text_to_image = original_text_to_image

        logger.info("Successfully instrumented Hugging Face InferenceClient")
        return True

    except Exception as e:
        logger.error(f"Failed to instrument Hugging Face: {e}")
        return False


def uninstrument_huggingface():
    """Remove GenOps instrumentation from Hugging Face InferenceClient."""
    if not HAS_HUGGINGFACE:
        return False

    try:
        # Restore original methods if they exist
        if hasattr(InferenceClient, '_genops_original_text_generation'):
            InferenceClient.text_generation = InferenceClient._genops_original_text_generation
            delattr(InferenceClient, '_genops_original_text_generation')

        if hasattr(InferenceClient, '_genops_original_chat_completions_create'):
            if hasattr(InferenceClient, 'chat') and hasattr(InferenceClient.chat, 'completions'):
                InferenceClient.chat.completions.create = InferenceClient._genops_original_chat_completions_create
            delattr(InferenceClient, '_genops_original_chat_completions_create')

        if hasattr(InferenceClient, '_genops_original_feature_extraction'):
            InferenceClient.feature_extraction = InferenceClient._genops_original_feature_extraction
            delattr(InferenceClient, '_genops_original_feature_extraction')

        if hasattr(InferenceClient, '_genops_original_text_to_image'):
            InferenceClient.text_to_image = InferenceClient._genops_original_text_to_image
            delattr(InferenceClient, '_genops_original_text_to_image')

        logger.info("Successfully removed Hugging Face instrumentation")
        return True

    except Exception as e:
        logger.error(f"Failed to uninstrument Hugging Face: {e}")
        return False


# Convenience function for creating instrumented client
def create_instrumented_client(**client_kwargs) -> GenOpsHuggingFaceAdapter:
    """Create a GenOps-instrumented Hugging Face client."""
    return GenOpsHuggingFaceAdapter(**client_kwargs)


# Import and expose cost aggregation functionality
try:
    from genops.providers.huggingface_cost_aggregator import (
        HuggingFaceCallCost,
        HuggingFaceCostAggregator,
        HuggingFaceCostContext,
        HuggingFaceCostSummary,
        create_huggingface_cost_context,
        get_cost_aggregator,
    )
    from genops.providers.huggingface_workflow import (
        ProductionWorkflowSpan,
        production_workflow_context,
    )

    # Export all the components
    __all__ = [
        'GenOpsHuggingFaceAdapter',
        'instrument_huggingface',
        'uninstrument_huggingface',
        'create_instrumented_client',
        'HuggingFaceCallCost',
        'HuggingFaceCostSummary',
        'HuggingFaceCostAggregator',
        'HuggingFaceCostContext',
        'create_huggingface_cost_context',
        'get_cost_aggregator',
        'production_workflow_context',
        'ProductionWorkflowSpan',
    ]

except ImportError as e:
    logger.debug(f"Advanced components not available: {e}")
    __all__ = [
        'GenOpsHuggingFaceAdapter',
        'instrument_huggingface',
        'uninstrument_huggingface',
        'create_instrumented_client',
    ]
