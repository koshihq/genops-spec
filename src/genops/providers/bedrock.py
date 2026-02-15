#!/usr/bin/env python3
"""
GenOps AWS Bedrock Provider Integration

This module provides comprehensive AWS Bedrock integration for GenOps AI governance,
cost intelligence, and observability. It follows the established GenOps provider
pattern for consistent developer experience across all AI platforms.

Features:
- Multi-model support (Claude, Titan, Jurassic, Command, Llama, Cohere)
- Zero-code auto-instrumentation with instrument_bedrock()
- Regional cost optimization and intelligent model selection
- Streaming response support for real-time applications
- AWS IAM authentication with cross-account support
- Comprehensive governance and audit trail integration

Example usage:

    # Zero-code auto-instrumentation
    from genops.providers.bedrock import instrument_bedrock
    instrument_bedrock()

    # Your existing Bedrock code works unchanged with automatic governance
    import boto3
    bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
    response = bedrock.invoke_model(...)  # Now tracked with GenOps!

    # Manual adapter usage for advanced control
    from genops.providers.bedrock import GenOpsBedrockAdapter

    adapter = GenOpsBedrockAdapter(region_name='us-east-1')
    response = adapter.text_generation(
        prompt="Explain quantum computing",
        model_id="anthropic.claude-3-haiku-20240307-v1:0",
        team="research-team",
        project="quantum-ai",
        customer_id="enterprise-123"
    )
"""

import json
import logging
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Optional, Union

try:
    import boto3
    from botocore.exceptions import (  # noqa: F401
        BotoCoreError,
        ClientError,
        NoCredentialsError,
    )

    BEDROCK_AVAILABLE = True
except ImportError:
    BEDROCK_AVAILABLE = False

try:
    from genops.core.base_provider import BaseProvider, OperationContext
    from genops.core.telemetry import GenOpsTelemetry
    from genops.providers.bedrock_pricing import (
        BEDROCK_MODELS,
        calculate_bedrock_cost,
        compare_bedrock_models,  # noqa: F401
        get_bedrock_model_info,  # noqa: F401
    )
    from genops.providers.bedrock_validation import (
        BedrockValidationResult,
        validate_bedrock_setup,
    )
    from genops.providers.bedrock_validation import (
        print_validation_result as _print_validation_result,
    )

    GENOPS_AVAILABLE = True
except ImportError:
    GENOPS_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class BedrockOperationResult:
    """Result from a Bedrock operation with full telemetry context."""

    content: str
    model_id: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    region: str
    operation_id: str
    governance_attributes: dict[str, str]
    raw_response: Optional[dict] = None


class GenOpsBedrockAdapter(BaseProvider):
    """
    GenOps adapter for AWS Bedrock with comprehensive AI governance.

    This adapter provides unified instrumentation for all Bedrock models
    while maintaining the native AWS SDK experience. It automatically
    captures costs, performance metrics, and governance attributes.
    """

    def __init__(
        self,
        region_name: str = "us-east-1",
        profile_name: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        enable_streaming: bool = True,
        default_model: str = "anthropic.claude-3-haiku-20240307-v1:0",
        **kwargs,
    ):
        """
        Initialize the GenOps Bedrock adapter.

        Args:
            region_name: AWS region for Bedrock operations
            profile_name: AWS profile name for authentication
            aws_access_key_id: AWS access key (optional, can use IAM roles)
            aws_secret_access_key: AWS secret key (optional)
            aws_session_token: AWS session token for temporary credentials
            endpoint_url: Custom endpoint URL for Bedrock (for testing)
            enable_streaming: Enable streaming response support
            default_model: Default model ID for operations
            **kwargs: Additional arguments passed to boto3 client
        """
        super().__init__()

        if not BEDROCK_AVAILABLE:
            raise ImportError(
                "AWS Bedrock dependencies not available. Install with: "
                "pip install boto3 botocore"
            )

        if not GENOPS_AVAILABLE:
            logger.warning("GenOps core not available, running in basic mode")

        self.region_name = region_name
        self.profile_name = profile_name
        self.enable_streaming = enable_streaming
        self.default_model = default_model

        # Initialize AWS session and clients
        session_kwargs = {}
        if profile_name:
            session_kwargs["profile_name"] = profile_name

        self.session = boto3.Session(**session_kwargs)

        client_kwargs = {"region_name": region_name, **kwargs}

        if aws_access_key_id:
            client_kwargs["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            client_kwargs["aws_secret_access_key"] = aws_secret_access_key
        if aws_session_token:
            client_kwargs["aws_session_token"] = aws_session_token
        if endpoint_url:
            client_kwargs["endpoint_url"] = endpoint_url

        try:
            self.bedrock_runtime = self.session.client(
                "bedrock-runtime", **client_kwargs
            )
            self.bedrock_client = self.session.client("bedrock", **client_kwargs)
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock clients: {e}")
            raise

        # Initialize telemetry
        if GENOPS_AVAILABLE:
            self.telemetry = GenOpsTelemetry()
        else:
            self.telemetry = None

        logger.info(f"GenOps Bedrock adapter initialized for region: {region_name}")

    def is_available(self) -> bool:
        """Check if Bedrock is available and accessible."""
        if not BEDROCK_AVAILABLE:
            return False

        try:
            # Try to list foundation models as availability check
            response = self.bedrock_client.list_foundation_models()
            return len(response.get("modelSummaries", [])) > 0
        except Exception as e:
            logger.warning(f"Bedrock availability check failed: {e}")
            return False

    def get_supported_models(self) -> list[str]:
        """Get list of supported Bedrock model IDs."""
        try:
            response = self.bedrock_client.list_foundation_models()
            return [model["modelId"] for model in response.get("modelSummaries", [])]
        except Exception as e:
            logger.warning(f"Failed to fetch supported models: {e}")
            return list(BEDROCK_MODELS.keys())

    def get_supported_tasks(self) -> list[str]:
        """Get list of supported AI tasks."""
        return [
            "text-generation",
            "chat-completion",
            "text-embedding",
            "text-summarization",
            "question-answering",
            "content-moderation",
            "streaming-generation",
        ]

    def detect_model_provider(self, model_id: str) -> str:
        """Detect the underlying provider for a Bedrock model ID."""
        model_id_lower = model_id.lower()

        if "anthropic" in model_id_lower or "claude" in model_id_lower:
            return "anthropic"
        elif "amazon" in model_id_lower or "titan" in model_id_lower:
            return "amazon"
        elif "ai21" in model_id_lower or "jurassic" in model_id_lower:
            return "ai21"
        elif "cohere" in model_id_lower or "command" in model_id_lower:
            return "cohere"
        elif "meta" in model_id_lower or "llama" in model_id_lower:
            return "meta"
        elif "mistral" in model_id_lower:
            return "mistral"
        else:
            return "bedrock"

    def _create_operation_context(  # type: ignore[override]
        self, operation_name: str, model_id: str, **governance_attrs
    ) -> OperationContext:
        """Create operation context with Bedrock-specific attributes."""
        operation_id = str(uuid.uuid4())

        context = OperationContext(  # type: ignore[call-arg]
            operation_id=operation_id,
            operation_name=operation_name,
            provider="bedrock",
            model=model_id,
            region=self.region_name,
            **governance_attrs,
        )

        return context

    def _calculate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        This is a rough approximation. For production use, consider
        integrating with model-specific tokenizers.
        """
        # Rough approximation: ~4 characters per token for most models
        return max(1, len(text) // 4)

    def _extract_response_content(
        self, response: dict, model_id: str
    ) -> tuple[str, int]:
        """
        Extract content and output tokens from Bedrock response.

        Different models have different response formats, so we handle each.
        """
        try:
            response_body = json.loads(response["body"].read())
        except Exception:
            response_body = response.get("body", {})

        provider = self.detect_model_provider(model_id)

        if provider == "anthropic":
            # Claude models
            content = response_body.get("completion", "")
            output_tokens = response_body.get("usage", {}).get(
                "output_tokens", self._calculate_tokens(content)
            )
        elif provider == "amazon":
            # Titan models
            results = response_body.get("results", [])
            content = results[0].get("outputText", "") if results else ""
            output_tokens = response_body.get(
                "inputTextTokenCount", self._calculate_tokens(content)
            )
        elif provider == "ai21":
            # Jurassic models
            completions = response_body.get("completions", [])
            content = (
                completions[0].get("data", {}).get("text", "") if completions else ""
            )
            output_tokens = self._calculate_tokens(content)
        elif provider == "cohere":
            # Command models
            generations = response_body.get("generations", [])
            content = generations[0].get("text", "") if generations else ""
            output_tokens = self._calculate_tokens(content)
        elif provider == "meta":
            # Llama models
            content = response_body.get("generation", "")
            output_tokens = response_body.get(
                "generation_token_count", self._calculate_tokens(content)
            )
        else:
            # Generic handling
            content = str(response_body)
            output_tokens = self._calculate_tokens(content)

        return content, output_tokens

    def text_generation(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        max_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop_sequences: Optional[list[str]] = None,
        stream: bool = False,
        **governance_attrs,
    ) -> Union[BedrockOperationResult, Iterator[str]]:
        """
        Generate text using Bedrock models with comprehensive governance.

        Args:
            prompt: Input text prompt
            model_id: Bedrock model ID (defaults to instance default)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
            top_p: Top-p sampling parameter
            stop_sequences: List of stop sequences
            stream: Enable streaming response
            **governance_attrs: Governance attributes (team, project, customer_id, etc.)

        Returns:
            BedrockOperationResult with full telemetry or streaming iterator
        """
        model_id = model_id or self.default_model
        operation_start = time.time()

        # Create operation context
        context = self._create_operation_context(
            "bedrock.text_generation", model_id, **governance_attrs
        )

        if self.telemetry:
            with self.telemetry.trace_operation(
                operation_name=f"bedrock.text_generation.{model_id}",
                **context.to_dict(),
            ) as span:
                return self._execute_text_generation(
                    span,
                    context,
                    prompt,
                    model_id,
                    max_tokens,
                    temperature,
                    top_p,
                    stop_sequences,
                    stream,
                    operation_start,
                )
        else:
            return self._execute_text_generation(
                None,
                context,
                prompt,
                model_id,
                max_tokens,
                temperature,
                top_p,
                stop_sequences,
                stream,
                operation_start,
            )

    def _execute_text_generation(
        self,
        span,
        context: OperationContext,
        prompt: str,
        model_id: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_sequences: Optional[list[str]],
        stream: bool,
        operation_start: float,
    ) -> Union[BedrockOperationResult, Iterator[str]]:
        """Execute text generation with telemetry."""

        try:
            # Prepare model-specific request body
            request_body = self._prepare_text_generation_body(
                prompt, model_id, max_tokens, temperature, top_p, stop_sequences
            )

            # Set telemetry attributes
            if span:
                span.set_attribute("bedrock.model_id", model_id)
                span.set_attribute("bedrock.region", self.region_name)
                span.set_attribute("bedrock.max_tokens", max_tokens)
                span.set_attribute("bedrock.temperature", temperature)
                span.set_attribute("bedrock.stream", stream)

            input_tokens = self._calculate_tokens(prompt)

            if stream and self.enable_streaming:
                return self._stream_text_generation(
                    span, context, model_id, request_body, input_tokens, operation_start
                )
            else:
                return self._invoke_text_generation(
                    span, context, model_id, request_body, input_tokens, operation_start
                )

        except Exception as e:
            if span:
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
            logger.error(f"Bedrock text generation failed: {e}")
            raise

    def _prepare_text_generation_body(
        self,
        prompt: str,
        model_id: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_sequences: Optional[list[str]],
    ) -> str:
        """Prepare model-specific request body."""

        provider = self.detect_model_provider(model_id)

        if provider == "anthropic":
            # Claude models
            body = {
                "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                "max_tokens_to_sample": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
            if stop_sequences:
                body["stop_sequences"] = stop_sequences

        elif provider == "amazon":
            # Titan models
            body = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": temperature,
                    "topP": top_p,
                },
            }
            if stop_sequences:
                body["textGenerationConfig"]["stopSequences"] = stop_sequences

        elif provider == "ai21":
            # Jurassic models
            body = {
                "prompt": prompt,
                "maxTokens": max_tokens,
                "temperature": temperature,
                "topP": top_p,
            }
            if stop_sequences:
                body["stopSequences"] = stop_sequences

        elif provider == "cohere":
            # Command models
            body = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "p": top_p,
            }
            if stop_sequences:
                body["stop_sequences"] = stop_sequences

        elif provider == "meta":
            # Llama models
            body = {
                "prompt": prompt,
                "max_gen_len": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }

        else:
            # Generic fallback
            body = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }

        return json.dumps(body)

    def _invoke_text_generation(
        self,
        span,
        context: OperationContext,
        model_id: str,
        request_body: str,
        input_tokens: int,
        operation_start: float,
    ) -> BedrockOperationResult:
        """Invoke non-streaming text generation."""

        try:
            response = self.bedrock_runtime.invoke_model(
                modelId=model_id,
                body=request_body,
                contentType="application/json",
                accept="application/json",
            )

            # Extract response content and tokens
            content, output_tokens = self._extract_response_content(response, model_id)

            # Calculate metrics
            latency_ms = (time.time() - operation_start) * 1000
            cost_usd = calculate_bedrock_cost(
                model_id=model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                region=self.region_name,
            )

            # Set telemetry metrics
            if span:
                span.set_attribute("bedrock.input_tokens", input_tokens)
                span.set_attribute("bedrock.output_tokens", output_tokens)
                span.set_attribute("bedrock.latency_ms", latency_ms)
                span.set_attribute("bedrock.cost_usd", cost_usd)
                span.set_attribute("bedrock.success", True)

            # Create result
            result = BedrockOperationResult(
                content=content,
                model_id=model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cost_usd=cost_usd,
                region=self.region_name,
                operation_id=context.operation_id,
                governance_attributes=context.governance_attributes,
                raw_response=response,
            )

            return result

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))

            if span:
                span.set_attribute("error", True)
                span.set_attribute("error.type", error_code)
                span.set_attribute("error.message", error_message)

            logger.error(f"Bedrock API error [{error_code}]: {error_message}")
            raise

    def _stream_text_generation(
        self,
        span,
        context: OperationContext,
        model_id: str,
        request_body: str,
        input_tokens: int,
        operation_start: float,
    ) -> Iterator[str]:
        """Stream text generation with telemetry tracking."""

        try:
            response = self.bedrock_runtime.invoke_model_with_response_stream(
                modelId=model_id,
                body=request_body,
                contentType="application/json",
                accept="application/json",
            )

            output_tokens = 0
            full_content = ""

            for event in response["body"]:
                if "chunk" in event:
                    chunk_data = json.loads(event["chunk"]["bytes"])

                    # Extract chunk content based on provider
                    chunk_text = self._extract_chunk_content(chunk_data, model_id)

                    if chunk_text:
                        full_content += chunk_text
                        output_tokens += self._calculate_tokens(chunk_text)
                        yield chunk_text

            # Final telemetry update
            latency_ms = (time.time() - operation_start) * 1000
            cost_usd = calculate_bedrock_cost(
                model_id=model_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                region=self.region_name,
            )

            if span:
                span.set_attribute("bedrock.input_tokens", input_tokens)
                span.set_attribute("bedrock.output_tokens", output_tokens)
                span.set_attribute("bedrock.latency_ms", latency_ms)
                span.set_attribute("bedrock.cost_usd", cost_usd)
                span.set_attribute("bedrock.success", True)
                span.set_attribute("bedrock.streaming", True)

        except Exception as e:
            if span:
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))
            logger.error(f"Bedrock streaming failed: {e}")
            raise

    def _extract_chunk_content(self, chunk_data: dict, model_id: str) -> str:
        """Extract content from streaming chunk based on model provider."""
        provider = self.detect_model_provider(model_id)

        if provider == "anthropic":
            return chunk_data.get("completion", "")
        elif provider == "amazon":
            return chunk_data.get("outputText", "")
        elif provider == "cohere":
            generations = chunk_data.get("generations", [])
            return generations[0].get("text", "") if generations else ""
        else:
            # Generic extraction
            return chunk_data.get("text", chunk_data.get("content", ""))

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        model_id: Optional[str] = None,
        max_tokens: int = 200,
        temperature: float = 0.7,
        **governance_attrs,
    ) -> BedrockOperationResult:
        """
        Perform chat completion using Bedrock models.

        Converts chat messages to appropriate prompt format for each model.
        """
        model_id = model_id or self.default_model

        # Convert messages to prompt format
        prompt = self._messages_to_prompt(messages, model_id)

        return self.text_generation(  # type: ignore
            prompt=prompt,
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            **governance_attrs,
        )

    def _messages_to_prompt(self, messages: list[dict[str, str]], model_id: str) -> str:
        """Convert chat messages to model-specific prompt format."""
        provider = self.detect_model_provider(model_id)

        if provider == "anthropic":
            # Claude format
            prompt_parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if role == "system":
                    prompt_parts.append(f"System: {content}")
                elif role == "user":
                    prompt_parts.append(f"Human: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")

            return "\n\n" + "\n\n".join(prompt_parts) + "\n\nAssistant:"

        else:
            # Generic format for other models
            conversation = []
            for msg in messages:
                role = msg.get("role", "user").title()
                content = msg.get("content", "")
                conversation.append(f"{role}: {content}")

            return "\n".join(conversation) + "\nAssistant:"

    def get_performance_config(self) -> dict[str, Any]:
        """Get current performance configuration."""
        return {
            "provider": "bedrock",
            "region": self.region_name,
            "streaming_enabled": self.enable_streaming,
            "default_model": self.default_model,
            "telemetry_enabled": self.telemetry is not None,
            "profile_name": self.profile_name,
        }

    def validate_setup(self) -> dict[str, Any]:
        """
        Validate that the Bedrock adapter is properly configured.

        Returns:
            Dict containing validation results with keys:
            - 'valid': bool indicating if setup is valid
            - 'errors': list of error messages
            - 'warnings': list of warning messages
            - 'recommendations': list of recommendations
        """
        if not BEDROCK_AVAILABLE:
            return {
                "valid": False,
                "errors": [
                    "Bedrock dependencies not available - install with: pip install boto3"
                ],
                "warnings": [],
                "recommendations": ["Run: pip install genops-ai[bedrock]"],
            }

        try:
            # Use the module-level validation function
            result = validate_bedrock_setup()

            # Convert to the expected format
            return {
                "valid": result.success,
                "errors": result.errors,
                "warnings": result.warnings,
                "recommendations": result.recommendations,
            }
        except Exception as e:
            return {
                "valid": False,
                "errors": [f"Validation failed: {str(e)}"],
                "warnings": [],
                "recommendations": ["Check AWS credentials and region configuration"],
            }


# Auto-instrumentation support
_original_invoke_model = None
_original_invoke_model_with_response_stream = None
_instrumentation_enabled = False


def instrument_bedrock():
    """
    Enable zero-code auto-instrumentation for AWS Bedrock.

    This function patches boto3 Bedrock client methods to automatically
    add GenOps telemetry without requiring any code changes.

    Example:
        from genops.providers.bedrock import instrument_bedrock
        instrument_bedrock()

        # Your existing Bedrock code now automatically has governance
        import boto3
        bedrock = boto3.client('bedrock-runtime')
        response = bedrock.invoke_model(...)  # Automatically tracked!
    """
    global \
        _instrumentation_enabled, \
        _original_invoke_model, \
        _original_invoke_model_with_response_stream

    if _instrumentation_enabled:
        logger.info("Bedrock auto-instrumentation already enabled")
        return

    if not BEDROCK_AVAILABLE:
        logger.warning("Cannot enable Bedrock instrumentation - boto3 not available")
        return

    try:
        import boto3.session

        # Store original methods
        original_client = boto3.session.Session.client

        def instrumented_client(self, service_name, *args, **kwargs):
            """Instrumented client factory that adds GenOps tracking."""
            client = original_client(self, service_name, *args, **kwargs)

            if service_name == "bedrock-runtime":
                # Wrap the invoke_model method
                original_invoke = client.invoke_model
                original_invoke_stream = client.invoke_model_with_response_stream

                def instrumented_invoke_model(*args, **kwargs):
                    # Extract basic info for telemetry
                    model_id = kwargs.get("modelId", args[0] if args else "unknown")

                    if GENOPS_AVAILABLE:
                        telemetry = GenOpsTelemetry()
                        with telemetry.trace_operation(
                            operation_name=f"bedrock.invoke_model.{model_id}",
                            provider="bedrock",
                            model=model_id,
                        ) as span:
                            span.set_attribute("bedrock.auto_instrumented", True)
                            return original_invoke(*args, **kwargs)
                    else:
                        return original_invoke(*args, **kwargs)

                def instrumented_invoke_model_stream(*args, **kwargs):
                    # Extract basic info for telemetry
                    model_id = kwargs.get("modelId", args[0] if args else "unknown")

                    if GENOPS_AVAILABLE:
                        telemetry = GenOpsTelemetry()
                        with telemetry.trace_operation(
                            operation_name=f"bedrock.invoke_model_stream.{model_id}",
                            provider="bedrock",
                            model=model_id,
                        ) as span:
                            span.set_attribute("bedrock.auto_instrumented", True)
                            span.set_attribute("bedrock.streaming", True)
                            return original_invoke_stream(*args, **kwargs)
                    else:
                        return original_invoke_stream(*args, **kwargs)

                client.invoke_model = instrumented_invoke_model
                client.invoke_model_with_response_stream = (
                    instrumented_invoke_model_stream
                )

            return client

        # Apply instrumentation
        boto3.session.Session.client = instrumented_client
        _instrumentation_enabled = True

        logger.info("✅ Bedrock auto-instrumentation enabled successfully")
        logger.info(
            "   All boto3 bedrock-runtime client calls will now include GenOps telemetry"
        )

    except Exception as e:
        logger.error(f"Failed to enable Bedrock auto-instrumentation: {e}")
        raise


def auto_instrument_bedrock():
    """
    Alias for instrument_bedrock() for compatibility with other providers.

    Enables zero-code auto-instrumentation for AWS Bedrock.
    """
    return instrument_bedrock()


def validate_setup() -> "BedrockValidationResult":
    """
    Validate Bedrock setup and configuration.

    Returns comprehensive validation result with actionable feedback.
    """
    if not BEDROCK_AVAILABLE:
        from types import SimpleNamespace

        return SimpleNamespace(  # type: ignore[return-value]
            success=False,
            errors=[
                "Bedrock dependencies not available - install with: pip install boto3"
            ],
            warnings=[],
            recommendations=["Run: pip install genops-ai[bedrock]"],
        )

    return validate_bedrock_setup()


def print_validation_result(
    result: "BedrockValidationResult", detailed: bool = False
) -> None:
    """
    Print validation result in user-friendly format.

    Wrapper function to maintain consistent API with other providers.
    """
    if BEDROCK_AVAILABLE:
        _print_validation_result(result, detailed=detailed)
    else:
        # Fallback for when bedrock validation is not available
        if hasattr(result, "success"):
            if result.success:
                print("✅ Bedrock setup validation successful")
            else:
                print("❌ Bedrock setup validation failed")
                for error in getattr(result, "errors", []):
                    print(f"   - {error}")
        else:
            print(f"Validation result: {result}")


def quick_validate() -> bool:
    """Quick validation check for Bedrock setup."""
    try:
        result = validate_setup()
        if result.success:
            print("✅ Bedrock setup validation successful")
            return True
        else:
            print(f"❌ Bedrock setup validation failed: {result.errors}")
            return False
    except Exception as e:
        print(f"❌ Bedrock validation error: {e}")
        return False


# Export main classes and functions
__all__ = [
    "GenOpsBedrockAdapter",
    "BedrockOperationResult",
    "instrument_bedrock",
    "auto_instrument_bedrock",
    "validate_setup",
    "print_validation_result",
    "quick_validate",
    "BEDROCK_AVAILABLE",
]
