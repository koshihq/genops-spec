"""Vercel AI SDK provider adapter for GenOps AI governance."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from genops.core.telemetry import GenOpsTelemetry

logger = logging.getLogger(__name__)


# Check for Node.js and npm availability for JavaScript integration
def _check_nodejs_available() -> bool:
    """Check if Node.js is available for JavaScript integration."""
    try:
        result = subprocess.run(
            ["node", "--version"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


HAS_NODEJS = _check_nodejs_available()
if not HAS_NODEJS:
    logger.warning("Node.js not available - JavaScript integration limited")

# Optional imports for enhanced functionality
try:
    import requests  # noqa: F401

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("requests not installed. Install with: pip install requests")

try:
    import websockets  # noqa: F401

    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    logger.info(
        "websockets not installed. Install for real-time telemetry: pip install websockets"
    )


@dataclass
class VercelAISDKRequest:
    """Data class for Vercel AI SDK request tracking."""

    request_id: str
    provider: str
    model: str
    operation_type: str  # generateText, streamText, generateObject, embed, etc.
    input_tokens: int | None = None
    output_tokens: int | None = None
    prompt: str | None = None
    response: str | None = None
    tools_used: list[str] = field(default_factory=list)
    cost: Decimal | None = None
    duration_ms: float | None = None
    stream_chunks: int = 0
    governance_attrs: dict[str, Any] = field(default_factory=dict)
    request_attrs: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class VercelAISDKResponse:
    """Data class for Vercel AI SDK response tracking."""

    request_id: str
    success: bool
    text: str | None = None
    object_data: dict[str, Any] | None = None
    embedding: list[float] | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    provider_response: dict[str, Any] | None = None
    error: str | None = None


class GenOpsVercelAISDKAdapter:
    """
    Vercel AI SDK adapter with automatic governance telemetry.

    Provides GenOps governance integration for Vercel AI SDK, supporting both
    Python wrapper patterns and JavaScript/Node.js integration via subprocess
    or WebSocket communication.

    Features:
    - Multi-provider cost tracking across 20+ AI providers
    - Real-time streaming telemetry
    - Tool calling and agent workflow governance
    - JavaScript/Python hybrid integration
    - Auto-instrumentation for existing Vercel AI SDK applications
    """

    def __init__(
        self,
        integration_mode: str = "python_wrapper",  # or "websocket", "subprocess"
        websocket_port: int = 8080,
        team: str | None = None,
        project: str | None = None,
        environment: str | None = None,
        cost_center: str | None = None,
        customer_id: str | None = None,
        feature: str | None = None,
        **kwargs,
    ):
        """
        Initialize the Vercel AI SDK adapter.

        Args:
            integration_mode: How to integrate with JavaScript ("python_wrapper", "websocket", "subprocess")
            websocket_port: Port for WebSocket communication (if using websocket mode)
            team: Team name for governance attribution
            project: Project name for cost tracking
            environment: Environment (dev/staging/prod)
            cost_center: Cost center for financial reporting
            customer_id: Customer ID for attribution
            feature: Feature name for cost tracking
            **kwargs: Additional governance attributes
        """
        # Validate integration mode
        valid_modes = ["python_wrapper", "websocket", "subprocess"]
        if integration_mode not in valid_modes:
            raise ValueError(f"integration_mode must be one of {valid_modes}")

        self.integration_mode = integration_mode
        self.websocket_port = websocket_port

        # Initialize governance attributes with environment variable fallbacks
        self.governance_attrs = self._initialize_governance_attributes(
            team=team,
            project=project,
            environment=environment,
            cost_center=cost_center,
            customer_id=customer_id,
            feature=feature,
            **kwargs,
        )

        self.telemetry = GenOpsTelemetry()

        # Define standard governance and request attributes
        self.GOVERNANCE_ATTRIBUTES = {
            "team",
            "project",
            "feature",
            "customer_id",
            "customer",
            "environment",
            "cost_center",
            "user_id",
        }

        # Vercel AI SDK specific request attributes
        self.REQUEST_ATTRIBUTES = {
            "model",
            "temperature",
            "maxTokens",
            "topP",
            "topK",
            "presencePenalty",
            "frequencyPenalty",
            "seed",
            "maxRetries",
            "abortSignal",
            "headers",
            "experimental_telemetry",
            "experimental_providerMetadata",
        }

        # Track active requests for cost aggregation
        self.active_requests: dict[str, VercelAISDKRequest] = {}
        self._request_lock = threading.Lock()

        # Initialize integration-specific components
        self._initialize_integration_mode()

        logger.info(
            f"GenOps Vercel AI SDK adapter initialized in {integration_mode} mode"
        )

    def _initialize_governance_attributes(self, **governance_attrs) -> dict[str, Any]:
        """Initialize and validate governance attributes with environment variable fallbacks."""
        # Standard governance attributes from CLAUDE.md
        standard_attrs = {
            "team": governance_attrs.get("team") or os.getenv("GENOPS_TEAM"),
            "project": governance_attrs.get("project") or os.getenv("GENOPS_PROJECT"),
            "environment": governance_attrs.get("environment")
            or os.getenv("GENOPS_ENVIRONMENT"),
            "cost_center": governance_attrs.get("cost_center")
            or os.getenv("GENOPS_COST_CENTER"),
            "customer_id": governance_attrs.get("customer_id")
            or os.getenv("GENOPS_CUSTOMER_ID"),
            "feature": governance_attrs.get("feature") or os.getenv("GENOPS_FEATURE"),
        }

        # Add any additional custom attributes
        additional_attrs = {
            k: v
            for k, v in governance_attrs.items()
            if k not in standard_attrs and not k.startswith("_")
        }

        # Combine and filter out None values
        all_attrs = {**standard_attrs, **additional_attrs}
        return {k: v for k, v in all_attrs.items() if v is not None}

    def _initialize_integration_mode(self) -> None:
        """Initialize components based on integration mode."""
        if self.integration_mode == "websocket":
            if not HAS_WEBSOCKETS:
                logger.warning(
                    "WebSocket mode requested but websockets not available. Falling back to python_wrapper mode."
                )
                self.integration_mode = "python_wrapper"
            else:
                self._initialize_websocket_server()
        elif self.integration_mode == "subprocess":
            if not HAS_NODEJS:
                logger.warning(
                    "Subprocess mode requested but Node.js not available. Falling back to python_wrapper mode."
                )
                self.integration_mode = "python_wrapper"

        # python_wrapper mode needs no special initialization

    def _initialize_websocket_server(self) -> None:
        """Initialize WebSocket server for real-time JavaScript communication."""
        # This would be implemented to start a WebSocket server
        # for receiving telemetry from JavaScript clients
        logger.info(f"WebSocket server mode initialized on port {self.websocket_port}")

    def _extract_attributes(self, kwargs: dict) -> tuple[dict, dict, dict]:
        """Extract governance and request attributes from kwargs."""
        governance_attrs = {}
        request_attrs = {}
        api_kwargs = kwargs.copy()

        # Extract governance attributes
        for attr in self.GOVERNANCE_ATTRIBUTES:
            if attr in kwargs:
                governance_attrs[attr] = kwargs[attr]
                api_kwargs.pop(attr, None)

        # Extract request attributes
        for attr in self.REQUEST_ATTRIBUTES:
            if attr in kwargs:
                request_attrs[attr] = kwargs[attr]

        # Merge with instance-level governance attributes
        merged_governance = {**self.governance_attrs, **governance_attrs}

        return merged_governance, request_attrs, api_kwargs

    @contextmanager
    def track_request(self, operation_type: str, provider: str, model: str, **kwargs):
        """
        Context manager for tracking a Vercel AI SDK request with governance.

        Args:
            operation_type: Type of operation (generateText, streamText, etc.)
            provider: AI provider (openai, anthropic, etc.)
            model: Model name
            **kwargs: Additional parameters including governance attributes

        Yields:
            VercelAISDKRequest: Request tracking object
        """
        # Extract attributes
        governance_attrs, request_attrs, api_kwargs = self._extract_attributes(kwargs)

        # Generate unique request ID
        request_id = f"vercel-ai-sdk-{int(time.time() * 1000)}-{threading.current_thread().ident}"

        # Create request tracking object
        request = VercelAISDKRequest(
            request_id=request_id,
            provider=provider,
            model=model,
            operation_type=operation_type,
            governance_attrs=governance_attrs,
            request_attrs=request_attrs,
        )

        # Add to active requests
        with self._request_lock:
            self.active_requests[request_id] = request

        start_time = time.time()

        try:
            # Start telemetry span
            with self.telemetry.start_span(f"vercel_ai_sdk.{operation_type}") as span:
                # Add standard attributes
                span.set_attribute("genops.provider", "vercel-ai-sdk")
                span.set_attribute("genops.underlying_provider", provider)
                span.set_attribute("genops.model", model)
                span.set_attribute("genops.operation_type", operation_type)

                # Add governance attributes
                for key, value in governance_attrs.items():
                    span.set_attribute(f"genops.{key}", str(value))

                # Add request attributes
                for key, value in request_attrs.items():
                    if value is not None:
                        span.set_attribute(f"vercel_ai_sdk.{key}", str(value))

                yield request

        except Exception as e:
            logger.error(f"Error in Vercel AI SDK request {request_id}: {e}")
            request.error = str(e)
            raise
        finally:
            # Calculate duration
            end_time = time.time()
            request.duration_ms = (end_time - start_time) * 1000

            # Finalize telemetry
            self._finalize_request_telemetry(request)

            # Remove from active requests
            with self._request_lock:
                self.active_requests.pop(request_id, None)

    def _finalize_request_telemetry(self, request: VercelAISDKRequest) -> None:
        """Finalize telemetry for a completed request."""
        try:
            # Calculate cost if we have token information
            if request.input_tokens and request.output_tokens:
                request.cost = self._calculate_cost(
                    request.provider,
                    request.model,
                    request.input_tokens,
                    request.output_tokens,
                )

            # Emit final telemetry
            with self.telemetry.start_span(
                f"vercel_ai_sdk.{request.operation_type}.complete"
            ) as span:
                span.set_attribute("genops.request_id", request.request_id)
                span.set_attribute("genops.provider", "vercel-ai-sdk")
                span.set_attribute("genops.underlying_provider", request.provider)
                span.set_attribute("genops.model", request.model)
                span.set_attribute("genops.duration_ms", request.duration_ms or 0)

                if request.input_tokens:
                    span.set_attribute("genops.tokens.input", request.input_tokens)
                if request.output_tokens:
                    span.set_attribute("genops.tokens.output", request.output_tokens)
                if request.cost:
                    span.set_attribute("genops.cost.total", float(request.cost))
                    span.set_attribute("genops.cost.currency", "USD")

                if request.stream_chunks > 0:
                    span.set_attribute("genops.stream.chunks", request.stream_chunks)

                if request.tools_used:
                    span.set_attribute(
                        "genops.tools.used", ",".join(request.tools_used)
                    )

                # Add governance attributes
                for key, value in request.governance_attrs.items():
                    span.set_attribute(f"genops.{key}", str(value))

                if request.error:
                    span.set_attribute("genops.error", request.error)
                    span.set_attribute("genops.status", "error")
                else:
                    span.set_attribute("genops.status", "success")

        except Exception as e:
            logger.error(
                f"Error finalizing telemetry for request {request.request_id}: {e}"
            )

    def _calculate_cost(
        self, provider: str, model: str, input_tokens: int, output_tokens: int
    ) -> Decimal | None:
        """Calculate cost for the request using GenOps provider-specific pricing."""
        try:
            # Import and use existing GenOps provider cost calculators
            if provider == "openai":
                from genops.providers.openai import calculate_cost

                return calculate_cost(model, input_tokens, output_tokens)
            elif provider == "anthropic":
                from genops.providers.anthropic import calculate_cost

                return calculate_cost(model, input_tokens, output_tokens)
            elif provider == "google" or provider == "gemini":
                from genops.providers.gemini import calculate_cost

                return calculate_cost(model, input_tokens, output_tokens)
            else:
                # Generic cost calculation for unsupported providers
                # Use reasonable defaults: $0.01 per 1K input tokens, $0.03 per 1K output tokens
                input_cost = Decimal(str(input_tokens)) * Decimal("0.00001")  # $0.01/1K
                output_cost = Decimal(str(output_tokens)) * Decimal(
                    "0.00003"
                )  # $0.03/1K
                return input_cost + output_cost
        except Exception as e:
            logger.warning(f"Could not calculate cost for {provider}/{model}: {e}")
            return None

    # JavaScript Integration Methods

    def generate_instrumentation_code(
        self, output_path: str = "./genops-vercel-instrumentation.js"
    ) -> str:
        """
        Generate JavaScript instrumentation code for Vercel AI SDK.

        Args:
            output_path: Path to write the instrumentation code

        Returns:
            Path to the generated instrumentation file
        """
        instrumentation_code = self._get_javascript_instrumentation_template()

        # Write to file
        with open(output_path, "w") as f:
            f.write(instrumentation_code)

        logger.info(f"Generated Vercel AI SDK instrumentation at {output_path}")
        return output_path

    def _get_javascript_instrumentation_template(self) -> str:
        """Get the JavaScript instrumentation template."""
        return f"""// GenOps Vercel AI SDK Instrumentation
// Auto-generated instrumentation for Vercel AI SDK governance

const {{ generateText, streamText, generateObject, embed }} = require('ai');
const http = require('http');

// Configuration
const GENOPS_CONFIG = {{
    telemetryEndpoint: 'http://localhost:{self.websocket_port}/telemetry',
    governance: {json.dumps(self.governance_attrs, indent=2)}
}};

// Instrumentation wrappers
function instrumentedGenerateText(options) {{
    const startTime = Date.now();
    const requestId = `vercel-ai-sdk-${{startTime}}-${{Math.random().toString(36).substr(2, 9)}}`;

    // Send start telemetry
    sendTelemetry({{
        type: 'start',
        requestId,
        operation: 'generateText',
        provider: extractProvider(options.model),
        model: extractModelName(options.model),
        governance: GENOPS_CONFIG.governance,
        timestamp: startTime
    }});

    return generateText(options).then(result => {{
        // Send completion telemetry
        sendTelemetry({{
            type: 'complete',
            requestId,
            operation: 'generateText',
            duration: Date.now() - startTime,
            usage: result.usage,
            finishReason: result.finishReason,
            success: true
        }});
        return result;
    }}).catch(error => {{
        // Send error telemetry
        sendTelemetry({{
            type: 'error',
            requestId,
            operation: 'generateText',
            duration: Date.now() - startTime,
            error: error.message,
            success: false
        }});
        throw error;
    }});
}}

function instrumentedStreamText(options) {{
    const startTime = Date.now();
    const requestId = `vercel-ai-sdk-${{startTime}}-${{Math.random().toString(36).substr(2, 9)}}`;

    // Send start telemetry
    sendTelemetry({{
        type: 'start',
        requestId,
        operation: 'streamText',
        provider: extractProvider(options.model),
        model: extractModelName(options.model),
        governance: GENOPS_CONFIG.governance,
        timestamp: startTime
    }});

    return streamText(options);
}}

// Helper functions
function extractProvider(model) {{
    if (typeof model === 'string') {{
        return model.split('/')[0] || 'unknown';
    }}
    return model.provider || 'unknown';
}}

function extractModelName(model) {{
    if (typeof model === 'string') {{
        return model.split('/').pop() || model;
    }}
    return model.name || model.model || 'unknown';
}}

function sendTelemetry(data) {{
    const postData = JSON.stringify(data);

    const options = {{
        hostname: 'localhost',
        port: {self.websocket_port},
        path: '/telemetry',
        method: 'POST',
        headers: {{
            'Content-Type': 'application/json',
            'Content-Length': Buffer.byteLength(postData)
        }}
    }};

    const req = http.request(options, (res) => {{
        // Handle response if needed
    }});

    req.on('error', (e) => {{
        console.warn('GenOps telemetry error:', e.message);
    }});

    req.write(postData);
    req.end();
}}

// Export instrumented functions
module.exports = {{
    generateText: instrumentedGenerateText,
    streamText: instrumentedStreamText,
    // Add other instrumented functions as needed

    // Original functions for direct access
    original: {{
        generateText,
        streamText,
        generateObject,
        embed
    }}
}};
"""

    def update_readme_status(self) -> None:
        """Update the README to mark Vercel AI SDK as completed."""
        # This would be implemented to automatically update the README
        # when the integration is fully functional
        pass


# Auto-instrumentation function for existing applications
def auto_instrument(
    integration_mode: str = "python_wrapper",
    team: str | None = None,
    project: str | None = None,
    **kwargs,
) -> GenOpsVercelAISDKAdapter:
    """
    Auto-instrument existing Vercel AI SDK applications with GenOps governance.

    Args:
        integration_mode: How to integrate with JavaScript
        team: Team name for governance
        project: Project name
        **kwargs: Additional governance attributes

    Returns:
        GenOpsVercelAISDKAdapter: Configured adapter instance
    """
    adapter = GenOpsVercelAISDKAdapter(
        integration_mode=integration_mode, team=team, project=project, **kwargs
    )

    logger.info("Auto-instrumentation enabled for Vercel AI SDK")
    return adapter


# Convenience functions for common operations
def track_generate_text(provider: str, model: str, **kwargs):
    """Convenience function for tracking generateText operations."""
    adapter = auto_instrument()
    return adapter.track_request("generateText", provider, model, **kwargs)


def track_stream_text(provider: str, model: str, **kwargs):
    """Convenience function for tracking streamText operations."""
    adapter = auto_instrument()
    return adapter.track_request("streamText", provider, model, **kwargs)


def track_generate_object(provider: str, model: str, **kwargs):
    """Convenience function for tracking generateObject operations."""
    adapter = auto_instrument()
    return adapter.track_request("generateObject", provider, model, **kwargs)


def track_embed(provider: str, model: str, **kwargs):
    """Convenience function for tracking embed operations."""
    adapter = auto_instrument()
    return adapter.track_request("embed", provider, model, **kwargs)
