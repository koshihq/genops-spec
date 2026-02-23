"""Flowise provider adapter for GenOps AI governance."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin

from genops.core.telemetry import GenOpsTelemetry


@dataclass
class FlowiseConfig:
    """Configuration for Flowise connection."""

    base_url: str = "http://localhost:3000"
    api_key: str | None = None
    timeout: int = 30
    max_retries: int = 3
    verify_ssl: bool = True
    headers: dict[str, str] = field(default_factory=dict)


logger = logging.getLogger(__name__)

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("requests not installed. Install with: pip install requests")


class GenOpsFlowiseAdapter:
    """Flowise adapter with automatic governance telemetry."""

    def __init__(
        self,
        base_url: str = "http://localhost:3000",
        api_key: str | None = None,
        team: str | None = None,
        project: str | None = None,
        environment: str | None = None,
        cost_center: str | None = None,
        customer_id: str | None = None,
        feature: str | None = None,
        **kwargs,
    ):
        if not HAS_REQUESTS:
            raise ImportError(
                "requests package not found. Install with: pip install requests"
            )

        # Auto-detect from environment if not provided
        self.base_url = base_url or os.getenv(
            "FLOWISE_BASE_URL", "http://localhost:3000"
        )
        self.api_key = api_key or os.getenv("FLOWISE_API_KEY")

        # Flowise API key is optional for local development but required for production
        if not self.api_key and self.base_url != "http://localhost:3000":
            logger.warning(
                "Flowise API key not provided. Set api_key parameter or FLOWISE_API_KEY environment variable. "
                "API key is required for production Flowise instances."
            )

        self.base_url = self.base_url.rstrip("/")
        self.session = requests.Session()

        # Set up headers if API key is provided
        if self.api_key:
            self.session.headers.update(
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
            )
        else:
            self.session.headers.update({"Content-Type": "application/json"})

        # Initialize governance attributes with defaults and validation
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

        # Define governance and request attributes
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
        self.REQUEST_ATTRIBUTES = {"stream", "timeout", "sessionId", "overrideConfig"}

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

    def _validate_governance_attributes(self, attrs: dict[str, Any]) -> list[str]:
        """Validate governance attributes and return list of warnings/errors."""
        warnings = []

        # Check for required governance attributes for cost attribution
        if not attrs.get("team"):
            warnings.append(
                "Missing 'team' attribute - cost attribution may be less accurate"
            )

        if not attrs.get("project"):
            warnings.append(
                "Missing 'project' attribute - project-level cost tracking unavailable"
            )

        # Validate attribute formats
        for attr_name, value in attrs.items():
            if not isinstance(value, (str, int, float, bool)):
                warnings.append(
                    f"Governance attribute '{attr_name}' should be a simple type (str, int, float, bool), got {type(value)}"
                )

            if isinstance(value, str) and len(value) > 100:
                warnings.append(
                    f"Governance attribute '{attr_name}' is very long ({len(value)} chars) - consider shortening"
                )

        return warnings

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

        # Merge with instance-level governance attributes
        merged_governance = {**self.governance_attrs, **governance_attrs}

        # Validate governance attributes
        validation_warnings = self._validate_governance_attributes(merged_governance)
        if validation_warnings:
            for warning in validation_warnings[:3]:  # Limit to first 3 warnings
                logger.warning(f"Governance validation: {warning}")

        return merged_governance, request_attrs, api_kwargs

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: dict | None = None,
        params: dict | None = None,
    ) -> dict[str, Any]:
        """Make HTTP request to Flowise API with standardized error handling."""
        url = urljoin(self.base_url, endpoint)

        try:
            response = self.session.request(method, url, json=data, params=params)
            response.raise_for_status()

            # Handle different response types
            content_type = response.headers.get("content-type", "")
            if "application/json" in content_type:
                return response.json()
            else:
                return {"content": response.text, "status_code": response.status_code}

        except requests.exceptions.ConnectionError as e:
            error_msg = f"Unable to connect to Flowise at {self.base_url}. Check your connection and verify Flowise is running."
            logger.error(f"Connection error: {error_msg}")
            raise ConnectionError(error_msg) from e
        except requests.exceptions.Timeout as e:
            error_msg = "Request to Flowise API timed out. The service may be experiencing high load."
            logger.error(f"Timeout error: {error_msg}")
            raise TimeoutError(error_msg) from e
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else "Unknown"

            if status_code == 401:
                error_msg = "Authentication failed with Flowise API. Verify your FLOWISE_API_KEY is correct."
            elif status_code == 404:
                error_msg = f"Flowise resource not found: {endpoint}. Check your chatflow ID and endpoint path."
            elif status_code == 429:
                error_msg = "Rate limit exceeded for Flowise API. Please retry after a brief delay."
            elif 500 <= status_code < 600:
                error_msg = f"Flowise API server error (HTTP {status_code}). This is a temporary issue with the service."
            else:
                error_msg = f"Flowise API request failed with HTTP {status_code}. Response: {e.response.text[:200] if e.response else 'No response body'}"

            logger.error(f"HTTP error: {error_msg}")
            raise requests.exceptions.HTTPError(error_msg) from e
        except requests.RequestException as e:
            error_msg = f"Unexpected error communicating with Flowise API: {str(e)}"
            logger.error(f"Request error: {error_msg}")
            raise RuntimeError(error_msg) from e

    def predict_flow(self, chatflow_id: str, question: str, **kwargs) -> Any:
        """Execute a Flowise chatflow with governance tracking."""
        governance_attrs, request_attrs, api_kwargs = self._extract_attributes(kwargs)

        # Extract prediction parameters
        session_id = api_kwargs.get("sessionId")
        override_config = api_kwargs.get("overrideConfig", {})
        history = api_kwargs.get("history", [])
        stream = api_kwargs.get("stream", False)

        # Estimate input tokens (rough approximation for cost tracking)
        estimated_input_tokens = len(question.split()) * 1.3
        if history:
            for msg in history:
                if isinstance(msg, dict) and "message" in msg:
                    estimated_input_tokens += len(str(msg["message"]).split()) * 1.3

        operation_name = "flowise.flow.predict"

        # Add governance attributes to trace_operation
        trace_attrs = {
            "operation_name": operation_name,
            "operation_type": "ai.flow_execution",
            "provider": "flowise",
            "chatflow_id": chatflow_id,
            "session_id": session_id or "none",
            "stream": stream,
            "tokens_estimated_input": int(estimated_input_tokens),
            "question_length": len(question),
            "history_length": len(history),
            "has_override_config": bool(override_config),
        }

        # Add effective attributes (defaults + context + governance)
        try:
            from genops.core.context import get_effective_attributes

            effective_attrs = get_effective_attributes(**governance_attrs)
            trace_attrs.update(effective_attrs)
        except ImportError:
            logger.debug(
                "Context module not available, proceeding without context attributes"
            )

        # Execute flow
        with self.telemetry.trace_operation(operation_name, **trace_attrs) as span:
            try:
                prediction_data = {"question": question}

                # Add optional parameters
                if session_id:
                    prediction_data["sessionId"] = session_id
                if override_config:
                    prediction_data["overrideConfig"] = override_config
                if history:
                    prediction_data["history"] = history

                endpoint = f"/api/v1/prediction/{chatflow_id}"
                response = self._make_request("POST", endpoint, prediction_data)

                # Update span with response data
                if response and isinstance(response, dict):
                    # Extract response text for token estimation
                    response_text = ""
                    if "text" in response:
                        response_text = str(response["text"])
                    elif "answer" in response:
                        response_text = str(response["answer"])
                    elif "content" in response:
                        response_text = str(response["content"])

                    if response_text:
                        estimated_output_tokens = len(response_text.split()) * 1.3
                        span.set_attribute(
                            "tokens_estimated_output", int(estimated_output_tokens)
                        )
                        span.set_attribute("response_length", len(response_text))

                    # Track session information if available
                    if "sessionId" in response:
                        span.set_attribute("response_session_id", response["sessionId"])

                    # Track any additional metadata
                    if "chatId" in response:
                        span.set_attribute("chat_id", response["chatId"])

                return response

            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error_message", str(e))
                logger.error(f"Error executing Flowise flow: {e}")
                raise

    def get_chatflows(self, **kwargs) -> Any:
        """Get list of available chatflows with governance tracking."""
        governance_attrs, request_attrs, api_kwargs = self._extract_attributes(kwargs)

        operation_name = "flowise.chatflows.list"

        # Add governance attributes to trace_operation
        trace_attrs = {
            "operation_name": operation_name,
            "operation_type": "ai.management",
            "provider": "flowise",
        }

        # Add effective attributes (defaults + context + governance)
        try:
            from genops.core.context import get_effective_attributes

            effective_attrs = get_effective_attributes(**governance_attrs)
            trace_attrs.update(effective_attrs)
        except ImportError:
            logger.debug(
                "Context module not available, proceeding without context attributes"
            )

        # Get chatflows
        with self.telemetry.trace_operation(operation_name, **trace_attrs) as span:
            try:
                response = self._make_request("GET", "/api/v1/chatflows")

                # Update span with response data
                if response and isinstance(response, list):
                    span.set_attribute("chatflows_count", len(response))

                    # Extract chatflow names for debugging
                    if response:
                        chatflow_names = [
                            cf.get("name", "unnamed") for cf in response[:5]
                        ]  # First 5
                        span.set_attribute(
                            "sample_chatflow_names", json.dumps(chatflow_names)
                        )

                return response

            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error_message", str(e))
                logger.error(f"Error getting Flowise chatflows: {e}")
                raise

    def get_chatflow(self, chatflow_id: str, **kwargs) -> Any:
        """Get specific chatflow details with governance tracking."""
        governance_attrs, request_attrs, api_kwargs = self._extract_attributes(kwargs)

        operation_name = "flowise.chatflow.get"

        # Add governance attributes to trace_operation
        trace_attrs = {
            "operation_name": operation_name,
            "operation_type": "ai.management",
            "provider": "flowise",
            "chatflow_id": chatflow_id,
        }

        # Add effective attributes (defaults + context + governance)
        try:
            from genops.core.context import get_effective_attributes

            effective_attrs = get_effective_attributes(**governance_attrs)
            trace_attrs.update(effective_attrs)
        except ImportError:
            logger.debug(
                "Context module not available, proceeding without context attributes"
            )

        # Get chatflow
        with self.telemetry.trace_operation(operation_name, **trace_attrs) as span:
            try:
                response = self._make_request("GET", f"/api/v1/chatflows/{chatflow_id}")

                # Update span with response data
                if response and isinstance(response, dict):
                    span.set_attribute("chatflow_name", response.get("name", "unknown"))
                    span.set_attribute(
                        "chatflow_category", response.get("category", "unknown")
                    )

                    # Track flow complexity
                    if "flowData" in response:
                        try:
                            flow_data = (
                                json.loads(response["flowData"])
                                if isinstance(response["flowData"], str)
                                else response["flowData"]
                            )
                            if isinstance(flow_data, dict) and "nodes" in flow_data:
                                span.set_attribute(
                                    "nodes_count", len(flow_data["nodes"])
                                )
                        except (json.JSONDecodeError, KeyError, TypeError):
                            logger.debug("Could not parse flowData for node count")

                return response

            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error_message", str(e))
                logger.error(f"Error getting Flowise chatflow: {e}")
                raise

    def get_chat_messages(
        self, chatflow_id: str, session_id: str | None = None, **kwargs
    ) -> Any:
        """Get chat message history with governance tracking."""
        governance_attrs, request_attrs, api_kwargs = self._extract_attributes(kwargs)

        operation_name = "flowise.messages.get"

        # Add governance attributes to trace_operation
        trace_attrs = {
            "operation_name": operation_name,
            "operation_type": "ai.management",
            "provider": "flowise",
            "chatflow_id": chatflow_id,
            "session_id": session_id or "none",
        }

        # Add effective attributes (defaults + context + governance)
        try:
            from genops.core.context import get_effective_attributes

            effective_attrs = get_effective_attributes(**governance_attrs)
            trace_attrs.update(effective_attrs)
        except ImportError:
            logger.debug(
                "Context module not available, proceeding without context attributes"
            )

        # Get messages
        with self.telemetry.trace_operation(operation_name, **trace_attrs) as span:
            try:
                endpoint = f"/api/v1/chatmessage/{chatflow_id}"
                params = {}
                if session_id:
                    params["sessionId"] = session_id

                response = self._make_request("GET", endpoint, params=params)

                # Update span with response data
                if response and isinstance(response, list):
                    span.set_attribute("messages_count", len(response))

                    # Calculate total tokens from messages for cost tracking
                    total_tokens = 0
                    for msg in response:
                        if isinstance(msg, dict):
                            message_text = msg.get("message", "") + msg.get(
                                "answer", ""
                            )
                            total_tokens += len(message_text.split()) * 1.3

                    if total_tokens > 0:
                        span.set_attribute("total_estimated_tokens", int(total_tokens))

                return response

            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error_message", str(e))
                logger.error(f"Error getting Flowise chat messages: {e}")
                raise

    def delete_chat_messages(
        self, chatflow_id: str, session_id: str | None = None, **kwargs
    ) -> Any:
        """Delete chat message history with governance tracking."""
        governance_attrs, request_attrs, api_kwargs = self._extract_attributes(kwargs)

        operation_name = "flowise.messages.delete"

        # Add governance attributes to trace_operation
        trace_attrs = {
            "operation_name": operation_name,
            "operation_type": "ai.management",
            "provider": "flowise",
            "chatflow_id": chatflow_id,
            "session_id": session_id or "all",
        }

        # Add effective attributes (defaults + context + governance)
        try:
            from genops.core.context import get_effective_attributes

            effective_attrs = get_effective_attributes(**governance_attrs)
            trace_attrs.update(effective_attrs)
        except ImportError:
            logger.debug(
                "Context module not available, proceeding without context attributes"
            )

        # Delete messages
        with self.telemetry.trace_operation(operation_name, **trace_attrs) as span:
            try:
                endpoint = f"/api/v1/chatmessage/{chatflow_id}"
                params = {}
                if session_id:
                    params["sessionId"] = session_id

                response = self._make_request("DELETE", endpoint, params=params)

                # Update span with success indicator
                span.set_attribute("deletion_successful", True)

                return response

            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error_message", str(e))
                logger.error(f"Error deleting Flowise chat messages: {e}")
                raise


def instrument_flowise(
    base_url: str | None = None, api_key: str | None = None, **kwargs
) -> GenOpsFlowiseAdapter:
    """
    Create instrumented Flowise adapter with automatic environment detection.

    Args:
        base_url: Flowise instance URL (auto-detected from FLOWISE_BASE_URL if not provided)
        api_key: Flowise API key (auto-detected from FLOWISE_API_KEY if not provided)
        **kwargs: Additional configuration options and governance attributes

    Returns:
        GenOpsFlowiseAdapter instance with telemetry enabled

    Examples:
        # Using environment variables (recommended)
        flowise = instrument_flowise()

        # Explicit configuration
        flowise = instrument_flowise(
            base_url="http://localhost:3000",
            api_key="your_api_key"
        )

        # With governance attributes
        flowise = instrument_flowise(
            team="ai-team",
            project="customer-support",
            environment="production"
        )
    """
    return GenOpsFlowiseAdapter(base_url=base_url, api_key=api_key, **kwargs)  # type: ignore


def auto_instrument(**config) -> bool:
    """
    Universal auto-instrumentation function for Flowise.

    Automatically instruments HTTP requests to Flowise API endpoints with
    GenOps governance telemetry. Works with any HTTP client (requests, httpx, urllib).

    Args:
        **config: Configuration options for instrumentation
            - base_url: Optional Flowise base URL override
            - api_key: Optional API key override
            - team: Default team for governance attribution
            - project: Default project for governance attribution
            - environment: Default environment (dev/staging/prod)
            - enable_console_export: Show telemetry in console for debugging

    Returns:
        True if instrumentation was successful, False otherwise
    """
    try:
        logger.info("Activating Flowise auto-instrumentation...")

        # Import required modules
        import os

        from genops.core.context import get_effective_attributes
        from genops.core.telemetry import GenOpsTelemetry

        # Get configuration from environment and config params
        base_url = config.get("base_url") or os.getenv(
            "FLOWISE_BASE_URL", "http://localhost:3000"
        )
        config.get("api_key") or os.getenv("FLOWISE_API_KEY")

        # Initialize telemetry
        telemetry = GenOpsTelemetry()

        # Store original requests.Session.request method
        if not hasattr(auto_instrument, "_original_request"):
            import requests

            auto_instrument._original_request = requests.Session.request

        def instrumented_request(self, method, url, **kwargs):
            """Instrumented version of requests.Session.request for Flowise API calls."""

            # Check if this is a Flowise API call
            base_domain = (
                base_url.replace("http://", "").replace("https://", "").split("/")[0]
            )
            if base_domain not in url or "/api/v1/" not in url:
                # Not a Flowise API call, use original method
                return auto_instrument._original_request(self, method, url, **kwargs)

            # Extract operation from URL
            operation_type = "unknown"
            if "/prediction/" in url:
                operation_type = "flow_predict"
            elif "/chatflows" in url:
                if method.upper() == "GET":
                    operation_type = (
                        "chatflows_list"
                        if url.endswith("/chatflows")
                        else "chatflow_get"
                    )
                else:
                    operation_type = "chatflow_operation"
            elif "/chatmessage/" in url:
                if method.upper() == "GET":
                    operation_type = "messages_get"
                elif method.upper() == "DELETE":
                    operation_type = "messages_delete"
                else:
                    operation_type = "message_operation"

            # Get governance attributes
            governance_attrs = get_effective_attributes(
                team=config.get("team"),
                project=config.get("project"),
                environment=config.get("environment"),
                **{
                    k: v
                    for k, v in config.items()
                    if k in {"customer_id", "cost_center", "user_id", "feature"}
                },
            )

            # Validate governance attributes (silent validation for auto-instrumentation)
            if not governance_attrs.get("team"):
                logger.debug(
                    "Auto-instrumentation: Missing team attribute - cost attribution may be less accurate"
                )
            if not governance_attrs.get("project"):
                logger.debug(
                    "Auto-instrumentation: Missing project attribute - project-level cost tracking unavailable"
                )

            # Create telemetry span
            operation_name = f"flowise.{operation_type}"

            trace_attrs = {
                "operation_name": operation_name,
                "operation_type": "ai.flowise_api",
                "provider": "flowise",
                "http.method": method.upper(),
                "http.url": url,
                **governance_attrs,
            }

            with telemetry.trace_operation(operation_name, **trace_attrs) as span:
                try:
                    # Make the actual request
                    response = auto_instrument._original_request(
                        self, method, url, **kwargs
                    )

                    # Record response details
                    span.set_attribute("http.status_code", response.status_code)

                    if response.status_code >= 400:
                        span.set_attribute("error", True)
                        span.set_attribute(
                            "error_message", f"HTTP {response.status_code}"
                        )

                    # Try to extract meaningful data from response
                    try:
                        if response.headers.get("content-type", "").startswith(
                            "application/json"
                        ):
                            response_data = response.json()

                            # Extract operation-specific metrics
                            if operation_type == "flow_predict" and isinstance(
                                response_data, dict
                            ):
                                # Track prediction response
                                if "text" in response_data or "answer" in response_data:
                                    response_text = response_data.get(
                                        "text"
                                    ) or response_data.get("answer", "")
                                    if response_text:
                                        estimated_tokens = (
                                            len(str(response_text).split()) * 1.3
                                        )
                                        span.set_attribute(
                                            "tokens_estimated_output",
                                            int(estimated_tokens),
                                        )

                            elif operation_type == "chatflows_list" and isinstance(
                                response_data, list
                            ):
                                span.set_attribute(
                                    "chatflows_count", len(response_data)
                                )

                            elif operation_type == "chatflow_get" and isinstance(
                                response_data, dict
                            ):
                                span.set_attribute(
                                    "chatflow_name",
                                    response_data.get("name", "unknown"),
                                )

                            elif operation_type == "messages_get" and isinstance(
                                response_data, list
                            ):
                                span.set_attribute("messages_count", len(response_data))

                    except Exception as parse_error:
                        logger.debug(f"Could not parse Flowise response: {parse_error}")

                    return response

                except Exception as e:
                    span.set_attribute("error", True)
                    span.set_attribute("error_message", str(e))
                    logger.error(f"Flowise API request failed: {e}")
                    raise

        # Monkey patch requests.Session.request
        import requests

        requests.Session.request = instrumented_request

        logger.info("✅ Flowise auto-instrumentation activated successfully")
        logger.info(
            f"   All HTTP requests to {base_url}/api/v1 will be automatically tracked"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to activate Flowise auto-instrumentation: {e}")
        return False


def disable_auto_instrument():
    """Disable auto-instrumentation and restore original HTTP methods."""
    try:
        if hasattr(auto_instrument, "_original_request"):
            import requests

            requests.Session.request = auto_instrument._original_request
            delattr(auto_instrument, "_original_request")
            logger.info("Flowise auto-instrumentation disabled")
            return True
    except Exception as e:
        logger.error(f"Failed to disable Flowise auto-instrumentation: {e}")
        return False
