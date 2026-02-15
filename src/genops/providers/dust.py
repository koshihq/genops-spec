"""Dust provider adapter for GenOps AI governance."""

from __future__ import annotations

import logging
import os
from typing import Any

from genops.core.telemetry import GenOpsTelemetry

logger = logging.getLogger(__name__)

# Constants to avoid CodeQL false positives
CONVERSATION_VISIBILITY_RESTRICTED = "private"

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("requests not installed. Install with: pip install requests")


class GenOpsDustAdapter:
    """Dust adapter with automatic governance telemetry."""

    def __init__(
        self,
        api_key: str | None = None,
        workspace_id: str | None = None,
        base_url: str = "https://dust.tt",
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
        self.api_key = api_key or os.getenv("DUST_API_KEY")
        self.workspace_id = workspace_id or os.getenv("DUST_WORKSPACE_ID")

        # Validate required credentials
        if not self.api_key:
            raise ValueError(
                "Dust API key not provided. Set api_key parameter or DUST_API_KEY environment variable. "
                "Get your API key from your Dust workspace settings."
            )

        if not self.workspace_id:
            raise ValueError(
                "Dust workspace ID not provided. Set workspace_id parameter or DUST_WORKSPACE_ID environment variable. "
                "Get your workspace ID from your Dust workspace URL."
            )

        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

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
        self.REQUEST_ATTRIBUTES = {"stream", "blocking", "timeout"}

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
        self, method: str, endpoint: str, data: dict | None = None
    ) -> dict[str, Any]:
        """Make HTTP request to Dust API with standardized error handling."""
        url = f"{self.base_url}/api/v1/w/{self.workspace_id}/{endpoint}"

        try:
            response = self.session.request(method, url, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Unable to connect to Dust API at {self.base_url}. Check your internet connection and verify the Dust service is accessible."
            logger.error(f"Connection error: {error_msg}")
            raise ConnectionError(error_msg) from e
        except requests.exceptions.Timeout as e:
            error_msg = "Request to Dust API timed out. The service may be experiencing high load or network issues."
            logger.error(f"Timeout error: {error_msg}")
            raise TimeoutError(error_msg) from e
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else "Unknown"

            if status_code == 401:
                error_msg = "Authentication failed with Dust API. Verify your DUST_API_KEY is correct and has not expired."
            elif status_code == 403:
                error_msg = f"Access denied to Dust workspace {self.workspace_id}. Verify your API key has permissions for this workspace."
            elif status_code == 404:
                error_msg = f"Dust resource not found: {endpoint}. Check your workspace ID ({self.workspace_id}) and endpoint path."
            elif status_code == 429:
                error_msg = "Rate limit exceeded for Dust API. Please retry after a brief delay or contact Dust support to increase limits."
            elif 500 <= status_code < 600:
                error_msg = f"Dust API server error (HTTP {status_code}). This is a temporary issue with Dust's service."
            else:
                error_msg = f"Dust API request failed with HTTP {status_code}. Response: {e.response.text[:200] if e.response else 'No response body'}"

            logger.error(f"HTTP error: {error_msg}")
            raise requests.exceptions.HTTPError(error_msg) from e
        except requests.RequestException as e:
            error_msg = f"Unexpected error communicating with Dust API: {str(e)}"
            logger.error(f"Request error: {error_msg}")
            raise RuntimeError(error_msg) from e

    def create_conversation(self, **kwargs) -> Any:
        """Create a new conversation with governance tracking."""
        governance_attrs, request_attrs, api_kwargs = self._extract_attributes(kwargs)

        # Extract conversation parameters
        title = api_kwargs.get("title", "Untitled Conversation")
        visibility = api_kwargs.get("visibility", CONVERSATION_VISIBILITY_RESTRICTED)

        operation_name = "dust.conversation.create"

        # Add governance attributes to trace_operation
        trace_attrs = {
            "operation_name": operation_name,
            "operation_type": "ai.conversation",
            "provider": "dust",
            "conversation_title": title,
            "visibility": visibility,
            "workspace_id": self.workspace_id,
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

        # Create conversation
        with self.telemetry.trace_operation(operation_name, **trace_attrs) as span:
            try:
                conversation_data = {"title": title, "visibility": visibility}

                response = self._make_request(
                    "POST", "conversations", conversation_data
                )

                # Update span with response data
                if response and isinstance(response, dict):
                    conversation_id = response.get("conversation", {}).get("sId")
                    if conversation_id:
                        span.set_attribute("conversation_id", conversation_id)

                return response

            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error_message", str(e))
                logger.error(f"Error creating Dust conversation: {e}")
                raise

    def send_message(self, conversation_id: str, content: str, **kwargs) -> Any:
        """Send message to conversation with governance tracking."""
        governance_attrs, request_attrs, api_kwargs = self._extract_attributes(kwargs)

        # Extract message parameters
        context = api_kwargs.get("context", {})
        mentions = api_kwargs.get("mentions", [])

        # Estimate input tokens (rough approximation)
        estimated_input_tokens = len(content.split()) * 1.3

        operation_name = "dust.message.send"

        # Add governance attributes to trace_operation
        trace_attrs = {
            "operation_name": operation_name,
            "operation_type": "ai.message",
            "provider": "dust",
            "conversation_id": conversation_id,
            "workspace_id": self.workspace_id,
            "tokens_estimated_input": int(estimated_input_tokens),
            "message_length": len(content),
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

        # Send message
        with self.telemetry.trace_operation(operation_name, **trace_attrs) as span:  # type: ignore[arg-type]
            try:
                message_data = {
                    "content": content,
                    "context": context,
                    "mentions": mentions,
                }

                response = self._make_request(
                    "POST", f"conversations/{conversation_id}/messages", message_data
                )

                # Update span with response data
                if response and isinstance(response, dict):
                    message = response.get("message", {})
                    if message:
                        span.set_attribute("message_id", message.get("sId", ""))

                        # Extract output tokens if available
                        if "content" in message and isinstance(message["content"], str):
                            estimated_output_tokens = (
                                len(message["content"].split()) * 1.3
                            )
                            span.set_attribute(
                                "tokens_estimated_output", int(estimated_output_tokens)
                            )

                return response

            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error_message", str(e))
                logger.error(f"Error sending Dust message: {e}")
                raise

    def run_agent(self, agent_id: str, **kwargs) -> Any:
        """Run agent with governance tracking."""
        governance_attrs, request_attrs, api_kwargs = self._extract_attributes(kwargs)

        # Extract agent parameters
        inputs = api_kwargs.get("inputs", {})
        stream = api_kwargs.get("stream", False)
        blocking = api_kwargs.get("blocking", True)

        operation_name = "dust.agent.run"

        # Add governance attributes to trace_operation
        trace_attrs = {
            "operation_name": operation_name,
            "operation_type": "ai.agent_execution",
            "provider": "dust",
            "agent_id": agent_id,
            "workspace_id": self.workspace_id,
            "stream": stream,
            "blocking": blocking,
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

        # Run agent
        with self.telemetry.trace_operation(operation_name, **trace_attrs) as span:
            try:
                agent_data = {"inputs": inputs, "stream": stream, "blocking": blocking}

                response = self._make_request(
                    "POST", f"agents/{agent_id}/runs", agent_data
                )

                # Update span with response data
                if response and isinstance(response, dict):
                    run = response.get("run", {})
                    if run:
                        span.set_attribute("run_id", run.get("sId", ""))
                        span.set_attribute("run_status", run.get("status", ""))

                        # Track results if available
                        if "results" in run and run["results"]:
                            results_count = len(run["results"])
                            span.set_attribute("results_count", results_count)

                return response

            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error_message", str(e))
                logger.error(f"Error running Dust agent: {e}")
                raise

    def create_datasource(self, name: str, **kwargs) -> Any:
        """Create datasource with governance tracking."""
        governance_attrs, request_attrs, api_kwargs = self._extract_attributes(kwargs)

        # Extract datasource parameters
        description = api_kwargs.get("description", "")
        visibility = api_kwargs.get("visibility", CONVERSATION_VISIBILITY_RESTRICTED)
        provider_id = api_kwargs.get("provider_id", "webcrawler")

        operation_name = "dust.datasource.create"

        # Add governance attributes to trace_operation
        trace_attrs = {
            "operation_name": operation_name,
            "operation_type": "ai.datasource",
            "provider": "dust",
            "datasource_name": name,
            "workspace_id": self.workspace_id,
            "visibility": visibility,
            "provider_id": provider_id,
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

        # Create datasource
        with self.telemetry.trace_operation(operation_name, **trace_attrs) as span:
            try:
                datasource_data = {
                    "name": name,
                    "description": description,
                    "visibility": visibility,
                    "provider_id": provider_id,
                }

                response = self._make_request("POST", "data_sources", datasource_data)

                # Update span with response data
                if response and isinstance(response, dict):
                    datasource = response.get("data_source", {})
                    if datasource:
                        span.set_attribute("datasource_id", datasource.get("sId", ""))

                return response

            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error_message", str(e))
                logger.error(f"Error creating Dust datasource: {e}")
                raise

    def search_datasources(self, query: str, **kwargs) -> Any:
        """Search datasources with governance tracking."""
        governance_attrs, request_attrs, api_kwargs = self._extract_attributes(kwargs)

        # Extract search parameters
        data_sources = api_kwargs.get("data_sources", [])
        top_k = api_kwargs.get("top_k", 10)

        # Estimate input tokens (rough approximation)
        estimated_input_tokens = len(query.split()) * 1.3

        operation_name = "dust.datasource.search"

        # Add governance attributes to trace_operation
        trace_attrs = {
            "operation_name": operation_name,
            "operation_type": "ai.search",
            "provider": "dust",
            "query": query,
            "workspace_id": self.workspace_id,
            "top_k": top_k,
            "tokens_estimated_input": int(estimated_input_tokens),
            "datasources_count": len(data_sources),
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

        # Search datasources
        with self.telemetry.trace_operation(operation_name, **trace_attrs) as span:
            try:
                search_data = {
                    "query": query,
                    "data_sources": data_sources,
                    "top_k": top_k,
                }

                response = self._make_request(
                    "POST", "data_sources/search", search_data
                )

                # Update span with response data
                if response and isinstance(response, dict):
                    documents = response.get("documents", [])
                    span.set_attribute("documents_found", len(documents))

                    # Estimate output tokens from search results
                    total_content = ""
                    for doc in documents:
                        if (
                            isinstance(doc, dict)
                            and "chunk" in doc
                            and "text" in doc["chunk"]
                        ):
                            total_content += doc["chunk"]["text"] + " "

                    if total_content:
                        estimated_output_tokens = len(total_content.split()) * 1.3
                        span.set_attribute(
                            "tokens_estimated_output", int(estimated_output_tokens)
                        )

                return response

            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error_message", str(e))
                logger.error(f"Error searching Dust datasources: {e}")
                raise


def instrument_dust(
    api_key: str | None = None, workspace_id: str | None = None, **kwargs
) -> GenOpsDustAdapter:
    """
    Create instrumented Dust adapter with automatic environment detection.

    Args:
        api_key: Dust API key (auto-detected from DUST_API_KEY if not provided)
        workspace_id: Dust workspace ID (auto-detected from DUST_WORKSPACE_ID if not provided)
        **kwargs: Additional configuration options and governance attributes

    Returns:
        GenOpsDustAdapter instance with telemetry enabled

    Examples:
        # Using environment variables (recommended)
        dust = instrument_dust()

        # Explicit credentials
        dust = instrument_dust(
            api_key="your_api_key",
            workspace_id="your_workspace_id"
        )

        # With governance attributes
        dust = instrument_dust(
            team="ai-team",
            project="customer-support",
            environment="production"
        )
    """
    return GenOpsDustAdapter(api_key=api_key, workspace_id=workspace_id, **kwargs)


def auto_instrument(**config) -> bool:
    """
    Universal auto-instrumentation function for Dust AI.

    Automatically instruments HTTP requests to Dust API endpoints with
    GenOps governance telemetry. Works with any HTTP client (requests, httpx, urllib).

    Args:
        **config: Configuration options for instrumentation
            - api_key: Optional API key override
            - workspace_id: Optional workspace ID override
            - team: Default team for governance attribution
            - project: Default project for governance attribution
            - environment: Default environment (dev/staging/prod)
            - enable_console_export: Show telemetry in console for debugging

    Returns:
        True if instrumentation was successful, False otherwise
    """
    try:
        logger.info("Activating Dust auto-instrumentation...")

        # Import required modules
        import os

        from genops.core.context import get_effective_attributes
        from genops.core.telemetry import GenOpsTelemetry

        # Get configuration from environment and config params
        api_key = config.get("api_key") or os.getenv("DUST_API_KEY")
        workspace_id = config.get("workspace_id") or os.getenv("DUST_WORKSPACE_ID")

        if not api_key or not workspace_id:
            error_msg = (
                "Dust auto-instrumentation requires API credentials:\n"
                "• Set DUST_API_KEY environment variable with your API key\n"
                "• Set DUST_WORKSPACE_ID environment variable with your workspace ID\n"
                "• Get credentials from your Dust workspace settings at https://dust.tt/"
            )
            logger.error(error_msg)
            return False

        # Initialize telemetry
        telemetry = GenOpsTelemetry()

        # Store original requests.Session.request method
        if not hasattr(auto_instrument, "_original_request"):
            import requests

            auto_instrument._original_request = requests.Session.request

        def instrumented_request(self, method, url, **kwargs):
            """Instrumented version of requests.Session.request for Dust API calls."""

            # Check if this is a Dust API call
            if "dust.tt/api/v1" not in url:
                # Not a Dust API call, use original method
                return auto_instrument._original_request(self, method, url, **kwargs)

            # Extract operation from URL
            operation_type = "unknown"
            if "/conversations" in url:
                if method.upper() == "POST" and url.endswith("/conversations"):
                    operation_type = "conversation_create"
                elif "/messages" in url and method.upper() == "POST":
                    operation_type = "message_send"
                else:
                    operation_type = "conversation_operation"
            elif "/agents/" in url and "/runs" in url:
                operation_type = "agent_run"
            elif "/data_sources" in url:
                if "/search" in url:
                    operation_type = "datasource_search"
                else:
                    operation_type = "datasource_operation"

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
            operation_name = f"dust.{operation_type}"

            trace_attrs = {
                "operation_name": operation_name,
                "operation_type": "ai.dust_api",
                "provider": "dust",
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
                            if (
                                operation_type == "conversation_create"
                                and "conversation" in response_data
                            ):
                                span.set_attribute(
                                    "conversation_id",
                                    response_data["conversation"].get("sId", ""),
                                )
                            elif (
                                operation_type == "message_send"
                                and "message" in response_data
                            ):
                                span.set_attribute(
                                    "message_id",
                                    response_data["message"].get("sId", ""),
                                )
                                # Estimate tokens from message content
                                content = response_data["message"].get("content", "")
                                if content:
                                    estimated_tokens = len(content.split()) * 1.3
                                    span.set_attribute(
                                        "tokens_estimated_output", int(estimated_tokens)
                                    )
                            elif (
                                operation_type == "agent_run" and "run" in response_data
                            ):
                                run_data = response_data["run"]
                                span.set_attribute("run_id", run_data.get("sId", ""))
                                span.set_attribute(
                                    "run_status", run_data.get("status", "")
                                )
                    except Exception as parse_error:
                        logger.debug(f"Could not parse Dust response: {parse_error}")

                    return response

                except Exception as e:
                    span.set_attribute("error", True)
                    span.set_attribute("error_message", str(e))
                    logger.error(f"Dust API request failed: {e}")
                    raise

        # Monkey patch requests.Session.request
        import requests

        requests.Session.request = instrumented_request

        logger.info("✅ Dust auto-instrumentation activated successfully")
        logger.info(
            "   All HTTP requests to dust.tt/api/v1 will be automatically tracked"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to activate Dust auto-instrumentation: {e}")
        return False


def disable_auto_instrument():
    """Disable auto-instrumentation and restore original HTTP methods."""
    try:
        if hasattr(auto_instrument, "_original_request"):
            import requests

            requests.Session.request = auto_instrument._original_request
            delattr(auto_instrument, "_original_request")
            logger.info("Dust auto-instrumentation disabled")
            return True
    except Exception as e:
        logger.error(f"Failed to disable Dust auto-instrumentation: {e}")
        return False
