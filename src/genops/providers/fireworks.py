"""
Fireworks AI Provider Adapter for GenOps AI Governance

Provides comprehensive governance for Fireworks AI operations including:
- Access to 100+ models across all modalities (text, vision, audio, embeddings)
- OpenAI-compatible API with 4x faster inference via Fireattention kernels
- Enterprise governance with SOC 2, GDPR, HIPAA compliance support
- Multi-modal support with structured outputs and function calling
- Zero-code auto-instrumentation for existing Fireworks integrations
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from genops.core.exceptions import (
    GenOpsBudgetExceededError,
    GenOpsConfigurationError,
    GenOpsValidationError,
)

# Core GenOps imports
from genops.core.telemetry import GenOpsTelemetry

# Import Fireworks pricing calculator
from .fireworks_pricing import FireworksPricingCalculator

logger = logging.getLogger(__name__)

# Optional Fireworks AI dependencies
try:
    from fireworks.client import Fireworks
    HAS_FIREWORKS = True
except ImportError:
    HAS_FIREWORKS = False
    Fireworks = None
    logger.warning("Fireworks AI client not installed. Install with: pip install fireworks-ai")

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logger.warning("OpenAI client not installed. Install with: pip install openai")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("Requests not installed. Install with: pip install requests")


class FireworksModel(Enum):
    """Popular Fireworks AI models with their characteristics."""

    # Chat Models - Llama Family
    LLAMA_3_1_8B_INSTRUCT = "accounts/fireworks/models/llama-v3p1-8b-instruct"
    LLAMA_3_1_70B_INSTRUCT = "accounts/fireworks/models/llama-v3p1-70b-instruct"
    LLAMA_3_1_405B_INSTRUCT = "accounts/fireworks/models/llama-v3p1-405b-instruct"
    LLAMA_3_2_1B_INSTRUCT = "accounts/fireworks/models/llama-v3p2-1b-instruct"
    LLAMA_3_2_3B_INSTRUCT = "accounts/fireworks/models/llama-v3p2-3b-instruct"

    # Reasoning Models
    DEEPSEEK_R1 = "accounts/deepseek-ai/models/deepseek-r1"
    DEEPSEEK_R1_DISTILL = "accounts/deepseek-ai/models/deepseek-r1-distill-llama-70b"

    # Code Generation Models
    DEEPSEEK_CODER_V2_LITE = "accounts/deepseek-ai/models/deepseek-coder-v2-lite-instruct"
    QWEN_CODER_32B = "accounts/qwen/models/qwen2p5-coder-32b-instruct"
    CODELLAMA_70B_INSTRUCT = "accounts/codellama/models/codellama-70b-instruct"

    # Multimodal Models
    QWEN_VL_72B = "accounts/qwen/models/qwen2-vl-72b-instruct"
    LLAMA_VISION_11B = "accounts/fireworks/models/llama-v3p2-11b-vision-instruct"
    PIXTRAL_12B = "accounts/mistral/models/pixtral-12b-2409"

    # Language Models - Mixtral
    MIXTRAL_8X7B = "accounts/fireworks/models/mixtral-8x7b-instruct"
    MIXTRAL_8X22B = "accounts/fireworks/models/mixtral-8x22b-instruct"

    # Embedding Models
    NOMIC_EMBED_TEXT = "accounts/fireworks/models/nomic-embed-text-v1p5"
    BGE_BASE_EN_V15 = "accounts/fireworks/models/bge-base-en-v1p5"

    # Audio Models
    WHISPER_V3 = "accounts/fireworks/models/whisper-v3"


class FireworksTaskType(Enum):
    """Task types for Fireworks AI operations."""
    CHAT_COMPLETION = "chat_completion"
    TEXT_COMPLETION = "text_completion"
    EMBEDDINGS = "embeddings"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    VISION_ANALYSIS = "vision_analysis"
    FUNCTION_CALLING = "function_calling"
    STRUCTURED_OUTPUT = "structured_output"
    BATCH_INFERENCE = "batch_inference"


@dataclass
class FireworksResult:
    """Fireworks AI result with governance metadata."""
    prompt: str
    response: str
    model_used: str
    task_type: FireworksTaskType
    tokens_used: int
    cost: Decimal
    execution_time_seconds: float
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    governance_attributes: Dict[str, Any] = None

    def __post_init__(self):
        if self.governance_attributes is None:
            self.governance_attributes = {}


@dataclass
class FireworksSessionContext:
    """Session context for tracking multiple Fireworks operations."""
    session_name: str
    session_id: str
    start_time: datetime
    total_operations: int = 0
    total_cost: Decimal = Decimal("0.00")
    total_tokens: int = 0
    operations_by_model: Dict[str, int] = None

    def __post_init__(self):
        if self.operations_by_model is None:
            self.operations_by_model = {}


class GenOpsFireworksAdapter:
    """
    GenOps governance adapter for Fireworks AI with comprehensive cost tracking,
    budget enforcement, and enterprise-grade governance controls.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.fireworks.ai/inference/v1",
        team: Optional[str] = None,
        project: Optional[str] = None,
        environment: str = "development",
        customer_id: Optional[str] = None,
        cost_center: Optional[str] = None,
        daily_budget_limit: float = 100.0,
        monthly_budget_limit: float = 2000.0,
        governance_policy: str = "advisory",  # advisory, enforced, strict
        enable_governance: bool = True,
        enable_cost_alerts: bool = True,
        enable_performance_monitoring: bool = True,
        tags: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Initialize Fireworks adapter with governance configuration.
        
        Args:
            api_key: Fireworks API key (or set FIREWORKS_API_KEY env var)
            base_url: Fireworks API base URL
            team: Team name for cost attribution
            project: Project name for tracking
            environment: Environment (development, staging, production)
            customer_id: Customer ID for multi-tenant billing
            cost_center: Cost center for financial reporting
            daily_budget_limit: Daily spending limit in USD
            monthly_budget_limit: Monthly spending limit in USD
            governance_policy: Policy enforcement level
            enable_governance: Enable governance tracking
            enable_cost_alerts: Enable cost alerting
            enable_performance_monitoring: Enable performance tracking
            tags: Additional tags for attribution
        """
        # API Configuration
        self.api_key = api_key or os.getenv("FIREWORKS_API_KEY")
        self.base_url = base_url

        if not self.api_key:
            raise GenOpsConfigurationError(
                "Fireworks API key required. Set FIREWORKS_API_KEY environment variable "
                "or pass api_key parameter. Get your key from: https://fireworks.ai/api-keys"
            )

        # Governance Configuration
        self.team = team or os.getenv("GENOPS_TEAM", "default-team")
        self.project = project or os.getenv("GENOPS_PROJECT", "fireworks-project")
        self.environment = environment or os.getenv("GENOPS_ENVIRONMENT", "development")
        self.customer_id = customer_id
        self.cost_center = cost_center
        self.tags = tags or {}

        # Budget Configuration
        self.daily_budget_limit = daily_budget_limit
        self.monthly_budget_limit = monthly_budget_limit
        self.governance_policy = governance_policy
        self.enable_governance = enable_governance
        self.enable_cost_alerts = enable_cost_alerts
        self.enable_performance_monitoring = enable_performance_monitoring

        # Initialize clients
        self._init_clients()

        # Initialize pricing calculator
        self.pricing_calc = FireworksPricingCalculator()

        # Initialize telemetry
        self.telemetry = GenOpsTelemetry(tracer_name="fireworks")

        # Session tracking
        self.active_sessions: Dict[str, FireworksSessionContext] = {}

        # Cost tracking
        self._daily_costs = Decimal("0.00")
        self._monthly_costs = Decimal("0.00")

        logger.info(f"GenOps Fireworks adapter initialized for team={self.team}, project={self.project}")

    def _init_clients(self):
        """Initialize Fireworks AI clients."""
        if not HAS_FIREWORKS:
            raise GenOpsConfigurationError(
                "Fireworks AI client not installed. Install with: pip install fireworks-ai"
            )

        # Initialize Fireworks client
        self.client = Fireworks(api_key=self.api_key)

        # Initialize OpenAI-compatible client for compatibility
        if HAS_OPENAI:
            self.openai_client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        else:
            self.openai_client = None
            logger.warning("OpenAI client not available for OpenAI-compatible interface")

    def chat_with_governance(
        self,
        messages: List[Dict[str, Any]],
        model: Union[str, FireworksModel] = FireworksModel.LLAMA_3_1_8B_INSTRUCT,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        # Governance parameters
        session_id: Optional[str] = None,
        feature: Optional[str] = None,
        use_case: Optional[str] = None,
        customer_id: Optional[str] = None,
        cost_center: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> FireworksResult:
        """
        Perform chat completion with comprehensive governance tracking.
        
        Args:
            messages: Chat messages in OpenAI format
            model: Model to use for completion
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            functions: Available functions for function calling
            function_call: Function call configuration
            tools: Available tools for tool calling
            tool_choice: Tool choice configuration
            response_format: Response format specification
            stream: Whether to stream the response
            session_id: Session ID for grouping operations
            feature: Feature name for attribution
            use_case: Use case for categorization
            customer_id: Customer ID for billing attribution
            cost_center: Cost center for financial reporting
            tags: Additional tags for attribution
            **kwargs: Additional parameters for the API
        
        Returns:
            FireworksResult with response and governance metadata
        """
        start_time = time.time()

        # Resolve model
        model_name = model.value if isinstance(model, FireworksModel) else model

        # Prepare request
        request_data = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stream": stream,
            **kwargs
        }

        if max_tokens is not None:
            request_data["max_tokens"] = max_tokens

        if functions:
            request_data["functions"] = functions
        if function_call:
            request_data["function_call"] = function_call
        if tools:
            request_data["tools"] = tools
        if tool_choice:
            request_data["tool_choice"] = tool_choice
        if response_format:
            request_data["response_format"] = response_format

        # Governance attributes
        governance_attrs = {
            "team": self.team,
            "project": self.project,
            "environment": self.environment,
            "model": model_name,
            "task_type": FireworksTaskType.CHAT_COMPLETION.value,
            **self.tags
        }

        if session_id:
            governance_attrs["session_id"] = session_id
        if feature:
            governance_attrs["feature"] = feature
        if use_case:
            governance_attrs["use_case"] = use_case
        if customer_id or self.customer_id:
            governance_attrs["customer_id"] = customer_id or self.customer_id
        if cost_center or self.cost_center:
            governance_attrs["cost_center"] = cost_center or self.cost_center
        if tags:
            governance_attrs.update(tags)

        # Pre-execution governance checks
        if self.enable_governance and self.governance_policy in ["enforced", "strict"]:
            self._check_budget_compliance(governance_attrs)

        try:
            # Execute request
            if stream:
                return self._handle_streaming_completion(request_data, governance_attrs, start_time)
            else:
                response = self.client.chat.completions.create(**request_data)

                # Calculate execution time
                execution_time = time.time() - start_time

                # Extract response content
                response_content = ""
                if response.choices and len(response.choices) > 0:
                    if response.choices[0].message:
                        response_content = response.choices[0].message.content or ""

                # Calculate tokens and cost
                input_tokens = response.usage.prompt_tokens if response.usage else 0
                output_tokens = response.usage.completion_tokens if response.usage else 0
                total_tokens = input_tokens + output_tokens

                cost = self.pricing_calc.estimate_chat_cost(
                    model_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens
                )

                # Create result
                result = FireworksResult(
                    prompt=self._extract_prompt_text(messages),
                    response=response_content,
                    model_used=model_name,
                    task_type=FireworksTaskType.CHAT_COMPLETION,
                    tokens_used=total_tokens,
                    cost=cost,
                    execution_time_seconds=execution_time,
                    session_id=session_id,
                    request_id=getattr(response, 'id', None),
                    governance_attributes=governance_attrs
                )

                # Update session tracking
                if session_id and session_id in self.active_sessions:
                    session = self.active_sessions[session_id]
                    session.total_operations += 1
                    session.total_cost += cost
                    session.total_tokens += total_tokens
                    session.operations_by_model[model_name] = session.operations_by_model.get(model_name, 0) + 1

                # Update cost tracking
                self._daily_costs += cost
                self._monthly_costs += cost

                # Emit telemetry
                self._emit_completion_telemetry(result, governance_attrs)

                # Post-execution governance checks
                if self.enable_cost_alerts:
                    self._check_cost_alerts(cost, governance_attrs)

                return result

        except Exception as e:
            execution_time = time.time() - start_time

            # Emit error telemetry
            error_attrs = {**governance_attrs, "error": str(e), "execution_time": execution_time}
            self.telemetry.record_error("fireworks_chat_completion_failed", error_attrs)

            if isinstance(e, (GenOpsBudgetExceededError, GenOpsValidationError)):
                raise

            raise GenOpsValidationError(f"Fireworks chat completion failed: {e}") from e

    def _handle_streaming_completion(
        self,
        request_data: Dict[str, Any],
        governance_attrs: Dict[str, Any],
        start_time: float
    ) -> FireworksResult:
        """Handle streaming chat completion with governance tracking."""
        response_content = ""
        total_tokens = 0

        try:
            stream = self.client.chat.completions.create(**request_data)

            for chunk in stream:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        response_content += delta.content

            execution_time = time.time() - start_time

            # Estimate tokens for streaming (approximate)
            estimated_input_tokens = len(str(request_data.get("messages", ""))) // 4
            estimated_output_tokens = len(response_content) // 4
            total_tokens = estimated_input_tokens + estimated_output_tokens

            # Calculate cost
            cost = self.pricing_calc.estimate_chat_cost(
                request_data["model"],
                input_tokens=estimated_input_tokens,
                output_tokens=estimated_output_tokens
            )

            # Create result
            result = FireworksResult(
                prompt=self._extract_prompt_text(request_data.get("messages", [])),
                response=response_content,
                model_used=request_data["model"],
                task_type=FireworksTaskType.CHAT_COMPLETION,
                tokens_used=total_tokens,
                cost=cost,
                execution_time_seconds=execution_time,
                session_id=governance_attrs.get("session_id"),
                governance_attributes=governance_attrs
            )

            # Update tracking
            self._daily_costs += cost
            self._monthly_costs += cost

            # Emit telemetry
            self._emit_completion_telemetry(result, governance_attrs)

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            error_attrs = {**governance_attrs, "error": str(e), "execution_time": execution_time}
            self.telemetry.record_error("fireworks_streaming_completion_failed", error_attrs)
            raise GenOpsValidationError(f"Fireworks streaming completion failed: {e}") from e

    def embeddings_with_governance(
        self,
        input_texts: Union[str, List[str]],
        model: Union[str, FireworksModel] = FireworksModel.NOMIC_EMBED_TEXT,
        # Governance parameters
        session_id: Optional[str] = None,
        feature: Optional[str] = None,
        use_case: Optional[str] = None,
        customer_id: Optional[str] = None,
        **kwargs
    ) -> FireworksResult:
        """
        Generate embeddings with governance tracking.
        
        Args:
            input_texts: Text(s) to embed
            model: Embedding model to use
            session_id: Session ID for grouping operations
            feature: Feature name for attribution
            use_case: Use case for categorization
            customer_id: Customer ID for billing attribution
            **kwargs: Additional parameters for the API
        
        Returns:
            FireworksResult with embeddings and governance metadata
        """
        start_time = time.time()

        # Resolve model
        model_name = model.value if isinstance(model, FireworksModel) else model

        # Ensure input is list
        if isinstance(input_texts, str):
            input_texts = [input_texts]

        # Governance attributes
        governance_attrs = {
            "team": self.team,
            "project": self.project,
            "environment": self.environment,
            "model": model_name,
            "task_type": FireworksTaskType.EMBEDDINGS.value,
            "input_count": len(input_texts),
            **self.tags
        }

        if session_id:
            governance_attrs["session_id"] = session_id
        if feature:
            governance_attrs["feature"] = feature
        if use_case:
            governance_attrs["use_case"] = use_case
        if customer_id or self.customer_id:
            governance_attrs["customer_id"] = customer_id or self.customer_id

        try:
            # Execute embedding request
            response = self.client.embeddings.create(
                model=model_name,
                input=input_texts,
                **kwargs
            )

            execution_time = time.time() - start_time

            # Calculate tokens and cost (embeddings typically charge per input token)
            estimated_tokens = sum(len(text) // 4 for text in input_texts)  # Rough estimate
            cost = self.pricing_calc.estimate_embedding_cost(model_name, estimated_tokens)

            # Create result
            result = FireworksResult(
                prompt=f"Embedding {len(input_texts)} texts",
                response=f"Generated {len(response.data)} embeddings",
                model_used=model_name,
                task_type=FireworksTaskType.EMBEDDINGS,
                tokens_used=estimated_tokens,
                cost=cost,
                execution_time_seconds=execution_time,
                session_id=session_id,
                governance_attributes=governance_attrs
            )

            # Update tracking
            self._daily_costs += cost
            self._monthly_costs += cost

            # Emit telemetry
            self._emit_completion_telemetry(result, governance_attrs)

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            error_attrs = {**governance_attrs, "error": str(e), "execution_time": execution_time}
            self.telemetry.record_error("fireworks_embeddings_failed", error_attrs)
            raise GenOpsValidationError(f"Fireworks embeddings failed: {e}") from e

    @contextmanager
    def track_session(
        self,
        session_name: str,
        customer_id: Optional[str] = None,
        cost_center: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> Iterator[FireworksSessionContext]:
        """
        Context manager for tracking multiple related Fireworks operations.
        
        Args:
            session_name: Name for the session
            customer_id: Customer ID for billing attribution
            cost_center: Cost center for financial reporting
            tags: Additional tags for attribution
        
        Yields:
            FireworksSessionContext for the session
        """
        session_id = str(uuid.uuid4())
        session = FireworksSessionContext(
            session_name=session_name,
            session_id=session_id,
            start_time=datetime.now(timezone.utc)
        )

        self.active_sessions[session_id] = session

        # Emit session start telemetry
        session_attrs = {
            "team": self.team,
            "project": self.project,
            "environment": self.environment,
            "session_name": session_name,
            "session_id": session_id,
            **self.tags
        }

        if customer_id or self.customer_id:
            session_attrs["customer_id"] = customer_id or self.customer_id
        if cost_center or self.cost_center:
            session_attrs["cost_center"] = cost_center or self.cost_center
        if tags:
            session_attrs.update(tags)

        self.telemetry.record_event("fireworks_session_started", session_attrs)

        try:
            yield session
        finally:
            # Calculate session duration
            end_time = datetime.now(timezone.utc)
            duration = (end_time - session.start_time).total_seconds()

            # Update session context
            session_attrs.update({
                "duration_seconds": duration,
                "total_operations": session.total_operations,
                "total_cost": float(session.total_cost),
                "total_tokens": session.total_tokens,
                "operations_by_model": session.operations_by_model
            })

            # Emit session completion telemetry
            self.telemetry.record_event("fireworks_session_completed", session_attrs)

            # Clean up session
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get current cost summary and budget utilization."""
        daily_utilization = (float(self._daily_costs) / self.daily_budget_limit) * 100
        monthly_utilization = (float(self._monthly_costs) / self.monthly_budget_limit) * 100

        return {
            "daily_costs": float(self._daily_costs),
            "monthly_costs": float(self._monthly_costs),
            "daily_budget_limit": self.daily_budget_limit,
            "monthly_budget_limit": self.monthly_budget_limit,
            "daily_budget_utilization": daily_utilization,
            "monthly_budget_utilization": monthly_utilization,
            "active_sessions": len(self.active_sessions),
            "governance_policy": self.governance_policy
        }

    def _extract_prompt_text(self, messages: List[Dict[str, Any]]) -> str:
        """Extract prompt text from messages for tracking."""
        if not messages:
            return ""

        # Get the last user message as the primary prompt
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, str):
                    return content[:200] + "..." if len(content) > 200 else content
                elif isinstance(content, list):
                    # Handle multimodal content
                    text_parts = [item.get("text", "") for item in content if item.get("type") == "text"]
                    full_text = " ".join(text_parts)
                    return full_text[:200] + "..." if len(full_text) > 200 else full_text

        return "No user message found"

    def _check_budget_compliance(self, governance_attrs: Dict[str, Any]):
        """Check budget compliance before operations."""
        daily_utilization = (float(self._daily_costs) / self.daily_budget_limit) * 100
        monthly_utilization = (float(self._monthly_costs) / self.monthly_budget_limit) * 100

        if self.governance_policy == "strict":
            if daily_utilization >= 95.0:
                raise GenOpsBudgetExceededError(
                    f"Daily budget 95% exceeded ({daily_utilization:.1f}%). Operation blocked."
                )
            if monthly_utilization >= 95.0:
                raise GenOpsBudgetExceededError(
                    f"Monthly budget 95% exceeded ({monthly_utilization:.1f}%). Operation blocked."
                )

        elif self.governance_policy == "enforced":
            if daily_utilization >= 100.0:
                raise GenOpsBudgetExceededError(
                    f"Daily budget exceeded ({daily_utilization:.1f}%). Operation blocked."
                )
            if monthly_utilization >= 100.0:
                raise GenOpsBudgetExceededError(
                    f"Monthly budget exceeded ({monthly_utilization:.1f}%). Operation blocked."
                )

    def _check_cost_alerts(self, operation_cost: Decimal, governance_attrs: Dict[str, Any]):
        """Check for cost alerts after operations."""
        daily_utilization = (float(self._daily_costs) / self.daily_budget_limit) * 100
        monthly_utilization = (float(self._monthly_costs) / self.monthly_budget_limit) * 100

        # High cost operation alert
        if float(operation_cost) > 1.0:  # Operations over $1.00
            alert_attrs = {
                **governance_attrs,
                "operation_cost": float(operation_cost),
                "alert_type": "high_cost_operation"
            }
            self.telemetry.record_event("fireworks_cost_alert", alert_attrs)

        # Budget utilization alerts
        if daily_utilization > 80.0:
            alert_attrs = {
                **governance_attrs,
                "daily_utilization": daily_utilization,
                "alert_type": "high_daily_budget_utilization"
            }
            self.telemetry.record_event("fireworks_budget_alert", alert_attrs)

        if monthly_utilization > 80.0:
            alert_attrs = {
                **governance_attrs,
                "monthly_utilization": monthly_utilization,
                "alert_type": "high_monthly_budget_utilization"
            }
            self.telemetry.record_event("fireworks_budget_alert", alert_attrs)

    def _emit_completion_telemetry(self, result: FireworksResult, governance_attrs: Dict[str, Any]):
        """Emit telemetry for completed operations."""
        telemetry_attrs = {
            **governance_attrs,
            "model_used": result.model_used,
            "tokens_used": result.tokens_used,
            "cost": float(result.cost),
            "execution_time": result.execution_time_seconds,
            "task_type": result.task_type.value
        }

        if result.request_id:
            telemetry_attrs["request_id"] = result.request_id

        # Record the completion event
        self.telemetry.record_event("fireworks_completion", telemetry_attrs)

        # Record cost tracking
        self.telemetry.record_metric("fireworks_cost", float(result.cost), telemetry_attrs)
        self.telemetry.record_metric("fireworks_tokens", result.tokens_used, telemetry_attrs)
        self.telemetry.record_metric("fireworks_latency", result.execution_time_seconds, telemetry_attrs)


def auto_instrument():
    """
    Auto-instrumentation for existing Fireworks AI applications.
    
    Adds governance tracking to existing Fireworks code with zero changes required.
    Simply import and call this function to enable automatic tracking.
    """
    if not HAS_FIREWORKS:
        logger.warning("Fireworks AI not installed. Auto-instrumentation skipped.")
        return

    # Store original methods
    original_chat_create = None
    original_embeddings_create = None

    try:
        # Get original methods
        if hasattr(Fireworks, 'chat') and hasattr(Fireworks.chat, 'completions'):
            original_chat_create = Fireworks.chat.completions.create

        if hasattr(Fireworks, 'embeddings'):
            original_embeddings_create = Fireworks.embeddings.create

        # Create governance adapter
        adapter = GenOpsFireworksAdapter(
            team=os.getenv("GENOPS_TEAM", "auto-instrumented"),
            project=os.getenv("GENOPS_PROJECT", "fireworks-auto"),
            enable_governance=True,
            governance_policy="advisory"  # Non-blocking by default
        )

        def instrumented_chat_create(self, **kwargs):
            """Instrumented chat completion with governance tracking."""
            messages = kwargs.get('messages', [])
            model = kwargs.get('model', FireworksModel.LLAMA_3_1_8B_INSTRUCT.value)

            try:
                result = adapter.chat_with_governance(
                    messages=messages,
                    model=model,
                    **{k: v for k, v in kwargs.items() if k not in ['messages', 'model']}
                )

                # Return original format for compatibility
                class MockResponse:
                    def __init__(self, content, model, tokens):
                        self.choices = [MockChoice(content)]
                        self.model = model
                        self.usage = MockUsage(tokens)
                        self.id = result.request_id

                class MockChoice:
                    def __init__(self, content):
                        self.message = MockMessage(content)

                class MockMessage:
                    def __init__(self, content):
                        self.content = content
                        self.role = "assistant"

                class MockUsage:
                    def __init__(self, total_tokens):
                        self.total_tokens = total_tokens
                        self.prompt_tokens = total_tokens // 2
                        self.completion_tokens = total_tokens - self.prompt_tokens

                return MockResponse(result.response, result.model_used, result.tokens_used)

            except Exception as e:
                logger.warning(f"Governance tracking failed, falling back to original: {e}")
                return original_chat_create(self, **kwargs)

        def instrumented_embeddings_create(self, **kwargs):
            """Instrumented embeddings with governance tracking."""
            try:
                input_texts = kwargs.get('input', [])
                model = kwargs.get('model', FireworksModel.NOMIC_EMBED_TEXT.value)

                result = adapter.embeddings_with_governance(
                    input_texts=input_texts,
                    model=model,
                    **{k: v for k, v in kwargs.items() if k not in ['input', 'model']}
                )

                # Fall back to original for actual embeddings
                return original_embeddings_create(self, **kwargs)

            except Exception as e:
                logger.warning(f"Embedding governance tracking failed: {e}")
                return original_embeddings_create(self, **kwargs)

        # Monkey patch methods
        if original_chat_create:
            Fireworks.chat.completions.create = instrumented_chat_create
            logger.info("✅ Fireworks chat completions auto-instrumented with GenOps governance")

        if original_embeddings_create:
            Fireworks.embeddings.create = instrumented_embeddings_create
            logger.info("✅ Fireworks embeddings auto-instrumented with GenOps governance")

        logger.info("🎉 Fireworks AI auto-instrumentation complete!")

    except Exception as e:
        logger.error(f"Auto-instrumentation failed: {e}")


# Export key classes and functions
__all__ = [
    "GenOpsFireworksAdapter",
    "FireworksModel",
    "FireworksTaskType",
    "FireworksResult",
    "FireworksSessionContext",
    "auto_instrument"
]
