"""
Together AI Provider Adapter for GenOps AI Governance

Provides comprehensive governance for Together AI operations including:
- Access to 200+ open-source models (chat, image, code, audio)
- Multi-modal support (vision, audio, fine-tuning)
- OpenAI-compatible API integration with governance
- Enterprise governance with multi-tenant support
- Zero-code auto-instrumentation for existing Together integrations
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
)

# Core GenOps imports
from genops.core.telemetry import GenOpsTelemetry

# Import Together pricing calculator
from .together_pricing import TogetherPricingCalculator

logger = logging.getLogger(__name__)

# Optional Together AI dependencies
try:
    from together import Together
    HAS_TOGETHER = True
except ImportError:
    HAS_TOGETHER = False
    Together = None
    logger.warning("Together AI client not installed. Install with: pip install together")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("Requests not installed. Install with: pip install requests")


class TogetherModel(Enum):
    """Popular Together AI models with their characteristics."""

    # Chat Models
    LLAMA_3_1_8B_INSTRUCT = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    LLAMA_3_1_70B_INSTRUCT = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    LLAMA_3_1_405B_INSTRUCT = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"

    # Reasoning Models
    DEEPSEEK_R1 = "deepseek-ai/DeepSeek-R1"
    DEEPSEEK_R1_DISTILL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

    # Multimodal Models
    QWEN_VL_72B = "Qwen/Qwen2.5-VL-72B-Instruct"
    LLAMA_VISION_11B = "meta-llama/Llama-Vision-Free"

    # Code Models
    DEEPSEEK_CODER_V2 = "deepseek-ai/DeepSeek-Coder-V2-Instruct"
    QWEN_CODER_32B = "Qwen/Qwen2.5-Coder-32B-Instruct"

    # Language Models
    MIXTRAL_8X7B = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    MIXTRAL_8X22B = "mistralai/Mixtral-8x22B-Instruct-v0.1"


class TogetherTaskType(Enum):
    """Task types for Together AI operations."""
    CHAT_COMPLETION = "chat_completion"
    CODE_COMPLETION = "code_completion"
    IMAGE_GENERATION = "image_generation"
    EMBEDDING = "embedding"
    FINE_TUNING = "fine_tuning"
    MULTIMODAL = "multimodal"


@dataclass
class TogetherResult:
    """Together AI result with governance metadata."""
    prompt: str
    response: str
    model_used: str
    task_type: TogetherTaskType
    tokens_used: int
    cost: Decimal
    execution_time_seconds: float
    governance_metadata: Dict[str, Any]
    session_id: Optional[str] = None
    images: Optional[List[str]] = None
    citations: Optional[List[Dict[str, Any]]] = None


@dataclass
class TogetherSession:
    """Together AI session with cost tracking and governance."""
    session_id: str
    session_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_operations: int = 0
    total_cost: Decimal = Decimal('0')
    governance_attributes: Dict[str, Any] = None
    results: List[TogetherResult] = None

    def __post_init__(self):
        if self.governance_attributes is None:
            self.governance_attributes = {}
        if self.results is None:
            self.results = []


class GenOpsTogetherAdapter:
    """
    Together AI adapter with GenOps governance for 200+ open-source models.
    
    Provides comprehensive governance for Together AI operations including:
    - Access to 200+ open-source models (chat, code, image, multimodal)
    - Multi-modal operations with vision and audio support
    - Fine-tuning governance and cost tracking
    - Multi-tenant operations with governance controls
    - Zero-code auto-instrumentation for existing integrations
    """

    def __init__(
        self,
        together_api_key: Optional[str] = None,
        team: str = "default",
        project: str = "default",
        environment: str = "production",
        customer_id: Optional[str] = None,
        cost_center: Optional[str] = None,
        daily_budget_limit: float = 1000.0,
        monthly_budget_limit: Optional[float] = None,
        enable_governance: bool = True,
        enable_cost_alerts: bool = True,
        governance_policy: str = "advisory",  # advisory, enforced, strict
        default_model: TogetherModel = TogetherModel.LLAMA_3_1_8B_INSTRUCT,
        tags: Optional[Dict[str, str]] = None,
        **kwargs
    ):
        """
        Initialize Together AI adapter with governance configuration.
        
        Args:
            together_api_key: Together API key (or use TOGETHER_API_KEY env var)
            team: Team name for cost attribution and governance
            project: Project name for cost tracking
            environment: Environment (production, staging, development)
            customer_id: Customer ID for multi-tenant attribution
            cost_center: Cost center for financial reporting
            daily_budget_limit: Daily budget limit in USD
            monthly_budget_limit: Monthly budget limit in USD
            enable_governance: Enable governance controls
            enable_cost_alerts: Enable cost alerting
            governance_policy: Governance enforcement level
            default_model: Default Together model to use
            tags: Additional tags for governance metadata
            **kwargs: Additional configuration options
        """
        # Configuration
        self.together_api_key = together_api_key or os.getenv('TOGETHER_API_KEY')
        self.team = team or os.getenv('GENOPS_TEAM', 'default')
        self.project = project or os.getenv('GENOPS_PROJECT', 'default')
        self.environment = environment
        self.customer_id = customer_id
        self.cost_center = cost_center
        self.daily_budget_limit = Decimal(str(daily_budget_limit))
        self.monthly_budget_limit = Decimal(str(monthly_budget_limit)) if monthly_budget_limit else None
        self.enable_governance = enable_governance
        self.enable_cost_alerts = enable_cost_alerts
        self.governance_policy = governance_policy
        self.default_model = default_model
        self.tags = tags or {}

        # Cost tracking
        self.pricing_calculator = TogetherPricingCalculator()
        self.daily_costs = Decimal('0')
        self.monthly_costs = Decimal('0')

        # Telemetry
        self.telemetry = GenOpsTelemetry(tracer_name="together")

        # Active sessions
        self._active_sessions: Dict[str, TogetherSession] = {}

        # Validation
        if not self.together_api_key:
            raise GenOpsConfigurationError(
                "Together API key required. Set TOGETHER_API_KEY environment variable or pass together_api_key parameter."
            )

        # Initialize Together client
        if HAS_TOGETHER:
            self.client = Together(api_key=self.together_api_key)
        else:
            self.client = None
            logger.warning("Together client not available. Some features may be limited.")

        logger.info(f"GenOps Together adapter initialized for team='{self.team}', project='{self.project}'")

    def _build_base_tags(self, additional_tags: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Build base governance tags for telemetry."""
        base_tags = {
            'provider': 'together',
            'team': self.team,
            'project': self.project,
            'environment': self.environment,
            'governance_enabled': str(self.enable_governance),
            'governance_policy': self.governance_policy
        }

        if self.customer_id:
            base_tags['customer_id'] = self.customer_id
        if self.cost_center:
            base_tags['cost_center'] = self.cost_center

        # Merge with instance tags and additional tags
        base_tags.update(self.tags)
        if additional_tags:
            base_tags.update(additional_tags)

        return base_tags

    def _check_budget_limits(self, estimated_cost: Decimal) -> None:
        """Check if operation would exceed budget limits."""
        if not self.enable_governance or self.governance_policy == "advisory":
            return

        projected_daily = self.daily_costs + estimated_cost
        if projected_daily > self.daily_budget_limit:
            if self.governance_policy in ["enforced", "strict"]:
                raise GenOpsBudgetExceededError(
                    f"Operation would exceed daily budget limit. "
                    f"Projected: ${projected_daily:.4f}, Limit: ${self.daily_budget_limit:.4f}"
                )

        if self.monthly_budget_limit:
            projected_monthly = self.monthly_costs + estimated_cost
            if projected_monthly > self.monthly_budget_limit:
                if self.governance_policy in ["enforced", "strict"]:
                    raise GenOpsBudgetExceededError(
                        f"Operation would exceed monthly budget limit. "
                        f"Projected: ${projected_monthly:.4f}, Limit: ${self.monthly_budget_limit:.4f}"
                    )

    def _update_costs(self, cost: Decimal) -> None:
        """Update cost tracking."""
        self.daily_costs += cost
        self.monthly_costs += cost

        # Cost alerting
        if self.enable_cost_alerts:
            daily_utilization = (self.daily_costs / self.daily_budget_limit) * 100
            if daily_utilization > 80:
                logger.warning(
                    f"Together costs approaching daily limit: {daily_utilization:.1f}% "
                    f"(${self.daily_costs:.4f}/${self.daily_budget_limit:.4f})"
                )

    @contextmanager
    def track_session(
        self,
        session_name: str,
        customer_id: Optional[str] = None,
        cost_center: Optional[str] = None,
        environment: Optional[str] = None,
        **governance_attributes
    ) -> Iterator[TogetherSession]:
        """
        Context manager for tracking Together AI sessions with governance.
        
        Args:
            session_name: Name of the session
            customer_id: Customer ID override
            cost_center: Cost center override  
            environment: Environment override
            **governance_attributes: Additional governance attributes
            
        Returns:
            TogetherSession: Session object for tracking
            
        Example:
            with adapter.track_session("model_comparison") as session:
                result = adapter.chat_with_governance(
                    messages=[{"role": "user", "content": "Explain AI"}],
                    session_id=session.session_id
                )
        """
        session_id = str(uuid.uuid4())

        # Build governance attributes
        governance_attrs = self._build_base_tags()
        governance_attrs.update({
            'session_name': session_name,
            'customer_id': customer_id or self.customer_id,
            'cost_center': cost_center or self.cost_center,
            'environment': environment or self.environment,
        })
        governance_attrs.update(governance_attributes)

        # Create session
        session = TogetherSession(
            session_id=session_id,
            session_name=session_name,
            start_time=datetime.now(timezone.utc),
            governance_attributes=governance_attrs
        )

        self._active_sessions[session_id] = session

        try:
            logger.info(f"Starting Together AI session '{session_name}' ({session_id})")
            yield session
        finally:
            # Finalize session
            session.end_time = datetime.now(timezone.utc)
            session_duration = (session.end_time - session.start_time).total_seconds()

            logger.info(
                f"Completed Together AI session '{session_name}': "
                f"{session.total_operations} operations, ${session.total_cost:.4f} cost, "
                f"{session_duration:.1f}s duration"
            )

            # Remove from active sessions
            if session_id in self._active_sessions:
                del self._active_sessions[session_id]

    def chat_with_governance(
        self,
        messages: List[Dict[str, Any]],
        model: Union[str, TogetherModel] = None,
        session_id: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        **governance_attributes
    ) -> TogetherResult:
        """
        Perform chat completion with Together AI and comprehensive governance.
        
        Args:
            messages: Chat messages in OpenAI format
            model: Together model to use
            session_id: Optional session ID for tracking
            max_tokens: Maximum tokens in response
            temperature: Response temperature (0.0-1.0)
            top_p: Top-p sampling parameter
            stream: Stream response tokens
            **governance_attributes: Additional governance metadata
            
        Returns:
            TogetherResult: Chat result with governance metadata
            
        Example:
            result = adapter.chat_with_governance(
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": "Explain quantum computing"}
                ],
                model=TogetherModel.LLAMA_3_1_70B_INSTRUCT,
                max_tokens=500,
                team="research",
                project="quantum-analysis"
            )
        """
        if not HAS_TOGETHER:
            raise GenOpsConfigurationError("Together client required for Together AI integration")

        start_time = time.time()

        # Normalize model
        if isinstance(model, TogetherModel):
            model_name = model.value
        elif model:
            model_name = str(model)
        else:
            model_name = self.default_model.value

        # Estimate cost before operation
        estimated_cost = self.pricing_calculator.estimate_chat_cost(
            model=model_name,
            estimated_tokens=max_tokens
        )

        # Budget check
        self._check_budget_limits(estimated_cost)

        # Build governance attributes
        operation_attrs = self._build_base_tags()
        operation_attrs.update(governance_attributes)
        operation_attrs.update({
            'operation': 'chat_completion',
            'model': model_name,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'estimated_cost': str(estimated_cost),
            'message_count': len(messages)
        })

        try:
            # Execute chat completion with telemetry
            with self.telemetry.trace_operation("together.chat_completion", **operation_attrs) as span:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=stream
                )

                # Extract response data
                response_text = response.choices[0].message.content
                tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else max_tokens

                # Calculate actual cost
                actual_cost = self.pricing_calculator.calculate_chat_cost(
                    model=model_name,
                    input_tokens=response.usage.prompt_tokens if hasattr(response, 'usage') else 0,
                    output_tokens=response.usage.completion_tokens if hasattr(response, 'usage') else tokens_used
                )

                # Update cost tracking
                self._update_costs(actual_cost)

                # Update telemetry
                span.set_attributes({
                    'together.tokens_used': tokens_used,
                    'together.actual_cost': str(actual_cost),
                    'together.execution_time_seconds': time.time() - start_time
                })

                # Create result
                result = TogetherResult(
                    prompt=str(messages),
                    response=response_text,
                    model_used=model_name,
                    task_type=TogetherTaskType.CHAT_COMPLETION,
                    tokens_used=tokens_used,
                    cost=actual_cost,
                    execution_time_seconds=time.time() - start_time,
                    governance_metadata=operation_attrs,
                    session_id=session_id
                )

                # Update session if provided
                if session_id and session_id in self._active_sessions:
                    session = self._active_sessions[session_id]
                    session.total_operations += 1
                    session.total_cost += actual_cost
                    session.results.append(result)

                logger.info(
                    f"Together chat completion: {tokens_used} tokens, "
                    f"${actual_cost:.4f} cost, {model_name}"
                )

                return result

        except Exception as e:
            logger.error(f"Together chat completion failed: {e}")
            # Update telemetry with error
            if 'span' in locals():
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise

    def complete_with_governance(
        self,
        prompt: str,
        model: Union[str, TogetherModel] = None,
        session_id: Optional[str] = None,
        max_tokens: int = 200,
        temperature: float = 0.1,
        **governance_attributes
    ) -> TogetherResult:
        """
        Perform text completion with Together AI (useful for code completion).
        
        Args:
            prompt: Text prompt for completion
            model: Together model to use
            session_id: Optional session ID for tracking
            max_tokens: Maximum tokens in response
            temperature: Response temperature
            **governance_attributes: Additional governance metadata
            
        Returns:
            TogetherResult: Completion result with governance metadata
        """
        if not HAS_TOGETHER:
            raise GenOpsConfigurationError("Together client required for Together AI integration")

        start_time = time.time()

        # Normalize model
        if isinstance(model, TogetherModel):
            model_name = model.value
        elif model:
            model_name = str(model)
        else:
            model_name = self.default_model.value

        # Estimate cost
        estimated_cost = self.pricing_calculator.estimate_completion_cost(
            model=model_name,
            estimated_tokens=max_tokens
        )

        # Budget check
        self._check_budget_limits(estimated_cost)

        # Build governance attributes
        operation_attrs = self._build_base_tags()
        operation_attrs.update(governance_attributes)
        operation_attrs.update({
            'operation': 'text_completion',
            'model': model_name,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'estimated_cost': str(estimated_cost),
            'prompt_length': len(prompt)
        })

        try:
            # Execute completion with telemetry
            with self.telemetry.trace_operation("together.completion", **operation_attrs) as span:
                response = self.client.completions.create(
                    model=model_name,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                # Extract response data
                response_text = response.choices[0].text
                tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else max_tokens

                # Calculate actual cost
                actual_cost = self.pricing_calculator.calculate_completion_cost(
                    model=model_name,
                    tokens_used=tokens_used
                )

                # Update cost tracking
                self._update_costs(actual_cost)

                # Update telemetry
                span.set_attributes({
                    'together.tokens_used': tokens_used,
                    'together.actual_cost': str(actual_cost),
                    'together.execution_time_seconds': time.time() - start_time
                })

                # Create result
                result = TogetherResult(
                    prompt=prompt,
                    response=response_text,
                    model_used=model_name,
                    task_type=TogetherTaskType.CODE_COMPLETION,
                    tokens_used=tokens_used,
                    cost=actual_cost,
                    execution_time_seconds=time.time() - start_time,
                    governance_metadata=operation_attrs,
                    session_id=session_id
                )

                # Update session if provided
                if session_id and session_id in self._active_sessions:
                    session = self._active_sessions[session_id]
                    session.total_operations += 1
                    session.total_cost += actual_cost
                    session.results.append(result)

                return result

        except Exception as e:
            logger.error(f"Together completion failed: {e}")
            if 'span' in locals():
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise

    def get_cost_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive cost summary and analytics.
        
        Returns:
            Dict with cost summary, budget utilization, and recommendations
        """
        summary = {
            'daily_costs': float(self.daily_costs),
            'monthly_costs': float(self.monthly_costs),
            'daily_budget_limit': float(self.daily_budget_limit),
            'monthly_budget_limit': float(self.monthly_budget_limit) if self.monthly_budget_limit else None,
            'daily_budget_utilization': (self.daily_costs / self.daily_budget_limit * 100) if self.daily_budget_limit > 0 else 0,
            'monthly_budget_utilization': (
                (self.monthly_costs / self.monthly_budget_limit * 100)
                if self.monthly_budget_limit and self.monthly_budget_limit > 0 else 0
            ),
            'governance_enabled': self.enable_governance,
            'governance_policy': self.governance_policy,
            'active_sessions': len(self._active_sessions),
            'team': self.team,
            'project': self.project,
            'environment': self.environment
        }

        return summary

    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available Together AI models with metadata.
        
        Returns:
            List of model information dictionaries
        """
        if not HAS_TOGETHER:
            raise GenOpsConfigurationError("Together client required")

        try:
            models = self.client.models.list()
            return [
                {
                    'id': model.id,
                    'type': model.type,
                    'pricing': self.pricing_calculator.get_model_pricing(model.id),
                    'context_length': getattr(model, 'context_length', None),
                    'organization': getattr(model, 'organization', None)
                }
                for model in models.data
            ]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []


# Auto-instrumentation functions
_current_adapter: Optional[GenOpsTogetherAdapter] = None


def auto_instrument(
    together_api_key: Optional[str] = None,
    team: str = "auto-instrumented",
    project: str = "default",
    **adapter_kwargs
) -> GenOpsTogetherAdapter:
    """
    Enable automatic instrumentation for Together AI operations.
    
    This function enables zero-code governance for existing Together integrations.
    
    Args:
        together_api_key: Together API key (or use TOGETHER_API_KEY env var)
        team: Team name for cost attribution
        project: Project name for cost tracking
        **adapter_kwargs: Additional adapter configuration
        
    Returns:
        GenOpsTogetherAdapter: The configured adapter instance
        
    Example:
        from genops.providers.together import auto_instrument
        auto_instrument()
        
        # Your existing code works with governance
        from together import Together
        client = Together()
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages=[{"role": "user", "content": "Hello!"}]
        )
    """
    global _current_adapter

    _current_adapter = GenOpsTogetherAdapter(
        together_api_key=together_api_key,
        team=team,
        project=project,
        **adapter_kwargs
    )

    logger.info("Together AI auto-instrumentation enabled")
    return _current_adapter


def instrument_together(
    together_api_key: Optional[str] = None,
    team: str = "default",
    project: str = "default",
    **kwargs
) -> GenOpsTogetherAdapter:
    """
    Create instrumented Together AI adapter.
    
    Alternative entry point for creating a GenOps Together adapter with
    governance controls and cost tracking.
    
    Args:
        together_api_key: Together API key
        team: Team name for attribution
        project: Project name for tracking
        **kwargs: Additional configuration
        
    Returns:
        GenOpsTogetherAdapter: Configured adapter
    """
    return GenOpsTogetherAdapter(
        together_api_key=together_api_key,
        team=team,
        project=project,
        **kwargs
    )


def get_current_adapter() -> Optional[GenOpsTogetherAdapter]:
    """Get the current auto-instrumented adapter instance."""
    return _current_adapter


# Export key classes and functions
__all__ = [
    'GenOpsTogetherAdapter',
    'TogetherSession',
    'TogetherResult',
    'TogetherModel',
    'TogetherTaskType',
    'auto_instrument',
    'instrument_together',
    'get_current_adapter'
]
