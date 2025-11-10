#!/usr/bin/env python3
"""
Haystack AI Framework Adapter for GenOps Governance

Provides comprehensive governance telemetry for Haystack AI orchestration framework,
including pipeline-level tracking, component monitoring, and multi-provider cost aggregation.

Usage:
    from genops.providers.haystack_adapter import GenOpsHaystackAdapter
    
    adapter = GenOpsHaystackAdapter(
        team="ai-research",
        project="rag-system",
        daily_budget_limit=100.0
    )
    
    # Track entire pipeline execution
    with adapter.track_pipeline("document-qa") as context:
        result = pipeline.run({"query": "What is retrieval augmented generation?"})
        print(f"Total cost: ${context.total_cost:.6f}")

Features:
    - End-to-end pipeline governance and cost tracking
    - Component-level instrumentation and performance monitoring
    - Multi-provider cost aggregation (OpenAI, Anthropic, Hugging Face, etc.)
    - RAG workflow specialization with retrieval and generation tracking
    - Agent workflow governance with decision and tool usage monitoring
    - Enterprise compliance patterns and multi-tenant governance
"""

import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Union, TYPE_CHECKING

# TYPE_CHECKING imports to avoid circular imports
if TYPE_CHECKING:
    from genops.providers.haystack_cost_aggregator import HaystackCostAggregator
    from genops.providers.haystack_monitor import HaystackMonitor
from datetime import datetime
import random
from functools import wraps

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# GenOps core imports
from genops.core.telemetry import GenOpsTelemetry

logger = logging.getLogger(__name__)

# Check for Haystack availability
try:
    import haystack
    from haystack import Pipeline, component
    from haystack.core.component import Component
    HAS_HAYSTACK = True
    logger.info(f"Haystack {haystack.__version__} detected")
except ImportError:
    HAS_HAYSTACK = False
    Pipeline = None
    Component = None
    component = None
    logger.warning("Haystack not installed. Install with: pip install haystack-ai")


@dataclass
class HaystackComponentResult:
    """Result from a tracked Haystack component execution."""
    component_name: str
    component_type: str
    execution_time_seconds: float
    cost: Decimal
    provider: Optional[str] = None
    model: Optional[str] = None
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    status: str = "success"
    error_message: Optional[str] = None
    custom_attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HaystackPipelineResult:
    """Result from a tracked Haystack pipeline execution."""
    pipeline_name: str
    pipeline_id: str
    total_cost: Decimal
    total_execution_time_seconds: float
    component_results: List[HaystackComponentResult]
    cost_by_provider: Dict[str, Decimal]
    cost_by_component: Dict[str, Decimal]
    total_components: int
    successful_components: int
    failed_components: int
    governance_attributes: Dict[str, Any]
    start_time: datetime
    end_time: datetime


@dataclass 
class HaystackSessionContext:
    """Context for tracking multi-pipeline sessions in Haystack workflows."""
    session_id: str
    session_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_pipelines: int = 0
    total_cost: Decimal = Decimal('0')
    governance_attributes: Dict[str, Any] = field(default_factory=dict)
    pipeline_results: List[HaystackPipelineResult] = field(default_factory=list)
    
    def add_pipeline_result(self, result: HaystackPipelineResult):
        """Add a pipeline result to the session."""
        self.pipeline_results.append(result)
        self.total_pipelines += 1
        self.total_cost += result.total_cost


class HaystackPipelineContext:
    """Context manager for tracking Haystack pipeline execution."""
    
    def __init__(self, adapter: 'GenOpsHaystackAdapter', pipeline_name: str, 
                 pipeline_id: str, **governance_attrs):
        self.adapter = adapter
        self.pipeline_name = pipeline_name
        self.pipeline_id = pipeline_id
        self.governance_attrs = governance_attrs
        
        # Tracking state
        self.start_time = None
        self.end_time = None
        self.component_results: List[HaystackComponentResult] = []
        self.total_cost = Decimal('0')
        self.span = None
        
    def __enter__(self):
        """Start pipeline tracking."""
        self.start_time = datetime.utcnow()
        
        # Create OpenTelemetry span for the entire pipeline
        self.span = self.adapter.telemetry.tracer.start_span(
            f"haystack.pipeline.{self.pipeline_name}"
        )
        
        # Set pipeline attributes
        self.span.set_attribute("genops.provider", "haystack")
        self.span.set_attribute("genops.pipeline.name", self.pipeline_name)
        self.span.set_attribute("genops.pipeline.id", self.pipeline_id)
        self.span.set_attribute("genops.framework", "haystack")
        
        # Set governance attributes
        for key, value in self.governance_attrs.items():
            if value is not None:
                self.span.set_attribute(f"genops.{key}", str(value))
        
        # Set adapter-level governance attributes
        self.span.set_attribute("genops.team", self.adapter.team)
        self.span.set_attribute("genops.project", self.adapter.project)
        self.span.set_attribute("genops.environment", self.adapter.environment)
        
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Complete pipeline tracking."""
        self.end_time = datetime.utcnow()
        
        # Calculate totals
        total_execution_time = (self.end_time - self.start_time).total_seconds()
        
        # Aggregate costs by provider and component
        cost_by_provider = {}
        cost_by_component = {}
        
        for result in self.component_results:
            # By provider
            if result.provider:
                cost_by_provider[result.provider] = (
                    cost_by_provider.get(result.provider, Decimal('0')) + result.cost
                )
            
            # By component
            cost_by_component[result.component_name] = (
                cost_by_component.get(result.component_name, Decimal('0')) + result.cost
            )
        
        # Create pipeline result
        pipeline_result = HaystackPipelineResult(
            pipeline_name=self.pipeline_name,
            pipeline_id=self.pipeline_id,
            total_cost=self.total_cost,
            total_execution_time_seconds=total_execution_time,
            component_results=self.component_results,
            cost_by_provider=cost_by_provider,
            cost_by_component=cost_by_component,
            total_components=len(self.component_results),
            successful_components=len([r for r in self.component_results if r.status == "success"]),
            failed_components=len([r for r in self.component_results if r.status == "error"]),
            governance_attributes=self.governance_attrs,
            start_time=self.start_time,
            end_time=self.end_time
        )
        
        # Set final span attributes
        self.span.set_attribute("genops.cost.total", float(self.total_cost))
        self.span.set_attribute("genops.pipeline.components.total", len(self.component_results))
        self.span.set_attribute("genops.pipeline.components.successful", pipeline_result.successful_components)
        self.span.set_attribute("genops.pipeline.components.failed", pipeline_result.failed_components)
        self.span.set_attribute("genops.pipeline.execution_time_seconds", total_execution_time)
        
        # Set provider cost breakdown
        for provider, cost in cost_by_provider.items():
            self.span.set_attribute(f"genops.cost.provider.{provider}", float(cost))
        
        # Set span status
        if exc_type is None:
            self.span.set_status(Status(StatusCode.OK))
        else:
            self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
            self.span.record_exception(exc_val)
        
        # Finish span
        self.span.end()
        
        # Store result in adapter
        self.adapter._pipeline_results[self.pipeline_id] = pipeline_result
        
        # Update adapter totals
        self.adapter._daily_costs += self.total_cost
        
        return pipeline_result
    
    def add_component_result(self, result: HaystackComponentResult):
        """Add a component execution result to the pipeline tracking."""
        self.component_results.append(result)
        self.total_cost += result.cost
        
        # Create span for the component
        with self.adapter.telemetry.tracer.start_as_current_span(
            f"haystack.component.{result.component_name}"
        ) as component_span:
            component_span.set_attribute("genops.component.name", result.component_name)
            component_span.set_attribute("genops.component.type", result.component_type)
            component_span.set_attribute("genops.cost.total", float(result.cost))
            component_span.set_attribute("genops.execution_time_seconds", result.execution_time_seconds)
            
            if result.provider:
                component_span.set_attribute("genops.provider", result.provider)
            if result.model:
                component_span.set_attribute("genops.model", result.model)
            if result.tokens_input:
                component_span.set_attribute("genops.tokens.input", result.tokens_input)
            if result.tokens_output:
                component_span.set_attribute("genops.tokens.output", result.tokens_output)
            
            # Set custom attributes
            for key, value in result.custom_attributes.items():
                component_span.set_attribute(f"genops.component.{key}", str(value))
            
            # Set status
            if result.status == "error":
                component_span.set_status(Status(StatusCode.ERROR, result.error_message or "Component failed"))
            else:
                component_span.set_status(Status(StatusCode.OK))


class GenOpsHaystackAdapter:
    """
    Haystack AI framework adapter with comprehensive GenOps governance.
    
    Provides end-to-end tracking for Haystack pipelines including:
    - Pipeline-level cost and performance monitoring
    - Component-level instrumentation and telemetry
    - Multi-provider cost aggregation
    - RAG workflow specialization
    - Agent workflow governance
    - Enterprise compliance and multi-tenant support
    """
    
    def __init__(
        self,
        team: str = "default-team",
        project: str = "haystack-integration",
        environment: str = "development",
        daily_budget_limit: float = 100.0,
        monthly_budget_limit: Optional[float] = None,
        governance_policy: str = "advisory",  # "advisory", "enforcing", "monitoring"
        enable_cost_alerts: bool = True,
        enable_component_tracking: bool = True,
        enable_pipeline_caching: bool = True,
        **kwargs
    ):
        """
        Initialize Haystack adapter with governance configuration.
        
        Args:
            team: Team name for cost attribution
            project: Project name for cost attribution
            environment: Environment (development, staging, production)
            daily_budget_limit: Daily spending limit in USD
            monthly_budget_limit: Monthly spending limit in USD
            governance_policy: Policy enforcement level
            enable_cost_alerts: Enable cost alert notifications
            enable_component_tracking: Enable individual component tracking
            enable_pipeline_caching: Enable pipeline result caching
            **kwargs: Additional configuration options
        """
        if not HAS_HAYSTACK:
            error_msg = (
                "❌ Haystack AI framework not installed or not compatible.\n\n"
                "🔧 Quick Fix:\n"
                "   pip install haystack-ai\n\n"
                "📋 For specific providers, also install:\n"
                "   pip install openai anthropic cohere-ai transformers\n\n"
                "🔍 Validate your setup:\n"
                "   python scripts/validate_setup.py --fix-issues\n\n"
                "📚 Documentation: https://docs.haystack.deepset.ai/docs/installation"
            )
            logger.warning(error_msg)
            
            # In strict mode, raise an exception with actionable guidance
            if kwargs.get('strict_mode', False):
                raise ImportError(
                    f"Haystack AI framework is required but not installed. {error_msg}"
                )
        
        self.team = team
        self.project = project
        self.environment = environment
        self.daily_budget_limit = daily_budget_limit
        self.monthly_budget_limit = monthly_budget_limit or (daily_budget_limit * 30)
        self.governance_policy = governance_policy
        self.enable_cost_alerts = enable_cost_alerts
        self.enable_component_tracking = enable_component_tracking
        self.enable_pipeline_caching = enable_pipeline_caching
        
        # Initialize telemetry
        self.telemetry = GenOpsTelemetry(tracer_name="haystack")
        
        # Initialize error tracking
        self.initialization_errors = []
        
        # Initialize cost aggregator and monitor with lazy imports
        self.cost_aggregator = None
        self.monitor = None
        self._lazy_init_components(team, project)
        
        
        # Cost tracking
        self._daily_costs = Decimal("0.00")
        self._monthly_costs = Decimal("0.00")
        self._pipeline_results: Dict[str, HaystackPipelineResult] = {}
        self._active_sessions: Dict[str, HaystackSessionContext] = {}
        
        # Component type registry for cost estimation
        self._component_cost_registry = {
            'OpenAIGenerator': {'provider': 'openai', 'cost_per_token': 0.00002},
            'AnthropicGenerator': {'provider': 'anthropic', 'cost_per_token': 0.00001},
            'HuggingFaceGenerator': {'provider': 'huggingface', 'cost_per_token': 0.000001},
            'CohereGenerator': {'provider': 'cohere', 'cost_per_token': 0.00001},
            'EmbeddingRetriever': {'provider': 'generic', 'cost_per_operation': 0.0001},
            'InMemoryDocumentStore': {'provider': 'local', 'cost_per_operation': 0.0},
        }
        
        # Enhanced error handling configuration
        self.retry_config = {
            'max_retries': kwargs.get('max_retries', 3),
            'base_delay': kwargs.get('retry_base_delay', 1.0),
            'max_delay': kwargs.get('retry_max_delay', 60.0),
            'backoff_factor': kwargs.get('retry_backoff_factor', 2.0),
            'jitter': kwargs.get('retry_jitter', True)
        }
        
        # Error tracking and diagnostics
        self.error_stats = {
            'total_errors': 0,
            'retry_attempts': 0,
            'error_types': {},
            'component_failures': {},
            'provider_failures': {}
        }
        
        logger.info(f"GenOps Haystack adapter initialized for team '{team}', project '{project}'")
    
    def _lazy_init_components(self, team: str, project: str):
        """Lazily initialize components to avoid circular imports."""
        try:
            from genops.providers.haystack_cost_aggregator import HaystackCostAggregator
            from genops.providers.haystack_monitor import HaystackMonitor
            
            self.cost_aggregator = HaystackCostAggregator()
            self.monitor = HaystackMonitor(team=team, project=project)
            
        except ImportError as e:
            error_msg = (
                f"❌ GenOps Haystack components not available: {e}\n\n"
                "🔧 Quick Fix:\n"
                "   pip install --upgrade genops-ai[haystack]\n\n"
                "🔍 Validate installation:\n"
                "   python scripts/validate_setup.py\n\n"
                "📚 If issues persist, see: docs/troubleshooting.md"
            )
            logger.warning(error_msg)
            
            # Store error details for diagnostics
            self.initialization_errors = [f"Cost aggregator/monitor: {e}"]
            
        except Exception as e:
            error_msg = (
                f"❌ Unexpected error initializing GenOps components: {e}\n\n"
                "🔧 Troubleshooting steps:\n"
                "   1. python scripts/validate_setup.py --detailed\n"
                "   2. pip install --force-reinstall genops-ai[haystack]\n"
                "   3. Check OpenTelemetry configuration\n\n"
                "📚 Documentation: docs/integrations/haystack.md#troubleshooting"
            )
            logger.error(error_msg)
            
            self.initialization_errors = [f"Unexpected error: {e}"]
    
    def _ensure_components_initialized(self):
        """Ensure components are initialized before use."""
        if self.cost_aggregator is None or self.monitor is None:
            self._lazy_init_components(self.team, self.project)
    
    @contextmanager
    def track_pipeline(self, pipeline_name: str, **governance_attrs):
        """
        Context manager for tracking a complete Haystack pipeline execution.
        
        Args:
            pipeline_name: Name of the pipeline being executed
            **governance_attrs: Additional governance attributes
            
        Yields:
            HaystackPipelineContext: Context for tracking the pipeline
            
        Example:
            with adapter.track_pipeline("rag-qa", customer_id="customer-123") as context:
                result = pipeline.run({"query": "What is RAG?"})
                print(f"Total cost: ${context.total_cost:.6f}")
        """
        pipeline_id = str(uuid.uuid4())
        
        # Merge governance attributes with adapter defaults
        merged_attrs = {
            "team": self.team,
            "project": self.project,
            "environment": self.environment,
            **governance_attrs
        }
        
        with HaystackPipelineContext(
            self, pipeline_name, pipeline_id, **merged_attrs
        ) as context:
            yield context
    
    @contextmanager 
    def track_session(self, session_name: str, **governance_attrs):
        """
        Context manager for tracking multi-pipeline sessions.
        
        Args:
            session_name: Name of the session
            **governance_attrs: Additional governance attributes
            
        Yields:
            HaystackSessionContext: Session context for tracking
            
        Example:
            with adapter.track_session("research-experiment") as session:
                # Run multiple pipelines
                for query in queries:
                    with adapter.track_pipeline("qa-pipeline") as pipeline_ctx:
                        result = pipeline.run({"query": query})
                        
                print(f"Session cost: ${session.total_cost:.6f}")
        """
        session_id = str(uuid.uuid4())
        
        session_context = HaystackSessionContext(
            session_id=session_id,
            session_name=session_name,
            start_time=datetime.utcnow(),
            governance_attributes=governance_attrs
        )
        
        self._active_sessions[session_id] = session_context
        
        try:
            yield session_context
        finally:
            session_context.end_time = datetime.utcnow()
            
            # Create session telemetry span
            with self.telemetry.tracer.start_as_current_span(
                f"haystack.session.{session_name}"
            ) as session_span:
                session_span.set_attribute("genops.session.name", session_name)
                session_span.set_attribute("genops.session.id", session_id)
                session_span.set_attribute("genops.session.total_pipelines", session_context.total_pipelines)
                session_span.set_attribute("genops.session.total_cost", float(session_context.total_cost))
                
                # Set governance attributes
                for key, value in governance_attrs.items():
                    if value is not None:
                        session_span.set_attribute(f"genops.{key}", str(value))
                        
                session_span.set_status(Status(StatusCode.OK))
            
            del self._active_sessions[session_id]
    
    def estimate_component_cost(self, component_name: str, component_type: str, 
                              tokens_input: int = 0, tokens_output: int = 0,
                              operations: int = 1) -> tuple[Decimal, str]:
        """
        Estimate cost for a Haystack component execution.
        
        Args:
            component_name: Name of the component
            component_type: Type/class of the component
            tokens_input: Number of input tokens
            tokens_output: Number of output tokens
            operations: Number of operations performed
            
        Returns:
            Tuple of (estimated_cost, provider)
        """
        # Look up component in registry
        registry_entry = self._component_cost_registry.get(component_type)
        if not registry_entry:
            # Default estimation for unknown components
            return Decimal("0.001"), "unknown"
        
        provider = registry_entry['provider']
        
        if 'cost_per_token' in registry_entry:
            # Token-based pricing
            cost_per_token = Decimal(str(registry_entry['cost_per_token']))
            total_tokens = tokens_input + tokens_output
            total_cost = cost_per_token * total_tokens
        else:
            # Operation-based pricing
            cost_per_operation = Decimal(str(registry_entry['cost_per_operation']))
            total_cost = cost_per_operation * operations
        
        return total_cost, provider
    
    def track_component_execution(self, component_name: str, component_type: str,
                                execution_func, *args, **kwargs) -> HaystackComponentResult:
        """
        Track the execution of a Haystack component with enhanced error handling and retry logic.
        
        Args:
            component_name: Name of the component
            component_type: Type/class of the component
            execution_func: Function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            HaystackComponentResult: Tracking result with cost and performance data
        """
        return self._execute_with_retry(
            component_name, component_type, execution_func, *args, **kwargs
        )
    
    def _execute_with_retry(self, component_name: str, component_type: str,
                           execution_func, *args, **kwargs) -> HaystackComponentResult:
        """
        Execute component with intelligent retry logic and comprehensive error handling.
        """
        start_time = time.time()
        last_exception = None
        retry_count = 0
        
        for attempt in range(self.retry_config['max_retries'] + 1):
            try:
                # Add artificial delay for retries
                if attempt > 0:
                    delay = self._calculate_retry_delay(attempt)
                    logger.info(
                        f"Retrying {component_name} (attempt {attempt + 1}/{self.retry_config['max_retries'] + 1}) "
                        f"after {delay:.2f}s delay"
                    )
                    time.sleep(delay)
                    retry_count += 1
                    self.error_stats['retry_attempts'] += 1
                
                # Execute the component
                result = execution_func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Extract token information if available
                tokens_input, tokens_output = self._extract_token_usage(result)
                
                # Estimate cost
                estimated_cost, provider = self.estimate_component_cost(
                    component_name, component_type, tokens_input, tokens_output
                )
                
                # Log successful execution after retries
                if retry_count > 0:
                    logger.info(
                        f"Component {component_name} succeeded after {retry_count} retries "
                        f"(total time: {execution_time:.2f}s)"
                    )
                
                return HaystackComponentResult(
                    component_name=component_name,
                    component_type=component_type,
                    execution_time_seconds=execution_time,
                    cost=estimated_cost,
                    provider=provider,
                    tokens_input=tokens_input if tokens_input > 0 else None,
                    tokens_output=tokens_output if tokens_output > 0 else None,
                    status="success",
                    custom_attributes={
                        'retry_count': retry_count,
                        'final_attempt': attempt + 1
                    }
                )
                
            except Exception as e:
                last_exception = e
                error_type = type(e).__name__
                
                # Track error statistics
                self._track_error(component_name, component_type, error_type, str(e))
                
                # Check if error is retryable
                if not self._is_retryable_error(e) or attempt >= self.retry_config['max_retries']:
                    break
                
                logger.warning(
                    f"Component {component_name} failed with {error_type}: {str(e)}. "
                    f"Will retry (attempt {attempt + 1}/{self.retry_config['max_retries']})..."
                )
        
        # All retries exhausted or non-retryable error
        execution_time = time.time() - start_time
        error_message = str(last_exception) if last_exception else "Unknown error"
        
        logger.error(
            f"Component {component_name} failed permanently after {retry_count} retries. "
            f"Final error: {error_message}"
        )
        
        return HaystackComponentResult(
            component_name=component_name,
            component_type=component_type,
            execution_time_seconds=execution_time,
            cost=Decimal("0.00"),
            status="error",
            error_message=error_message,
            custom_attributes={
                'retry_count': retry_count,
                'error_type': type(last_exception).__name__ if last_exception else 'Unknown',
                'retryable': self._is_retryable_error(last_exception) if last_exception else False
            }
        )
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """
        Calculate retry delay with exponential backoff and optional jitter.
        """
        delay = min(
            self.retry_config['base_delay'] * (self.retry_config['backoff_factor'] ** (attempt - 1)),
            self.retry_config['max_delay']
        )
        
        # Add jitter to avoid thundering herd
        if self.retry_config['jitter']:
            delay *= (0.5 + random.random() * 0.5)
        
        return delay
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retryable based on error type and message.
        """
        retryable_errors = {
            'ConnectionError', 'TimeoutError', 'HTTPError', 
            'ServiceUnavailableError', 'RateLimitError',
            'APIError', 'NetworkError', 'TemporaryFailure'
        }
        
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Check error type
        if error_type in retryable_errors:
            return True
        
        # Check error message for retryable patterns
        retryable_patterns = [
            'timeout', 'connection', 'network', 'rate limit', 
            'service unavailable', 'temporary', 'retry', 
            'busy', 'overload', 'throttle'
        ]
        
        return any(pattern in error_message for pattern in retryable_patterns)
    
    def _extract_token_usage(self, result: Any) -> tuple[int, int]:
        """
        Extract token usage information from component result.
        """
        tokens_input = 0
        tokens_output = 0
        
        if isinstance(result, dict):
            if 'usage' in result:
                usage = result['usage']
                tokens_input = usage.get('prompt_tokens', 0)
                tokens_output = usage.get('completion_tokens', 0)
            elif 'meta' in result:
                meta = result['meta']
                tokens_input = meta.get('prompt_tokens', 0)  
                tokens_output = meta.get('completion_tokens', 0)
            # Check for OpenAI-style usage in nested structures
            elif hasattr(result, 'get'):
                for key in result.keys():
                    if isinstance(result[key], dict) and 'usage' in result[key]:
                        usage = result[key]['usage']
                        tokens_input = usage.get('prompt_tokens', 0)
                        tokens_output = usage.get('completion_tokens', 0)
                        break
        
        return tokens_input, tokens_output
    
    def _track_error(self, component_name: str, component_type: str, 
                    error_type: str, error_message: str):
        """
        Track error statistics for diagnostics and monitoring.
        """
        self.error_stats['total_errors'] += 1
        
        # Track by error type
        if error_type not in self.error_stats['error_types']:
            self.error_stats['error_types'][error_type] = 0
        self.error_stats['error_types'][error_type] += 1
        
        # Track by component
        if component_name not in self.error_stats['component_failures']:
            self.error_stats['component_failures'][component_name] = 0
        self.error_stats['component_failures'][component_name] += 1
        
        # Track by provider (if identifiable)
        registry_entry = self._component_cost_registry.get(component_type)
        if registry_entry:
            provider = registry_entry['provider']
            if provider not in self.error_stats['provider_failures']:
                self.error_stats['provider_failures'][provider] = 0
            self.error_stats['provider_failures'][provider] += 1
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive cost summary for the current period with error diagnostics.
        
        Returns:
            Dictionary with cost breakdown, budget utilization, and error statistics
        """
        self._ensure_components_initialized()
        daily_budget_utilization = (
            (float(self._daily_costs) / self.daily_budget_limit) * 100 
            if self.daily_budget_limit > 0 else 0
        )
        
        monthly_budget_utilization = (
            (float(self._monthly_costs) / self.monthly_budget_limit) * 100
            if self.monthly_budget_limit > 0 else 0
        )
        
        # Aggregate costs by provider across all pipelines
        cost_by_provider = {}
        total_pipelines = len(self._pipeline_results)
        total_components = 0
        successful_components = 0
        failed_components = 0
        
        for pipeline_result in self._pipeline_results.values():
            for provider, cost in pipeline_result.cost_by_provider.items():
                cost_by_provider[provider] = cost_by_provider.get(provider, Decimal('0')) + cost
            
            total_components += pipeline_result.total_components
            successful_components += pipeline_result.successful_components
            failed_components += pipeline_result.failed_components
        
        # Calculate reliability metrics
        success_rate = (
            (successful_components / total_components * 100) 
            if total_components > 0 else 100.0
        )
        
        return {
            "daily_costs": float(self._daily_costs),
            "monthly_costs": float(self._monthly_costs),
            "daily_budget_limit": self.daily_budget_limit,
            "monthly_budget_limit": self.monthly_budget_limit,
            "daily_budget_utilization": daily_budget_utilization,
            "monthly_budget_utilization": monthly_budget_utilization,
            "cost_by_provider": {k: float(v) for k, v in cost_by_provider.items()},
            "total_pipelines_executed": total_pipelines,
            "total_components_executed": total_components,
            "successful_components": successful_components,
            "failed_components": failed_components,
            "success_rate_percent": success_rate,
            "governance_policy": self.governance_policy,
            "team": self.team,
            "project": self.project,
            "environment": self.environment,
            "error_statistics": self.get_error_diagnostics(),
            "retry_configuration": self.retry_config
        }
    
    def get_error_diagnostics(self) -> Dict[str, Any]:
        """
        Get comprehensive error diagnostics and failure analysis.
        
        Returns:
            Dictionary with detailed error statistics and recommendations
        """
        total_operations = sum(
            pipeline.total_components for pipeline in self._pipeline_results.values()
        )
        
        error_rate = (
            (self.error_stats['total_errors'] / total_operations * 100)
            if total_operations > 0 else 0.0
        )
        
        # Generate recommendations based on error patterns
        recommendations = self._generate_error_recommendations()
        
        # Find most problematic components and providers
        most_problematic_component = max(
            self.error_stats['component_failures'].items(),
            key=lambda x: x[1],
            default=('none', 0)
        )
        
        most_problematic_provider = max(
            self.error_stats['provider_failures'].items(),
            key=lambda x: x[1],
            default=('none', 0)
        )
        
        return {
            "total_errors": self.error_stats['total_errors'],
            "retry_attempts": self.error_stats['retry_attempts'],
            "error_rate_percent": error_rate,
            "error_types": dict(self.error_stats['error_types']),
            "component_failures": dict(self.error_stats['component_failures']),
            "provider_failures": dict(self.error_stats['provider_failures']),
            "most_problematic_component": {
                "name": most_problematic_component[0],
                "failure_count": most_problematic_component[1]
            },
            "most_problematic_provider": {
                "name": most_problematic_provider[0],
                "failure_count": most_problematic_provider[1]
            },
            "recommendations": recommendations
        }
    
    def _generate_error_recommendations(self) -> List[str]:
        """
        Generate actionable recommendations based on error patterns.
        """
        recommendations = []
        
        # High error rate recommendation
        total_ops = sum(pipeline.total_components for pipeline in self._pipeline_results.values())
        if total_ops > 0:
            error_rate = (self.error_stats['total_errors'] / total_ops) * 100
            if error_rate > 10:
                recommendations.append(
                    f"High error rate detected ({error_rate:.1f}%). Consider reviewing component configurations and provider connectivity."
                )
        
        # High retry rate recommendation
        if self.error_stats['retry_attempts'] > 10:
            recommendations.append(
                f"High retry count ({self.error_stats['retry_attempts']}) detected. "
                "Consider increasing timeout values or checking network stability."
            )
        
        # Component-specific recommendations
        for component, failures in self.error_stats['component_failures'].items():
            if failures > 5:
                recommendations.append(
                    f"Component '{component}' has {failures} failures. "
                    "Review configuration and input validation."
                )
        
        # Provider-specific recommendations
        for provider, failures in self.error_stats['provider_failures'].items():
            if failures > 3:
                recommendations.append(
                    f"Provider '{provider}' has {failures} failures. "
                    "Check API keys, rate limits, and service status."
                )
        
        # Error type specific recommendations
        if 'ConnectionError' in self.error_stats['error_types']:
            recommendations.append(
                "Connection errors detected. Verify network connectivity and firewall settings."
            )
        
        if 'RateLimitError' in self.error_stats['error_types']:
            recommendations.append(
                "Rate limit errors detected. Consider implementing request throttling or upgrading service tier."
            )
        
        if not recommendations:
            recommendations.append(
                "System operating normally. Error rates are within acceptable limits."
            )
        
        return recommendations
    
    def get_pipeline_result(self, pipeline_id: str) -> Optional[HaystackPipelineResult]:
        """Get a specific pipeline execution result."""
        return self._pipeline_results.get(pipeline_id)
    
    def get_recent_pipeline_results(self, limit: int = 10) -> List[HaystackPipelineResult]:
        """Get the most recent pipeline execution results."""
        return sorted(
            self._pipeline_results.values(),
            key=lambda x: x.end_time,
            reverse=True
        )[:limit]
    
    def get_initialization_status(self) -> Dict[str, Any]:
        """
        Get detailed initialization status with actionable error messages.
        
        Returns:
            Dictionary with initialization status and fix suggestions
        """
        status = {
            "initialized": True,
            "errors": [],
            "warnings": [],
            "component_status": {},
            "fix_suggestions": []
        }
        
        # Check for initialization errors
        if hasattr(self, 'initialization_errors') and self.initialization_errors:
            status["initialized"] = False
            status["errors"] = self.initialization_errors
        
        # Check component availability
        components = {
            "haystack": HAS_HAYSTACK,
            "cost_aggregator": self.cost_aggregator is not None,
            "monitor": self.monitor is not None,
            "telemetry": self.telemetry is not None
        }
        
        for component, available in components.items():
            status["component_status"][component] = "available" if available else "unavailable"
            if not available:
                if component == "haystack":
                    status["fix_suggestions"].append({
                        "issue": "Haystack AI framework not available",
                        "fix": "pip install haystack-ai",
                        "priority": "high",
                        "validation": "python scripts/validate_setup.py --provider openai"
                    })
                elif component in ["cost_aggregator", "monitor"]:
                    status["fix_suggestions"].append({
                        "issue": f"GenOps {component} not available",
                        "fix": "pip install --upgrade genops-ai[haystack]",
                        "priority": "medium",
                        "validation": "python scripts/validate_setup.py"
                    })
        
        # Generate summary message
        if not status["initialized"] or not all(components.values()):
            status["summary"] = "⚠️ Initialization incomplete - some features may not work properly"
        else:
            status["summary"] = "✅ All components initialized successfully"
        
        return status
    
    def print_initialization_status(self):
        """Print user-friendly initialization status with fix suggestions."""
        status = self.get_initialization_status()
        
        print(f"\n🔍 GenOps Haystack Adapter Status")
        print("-" * 40)
        print(f"{status['summary']}")
        
        if status["errors"]:
            print(f"\n❌ Initialization Errors:")
            for error in status["errors"]:
                print(f"   • {error}")
        
        if status["fix_suggestions"]:
            print(f"\n🔧 Recommended Fixes:")
            for i, fix in enumerate(status["fix_suggestions"], 1):
                print(f"   {i}. {fix['issue']}")
                print(f"      Fix: {fix['fix']}")
                if fix.get("validation"):
                    print(f"      Validate: {fix['validation']}")
                print()
        
        if status["initialized"] and not status["errors"]:
            print(f"\n🎉 Ready to use! Try:")
            print(f"   with adapter.track_pipeline('my-pipeline') as context:")
            print(f"       # Your Haystack code here")


# Auto-instrumentation function for easy setup
def auto_instrument():
    """
    Automatically instrument Haystack pipelines with GenOps governance tracking.
    
    This function patches Haystack's Pipeline class to automatically track
    all pipeline executions with minimal code changes.
    
    Usage:
        from genops.providers.haystack_adapter import auto_instrument
        auto_instrument()
        
        # Your existing Haystack code works unchanged
        pipeline = Pipeline()
        # ... add components ...
        result = pipeline.run({"query": "What is RAG?"})
        # ✅ Automatic cost tracking and governance added!
    """
    if not HAS_HAYSTACK:
        logger.warning("Cannot auto-instrument: Haystack not installed")
        return
    
    # Create a default adapter
    default_adapter = GenOpsHaystackAdapter()
    
    # Store original Pipeline.run method
    original_run = Pipeline.run
    
    def instrumented_run(self, inputs: Dict[str, Any], **kwargs):
        """Instrumented version of Pipeline.run with governance tracking."""
        pipeline_name = getattr(self, 'name', 'unknown-pipeline')
        
        with default_adapter.track_pipeline(pipeline_name) as context:
            # Execute original pipeline
            result = original_run(self, inputs, **kwargs)
            
            # Try to extract component information from pipeline
            if hasattr(self, 'graph') and hasattr(self.graph, 'nodes'):
                for node_name in self.graph.nodes():
                    # Create a dummy component result for tracking
                    component_result = HaystackComponentResult(
                        component_name=node_name,
                        component_type="GenericComponent",
                        execution_time_seconds=0.1,  # Placeholder
                        cost=Decimal("0.001"),  # Placeholder
                        provider="haystack"
                    )
                    context.add_component_result(component_result)
            
            return result
    
    # Monkey patch the Pipeline.run method
    Pipeline.run = instrumented_run
    
    logger.info("Haystack auto-instrumentation enabled - all pipeline executions will be tracked")


# Component mixin for building GenOps-aware custom components
class GenOpsComponentMixin:
    """
    Mixin class for building GenOps-aware Haystack components.
    
    Example:
        from haystack import component
        from genops.providers.haystack_adapter import GenOpsComponentMixin
        
        @component
        class MyCustomComponent(GenOpsComponentMixin):
            def run(self, text: str):
                with self.track_operation("custom-processing") as span:
                    result = self.process_text(text)
                    span.record_cost(cost=0.001, provider="custom")
                    return {"output": result}
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.telemetry = GenOpsTelemetry(tracer_name="haystack-component")
    
    @contextmanager
    def track_operation(self, operation_name: str, **attributes):
        """
        Track an operation within a custom component.
        
        Args:
            operation_name: Name of the operation
            **attributes: Additional telemetry attributes
            
        Yields:
            OpenTelemetry span for recording metrics
        """
        with self.telemetry.trace_operation(
            f"haystack.component.{operation_name}",
            operation_type="ai.component",
            **attributes
        ) as span:
            yield span


    def with_retry_decorator(self, max_retries: Optional[int] = None, 
                           base_delay: Optional[float] = None):
        """
        Decorator factory for adding retry logic to external functions.
        
        Args:
            max_retries: Override default max retry attempts
            base_delay: Override default base delay
            
        Returns:
            Decorator function for adding retry logic
            
        Example:
            @adapter.with_retry_decorator(max_retries=5, base_delay=2.0)
            def my_ai_function():
                return call_ai_service()
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Temporarily override retry config if specified
                original_max_retries = self.retry_config['max_retries']
                original_base_delay = self.retry_config['base_delay']
                
                if max_retries is not None:
                    self.retry_config['max_retries'] = max_retries
                if base_delay is not None:
                    self.retry_config['base_delay'] = base_delay
                
                try:
                    # Use the existing retry mechanism
                    result = self._execute_with_retry(
                        func.__name__, 
                        'DecoratedFunction',
                        func,
                        *args, 
                        **kwargs
                    )
                    
                    # Return the actual result, not the HaystackComponentResult
                    if result.status == 'success':
                        return func(*args, **kwargs)  # Execute one final time to get actual result
                    else:
                        raise Exception(result.error_message)
                        
                finally:
                    # Restore original config
                    self.retry_config['max_retries'] = original_max_retries
                    self.retry_config['base_delay'] = original_base_delay
                    
            return wrapper
        return decorator
    
    def reset_error_stats(self):
        """
        Reset error tracking statistics. Useful for testing or periodic cleanup.
        """
        self.error_stats = {
            'total_errors': 0,
            'retry_attempts': 0,
            'error_types': {},
            'component_failures': {},
            'provider_failures': {}
        }
        logger.info("Error statistics reset")
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get overall health status of the adapter and its components.
        
        Returns:
            Dictionary with health indicators and status
        """
        total_operations = sum(
            pipeline.total_components for pipeline in self._pipeline_results.values()
        )
        
        if total_operations == 0:
            return {
                "status": "healthy",
                "reason": "No operations executed yet",
                "error_rate": 0.0,
                "retry_rate": 0.0,
                "recommendations": []
            }
        
        error_rate = (self.error_stats['total_errors'] / total_operations) * 100
        retry_rate = (self.error_stats['retry_attempts'] / total_operations) * 100
        
        # Determine health status based on error rates
        if error_rate > 20:
            status = "unhealthy"
            reason = f"High error rate: {error_rate:.1f}%"
        elif error_rate > 10:
            status = "degraded"
            reason = f"Elevated error rate: {error_rate:.1f}%"
        elif retry_rate > 30:
            status = "degraded"
            reason = f"High retry rate: {retry_rate:.1f}%"
        else:
            status = "healthy"
            reason = "Operating within normal parameters"
        
        return {
            "status": status,
            "reason": reason,
            "error_rate": error_rate,
            "retry_rate": retry_rate,
            "total_operations": total_operations,
            "total_errors": self.error_stats['total_errors'],
            "recommendations": self._generate_error_recommendations()
        }


# Export main classes and functions
__all__ = [
    'GenOpsHaystackAdapter',
    'HaystackComponentResult', 
    'HaystackPipelineResult',
    'HaystackSessionContext',
    'HaystackPipelineContext',
    'GenOpsComponentMixin',
    'auto_instrument'
]