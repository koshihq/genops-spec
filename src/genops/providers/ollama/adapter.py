"""Ollama provider adapter for GenOps AI governance."""

from __future__ import annotations

import logging
import time
import uuid
import json
import requests
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from contextlib import contextmanager

from genops.providers.base import BaseFrameworkProvider
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# Check for Ollama availability
try:
    # Try to import ollama client if available
    import ollama
    HAS_OLLAMA_CLIENT = True
except ImportError:
    HAS_OLLAMA_CLIENT = False
    logger.info("Ollama client not installed. Install with: pip install ollama")


@dataclass
class OllamaOperation:
    """Represents a single Ollama operation for resource tracking."""
    
    operation_id: str
    operation_type: str  # 'generate', 'chat', 'embed', 'pull_model', 'list_models'
    model: str
    start_time: float
    end_time: Optional[float] = None
    
    # Input/output data
    prompt: Optional[str] = None
    response: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    
    # Resource metrics (Ollama-specific)
    inference_time_ms: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    model_load_time_ms: Optional[float] = None
    
    # Cost attribution (infrastructure costs)
    infrastructure_cost: Optional[float] = None
    gpu_hours: Optional[float] = None
    cpu_hours: Optional[float] = None
    
    # Governance attributes
    governance_attributes: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.governance_attributes is None:
            self.governance_attributes = {}
    
    @property
    def duration_ms(self) -> float:
        """Calculate operation duration in milliseconds."""
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000


@dataclass
class LocalModelMetrics:
    """Comprehensive metrics for local Ollama model operations."""
    
    model_name: str
    total_operations: int
    total_inference_time_ms: float
    
    # Resource utilization
    avg_gpu_memory_mb: float = 0.0
    avg_cpu_usage_percent: float = 0.0
    avg_inference_latency_ms: float = 0.0
    
    # Token statistics
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    avg_tokens_per_second: float = 0.0
    
    # Infrastructure costs
    total_infrastructure_cost: float = 0.0
    cost_per_operation: float = 0.0
    gpu_hours_consumed: float = 0.0
    
    # Quality metrics
    success_rate: float = 100.0
    error_count: int = 0
    
    # Model efficiency
    tokens_per_gpu_hour: float = 0.0
    operations_per_dollar: float = 0.0


class GenOpsOllamaAdapter(BaseFrameworkProvider):
    """
    GenOps adapter for Ollama with comprehensive local model governance.
    
    Provides resource tracking, cost attribution, and performance optimization for:
    - Local model inference and resource utilization
    - Infrastructure cost attribution (GPU time, electricity, compute)
    - Model performance optimization and comparison
    - Team-based resource allocation and governance
    """

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        telemetry_enabled: bool = True,
        cost_tracking_enabled: bool = True,
        debug: bool = False,
        # Infrastructure cost rates (USD)
        gpu_hour_rate: float = 0.50,  # $0.50/hour for GPU usage
        cpu_hour_rate: float = 0.05,  # $0.05/hour for CPU usage
        electricity_rate: float = 0.12,  # $0.12/kWh
        **governance_defaults
    ):
        """
        Initialize GenOps Ollama adapter.
        
        Args:
            ollama_base_url: Base URL for Ollama server
            telemetry_enabled: Enable OpenTelemetry export
            cost_tracking_enabled: Enable infrastructure cost calculation
            debug: Enable debug logging
            gpu_hour_rate: Cost per GPU hour in USD
            cpu_hour_rate: Cost per CPU hour in USD  
            electricity_rate: Electricity cost per kWh
            **governance_defaults: Default governance attributes
        """
        super().__init__()
        
        self.ollama_base_url = ollama_base_url.rstrip('/')
        self.telemetry_enabled = telemetry_enabled
        self.cost_tracking_enabled = cost_tracking_enabled
        self.debug = debug
        self.governance_defaults = governance_defaults
        
        # Infrastructure cost rates
        self.gpu_hour_rate = gpu_hour_rate
        self.cpu_hour_rate = cpu_hour_rate
        self.electricity_rate = electricity_rate
        
        # Operation tracking
        self.operations: List[OllamaOperation] = []
        self.model_metrics: Dict[str, LocalModelMetrics] = {}
        
        # Current operation context
        self._governance_context: Dict[str, Any] = {}
        
        # Initialize Ollama client if available
        self.client = None
        if HAS_OLLAMA_CLIENT:
            try:
                self.client = ollama.Client(host=ollama_base_url)
                self._test_connection()
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama client: {e}")
        else:
            logger.info("Using HTTP client for Ollama communication")

    def _test_connection(self):
        """Test connection to Ollama server."""
        try:
            if self.client:
                # Test with client
                self.client.list()
            else:
                # Test with HTTP request
                response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
                response.raise_for_status()
            
            logger.info(f"Successfully connected to Ollama server at {self.ollama_base_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Ollama server: {e}")
            raise ConnectionError(f"Cannot connect to Ollama server at {self.ollama_base_url}: {e}")

    def get_current_governance_context(self) -> Dict[str, Any]:
        """Get current governance context for operations."""
        return {**self.governance_defaults, **self._governance_context}

    @contextmanager
    def governance_context(self, **attributes):
        """Context manager to set governance attributes for operations."""
        old_context = self._governance_context.copy()
        self._governance_context.update(attributes)
        try:
            yield
        finally:
            self._governance_context = old_context

    def list_models(self, **governance_attrs) -> List[Dict[str, Any]]:
        """
        List available Ollama models with governance tracking.
        
        Args:
            **governance_attrs: Governance attributes for operation tracking
            
        Returns:
            List of available models with metadata
        """
        with self.governance_context(**governance_attrs):
            operation = OllamaOperation(
                operation_id=str(uuid.uuid4()),
                operation_type="list_models",
                model="system",
                start_time=time.time(),
                governance_attributes=self.get_current_governance_context()
            )
            
            with tracer.start_as_current_span("ollama.list_models") as span:
                span.set_attributes({
                    "genops.operation_id": operation.operation_id,
                    "genops.operation_type": "list_models",
                    "genops.framework": "ollama",
                    "genops.server_url": self.ollama_base_url,
                    **operation.governance_attributes
                })
                
                try:
                    if self.client:
                        # Use ollama client
                        models_response = self.client.list()
                        models = models_response.get('models', [])
                    else:
                        # Use HTTP API
                        response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=10)
                        response.raise_for_status()
                        models = response.json().get('models', [])
                    
                    operation.end_time = time.time()
                    span.set_attribute("genops.models_count", len(models))
                    span.set_attribute("genops.success", True)
                    
                    # Calculate infrastructure cost
                    if self.cost_tracking_enabled:
                        operation.infrastructure_cost = self._calculate_operation_cost(operation)
                    
                    self.operations.append(operation)
                    
                    logger.info(f"Listed {len(models)} available Ollama models")
                    return models
                    
                except Exception as e:
                    operation.end_time = time.time()
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    logger.error(f"Failed to list Ollama models: {e}")
                    raise

    def generate(
        self,
        model: str,
        prompt: str,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text with Ollama model and comprehensive tracking.
        
        Args:
            model: Model name to use for generation
            prompt: Input prompt
            stream: Whether to stream the response
            **kwargs: Additional parameters including governance attributes
            
        Returns:
            Generation response with tracking metadata
        """
        # Extract governance attributes
        governance_attrs = {k: v for k, v in kwargs.items() if k.startswith(('team', 'project', 'customer', 'environment'))}
        generation_kwargs = {k: v for k, v in kwargs.items() if not k.startswith(('team', 'project', 'customer', 'environment'))}
        
        with self.governance_context(**governance_attrs):
            operation = OllamaOperation(
                operation_id=str(uuid.uuid4()),
                operation_type="generate",
                model=model,
                prompt=prompt,
                start_time=time.time(),
                governance_attributes=self.get_current_governance_context()
            )
            
            with tracer.start_as_current_span("ollama.generate") as span:
                span.set_attributes({
                    "genops.operation_id": operation.operation_id,
                    "genops.operation_type": "generate",
                    "genops.framework": "ollama",
                    "genops.model": model,
                    "genops.prompt_length": len(prompt),
                    "genops.stream": stream,
                    **operation.governance_attributes
                })
                
                try:
                    # Record inference start time for latency measurement
                    inference_start = time.time()
                    
                    if self.client:
                        # Use ollama client
                        response = self.client.generate(
                            model=model,
                            prompt=prompt,
                            stream=stream,
                            **generation_kwargs
                        )
                    else:
                        # Use HTTP API
                        payload = {
                            "model": model,
                            "prompt": prompt,
                            "stream": stream,
                            **generation_kwargs
                        }
                        
                        http_response = requests.post(
                            f"{self.ollama_base_url}/api/generate",
                            json=payload,
                            timeout=300  # 5 minute timeout for generation
                        )
                        http_response.raise_for_status()
                        response = http_response.json()
                    
                    inference_end = time.time()
                    operation.inference_time_ms = (inference_end - inference_start) * 1000
                    operation.end_time = time.time()
                    
                    # Extract response details
                    if isinstance(response, dict):
                        operation.response = response.get('response', '')
                        
                        # Extract token counts if available
                        if 'eval_count' in response:
                            operation.output_tokens = response['eval_count']
                        if 'prompt_eval_count' in response:
                            operation.input_tokens = response['prompt_eval_count']
                    
                    # Calculate infrastructure cost
                    if self.cost_tracking_enabled:
                        operation.infrastructure_cost = self._calculate_operation_cost(operation)
                        operation.gpu_hours = operation.duration_ms / (1000 * 3600)  # Convert to hours
                        operation.cpu_hours = operation.duration_ms / (1000 * 3600)
                    
                    # Update telemetry
                    span.set_attributes({
                        "genops.success": True,
                        "genops.inference_time_ms": operation.inference_time_ms,
                        "genops.input_tokens": operation.input_tokens or 0,
                        "genops.output_tokens": operation.output_tokens or 0,
                        "genops.infrastructure_cost": operation.infrastructure_cost or 0.0
                    })
                    
                    if operation.response:
                        span.set_attribute("genops.response_length", len(operation.response))
                    
                    # Store operation and update metrics
                    self.operations.append(operation)
                    self._update_model_metrics(model, operation)
                    
                    logger.info(f"Generated text with model {model}: {operation.inference_time_ms:.0f}ms")
                    return response
                    
                except Exception as e:
                    operation.end_time = time.time()
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    logger.error(f"Failed to generate with Ollama model {model}: {e}")
                    
                    # Still record the failed operation for metrics
                    self.operations.append(operation)
                    raise

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Chat with Ollama model and comprehensive tracking.
        
        Args:
            model: Model name to use for chat
            messages: List of messages in OpenAI chat format
            stream: Whether to stream the response
            **kwargs: Additional parameters including governance attributes
            
        Returns:
            Chat response with tracking metadata
        """
        # Extract governance attributes
        governance_attrs = {k: v for k, v in kwargs.items() if k.startswith(('team', 'project', 'customer', 'environment'))}
        chat_kwargs = {k: v for k, v in kwargs.items() if not k.startswith(('team', 'project', 'customer', 'environment'))}
        
        with self.governance_context(**governance_attrs):
            # Create prompt from messages for tracking
            prompt_text = json.dumps(messages) if messages else ""
            
            operation = OllamaOperation(
                operation_id=str(uuid.uuid4()),
                operation_type="chat",
                model=model,
                prompt=prompt_text,
                start_time=time.time(),
                governance_attributes=self.get_current_governance_context()
            )
            
            with tracer.start_as_current_span("ollama.chat") as span:
                span.set_attributes({
                    "genops.operation_id": operation.operation_id,
                    "genops.operation_type": "chat",
                    "genops.framework": "ollama",
                    "genops.model": model,
                    "genops.messages_count": len(messages),
                    "genops.stream": stream,
                    **operation.governance_attributes
                })
                
                try:
                    # Record inference start time
                    inference_start = time.time()
                    
                    if self.client:
                        # Use ollama client
                        response = self.client.chat(
                            model=model,
                            messages=messages,
                            stream=stream,
                            **chat_kwargs
                        )
                    else:
                        # Use HTTP API
                        payload = {
                            "model": model,
                            "messages": messages,
                            "stream": stream,
                            **chat_kwargs
                        }
                        
                        http_response = requests.post(
                            f"{self.ollama_base_url}/api/chat",
                            json=payload,
                            timeout=300
                        )
                        http_response.raise_for_status()
                        response = http_response.json()
                    
                    inference_end = time.time()
                    operation.inference_time_ms = (inference_end - inference_start) * 1000
                    operation.end_time = time.time()
                    
                    # Extract response details
                    if isinstance(response, dict):
                        if 'message' in response:
                            operation.response = response['message'].get('content', '')
                        
                        # Extract token counts if available
                        if 'eval_count' in response:
                            operation.output_tokens = response['eval_count']
                        if 'prompt_eval_count' in response:
                            operation.input_tokens = response['prompt_eval_count']
                    
                    # Calculate infrastructure cost
                    if self.cost_tracking_enabled:
                        operation.infrastructure_cost = self._calculate_operation_cost(operation)
                        operation.gpu_hours = operation.duration_ms / (1000 * 3600)
                        operation.cpu_hours = operation.duration_ms / (1000 * 3600)
                    
                    # Update telemetry
                    span.set_attributes({
                        "genops.success": True,
                        "genops.inference_time_ms": operation.inference_time_ms,
                        "genops.input_tokens": operation.input_tokens or 0,
                        "genops.output_tokens": operation.output_tokens or 0,
                        "genops.infrastructure_cost": operation.infrastructure_cost or 0.0
                    })
                    
                    if operation.response:
                        span.set_attribute("genops.response_length", len(operation.response))
                    
                    # Store operation and update metrics
                    self.operations.append(operation)
                    self._update_model_metrics(model, operation)
                    
                    logger.info(f"Chat with model {model}: {operation.inference_time_ms:.0f}ms")
                    return response
                    
                except Exception as e:
                    operation.end_time = time.time()
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    logger.error(f"Failed to chat with Ollama model {model}: {e}")
                    
                    # Still record the failed operation
                    self.operations.append(operation)
                    raise

    def _calculate_operation_cost(self, operation: OllamaOperation) -> float:
        """
        Calculate infrastructure cost for an operation.
        
        For Ollama, costs are based on:
        - GPU usage time
        - CPU usage time  
        - Electricity consumption
        - Infrastructure amortization
        """
        if not operation.end_time:
            return 0.0
        
        duration_hours = (operation.end_time - operation.start_time) / 3600
        
        # Base infrastructure cost (GPU + CPU time)
        base_cost = (self.gpu_hour_rate + self.cpu_hour_rate) * duration_hours
        
        # Add electricity cost estimate (rough approximation)
        # Assume 300W GPU + 100W CPU = 0.4kW
        electricity_cost = 0.4 * duration_hours * self.electricity_rate
        
        total_cost = base_cost + electricity_cost
        
        # Adjust based on model complexity (rough heuristic)
        if operation.model:
            if 'large' in operation.model.lower() or '70b' in operation.model.lower():
                total_cost *= 2.0  # Large models cost more
            elif 'small' in operation.model.lower() or '7b' in operation.model.lower():
                total_cost *= 0.5  # Small models cost less
        
        return round(total_cost, 6)

    def _update_model_metrics(self, model: str, operation: OllamaOperation):
        """Update aggregated metrics for a model."""
        if model not in self.model_metrics:
            self.model_metrics[model] = LocalModelMetrics(
                model_name=model,
                total_operations=0,
                total_inference_time_ms=0.0
            )
        
        metrics = self.model_metrics[model]
        metrics.total_operations += 1
        
        if operation.inference_time_ms:
            metrics.total_inference_time_ms += operation.inference_time_ms
            metrics.avg_inference_latency_ms = metrics.total_inference_time_ms / metrics.total_operations
        
        if operation.input_tokens:
            metrics.total_input_tokens += operation.input_tokens
        if operation.output_tokens:
            metrics.total_output_tokens += operation.output_tokens
        
        if operation.infrastructure_cost:
            metrics.total_infrastructure_cost += operation.infrastructure_cost
            metrics.cost_per_operation = metrics.total_infrastructure_cost / metrics.total_operations
        
        if operation.gpu_hours:
            metrics.gpu_hours_consumed += operation.gpu_hours
            
            # Calculate efficiency metrics
            if metrics.gpu_hours_consumed > 0:
                total_tokens = metrics.total_input_tokens + metrics.total_output_tokens
                metrics.tokens_per_gpu_hour = total_tokens / metrics.gpu_hours_consumed
                
                if metrics.total_infrastructure_cost > 0:
                    metrics.operations_per_dollar = metrics.total_operations / metrics.total_infrastructure_cost

    def get_model_metrics(self, model: Optional[str] = None) -> Union[LocalModelMetrics, Dict[str, LocalModelMetrics]]:
        """
        Get metrics for a specific model or all models.
        
        Args:
            model: Model name to get metrics for, or None for all models
            
        Returns:
            Model metrics for specified model or all models
        """
        if model:
            return self.model_metrics.get(model)
        return self.model_metrics

    def get_operation_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all tracked operations."""
        if not self.operations:
            return {
                "total_operations": 0,
                "total_infrastructure_cost": 0.0,
                "models_used": [],
                "avg_inference_time_ms": 0.0
            }
        
        total_cost = sum(op.infrastructure_cost or 0.0 for op in self.operations)
        total_inference_time = sum(op.inference_time_ms or 0.0 for op in self.operations)
        models_used = list(set(op.model for op in self.operations))
        
        successful_ops = [op for op in self.operations if op.end_time and op.response]
        success_rate = len(successful_ops) / len(self.operations) * 100
        
        return {
            "total_operations": len(self.operations),
            "total_infrastructure_cost": total_cost,
            "avg_cost_per_operation": total_cost / len(self.operations) if self.operations else 0.0,
            "models_used": models_used,
            "unique_models_count": len(models_used),
            "avg_inference_time_ms": total_inference_time / len(self.operations) if self.operations else 0.0,
            "success_rate_percent": success_rate,
            "total_gpu_hours": sum(op.gpu_hours or 0.0 for op in self.operations),
            "total_tokens": sum((op.input_tokens or 0) + (op.output_tokens or 0) for op in self.operations),
            "operations": [asdict(op) for op in self.operations]
        }

def instrument_ollama(
    ollama_base_url: str = "http://localhost:11434",
    telemetry_enabled: bool = True,
    cost_tracking_enabled: bool = True,
    **governance_defaults
) -> GenOpsOllamaAdapter:
    """
    Create and configure a GenOps Ollama adapter.
    
    Args:
        ollama_base_url: Base URL for Ollama server
        telemetry_enabled: Enable OpenTelemetry export
        cost_tracking_enabled: Enable infrastructure cost tracking
        **governance_defaults: Default governance attributes
        
    Returns:
        Configured GenOpsOllamaAdapter instance
        
    Example:
        adapter = instrument_ollama(team="ai-research", project="local-models")
        response = adapter.generate("llama2", "What is machine learning?")
    """
    return GenOpsOllamaAdapter(
        ollama_base_url=ollama_base_url,
        telemetry_enabled=telemetry_enabled,
        cost_tracking_enabled=cost_tracking_enabled,
        **governance_defaults
    )


# Export main classes and functions
__all__ = [
    "GenOpsOllamaAdapter",
    "OllamaOperation", 
    "LocalModelMetrics",
    "instrument_ollama"
]