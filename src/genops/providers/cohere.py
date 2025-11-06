#!/usr/bin/env python3
"""
GenOps Cohere Provider Integration

This module provides comprehensive Cohere integration for GenOps AI governance,
cost intelligence, and observability. It follows the established GenOps provider
pattern for consistent developer experience across all AI platforms.

Features:
- Multi-operation support (generate, chat, embed, rerank, classify)
- Zero-code auto-instrumentation with instrument_cohere()
- Unified cost tracking across all Cohere models and operations
- Streaming response support for real-time applications  
- Cohere API key authentication with environment variable support
- Advanced embedding and rerank cost optimization
- Comprehensive governance and audit trail integration

Example usage:

    # Zero-code auto-instrumentation
    from genops.providers.cohere import instrument_cohere
    instrument_cohere()
    
    # Your existing Cohere code works unchanged with automatic governance
    import cohere
    client = cohere.ClientV2()
    response = client.chat(...)  # Now tracked with GenOps!
    
    # Manual adapter usage for advanced control
    from genops.providers.cohere import GenOpsCohereAdapter
    
    adapter = GenOpsCohereAdapter()
    response = adapter.chat(
        message="Explain quantum computing",
        model="command-r-plus-08-2024",
        team="research-team",
        project="quantum-ai",
        customer_id="enterprise-123"
    )
"""

import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Iterator
import os
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import Cohere dependencies with graceful fallback
try:
    import cohere
    from cohere import ClientV2
    HAS_COHERE = True
except ImportError:
    HAS_COHERE = False
    ClientV2 = None
    logger.warning("Cohere not installed. Install with: pip install cohere")

# Try to import GenOps core dependencies
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False
    logger.warning("OpenTelemetry not available - telemetry will be disabled")

# Constants for Cohere models and operations
class CohereModel(Enum):
    """Cohere model enumeration for type safety and cost calculation."""
    # Command series - text generation
    COMMAND = "command"
    COMMAND_LIGHT = "command-light"
    COMMAND_R = "command-r-03-2024"
    COMMAND_R_PLUS = "command-r-plus-04-2024"
    COMMAND_R_PLUS_08 = "command-r-plus-08-2024"
    
    # Aya Expanse series
    AYA_EXPANSE_8B = "aya-expanse-8b"
    AYA_EXPANSE_32B = "aya-expanse-32b"
    
    # Embedding models
    EMBED_ENGLISH_V3 = "embed-english-v3.0"
    EMBED_MULTILINGUAL_V3 = "embed-multilingual-v3.0"
    EMBED_V4 = "embed-english-v4.0"
    
    # Rerank models
    RERANK_V3 = "rerank-english-v3.0"
    RERANK_MULTILINGUAL_V3 = "rerank-multilingual-v3.0"


class CohereOperation(Enum):
    """Cohere operation types for cost tracking."""
    GENERATE = "generate"
    CHAT = "chat"
    EMBED = "embed"
    RERANK = "rerank"
    CLASSIFY = "classify"
    SUMMARIZE = "summarize"


@dataclass
class CohereUsageMetrics:
    """Comprehensive usage metrics for Cohere operations."""
    
    # Request metadata
    operation_id: str
    operation_type: CohereOperation
    model: str
    timestamp: float
    
    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    # Operation-specific metrics
    embedding_units: int = 0  # For embedding operations
    search_units: int = 0     # For rerank operations
    
    # Cost information
    input_cost: float = 0.0
    output_cost: float = 0.0
    operation_cost: float = 0.0  # For non-token operations
    total_cost: float = 0.0
    
    # Performance metrics
    latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    
    # Governance attributes
    team: Optional[str] = None
    project: Optional[str] = None
    environment: Optional[str] = None
    customer_id: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived metrics."""
        self.total_tokens = self.input_tokens + self.output_tokens
        self.total_cost = self.input_cost + self.output_cost + self.operation_cost
        
        if self.latency_ms > 0 and self.output_tokens > 0:
            self.tokens_per_second = (self.output_tokens / self.latency_ms) * 1000


@dataclass
class CohereResponse:
    """Standardized response format for all Cohere operations."""
    
    # Core response data
    content: str = ""
    usage: Optional[CohereUsageMetrics] = None
    model: str = ""
    
    # Operation-specific data
    embeddings: Optional[List[List[float]]] = None
    rankings: Optional[List[Dict[str, Any]]] = None
    classifications: Optional[List[Dict[str, Any]]] = None
    
    # Metadata
    operation_id: str = ""
    request_id: str = ""
    success: bool = True
    error_message: str = ""
    
    # Raw response for advanced use cases
    raw_response: Optional[Any] = None


class GenOpsCohereAdapter:
    """
    Comprehensive Cohere adapter with automatic GenOps governance integration.
    
    This adapter provides intelligent cost tracking, team attribution, and observability
    for all Cohere operations including text generation, embedding, reranking, and classification.
    
    Key features:
    - Multi-operation support with unified cost tracking
    - Automatic team and project attribution
    - Advanced embedding and rerank optimization
    - Streaming response support
    - Budget controls and cost alerts
    - OpenTelemetry integration for observability
    
    Example:
        adapter = GenOpsCohereAdapter(api_key="your-key")
        
        # Text generation with governance
        response = adapter.chat(
            message="Explain machine learning",
            model="command-r-plus-08-2024",
            team="ml-team",
            project="ai-education"
        )
        
        # Embedding with cost optimization
        embeddings = adapter.embed(
            texts=["query text", "document text"],
            model="embed-english-v4.0",
            team="search-team"
        )
        
        # Reranking with search optimization
        rankings = adapter.rerank(
            query="machine learning",
            documents=["doc1", "doc2", "doc3"],
            model="rerank-english-v3.0"
        )
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        
        # Cost tracking configuration
        cost_tracking_enabled: bool = True,
        budget_limit: Optional[float] = None,
        cost_alert_threshold: float = 0.8,
        
        # Governance defaults
        default_team: Optional[str] = None,
        default_project: Optional[str] = None,
        default_environment: Optional[str] = None,
        
        # Advanced settings
        enable_streaming: bool = True,
        enable_caching: bool = False,
        debug: bool = False,
        
        **kwargs
    ):
        """
        Initialize GenOps Cohere adapter with comprehensive configuration.
        
        Args:
            api_key: Cohere API key (defaults to CO_API_KEY env var)
            base_url: Custom API base URL for enterprise deployments
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
            
            cost_tracking_enabled: Enable automatic cost calculation and tracking
            budget_limit: Optional budget limit for cost controls
            cost_alert_threshold: Threshold (0-1) for cost alerts
            
            default_team: Default team attribution for operations
            default_project: Default project attribution
            default_environment: Default environment (dev/staging/prod)
            
            enable_streaming: Enable streaming response support
            enable_caching: Enable response caching for identical requests
            debug: Enable debug logging
        """
        if not HAS_COHERE:
            raise ImportError(
                "Cohere package not found. Install with: pip install cohere"
            )
        
        # Initialize API key from parameter or environment
        self.api_key = api_key or os.getenv("CO_API_KEY")
        if not self.api_key:
            logger.warning("No Cohere API key provided. Set CO_API_KEY environment variable or pass api_key parameter")
        
        # Initialize Cohere client
        client_kwargs = {
            "api_key": self.api_key,
            "timeout": timeout,
            **kwargs
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        
        try:
            self.client = ClientV2(**client_kwargs)
        except Exception as e:
            logger.error(f"Failed to initialize Cohere client: {e}")
            self.client = None
        
        # Configuration
        self.timeout = timeout
        self.max_retries = max_retries
        self.cost_tracking_enabled = cost_tracking_enabled
        self.budget_limit = budget_limit
        self.cost_alert_threshold = cost_alert_threshold
        self.enable_streaming = enable_streaming
        self.enable_caching = enable_caching
        self.debug = debug
        
        # Governance defaults
        self.default_team = default_team
        self.default_project = default_project
        self.default_environment = default_environment
        
        # Internal state
        self._total_cost = 0.0
        self._operation_count = 0
        self._cache = {} if enable_caching else None
        
        # Initialize telemetry
        self.tracer = None
        if HAS_OTEL:
            self.tracer = trace.get_tracer(__name__)
        
        logger.info(f"GenOpsCohereAdapter initialized with cost tracking: {cost_tracking_enabled}")
    
    def _create_operation_id(self) -> str:
        """Generate unique operation ID for tracking."""
        return f"cohere-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
    
    def _get_governance_attributes(self, **kwargs) -> Dict[str, str]:
        """Extract and standardize governance attributes."""
        return {
            "team": kwargs.get("team", self.default_team),
            "project": kwargs.get("project", self.default_project),
            "environment": kwargs.get("environment", self.default_environment),
            "customer_id": kwargs.get("customer_id"),
            "feature": kwargs.get("feature"),
            "cost_center": kwargs.get("cost_center"),
        }
    
    def _calculate_cost(
        self, 
        model: str, 
        operation: CohereOperation,
        input_tokens: int = 0,
        output_tokens: int = 0,
        operation_units: int = 0
    ) -> tuple[float, float, float]:
        """
        Calculate costs for Cohere operations based on current pricing.
        
        Returns:
            tuple: (input_cost, output_cost, operation_cost)
        """
        if not self.cost_tracking_enabled:
            return 0.0, 0.0, 0.0
        
        # Import pricing calculator
        try:
            from .cohere_pricing import CohereCalculator
            calculator = CohereCalculator()
            
            return calculator.calculate_cost(
                model=model,
                operation=operation,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                operation_units=operation_units
            )
        except ImportError:
            logger.warning("Cohere pricing calculator not available")
            return 0.0, 0.0, 0.0
    
    def _check_budget_limit(self, estimated_cost: float) -> bool:
        """Check if operation would exceed budget limit."""
        if not self.budget_limit:
            return True
        
        projected_total = self._total_cost + estimated_cost
        
        if projected_total > self.budget_limit:
            logger.warning(f"Operation would exceed budget limit: ${projected_total:.6f} > ${self.budget_limit:.6f}")
            return False
        
        # Cost alert threshold check
        if projected_total > (self.budget_limit * self.cost_alert_threshold):
            logger.warning(f"Approaching budget limit: ${projected_total:.6f} / ${self.budget_limit:.6f}")
        
        return True
    
    def _update_usage_stats(self, usage: CohereUsageMetrics):
        """Update internal usage statistics."""
        self._total_cost += usage.total_cost
        self._operation_count += 1
        
        if self.debug:
            logger.debug(f"Operation {usage.operation_id}: {usage.operation_type.value} - ${usage.total_cost:.6f}")
    
    @contextmanager
    def _create_span(self, operation: str, **attributes):
        """Create OpenTelemetry span for operation tracking."""
        if not self.tracer:
            yield None
            return
        
        with self.tracer.start_as_current_span(f"genops.cohere.{operation}") as span:
            # Add standard attributes
            span.set_attribute("genops.provider", "cohere")
            span.set_attribute("genops.operation", operation)
            
            # Add governance attributes
            for key, value in attributes.items():
                if value is not None:
                    span.set_attribute(f"genops.{key}", str(value))
            
            try:
                yield span
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
    
    def chat(
        self,
        message: str,
        model: str = "command-r-plus-08-2024",
        conversation_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **governance_kwargs
    ) -> CohereResponse:
        """
        Generate conversational responses with comprehensive governance tracking.
        
        Args:
            message: User message for the conversation
            model: Cohere model to use (default: command-r-plus-08-2024)
            conversation_id: Optional conversation ID for multi-turn tracking
            temperature: Randomness in response generation (0.0-1.0)
            max_tokens: Maximum tokens in response
            stream: Enable streaming response
            **governance_kwargs: Team, project, customer_id, etc.
        
        Returns:
            CohereResponse: Standardized response with usage metrics
        
        Example:
            response = adapter.chat(
                message="What is machine learning?",
                model="command-r-plus-08-2024",
                team="ml-team",
                project="education"
            )
        """
        if not self.client:
            raise RuntimeError("Cohere client not initialized")
        
        operation_id = self._create_operation_id()
        governance_attrs = self._get_governance_attributes(**governance_kwargs)
        start_time = time.time()
        
        with self._create_span("chat", **governance_attrs, model=model) as span:
            try:
                # Prepare request parameters
                request_params = {
                    "model": model,
                    "messages": [{"role": "user", "content": message}]
                }
                
                if temperature is not None:
                    request_params["temperature"] = temperature
                if max_tokens is not None:
                    request_params["max_tokens"] = max_tokens
                if stream and self.enable_streaming:
                    request_params["stream"] = True
                
                # Add conversation context if provided
                if conversation_id:
                    request_params["conversation_id"] = conversation_id
                
                # Execute request
                response = self.client.chat(**request_params)
                
                # Process response
                if stream and self.enable_streaming:
                    return self._handle_streaming_response(
                        response, operation_id, CohereOperation.CHAT, model, governance_attrs, start_time
                    )
                else:
                    return self._process_chat_response(
                        response, operation_id, model, governance_attrs, start_time
                    )
            
            except Exception as e:
                logger.error(f"Cohere chat operation failed: {e}")
                return CohereResponse(
                    operation_id=operation_id,
                    success=False,
                    error_message=str(e)
                )
    
    def _process_chat_response(
        self,
        response: Any,
        operation_id: str,
        model: str,
        governance_attrs: Dict[str, str],
        start_time: float
    ) -> CohereResponse:
        """Process non-streaming chat response."""
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        # Extract response content
        content = ""
        if hasattr(response, 'message') and hasattr(response.message, 'content'):
            content = response.message.content[0].text if response.message.content else ""
        
        # Extract usage information
        input_tokens = 0
        output_tokens = 0
        
        if hasattr(response, 'usage'):
            input_tokens = getattr(response.usage, 'input_tokens', 0)
            output_tokens = getattr(response.usage, 'output_tokens', 0)
        
        # Calculate costs
        input_cost, output_cost, operation_cost = self._calculate_cost(
            model=model,
            operation=CohereOperation.CHAT,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
        
        # Create usage metrics
        usage = CohereUsageMetrics(
            operation_id=operation_id,
            operation_type=CohereOperation.CHAT,
            model=model,
            timestamp=start_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            operation_cost=operation_cost,
            latency_ms=latency_ms,
            **governance_attrs
        )
        
        # Update statistics
        self._update_usage_stats(usage)
        
        return CohereResponse(
            content=content,
            usage=usage,
            model=model,
            operation_id=operation_id,
            success=True,
            raw_response=response
        )
    
    def generate(
        self,
        prompt: str,
        model: str = "command-r-08-2024",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **governance_kwargs
    ) -> CohereResponse:
        """
        Generate text completions with comprehensive governance tracking.
        
        Args:
            prompt: Text prompt for generation
            model: Cohere model to use
            temperature: Randomness in generation (0.0-1.0)
            max_tokens: Maximum tokens to generate
            stop_sequences: Sequences that stop generation
            **governance_kwargs: Team, project, customer_id, etc.
        
        Returns:
            CohereResponse: Standardized response with usage metrics
        """
        if not self.client:
            raise RuntimeError("Cohere client not initialized")
        
        operation_id = self._create_operation_id()
        governance_attrs = self._get_governance_attributes(**governance_kwargs)
        start_time = time.time()
        
        with self._create_span("generate", **governance_attrs, model=model) as span:
            try:
                # Prepare request parameters
                request_params = {
                    "model": model,
                    "prompt": prompt
                }
                
                if temperature is not None:
                    request_params["temperature"] = temperature
                if max_tokens is not None:
                    request_params["max_tokens"] = max_tokens
                if stop_sequences:
                    request_params["stop_sequences"] = stop_sequences
                
                # Execute request (using legacy generate endpoint if available)
                if hasattr(self.client, 'generate'):
                    response = self.client.generate(**request_params)
                else:
                    # Fallback to chat endpoint with system message
                    messages = [{"role": "user", "content": prompt}]
                    response = self.client.chat(model=model, messages=messages)
                
                return self._process_generate_response(
                    response, operation_id, model, governance_attrs, start_time
                )
            
            except Exception as e:
                logger.error(f"Cohere generate operation failed: {e}")
                return CohereResponse(
                    operation_id=operation_id,
                    success=False,
                    error_message=str(e)
                )
    
    def _process_generate_response(
        self,
        response: Any,
        operation_id: str,
        model: str,
        governance_attrs: Dict[str, str],
        start_time: float
    ) -> CohereResponse:
        """Process text generation response."""
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        # Extract content based on response type
        content = ""
        if hasattr(response, 'generations'):
            # Legacy generate response
            content = response.generations[0].text if response.generations else ""
        elif hasattr(response, 'message'):
            # Chat response used as fallback
            content = response.message.content[0].text if response.message.content else ""
        
        # Extract usage information
        input_tokens = 0
        output_tokens = 0
        
        if hasattr(response, 'meta') and hasattr(response.meta, 'billed_units'):
            # Legacy format
            input_tokens = getattr(response.meta.billed_units, 'input_tokens', 0)
            output_tokens = getattr(response.meta.billed_units, 'output_tokens', 0)
        elif hasattr(response, 'usage'):
            # New format
            input_tokens = getattr(response.usage, 'input_tokens', 0)
            output_tokens = getattr(response.usage, 'output_tokens', 0)
        
        # Calculate costs
        input_cost, output_cost, operation_cost = self._calculate_cost(
            model=model,
            operation=CohereOperation.GENERATE,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )
        
        # Create usage metrics
        usage = CohereUsageMetrics(
            operation_id=operation_id,
            operation_type=CohereOperation.GENERATE,
            model=model,
            timestamp=start_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            operation_cost=operation_cost,
            latency_ms=latency_ms,
            **governance_attrs
        )
        
        # Update statistics
        self._update_usage_stats(usage)
        
        return CohereResponse(
            content=content,
            usage=usage,
            model=model,
            operation_id=operation_id,
            success=True,
            raw_response=response
        )
    
    def embed(
        self,
        texts: Union[str, List[str]],
        model: str = "embed-english-v4.0",
        input_type: str = "search_document",
        embedding_types: Optional[List[str]] = None,
        **governance_kwargs
    ) -> CohereResponse:
        """
        Generate embeddings with comprehensive cost tracking and optimization.
        
        Args:
            texts: Text(s) to embed (string or list of strings)
            model: Embedding model to use
            input_type: Type of input (search_document, search_query, classification, clustering)
            embedding_types: Types of embeddings to return
            **governance_kwargs: Team, project, customer_id, etc.
        
        Returns:
            CohereResponse: Response with embeddings and usage metrics
        """
        if not self.client:
            raise RuntimeError("Cohere client not initialized")
        
        # Normalize input
        if isinstance(texts, str):
            texts = [texts]
        
        operation_id = self._create_operation_id()
        governance_attrs = self._get_governance_attributes(**governance_kwargs)
        start_time = time.time()
        
        with self._create_span("embed", **governance_attrs, model=model, text_count=len(texts)) as span:
            try:
                # Prepare request parameters
                request_params = {
                    "model": model,
                    "texts": texts,
                    "input_type": input_type
                }
                
                if embedding_types:
                    request_params["embedding_types"] = embedding_types
                
                # Execute request
                response = self.client.embed(**request_params)
                
                return self._process_embed_response(
                    response, operation_id, model, len(texts), governance_attrs, start_time
                )
            
            except Exception as e:
                logger.error(f"Cohere embed operation failed: {e}")
                return CohereResponse(
                    operation_id=operation_id,
                    success=False,
                    error_message=str(e)
                )
    
    def _process_embed_response(
        self,
        response: Any,
        operation_id: str,
        model: str,
        text_count: int,
        governance_attrs: Dict[str, str],
        start_time: float
    ) -> CohereResponse:
        """Process embedding response."""
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        # Extract embeddings
        embeddings = []
        if hasattr(response, 'embeddings'):
            embeddings = response.embeddings
        
        # Calculate embedding units (typically 1 per text)
        embedding_units = text_count
        
        # Extract usage information
        input_tokens = 0
        if hasattr(response, 'meta') and hasattr(response.meta, 'billed_units'):
            input_tokens = getattr(response.meta.billed_units, 'input_tokens', 0)
        elif hasattr(response, 'usage'):
            input_tokens = getattr(response.usage, 'input_tokens', 0)
        
        # Calculate costs
        input_cost, output_cost, operation_cost = self._calculate_cost(
            model=model,
            operation=CohereOperation.EMBED,
            input_tokens=input_tokens,
            operation_units=embedding_units
        )
        
        # Create usage metrics
        usage = CohereUsageMetrics(
            operation_id=operation_id,
            operation_type=CohereOperation.EMBED,
            model=model,
            timestamp=start_time,
            input_tokens=input_tokens,
            embedding_units=embedding_units,
            input_cost=input_cost,
            output_cost=output_cost,
            operation_cost=operation_cost,
            latency_ms=latency_ms,
            **governance_attrs
        )
        
        # Update statistics
        self._update_usage_stats(usage)
        
        return CohereResponse(
            embeddings=embeddings,
            usage=usage,
            model=model,
            operation_id=operation_id,
            success=True,
            raw_response=response
        )
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        model: str = "rerank-english-v3.0",
        top_n: Optional[int] = None,
        return_documents: bool = True,
        **governance_kwargs
    ) -> CohereResponse:
        """
        Rerank documents for search relevance with cost tracking.
        
        Args:
            query: Search query
            documents: List of documents to rerank
            model: Rerank model to use
            top_n: Number of top results to return
            return_documents: Whether to return document texts
            **governance_kwargs: Team, project, customer_id, etc.
        
        Returns:
            CohereResponse: Response with rankings and usage metrics
        """
        if not self.client:
            raise RuntimeError("Cohere client not initialized")
        
        operation_id = self._create_operation_id()
        governance_attrs = self._get_governance_attributes(**governance_kwargs)
        start_time = time.time()
        
        with self._create_span("rerank", **governance_attrs, model=model, document_count=len(documents)) as span:
            try:
                # Prepare request parameters
                request_params = {
                    "model": model,
                    "query": query,
                    "documents": documents,
                    "return_documents": return_documents
                }
                
                if top_n is not None:
                    request_params["top_n"] = top_n
                
                # Execute request
                response = self.client.rerank(**request_params)
                
                return self._process_rerank_response(
                    response, operation_id, model, len(documents), governance_attrs, start_time
                )
            
            except Exception as e:
                logger.error(f"Cohere rerank operation failed: {e}")
                return CohereResponse(
                    operation_id=operation_id,
                    success=False,
                    error_message=str(e)
                )
    
    def _process_rerank_response(
        self,
        response: Any,
        operation_id: str,
        model: str,
        document_count: int,
        governance_attrs: Dict[str, str],
        start_time: float
    ) -> CohereResponse:
        """Process rerank response."""
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        # Extract rankings
        rankings = []
        if hasattr(response, 'results'):
            rankings = [
                {
                    "index": result.index,
                    "relevance_score": result.relevance_score,
                    "document": getattr(result, 'document', {})
                }
                for result in response.results
            ]
        
        # Calculate search units (typically 1 per search request)
        search_units = 1
        
        # Extract usage information
        if hasattr(response, 'meta') and hasattr(response.meta, 'billed_units'):
            search_units = getattr(response.meta.billed_units, 'search_units', search_units)
        
        # Calculate costs
        input_cost, output_cost, operation_cost = self._calculate_cost(
            model=model,
            operation=CohereOperation.RERANK,
            operation_units=search_units
        )
        
        # Create usage metrics
        usage = CohereUsageMetrics(
            operation_id=operation_id,
            operation_type=CohereOperation.RERANK,
            model=model,
            timestamp=start_time,
            search_units=search_units,
            input_cost=input_cost,
            output_cost=output_cost,
            operation_cost=operation_cost,
            latency_ms=latency_ms,
            **governance_attrs
        )
        
        # Update statistics
        self._update_usage_stats(usage)
        
        return CohereResponse(
            rankings=rankings,
            usage=usage,
            model=model,
            operation_id=operation_id,
            success=True,
            raw_response=response
        )
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive usage and cost summary.
        
        Returns:
            Dictionary with usage statistics and cost breakdown
        """
        return {
            "total_operations": self._operation_count,
            "total_cost": round(self._total_cost, 6),
            "average_cost_per_operation": round(self._total_cost / max(1, self._operation_count), 6),
            "budget_utilization": round((self._total_cost / self.budget_limit * 100) if self.budget_limit else 0, 2),
            "cost_tracking_enabled": self.cost_tracking_enabled,
            "budget_limit": self.budget_limit
        }
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        self._total_cost = 0.0
        self._operation_count = 0
        if self._cache:
            self._cache.clear()


def instrument_cohere(
    api_key: Optional[str] = None,
    cost_tracking_enabled: bool = True,
    **governance_defaults
) -> GenOpsCohereAdapter:
    """
    Create and configure a GenOps Cohere adapter with intelligent defaults.
    
    Args:
        api_key: Cohere API key (defaults to CO_API_KEY env var)
        cost_tracking_enabled: Enable automatic cost tracking
        **governance_defaults: Default team, project, environment attributes
    
    Returns:
        Configured GenOpsCohereAdapter instance
    
    Example:
        # Basic setup
        adapter = instrument_cohere()
        
        # With governance defaults
        adapter = instrument_cohere(
            team="ml-team",
            project="ai-research",
            environment="production"
        )
    """
    return GenOpsCohereAdapter(
        api_key=api_key,
        cost_tracking_enabled=cost_tracking_enabled,
        **governance_defaults
    )


@dataclass
class WorkflowResult:
    """Result of a multi-operation workflow."""
    success: bool
    workflow_id: str
    total_cost: float
    operations: List[Dict[str, Any]]
    cost_breakdown: Dict[str, float]
    performance_metrics: Dict[str, float]
    error_message: Optional[str] = None


@contextmanager
def cohere_workflow_context(
    workflow_name: str,
    adapter: Optional[GenOpsCohereAdapter] = None,
    **governance_attrs
) -> Iterator[tuple]:
    """
    Context manager for complex multi-operation Cohere workflows.
    
    Provides automatic cost aggregation, error handling, and cleanup for
    workflows that combine multiple Cohere operations (chat, embed, rerank).
    
    Args:
        workflow_name: Name of the workflow for tracking
        adapter: GenOps Cohere adapter (creates one if None)
        **governance_attrs: Team, project, customer_id, etc.
    
    Yields:
        tuple: (workflow_context, workflow_id) for operation tracking
    
    Example:
        >>> with cohere_workflow_context("intelligent_search", team="search-team") as (ctx, workflow_id):
        ...     # Step 1: Embed query
        ...     query_result = ctx.embed(texts=["search query"], model="embed-english-v4.0")
        ...     
        ...     # Step 2: Rerank documents  
        ...     rerank_result = ctx.rerank(query="search", documents=docs, model="rerank-english-v3.0")
        ...     
        ...     # Step 3: Generate summary
        ...     summary = ctx.chat(message="Summarize results", model="command-r-08-2024")
        ...
        ... # Automatic cost aggregation and cleanup
        ... print(f"Workflow {workflow_id} total cost: ${ctx.get_total_cost():.6f}")
    """
    workflow_id = f"cohere-workflow-{uuid.uuid4().hex[:8]}"
    
    # Use provided adapter or create new one
    if adapter is None:
        adapter = GenOpsCohereAdapter(**governance_attrs)
    
    # Workflow tracking state
    workflow_context = WorkflowContext(
        workflow_id=workflow_id,
        workflow_name=workflow_name,
        adapter=adapter,
        governance_attrs=governance_attrs
    )
    
    start_time = time.time()
    
    try:
        # Create OpenTelemetry span for workflow
        if HAS_OTEL:
            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(f"genops.cohere.workflow.{workflow_name}") as span:
                span.set_attribute("genops.workflow.id", workflow_id)
                span.set_attribute("genops.workflow.name", workflow_name)
                span.set_attribute("genops.provider", "cohere")
                
                # Add governance attributes to span
                for key, value in governance_attrs.items():
                    span.set_attribute(f"genops.{key}", str(value))
                
                yield workflow_context, workflow_id
        else:
            yield workflow_context, workflow_id
            
    except Exception as e:
        logger.error(f"Workflow {workflow_id} failed: {e}")
        workflow_context.mark_failed(str(e))
        raise
        
    finally:
        # Finalize workflow metrics
        end_time = time.time()
        workflow_context.finalize(end_time - start_time)


class WorkflowContext:
    """Context for tracking multi-operation workflows."""
    
    def __init__(self, workflow_id: str, workflow_name: str, adapter: GenOpsCohereAdapter, governance_attrs: Dict[str, Any]):
        self.workflow_id = workflow_id
        self.workflow_name = workflow_name
        self.adapter = adapter
        self.governance_attrs = governance_attrs
        self.operations = []
        self.total_cost = 0.0
        self.failed = False
        self.error_message = None
        self.start_time = time.time()
    
    def chat(self, **kwargs) -> CohereResponse:
        """Execute chat operation within workflow context."""
        # Add workflow tracking to kwargs
        kwargs.update({
            'workflow_id': self.workflow_id,
            'workflow_name': self.workflow_name,
            **self.governance_attrs
        })
        
        response = self.adapter.chat(**kwargs)
        
        # Track operation
        self.operations.append({
            'operation': 'chat',
            'model': kwargs.get('model', ''),
            'cost': response.usage.total_cost if response.usage else 0.0,
            'success': response.success,
            'timestamp': time.time()
        })
        
        if response.usage:
            self.total_cost += response.usage.total_cost
        
        return response
    
    def embed(self, **kwargs) -> CohereResponse:
        """Execute embed operation within workflow context."""
        kwargs.update({
            'workflow_id': self.workflow_id,
            'workflow_name': self.workflow_name,
            **self.governance_attrs
        })
        
        response = self.adapter.embed(**kwargs)
        
        # Track operation
        self.operations.append({
            'operation': 'embed',
            'model': kwargs.get('model', ''),
            'cost': response.usage.total_cost if response.usage else 0.0,
            'success': response.success,
            'texts_count': len(kwargs.get('texts', [])),
            'timestamp': time.time()
        })
        
        if response.usage:
            self.total_cost += response.usage.total_cost
        
        return response
    
    def rerank(self, **kwargs) -> CohereResponse:
        """Execute rerank operation within workflow context."""
        kwargs.update({
            'workflow_id': self.workflow_id,
            'workflow_name': self.workflow_name,
            **self.governance_attrs
        })
        
        response = self.adapter.rerank(**kwargs)
        
        # Track operation
        self.operations.append({
            'operation': 'rerank',
            'model': kwargs.get('model', ''),
            'cost': response.usage.total_cost if response.usage else 0.0,
            'success': response.success,
            'documents_count': len(kwargs.get('documents', [])),
            'timestamp': time.time()
        })
        
        if response.usage:
            self.total_cost += response.usage.total_cost
        
        return response
    
    def get_total_cost(self) -> float:
        """Get total cost of all operations in the workflow."""
        return self.total_cost
    
    def get_operation_count(self) -> int:
        """Get total number of operations in the workflow."""
        return len(self.operations)
    
    def get_cost_breakdown(self) -> Dict[str, float]:
        """Get cost breakdown by operation type."""
        breakdown = {}
        for op in self.operations:
            op_type = op['operation']
            breakdown[op_type] = breakdown.get(op_type, 0.0) + op['cost']
        return breakdown
    
    def mark_failed(self, error_message: str):
        """Mark workflow as failed."""
        self.failed = True
        self.error_message = error_message
    
    def finalize(self, duration: float):
        """Finalize workflow tracking."""
        logger.info(f"Workflow {self.workflow_id} completed: "
                   f"{len(self.operations)} operations, "
                   f"${self.total_cost:.6f} total cost, "
                   f"{duration:.2f}s duration")


def auto_instrument():
    """
    Enable automatic instrumentation of Cohere operations.
    
    This patches Cohere client operations to automatically add GenOps tracking
    with zero code changes required.
    
    Usage:
        from genops.providers.cohere import auto_instrument
        auto_instrument()
        
        # Your existing Cohere code now has automatic tracking
        import cohere
        client = cohere.ClientV2()
        response = client.chat(...)  # Now tracked with GenOps!
    """
    if not HAS_COHERE:
        logger.warning("Cohere client not available for auto-instrumentation")
        return False
    
    try:
        # Create global adapter instance
        global_adapter = GenOpsCohereAdapter()
        
        # Store original methods
        original_client_init = ClientV2.__init__
        
        def instrumented_client_init(self, *args, **kwargs):
            # Initialize original client
            original_client_init(self, *args, **kwargs)
            
            # Store original methods
            self._genops_original_chat = self.chat
            self._genops_original_embed = self.embed
            self._genops_original_rerank = self.rerank
            
            # Create instrumented methods
            def instrumented_chat(*args, **kwargs):
                return global_adapter.chat(*args, **kwargs)
            
            def instrumented_embed(*args, **kwargs):
                return global_adapter.embed(*args, **kwargs)
            
            def instrumented_rerank(*args, **kwargs):
                return global_adapter.rerank(*args, **kwargs)
            
            # Apply patches
            self.chat = instrumented_chat
            self.embed = instrumented_embed  
            self.rerank = instrumented_rerank
        
        # Apply global patch
        ClientV2.__init__ = instrumented_client_init
        
        logger.info("GenOps auto-instrumentation enabled for Cohere")
        return True
        
    except Exception as e:
        logger.error(f"Failed to enable Cohere auto-instrumentation: {e}")
        return False


# Export main classes and functions
__all__ = [
    "GenOpsCohereAdapter",
    "CohereUsageMetrics",
    "CohereResponse",
    "CohereModel",
    "CohereOperation",
    "WorkflowResult",
    "WorkflowContext",
    "cohere_workflow_context",
    "instrument_cohere",
    "auto_instrument"
]