"""Model management and optimization for Ollama local deployments."""

import logging
import time
import statistics
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import json

logger = logging.getLogger(__name__)

# Try to import Ollama client if available
try:
    import ollama
    HAS_OLLAMA_CLIENT = True
except ImportError:
    HAS_OLLAMA_CLIENT = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class ModelSize(Enum):
    """Model size categories for optimization."""
    TINY = "tiny"      # <1B parameters
    SMALL = "small"    # 1B-7B parameters  
    MEDIUM = "medium"  # 7B-13B parameters
    LARGE = "large"    # 13B-33B parameters
    XLARGE = "xlarge"  # 33B+ parameters


class ModelType(Enum):
    """Model type categories."""
    CHAT = "chat"
    CODE = "code"
    INSTRUCT = "instruct"
    EMBEDDING = "embedding"
    MULTIMODAL = "multimodal"
    SPECIALIZED = "specialized"


@dataclass
class ModelInfo:
    """Information about an Ollama model."""
    
    name: str
    size_gb: float
    parameter_count: Optional[str] = None
    family: Optional[str] = None
    format: Optional[str] = None
    
    # Performance characteristics
    avg_tokens_per_second: float = 0.0
    avg_memory_usage_mb: float = 0.0
    avg_inference_latency_ms: float = 0.0
    
    # Usage statistics
    total_inferences: int = 0
    total_runtime_hours: float = 0.0
    last_used: Optional[float] = None
    
    # Cost efficiency
    cost_per_inference: float = 0.0
    tokens_per_dollar: float = 0.0
    
    # Quality metrics
    success_rate: float = 100.0
    error_count: int = 0
    
    # Model categorization
    size_category: ModelSize = ModelSize.MEDIUM
    model_type: ModelType = ModelType.CHAT
    
    # Optimization recommendations
    recommended_for: List[str] = field(default_factory=list)
    optimization_notes: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize calculated fields."""
        self._categorize_model()
    
    def _categorize_model(self):
        """Automatically categorize model based on name and size."""
        name_lower = self.name.lower()
        
        # Determine model type
        if any(keyword in name_lower for keyword in ["code", "codellama", "starcoder"]):
            self.model_type = ModelType.CODE
        elif any(keyword in name_lower for keyword in ["instruct", "chat"]):
            self.model_type = ModelType.INSTRUCT if "instruct" in name_lower else ModelType.CHAT
        elif "embed" in name_lower:
            self.model_type = ModelType.EMBEDDING
        elif any(keyword in name_lower for keyword in ["vision", "multimodal", "llava"]):
            self.model_type = ModelType.MULTIMODAL
        else:
            self.model_type = ModelType.CHAT
        
        # Determine size category
        if self.size_gb < 1.0:
            self.size_category = ModelSize.TINY
        elif self.size_gb < 4.0:
            self.size_category = ModelSize.SMALL
        elif self.size_gb < 8.0:
            self.size_category = ModelSize.MEDIUM
        elif self.size_gb < 20.0:
            self.size_category = ModelSize.LARGE
        else:
            self.size_category = ModelSize.XLARGE
    
    def update_performance_stats(self, inference_time_ms: float, tokens: int = 0, memory_mb: float = 0.0, cost: float = 0.0):
        """Update performance statistics with new inference data."""
        self.total_inferences += 1
        self.last_used = time.time()
        
        # Update averages
        if inference_time_ms > 0:
            self.avg_inference_latency_ms = (
                (self.avg_inference_latency_ms * (self.total_inferences - 1) + inference_time_ms)
                / self.total_inferences
            )
        
        if tokens > 0 and inference_time_ms > 0:
            tokens_per_second = tokens / (inference_time_ms / 1000)
            self.avg_tokens_per_second = (
                (self.avg_tokens_per_second * (self.total_inferences - 1) + tokens_per_second)
                / self.total_inferences
            )
        
        if memory_mb > 0:
            self.avg_memory_usage_mb = (
                (self.avg_memory_usage_mb * (self.total_inferences - 1) + memory_mb)
                / self.total_inferences
            )
        
        if cost > 0:
            self.cost_per_inference = (
                (self.cost_per_inference * (self.total_inferences - 1) + cost)
                / self.total_inferences
            )
            
            if tokens > 0 and cost > 0:
                self.tokens_per_dollar = tokens / cost
    
    def mark_error(self):
        """Mark an inference error."""
        self.error_count += 1
        self.success_rate = ((self.total_inferences - self.error_count) / max(self.total_inferences, 1)) * 100


@dataclass
class ModelOptimizer:
    """Optimization recommendations for models."""
    
    model_name: str
    current_performance: Dict[str, float]
    optimization_opportunities: List[str] = field(default_factory=list)
    alternative_models: List[str] = field(default_factory=list)
    cost_savings_potential: float = 0.0
    performance_improvement_potential: float = 0.0
    
    def add_recommendation(self, category: str, description: str, impact: str = "medium"):
        """Add an optimization recommendation."""
        self.optimization_opportunities.append(f"[{impact.upper()}] {category}: {description}")
    
    def suggest_alternative(self, model_name: str, reason: str):
        """Suggest an alternative model with reasoning."""
        self.alternative_models.append(f"{model_name} ({reason})")


@dataclass
class ModelComparison:
    """Comparison between multiple models."""
    
    models: List[str]
    comparison_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    recommendations: Dict[str, str] = field(default_factory=dict)
    best_for_cost: Optional[str] = None
    best_for_speed: Optional[str] = None
    best_for_quality: Optional[str] = None
    
    def add_metric(self, metric_name: str, model_values: Dict[str, float]):
        """Add a comparison metric for all models."""
        self.comparison_metrics[metric_name] = model_values
        
        # Update best performers
        if metric_name == "cost_per_inference":
            self.best_for_cost = min(model_values.keys(), key=lambda m: model_values[m])
        elif metric_name == "avg_tokens_per_second":
            self.best_for_speed = max(model_values.keys(), key=lambda m: model_values[m])
        elif metric_name == "success_rate":
            self.best_for_quality = max(model_values.keys(), key=lambda m: model_values[m])


class OllamaModelManager:
    """
    Comprehensive model management for Ollama deployments.
    
    Handles:
    - Model discovery and cataloging
    - Performance tracking and optimization
    - Cost analysis and recommendations
    - Model comparison and selection
    - Lifecycle management and maintenance
    """
    
    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        enable_auto_optimization: bool = True,
        track_performance_history: bool = True,
        history_size: int = 1000
    ):
        """
        Initialize model manager.
        
        Args:
            ollama_base_url: Base URL for Ollama server
            enable_auto_optimization: Enable automatic optimization recommendations
            track_performance_history: Track detailed performance history
            history_size: Number of historical data points to keep
        """
        self.ollama_base_url = ollama_base_url.rstrip('/')
        self.enable_auto_optimization = enable_auto_optimization
        self.track_performance_history = track_performance_history
        self.history_size = history_size
        
        # Model tracking
        self.models: Dict[str, ModelInfo] = {}
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        
        # Optimization tracking
        self.optimization_cache: Dict[str, ModelOptimizer] = {}
        self.last_optimization_check: float = 0.0
        self.optimization_interval: float = 3600.0  # 1 hour
        
        # Initialize Ollama client
        self.client = None
        if HAS_OLLAMA_CLIENT:
            try:
                self.client = ollama.Client(host=ollama_base_url)
            except Exception as e:
                logger.warning(f"Failed to initialize Ollama client: {e}")
        
        logger.info(f"Initialized Ollama model manager (optimization: {enable_auto_optimization})")
    
    def discover_models(self) -> List[ModelInfo]:
        """
        Discover and catalog all available Ollama models.
        
        Returns:
            List of discovered models with metadata
        """
        models = []
        
        try:
            if self.client:
                # Use ollama client
                response = self.client.list()
                model_list = response.get('models', [])
            else:
                # Use HTTP API
                if not HAS_REQUESTS:
                    raise ImportError("requests library required for HTTP API")
                
                response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=10)
                response.raise_for_status()
                model_list = response.json().get('models', [])
            
            for model_data in model_list:
                model_name = model_data.get('name', 'unknown')
                size_bytes = model_data.get('size', 0)
                size_gb = size_bytes / (1024**3) if size_bytes > 0 else 0.0
                
                # Extract additional metadata
                details = model_data.get('details', {})
                parameter_count = details.get('parameter_size', None)
                family = details.get('family', None)
                format = details.get('format', None)
                
                model_info = ModelInfo(
                    name=model_name,
                    size_gb=size_gb,
                    parameter_count=parameter_count,
                    family=family,
                    format=format
                )
                
                models.append(model_info)
                self.models[model_name] = model_info
            
            logger.info(f"Discovered {len(models)} Ollama models")
            return models
            
        except Exception as e:
            logger.error(f"Failed to discover Ollama models: {e}")
            return []
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        return self.models.get(model_name)
    
    def update_model_performance(self, model_name: str, **performance_data):
        """
        Update performance metrics for a model.
        
        Args:
            model_name: Name of the model
            **performance_data: Performance metrics (inference_time_ms, tokens, memory_mb, cost)
        """
        if model_name not in self.models:
            # Create basic model info if not exists
            self.models[model_name] = ModelInfo(
                name=model_name,
                size_gb=0.0  # Will be updated when discovered
            )
        
        model = self.models[model_name]
        
        # Update performance statistics
        inference_time = performance_data.get('inference_time_ms', 0.0)
        tokens = performance_data.get('tokens', 0)
        memory_mb = performance_data.get('memory_mb', 0.0)
        cost = performance_data.get('cost', 0.0)
        
        if inference_time > 0:
            model.update_performance_stats(inference_time, tokens, memory_mb, cost)
        
        # Track performance history
        if self.track_performance_history and inference_time > 0:
            history_entry = {
                'timestamp': time.time(),
                'inference_time_ms': inference_time,
                'tokens': tokens,
                'memory_mb': memory_mb,
                'cost': cost
            }
            self.performance_history[model_name].append(history_entry)
        
        # Check for optimization opportunities
        if self.enable_auto_optimization:
            self._check_optimization_opportunities(model_name)
    
    def mark_model_error(self, model_name: str, error_type: str = "inference"):
        """Mark an error for a model."""
        if model_name in self.models:
            self.models[model_name].mark_error()
            logger.debug(f"Marked error for model {model_name}: {error_type}")
    
    def get_model_performance_summary(self, model_name: str = None) -> Dict[str, Any]:
        """
        Get performance summary for specific model or all models.
        
        Args:
            model_name: Specific model name, or None for all models
            
        Returns:
            Performance summary data
        """
        if model_name:
            if model_name not in self.models:
                return {}
            
            model = self.models[model_name]
            return {
                'model_name': model.name,
                'total_inferences': model.total_inferences,
                'avg_inference_latency_ms': model.avg_inference_latency_ms,
                'avg_tokens_per_second': model.avg_tokens_per_second,
                'avg_memory_usage_mb': model.avg_memory_usage_mb,
                'cost_per_inference': model.cost_per_inference,
                'success_rate': model.success_rate,
                'size_category': model.size_category.value,
                'model_type': model.model_type.value,
                'last_used': model.last_used
            }
        else:
            # Summary for all models
            summaries = {}
            for name, model in self.models.items():
                summaries[name] = self.get_model_performance_summary(name)
            return summaries
    
    def compare_models(self, model_names: List[str], metrics: List[str] = None) -> ModelComparison:
        """
        Compare performance across multiple models.
        
        Args:
            model_names: List of model names to compare
            metrics: Specific metrics to compare (default: all key metrics)
            
        Returns:
            Model comparison with recommendations
        """
        if metrics is None:
            metrics = [
                'avg_inference_latency_ms',
                'avg_tokens_per_second', 
                'cost_per_inference',
                'success_rate',
                'avg_memory_usage_mb'
            ]
        
        comparison = ModelComparison(models=model_names)
        
        for metric in metrics:
            metric_values = {}
            for model_name in model_names:
                if model_name in self.models:
                    model = self.models[model_name]
                    metric_values[model_name] = getattr(model, metric, 0.0)
            
            if metric_values:
                comparison.add_metric(metric, metric_values)
        
        # Add recommendations
        if comparison.best_for_cost:
            comparison.recommendations['cost'] = f"Use {comparison.best_for_cost} for lowest cost per inference"
        if comparison.best_for_speed:
            comparison.recommendations['speed'] = f"Use {comparison.best_for_speed} for highest throughput"
        if comparison.best_for_quality:
            comparison.recommendations['quality'] = f"Use {comparison.best_for_quality} for highest success rate"
        
        return comparison
    
    def get_optimization_recommendations(self, model_name: str = None) -> Dict[str, ModelOptimizer]:
        """
        Get optimization recommendations for specific model or all models.
        
        Args:
            model_name: Specific model name, or None for all models
            
        Returns:
            Optimization recommendations
        """
        if model_name:
            if model_name not in self.optimization_cache:
                self._generate_optimization_recommendations(model_name)
            return {model_name: self.optimization_cache.get(model_name)}
        else:
            # Generate recommendations for all models
            recommendations = {}
            for name in self.models.keys():
                if name not in self.optimization_cache:
                    self._generate_optimization_recommendations(name)
                if name in self.optimization_cache:
                    recommendations[name] = self.optimization_cache[name]
            return recommendations
    
    def _check_optimization_opportunities(self, model_name: str):
        """Check if it's time to update optimization recommendations."""
        current_time = time.time()
        if current_time - self.last_optimization_check > self.optimization_interval:
            self._generate_optimization_recommendations(model_name)
            self.last_optimization_check = current_time
    
    def _generate_optimization_recommendations(self, model_name: str):
        """Generate optimization recommendations for a model."""
        if model_name not in self.models:
            return
        
        model = self.models[model_name]
        
        optimizer = ModelOptimizer(
            model_name=model_name,
            current_performance={
                'latency_ms': model.avg_inference_latency_ms,
                'tokens_per_second': model.avg_tokens_per_second,
                'memory_usage_mb': model.avg_memory_usage_mb,
                'cost_per_inference': model.cost_per_inference,
                'success_rate': model.success_rate
            }
        )
        
        # Performance recommendations
        if model.avg_inference_latency_ms > 5000:  # >5 seconds
            optimizer.add_recommendation(
                "Latency", 
                f"High latency ({model.avg_inference_latency_ms:.0f}ms) - consider using quantized version or smaller model",
                "high"
            )
        
        if model.avg_tokens_per_second < 5:  # <5 tokens/sec
            optimizer.add_recommendation(
                "Throughput",
                f"Low throughput ({model.avg_tokens_per_second:.1f} tokens/sec) - check GPU utilization or use faster model",
                "high"
            )
        
        if model.avg_memory_usage_mb > 8000:  # >8GB
            optimizer.add_recommendation(
                "Memory",
                f"High memory usage ({model.avg_memory_usage_mb:.0f}MB) - consider using quantized version",
                "medium"
            )
        
        # Cost optimization
        if model.cost_per_inference > 0.01:  # >1 cent per inference
            optimizer.add_recommendation(
                "Cost",
                f"High cost per inference (${model.cost_per_inference:.4f}) - evaluate smaller models for simple tasks",
                "medium"
            )
        
        # Quality issues
        if model.success_rate < 90:
            optimizer.add_recommendation(
                "Reliability",
                f"Low success rate ({model.success_rate:.1f}%) - investigate error patterns",
                "high"
            )
        
        # Model alternatives
        self._suggest_model_alternatives(optimizer, model)
        
        self.optimization_cache[model_name] = optimizer
    
    def _suggest_model_alternatives(self, optimizer: ModelOptimizer, model: ModelInfo):
        """Suggest alternative models based on current model performance."""
        model_name_lower = model.name.lower()
        
        # Suggest smaller models for cost optimization
        if model.size_category in [ModelSize.LARGE, ModelSize.XLARGE]:
            if model.cost_per_inference > 0.005:
                optimizer.suggest_alternative(
                    "llama3.2:3b", 
                    "smaller model with good performance for most tasks"
                )
        
        # Suggest faster models for latency issues
        if model.avg_inference_latency_ms > 3000:
            if "llama" in model_name_lower:
                optimizer.suggest_alternative(
                    "llama3.2:1b",
                    "fastest LLaMA variant for simple tasks"
                )
        
        # Suggest specialized models
        if model.model_type == ModelType.CHAT and "code" not in model_name_lower:
            optimizer.suggest_alternative(
                "codellama:7b",
                "specialized for code-related tasks"
            )
    
    def get_model_usage_analytics(self, days: int = 30) -> Dict[str, Any]:
        """
        Get model usage analytics over specified time period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Usage analytics summary
        """
        cutoff_time = time.time() - (days * 24 * 3600)
        
        analytics = {
            'analysis_period_days': days,
            'total_models': len(self.models),
            'active_models': 0,
            'total_inferences': 0,
            'total_cost': 0.0,
            'models_by_usage': [],
            'models_by_cost': [],
            'performance_trends': {}
        }
        
        for model_name, model in self.models.items():
            # Check if model was used in analysis period
            if model.last_used and model.last_used > cutoff_time:
                analytics['active_models'] += 1
            
            analytics['total_inferences'] += model.total_inferences
            analytics['total_cost'] += model.cost_per_inference * model.total_inferences
            
            # Add to usage ranking
            analytics['models_by_usage'].append({
                'model': model_name,
                'inferences': model.total_inferences,
                'avg_latency_ms': model.avg_inference_latency_ms,
                'success_rate': model.success_rate
            })
            
            # Add to cost ranking  
            analytics['models_by_cost'].append({
                'model': model_name,
                'total_cost': model.cost_per_inference * model.total_inferences,
                'cost_per_inference': model.cost_per_inference
            })
        
        # Sort rankings
        analytics['models_by_usage'].sort(key=lambda x: x['inferences'], reverse=True)
        analytics['models_by_cost'].sort(key=lambda x: x['total_cost'], reverse=True)
        
        return analytics
    
    def export_model_data(self, format: str = "json") -> str:
        """
        Export model data for backup or analysis.
        
        Args:
            format: Export format ("json" or "csv")
            
        Returns:
            Exported data as string
        """
        if format.lower() == "json":
            export_data = {
                'export_timestamp': time.time(),
                'models': {}
            }
            
            for name, model in self.models.items():
                export_data['models'][name] = {
                    'name': model.name,
                    'size_gb': model.size_gb,
                    'parameter_count': model.parameter_count,
                    'avg_tokens_per_second': model.avg_tokens_per_second,
                    'avg_inference_latency_ms': model.avg_inference_latency_ms,
                    'avg_memory_usage_mb': model.avg_memory_usage_mb,
                    'total_inferences': model.total_inferences,
                    'cost_per_inference': model.cost_per_inference,
                    'success_rate': model.success_rate,
                    'size_category': model.size_category.value,
                    'model_type': model.model_type.value,
                    'last_used': model.last_used
                }
            
            return json.dumps(export_data, indent=2)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global model manager instance
_global_manager: Optional[OllamaModelManager] = None


def get_model_manager() -> OllamaModelManager:
    """Get or create global model manager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = OllamaModelManager()
    return _global_manager


def set_model_manager(manager: OllamaModelManager) -> None:
    """Set global model manager instance."""
    global _global_manager
    _global_manager = manager


def create_model_manager(**kwargs) -> OllamaModelManager:
    """Create a new model manager with specified configuration."""
    return OllamaModelManager(**kwargs)


# Export main classes and functions
__all__ = [
    "OllamaModelManager",
    "ModelInfo",
    "ModelOptimizer", 
    "ModelComparison",
    "ModelSize",
    "ModelType",
    "get_model_manager",
    "set_model_manager",
    "create_model_manager"
]