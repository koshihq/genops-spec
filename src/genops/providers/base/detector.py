"""Framework detection utilities for GenOps AI governance."""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class FrameworkInfo:
    """Information about a detected framework."""
    
    def __init__(
        self,
        name: str,
        import_path: str,
        version: Optional[str] = None,
        framework_type: str = "unknown",
        available: bool = False,
        module_obj: Optional[Any] = None
    ):
        self.name = name
        self.import_path = import_path
        self.version = version
        self.framework_type = framework_type
        self.available = available
        self.module_obj = module_obj
        
    def __str__(self) -> str:
        status = "✓" if self.available else "✗"
        version_str = f" (v{self.version})" if self.version else ""
        return f"{status} {self.name}{version_str} [{self.framework_type}]"
        
    def __repr__(self) -> str:
        return f"FrameworkInfo(name='{self.name}', available={self.available}, version='{self.version}')"


class FrameworkDetector:
    """Automatic detection of installed AI frameworks."""
    
    # Registry of known frameworks with detection metadata
    FRAMEWORKS = {
        # Orchestration Frameworks
        'langchain': {
            'import_path': 'langchain',
            'version_attr': '__version__',
            'framework_type': 'orchestration',
            'description': 'LLM application orchestration framework'
        },
        'langchain_community': {
            'import_path': 'langchain_community',
            'version_attr': '__version__',
            'framework_type': 'orchestration', 
            'description': 'LangChain community integrations'
        },
        'haystack': {
            'import_path': 'haystack',
            'version_attr': '__version__',
            'framework_type': 'orchestration',
            'description': 'Pipeline framework for LLM applications'
        },
        
        # Training Frameworks
        'torch': {
            'import_path': 'torch',
            'version_attr': '__version__',
            'framework_type': 'training',
            'description': 'PyTorch deep learning framework'
        },
        'tensorflow': {
            'import_path': 'tensorflow',
            'version_attr': '__version__',
            'framework_type': 'training',
            'description': 'TensorFlow machine learning platform'
        },
        
        # Inference/Model Frameworks  
        'transformers': {
            'import_path': 'transformers',
            'version_attr': '__version__',
            'framework_type': 'inference',
            'description': 'HuggingFace Transformers library'
        },
        'sentence_transformers': {
            'import_path': 'sentence_transformers',
            'version_attr': '__version__',
            'framework_type': 'inference',
            'description': 'Sentence embeddings with transformers'
        },
        
        # Vector/Retrieval Frameworks
        'llamaindex': {
            'import_path': 'llama_index',
            'version_attr': '__version__',
            'framework_type': 'vector',
            'description': 'LlamaIndex data ingestion and retrieval'
        },
        'chromadb': {
            'import_path': 'chromadb',
            'version_attr': '__version__',
            'framework_type': 'vector',
            'description': 'Chroma vector database'
        },
        'pinecone': {
            'import_path': 'pinecone',
            'version_attr': '__version__',
            'framework_type': 'vector',
            'description': 'Pinecone vector database client'
        },
        'weaviate': {
            'import_path': 'weaviate',
            'version_attr': '__version__',
            'framework_type': 'vector',
            'description': 'Weaviate vector database client'
        },
        
        # Multimodal Frameworks
        'nemo_toolkit': {
            'import_path': 'nemo',
            'version_attr': '__version__',
            'framework_type': 'multimodal',
            'description': 'NVIDIA NeMo toolkit'
        },
        
        # AutoML Frameworks
        'nni': {
            'import_path': 'nni',
            'version_attr': '__version__',
            'framework_type': 'automl',
            'description': 'Neural Network Intelligence AutoML toolkit'
        },
        'optuna': {
            'import_path': 'optuna',
            'version_attr': '__version__',
            'framework_type': 'automl',
            'description': 'Hyperparameter optimization framework'
        },
        
        # Distributed Training Frameworks
        'horovod': {
            'import_path': 'horovod.torch',  # Common entry point
            'version_attr': '__version__',
            'framework_type': 'distributed',
            'description': 'Distributed training framework'
        },
        'ray': {
            'import_path': 'ray',
            'version_attr': '__version__',
            'framework_type': 'distributed',
            'description': 'Distributed computing framework'
        },
        'deepspeed': {
            'import_path': 'deepspeed',
            'version_attr': '__version__',
            'framework_type': 'distributed',
            'description': 'Microsoft DeepSpeed distributed training'
        },
        
        # Legacy/Other Frameworks
        'mxnet': {
            'import_path': 'mxnet',
            'version_attr': '__version__',
            'framework_type': 'training',
            'description': 'Apache MXNet deep learning framework'
        },
        'jax': {
            'import_path': 'jax',
            'version_attr': '__version__',
            'framework_type': 'training', 
            'description': 'JAX NumPy-compatible ML framework'
        },
        'flax': {
            'import_path': 'flax',
            'version_attr': '__version__',
            'framework_type': 'training',
            'description': 'Flax neural network library for JAX'
        }
    }
    
    def __init__(self):
        self._detected_frameworks: Optional[Dict[str, FrameworkInfo]] = None
        
    def detect_all_frameworks(self, force_refresh: bool = False) -> Dict[str, FrameworkInfo]:
        """
        Detect all available frameworks.
        
        Args:
            force_refresh: Force re-detection even if already cached
            
        Returns:
            Dictionary of framework_name -> FrameworkInfo
        """
        if self._detected_frameworks is None or force_refresh:
            self._detected_frameworks = {}
            
            for name, config in self.FRAMEWORKS.items():
                framework_info = self.detect_framework(name, config)
                self._detected_frameworks[name] = framework_info
                
        return self._detected_frameworks
        
    def detect_framework(self, name: str, config: Dict) -> FrameworkInfo:
        """
        Detect a specific framework.
        
        Args:
            name: Framework name
            config: Framework configuration dict
            
        Returns:
            FrameworkInfo instance
        """
        import_path = config['import_path']
        version_attr = config.get('version_attr', '__version__')
        framework_type = config.get('framework_type', 'unknown')
        
        try:
            module = importlib.import_module(import_path)
            version = getattr(module, version_attr, None)
            
            framework_info = FrameworkInfo(
                name=name,
                import_path=import_path,
                version=version,
                framework_type=framework_type,
                available=True,
                module_obj=module
            )
            
            logger.debug(f"✓ Detected {name} v{version}")
            return framework_info
            
        except ImportError as e:
            logger.debug(f"✗ {name} not available: {e}")
            return FrameworkInfo(
                name=name,
                import_path=import_path,
                framework_type=framework_type,
                available=False
            )
        except Exception as e:
            logger.warning(f"Error detecting {name}: {e}")
            return FrameworkInfo(
                name=name,
                import_path=import_path,
                framework_type=framework_type,
                available=False
            )
            
    def get_available_frameworks(self, framework_type: Optional[str] = None) -> List[FrameworkInfo]:
        """
        Get list of available frameworks, optionally filtered by type.
        
        Args:
            framework_type: Filter by framework type (orchestration, training, etc.)
            
        Returns:
            List of available FrameworkInfo instances
        """
        all_frameworks = self.detect_all_frameworks()
        available = [info for info in all_frameworks.values() if info.available]
        
        if framework_type:
            available = [info for info in available if info.framework_type == framework_type]
            
        return available
        
    def get_framework_types(self) -> Set[str]:
        """
        Get set of all framework types.
        
        Returns:
            Set of framework type strings
        """
        return {config['framework_type'] for config in self.FRAMEWORKS.values()}
        
    def is_framework_available(self, name: str) -> bool:
        """
        Check if a specific framework is available.
        
        Args:
            name: Framework name
            
        Returns:
            True if framework is available, False otherwise
        """
        frameworks = self.detect_all_frameworks()
        return frameworks.get(name, FrameworkInfo(name, '', available=False)).available
        
    def get_framework_version(self, name: str) -> Optional[str]:
        """
        Get version of a specific framework.
        
        Args:
            name: Framework name
            
        Returns:
            Version string if available, None otherwise
        """
        frameworks = self.detect_all_frameworks()
        framework_info = frameworks.get(name)
        return framework_info.version if framework_info and framework_info.available else None
        
    def print_detection_summary(self) -> None:
        """Print a summary of detected frameworks."""
        frameworks = self.detect_all_frameworks()
        
        print("\n🔍 GenOps Framework Detection Summary")
        print("=" * 50)
        
        # Group by framework type
        by_type: Dict[str, List[FrameworkInfo]] = {}
        for info in frameworks.values():
            if info.framework_type not in by_type:
                by_type[info.framework_type] = []
            by_type[info.framework_type].append(info)
            
        for framework_type in sorted(by_type.keys()):
            print(f"\n📦 {framework_type.title()} Frameworks:")
            for info in sorted(by_type[framework_type], key=lambda x: x.name):
                print(f"  {info}")
                
        # Summary stats
        total = len(frameworks)
        available = len([f for f in frameworks.values() if f.available])
        print(f"\n📊 Summary: {available}/{total} frameworks available")
        
    def add_custom_framework(
        self, 
        name: str,
        import_path: str,
        framework_type: str = "custom",
        version_attr: str = "__version__",
        description: str = ""
    ) -> None:
        """
        Add a custom framework to the detection registry.
        
        Args:
            name: Framework name
            import_path: Python import path
            framework_type: Framework category
            version_attr: Attribute name for version detection
            description: Framework description
        """
        self.FRAMEWORKS[name] = {
            'import_path': import_path,
            'version_attr': version_attr,
            'framework_type': framework_type,
            'description': description or f"Custom {framework_type} framework"
        }
        
        # Clear cache to force re-detection
        self._detected_frameworks = None
        logger.info(f"Added custom framework: {name}")


# Singleton detector instance
_detector_instance: Optional[FrameworkDetector] = None


def get_framework_detector() -> FrameworkDetector:
    """Get the global framework detector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = FrameworkDetector()
    return _detector_instance


def detect_frameworks() -> Dict[str, FrameworkInfo]:
    """Convenience function to detect all frameworks."""
    return get_framework_detector().detect_all_frameworks()


def is_framework_available(name: str) -> bool:
    """Convenience function to check if a framework is available."""
    return get_framework_detector().is_framework_available(name)