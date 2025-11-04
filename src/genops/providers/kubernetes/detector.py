#!/usr/bin/env python3
"""
🔍 Kubernetes Environment Detector

Detects Kubernetes environment and extracts cluster metadata for governance telemetry.
Integrates with OpenTelemetry's Kubernetes resource detection for comprehensive attribution.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Kubernetes service account token path
K8S_TOKEN_PATH = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
K8S_NAMESPACE_PATH = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
K8S_CA_CERT_PATH = Path("/var/run/secrets/kubernetes.io/serviceaccount/ca.crt")


@dataclass
class KubernetesContext:
    """Kubernetes runtime context information."""
    
    # Pod identification
    pod_name: Optional[str] = None
    pod_namespace: Optional[str] = None
    pod_uid: Optional[str] = None
    
    # Node information
    node_name: Optional[str] = None
    node_ip: Optional[str] = None
    
    # Service account
    service_account: Optional[str] = None
    
    # Container runtime
    container_name: Optional[str] = None
    container_id: Optional[str] = None
    
    # Cluster metadata
    cluster_name: Optional[str] = None
    cluster_uid: Optional[str] = None
    
    # Labels and annotations
    labels: Dict[str, str] = None
    annotations: Dict[str, str] = None
    
    # Runtime state
    is_running_in_kubernetes: bool = False
    has_service_account: bool = False
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}
        if self.annotations is None:
            self.annotations = {}


class KubernetesDetector:
    """Detects Kubernetes environment and extracts governance-relevant metadata."""
    
    def __init__(self):
        self.context = KubernetesContext()
        self._detect_environment()
    
    def _detect_environment(self) -> None:
        """Detect if running in Kubernetes and extract context information."""
        
        logger.debug("Detecting Kubernetes environment...")
        
        # Check for Kubernetes service account token
        if K8S_TOKEN_PATH.exists():
            self.context.is_running_in_kubernetes = True
            self.context.has_service_account = True
            logger.debug("✅ Kubernetes service account detected")
            self._extract_service_account_info()
        
        # Extract environment variables set by Kubernetes
        self._extract_pod_info()
        self._extract_node_info()
        self._extract_container_info()
        
        # Try to detect additional cluster metadata
        self._detect_cluster_info()
        
        if self.context.is_running_in_kubernetes:
            logger.info(f"🚢 Kubernetes context detected: {self.context.pod_namespace}/{self.context.pod_name}")
        else:
            logger.debug("Not running in Kubernetes environment")
    
    def _extract_service_account_info(self) -> None:
        """Extract service account information."""
        
        try:
            # Read namespace from service account
            if K8S_NAMESPACE_PATH.exists():
                namespace = K8S_NAMESPACE_PATH.read_text().strip()
                self.context.pod_namespace = namespace
                logger.debug(f"Service account namespace: {namespace}")
        except Exception as e:
            logger.warning(f"Failed to read service account namespace: {e}")
    
    def _extract_pod_info(self) -> None:
        """Extract pod information from environment variables."""
        
        # Pod metadata (set by Kubernetes downward API)
        self.context.pod_name = os.getenv("K8S_POD_NAME") or os.getenv("HOSTNAME")
        self.context.pod_namespace = (
            self.context.pod_namespace or 
            os.getenv("K8S_NAMESPACE") or 
            os.getenv("POD_NAMESPACE")
        )
        self.context.pod_uid = os.getenv("K8S_POD_UID") or os.getenv("POD_UID")
        
        # Detect if we're in a pod even without explicit environment variables
        if self.context.pod_name and not self.context.is_running_in_kubernetes:
            # Check for other Kubernetes indicators
            if any(Path(p).exists() for p in ["/proc/1/cgroup", "/etc/resolv.conf"]):
                try:
                    # Check cgroup for Kubernetes patterns
                    cgroup_content = Path("/proc/1/cgroup").read_text()
                    if any(pattern in cgroup_content for pattern in ["kubepods", "docker", "containerd"]):
                        self.context.is_running_in_kubernetes = True
                        logger.debug("✅ Kubernetes detected via cgroup analysis")
                except Exception as e:
                    logger.debug(f"Cgroup analysis failed: {e}")
    
    def _extract_node_info(self) -> None:
        """Extract node information from environment variables."""
        
        self.context.node_name = os.getenv("K8S_NODE_NAME") or os.getenv("NODE_NAME")
        self.context.node_ip = os.getenv("K8S_NODE_IP") or os.getenv("NODE_IP")
    
    def _extract_container_info(self) -> None:
        """Extract container runtime information."""
        
        self.context.container_name = os.getenv("CONTAINER_NAME")
        
        # Try to extract container ID from cgroup
        try:
            cgroup_path = Path("/proc/self/cgroup")
            if cgroup_path.exists():
                cgroup_content = cgroup_path.read_text()
                
                # Look for Docker container ID pattern
                for line in cgroup_content.split('\n'):
                    if 'docker' in line and '/' in line:
                        parts = line.split('/')
                        if len(parts) > 1:
                            container_id = parts[-1].strip()
                            if len(container_id) >= 12:  # Docker container IDs are at least 12 chars
                                self.context.container_id = container_id[:12]
                                break
        except Exception as e:
            logger.debug(f"Container ID detection failed: {e}")
    
    def _detect_cluster_info(self) -> None:
        """Attempt to detect cluster-level information."""
        
        # Cluster name from environment or metadata
        self.context.cluster_name = (
            os.getenv("CLUSTER_NAME") or 
            os.getenv("K8S_CLUSTER_NAME")
        )
        
        # Try to extract from cloud provider metadata
        self._detect_cloud_metadata()
    
    def _detect_cloud_metadata(self) -> None:
        """Detect cloud provider metadata if available."""
        
        # This is a placeholder for cloud provider specific detection
        # In production, you might query cloud provider metadata services
        
        # EKS cluster detection
        if os.getenv("AWS_REGION") and not self.context.cluster_name:
            # Could query AWS metadata service for EKS cluster name
            pass
        
        # GKE cluster detection  
        if os.getenv("GOOGLE_CLOUD_PROJECT") and not self.context.cluster_name:
            # Could query GCP metadata service for GKE cluster name
            pass
        
        # AKS cluster detection
        if os.getenv("AZURE_SUBSCRIPTION_ID") and not self.context.cluster_name:
            # Could query Azure metadata service for AKS cluster name
            pass
    
    def get_governance_attributes(self) -> Dict[str, str]:
        """Get Kubernetes context as governance telemetry attributes."""
        
        attributes = {}
        
        # Core Kubernetes attributes
        if self.context.pod_name:
            attributes["k8s.pod.name"] = self.context.pod_name
        if self.context.pod_namespace:
            attributes["k8s.namespace.name"] = self.context.pod_namespace
        if self.context.pod_uid:
            attributes["k8s.pod.uid"] = self.context.pod_uid
        
        if self.context.node_name:
            attributes["k8s.node.name"] = self.context.node_name
        if self.context.node_ip:
            attributes["k8s.node.ip"] = self.context.node_ip
        
        if self.context.container_name:
            attributes["k8s.container.name"] = self.context.container_name
        if self.context.container_id:
            attributes["container.id"] = self.context.container_id
        
        if self.context.cluster_name:
            attributes["k8s.cluster.name"] = self.context.cluster_name
        
        # GenOps-specific attributes
        attributes["genops.runtime"] = "kubernetes"
        attributes["genops.k8s.detected"] = str(self.context.is_running_in_kubernetes)
        
        if self.context.has_service_account:
            attributes["genops.k8s.service_account"] = "true"
        
        # Add resource context for multi-tenancy
        if self.context.pod_namespace:
            attributes["genops.tenant"] = self.context.pod_namespace
        
        return attributes
    
    def get_resource_context(self) -> Dict[str, str]:
        """Get resource context for OpenTelemetry resource attributes."""
        
        resource_attrs = {}
        
        # Standard OpenTelemetry resource attributes
        if self.context.pod_name:
            resource_attrs["k8s.pod.name"] = self.context.pod_name
        if self.context.pod_namespace:
            resource_attrs["k8s.namespace.name"] = self.context.pod_namespace
        if self.context.pod_uid:
            resource_attrs["k8s.pod.uid"] = self.context.pod_uid
        
        if self.context.node_name:
            resource_attrs["k8s.node.name"] = self.context.node_name
        
        if self.context.cluster_name:
            resource_attrs["k8s.cluster.name"] = self.context.cluster_name
        
        if self.context.container_name:
            resource_attrs["k8s.container.name"] = self.context.container_name
        
        return resource_attrs
    
    def is_kubernetes(self) -> bool:
        """Check if running in Kubernetes environment."""
        return self.context.is_running_in_kubernetes
    
    def get_namespace(self) -> Optional[str]:
        """Get current Kubernetes namespace."""
        return self.context.pod_namespace
    
    def get_pod_name(self) -> Optional[str]:
        """Get current pod name."""
        return self.context.pod_name
    
    def get_node_name(self) -> Optional[str]:
        """Get current node name."""
        return self.context.node_name