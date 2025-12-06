#!/usr/bin/env python3
"""
Raindrop AI Cost Calculation Engine

This module provides comprehensive cost calculation for Raindrop AI operations
with GenOps governance. It handles pricing for agent interactions, performance
signals, alerts, and dashboard analytics with accurate cost modeling.

Features:
- Agent interaction cost calculation with variable pricing tiers
- Performance signal monitoring costs with complexity-based pricing
- Alert creation and management cost modeling
- Dashboard analytics and deep search operation costs
- Volume discount calculation and optimization
- Multi-currency support with automatic conversion
- Custom pricing model support for enterprise deployments

Author: GenOps AI Contributors
License: Apache 2.0
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class RaindropOperationType(Enum):
    """Types of Raindrop AI operations with cost implications."""
    AGENT_INTERACTION = "agent_interaction"
    PERFORMANCE_SIGNAL = "performance_signal"
    ALERT_CREATION = "alert_creation"
    ALERT_MANAGEMENT = "alert_management"
    DEEP_SEARCH = "deep_search"
    EXPERIMENT = "experiment"
    DASHBOARD_ANALYTICS = "dashboard_analytics"

@dataclass
class RaindropCostResult:
    """Result of a Raindrop AI cost calculation."""
    operation_type: str
    base_cost: Decimal
    volume_discount: Decimal = field(default_factory=lambda: Decimal('0.00'))
    total_cost: Decimal = field(default_factory=lambda: Decimal('0.00'))
    currency: str = "USD"
    
    # Operation-specific details
    agent_id: Optional[str] = None
    signal_name: Optional[str] = None
    alert_name: Optional[str] = None
    search_query: Optional[str] = None
    experiment_name: Optional[str] = None
    
    # Pricing details
    unit_count: int = 1
    unit_price: Decimal = field(default_factory=lambda: Decimal('0.00'))
    pricing_tier: str = "standard"
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    calculation_notes: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate total cost after initialization."""
        if self.total_cost == Decimal('0.00'):
            self.total_cost = self.base_cost - self.volume_discount

@dataclass
class RaindropPricingConfig:
    """Configuration for Raindrop AI pricing calculations."""
    
    # Base costs per operation (USD)
    agent_interaction_base_cost: Decimal = field(default_factory=lambda: Decimal('0.001'))
    performance_signal_base_cost: Decimal = field(default_factory=lambda: Decimal('0.01'))
    alert_creation_cost: Decimal = field(default_factory=lambda: Decimal('0.05'))
    alert_management_daily_cost: Decimal = field(default_factory=lambda: Decimal('0.10'))
    deep_search_base_cost: Decimal = field(default_factory=lambda: Decimal('0.02'))
    experiment_base_cost: Decimal = field(default_factory=lambda: Decimal('0.15'))
    dashboard_analytics_daily_cost: Decimal = field(default_factory=lambda: Decimal('0.10'))
    
    # Volume discount tiers (monthly interaction thresholds)
    volume_tiers: Dict[int, Decimal] = field(default_factory=lambda: {
        1000: Decimal('0.05'),      # 5% discount for 1K+ interactions
        10000: Decimal('0.10'),     # 10% discount for 10K+ interactions
        100000: Decimal('0.15'),    # 15% discount for 100K+ interactions
        1000000: Decimal('0.25')    # 25% discount for 1M+ interactions
    })
    
    # Performance signal complexity multipliers
    signal_complexity_multipliers: Dict[str, Decimal] = field(default_factory=lambda: {
        'simple': Decimal('1.0'),    # Basic metrics (latency, error rate)
        'moderate': Decimal('1.5'),  # Advanced metrics (accuracy, F1-score)
        'complex': Decimal('2.0'),   # Custom metrics with ML evaluation
        'enterprise': Decimal('3.0') # Complex evaluation with custom models
    })
    
    # Alert complexity multipliers
    alert_complexity_multipliers: Dict[str, Decimal] = field(default_factory=lambda: {
        'simple': Decimal('1.0'),     # Basic threshold alerts
        'moderate': Decimal('1.5'),   # Multi-condition alerts
        'complex': Decimal('2.5'),    # ML-based anomaly detection
        'enterprise': Decimal('4.0')  # Custom alert logic with integrations
    })
    
    # Search operation multipliers
    search_complexity_multipliers: Dict[str, Decimal] = field(default_factory=lambda: {
        'basic': Decimal('1.0'),      # Simple text search
        'advanced': Decimal('2.0'),   # Semantic search with filters
        'analytical': Decimal('3.0'), # Complex analytical queries
        'enterprise': Decimal('5.0')  # AI-powered insights and recommendations
    })
    
    # Currency conversion rates (from USD)
    currency_rates: Dict[str, Decimal] = field(default_factory=lambda: {
        'USD': Decimal('1.00'),
        'EUR': Decimal('0.85'),
        'GBP': Decimal('0.75'),
        'CAD': Decimal('1.25'),
        'AUD': Decimal('1.35')
    })

class RaindropPricingCalculator:
    """
    Comprehensive cost calculation engine for Raindrop AI operations.
    
    Provides accurate cost modeling for all Raindrop AI features with
    support for volume discounts, complexity-based pricing, and enterprise customizations.
    """
    
    def __init__(self, pricing_config: Optional[RaindropPricingConfig] = None):
        """
        Initialize the pricing calculator.
        
        Args:
            pricing_config: Custom pricing configuration (uses defaults if None)
        """
        self.config = pricing_config or RaindropPricingConfig()
        self.monthly_interaction_count = 0  # Track for volume discounts
        
    def calculate_interaction_cost(
        self,
        agent_id: str,
        interaction_data: Dict[str, Any],
        complexity: str = "simple",
        currency: str = "USD"
    ) -> RaindropCostResult:
        """
        Calculate cost for an agent interaction.
        
        Args:
            agent_id: Identifier for the agent
            interaction_data: Interaction data including input/output and signals
            complexity: Complexity level (simple, moderate, complex, enterprise)
            currency: Target currency for cost calculation
            
        Returns:
            RaindropCostResult: Detailed cost calculation result
        """
        # Determine complexity multiplier
        complexity_multiplier = self.config.signal_complexity_multipliers.get(
            complexity, Decimal('1.0')
        )
        
        # Calculate base cost
        base_cost = self.config.agent_interaction_base_cost * complexity_multiplier
        
        # Apply data size multiplier based on interaction data size
        data_size_multiplier = self._calculate_data_size_multiplier(interaction_data)
        base_cost *= data_size_multiplier
        
        # Calculate volume discount
        volume_discount = self._calculate_volume_discount(base_cost)
        
        # Convert currency if needed
        final_cost = self._convert_currency(base_cost - volume_discount, currency)
        
        notes = [
            f"Complexity: {complexity} (multiplier: {complexity_multiplier})",
            f"Data size multiplier: {data_size_multiplier}",
            f"Volume discount: ${volume_discount:.4f}"
        ]
        
        return RaindropCostResult(
            operation_type=RaindropOperationType.AGENT_INTERACTION.value,
            base_cost=base_cost,
            volume_discount=volume_discount,
            total_cost=final_cost,
            currency=currency,
            agent_id=agent_id,
            unit_price=self.config.agent_interaction_base_cost,
            pricing_tier=complexity,
            calculation_notes=notes
        )
    
    def calculate_signal_cost(
        self,
        signal_name: str,
        signal_data: Dict[str, Any],
        complexity: str = "simple",
        currency: str = "USD"
    ) -> RaindropCostResult:
        """
        Calculate cost for performance signal monitoring.
        
        Args:
            signal_name: Name of the performance signal
            signal_data: Signal configuration and evaluation data
            complexity: Signal complexity level
            currency: Target currency for cost calculation
            
        Returns:
            RaindropCostResult: Detailed cost calculation result
        """
        # Get complexity multiplier
        complexity_multiplier = self.config.signal_complexity_multipliers.get(
            complexity, Decimal('1.0')
        )
        
        # Calculate base cost
        base_cost = self.config.performance_signal_base_cost * complexity_multiplier
        
        # Apply signal frequency multiplier
        frequency_multiplier = self._calculate_signal_frequency_multiplier(signal_data)
        base_cost *= frequency_multiplier
        
        # Calculate volume discount
        volume_discount = self._calculate_volume_discount(base_cost)
        
        # Convert currency
        final_cost = self._convert_currency(base_cost - volume_discount, currency)
        
        notes = [
            f"Signal complexity: {complexity} (multiplier: {complexity_multiplier})",
            f"Frequency multiplier: {frequency_multiplier}",
            f"Volume discount: ${volume_discount:.4f}"
        ]
        
        return RaindropCostResult(
            operation_type=RaindropOperationType.PERFORMANCE_SIGNAL.value,
            base_cost=base_cost,
            volume_discount=volume_discount,
            total_cost=final_cost,
            currency=currency,
            signal_name=signal_name,
            unit_price=self.config.performance_signal_base_cost,
            pricing_tier=complexity,
            calculation_notes=notes
        )
    
    def calculate_alert_cost(
        self,
        alert_name: str,
        alert_config: Dict[str, Any],
        complexity: str = "simple",
        currency: str = "USD"
    ) -> RaindropCostResult:
        """
        Calculate cost for alert creation and management.
        
        Args:
            alert_name: Name of the alert
            alert_config: Alert configuration including conditions and actions
            complexity: Alert complexity level
            currency: Target currency for cost calculation
            
        Returns:
            RaindropCostResult: Detailed cost calculation result
        """
        # Get complexity multiplier
        complexity_multiplier = self.config.alert_complexity_multipliers.get(
            complexity, Decimal('1.0')
        )
        
        # Calculate base cost (creation + daily management)
        creation_cost = self.config.alert_creation_cost * complexity_multiplier
        daily_management_cost = self.config.alert_management_daily_cost * complexity_multiplier
        base_cost = creation_cost + daily_management_cost
        
        # Apply notification multiplier based on alert configuration
        notification_multiplier = self._calculate_notification_multiplier(alert_config)
        base_cost *= notification_multiplier
        
        # Calculate volume discount
        volume_discount = self._calculate_volume_discount(base_cost)
        
        # Convert currency
        final_cost = self._convert_currency(base_cost - volume_discount, currency)
        
        notes = [
            f"Alert complexity: {complexity} (multiplier: {complexity_multiplier})",
            f"Notification multiplier: {notification_multiplier}",
            f"Creation cost: ${creation_cost:.4f}, Daily management: ${daily_management_cost:.4f}",
            f"Volume discount: ${volume_discount:.4f}"
        ]
        
        return RaindropCostResult(
            operation_type=RaindropOperationType.ALERT_CREATION.value,
            base_cost=base_cost,
            volume_discount=volume_discount,
            total_cost=final_cost,
            currency=currency,
            alert_name=alert_name,
            unit_price=self.config.alert_creation_cost,
            pricing_tier=complexity,
            calculation_notes=notes
        )
    
    def calculate_search_cost(
        self,
        search_query: str,
        search_config: Dict[str, Any],
        complexity: str = "basic",
        currency: str = "USD"
    ) -> RaindropCostResult:
        """
        Calculate cost for deep search operations.
        
        Args:
            search_query: The search query string
            search_config: Search configuration including filters and scope
            complexity: Search complexity level
            currency: Target currency for cost calculation
            
        Returns:
            RaindropCostResult: Detailed cost calculation result
        """
        # Get complexity multiplier
        complexity_multiplier = self.config.search_complexity_multipliers.get(
            complexity, Decimal('1.0')
        )
        
        # Calculate base cost
        base_cost = self.config.deep_search_base_cost * complexity_multiplier
        
        # Apply search scope multiplier
        scope_multiplier = self._calculate_search_scope_multiplier(search_config)
        base_cost *= scope_multiplier
        
        # Calculate volume discount
        volume_discount = self._calculate_volume_discount(base_cost)
        
        # Convert currency
        final_cost = self._convert_currency(base_cost - volume_discount, currency)
        
        notes = [
            f"Search complexity: {complexity} (multiplier: {complexity_multiplier})",
            f"Scope multiplier: {scope_multiplier}",
            f"Query length: {len(search_query)} characters",
            f"Volume discount: ${volume_discount:.4f}"
        ]
        
        return RaindropCostResult(
            operation_type=RaindropOperationType.DEEP_SEARCH.value,
            base_cost=base_cost,
            volume_discount=volume_discount,
            total_cost=final_cost,
            currency=currency,
            search_query=search_query,
            unit_price=self.config.deep_search_base_cost,
            pricing_tier=complexity,
            calculation_notes=notes
        )
    
    def calculate_experiment_cost(
        self,
        experiment_name: str,
        experiment_config: Dict[str, Any],
        currency: str = "USD"
    ) -> RaindropCostResult:
        """
        Calculate cost for A/B testing experiments.
        
        Args:
            experiment_name: Name of the experiment
            experiment_config: Experiment configuration
            currency: Target currency for cost calculation
            
        Returns:
            RaindropCostResult: Detailed cost calculation result
        """
        # Base experiment cost
        base_cost = self.config.experiment_base_cost
        
        # Apply duration multiplier based on experiment configuration
        duration_multiplier = self._calculate_experiment_duration_multiplier(experiment_config)
        base_cost *= duration_multiplier
        
        # Apply complexity multiplier based on number of variants and metrics
        complexity_multiplier = self._calculate_experiment_complexity_multiplier(experiment_config)
        base_cost *= complexity_multiplier
        
        # Calculate volume discount
        volume_discount = self._calculate_volume_discount(base_cost)
        
        # Convert currency
        final_cost = self._convert_currency(base_cost - volume_discount, currency)
        
        notes = [
            f"Duration multiplier: {duration_multiplier}",
            f"Complexity multiplier: {complexity_multiplier}",
            f"Volume discount: ${volume_discount:.4f}"
        ]
        
        return RaindropCostResult(
            operation_type=RaindropOperationType.EXPERIMENT.value,
            base_cost=base_cost,
            volume_discount=volume_discount,
            total_cost=final_cost,
            currency=currency,
            experiment_name=experiment_name,
            unit_price=self.config.experiment_base_cost,
            calculation_notes=notes
        )
    
    def _calculate_data_size_multiplier(self, interaction_data: Dict[str, Any]) -> Decimal:
        """Calculate cost multiplier based on interaction data size."""
        try:
            # Estimate data size (simplified calculation)
            data_size = len(str(interaction_data))
            
            if data_size < 1000:  # < 1KB
                return Decimal('1.0')
            elif data_size < 10000:  # < 10KB
                return Decimal('1.2')
            elif data_size < 100000:  # < 100KB
                return Decimal('1.5')
            else:  # >= 100KB
                return Decimal('2.0')
        except Exception:
            return Decimal('1.0')
    
    def _calculate_signal_frequency_multiplier(self, signal_data: Dict[str, Any]) -> Decimal:
        """Calculate cost multiplier based on signal monitoring frequency."""
        frequency = signal_data.get('monitoring_frequency', 'standard')
        
        frequency_multipliers = {
            'low': Decimal('0.8'),       # Weekly or less frequent
            'standard': Decimal('1.0'),  # Daily monitoring
            'high': Decimal('1.5'),      # Hourly monitoring
            'realtime': Decimal('2.5')   # Real-time monitoring
        }
        
        return frequency_multipliers.get(frequency, Decimal('1.0'))
    
    def _calculate_notification_multiplier(self, alert_config: Dict[str, Any]) -> Decimal:
        """Calculate cost multiplier based on notification configuration."""
        notification_count = len(alert_config.get('notification_channels', []))
        
        if notification_count == 0:
            return Decimal('1.0')
        elif notification_count <= 2:
            return Decimal('1.2')
        elif notification_count <= 5:
            return Decimal('1.5')
        else:
            return Decimal('2.0')
    
    def _calculate_search_scope_multiplier(self, search_config: Dict[str, Any]) -> Decimal:
        """Calculate cost multiplier based on search scope."""
        scope = search_config.get('scope', 'single_agent')
        
        scope_multipliers = {
            'single_agent': Decimal('1.0'),
            'agent_group': Decimal('1.5'),
            'project': Decimal('2.0'),
            'organization': Decimal('3.0')
        }
        
        return scope_multipliers.get(scope, Decimal('1.0'))
    
    def _calculate_experiment_duration_multiplier(self, experiment_config: Dict[str, Any]) -> Decimal:
        """Calculate cost multiplier based on experiment duration."""
        duration_days = experiment_config.get('duration_days', 7)
        
        if duration_days <= 3:
            return Decimal('1.0')
        elif duration_days <= 7:
            return Decimal('1.2')
        elif duration_days <= 30:
            return Decimal('1.5')
        else:
            return Decimal('2.0')
    
    def _calculate_experiment_complexity_multiplier(self, experiment_config: Dict[str, Any]) -> Decimal:
        """Calculate cost multiplier based on experiment complexity."""
        variant_count = len(experiment_config.get('variants', []))
        metric_count = len(experiment_config.get('metrics', []))
        
        complexity_score = variant_count + metric_count
        
        if complexity_score <= 3:
            return Decimal('1.0')
        elif complexity_score <= 6:
            return Decimal('1.3')
        elif complexity_score <= 10:
            return Decimal('1.7')
        else:
            return Decimal('2.5')
    
    def _calculate_volume_discount(self, base_cost: Decimal) -> Decimal:
        """Calculate volume discount based on monthly usage."""
        if self.monthly_interaction_count == 0:
            return Decimal('0.00')
        
        # Find the highest applicable discount tier
        applicable_discount = Decimal('0.00')
        for threshold, discount_rate in sorted(self.config.volume_tiers.items()):
            if self.monthly_interaction_count >= threshold:
                applicable_discount = discount_rate
            else:
                break
        
        return base_cost * applicable_discount
    
    def _convert_currency(self, amount_usd: Decimal, target_currency: str) -> Decimal:
        """Convert USD amount to target currency."""
        if target_currency == 'USD':
            return amount_usd
        
        rate = self.config.currency_rates.get(target_currency, Decimal('1.0'))
        converted = amount_usd * rate
        
        # Round to 4 decimal places
        return converted.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
    
    def update_monthly_volume(self, interaction_count: int) -> None:
        """Update the monthly interaction count for volume discount calculation."""
        self.monthly_interaction_count = interaction_count
    
    def get_volume_discount_info(self) -> Dict[str, Any]:
        """Get current volume discount information."""
        current_discount_rate = Decimal('0.00')
        next_tier_threshold = None
        next_tier_discount = None
        
        # Find current discount tier
        for threshold, discount_rate in sorted(self.config.volume_tiers.items()):
            if self.monthly_interaction_count >= threshold:
                current_discount_rate = discount_rate
            elif next_tier_threshold is None:
                next_tier_threshold = threshold
                next_tier_discount = discount_rate
                break
        
        return {
            'current_monthly_interactions': self.monthly_interaction_count,
            'current_discount_rate': float(current_discount_rate),
            'current_discount_percentage': float(current_discount_rate * 100),
            'next_tier_threshold': next_tier_threshold,
            'next_tier_discount_rate': float(next_tier_discount) if next_tier_discount else None,
            'next_tier_discount_percentage': float(next_tier_discount * 100) if next_tier_discount else None
        }

# Export main classes
__all__ = [
    'RaindropPricingCalculator',
    'RaindropCostResult',
    'RaindropPricingConfig',
    'RaindropOperationType'
]