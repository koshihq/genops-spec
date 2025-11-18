"""Test suite for Dust pricing engine."""

import pytest
from genops.providers.dust_pricing import (
    DustPricingEngine,
    DustPricing,
    DustCostBreakdown,
    calculate_dust_cost,
    get_dust_pricing_info
)


class TestDustPricing:
    """Test cases for DustPricing dataclass."""

    def test_dust_pricing_creation(self):
        """Test DustPricing dataclass creation."""
        pricing = DustPricing(
            pro_monthly_per_user=29.0,
            enterprise_monthly_per_user=None,
            currency="EUR",
            billing_model="per_user"
        )
        
        assert pricing.pro_monthly_per_user == 29.0
        assert pricing.enterprise_monthly_per_user is None
        assert pricing.currency == "EUR"
        assert pricing.billing_model == "per_user"

    def test_dust_pricing_defaults(self):
        """Test DustPricing default values."""
        pricing = DustPricing(pro_monthly_per_user=29.0, enterprise_monthly_per_user=None)
        
        assert pricing.currency == "EUR"
        assert pricing.billing_model == "per_user"


class TestDustCostBreakdown:
    """Test cases for DustCostBreakdown dataclass."""

    def test_cost_breakdown_creation(self):
        """Test DustCostBreakdown creation with all fields."""
        breakdown = DustCostBreakdown(
            operation_type="conversation",
            operation_count=10,
            estimated_tokens=5000,
            user_count=5,
            monthly_subscription_cost=145.0,
            estimated_api_cost=2.5,
            total_cost=147.5,
            currency="EUR",
            billing_period="monthly"
        )
        
        assert breakdown.operation_type == "conversation"
        assert breakdown.operation_count == 10
        assert breakdown.estimated_tokens == 5000
        assert breakdown.user_count == 5
        assert breakdown.monthly_subscription_cost == 145.0
        assert breakdown.estimated_api_cost == 2.5
        assert breakdown.total_cost == 147.5
        assert breakdown.currency == "EUR"
        assert breakdown.billing_period == "monthly"

    def test_cost_breakdown_defaults(self):
        """Test DustCostBreakdown default values."""
        breakdown = DustCostBreakdown(
            operation_type="message",
            operation_count=5,
            estimated_tokens=1000,
            user_count=2,
            monthly_subscription_cost=58.0
        )
        
        assert breakdown.estimated_api_cost == 0.0
        assert breakdown.total_cost == 0.0
        assert breakdown.currency == "EUR"
        assert breakdown.billing_period == "monthly"


class TestDustPricingEngine:
    """Test cases for DustPricingEngine."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = DustPricingEngine()

    def test_engine_initialization(self):
        """Test pricing engine initialization."""
        engine = DustPricingEngine()
        
        assert engine.pricing is not None
        assert engine.pricing.pro_monthly_per_user == 29.0
        assert engine.pricing.enterprise_monthly_per_user is None
        assert engine.pricing.currency == "EUR"
        assert engine.pricing.billing_model == "per_user"

    def test_calculate_subscription_cost_pro_monthly(self):
        """Test Pro plan monthly subscription cost calculation."""
        engine = DustPricingEngine()
        
        cost = engine.calculate_subscription_cost(
            user_count=5,
            plan_type="pro",
            billing_period="monthly"
        )
        
        assert cost == 145.0  # 5 users * €29/user/month

    def test_calculate_subscription_cost_pro_annual(self):
        """Test Pro plan annual subscription cost calculation with discount."""
        engine = DustPricingEngine()
        
        cost = engine.calculate_subscription_cost(
            user_count=3,
            plan_type="pro",
            billing_period="annual"
        )
        
        # 3 users * €29/month * 12 months * 0.9 (10% annual discount)
        expected = 3 * 29 * 12 * 0.9
        assert cost == expected

    def test_calculate_subscription_cost_enterprise(self):
        """Test Enterprise plan subscription cost (custom pricing)."""
        engine = DustPricingEngine()
        
        cost = engine.calculate_subscription_cost(
            user_count=50,
            plan_type="enterprise"
        )
        
        assert cost == 0.0  # Enterprise pricing is custom

    def test_calculate_subscription_cost_invalid_plan(self):
        """Test subscription cost calculation with invalid plan type."""
        engine = DustPricingEngine()
        
        with pytest.raises(ValueError, match="Unknown plan type"):
            engine.calculate_subscription_cost(
                user_count=5,
                plan_type="invalid_plan"
            )

    def test_calculate_operation_cost_pro_plan(self):
        """Test operation cost calculation for Pro plan."""
        engine = DustPricingEngine()
        
        breakdown = engine.calculate_operation_cost(
            operation_type="conversation",
            operation_count=10,
            estimated_tokens=5000,
            user_count=3,
            plan_type="pro"
        )
        
        assert breakdown.operation_type == "conversation"
        assert breakdown.operation_count == 10
        assert breakdown.estimated_tokens == 5000
        assert breakdown.user_count == 3
        assert breakdown.monthly_subscription_cost == 87.0  # 3 * €29
        assert breakdown.estimated_api_cost == 0.0  # Pro plan includes API usage
        assert breakdown.total_cost == 87.0
        assert breakdown.currency == "EUR"

    def test_calculate_operation_cost_enterprise_plan(self):
        """Test operation cost calculation for Enterprise plan."""
        engine = DustPricingEngine()
        
        breakdown = engine.calculate_operation_cost(
            operation_type="agent_execution",
            operation_count=50,
            estimated_tokens=25000,
            user_count=20,
            plan_type="enterprise"
        )
        
        assert breakdown.operation_type == "agent_execution"
        assert breakdown.monthly_subscription_cost == 0.0  # Enterprise custom pricing
        assert breakdown.estimated_api_cost > 0.0  # Should have API costs
        assert breakdown.total_cost > 0.0

    def test_estimate_enterprise_api_cost(self):
        """Test enterprise API cost estimation."""
        engine = DustPricingEngine()
        
        # Test conversation operations
        cost = engine._estimate_enterprise_api_cost("conversation", 10, 1000)
        assert cost > 0
        
        # Test agent execution operations
        cost = engine._estimate_enterprise_api_cost("agent_execution", 5, 5000)
        assert cost > 0
        
        # Test unknown operation type
        cost = engine._estimate_enterprise_api_cost("unknown", 1, 100)
        assert cost > 0  # Should use default rate

    def test_get_cost_optimization_insights_low_utilization(self):
        """Test cost optimization insights for low user utilization."""
        engine = DustPricingEngine()
        
        usage_stats = {
            "active_users": 3,
            "total_users": 10,
            "total_operations": 1000,
            "conversations": 400,
            "agent_runs": 300,
            "searches": 300
        }
        
        insights = engine.get_cost_optimization_insights(usage_stats)
        
        assert "user_optimization" in insights
        assert "30.0%" in insights["user_optimization"]  # 3/10 = 30%
        assert "Low user utilization" in insights["user_optimization"]

    def test_get_cost_optimization_insights_high_utilization(self):
        """Test cost optimization insights for high user utilization."""
        engine = DustPricingEngine()
        
        usage_stats = {
            "active_users": 95,
            "total_users": 100,
            "total_operations": 10000,
            "conversations": 3000,
            "agent_runs": 4000,
            "searches": 3000
        }
        
        insights = engine.get_cost_optimization_insights(usage_stats)
        
        assert "user_optimization" in insights
        assert "95.0%" in insights["user_optimization"]  # 95/100 = 95%
        assert "Well-optimized" in insights["user_optimization"]

    def test_get_cost_optimization_insights_heavy_agent_usage(self):
        """Test cost optimization insights for heavy agent usage."""
        engine = DustPricingEngine()
        
        usage_stats = {
            "active_users": 10,
            "total_users": 10,
            "total_operations": 1000,
            "conversations": 100,
            "agent_runs": 800,  # 80% of operations are agent runs
            "searches": 100
        }
        
        insights = engine.get_cost_optimization_insights(usage_stats)
        
        assert "usage_pattern" in insights
        assert "Heavy agent usage" in insights["usage_pattern"]

    def test_get_cost_optimization_insights_high_search_volume(self):
        """Test cost optimization insights for high search volume."""
        engine = DustPricingEngine()
        
        usage_stats = {
            "active_users": 5,
            "total_users": 5,
            "total_operations": 1000,
            "conversations": 200,
            "agent_runs": 200,
            "searches": 600  # 60% of operations are searches
        }
        
        insights = engine.get_cost_optimization_insights(usage_stats)
        
        assert "search_optimization" in insights
        assert "High search volume" in insights["search_optimization"]

    def test_get_cost_optimization_insights_enterprise_recommendation(self):
        """Test cost optimization insights for enterprise plan recommendation."""
        engine = DustPricingEngine()
        
        usage_stats = {
            "active_users": 75,
            "total_users": 75,  # Large team
            "total_operations": 50000,
            "conversations": 15000,
            "agent_runs": 20000,
            "searches": 15000
        }
        
        insights = engine.get_cost_optimization_insights(usage_stats)
        
        assert "plan_recommendation" in insights
        assert "Enterprise plan" in insights["plan_recommendation"]
        assert "50 users" in insights["plan_recommendation"]

    def test_estimate_monthly_cost_pro_plan(self):
        """Test monthly cost estimation for Pro plan."""
        engine = DustPricingEngine()
        
        usage_forecast = {
            "conversations": 100,
            "agent_runs": 200,
            "searches": 150
        }
        
        estimate = engine.estimate_monthly_cost(
            user_count=5,
            usage_forecast=usage_forecast,
            plan_type="pro"
        )
        
        assert estimate["user_count"] == 5
        assert estimate["plan_type"] == "pro"
        assert estimate["base_subscription"] == 145.0  # 5 * €29
        assert estimate["api_costs"] == 0.0  # Pro plan includes API usage
        assert estimate["total_monthly_cost"] == 145.0
        assert estimate["currency"] == "EUR"
        assert estimate["cost_per_user"] == 29.0  # 145/5
        assert "operation_breakdown" in estimate

    def test_estimate_monthly_cost_enterprise_plan(self):
        """Test monthly cost estimation for Enterprise plan."""
        engine = DustPricingEngine()
        
        usage_forecast = {
            "conversations": 500,
            "agent_runs": 1000,
            "searches": 750,
            "datasource_creation": 10
        }
        
        estimate = engine.estimate_monthly_cost(
            user_count=25,
            usage_forecast=usage_forecast,
            plan_type="enterprise"
        )
        
        assert estimate["user_count"] == 25
        assert estimate["plan_type"] == "enterprise"
        assert estimate["base_subscription"] == 0.0  # Enterprise custom pricing
        assert estimate["api_costs"] > 0.0  # Should have API costs
        assert estimate["total_monthly_cost"] > 0.0
        assert "operation_breakdown" in estimate
        assert len(estimate["operation_breakdown"]) == 4


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    def test_calculate_dust_cost(self):
        """Test calculate_dust_cost convenience function."""
        cost = calculate_dust_cost(
            operation_type="message",
            operation_count=25,
            estimated_tokens=12500,
            user_count=4,
            plan_type="pro"
        )
        
        assert isinstance(cost, DustCostBreakdown)
        assert cost.operation_type == "message"
        assert cost.operation_count == 25
        assert cost.estimated_tokens == 12500
        assert cost.user_count == 4
        assert cost.monthly_subscription_cost == 116.0  # 4 * €29

    def test_get_dust_pricing_info(self):
        """Test get_dust_pricing_info convenience function."""
        pricing = get_dust_pricing_info()
        
        assert isinstance(pricing, DustPricing)
        assert pricing.pro_monthly_per_user == 29.0
        assert pricing.currency == "EUR"
        assert pricing.billing_model == "per_user"

    def test_calculate_dust_cost_with_kwargs(self):
        """Test calculate_dust_cost with additional kwargs."""
        cost = calculate_dust_cost(
            operation_type="datasource_search",
            operation_count=100,
            user_count=2,
            billing_period="annual",  # Additional kwarg
            custom_param="test"  # Additional kwarg
        )
        
        assert cost.operation_type == "datasource_search"
        assert cost.user_count == 2


class TestEdgeCases:
    """Test cases for edge cases and error conditions."""

    def test_zero_users(self):
        """Test cost calculation with zero users."""
        engine = DustPricingEngine()
        
        cost = engine.calculate_subscription_cost(
            user_count=0,
            plan_type="pro"
        )
        
        assert cost == 0.0

    def test_zero_operations(self):
        """Test cost calculation with zero operations."""
        engine = DustPricingEngine()
        
        breakdown = engine.calculate_operation_cost(
            operation_type="conversation",
            operation_count=0,
            user_count=1
        )
        
        assert breakdown.operation_count == 0
        assert breakdown.monthly_subscription_cost == 29.0  # Still pay for user

    def test_negative_values_handling(self):
        """Test handling of negative input values."""
        engine = DustPricingEngine()
        
        # Negative user count should not cause errors
        cost = engine.calculate_subscription_cost(
            user_count=-1,
            plan_type="pro"
        )
        
        assert cost == -29.0  # Calculation still works but gives negative result

    def test_empty_usage_stats(self):
        """Test optimization insights with empty usage stats."""
        engine = DustPricingEngine()
        
        insights = engine.get_cost_optimization_insights({})
        
        # Should not crash and should return some insights
        assert isinstance(insights, dict)

    def test_empty_usage_forecast(self):
        """Test monthly cost estimation with empty usage forecast."""
        engine = DustPricingEngine()
        
        estimate = engine.estimate_monthly_cost(
            user_count=3,
            usage_forecast={},
            plan_type="pro"
        )
        
        assert estimate["user_count"] == 3
        assert estimate["base_subscription"] == 87.0  # 3 * €29
        assert estimate["api_costs"] == 0.0
        assert estimate["total_monthly_cost"] == 87.0
        assert estimate["operation_breakdown"] == {}

    def test_case_insensitive_plan_types(self):
        """Test case insensitive plan type handling."""
        engine = DustPricingEngine()
        
        cost_upper = engine.calculate_subscription_cost(
            user_count=2,
            plan_type="PRO"
        )
        
        cost_mixed = engine.calculate_subscription_cost(
            user_count=2,
            plan_type="Pro"
        )
        
        cost_lower = engine.calculate_subscription_cost(
            user_count=2,
            plan_type="pro"
        )
        
        assert cost_upper == cost_mixed == cost_lower == 58.0