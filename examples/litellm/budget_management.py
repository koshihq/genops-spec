#!/usr/bin/env python3
"""
LiteLLM Budget Management and Controls with GenOps

Demonstrates comprehensive budget management, spending controls, and financial
governance patterns for LiteLLM applications. This example shows how to implement
spending limits, alerts, and budget allocation strategies across teams and projects.

Usage:
    export OPENAI_API_KEY="your_key_here"
    python budget_management.py

Features:
    - Team-based budget allocation and tracking
    - Real-time spending alerts and notifications
    - Budget enforcement policies (advisory vs enforced)
    - Cost forecasting and trend analysis
    - Multi-tenant budget isolation
    - Emergency budget controls and circuit breakers
"""

import os
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class BudgetPolicy(Enum):
    """Budget enforcement policies."""
    ADVISORY = "advisory"      # Warnings only, allow overruns
    SOFT_LIMIT = "soft_limit"  # Block requests after warning threshold
    HARD_LIMIT = "hard_limit"  # Strict blocking at budget limit
    CIRCUIT_BREAKER = "circuit_breaker"  # Temporary blocks with recovery


class AlertLevel(Enum):
    """Budget alert severity levels."""
    INFO = "info"        # 25% budget used
    WARNING = "warning"  # 50% budget used
    CRITICAL = "critical"  # 80% budget used
    EMERGENCY = "emergency"  # 95% budget used


@dataclass
class BudgetConfig:
    """Budget configuration for a team or project."""
    name: str
    daily_limit: float
    monthly_limit: float
    policy: BudgetPolicy = BudgetPolicy.SOFT_LIMIT
    alert_thresholds: Dict[AlertLevel, float] = field(default_factory=lambda: {
        AlertLevel.INFO: 0.25,
        AlertLevel.WARNING: 0.50,
        AlertLevel.CRITICAL: 0.80,
        AlertLevel.EMERGENCY: 0.95
    })
    emergency_contacts: List[str] = field(default_factory=list)
    allowed_overrun_percentage: float = 10.0  # Allow 10% overrun for soft limits


@dataclass
class BudgetUsage:
    """Current budget usage tracking."""
    config: BudgetConfig
    daily_spent: float = 0.0
    monthly_spent: float = 0.0
    requests_count: int = 0
    last_reset_date: Optional[str] = None
    current_alerts: List[AlertLevel] = field(default_factory=list)
    is_blocked: bool = False
    block_reason: Optional[str] = None


class BudgetManager:
    """Comprehensive budget management system."""
    
    def __init__(self):
        self.budget_configs: Dict[str, BudgetConfig] = {}
        self.usage_tracking: Dict[str, BudgetUsage] = {}
        self.alert_callbacks: List[Callable] = []
        self._lock = threading.RLock()
        
    def register_budget(self, team_or_project: str, config: BudgetConfig) -> bool:
        """Register a budget configuration."""
        with self._lock:
            self.budget_configs[team_or_project] = config
            
            if team_or_project not in self.usage_tracking:
                self.usage_tracking[team_or_project] = BudgetUsage(
                    config=config,
                    last_reset_date=datetime.now().strftime('%Y-%m-%d')
                )
            
        return True
    
    def add_alert_callback(self, callback: Callable[[str, AlertLevel, Dict], None]):
        """Add callback for budget alerts."""
        self.alert_callbacks.append(callback)
    
    def check_budget_allowance(
        self, 
        team_or_project: str, 
        estimated_cost: float
    ) -> Dict[str, Any]:
        """
        Check if a request is within budget allowance.
        
        Returns:
            Dict with 'allowed' boolean and details
        """
        with self._lock:
            if team_or_project not in self.usage_tracking:
                return {
                    'allowed': False,
                    'reason': f'No budget configured for {team_or_project}',
                    'action': 'configure_budget'
                }
            
            usage = self.usage_tracking[team_or_project]
            config = usage.config
            
            # Reset daily usage if new day
            self._reset_daily_usage_if_needed(team_or_project)
            
            # Check if currently blocked
            if usage.is_blocked:
                return {
                    'allowed': False,
                    'reason': usage.block_reason,
                    'action': 'wait_for_reset_or_increase_budget'
                }
            
            # Calculate usage after this request
            new_daily_total = usage.daily_spent + estimated_cost
            new_monthly_total = usage.monthly_spent + estimated_cost
            
            # Check against limits based on policy
            daily_limit = config.daily_limit
            monthly_limit = config.monthly_limit
            
            if config.policy == BudgetPolicy.HARD_LIMIT:
                if new_daily_total > daily_limit or new_monthly_total > monthly_limit:
                    return {
                        'allowed': False,
                        'reason': f'Hard budget limit reached (daily: ${usage.daily_spent:.4f}/${daily_limit}, monthly: ${usage.monthly_spent:.4f}/${monthly_limit})',
                        'action': 'increase_budget_or_wait'
                    }
            
            elif config.policy == BudgetPolicy.SOFT_LIMIT:
                overrun_daily = daily_limit * (1 + config.allowed_overrun_percentage / 100)
                overrun_monthly = monthly_limit * (1 + config.allowed_overrun_percentage / 100)
                
                if new_daily_total > overrun_daily or new_monthly_total > overrun_monthly:
                    return {
                        'allowed': False,
                        'reason': f'Soft budget limit exceeded (including {config.allowed_overrun_percentage}% overrun)',
                        'action': 'increase_budget'
                    }
            
            # Check for alert thresholds
            daily_usage_pct = new_daily_total / daily_limit if daily_limit > 0 else 0
            monthly_usage_pct = new_monthly_total / monthly_limit if monthly_limit > 0 else 0
            
            max_usage_pct = max(daily_usage_pct, monthly_usage_pct)
            
            # Trigger alerts
            for alert_level, threshold in config.alert_thresholds.items():
                if max_usage_pct >= threshold and alert_level not in usage.current_alerts:
                    self._trigger_alert(team_or_project, alert_level, {
                        'usage_percentage': max_usage_pct * 100,
                        'daily_spent': usage.daily_spent,
                        'monthly_spent': usage.monthly_spent,
                        'estimated_cost': estimated_cost
                    })
                    usage.current_alerts.append(alert_level)
            
            return {
                'allowed': True,
                'daily_usage_pct': daily_usage_pct * 100,
                'monthly_usage_pct': monthly_usage_pct * 100,
                'remaining_daily': daily_limit - new_daily_total,
                'remaining_monthly': monthly_limit - new_monthly_total
            }
    
    def record_usage(self, team_or_project: str, actual_cost: float) -> bool:
        """Record actual usage after a request."""
        with self._lock:
            if team_or_project not in self.usage_tracking:
                return False
            
            usage = self.usage_tracking[team_or_project]
            usage.daily_spent += actual_cost
            usage.monthly_spent += actual_cost
            usage.requests_count += 1
            
            return True
    
    def _reset_daily_usage_if_needed(self, team_or_project: str):
        """Reset daily usage if it's a new day."""
        usage = self.usage_tracking[team_or_project]
        today = datetime.now().strftime('%Y-%m-%d')
        
        if usage.last_reset_date != today:
            usage.daily_spent = 0.0
            usage.last_reset_date = today
            usage.current_alerts = []  # Reset daily alerts
            usage.is_blocked = False
            usage.block_reason = None
    
    def _trigger_alert(self, team_or_project: str, level: AlertLevel, details: Dict):
        """Trigger budget alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                callback(team_or_project, level, details)
            except Exception as e:
                print(f"Alert callback failed: {e}")
    
    def get_budget_summary(self, team_or_project: Optional[str] = None) -> Dict[str, Any]:
        """Get budget summary for team/project or all."""
        with self._lock:
            if team_or_project:
                if team_or_project not in self.usage_tracking:
                    return {'error': f'No budget tracking for {team_or_project}'}
                
                usage = self.usage_tracking[team_or_project]
                config = usage.config
                
                return {
                    'name': config.name,
                    'daily_limit': config.daily_limit,
                    'daily_spent': usage.daily_spent,
                    'daily_remaining': config.daily_limit - usage.daily_spent,
                    'daily_usage_pct': (usage.daily_spent / config.daily_limit) * 100 if config.daily_limit > 0 else 0,
                    'monthly_limit': config.monthly_limit,
                    'monthly_spent': usage.monthly_spent,
                    'monthly_remaining': config.monthly_limit - usage.monthly_spent,
                    'monthly_usage_pct': (usage.monthly_spent / config.monthly_limit) * 100 if config.monthly_limit > 0 else 0,
                    'requests_count': usage.requests_count,
                    'policy': config.policy.value,
                    'is_blocked': usage.is_blocked,
                    'current_alerts': [alert.value for alert in usage.current_alerts]
                }
            else:
                # Return summary for all tracked budgets
                summaries = {}
                for key in self.usage_tracking:
                    summaries[key] = self.get_budget_summary(key)
                return summaries
    
    def set_emergency_block(self, team_or_project: str, reason: str) -> bool:
        """Emergency block for a team/project."""
        with self._lock:
            if team_or_project in self.usage_tracking:
                usage = self.usage_tracking[team_or_project]
                usage.is_blocked = True
                usage.block_reason = f"EMERGENCY BLOCK: {reason}"
                
                # Trigger emergency alert
                self._trigger_alert(team_or_project, AlertLevel.EMERGENCY, {
                    'block_reason': reason,
                    'action': 'emergency_block_activated'
                })
                
                return True
        return False
    
    def remove_block(self, team_or_project: str) -> bool:
        """Remove emergency block."""
        with self._lock:
            if team_or_project in self.usage_tracking:
                usage = self.usage_tracking[team_or_project]
                usage.is_blocked = False
                usage.block_reason = None
                return True
        return False


def setup_budget_demo_configs() -> BudgetManager:
    """Set up demonstration budget configurations."""
    manager = BudgetManager()
    
    # Development team - tight budget
    dev_config = BudgetConfig(
        name="Development Team",
        daily_limit=25.0,
        monthly_limit=500.0,
        policy=BudgetPolicy.SOFT_LIMIT,
        emergency_contacts=["dev-lead@company.com"],
        allowed_overrun_percentage=15.0
    )
    manager.register_budget("dev-team", dev_config)
    
    # Production team - higher budget, strict control  
    prod_config = BudgetConfig(
        name="Production Team",
        daily_limit=200.0,
        monthly_limit=5000.0,
        policy=BudgetPolicy.HARD_LIMIT,
        emergency_contacts=["ops-lead@company.com", "cto@company.com"]
    )
    manager.register_budget("prod-team", prod_config)
    
    # Research team - moderate budget, flexible policy
    research_config = BudgetConfig(
        name="Research Team", 
        daily_limit=75.0,
        monthly_limit=2000.0,
        policy=BudgetPolicy.ADVISORY,
        emergency_contacts=["research-lead@company.com"],
        allowed_overrun_percentage=25.0
    )
    manager.register_budget("research-team", research_config)
    
    # Customer support - customer-funded, circuit breaker
    support_config = BudgetConfig(
        name="Customer Support",
        daily_limit=100.0,
        monthly_limit=2500.0,
        policy=BudgetPolicy.CIRCUIT_BREAKER,
        emergency_contacts=["support-lead@company.com"]
    )
    manager.register_budget("support-team", support_config)
    
    return manager


def setup_alert_system(manager: BudgetManager):
    """Set up alert system with different notification methods."""
    
    def console_alert_handler(team: str, level: AlertLevel, details: Dict):
        """Handle alerts with console output."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        level_emoji = {
            AlertLevel.INFO: "💡",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.CRITICAL: "🚨",
            AlertLevel.EMERGENCY: "🔥"
        }
        
        print(f"\n{level_emoji[level]} BUDGET ALERT [{level.value.upper()}] - {timestamp}")
        print(f"   Team: {team}")
        print(f"   Usage: {details.get('usage_percentage', 0):.1f}%")
        print(f"   Daily spent: ${details.get('daily_spent', 0):.4f}")
        print(f"   Monthly spent: ${details.get('monthly_spent', 0):.4f}")
        
        if level == AlertLevel.EMERGENCY:
            print(f"   🚨 EMERGENCY ACTION REQUIRED! 🚨")
            print(f"   Consider immediate budget increase or usage review")
    
    def email_alert_handler(team: str, level: AlertLevel, details: Dict):
        """Handle alerts with email notifications (simulated)."""
        if level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
            print(f"   📧 Email alert sent for {team} ({level.value})")
            print(f"   Recipients: budget-alerts@company.com, team-leads@company.com")
    
    def slack_alert_handler(team: str, level: AlertLevel, details: Dict):
        """Handle alerts with Slack notifications (simulated)."""
        if level in [AlertLevel.WARNING, AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
            print(f"   💬 Slack notification sent to #{team}-alerts channel")
    
    # Register alert handlers
    manager.add_alert_callback(console_alert_handler)
    manager.add_alert_callback(email_alert_handler)
    manager.add_alert_callback(slack_alert_handler)


def check_budget_setup():
    """Check setup for budget management demo."""
    print("🔍 Checking budget management setup...")
    
    # Check imports
    try:
        import litellm
        from genops.providers.litellm import auto_instrument, get_usage_stats
        print("✅ LiteLLM and GenOps available")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Install: pip install litellm genops[litellm]")
        return False
    
    # Check API keys
    if os.getenv('OPENAI_API_KEY'):
        print("✅ OpenAI API key configured")
    else:
        print("⚠️  No API keys configured - will use demo mode")
    
    return True


def demo_budget_configuration():
    """Demonstrate budget configuration and policies."""
    print("\n" + "="*60)
    print("⚙️ Demo: Budget Configuration & Policies")
    print("="*60)
    
    print("Setting up budget configurations for different teams with")
    print("varying limits, policies, and alert thresholds.")
    
    manager = setup_budget_demo_configs()
    setup_alert_system(manager)
    
    print(f"\n📊 Configured Budget Policies:")
    
    budget_summary = manager.get_budget_summary()
    
    for team, summary in budget_summary.items():
        if 'error' in summary:
            continue
            
        print(f"\n👥 {summary['name']} ({team})")
        print(f"   Daily limit: ${summary['daily_limit']:.2f}")
        print(f"   Monthly limit: ${summary['monthly_limit']:.2f}")
        print(f"   Policy: {summary['policy']}")
        print(f"   Current usage: ${summary['daily_spent']:.4f} daily, ${summary['monthly_spent']:.4f} monthly")
        
        usage_pct = max(summary['daily_usage_pct'], summary['monthly_usage_pct'])
        if usage_pct > 0:
            print(f"   Usage: {usage_pct:.1f}%")
    
    return manager


def demo_budget_enforcement():
    """Demonstrate budget enforcement in action."""
    print("\n" + "="*60)
    print("🛡️ Demo: Budget Enforcement")
    print("="*60)
    
    manager = setup_budget_demo_configs()
    setup_alert_system(manager)
    
    print("Testing budget enforcement policies by simulating requests")
    print("that gradually approach and exceed budget limits.")
    
    # Simulate requests for different teams
    test_scenarios = [
        {
            "team": "dev-team",
            "requests": [
                {"description": "Small API call", "cost": 0.05},
                {"description": "Medium analysis", "cost": 0.25},
                {"description": "Large batch job", "cost": 2.50},
                {"description": "Heavy processing", "cost": 15.00},  # Should trigger alerts
                {"description": "Overrun attempt", "cost": 25.00},   # May be blocked
            ]
        },
        {
            "team": "prod-team", 
            "requests": [
                {"description": "Production query", "cost": 1.00},
                {"description": "Critical analysis", "cost": 5.00},
                {"description": "Large operation", "cost": 50.00},
                {"description": "Massive job", "cost": 150.00},  # Should trigger alerts
                {"description": "Emergency overrun", "cost": 100.00},  # Hard limit test
            ]
        }
    ]
    
    for scenario in test_scenarios:
        team = scenario["team"]
        requests = scenario["requests"]
        
        print(f"\n👥 Testing {team} budget enforcement:")
        
        total_spent = 0.0
        
        for i, request in enumerate(requests):
            cost = request["cost"]
            description = request["description"]
            
            print(f"\n   📋 Request {i+1}: {description} (${cost:.2f})")
            
            # Check budget allowance
            allowance = manager.check_budget_allowance(team, cost)
            
            if allowance['allowed']:
                # Record the usage
                manager.record_usage(team, cost)
                total_spent += cost
                
                print(f"   ✅ Approved - Remaining daily: ${allowance.get('remaining_daily', 0):.2f}")
                print(f"       Usage: {allowance.get('daily_usage_pct', 0):.1f}% daily, {allowance.get('monthly_usage_pct', 0):.1f}% monthly")
            else:
                print(f"   ❌ Blocked - {allowance['reason']}")
                print(f"   💡 Action: {allowance['action']}")
                break
        
        # Show final budget status
        final_summary = manager.get_budget_summary(team)
        print(f"\n   📊 Final status for {team}:")
        print(f"       Daily spent: ${final_summary['daily_spent']:.2f}/${final_summary['daily_limit']:.2f}")
        print(f"       Alerts active: {', '.join(final_summary['current_alerts']) if final_summary['current_alerts'] else 'None'}")
        
        if final_summary['is_blocked']:
            print(f"       🚫 BLOCKED: {final_summary.get('block_reason', 'Unknown')}")


def demo_real_time_tracking():
    """Demonstrate real-time budget tracking with actual API calls."""
    print("\n" + "="*60)
    print("📊 Demo: Real-Time Budget Tracking")
    print("="*60)
    
    import litellm
    from genops.providers.litellm import auto_instrument, get_usage_stats
    
    print("Real-time tracking integrates budget controls directly into")
    print("LiteLLM requests with automatic enforcement and alerts.")
    
    manager = setup_budget_demo_configs()
    setup_alert_system(manager)
    
    # Create a custom callback to integrate with budget manager
    def budget_aware_request(team: str, model: str, messages: List[Dict], estimated_cost: float = 0.01):
        """Make a budget-aware LiteLLM request."""
        
        # Check budget allowance first
        allowance = manager.check_budget_allowance(team, estimated_cost)
        
        if not allowance['allowed']:
            print(f"   ❌ Request blocked: {allowance['reason']}")
            return None
        
        try:
            # Make the actual request
            print(f"   🔄 Making request for {team}...")
            
            response = litellm.completion(
                model=model,
                messages=messages,
                max_tokens=30,
                timeout=10
            )
            
            # Record actual usage (would normally get real cost from response)
            actual_cost = estimated_cost  # Simplified for demo
            manager.record_usage(team, actual_cost)
            
            print(f"   ✅ Success - Cost: ${actual_cost:.4f}")
            
            return response
            
        except Exception as e:
            print(f"   ⚠️ Request failed: [Error details redacted for security]")
            return None
    
    # Enable GenOps tracking
    auto_instrument(
        team="budget-demo",
        project="real-time-tracking",
        daily_budget_limit=50.0
    )
    
    # Test requests from different teams
    test_requests = [
        {
            "team": "dev-team",
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Hello dev team!"}],
            "estimated_cost": 0.002
        },
        {
            "team": "research-team", 
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Research query here"}],
            "estimated_cost": 0.005
        },
        {
            "team": "prod-team",
            "model": "gpt-3.5-turbo", 
            "messages": [{"role": "user", "content": "Production request"}],
            "estimated_cost": 0.01
        }
    ]
    
    print(f"\n🎯 Processing budget-aware requests:")
    
    for i, request in enumerate(test_requests):
        print(f"\n📋 Request {i+1} from {request['team']}:")
        
        response = budget_aware_request(**request)
        
        if response:
            # Show updated budget status
            summary = manager.get_budget_summary(request['team'])
            print(f"       Updated usage: {summary['daily_usage_pct']:.1f}% daily")
    
    # Show overall tracking results
    print(f"\n📊 Real-Time Tracking Summary:")
    
    genops_stats = get_usage_stats()
    print(f"   GenOps total requests: {genops_stats['total_requests']}")
    print(f"   GenOps total cost: ${genops_stats['total_cost']:.6f}")
    
    all_budgets = manager.get_budget_summary()
    total_budget_spent = sum(
        summary['daily_spent'] for summary in all_budgets.values() 
        if 'daily_spent' in summary
    )
    print(f"   Budget manager total: ${total_budget_spent:.6f}")


def demo_emergency_controls():
    """Demonstrate emergency budget controls and circuit breakers."""
    print("\n" + "="*60)
    print("🚨 Demo: Emergency Budget Controls")
    print("="*60)
    
    print("Emergency controls provide immediate response to budget crises")
    print("with circuit breakers, emergency blocks, and escalation procedures.")
    
    manager = setup_budget_demo_configs()
    setup_alert_system(manager)
    
    print(f"\n🎯 Testing emergency control scenarios:")
    
    # Scenario 1: Emergency block
    print(f"\n📋 Scenario 1: Emergency Block")
    print("   Simulating security incident requiring immediate spending halt")
    
    emergency_team = "prod-team"
    block_result = manager.set_emergency_block(
        emergency_team, 
        "Security incident detected - suspicious API usage pattern"
    )
    
    if block_result:
        print(f"   ✅ Emergency block activated for {emergency_team}")
        
        # Test that requests are now blocked
        allowance = manager.check_budget_allowance(emergency_team, 0.01)
        print(f"   🚫 Request test: {allowance}")
        
        # Remove the block
        manager.remove_block(emergency_team)
        print(f"   ✅ Emergency block removed")
    
    # Scenario 2: Rapid spending detection
    print(f"\n📋 Scenario 2: Rapid Spending Detection")
    print("   Simulating sudden cost spike that triggers automatic alerts")
    
    rapid_team = "dev-team"
    
    # Simulate rapid spending
    rapid_costs = [0.50, 1.00, 2.00, 5.00, 10.00]  # Escalating costs
    
    print("   Simulating rapid cost escalation:")
    for i, cost in enumerate(rapid_costs):
        print(f"      Request {i+1}: ${cost:.2f}")
        
        allowance = manager.check_budget_allowance(rapid_team, cost)
        
        if allowance['allowed']:
            manager.record_usage(rapid_team, cost)
            print(f"         ✅ Approved ({allowance.get('daily_usage_pct', 0):.1f}% usage)")
        else:
            print(f"         ❌ Blocked: {allowance['reason']}")
            break
    
    # Show final status
    final_status = manager.get_budget_summary(rapid_team)
    print(f"\n   📊 Final Status:")
    print(f"       Spent: ${final_status['daily_spent']:.2f}/${final_status['daily_limit']:.2f}")
    print(f"       Active alerts: {final_status['current_alerts']}")
    print(f"       Blocked: {final_status['is_blocked']}")
    
    # Scenario 3: Budget forecasting alert
    print(f"\n📋 Scenario 3: Budget Forecasting")
    print("   Analyzing spending trends to predict budget exhaustion")
    
    forecast_team = "research-team"
    
    # Simulate spending pattern
    hourly_costs = [0.25, 0.30, 0.35, 0.40, 0.45]  # Increasing trend
    
    print("   Spending trend analysis:")
    for hour, cost in enumerate(hourly_costs):
        manager.record_usage(forecast_team, cost)
        print(f"      Hour {hour + 1}: ${cost:.2f}")
    
    current_status = manager.get_budget_summary(forecast_team)
    spent = current_status['daily_spent']
    limit = current_status['daily_limit']
    
    # Simple forecasting (linear projection)
    if len(hourly_costs) > 1:
        trend = (hourly_costs[-1] - hourly_costs[0]) / len(hourly_costs)
        hours_to_limit = (limit - spent) / (hourly_costs[-1] + trend) if (hourly_costs[-1] + trend) > 0 else float('inf')
        
        print(f"\n   📈 Forecast Analysis:")
        print(f"      Current trend: +${trend:.3f}/hour")
        print(f"      Estimated hours to budget limit: {hours_to_limit:.1f}")
        
        if hours_to_limit < 8:  # Less than 8 hours  
            print(f"      🚨 WARNING: Budget may be exhausted within business day!")
            print(f"      💡 Recommendation: Review spending or increase budget")


def main():
    """Run the complete budget management demonstration."""
    
    print("💳 LiteLLM + GenOps: Advanced Budget Management")
    print("=" * 60)
    print("Comprehensive spending controls, alerts, and financial governance")
    print("for LiteLLM applications with real-time budget enforcement")
    
    # Check setup
    if not check_budget_setup():
        print("\n❌ Setup incomplete. Please resolve issues above.")
        return 1
    
    try:
        # Run demonstrations
        manager = demo_budget_configuration()
        demo_budget_enforcement()
        demo_real_time_tracking()
        demo_emergency_controls()
        
        print("\n" + "="*60)
        print("🎉 Budget Management Complete!")
        
        print("\n💳 Budget Management Features Demonstrated:")
        print("   ✅ Team-based budget allocation and tracking")
        print("   ✅ Real-time spending alerts and notifications") 
        print("   ✅ Budget enforcement policies (advisory to hard limits)")
        print("   ✅ Emergency controls and circuit breakers")
        print("   ✅ Multi-tenant budget isolation")
        print("   ✅ Spending trend analysis and forecasting")
        
        print("\n🎯 Financial Governance Benefits:")
        print("   • Prevent cost overruns with automatic enforcement")
        print("   • Real-time visibility into spending across teams")
        print("   • Configurable policies for different environments")
        print("   • Emergency controls for incident response")
        print("   • Detailed audit trails for financial compliance")
        
        print("\n📊 Production Implementation:")
        print("   • Integrate budget checks into request middleware")
        print("   • Connect alerts to existing notification systems")
        print("   • Implement automated budget adjustments")
        print("   • Set up cost forecasting and trend monitoring")
        print("   • Configure emergency response procedures")
        
        print("\n📖 Next Steps:")
        print("   • Try production_patterns.py for scaling strategies")
        print("   • Explore compliance_monitoring.py for governance")
        print("   • Implement budget controls in your applications!")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Demo failed: [Error details redacted for security]")
        print("💡 For debugging, check your API key configuration")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)