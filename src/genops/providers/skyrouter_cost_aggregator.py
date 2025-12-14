#!/usr/bin/env python3
"""
SkyRouter Cost Aggregation and Analysis Engine

This module provides advanced cost aggregation, analysis, and optimization
capabilities for SkyRouter multi-model routing operations. It tracks costs
across teams, projects, models, and routing strategies, providing insights
for cost optimization and budget management.

Features:
- Multi-dimensional cost aggregation (team, project, model, route)
- Real-time budget monitoring and alerting
- Cost optimization recommendations  
- Route efficiency analysis and suggestions
- Multi-model cost comparison and insights
- Historical cost trend analysis
- Automated cost optimization strategies

Author: GenOps AI Contributors  
License: Apache 2.0
"""

import os
import time
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class SkyRouterCostSummary:
    """Summary of SkyRouter costs across multiple dimensions."""
    total_cost: Decimal
    total_operations: int
    cost_by_team: Dict[str, Decimal]
    cost_by_project: Dict[str, Decimal]
    cost_by_model: Dict[str, Decimal]
    cost_by_route: Dict[str, Decimal]
    cost_by_operation_type: Dict[str, Decimal]
    optimization_savings: Decimal
    average_cost_per_operation: Decimal
    start_time: datetime
    end_time: datetime

@dataclass
class SkyRouterBudgetAlert:
    """Budget alert for cost monitoring."""
    alert_type: str  # "warning", "critical", "budget_exceeded"
    message: str
    current_cost: Decimal
    budget_limit: Decimal
    utilization_percentage: float
    recommended_action: str
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class SkyRouterCostOptimization:
    """Cost optimization recommendation."""
    optimization_type: str
    title: str
    description: str
    potential_savings: Decimal
    effort_level: str  # "low", "medium", "high"
    priority_score: float
    implementation_guide: str
    affected_operations: List[str]

class SkyRouterCostAggregator:
    """Advanced cost aggregation and analysis for SkyRouter operations."""
    
    def __init__(
        self,
        team: str = "default",
        project: str = "default",
        daily_budget_limit: Optional[float] = None,
        enable_cost_alerts: bool = True,
        cost_history_days: int = 30
    ):
        """
        Initialize cost aggregator for SkyRouter operations.
        
        Args:
            team: Primary team for cost attribution
            project: Primary project for cost attribution  
            daily_budget_limit: Daily budget limit in USD
            enable_cost_alerts: Enable budget monitoring and alerts
            cost_history_days: Number of days to retain cost history
        """
        self.team = team
        self.project = project
        self.daily_budget_limit = daily_budget_limit
        self.enable_cost_alerts = enable_cost_alerts
        self.cost_history_days = cost_history_days
        
        # Cost tracking storage
        self.operations: List[Dict[str, Any]] = []
        self.daily_costs: Dict[str, Decimal] = defaultdict(lambda: Decimal('0'))
        self.team_budgets: Dict[str, float] = {}
        self.project_budgets: Dict[str, float] = {}
        
        # Alert tracking
        self.alerts: List[SkyRouterBudgetAlert] = []
        self.last_alert_check = time.time()
        
        # Optimization tracking
        self.optimization_history: List[SkyRouterCostOptimization] = []
        
        logger.info(f"SkyRouter cost aggregator initialized for team: {team}, project: {project}")
    
    def add_operation_cost(
        self,
        operation_type: str,
        cost: float,
        model: str,
        team: str,
        project: str,
        route: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add cost data for a SkyRouter operation."""
        operation_data = {
            "timestamp": time.time(),
            "operation_type": operation_type,
            "cost": Decimal(str(cost)),
            "model": model,
            "team": team,
            "project": project,
            "route": route or "default",
            "metadata": metadata or {},
            "date": datetime.now().strftime("%Y-%m-%d")
        }
        
        self.operations.append(operation_data)
        
        # Update daily costs
        today = datetime.now().strftime("%Y-%m-%d")
        self.daily_costs[today] += Decimal(str(cost))
        
        # Check budget alerts if enabled
        if self.enable_cost_alerts:
            self._check_budget_alerts()
        
        # Cleanup old data
        self._cleanup_old_data()
        
        logger.debug(f"Added operation cost: {operation_type}, ${cost:.3f}, model: {model}")
    
    def get_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> SkyRouterCostSummary:
        """Get comprehensive cost summary for specified time period."""
        
        # Default to last 24 hours if no dates specified
        if not end_date:
            end_date = datetime.now()
        if not start_date:
            start_date = end_date - timedelta(days=1)
        
        # Filter operations by date range
        start_timestamp = start_date.timestamp()
        end_timestamp = end_date.timestamp()
        
        filtered_ops = [
            op for op in self.operations
            if start_timestamp <= op["timestamp"] <= end_timestamp
        ]
        
        if not filtered_ops:
            return SkyRouterCostSummary(
                total_cost=Decimal('0'),
                total_operations=0,
                cost_by_team={},
                cost_by_project={},
                cost_by_model={},
                cost_by_route={},
                cost_by_operation_type={},
                optimization_savings=Decimal('0'),
                average_cost_per_operation=Decimal('0'),
                start_time=start_date,
                end_time=end_date
            )
        
        # Aggregate costs by dimensions
        cost_by_team = defaultdict(lambda: Decimal('0'))
        cost_by_project = defaultdict(lambda: Decimal('0'))
        cost_by_model = defaultdict(lambda: Decimal('0'))
        cost_by_route = defaultdict(lambda: Decimal('0'))
        cost_by_operation_type = defaultdict(lambda: Decimal('0'))
        
        total_cost = Decimal('0')
        total_savings = Decimal('0')
        
        for op in filtered_ops:
            cost = op["cost"]
            total_cost += cost
            
            cost_by_team[op["team"]] += cost
            cost_by_project[op["project"]] += cost
            cost_by_model[op["model"]] += cost
            cost_by_route[op["route"]] += cost
            cost_by_operation_type[op["operation_type"]] += cost
            
            # Track optimization savings
            if "optimization_savings" in op["metadata"]:
                total_savings += Decimal(str(op["metadata"]["optimization_savings"]))
        
        average_cost = total_cost / len(filtered_ops) if filtered_ops else Decimal('0')
        
        return SkyRouterCostSummary(
            total_cost=total_cost,
            total_operations=len(filtered_ops),
            cost_by_team=dict(cost_by_team),
            cost_by_project=dict(cost_by_project),
            cost_by_model=dict(cost_by_model),
            cost_by_route=dict(cost_by_route),
            cost_by_operation_type=dict(cost_by_operation_type),
            optimization_savings=total_savings,
            average_cost_per_operation=average_cost,
            start_time=start_date,
            end_time=end_date
        )
    
    def set_team_budget(self, team: str, daily_limit: float):
        """Set daily budget limit for a team."""
        self.team_budgets[team] = daily_limit
        logger.info(f"Set daily budget for team {team}: ${daily_limit:.2f}")
    
    def set_project_budget(self, project: str, daily_limit: float):
        """Set daily budget limit for a project."""
        self.project_budgets[project] = daily_limit
        logger.info(f"Set daily budget for project {project}: ${daily_limit:.2f}")
    
    def check_budget_status(self) -> Dict[str, Any]:
        """Check current budget status and return alerts."""
        today = datetime.now().strftime("%Y-%m-%d")
        current_daily_cost = self.daily_costs[today]
        
        budget_alerts = []
        
        # Check overall daily budget
        if self.daily_budget_limit and current_daily_cost >= Decimal(str(self.daily_budget_limit * 0.8)):
            utilization = float(current_daily_cost / Decimal(str(self.daily_budget_limit))) * 100
            
            if current_daily_cost >= Decimal(str(self.daily_budget_limit)):
                alert_type = "budget_exceeded"
                message = f"Daily budget exceeded: ${current_daily_cost:.2f} > ${self.daily_budget_limit:.2f}"
            elif utilization >= 90:
                alert_type = "critical"
                message = f"Daily budget critical: {utilization:.1f}% used"
            else:
                alert_type = "warning"
                message = f"Daily budget warning: {utilization:.1f}% used"
            
            budget_alerts.append({
                "type": alert_type,
                "message": message,
                "cost": float(current_daily_cost),
                "limit": self.daily_budget_limit,
                "utilization": utilization
            })
        
        # Check team budgets
        team_costs = self._get_daily_team_costs(today)
        for team, team_cost in team_costs.items():
            if team in self.team_budgets:
                limit = self.team_budgets[team]
                if team_cost >= Decimal(str(limit * 0.8)):
                    utilization = float(team_cost / Decimal(str(limit))) * 100
                    budget_alerts.append({
                        "type": "team_budget_warning",
                        "message": f"Team {team} budget: {utilization:.1f}% used",
                        "team": team,
                        "cost": float(team_cost),
                        "limit": limit,
                        "utilization": utilization
                    })
        
        # Check project budgets
        project_costs = self._get_daily_project_costs(today)
        for project, project_cost in project_costs.items():
            if project in self.project_budgets:
                limit = self.project_budgets[project]
                if project_cost >= Decimal(str(limit * 0.8)):
                    utilization = float(project_cost / Decimal(str(limit))) * 100
                    budget_alerts.append({
                        "type": "project_budget_warning",
                        "message": f"Project {project} budget: {utilization:.1f}% used",
                        "project": project,
                        "cost": float(project_cost),
                        "limit": limit,
                        "utilization": utilization
                    })
        
        return {
            "current_daily_cost": float(current_daily_cost),
            "daily_budget_limit": self.daily_budget_limit,
            "budget_alerts": budget_alerts,
            "team_costs": {k: float(v) for k, v in team_costs.items()},
            "project_costs": {k: float(v) for k, v in project_costs.items()}
        }
    
    def get_cost_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Generate cost optimization recommendations based on usage patterns."""
        recommendations = []
        
        # Analyze recent operations (last 7 days)
        week_ago = datetime.now() - timedelta(days=7)
        summary = self.get_summary(start_date=week_ago)
        
        if summary.total_operations == 0:
            return recommendations
        
        # 1. Model optimization recommendations
        model_costs = summary.cost_by_model
        if len(model_costs) > 1:
            sorted_models = sorted(model_costs.items(), key=lambda x: x[1], reverse=True)
            most_expensive = sorted_models[0]
            
            if most_expensive[1] > summary.total_cost * Decimal('0.3'):  # >30% of costs
                recommendations.append({
                    "optimization_type": "model_optimization",
                    "title": "Consider model alternatives for high-cost operations",
                    "description": f"Model '{most_expensive[0]}' accounts for {float(most_expensive[1]/summary.total_cost)*100:.1f}% of costs",
                    "potential_savings": float(most_expensive[1] * Decimal('0.2')),  # Estimate 20% savings
                    "effort_level": "medium",
                    "priority_score": 85.0,
                    "implementation_guide": "Evaluate alternative models with similar performance but lower costs"
                })
        
        # 2. Route optimization recommendations
        route_costs = summary.cost_by_route
        if "balanced" in route_costs and "cost_optimized" in route_costs:
            balanced_cost = route_costs["balanced"]
            cost_optimized = route_costs.get("cost_optimized", Decimal('0'))
            
            if balanced_cost > cost_optimized * Decimal('1.15'):  # 15% more expensive
                potential_savings = balanced_cost - cost_optimized
                recommendations.append({
                    "optimization_type": "route_optimization",
                    "title": "Switch to cost-optimized routing strategy",
                    "description": "Cost-optimized routing shows lower costs than balanced approach",
                    "potential_savings": float(potential_savings),
                    "effort_level": "low",
                    "priority_score": 75.0,
                    "implementation_guide": "Update routing strategy to 'cost_optimized' in adapter configuration"
                })
        
        # 3. Volume discount optimization
        total_operations = summary.total_operations
        avg_cost = summary.average_cost_per_operation
        
        if total_operations < 1000 and avg_cost > Decimal('0.01'):
            recommendations.append({
                "optimization_type": "volume_optimization",
                "title": "Increase operation volume to unlock volume discounts",
                "description": f"Current volume ({total_operations} ops) may not qualify for volume discounts",
                "potential_savings": float(summary.total_cost * Decimal('0.1')),  # Estimate 10% savings
                "effort_level": "high",
                "priority_score": 60.0,
                "implementation_guide": "Consolidate operations or batch requests to increase volume"
            })
        
        # 4. Multi-model routing recommendations
        if summary.cost_by_operation_type.get("model_call", 0) > summary.cost_by_operation_type.get("multi_model_routing", 0) * 2:
            recommendations.append({
                "optimization_type": "routing_strategy",
                "title": "Implement multi-model routing for cost optimization",
                "description": "Single model calls dominate usage - multi-model routing could reduce costs",
                "potential_savings": float(summary.total_cost * Decimal('0.15')),  # Estimate 15% savings
                "effort_level": "medium",
                "priority_score": 70.0,
                "implementation_guide": "Use track_multi_model_routing() instead of track_model_call() where possible"
            })
        
        # 5. Budget optimization
        if self.daily_budget_limit:
            current_utilization = float(summary.total_cost / (Decimal(str(self.daily_budget_limit)) * 7))  # 7 days
            
            if current_utilization < 0.5:  # Under 50% budget utilization
                recommendations.append({
                    "optimization_type": "budget_optimization",
                    "title": "Budget utilization is low - consider reallocating",
                    "description": f"Only using {current_utilization*100:.1f}% of available budget",
                    "potential_savings": 0.0,  # Not really savings, but optimization
                    "effort_level": "low",
                    "priority_score": 40.0,
                    "implementation_guide": "Consider reallocating unused budget or increasing operation volume"
                })
        
        # Sort by priority score
        recommendations.sort(key=lambda x: x["priority_score"], reverse=True)
        
        return recommendations
    
    def _check_budget_alerts(self):
        """Check and generate budget alerts if necessary."""
        current_time = time.time()
        
        # Only check alerts every 5 minutes to avoid spam
        if current_time - self.last_alert_check < 300:
            return
        
        self.last_alert_check = current_time
        
        budget_status = self.check_budget_status()
        
        for alert_data in budget_status["budget_alerts"]:
            alert = SkyRouterBudgetAlert(
                alert_type=alert_data["type"],
                message=alert_data["message"],
                current_cost=Decimal(str(alert_data["cost"])),
                budget_limit=Decimal(str(alert_data["limit"])),
                utilization_percentage=alert_data["utilization"],
                recommended_action=self._get_recommended_action(alert_data)
            )
            
            self.alerts.append(alert)
            
            # Log the alert
            if alert.alert_type == "budget_exceeded":
                logger.error(f"Budget exceeded: {alert.message}")
            elif alert.alert_type == "critical":
                logger.warning(f"Budget critical: {alert.message}")
            else:
                logger.info(f"Budget warning: {alert.message}")
        
        # Keep only recent alerts (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.alerts = [alert for alert in self.alerts if alert.timestamp >= cutoff_time]
    
    def _get_recommended_action(self, alert_data: Dict[str, Any]) -> str:
        """Get recommended action for a budget alert."""
        alert_type = alert_data["type"]
        utilization = alert_data["utilization"]
        
        if alert_type == "budget_exceeded":
            return "Immediate action required: Stop operations or increase budget"
        elif utilization >= 90:
            return "Reduce operation frequency or switch to cost-optimized routing"
        elif utilization >= 80:
            return "Monitor usage closely and consider cost optimization"
        else:
            return "Review cost optimization recommendations"
    
    def _get_daily_team_costs(self, date: str) -> Dict[str, Decimal]:
        """Get daily costs broken down by team."""
        team_costs = defaultdict(lambda: Decimal('0'))
        
        for op in self.operations:
            if op["date"] == date:
                team_costs[op["team"]] += op["cost"]
        
        return dict(team_costs)
    
    def _get_daily_project_costs(self, date: str) -> Dict[str, Decimal]:
        """Get daily costs broken down by project."""
        project_costs = defaultdict(lambda: Decimal('0'))
        
        for op in self.operations:
            if op["date"] == date:
                project_costs[op["project"]] += op["cost"]
        
        return dict(project_costs)
    
    def _cleanup_old_data(self):
        """Clean up old cost data beyond retention period."""
        cutoff_date = datetime.now() - timedelta(days=self.cost_history_days)
        cutoff_timestamp = cutoff_date.timestamp()
        
        # Remove old operations
        self.operations = [
            op for op in self.operations
            if op["timestamp"] >= cutoff_timestamp
        ]
        
        # Remove old daily costs
        cutoff_date_str = cutoff_date.strftime("%Y-%m-%d")
        dates_to_remove = [
            date for date in self.daily_costs.keys()
            if date < cutoff_date_str
        ]
        
        for date in dates_to_remove:
            del self.daily_costs[date]
    
    def export_cost_data(self, format: str = "json") -> str:
        """Export cost data for external analysis."""
        summary = self.get_summary()
        
        export_data = {
            "summary": {
                "total_cost": float(summary.total_cost),
                "total_operations": summary.total_operations,
                "average_cost_per_operation": float(summary.average_cost_per_operation),
                "optimization_savings": float(summary.optimization_savings),
                "period": {
                    "start": summary.start_time.isoformat(),
                    "end": summary.end_time.isoformat()
                }
            },
            "cost_breakdowns": {
                "by_team": {k: float(v) for k, v in summary.cost_by_team.items()},
                "by_project": {k: float(v) for k, v in summary.cost_by_project.items()},
                "by_model": {k: float(v) for k, v in summary.cost_by_model.items()},
                "by_route": {k: float(v) for k, v in summary.cost_by_route.items()},
                "by_operation_type": {k: float(v) for k, v in summary.cost_by_operation_type.items()}
            },
            "optimization_recommendations": self.get_cost_optimization_recommendations(),
            "budget_status": self.check_budget_status(),
            "recent_alerts": [
                {
                    "type": alert.alert_type,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat(),
                    "utilization": alert.utilization_percentage
                }
                for alert in self.alerts[-10:]  # Last 10 alerts
            ]
        }
        
        if format == "json":
            return json.dumps(export_data, indent=2, default=str)
        elif format == "csv":
            # Simple CSV export of operations
            lines = ["timestamp,operation_type,cost,model,team,project,route"]
            for op in self.operations:
                lines.append(f"{op['timestamp']},{op['operation_type']},{op['cost']},{op['model']},{op['team']},{op['project']},{op['route']}")
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_cost_trends(self, days: int = 7) -> Dict[str, Any]:
        """Analyze cost trends over specified number of days."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get daily cost data
        daily_costs = {}
        current_date = start_date
        
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            daily_costs[date_str] = float(self.daily_costs.get(date_str, Decimal('0')))
            current_date += timedelta(days=1)
        
        # Calculate trend metrics
        costs = list(daily_costs.values())
        if len(costs) >= 2:
            # Simple trend calculation
            recent_avg = sum(costs[-3:]) / min(3, len(costs))  # Last 3 days average
            older_avg = sum(costs[:-3]) / max(1, len(costs) - 3) if len(costs) > 3 else costs[0]
            
            trend_direction = "increasing" if recent_avg > older_avg * 1.1 else "decreasing" if recent_avg < older_avg * 0.9 else "stable"
            trend_percentage = ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
        else:
            trend_direction = "insufficient_data"
            trend_percentage = 0
        
        return {
            "period_days": days,
            "daily_costs": daily_costs,
            "total_cost": sum(costs),
            "average_daily_cost": sum(costs) / len(costs) if costs else 0,
            "trend_direction": trend_direction,
            "trend_percentage": round(trend_percentage, 2),
            "highest_day": max(daily_costs.items(), key=lambda x: x[1]) if daily_costs else None,
            "lowest_day": min(daily_costs.items(), key=lambda x: x[1]) if daily_costs else None
        }