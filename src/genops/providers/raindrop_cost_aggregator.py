#!/usr/bin/env python3
"""
Raindrop AI Cost Aggregation Engine

This module provides comprehensive cost aggregation for Raindrop AI operations
across multiple agents, sessions, and time periods. It enables unified cost tracking,
team attribution, and financial reporting for AI agent monitoring workflows.

Features:
- Multi-agent cost aggregation with unified reporting
- Session-based cost tracking and attribution
- Real-time cost monitoring and budget enforcement
- Team and project cost breakdowns
- Time-series cost analysis and forecasting
- Cross-provider cost comparison and optimization
- Enterprise-ready financial reporting integration

Author: GenOps AI Contributors
License: Apache 2.0
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple
from decimal import Decimal
from datetime import datetime, timedelta
import logging

from .raindrop_pricing import RaindropCostResult, RaindropOperationType

logger = logging.getLogger(__name__)

@dataclass
class RaindropSessionSummary:
    """Summary of costs and metrics for a single monitoring session."""
    session_id: str
    session_name: str
    total_cost: float
    operation_count: int
    duration_seconds: float
    operations: List[Dict[str, Any]]
    governance_attributes: Dict[str, str]
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    @property
    def cost_per_operation(self) -> float:
        """Average cost per operation in this session."""
        return self.total_cost / max(1, self.operation_count)
    
    @property
    def operations_per_hour(self) -> float:
        """Operations per hour rate for this session."""
        hours = max(self.duration_seconds / 3600, 1/3600)  # Minimum 1 second
        return self.operation_count / hours
    
    @property
    def team(self) -> str:
        """Team from governance attributes."""
        return self.governance_attributes.get('genops.team', 'unknown')
    
    @property
    def project(self) -> str:
        """Project from governance attributes."""
        return self.governance_attributes.get('genops.project', 'unknown')
    
    @property
    def environment(self) -> str:
        """Environment from governance attributes."""
        return self.governance_attributes.get('genops.environment', 'unknown')

@dataclass
class RaindropCostSummary:
    """Aggregated cost summary across multiple sessions and operations."""
    total_cost: Decimal = field(default_factory=lambda: Decimal('0.00'))
    cost_by_operation_type: Dict[str, Decimal] = field(default_factory=lambda: defaultdict(lambda: Decimal('0.00')))
    cost_by_team: Dict[str, Decimal] = field(default_factory=lambda: defaultdict(lambda: Decimal('0.00')))
    cost_by_project: Dict[str, Decimal] = field(default_factory=lambda: defaultdict(lambda: Decimal('0.00')))
    cost_by_environment: Dict[str, Decimal] = field(default_factory=lambda: defaultdict(lambda: Decimal('0.00')))
    cost_by_agent: Dict[str, Decimal] = field(default_factory=lambda: defaultdict(lambda: Decimal('0.00')))
    
    session_count: int = 0
    total_operations: int = 0
    total_duration_seconds: float = 0.0
    
    unique_teams: Set[str] = field(default_factory=set)
    unique_projects: Set[str] = field(default_factory=set)
    unique_agents: Set[str] = field(default_factory=set)
    
    currency: str = "USD"
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    
    @property
    def average_cost_per_session(self) -> Decimal:
        """Average cost per session."""
        if self.session_count == 0:
            return Decimal('0.00')
        return self.total_cost / self.session_count
    
    @property
    def average_cost_per_operation(self) -> Decimal:
        """Average cost per operation."""
        if self.total_operations == 0:
            return Decimal('0.00')
        return self.total_cost / self.total_operations
    
    @property
    def operations_per_hour(self) -> float:
        """Average operations per hour."""
        if self.total_duration_seconds == 0:
            return 0.0
        hours = self.total_duration_seconds / 3600
        return self.total_operations / hours

class RaindropCostAggregator:
    """
    Comprehensive cost aggregation engine for Raindrop AI operations.
    
    Tracks costs across multiple sessions, teams, and time periods with
    real-time aggregation and enterprise-ready reporting capabilities.
    """
    
    def __init__(self):
        """Initialize the cost aggregator."""
        self.sessions: Dict[str, RaindropSessionSummary] = {}
        self.cost_history: List[Tuple[float, RaindropCostResult]] = []
        self.daily_costs: Dict[str, Decimal] = defaultdict(lambda: Decimal('0.00'))  # date -> cost
        self.team_budgets: Dict[str, Decimal] = {}  # team -> daily budget
        self.project_budgets: Dict[str, Decimal] = {}  # project -> daily budget
        
    def add_session(self, session: RaindropSessionSummary) -> None:
        """
        Add a completed session to the aggregator.
        
        Args:
            session: Completed session summary with cost and operation details
        """
        self.sessions[session.session_id] = session
        
        # Update cost history for each operation in the session
        for operation in session.operations:
            cost_result = RaindropCostResult(
                operation_type=operation['type'],
                base_cost=Decimal(str(operation['cost'])),
                total_cost=Decimal(str(operation['cost'])),
                timestamp=operation['timestamp']
            )
            self.cost_history.append((operation['timestamp'], cost_result))
        
        # Update daily costs
        session_date = datetime.fromtimestamp(session.start_time).strftime('%Y-%m-%d')
        self.daily_costs[session_date] += Decimal(str(session.total_cost))
        
        logger.debug(f"Added session {session.session_id} with cost ${session.total_cost}")
    
    def add_cost_result(self, cost_result: RaindropCostResult) -> None:
        """
        Add an individual cost result to the aggregator.
        
        Args:
            cost_result: Individual operation cost result
        """
        self.cost_history.append((cost_result.timestamp, cost_result))
        
        # Update daily costs
        result_date = datetime.fromtimestamp(cost_result.timestamp).strftime('%Y-%m-%d')
        self.daily_costs[result_date] += cost_result.total_cost
        
        logger.debug(f"Added cost result: {cost_result.operation_type} - ${cost_result.total_cost}")
    
    def get_summary(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        team_filter: Optional[str] = None,
        project_filter: Optional[str] = None
    ) -> RaindropCostSummary:
        """
        Generate aggregated cost summary with optional filtering.
        
        Args:
            start_time: Start timestamp for filtering (optional)
            end_time: End timestamp for filtering (optional)
            team_filter: Filter by specific team (optional)
            project_filter: Filter by specific project (optional)
            
        Returns:
            RaindropCostSummary: Aggregated cost summary
        """
        summary = RaindropCostSummary()
        
        # Filter sessions based on criteria
        filtered_sessions = self._filter_sessions(start_time, end_time, team_filter, project_filter)
        
        # Aggregate costs from filtered sessions
        for session in filtered_sessions:
            session_cost = Decimal(str(session.total_cost))
            summary.total_cost += session_cost
            summary.session_count += 1
            summary.total_operations += session.operation_count
            summary.total_duration_seconds += session.duration_seconds
            
            # Aggregate by team, project, environment
            summary.cost_by_team[session.team] += session_cost
            summary.cost_by_project[session.project] += session_cost
            summary.cost_by_environment[session.environment] += session_cost
            
            # Track unique identifiers
            summary.unique_teams.add(session.team)
            summary.unique_projects.add(session.project)
            
            # Aggregate by operation type and agent
            for operation in session.operations:
                op_cost = Decimal(str(operation['cost']))
                summary.cost_by_operation_type[operation['type']] += op_cost
                
                # Track agent costs if available
                if 'agent_id' in operation:
                    summary.cost_by_agent[operation['agent_id']] += op_cost
                    summary.unique_agents.add(operation['agent_id'])
        
        # Set time period
        if filtered_sessions:
            summary.period_start = datetime.fromtimestamp(min(s.start_time for s in filtered_sessions))
            summary.period_end = datetime.fromtimestamp(max(s.start_time for s in filtered_sessions))
        
        return summary
    
    def get_daily_costs(self, days: int = 30) -> Dict[str, float]:
        """
        Get daily cost breakdown for the last N days.
        
        Args:
            days: Number of days to include in the breakdown
            
        Returns:
            Dict mapping dates to daily costs
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        daily_breakdown = {}
        for i in range(days):
            date = start_date + timedelta(days=i)
            date_str = date.strftime('%Y-%m-%d')
            daily_breakdown[date_str] = float(self.daily_costs.get(date_str, Decimal('0.00')))
        
        return daily_breakdown
    
    def get_cost_breakdown_by_team(self, days: int = 30) -> Dict[str, Dict[str, float]]:
        """
        Get cost breakdown by team for the last N days.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dict mapping teams to their daily cost breakdowns
        """
        end_date = datetime.now()
        start_time = (end_date - timedelta(days=days)).timestamp()
        
        team_breakdown = defaultdict(lambda: defaultdict(float))
        
        filtered_sessions = self._filter_sessions(start_time=start_time)
        
        for session in filtered_sessions:
            date_str = datetime.fromtimestamp(session.start_time).strftime('%Y-%m-%d')
            team_breakdown[session.team][date_str] += session.total_cost
        
        return dict(team_breakdown)
    
    def get_cost_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate cost optimization recommendations based on usage patterns.
        
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        summary = self.get_summary()
        
        # Recommendation 1: High-frequency agent monitoring optimization
        if summary.operations_per_hour > 1000:
            recommendations.append({
                'type': 'optimization',
                'category': 'frequency',
                'title': 'Optimize High-Frequency Agent Monitoring',
                'description': f'Your agents are performing {summary.operations_per_hour:.0f} operations/hour. Consider implementing intelligent sampling.',
                'potential_savings': float(summary.total_cost * Decimal('0.30')),
                'effort_level': 'Medium',
                'priority_score': 85.0,
                'actions': [
                    'Implement intelligent sampling to reduce monitoring frequency',
                    'Use batch processing for agent interactions',
                    'Optimize performance signal collection'
                ]
            })
        
        # Recommendation 2: Team cost optimization
        if len(summary.cost_by_team) > 1:
            highest_cost_team = max(summary.cost_by_team.items(), key=lambda x: x[1])
            if highest_cost_team[1] > summary.total_cost * Decimal('0.50'):
                recommendations.append({
                    'type': 'optimization',
                    'category': 'team_costs',
                    'title': f'Optimize {highest_cost_team[0]} Team Costs',
                    'description': f'Team {highest_cost_team[0]} accounts for {float(highest_cost_team[1]/summary.total_cost*100):.1f}% of total costs.',
                    'potential_savings': float(highest_cost_team[1] * Decimal('0.20')),
                    'effort_level': 'Low',
                    'priority_score': 70.0,
                    'actions': [
                        'Review monitoring frequency for this team',
                        'Implement team-specific budget controls',
                        'Optimize alert configurations'
                    ]
                })
        
        # Recommendation 3: Operation type optimization
        op_costs = summary.cost_by_operation_type
        if op_costs:
            highest_cost_op = max(op_costs.items(), key=lambda x: x[1])
            if highest_cost_op[1] > summary.total_cost * Decimal('0.40'):
                recommendations.append({
                    'type': 'optimization',
                    'category': 'operations',
                    'title': f'Optimize {highest_cost_op[0]} Operations',
                    'description': f'{highest_cost_op[0]} operations account for {float(highest_cost_op[1]/summary.total_cost*100):.1f}% of costs.',
                    'potential_savings': float(highest_cost_op[1] * Decimal('0.25')),
                    'effort_level': 'Medium',
                    'priority_score': 75.0,
                    'actions': [
                        f'Review {highest_cost_op[0]} operation frequency',
                        'Consider batching or sampling strategies',
                        'Optimize data payload sizes'
                    ]
                })
        
        # Sort by priority score
        recommendations.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return recommendations
    
    def check_budget_status(self) -> Dict[str, Any]:
        """
        Check current budget status for teams and projects.
        
        Returns:
            Budget status information including alerts and recommendations
        """
        today = datetime.now().strftime('%Y-%m-%d')
        today_cost = self.daily_costs.get(today, Decimal('0.00'))
        
        summary = self.get_summary()
        budget_status = {
            'today_total_cost': float(today_cost),
            'team_status': {},
            'project_status': {},
            'budget_alerts': [],
            'recommendations': []
        }
        
        # Check team budgets
        for team, budget in self.team_budgets.items():
            team_today_cost = self._get_team_daily_cost(team, today)
            utilization = float(team_today_cost / budget) if budget > 0 else 0
            
            budget_status['team_status'][team] = {
                'daily_budget': float(budget),
                'today_cost': float(team_today_cost),
                'utilization': utilization,
                'remaining': float(budget - team_today_cost)
            }
            
            # Generate alerts
            if utilization >= 0.9:
                budget_status['budget_alerts'].append({
                    'type': 'team_budget_exceeded',
                    'team': team,
                    'message': f'Team {team} has used {utilization*100:.1f}% of daily budget'
                })
            elif utilization >= 0.8:
                budget_status['budget_alerts'].append({
                    'type': 'team_budget_warning',
                    'team': team,
                    'message': f'Team {team} approaching budget limit ({utilization*100:.1f}% used)'
                })
        
        # Check project budgets
        for project, budget in self.project_budgets.items():
            project_today_cost = self._get_project_daily_cost(project, today)
            utilization = float(project_today_cost / budget) if budget > 0 else 0
            
            budget_status['project_status'][project] = {
                'daily_budget': float(budget),
                'today_cost': float(project_today_cost),
                'utilization': utilization,
                'remaining': float(budget - project_today_cost)
            }
            
            # Generate alerts
            if utilization >= 0.9:
                budget_status['budget_alerts'].append({
                    'type': 'project_budget_exceeded',
                    'project': project,
                    'message': f'Project {project} has used {utilization*100:.1f}% of daily budget'
                })
            elif utilization >= 0.8:
                budget_status['budget_alerts'].append({
                    'type': 'project_budget_warning',
                    'project': project,
                    'message': f'Project {project} approaching budget limit ({utilization*100:.1f}% used)'
                })
        
        return budget_status
    
    def set_team_budget(self, team: str, daily_budget: float) -> None:
        """Set daily budget for a team."""
        self.team_budgets[team] = Decimal(str(daily_budget))
        logger.info(f"Set daily budget for team {team}: ${daily_budget}")
    
    def set_project_budget(self, project: str, daily_budget: float) -> None:
        """Set daily budget for a project."""
        self.project_budgets[project] = Decimal(str(daily_budget))
        logger.info(f"Set daily budget for project {project}: ${daily_budget}")
    
    def _filter_sessions(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        team_filter: Optional[str] = None,
        project_filter: Optional[str] = None
    ) -> List[RaindropSessionSummary]:
        """Filter sessions based on criteria."""
        filtered = []
        
        for session in self.sessions.values():
            # Time filtering
            if start_time and session.start_time < start_time:
                continue
            if end_time and session.start_time > end_time:
                continue
            
            # Team filtering
            if team_filter and session.team != team_filter:
                continue
            
            # Project filtering
            if project_filter and session.project != project_filter:
                continue
            
            filtered.append(session)
        
        return filtered
    
    def _get_team_daily_cost(self, team: str, date: str) -> Decimal:
        """Get daily cost for a specific team."""
        daily_cost = Decimal('0.00')
        
        for session in self.sessions.values():
            session_date = datetime.fromtimestamp(session.start_time).strftime('%Y-%m-%d')
            if session_date == date and session.team == team:
                daily_cost += Decimal(str(session.total_cost))
        
        return daily_cost
    
    def _get_project_daily_cost(self, project: str, date: str) -> Decimal:
        """Get daily cost for a specific project."""
        daily_cost = Decimal('0.00')
        
        for session in self.sessions.values():
            session_date = datetime.fromtimestamp(session.start_time).strftime('%Y-%m-%d')
            if session_date == date and session.project == project:
                daily_cost += Decimal(str(session.total_cost))
        
        return daily_cost

# Export main classes
__all__ = [
    'RaindropCostAggregator',
    'RaindropSessionSummary',
    'RaindropCostSummary'
]