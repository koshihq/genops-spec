#!/usr/bin/env python3
"""
AutoGen Conversation Monitoring for GenOps Governance

Specialized monitoring for AutoGen conversation flows, agent interactions,
group chat orchestration, and code execution patterns.

Usage:
    from genops.providers.autogen.conversation_monitor import AutoGenConversationMonitor
    
    monitor = AutoGenConversationMonitor(
        team="ai-research",
        project="multi-agent-conversations"
    )
    
    # Track conversation metrics
    with monitor.track_conversation("user-assistant") as tracker:
        tracker.add_turn("assistant", 150, 2.5)  # tokens, response_time_ms
        tracker.add_code_execution("python", True)
        tracker.add_function_call("web_search", {"query": "AI research"})

Features:
    - Conversation flow tracking with turn-by-turn analysis
    - Agent interaction patterns and collaboration metrics
    - Group chat orchestration monitoring
    - Code execution tracking and success rates
    - Function calling telemetry and usage patterns
    - Performance analysis and bottleneck identification
    - Conversation quality scoring and optimization insights
"""

import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Union, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import threading
import statistics

logger = logging.getLogger(__name__)


class ConversationStatus(Enum):
    """Status of a conversation."""
    ACTIVE = "active"
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"


class MessageType(Enum):
    """Types of messages in AutoGen conversations."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION_CALL = "function_call"
    FUNCTION_RESPONSE = "function_response"
    CODE_EXECUTION = "code_execution"
    CODE_RESULT = "code_result"


@dataclass
class ConversationTurn:
    """Single turn in an AutoGen conversation."""
    turn_number: int
    agent_name: str
    message_type: MessageType
    timestamp: datetime
    response_time_ms: float
    tokens_used: int
    cost: Decimal
    message_length: int
    function_calls: List[str] = field(default_factory=list)
    code_executed: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentInteractionMetrics:
    """Metrics for agent interactions within conversations."""
    agent_name: str
    total_turns: int
    total_tokens: int
    total_cost: Decimal
    avg_response_time_ms: float
    messages_sent: int
    messages_received: int
    function_calls_made: int
    code_executions: int
    success_rate: float
    collaboration_score: float
    efficiency_score: float
    last_active: datetime


@dataclass
class ConversationMetrics:
    """Comprehensive metrics for a conversation."""
    conversation_id: str
    status: ConversationStatus
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: float
    turns_count: int
    participants: List[str]
    total_tokens: int
    total_cost: Decimal
    avg_response_time_ms: float
    cost_per_turn: Decimal
    code_executions_count: int
    function_calls_count: int
    success_rate: float
    quality_score: float
    agent_participation: Dict[str, float]
    turn_distribution: Dict[str, int]
    conversation_turns: List[ConversationTurn] = field(default_factory=list)


@dataclass
class GroupChatMetrics:
    """Metrics specific to AutoGen group chat sessions."""
    group_chat_id: str
    participants: List[str]
    speaker_transitions: int
    coordination_overhead_ms: float
    parallel_efficiency: float
    dominant_speaker: str
    quiet_participants: List[str]
    turn_balance_score: float
    consensus_quality: float
    group_dynamics_score: float


@dataclass
class CodeExecutionMetrics:
    """Metrics for code execution within conversations."""
    total_executions: int
    successful_executions: int
    failed_executions: int
    languages_used: Set[str]
    avg_execution_time_ms: float
    success_rate: float
    error_types: Dict[str, int]
    resource_usage: Dict[str, Any]


class ConversationTracker:
    """Tracks metrics for a single conversation."""
    
    def __init__(self, conversation_id: str, monitor: 'AutoGenConversationMonitor'):
        self.conversation_id = conversation_id
        self.monitor = monitor
        self.start_time = datetime.now()
        self.status = ConversationStatus.ACTIVE
        self.turns = []
        self.participants = set()
        self.current_turn = 0
        self.total_tokens = 0
        self.total_cost = Decimal('0')
        self.code_executions = 0
        self.function_calls = 0
        self.response_times = deque(maxlen=100)  # Rolling window for response times
        self.errors = []
        
    def add_turn(
        self,
        agent_name: str,
        tokens_used: int,
        response_time_ms: float,
        message_type: MessageType = MessageType.ASSISTANT,
        cost: Decimal = Decimal('0'),
        message_length: int = 0,
        **metadata
    ):
        """Add a conversation turn with metrics."""
        self.current_turn += 1
        self.participants.add(agent_name)
        self.total_tokens += tokens_used
        self.total_cost += cost
        self.response_times.append(response_time_ms)
        
        turn = ConversationTurn(
            turn_number=self.current_turn,
            agent_name=agent_name,
            message_type=message_type,
            timestamp=datetime.now(),
            response_time_ms=response_time_ms,
            tokens_used=tokens_used,
            cost=cost,
            message_length=message_length,
            metadata=metadata
        )
        
        self.turns.append(turn)
        logger.debug(f"Added turn {self.current_turn} for {agent_name} in {self.conversation_id}")
        
    def add_code_execution(self, language: str, success: bool, execution_time_ms: float = 0):
        """Record a code execution event."""
        self.code_executions += 1
        
        if self.turns:
            # Associate with the most recent turn
            self.turns[-1].code_executed = True
            if not success:
                self.turns[-1].error = f"Code execution failed ({language})"
                
        logger.debug(f"Code execution: {language} ({'success' if success else 'failed'})")
        
    def add_function_call(self, function_name: str, parameters: Dict[str, Any] = None):
        """Record a function call event."""
        self.function_calls += 1
        
        if self.turns:
            # Associate with the most recent turn
            self.turns[-1].function_calls.append(function_name)
            
        logger.debug(f"Function call: {function_name}")
        
    def add_error(self, error_msg: str):
        """Record an error in the conversation."""
        self.errors.append(error_msg)
        if self.turns:
            self.turns[-1].error = error_msg
            
    def get_metrics(self) -> ConversationMetrics:
        """Get comprehensive metrics for this conversation."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        # Calculate participation distribution
        agent_turns = defaultdict(int)
        for turn in self.turns:
            agent_turns[turn.agent_name] += 1
            
        total_turns = len(self.turns)
        agent_participation = {
            agent: count / max(total_turns, 1)
            for agent, count in agent_turns.items()
        }
        
        # Calculate quality score
        quality_score = self._calculate_quality_score()
        
        # Calculate success rate
        error_count = len([t for t in self.turns if t.error])
        success_rate = 1.0 - (error_count / max(total_turns, 1))
        
        return ConversationMetrics(
            conversation_id=self.conversation_id,
            status=self.status,
            start_time=self.start_time,
            end_time=end_time,
            duration_seconds=duration,
            turns_count=total_turns,
            participants=list(self.participants),
            total_tokens=self.total_tokens,
            total_cost=self.total_cost,
            avg_response_time_ms=statistics.mean(self.response_times) if self.response_times else 0.0,
            cost_per_turn=self.total_cost / max(total_turns, 1),
            code_executions_count=self.code_executions,
            function_calls_count=self.function_calls,
            success_rate=success_rate,
            quality_score=quality_score,
            agent_participation=agent_participation,
            turn_distribution=dict(agent_turns),
            conversation_turns=self.turns.copy()
        )
        
    def _calculate_quality_score(self) -> float:
        """Calculate conversation quality score based on various factors."""
        if not self.turns:
            return 0.0
            
        # Factors for quality scoring
        response_time_score = self._response_time_score()
        error_penalty = len(self.errors) / max(len(self.turns), 1)
        participation_balance = self._participation_balance_score()
        efficiency_score = self._efficiency_score()
        
        # Weighted average
        quality = (
            response_time_score * 0.3 +
            (1 - error_penalty) * 0.3 +
            participation_balance * 0.2 +
            efficiency_score * 0.2
        )
        
        return max(0.0, min(1.0, quality))
        
    def _response_time_score(self) -> float:
        """Score based on response times (faster is better)."""
        if not self.response_times:
            return 0.5
            
        avg_time = statistics.mean(self.response_times)
        # Normalize to 0-1 scale (assume 5000ms is poor, 500ms is excellent)
        return max(0.0, min(1.0, (5000 - avg_time) / 4500))
        
    def _participation_balance_score(self) -> float:
        """Score based on balanced participation."""
        if len(self.participants) <= 1:
            return 1.0
            
        agent_turns = defaultdict(int)
        for turn in self.turns:
            agent_turns[turn.agent_name] += 1
            
        turn_counts = list(agent_turns.values())
        if not turn_counts:
            return 0.0
            
        # Use coefficient of variation (lower is better balanced)
        mean_turns = statistics.mean(turn_counts)
        if mean_turns == 0:
            return 0.0
            
        std_dev = statistics.stdev(turn_counts) if len(turn_counts) > 1 else 0
        cv = std_dev / mean_turns
        
        # Normalize CV to 0-1 score (0 CV = perfect balance = score 1)
        return max(0.0, min(1.0, 1.0 - cv))
        
    def _efficiency_score(self) -> float:
        """Score based on tokens per turn and function usage."""
        if not self.turns:
            return 0.0
            
        avg_tokens_per_turn = self.total_tokens / len(self.turns)
        function_usage_rate = self.function_calls / max(len(self.turns), 1)
        
        # Balance token efficiency with function usage
        token_efficiency = min(1.0, avg_tokens_per_turn / 500)  # Normalize around 500 tokens
        function_bonus = min(0.2, function_usage_rate * 0.2)  # Bonus for function usage
        
        return token_efficiency + function_bonus


class AutoGenConversationMonitor:
    """
    Comprehensive monitoring for AutoGen conversation flows and agent interactions.
    
    Tracks conversation metrics, agent performance, group chat dynamics,
    code execution patterns, and provides optimization insights.
    """
    
    def __init__(
        self,
        team: str,
        project: str,
        max_concurrent_conversations: int = 100,
        metrics_retention_hours: int = 24
    ):
        """
        Initialize conversation monitor.
        
        Args:
            team: Team name for attribution
            project: Project name for attribution
            max_concurrent_conversations: Maximum concurrent conversation trackers
            metrics_retention_hours: How long to retain detailed metrics
        """
        self.team = team
        self.project = project
        self.max_concurrent = max_concurrent_conversations
        self.retention_hours = metrics_retention_hours
        
        # Active conversation tracking
        self.active_conversations: Dict[str, ConversationTracker] = {}
        self.completed_conversations: Dict[str, ConversationMetrics] = {}
        
        # Agent performance tracking
        self.agent_metrics: Dict[str, AgentInteractionMetrics] = {}
        
        # Group chat tracking
        self.group_chat_metrics: Dict[str, GroupChatMetrics] = {}
        
        # Code execution tracking
        self.code_execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'languages_used': set(),
            'avg_execution_time_ms': 0.0,
            'error_types': defaultdict(int)
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background cleanup
        self._last_cleanup = datetime.now()
        
        logger.info(f"Initialized AutoGen conversation monitor - Team: {team}, Project: {project}")
        
    @contextmanager
    def track_conversation(self, conversation_id: str):
        """
        Context manager for tracking a conversation.
        
        Args:
            conversation_id: Unique identifier for the conversation
            
        Yields:
            ConversationTracker: Tracker for the conversation
        """
        with self._lock:
            if len(self.active_conversations) >= self.max_concurrent:
                self._cleanup_old_conversations()
                
            tracker = ConversationTracker(conversation_id, self)
            self.active_conversations[conversation_id] = tracker
            
        try:
            yield tracker
        finally:
            with self._lock:
                if conversation_id in self.active_conversations:
                    # Move to completed
                    tracker = self.active_conversations.pop(conversation_id)
                    tracker.status = ConversationStatus.COMPLETED
                    metrics = tracker.get_metrics()
                    self.completed_conversations[conversation_id] = metrics
                    
                    # Update agent metrics
                    self._update_agent_metrics(metrics)
                    
                    logger.info(f"Completed conversation tracking: {conversation_id}")
                    
    def get_conversation_analysis(self, conversation_id: str) -> Optional[ConversationMetrics]:
        """Get detailed analysis for a specific conversation."""
        with self._lock:
            # Check active conversations first
            if conversation_id in self.active_conversations:
                return self.active_conversations[conversation_id].get_metrics()
                
            # Check completed conversations
            return self.completed_conversations.get(conversation_id)
            
    def get_agent_metrics(self, agent_name: str) -> Optional[AgentInteractionMetrics]:
        """Get performance metrics for a specific agent."""
        with self._lock:
            return self.agent_metrics.get(agent_name)
            
    def get_conversation_summary(self, time_period_hours: int = 24) -> Dict[str, Any]:
        """
        Get summary of conversation activity over a time period.
        
        Args:
            time_period_hours: Time period for analysis
            
        Returns:
            Dictionary with conversation summary metrics
        """
        with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=time_period_hours)
            
            # Filter recent conversations
            recent_conversations = [
                metrics for metrics in self.completed_conversations.values()
                if metrics.start_time >= cutoff_time
            ]
            
            if not recent_conversations:
                return {
                    "total_conversations": 0,
                    "avg_duration_seconds": 0,
                    "avg_turns_per_conversation": 0,
                    "avg_cost_per_conversation": 0,
                    "success_rate": 0,
                    "quality_score": 0
                }
                
            # Calculate aggregated metrics
            total_conversations = len(recent_conversations)
            avg_duration = statistics.mean(conv.duration_seconds for conv in recent_conversations)
            avg_turns = statistics.mean(conv.turns_count for conv in recent_conversations)
            avg_cost = statistics.mean(float(conv.total_cost) for conv in recent_conversations)
            avg_success_rate = statistics.mean(conv.success_rate for conv in recent_conversations)
            avg_quality_score = statistics.mean(conv.quality_score for conv in recent_conversations)
            
            # Agent participation analysis
            agent_participation = defaultdict(int)
            for conv in recent_conversations:
                for agent in conv.participants:
                    agent_participation[agent] += 1
                    
            return {
                "time_period_hours": time_period_hours,
                "total_conversations": total_conversations,
                "avg_duration_seconds": avg_duration,
                "avg_turns_per_conversation": avg_turns,
                "avg_cost_per_conversation": avg_cost,
                "success_rate": avg_success_rate,
                "quality_score": avg_quality_score,
                "total_tokens": sum(conv.total_tokens for conv in recent_conversations),
                "total_cost": sum(float(conv.total_cost) for conv in recent_conversations),
                "code_executions": sum(conv.code_executions_count for conv in recent_conversations),
                "function_calls": sum(conv.function_calls_count for conv in recent_conversations),
                "most_active_agents": dict(sorted(agent_participation.items(), 
                                                 key=lambda x: x[1], reverse=True)[:5]),
                "active_conversations": len(self.active_conversations)
            }
            
    def _update_agent_metrics(self, conversation_metrics: ConversationMetrics):
        """Update agent metrics based on completed conversation."""
        for turn in conversation_metrics.conversation_turns:
            agent_name = turn.agent_name
            
            if agent_name not in self.agent_metrics:
                self.agent_metrics[agent_name] = AgentInteractionMetrics(
                    agent_name=agent_name,
                    total_turns=0,
                    total_tokens=0,
                    total_cost=Decimal('0'),
                    avg_response_time_ms=0.0,
                    messages_sent=0,
                    messages_received=0,
                    function_calls_made=0,
                    code_executions=0,
                    success_rate=1.0,
                    collaboration_score=0.0,
                    efficiency_score=0.0,
                    last_active=datetime.now()
                )
                
            metrics = self.agent_metrics[agent_name]
            
            # Update metrics
            metrics.total_turns += 1
            metrics.total_tokens += turn.tokens_used
            metrics.total_cost += turn.cost
            metrics.function_calls_made += len(turn.function_calls)
            if turn.code_executed:
                metrics.code_executions += 1
            metrics.last_active = turn.timestamp
            
            # Update running averages
            metrics.avg_response_time_ms = (
                (metrics.avg_response_time_ms * (metrics.total_turns - 1) + turn.response_time_ms) 
                / metrics.total_turns
            )
            
            # Update success rate
            if turn.error:
                error_rate = 1 / metrics.total_turns
                metrics.success_rate = max(0.0, metrics.success_rate - error_rate)
                
    def _cleanup_old_conversations(self):
        """Clean up old conversation data to manage memory."""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        # Remove old completed conversations
        old_conversations = [
            conv_id for conv_id, metrics in self.completed_conversations.items()
            if metrics.end_time and metrics.end_time < cutoff_time
        ]
        
        for conv_id in old_conversations:
            del self.completed_conversations[conv_id]
            
        logger.debug(f"Cleaned up {len(old_conversations)} old conversations")
        
    def export_metrics(self, format_type: str = "dict") -> Union[Dict, str]:
        """
        Export conversation metrics in various formats.
        
        Args:
            format_type: Export format ("dict", "json")
            
        Returns:
            Metrics data in requested format
        """
        with self._lock:
            data = {
                "team": self.team,
                "project": self.project,
                "active_conversations": len(self.active_conversations),
                "completed_conversations": len(self.completed_conversations),
                "total_agents_tracked": len(self.agent_metrics),
                "agent_metrics": {
                    name: {
                        "total_turns": metrics.total_turns,
                        "total_tokens": metrics.total_tokens,
                        "total_cost": str(metrics.total_cost),
                        "avg_response_time_ms": metrics.avg_response_time_ms,
                        "function_calls_made": metrics.function_calls_made,
                        "code_executions": metrics.code_executions,
                        "success_rate": metrics.success_rate,
                        "last_active": metrics.last_active.isoformat()
                    }
                    for name, metrics in self.agent_metrics.items()
                },
                "code_execution_stats": {
                    **{k: v for k, v in self.code_execution_stats.items() if k != 'languages_used'},
                    'languages_used': list(self.code_execution_stats['languages_used'])
                }
            }
            
            if format_type == "dict":
                return data
            elif format_type == "json":
                import json
                return json.dumps(data, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
    def reset_metrics(self):
        """Reset all metrics (useful for testing)."""
        with self._lock:
            self.active_conversations.clear()
            self.completed_conversations.clear()
            self.agent_metrics.clear()
            self.group_chat_metrics.clear()
            self.code_execution_stats = {
                'total_executions': 0,
                'successful_executions': 0,
                'failed_executions': 0,
                'languages_used': set(),
                'avg_execution_time_ms': 0.0,
                'error_types': defaultdict(int)
            }
            logger.info("Reset all conversation metrics")