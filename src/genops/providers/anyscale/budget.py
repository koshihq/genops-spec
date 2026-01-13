"""Budget constraint enforcement for Anyscale operations."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, Callable

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """Raised when a budget limit would be exceeded."""
    pass


@dataclass
class BudgetPeriod:
    """Represents a budget period with usage tracking."""

    period_type: str  # 'hourly', 'daily', 'weekly', 'monthly'
    limit_usd: float
    current_usage: float = 0.0
    period_start: float = field(default_factory=time.time)
    request_count: int = 0

    def is_expired(self) -> bool:
        """Check if budget period has expired."""
        now = time.time()
        elapsed_seconds = now - self.period_start

        if self.period_type == 'hourly':
            return elapsed_seconds > 3600
        elif self.period_type == 'daily':
            return elapsed_seconds > 86400
        elif self.period_type == 'weekly':
            return elapsed_seconds > 604800
        elif self.period_type == 'monthly':
            return elapsed_seconds > 2592000  # 30 days
        return False

    def reset(self) -> None:
        """Reset budget period."""
        self.current_usage = 0.0
        self.period_start = time.time()
        self.request_count = 0

    def get_remaining(self) -> float:
        """Get remaining budget."""
        return max(0.0, self.limit_usd - self.current_usage)

    def get_usage_percentage(self) -> float:
        """Get usage as percentage of limit."""
        if self.limit_usd <= 0:
            return 0.0
        return (self.current_usage / self.limit_usd) * 100


class BudgetManager:
    """
    Manage budget constraints for Anyscale operations.

    Supports multiple budget periods (hourly, daily, weekly, monthly)
    with proactive enforcement and alerting.
    """

    def __init__(
        self,
        hourly_limit_usd: Optional[float] = None,
        daily_limit_usd: Optional[float] = None,
        weekly_limit_usd: Optional[float] = None,
        monthly_limit_usd: Optional[float] = None,
        alert_thresholds: Optional[list[float]] = None,
        alert_callback: Optional[Callable[[str, float, float], None]] = None
    ):
        """
        Initialize budget manager.

        Args:
            hourly_limit_usd: Hourly budget limit in USD
            daily_limit_usd: Daily budget limit in USD
            weekly_limit_usd: Weekly budget limit in USD
            monthly_limit_usd: Monthly budget limit in USD
            alert_thresholds: Budget usage percentages to trigger alerts (e.g., [0.5, 0.75, 0.9])
            alert_callback: Function to call when alert threshold reached
                            Signature: callback(period_type: str, usage_pct: float, limit: float)
        """
        self.periods: Dict[str, BudgetPeriod] = {}

        if hourly_limit_usd:
            self.periods['hourly'] = BudgetPeriod('hourly', hourly_limit_usd)
        if daily_limit_usd:
            self.periods['daily'] = BudgetPeriod('daily', daily_limit_usd)
        if weekly_limit_usd:
            self.periods['weekly'] = BudgetPeriod('weekly', weekly_limit_usd)
        if monthly_limit_usd:
            self.periods['monthly'] = BudgetPeriod('monthly', monthly_limit_usd)

        # Alert configuration
        self.alert_thresholds = alert_thresholds or [0.5, 0.75, 0.9, 1.0]
        self.alert_callback = alert_callback
        self._alerts_sent: Dict[str, set[float]] = {
            period: set() for period in self.periods.keys()
        }

        # Total lifetime tracking
        self.total_lifetime_cost = 0.0
        self.total_lifetime_requests = 0

        logger.info(f"Budget manager initialized with periods: {list(self.periods.keys())}")

    def _check_and_reset_expired_periods(self) -> None:
        """Check and reset any expired budget periods."""
        for period_type, period in self.periods.items():
            if period.is_expired():
                logger.info(
                    f"Budget period '{period_type}' expired. "
                    f"Final usage: ${period.current_usage:.6f}/${period.limit_usd:.2f}"
                )
                period.reset()
                self._alerts_sent[period_type].clear()

    def check_budget_availability(self, estimated_cost: float) -> tuple[bool, Optional[str]]:
        """
        Check if estimated cost would exceed any budget limits.

        Args:
            estimated_cost: Estimated cost in USD for the operation

        Returns:
            Tuple of (allowed: bool, reason: Optional[str])
        """
        self._check_and_reset_expired_periods()

        for period_type, period in self.periods.items():
            new_usage = period.current_usage + estimated_cost

            if new_usage > period.limit_usd:
                return False, (
                    f"{period_type.capitalize()} budget would be exceeded: "
                    f"${new_usage:.6f} > ${period.limit_usd:.2f} "
                    f"(current: ${period.current_usage:.6f}, operation: ${estimated_cost:.6f})"
                )

        return True, None

    def record_cost(self, actual_cost: float) -> None:
        """
        Record actual cost and update all budget periods.

        Args:
            actual_cost: Actual cost in USD
        """
        self._check_and_reset_expired_periods()

        # Update all periods
        for period_type, period in self.periods.items():
            period.current_usage += actual_cost
            period.request_count += 1

            # Check alert thresholds
            usage_pct = period.get_usage_percentage()
            for threshold in self.alert_thresholds:
                threshold_pct = threshold * 100

                # Check if we've crossed this threshold and haven't sent alert yet
                if usage_pct >= threshold_pct and threshold not in self._alerts_sent[period_type]:
                    self._send_alert(period_type, usage_pct, period.limit_usd)
                    self._alerts_sent[period_type].add(threshold)

        # Update lifetime tracking
        self.total_lifetime_cost += actual_cost
        self.total_lifetime_requests += 1

        logger.debug(f"Recorded cost: ${actual_cost:.8f}")

    def _send_alert(self, period_type: str, usage_pct: float, limit: float) -> None:
        """Send budget alert."""
        message = (
            f"Budget Alert: {period_type} usage at {usage_pct:.1f}% "
            f"(limit: ${limit:.2f})"
        )

        logger.warning(message)

        if self.alert_callback:
            try:
                self.alert_callback(period_type, usage_pct, limit)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def get_budget_status(self) -> Dict:
        """
        Get current budget status for all periods.

        Returns:
            Dict with budget status for each period
        """
        self._check_and_reset_expired_periods()

        status = {
            'periods': {},
            'lifetime': {
                'total_cost': self.total_lifetime_cost,
                'total_requests': self.total_lifetime_requests,
                'avg_cost_per_request': (
                    self.total_lifetime_cost / self.total_lifetime_requests
                    if self.total_lifetime_requests > 0 else 0.0
                )
            }
        }

        for period_type, period in self.periods.items():
            status['periods'][period_type] = {
                'limit': period.limit_usd,
                'current_usage': period.current_usage,
                'remaining': period.get_remaining(),
                'usage_percentage': period.get_usage_percentage(),
                'request_count': period.request_count,
                'period_start': datetime.fromtimestamp(period.period_start).isoformat()
            }

        return status

    def print_budget_status(self) -> None:
        """Print formatted budget status."""
        status = self.get_budget_status()

        print("\n" + "=" * 70)
        print("BUDGET STATUS")
        print("=" * 70)

        # Period budgets
        for period_type, period_status in status['periods'].items():
            print(f"\n{period_type.upper()} BUDGET:")
            print(f"   Limit:      ${period_status['limit']:.2f}")
            print(f"   Used:       ${period_status['current_usage']:.6f}")
            print(f"   Remaining:  ${period_status['remaining']:.6f}")
            print(f"   Usage:      {period_status['usage_percentage']:.1f}%")
            print(f"   Requests:   {period_status['request_count']}")

            # Visual progress bar
            usage_pct = period_status['usage_percentage']
            bar_length = 40
            filled = int(bar_length * usage_pct / 100)
            bar = "█" * filled + "░" * (bar_length - filled)

            # Color indicator
            if usage_pct >= 90:
                indicator = "🔴"
            elif usage_pct >= 75:
                indicator = "🟡"
            else:
                indicator = "🟢"

            print(f"   {indicator} [{bar}] {usage_pct:.1f}%")

        # Lifetime stats
        print(f"\nLIFETIME STATS:")
        print(f"   Total Cost:  ${status['lifetime']['total_cost']:.6f}")
        print(f"   Requests:    {status['lifetime']['total_requests']}")
        print(f"   Avg/Request: ${status['lifetime']['avg_cost_per_request']:.8f}")

        print("=" * 70 + "\n")

    def reset_period(self, period_type: str) -> None:
        """
        Manually reset a specific budget period.

        Args:
            period_type: Type of period to reset ('hourly', 'daily', 'weekly', 'monthly')
        """
        if period_type in self.periods:
            self.periods[period_type].reset()
            self._alerts_sent[period_type].clear()
            logger.info(f"Manually reset {period_type} budget period")
        else:
            raise ValueError(f"Unknown period type: {period_type}")

    def set_alert_callback(self, callback: Callable[[str, float, float], None]) -> None:
        """
        Set or update the alert callback function.

        Args:
            callback: Function to call when alert threshold reached
        """
        self.alert_callback = callback
        logger.info("Budget alert callback updated")


def create_budget_manager(
    hourly_limit: Optional[float] = None,
    daily_limit: Optional[float] = None,
    weekly_limit: Optional[float] = None,
    monthly_limit: Optional[float] = None,
    **kwargs
) -> BudgetManager:
    """
    Factory function to create a budget manager.

    Args:
        hourly_limit: Hourly budget limit in USD
        daily_limit: Daily budget limit in USD
        weekly_limit: Weekly budget limit in USD
        monthly_limit: Monthly budget limit in USD
        **kwargs: Additional arguments for BudgetManager

    Returns:
        BudgetManager instance
    """
    return BudgetManager(
        hourly_limit_usd=hourly_limit,
        daily_limit_usd=daily_limit,
        weekly_limit_usd=weekly_limit,
        monthly_limit_usd=monthly_limit,
        **kwargs
    )


# Export public API
__all__ = [
    'BudgetManager',
    'BudgetPeriod',
    'BudgetExceededError',
    'create_budget_manager',
]
