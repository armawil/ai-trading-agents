"""Simple backtest runner skeleton."""

from dataclasses import dataclass

from ai_trading_research.strategies.base_strategy import BaseStrategy


@dataclass
class BacktestRunner:
    """Coordinates strategy evaluation over historical datasets."""

    strategy: BaseStrategy

    def run(self) -> dict[str, float]:
        """Run a placeholder backtest and return stub metrics."""
        return {
            "strategy": self.strategy.name,
            "return_pct": 0.0,
            "max_drawdown_pct": 0.0,
        }
