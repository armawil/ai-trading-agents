#!/usr/bin/env python3
"""Example script for running a placeholder backtest."""

from ai_trading_research.backtests.runner import BacktestRunner
from ai_trading_research.strategies.base_strategy import BaseStrategy


def main() -> None:
    strategy = BaseStrategy(name="demo-strategy")
    metrics = BacktestRunner(strategy=strategy).run()
    print(metrics)


if __name__ == "__main__":
    main()
