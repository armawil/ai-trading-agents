"""Grid-search optimizer for trading strategies."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

from backtester.backtest_engine import BacktestEngine
from strategies.base_strategy import Strategy


class StrategyOptimizer:
    """Brute-force parameter optimization via cartesian grid search."""

    def __init__(self, engine: BacktestEngine, strategy_cls: type[Strategy]) -> None:
        self.engine = engine
        self.strategy_cls = strategy_cls

    def grid_search(
        self,
        data: pd.DataFrame,
        parameter_grid: dict[str, list[Any]],
        top_n: int = 10,
        sort_by: str = "sharpe_ratio",
        output_csv: str | None = None,
    ) -> pd.DataFrame:
        """Run every parameter combination and rank results."""
        param_names = list(parameter_grid.keys())
        combinations = list(product(*(parameter_grid[name] for name in param_names)))

        records = []
        for combo in combinations:
            params = dict(zip(param_names, combo))

            # Separate engine params (risk controls) from strategy params.
            strategy_kwargs = {
                k: v
                for k, v in params.items()
                if k not in {"stop_loss_pct", "take_profit_pct", "slippage_perc", "commission_per_contract"}
            }
            engine_kwargs = {k: v for k, v in params.items() if k not in strategy_kwargs}

            strategy = self.strategy_cls(**strategy_kwargs)
            local_engine = BacktestEngine(
                initial_capital=self.engine.initial_capital,
                position_size=self.engine.position_size,
                stop_loss_pct=engine_kwargs.get("stop_loss_pct", self.engine.stop_loss_pct),
                take_profit_pct=engine_kwargs.get("take_profit_pct", self.engine.take_profit_pct),
                slippage_perc=engine_kwargs.get("slippage_perc", self.engine.slippage_perc),
                commission_per_contract=engine_kwargs.get(
                    "commission_per_contract", self.engine.commission_per_contract
                ),
                session_start=self.engine.session_start,
                session_end=self.engine.session_end,
                session_tz=self.engine.session_tz,
            )

            signals = strategy.generate_signals(data)
            _, _, metrics = local_engine.run(data, signals)

            record = {**params, **metrics}
            records.append(record)

        results = pd.DataFrame(records)
        results = results.sort_values(sort_by, ascending=False).reset_index(drop=True)

        if output_csv:
            Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(output_csv, index=False)

        return results.head(top_n) if top_n else results
