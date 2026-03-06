"""Walk-forward analysis for validating strategy generalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from backtester.backtest_engine import BacktestEngine
from optimizer.strategy_optimizer import StrategyOptimizer
from strategies.base_strategy import Strategy


@dataclass
class WalkForwardResult:
    """Container for each train/test fold output."""

    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict[str, Any]
    train_sharpe: float
    test_sharpe: float
    test_total_return: float
    test_max_drawdown: float


class WalkForwardAnalyzer:
    """Performs rolling optimization and out-of-sample evaluation."""

    def __init__(self, engine: BacktestEngine, strategy_cls: type[Strategy]) -> None:
        self.engine = engine
        self.strategy_cls = strategy_cls

    def run(
        self,
        data: pd.DataFrame,
        parameter_grid: dict[str, list[Any]],
        train_bars: int,
        test_bars: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run walk-forward cycles across the dataset."""
        results: list[WalkForwardResult] = []
        test_curves = []

        fold = 0
        for start in range(0, len(data) - train_bars - test_bars + 1, test_bars):
            fold += 1
            train_data = data.iloc[start : start + train_bars]
            test_data = data.iloc[start + train_bars : start + train_bars + test_bars]

            optimizer = StrategyOptimizer(self.engine, self.strategy_cls)
            ranked = optimizer.grid_search(train_data, parameter_grid, top_n=1)
            best_row = ranked.iloc[0].to_dict()

            best_params = {k: best_row[k] for k in parameter_grid.keys()}
            strategy_kwargs = {
                k: v for k, v in best_params.items() if k not in {"stop_loss_pct", "take_profit_pct"}
            }
            local_engine = BacktestEngine(
                initial_capital=self.engine.initial_capital,
                position_size=self.engine.position_size,
                stop_loss_pct=best_params.get("stop_loss_pct", self.engine.stop_loss_pct),
                take_profit_pct=best_params.get("take_profit_pct", self.engine.take_profit_pct),
                slippage_perc=self.engine.slippage_perc,
                commission_per_contract=self.engine.commission_per_contract,
                session_start=self.engine.session_start,
                session_end=self.engine.session_end,
                session_tz=self.engine.session_tz,
            )

            strategy = self.strategy_cls(**strategy_kwargs)
            signals = strategy.generate_signals(test_data)
            test_curve, _, test_metrics = local_engine.run(test_data, signals)
            test_curve = test_curve.rename(columns={"equity": f"equity_fold_{fold}"})
            test_curves.append(test_curve)

            results.append(
                WalkForwardResult(
                    fold=fold,
                    train_start=train_data.index[0],
                    train_end=train_data.index[-1],
                    test_start=test_data.index[0],
                    test_end=test_data.index[-1],
                    best_params=best_params,
                    train_sharpe=float(best_row.get("sharpe_ratio", 0.0)),
                    test_sharpe=test_metrics["sharpe_ratio"],
                    test_total_return=test_metrics["total_return"],
                    test_max_drawdown=test_metrics["max_drawdown"],
                )
            )

        summary = pd.DataFrame([r.__dict__ for r in results])
        combined_curve = pd.concat(test_curves, axis=1) if test_curves else pd.DataFrame()

        return summary, combined_curve
