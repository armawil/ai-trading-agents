"""Monte Carlo robustness simulations for trade-level results."""

from __future__ import annotations

import numpy as np
import pandas as pd


class MonteCarloSimulator:
    """Generates randomized equity trajectories from historical trades."""

    def __init__(self, initial_capital: float = 100_000.0) -> None:
        self.initial_capital = initial_capital

    def simulate(
        self,
        trade_pnls: list[float],
        n_simulations: int = 5_000,
        method: str = "bootstrap",
        ruin_threshold: float = 0.7,
        confidence: float = 0.95,
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        """Run Monte Carlo simulation and return equity paths and summary stats."""
        if not trade_pnls:
            raise ValueError("trade_pnls must not be empty")

        pnl_array = np.array(trade_pnls, dtype=float)
        n_trades = len(pnl_array)
        equity_curves = np.zeros((n_simulations, n_trades + 1), dtype=float)
        equity_curves[:, 0] = self.initial_capital

        for i in range(n_simulations):
            if method == "random_order":
                sampled = np.random.permutation(pnl_array)
            elif method == "bootstrap":
                sampled = np.random.choice(pnl_array, size=n_trades, replace=True)
            else:
                raise ValueError("method must be 'random_order' or 'bootstrap'")

            equity_curves[i, 1:] = self.initial_capital + np.cumsum(sampled)

        curve_df = pd.DataFrame(equity_curves.T)

        ending_equity = equity_curves[:, -1]
        running_max = np.maximum.accumulate(equity_curves, axis=1)
        drawdowns = equity_curves / running_max - 1.0
        max_dd_per_curve = drawdowns.min(axis=1)

        ruin_level = self.initial_capital * ruin_threshold
        probability_of_ruin = float((equity_curves.min(axis=1) <= ruin_level).mean())

        alpha = 1.0 - confidence
        lower_ci = float(np.quantile(ending_equity, alpha / 2))
        upper_ci = float(np.quantile(ending_equity, 1 - alpha / 2))

        stats = {
            "probability_of_ruin": probability_of_ruin,
            "expected_drawdown": float(max_dd_per_curve.mean()),
            "ending_equity_mean": float(ending_equity.mean()),
            "ending_equity_median": float(np.median(ending_equity)),
            "ending_equity_ci_lower": lower_ci,
            "ending_equity_ci_upper": upper_ci,
        }

        return curve_df, stats
