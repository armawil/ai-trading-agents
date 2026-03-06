"""Modular strategy optimizer with backtesting + grid search.

Usage:
    python optimizers/strategy_optimizer.py \
        --data data/sample_price_data.csv \
        --strategy strategies.example_strategy \
        --output optimization_results.csv
"""

from __future__ import annotations

import argparse
import csv
import importlib
import itertools
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_prices(csv_path: Path) -> list[float]:
    prices: list[float] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "close" not in (reader.fieldnames or []):
            raise ValueError("Price CSV must contain a 'close' column")
        for row in reader:
            prices.append(float(row["close"]))
    if len(prices) < 2:
        raise ValueError("Need at least 2 rows of price data")
    return prices


def daily_returns(prices: list[float]) -> list[float]:
    return [(prices[i] / prices[i - 1]) - 1 for i in range(1, len(prices))]


def backtest(prices: list[float], positions: list[int]) -> dict[str, float]:
    if len(positions) != len(prices):
        raise ValueError("positions length must match prices length")

    market_rets = daily_returns(prices)
    strategy_rets = [market_rets[i] * positions[i] for i in range(len(market_rets))]

    equity = [1.0]
    for r in strategy_rets:
        equity.append(equity[-1] * (1 + r))

    total_return = equity[-1] - 1

    mean_ret = sum(strategy_rets) / len(strategy_rets) if strategy_rets else 0.0
    std_ret = (
        math.sqrt(sum((r - mean_ret) ** 2 for r in strategy_rets) / len(strategy_rets))
        if strategy_rets
        else 0.0
    )
    sharpe_ratio = (mean_ret / std_ret) * math.sqrt(252) if std_ret > 0 else 0.0

    gross_profit = sum(r for r in strategy_rets if r > 0)
    gross_loss = abs(sum(r for r in strategy_rets if r < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    peak = equity[0]
    max_drawdown = 0.0
    for value in equity:
        peak = max(peak, value)
        drawdown = (value / peak) - 1
        max_drawdown = min(max_drawdown, drawdown)

    return {
        "sharpe_ratio": sharpe_ratio,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "total_return": total_return,
    }


def grid_search(parameter_space: dict[str, list[Any]]):
    keys = list(parameter_space.keys())
    values = [parameter_space[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def optimize(prices: list[float], strategy_module: Any) -> tuple[dict[str, Any], dict[str, float], list[dict[str, Any]]]:
    parameter_space = strategy_module.parameter_grid()
    all_results: list[dict[str, Any]] = []

    best_params: dict[str, Any] | None = None
    best_metrics: dict[str, float] | None = None
    best_score = float("-inf")

    for params in grid_search(parameter_space):
        positions = strategy_module.generate_positions(prices, params)
        metrics = backtest(prices, positions)

        # Score by total return first, then Sharpe as tie-breaker.
        score = metrics["total_return"] + 0.001 * metrics["sharpe_ratio"]

        row = {**params, **metrics}
        all_results.append(row)

        if score > best_score:
            best_score = score
            best_params = params
            best_metrics = metrics

    if best_params is None or best_metrics is None:
        raise RuntimeError("No optimization results generated")

    return best_params, best_metrics, all_results


def write_results(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No rows to save")

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize trading strategy parameters via grid search")
    parser.add_argument("--data", type=Path, default=Path("data/sample_price_data.csv"), help="Path to price CSV")
    parser.add_argument(
        "--strategy",
        default="strategies.example_strategy",
        help="Python module path for strategy (e.g. strategies.example_strategy)",
    )
    parser.add_argument("--output", type=Path, default=Path("optimization_results.csv"), help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strategy_module = importlib.import_module(args.strategy)

    prices = load_prices(args.data)
    best_params, best_metrics, all_results = optimize(prices, strategy_module)
    write_results(all_results, args.output)

    print("Best parameter set:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    print("\nPerformance metrics:")
    for k, v in best_metrics.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    print(f"\nSaved optimization results to: {args.output}")


if __name__ == "__main__":
    main()
