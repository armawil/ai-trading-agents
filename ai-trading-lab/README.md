# AI Trading Lab (NQ Futures Research Platform)

A modular quantitative trading research environment focused on **Nasdaq futures (NQ)** strategy development, backtesting, optimization, and robustness testing.

## Project Structure

```text
ai-trading-lab/
│
├── backtester/
│   └── backtest_engine.py
├── strategies/
│   ├── base_strategy.py
│   └── ema_strategy.py
├── optimizer/
│   └── strategy_optimizer.py
├── walkforward/
│   └── walk_forward.py
├── montecarlo/
│   └── monte_carlo.py
├── visualization/
│   └── stability_heatmap.py
├── data/
│   └── nq_sample_data.csv
├── results/
├── requirements.txt
└── README.md
```

## Features

### 1) NQ Futures Backtester
- Event-driven trade loop using OHLCV bars.
- Long and short trade support.
- Stop loss and take profit controls.
- Slippage and commission modeling.
- Session filter (e.g. trade only 7:30–10:30 New York time).
- Performance metrics:
  - total return
  - Sharpe ratio
  - max drawdown
  - profit factor
  - win rate

### 2) Strategy Optimizer
- Grid-search parameter sweeps.
- Supports strategy parameters and risk controls (SL/TP/slippage/commission).
- Saves optimization table to CSV.
- Ranks combinations by selectable metric (default Sharpe ratio).

### 3) Walk-Forward Testing
- Rolling train/test windows.
- Optimize on in-sample data.
- Validate on unseen out-of-sample windows.
- Fold-level performance summary output.

### 4) Monte Carlo Simulation
- Randomized trade order simulation.
- Bootstrap sampling simulation.
- Thousands of simulated equity trajectories.
- Robustness outputs:
  - probability of ruin
  - expected drawdown
  - confidence interval for ending equity

### 5) Parameter Stability Maps
- Heatmap of sensitivity across parameter pairs.
- Useful for identifying stable vs overfit parameter regions.

### 6) Strategy Interface
- Common base strategy contract via `Strategy.generate_signals(data)`.
- Example strategy included: EMA crossover.

### 7) Data Handling
- CSV loader expects: `datetime, open, high, low, close, volume`.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from datetime import time

from backtester.backtest_engine import BacktestEngine
from strategies.ema_strategy import EMACrossoverStrategy
from optimizer.strategy_optimizer import StrategyOptimizer
from walkforward.walk_forward import WalkForwardAnalyzer
from montecarlo.monte_carlo import MonteCarloSimulator
from visualization.stability_heatmap import plot_stability_heatmap

# 1) Load data and run a baseline backtest
engine = BacktestEngine(
    initial_capital=100_000,
    stop_loss_pct=0.004,
    take_profit_pct=0.008,
    slippage_perc=0.0001,
    commission_per_contract=2.5,
    session_start=time(7, 30),
    session_end=time(10, 30),
)

data = engine.load_data("data/nq_sample_data.csv")
strategy = EMACrossoverStrategy(fast_length=12, slow_length=26, threshold=0.0002)
signals = strategy.generate_signals(data)
equity_curve, trades, metrics = engine.run(data, signals)
print("Baseline metrics:", metrics)

# 2) Optimize parameters
optimizer = StrategyOptimizer(engine, EMACrossoverStrategy)
param_grid = {
    "fast_length": [8, 12, 16],
    "slow_length": [24, 30, 40],
    "threshold": [0.0, 0.0002, 0.0005],
    "stop_loss_pct": [0.003, 0.004, 0.006],
    "take_profit_pct": [0.006, 0.008, 0.012],
}
results = optimizer.grid_search(data, param_grid, top_n=20, output_csv="results/optimization_results.csv")
print(results.head())

# 3) Walk-forward analysis
wfa = WalkForwardAnalyzer(engine, EMACrossoverStrategy)
wf_summary, wf_curves = wfa.run(data, param_grid, train_bars=400, test_bars=120)
print(wf_summary)

# 4) Monte Carlo on realized trade PnL
trade_pnls = [t.pnl for t in trades]
if trade_pnls:
    mc = MonteCarloSimulator(initial_capital=100_000)
    mc_curves, mc_stats = mc.simulate(trade_pnls, n_simulations=3000, method="bootstrap")
    print(mc_stats)

# 5) Stability heatmap example
plot_stability_heatmap(
    optimization_results=results,
    x_param="fast_length",
    y_param="stop_loss_pct",
    metric="sharpe_ratio",
    output_path="results/stability_map.png",
)
```

## Notes for Extension
- Add new strategies by inheriting from `strategies.base_strategy.Strategy`.
- Add new metrics in `BacktestEngine.calculate_metrics`.
- Add additional robustness tests alongside Monte Carlo module.

