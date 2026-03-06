# NQ Futures Backtesting Engine

A lightweight Python backtesting engine focused on **Nasdaq-100 E-mini futures (NQ)** strategies.

## Features
- Contract-aware PnL for NQ (`0.25` tick size, `$5` per tick)
- Bar-based simulation with deterministic execution
- Strategy interface for pluggable trading logic
- Costs model (`commission` + `slippage` in ticks)
- Performance metrics (return, drawdown, Sharpe, win rate)

## Quick start

```python
from datetime import datetime, timedelta

from backtesting.contracts import NQ_CONTRACT
from backtesting.data import Bar
from backtesting.engine import BacktestConfig, BacktestEngine
from backtesting.strategies import MovingAverageCrossStrategy

bars = []
start = datetime(2024, 1, 1)
price = 16500.0
for i in range(120):
    drift = 1 if i > 20 else -1
    price += drift * 2
    bars.append(
        Bar(
            timestamp=start + timedelta(minutes=i),
            open=price - 0.5,
            high=price + 1.0,
            low=price - 1.0,
            close=price,
            volume=100,
        )
    )

engine = BacktestEngine(
    contract=NQ_CONTRACT,
    config=BacktestConfig(initial_capital=50_000, commission_per_side=1.25, slippage_ticks=1),
)
result = engine.run(bars, MovingAverageCrossStrategy(fast=5, slow=20))

print(result.summary)
```

## Testing

```bash
pytest -q
```
