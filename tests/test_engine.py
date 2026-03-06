from datetime import datetime, timedelta

from backtesting.contracts import NQ_CONTRACT
from backtesting.data import Bar, Side
from backtesting.engine import BacktestConfig, BacktestEngine
from backtesting.metrics import max_drawdown, sharpe_ratio
from backtesting.strategies import MovingAverageCrossStrategy


def make_bars() -> list[Bar]:
    start = datetime(2024, 1, 1)
    price = 16_000.0
    bars = []
    for i in range(140):
        if i < 40:
            price -= 2.0
        elif i < 90:
            price += 3.0
        else:
            price -= 2.5
        bars.append(
            Bar(
                timestamp=start + timedelta(minutes=i),
                open=price - 0.25,
                high=price + 1.0,
                low=price - 1.0,
                close=price,
                volume=1000,
            )
        )
    return bars


def test_backtest_runs_and_produces_trades() -> None:
    engine = BacktestEngine(NQ_CONTRACT, BacktestConfig(initial_capital=100_000, commission_per_side=1.25, slippage_ticks=1))
    strategy = MovingAverageCrossStrategy(fast=5, slow=20, qty=1)

    result = engine.run(make_bars(), strategy)

    assert len(result.trades) >= 1
    assert result.summary["initial_capital"] == 100_000
    assert 0 <= result.summary["max_drawdown"] <= 1
    assert result.summary["sharpe"] == result.summary["sharpe"]  # not NaN


def test_contract_pnl_conversion() -> None:
    # 1 point move in NQ = 4 ticks * $5 = $20 per contract
    pnl = NQ_CONTRACT.pnl_from_price_delta(1.0, 1)
    assert pnl == 20.0


def test_metrics_utilities() -> None:
    equity = [100, 110, 90, 120]
    assert round(max_drawdown(equity), 4) == round((110 - 90) / 110, 4)

    sr = sharpe_ratio([0.01, -0.005, 0.02, -0.01])
    assert isinstance(sr, float)


def test_strategy_validation() -> None:
    try:
        MovingAverageCrossStrategy(fast=20, slow=10)
        assert False, "Expected ValueError"
    except ValueError:
        assert True


def test_side_values_stable() -> None:
    assert Side.LONG.value == "long"
    assert Side.SHORT.value == "short"
