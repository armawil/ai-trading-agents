from dataclasses import dataclass

from .contracts import FuturesContract
from .data import Bar, Side
from .metrics import max_drawdown, sharpe_ratio
from .strategy import Strategy


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 50_000.0
    commission_per_side: float = 1.25
    slippage_ticks: int = 1


@dataclass(frozen=True)
class Trade:
    entry_price: float
    exit_price: float
    qty: int
    side: Side
    pnl: float


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: list[float]
    trades: list[Trade]
    summary: dict[str, float]


class BacktestEngine:
    def __init__(self, contract: FuturesContract, config: BacktestConfig | None = None):
        self.contract = contract
        self.config = config or BacktestConfig()

    def _execution_price(self, bar: Bar, side: Side) -> float:
        slip = self.config.slippage_ticks * self.contract.tick_size
        if side == Side.LONG:
            return bar.open + slip
        if side == Side.SHORT:
            return bar.open - slip
        return bar.open

    def run(self, bars: list[Bar], strategy: Strategy) -> BacktestResult:
        if len(bars) < 2:
            raise ValueError("Need at least two bars to backtest.")

        equity = self.config.initial_capital
        equity_curve = [equity]
        position_side = Side.FLAT
        position_qty = 0
        entry_price = 0.0
        trades: list[Trade] = []

        for i in range(len(bars) - 1):
            bar = bars[i]
            next_bar = bars[i + 1]
            signal = strategy.on_bar(bar)

            if signal is not None and signal.side != position_side:
                if position_side != Side.FLAT:
                    exit_price = self._execution_price(next_bar, Side.SHORT if position_side == Side.LONG else Side.LONG)
                    direction = 1 if position_side == Side.LONG else -1
                    gross = self.contract.pnl_from_price_delta((exit_price - entry_price) * direction, position_qty)
                    costs = 2 * self.config.commission_per_side * position_qty
                    net = gross - costs
                    equity += net
                    trades.append(
                        Trade(
                            entry_price=entry_price,
                            exit_price=exit_price,
                            qty=position_qty,
                            side=position_side,
                            pnl=net,
                        )
                    )
                    position_side = Side.FLAT
                    position_qty = 0

                if signal.side != Side.FLAT and signal.qty > 0:
                    entry_price = self._execution_price(next_bar, signal.side)
                    position_side = signal.side
                    position_qty = signal.qty
                    equity -= self.config.commission_per_side * position_qty

            mark_to_market = equity
            if position_side != Side.FLAT:
                direction = 1 if position_side == Side.LONG else -1
                unrealized = self.contract.pnl_from_price_delta((bar.close - entry_price) * direction, position_qty)
                mark_to_market += unrealized

            equity_curve.append(mark_to_market)

        if position_side != Side.FLAT:
            last = bars[-1]
            exit_price = last.close
            direction = 1 if position_side == Side.LONG else -1
            gross = self.contract.pnl_from_price_delta((exit_price - entry_price) * direction, position_qty)
            costs = self.config.commission_per_side * position_qty
            net = gross - costs
            equity += net
            trades.append(
                Trade(
                    entry_price=entry_price,
                    exit_price=exit_price,
                    qty=position_qty,
                    side=position_side,
                    pnl=net,
                )
            )
            equity_curve[-1] = equity

        total_return = (equity - self.config.initial_capital) / self.config.initial_capital
        period_returns: list[float] = []
        for idx in range(1, len(equity_curve)):
            prev = equity_curve[idx - 1]
            curr = equity_curve[idx]
            period_returns.append((curr - prev) / prev if prev else 0.0)

        wins = sum(1 for t in trades if t.pnl > 0)
        summary = {
            "initial_capital": self.config.initial_capital,
            "ending_equity": equity,
            "total_return": total_return,
            "max_drawdown": max_drawdown(equity_curve),
            "sharpe": sharpe_ratio(period_returns),
            "trades": float(len(trades)),
            "win_rate": (wins / len(trades)) if trades else 0.0,
        }
        return BacktestResult(equity_curve=equity_curve, trades=trades, summary=summary)
