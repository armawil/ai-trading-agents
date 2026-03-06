"""Event-driven backtesting engine for OHLCV futures data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Trade:
    """Stores a completed trade for analysis and simulation."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    return_pct: float
    reason: str


class BacktestEngine:
    """Event-driven execution model for long/short strategy testing."""

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        position_size: float = 1.0,
        stop_loss_pct: float = 0.005,
        take_profit_pct: float = 0.01,
        slippage_perc: float = 0.0001,
        commission_per_contract: float = 2.5,
        session_start: Optional[time] = None,
        session_end: Optional[time] = None,
        session_tz: str = "America/New_York",
    ) -> None:
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.slippage_perc = slippage_perc
        self.commission_per_contract = commission_per_contract
        self.session_start = session_start
        self.session_end = session_end
        self.session_tz = session_tz

    @staticmethod
    def load_data(csv_path: str) -> pd.DataFrame:
        """Load CSV with datetime, open, high, low, close, volume columns."""
        data = pd.read_csv(csv_path)
        data.columns = [col.lower() for col in data.columns]
        required = ["datetime", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        data["datetime"] = pd.to_datetime(data["datetime"], utc=True)
        data = data.set_index("datetime").sort_index()
        return data

    def _is_in_session(self, timestamp: pd.Timestamp) -> bool:
        if self.session_start is None or self.session_end is None:
            return True
        local_time = timestamp.tz_convert(self.session_tz).time()
        return self.session_start <= local_time <= self.session_end

    def run(self, data: pd.DataFrame, signals: pd.DataFrame) -> tuple[pd.DataFrame, list[Trade], dict[str, float]]:
        """Run an event-driven simulation and return equity curve, trades, metrics."""
        if "signal" not in signals.columns:
            raise ValueError("signals DataFrame must contain a 'signal' column")

        merged = data.join(signals[["signal"]], how="left").fillna({"signal": 0})

        cash = self.initial_capital
        equity = self.initial_capital
        position = 0
        entry_price = np.nan
        entry_time = None
        trades: list[Trade] = []
        curve_records = []

        for timestamp, row in merged.iterrows():
            signal = int(np.sign(row["signal"]))
            open_price, high_price, low_price, close_price = row[["open", "high", "low", "close"]]

            # Mark-to-market using close.
            if position != 0:
                unrealized = position * self.position_size * (close_price - entry_price)
                equity = cash + unrealized
            else:
                equity = cash

            if position == 0 and signal != 0 and self._is_in_session(timestamp):
                fill_price = open_price * (1 + self.slippage_perc * signal)
                entry_price = fill_price
                entry_time = timestamp
                position = signal
                cash -= self.commission_per_contract * self.position_size

            elif position != 0:
                stop_price = entry_price * (1 - self.stop_loss_pct * position)
                target_price = entry_price * (1 + self.take_profit_pct * position)

                stop_hit = low_price <= stop_price if position == 1 else high_price >= stop_price
                target_hit = high_price >= target_price if position == 1 else low_price <= target_price
                reverse_signal = signal == -position and self._is_in_session(timestamp)

                exit_reason = None
                if stop_hit:
                    exit_reason = "stop_loss"
                    raw_exit = stop_price
                elif target_hit:
                    exit_reason = "take_profit"
                    raw_exit = target_price
                elif reverse_signal:
                    exit_reason = "signal_flip"
                    raw_exit = open_price
                elif not self._is_in_session(timestamp):
                    exit_reason = "session_end"
                    raw_exit = close_price

                if exit_reason is not None:
                    fill_price = raw_exit * (1 - self.slippage_perc * position)
                    pnl = position * self.position_size * (fill_price - entry_price)
                    gross_notional = abs(entry_price * self.position_size)
                    trade_return = pnl / gross_notional if gross_notional else 0.0
                    cash += pnl
                    cash -= self.commission_per_contract * self.position_size

                    trades.append(
                        Trade(
                            entry_time=entry_time,
                            exit_time=timestamp,
                            direction=position,
                            entry_price=float(entry_price),
                            exit_price=float(fill_price),
                            size=self.position_size,
                            pnl=float(pnl - (2 * self.commission_per_contract * self.position_size)),
                            return_pct=float(trade_return),
                            reason=exit_reason,
                        )
                    )
                    position = 0
                    entry_price = np.nan
                    entry_time = None
                    equity = cash

            curve_records.append({"datetime": timestamp, "equity": equity})

        equity_curve = pd.DataFrame(curve_records).set_index("datetime")
        metrics = self.calculate_metrics(equity_curve, trades)
        return equity_curve, trades, metrics

    def calculate_metrics(self, equity_curve: pd.DataFrame, trades: list[Trade]) -> dict[str, float]:
        """Calculate core performance metrics for strategy evaluation."""
        if equity_curve.empty:
            return {
                "total_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "profit_factor": 0.0,
                "win_rate": 0.0,
                "num_trades": 0,
            }

        returns = equity_curve["equity"].pct_change().dropna()
        sharpe = 0.0
        if not returns.empty and returns.std() > 0:
            sharpe = np.sqrt(252 * 24 * 60) * returns.mean() / returns.std()

        running_max = equity_curve["equity"].cummax()
        drawdown = equity_curve["equity"] / running_max - 1.0
        max_drawdown = drawdown.min() if not drawdown.empty else 0.0

        total_return = equity_curve["equity"].iloc[-1] / self.initial_capital - 1

        pnls = np.array([t.pnl for t in trades], dtype=float)
        gross_profit = pnls[pnls > 0].sum() if pnls.size else 0.0
        gross_loss = abs(pnls[pnls < 0].sum()) if pnls.size else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf if gross_profit > 0 else 0.0

        wins = (pnls > 0).sum() if pnls.size else 0
        win_rate = wins / len(trades) if trades else 0.0

        return {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "profit_factor": float(profit_factor),
            "win_rate": float(win_rate),
            "num_trades": len(trades),
        }
