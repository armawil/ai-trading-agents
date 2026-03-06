"""EMA crossover example strategy."""

from __future__ import annotations

import pandas as pd

from .base_strategy import Strategy


class EMACrossoverStrategy(Strategy):
    """Simple EMA crossover strategy with optional momentum threshold.

    Parameters
    ----------
    fast_length : int
        Fast EMA period.
    slow_length : int
        Slow EMA period.
    threshold : float
        Minimum percentage distance between EMAs to trigger signals.
    """

    def __init__(self, fast_length: int = 12, slow_length: int = 26, threshold: float = 0.0) -> None:
        if fast_length >= slow_length:
            raise ValueError("fast_length must be smaller than slow_length")
        self.fast_length = fast_length
        self.slow_length = slow_length
        self.threshold = threshold

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate directional signals from EMA crossover."""
        required_columns = {"close"}
        if not required_columns.issubset(data.columns):
            raise ValueError(f"Data must contain columns: {required_columns}")

        signals = pd.DataFrame(index=data.index)
        signals["ema_fast"] = data["close"].ewm(span=self.fast_length, adjust=False).mean()
        signals["ema_slow"] = data["close"].ewm(span=self.slow_length, adjust=False).mean()

        # Relative distance controls how sensitive entries are.
        ema_distance = (signals["ema_fast"] - signals["ema_slow"]) / signals["ema_slow"]

        signals["signal"] = 0
        signals.loc[ema_distance > self.threshold, "signal"] = 1
        signals.loc[ema_distance < -self.threshold, "signal"] = -1

        return signals[["signal", "ema_fast", "ema_slow"]]
