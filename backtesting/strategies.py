from collections import deque

from .data import Bar, Signal, Side
from .strategy import Strategy


class MovingAverageCrossStrategy(Strategy):
    def __init__(self, fast: int = 10, slow: int = 30, qty: int = 1):
        if fast >= slow:
            raise ValueError("fast period must be lower than slow period")
        self.fast = fast
        self.slow = slow
        self.qty = qty
        self.window = deque(maxlen=slow)
        self.prev_fast = None
        self.prev_slow = None

    def _sma(self, n: int) -> float:
        values = list(self.window)[-n:]
        return sum(values) / len(values)

    def on_bar(self, bar: Bar) -> Signal | None:
        self.window.append(bar.close)
        if len(self.window) < self.slow:
            return None

        fast_ma = self._sma(self.fast)
        slow_ma = self._sma(self.slow)

        signal = None
        if self.prev_fast is not None and self.prev_slow is not None:
            if self.prev_fast <= self.prev_slow and fast_ma > slow_ma:
                signal = Signal(side=Side.LONG, qty=self.qty)
            elif self.prev_fast >= self.prev_slow and fast_ma < slow_ma:
                signal = Signal(side=Side.SHORT, qty=self.qty)

        self.prev_fast = fast_ma
        self.prev_slow = slow_ma
        return signal
