from abc import ABC, abstractmethod

from .data import Bar, Signal


class Strategy(ABC):
    @abstractmethod
    def on_bar(self, bar: Bar) -> Signal | None:
        """Return desired position change signal for the current bar."""
