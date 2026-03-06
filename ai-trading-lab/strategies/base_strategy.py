"""Base strategy interface for trading models."""

from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd


class Strategy(ABC):
    """Standard interface all strategies must implement."""

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return signal DataFrame indexed like data with a `signal` column.

        Signal semantics:
        - 1 for long bias
        - -1 for short bias
        - 0 for flat / no position
        """

    def __repr__(self) -> str:
        return self.__class__.__name__
