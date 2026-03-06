"""NQ futures backtesting engine package."""

from .contracts import FuturesContract, NQ_CONTRACT
from .data import Bar, Signal, Side
from .engine import BacktestConfig, BacktestEngine, BacktestResult

__all__ = [
    "FuturesContract",
    "NQ_CONTRACT",
    "Bar",
    "Signal",
    "Side",
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
]
