from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass(frozen=True)
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Signal:
    side: Side
    qty: int = 1
