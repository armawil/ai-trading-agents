from dataclasses import dataclass


@dataclass(frozen=True)
class FuturesContract:
    symbol: str
    tick_size: float
    tick_value: float
    point_value: float

    def pnl_from_price_delta(self, delta: float, qty: int) -> float:
        ticks = delta / self.tick_size
        return ticks * self.tick_value * qty


NQ_CONTRACT = FuturesContract(
    symbol="NQ",
    tick_size=0.25,
    tick_value=5.0,
    point_value=20.0,
)
