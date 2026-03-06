"""Example EMA crossover strategy module.

New strategies should expose:
- parameter_grid() -> dict[str, list]
- generate_positions(prices: list[float], params: dict) -> list[int]
"""

from __future__ import annotations


def _ema(values: list[float], length: int) -> list[float]:
    if length <= 0:
        raise ValueError("EMA length must be > 0")

    alpha = 2 / (length + 1)
    ema_values: list[float] = []
    prev = values[0]
    for value in values:
        prev = alpha * value + (1 - alpha) * prev
        ema_values.append(prev)
    return ema_values


def parameter_grid() -> dict[str, list[float | int]]:
    """Search space used by the optimizer for this strategy."""
    return {
        "ema_length": [5, 10, 15, 20],
        "stop_loss": [0.01, 0.02, 0.03],
        "take_profit": [0.02, 0.04, 0.06],
    }


def generate_positions(prices: list[float], params: dict[str, float | int]) -> list[int]:
    """Generate daily positions: 1 for long, 0 for flat.

    Rules:
    - Enter long when close > EMA.
    - Exit long on stop-loss / take-profit or when close < EMA.
    """
    if not prices:
        return []

    ema_length = int(params["ema_length"])
    stop_loss = float(params["stop_loss"])
    take_profit = float(params["take_profit"])

    ema_values = _ema(prices, ema_length)
    positions = [0] * len(prices)

    in_position = False
    entry_price = 0.0

    for i, price in enumerate(prices):
        ema = ema_values[i]

        if in_position:
            move = (price - entry_price) / entry_price
            if move <= -stop_loss or move >= take_profit or price < ema:
                in_position = False

        if not in_position and price > ema:
            in_position = True
            entry_price = price

        positions[i] = 1 if in_position else 0

    return positions
