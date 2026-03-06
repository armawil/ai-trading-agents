"""Base strategy primitives for signal generation."""

from dataclasses import dataclass
from typing import Any


@dataclass
class BaseStrategy:
    """Minimal strategy interface."""

    name: str

    def generate_signal(self, market_state: dict[str, Any]) -> str:
        """Return a placeholder directional signal."""
        _ = market_state
        return "HOLD"
