from math import sqrt


def max_drawdown(equity_curve: list[float]) -> float:
    peak = float("-inf")
    max_dd = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return max_dd


def sharpe_ratio(period_returns: list[float], periods_per_year: int = 252) -> float:
    if len(period_returns) < 2:
        return 0.0
    mean = sum(period_returns) / len(period_returns)
    variance = sum((x - mean) ** 2 for x in period_returns) / (len(period_returns) - 1)
    if variance == 0:
        return 0.0
    std = variance ** 0.5
    return (mean / std) * sqrt(periods_per_year)
