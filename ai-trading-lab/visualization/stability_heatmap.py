"""Parameter stability map plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_stability_heatmap(
    optimization_results: pd.DataFrame,
    x_param: str,
    y_param: str,
    metric: str = "sharpe_ratio",
    output_path: str | None = None,
) -> pd.DataFrame:
    """Create and optionally save a heatmap showing parameter sensitivity."""
    pivot = optimization_results.pivot_table(index=y_param, columns=x_param, values=metric, aggfunc="mean")

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, cmap="viridis", fmt=".2f")
    plt.title(f"Parameter Stability Map: {y_param} vs {x_param} ({metric})")
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150)
    plt.close()

    return pivot
