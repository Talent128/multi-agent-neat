from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt


def style_axis(ax, *, title: str = "", xlabel: str = "", ylabel: str = ""):
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def plot_line(ax, x, y, *, label: str = "", color=None, linewidth: float = 2.0, alpha: float = 1.0):
    ax.plot(x, y, label=label, color=color, linewidth=linewidth, alpha=alpha)


def plot_band(ax, x, low, high, *, color=None, alpha: float = 0.18, label: str = ""):
    ax.fill_between(x, low, high, color=color, alpha=alpha, label=label)


def plot_quartiles(
    ax,
    x,
    values,
    *,
    label: str = "",
    color=None,
    alpha: float = 0.18,
    linewidth: float = 2.2,
):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        raise ValueError("values must be a 2D array with shape [n_steps, n_runs]")

    median = np.nanpercentile(values, 50, axis=1)
    lower = np.nanpercentile(values, 25, axis=1)
    upper = np.nanpercentile(values, 75, axis=1)

    plot_band(ax, x, lower, upper, color=color, alpha=alpha)
    plot_line(ax, x, median, label=label, color=color, linewidth=linewidth)
    return median, lower, upper


def plot_histogram(ax, values, *, bins: int = 12, color: str = "#4C72B0", alpha: float = 0.75):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    ax.hist(values, bins=bins, color=color, alpha=alpha, edgecolor="white")
    style_axis(ax)
    return values


def finalize_figure(fig, output_file: str):
    fig.tight_layout()
    fig.savefig(output_file, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
