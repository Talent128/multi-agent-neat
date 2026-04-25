from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "multi-agent-neat-mpl"))

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt

if __package__ in (None, ""):
    import sys

    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from experiment.vis.data import (
        PURE_NEAT_BRANCH,
        ensure_output_dir,
        format_result_dir_label,
        load_global_best_genome,
        load_run_history,
        load_species_history,
        resolve_task_dir,
    )
    from experiment.vis.plotting import finalize_figure, plot_band, plot_line, style_axis
else:
    from .data import (
        PURE_NEAT_BRANCH,
        ensure_output_dir,
        format_result_dir_label,
        load_global_best_genome,
        load_run_history,
        load_species_history,
        resolve_task_dir,
    )
    from .plotting import finalize_figure, plot_band, plot_line, style_axis


@dataclass(frozen=True)
class PopulationData:
    generations: np.ndarray
    fitness_avg: np.ndarray
    fitness_std: np.ndarray
    fitness_best: np.ndarray


def build_population_data(history: list[dict]) -> PopulationData:
    return PopulationData(
        generations=np.asarray([int(record["generation"]) for record in history], dtype=int),
        fitness_avg=np.asarray([record.get("fitness_avg", np.nan) for record in history], dtype=float),
        fitness_std=np.asarray([record.get("fitness_std", np.nan) for record in history], dtype=float),
        fitness_best=np.asarray([record.get("fitness_best", np.nan) for record in history], dtype=float),
    )


def _resolve_best_generation(task_dir: str):
    try:
        _, best_generation, _, _ = load_global_best_genome(task_dir)
        return int(best_generation)
    except FileNotFoundError:
        return None


def _to_optional_float(value):
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def _to_optional_int(value):
    if value is None:
        return None
    value = int(value)
    return value


def _series_summary(generations, values, *, label: str):
    generations = np.asarray(generations, dtype=int)
    values = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(values)
    if generations.size == 0 or not finite_mask.any():
        return {
            "label": label,
            "initial": None,
            "final": None,
            "min": None,
            "max": None,
        }

    finite_generations = generations[finite_mask]
    finite_values = values[finite_mask]
    min_idx = int(np.argmin(finite_values))
    max_idx = int(np.argmax(finite_values))
    return {
        "label": label,
        "initial": {
            "generation": int(finite_generations[0]),
            "value": _to_optional_float(finite_values[0]),
        },
        "final": {
            "generation": int(finite_generations[-1]),
            "value": _to_optional_float(finite_values[-1]),
        },
        "min": {
            "generation": int(finite_generations[min_idx]),
            "value": _to_optional_float(finite_values[min_idx]),
        },
        "max": {
            "generation": int(finite_generations[max_idx]),
            "value": _to_optional_float(finite_values[max_idx]),
        },
    }


def _collect_output_files(output_dir: str):
    if not os.path.isdir(output_dir):
        return []
    return sorted(
        filename
        for filename in os.listdir(output_dir)
        if filename.endswith(".png")
    )


def _build_analysis_summary(
    task_dir: str,
    history: list[dict],
    output_dir: str,
    *,
    best_generation: int | None,
):
    generations = np.asarray([int(record["generation"]) for record in history], dtype=int)
    total_generations = int(generations[-1]) + 1 if generations.size else 0

    fitness_avg = np.asarray([record.get("fitness_avg", np.nan) for record in history], dtype=float)
    fitness_std = np.asarray([record.get("fitness_std", np.nan) for record in history], dtype=float)
    fitness_best = np.asarray([record.get("fitness_best", np.nan) for record in history], dtype=float)

    best_mean = np.asarray([record.get("best_mean", np.nan) for record in history], dtype=float)
    best_std = np.asarray([record.get("best_std", np.nan) for record in history], dtype=float)
    best_min = np.asarray([record.get("best_min", np.nan) for record in history], dtype=float)
    best_max = np.asarray([record.get("best_max", np.nan) for record in history], dtype=float)

    n_neurons = np.asarray([record.get("n_neurons_best", np.nan) for record in history], dtype=float)
    n_conns = np.asarray([record.get("n_conns_best", np.nan) for record in history], dtype=float)
    time_elapsed = np.asarray([record.get("time_elapsed", np.nan) for record in history], dtype=float)
    n_species_from_logs = np.asarray([record.get("n_species", np.nan) for record in history], dtype=float)

    species_generations, species_curves, species_ids = load_species_history(task_dir)
    final_species_sizes = {}
    largest_species = None
    if species_curves.size > 0:
        final_idx = species_curves.shape[1] - 1
        final_species_sizes = {
            str(species_id): int(species_curves[idx, final_idx])
            for idx, species_id in enumerate(species_ids)
            if species_curves[idx, final_idx] > 0
        }
        max_flat_idx = int(np.argmax(species_curves))
        species_idx, generation_idx = np.unravel_index(max_flat_idx, species_curves.shape)
        largest_species = {
            "species_id": int(species_ids[species_idx]),
            "generation": int(species_generations[generation_idx]),
            "size": int(species_curves[species_idx, generation_idx]),
        }

    summary = {
        "config": format_result_dir_label(task_dir),
        "task_dir": task_dir,
        "total_generations": total_generations,
        "global_best_generation": _to_optional_int(best_generation),
        "population": {
            "fitness_avg": _series_summary(generations, fitness_avg, label="fitness_avg"),
            "fitness_std": _series_summary(generations, fitness_std, label="fitness_std"),
            "fitness_best": _series_summary(generations, fitness_best, label="fitness_best"),
            "avg_improvement": (
                _to_optional_float(fitness_avg[-1] - fitness_avg[0])
                if generations.size and np.isfinite(fitness_avg[[0, -1]]).all()
                else None
            ),
            "best_improvement": (
                _to_optional_float(fitness_best[-1] - fitness_best[0])
                if generations.size and np.isfinite(fitness_best[[0, -1]]).all()
                else None
            ),
        },
        "evaluation": {
            "best_mean": _series_summary(generations, best_mean, label="best_mean"),
            "best_std": _series_summary(generations, best_std, label="best_std"),
            "best_min": _series_summary(generations, best_min, label="best_min"),
            "best_max": _series_summary(generations, best_max, label="best_max"),
        },
        "complexity": {
            "n_neurons_best": _series_summary(generations, n_neurons, label="n_neurons_best"),
            "n_conns_best": _series_summary(generations, n_conns, label="n_conns_best"),
        },
        "species": {
            "logged_n_species": _series_summary(generations, n_species_from_logs, label="n_species"),
            "tracked_species_ids": [int(species_id) for species_id in species_ids],
            "n_species_final": len(final_species_sizes),
            "n_species_peak": int(species_curves.shape[0]) if species_curves.size else 0,
            "final_sizes": final_species_sizes,
            "largest_species": largest_species,
        },
        "runtime": {
            "time_elapsed": _series_summary(generations, time_elapsed, label="time_elapsed"),
            "total_recorded_seconds": _to_optional_float(np.nansum(time_elapsed)) if time_elapsed.size else None,
        },
        "outputs": {
            "plots": _collect_output_files(output_dir),
            "summary_json": "analysis_summary.json",
        },
    }
    return summary


def _write_analysis_summary(
    task_dir: str,
    output_dir: str,
    history: list[dict],
    *,
    best_generation: int | None,
):
    summary = _build_analysis_summary(
        task_dir,
        history,
        output_dir,
        best_generation=best_generation,
    )
    output_file = os.path.join(output_dir, "analysis_summary.json")
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved: {output_file}")
    return output_file


def _style_generation_axis(ax, generations):
    generations = np.asarray(generations, dtype=int)
    if generations.size == 0:
        return

    max_gen = int(generations[-1])
    total_generations = max_gen + 1
    step = max(1, int(round(total_generations / 10)))
    ticks = list(range(0, total_generations + step, step))
    if ticks[-1] > total_generations:
        ticks[-1] = total_generations
    elif ticks[-1] < total_generations:
        ticks.append(total_generations)

    x_max = total_generations + step * 0.2
    ax.set_xlim(left=-step * 0.2, right=x_max)
    ax.set_xticks(ticks)


def _mark_best_generation(ax, best_generation):
    if best_generation is None:
        return
    ax.axvline(best_generation, color="#111111", linestyle="--", linewidth=1.0, alpha=0.45)


def _draw_population_axis(ax, population_data: PopulationData, best_generation=None):
    if np.isfinite(population_data.fitness_avg).any():
        ax.plot(
            population_data.generations,
            population_data.fitness_avg,
            color="#2196F3",
            linewidth=2.0,
            label="Population Mean",
        )
    if np.isfinite(population_data.fitness_avg).any() and np.isfinite(population_data.fitness_std).any():
        ax.fill_between(
            population_data.generations,
            population_data.fitness_avg - population_data.fitness_std,
            population_data.fitness_avg + population_data.fitness_std,
            color="#2196F3",
            alpha=0.28,
            label="Population Std",
        )
    if np.isfinite(population_data.fitness_best).any():
        ax.plot(
            population_data.generations,
            population_data.fitness_best,
            color="#FF5722",
            linewidth=2.0,
            linestyle="--",
            label="Best Individual",
        )

    style_axis(ax, title="Population Evolution", xlabel="Generation", ylabel="Fitness")
    _style_generation_axis(ax, population_data.generations)
    _mark_best_generation(ax, best_generation)
    ax.legend(loc="lower right")


def _draw_spread_axis(ax, history: list[dict], best_generation=None):
    generations = np.asarray([int(record["generation"]) for record in history], dtype=int)
    best_mean = np.asarray([record.get("best_mean", np.nan) for record in history], dtype=float)
    best_std = np.asarray([record.get("best_std", np.nan) for record in history], dtype=float)
    best_min = np.asarray([record.get("best_min", np.nan) for record in history], dtype=float)
    best_max = np.asarray([record.get("best_max", np.nan) for record in history], dtype=float)

    if np.isfinite(best_min).any() and np.isfinite(best_max).any():
        plot_band(ax, generations, best_min, best_max, color="#7F7F7F", alpha=0.14, label="best_min/max")
    if np.isfinite(best_mean).any() and np.isfinite(best_std).any():
        plot_band(
            ax,
            generations,
            best_mean - best_std,
            best_mean + best_std,
            color="#D62728",
            alpha=0.18,
            label="best_mean ± std",
        )
        plot_line(ax, generations, best_mean, label="best_mean", color="#D62728")

    style_axis(ax, title="Evaluation Spread", xlabel="Generation", ylabel="Fitness")
    _style_generation_axis(ax, generations)
    _mark_best_generation(ax, best_generation)
    ax.legend(loc="best")


def _draw_complexity_axis(ax, history: list[dict], best_generation=None):
    generations = np.asarray([int(record["generation"]) for record in history], dtype=int)
    n_neurons = np.asarray([record.get("n_neurons_best", np.nan) for record in history], dtype=float)
    n_conns = np.asarray([record.get("n_conns_best", np.nan) for record in history], dtype=float)

    if np.isfinite(n_neurons).any():
        plot_line(ax, generations, n_neurons, label="neurons", color="#2CA02C")
    if np.isfinite(n_conns).any():
        plot_line(ax, generations, n_conns, label="connections", color="#FF7F0E")

    style_axis(ax, title="Complexity", xlabel="Generation", ylabel="Count")
    _style_generation_axis(ax, generations)
    _mark_best_generation(ax, best_generation)
    ax.legend(loc="best")


def _draw_species_axis(ax, generations, curves, best_generation=None):
    generations = np.asarray(generations, dtype=int)
    curves = np.asarray(curves, dtype=float)

    if generations.size == 0 or curves.size == 0:
        style_axis(ax, title="Species Distribution", xlabel="Generation", ylabel="Size per Species")
        ax.text(0.5, 0.5, "No checkpoint species data", ha="center", va="center", transform=ax.transAxes, color="#666666")
        return

    colors = plt.cm.tab20(np.linspace(0.0, 1.0, max(curves.shape[0], 1)))
    ax.stackplot(generations, *curves, colors=colors, alpha=0.9)
    style_axis(ax, title="Species Distribution", xlabel="Generation", ylabel="Size per Species")
    _style_generation_axis(ax, generations)
    _mark_best_generation(ax, best_generation)


def _single_run_output_dir(task_dir: str, output_dir: str | None):
    task_dir = resolve_task_dir(task_dir)
    output_dir = ensure_output_dir(task_dir, "result_analysis") if output_dir is None else output_dir
    os.makedirs(output_dir, exist_ok=True)
    return task_dir, output_dir


def plot_population_evolution_panel(task_dir: str, output_dir: str | None = None):
    task_dir, output_dir = _single_run_output_dir(task_dir, output_dir)
    history = load_run_history(task_dir)
    if not history:
        raise FileNotFoundError(f"No logs found in {task_dir}")
    best_generation = _resolve_best_generation(task_dir)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    _draw_population_axis(ax, build_population_data(history), best_generation=best_generation)

    output_file = os.path.join(output_dir, "population_evolution.png")
    finalize_figure(fig, output_file)
    print(f"saved: {output_file}")
    _write_analysis_summary(task_dir, output_dir, history, best_generation=best_generation)
    return output_file


def plot_evaluation_spread_panel(task_dir: str, output_dir: str | None = None):
    task_dir, output_dir = _single_run_output_dir(task_dir, output_dir)
    history = load_run_history(task_dir)
    if not history:
        raise FileNotFoundError(f"No logs found in {task_dir}")
    best_generation = _resolve_best_generation(task_dir)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    _draw_spread_axis(ax, history, best_generation=best_generation)

    output_file = os.path.join(output_dir, "evaluation_spread.png")
    finalize_figure(fig, output_file)
    print(f"saved: {output_file}")
    _write_analysis_summary(task_dir, output_dir, history, best_generation=best_generation)
    return output_file


def plot_complexity_panel(task_dir: str, output_dir: str | None = None):
    task_dir, output_dir = _single_run_output_dir(task_dir, output_dir)
    history = load_run_history(task_dir)
    if not history:
        raise FileNotFoundError(f"No logs found in {task_dir}")
    best_generation = _resolve_best_generation(task_dir)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    _draw_complexity_axis(ax, history, best_generation=best_generation)

    output_file = os.path.join(output_dir, "complexity_panel.png")
    finalize_figure(fig, output_file)
    print(f"saved: {output_file}")
    _write_analysis_summary(task_dir, output_dir, history, best_generation=best_generation)
    return output_file


def plot_species_distribution_panel(task_dir: str, output_dir: str | None = None):
    task_dir, output_dir = _single_run_output_dir(task_dir, output_dir)
    history = load_run_history(task_dir)
    if not history:
        raise FileNotFoundError(f"No logs found in {task_dir}")
    best_generation = _resolve_best_generation(task_dir)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    generations, curves, _ = load_species_history(task_dir)
    _draw_species_axis(ax, generations, curves, best_generation=best_generation)

    output_file = os.path.join(output_dir, "species_distribution.png")
    finalize_figure(fig, output_file)
    print(f"saved: {output_file}")
    _write_analysis_summary(task_dir, output_dir, history, best_generation=best_generation)
    return output_file


def plot_run_dashboard(task_dir: str, output_dir: str | None = None):
    task_dir, output_dir = _single_run_output_dir(task_dir, output_dir)
    history = load_run_history(task_dir)
    if not history:
        raise FileNotFoundError(f"No logs found in {task_dir}")

    population_data = build_population_data(history)
    species_generations, species_curves, _ = load_species_history(task_dir)
    best_generation = _resolve_best_generation(task_dir)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    task_name = format_result_dir_label(task_dir)
    fig.suptitle(f"Run Dashboard: {task_name}", fontsize=14, fontweight="bold")

    _draw_population_axis(axes[0, 0], population_data, best_generation=best_generation)
    _draw_spread_axis(axes[0, 1], history, best_generation=best_generation)
    _draw_complexity_axis(axes[1, 0], history, best_generation=best_generation)
    _draw_species_axis(axes[1, 1], species_generations, species_curves, best_generation=best_generation)

    output_file = os.path.join(output_dir, "analysis_dashboard.png")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_file, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {output_file}")
    _write_analysis_summary(task_dir, output_dir, history, best_generation=best_generation)
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Result analysis tools for multi-agent NEAT runs.")
    parser.add_argument("task_dir", help="Task root, pure_neat branch directory, or absolute path")
    parser.add_argument(
        "--branch",
        choices=(PURE_NEAT_BRANCH,),
        default=None,
        help="Branch under a task root. Currently supported: pure_neat.",
    )
    parser.add_argument("--dashboard", action="store_true", help="Plot a single-run dashboard")
    parser.add_argument("--panels", action="store_true", help="Plot all four dashboard panels as standalone figures")
    parser.add_argument("--population-panel", action="store_true", help="Plot the population evolution panel only")
    parser.add_argument("--spread-panel", action="store_true", help="Plot the evaluation spread panel only")
    parser.add_argument("--complexity-panel", action="store_true", help="Plot the complexity panel only")
    parser.add_argument("--species-panel", action="store_true", help="Plot the species distribution panel only")
    parser.add_argument("--output-dir", help="Directory to save plots to")
    args = parser.parse_args()

    task_dir = resolve_task_dir(args.task_dir, branch=args.branch)

    panel_mode = (
        args.panels
        or args.population_panel
        or args.spread_panel
        or args.complexity_panel
        or args.species_panel
    )
    explicit_mode = args.dashboard or panel_mode
    if not explicit_mode:
        args.dashboard = True

    if args.dashboard:
        plot_run_dashboard(task_dir, output_dir=args.output_dir)
    if args.panels or args.population_panel:
        plot_population_evolution_panel(task_dir, output_dir=args.output_dir)
    if args.panels or args.spread_panel:
        plot_evaluation_spread_panel(task_dir, output_dir=args.output_dir)
    if args.panels or args.complexity_panel:
        plot_complexity_panel(task_dir, output_dir=args.output_dir)
    if args.panels or args.species_panel:
        plot_species_distribution_panel(task_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
