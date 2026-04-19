from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "multi-agent-neat-mpl"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

if __package__ in (None, ""):
    import sys

    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from experiment.vis.data import (
        ensure_output_dir,
        get_checkpoint_paths,
        load_best_genome_from_checkpoint,
        load_best_genome_from_generation,
        load_global_best_genome,
        resolve_task_dir,
    )
    from experiment.vis.plotting import finalize_figure
else:
    from .data import (
        ensure_output_dir,
        get_checkpoint_paths,
        load_best_genome_from_checkpoint,
        load_best_genome_from_generation,
        load_global_best_genome,
        resolve_task_dir,
    )
    from .plotting import finalize_figure


ACTIVATION_LABELS = {
    "sigmoid": "sig",
    "tanh": "tanh",
    "relu": "relu",
    "identity": "id",
    "sin": "sin",
    "gauss": "gau",
    "abs": "abs",
    "clamped": "clamp",
    "inv": "inv",
    "log": "log",
    "exp": "exp",
}


def short_activation_label(name: str) -> str:
    return ACTIVATION_LABELS.get(name, name[:6])


def get_genome_structure(genome, config) -> dict:
    genome_config = config.genome_config
    input_keys = list(genome_config.input_keys)
    output_keys = list(genome_config.output_keys)
    hidden_keys = sorted(key for key in genome.nodes.keys() if key not in output_keys)

    connections = []
    for conn_key, conn in genome.connections.items():
        connections.append(
            {
                "from": conn_key[0],
                "to": conn_key[1],
                "weight": conn.weight,
                "enabled": conn.enabled,
            }
        )

    nodes = {}
    for key in input_keys:
        nodes[key] = {"type": "input", "bias": 0.0, "response": 1.0, "activation": "input"}
    for key, node in genome.nodes.items():
        nodes[key] = {
            "type": "output" if key in output_keys else "hidden",
            "bias": node.bias,
            "response": node.response,
            "activation": getattr(node, "activation", "unknown"),
        }

    return {
        "input_keys": input_keys,
        "output_keys": output_keys,
        "hidden_keys": hidden_keys,
        "nodes": nodes,
        "connections": connections,
        "n_inputs": len(input_keys),
        "n_hidden": len(hidden_keys),
        "n_outputs": len(output_keys),
        "n_enabled_conns": sum(1 for conn in connections if conn["enabled"]),
        "n_total_conns": len(connections),
    }


def c_linspace(start: float, end: float, n_items: int):
    if n_items == 1:
        return np.array([(start + end) / 2.0])
    return np.linspace(start, end, n_items)


def _sorted_nodes(nodes):
    return sorted(nodes, key=lambda node_key: (isinstance(node_key, str), node_key))


def _compute_node_layers(structure: dict):
    input_keys = set(structure["input_keys"])
    output_keys = set(structure["output_keys"])
    nodes = list(structure["nodes"].keys())

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    for conn in structure["connections"]:
        if not conn["enabled"]:
            continue
        if conn["from"] == conn["to"]:
            continue
        if conn["to"] in input_keys:
            continue
        if conn["from"] in output_keys:
            continue
        graph.add_edge(conn["from"], conn["to"])

    sccs = list(nx.strongly_connected_components(graph))
    component_of = {}
    component_sizes = {}
    for idx, component in enumerate(sccs):
        component_sizes[idx] = len(component)
        for node_key in component:
            component_of[node_key] = idx

    condensed = nx.DiGraph()
    condensed.add_nodes_from(range(len(sccs)))
    for u, v in graph.edges():
        src_comp = component_of[u]
        dst_comp = component_of[v]
        if src_comp != dst_comp:
            condensed.add_edge(src_comp, dst_comp)

    component_layers = {}
    for component_idx in nx.topological_sort(condensed):
        component_nodes = sccs[component_idx]
        if any(node_key in input_keys for node_key in component_nodes):
            component_layers[component_idx] = 0
            continue
        predecessor_layers = [component_layers[pred] for pred in condensed.predecessors(component_idx)]
        component_layers[component_idx] = max(predecessor_layers, default=0) + (1 if predecessor_layers else 1)

    node_layers = {}
    for node_key in nodes:
        if node_key in input_keys:
            node_layers[node_key] = 0
        elif node_key in output_keys:
            node_layers[node_key] = 0
        else:
            node_layers[node_key] = component_layers.get(component_of[node_key], 1)

    max_hidden_layer = max((layer for node_key, layer in node_layers.items() if node_key not in output_keys), default=0)
    output_layer = max_hidden_layer + 1
    for node_key in output_keys:
        node_layers[node_key] = output_layer

    return node_layers, component_of, component_sizes


def _compute_positions(structure: dict):
    node_layers, component_of, component_sizes = _compute_node_layers(structure)
    layers = sorted(set(node_layers.values()))
    max_layer = max(layers, default=1)
    fig_wide = 10.0
    fig_long = 5.0

    positions = {}
    layer_groups = []
    for layer in layers:
        layer_nodes = [node_key for node_key, node_layer in node_layers.items() if node_layer == layer]
        layer_nodes = _sorted_nodes(layer_nodes)
        layer_groups.append(layer_nodes)

    for layer_idx, layer_nodes in enumerate(layer_groups):
        layer = node_layers[layer_nodes[0]]
        x_coord = 0.0 if max_layer == 0 else (layer / max_layer) * fig_wide
        if layer_idx == 0:
            y_coords = c_linspace(-2.0, fig_long + 2.0, len(layer_nodes))
        elif layer_idx % 2 == 0:
            y_coords = c_linspace(0.0, fig_long, len(layer_nodes))
        else:
            y_coords = c_linspace(-1.0, fig_long + 1.0, len(layer_nodes))

        for node_key, y_coord in zip(layer_nodes, y_coords):
            positions[node_key] = (x_coord, y_coord)

    return positions, node_layers, component_of, component_sizes, fig_wide, fig_long


def _is_recurrent(conn: dict, structure: dict, node_layers: dict, component_of: dict, component_sizes: dict) -> bool:
    from_key = conn["from"]
    to_key = conn["to"]
    input_keys = set(structure["input_keys"])
    output_keys = set(structure["output_keys"])

    if from_key == to_key:
        return True
    if from_key in output_keys:
        return True
    if to_key in input_keys:
        return True
    if component_of.get(from_key) == component_of.get(to_key) and component_sizes.get(component_of.get(from_key), 1) > 1:
        return True
    return node_layers.get(from_key, 0) >= node_layers.get(to_key, 0)


def _activation_labels(structure: dict):
    labels = {}
    for node_key, node_info in structure["nodes"].items():
        if node_info["type"] == "input":
            labels[node_key] = ""
        else:
            labels[node_key] = f"({short_activation_label(node_info['activation'])})"
    return labels


def _compact_scale(structure: dict, node_layers: dict) -> float:
    layer_counts = {}
    for node_key, layer in node_layers.items():
        layer_counts[layer] = layer_counts.get(layer, 0) + 1

    max_layer_count = max(layer_counts.values(), default=1)
    # Densely stacked input layers need smaller nodes in evolution subplots.
    return max(0.58, min(0.90, 8.0 / max_layer_count))


def _io_labels(structure: dict):
    input_labels = {node_key: f"obs_{idx}" for idx, node_key in enumerate(structure["input_keys"])}
    output_labels = {node_key: f"act_{idx}" for idx, node_key in enumerate(structure["output_keys"])}
    return input_labels, output_labels


def _draw_io_annotations(ax: plt.Axes, positions: dict, structure: dict, fig_wide: float):
    input_labels, output_labels = _io_labels(structure)

    for node_key, label in input_labels.items():
        x_coord, y_coord = positions[node_key]
        node_edge_x = x_coord - 0.42
        label_x = x_coord - 1.95
        ax.text(label_x, y_coord - 0.10, label, fontsize=8, ha="left", va="center")
        ax.annotate(
            "",
            xy=(node_edge_x, y_coord),
            xytext=(label_x + 0.55, y_coord),
            arrowprops=dict(arrowstyle="->", color="black", linewidth=1.3, shrinkA=0, shrinkB=0),
        )

    for node_key, label in output_labels.items():
        x_coord, y_coord = positions[node_key]
        start_x = x_coord + 0.42
        elbow_x = min(fig_wide + 1.10, x_coord + 1.15)
        arrow_top_y = y_coord + 0.82
        ax.plot([start_x, elbow_x], [y_coord, y_coord], color="black", linewidth=1.3, zorder=3)
        ax.annotate(
            "",
            xy=(elbow_x, arrow_top_y),
            xytext=(elbow_x, y_coord),
            arrowprops=dict(arrowstyle="->", color="black", linewidth=1.3, shrinkA=0, shrinkB=0),
        )
        ax.text(elbow_x, arrow_top_y + 0.08, label, fontsize=8, ha="center", va="bottom")


def _point_to_segment_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)

    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def _line_hits_other_node(edge, positions: dict, structure: dict, threshold: float = 0.5) -> bool:
    src, dst = edge
    x1, y1 = positions[src]
    x2, y2 = positions[dst]
    for node_key in structure["nodes"]:
        if node_key in edge:
            continue
        px, py = positions[node_key]
        if _point_to_segment_distance(px, py, x1, y1, x2, y2) < threshold:
            return True
    return False


def _fan_offsets(count: int) -> list[float]:
    if count <= 1:
        return [0.0]
    step = 0.065
    center = (count - 1) / 2.0
    return [(idx - center) * step for idx in range(count)]


def _clamp_forward_rad(rad: float, conn: dict, positions: dict, structure: dict) -> float:
    source_type = structure["nodes"][conn["from"]]["type"]
    target_type = structure["nodes"][conn["to"]]["type"]
    x1, _ = positions[conn["from"]]
    x2, _ = positions[conn["to"]]
    x_span = abs(x2 - x1)

    limit = 0.30
    if source_type == "input":
        limit = min(limit, 0.18)
    if target_type == "output":
        limit = min(limit, 0.18)
    if x_span < 3.0:
        limit = min(limit, 0.12)

    return max(-limit, min(limit, rad))


def _forward_edge_color(conn: dict, structure: dict) -> str:
    target_type = structure["nodes"][conn["to"]]["type"]
    if target_type == "hidden":
        return "#4DB6AC"
    return "#F4D03F"


def _draw_curved_edge(
    ax: plt.Axes,
    start,
    end,
    color: str,
    *,
    rad: float,
    linestyle: str = "-",
    linewidth: float = 1.15,
    alpha: float = 0.8,
    shrink: float = 12.0,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=linewidth,
        color=color,
        linestyle=linestyle,
        alpha=alpha,
        shrinkA=shrink,
        shrinkB=shrink,
        connectionstyle=f"arc3,rad={rad}",
        zorder=1,
    )
    ax.add_patch(patch)


def _edge_shrink(node_size: float, compact_mode: bool, compact_scale: float) -> float:
    if compact_mode:
        return max(4.5, math.sqrt(node_size) / 1.8)
    return max(8.0, math.sqrt(node_size) / 1.8)


def _draw_self_loop(
    ax: plt.Axes,
    center,
    color: str,
    *,
    compact_mode: bool,
    compact_scale: float = 1.0,
    linestyle: str = "--",
    alpha: float = 0.75,
):
    x_coord, y_coord = center
    patch = FancyArrowPatch(
        (x_coord + 0.10, y_coord + 0.06),
        (x_coord - 0.02, y_coord + 0.08),
        arrowstyle="-|>",
        mutation_scale=(8 * compact_scale) if compact_mode else 10,
        linewidth=0.9 * compact_scale if compact_mode else 1.0,
        color=color,
        linestyle=linestyle,
        alpha=alpha,
        connectionstyle="arc3,rad=1.4",
        zorder=1,
    )
    ax.add_patch(patch)


def draw_genome_network(
    genome,
    config,
    ax: plt.Axes,
    *,
    title: str = "",
    compact_mode: bool = False,
    show_activations: bool = True,
):
    structure = get_genome_structure(genome, config)
    positions, node_layers, component_of, component_sizes, fig_wide, fig_long = _compute_positions(structure)
    compact_scale = _compact_scale(structure, node_layers) if compact_mode else 1.0

    graph = nx.DiGraph()
    graph.add_nodes_from(structure["nodes"].keys())

    forward_connections = []
    recurrent_connections = []
    for conn in structure["connections"]:
        if not conn["enabled"]:
            continue
        graph.add_edge(conn["from"], conn["to"])
        if _is_recurrent(conn, structure, node_layers, component_of, component_sizes):
            recurrent_connections.append(conn)
        else:
            forward_connections.append(conn)

    node_size = (420 * (compact_scale ** 2)) if compact_mode else 980
    edge_shrink = _edge_shrink(node_size, compact_mode, compact_scale)
    for node_type, color in [
        ("input", "#90EE90"),
        ("hidden", "#87CEEB"),
        ("output", "#FFB6C1"),
    ]:
        node_keys = [node_key for node_key, node_info in structure["nodes"].items() if node_info["type"] == node_type]
        if not node_keys:
            continue
        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=node_keys,
            node_color=color,
            node_shape="o",
            edgecolors="#333333",
            linewidths=(0.95 * compact_scale) if compact_mode else 1.4,
            node_size=node_size,
            ax=ax,
        )

    incoming_by_target = {}
    for conn in forward_connections:
        incoming_by_target.setdefault(conn["to"], []).append(conn)

    forward_edge_rads = {}
    for target_key, connections in incoming_by_target.items():
        ordered = sorted(connections, key=lambda conn: positions[conn["from"]][1])
        offsets = _fan_offsets(len(ordered))
        for idx, conn in enumerate(ordered):
            edge = (conn["from"], conn["to"])
            rad = offsets[idx]
            x1, y1 = positions[conn["from"]]
            x2, y2 = positions[conn["to"]]
            if abs(rad) < 0.05 and _line_hits_other_node(edge, positions, structure):
                rad = 0.10 if y2 >= y1 else -0.10
            elif _line_hits_other_node(edge, positions, structure):
                rad *= 1.15
            elif abs(y2 - y1) < 0.4 and len(ordered) == 1:
                rad = 0.06 if y2 >= y1 else -0.06
            forward_edge_rads[edge] = _clamp_forward_rad(rad, conn, positions, structure)

    for conn in forward_connections:
        edge = (conn["from"], conn["to"])
        _draw_curved_edge(
            ax,
            positions[conn["from"]],
            positions[conn["to"]],
            _forward_edge_color(conn, structure),
            rad=forward_edge_rads.get(edge, 0.0),
            linewidth=(0.65 * compact_scale) if compact_mode else 0.9,
            alpha=0.82,
            shrink=edge_shrink,
        )

    for conn in recurrent_connections:
        edge = (conn["from"], conn["to"])
        if conn["from"] == conn["to"]:
            _draw_self_loop(
                ax,
                positions[conn["from"]],
                "#777777",
                compact_mode=compact_mode,
                compact_scale=compact_scale,
            )
            continue

        x1, y1 = positions[conn["from"]]
        x2, y2 = positions[conn["to"]]
        rad = 0.28 if y2 >= y1 else -0.28
        if _line_hits_other_node(edge, positions, structure, threshold=0.45):
            rad *= 1.3
        _draw_curved_edge(
            ax,
            positions[conn["from"]],
            positions[conn["to"]],
            "#777777",
            rad=rad,
            linestyle="--",
            linewidth=(0.65 * compact_scale) if compact_mode else 0.85,
            alpha=0.74,
            shrink=edge_shrink,
        )

    if show_activations:
        activation_labels = _activation_labels(structure)
        if compact_mode:
            activation_labels = {
                node_key: label[1:-1] if label.startswith("(") and label.endswith(")") else label
                for node_key, label in activation_labels.items()
            }
        nx.draw_networkx_labels(
            graph,
            positions,
            labels=activation_labels,
            font_size=max(4.5, 6.4 * compact_scale) if compact_mode else 8,
            font_color="#111111",
            ax=ax,
        )

    if not compact_mode:
        _draw_io_annotations(ax, positions, structure, fig_wide)

    ax.set_xlim(-2.8, fig_wide + 2.5)
    ax.set_ylim(-2.5, fig_long + 2.5)
    ax.tick_params(axis="both", which="both", bottom=False, top=False, left=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if title:
        ax.set_title(title, fontsize=10 if compact_mode else 12, fontweight="bold", pad=5)

    stats_text = f"Hidden: {structure['n_hidden']}\nConns: {structure['n_enabled_conns']}"
    if genome.fitness is not None:
        stats_text += f"\nFitness: {genome.fitness:.2f}"

    if not compact_mode:
        ax.text(
            0.98,
            0.98,
            stats_text,
            ha="right",
            va="top",
            fontsize=8,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC", alpha=0.9),
            color="#333333",
        )
    else:
        short_stats = f"H:{structure['n_hidden']} C:{structure['n_enabled_conns']}"
        if genome.fitness is not None:
            short_stats += f" F:{genome.fitness:.1f}"
        ax.text(
            0.5,
            -0.02,
            short_stats,
            ha="center",
            va="top",
            fontsize=max(4.4, 5.8 * compact_scale),
            transform=ax.transAxes,
            color="#555555",
        )


def visualize_single_genome(
    task_dir: str,
    generation: Optional[int] = None,
    output_dir: Optional[str] = None,
):
    task_dir = resolve_task_dir(task_dir)
    output_dir = ensure_output_dir(task_dir, "genome_visualization") if output_dir is None else output_dir
    os.makedirs(output_dir, exist_ok=True)

    if generation is None:
        genome, resolved_generation, config, _ = load_global_best_genome(task_dir)
        selection_label = f"Global Best (Gen {resolved_generation})"
        output_name = "genome_global_best.png"
    else:
        genome, resolved_generation, config, checkpoint_file = load_best_genome_from_generation(task_dir, generation)
        selection_label = f"Generation {resolved_generation}"
        output_name = f"genome_gen{resolved_generation}.png"
        if resolved_generation != generation:
            print(f"gen {generation} missing, using {checkpoint_file}")

    fig, ax = plt.subplots(figsize=(10, 8))
    draw_genome_network(
        genome,
        config,
        ax,
        title=f"{os.path.basename(task_dir)}\n{selection_label}",
        compact_mode=False,
        show_activations=True,
    )

    legend_elements = [
        mpatches.Patch(color="#90EE90", ec="#333333", label="Input"),
        mpatches.Patch(color="#87CEEB", ec="#333333", label="Hidden"),
        mpatches.Patch(color="#FFB6C1", ec="#333333", label="Output"),
        Line2D([0], [0], color="#4DB6AC", lw=1.5, label="Forward to hidden"),
        Line2D([0], [0], color="#F4D03F", lw=1.5, label="Forward to output"),
        Line2D([0], [0], color="#777777", lw=1.5, linestyle="--", label="Recurrent"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.95, edgecolor="#CCCCCC")

    output_file = os.path.join(output_dir, output_name)
    finalize_figure(fig, output_file)
    print(f"saved: {output_file}")

    structure = get_genome_structure(genome, config)
    print(f"target: {selection_label}")
    print(f"fitness: {genome.fitness:.4f}" if genome.fitness is not None else "fitness: N/A")
    print(f"inputs: {structure['n_inputs']}, hidden: {structure['n_hidden']}, outputs: {structure['n_outputs']}")
    print(f"enabled_conns: {structure['n_enabled_conns']}, total_conns: {structure['n_total_conns']}")
    return output_file


def visualize_evolution(
    task_dir: str,
    interval: int = 25,
    output_dir: Optional[str] = None,
    max_generations: Optional[int] = None,
):
    task_dir = resolve_task_dir(task_dir)
    output_dir = ensure_output_dir(task_dir, "genome_visualization") if output_dir is None else output_dir
    checkpoint_paths = get_checkpoint_paths(task_dir)
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found in {task_dir}")

    generations = [int(os.path.basename(path).split("-")[-1]) for path in checkpoint_paths]
    if max_generations is not None:
        generations = [generation for generation in generations if generation <= max_generations]

    selected = []
    for generation in generations:
        if generation == 0 or generation == generations[-1] or generation % interval == 0:
            selected.append(generation)

    if len(selected) > 16:
        step = len(selected) // 16 + 1
        selected = selected[::step]
        if generations[-1] not in selected:
            selected.append(generations[-1])

    checkpoint_dir = os.path.join(task_dir, "checkpoints")
    selected_entries = []
    for generation in selected:
        checkpoint_file = os.path.join(checkpoint_dir, f"neat-checkpoint-{generation}")
        try:
            genome, _, config = load_best_genome_from_checkpoint(checkpoint_file)
            if genome is None:
                selected_entries.append({"generation": generation, "error": "No valid genome"})
                continue

            entry = {
                "generation": generation,
                "genome": genome,
                "config": config,
            }
            selected_entries.append(entry)
        except Exception:
            selected_entries.append({"generation": generation, "error": "Error"})

    n_plots = len(selected_entries)
    if n_plots == 0:
        raise ValueError("No checkpoints selected for evolution view.")

    n_cols = min(4, n_plots)
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.1 * n_cols, 4.0 * n_rows))
    axes = np.array(axes, dtype=object).reshape(n_rows, n_cols)

    fig.suptitle(f"Network Evolution: {os.path.basename(task_dir)}", fontsize=14, fontweight="bold")

    for idx, entry in enumerate(selected_entries):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        generation = entry["generation"]
        if "error" in entry:
            ax.axis("off")
            ax.text(0.5, 0.5, f"Gen {generation}\n{entry['error']}", ha="center", va="center", fontsize=8)
            continue

        draw_genome_network(
            entry["genome"],
            entry["config"],
            ax,
            title=f"Gen {generation}",
            compact_mode=True,
            show_activations=True,
        )

    for idx in range(n_plots, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    legend_elements = [
        mpatches.Patch(color="#90EE90", ec="#333333", label="Input"),
        mpatches.Patch(color="#87CEEB", ec="#333333", label="Hidden"),
        mpatches.Patch(color="#FFB6C1", ec="#333333", label="Output"),
        Line2D([0], [0], color="#4DB6AC", lw=1.5, label="Forward to hidden"),
        Line2D([0], [0], color="#F4D03F", lw=1.5, label="Forward to output"),
        Line2D([0], [0], color="#777777", lw=1.5, linestyle="--", label="Recurrent"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, 0.01), framealpha=0.95)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])

    output_file = os.path.join(output_dir, "network_evolution.png")
    fig.savefig(output_file, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved: {output_file}")
    print(f"generations: {[entry['generation'] for entry in selected_entries]}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Visualize NEAT genomes and topology evolution.")
    parser.add_argument("task_dir", help="Result directory name under results/ or an absolute path")
    parser.add_argument("-g", "--generation", type=int, default=None, help="Generation to visualize, default: global best")
    parser.add_argument("--evolution", action="store_true", help="Render topology evolution across checkpoints")
    parser.add_argument("--interval", type=int, default=25, help="Generation interval for evolution view")
    parser.add_argument("--all", action="store_true", help="Generate single genome and evolution plots")
    args = parser.parse_args()

    task_dir = resolve_task_dir(args.task_dir)
    print(f"[GenomeVis]\ntask_dir: {task_dir}")

    if args.all:
        visualize_single_genome(task_dir, args.generation)
        visualize_evolution(task_dir, args.interval)
    elif args.evolution:
        visualize_evolution(task_dir, args.interval)
    else:
        visualize_single_genome(task_dir, args.generation)

    print(f"output_dir: {os.path.join(task_dir, 'genome_visualization')}")


if __name__ == "__main__":
    main()
