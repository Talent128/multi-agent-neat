from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from experiment.runtime import (
    GLOBAL_BEST_PACKAGE_NAME,
    CHECKPOINT_PREFIX,
    list_checkpoints,
    load_checkpoint_payload,
    load_global_best_package,
    select_best_genome,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results"


def resolve_task_dir(task_dir: str) -> str:
    path = Path(task_dir)
    if path.exists():
        return str(path.resolve())

    candidate = RESULTS_ROOT / task_dir
    if candidate.exists():
        return str(candidate.resolve())

    raise FileNotFoundError(f"Result directory not found: {task_dir}")


def ensure_output_dir(task_dir: str, folder_name: str) -> str:
    output_dir = os.path.join(task_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def read_jsonl(file_path: str) -> list[dict]:
    if not os.path.exists(file_path):
        return []

    records = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def merge_history_by_generation(*record_groups: list[dict]) -> list[dict]:
    merged: dict[int, dict] = {}
    for records in record_groups:
        for record in records:
            generation = int(record["generation"])
            merged.setdefault(generation, {"generation": generation}).update(record)
    return [merged[generation] for generation in sorted(merged)]


def load_run_history(task_dir: str) -> list[dict]:
    task_dir = resolve_task_dir(task_dir)
    log_file = os.path.join(task_dir, "logs", "log.json")
    best_log_file = os.path.join(task_dir, "logs", "best_log.json")
    return merge_history_by_generation(read_jsonl(log_file), read_jsonl(best_log_file))


def get_checkpoint_paths(task_dir: str) -> list[str]:
    checkpoint_dir = os.path.join(resolve_task_dir(task_dir), "checkpoints")
    return [os.path.join(checkpoint_dir, filename) for filename in list_checkpoints(checkpoint_dir)]


def load_best_genome_from_checkpoint(checkpoint_file: str):
    generation, config, population, _ = load_checkpoint_payload(checkpoint_file)
    best_genome = select_best_genome(population)
    return best_genome, generation, config


def load_best_genome_from_generation(task_dir: str, generation: int):
    checkpoint_paths = get_checkpoint_paths(task_dir)
    if not checkpoint_paths:
        raise FileNotFoundError("No checkpoints found. Train the experiment first.")

    target_name = f"{CHECKPOINT_PREFIX}{generation}"
    checkpoint_file = None
    for path in checkpoint_paths:
        if os.path.basename(path) == target_name:
            checkpoint_file = path
            break

    if checkpoint_file is None:
        available = [int(os.path.basename(path).split("-")[-1]) for path in checkpoint_paths]
        resolved_generation = min(available, key=lambda value: abs(value - generation))
        checkpoint_file = os.path.join(
            os.path.dirname(checkpoint_paths[0]),
            f"{CHECKPOINT_PREFIX}{resolved_generation}",
        )
    else:
        resolved_generation = generation

    genome, _, config = load_best_genome_from_checkpoint(checkpoint_file)
    if genome is None:
        raise ValueError(f"No valid genome found in checkpoint: {checkpoint_file}")
    return genome, resolved_generation, config, checkpoint_file


def get_global_best_package_path(task_dir: str) -> str:
    return os.path.join(resolve_task_dir(task_dir), GLOBAL_BEST_PACKAGE_NAME)


def load_global_best_genome(task_dir: str):
    package_path = get_global_best_package_path(task_dir)
    package = load_global_best_package(package_path)
    if package is None:
        raise FileNotFoundError(f"Missing global best record: {package_path}")

    return (
        package["genome"],
        package.get("generation", -1),
        package["neat_config"],
        package_path,
    )

@lru_cache(maxsize=16)
def load_species_history(task_dir: str):
    task_dir = resolve_task_dir(task_dir)
    checkpoint_paths = get_checkpoint_paths(task_dir)
    if not checkpoint_paths:
        return np.array([], dtype=int), np.zeros((0, 0), dtype=float), []

    generations = []
    species_records = []
    species_ids = set()

    for checkpoint_path in checkpoint_paths:
        generation, _, _, species_set = load_checkpoint_payload(checkpoint_path)
        species_sizes = {
            int(species_id): len(species.members)
            for species_id, species in species_set.species.items()
        }
        generations.append(int(generation))
        species_records.append(species_sizes)
        species_ids.update(species_sizes.keys())

    ordered_species_ids = sorted(species_ids)
    curves = np.zeros((len(ordered_species_ids), len(generations)), dtype=float)
    species_index = {species_id: idx for idx, species_id in enumerate(ordered_species_ids)}

    for generation_idx, species_sizes in enumerate(species_records):
        for species_id, size in species_sizes.items():
            curves[species_index[int(species_id)], generation_idx] = float(size)

    return np.asarray(generations, dtype=int), curves, ordered_species_ids
