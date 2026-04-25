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
PURE_NEAT_BRANCH = "pure_neat"
EA_RL_BRANCH = "ea_rl"
DEFAULT_BRANCH = PURE_NEAT_BRANCH
BRANCH_NAMES = (PURE_NEAT_BRANCH, EA_RL_BRANCH)


def _existing_path(task_dir: str) -> Path | None:
    path = Path(task_dir).expanduser()
    candidates = [path] if path.is_absolute() else [
        Path.cwd() / path,
        PROJECT_ROOT / path,
        RESULTS_ROOT / path,
    ]

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists():
            return candidate.resolve()

    return None


def has_pure_neat_artifacts(path: str | Path) -> bool:
    path = Path(path)
    return (
        (path / "logs" / "log.json").exists()
        or (path / "logs" / "best_log.json").exists()
        or (path / GLOBAL_BEST_PACKAGE_NAME).exists()
        or (path / "recurrent.cfg").exists()
    )


def _validate_pure_neat_dir(path: Path, require_kind: str | None) -> Path:
    if require_kind in (None, PURE_NEAT_BRANCH) and has_pure_neat_artifacts(path):
        return path
    if require_kind is None:
        raise FileNotFoundError(f"Result directory has no pure_neat artifacts: {path}")
    raise FileNotFoundError(f"Result directory does not contain {require_kind} artifacts: {path}")


def resolve_task_dir(
    task_dir: str,
    *,
    branch: str | None = None,
    require_kind: str | None = None,
) -> str:
    """Resolve current results/<task_params>/<branch> paths.

    A task root such as results/transport_400_5_1_0.15_0.15_15.0 resolves
    to its pure_neat child unless a branch is specified explicitly.
    """
    base_path = _existing_path(task_dir)
    if base_path is None:
        raise FileNotFoundError(f"Result directory not found: {task_dir}")

    if branch is not None:
        if branch not in BRANCH_NAMES:
            raise ValueError(f"Unknown result branch: {branch}")
        branch_path = base_path if base_path.name == branch else base_path / branch
        if not branch_path.exists():
            raise FileNotFoundError(f"Result branch not found: {branch_path}")
        return str(_validate_pure_neat_dir(branch_path.resolve(), require_kind))

    if base_path.name == PURE_NEAT_BRANCH:
        return str(_validate_pure_neat_dir(base_path, require_kind))

    branch_path = base_path / DEFAULT_BRANCH
    if branch_path.exists():
        return str(_validate_pure_neat_dir(branch_path.resolve(), require_kind))

    raise FileNotFoundError(
        f"Expected current result layout at {base_path}/{DEFAULT_BRANCH}"
    )


def format_result_dir_label(task_dir: str | Path) -> str:
    path = Path(task_dir)
    if path.name in BRANCH_NAMES and path.parent.name != "results":
        return path.parent.name
    return path.name


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
    task_dir = resolve_task_dir(task_dir, require_kind=PURE_NEAT_BRANCH)
    log_file = os.path.join(task_dir, "logs", "log.json")
    best_log_file = os.path.join(task_dir, "logs", "best_log.json")
    return merge_history_by_generation(read_jsonl(log_file), read_jsonl(best_log_file))


def get_checkpoint_paths(task_dir: str) -> list[str]:
    checkpoint_dir = os.path.join(resolve_task_dir(task_dir, require_kind=PURE_NEAT_BRANCH), "checkpoints")
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
    return os.path.join(resolve_task_dir(task_dir, require_kind=PURE_NEAT_BRANCH), GLOBAL_BEST_PACKAGE_NAME)


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
    task_dir = resolve_task_dir(task_dir, require_kind=PURE_NEAT_BRANCH)
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
