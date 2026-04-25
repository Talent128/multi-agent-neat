from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence
import gzip
import os
import pickle
import random
import tempfile

import torch


CHECKPOINT_PREFIX = "neat-checkpoint-"
GLOBAL_BEST_PACKAGE_NAME = "global_best_genome.pkl"


@dataclass(frozen=True)
class ExperimentPaths:
    results_dir: str
    checkpoint_dir: str
    log_dir: str
    video_dir: str
    global_best_package_path: str


@dataclass(frozen=True)
class ResultTarget:
    genome: Any
    neat_config: Any
    generation: int
    label: str
    message: str


def ensure_results_layout(results_dir: str) -> ExperimentPaths:
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    log_dir = os.path.join(results_dir, "logs")
    video_dir = os.path.join(results_dir, "videos")

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    return ExperimentPaths(
        results_dir=results_dir,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        video_dir=video_dir,
        global_best_package_path=os.path.join(results_dir, GLOBAL_BEST_PACKAGE_NAME),
    )


def generate_results_dir_name(scenario_name: str, algorithm_name: str, task_config) -> str:
    from dataclasses import fields, is_dataclass

    task_params = []
    if is_dataclass(task_config):
        for field in fields(task_config):
            task_params.append(str(getattr(task_config, field.name)))
    else:
        for key in sorted(vars(task_config).keys()):
            task_params.append(str(getattr(task_config, key)))

    params_str = "_".join(task_params)
    return f"results/{scenario_name}_{algorithm_name}_{params_str}"


def generate_task_results_root_name(scenario_name: str, task_config) -> str:
    from dataclasses import fields, is_dataclass

    task_params = []
    if is_dataclass(task_config):
        for field in fields(task_config):
            task_params.append(str(getattr(task_config, field.name)))
    else:
        for key in sorted(vars(task_config).keys()):
            task_params.append(str(getattr(task_config, key)))

    params_str = "_".join(task_params)
    return f"results/{scenario_name}_{params_str}"


def generate_task_branch_results_dir_name(
    scenario_name: str,
    branch_name: str,
    task_config,
) -> str:
    return os.path.join(
        generate_task_results_root_name(scenario_name, task_config),
        branch_name,
    )


def load_neat_config_with_substitution(cfg_path, num_inputs, num_outputs, output_path=None):
    with open(cfg_path, "r") as f:
        content = f.read()

    content = content.replace("{num_inputs}", str(num_inputs))
    content = content.replace("{num_outputs}", str(num_outputs))

    if output_path is not None:
        with open(output_path, "w") as f:
            f.write(content)
        return output_path

    with tempfile.NamedTemporaryFile(mode="w", suffix=".cfg", delete=False) as f:
        f.write(content)
        return f.name


def print_block(title: str, rows: Sequence[tuple[str, Any]]):
    print(f"\n[{title}]")
    for key, value in rows:
        print(f"{key}: {value}")


def get_action_bounds(u_range, device, dtype, dynamics_type="Holonomic"):
    u_range_tensor = torch.as_tensor(u_range, device=device, dtype=dtype).flatten()

    if dynamics_type == "DiffDrive":
        if u_range_tensor.numel() == 1:
            u_range_tensor = u_range_tensor.repeat(2)
        u_min = -u_range_tensor[:2]
        u_max = u_range_tensor[:2]
        return u_min, u_max

    if u_range_tensor.numel() == 1:
        u_min = -u_range_tensor
        u_max = u_range_tensor
        return u_min, u_max

    return -u_range_tensor, u_range_tensor


def list_checkpoints(checkpoint_dir: str) -> list[str]:
    if not os.path.exists(checkpoint_dir):
        return []

    checkpoints = [
        filename for filename in os.listdir(checkpoint_dir)
        if filename.startswith(CHECKPOINT_PREFIX)
    ]
    checkpoints.sort(key=lambda name: int(name.split("-")[-1]))
    return checkpoints


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    checkpoints = list_checkpoints(checkpoint_dir)
    if not checkpoints:
        return None
    return os.path.join(checkpoint_dir, checkpoints[-1])


def restore_population_checkpoint(filename: str, population_cls):
    with gzip.open(filename) as f:
        generation, config, population, species_set, rndstate = pickle.load(f)
    random.setstate(rndstate)
    return population_cls(config, (population, species_set, generation))


def load_checkpoint_payload(filename: str):
    with gzip.open(filename) as f:
        generation, config, population, species_set, _ = pickle.load(f)
    return generation, config, population, species_set


def select_best_genome(population: Iterable[Any]):
    best_genome = None
    best_fitness = float("-inf")

    for genome in population.values():
        if genome.fitness is None:
            continue
        if best_genome is None or genome.fitness > best_fitness:
            best_fitness = genome.fitness
            best_genome = genome

    return best_genome


def load_global_best_package(package_path: str):
    if not os.path.exists(package_path):
        return None

    with open(package_path, "rb") as f:
        return pickle.load(f)


def load_generation_target(checkpoint_dir: str, target_generation: int) -> ResultTarget:
    checkpoints = list_checkpoints(checkpoint_dir)
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found. Train the experiment first.")

    checkpoint_name = f"{CHECKPOINT_PREFIX}{target_generation}"
    if checkpoint_name not in checkpoints:
        checkpoint_name = checkpoints[-1]
        resolved_generation = int(checkpoint_name.split("-")[-1])
        message = f"gen {target_generation} missing, using gen {resolved_generation}"
    else:
        resolved_generation = target_generation
        message = f"loading gen {resolved_generation}"

    checkpoint_file = os.path.join(checkpoint_dir, checkpoint_name)
    _, eval_config, population, _ = load_checkpoint_payload(checkpoint_file)
    best_genome = select_best_genome(population)
    if best_genome is None:
        raise ValueError("No valid genome found in checkpoint.")

    return ResultTarget(
        genome=best_genome,
        neat_config=eval_config,
        generation=resolved_generation,
        label=f"gen {resolved_generation}",
        message=f"{message} from {checkpoint_file}",
    )


def load_global_best_target(package_path: str) -> ResultTarget:
    package = load_global_best_package(package_path)
    if package is None:
        raise FileNotFoundError(
            f"Missing global best record: {package_path}. Run a fresh training job first."
        )

    generation = package.get("generation", -1)
    return ResultTarget(
        genome=package["genome"],
        neat_config=package["neat_config"],
        generation=generation,
        label=f"global best (gen {generation})",
        message=f"loading global best from {package_path}",
    )


def build_evaluator_kwargs(
    make_net,
    activate_net,
    make_env,
    n_steps: int,
    batch_size: int,
    render: bool,
    save_render: bool,
    video_dir: str,
    generation: int,
    env_seed: int,
    scenario_name: str,
):
    return {
        "make_net": make_net,
        "activate_net": activate_net,
        "make_env": make_env,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "render": render,
        "save_render": save_render,
        "video_dir": video_dir,
        "generation": generation,
        "env_seed": env_seed,
        "scenario_name": scenario_name,
    }
