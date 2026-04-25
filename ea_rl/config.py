from dataclasses import dataclass
from typing import Optional


@dataclass
class EaRlAlgorithmConfig:
    branch: str = "ea_rl"
    name: str = "evo_td3"
    status: str = "placeholder"
    message: str = "ea_rl branch implementation has been removed pending rewrite."


@dataclass
class EaRlExperimentConfig:
    device: str = "cpu"
    continuous_actions: bool = True
    results_dir: Optional[str] = None
    overwrite: bool = False
    render: bool = False
    save_render: bool = False
    collect_results: bool = True
    eval_interval: int = 10
    checkpoint_interval: int = 10
    save_replay: bool = False
    progress_interval: int = 1
    candidate_progress: bool = False
