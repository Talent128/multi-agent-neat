from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RecurrentNetConfig:
    """Hydra schema for RecurrentNet."""

    neat_config_path: str = "conf/algorithm/neat_config/recurrent.cfg"
    #batch_size: Optional[int] = None  # None -> fallback to experiment.trials
    activation: str = "sigmoid"
    prune_empty: bool = False
    use_current_activs: bool = False
    n_internal_steps: int = 1
    dtype: str = "float64"
    #device: str = "cpu"
