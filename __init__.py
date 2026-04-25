import importlib
import pkgutil
from pathlib import Path

_has_hydra = importlib.util.find_spec("hydra") is not None

if _has_hydra:
    from hydra.core.config_store import ConfigStore

    from experiment.experiment import ExperimentConfig
    from ea_rl.config import EaRlAlgorithmConfig, EaRlExperimentConfig
    from pytorch_neat.config import RecurrentNetConfig

    cs = ConfigStore.instance()
    # Algorithm schemas
    cs.store(group="algorithm", name="recurrent_config", node=RecurrentNetConfig)
    cs.store(group="algorithm/pure_neat", name="recurrent_config", node=RecurrentNetConfig)
    cs.store(group="algorithm/ea_rl", name="evo_rl_config", node=EaRlAlgorithmConfig)
    # Experiment schema
    cs.store(group="experiment", name="experiment_config", node=ExperimentConfig)
    cs.store(group="experiment", name="ea_rl_experiment_config", node=EaRlExperimentConfig)

    def _register_vmas_tasks():
        """Auto-register VMAS task TaskConfig dataclasses for Hydra."""

        base_pkg = "environments.vmas"
        pkg_path = Path(__file__).resolve().parent / "environments" / "vmas"
        if not pkg_path.exists():
            return

        for module_info in pkgutil.iter_modules([str(pkg_path)]):
            name = module_info.name
            if name.startswith("__"):
                continue
            module = importlib.import_module(f"{base_pkg}.{name}")
            task_config = getattr(module, "TaskConfig", None)
            if task_config is None:
                continue
            # YAML defaults expect names like vmas_<module>_config
            cs.store(group="task", name=f"vmas_{name}_config", node=task_config)

    _register_vmas_tasks()
