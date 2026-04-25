from __future__ import annotations

import os

from experiment.runtime import (
    ensure_results_layout,
    generate_task_branch_results_dir_name,
    print_block,
)


class EaRlExperiment:
    """Placeholder experiment for the ea_rl branch pending a full rewrite."""

    def __init__(
        self,
        task_name: str,
        algorithm_name: str,
        algorithm_config,
        task_config,
        experiment_config,
        seed: int = 0,
    ):
        self.task_name = task_name
        self.algorithm_name = algorithm_name
        self.algorithm_config = algorithm_config
        self.task_config = task_config
        self.config = experiment_config
        self.seed = int(seed)

        _, self.scenario_name = task_name.split("/")
        self._init_results_layout()

    @property
    def results_dir(self) -> str:
        return self.config.results_dir

    def _init_results_layout(self) -> None:
        if self.config.results_dir is None:
            self.config.results_dir = generate_task_branch_results_dir_name(
                self.scenario_name,
                "ea_rl",
                self.task_config,
            )
        paths = ensure_results_layout(self.config.results_dir)
        self.checkpoint_dir = paths.checkpoint_dir
        self.log_dir = paths.log_dir
        self.video_dir = paths.video_dir
        self.placeholder_note_path = os.path.join(self.log_dir, "ea_rl_placeholder.txt")

    def run(self) -> None:
        message = getattr(
            self.algorithm_config,
            "message",
            "ea_rl branch implementation has been removed pending rewrite.",
        )
        with open(self.placeholder_note_path, "w", encoding="utf-8") as handle:
            handle.write("EA-RL placeholder branch\n")
            handle.write(f"task={self.task_name}\n")
            handle.write(f"algorithm={self.algorithm_name}\n")
            handle.write(f"seed={self.seed}\n")
            handle.write(f"message={message}\n")

        print_block(
            "EA-RL Placeholder",
            [
                ("task", self.task_name),
                ("algo", self.algorithm_name),
                ("status", getattr(self.algorithm_config, "status", "placeholder")),
                ("message", message),
                ("results", self.results_dir),
            ],
        )
