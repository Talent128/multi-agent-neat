"""
NEAT进化实验模块

管理整个NEAT进化训练流程
"""
from dataclasses import dataclass
from typing import Optional
import os
import time
import multiprocessing

import neat
import torch
from vmas import make_env

from pytorch_neat.recurrent_net import RecurrentNet
from pytorch_neat.activations import normalize_output_groups

from .evaluator import GenomeEvaluator
from .batch_evaluator import BatchGenomeEvaluator
from .neat_reporter import LogReporter, GlobalBestReporter
from .runtime import (
    build_evaluator_kwargs,
    ensure_results_layout,
    find_latest_checkpoint,
    generate_results_dir_name,
    get_action_bounds,
    load_global_best_package,
    load_global_best_target,
    load_generation_target,
    load_neat_config_with_substitution,
    print_block,
    restore_population_checkpoint,
)
from .better_pop import better_Population


@dataclass
class ExperimentConfig:
    """实验配置类"""
    generations: int
    trials: int
    n_parallel: int

    device: str = "cpu"
    continuous_actions: bool = True
    overwrite: bool = False

    render: bool = False
    save_render: bool = False
    train_render: bool = False  # 训练时是否渲染（默认False，仅在result_of_experiment中渲染）
    show_gen: int = -1
    collect_results: bool = True

    checkpt_freq: int = 1
    results_dir: Optional[str] = None  # 结果目录，null表示自动生成


class Experiment:
    """NEAT进化实验类"""
    
    def __init__(
        self,
        task_name: str,
        algorithm_name: str,
        algorithm_config: dict,
        task_config: dict,
        experiment_config: ExperimentConfig,
        seed: int = 0,
    ):
        """初始化实验
        
        Args:
            task_name (str): 任务名称，如 "vmas/transport"
            algorithm_name (str): 算法名称，如 "recurrent"
            algorithm_config (dict): 算法配置字典
            task_config (dict): 任务配置字典
            experiment_config (ExperimentConfig): 实验配置
            seed (int): 随机种子
        """
        self.task_name = task_name
        self.algorithm_name = algorithm_name
        self.algorithm_config = algorithm_config
        self.task_config = task_config
        self.config = experiment_config
        self.seed = seed

        self.just_restored = False  # 记录是否刚从检查点恢复

        self.env_name, self.scenario_name = task_name.split("/")
        assert self.env_name == "vmas", f"仅支持vmas环境，当前: {self.env_name}"
        self.n_agents = getattr(task_config, 'n_agents', 'default')
        self._init_results_layout()
        self._init_space_dimensions()
        self._print_setup_summary()

        # 加载NEAT配置
        self.neat_config = self._load_neat_config()

        # 初始化种群
        self.population = None

    def _init_results_layout(self):
        if self.config.results_dir is None:
            self.config.results_dir = generate_results_dir_name(
                self.scenario_name, self.algorithm_name, self.task_config
            )

        paths = ensure_results_layout(self.config.results_dir)
        self.checkpoint_dir = paths.checkpoint_dir
        self.log_dir = paths.log_dir
        self.video_dir = paths.video_dir
        self.global_best_package_path = paths.global_best_package_path

    def _init_space_dimensions(self):
        test_env = self._make_env(num_envs=1)
        try:
            obs = test_env.reset()
            self.obs_dim = obs[0].shape[-1]

            if hasattr(test_env.agents[0], 'action_space') and hasattr(test_env.agents[0].action_space, 'shape'):
                self.action_dim = test_env.agents[0].action_space.shape[0]
            else:
                self.action_dim = 2
        finally:
            if hasattr(test_env, "close"):
                test_env.close()

    def _print_setup_summary(self):
        print_block(
            "Experiment",
            [
                ("task", self.task_name),
                ("algo", self.algorithm_name),
                ("obs_dim", self.obs_dim),
                ("action_dim", self.action_dim),
                ("agents", self.n_agents),
                ("trials", self.config.trials),
                ("n_parallel", self.config.n_parallel),
                ("device", self.config.device),
                ("results", self.config.results_dir),
            ],
        )

    def _load_neat_config(self):
        """加载NEAT配置，替换输入输出维度"""
        cfg_path = self.algorithm_config.neat_config_path

        neat_cfg_filename = os.path.basename(cfg_path)
        permanent_cfg_path = os.path.join(self.config.results_dir, neat_cfg_filename)

        if os.path.exists(permanent_cfg_path):
            temp_cfg_path = permanent_cfg_path
        else:
            temp_cfg_path = load_neat_config_with_substitution(
                cfg_path, self.obs_dim, self.action_dim, permanent_cfg_path
            )

        print(f"neat_cfg: {permanent_cfg_path}")

        return neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            temp_cfg_path
        )

    def _get_generation_env_seed(self, generation: int) -> int:
        """返回某一代评估使用的环境 seed。"""
        return int(self.seed + generation)

    def _n_steps(self) -> int:
        return getattr(self.task_config, 'max_steps', 200)

    def _build_evaluator_kwargs(
        self,
        *,
        generation: int,
        env_seed: int,
        render: bool,
        save_render: bool,
    ):
        return build_evaluator_kwargs(
            make_net=self._make_net,
            activate_net=self._activate_net,
            make_env=self._make_env,
            n_steps=self._n_steps(),
            batch_size=self.config.trials,
            render=render,
            save_render=save_render,
            video_dir=self.video_dir,
            generation=generation,
            env_seed=env_seed,
            scenario_name=self.scenario_name,
        )

    @staticmethod
    def _assign_eval_stats(genomes, results):
        genomes_dict = dict(genomes)
        for genome_id, stats in results:
            genomes_dict[genome_id].fitness = stats.mean
            genomes_dict[genome_id].eval_stats = stats

    def _make_env(self, num_envs=None, seed=None):
        """创建VMAS环境

        Args:
            num_envs (int, optional): 个体评估时并行环境数量，默认使用config.trials
            seed (int, optional): 环境随机种子，默认使用实验基础 seed

        Returns:
            vmas.Env: VMAS环境实例
        """
        if num_envs is None:
            num_envs = self.config.trials
        if seed is None:
            seed = self.seed

        # 准备环境参数（task_config是dataclass）
        from dataclasses import asdict
        env_kwargs = asdict(self.task_config)

        env = make_env(
            scenario=self.scenario_name,
            num_envs=num_envs,
            device=self.config.device,
            continuous_actions=self.config.continuous_actions,
            wrapper=None,
            seed=seed,
            **env_kwargs
        )

        return env

    def _make_net(self, genome, config, batch_size):
        """根据算法类型创建神经网络

        Args:
            genome: NEAT基因组
            config: NEAT配置
            batch_size (int): 批量大小（通常等于experiment.trials，即并行环境数）

        Returns:
            网络实例

        Note:
            - batch_size: 算法配置中的batch_size会被此处覆盖，使用experiment.trials
            - device: 使用experiment.device（会覆盖算法配置中的device）
        """
        device = self.config.device  # 使用实验配置中的device

        if self.algorithm_name == "recurrent":
            return RecurrentNet.create(
                genome,
                config,
                batch_size=batch_size,
                use_current_activs=getattr(self.algorithm_config, 'use_current_activs', False),
                n_internal_steps=getattr(self.algorithm_config, 'n_internal_steps', 1),
                prune_empty=getattr(self.algorithm_config, 'prune_empty', True),
                device=device,
            )
        else:
            raise ValueError(f"未知算法: {self.algorithm_name}")

    def _activate_net(self, net, obs, u_range, dynamics_type="Holonomic"):
        """激活网络获取动作

        Args:
            net: 神经网络
            obs (torch.Tensor): 观测，shape=(batch_size, obs_dim)
            u_range: 动作范围，含义取决于动力学模型：
                - Holonomic: float或list，表示[-u_range, u_range]的对称范围
                - DiffDrive: [最大线速度, 最大角速度]，网络输出直接映射到[-max, max]
            dynamics_type (str): 动力学模型类型，如 "Holonomic", "DiffDrive" 等

        Returns:
            torch.Tensor: 动作，shape=(batch_size, action_dim)
        """
        # 激活网络
        with torch.no_grad():
            output = net.activate(obs)

        if not hasattr(net, "output_activation_groups"):
            raise ValueError("Network is missing output_activation_groups; node activations are required.")

        normalized_output = normalize_output_groups(output, net.output_activation_groups)
        u_min, u_max = get_action_bounds(
            u_range,
            device=output.device,
            dtype=output.dtype,
            dynamics_type=dynamics_type,
        )
        action = normalized_output * (u_max - u_min) + u_min
        return action

    def _eval_genome_worker(self, args):
        """评估基因组的工作函数

        Args:
            args: (genome_id, genome, config, evaluator_kwargs)

        Returns:
            (genome_id, EvalStats): 返回详细统计，stats.mean 作为适应度
        """
        genome_id, genome, config, evaluator_kwargs = args

        # 创建评估器
        evaluator = GenomeEvaluator(**evaluator_kwargs)

        # 评估基因组，返回详细统计
        stats = evaluator.eval_genome(genome, config, debug=False)

        return genome_id, stats

    def _get_best_stats(self, genome, config, debug=False):
        """获取最佳个体的详细统计（直接从 genome.eval_stats 获取）

        Args:
            genome: NEAT基因组（最佳个体）
            config: NEAT配置（未使用，保持接口兼容）
            debug: 是否启用调试输出

        Returns:
            EvalStats: 训练时已计算的详细统计数据
        """
        if not hasattr(genome, 'eval_stats') or genome.eval_stats is None:
            raise ValueError("基因组缺少 eval_stats，请确保训练评估时已计算详细统计")
        
        if debug:
            stats = genome.eval_stats
            print(
                "best_stats: "
                f"mean={stats.mean:.4f}, std={stats.std:.4f}, "
                f"max={stats.max_val:.4f}, min={stats.min_val:.4f}"
            )
        return genome.eval_stats

    def _eval_genomes(self, genomes, config):
        """评估所有基因组（并行）,用于训练过程

        Args:
            genomes: 基因组列表 [(genome_id, genome), ...]
            config: NEAT配置
        """
        # 如果刚从检查点恢复，跳过这一代的评估（已有适应度）
        if self.just_restored:
            print("resume: skip first evaluation, reuse checkpoint fitness")
            self.just_restored = False
            return

        is_cuda = 'cuda' in self.config.device.lower()
        generation = getattr(self.population, "generation", 0)
        generation_env_seed = self._get_generation_env_seed(generation)
        
        if is_cuda and self.config.n_parallel > 1:
            self._eval_genomes_batch(genomes, config, generation_env_seed)
        else:
            self._eval_genomes_parallel(genomes, config, generation, generation_env_seed)
    
    def _eval_genomes_batch(self, genomes, config, generation_env_seed):
        """GPU批量评估：单进程，利用GPU并行能力
        
        使用 BatchGenomeEvaluator，通过 BatchedRecurrentNet 将多个网络参数
        打包成统一张量，使用 einsum 进行批量矩阵乘法，实现高效并行评估。
        
        注意：每次调用时动态创建评估器，评估完成后释放。
        这与CPU模式保持一致，避免在Experiment实例中持久存储vmas环境，
        从而防止checkpoint保存时的pickle序列化问题。
        """
        n_parallel = self.config.n_parallel

        batch_evaluator = BatchGenomeEvaluator(
            make_net=self._make_net,
            activate_net=self._activate_net,
            make_env=self._make_env,
            n_genomes_batch=n_parallel,
            trials_per_genome=self.config.trials,
            n_steps=self._n_steps(),
            device=self.config.device,
            env_seed=generation_env_seed,
        )

        for batch_start in range(0, len(genomes), n_parallel):
            batch_end = min(batch_start + n_parallel, len(genomes))
            batch_genomes = genomes[batch_start:batch_end]
            results = batch_evaluator.eval_genomes_batch(batch_genomes, config)
            self._assign_eval_stats(genomes, results)
    
    def _eval_genomes_parallel(self, genomes, config, generation, generation_env_seed):
        """CPU多进程并行评估"""
        evaluator_kwargs = self._build_evaluator_kwargs(
            generation=generation,
            env_seed=generation_env_seed,
            render=self.config.train_render,
            save_render=False,
        )

        args_list = [
            (genome_id, genome, config, evaluator_kwargs)
            for genome_id, genome in genomes
        ]

        if self.config.n_parallel > 1:
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(self.config.n_parallel) as pool:
                results = pool.map(self._eval_genome_worker, args_list)
        else:
            results = [self._eval_genome_worker(args) for args in args_list]

        self._assign_eval_stats(genomes, results)

    def run(self):
        if self.config.render:
            self.result_of_experiment()
            return

        need_train, checkpoint_file = self._resolve_training_state()
        if need_train:
            self._train(checkpoint_file)

        self.result_of_experiment()

    def _resolve_training_state(self):
        checkpoint_file = find_latest_checkpoint(self.checkpoint_dir)

        if self.config.overwrite:
            print("train: overwrite enabled, starting fresh")
            return True, None

        if checkpoint_file is None:
            print("train: no checkpoint, starting fresh")
            return True, None

        last_gen = int(checkpoint_file.split('-')[-1])
        if last_gen < self.config.generations - 1:
            print(f"train: resume from gen {last_gen}")
            return True, checkpoint_file

        print(f"train: already complete at gen {last_gen}")
        return False, checkpoint_file

    def _init_population(self, checkpoint_file=None):
        if checkpoint_file is not None:
            print(f"resume: {checkpoint_file}")
            self.population = restore_population_checkpoint(checkpoint_file, better_Population)
            self.just_restored = True
            start_generation = int(checkpoint_file.split('-')[-1])
        else:
            self.population = better_Population(self.neat_config)
            start_generation = 0

        return start_generation

    def _reset_global_best_record(self, checkpoint_file):
        reset_global_best = checkpoint_file is None
        if reset_global_best and os.path.exists(self.global_best_package_path):
            os.remove(self.global_best_package_path)
        return reset_global_best

    def _restore_global_best_record(self, checkpoint_file):
        if checkpoint_file is not None:
            global_best_package = load_global_best_package(self.global_best_package_path)
            if global_best_package is not None:
                self.population.best_genome = global_best_package["genome"]
                print(
                    "resume_best: "
                    f"gen={global_best_package.get('generation')}, "
                    f"fitness={global_best_package.get('fitness'):.4f}"
                )

    def _add_reporters(self, reset_global_best):
        self.population.add_reporter(neat.StdOutReporter(True))
        self.population.add_reporter(neat.StatisticsReporter())

        self.population.add_reporter(
            neat.Checkpointer(
                self.config.checkpt_freq,
                filename_prefix=os.path.join(self.checkpoint_dir, 'neat-checkpoint-')
            )
        )
        self.population.add_reporter(
            GlobalBestReporter(
                self.global_best_package_path,
                reset_existing=reset_global_best,
            )
        )

        if self.config.collect_results:
            log_file = os.path.join(self.log_dir, "log.json")
            logger = LogReporter(log_file, self._get_best_stats, eval_with_debug=False)
            if self.just_restored:
                logger.skip_next_log = True
            self.population.add_reporter(logger)
            print(f"log: {log_file}")
            print(f"best_log: {os.path.join(self.log_dir, 'best_log.json')}")

    def _finalize_winner(self, winner):
        global_best_package = self._get_global_best_package()
        if global_best_package is not None:
            return global_best_package["genome"]
        return winner

    def _print_training_summary(self, winner, elapsed_time):
        print_block(
            "Train",
            [
                ("minutes", f"{elapsed_time / 60:.2f}"),
                ("generations", self.config.generations),
                ("best_fitness", f"{winner.fitness:.4f}"),
            ],
        )

    def _train(self, checkpoint_file=None):
        print(f"\n[Train]\ngenerations: {self.config.generations}")

        start_generation = self._init_population(checkpoint_file)
        reset_global_best = self._reset_global_best_record(checkpoint_file)
        self._restore_global_best_record(checkpoint_file)
        self._add_reporters(reset_global_best)

        start_time = time.time()
        winner = self.population.run(
            self._eval_genomes,
            self.config.generations - start_generation
        )
        winner = self._finalize_winner(winner)
        elapsed_time = time.time() - start_time

        self._print_training_summary(winner, elapsed_time)
        return winner

    def _load_global_best_package(self):
        return load_global_best_package(self.global_best_package_path)

    def _get_global_best_package(self):
        package = self._load_global_best_package()
        if package is not None:
            package.setdefault("source", "saved_global_best")
            return package
        return None
    
    ####################################################################################################################
    # output functions
    ####################################################################################################################
    def _resolve_result_target(self):
        if self.config.show_gen >= 0:
            return load_generation_target(self.checkpoint_dir, self.config.show_gen)
        return load_global_best_target(self.global_best_package_path)

    def _result_env_seed(self, generation: int) -> int:
        if self.config.render:
            return time.time_ns() % (2**32)
        return self._get_generation_env_seed(generation)

    def _print_result_summary(self, target, stats, elapsed_time):
        print_block(
            "Result",
            [
                ("target", target.label),
                ("seconds", f"{elapsed_time:.2f}"),
                ("steps", self._n_steps()),
                ("trials", self.config.trials),
                ("device", self.config.device),
                ("fitness_mean", f"{stats.mean:.4f}"),
                ("fitness_std", f"{stats.std:.4f}"),
                ("fitness_max", f"{stats.max_val:.4f}"),
                ("fitness_min", f"{stats.min_val:.4f}"),
            ],
        )

    def result_of_experiment(self):
        target = self._resolve_result_target()
        print(f"\n[Result]\n{target.message}")

        start_time = time.time()
        evaluator = GenomeEvaluator(
            **self._build_evaluator_kwargs(
                generation=target.generation,
                env_seed=self._result_env_seed(target.generation),
                render=self.config.render,
                save_render=self.config.save_render,
            )
        )
        stats = evaluator.eval_genome(target.genome, target.neat_config, debug=True)
        elapsed_time = time.time() - start_time
        self._print_result_summary(target, stats, elapsed_time)
