"""
BenchMARL基线训练脚本

该脚本用于运行BenchMARL中的多种强化学习算法，
配置与PyTorch-NEAT进化算法对齐，便于后续指标对比。

使用方式:
    python benchmark/run_benchmarl_baseline.py --task transport --seed 42
    python benchmark/run_benchmarl_baseline.py --task navigation --algorithms mappo ippo --seed 42
    python benchmark/run_benchmarl_baseline.py --task all --seed 42  # 运行所有任务
"""

import sys
import os
import argparse
import json
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Any, List, Optional

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 添加BenchMARL路径
benchmarl_root = Path(__file__).parent.parent.parent / "BenchMARL"
sys.path.insert(0, str(benchmarl_root))

import yaml
from omegaconf import OmegaConf

# BenchMARL imports
from benchmarl.algorithms import (
    MappoConfig, IppoConfig, MaddpgConfig, IddpgConfig,
    MasacConfig, IsacConfig, QmixConfig, VdnConfig
)
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig


# 算法配置注册表
ALGORITHM_CONFIGS = {
    "mappo": MappoConfig,
    "ippo": IppoConfig,
    "maddpg": MaddpgConfig,
    "iddpg": IddpgConfig,
    "masac": MasacConfig,
    "isac": IsacConfig,
    "qmix": QmixConfig,
    "vdn": VdnConfig,
}

# VmasTask映射
VMAS_TASK_MAP = {
    "transport": VmasTask.TRANSPORT,
    "navigation": VmasTask.NAVIGATION,
    "dispersion": VmasTask.DISPERSION,
    "balance": VmasTask.BALANCE,
    "flocking": VmasTask.FLOCKING,
    "give_way": VmasTask.GIVE_WAY,
    "multi_give_way": VmasTask.MULTI_GIVE_WAY,
    "passage": VmasTask.PASSAGE,
    "joint_passage": VmasTask.JOINT_PASSAGE,
    "reverse_transport": VmasTask.REVERSE_TRANSPORT,
    "football": VmasTask.FOOTBALL,
    "discovery": VmasTask.DISCOVERY,
    "simple_spread": VmasTask.SIMPLE_SPREAD,
    "simple_tag": VmasTask.SIMPLE_TAG,
    "simple_adversary": VmasTask.SIMPLE_ADVERSARY,
}


def load_neat_config(task_name: str) -> Dict[str, Any]:
    """加载PyTorch-NEAT任务配置
    
    Args:
        task_name: 任务名称，如 "transport"
        
    Returns:
        任务配置字典
    """
    config_path = project_root / "conf" / "task" / "vmas" / f"{task_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"找不到任务配置文件: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 移除defaults字段
    config.pop('defaults', None)
    
    return config


def load_neat_experiment_config() -> Dict[str, Any]:
    """加载PyTorch-NEAT实验配置"""
    config_path = project_root / "conf" / "experiment" / "base_experiment.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 移除defaults字段
    config.pop('defaults', None)
    
    return config


def calculate_aligned_config(
    neat_task_config: Dict[str, Any],
    neat_exp_config: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    """计算与NEAT对齐的BenchMARL配置
    
    设计原则：
    - 只修改需要与NEAT对齐的参数
    - 其他参数使用BenchMARL默认配置
    - 不覆盖设备配置、学习率、批大小等超参数
    
    Args:
        neat_task_config: NEAT任务配置
        neat_exp_config: NEAT实验配置
        seed: 随机种子
        
    Returns:
        对齐后的BenchMARL实验配置参数（只包含需要对齐的参数）
    """
    # 从NEAT配置中提取关键参数
    generations = neat_exp_config.get('generations', 50)
    pop_size = neat_exp_config.get('n_parallel', 300)
    trials = neat_exp_config.get('trials', 200)
    max_steps = neat_task_config.get('max_steps', 200)
    
    # ============ 计算对齐参数 ============
    
    # 1. 总采样量对齐
    # NEAT总步数 = 代数 × 种群大小 × 每回合步数
    neat_total_frames = generations * pop_size * max_steps
    
    # 2. 评估间隔对齐
    # NEAT每代评估一次，对应的帧数 = pop_size × max_steps
    frames_per_generation = pop_size * max_steps
    
    # evaluation_interval（评估频率） 必须是 on_policy_collected_frames_per_batch 的整数倍
    # 使用BenchMARL默认的 on_policy_collected_frames_per_batch (60000)
    default_collected_frames = 60000
    eval_multiplier = max(1, round(frames_per_generation / default_collected_frames))
    evaluation_interval = eval_multiplier * default_collected_frames
    
    # ============ 只返回需要对齐的参数 ============
    # 其他参数（设备、学习率、批大小等）使用BenchMARL默认配置
    
    aligned_config = {
        # 训练总帧数 - 与NEAT对齐
        "max_n_frames": neat_total_frames,
        "max_n_iters": None,
        
        # 评估配置 - 与NEAT对齐
        "evaluation": True,
        "render": True,  
        "evaluation_interval": evaluation_interval,
        "evaluation_episodes": trials,  # 与NEAT的trials对齐
        "evaluation_deterministic_actions": True,
        "evaluation_static": False,  
        
        # 日志配置
        "loggers": ["csv"],  # 基线测试只用csv，不需要wandb
        "create_json": True,
        
        # 检查点配置
        "checkpoint_interval": 0,  # 不保存中间检查点
        "checkpoint_at_end": True,  # 只在结束时保存
        "keep_checkpoints_num": 3,
    }
    
    return aligned_config


def create_experiment(
    task_name: str,
    algorithm_name: str,
    seed: int,
    save_folder: str,
    neat_task_config: Dict[str, Any],
    aligned_exp_config: Dict[str, Any]
) -> Experiment:
    """创建BenchMARL实验
    
    Args:
        task_name: 任务名称
        algorithm_name: 算法名称
        seed: 随机种子
        save_folder: 保存目录
        neat_task_config: NEAT任务配置
        aligned_exp_config: 对齐的实验配置（只包含需要与NEAT对齐的参数）
        
    Returns:
        BenchMARL Experiment对象
        
    Note:
        设备配置、学习率、批大小等超参数使用BenchMARL默认配置
        （可在conf/experiment/base_experiment.yaml中调整）
    """
    # 加载基础实验配置（来自BenchMARL的yaml文件，已包含调优的超参数）
    experiment_config = ExperimentConfig.get_from_yaml()
    
    # 只更新需要与NEAT对齐的配置参数
    for key, value in aligned_exp_config.items():
        if hasattr(experiment_config, key):
            setattr(experiment_config, key, value)
    
    # 设置保存目录
    experiment_config.save_folder = save_folder
    
    # 获取任务
    vmas_task = VMAS_TASK_MAP.get(task_name)
    if vmas_task is None:
        raise ValueError(f"不支持的任务: {task_name}")
    
    # 创建任务配置 - 使用NEAT的任务参数（确保环境配置一致）
    task_config = neat_task_config.copy()
    task = vmas_task.get_task(config=task_config)
    
    # 获取算法配置（使用BenchMARL默认配置）
    algorithm_config_class = ALGORITHM_CONFIGS.get(algorithm_name)
    if algorithm_config_class is None:
        raise ValueError(f"不支持的算法: {algorithm_name}")
    algorithm_config = algorithm_config_class.get_from_yaml()
    
    # 创建模型配置（使用BenchMARL默认配置）
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()
    
    # 创建实验
    experiment = Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=model_config,
        critic_model_config=critic_model_config,
        seed=seed,
        config=experiment_config,
    )
    
    return experiment


# 从utils模块导入公共函数
from benchmark.utils import generate_benchmarl_folder_name as generate_folder_name


def run_single_experiment(
    task_name: str,
    algorithm_name: str,
    seed: int,
    output_dir: str,
    neat_task_config: Dict[str, Any],
    neat_exp_config: Dict[str, Any],
) -> str:
    """运行单个实验
    
    Args:
        task_name: 任务名称
        algorithm_name: 算法名称
        seed: 随机种子
        output_dir: 输出目录
        neat_task_config: 任务配置（启动时已锁定）
        neat_exp_config: 实验配置（启动时已锁定）
        
    Returns:
        实验结果目录路径
    """
    print(f"\n{'='*60}")
    print(f"运行实验: {algorithm_name} on {task_name} (seed={seed})")
    print(f"{'='*60}\n")
    
    # 计算对齐配置（只包含需要与NEAT对齐的参数）
    aligned_config = calculate_aligned_config(
        neat_task_config, neat_exp_config, seed
    )
    
    # 生成文件夹名称（参考NEAT命名方式，包含关键配置参数）
    folder_name = generate_folder_name(task_name, algorithm_name, neat_task_config, seed)
    save_folder = os.path.join(output_dir, folder_name)
    os.makedirs(save_folder, exist_ok=True)
    
    # 保存配置信息
    config_info = {
        "task_name": task_name,
        "algorithm_name": algorithm_name,
        "seed": seed,
        "neat_task_config": neat_task_config,
        "neat_exp_config": {k: v for k, v in neat_exp_config.items() 
                           if k not in ['defaults']},
        "aligned_params": aligned_config,
        "note": "其他参数使用BenchMARL默认配置(conf/experiment/base_experiment.yaml)",
    }
    with open(os.path.join(save_folder, "config_info.json"), 'w') as f:
        json.dump(config_info, f, indent=2, default=str)
    
    # 创建并运行实验
    try:
        experiment = create_experiment(
            task_name=task_name,
            algorithm_name=algorithm_name,
            seed=seed,
            save_folder=save_folder,
            neat_task_config=neat_task_config,
            aligned_exp_config=aligned_config,
        )
        
        # 打印实验配置信息
        print(f"📌 设备配置使用BenchMARL默认: ")
        print(f"   sampling_device: {experiment.config.sampling_device}")
        print(f"   train_device: {experiment.config.train_device}")
        print(f"📌 对齐的参数:")
        print(f"   max_n_frames: {aligned_config['max_n_frames']:,}")
        print(f"   evaluation_interval: {aligned_config['evaluation_interval']:,}")
        print(f"   evaluation_episodes: {aligned_config['evaluation_episodes']}")
        print()
        
        experiment.run()
        print(f"\n实验完成! 结果保存在: {save_folder}")
    except Exception as e:
        print(f"\n实验失败: {e}")
        import traceback
        traceback.print_exc()
        
    return save_folder


def get_available_tasks() -> List[str]:
    """获取可用的任务列表"""
    task_dir = project_root / "conf" / "task" / "vmas"
    tasks = []
    for f in task_dir.glob("*.yaml"):
        task_name = f.stem
        if task_name in VMAS_TASK_MAP:
            tasks.append(task_name)
    return sorted(tasks)


def main():
    parser = argparse.ArgumentParser(
        description="运行BenchMARL基线实验，配置与PyTorch-NEAT对齐",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
配置说明:
  - 只修改与NEAT对齐的参数: max_n_frames, evaluation_interval, evaluation_episodes
  - 其他参数(设备、学习率、批大小等)使用BenchMARL默认配置
  - 可在 conf/experiment/base_experiment.yaml 中调整BenchMARL配置

使用示例:
  python benchmark/run_benchmarl_baseline.py --task transport --algorithms ippo mappo
  python benchmark/run_benchmarl_baseline.py --task all --seed 42
        """
    )
    parser.add_argument(
        "--task", 
        type=str, 
        default="transport",
        help="任务名称，使用 'all' 运行所有可用任务"
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["ippo"],
        help="要运行的算法列表 (默认: ippo)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_results",
        help="输出目录"
    )
    parser.add_argument(
        "--list_tasks",
        action="store_true",
        help="列出所有可用任务"
    )
    parser.add_argument(
        "--list_algorithms",
        action="store_true",
        help="列出所有可用算法"
    )
    
    args = parser.parse_args()
    
    # 列出任务
    if args.list_tasks:
        print("可用任务:")
        for task in get_available_tasks():
            print(f"  - {task}")
        return
    
    # 列出算法
    if args.list_algorithms:
        print("可用算法:")
        for algo in ALGORITHM_CONFIGS.keys():
            print(f"  - {algo}")
        return
    
    # 创建输出目录
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 确定要运行的任务
    if args.task == "all":
        tasks = get_available_tasks()
    else:
        tasks = [args.task]
    
    # 验证算法
    for algo in args.algorithms:
        if algo not in ALGORITHM_CONFIGS:
            print(f"错误: 不支持的算法 '{algo}'")
            print(f"可用算法: {list(ALGORITHM_CONFIGS.keys())}")
            return
    
    
    # 预加载实验配置（所有任务共享）
    locked_exp_config = load_neat_experiment_config()
    
    # 预加载每个任务的配置
    locked_task_configs = {}
    for task in tasks:
        locked_task_configs[task] = load_neat_config(task)
    
    # 运行实验
    results = []
    total_experiments = len(tasks) * len(args.algorithms)
    current = 0
    
    for task in tasks:
        # 获取该任务的预加载配置
        task_config = locked_task_configs.get(task)
        if task_config is None:
            print(f"跳过任务 {task}：配置加载失败")
            continue
            
        for algo in args.algorithms:
            current += 1
            print(f"\n[{current}/{total_experiments}] 开始实验...")
            
            try:
                result_dir = run_single_experiment(
                    task_name=task,
                    algorithm_name=algo,
                    seed=args.seed,
                    output_dir=output_dir,
                    neat_task_config=task_config,
                    neat_exp_config=locked_exp_config,
                )
                results.append({
                    "task": task,
                    "algorithm": algo,
                    "seed": args.seed,
                    "status": "success",
                    "result_dir": result_dir
                })
            except Exception as e:
                results.append({
                    "task": task,
                    "algorithm": algo,
                    "seed": args.seed,
                    "status": "failed",
                    "error": str(e)
                })
    
    # 保存运行摘要
    summary_path = os.path.join(output_dir, "run_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("所有实验完成!")
    print(f"结果保存在: {output_dir}")
    print(f"摘要文件: {summary_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()