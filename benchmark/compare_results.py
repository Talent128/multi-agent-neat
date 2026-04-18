"""
结果对比分析脚本

该脚本用于对比PyTorch-NEAT（进化算法）和BenchMARL（强化学习）的训练结果。
生成曲线图和统计表格，便于分析两种方法的性能差异。

使用方式:
    python benchmark/compare_results.py --task transport --seed 42
    python benchmark/compare_results.py --neat_dir results/transport_recurrent_... --algorithms mappo ippo

    python benchmark/benchmark_script.py --task transport --algorithms iddpg ippo maddpg mappo qmix vdn --skip_training 若有已有数据结果，可跳过强化基线训练进行比较绘图
"""

import sys
import os
import argparse
import json
import glob
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import torch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 从utils模块导入公共函数
from benchmark.utils import (
    load_task_config_from_yaml,
    load_task_config_from_benchmarl,
    find_neat_results,
    find_benchmarl_results,
    get_config_name as get_neat_config_name
)

# 启发式策略场景映射
HEURISTIC_SCENARIOS = {
    'transport': 'vmas.scenarios.transport',
    'flocking': 'vmas.scenarios.flocking', 
    'navigation': 'vmas.scenarios.navigation',
    'balance': 'vmas.scenarios.balance',
    'discovery': 'vmas.scenarios.discovery',
    'give_way': 'vmas.scenarios.give_way',
    'joint_passage': 'vmas.scenarios.joint_passage',
}


def generate_heuristic_results_dir_name(
    scenario_name: str,
    n_steps: int,
    env_kwargs: Dict[str, Any],
) -> str:
    """生成启发式评估结果目录名。

    命名逻辑需要与 benchmark/run_heuristic_baseline.py 保持一致。
    """
    parts = [scenario_name, "heuristic", str(n_steps)]

    for key in sorted(env_kwargs.keys()):
        value = env_kwargs[key]
        if isinstance(value, bool):
            parts.append(str(value))
        elif isinstance(value, float):
            value_str = str(value)
            parts.append(value_str.rstrip("0").rstrip(".") if "." in value_str else value_str)
        else:
            parts.append(str(value))

    return "_".join(parts)


def find_heuristic_results(
    scenario_name: str,
    n_steps: Optional[int] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    查找已有的启发式评估结果目录
    
    Args:
        scenario_name: 场景名称
        n_steps: 仿真步数（可选，用于精确匹配）
        env_kwargs: 环境参数（可选，用于精确匹配）
        
    Returns:
        结果目录路径，如果不存在返回None
    """
    results_base = project_root / "benchmark_results"

    # 如果提供了完整参数，优先精确匹配，避免不同环境配置串用结果
    if n_steps is not None and env_kwargs is not None:
        dir_name = generate_heuristic_results_dir_name(
            scenario_name=scenario_name,
            n_steps=n_steps,
            env_kwargs=env_kwargs,
        )
        expected_path = results_base / dir_name
        if expected_path.exists() and (expected_path / "summary.json").exists():
            return str(expected_path)
    
    # 查找匹配的目录
    pattern = f"{scenario_name}_heuristic*"
    matches = [path for path in results_base.glob(pattern) if (path / "summary.json").exists()]
    
    if matches:
        # 返回最新的
        return str(sorted(matches, key=lambda x: x.stat().st_mtime, reverse=True)[0])
    
    return None


def load_heuristic_results(results_dir: str) -> Optional[float]:
    """
    从已有结果目录加载启发式评估结果
    
    Args:
        results_dir: 结果目录
        
    Returns:
        总回报，如果不存在返回None
    """
    summary_file = os.path.join(results_dir, 'summary.json')
    if not os.path.exists(summary_file):
        return None
    
    with open(summary_file, 'r') as f:
        data = json.load(f)
        return data.get('total_reward')


def get_or_compute_heuristic_reward(
    scenario_name: str,
    n_steps: int = 200,
    n_envs: int = 200,
    env_kwargs: dict = None,
    device: str = "cpu",
    force_compute: bool = False,
) -> Optional[float]:
    """
    获取启发式策略回报（优先使用已有结果）
    
    Args:
        scenario_name: 场景名称
        n_steps: 仿真步数
        n_envs: 并行环境数量
        env_kwargs: 环境参数
        device: 计算设备
        force_compute: 是否强制重新计算
        
    Returns:
        总回报
    """
    # 尝试从已有结果加载
    if not force_compute:
        heuristic_dir = find_heuristic_results(
            scenario_name=scenario_name,
            n_steps=n_steps,
            env_kwargs=env_kwargs,
        )
        if heuristic_dir:
            reward = load_heuristic_results(heuristic_dir)
            if reward is not None:
                print(f"从已有结果加载启发式策略回报: {heuristic_dir}")
                return reward
    
    # 没有已有结果，进行计算
    print(f"计算启发式策略回报...")
    return get_heuristic_reward(scenario_name, n_steps, n_envs, env_kwargs, device)


@dataclass
class MetricData:
    """指标数据类"""
    steps: np.ndarray  # 步数（横轴）
    mean: np.ndarray   # 均值
    std: np.ndarray    # 标准差
    max_val: np.ndarray  # 最大值
    min_val: Optional[np.ndarray] = None  # 最小值


def get_heuristic_reward(
    scenario_name: str,
    n_steps: int = 200,
    n_envs: int = 200,
    env_kwargs: dict = None,
    device: str = "cpu",
) -> float:
    """
    使用启发式策略运行VMAS场景并获取平均总回报
    
    Args:
        scenario_name: 场景名称字符串 (如 "transport", "flocking")
        n_steps: 仿真步数
        n_envs: 并行环境数量
        env_kwargs: 传递给场景的字典参数
        device: PyTorch计算设备
        
    Returns:
        float: 平均总回报（累计奖励的均值）
    """
    from vmas import make_env
    
    if env_kwargs is None:
        env_kwargs = {}
    
    # 动态导入对应场景的启发式策略
    if scenario_name not in HEURISTIC_SCENARIOS:
        print(f"警告: 场景 {scenario_name} 没有对应的启发式策略")
        return None
    
    try:
        module_path = HEURISTIC_SCENARIOS[scenario_name]
        module = __import__(module_path, fromlist=['HeuristicPolicy'])
        HeuristicPolicy = getattr(module, 'HeuristicPolicy')
    except (ImportError, AttributeError) as e:
        print(f"警告: 无法导入 {scenario_name} 的启发式策略: {e}")
        return None
    
    # 创建策略
    policy = HeuristicPolicy(continuous_action=True)
    
    # 创建环境
    env = make_env(
        scenario=scenario_name,
        num_envs=n_envs,
        device=device,
        continuous_actions=True,
        wrapper=None,
        **env_kwargs,
    )
    
    # 重置环境
    obs = env.reset()
    total_reward = 0.0
    
    # 运行仿真
    for _ in range(n_steps):
        # 为每个智能体计算动作
        actions = [None] * len(obs)
        for i in range(len(obs)):
            actions[i] = policy.compute_action(
                obs[i],
                u_range=env.agents[i].u_range
            )
        
        # 执行动作
        obs, rews, dones, info = env.step(actions)
        
        # 计算和累积奖励
        rewards = torch.stack(rews, dim=1)  # (n_envs, n_agents)
        global_reward = rewards.mean(dim=1)  # (n_envs,)
        mean_global_reward = global_reward.mean(dim=0)
        total_reward += mean_global_reward.item()
    
    return total_reward


@dataclass
class NEATBestData:
    """NEAT最佳个体数据类
    
    只包含最佳个体的详细统计，用于与RL进行公平对比。
    """
    steps: np.ndarray      # 步数（横轴）
    best_mean: np.ndarray  # 最佳个体适应度均值
    best_std: np.ndarray   # 最佳个体适应度标准差
    best_max: np.ndarray   # 最佳个体适应度最大值
    best_min: np.ndarray   # 最佳个体适应度最小值


def load_neat_best_results(log_dir: str) -> NEATBestData:
    """加载PyTorch-NEAT的最佳个体统计
    
    从 best_log.json 加载最佳个体的详细统计数据。
    
    Args:
        log_dir: NEAT日志目录路径（包含best_log.json）
        
    Returns:
        NEATBestData对象
    """
    best_log_file = os.path.join(log_dir, 'best_log.json')
    
    if not os.path.exists(best_log_file):
        raise FileNotFoundError(f"未找到 best_log.json: {best_log_file}")
    
    generations = []
    best_mean = []
    best_std = []
    best_max = []
    best_min = []
    
    with open(best_log_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                generations.append(data['generation'])
                best_mean.append(data.get('best_mean', 0))
                best_std.append(data.get('best_std', 0))
                best_max.append(data.get('best_max', 0))
                best_min.append(data.get('best_min', 0))
    
    print(f"已加载最佳个体详细统计: {best_log_file}")
    
    return NEATBestData(
        steps=np.array(generations),  # 先存代数，后面转换为步数
        best_mean=np.array(best_mean),
        best_std=np.array(best_std),
        best_max=np.array(best_max),
        best_min=np.array(best_min),
    )


def load_neat_config(results_dir: str) -> Tuple[int, int, int]:
    """从NEAT结果目录加载配置
    
    Args:
        results_dir: NEAT结果目录
        
    Returns:
        (pop_size, max_steps, trials) 元组
    """
    # 从.hydra目录加载配置
    hydra_dir = os.path.join(results_dir, '.hydra')
    if os.path.exists(hydra_dir):
        config_file = os.path.join(hydra_dir, 'config.yaml')
        if os.path.exists(config_file):
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                pop_size = config.get('experiment', {}).get('n_parallel', 150)
                max_steps = config.get('task', {}).get('max_steps', 200)
                trials = config.get('experiment', {}).get('trials', 60)
                return pop_size, max_steps, trials
    
    # 默认值
    return 150, 200, 60


def load_task_env_kwargs(results_dir: str) -> Tuple[str, int, dict]:
    """从NEAT结果目录加载任务配置和环境参数
    
    Args:
        results_dir: NEAT结果目录
        
    Returns:
        (scenario_name, max_steps, env_kwargs) 元组
    """
    hydra_dir = os.path.join(results_dir, '.hydra')
    
    # 从目录名中提取场景名
    dir_name = os.path.basename(results_dir)
    scenario_name = dir_name.split('_')[0]  # 第一部分是场景名
    
    max_steps = 200
    env_kwargs = {}
    
    if os.path.exists(hydra_dir):
        config_file = os.path.join(hydra_dir, 'config.yaml')
        if os.path.exists(config_file):
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
                
                # 获取max_steps
                max_steps = config.get('task', {}).get('max_steps', 200)
                
                # 获取环境参数（排除max_steps）
                task_config = config.get('task', {})
                env_kwargs = {k: v for k, v in task_config.items() if k != 'max_steps'}
    
    return scenario_name, max_steps, env_kwargs


def load_benchmarl_results(results_dir: str) -> MetricData:
    """加载BenchMARL的训练结果
    
    Args:
        results_dir: BenchMARL结果目录
        
    Returns:
        MetricData对象
    """
    # 查找JSON结果文件
    json_files = glob.glob(os.path.join(results_dir, "**/*.json"), recursive=True)
    
    # 过滤出评估结果文件
    eval_json = None
    for jf in json_files:
        if 'config_info' not in jf and 'run_summary' not in jf:
            try:
                with open(jf, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        def find_step_data(d):
                            if isinstance(d, dict):
                                for k, v in d.items():
                                    if k.startswith('step_'):
                                        return d
                                    result = find_step_data(v)
                                    if result:
                                        return result
                            return None
                        
                        step_data = find_step_data(data)
                        if step_data:
                            eval_json = jf
                            break
            except:
                continue
    
    if eval_json is None:
        return load_benchmarl_csv_results(results_dir)
    
    # 解析JSON结果
    with open(eval_json, 'r') as f:
        data = json.load(f)
    
    def extract_step_data(d):
        if isinstance(d, dict):
            step_data = {}
            for k, v in d.items():
                if k.startswith('step_'):
                    step_data[k] = v
                elif isinstance(v, dict):
                    result = extract_step_data(v)
                    if result:
                        return result
            if step_data:
                return step_data
        return None
    
    step_data = extract_step_data(data)
    
    if not step_data:
        return load_benchmarl_csv_results(results_dir)
    
    steps = []
    returns_mean = []
    returns_std = []
    returns_max = []
    returns_min = []
    
    for step_key in sorted(step_data.keys(), key=lambda x: int(x.split('_')[1])):
        step_info = step_data[step_key]
        step_count = step_info.get('step_count', 0)
        
        return_data = step_info.get('return', [])
        if isinstance(return_data, list) and len(return_data) > 0:
            steps.append(step_count)
            returns_mean.append(np.mean(return_data))
            returns_std.append(np.std(return_data))
            returns_max.append(np.max(return_data))
            returns_min.append(np.min(return_data))
    
    if not steps:
        return load_benchmarl_csv_results(results_dir)
    
    return MetricData(
        steps=np.array(steps),
        mean=np.array(returns_mean),
        std=np.array(returns_std),
        max_val=np.array(returns_max),
        min_val=np.array(returns_min),
    )


def load_benchmarl_csv_results(results_dir: str) -> MetricData:
    """从CSV文件加载BenchMARL结果
    
    注意：BenchMARL的CSV文件可能没有表头，格式为：
    evaluation_index,episode_reward_mean
    或
    step_count,episode_reward_mean
    
    需要根据文件名判断是评估数据还是训练数据。
    """
    import csv
    
    csv_files = glob.glob(os.path.join(results_dir, "**/*.csv"), recursive=True)
    
    if not csv_files:
        raise FileNotFoundError(f"在 {results_dir} 中未找到CSV结果文件")
    
    # 优先查找评估相关的CSV文件
    eval_csv = None
    for cf in csv_files:
        if 'eval' in cf.lower() and 'reward' in cf.lower() and 'mean' in cf.lower():
            eval_csv = cf
            break
    
    if eval_csv is None:
        # 回退：查找任何包含eval的CSV
        for cf in csv_files:
            if 'eval' in cf.lower():
                eval_csv = cf
                break
    
    if eval_csv is None:
        eval_csv = csv_files[0]
    
    steps = []
    returns_mean = []
    returns_std = []
    returns_max = []
    
    # 尝试读取CSV文件（可能没有表头）
    with open(eval_csv, 'r') as f:
        # 先读取第一行判断是否有表头
        first_line = f.readline().strip()
        f.seek(0)  # 重置文件指针
        
        # 检查第一行是否是数字（无表头）还是字段名（有表头）
        try:
            first_col = first_line.split(',')[0]
            float(first_col)  # 如果能转换为数字，说明没有表头
            has_header = False
        except (ValueError, IndexError):
            has_header = True
        
        if has_header:
            # 有表头，使用DictReader
            reader = csv.DictReader(f)
            for row in reader:
                step = None
                mean_return = None
                
                for key in row.keys():
                    if 'step' in key.lower() or 'iteration' in key.lower() or 'count' in key.lower():
                        try:
                            step = int(float(row[key]))
                        except:
                            pass
                    if 'return' in key.lower() and 'mean' in key.lower():
                        try:
                            mean_return = float(row[key])
                        except:
                            pass
                    elif 'return' in key.lower() and mean_return is None:
                        try:
                            mean_return = float(row[key])
                        except:
                            pass
                
                if step is not None and mean_return is not None:
                    steps.append(step)
                    returns_mean.append(mean_return)
                    
                    std_val = 0
                    max_val = mean_return
                    for key in row.keys():
                        if 'std' in key.lower():
                            try:
                                std_val = float(row[key])
                            except:
                                pass
                        if 'max' in key.lower():
                            try:
                                max_val = float(row[key])
                            except:
                                pass
                    returns_std.append(std_val)
                    returns_max.append(max_val)
        else:
            # 无表头，手动解析（格式：evaluation_index,episode_reward_mean）
            reader = csv.reader(f)
            for row_idx, row in enumerate(reader):
                if len(row) >= 2:
                    try:
                        # 第一列是评估索引或步数
                        step = int(float(row[0]))
                        # 第二列是episode_reward_mean
                        mean_return = float(row[1])
                        
                        steps.append(step)
                        returns_mean.append(mean_return)
                        returns_std.append(0)  # 如果没有std列，设为0
                        returns_max.append(mean_return)  # 如果没有max列，使用mean
                    except (ValueError, IndexError):
                        continue
    
    if not steps:
        raise ValueError(f"无法从 {eval_csv} 解析数据（解析到0条记录）")
    
    # 注意：BenchMARL的episode_reward_mean可能是整个episode的累积奖励
    # 但如果值太小（<10），可能需要检查是否是平均每步奖励
    # 这里先直接使用，后续可以根据需要调整
    
    return MetricData(
        steps=np.array(steps),
        mean=np.array(returns_mean),
        std=np.array(returns_std),
        max_val=np.array(returns_max),
    )


def plot_comparison(
    neat_data: NEATBestData,
    benchmarl_data: Dict[str, MetricData],
    task_name: str,
    output_path: str,
    title: Optional[str] = None,
    heuristic_reward: Optional[float] = None
):
    """绘制对比曲线图（只绘制均值对比）
    
    Args:
        neat_data: NEAT的最佳个体数据
        benchmarl_data: BenchMARL各算法的指标数据字典
        task_name: 任务名称
        output_path: 输出文件路径
        title: 图表标题
        heuristic_reward: 启发式策略的平均回报（可选）
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if title is None:
        title = f"Performance Comparison: {task_name}"
    
    # 计算需要的颜色数量（NEAT + BenchMARL算法 + Heuristic）
    n_colors = len(benchmarl_data) + 2 if heuristic_reward is not None else len(benchmarl_data) + 1
    colors = plt.cm.tab10(np.linspace(0, 1, n_colors))
    
    # 收集所有数据的最大步数，用于设置x轴范围
    max_steps = neat_data.steps[-1] if len(neat_data.steps) > 0 else 0
    for data in benchmarl_data.values():
        if len(data.steps) > 0:
            max_steps = max(max_steps, data.steps[-1])
    
    # NEAT最佳个体均值（带标准差阴影）
    ax.plot(neat_data.steps, neat_data.best_mean, 
             label='NEAT', color=colors[0], linewidth=2)
    if np.any(neat_data.best_std > 0):
        ax.fill_between(neat_data.steps, 
                         neat_data.best_mean - neat_data.best_std,
                         neat_data.best_mean + neat_data.best_std,
                         alpha=0.2, color=colors[0])
    
    # BenchMARL各算法
    for i, (algo_name, data) in enumerate(benchmarl_data.items(), 1):
        ax.plot(data.steps, data.mean, 
                 label=algo_name.upper(), color=colors[i], linewidth=2)
        if data.std is not None and np.any(data.std > 0):
            ax.fill_between(data.steps,
                             data.mean - data.std,
                             data.mean + data.std,
                             alpha=0.2, color=colors[i])
    
    # Heuristic水平线 - 从x=0开始绘制到最大步数
    if heuristic_reward is not None:
        heuristic_color = 'red'  # 使用红色来突出显示
        # 使用plot而不是axhline，这样可以控制起始点
        ax.plot([0, max_steps], [heuristic_reward, heuristic_reward], 
                color=heuristic_color, linestyle='-', linewidth=2, label='Heuristic')
    
    ax.set_xlabel('Steps', fontsize=12)
    ax.set_ylabel('Mean Return / Fitness', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"对比曲线图已保存: {output_path}")


def generate_comparison_table(
    neat_data: NEATBestData,
    benchmarl_data: Dict[str, MetricData],
    task_name: str,
    output_path: str,
    heuristic_reward: Optional[float] = None
):
    """生成对比统计表格"""
    results = []
    
    # NEAT统计
    neat_final_mean = neat_data.best_mean[-1] if len(neat_data.best_mean) > 0 else 0
    neat_final_std = neat_data.best_std[-1] if len(neat_data.best_std) > 0 else 0
    neat_best_ever_mean = np.max(neat_data.best_mean) if len(neat_data.best_mean) > 0 else 0
    
    results.append({
        "Algorithm": "NEAT",
        "Final Mean": f"{neat_final_mean:.4f}",
        "Final Std": f"{neat_final_std:.4f}",
        "Best Mean Ever": f"{neat_best_ever_mean:.4f}",
        "Total Steps": int(neat_data.steps[-1]) if len(neat_data.steps) > 0 else 0,
    })
    
    # BenchMARL各算法统计
    for algo_name, data in benchmarl_data.items():
        final_mean = data.mean[-1] if len(data.mean) > 0 else 0
        final_std = data.std[-1] if len(data.std) > 0 else 0
        best_ever_mean = np.max(data.mean) if len(data.mean) > 0 else 0
        
        results.append({
            "Algorithm": algo_name.upper(),
            "Final Mean": f"{final_mean:.4f}",
            "Final Std": f"{final_std:.4f}",
            "Best Mean Ever": f"{best_ever_mean:.4f}",
            "Total Steps": int(data.steps[-1]) if len(data.steps) > 0 else 0,
        })
    
    # Heuristic统计（如果有）
    if heuristic_reward is not None:
        results.append({
            "Algorithm": "Heuristic",
            "Final Mean": f"{heuristic_reward:.4f}",
            "Final Std": "N/A",
            "Best Mean Ever": f"{heuristic_reward:.4f}",
            "Total Steps": "N/A",
        })
    
    # 保存为JSON
    json_path = output_path.replace('.txt', '.json')
    with open(json_path, 'w') as f:
        json.dump({
            "task": task_name,
            "results": results,
        }, f, indent=2)
    
    # 保存为文本表格
    with open(output_path, 'w') as f:
        f.write(f"Performance Comparison: {task_name}\n")
        f.write("=" * 90 + "\n\n")
        
        headers = ["Algorithm", "Final Mean", "Final Std", "Best Mean Ever", "Total Steps"]
        col_widths = [12, 12, 12, 14, 12]
        
        header_line = " | ".join(h.center(w) for h, w in zip(headers, col_widths))
        f.write(header_line + "\n")
        f.write("-" * len(header_line) + "\n")
        
        for r in results:
            row_values = [
                r["Algorithm"].center(col_widths[0]),
                r["Final Mean"].center(col_widths[1]),
                r["Final Std"].center(col_widths[2]),
                r["Best Mean Ever"].center(col_widths[3]),
                str(r["Total Steps"]).center(col_widths[4]),
            ]
            f.write(" | ".join(row_values) + "\n")
        
        f.write("\n\nNotes:\n")
        f.write("- Final Mean: Mean return/fitness at the end of training\n")
        f.write("- Final Std: Standard deviation at the end of training\n")
        f.write("- Best Mean Ever: Best mean performance achieved during training\n")
        f.write("- Total Steps: Total environment interaction steps\n")
    
    print(f"统计表格已保存: {output_path}")
    print(f"JSON数据已保存: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="对比PyTorch-NEAT、BenchMARL和heuristic启发式策略的训练结果"
    )
    parser.add_argument("--neat_dir", type=str, default=None, help="NEAT结果目录路径")
    parser.add_argument("--task", type=str, default=None, help="任务名称（用于自动查找结果目录）")
    parser.add_argument("--algorithms", nargs="+", default=["mappo", "ippo"], help="要对比的算法列表")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--no-heuristic", action="store_true", help="不计算启发式策略的基准线")
    parser.add_argument("--force-heuristic", action="store_true", help="强制重新计算启发式策略（忽略已有结果）")
    parser.add_argument("--heuristic-envs", type=int, default=200, help="启发式策略并行环境数量")
    parser.add_argument("--device", type=str, default="cpu", help="启发式策略运行设备")
    
    args = parser.parse_args()
    # 处理参数中的连字符
    args.force_heuristic = getattr(args, 'force_heuristic', False)
    
    # 优先从任务配置文件加载配置（这是当前应该使用的配置）
    task_config = None
    if args.task:
        task_config = load_task_config_from_yaml(args.task)
        
        # 如果配置文件不存在，尝试从BenchMARL结果中读取配置（作为回退）
        if task_config is None and args.algorithms:
            bm_dir = find_benchmarl_results(args.task, args.algorithms[0], args.seed)
            if bm_dir:
                task_config = load_task_config_from_benchmarl(bm_dir)
    
    # 确定NEAT结果目录
    neat_dir = args.neat_dir
    if neat_dir is None and args.task:
        neat_dir = find_neat_results(args.task, task_config)
        if neat_dir is None:
            print(f"错误: 未找到任务 {args.task} 的NEAT结果")
            return
    
    if neat_dir is None:
        print("错误: 请指定 --neat_dir 或 --task 参数")
        return
    
    # 确定任务名称和配置名
    task_name = args.task
    if task_name is None:
        dir_name = os.path.basename(neat_dir)
        task_name = dir_name.split('_')[0]
    
    config_name = get_neat_config_name(neat_dir)
    
    print(f"\n{'='*60}")
    print(f"对比分析: {task_name}")
    print(f"配置: {config_name}")
    print(f"{'='*60}")
    print(f"NEAT结果目录: {neat_dir}")
    
    # 创建输出目录（在NEAT结果目录下）
    output_dir = os.path.join(neat_dir, "comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载NEAT配置
    pop_size, max_steps, trials = load_neat_config(neat_dir)
    
    # 加载NEAT最佳个体结果
    log_dir = os.path.join(neat_dir, "logs")
    neat_data = load_neat_best_results(log_dir)
    
    # 转换代数为步数: (generation + 1) * pop_size * max_steps
    # 代数从0开始，所以第N代完成后的步数是 (N + 1) * pop_size * max_steps
    neat_data.steps = (neat_data.steps + 1) * pop_size * max_steps
    
    print(f"NEAT数据点数: {len(neat_data.steps)}")
    print(f"NEAT步数范围: [{neat_data.steps[0]}, {neat_data.steps[-1]}]")
    
    # 加载BenchMARL结果（使用任务配置精确匹配）
    benchmarl_data = {}
    for algo in args.algorithms:
        bm_dir = find_benchmarl_results(task_name, algo, args.seed, task_config)
        if bm_dir:
            try:
                data = load_benchmarl_results(bm_dir)
                benchmarl_data[algo] = data
                print(f"BenchMARL {algo} 结果目录: {bm_dir}")
                print(f"BenchMARL {algo} 数据点数: {len(data.steps)}")
            except Exception as e:
                print(f"警告: 加载 {algo} 结果失败: {e}")
        else:
            print(f"警告: 未找到 {algo} 的结果目录")
    
    if not benchmarl_data:
        print("警告: 未找到BenchMARL结果，将只显示NEAT数据")
    
    # 获取启发式策略基准线（优先使用已有结果）
    heuristic_reward = None
    if not args.no_heuristic:
        print(f"\n获取 {task_name} 的启发式策略基准...")
        # 加载任务环境参数
        scenario_name, task_max_steps, env_kwargs = load_task_env_kwargs(neat_dir)
        try:
            heuristic_reward = get_or_compute_heuristic_reward(
                scenario_name=scenario_name,
                n_steps=task_max_steps,
                n_envs=args.heuristic_envs,
                env_kwargs=env_kwargs,
                device=args.device,
                force_compute=args.force_heuristic,
            )
            if heuristic_reward is not None:
                print(f"启发式策略平均回报: {heuristic_reward:.4f}")
            else:
                print(f"警告: {task_name} 没有对应的启发式策略")
        except Exception as e:
            print(f"警告: 获取启发式策略失败: {e}")
    
    # 生成对比图
    plot_path = os.path.join(output_dir, "comparison.png")
    plot_comparison(neat_data, benchmarl_data, task_name, plot_path, 
                   title=f"Performance Comparison: {config_name}",
                   heuristic_reward=heuristic_reward)
    
    # 生成统计表格
    table_path = os.path.join(output_dir, "statistics.txt")
    generate_comparison_table(neat_data, benchmarl_data, task_name, table_path,
                             heuristic_reward=heuristic_reward)
    
    print(f"\n{'='*60}")
    print("对比分析完成!")
    print(f"结果保存在: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
