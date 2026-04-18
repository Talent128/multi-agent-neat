"""
种群进化过程绘图脚本

该脚本使用 log.json 记录的数据绘制进化训练中种群进化的过程。
横坐标是代数，纵坐标是适应度（带标准差阴影）。

使用方式:
    python benchmark/plot_evolution.py --results_dir results/transport_recurrent_...
    python benchmark/plot_evolution.py --task transport --seed 42  # 自动查找
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 从utils模块导入公共函数
from benchmark.utils import (
    load_task_config_from_yaml,
    find_neat_results,
    get_config_name
)


@dataclass
class PopulationData:
    """种群数据类"""
    generations: np.ndarray    # 代数
    fitness_avg: np.ndarray    # 种群适应度均值
    fitness_std: np.ndarray    # 种群适应度标准差
    fitness_best: np.ndarray   # 最佳个体适应度


def load_population_log(log_file: str) -> PopulationData:
    """加载种群日志数据
    
    Args:
        log_file: log.json文件路径
        
    Returns:
        PopulationData对象
    """
    generations = []
    fitness_avg = []
    fitness_std = []
    fitness_best = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                generations.append(data['generation'])
                fitness_avg.append(data['fitness_avg'])
                fitness_std.append(data['fitness_std'])
                fitness_best.append(data['fitness_best'])
    
    return PopulationData(
        generations=np.array(generations),
        fitness_avg=np.array(fitness_avg),
        fitness_std=np.array(fitness_std),
        fitness_best=np.array(fitness_best),
    )


def plot_population_evolution(
    data: PopulationData,
    output_path: str,
    title: Optional[str] = None
):
    """绘制种群进化过程
    
    绘制两条曲线：
    1. 种群平均适应度（带标准差阴影）
    2. 每代最佳个体适应度
    
    Args:
        data: 种群数据
        output_path: 输出文件路径
        title: 图表标题
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    if title is None:
        title = "Population Evolution"
    
    # 种群平均适应度（带标准差阴影）
    ax.plot(data.generations, data.fitness_avg, 
            label='Population Mean', color='#2196F3', linewidth=2)
    ax.fill_between(data.generations, 
                    data.fitness_avg - data.fitness_std,
                    data.fitness_avg + data.fitness_std,
                    alpha=0.3, color='#2196F3', label='Population Std')
    
    # 每代最佳个体适应度
    ax.plot(data.generations, data.fitness_best, 
            label='Best Individual', color='#FF5722', linewidth=2, linestyle='--')
    
    ax.set_xlabel('Generation', fontsize=14)
    ax.set_ylabel('Fitness', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 设置x轴范围和刻度，确保显示最终代数值
    max_gen = int(data.generations[-1])
    # 代数加一
    total_generations = max_gen + 1
    
    # 计算刻度间隔：总代数除以10
    step = total_generations / 10
    step = int(round(step))
    
    # 生成刻度点（基于总代数）
    ticks = list(range(0, total_generations + step, step))
    # 确保最后一个刻度是总代数
    if ticks[-1] > total_generations:
        ticks[-1] = total_generations
    elif ticks[-1] < total_generations:
        ticks.append(total_generations)
    
    # 设置x轴范围，基于总代数并稍微扩展避免顶格
    x_max = total_generations + step * 0.2
    ax.set_xlim(left=-step*0.2, right=x_max)
    
    ax.set_xticks(ticks)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"种群进化图已保存: {output_path}")


def generate_evolution_summary(
    data: PopulationData,
    output_path: str,
    config_name: str
):
    """生成进化过程摘要
    
    Args:
        data: 种群数据
        output_path: 输出文件路径
        config_name: 配置名称
    """
    summary = {
        "config": config_name,
        "total_generations": int(data.generations[-1]) + 1,
        "initial": {
            "fitness_avg": float(data.fitness_avg[0]),
            "fitness_std": float(data.fitness_std[0]),
            "fitness_best": float(data.fitness_best[0]),
        },
        "final": {
            "fitness_avg": float(data.fitness_avg[-1]),
            "fitness_std": float(data.fitness_std[-1]),
            "fitness_best": float(data.fitness_best[-1]),
        },
        "best_ever": {
            "fitness": float(np.max(data.fitness_best)),
            "generation": int(data.generations[np.argmax(data.fitness_best)]),
        },
        "improvement": {
            "avg_improvement": float(data.fitness_avg[-1] - data.fitness_avg[0]),
            "best_improvement": float(data.fitness_best[-1] - data.fitness_best[0]),
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"进化摘要已保存: {output_path}")
    
    # 打印摘要
    print(f"\n{'='*50}")
    print(f"进化摘要: {config_name}")
    print(f"{'='*50}")
    print(f"总代数: {summary['total_generations']}")
    print(f"初始均值: {summary['initial']['fitness_avg']:.4f}")
    print(f"最终均值: {summary['final']['fitness_avg']:.4f}")
    print(f"均值提升: {summary['improvement']['avg_improvement']:.4f}")
    print(f"历史最佳: {summary['best_ever']['fitness']:.4f} (第{summary['best_ever']['generation']}代)")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="绘制种群进化过程"
    )
    parser.add_argument("--results_dir", type=str, default=None, help="NEAT结果目录路径")
    parser.add_argument("--task", type=str, default=None, help="任务名称（用于自动查找）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 确定结果目录
    results_dir = args.results_dir
    if results_dir is None and args.task:
        # 尝试从配置文件加载任务配置
        task_config = load_task_config_from_yaml(args.task)
        results_dir = find_neat_results(args.task, task_config)
        if results_dir is None:
            print(f"错误: 未找到任务 {args.task} 的结果")
            return
    
    if results_dir is None:
        print("错误: 请指定 --results_dir 或 --task 参数")
        return
    
    # 获取配置名
    config_name = get_config_name(results_dir)
    
    print(f"\n{'='*60}")
    print(f"种群进化分析")
    print(f"配置: {config_name}")
    print(f"{'='*60}")
    print(f"结果目录: {results_dir}")
    
    # 加载数据
    log_file = os.path.join(results_dir, "logs", "log.json")
    if not os.path.exists(log_file):
        print(f"错误: 日志文件不存在: {log_file}")
        return
    
    data = load_population_log(log_file)
    print(f"已加载 {len(data.generations)} 代数据")
    
    # 创建输出目录
    output_dir = os.path.join(results_dir, "evolution_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # 绘制种群进化过程
    plot_population_evolution(
        data, 
        os.path.join(output_dir, "population_evolution.png"),
        title=f"Population Evolution: {config_name}"
    )
    
    # 生成进化摘要
    generate_evolution_summary(
        data,
        os.path.join(output_dir, "evolution_summary.json"),
        config_name
    )
    
    print(f"\n结果保存在: {output_dir}")


if __name__ == "__main__":
    main()


