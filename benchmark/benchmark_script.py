"""
综合基准测试脚本

该脚本提供一站式的基准测试流程：
1. 运行 BenchMARL 强化学习基线
2. 生成种群进化过程图
3. 自动对比 NEAT 进化算法结果
4. 生成综合报告

使用方式:
    python benchmark/benchmark_script.py --task transport --algorithms mappo ippo --seed 42
    python benchmark/benchmark_script.py --task all --algorithms mappo ippo --seed 42
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# 从utils模块导入公共函数
from benchmark.utils import (
    load_task_config_from_yaml,
    find_neat_results
)


def run_benchmarl_baseline(task: str, algorithms: list, seed: int):
    """运行 BenchMARL 基线训练
    
    Note:
        设备配置使用BenchMARL默认配置（conf/experiment/base_experiment.yaml）
    """
    script_path = project_root / "benchmark" / "run_benchmarl_baseline.py"
    
    cmd = [
        sys.executable, str(script_path),
        "--task", task,
        "--algorithms", *algorithms,
        "--seed", str(seed),
    ]
    
    print(f"\n{'='*60}")
    print("运行 BenchMARL 基线训练")
    print(f"命令: {' '.join(cmd)}")
    print(f"📌 设备配置使用BenchMARL默认配置")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, cwd=str(project_root))
    return result.returncode == 0


def run_evolution_plot(task: str, seed: int):
    """运行种群进化过程绘图"""
    script_path = project_root / "benchmark" / "plot_evolution.py"
    
    cmd = [
        sys.executable, str(script_path),
        "--task", task,
        "--seed", str(seed)
    ]
    
    print(f"\n{'='*60}")
    print("绘制种群进化过程")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, cwd=str(project_root))
    return result.returncode == 0


def run_comparison(task: str, algorithms: list, seed: int):
    """运行结果对比分析"""
    script_path = project_root / "benchmark" / "compare_results.py"
    
    cmd = [
        sys.executable, str(script_path),
        "--task", task,
        "--algorithms", *algorithms,
        "--seed", str(seed)
    ]
    
    print(f"\n{'='*60}")
    print("运行结果对比分析")
    print(f"命令: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, cwd=str(project_root))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="综合基准测试脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
配置说明:
  - BenchMARL的设备、学习率等参数使用默认配置
  - 可在 conf/experiment/base_experiment.yaml 中调整
  - 只有与NEAT对齐的参数会被自动计算和覆盖

使用示例:
  python benchmark/benchmark_script.py --task transport --algorithms ippo mappo
  python benchmark/benchmark_script.py --task all --seed 42
        """
    )
    parser.add_argument(
        "--task",
        type=str,
        default="transport",
        help="任务名称，使用 'all' 运行所有任务"
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=["ippo"],
        help="要运行的强化学习算法 (默认: ippo)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="随机种子"
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="跳过BenchMARL训练"
    )
    parser.add_argument(
        "--skip_evolution_plot",
        action="store_true",
        help="跳过种群进化绘图"
    )
    parser.add_argument(
        "--skip_comparison",
        action="store_true",
        help="跳过对比分析"
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("PyTorch-NEAT vs BenchMARL 综合基准测试")
    print(f"任务: {args.task}")
    print(f"算法: {args.algorithms}")
    print(f"种子: {args.seed}")
    print(f"📌 设备配置使用BenchMARL默认配置")
    print(f"{'='*60}\n")
    
    # 确定任务列表
    if args.task == "all":
        from run_benchmarl_baseline import get_available_tasks
        tasks = get_available_tasks()
    else:
        tasks = [args.task]
    
    # 运行基线训练
    if not args.skip_training:
        for task in tasks:
            success = run_benchmarl_baseline(
                task, args.algorithms, args.seed
            )
            if not success:
                print(f"警告: {task} 的 BenchMARL 基线训练未完全成功")
    
    # 对每个任务运行后续分析
    for task in tasks:
        # 绘制种群进化过程
        if not args.skip_evolution_plot:
            # 尝试从配置文件加载任务配置
            task_config = load_task_config_from_yaml(task)
            neat_dir = find_neat_results(task, task_config)
            if neat_dir:
                run_evolution_plot(task, args.seed)
            else:
                print(f"警告: 未找到任务 {task} 的NEAT结果，跳过进化绘图")
        
        # 运行对比分析
        if not args.skip_comparison:
            run_comparison(task, args.algorithms, args.seed)
    
    # 打印结果位置
    print(f"\n{'='*60}")
    print("基准测试完成!")
    print(f"{'='*60}")
    print("\n结果保存位置:")
    print(f"  BenchMARL 结果: {project_root / 'benchmark_results'}")
    for task in tasks:
        neat_dir = find_neat_results(task)
        if neat_dir:
            print(f"  {task} 进化图: {neat_dir}/evolution_plots/")
            print(f"  {task} 对比结果: {neat_dir}/comparison/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

