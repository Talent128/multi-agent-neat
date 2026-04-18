"""
启发式策略基准评估脚本

该脚本用于评估VMAS场景的手工启发式策略性能，并将结果保存到benchmark_results目录。
支持渲染和视频保存。所有任务参数通过 --env_kwargs 传递。

使用方式:
    # 评估flocking场景
    python benchmark/run_heuristic_baseline.py --task flocking --n_steps 200 --env_kwargs '{"n_agents": 5, "n_obstacles": 0, "collision_reward": -0.1, "static_at_origin": true}' --render --save_video --device cuda
    
    # 评估transport场景（带环境参数）
    python benchmark/run_heuristic_baseline.py --task transport --n_steps 400 --env_kwargs '{"n_agents": 5, "n_packages": 1,"package_width": 0.15, "package_length": 0.15, "package_mass": 15}' --render --save_video --device cuda
    
    # 评估navigation场景
    python benchmark/run_heuristic_baseline.py --task navigation --n_steps 200 --env_kwargs '{"n_agents": 3, "collisions": true, "agents_with_same_goal": 1, "split_goals": false, "shared_rew": true, "observe_all_goals": false, "lidar_range": 0.35, "agent_radius": 0.1}' --render --save_video --device cuda

输出目录命名格式:
    {task}_heuristic_{n_steps}_{env_kwargs参数...}
    例如: transport_heuristic_200_4_1_0.15_0.15_10
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

# 修复scipy的C++ ABI版本问题：优先使用conda环境的libstdc++
# 这解决了 "CXXABI_1.3.15 not found" 错误
# 必须在导入任何可能依赖scipy的模块之前设置
if 'CONDA_PREFIX' in os.environ:
    conda_lib_path = os.path.join(os.environ['CONDA_PREFIX'], 'lib')
    if os.path.exists(conda_lib_path):
        # 设置LD_LIBRARY_PATH环境变量
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        if conda_lib_path not in current_ld_path:
            new_ld_path = f"{conda_lib_path}:{current_ld_path}" if current_ld_path else conda_lib_path
            os.environ['LD_LIBRARY_PATH'] = new_ld_path
            # 同时使用putenv确保子进程也能看到
            os.putenv('LD_LIBRARY_PATH', new_ld_path)
        
        # 使用ctypes预加载conda环境的libstdc++，确保优先使用
        try:
            import ctypes
            libstdcxx_path = os.path.join(conda_lib_path, 'libstdc++.so.6')
            if os.path.exists(libstdcxx_path):
                ctypes.CDLL(libstdcxx_path, mode=ctypes.RTLD_GLOBAL)
        except Exception:
            # 如果预加载失败，继续执行（LD_LIBRARY_PATH应该已经设置）
            pass

import torch
import numpy as np

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from vmas import make_env
from vmas.simulator.utils import save_video

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

# 默认环境参数（空字典，所有参数通过env_kwargs传递）
DEFAULT_ENV_KWARGS = {
    'transport': {},
    'flocking': {},
    'navigation': {},
    'give_way': {},
    'joint_passage': {},
    'balance': {},
    'discovery': {},
}


@dataclass
class HeuristicEvalResult:
    """启发式评估结果"""
    scenario: str
    total_reward: float  # 累计总回报
    mean_reward_per_step: float  # 每步平均回报
    n_steps: int
    n_envs: int
    env_kwargs: Dict[str, Any]
    eval_time: float  # 评估耗时(秒)
    
    def to_dict(self):
        return asdict(self)


def get_heuristic_policy(scenario_name: str):
    """
    获取场景对应的启发式策略类
    
    Args:
        scenario_name: 场景名称
        
    Returns:
        HeuristicPolicy类实例
    """
    if scenario_name not in HEURISTIC_SCENARIOS:
        raise ValueError(f"场景 {scenario_name} 没有对应的启发式策略。"
                        f"支持的场景: {list(HEURISTIC_SCENARIOS.keys())}")
    
    module_path = HEURISTIC_SCENARIOS[scenario_name]
    module = __import__(module_path, fromlist=['HeuristicPolicy'])
    HeuristicPolicy = getattr(module, 'HeuristicPolicy')
    return HeuristicPolicy(continuous_action=True)


def run_heuristic_evaluation(
    scenario_name: str,
    n_steps: int = 200,
    n_envs: int = 200,
    env_kwargs: Optional[Dict] = None,
    device: str = "cpu",
    render: bool = False,
    save_render: bool = False,
    video_dir: Optional[str] = None,
) -> tuple:
    """
    运行启发式策略评估
    
    Args:
        scenario_name: 场景名称
        n_steps: 仿真步数
        n_envs: 并行环境数量
        env_kwargs: 环境参数
        device: 计算设备
        render: 是否渲染
        save_render: 是否保存视频
        video_dir: 视频保存目录
        
    Returns:
        (HeuristicEvalResult, frame_list) 元组
    """
    if env_kwargs is None:
        env_kwargs = DEFAULT_ENV_KWARGS.get(scenario_name, {})
    
    # 获取启发式策略
    policy = get_heuristic_policy(scenario_name)
    
    # 创建环境
    env = make_env(
        scenario=scenario_name,
        num_envs=n_envs,
        device=device,
        continuous_actions=True,
        wrapper=None,
        **env_kwargs,
    )
    
    # 准备渲染
    frame_list = []
    start_time = time.time()
    
    # 重置环境
    obs = env.reset()
    total_reward = 0.0
    step_rewards = []
    
    # 运行仿真
    for step in range(n_steps):
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
        mean_global_reward = global_reward.mean(dim=0).item()
        total_reward += mean_global_reward
        step_rewards.append(mean_global_reward)
        
        # 渲染
        if render:
            frame_list.append(
                env.render(
                    mode="rgb_array",
                    agent_index_focus=None,
                    visualize_when_rgb=True,
                )
            )
    
    eval_time = time.time() - start_time
    
    # 保存视频
    if render and save_render and video_dir:
        os.makedirs(video_dir, exist_ok=True)
        video_basename = f"{scenario_name}_heuristic"
        video_path = os.path.join(video_dir, f"{video_basename}.mp4")
        
        # save_video会在当前目录保存，需要切换到video_dir目录
        old_cwd = os.getcwd()
        try:
            os.chdir(video_dir)
            fps = int(1 / env.scenario.world.dt)
            save_video(video_basename, frame_list, fps)
            print(f"视频已保存: {video_path}")
        finally:
            os.chdir(old_cwd)
    
    # 创建结果
    result = HeuristicEvalResult(
        scenario=scenario_name,
        total_reward=total_reward,
        mean_reward_per_step=total_reward / n_steps,
        n_steps=n_steps,
        n_envs=n_envs,
        env_kwargs=env_kwargs,
        eval_time=eval_time,
    )
    
    return result, frame_list, step_rewards


def generate_results_dir_name(scenario_name: str, n_steps: int, env_kwargs: Dict) -> str:
    """生成结果目录名称
    
    格式: {scenario}_heuristic_{n_steps}_{所有env_kwargs参数}
    """
    parts = [scenario_name, 'heuristic', str(n_steps)]
    
    # 添加所有env_kwargs参数（按key排序保证一致性）
    for key in sorted(env_kwargs.keys()):
        value = env_kwargs[key]
        # 处理不同类型的值
        if isinstance(value, bool):
            parts.append(str(value))
        elif isinstance(value, float):
            # 移除不必要的小数位
            parts.append(str(value).rstrip('0').rstrip('.') if '.' in str(value) else str(value))
        else:
            parts.append(str(value))
    
    return '_'.join(parts)


def save_heuristic_results(
    result: HeuristicEvalResult,
    step_rewards: list,
    output_dir: str,
):
    """
    保存启发式评估结果
    
    Args:
        result: 评估结果
        step_rewards: 每步奖励列表
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存摘要结果
    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"摘要结果已保存: {summary_file}")
    
    # 保存详细步骤奖励（用于绘图）
    rewards_file = os.path.join(output_dir, 'step_rewards.json')
    cumulative_rewards = np.cumsum(step_rewards).tolist()
    with open(rewards_file, 'w') as f:
        json.dump({
            'step_rewards': step_rewards,
            'cumulative_rewards': cumulative_rewards,
            'total_reward': result.total_reward,
            'n_steps': result.n_steps,
        }, f, indent=2)
    print(f"步骤奖励已保存: {rewards_file}")


def load_heuristic_results(results_dir: str) -> Optional[Dict]:
    """
    加载已有的启发式评估结果
    
    Args:
        results_dir: 结果目录
        
    Returns:
        结果字典，如果不存在返回None
    """
    summary_file = os.path.join(results_dir, 'summary.json')
    if not os.path.exists(summary_file):
        return None
    
    with open(summary_file, 'r') as f:
        return json.load(f)


def find_heuristic_results(scenario_name: str, n_steps: int = None, env_kwargs: Dict = None) -> Optional[str]:
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
    
    # 如果提供了完整参数，尝试精确匹配
    if n_steps is not None and env_kwargs is not None:
        dir_name = generate_results_dir_name(scenario_name, n_steps, env_kwargs)
        results_dir = results_base / dir_name
        if results_dir.exists() and (results_dir / 'summary.json').exists():
            return str(results_dir)
    
    # 尝试模糊匹配（返回最新的匹配结果）
    pattern = f"{scenario_name}_heuristic_*"
    matches = list(results_base.glob(pattern))
    if matches:
        # 返回最新的
        return str(sorted(matches, key=lambda x: x.stat().st_mtime, reverse=True)[0])
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description='启发式策略基准评估',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 评估transport场景
    python benchmark/run_heuristic_baseline.py --task transport --env_kwargs '{"n_agents": 4, "n_packages": 1, "package_mass": 10}'
    
    # 评估flocking场景
    python benchmark/run_heuristic_baseline.py --task flocking --env_kwargs '{"n_agents": 5, "n_obstacles": 3}'
    
    # 评估并保存视频
    python benchmark/run_heuristic_baseline.py --task transport --env_kwargs '{"n_agents": 4}' --render --save_video
    
    # 使用GPU
    python benchmark/run_heuristic_baseline.py --task transport --env_kwargs '{"n_agents": 4}' --device cuda
    
    # 强制重新评估
    python benchmark/run_heuristic_baseline.py --task transport --env_kwargs '{"n_agents": 4}' --force
        """
    )
    
    parser.add_argument('--task', type=str, required=True,
                       help=f'场景名称，支持: {list(HEURISTIC_SCENARIOS.keys())}')
    parser.add_argument('--n_steps', type=int, default=200, help='仿真步数')
    parser.add_argument('--n_envs', type=int, default=200, help='并行环境数量')
    parser.add_argument('--device', type=str, default='cpu', help='计算设备')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    # 环境参数（通过JSON传递所有参数）
    parser.add_argument('--env_kwargs', type=str, default='{}', 
                       help='JSON格式的环境参数，例如: \'{"n_agents": 4, "n_packages": 1}\'')
    
    # 渲染选项
    parser.add_argument('--render', action='store_true', help='启用渲染')
    parser.add_argument('--save_video', action='store_true', help='保存视频')
    
    # 输出选项
    parser.add_argument('--force', action='store_true', help='强制重新评估（忽略已有结果）')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 解析环境参数（从JSON字符串）
    try:
        env_kwargs = json.loads(args.env_kwargs)
    except json.JSONDecodeError as e:
        print(f"错误: 无法解析 env_kwargs: {e}")
        return
    
    # 生成输出目录（使用步数和所有env_kwargs参数）
    results_base = project_root / "benchmark_results"
    dir_name = generate_results_dir_name(args.task, args.n_steps, env_kwargs)
    output_dir = results_base / dir_name
    video_dir = output_dir / "videos" if args.save_video else None
    
    print(f"\n{'='*60}")
    print(f"启发式策略评估: {args.task}")
    print(f"{'='*60}")
    print(f"环境参数: {env_kwargs}")
    print(f"仿真步数: {args.n_steps}")
    print(f"并行环境: {args.n_envs}")
    print(f"设备: {args.device}")
    print(f"输出目录: {output_dir}")
    
    # 检查是否已有结果
    if not args.force and output_dir.exists():
        existing_result = load_heuristic_results(str(output_dir))
        if existing_result:
            print(f"\n已存在评估结果，使用 --force 强制重新评估")
            print(f"总回报: {existing_result['total_reward']:.4f}")
            print(f"{'='*60}")
            return
    
    # 运行评估
    print(f"\n开始评估...")
    try:
        result, frame_list, step_rewards = run_heuristic_evaluation(
            scenario_name=args.task,
            n_steps=args.n_steps,
            n_envs=args.n_envs,
            env_kwargs=env_kwargs,
            device=args.device,
            render=args.render,
            save_render=args.save_video,
            video_dir=str(video_dir) if video_dir else None,
        )
    except Exception as e:
        print(f"评估失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 保存结果
    save_heuristic_results(result, step_rewards, str(output_dir))
    
    print(f"\n{'='*60}")
    print("评估完成!")
    print(f"{'='*60}")
    print(f"场景: {args.task}")
    print(f"总回报: {result.total_reward:.4f}")
    print(f"每步平均回报: {result.mean_reward_per_step:.4f}")
    print(f"评估耗时: {result.eval_time:.2f}秒")
    print(f"结果保存: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

