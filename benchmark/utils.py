"""
Benchmark工具函数模块

提供benchmark目录下多个脚本共享的公共函数
"""
import os
import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

# 添加项目路径
project_root = Path(__file__).parent.parent


def load_task_config_from_yaml(task_name: str) -> Optional[dict]:
    """从任务配置文件加载配置
    
    Args:
        task_name: 任务名称，如 "flocking"
        
    Returns:
        任务配置字典，如果文件不存在返回None
    """
    config_path = project_root / "conf" / "task" / "vmas" / f"{task_name}.yaml"
    if not config_path.exists():
        return None
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 移除defaults字段
    config.pop('defaults', None)
    return config


def load_task_config_from_benchmarl(benchmarl_dir: str) -> Optional[dict]:
    """从BenchMARL结果目录的config_info.json加载任务配置
    
    Args:
        benchmarl_dir: BenchMARL结果目录
        
    Returns:
        任务配置字典，如果文件不存在返回None
    """
    config_info_path = os.path.join(benchmarl_dir, 'config_info.json')
    if not os.path.exists(config_info_path):
        return None
    
    with open(config_info_path, 'r') as f:
        config_info = json.load(f)
    
    return config_info.get('neat_task_config')


def find_neat_results(
    task_name: str, 
    task_config: Optional[dict] = None, 
    algorithm_name: str = "recurrent"
) -> Optional[str]:
    """查找NEAT结果目录
    
    Args:
        task_name: 任务名称，如 "flocking"
        task_config: 任务配置字典（可选），如果提供则精确匹配
        algorithm_name: 算法名称，默认为 "recurrent"
        
    Returns:
        结果目录路径，如果未找到返回None
    """
    results_dir = project_root / "results"
    
    # 如果提供了任务配置，尝试精确匹配
    if task_config is not None:
        from experiment.runtime import generate_results_dir_name
        
        # 创建任务配置对象（可以是普通对象或dataclass）
        # 先尝试从environments.vmas模块导入TaskConfig
        try:
            module_name = f"environments.vmas.{task_name}"
            module = __import__(module_name, fromlist=['TaskConfig'])
            TaskConfig = getattr(module, 'TaskConfig', None)
            
            if TaskConfig is not None:
                # 使用dataclass
                task_config_obj = TaskConfig(**task_config)
            else:
                # 使用普通对象
                class TaskConfigObj:
                    pass
                task_config_obj = TaskConfigObj()
                for k, v in task_config.items():
                    setattr(task_config_obj, k, v)
        except (ImportError, AttributeError):
            # 如果导入失败，使用普通对象
            class TaskConfigObj:
                pass
            task_config_obj = TaskConfigObj()
            for k, v in task_config.items():
                setattr(task_config_obj, k, v)
        
        # 生成预期的目录名
        expected_dir = generate_results_dir_name(task_name, algorithm_name, task_config_obj)
        expected_dir_name = os.path.basename(expected_dir)
        expected_path = results_dir / expected_dir_name
        
        if expected_path.exists() and expected_path.is_dir():
            return str(expected_path)
    
    # 如果没有提供配置或精确匹配失败，尝试从配置文件加载
    if task_config is None:
        task_config = load_task_config_from_yaml(task_name)
        if task_config is not None:
            # 递归调用，使用加载的配置
            return find_neat_results(task_name, task_config, algorithm_name)
    
    # 回退到原有行为：查找所有匹配的目录，返回最新的
    pattern = f"{task_name}_*"
    matches = list(results_dir.glob(pattern))
    if matches:
        # 排除非目录文件，按修改时间排序，返回最新的
        dirs = [m for m in matches if m.is_dir()]
        if dirs:
            return str(sorted(dirs, key=lambda x: x.stat().st_mtime, reverse=True)[0])
    
    return None


def generate_benchmarl_folder_name(
    task_name: str,
    algorithm_name: str,
    task_config: Dict[str, Any],
    seed: int
) -> str:
    """生成BenchMARL结果目录名
    
    命名格式: {task}_{algorithm}_{key_params}_seed{seed}
    与run_benchmarl_baseline.py中的逻辑一致
    
    Args:
        task_name: 任务名称
        algorithm_name: 算法名称
        task_config: 任务配置字典
        seed: 随机种子
        
    Returns:
        目录名
    """
    # 提取关键配置参数
    key_params = []
    
    # 通用参数
    if 'max_steps' in task_config:
        key_params.append(str(task_config['max_steps']))
    if 'n_agents' in task_config:
        key_params.append(str(task_config['n_agents']))
    
    # 任务特定参数
    task_specific_keys = {
        'transport': ['n_packages', 'package_width', 'package_length', 'package_mass'],
        'navigation': ['collisions', 'agents_with_same_goal', 'split_goals', 'shared_rew', 
                       'observe_all_goals', 'lidar_range', 'agent_radius'],
        'dispersion': ['n_food', 'share_reward', 'food_radius', 'penalise_by_time'],
        'flocking': ['n_obstacles', 'collision_reward', 'static_at_origin'],
        'balance': ['random_package_pos_on_line', 'package_mass'],
        'give_way': ['mirror_passage', 'observe_rel_pos', 'done_on_completion', 'final_reward'],
    }
    
    specific_keys = task_specific_keys.get(task_name, [])
    for key in specific_keys:
        if key in task_config:
            val = task_config[key]
            # 格式化值
            if isinstance(val, bool):
                key_params.append(str(val))
            elif isinstance(val, float):
                # 处理浮点数，去除不必要的精度
                if val == int(val):
                    key_params.append(str(int(val)))
                else:
                    key_params.append(f"{val}".rstrip('0').rstrip('.'))
            else:
                key_params.append(str(val))
    
    # 构建文件夹名称
    params_str = "_".join(key_params) if key_params else ""
    if params_str:
        folder_name = f"{task_name}_{algorithm_name}_{params_str}_seed{seed}"
    else:
        folder_name = f"{task_name}_{algorithm_name}_seed{seed}"
    
    return folder_name


def find_benchmarl_results(
    task_name: str, 
    algorithm: str, 
    seed: int,
    task_config: Optional[dict] = None
) -> Optional[str]:
    """查找BenchMARL结果目录
    
    如果提供了task_config，则精确匹配；否则使用模糊匹配（回退行为）
    
    Args:
        task_name: 任务名称
        algorithm: 算法名称
        seed: 随机种子
        task_config: 任务配置字典（可选），如果提供则精确匹配
        
    Returns:
        结果目录路径，如果未找到返回None
    """
    results_dir = project_root / "benchmark_results"
    
    # 如果提供了任务配置，尝试精确匹配
    if task_config is not None:
        expected_dir_name = generate_benchmarl_folder_name(
            task_name, algorithm, task_config, seed
        )
        expected_path = results_dir / expected_dir_name
        
        if expected_path.exists() and expected_path.is_dir():
            return str(expected_path)
    
    # 回退到模糊匹配（原有行为）
    pattern = f"{task_name}_{algorithm}_*_seed{seed}"
    matches = list(results_dir.glob(pattern))
    if matches:
        return str(sorted(matches, key=lambda x: x.stat().st_mtime, reverse=True)[0])
    
    return None


def get_config_name(results_dir: str) -> str:
    """从结果目录名提取配置名称
    
    Args:
        results_dir: 结果目录路径
        
    Returns:
        配置名称（目录名）
    """
    dir_name = os.path.basename(results_dir)
    return dir_name
