import random
import numpy as np
import torch
import tempfile
import os

def seed_everything(seed: int):
    r"""Sets the seed for generating random numbers in :pytorch:`PyTorch`,
    :obj:`numpy` and :python:`Python`.
    调用重置随机状态：即使中间有其他操作改变了随机状态，也能在关键阶段恢复
    Args:
        seed (int): The desired seed.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def generate_results_dir_name(scenario_name: str, algorithm_name: str, task_config) -> str:
    """根据任务配置参数生成结果目录名
    
    格式: results/task_name_algorithm_name_param1_param2_...
    例如: results/flocking_recurrent_200_4_3_-0.1_True
    
    Args:
        scenario_name (str): 场景/任务名称，如 'flocking'
        algorithm_name (str): 算法名称，如 'recurrent'
        task_config: 任务配置对象（dataclass 或普通对象）
        
    Returns:
        str: 结果目录路径
    """
    from dataclasses import fields, is_dataclass
    
    # 获取任务配置参数值（按字段顺序）
    task_params = []
    if is_dataclass(task_config):
        for field in fields(task_config):
            value = getattr(task_config, field.name)
            task_params.append(str(value))
    else:
        # 如果不是dataclass，尝试按排序键顺序
        for key in sorted(vars(task_config).keys()):
            value = getattr(task_config, key)
            task_params.append(str(value))
    
    # 构建目录名: task_algorithm_params
    params_str = "_".join(task_params)
    dir_name = f"{scenario_name}_{algorithm_name}_{params_str}"
    
    return f"results/{dir_name}"


def load_neat_config_with_substitution(cfg_path, num_inputs, num_outputs, output_path=None):
    """加载NEAT配置文件并替换占位符
    
    Args:
        cfg_path (str): NEAT配置文件路径(.cfg)
        num_inputs (int): 输入神经元数量
        num_outputs (int): 输出神经元数量
        output_path (str, optional): 输出文件路径，如果为None则创建临时文件
        
    Returns:
        str: 配置文件路径
    """
    # 读取配置文件
    with open(cfg_path, 'r') as f:
        content = f.read()
    
    # 替换占位符
    content = content.replace('{num_inputs}', str(num_inputs))
    content = content.replace('{num_outputs}', str(num_outputs))
    
    # 保存到指定路径或临时文件
    if output_path is not None:
        with open(output_path, 'w') as f:
            f.write(content)
        return output_path
    else:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as f:
            f.write(content)
            temp_path = f.name
        return temp_path
