# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

"""
激活函数模块：提供神经网络常用的激活函数。

除了具体激活函数本身，这个模块还提供：
1. 按节点名字解析激活函数
2. 对最后一维按节点分组应用不同激活函数
3. 按激活函数理论输出范围，将输出归一化到 [0, 1]

注意：大部分激活函数都进行了缩放（乘以常数），以增强信号强度。
"""

from collections import OrderedDict
from typing import Iterable, Sequence, Tuple

import torch
import torch.nn.functional as F


def _finite_clamp(
    x: torch.Tensor,
    min_value: float | None = None,
    max_value: float | None = None,
    *,
    nan_value: float = 0.0,
) -> torch.Tensor:
    """将张量收敛到有限区间，避免递归网络传播 inf/nan。"""
    posinf_value = max_value if max_value is not None else 1e6
    neginf_value = min_value if min_value is not None else -1e6
    x = torch.nan_to_num(x, nan=nan_value, posinf=posinf_value, neginf=neginf_value)
    if min_value is None and max_value is None:
        return x
    if min_value is None:
        return torch.clamp_max(x, max_value)
    if max_value is None:
        return torch.clamp_min(x, min_value)
    return torch.clamp(x, min=min_value, max=max_value)


def sigmoid_activation(x):
    """
    Sigmoid激活函数（放大版）
    
    公式: σ(5x) = 1 / (1 + e^(-5x))
    输出范围: (0, 1)
    
    特点：
    - 输出有界，适合作为概率输出
    - 乘以5使函数更陡峭，增强信号
    - 易饱和（梯度消失），但在NEAT中不影响（无反向传播）
    
    Args:
        x: 输入张量
        
    Returns:
        torch.Tensor: 激活后的张量，值域(0, 1)
    """
    # 对齐 neat-python: 先乘 5，再截断到 [-60, 60]。
    x = _finite_clamp(5.0 * x, -60.0, 60.0)
    return torch.sigmoid(x)


def tanh_activation(x):
    """
    双曲正切激活函数（放大版）
    
    公式: tanh(2.5x)
    输出范围: (-1, 1)
    
    特点：
    - 输出以0为中心，比sigmoid更好
    - 乘以2.5使函数更陡峭
    - 适合大多数任务
    
    Args:
        x: 输入张量
        
    Returns:
        torch.Tensor: 激活后的张量，值域(-1, 1)
    """
    # 对齐 neat-python: 先乘 2.5，再截断到 [-60, 60]。
    x = _finite_clamp(2.5 * x, -60.0, 60.0)
    return torch.tanh(x)


def abs_activation(x):
    """
    绝对值激活函数
    
    公式: |x|
    输出范围: [0, ∞)
    
    特点：
    - 非单调函数，可以产生有趣的行为
    - 在x=0处不可微（但NEAT不需要梯度）
    - 适合某些特殊任务
    
    Args:
        x: 输入张量
        
    Returns:
        torch.Tensor: 激活后的张量，所有值非负
    """
    x = _finite_clamp(x, -1e6, 1e6)
    return torch.abs(x)


def gauss_activation(x):
    """
    高斯（径向基）激活函数
    
    公式: exp(-5x²)
    输出范围: (0, 1]
    
    特点：
    - 在x=0处最大值为1
    - 对称的钟形曲线
    - 可以检测输入是否接近0
    - 适合模式识别任务
    
    Args:
        x: 输入张量
        
    Returns:
        torch.Tensor: 激活后的张量，在x=0时最大
    """
    # 对齐 neat-python: 输入先截断到 [-3.4, 3.4]。
    x = _finite_clamp(x, -3.4, 3.4)
    return torch.exp(-5.0 * x.square())


def identity_activation(x):
    """
    恒等激活函数（线性）
    
    公式: f(x) = x
    输出范围: (-∞, ∞)
    
    特点：
    - 不改变输入
    - 保持线性特性
    - 常用于输出层或CPPN
    
    Args:
        x: 输入张量
        
    Returns:
        torch.Tensor: 输入本身
    """
    return _finite_clamp(x, -1e6, 1e6)


def sin_activation(x):
    """
    正弦激活函数
    
    公式: sin(x)
    输出范围: [-1, 1]
    
    特点：
    - 周期性函数，可以产生振荡行为
    - 在CPPN中很有用（生成重复模式）
    - 可以帮助探索更复杂的行为
    
    Args:
        x: 输入张量
        
    Returns:
        torch.Tensor: 激活后的张量，周期性振荡
    """
    # 对齐 neat-python: 先乘 5，再截断到 [-60, 60]。
    x = _finite_clamp(5.0 * x, -60.0, 60.0)
    return torch.sin(x)


def relu_activation(x):
    """
    ReLU（修正线性单元）激活函数
    
    公式: max(0, x)
    输出范围: [0, ∞)
    
    特点：
    - 深度学习中最常用的激活函数
    - 计算简单，不会饱和（正值部分）
    - 可能导致"神经元死亡"（负值全为0）
    
    Args:
        x: 输入张量
        
    Returns:
        torch.Tensor: 激活后的张量，负值变为0
    """
    x = _finite_clamp(x, -1e6, 1e6)
    return F.relu(x)


# 字符串到激活函数的映射字典
# 用于从配置文件中根据名称选择激活函数
str_to_activation = {
    'sigmoid': sigmoid_activation,
    'tanh': tanh_activation,
    'abs': abs_activation,
    'gauss': gauss_activation,
    'identity': identity_activation,
    'sin': sin_activation,
    'relu': relu_activation,
}


# 动作映射时需要知道激活函数的理论输出范围。
# 对于无界激活，无法直接线性映射到动作空间，这里先做单调压缩：
# - identity: 先用 tanh 压到 [-1, 1]
# - relu/abs: 先用 1 - exp(-x) 压到 [0, 1)
activation_output_mode = {
    'sigmoid': 'unit',
    'gauss': 'unit',
    'tanh': 'signed',
    'sin': 'signed',
    'identity': 'unbounded',
    'relu': 'nonnegative',
    'abs': 'nonnegative',
}


def resolve_activation(name: str):
    """根据名称解析激活函数，不支持时抛出明确异常。"""
    activation = str_to_activation.get(name)
    if activation is None:
        raise ValueError(
            f"Unsupported activation '{name}'. Supported values: {sorted(str_to_activation)}"
        )
    return activation


def build_activation_groups(
    activation_names: Sequence[str], device=None
) -> Tuple[Tuple[str, torch.Tensor], ...]:
    """按激活函数名分组，便于对最后一维成组应用不同激活函数。"""
    grouped = OrderedDict()
    for idx, name in enumerate(activation_names):
        resolve_activation(name)
        grouped.setdefault(name, []).append(idx)

    groups = []
    for name, indices in grouped.items():
        groups.append(
            (
                name,
                torch.tensor(indices, dtype=torch.long, device=device),
            )
        )
    return tuple(groups)


def build_batched_activation_masks(
    activation_names_per_sample: Sequence[Sequence[str]], device=None
) -> Tuple[Tuple[str, torch.Tensor], ...]:
    """为形如 (sample, node) 的激活名矩阵构建按激活类型聚合的布尔 mask。"""
    n_samples = len(activation_names_per_sample)
    width = max((len(names) for names in activation_names_per_sample), default=0)

    grouped = OrderedDict()
    for sample_idx, activation_names in enumerate(activation_names_per_sample):
        for node_idx, name in enumerate(activation_names):
            resolve_activation(name)
            grouped.setdefault(name, [[], []])
            grouped[name][0].append(sample_idx)
            grouped[name][1].append(node_idx)

    masks = []
    for name, (sample_indices, node_indices) in grouped.items():
        mask = torch.zeros((n_samples, width), dtype=torch.bool, device=device)
        if sample_indices:
            mask[
                torch.tensor(sample_indices, dtype=torch.long, device=device),
                torch.tensor(node_indices, dtype=torch.long, device=device),
            ] = True
        masks.append((name, mask))
    return tuple(masks)


def apply_activation_groups(
    values: torch.Tensor,
    activation_groups: Iterable[Tuple[str, torch.Tensor]],
) -> torch.Tensor:
    """对张量最后一维的不同节点，按各自激活函数分别处理。"""
    if values.shape[-1] == 0:
        return values

    activated = torch.zeros_like(values)
    for name, indices in activation_groups:
        if indices.numel() == 0:
            continue
        fn = resolve_activation(name)
        indices = indices.to(device=values.device)
        activated[..., indices] = fn(values.index_select(-1, indices))
    return activated


def apply_activation_masks(
    values: torch.Tensor,
    activation_masks: Iterable[Tuple[str, torch.Tensor]],
) -> torch.Tensor:
    """对 (sample, batch, node) 张量按预构建 mask 向量化应用不同激活函数。"""
    if values.shape[-1] == 0:
        return values

    activated = torch.zeros_like(values)
    for name, mask in activation_masks:
        if mask.numel() == 0:
            continue
        fn = resolve_activation(name)
        mask = mask.to(device=values.device).unsqueeze(1)
        activated = activated + fn(values) * mask.to(dtype=values.dtype)
    return activated


def apply_named_activations(values: torch.Tensor, activation_names: Sequence[str]) -> torch.Tensor:
    """便捷封装：按节点激活函数名字应用到最后一维。"""
    return apply_activation_groups(
        values, build_activation_groups(activation_names, device=values.device)
    )


def _normalize_values_for_activation(name: str, values: torch.Tensor) -> torch.Tensor:
    """将某个激活函数的输出压到 [0, 1]，便于后续统一映射到动作空间。"""
    mode = activation_output_mode.get(name)
    if mode is None:
        raise ValueError(
            f"Activation '{name}' does not have a defined output normalization rule."
        )

    values = _finite_clamp(values, -1e6, 1e6)

    if mode == 'unit':
        return values.clamp(0.0, 1.0)
    if mode == 'signed':
        return (values.clamp(-1.0, 1.0) + 1.0) * 0.5
    if mode == 'nonnegative':
        positive = values.clamp_min(0.0)
        return 1.0 - torch.exp(-positive)
    if mode == 'unbounded':
        return (torch.tanh(values) + 1.0) * 0.5

    raise ValueError(f"Unknown activation output mode '{mode}' for activation '{name}'.")


def normalize_output_groups(
    values: torch.Tensor,
    activation_groups: Iterable[Tuple[str, torch.Tensor]],
) -> torch.Tensor:
    """按输出节点激活函数，把张量最后一维归一化到 [0, 1]。"""
    if values.shape[-1] == 0:
        return values

    normalized = torch.zeros_like(values)
    for name, indices in activation_groups:
        if indices.numel() == 0:
            continue
        indices = indices.to(device=values.device)
        normalized[..., indices] = _normalize_values_for_activation(
            name, values.index_select(-1, indices)
        )
    return normalized


def normalize_output_masks(
    values: torch.Tensor,
    activation_masks: Iterable[Tuple[str, torch.Tensor]],
) -> torch.Tensor:
    """对 (sample, batch, node) 张量按预构建 mask 向量化归一化到 [0, 1]。"""
    if values.shape[-1] == 0:
        return values

    normalized = torch.zeros_like(values)
    for name, mask in activation_masks:
        if mask.numel() == 0:
            continue
        mask = mask.to(device=values.device).unsqueeze(1)
        normalized = normalized + _normalize_values_for_activation(
            name, values
        ) * mask.to(dtype=values.dtype)
    return normalized
