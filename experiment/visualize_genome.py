"""
NEAT 基因组网络可视化工具

用于可视化 NEAT 训练出来的网络拓扑结构。

功能:
    1. 可视化单个网络拓扑结构（指定代数或默认最后一代）
    2. 以一张图展示整个训练过程中基因组网络的变化过程

使用方式:
    # 可视化最后一代的最佳基因组
    python visualize_genome.py transport_recurrent_200_4_1_0.15_0.15_10.0_seed42
    
    # 可视化指定代数的最佳基因组
    python visualize_genome.py transport_recurrent_200_4_1_0.15_0.15_10.0_seed42 --generation 50
    
    # 展示网络进化过程（多代对比）
    python visualize_genome.py transport_recurrent_200_4_1_0.15_0.15_10.0_seed42 --evolution --interval 25
"""

import os
import sys
import gzip
import pickle
import argparse
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib
matplotlib.use('Agg')

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_checkpoint(checkpoint_file: str):
    """
    从检查点文件加载种群信息
    
    Args:
        checkpoint_file: 检查点文件路径
        
    Returns:
        (generation, config, population, species_set) 元组
    """
    with gzip.open(checkpoint_file) as f:
        generation, config, population, species_set, rndstate = pickle.load(f)
    return generation, config, population, species_set


def get_best_genome_from_checkpoint(checkpoint_file: str):
    """
    从检查点中获取最佳基因组
    
    Args:
        checkpoint_file: 检查点文件路径
        
    Returns:
        (best_genome, generation, config) 元组
    """
    generation, config, population, species_set = load_checkpoint(checkpoint_file)
    
    # 找到适应度最高的基因组
    best_genome = None
    best_fitness = float('-inf')
    
    for genome_id, genome in population.items():
        if genome.fitness is not None and genome.fitness > best_fitness:
            best_fitness = genome.fitness
            best_genome = genome
    
    return best_genome, generation, config


def find_checkpoints(checkpoint_dir: str) -> List[str]:
    """
    查找所有检查点文件
    
    Args:
        checkpoint_dir: 检查点目录
        
    Returns:
        按代数排序的检查点文件路径列表
    """
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('neat-checkpoint-')]
    # 按代数排序
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
    return [os.path.join(checkpoint_dir, cp) for cp in checkpoints]


def get_genome_structure(genome, config) -> Dict:
    """
    分析基因组结构
    
    Args:
        genome: NEAT基因组
        config: NEAT配置
        
    Returns:
        包含节点和连接信息的字典
    """
    genome_config = config.genome_config
    
    # 分类节点
    input_keys = list(genome_config.input_keys)
    output_keys = list(genome_config.output_keys)
    hidden_keys = [k for k in genome.nodes.keys() if k not in output_keys]
    
    # 收集连接信息
    connections = []
    for conn_key, conn in genome.connections.items():
        connections.append({
            'from': conn_key[0],
            'to': conn_key[1],
            'weight': conn.weight,
            'enabled': conn.enabled,
        })
    
    # 收集节点信息
    nodes = {}
    for key in input_keys:
        nodes[key] = {'type': 'input', 'bias': 0.0, 'response': 1.0}
    for key, node in genome.nodes.items():
        node_type = 'output' if key in output_keys else 'hidden'
        nodes[key] = {
            'type': node_type,
            'bias': node.bias,
            'response': node.response,
            'activation': node.activation if hasattr(node, 'activation') else 'sigmoid',
        }
    
    return {
        'input_keys': input_keys,
        'output_keys': output_keys,
        'hidden_keys': hidden_keys,
        'nodes': nodes,
        'connections': connections,
        'n_inputs': len(input_keys),
        'n_hidden': len(hidden_keys),
        'n_outputs': len(output_keys),
        'n_enabled_conns': sum(1 for c in connections if c['enabled']),
        'n_total_conns': len(connections),
    }


def draw_genome_network(
    genome, 
    config, 
    ax: plt.Axes,
    title: str = "",
    show_weights: bool = False,
    compact_mode: bool = False,
):
    """
    在给定的matplotlib axes上绘制基因组网络拓扑（使用曲线连接）
    
    Args:
        genome: NEAT基因组
        config: NEAT配置
        ax: matplotlib Axes对象
        title: 图表标题
        show_weights: 是否显示连接权重
        compact_mode: 紧凑模式（用于进化展示多图）
    
    图例说明:
        - 节点颜色: 绿色=输入, 蓝色=隐藏, 粉色=输出
        - 实线曲线: 前向连接
        - 虚线曲线: 循环连接（recurrent）
    """
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.path import Path
    import matplotlib.patches as mpatches_local
    
    structure = get_genome_structure(genome, config)
    
    input_keys = structure['input_keys']
    output_keys = structure['output_keys']
    hidden_keys = structure['hidden_keys']
    connections = structure['connections']
    
    # 根据模式调整参数
    if compact_mode:
        node_radius = 0.028
        label_fontsize = 6
    else:
        node_radius = 0.038
        label_fontsize = 9
    
    # 计算节点位置（增加间距）
    positions = {}
    
    # 输入层（左侧）- 增加间距
    n_inputs = len(input_keys)
    input_spacing = min(0.09, 1.1 / max(n_inputs, 1))  # 增加间距
    input_start_y = 0.5 - (n_inputs - 1) * input_spacing / 2
    for i, key in enumerate(input_keys):
        y = input_start_y + i * input_spacing
        positions[key] = (0.1, y)
    
    # 输出层（右侧）- 增加间距
    n_outputs = len(output_keys)
    output_spacing = min(0.22, 0.6 / max(n_outputs, 1))  # 增加间距
    output_start_y = 0.5 - (n_outputs - 1) * output_spacing / 2
    for i, key in enumerate(output_keys):
        y = output_start_y + i * output_spacing
        positions[key] = (0.9, y)
    
    # 隐藏层（中间）- 增加间距
    if hidden_keys:
        n_hidden = len(hidden_keys)
        if n_hidden <= 4:
            n_cols = 1
        elif n_hidden <= 10:
            n_cols = 2
        else:
            n_cols = 3
        
        nodes_per_col = (n_hidden + n_cols - 1) // n_cols
        col_x_positions = np.linspace(0.38, 0.62, n_cols) if n_cols > 1 else [0.5]
        
        for i, key in enumerate(hidden_keys):
            col = i // nodes_per_col
            row = i % nodes_per_col
            actual_nodes_in_col = min(nodes_per_col, n_hidden - col * nodes_per_col)
            col_spacing = min(0.14, 0.75 / max(actual_nodes_in_col, 1))  # 增加间距
            col_start_y = 0.5 - (actual_nodes_in_col - 1) * col_spacing / 2
            x = col_x_positions[min(col, len(col_x_positions) - 1)]
            y = col_start_y + row * col_spacing
            positions[key] = (x, y)
    
    # 节点颜色
    colors = {
        'input': '#90EE90',    # 浅绿色
        'hidden': '#87CEEB',   # 天蓝色
        'output': '#FFB6C1',   # 浅粉色
    }
    
    # 统一的连接颜色
    conn_color = '#4A4A4A'  # 深灰色
    recurrent_color = '#888888'  # 浅灰色
    
    # 绘制所有连接（使用贝塞尔曲线）
    for conn in connections:
        if not conn['enabled']:
            continue
        
        from_key = conn['from']
        to_key = conn['to']
        weight = conn['weight']
        
        if from_key not in positions or to_key not in positions:
            continue
        
        x1, y1 = positions[from_key]
        x2, y2 = positions[to_key]
        
        # 判断是否为循环连接
        is_recurrent = (from_key in output_keys and to_key in hidden_keys) or \
                       (from_key in output_keys and to_key in output_keys and from_key != to_key) or \
                       (from_key == to_key)
        
        if not is_recurrent and from_key in hidden_keys and to_key in hidden_keys:
            if positions[from_key][0] >= positions[to_key][0]:
                is_recurrent = True
        
        # 线条样式
        if is_recurrent:
            linestyle = '--'
            color = recurrent_color
            alpha = 0.6
        else:
            linestyle = '-'
            color = conn_color
            alpha = 0.75
        
        # 线宽与权重绝对值成正比
        linewidth = max(0.6, min(2.0, abs(weight) * 0.5 + 0.4)) if not compact_mode else \
                    max(0.4, min(1.2, abs(weight) * 0.3 + 0.3))
        
        # 计算贝塞尔曲线控制点（使曲线绕过可能的节点）
        dx = x2 - x1
        dy = y2 - y1
        
        # 根据连接方向调整曲线弯曲程度
        if abs(dy) < 0.01:  # 几乎水平
            curve_offset = 0.08 * np.sign(np.random.randn() + 0.1)  # 轻微随机弯曲
        else:
            curve_offset = 0.05 * np.sign(dy)
        
        # 控制点
        ctrl_x = (x1 + x2) / 2
        ctrl_y = (y1 + y2) / 2 + curve_offset
        
        # 对于循环连接，使用更大的弯曲
        if is_recurrent:
            if from_key == to_key:  # 自连接
                ctrl_x = x1 + 0.08
                ctrl_y = y1 + 0.1
                x2, y2 = x1 + 0.02, y1 + 0.03
            else:
                curve_offset = 0.12
                ctrl_y = max(y1, y2) + curve_offset
        
        # 绘制贝塞尔曲线
        verts = [(x1, y1), (ctrl_x, ctrl_y), (x2, y2)]
        codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
        path = Path(verts, codes)
        
        # 创建路径补丁
        patch = FancyArrowPatch(
            path=path,
            arrowstyle='-|>',
            mutation_scale=6 if compact_mode else 10,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            shrinkA=node_radius * 100 * 0.9,
            shrinkB=node_radius * 100 * 0.9,
            zorder=1,
        )
        ax.add_patch(patch)
        
        # 显示权重
        if show_weights and not compact_mode:
            mid_x = ctrl_x + 0.02
            mid_y = ctrl_y
            ax.text(mid_x, mid_y, f'{weight:.1f}', fontsize=5, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.1', facecolor='white', 
                            edgecolor='none', alpha=0.8), zorder=2)
    
    # 绘制节点
    for key, pos in positions.items():
        if key in input_keys:
            node_type = 'input'
            label = str(abs(key))
        elif key in output_keys:
            node_type = 'output'
            label = str(key)
        else:
            node_type = 'hidden'
            label = ''
        
        circle = plt.Circle(pos, node_radius, color=colors[node_type], 
                           ec='#333333', linewidth=1.5, zorder=10)
        ax.add_patch(circle)
        
        if label:
            ax.text(pos[0], pos[1], label, ha='center', va='center', 
                   fontsize=label_fontsize, fontweight='bold', color='#333333', zorder=11)
    
    # 设置图形属性
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.08, 1.08)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 添加标题
    if title:
        ax.set_title(title, fontsize=10 if compact_mode else 12, fontweight='bold', pad=5)
    
    # 添加统计信息（右上角，仅非紧凑模式）
    stats_text = f"Hidden: {structure['n_hidden']}\nConns: {structure['n_enabled_conns']}"
    if genome.fitness is not None:
        stats_text += f"\nFitness: {genome.fitness:.2f}"
    
    if not compact_mode:
        ax.text(0.98, 0.98, stats_text, ha='right', va='top', 
               fontsize=8, transform=ax.transAxes,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                        edgecolor='#CCCCCC', alpha=0.9),
               color='#333333')
    else:
        # 紧凑模式下的简短统计
        short_stats = f"H:{structure['n_hidden']} C:{structure['n_enabled_conns']}"
        if genome.fitness is not None:
            short_stats += f" F:{genome.fitness:.1f}"
        ax.text(0.5, -0.02, short_stats, ha='center', va='top', 
               fontsize=6, transform=ax.transAxes, color='#555555')


def visualize_single_genome(
    task_dir: str,
    generation: Optional[int] = None,
    output_dir: Optional[str] = None,
    show_weights: bool = False,
):
    """
    可视化单个基因组网络
    
    Args:
        task_dir: 任务结果目录
        generation: 指定代数，None表示最后一代
        output_dir: 输出目录，None则创建在task_dir下
        show_weights: 是否显示连接权重
    """
    # 设置路径
    checkpoint_dir = os.path.join(task_dir, 'checkpoints')
    if output_dir is None:
        output_dir = os.path.join(task_dir, 'genome_visualization')
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找检查点
    checkpoints = find_checkpoints(checkpoint_dir)
    if not checkpoints:
        raise FileNotFoundError(f"未找到检查点文件: {checkpoint_dir}")
    
    # 确定要使用的检查点
    if generation is not None:
        target_name = f'neat-checkpoint-{generation}'
        checkpoint_file = None
        for cp in checkpoints:
            if os.path.basename(cp) == target_name:
                checkpoint_file = cp
                break
        if checkpoint_file is None:
            # 找最接近的
            available_gens = [int(os.path.basename(cp).split('-')[-1]) for cp in checkpoints]
            closest = min(available_gens, key=lambda x: abs(x - generation))
            print(f"未找到第{generation}代，使用最接近的第{closest}代")
            checkpoint_file = os.path.join(checkpoint_dir, f'neat-checkpoint-{closest}')
            generation = closest
    else:
        checkpoint_file = checkpoints[-1]
        generation = int(os.path.basename(checkpoint_file).split('-')[-1])
    
    # 加载最佳基因组
    best_genome, gen, config = get_best_genome_from_checkpoint(checkpoint_file)
    if best_genome is None:
        raise ValueError(f"未找到有效的基因组")
    
    # 获取任务名
    task_name = os.path.basename(task_dir)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制网络
    title = f"{task_name}\nGeneration {generation}"
    draw_genome_network(best_genome, config, ax, title=title, show_weights=show_weights, compact_mode=False)
    
    # 添加图例（右下角，简洁版）
    legend_elements = [
        # 节点类型
        mpatches.Patch(color='#90EE90', ec='#333333', label='Input'),
        mpatches.Patch(color='#87CEEB', ec='#333333', label='Hidden'),
        mpatches.Patch(color='#FFB6C1', ec='#333333', label='Output'),
        # 连接类型
        Line2D([0], [0], color='#4A4A4A', lw=1.5, label='Forward (thick=strong)'),
        Line2D([0], [0], color='#888888', lw=1.5, linestyle='--', label='Recurrent'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8, 
             framealpha=0.95, edgecolor='#CCCCCC')
    
    # 保存图像
    output_file = os.path.join(output_dir, f'genome_gen{generation}.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"基因组可视化已保存: {output_file}")
    
    # 打印基因组信息
    structure = get_genome_structure(best_genome, config)
    print(f"\n基因组结构信息:")
    print(f"  代数: {generation}")
    print(f"  适应度: {best_genome.fitness:.4f}" if best_genome.fitness else "  适应度: N/A")
    print(f"  输入节点: {structure['n_inputs']}")
    print(f"  隐藏节点: {structure['n_hidden']}")
    print(f"  输出节点: {structure['n_outputs']}")
    print(f"  启用连接: {structure['n_enabled_conns']}")
    print(f"  总连接数: {structure['n_total_conns']}")


def visualize_evolution(
    task_dir: str,
    interval: int = 25,
    output_dir: Optional[str] = None,
    max_generations: Optional[int] = None,
):
    """
    可视化网络进化过程
    
    Args:
        task_dir: 任务结果目录
        interval: 每隔多少代显示一次
        output_dir: 输出目录
        max_generations: 最大显示代数，None表示全部
    """
    # 设置路径
    checkpoint_dir = os.path.join(task_dir, 'checkpoints')
    if output_dir is None:
        output_dir = os.path.join(task_dir, 'genome_visualization')
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找检查点
    checkpoints = find_checkpoints(checkpoint_dir)
    if not checkpoints:
        raise FileNotFoundError(f"未找到检查点文件: {checkpoint_dir}")
    
    # 获取可用代数
    all_generations = [int(os.path.basename(cp).split('-')[-1]) for cp in checkpoints]
    
    # 选择要展示的代数
    if max_generations is not None:
        all_generations = [g for g in all_generations if g <= max_generations]
    
    selected_generations = []
    for g in all_generations:
        if g == 0 or g % interval == 0 or g == all_generations[-1]:
            selected_generations.append(g)
    
    # 确保不会太多
    if len(selected_generations) > 16:
        step = len(selected_generations) // 16 + 1
        selected_generations = selected_generations[::step]
        if all_generations[-1] not in selected_generations:
            selected_generations.append(all_generations[-1])
    
    n_plots = len(selected_generations)
    if n_plots == 0:
        print("没有可用的检查点")
        return
    
    # 计算网格布局
    n_cols = min(4, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    # 创建大图
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_plots == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # 获取任务名
    task_name = os.path.basename(task_dir)
    fig.suptitle(f'Network Evolution: {task_name}', fontsize=14, fontweight='bold')
    
    # 加载并绘制每个代数的网络
    for idx, gen in enumerate(selected_generations):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        checkpoint_file = os.path.join(checkpoint_dir, f'neat-checkpoint-{gen}')
        if not os.path.exists(checkpoint_file):
            ax.axis('off')
            ax.text(0.5, 0.5, f'Gen {gen}\nNot found', ha='center', va='center')
            continue
        
        try:
            best_genome, _, config = get_best_genome_from_checkpoint(checkpoint_file)
            if best_genome is not None:
                draw_genome_network(best_genome, config, ax, title=f'Gen {gen}', compact_mode=True)
            else:
                ax.axis('off')
                ax.text(0.5, 0.5, f'Gen {gen}\nNo valid genome', ha='center', va='center', fontsize=8)
        except Exception as e:
            ax.axis('off')
            ax.text(0.5, 0.5, f'Gen {gen}\nError', ha='center', va='center', fontsize=8)
    
    # 隐藏多余的子图
    for idx in range(n_plots, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    # 添加全局图例（简洁版）
    legend_elements = [
        mpatches.Patch(color='#90EE90', ec='#333333', label='Input'),
        mpatches.Patch(color='#87CEEB', ec='#333333', label='Hidden'),
        mpatches.Patch(color='#FFB6C1', ec='#333333', label='Output'),
        Line2D([0], [0], color='#4A4A4A', lw=1.5, label='Forward'),
        Line2D([0], [0], color='#888888', lw=1.5, linestyle='--', label='Recurrent'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=8,
              bbox_to_anchor=(0.5, 0.01), framealpha=0.95)
    
    # 保存
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    output_file = os.path.join(output_dir, 'network_evolution.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"网络进化可视化已保存: {output_file}")
    print(f"展示的代数: {selected_generations}")


def plot_complexity_evolution(task_dir: str, output_dir: Optional[str] = None):
    """
    绘制网络复杂度（节点数和连接数）随代数变化的曲线
    
    Args:
        task_dir: 任务结果目录
        output_dir: 输出目录
    """
    checkpoint_dir = os.path.join(task_dir, 'checkpoints')
    if output_dir is None:
        output_dir = os.path.join(task_dir, 'genome_visualization')
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoints = find_checkpoints(checkpoint_dir)
    if not checkpoints:
        raise FileNotFoundError(f"未找到检查点文件: {checkpoint_dir}")
    
    generations = []
    n_hidden_list = []
    n_conns_list = []
    fitness_list = []
    
    for cp in checkpoints:
        gen = int(os.path.basename(cp).split('-')[-1])
        try:
            best_genome, _, config = get_best_genome_from_checkpoint(cp)
            if best_genome is not None:
                structure = get_genome_structure(best_genome, config)
                generations.append(gen)
                n_hidden_list.append(structure['n_hidden'])
                n_conns_list.append(structure['n_enabled_conns'])
                fitness_list.append(best_genome.fitness if best_genome.fitness else 0)
        except:
            continue
    
    if not generations:
        print("无法提取网络复杂度数据")
        return
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # 上图：适应度
    ax1.plot(generations, fitness_list, 'b-', linewidth=2, label='Best Fitness')
    ax1.fill_between(generations, fitness_list, alpha=0.3)
    ax1.set_ylabel('Fitness', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Network Complexity Evolution: {os.path.basename(task_dir)}', 
                  fontsize=14, fontweight='bold')
    
    # 下图：网络复杂度
    ax2.plot(generations, n_hidden_list, 'g-', linewidth=2, label='Hidden Nodes')
    ax2.plot(generations, n_conns_list, 'r-', linewidth=2, label='Connections')
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'complexity_evolution.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"复杂度演化曲线已保存: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='NEAT 基因组网络可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 可视化最后一代的最佳基因组
    python visualize_genome.py transport_recurrent_200_4_1_0.15_0.15_10.0_seed42
    
    # 可视化指定代数的最佳基因组
    python visualize_genome.py transport_recurrent_200_4_1_0.15_0.15_10.0_seed42 -g 50
    
    # 展示网络进化过程
    python visualize_genome.py transport_recurrent_200_4_1_0.15_0.15_10.0_seed42 --evolution
    
    # 展示网络进化过程，每50代显示一次
    python visualize_genome.py transport_recurrent_200_4_1_0.15_0.15_10.0_seed42 --evolution --interval 50
    
    # 显示连接权重
    python visualize_genome.py transport_recurrent_200_4_1_0.15_0.15_10.0_seed42 --weights
    
    # 绘制复杂度演化曲线
    python visualize_genome.py transport_recurrent_200_4_1_0.15_0.15_10.0_seed42 --complexity
        """
    )
    
    parser.add_argument('task_dir', type=str, 
                        help='任务结果目录名称（在results/目录下）')
    parser.add_argument('-g', '--generation', type=int, default=None,
                        help='指定要可视化的代数，不指定则使用最后一代')
    parser.add_argument('--evolution', action='store_true',
                        help='展示整个训练过程的网络进化')
    parser.add_argument('--interval', type=int, default=25,
                        help='进化展示时每隔多少代显示一次（默认25）')
    parser.add_argument('--weights', action='store_true',
                        help='显示连接权重')
    parser.add_argument('--complexity', action='store_true',
                        help='绘制网络复杂度演化曲线')
    parser.add_argument('--all', action='store_true',
                        help='执行所有可视化（单个网络 + 进化过程 + 复杂度曲线）')
    
    args = parser.parse_args()
    
    # 处理任务目录路径
    if os.path.isabs(args.task_dir):
        task_dir = args.task_dir
    else:
        # 相对路径，假设在 results/ 下
        task_dir = os.path.join(project_root, 'results', args.task_dir)
    
    if not os.path.exists(task_dir):
        print(f"错误: 任务目录不存在: {task_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"NEAT 基因组可视化")
    print(f"任务目录: {task_dir}")
    print(f"{'='*60}\n")
    
    # 执行可视化
    try:
        if args.all:
            # 执行所有可视化
            visualize_single_genome(task_dir, args.generation, show_weights=args.weights)
            visualize_evolution(task_dir, args.interval)
            plot_complexity_evolution(task_dir)
        elif args.evolution:
            visualize_evolution(task_dir, args.interval)
        elif args.complexity:
            plot_complexity_evolution(task_dir)
        else:
            visualize_single_genome(task_dir, args.generation, show_weights=args.weights)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n{'='*60}")
    print("可视化完成!")
    print(f"输出目录: {os.path.join(task_dir, 'genome_visualization')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
