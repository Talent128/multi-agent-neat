"""
从NEAT检查点中提取指定代数（或最后一代）适应度最高的基因组并保存

用法:
    python extract_best_genome.py                    # 提取最后一代
    python extract_best_genome.py --generation 50   # 提取第50代
    python extract_best_genome.py -g 100            # 提取第100代

将从 transport 和 flocking 任务的检查点中提取最佳基因组。
"""

import os
import gzip
import pickle
import neat
import argparse


def find_checkpoint(checkpoint_dir, generation=None):
    """查找检查点文件
    
    Args:
        checkpoint_dir: 检查点目录
        generation: 指定代数，None表示取最新的
        
    Returns:
        str: 检查点文件路径，未找到返回None
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('neat-checkpoint-')]
    if not checkpoints:
        return None
    
    # 按代数排序
    checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
    
    if generation is not None:
        # 查找指定代数的检查点
        target_name = f'neat-checkpoint-{generation}'
        if target_name in checkpoints:
            return os.path.join(checkpoint_dir, target_name)
        else:
            # 列出可用的代数
            available_gens = [int(f.split('-')[-1]) for f in checkpoints]
            print(f"  警告: 未找到第{generation}代检查点")
            print(f"  可用代数: {available_gens[:10]}{'...' if len(available_gens) > 10 else ''} (共{len(available_gens)}个)")
            # 找最接近的
            closest = min(available_gens, key=lambda x: abs(x - generation))
            print(f"  使用最接近的代数: {closest}")
            return os.path.join(checkpoint_dir, f'neat-checkpoint-{closest}')
    else:
        # 取最后一个
        return os.path.join(checkpoint_dir, checkpoints[-1])


def find_latest_checkpoint(checkpoint_dir):
    """查找最新的检查点文件（向后兼容）"""
    return find_checkpoint(checkpoint_dir, generation=None)


def extract_best_genome_from_checkpoint(checkpoint_file):
    """
    从检查点中提取最佳基因组
    
    Args:
        checkpoint_file: 检查点文件路径
        
    Returns:
        tuple: (best_genome, generation, neat_config)
    """
    # 加载检查点
    with gzip.open(checkpoint_file) as f:
        generation, config, population, species_set, rndstate = pickle.load(f)
    
    # 找到适应度最高的基因组
    best_genome = None
    best_fitness = float('-inf')
    
    for genome_id, genome in population.items():
        if genome.fitness is not None and genome.fitness > best_fitness:
            best_fitness = genome.fitness
            best_genome = genome
    
    return best_genome, generation, config


def save_best_genome(genome, neat_config, output_dir, task_name, generation):
    """
    保存最佳基因组和相关配置
    
    Args:
        genome: NEAT基因组对象
        neat_config: NEAT配置对象
        output_dir: 输出目录
        task_name: 任务名称
        generation: 代数
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存基因组（仅基因组对象）
    genome_file = os.path.join(output_dir, f"{task_name}_best_genome_gen{generation}.pkl")
    with open(genome_file, 'wb') as f:
        pickle.dump(genome, f)
    print(f"基因组已保存至: {genome_file}")
    
    # 保存完整包（基因组 + 配置，方便直接加载使用）
    full_package_file = os.path.join(output_dir, f"{task_name}_genome_package_gen{generation}.pkl")
    package = {
        'genome': genome,
        'neat_config': neat_config,
        'generation': generation,
        'fitness': genome.fitness,
        'task_name': task_name,
        'genome_info': {
            'num_nodes': len(genome.nodes),
            'num_connections': len(genome.connections),
            'num_enabled_connections': sum(1 for c in genome.connections.values() if c.enabled),
        }
    }
    with open(full_package_file, 'wb') as f:
        pickle.dump(package, f)
    print(f"完整包（基因组+配置）已保存至: {full_package_file}")
    
    return genome_file, full_package_file


def print_genome_info(genome, task_name, generation):
    """打印基因组信息"""
    print(f"\n{'='*60}")
    print(f"任务: {task_name}")
    print(f"代数: {generation}")
    print(f"适应度: {genome.fitness:.4f}" if genome.fitness else "适应度: N/A")
    print(f"节点数量: {len(genome.nodes)}")
    print(f"连接数量: {len(genome.connections)}")
    print(f"启用的连接: {sum(1 for c in genome.connections.values() if c.enabled)}")
    print(f"{'='*60}")
    
    # 打印节点信息
    print("\n节点信息:")
    for node_id, node in sorted(genome.nodes.items()):
        print(f"  节点 {node_id}: bias={node.bias:.4f}, response={node.response:.4f}, activation={node.activation}")
    
    # 打印连接信息（只显示启用的）
    print("\n启用的连接:")
    for conn_key, conn in sorted(genome.connections.items()):
        if conn.enabled:
            print(f"  {conn_key[0]} -> {conn_key[1]}: weight={conn.weight:.4f}")


def list_available_generations(checkpoint_dir):
    """列出检查点目录中可用的代数"""
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('neat-checkpoint-')]
    generations = sorted([int(f.split('-')[-1]) for f in checkpoints])
    return generations


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='从NEAT检查点中提取指定代数的最佳基因组',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python extract_best_genome.py                     # 提取最后一代
    python extract_best_genome.py -g 50               # 提取第50代
    python extract_best_genome.py --generation 100    # 提取第100代
    python extract_best_genome.py --list              # 列出可用代数
        """
    )
    parser.add_argument('-g', '--generation', type=int, default=None,
                        help='指定要提取的代数，不指定则提取最后一代')
    parser.add_argument('--list', action='store_true',
                        help='列出所有可用的代数检查点')
    args = parser.parse_args()
    
    # 结果目录
    results_base = "results"
    
    # 定义要提取的任务及其结果目录
    tasks = [
        {
            'name': 'transport',
            'result_dirs': [
                'transport_recurrent_400_5_1_0.15_0.15_15.0',
            ]
        },
        # {
        #     'name': 'flocking',
        #     'result_dirs': [
        #         'flocking_recurrent_300_5_3_-0.1_True_seed42',
        #     ]
        # },
        # {
        #     'name': 'navigation',
        #     'result_dirs': [
        #         'navigation_recurrent_200_4_True_1_False_True_False_0.35_0.1_seed42',
        #     ]
        # }
    ]
    
    # 如果只是列出可用代数
    if args.list:
        print("\n可用的检查点代数:")
        print("=" * 60)
        for task in tasks:
            for result_dir in task['result_dirs']:
                checkpoint_dir = os.path.join(results_base, result_dir, 'checkpoints')
                gens = list_available_generations(checkpoint_dir)
                if gens:
                    print(f"\n{task['name']} ({result_dir}):")
                    print(f"  代数范围: {gens[0]} - {gens[-1]} (共{len(gens)}个检查点)")
                    if len(gens) <= 20:
                        print(f"  所有代数: {gens}")
                    else:
                        print(f"  前10个: {gens[:10]}")
                        print(f"  后10个: {gens[-10:]}")
                else:
                    print(f"\n{task['name']}: 无检查点")
        return
    
    # 创建输出目录
    output_dir = "extracted_genomes"
    
    target_generation = args.generation
    if target_generation is not None:
        print(f"\n目标代数: {target_generation}")
    else:
        print("\n提取最后一代的最佳基因组")
    
    for task in tasks:
        task_name = task['name']
        print(f"\n{'#'*60}")
        print(f"# 处理任务: {task_name}")
        print(f"{'#'*60}")
        
        for result_dir in task['result_dirs']:
            checkpoint_dir = os.path.join(results_base, result_dir, 'checkpoints')
            
            # 查找指定代数或最新的检查点
            checkpoint_file = find_checkpoint(checkpoint_dir, generation=target_generation)
            if checkpoint_file is None:
                print(f"警告: 未找到检查点 - {checkpoint_dir}")
                continue
            
            print(f"\n加载检查点: {checkpoint_file}")
            
            # 提取最佳基因组
            best_genome, generation, neat_config = extract_best_genome_from_checkpoint(checkpoint_file)
            
            if best_genome is None:
                print(f"警告: 未找到有效的基因组")
                continue
            
            # 打印基因组信息
            print_genome_info(best_genome, task_name, generation)
            
            # 保存基因组
            genome_file, package_file = save_best_genome(
                best_genome, neat_config, output_dir, task_name, generation
            )
    
    print(f"\n{'='*60}")
    print("所有最佳基因组已提取完成！")
    print(f"保存目录: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

