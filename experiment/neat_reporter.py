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
NEAT训练日志记录器

这个模块提供了一个报告器类，用于记录NEAT训练过程中的详细统计信息到JSON文件。
每一代的信息以单独的JSON行记录，便于后续分析、绘图和监控训练进度。


输出文件：
- log.json: 每代的种群统计日志（种群适应度平均,种群适应度标准差,最佳个体适应度,种群大小,物种数量,时间消耗等）
- best_log.json: 每代最佳个体的详细统计（最佳个体适应度平均,最佳个体适应度标准差,最佳个体适应度最大值,最佳个体适应度最小值,最佳个体神经元数量,最佳个体连接数量等）
"""

import copy
import json  # JSON序列化，用于保存日志
import os
import pickle
import time  # 时间测量，用于统计每代耗时
from pprint import pprint  # 美化打印，用于终端输出

import numpy as np  # 数值计算，用于统计均值和标准差
from neat.reporting import BaseReporter  # NEAT基础报告器类


class LogReporter(BaseReporter):
    """
    JSON日志报告器类
    
    继承自NEAT的BaseReporter，在训练过程中的关键时刻被调用，
    收集并记录各种统计信息到JSON文件。
    
    核心功能：
    1. 记录每代的种群适应度统计（fitness_avg, fitness_std等）
    2. 从 genome.eval_stats 获取最佳个体的详细统计（best_mean, best_std等）
    3. 记录种群和物种信息
    4. 跟踪训练时间
    5. 记录最佳网络的结构信息
    """
    def __init__(self, fnm, get_best_stats, eval_with_debug=False):
        """
        初始化日志报告器
        
        Args:
            fnm (str): log.json的文件路径
            get_best_stats (callable): 获取最佳个体统计的函数
                签名: get_best_stats(genome, config, debug=False) -> EvalStats
                注意：此函数从 genome.eval_stats 获取，无需重新评估
            eval_with_debug (bool): 是否启用调试输出
        """
        self.log_filename = fnm  # 保存文件名（而不是文件对象，以支持多进程pickle）
        # best_log.json 用于记录最佳个体的详细统计
        self.best_log_filename = fnm.replace('log.json', 'best_log.json')  
        self.generation = None  # 当前代数
        self.generation_start_time = None  # 当前代开始时间
        self.generation_times = []  # 最近几代的耗时列表（用于计算平均值）
        self.num_extinctions = 0 # 累计物种灭绝次数
        self.get_best_stats = get_best_stats  # 获取最佳个体统计函数
        self.eval_with_debug = eval_with_debug  # 是否启用调试输出
        self.log_dict = {}  # 当前代的日志字典
        self.best_log_dict = {}  # 当前代最佳个体的详细统计字典
        self.skip_next_log = False  # 是否跳过本次日志记录（检查点恢复时）

    def start_generation(self, generation):
        """
        在每代开始时调用
        
        记录代数并开始计时。
        
        Args:
            generation (int): 当前代数
        """
        self.log_dict["generation"] = generation  # 记录代数
        self.generation_start_time = time.time()  # 记录开始时间

    def end_generation(self, config, population, species_set):
        """
        在每代结束时调用
        
        将统计信息写入log.json和best_log.json。
        
        Args:
            config: NEAT配置对象
            population (dict): 当前种群 {genome_id: genome}
            species_set: 物种集合对象
        """
        # 如果设置了跳过标志（检查点恢复时），则跳过本次日志记录
        if self.skip_next_log:
            self.skip_next_log = False
            self.log_dict = {}
            self.best_log_dict = {}
            print("跳过本代日志记录（从检查点恢复）")
            return

        # 记录种群大小和物种数量
        self.log_dict["pop_size"] = len(population)
        self.log_dict["n_species"] = len(species_set.species)

        # 计算并记录本代耗时（纯训练时间）
        elapsed = time.time() - self.generation_start_time
        self.log_dict["time_elapsed"] = elapsed

        # 维护最近10代的耗时，用于计算平均值
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        self.log_dict["time_elapsed_avg"] = np.mean(self.generation_times)

        # 记录累计灭绝次数
        self.log_dict["n_extinctions"] = self.num_extinctions

        # 美化打印到终端
        pprint(self.log_dict)
        
        # 写入log.json
        with open(self.log_filename, "a") as log_file:
            log_file.write(json.dumps(self.log_dict) + "\n")
        
        # 写入best_log.json（最佳个体详细统计）
        if self.best_log_dict:
            with open(self.best_log_filename, "a") as best_log_file:
                best_log_file.write(json.dumps(self.best_log_dict) + "\n")
            self.best_log_dict = {}

    def post_evaluate(self, config, population, species, best_genome):
        """
        在评估完所有个体后调用
        
        计算种群适应度统计，从 genome.eval_stats 获取最佳个体的详细统计。
        
        Args:
            config: NEAT配置对象
            population (dict): 当前种群 {genome_id: genome}
            species: 物种集合（未使用）
            best_genome: 本代最佳基因组
        """
        # 收集所有个体的适应度，计算种群统计
        fitnesses = [c.fitness for c in population.values()]
        self.log_dict["fitness_avg"] = np.mean(fitnesses)
        self.log_dict["fitness_std"] = np.std(fitnesses)
        self.log_dict["fitness_best"] = best_genome.fitness

        print("=" * 50 + " Best Genome: " + "=" * 50)
        
        if self.eval_with_debug:
            print(best_genome)

        # 获取并记录最佳网络的结构信息
        n_neurons_best, n_conns_best = best_genome.size()

        # ========== 获取最佳个体的详细统计（直接从 genome.eval_stats 获取，无需重新评估）==========
        # 训练评估时已经计算了详细统计，这里直接获取
        eval_stats = self.get_best_stats(
            best_genome, config, debug=self.eval_with_debug
        )
        
        # 记录到best_log（用于与RL对比）
        self.best_log_dict["generation"] = self.log_dict["generation"]
        self.best_log_dict["best_mean"] = eval_stats.mean
        self.best_log_dict["best_std"] = eval_stats.std
        self.best_log_dict["best_max"] = eval_stats.max_val
        self.best_log_dict["best_min"] = eval_stats.min_val
        self.best_log_dict["best_median"] = eval_stats.median
        self.best_log_dict["n_episodes"] = eval_stats.n_episodes
        self.best_log_dict["n_neurons_best"] = n_neurons_best
        self.best_log_dict["n_conns_best"] = n_conns_best
        

    def complete_extinction(self):
        """
        当一个物种完全灭绝时调用
        
        递增灭绝计数器。物种灭绝意味着该物种的所有成员都被淘汰，
        这可能发生在物种长期停滞或表现不佳时。
        """
        self.num_extinctions += 1  # 递增灭绝计数

    def found_solution(self, config, generation, best):
        """
        当找到满足适应度阈值的解时调用
        
        这是BaseReporter接口的一部分，但本实现中未使用。
        
        Args:
            config: NEAT配置对象
            generation (int): 找到解的代数
            best: 最佳基因组
        """
        pass  # 不需要额外操作

    def species_stagnant(self, sid, species):
        """
        当一个物种停滞不前时调用
        
        物种停滞意味着该物种在一定代数内没有适应度提升。
        这是BaseReporter接口的一部分，但本实现中未使用。
        
        Args:
            sid: 物种ID
            species: 物种对象
        """
        pass  # 不需要额外操作


class GlobalBestReporter(BaseReporter):
    """训练期间持久化全局最优基因组，支持断点续训后继续比较。"""

    def __init__(self, package_path, reset_existing=False):
        self.package_path = package_path
        self.current_generation = None
        self.best_fitness = None
        self.best_generation = None

        if reset_existing and os.path.exists(self.package_path):
            os.remove(self.package_path)

        if not reset_existing:
            self._load_existing_package()

    def _load_existing_package(self):
        if not os.path.exists(self.package_path):
            return

        try:
            with open(self.package_path, "rb") as f:
                package = pickle.load(f)
        except Exception as exc:
            print(f"警告：读取全局最优基因组失败，忽略已有记录: {exc}")
            return

        self.best_fitness = package.get("fitness")
        self.best_generation = package.get("generation")

    def _save_package(self, best_genome, config):
        package = {
            "genome": copy.deepcopy(best_genome),
            "neat_config": config,
            "generation": self.current_generation,
            "fitness": best_genome.fitness,
            "genome_key": getattr(best_genome, "key", None),
        }

        tmp_path = f"{self.package_path}.tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(package, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, self.package_path)

    def start_generation(self, generation):
        self.current_generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        if best_genome is None or best_genome.fitness is None:
            return

        if self.best_fitness is None or best_genome.fitness > self.best_fitness:
            self._save_package(best_genome, config)
            self.best_fitness = best_genome.fitness
            self.best_generation = self.current_generation
            print(
                f"更新全局最优基因组: generation={self.current_generation}, "
                f"fitness={best_genome.fitness:.4f}"
            )
