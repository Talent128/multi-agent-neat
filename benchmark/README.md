# Benchmark 基线对比实验

本目录包含用于对比 PyTorch-NEAT（进化算法）和 BenchMARL（强化学习）性能的脚本。

## 文件说明

- `run_benchmarl_baseline.py`: 运行 BenchMARL 强化学习基线实验
- `compare_results.py`: 对比分析两种方法的结果，生成曲线图和统计表格
- `plot_evolution.py`: 绘制种群进化过程图
- `benchmark_script.py`: 综合基准测试脚本（训练+绘图+对比一站式）

## ⚠️ 重要：评估机制差异与公平对比

### 评估机制差异

**PyTorch-NEAT (进化算法)**:
- 每代评估**整个种群**（`pop_size` 个不同的网络）
- 每个个体在 `trials` 个并行环境中评估，得到详细统计
- `fitness_avg`: 种群所有个体适应度的均值（多个不同网络）
- `fitness_best`: 最佳个体的适应度

**BenchMARL (强化学习)**:
- 每次评估**单一策略网络**
- 用 `evaluation_episodes` 次评估估计其期望回报
- `mean`: 策略在多次评估上的均值
- `std`: 策略在多次评估上的标准差

### ✅ 公平对比方案

为了进行公平对比，NEAT 现在会记录**最佳个体在多个环境上的详细统计**：

| NEAT | RL | 说明 |
|------|-----|-----|
| `best_mean` | `mean` | 最佳个体/策略在多次评估上的均值 |
| `best_std` | `std` | 最佳个体/策略在多次评估上的标准差 |
| `best_max` | `max_val` | 最佳个体/策略在多次评估上的最大值 |

**公平对比**：NEAT 的 `best_*` 统计 vs RL 的评估统计
- 两者都是"**单个网络/策略**在**多次评估**上的统计"
- 避免了 NEAT 种群统计与 RL 单策略的不公平对比

## 结果文件组织

### NEAT 结果目录结构
```
results/{task}_{algorithm}_{params}_seed{seed}/
├── logs/
│   ├── log.json       # 种群统计日志
│   └── best_log.json  # 最佳个体详细统计
├── evolution_plots/   # 种群进化图（由 plot_evolution.py 生成）
│   ├── population_evolution.png
│   └── evolution_summary.json
└── comparison/        # 对比结果（由 compare_results.py 生成）
    ├── comparison.png
    ├── statistics.txt
    └── statistics.json
```

### BenchMARL 结果
```
benchmark_results/{task}_{algorithm}_{params}_seed{seed}/
```
BenchMARL 生成的所有 CSV 文件类型
| 类别     | 文件名                                     | 说明                | 用途           |
|----------|--------------------------------------------|---------------------|----------------|
| 评估数据 | eval_reward_episode_reward_mean.csv         | 评估平均奖励        | 主要对比指标   |
| 评估数据 | eval_reward_episode_reward_max.csv          | 评估最大奖励        | 对比           |
| 评估数据 | eval_reward_episode_reward_min.csv          | 评估最小奖励        | 对比           |
| 评估数据 | eval_reward_episode_len_mean.csv            | 评估 episode 长度   | 参考           |
| 收集数据 | collection_reward_*.csv                     | 训练时收集的奖励    | 包含探索噪声   |
| 收集数据 | collection_agents_reward_*.csv              | 按智能体分的奖励    | 详细分析       |
| 训练数据 | train_agents_loss_critic.csv                | Critic 损失         | 训练监控       |
| 训练数据 | train_agents_loss_objective.csv             | Actor 损失          | 训练监控       |
| 训练数据 | train_agents_entropy.csv                    | 策略熵              | 探索程度       |
| 训练数据 | train_agents_clip_fraction.csv              | PPO clip 比例       | 训练稳定性     |
| 训练数据 | train_agents_kl_approx.csv                  | KL 散度             | 策略更新幅度   |
| 训练数据 | train_agents_explained_variance.csv         | 解释方差            | 价值估计质量   |
| 训练数据 | train_agents_ESS.csv                        | 有效样本大小        | 重要性采样效率 |
| 计时     | timers_collection_time.csv                  | 收集耗时            | 性能分析       |
| 计时     | timers_training_time.csv                    | 训练耗时            | 性能分析       |
| 计时     | timers_evaluation_time.csv                  | 评估耗时            | 性能分析       |
| 计数     | counters_total_frames.csv                   | 总帧数              | 进度跟踪       |
| 计数     | counters_iter.csv                           | 迭代数              | 进度跟踪       |

## 日志文件格式

### log.json - 种群统计
```json
{
  "generation": 0,
  "fitness_avg": 1.23,    // 种群均值（多个网络）
  "fitness_std": 0.45,    // 种群标准差
  "fitness_best": 2.34,   // 最佳个体适应度
  "pop_size": 150,
  "n_species": 3,
  "time_elapsed": 12.5
}
```

### best_log.json - 最佳个体详细统计（用于对比）
```json
{
  "generation": 0,
  "best_mean": 2.34,      // 最佳个体多环境均值（用于对比）
  "best_std": 0.12,       // 最佳个体多环境标准差（用于对比）
  "best_max": 2.56,       // 最佳个体多环境最大值（用于对比）
  "best_min": 2.10,       // 最佳个体多环境最小值
  "best_median": 2.30,    // 最佳个体多环境中位数
  "n_episodes": 60,       // 评估环境/回合数
  "n_neurons_best": 5,    // 最佳个体神经元数
  "n_conns_best": 12      // 最佳个体连接数
}
```

## 使用方法

### 1. 运行 BenchMARL 基线

```bash
# 单个任务，默认算法 (mappo, ippo)
python benchmark/run_benchmarl_baseline.py --task transport --seed 42

# 指定多个算法
python benchmark/run_benchmarl_baseline.py --task navigation --algorithms mappo ippo qmix --seed 42

# 运行所有可用任务
python benchmark/run_benchmarl_baseline.py --task all --seed 42

# 使用 GPU
python benchmark/run_benchmarl_baseline.py --task transport --device cuda --seed 42

# 查看可用任务和算法
python benchmark/run_benchmarl_baseline.py --list_tasks
python benchmark/run_benchmarl_baseline.py --list_algorithms
```

### 2. 绘制种群进化过程

```bash
# 自动查找结果目录
python benchmark/plot_evolution.py --task transport --seed 42

# 指定结果目录
python benchmark/plot_evolution.py --results_dir results/transport_recurrent_...
```

### 3. 对比分析

```bash
# 自动查找结果目录
python benchmark/compare_results.py --task transport --seed 42

# 指定结果目录
python benchmark/compare_results.py --neat_dir results/transport_recurrent_... --algorithms mappo ippo
```

### 4. 一站式运行

```bash
# 训练 + 进化图 + 对比分析
python benchmark/benchmark_script.py --task transport --seed 42

# 只运行对比分析（跳过训练）
python benchmark/benchmark_script.py --task transport --skip_training --seed 42

# 跳过进化绘图
python benchmark/benchmark_script.py --task transport --skip_evolution_plot --seed 42
```

## 输出说明

### 对比曲线图
只绘制均值对比（带标准差阴影），用于公平对比 NEAT 最佳个体与 RL 策略。

### 种群进化图
1. **population_evolution.png**: 种群平均适应度（带±1σ标准差阴影）和最佳个体适应度随代数变化
2. **evolution_summary.json**: 进化过程摘要（初始/最终适应度、改进幅度等）

## 配置对齐参数

基线训练脚本会自动从 PyTorch-NEAT 的配置中读取以下参数并对齐:

| 配置项 | 说明 |
|-------|-----|
| `max_n_frames` | 总训练步数 = generations × pop_size × max_steps |
| `evaluation_interval` | 评估间隔，对齐 NEAT 每代的步数（取 collected_frames_per_batch 的整数倍） |
| `evaluation_episodes` | 评估回合数 = NEAT 的 trials 参数 |
| `seed` | 随机种子 |
| `device` | 计算设备 |

## 步数计算

NEAT 步数计算公式：
```
steps = (generation + 1) × pop_size × max_steps
```

其中 generation 从 0 开始，所以第 N 代完成后的总步数是 `(N + 1) × pop_size × max_steps`。

## 支持的任务

当前支持的 VMAS 任务（需要在 `conf/task/vmas/` 目录下有对应配置）:

- transport
- navigation
- dispersion
- balance
- flocking
- give_way
- passage
- 等...

## 支持的算法

- **On-Policy**: mappo, ippo
- **Off-Policy (Actor-Critic)**: maddpg, iddpg, masac, isac
- **Value-Based**: qmix, vdn

## 注意事项

1. 确保已安装 BenchMARL 及其依赖
2. 确保 PyTorch-NEAT 的配置文件存在
3. GPU 训练需要足够的显存
4. 长时间训练建议使用 screen 或 tmux
