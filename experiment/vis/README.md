# `experiment/vis` 使用说明

本目录提供 pure NEAT 训练结果的可视化工具：

- 网络结构可视化：查看全局最优基因组、指定代基因组、网络拓扑演化
- 结果分析：查看单次训练 dashboard，以及四张分析子图的独立导出

## 当前结果目录

当前训练结果使用分支目录：

```text
results/<task_params>/<branch>
```

例如当前仓库中的 transport 结果：

```text
results/transport_400_5_1_0.15_0.15_15.0/
  pure_neat/
    checkpoints/
    logs/log.json
    logs/best_log.json
    global_best_genome.pkl
    recurrent.cfg
  ea_rl/
    checkpoints/
    logs/metrics.jsonl
    logs/eval.jsonl
    best_actor.pt
```

本轮适配覆盖 `pure_neat`。下面这些输入会解析到同一个 pure NEAT 结果目录：

```bash
python experiment/vis/analyze_results.py transport_400_5_1_0.15_0.15_15.0
python experiment/vis/analyze_results.py results/transport_400_5_1_0.15_0.15_15.0
python experiment/vis/analyze_results.py results/transport_400_5_1_0.15_0.15_15.0/pure_neat
python experiment/vis/analyze_results.py /home/zjc/multi-agent-neat/results/transport_400_5_1_0.15_0.15_15.0/pure_neat
```

如果传入任务根目录 `results/<task_params>`，工具会默认选择 `pure_neat` 子目录。也可以显式写：

```bash
python experiment/vis/analyze_results.py transport_400_5_1_0.15_0.15_15.0 --branch pure_neat
```

## 脚本说明

- `visualize_genome.py`
  - 网络结构可视化 CLI
  - 读取 `global_best_genome.pkl`、`checkpoints/neat-checkpoint-*`、`recurrent.cfg`
- `result_analysis.py`
  - 结果分析 CLI 主实现
  - 读取 `logs/log.json`、`logs/best_log.json` 和 checkpoint 中的物种信息
- `analyze_results.py`
  - `result_analysis.py` 的薄包装入口
- `data.py`
  - 统一解析当前 `results/<task_params>/pure_neat` 目录
- `plotting.py`
  - 基础绘图辅助

## 环境要求

建议在项目使用的 Conda 环境中运行，例如：

```bash
conda activate NEAT-ROBO
```

依赖至少包括：

- `matplotlib`
- `numpy`
- `networkx`
- `neat-python`

## 1. 网络结构可视化

入口：

```bash
python experiment/vis/visualize_genome.py <task_params_or_pure_neat_dir> [options]
```

### 1.1 默认：全局最优基因组

```bash
python experiment/vis/visualize_genome.py transport_400_5_1_0.15_0.15_15.0
```

输出：

```text
results/transport_400_5_1_0.15_0.15_15.0/pure_neat/genome_visualization/genome_global_best.png
```

### 1.2 指定某一代

```bash
python experiment/vis/visualize_genome.py transport_400_5_1_0.15_0.15_15.0 -g 50
```

输出：

```text
results/transport_400_5_1_0.15_0.15_15.0/pure_neat/genome_visualization/genome_gen50.png
```

如果该代不存在，会自动回退到最接近的 checkpoint。

### 1.3 网络演化图

```bash
python experiment/vis/visualize_genome.py \
  transport_400_5_1_0.15_0.15_15.0 \
  --evolution \
  --interval 25
```

输出：

```text
results/transport_400_5_1_0.15_0.15_15.0/pure_neat/genome_visualization/network_evolution.png
```

### 1.4 一次生成全部网络图

```bash
python experiment/vis/visualize_genome.py transport_400_5_1_0.15_0.15_15.0 --all
```

### 1.5 保存到自定义目录

```bash
python experiment/vis/visualize_genome.py \
  transport_400_5_1_0.15_0.15_15.0 \
  --output-dir /tmp/neat_genome_vis
```

## 2. 结果分析

推荐入口：

```bash
python experiment/vis/analyze_results.py <task_params_or_pure_neat_dir> [options]
```

### 2.1 单次训练 dashboard

```bash
python experiment/vis/analyze_results.py transport_400_5_1_0.15_0.15_15.0 --dashboard
```

输出：

```text
results/transport_400_5_1_0.15_0.15_15.0/pure_neat/result_analysis/analysis_dashboard.png
results/transport_400_5_1_0.15_0.15_15.0/pure_neat/result_analysis/analysis_summary.json
```

dashboard 包括：

- 种群适应度均值、标准差和每代最佳个体
- 最佳个体评估均值、标准差、最小值、最大值
- 最优网络复杂度变化
- checkpoint 中记录的物种分布

### 2.2 独立导出四张子图

```bash
python experiment/vis/analyze_results.py \
  transport_400_5_1_0.15_0.15_15.0 \
  --panels
```

输出：

```text
population_evolution.png
evaluation_spread.png
complexity_panel.png
species_distribution.png
analysis_summary.json
```

只导出某一张：

```bash
python experiment/vis/analyze_results.py <task_params_or_pure_neat_dir> --population-panel
python experiment/vis/analyze_results.py <task_params_or_pure_neat_dir> --spread-panel
python experiment/vis/analyze_results.py <task_params_or_pure_neat_dir> --complexity-panel
python experiment/vis/analyze_results.py <task_params_or_pure_neat_dir> --species-panel
```

不显式指定模式时，默认执行 `--dashboard`。

### 2.3 保存到自定义目录

```bash
python experiment/vis/analyze_results.py \
  transport_400_5_1_0.15_0.15_15.0 \
  --dashboard \
  --output-dir /tmp/neat_result_analysis
```

## 3. Python API

```python
from experiment.vis import visualize_single_genome, visualize_evolution

visualize_single_genome("transport_400_5_1_0.15_0.15_15.0")
visualize_evolution("transport_400_5_1_0.15_0.15_15.0", interval=20)
```

```python
from experiment.vis import (
    plot_complexity_panel,
    plot_evaluation_spread_panel,
    plot_population_evolution_panel,
    plot_species_distribution_panel,
    plot_run_dashboard,
)

plot_run_dashboard("transport_400_5_1_0.15_0.15_15.0")
plot_population_evolution_panel("transport_400_5_1_0.15_0.15_15.0")
plot_evaluation_spread_panel("transport_400_5_1_0.15_0.15_15.0")
plot_complexity_panel("transport_400_5_1_0.15_0.15_15.0")
plot_species_distribution_panel("transport_400_5_1_0.15_0.15_15.0")
```

## 4. 常用指标

这些指标来自 `pure_neat/logs/log.json` 与 `pure_neat/logs/best_log.json`：

- `fitness_avg`
- `fitness_std`
- `fitness_best`
- `best_mean`
- `best_std`
- `best_max`
- `best_min`
- `best_median`
- `n_species`
- `time_elapsed`
- `n_neurons_best`
- `n_conns_best`

`analysis_summary.json` 会额外记录：

- 总代数与全局最优代数
- 种群适应度初值、末值、最值和提升量
- 最佳个体评估指标摘要
- 最优网络复杂度摘要
- 物种数量和最终物种规模分布
- 已生成的图文件列表

## 5. 注意事项

- 当前适配目标是 `pure_neat`，`ea_rl` 的 `metrics.jsonl` / `eval.jsonl` 不在本工具本轮绘图范围内
- `log.json` 与 `best_log.json` 是 JSONL，不是单个 JSON 对象
- 默认全局最优依赖 `global_best_genome.pkl`
- `--evolution` 在 checkpoint 很多时会自动稀疏采样
- 旧单层结果目录不再兼容；请传入 `results/<task_params>` 或 `results/<task_params>/pure_neat`

## 6. 快速检查

```bash
python experiment/vis/visualize_genome.py -h
python experiment/vis/analyze_results.py -h
```
