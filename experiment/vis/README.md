# `experiment/vis` 使用说明

本目录提供两类工具：

- 网络结构可视化：查看单个基因组拓扑、进化过程
- 结果分析：查看单次训练总览、四张子图独立导出

默认假设结果目录位于项目根目录下的 `results/`。

## 目录结构

- `visualize_genome.py`
  - 网络结构可视化 CLI
  - 支持单个基因组、进化过程
- `result_analysis.py`
  - 结果分析 CLI 主实现
  - 支持单次训练总览、四张子图独立导出
- `analyze_results.py`
  - `result_analysis.py` 的薄包装入口
  - 直接执行它即可，不需要手动 `python -m`
- `data.py`
  - 数据读取辅助
  - 负责读取 `checkpoint`、`global_best_genome.pkl`、`log.json`、`best_log.json`
- `plotting.py`
  - 基础绘图辅助
  - 提供折线、区间带、分位数、直方图等基础图元
- `__init__.py`
  - 统一导出常用 API

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

## 结果目录输入规则

这两个 CLI 都支持两种输入方式：

1. 直接给绝对路径
2. 只给 `results/` 下的目录名

例如这两种等价：

```bash
python experiment/vis/visualize_genome.py transport_recurrent_410_5_1_0.15_0.15_15.0
python experiment/vis/visualize_genome.py /home/zjc/multi-agent-neat/results/transport_recurrent_410_5_1_0.15_0.15_15.0
```

## 1. 网络结构可视化

入口：

```bash
python experiment/vis/visualize_genome.py <result_dir> [options]
```

### 1.1 默认：显示全局最优基因组

```bash
python experiment/vis/visualize_genome.py transport_recurrent_410_5_1_0.15_0.15_15.0
```

输出：

- `results/<result_dir>/genome_visualization/genome_global_best.png`

### 1.2 指定某一代的最佳基因组

```bash
python experiment/vis/visualize_genome.py transport_recurrent_410_5_1_0.15_0.15_15.0 -g 50
```

输出：

- `results/<result_dir>/genome_visualization/genome_gen50.png`

如果该代不存在，会自动回退到最接近的 checkpoint。

### 1.3 展示网络进化过程

```bash
python experiment/vis/visualize_genome.py transport_recurrent_410_5_1_0.15_0.15_15.0 --evolution --interval 25
```

输出：

- `results/<result_dir>/genome_visualization/network_evolution.png`

说明：

- `--interval` 控制每隔多少代抽样一个 checkpoint
- 总子图过多时会自动稀疏采样

### 1.4 一次生成全部网络图

```bash
python experiment/vis/visualize_genome.py transport_recurrent_410_5_1_0.15_0.15_15.0 --all
```

会生成：

- `genome_global_best.png` 或 `genome_gen*.png`
- `network_evolution.png`

### 1.5 可选参数

```bash
python experiment/vis/visualize_genome.py -h
```

主要参数：

- `-g, --generation`
  - 指定要展示的代数
- `--evolution`
  - 绘制多代网络进化图
- `--interval`
  - 进化图抽样间隔
- `--all`
  - 一次生成所有网络相关图

## 2. 结果分析

推荐入口：

```bash
python experiment/vis/analyze_results.py <args>
```

`analyze_results.py` 只是转调 `result_analysis.py`，两者参数完全一致。

### 2.1 单次训练：Dashboard

```bash
python experiment/vis/analyze_results.py transport_recurrent_410_5_1_0.15_0.15_15.0 --dashboard
```

输出：

- `results/<result_dir>/result_analysis/analysis_dashboard.png`

内容包括：

- 种群进化过程（均值 + 标准差阴影 + 每代最佳）
- 最佳个体评估区间
- 网络复杂度变化
- 物种分布堆叠图

### 2.2 单次训练：独立导出四张子图

一次导出全部四张：

```bash
python experiment/vis/analyze_results.py \
  transport_recurrent_410_5_1_0.15_0.15_15.0 \
  --panels
```

输出：

- `results/<result_dir>/result_analysis/population_evolution.png`
- `results/<result_dir>/result_analysis/evaluation_spread.png`
- `results/<result_dir>/result_analysis/complexity_panel.png`
- `results/<result_dir>/result_analysis/species_distribution.png`

只导出某一张：

```bash
python experiment/vis/analyze_results.py <result_dir> --population-panel
python experiment/vis/analyze_results.py <result_dir> --spread-panel
python experiment/vis/analyze_results.py <result_dir> --complexity-panel
python experiment/vis/analyze_results.py <result_dir> --species-panel
```

### 2.3 不显式指定模式时的默认行为

- 如果不显式指定模式：
  - 默认执行 `--dashboard`

### 2.4 常用指标

这些指标来自 `log.json` 与 `best_log.json`：

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

## 3. Python API 用法

### 3.1 网络图 API

```python
from experiment.vis import visualize_single_genome, visualize_evolution

visualize_single_genome("transport_recurrent_410_5_1_0.15_0.15_15.0")
visualize_evolution("transport_recurrent_410_5_1_0.15_0.15_15.0", interval=20)
```

### 3.2 结果分析 API

```python
from experiment.vis import (
    plot_complexity_panel,
    plot_evaluation_spread_panel,
    plot_population_evolution_panel,
    plot_species_distribution_panel,
    plot_run_dashboard,
)

plot_run_dashboard("transport_recurrent_410_5_1_0.15_0.15_15.0")
plot_population_evolution_panel("transport_recurrent_410_5_1_0.15_0.15_15.0")
plot_evaluation_spread_panel("transport_recurrent_410_5_1_0.15_0.15_15.0")
plot_complexity_panel("transport_recurrent_410_5_1_0.15_0.15_15.0")
plot_species_distribution_panel("transport_recurrent_410_5_1_0.15_0.15_15.0")
```

## 4. 内部辅助模块

这些模块通常不给终端直接调用，但在扩展新可视化时会用到。

### `data.py`

常用函数：

- `resolve_task_dir(task_dir)`
- `load_run_history(task_dir)`
- `load_species_history(task_dir)`
- `get_checkpoint_paths(task_dir)`
- `load_best_genome_from_generation(task_dir, generation)`
- `load_global_best_genome(task_dir)`

### `plotting.py`

常用函数：

- `plot_line(...)`
- `plot_band(...)`
- `style_axis(...)`
- `finalize_figure(...)`

## 5. 推荐工作流

### 看单次训练效果

```bash
python experiment/vis/visualize_genome.py <result_dir>
python experiment/vis/analyze_results.py <result_dir> --dashboard
```

### 看结构如何演化

```bash
python experiment/vis/visualize_genome.py <result_dir> --evolution --interval 20
```

## 6. 注意事项

- `log.json` 与 `best_log.json` 是 JSONL，不是单个 JSON 对象
- 默认全局最优依赖 `global_best_genome.pkl`
- 如果 checkpoint 太多，`--evolution` 会自动下采样
- 如果想集中保存单次训练分析图，可使用 `--output-dir`

## 7. 快速检查

查看帮助：

```bash
python experiment/vis/visualize_genome.py -h
python experiment/vis/analyze_results.py -h
```
