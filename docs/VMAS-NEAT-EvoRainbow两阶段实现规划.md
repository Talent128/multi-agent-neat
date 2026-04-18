# VMAS + NEAT + EvoRainbow 两阶段实现规划

本文档基于当前工作区 `/home/zjc/multi-agent-neat` 与参考项目 `/home/zjc/EvoRainbow` 的实际代码实现整理，目标是先给出可落地的技术路线，再进入代码实现。

## 1. 目标重述

你要的不是“NEAT 训练完以后单独再跑一个 RL baseline”，而是一个明确的两阶段系统：

1. 第一阶段：在当前项目里继续使用 NEAT，对 VMAS 任务的网络参数和网络结构一起进化。
2. 第一阶段结束后：在第 `N` 代冻结最优个体的拓扑，只保留参数可训练。
3. 第一阶段训练期间：对精英个体额外采集轨迹，构造 critic 预训练数据。
4. 第二阶段：以第一阶段最优网络为 actor 初值，参考 `EvoRainbow` 的思路，做 EA + RL 双向耦合训练。
5. 多智能体场景下：要按 MARL 方式实现，但应优先使用“参数共享 actor + 集中式 critic”的 CTDE 方案。

我的结论先写在前面：

- 第一阶段继续保留当前仓库的 NEAT 结构搜索最合适。
- 第二阶段不建议直接套现成 BenchMARL/PPO 主循环，而应该实现一个自定义的“参数共享 actor + 集中式 twin critic + CEM/EA 分支”的离策略框架。
- `EvoRainbow` 里真正值得直接复用的是 `sepCEM`、参数向量化/注入思路、共享 replay buffer、`RL -> EA` 注入和 `elite -> RL` 软更新机制。
- `EvoRainbow` 里的 actor/critic 网络本体不能直接照搬，因为它是单智能体 MLP，和当前固定拓扑的 NEAT recurrent actor 不同。

## 2. 当前项目的具体实现分析

### 2.1 训练入口与配置组织

当前仓库的主入口是：

- `train.py`
- `hydra_config.py`
- `conf/config.yaml`
- `conf/experiment/base_experiment.yaml`
- `conf/algorithm/recurrent.yaml`
- `conf/algorithm/neat_config/recurrent.cfg`

整体结构是 Hydra 组合配置：

- `algorithm=recurrent`
- `task=vmas/...`
- `experiment=base_experiment`

`train.py` 只负责：

1. 读取 Hydra 运行时选择。
2. 通过 `load_experiment_from_hydra()` 创建 `Experiment`。
3. 复制 `.hydra` 配置到结果目录。
4. 调用 `experiment.run()`。

### 2.2 当前 NEAT 训练主链路

核心类在 `experiment/experiment.py`，当前进化流程是：

1. `Experiment.__init__`
   - 解析任务名。
   - 创建结果目录。
   - 建一个 VMAS 测试环境拿到 `obs_dim` 和 `action_dim`。
   - 用实际输入输出维度替换 `recurrent.cfg` 占位符。
   - 生成 `neat.Config`。

2. `Experiment._train`
   - 从 checkpoint 恢复或新建 `better_Population`。
   - 注册 `StdOutReporter`、`StatisticsReporter`、`Checkpointer`、`LogReporter`。
   - 调用 `population.run(self._eval_genomes, generations)`。

3. `better_Population.run`
   - 调用 `_eval_genomes()` 评估整代种群。
   - 记录 best genome。
   - 调用 `reproduction.reproduce()` 生成下一代。
   - 再做 speciation。

4. `Experiment._eval_genomes`
   - CPU 模式走 `GenomeEvaluator` 单基因组评估。
   - GPU 模式走 `BatchGenomeEvaluator` 批量评估。
   - 评估后把 `stats.mean` 写入 `genome.fitness`，把完整统计写入 `genome.eval_stats`。

### 2.3 当前种群评估与 fitness 定义

单基因组评估逻辑在 `experiment/evaluator.py`：

1. 用 `self.make_net(genome, config, batch_size)` 创建 `RecurrentNet`。
2. 重置 VMAS 环境。
3. 每个环境步，对每个 agent 依次调用同一个 `net` 生成动作。
4. 环境返回 `rews` 后，做：
   - `rewards = torch.stack(rews, dim=1)`
   - `global_reward = rewards.mean(dim=1)`
   - 用所有 agent 奖励均值作为该环境的团队奖励。
5. 在 `trials` 个并行环境上累计回报。
6. 最终 fitness 使用 `EvalStats.mean`。

这意味着当前项目的优化目标是：

- 单个共享控制器
- 在多个并行 VMAS episode 上的平均团队回报

### 2.4 当前项目已经默认做了参数共享

这一点很关键。

当前第一阶段不是“每个 agent 一个独立策略”，而是：

- 一个 genome
- 一个 `RecurrentNet`
- 所有 agent 共用这一套参数

所以从实现上说，第一阶段已经是“参数共享的多智能体控制器”。

这也是为什么第二阶段如果想无缝承接第一阶段，actor 也必须继续参数共享，否则 warm start 会发生结构性偏移。

### 2.5 一个必须明确写出来的实现细节

当前 recurrent policy 的“隐藏状态”并不是标准的“每个 agent 各自一份 RNN state”。

原因在于：

- `GenomeEvaluator` 在每个时间步里按 `for i in range(len(obs))` 顺序，对同一个 `net` 连续调用 `activate(obs[i])`。
- `RecurrentNet` 内部的 `activs` 和 `outputs` 是网络对象级别状态，没有 agent 维度。

这会带来一个非常重要的现象：

- 当前第一阶段实际上是“参数共享 + 跨 agent 共享 recurrent state”。
- agent 1 的 forward 会看到 agent 0 刚刚写进去的隐状态。

这不一定是错，但它不是标准 MARL actor 语义，必须在第二阶段设计里显式处理。

### 2.6 当前项目已存在的可复用桥接代码

当前仓库已经不是纯 NEAT 项目了，实际上已经有两块和第二阶段高度相关的基础设施：

1. `pytorch_neat/differentiable_recurrent_net.py`
   - 把 NEAT recurrent 网络改成可微版本。
   - 支持参数回写 genome。
   - 非常适合作为“冻结拓扑后 actor 核心”。

2. `benchmark/run_neat_rl_finetune.py`
   - 已经能从 checkpoint 中抽取 best genome。
   - 已经能把 genome 打包成 `genome_package.pkl`。
   - 已经能把 NEAT actor warm start 给 BenchMARL。

3. `pytorch_neat/benchmarl_recurrent_policy.py`
   - 已经实现了“从 frozen genome 构造 actor”的模型包装器。
   - 明确要求 `share_params=True`。

### 2.7 当前项目对第二阶段的主要限制

当前实现离目标还差几块：

1. 没有“冻结第 N 代并进入第二阶段”的统一 orchestrator。
2. 没有第一阶段精英轨迹采集与持久化。
3. 没有为 VMAS 多智能体写好的集中式 critic。
4. 没有共享 replay buffer 的 EA/RL 双向耦合循环。
5. 没有把 `EvoRainbow` 的 CEM 支路接到当前 fixed-topology actor 上。
6. 没有解决“当前共享 recurrent state 语义”和“标准 MARL per-agent recurrent state 语义”的一致性问题。

## 3. 当前项目的进化实现流程总结

基于实际代码，当前 NEAT 进化流程可以抽象成下面这条链：

1. Hydra 加载 `task`、`algorithm`、`experiment` 配置。
2. `Experiment` 根据 VMAS 环境自动确定输入输出维度。
3. 用替换后的 `recurrent.cfg` 初始化 `neat.Config`。
4. 新建或恢复 `better_Population`。
5. 对整代 population 做评估：
   - CPU：逐个 genome
   - GPU：`BatchGenomeEvaluator`
6. 每个 genome 在 `trials` 个并行 VMAS 环境中跑满 `max_steps`。
7. fitness 取团队平均回报均值。
8. reporter 记录：
   - 整体种群均值/方差
   - 本代 best genome 的详细统计
9. reproduction + mutation + crossover + speciation 生成下一代。
10. 每代 checkpoint 保存完整 population。

当前“进化的是”：

- 拓扑：节点增删、连接增删
- 参数：连接权重、bias 等

当前“没有进化的是”：

- 任何 critic
- 任何共享 replay buffer
- 任何 RL 分支

## 4. EvoRainbow 项目的具体实现分析

参考重点在：

- `MUJOCO/EvoRainbow_core/agent.py`
- `MUJOCO/EvoRainbow_core/EvoRainbow_Algs.py`
- `MUJOCO/EvoRainbow_core/ES.py`
- `MUJOCO/EvoRainbow_core/utils.py`
- `MUJOCO/EvoRainbow_core/replay_memory.py`
- `MetaWorld/EvoRainbow_Exp_sac_agent.py`

### 4.1 EvoRainbow 的主循环

`EvoRainbow_core/agent.py::Agent.train()` 的逻辑非常清晰：

1. 用 `sepCEM.ask()` 从参数分布采样一批 actor 参数。
2. 把这些参数注入种群 actor。
3. EA 分支评估整个人口，得到 fitness。
4. `sepCEM.tell()` 用 fitness 反向更新分布均值/协方差。
5. RL 分支单独与环境交互，把轨迹塞进共享 replay buffer。
6. RL 用 replay buffer 更新 actor/critic。
7. 如果 EA elite 比 RL 好，elite 可软更新 RL。
8. 周期性地把 RL actor 注入 EA 种群，通常替换最差个体。

也就是说，`EvoRainbow` 的关键不只是 “EA 和 RL 同时存在”，而是三层耦合：

- 数据层：EA/RL 共用 replay buffer
- 策略层：RL actor 注入 population
- 参数层：elite 软更新 RL

### 4.2 EvoRainbow 中真正值得复用的东西

#### 4.2.1 `sepCEM`

位置：

- `/home/zjc/EvoRainbow/MUJOCO/EvoRainbow_core/ES.py`

这是第二阶段 EA 分支最值得直接拿来的组件。

原因：

- 第一阶段结束后你已经冻结拓扑，第二阶段只优化参数。
- 这时继续用 full NEAT 做结构搜索反而不自然。
- `sepCEM` 正好适合“固定维度参数向量”的搜索。

#### 4.2.2 参数向量化与参数注入思路

位置：

- `/home/zjc/EvoRainbow/MUJOCO/EvoRainbow_core/EvoRainbow_Algs.py`
- `/home/zjc/EvoRainbow/MetaWorld/models.py`

这两个地方都实现了：

- `get_params()`
- `set_params()`
- `get_size()`
- `inject_parameters()`
- `extract_parameters()`

第二阶段的 fixed-topology NEAT actor 也必须有这套能力。

#### 4.2.3 共享 replay buffer 形态

位置：

- `/home/zjc/EvoRainbow/MUJOCO/EvoRainbow_core/utils.py::ReplayBuffer`

虽然这里是单智能体 transition，但设计思想可以直接复用：

- EA 和 RL 都往同一个离策略 buffer 写入数据。
- critic 只依赖 replay buffer，而不是只依赖 RL 自采样本。

#### 4.2.4 RL -> EA 注入和 elite -> RL 软更新

位置：

- `/home/zjc/EvoRainbow/MUJOCO/EvoRainbow_core/agent.py`
- `/home/zjc/EvoRainbow/MetaWorld/EvoRainbow_Exp_sac_agent.py`

这两条交互正是你在 `docs/两阶段EA+RL协同.txt` 里写的逻辑映射：

- `rl_to_evo()`: 把 RL actor 拷到种群里
- `EA_tau soft update`: elite 参数软更新 RL actor

### 4.3 EvoRainbow 中不能直接照搬的部分

#### 4.3.1 Actor / Critic 网络本体

不能直接复制的原因：

1. `EvoRainbow` 的 actor 是固定 MLP。
2. 当前项目第一阶段输出的是 NEAT recurrent 拓扑。
3. 第二阶段要保留第一阶段结构，不能换成普通 MLP。

#### 4.3.2 单智能体环境循环

`EvoRainbow` 默认是 Gym/MuJoCo 或 MetaWorld 单智能体接口：

- `state`
- `action`
- `next_state`
- `reward`
- `done`

当前项目是 VMAS 多智能体：

- `obs` 是 list，每个 agent 一份
- `rews` 也是 list
- 还要明确局部观测、联合观测、团队奖励、集中式 critic 输入格式

所以它的环境采样循环只能借鉴，不宜直接复制。

#### 4.3.3 `shared_state_embedding` / `Policy_Value_Network`

这些模块强依赖：

- 固定 latent size
- 单智能体输入形态
- actor 为普通线性头

不适合直接插到 frozen NEAT recurrent actor 上。

## 5. 多智能体下是否应该使用 MARL + 参数共享

结论：

- 是，应该用 MARL。
- 但不是“多套 actor 各自训练”的那种，而是“参数共享 actor + 集中式 critic”的 CTDE。

### 5.1 为什么必须是 MARL

因为第二阶段你要训练 critic，而 critic 需要看到多智能体的联合状态与联合动作关系。

如果还按单智能体 RL 去做，会丢失：

- agent 间协作约束
- 联合动作对团队奖励的贡献
- 非平稳性处理

### 5.2 为什么 actor 应该继续参数共享

原因有四个：

1. 第一阶段本来就是共享一个 genome。
2. VMAS 里的 `transport`、`flocking`、`navigation` 这类任务，agent 通常是同构 cooperative agents。
3. 参数共享能大幅降低第二阶段参数维度，EA 分支更稳定。
4. 这样可以直接用第一阶段 best genome 做 actor warm start。

### 5.3 为什么 critic 要集中式

因为你想做的是：

- EA 轨迹喂 critic
- RL 用 critic 做 actor 改进
- EA / RL 共用 buffer

这天然要求一个 CTDE critic，输入至少要包含：

- 所有 agent 的局部观测拼接，或环境可访问的全局状态
- 所有 agent 的动作拼接

### 5.4 一个范围限定

这个结论优先适用于“同构 cooperative 任务”，例如：

- `transport`
- `flocking`
- `navigation`
- `dispersion`
- `balance`

如果后面要扩展到：

- 对抗任务
- 异构角色任务
- 通信任务

那就不能全局只用一个共享 actor，而应该变成“按 agent group 共享”。

## 6. 推荐的总体实现路线

### 6.1 总体设计

建议分成两个显式阶段，而不是把所有逻辑直接糊进 `Experiment`：

#### 阶段一：NEAT 结构搜索阶段

保留现有 `Experiment` 为主干，只新增：

1. 精英轨迹采集
2. 冻结代导出
3. frozen actor 参数化接口

#### 阶段二：固定拓扑的 EA + RL 协同阶段

新建单独模块，核心是：

1. fixed-topology shared recurrent actor
2. centralized twin critic
3. shared replay buffer
4. CEM population branch
5. RL branch
6. EA -> RL / RL -> EA 双向同步

### 6.2 为什么第二阶段不建议直接基于 PPO / IPPO / MAPPO 主循环

虽然当前仓库已有 `benchmark/run_neat_rl_finetune.py`，但那更适合作为“warm start RL baseline”，不适合作为你要的最终双向协同主框架，原因有两个：

1. 你要让 EA 产生的轨迹直接喂给 critic，这天然更适合离策略 actor-critic。
2. 你要做 `RL -> EA` 注入和 `elite -> RL` 参数软更新，这更接近 `EvoRainbow`/TD3/CEM，而不是 PPO。

所以第二阶段建议主算法选：

- 参数共享 shared actor
- centralized twin critic
- TD3 风格离策略更新
- CEM 作为 EA 支路

可以把它理解成：

- “多智能体参数共享版 EvoRainbow”
- 或者 “PS-MATD3 + CEM over frozen NEAT actor”

## 7. 分模块实施方案

### 7.1 模块 A：冻结拓扑与 actor 导出

#### 目标

把第 `N` 代最佳 genome 变成第二阶段可直接消费的 frozen actor package。

#### 建议落位

- 复用 `extract_best_genome.py`
- 新增 `stage2/frozen_actor.py`
- 新增 `stage2/genome_package.py`

#### 关键实现

1. 从 checkpoint 提取指定代数 best genome。
2. 构造 `FrozenGenomePackage`：
   - genome
   - neat_config
   - generation
   - task config
   - obs/action dims
   - activation mapping
   - recurrent-state mode
3. 提供 fixed-topology actor 的：
   - `forward()`
   - `get_param_vector()`
   - `set_param_vector()`
   - `clone_from_vector()`

#### 特别要求

这里必须把“recurrent-state mode”显式写进 package：

- `legacy_shared_across_agents`
- `per_agent`

第一版建议默认保持 `legacy_shared_across_agents`，先保证和第一阶段行为一致。

### 7.2 模块 B：第一阶段精英轨迹采集

#### 目标

在 NEAT 阶段为第二阶段 critic 预训练准备数据。

#### 推荐方案

不要在“评估所有 genome”时把所有轨迹都搬出来，那样会很重。

建议做法：

1. 先按当前方式完成整代 fitness 评估。
2. 每代选 top-k 或 top-fraction 精英。
3. 对这些精英单独再跑一次 trajectory collection。
4. 只把精英轨迹写入 `elite_dataset`。

#### 数据字段建议

每条 transition 至少包含：

- `generation`
- `genome_id`
- `is_elite`
- `local_obs`: `[n_agents, obs_dim]`
- `joint_obs`: `[n_agents * obs_dim]`
- `actions`: `[n_agents, action_dim]`
- `joint_actions`: `[n_agents * action_dim]`
- `agent_rewards`: `[n_agents]`
- `team_reward`
- `next_local_obs`
- `next_joint_obs`
- `done`
- `return_to_go`

#### 为什么用 team_reward

当前第一阶段 fitness 就是 `agent rewards` 的均值累计，所以第二阶段 critic 预训练最好先保持同一奖励定义，避免目标漂移。

### 7.3 模块 C：critic 预训练

#### 目标

用第一阶段精英轨迹给第二阶段 critic 一个合理初始化。

#### 建议形式

critic 用 centralized twin Q：

- 输入：`joint_obs`, `joint_actions`
- 输出：`Q1`, `Q2`

预训练目标建议先用 Monte-Carlo return 回归，而不是一上来就 bootstrap TD：

- 更简单
- 不依赖已训练 target critic
- 非常适合离线 elite dataset 冷启动

#### 实现步骤

1. 从 `elite_dataset` 加载 transition。
2. 计算每条 transition 的 `return_to_go`。
3. 用 `MSE(Q1, G_t) + MSE(Q2, G_t)` 做预训练。
4. 保存 pretrained critic checkpoint。

### 7.4 模块 D：第二阶段 EA 分支

#### 目标

在 frozen actor 参数空间里继续全局搜索。

#### 推荐算法

直接采用 `sepCEM`。

#### 复用来源

- `/home/zjc/EvoRainbow/MUJOCO/EvoRainbow_core/ES.py::sepCEM`

#### 实现方式

1. 以 frozen best actor 参数向量作为 `mu_init`。
2. 每轮 `ask(pop_size)` 采样出一组参数向量。
3. 将参数注入 population actor。
4. 在 VMAS 中评估 population。
5. fitness 使用团队回报。
6. `tell(solutions, fitnesses)` 更新分布。

#### 进一步优化

第二阶段评估器建议复用当前 GPU 批量评估思路，新增一个：

- `stage2/batched_population_evaluator.py`

核心思想和 `BatchGenomeEvaluator` 一样：

- 一次性开 `n_candidates * trials_per_candidate` 个 VMAS env
- 每个 candidate 占一个连续 block
- 同代 trial 初始状态广播一致

### 7.5 模块 E：第二阶段 RL 分支

#### 目标

让 critic 利用 replay buffer 对 actor 做局部可微精修。

#### 推荐结构

- shared recurrent actor：沿用 frozen NEAT actor
- centralized twin critic：新写
- target actor / target critic：TD3 风格
- exploration：动作高斯噪声

#### 为什么不是直接用 `BenchmarlRecurrentPolicy`

因为它当前默认是“每个 agent 一份 recurrent state”，而第一阶段真实语义是“跨 agent 共享 recurrent state”。

第二阶段如果直接切过去，会导致：

- warm start 参数相同
- 但时序状态语义变了

这会让“冻结网络作为 actor 初值”在行为上不严格成立。

所以第二阶段 actor 最好单独实现一个与第一阶段一致的 wrapper。

### 7.6 模块 F：EA 与 RL 的交互机制

这一块直接按你的文档和 EvoRainbow 映射：

#### F1. EA -> RL：经验共享

EA population rollout 后，把 transition 写入共享 replay buffer。

第一版建议不要把所有个体都无脑入库，默认策略：

- RL 自己的 rollout 全部入库
- EA 只把 top-k 或 top-fraction 个体轨迹入库

这样能减少低质量样本污染 critic。

#### F2. RL -> EA：策略注入

每隔 `sync_period`：

1. 取 RL actor 当前参数。
2. 替换 EA population 里最差个体。
3. 让该个体参与下一轮 CEM 竞争。

这部分几乎可以直接照 `EvoRainbow_core/agent.py::rl_to_evo()` 逻辑改写。

#### F3. EA -> RL：精英软更新

每轮或每隔若干轮：

1. 找到 EA elite。
2. 若 elite 明显优于 RL actor，则执行：
   - `theta_rl <- tau * theta_elite + (1 - tau) * theta_rl`
3. 同步到 `target_actor`。

这一步在 frozen topology 条件下才成立。

## 8. 关键设计决策

### 8.1 第二阶段 EA 是否继续用 NEAT

不建议。

原因：

1. 你已经明确要冻结拓扑。
2. 拓扑冻结后，NEAT 的结构创新优势基本消失。
3. 第二阶段更适合参数分布搜索。
4. `EvoRainbow` 的 `sepCEM` 已经正好解决这个问题。

所以推荐：

- 第一阶段：NEAT
- 第二阶段：CEM/sepCEM

### 8.2 第二阶段是否保留第一阶段的共享 recurrent state 语义

建议第一版保留，理由：

1. 这能保证 frozen actor 行为严格继承第一阶段。
2. 这比“直接切换到 per-agent RNN state”风险更低。
3. 后续可以再做一个可切换选项。

但必须在配置里显式化，不能继续保持“代码隐含语义”。

### 8.3 是否继续用 BenchMARL

建议分开看：

- 作为 RL baseline 和 warm-start 验证工具：继续保留，很有价值。
- 作为目标中的双向 EA+RL 主训练框架：不建议作为第一实现路径。

原因是你要的交互粒度已经超过“普通 benchmark finetune”。

## 9. 建议的代码落位

建议新增一个明确的 `stage2/` 子包，避免把双阶段逻辑继续塞进 `experiment/`：

- `stage2/config.py`
- `stage2/frozen_actor.py`
- `stage2/frozen_actor_package.py`
- `stage2/trajectory_collector.py`
- `stage2/elite_dataset.py`
- `stage2/replay_buffer.py`
- `stage2/centralized_critic.py`
- `stage2/cem.py`
- `stage2/population.py`
- `stage2/batched_population_evaluator.py`
- `stage2/pretrain_critic.py`
- `stage2/trainer.py`
- `stage2/checkpoint.py`

训练入口建议新增：

- `train_two_stage.py`

Hydra 配置建议新增：

- `conf/two_stage/base.yaml`
- `conf/two_stage/task/vmas/...`

## 10. 单元测试规划

这部分必须跟着实现分阶段落地，不能最后再补。

### 10.1 冻结导出与 actor 一致性

建议新增：

- `tests/test_frozen_actor_package.py`
- `tests/test_frozen_actor_rollout_equivalence.py`

测试点：

1. 从 checkpoint 提取的 genome package 可序列化/反序列化。
2. frozen actor 的参数向量 `get/set` 可逆。
3. frozen actor 在相同输入下输出与 `RecurrentNet` 一致。
4. 如果使用 `legacy_shared_across_agents` 模式，整步 rollout 与当前 `GenomeEvaluator` 行为一致。

### 10.2 精英轨迹采集

建议新增：

- `tests/test_elite_trajectory_collection.py`

测试点：

1. top-k 精英筛选正确。
2. 轨迹字段 shape 正确。
3. `joint_obs` / `joint_actions` 拼接正确。
4. `return_to_go` 计算正确。
5. 同 seed 下轨迹采样可复现。

### 10.3 critic 预训练

建议新增：

- `tests/test_critic_pretraining.py`

测试点：

1. critic 能在小 synthetic dataset 上过拟合。
2. twin critic 输出 shape 正确。
3. 预训练 loss 随迭代下降。
4. 保存/加载 checkpoint 后数值一致。

### 10.4 CEM 支路

建议新增：

- `tests/test_sep_cem_on_frozen_actor.py`

测试点：

1. `ask()` 返回参数维度正确。
2. `tell()` 后分布均值发生更新。
3. elitism 生效。
4. 参数注入到 frozen actor 后 forward 正常。

### 10.5 centralized replay buffer

建议新增：

- `tests/test_stage2_replay_buffer.py`

测试点：

1. 能写入来自 EA 与 RL 的 transition。
2. sample 后 batch tensor shape 正确。
3. `joint_obs` / `joint_actions` / `team_reward` 对齐正确。
4. ring buffer 覆盖逻辑正确。

### 10.6 EA/RL 交互

建议新增：

- `tests/test_stage2_sync_mechanisms.py`

测试点：

1. `rl_to_ea` 能正确替换最差个体参数。
2. `elite_to_rl_soft_update` 结果正确。
3. 当拓扑被冻结时，参数软更新不触发 shape mismatch。
4. `sync_period` 调度正确。

### 10.7 batched population evaluator

建议新增：

- `tests/test_batched_population_evaluator.py`

测试点：

1. batched evaluator 和逐个 evaluator 输出一致。
2. 同代不同 candidate 共享相同 trial 初始状态。
3. 不同 generation seed 的初始状态不同。
4. 多 candidate 并行时 reward block 切分正确。

### 10.8 端到端 smoke test

建议新增：

- `tests/test_two_stage_smoke.py`

测试点：

1. 运行极小配置：
   - stage1 进化 2 代
   - freeze
   - critic 预训练 2 个 epoch
   - stage2 训练 2 个 iteration
2. 确认完整流程不报错。
3. 生成结果目录、checkpoint、日志、导出包。

## 11. 推荐实现顺序

按风险和依赖关系，建议这样落：

### 第 1 步

做 frozen actor package：

- checkpoint 提取
- actor 参数向量化
- rollout 一致性测试

### 第 2 步

补第一阶段精英轨迹采集：

- top-k 重新 rollout
- elite dataset 存盘
- RTG 计算

### 第 3 步

做 centralized critic 与 critic 预训练：

- 网络
- dataset loader
- pretrain script

### 第 4 步

接入 `sepCEM`，做固定拓扑 population：

- 参数采样
- 参数注入
- population evaluator

### 第 5 步

做 stage2 RL trainer：

- shared actor
- twin critic
- target net
- replay buffer

### 第 6 步

加双向同步：

- EA -> RL soft update
- RL -> EA inject worst replacement
- elite-only replay write policy

### 第 7 步

做端到端入口和完整日志：

- `train_two_stage.py`
- Hydra 配置
- checkpoint
- summary json

## 12. 最终建议

如果目标是“最稳地把当前项目推进到可运行的两阶段 EA+RL”，推荐第一版就按下面这个最小可行架构来做：

1. 第一阶段完全复用当前 `Experiment` + `better_Population` + `BatchGenomeEvaluator`。
2. 新增精英轨迹二次采集，不改主评估返回值。
3. 第 `N` 代导出 best genome package，冻结拓扑。
4. 第二阶段 actor 使用固定拓扑 recurrent actor，先保持当前共享 recurrent-state 语义。
5. critic 使用 centralized twin Q。
6. EA 支路直接使用 `sepCEM`。
7. RL 支路使用共享 actor 的 TD3 风格离策略更新。
8. 交互按 `docs/两阶段EA+RL协同.txt` 和 `EvoRainbow` 的三层融合来实现：
   - EA -> RL 经验共享
   - RL -> EA 策略注入
   - elite -> RL 参数软更新

这是最贴合你目标、也最贴合现有代码基础的路线。
