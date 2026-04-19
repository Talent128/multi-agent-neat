# Copyright (c) 2018 Uber Technologies, Inc.
# Modified for gradient-based fine-tuning support
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
可微分循环神经网络实现

支持反向传播的RecurrentNet版本，用于在NEAT进化中对精英个体进行梯度微调。


使用方式：
1. 从genome创建可微分网络
2. 使用强化学习算法（如REINFORCE/PPO）收集轨迹并计算梯度
3. 进行参数更新
4. 将更新后的参数同步回genome
"""

import torch
import torch.nn as nn
import numpy as np
from .activations import build_activation_groups, apply_activation_groups


def dense_from_coo(shape, conns, dtype=torch.float32, device=None):
    """从COO格式创建稠密矩阵（可微分版本）"""
    if device is None:
        device = torch.device('cpu')
    mat = torch.zeros(shape, dtype=dtype, device=device)
    idxs, weights = conns
    if len(idxs) == 0:
        return mat
    rows, cols = np.array(idxs).transpose()
    mat[torch.tensor(rows, device=device), torch.tensor(cols, device=device)] = torch.tensor(
        weights, dtype=dtype, device=device)
    return mat


class DifferentiableRecurrentNet(nn.Module):
    """
    可微分循环神经网络
    
    继承自nn.Module，支持反向传播。
    可用于在NEAT进化过程中对精英个体进行梯度微调。
    
    与原版RecurrentNet的主要区别：
    1. 继承nn.Module
    2. 权重使用nn.Parameter包装
    3. 移除了torch.no_grad()，支持梯度计算
    4. 添加了参数同步回genome的方法
    """
    
    def __init__(self, n_inputs, n_hidden, n_outputs,
                 input_to_hidden, hidden_to_hidden, output_to_hidden,
                 input_to_output, hidden_to_output, output_to_output,
                 hidden_activation_names, output_activation_names,
                 hidden_responses, output_responses,
                 hidden_biases, output_biases,
                 batch_size=1,
                 use_current_activs=False,
                 n_internal_steps=1,
                 dtype=torch.float32,
                 device=None,
                 connection_keys=None):  # 保存连接键用于同步回genome
        """
        初始化可微分递归神经网络
        
        Args:
            connection_keys: dict，保存每个权重矩阵对应的连接键，格式为：
                {
                    'input_to_hidden': [(conn_key, row_idx, col_idx), ...],
                    'hidden_to_hidden': [...],
                    ...
                }
        """
        super().__init__()
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        
        self.use_current_activs = use_current_activs
        self.n_internal_steps = n_internal_steps
        self.dtype = dtype
        
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.hidden_activation_names = tuple(hidden_activation_names)
        self.output_activation_names = tuple(output_activation_names)
        self.hidden_activation_groups = build_activation_groups(
            self.hidden_activation_names, device=device
        )
        self.output_activation_groups = build_activation_groups(
            self.output_activation_names, device=device
        )
        
        # 保存连接键映射（用于同步回genome）
        self.connection_keys = connection_keys or {}
        
        # 使用nn.Parameter包装权重矩阵，使其可训练
        if n_hidden > 0:
            self.input_to_hidden = nn.Parameter(dense_from_coo(
                (n_hidden, n_inputs), input_to_hidden, dtype=dtype, device=device))
            self.hidden_to_hidden = nn.Parameter(dense_from_coo(
                (n_hidden, n_hidden), hidden_to_hidden, dtype=dtype, device=device))
            self.output_to_hidden = nn.Parameter(dense_from_coo(
                (n_hidden, n_outputs), output_to_hidden, dtype=dtype, device=device))
            self.hidden_to_output = nn.Parameter(dense_from_coo(
                (n_outputs, n_hidden), hidden_to_output, dtype=dtype, device=device))
            self.hidden_responses = nn.Parameter(
                torch.tensor(hidden_responses, dtype=dtype, device=device))
            self.hidden_biases = nn.Parameter(
                torch.tensor(hidden_biases, dtype=dtype, device=device))
        
        self.input_to_output = nn.Parameter(dense_from_coo(
            (n_outputs, n_inputs), input_to_output, dtype=dtype, device=device))
        self.output_to_output = nn.Parameter(dense_from_coo(
            (n_outputs, n_outputs), output_to_output, dtype=dtype, device=device))
        
        self.output_responses = nn.Parameter(
            torch.tensor(output_responses, dtype=dtype, device=device))
        self.output_biases = nn.Parameter(
            torch.tensor(output_biases, dtype=dtype, device=device))
        
        self.batch_size = batch_size
        self.reset(batch_size)

    def apply_hidden_activations(self, values):
        return apply_activation_groups(values, self.hidden_activation_groups)

    def apply_output_activations(self, values):
        return apply_activation_groups(values, self.output_activation_groups)
    
    def reset(self, batch_size=1):
        """重置隐藏状态"""
        self.batch_size = batch_size
        if self.n_hidden > 0:
            self.activs = torch.zeros(
                batch_size, self.n_hidden, dtype=self.dtype, device=self.device)
        else:
            self.activs = None
        self.outputs = torch.zeros(
            batch_size, self.n_outputs, dtype=self.dtype, device=self.device)
    
    def forward(self, inputs):
        """
        前向传播（支持梯度计算）
        
        Args:
            inputs: (batch_size, n_inputs) 输入张量
            
        Returns:
            outputs: (batch_size, n_outputs) 输出张量
        """
        # 处理输入格式
        if isinstance(inputs, list):
            inputs = np.array(inputs)
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).to(dtype=self.dtype, device=self.device)
        elif isinstance(inputs, torch.Tensor):
            inputs = inputs.to(dtype=self.dtype, device=self.device)
        else:
            inputs = torch.tensor(inputs, dtype=self.dtype, device=self.device)
        
        # 分离隐藏状态以避免梯度累积（可选）
        activs_for_output = self.activs
        
        if self.n_hidden > 0:
            for _ in range(self.n_internal_steps):
                hidden_pre = self.hidden_responses * (
                    self.input_to_hidden.mm(inputs.t()).t() +
                    self.hidden_to_hidden.mm(self.activs.t()).t() +
                    self.output_to_hidden.mm(self.outputs.t()).t()
                ) + self.hidden_biases
                self.activs = self.apply_hidden_activations(hidden_pre)
            if self.use_current_activs:
                activs_for_output = self.activs
        
        output_inputs = (self.input_to_output.mm(inputs.t()).t() +
                         self.output_to_output.mm(self.outputs.t()).t())
        if self.n_hidden > 0:
            output_inputs += self.hidden_to_output.mm(activs_for_output.t()).t()
        
        output_pre = self.output_responses * output_inputs + self.output_biases
        self.outputs = self.apply_output_activations(output_pre)
        
        return self.outputs
    
    def activate(self, inputs):
        """兼容原版API的激活方法（无梯度）"""
        with torch.no_grad():
            return self.forward(inputs)
    
    def activate_with_grad(self, inputs):
        """带梯度的激活方法（用于训练）"""
        return self.forward(inputs)
    
    def sync_to_genome(self, genome):
        """
        将优化后的参数同步回genome
        
        这是拉马克进化的关键步骤：将学习到的权重写回基因组，
        使得进化可以继承这些改进。
        
        Args:
            genome: NEAT genome对象
        """
        with torch.no_grad():
            # 同步连接权重
            for conn_key, conn in genome.connections.items():
                if not conn.enabled:
                    continue
                    
                # 查找该连接在哪个权重矩阵中
                for matrix_name, key_mappings in self.connection_keys.items():
                    for stored_key, row_idx, col_idx in key_mappings:
                        if stored_key == conn_key:
                            # 从对应的参数矩阵获取更新后的权重
                            param = getattr(self, matrix_name)
                            conn.weight = param[row_idx, col_idx].item()
                            break
            
            # 同步节点参数（bias和response）
            from neat.graphs import required_for_output
            genome_config = genome  # 需要config来获取output_keys
            
            # 这里需要根据具体实现调整
            # 简化版：直接通过索引同步hidden和output节点

    @staticmethod
    def create(genome, config, batch_size=1,
               prune_empty=False, use_current_activs=False, n_internal_steps=1,
               device=None):
        """
        从NEAT基因组创建可微分递归神经网络
        
        与原版RecurrentNet.create相同的接口，但返回可微分版本。
        """
        from neat.graphs import required_for_output
        
        genome_config = config.genome_config
        required = required_for_output(
            genome_config.input_keys, genome_config.output_keys, genome.connections)
        if prune_empty:
            nonempty = {conn.key[1] for conn in genome.connections.values() if conn.enabled}.union(
                set(genome_config.input_keys))
        
        input_keys = list(genome_config.input_keys)
        hidden_keys = [k for k in genome.nodes.keys()
                       if k not in genome_config.output_keys]
        output_keys = list(genome_config.output_keys)
        hidden_activation_names = [
            genome.nodes[k].activation
            for k in hidden_keys
        ]
        output_activation_names = [
            genome.nodes[k].activation
            for k in output_keys
        ]
        
        hidden_responses = [genome.nodes[k].response for k in hidden_keys]
        output_responses = [genome.nodes[k].response for k in output_keys]
        
        hidden_biases = [genome.nodes[k].bias for k in hidden_keys]
        output_biases = [genome.nodes[k].bias for k in output_keys]
        
        if prune_empty:
            for i, key in enumerate(output_keys):
                if key not in nonempty:
                    output_biases[i] = 0.0
        
        n_inputs = len(input_keys)
        n_hidden = len(hidden_keys)
        n_outputs = len(output_keys)
        
        input_key_to_idx = {k: i for i, k in enumerate(input_keys)}
        hidden_key_to_idx = {k: i for i, k in enumerate(hidden_keys)}
        output_key_to_idx = {k: i for i, k in enumerate(output_keys)}
        
        def key_to_idx(key):
            if key in input_keys:
                return input_key_to_idx[key]
            elif key in hidden_keys:
                return hidden_key_to_idx[key]
            elif key in output_keys:
                return output_key_to_idx[key]
        
        input_to_hidden = ([], [])
        hidden_to_hidden = ([], [])
        output_to_hidden = ([], [])
        input_to_output = ([], [])
        hidden_to_output = ([], [])
        output_to_output = ([], [])
        
        # 保存连接键映射
        connection_keys = {
            'input_to_hidden': [],
            'hidden_to_hidden': [],
            'output_to_hidden': [],
            'input_to_output': [],
            'hidden_to_output': [],
            'output_to_output': [],
        }
        
        for conn in genome.connections.values():
            if not conn.enabled:
                continue
            
            i_key, o_key = conn.key
            if o_key not in required and i_key not in required:
                continue
            if prune_empty and i_key not in nonempty:
                continue
            
            i_idx = key_to_idx(i_key)
            o_idx = key_to_idx(o_key)
            
            if i_key in input_keys and o_key in hidden_keys:
                idxs, vals = input_to_hidden
                matrix_name = 'input_to_hidden'
            elif i_key in hidden_keys and o_key in hidden_keys:
                idxs, vals = hidden_to_hidden
                matrix_name = 'hidden_to_hidden'
            elif i_key in output_keys and o_key in hidden_keys:
                idxs, vals = output_to_hidden
                matrix_name = 'output_to_hidden'
            elif i_key in input_keys and o_key in output_keys:
                idxs, vals = input_to_output
                matrix_name = 'input_to_output'
            elif i_key in hidden_keys and o_key in output_keys:
                idxs, vals = hidden_to_output
                matrix_name = 'hidden_to_output'
            elif i_key in output_keys and o_key in output_keys:
                idxs, vals = output_to_output
                matrix_name = 'output_to_output'
            else:
                raise ValueError(
                    'Invalid connection from key {} to key {}'.format(i_key, o_key))
            
            idxs.append((o_idx, i_idx))
            vals.append(conn.weight)
            
            # 保存连接键映射
            connection_keys[matrix_name].append((conn.key, o_idx, i_idx))
        
        return DifferentiableRecurrentNet(
            n_inputs, n_hidden, n_outputs,
            input_to_hidden, hidden_to_hidden, output_to_hidden,
            input_to_output, hidden_to_output, output_to_output,
            hidden_activation_names, output_activation_names,
            hidden_responses, output_responses,
            hidden_biases, output_biases,
            batch_size=batch_size,
            use_current_activs=use_current_activs,
            n_internal_steps=n_internal_steps,
            device=device,
            connection_keys=connection_keys)


class ReinforcementLearningFinetuner:
    """
    强化学习微调器
    
    使用策略梯度方法对精英个体进行微调。
    支持REINFORCE和简化版PPO算法。
    
    使用方式：
    ```python
    finetuner = ReinforcementLearningFinetuner(
        learning_rate=1e-4,
        n_finetune_episodes=10,
        algorithm='reinforce'
    )
    
    # 在每代进化后对精英进行微调
    for elite_genome in elites:
        finetuner.finetune(elite_genome, config, env, make_net_fn)
    ```
    """
    
    def __init__(
        self,
        learning_rate: float = 1e-4,
        n_finetune_episodes: int = 10,
        gamma: float = 0.99,
        algorithm: str = 'reinforce',
        max_grad_norm: float = 0.5,
        entropy_coef: float = 0.01,
    ):
        """
        初始化强化学习微调器
        
        Args:
            learning_rate: 学习率
            n_finetune_episodes: 微调的episode数量
            gamma: 折扣因子
            algorithm: 算法类型 ('reinforce' 或 'ppo')
            max_grad_norm: 梯度裁剪阈值
            entropy_coef: 熵正则化系数
        """
        self.learning_rate = learning_rate
        self.n_finetune_episodes = n_finetune_episodes
        self.gamma = gamma
        self.algorithm = algorithm
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef
    
    def finetune(self, genome, config, env, make_net_fn=None, batch_size=1, 
                 activate_net_fn=None, sync_back=True):
        """
        对单个基因组进行强化学习微调
        
        Args:
            genome: NEAT genome对象
            config: NEAT配置
            env: 环境实例
            make_net_fn: 可选，自定义网络创建函数
            batch_size: 批量大小
            activate_net_fn: 可选，自定义激活函数
            sync_back: 是否将优化后的参数同步回genome
            
        Returns:
            float: 微调后的平均奖励
        """
        # 创建可微分网络
        net = DifferentiableRecurrentNet.create(
            genome, config, batch_size=batch_size)
        
        # 创建优化器
        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        
        total_rewards = []
        
        for episode in range(self.n_finetune_episodes):
            # 收集轨迹
            log_probs, rewards = self._collect_trajectory(
                net, env, activate_net_fn)
            
            # 计算returns
            returns = self._compute_returns(rewards)
            
            # 计算策略梯度损失
            if self.algorithm == 'reinforce':
                loss = self._reinforce_loss(log_probs, returns)
            elif self.algorithm == 'ppo':
                loss = self._ppo_loss(log_probs, returns)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(net.parameters(), self.max_grad_norm)
            
            optimizer.step()
            
            total_rewards.append(sum(rewards))
            
            # 重置网络状态
            net.reset(batch_size)
        
        # 将优化后的参数同步回genome
        if sync_back:
            net.sync_to_genome(genome)
        
        return np.mean(total_rewards)
    
    def _collect_trajectory(self, net, env, activate_net_fn=None):
        """收集一条轨迹"""
        log_probs = []
        rewards = []
        
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]  # gymnasium API
        
        done = False
        while not done:
            # 前向传播（带梯度）
            action_logits = net.activate_with_grad(obs)
            
            # 采样动作（这里简化处理，假设是连续动作）
            # 实际应用中需要根据动作空间类型调整
            action_dist = torch.distributions.Normal(action_logits, 0.5)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum()
            
            log_probs.append(log_prob)
            
            # 执行动作
            if hasattr(action, 'detach'):
                action_np = action.detach().cpu().numpy()
            else:
                action_np = action
            
            obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            
            rewards.append(reward)
        
        return log_probs, rewards
    
    def _compute_returns(self, rewards):
        """计算折扣回报"""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        # 标准化
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns
    
    def _reinforce_loss(self, log_probs, returns):
        """REINFORCE损失函数"""
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        return torch.stack(policy_loss).sum()
    
    def _ppo_loss(self, log_probs, returns):
        """简化版PPO损失（单步版本）"""
        # 完整PPO需要保存旧策略的log_prob，这里简化处理
        return self._reinforce_loss(log_probs, returns)


def finetune_elites_in_generation(
    population,
    config,
    make_env_fn,
    n_elites: int = 5,
    finetune_steps: int = 10,
    learning_rate: float = 1e-4,
):
    """
    在每代进化后对精英个体进行微调的辅助函数
    
    可以在NEAT的Reporter中调用，或者修改Population的run方法。
    
    Args:
        population: NEAT Population对象
        config: NEAT配置
        make_env_fn: 环境创建函数
        n_elites: 要微调的精英数量
        finetune_steps: 微调的episode数量
        learning_rate: 学习率
        
    Returns:
        list: 微调后的精英genome列表
    """
    # 获取当前代的精英
    genomes = list(population.population.items())
    genomes.sort(key=lambda x: x[1].fitness if x[1].fitness else float('-inf'), 
                 reverse=True)
    elites = [genome for _, genome in genomes[:n_elites]]
    
    # 创建微调器
    finetuner = ReinforcementLearningFinetuner(
        learning_rate=learning_rate,
        n_finetune_episodes=finetune_steps,
    )
    
    # 对每个精英进行微调
    env = make_env_fn()
    for elite in elites:
        original_fitness = elite.fitness
        new_fitness = finetuner.finetune(elite, config, env, sync_back=True)
        elite.fitness = new_fitness  # 更新适应度
        print(f"Elite finetuned: {original_fitness:.2f} -> {new_fitness:.2f}")
    
    return elites
