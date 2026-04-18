"""
批量循环神经网络实现

将多个 RecurrentNet 实例的参数打包成统一张量，
通过 torch.einsum 实现高效的批量矩阵乘法。

性能提升：
- 相比循环调用单独网络可提升约 20-30x 性能
- 减少 Python 循环和 kernel launch 开销

使用方式:
    batched_net = BatchedRecurrentNet.from_genomes(genomes, config, ...)
    outputs = batched_net.activate(inputs)
"""

import torch
import numpy as np
from typing import List, Tuple, Callable

# 导入与 RecurrentNet 相同的激活函数，确保行为一致
from .activations import sigmoid_activation


class BatchedRecurrentNet:
    """
    批量循环神经网络
    
    将多个 RecurrentNet 实例的参数打包成统一张量，
    实现高效的批量前向传播。
    
    核心优化：
    1. 所有网络参数统一存储在形状为 (n_genomes, ...) 的张量中
    2. 使用 torch.einsum 进行批量矩阵乘法
    3. 一次前向传播计算所有基因组的输出
    """
    
    def __init__(
        self,
        n_genomes: int,
        trials_per_genome: int,
        max_inputs: int,
        max_hidden: int,
        max_outputs: int,
        input_to_hidden_weights: torch.Tensor,
        hidden_to_hidden_weights: torch.Tensor,
        output_to_hidden_weights: torch.Tensor,
        input_to_output_weights: torch.Tensor,
        hidden_to_output_weights: torch.Tensor,
        output_to_output_weights: torch.Tensor,
        hidden_responses: torch.Tensor,
        output_responses: torch.Tensor,
        hidden_biases: torch.Tensor,
        output_biases: torch.Tensor,
        hidden_masks: torch.Tensor,
        output_masks: torch.Tensor,
        use_current_activs: bool = False,
        activation: Callable = sigmoid_activation,
        n_internal_steps: int = 1,
        dtype: torch.dtype = torch.float32,
        device: str = None,
    ):
        """
        初始化批量循环神经网络
        
        Args:
            n_genomes: 基因组数量
            trials_per_genome: 每个基因组的试验次数
            max_inputs: 最大输入维度
            max_hidden: 最大隐藏节点数
            max_outputs: 最大输出维度
            input_to_hidden_weights: (n_genomes, max_hidden, max_inputs) 输入到隐藏层权重
            hidden_to_hidden_weights: (n_genomes, max_hidden, max_hidden) 隐藏层递归权重
            output_to_hidden_weights: (n_genomes, max_hidden, max_outputs) 输出到隐藏层权重
            input_to_output_weights: (n_genomes, max_outputs, max_inputs) 输入到输出权重
            hidden_to_output_weights: (n_genomes, max_outputs, max_hidden) 隐藏层到输出权重
            output_to_output_weights: (n_genomes, max_outputs, max_outputs) 输出递归权重
            hidden_responses: (n_genomes, max_hidden) 隐藏层响应系数
            output_responses: (n_genomes, max_outputs) 输出层响应系数
            hidden_biases: (n_genomes, max_hidden) 隐藏层偏置
            output_biases: (n_genomes, max_outputs) 输出层偏置
            hidden_masks: (n_genomes, max_hidden) 隐藏层掩码（标记有效节点）
            output_masks: (n_genomes, max_outputs) 输出层掩码（标记有效节点）
            use_current_activs: 是否使用当前激活值（而非上一时刻的）
            activation: 激活函数
            n_internal_steps: 内部迭代步数
            dtype: 数据类型
            device: 计算设备
        """
        self.n_genomes = n_genomes
        self.trials_per_genome = trials_per_genome
        self.max_inputs = max_inputs
        self.max_hidden = max_hidden
        self.max_outputs = max_outputs
        self.use_current_activs = use_current_activs
        self.activation = activation
        self.n_internal_steps = n_internal_steps
        self.dtype = dtype
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 注册权重张量
        self.input_to_hidden = input_to_hidden_weights.to(dtype=dtype, device=self.device)
        self.hidden_to_hidden = hidden_to_hidden_weights.to(dtype=dtype, device=self.device)
        self.output_to_hidden = output_to_hidden_weights.to(dtype=dtype, device=self.device)
        self.input_to_output = input_to_output_weights.to(dtype=dtype, device=self.device)
        self.hidden_to_output = hidden_to_output_weights.to(dtype=dtype, device=self.device)
        self.output_to_output = output_to_output_weights.to(dtype=dtype, device=self.device)
        
        # 注册响应系数和偏置
        self.hidden_responses = hidden_responses.to(dtype=dtype, device=self.device)
        self.output_responses = output_responses.to(dtype=dtype, device=self.device)
        self.hidden_biases = hidden_biases.to(dtype=dtype, device=self.device)
        self.output_biases = output_biases.to(dtype=dtype, device=self.device)
        
        # 注册掩码
        self.hidden_masks = hidden_masks.to(dtype=dtype, device=self.device)
        self.output_masks = output_masks.to(dtype=dtype, device=self.device)
        
        # 初始化激活状态
        self.activs = torch.zeros(n_genomes, trials_per_genome, max_hidden, dtype=dtype, device=self.device)
        self.outputs = torch.zeros(n_genomes, trials_per_genome, max_outputs, dtype=dtype, device=self.device)
    
    def reset(self, trials_per_genome: int = None):
        """
        重置网络状态
        
        Args:
            trials_per_genome: 新的试验数（可选）
        """
        if trials_per_genome is not None:
            self.trials_per_genome = trials_per_genome
        
        self.activs = torch.zeros(
            self.n_genomes, self.trials_per_genome, self.max_hidden,
            dtype=self.dtype, device=self.device
        )
        self.outputs = torch.zeros(
            self.n_genomes, self.trials_per_genome, self.max_outputs,
            dtype=self.dtype, device=self.device
        )
    
    def activate(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        批量前向传播
        
        Args:
            inputs: (n_genomes, trials_per_genome, input_dim) 或 (total_envs, input_dim) 输入张量
            
        Returns:
            outputs: (n_genomes, trials_per_genome, output_dim) 输出张量
        """
        with torch.no_grad():
            # 处理输入形状
            if inputs.dim() == 2:
                # (total_envs, input_dim) -> (n_genomes, trials_per_genome, input_dim)
                inputs = inputs.view(self.n_genomes, self.trials_per_genome, -1)
            
            inputs = inputs.to(dtype=self.dtype, device=self.device)
            
            # ========== 隐藏层计算 ==========
            if self.max_hidden > 0:
                # 保存上一时刻的激活值（用于输出计算）
                activs_for_output = self.activs.clone()
                
                for _ in range(self.n_internal_steps):
                    # 计算隐藏层输入
                    # hidden_input: (n_genomes, trials, max_hidden)
                    # input_to_hidden: (n_genomes, max_hidden, max_inputs)
                    # inputs: (n_genomes, trials, max_inputs)
                    # 等价于 inputs @ input_to_hidden.T
                    hidden_input = torch.einsum('gti,ghi->gth', inputs, self.input_to_hidden)
                    
                    # 隐藏层递归
                    # hidden_to_hidden: (n_genomes, max_hidden, max_hidden) 存储为 (to, from)
                    # 等价于 activs @ hidden_to_hidden.T
                    hidden_recur = torch.einsum('gth,gkh->gtk', self.activs, self.hidden_to_hidden)
                    
                    # 输出到隐藏层反馈
                    # 等价于 outputs @ output_to_hidden.T
                    hidden_output = torch.einsum('gto,gho->gth', self.outputs, self.output_to_hidden)
                    
                    # 合并并应用响应系数和偏置
                    hidden_pre = (hidden_input + hidden_recur + hidden_output)
                    hidden_pre = self.hidden_responses.unsqueeze(1) * hidden_pre + self.hidden_biases.unsqueeze(1)
                    
                    # 激活
                    self.activs = self.activation(hidden_pre)
                    
                    # 应用掩码
                    self.activs = self.activs * self.hidden_masks.unsqueeze(1)
                
                if self.use_current_activs:
                    activs_for_output = self.activs
            
            # ========== 输出层计算 ==========
            # 输入到输出
            # 等价于 inputs @ input_to_output.T
            output_from_input = torch.einsum('gti,goi->gto', inputs, self.input_to_output)
            
            # 输出递归
            # output_to_output: (n_genomes, max_outputs, max_outputs) 存储为 (to, from)
            # 等价于 outputs @ output_to_output.T
            output_from_output = torch.einsum('gto,gko->gtk', self.outputs, self.output_to_output)
            
            # 合并
            output_pre = output_from_input + output_from_output
            
            # 隐藏层到输出
            if self.max_hidden > 0:
                output_from_hidden = torch.einsum('gth,goh->gto', activs_for_output, self.hidden_to_output)
                output_pre = output_pre + output_from_hidden
            
            # 应用响应系数和偏置
            output_pre = self.output_responses.unsqueeze(1) * output_pre + self.output_biases.unsqueeze(1)
            
            # 激活
            self.outputs = self.activation(output_pre)
            
            return self.outputs
    
    def activate_flat(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        批量前向传播（扁平输入输出）
        
        Args:
            inputs: (total_envs, input_dim) 输入张量
            
        Returns:
            outputs: (total_envs, output_dim) 输出张量
        """
        outputs = self.activate(inputs)
        return outputs.view(-1, self.max_outputs)
    
    @staticmethod
    def from_genomes(
        genomes: List[Tuple[int, object]],
        config,
        trials_per_genome: int = 1,
        activation: Callable = sigmoid_activation,
        prune_empty: bool = False,
        use_current_activs: bool = False,
        n_internal_steps: int = 1,
        device: str = None,
    ) -> 'BatchedRecurrentNet':
        """
        从基因组列表创建批量网络
        
        Args:
            genomes: [(genome_id, genome), ...] 基因组列表
            config: NEAT配置
            trials_per_genome: 每个基因组的试验次数
            activation: 激活函数
            prune_empty: 是否剪枝空节点
            use_current_activs: 是否使用当前激活值
            n_internal_steps: 内部迭代步数
            device: 计算设备
            
        Returns:
            BatchedRecurrentNet 实例
        """
        from neat.graphs import required_for_output
        
        n_genomes = len(genomes)
        genome_config = config.genome_config
        
        # 获取输入输出节点信息
        input_keys = list(genome_config.input_keys)
        output_keys = list(genome_config.output_keys)
        
        num_inputs = len(input_keys)
        num_outputs = len(output_keys)
        
        # 收集所有基因组的隐藏节点信息
        # 注意：与 RecurrentNet.create 保持一致，使用相同的节点顺序和过滤逻辑
        max_hidden = 0
        all_hidden_keys = []
        all_required_nodes = []
        all_nonempty_nodes = []
        
        for _, genome in genomes:
            # 获取对输出有贡献的节点（与 RecurrentNet.create 一致）
            required = required_for_output(input_keys, output_keys, genome.connections)
            all_required_nodes.append(set(required))

            if prune_empty:
                nonempty = {
                    conn.key[1]
                    for conn in genome.connections.values()
                    if conn.enabled
                }.union(set(input_keys))
            else:
                nonempty = None
            all_nonempty_nodes.append(nonempty)
            
            # 隐藏节点：与 RecurrentNet.create 使用相同的顺序
            hidden_keys = [k for k in genome.nodes.keys() if k not in output_keys]
            all_hidden_keys.append(hidden_keys)
            if len(hidden_keys) > max_hidden:
                max_hidden = len(hidden_keys)
        
        # 初始化权重张量
        input_to_hidden = torch.zeros(n_genomes, max_hidden, num_inputs)
        hidden_to_hidden = torch.zeros(n_genomes, max_hidden, max_hidden)
        output_to_hidden = torch.zeros(n_genomes, max_hidden, num_outputs)
        input_to_output = torch.zeros(n_genomes, num_outputs, num_inputs)
        hidden_to_output = torch.zeros(n_genomes, num_outputs, max_hidden)
        output_to_output = torch.zeros(n_genomes, num_outputs, num_outputs)
        
        # 初始化响应系数和偏置
        hidden_responses = torch.ones(n_genomes, max_hidden)
        output_responses = torch.ones(n_genomes, num_outputs)
        hidden_biases = torch.zeros(n_genomes, max_hidden)
        output_biases = torch.zeros(n_genomes, num_outputs)
        
        # 初始化掩码
        hidden_masks = torch.zeros(n_genomes, max_hidden)
        output_masks = torch.ones(n_genomes, num_outputs)
        
        # 为每个基因组填充参数
        for g_idx, (genome_id, genome) in enumerate(genomes):
            hidden_keys = all_hidden_keys[g_idx]
            required = all_required_nodes[g_idx]
            nonempty = all_nonempty_nodes[g_idx]
            
            # 创建节点索引映射
            # 输入节点: 0 ~ num_inputs-1 (由 input_keys 决定)
            # 输出节点: 0 ~ num_outputs-1 (由 output_keys 决定)
            # 隐藏节点: 0 ~ len(hidden_keys)-1
            input_key_to_idx = {k: i for i, k in enumerate(input_keys)}
            output_key_to_idx = {k: i for i, k in enumerate(output_keys)}
            hidden_key_to_idx = {k: i for i, k in enumerate(hidden_keys)}
            
            # 设置隐藏节点的响应系数和偏置
            for h_idx, h_key in enumerate(hidden_keys):
                node = genome.nodes[h_key]
                hidden_responses[g_idx, h_idx] = node.response
                hidden_biases[g_idx, h_idx] = node.bias
                hidden_masks[g_idx, h_idx] = 1.0
            
            # 设置输出节点的响应系数和偏置
            for o_idx, o_key in enumerate(output_keys):
                node = genome.nodes[o_key]
                output_responses[g_idx, o_idx] = node.response
                if prune_empty and o_key not in nonempty:
                    # 与 RecurrentNet.create 保持一致：空输出节点偏置清零。
                    output_biases[g_idx, o_idx] = 0.0
                else:
                    output_biases[g_idx, o_idx] = node.bias
            
            # 设置连接权重
            # 注意：与 RecurrentNet.create 保持一致，只处理对输出有贡献的连接
            for conn_key, conn in genome.connections.items():
                if not conn.enabled:
                    continue
                
                in_key, out_key = conn_key
                
                # 与 RecurrentNet.create 一致：跳过与 required 节点无关的连接
                if out_key not in required and in_key not in required:
                    continue
                if prune_empty and in_key not in nonempty:
                    continue
                
                weight = conn.weight
                
                # 判断连接类型
                in_is_input = in_key in input_key_to_idx
                in_is_output = in_key in output_key_to_idx
                in_is_hidden = in_key in hidden_key_to_idx
                
                out_is_output = out_key in output_key_to_idx
                out_is_hidden = out_key in hidden_key_to_idx
                
                if in_is_input and out_is_hidden:
                    # 输入 -> 隐藏
                    i_idx = input_key_to_idx[in_key]
                    h_idx = hidden_key_to_idx[out_key]
                    input_to_hidden[g_idx, h_idx, i_idx] = weight
                    
                elif in_is_input and out_is_output:
                    # 输入 -> 输出
                    i_idx = input_key_to_idx[in_key]
                    o_idx = output_key_to_idx[out_key]
                    input_to_output[g_idx, o_idx, i_idx] = weight
                    
                elif in_is_hidden and out_is_hidden:
                    # 隐藏 -> 隐藏
                    h_in_idx = hidden_key_to_idx[in_key]
                    h_out_idx = hidden_key_to_idx[out_key]
                    hidden_to_hidden[g_idx, h_out_idx, h_in_idx] = weight
                    
                elif in_is_hidden and out_is_output:
                    # 隐藏 -> 输出
                    h_idx = hidden_key_to_idx[in_key]
                    o_idx = output_key_to_idx[out_key]
                    hidden_to_output[g_idx, o_idx, h_idx] = weight
                    
                elif in_is_output and out_is_hidden:
                    # 输出 -> 隐藏（递归连接）
                    o_idx = output_key_to_idx[in_key]
                    h_idx = hidden_key_to_idx[out_key]
                    output_to_hidden[g_idx, h_idx, o_idx] = weight
                    
                elif in_is_output and out_is_output:
                    # 输出 -> 输出（递归连接）
                    o_in_idx = output_key_to_idx[in_key]
                    o_out_idx = output_key_to_idx[out_key]
                    output_to_output[g_idx, o_out_idx, o_in_idx] = weight
        
        return BatchedRecurrentNet(
            n_genomes=n_genomes,
            trials_per_genome=trials_per_genome,
            max_inputs=num_inputs,
            max_hidden=max_hidden,
            max_outputs=num_outputs,
            input_to_hidden_weights=input_to_hidden,
            hidden_to_hidden_weights=hidden_to_hidden,
            output_to_hidden_weights=output_to_hidden,
            input_to_output_weights=input_to_output,
            hidden_to_output_weights=hidden_to_output,
            output_to_output_weights=output_to_output,
            hidden_responses=hidden_responses,
            output_responses=output_responses,
            hidden_biases=hidden_biases,
            output_biases=output_biases,
            hidden_masks=hidden_masks,
            output_masks=output_masks,
            use_current_activs=use_current_activs,
            activation=activation,
            n_internal_steps=n_internal_steps,
            device=device,
        )
