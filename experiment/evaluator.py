import numpy as np
import torch
from vmas.simulator.utils import save_video  # 视频保存工具
import time
import os
from dataclasses import dataclass


@dataclass
class EvalStats:
    """评估统计数据类
    
    用于存储单个基因组在多个环境上的详细评估统计，
    便于与强化学习的评估指标进行公平对比。
    """
    mean: float      # 均值（所有环境的平均回报）
    std: float       # 标准差
    max_val: float   # 最大值
    min_val: float   # 最小值
    median: float    # 中位数
    n_episodes: int  # 评估的环境/回合数


class GenomeEvaluator:
    def __init__(
        self,
        make_net,
        activate_net,
        batch_size=5,
        n_steps=None,
        make_env=None,
        render=False,
        save_render=False,
        video_dir=None,
        generation=-1,
        env_seed=None,
        scenario_name="",
    ):
        # 创建或使用提供的环境列表
        self.make_env = make_env
        self.env_seed = env_seed
        self.env = self._build_env()
        
        # 保存网络创建和激活函数
        self.make_net = make_net
        self.activate_net = activate_net
        
        # 保存批量大小和最大步数
        self.batch_size = batch_size
        self.n_steps = n_steps

        # 渲染相关
        self.render = render
        self.save_render = save_render
        self.video_dir = video_dir
        self.generation = generation
        self.scenario_name = scenario_name

        assert not (save_render and not render), "要保存视频，必须启用渲染（render=True）"

    def _build_env(self):
        if self.env_seed is None:
            return self.make_env()
        return self.make_env(seed=self.env_seed)

    def eval_genome(self, genome, config, debug=False) -> EvalStats:
        """
        评估单个基因组，计算详细统计数据，采用 stats.mean 作为适应度
        
        Args:
            genome: NEAT基因组
            config: NEAT配置
            debug: 是否启用调试输出
            
        Returns:
            EvalStats: 包含 mean, std, max_val, min_val, median, n_episodes
        """
        # 根据基因组创建神经网络
        net = self.make_net(genome, config, self.batch_size)

        # ========== 准备渲染和统计 ==========
        frame_list = []  # 存储渲染帧，用于创建GIF或视频
        init_time = time.time()  # 记录开始时间
        step = 0  # 步数计数器

        # 记录每个环境的累积奖励: (n_envs,)
        # 用于计算详细统计
        env_rewards = torch.zeros(self.batch_size, device=self.env.device)
        
        # 重置环境，获取初始观测,obs是一个列表，每个元素对应一个智能体的观测,每个观测的形状: (n_envs, obs_dim)
        if self.env_seed is None:
            obs = self.env.reset()
        else:
            obs = self.env.reset(seed=self.env_seed)
        
        # ========== 主循环 ==========
        for _ in range(self.n_steps):
            step += 1
            
            # ===== 为每个智能体计算动作 =====
            # 创建动作列表，初始化为None
            actions = [None] * len(obs)
            
            # 遍历所有智能体的观测
            for i in range(len(obs)):
                # 使用策略计算动作
                # obs[i]: 第i个智能体在所有环境中的观测 (n_envs, obs_dim)
                # u_range: 动作的允许范围，含义取决于动力学模型
                # dynamics_type: 动力学模型类型（如Holonomic, DiffDrive）
                # 返回: (n_envs, action_dim) 的动作张量
                #print(f"vmas-Obs for agent {i} at step {step}: {obs[i]}")
                dynamics_type = type(self.env.agents[i].dynamics).__name__
                actions[i] = self.activate_net(
                    net,
                    obs[i],
                    u_range=self.env.agents[i].u_range,
                    dynamics_type=dynamics_type
                )
            #print(f"Step {step}, neat—Actions: {actions}")

            # ===== 执行动作，获取下一步状态 =====
            # step()返回四个值：
            # - obs: 新观测列表
            # - rews: 奖励列表，每个元素形状 (n_envs,)
            # - dones: 终止标志，形状 (n_envs,)
            # - info: 额外信息字典列表
            obs, rews, dones, info = self.env.step(actions)
            
            # ===== 计算和累积奖励 =====
            # 将奖励列表堆叠成张量
            # rewards 形状: (n_envs, n_agents)
            rewards = torch.stack(rews, dim=1)
            #print(f"Step {step}, Rewards: {rewards}")
            
            # 计算每个环境的全局奖励（所有智能体的平均）
            # global_reward 形状: (n_envs,)
            global_reward = rewards.mean(dim=1)
            
            # 累积每个环境的奖励
            env_rewards += global_reward

            # =====  渲染（如果启用） =====
            if self.render:
                # 渲染当前帧
                # mode="rgb_array": 返回numpy数组而不是显示窗口
                # agent_index_focus=None: 相机不跟随特定智能体
                # visualize_when_rgb=True: 在RGB模式下显示可视化信息
                frame_list.append(
                    self.env.render(
                        mode="rgb_array",
                        agent_index_focus=None,
                        visualize_when_rgb=True,
                    )
                )

        # 计算总耗时
        total_time = time.time() - init_time
        
        # 计算详细统计量
        env_rewards_np = env_rewards.cpu().numpy()
        stats = EvalStats(
            mean=float(np.mean(env_rewards_np)),
            std=float(np.std(env_rewards_np)),
            max_val=float(np.max(env_rewards_np)),
            min_val=float(np.min(env_rewards_np)),
            median=float(np.median(env_rewards_np)),
            n_episodes=self.batch_size,
        )
        
        # 如果需要保存视频
        if self.render and self.save_render:
            video_basename = f"{self.scenario_name}_gen{self.generation}_{stats.mean:.2f}"
            
            old_cwd = os.getcwd()
            try:
                os.chdir(self.video_dir)        #save_video会在当前目录保存，需要切换到video_dir目录保存视频
                # fps = 1 / dt，dt是仿真时间步长
                fps = int(1 / self.env.scenario.world.dt)
                save_video(video_basename, frame_list, fps)
                video_path = os.path.join(self.video_dir, f"{video_basename}.mp4")
                if debug:
                    print(f"视频已保存至: {video_path}")
            finally:
                os.chdir(old_cwd)
        
        if debug:
            print(f"详细评估统计: mean={stats.mean:.4f}, std={stats.std:.4f}, "
                  f"max={stats.max_val:.4f}, min={stats.min_val:.4f}")
        
        return stats
