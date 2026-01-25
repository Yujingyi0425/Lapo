"""
Based on https://github.com/sfujim/BCQ
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
"""
             ┌──────────────────────┐
             │   Replay Buffer      │
             │  (state, action,     │
             │   reward, next_state)│
             └──────────┬───────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
  ┌─────▼──────┐  ┌─────▼──────┐  ┌─────▼──────┐
  │  ActorVAE  │  │   Actor    │  │  Critic    │
  │            │  │            │  │            │
  │ Encoder:   │  │ π(s)→z     │  │ Q(s,a)→q   │
  │ (s,a)→z    │  │            │  │ V(s)→v     │
  │            │  │ z∈ℝ^d_lat │  │            │
  │ Decoder:   │  └─────┬──────┘  └────────────┘
  │ (s,z)→a    │        │
  └────────────┘        │
        ▲               │
        │               │
  ┌─────┴───────────────▼─────┐
  │   Training Loop:          │
  │ 1. Train Critic           │
  │ 2. Train ActorVAE (adv)   │
  │ 3. Train Actor (-Q)       │
  │ 4. Soft Update Targets    │
  └───────────────────────────┘
"""
class Actor(nn.Module):      #动作策略网络，负责生成动作分布
    """
    输入:state: 状态张量。
    输出:潜在向量 (Latent Vector): 
    注意，尽管代码中函数名为 select_action 或变量名为 a，但这个网络输出的实际上是输入给 VAE 解码器的潜在变量z。
    约束: 输出经过 tanh 激活函数并乘以 max_latent_action，被限制在一定范围内。
    在VAE的隐空间中学习最优策略
    比直接在动作空间学习更稳定（避免离策分布问题）
    通过 actor_vae.decode() 将隐向量转换回动作空间
    """
    
    # 在隐空间中学习策略 ：state → latent_action
    def __init__(self, state_dim, latent_dim, max_action, device):
        super(Actor, self).__init__()
        hidden_size = (256, 256, 256)

        self.pi1 = nn.Linear(state_dim, hidden_size[0])
        self.pi2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.pi3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.pi4 = nn.Linear(hidden_size[2], latent_dim)    # 输出隐向量

        self.max_action = max_action            

    def forward(self, state):
        a = F.relu(self.pi1(state))
        a = F.relu(self.pi2(a))
        a = F.relu(self.pi3(a))
        a = self.pi4(a)
        a = self.max_action * torch.tanh(a)    # 约束在[-max_action, max_action]

        return a

class ActorVAE(nn.Module):
    """
    训练阶段 (Forward):
        输入:
            state: 状态张量 (Batch Size, State Dim)。
            action: 真实动作张量 (Batch Size, Action Dim)。
        输出:
            u: 重构的动作 (Reconstructed Action)。
            z: 采样的潜在变量 (Latent Variable)。
            mean: 潜在分布的均值。
            log_var: 潜在分布的对数方差。
    生成/解码阶段 (Decode):
        输入:
            state: 状态。
            z: 潜在变量 (Latent Vector)。如果未提供 z，则会从标准正态分布中采样或被截断。
        输出:
             a: 解码后的物理动作。

    条件变分自编码器
        Forward:
    state + action → [e1,e2,e3] → mean, log_var
                                   ↓ (重参数化)
                                z = μ + σ·ε  (ε~N(0,1))
                                   ↓
    state + z → [d1,d2,d3,d4] → reconstructed_action
    """
    def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
        super(ActorVAE, self).__init__()
        hidden_size = (256, 256, 256)
        # 编码器：(state + action) → 隐变量
        self.e1 = nn.Linear(state_dim + action_dim, hidden_size[0])
        self.e2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.e3 = nn.Linear(hidden_size[1], hidden_size[2])

        self.mean = nn.Linear(hidden_size[2], latent_dim)        # μ（均值）
        self.log_var = nn.Linear(hidden_size[2], latent_dim)     # log(σ²)（log方差）
         # 解码器：(state + 隐变量) → 动作
        self.d1 = nn.Linear(state_dim + latent_dim, hidden_size[0])
        self.d2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.d3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.d4 = nn.Linear(hidden_size[2], action_dim)  

        self.max_action = max_action
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.device = device

    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))
        z = F.relu(self.e3(z))

        mean = self.mean(z)
        log_var = self.log_var(z)

        #作用：学习离线数据中动作的分布，生成高质量的动作候选
        std = torch.exp(log_var/2)                   # σ = exp(log_var/2) = exp(log(σ²)/2)
        z = mean + std * torch.randn_like(std)       # 重参数化技巧
 
        u = self.decode(state, z)

        return u, z, mean, log_var

    def decode(self, state, z=None, clip=None):
        # When sampling from the VAE, the latent vector is clipped
        if z is None:
            clip = self.max_action
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-clip, clip)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        a = F.relu(self.d3(a))
        a = self.d4(a)
        return a

class Critic(nn.Module):
    """
    用于评估状态和动作的价值。在这个实现中，它同时包含 Q 函数和 V 函数。
        Q 函数部分 (forward / q1 方法):
            输入: state 和 action 的拼接向量。
            输出: q1, q2 (两个标量值，用于 Double Q-learning)。
        V 函数部分 (v 方法):
            输入: state。
            输出: v (状态价值标量)。
    实现了 Double Q-learning 结构 (q1, q2)，用于评估 (s, a) 的价值
    # 双Q函数（Twin Delayed DDPG风格）
    self.l1-l4: (state + action) → Q1 value
    self.l5-l8: (state + action) → Q2 value
    
    # 价值函数（状态价值）
    self.v1-v4: state → V value

    三个输出：
    Q1(s,a)：第一个Q函数
    Q2(s,a)：第二个Q函数（用于过度估计修正）
    V(s)：状态价值函数
    作用：评估状态-动作对的质量和状态的价值
    """ 
    def __init__(self, state_dim, action_dim, device):
        super(Critic, self).__init__()

        hidden_size = (256, 256, 256)

        self.l1 = nn.Linear(state_dim + action_dim, hidden_size[0])
        self.l2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.l3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.l4 = nn.Linear(hidden_size[2], 1)

        self.l5 = nn.Linear(state_dim + action_dim, hidden_size[0])
        self.l6 = nn.Linear(hidden_size[0], hidden_size[1])
        self.l7 = nn.Linear(hidden_size[1], hidden_size[2])
        self.l8 = nn.Linear(hidden_size[2], 1)

        self.v1 = nn.Linear(state_dim, hidden_size[0])
        self.v2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.v3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.v4 = nn.Linear(hidden_size[2], 1)

    def forward(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = (self.l4(q1))

        q2 = F.relu(self.l5(torch.cat([state, action], 1)))
        q2 = F.relu(self.l6(q2))
        q2 = F.relu(self.l7(q2))
        q2 = (self.l8(q2))
        return q1, q2

    def q1(self, state, action):
        q1 = F.relu(self.l1(torch.cat([state, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = (self.l4(q1))
        return q1

    def v(self, state):
        v = F.relu(self.v1(state))
        v = F.relu(self.v2(v))
        v = F.relu(self.v3(v))
        v = (self.v4(v))
        return v

class Latent(object):
    """
    Step 1: Critic训练
    # 计算目标Q值：r + γ·V(s')
    target_Q = reward + γ * V(s')
    # 优化Q函数和V函数最小化MSE损失
    critic_loss = MSE(Q1, target_Q) + MSE(Q2, target_Q) + MSE(V, target_V)
    Step 2: ActorVAE训练（加权重的CVAE）
        # 计算优势函数：A = Q(s') - V(s)
        adv = (Q(s') - V(s))
        
        # 优势加权：正优势用expectile，负优势用1-expectile
        weights = where(adv > 0, expectile, 1-expectile)
        
        # VAE损失：加权重建损失 + KL散度
        recon_loss = MSE(recons_action, action)
        KL_loss = -0.5·Σ(1 + log_var - μ² - exp(log_var))
        actor_vae_loss = (recon_loss + β·KL_loss) * weights
    Step 3: Actor训练（最大化Q值）
        # 在隐空间中最大化Q值
        latent_action = Actor(state)
        actor_action = VAE.decode(state, latent_action)
        actor_loss = -Q1(state, actor_action).mean()
    Step 4: 软更新目标网络
        # 软更新目标网络参数
         target_param = τ * param + (1 - τ) * target_param
    """ 
    def __init__(self, state_dim, action_dim, latent_dim, max_action, min_v, max_v, replay_buffer, 
                 device, discount=0.99, tau=0.005, vae_lr=1e-4, actor_lr=1e-4, critic_lr=5e-4, 
                 max_latent_action=1, expectile=0.8, kl_beta=1.0, 
                 no_noise=True, doubleq_min=0.8):

        self.device = torch.device(device)
        self.actor_vae = ActorVAE(state_dim, action_dim, latent_dim, max_latent_action, self.device).to(self.device)
        self.actor_vae_target = copy.deepcopy(self.actor_vae)
        self.actorvae_optimizer = torch.optim.Adam(self.actor_vae.parameters(), lr=vae_lr)

        self.actor = Actor(state_dim, latent_dim, max_latent_action, self.device).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim, self.device).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.latent_dim = latent_dim
        self.max_action = max_action
        self.max_latent_action = max_latent_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.tau_vae = tau

        self.expectile = expectile
        self.kl_beta = kl_beta
        self.no_noise = no_noise
        self.doubleq_min = doubleq_min

        self.replay_buffer = replay_buffer
        self.min_v, self.max_v = min_v, max_v 

    def select_action(self, state):
        with torch.no_grad():
            state = self.replay_buffer.normalize_state(state)
            state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

            latent_a = self.actor(state)
            action = self.actor_vae_target.decode(state, z=latent_a).cpu().data.numpy().flatten()
                            
            action = self.replay_buffer.unnormalize_action(action)
        return action

    def kl_loss(self, mu, log_var):
        KL_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1).view(-1, 1)
        return KL_loss

    def get_target_q(self, state, actor_net, critic_net, use_noise=False):
        latent_action = actor_net(state)
        if use_noise:
            latent_action += (torch.randn_like(latent_action) * 0.1).clamp(-0.2, 0.2)
        actor_action = self.actor_vae_target.decode(state, z=latent_action)
        
        target_q1, target_q2 = critic_net(state, actor_action)
        target_q = torch.min(target_q1, target_q2)*self.doubleq_min + torch.max(target_q1, target_q2)*(1-self.doubleq_min)

        return target_q

    def train(self, iterations, batch_size=100):
        for it in range(iterations):
            # Sample replay buffer / batch
            state, action, next_state, reward, not_done = self.replay_buffer.sample(batch_size)

            # Critic Training
            with torch.no_grad():
                next_target_v = self.critic.v(next_state)
                target_Q = reward + not_done * self.discount * next_target_v         
                target_v = self.get_target_q(state, self.actor_target, self.critic_target, use_noise=True)

            current_Q1, current_Q2 = self.critic(state, action)
            current_v = self.critic.v(state)

            v_loss = F.mse_loss(current_v, target_v.clamp(self.min_v, self.max_v))
            critic_loss_1 = F.mse_loss(current_Q1, target_Q)
            critic_loss_2 = F.mse_loss(current_Q2, target_Q)
            critic_loss = critic_loss_1 + critic_loss_2 + v_loss
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # compute adv and weight
            current_v = self.critic.v(state)

            next_q = self.get_target_q(next_state, self.actor_target, self.critic_target)
            q_action = reward + not_done * self.discount * next_q
            adv = (q_action - current_v)
            weights = torch.where(adv > 0, self.expectile, 1-self.expectile)

            # train weighted CVAE
            recons_action, z_sample, mu, log_var = self.actor_vae(state, action)

            recons_loss_ori = F.mse_loss(recons_action, action, reduction='none')
            recon_loss = torch.sum(recons_loss_ori, 1).view(-1, 1)
            KL_loss = self.kl_loss(mu, log_var)
            actor_vae_loss = (recon_loss + KL_loss*self.kl_beta)*weights.detach()
            
            actor_vae_loss = actor_vae_loss.mean()
            self.actorvae_optimizer.zero_grad()
            actor_vae_loss.backward()
            self.actorvae_optimizer.step()

            # train latent policy 
            latent_actor_action = self.actor(state)
            actor_action = self.actor_vae_target.decode(state, z=latent_actor_action)
            q_pi = self.critic.q1(state, actor_action)
        
            actor_loss = -q_pi.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update Target Networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor_vae.parameters(), self.actor_vae_target.parameters()):
                target_param.data.copy_(self.tau_vae * param.data + (1 - self.tau_vae) * target_param.data)

        assert (np.abs(np.mean(target_Q.cpu().data.numpy())) < 1e6)

    def save(self, filename, directory):
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
        torch.save(self.critic_optimizer.state_dict(), '%s/%s_critic_optimizer.pth' % (directory, filename))
        torch.save(self.critic_target.state_dict(), '%s/%s_critic_target.pth' % (directory, filename))

        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.actor_optimizer.state_dict(), '%s/%s_actor_optimizer.pth' % (directory, filename))
        torch.save(self.actor_target.state_dict(), '%s/%s_actor_target.pth' % (directory, filename))

        torch.save(self.actor_vae.state_dict(), '%s/%s_actor_vae.pth' % (directory, filename))
        torch.save(self.actorvae_optimizer.state_dict(), '%s/%s_actor_vae_optimizer.pth' % (directory, filename))
        torch.save(self.actor_vae_target.state_dict(), '%s/%s_actor_vae_target.pth' % (directory, filename))


    def load(self, filename, directory):
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
        self.critic_optimizer.load_state_dict(torch.load('%s/%s_critic_optimizer.pth' % (directory, filename)))
        self.critic_target.load_state_dict(torch.load('%s/%s_critic_target.pth' % (directory, filename)))

        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.actor_optimizer.load_state_dict(torch.load('%s/%s_actor_optimizer.pth' % (directory, filename)))
        self.actor_target.load_state_dict(torch.load('%s/%s_actor_target.pth' % (directory, filename)))

        self.actor_vae.load_state_dict(torch.load('%s/%s_actor_vae.pth' % (directory, filename)))
        self.actorvae_optimizer.load_state_dict(torch.load('%s/%s_actor_vae_optimizer.pth' % (directory, filename)))
        self.actor_vae_target.load_state_dict(torch.load('%s/%s_actor_vae_target.pth' % (directory, filename)))