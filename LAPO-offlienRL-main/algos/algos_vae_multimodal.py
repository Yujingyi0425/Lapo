"""
多模态LAPO算法 - 支持图像和关节数据
基于 https://github.com/sfujim/BCQ
处理三路图像（左手腕、右手腕、全局）和16维关节数据
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torchvision import models


class ResNet18Encoder(nn.Module):
    """
    使用预训练的ResNet18作为图像编码器
    支持9通道输入（三张RGB图拼接）
    输入: (batch_size, 9, height, width) - 9通道拼接图像
    输出: (batch_size, feature_dim)
    """
    def __init__(self, output_dim=256, pretrained=True, in_channels=9):
        super(ResNet18Encoder, self).__init__()
        
        # 加载预训练的ResNet18
        resnet = models.resnet18(pretrained=pretrained)
        
        # 修改第一层卷积以支持9通道输入
        if in_channels != 3:
            # 原始第一层：3通道 → 64通道
            original_conv = resnet.conv1
            # 创建新的卷积层：9通道 → 64通道
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, 
                                     padding=3, bias=False)
            # 用平均方式初始化新增的权重
            if pretrained:
                weight = original_conv.weight.data
                resnet.conv1.weight.data[:, :3, :, :] = weight
                resnet.conv1.weight.data[:, 3:6, :, :] = weight
                resnet.conv1.weight.data[:, 6:9, :, :] = weight
        
        # 移除最后的全连接层
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        # resnet18的最后一层输出为512维特征
        self.fc = nn.Linear(512, output_dim)
    
    def forward(self, x):
        """
        输入: [batch, 9, H, W] - 拼接的9通道图像
        输出: [batch, output_dim] - 特征向量
        """
        x = self.resnet(x)
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc(x))
        return x


class ImageJointEncoder(nn.Module):
    """
    融合编码器：编码三张拼接的图像和关节数据
    
    图像处理流程:
        [Left RGB]   [Right RGB]  [Global RGB]
           |            |            |
           └────┬────────┴────┬───────┘
                │   拼接到9通道 │
                ▼              
            [9-channel image]
                │
            ResNet18(单个)
                │
            [image_feature_dim]
    
    输入: 
        - left_img, right_img, global_img: 各为 (batch_size, 3, H, W) RGB图像
        - joint: (batch_size, joint_dim=16) 关节数据
    输出: 
        - fused_feat: (batch_size, fusion_dim=256) 融合特征
    """
    def __init__(self, joint_dim=16, image_feature_dim=256, fusion_dim=256):
        super(ImageJointEncoder, self).__init__()
        
        # 单个ResNet18编码器处理9通道拼接图像
        self.image_encoder = ResNet18Encoder(output_dim=image_feature_dim, pretrained=True, in_channels=9)
        
        # 关节特征网络 (16维 → image_feature_dim维)
        self.joint_fc = nn.Sequential(
            nn.Linear(joint_dim, 128),
            nn.ReLU(),
            nn.Linear(128, image_feature_dim)
        )
        
        # 融合层：图像特征 + 关节特征 → 融合特征
        total_feature_dim = image_feature_dim * 2  # 图像 + 关节
        self.fusion_fc = nn.Sequential(
            nn.Linear(total_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, fusion_dim),
            nn.ReLU()
        )
    
    def forward(self, left_img, right_img, global_img, joint):
        """
        输入:
            left_img: [batch, 3, 224, 224]  (已在数据加载时resize)
            right_img: [batch, 3, 224, 224]
            global_img: [batch, 3, 224, 224]
            joint: [batch, 16]
        
        输出:
            fused_feat: [batch, 256]
        """
        # 1. 拼接三张图像为9通道
        # 沿着通道维度(dim=1)拼接：[3] + [3] + [3] = [9]
        concatenated_img = torch.cat([left_img, right_img, global_img], dim=1)
        
        # 2. 编码拼接的9通道图像
        image_feat = self.image_encoder(concatenated_img)  # [batch, image_feature_dim]
        
        # 3. 编码关节数据
        joint_feat = self.joint_fc(joint)  # [batch, image_feature_dim]
        
        # 4. 融合图像特征和关节特征
        fused = torch.cat([image_feat, joint_feat], dim=1)  # [batch, 2*image_feature_dim]
        fused_feat = self.fusion_fc(fused)  # [batch, fusion_dim]
        
        return fused_feat


class Actor(nn.Module):
    """
    动作策略网络 - 在隐空间中学习策略
    输入: 融合的观察特征
    输出: 潜在向量 (Latent Vector)
    """
    def __init__(self, obs_feature_dim, latent_dim, max_action, device):
        super(Actor, self).__init__()
        hidden_size = (256, 256, 256)
        
        self.pi1 = nn.Linear(obs_feature_dim, hidden_size[0])
        self.pi2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.pi3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.pi4 = nn.Linear(hidden_size[2], latent_dim)
        
        self.max_action = max_action
    
    def forward(self, obs_feature):
        a = F.relu(self.pi1(obs_feature))
        a = F.relu(self.pi2(a))
        a = F.relu(self.pi3(a))
        a = self.pi4(a)
        a = self.max_action * torch.tanh(a)
        
        return a


class ActorVAE(nn.Module):
    """
    条件变分自编码器 (CVAE)
    编码: (obs_feature + action) → (mean, log_var)
    采样: z = mean + std * epsilon
    解码: (obs_feature + z) → reconstructed_action
    """
    def __init__(self, obs_feature_dim, action_dim, latent_dim, max_action, device):
        super(ActorVAE, self).__init__()
        hidden_size = (256, 256, 256)
        
        # 编码器
        self.e1 = nn.Linear(obs_feature_dim + action_dim, hidden_size[0])
        self.e2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.e3 = nn.Linear(hidden_size[1], hidden_size[2])
        
        self.mean = nn.Linear(hidden_size[2], latent_dim)
        self.log_var = nn.Linear(hidden_size[2], latent_dim)
        
        # 解码器
        self.d1 = nn.Linear(obs_feature_dim + latent_dim, hidden_size[0])
        self.d2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.d3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.d4 = nn.Linear(hidden_size[2], action_dim)
        
        self.max_action = max_action
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.device = device
    
    def forward(self, obs_feature, action):
        # 编码
        z = F.relu(self.e1(torch.cat([obs_feature, action], 1)))
        z = F.relu(self.e2(z))
        z = F.relu(self.e3(z))
        
        mean = self.mean(z)
        log_var = self.log_var(z)
        
        # 重参数化
        std = torch.exp(log_var / 2)
        z_sample = mean + std * torch.randn_like(std)
        
        # 解码
        u = self.decode(obs_feature, z_sample)
        
        return u, z_sample, mean, log_var
    
    def decode(self, obs_feature, z=None, clip=None):
        if z is None:
            clip = self.max_action
            z = torch.randn((obs_feature.shape[0], self.latent_dim)).to(self.device).clamp(-clip, clip)
        
        a = F.relu(self.d1(torch.cat([obs_feature, z], 1)))
        a = F.relu(self.d2(a))
        a = F.relu(self.d3(a))
        a = self.d4(a)
        
        return a


class Critic(nn.Module):
    """
    评估网络 - 包含双Q函数和V函数
    Q网络: (obs_feature + action) → Q值
    V网络: (obs_feature) → V值
    """
    def __init__(self, obs_feature_dim, action_dim, device):
        super(Critic, self).__init__()
        hidden_size = (256, 256, 256)
        
        # Q函数1
        self.l1 = nn.Linear(obs_feature_dim + action_dim, hidden_size[0])
        self.l2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.l3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.l4 = nn.Linear(hidden_size[2], 1)
        
        # Q函数2
        self.l5 = nn.Linear(obs_feature_dim + action_dim, hidden_size[0])
        self.l6 = nn.Linear(hidden_size[0], hidden_size[1])
        self.l7 = nn.Linear(hidden_size[1], hidden_size[2])
        self.l8 = nn.Linear(hidden_size[2], 1)
        
        # V函数
        self.v1 = nn.Linear(obs_feature_dim, hidden_size[0])
        self.v2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.v3 = nn.Linear(hidden_size[1], hidden_size[2])
        self.v4 = nn.Linear(hidden_size[2], 1)
    
    def forward(self, obs_feature, action):
        q1 = F.relu(self.l1(torch.cat([obs_feature, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = self.l4(q1)
        
        q2 = F.relu(self.l5(torch.cat([obs_feature, action], 1)))
        q2 = F.relu(self.l6(q2))
        q2 = F.relu(self.l7(q2))
        q2 = self.l8(q2)
        
        return q1, q2
    
    def q1(self, obs_feature, action):
        q1 = F.relu(self.l1(torch.cat([obs_feature, action], 1)))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l3(q1))
        q1 = self.l4(q1)
        return q1
    
    def v(self, obs_feature):
        v = F.relu(self.v1(obs_feature))
        v = F.relu(self.v2(v))
        v = F.relu(self.v3(v))
        v = self.v4(v)
        return v



class MultimodalLatent(object):
    """
    多模态LAPO算法主类
    处理图像和关节数据的离线强化学习算法
    """
    def __init__(self, action_dim, latent_dim, max_action, min_v, max_v, replay_buffer,
                 device, joint_dim=16, obs_feature_dim=256, discount=0.99, tau=0.005,
                 vae_lr=1e-4, actor_lr=1e-4, critic_lr=5e-4, max_latent_action=1,
                 expectile=0.8, kl_beta=1.0, no_noise=True, doubleq_min=0.8):
        
        self.device = torch.device(device)
        self.joint_dim = joint_dim
        self.obs_feature_dim = obs_feature_dim
        
        # 1. 初始化观察编码器 (Main & Target)
        self.obs_encoder = ImageJointEncoder(
            joint_dim=joint_dim,
            image_feature_dim=128,
            fusion_dim=obs_feature_dim
        ).to(self.device)
        
        self.obs_encoder_target = copy.deepcopy(self.obs_encoder)
        
        # 2. 初始化VAE
        self.actor_vae = ActorVAE(obs_feature_dim, action_dim, latent_dim, max_latent_action, self.device).to(self.device)
        self.actor_vae_target = copy.deepcopy(self.actor_vae)
        self.actorvae_optimizer = torch.optim.Adam(self.actor_vae.parameters(), lr=vae_lr)
        
        # 3. 初始化Actor
        self.actor = Actor(obs_feature_dim, latent_dim, max_latent_action, self.device).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # 4. 初始化Critic
        self.critic = Critic(obs_feature_dim, action_dim, self.device).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        
        # Critic优化器包含：Critic参数 + 编码器参数
        # 这是为了让Critic的梯度能更新CNN提取特征
        self.critic_optimizer = torch.optim.Adam(
            [
                {'params': self.critic.parameters()},
                {'params': self.obs_encoder.parameters()}
            ], 
            lr=critic_lr
        )
        
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

    def _set_train_mode(self):
        """将所有网络设置为训练模式 (启用BatchNorm update/Dropout)"""
        self.obs_encoder.train()
        self.actor.train()
        self.actor_vae.train()
        self.critic.train()

    def _set_eval_mode(self):
        """将所有网络设置为评估模式 (锁定BatchNorm/禁用Dropout)"""
        self.obs_encoder.eval()
        self.actor.eval()
        self.actor_vae.eval()
        self.critic.eval()
        # target网络永远是eval模式
        self.obs_encoder_target.eval()
        self.actor_target.eval()
        self.actor_vae_target.eval()
        self.critic_target.eval()

    def encode_observation(self, left_img, right_img, global_img, joint, use_target=False):
        encoder = self.obs_encoder_target if use_target else self.obs_encoder
        obs_feature = encoder(left_img, right_img, global_img, joint)
        return obs_feature

    def select_action(self, left_img, right_img, global_img, joint):
        # [关键 1] 推理时必须切换到 eval 模式
        # 否则 ResNet 的 BatchNorm 层在 batch_size=1 时会因为计算不出方差而报错，
        # 或者使用错误的统计量导致输出动作剧烈抖动。
        self._set_eval_mode()
        
        with torch.no_grad():
            # [关键 2] 图像预处理：uint8 [0-255] -> float [0-1]
            # 是物理意义上的缩放，必须保留。
            left_img = torch.FloatTensor(left_img).unsqueeze(0).to(self.device) / 255.0
            right_img = torch.FloatTensor(right_img).unsqueeze(0).to(self.device) / 255.0
            global_img = torch.FloatTensor(global_img).unsqueeze(0).to(self.device) / 255.0
            
            # [关键 3] 关节数据处理：只转 Tensor，不归一化
            joint = torch.FloatTensor(joint).unsqueeze(0).to(self.device)
        
            # 1. 编码观察 (ResNet + MLP)
            obs_feature = self.encode_observation(left_img, right_img, global_img, joint)
            
            # 2. Actor 生成潜在变量 z
            latent_a = self.actor(obs_feature)
            
            # 3. VAE 解码器生成动作
            # 使用 target 网络解码通常更稳定
            action = self.actor_vae_target.decode(obs_feature, z=latent_a).cpu().data.numpy().flatten()
            
            # [关键 4] 移除反归一化
            # 因为我们在 utils 中已经禁用了 unnormalize_action，
            # 或者根据您的要求，直接输出网络预测的原始数值。
            # action = self.replay_buffer.unnormalize_action(action) 
            
        # [关键 5] 恢复训练模式
        # 如果这是一个在线交互过程，或者稍后还要继续 train，必须切回来。
        self._set_train_mode() 
        
        return action
    
    def kl_loss(self, mu, log_var):
        KL_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1).view(-1, 1)
        return KL_loss

    def get_target_q(self, obs_feature, actor_net, critic_net, use_noise=False):
        latent_action = actor_net(obs_feature)
        if use_noise:
            latent_action += (torch.randn_like(latent_action) * 0.1).clamp(-0.2, 0.2)
        actor_action = self.actor_vae_target.decode(obs_feature, z=latent_action)
        
        target_q1, target_q2 = critic_net(obs_feature, actor_action)
        target_q = torch.min(target_q1, target_q2) * self.doubleq_min + torch.max(target_q1, target_q2) * (1 - self.doubleq_min)
        return target_q
    
    def train(self, iterations, batch_size=100):
        # [关键] 确保进入训练循环时处于训练模式
        self._set_train_mode()
        
        for it in range(iterations):
            # 采样
            left_imgs, right_imgs, global_imgs, joints, actions, next_left_imgs, next_right_imgs, next_global_imgs, next_joints, rewards, not_done = self.replay_buffer.sample(batch_size)
            
            # 归一化图像 (0-255 -> 0-1)
            left_imgs = left_imgs / 255.0
            right_imgs = right_imgs / 255.0
            global_imgs = global_imgs / 255.0
            next_left_imgs = next_left_imgs / 255.0
            next_right_imgs = next_right_imgs / 255.0
            next_global_imgs = next_global_imgs / 255.0
            
            # 1. 编码观察
            obs_features = self.encode_observation(left_imgs, right_imgs, global_imgs, joints, use_target=False)
            
            with torch.no_grad():
                next_obs_features = self.encode_observation(next_left_imgs, next_right_imgs, next_global_imgs, next_joints, use_target=True)
            
            # ============ Critic训练 ============
            with torch.no_grad():
                next_target_v = self.critic.v(next_obs_features)
                target_Q = rewards + not_done * self.discount * next_target_v
                # 使用 detached features 计算 V-target，避免Actor影响Encoder
                target_v = self.get_target_q(obs_features.detach(), self.actor_target, self.critic_target, use_noise=True)
            
            current_Q1, current_Q2 = self.critic(obs_features, actions)
            current_v = self.critic.v(obs_features)
            
            v_loss = F.mse_loss(current_v, target_v.clamp(self.min_v, self.max_v))
            critic_loss_1 = F.mse_loss(current_Q1, target_Q)
            critic_loss_2 = F.mse_loss(current_Q2, target_Q)
            critic_loss = critic_loss_1 + critic_loss_2 + v_loss
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward() 
            self.critic_optimizer.step()
            
            # [关键] Detach特征用于后续网络
            obs_features_detached = obs_features.detach()
            
            # ============ ActorVAE训练 ============
            current_v = self.critic.v(obs_features_detached)
            next_q = self.get_target_q(next_obs_features, self.actor_target, self.critic_target)
            q_action = rewards + not_done * self.discount * next_q
            adv = (q_action - current_v)
            weights = torch.where(adv > 0, self.expectile, 1 - self.expectile)
            
            recons_action, z_sample, mu, log_var = self.actor_vae(obs_features_detached, actions)
            recons_loss_ori = F.mse_loss(recons_action, actions, reduction='none')
            recon_loss = torch.sum(recons_loss_ori, 1).view(-1, 1)
            KL_loss = self.kl_loss(mu, log_var)
            actor_vae_loss = (recon_loss + KL_loss * self.kl_beta) * weights.detach()
            
            actor_vae_loss = actor_vae_loss.mean()
            self.actorvae_optimizer.zero_grad()
            actor_vae_loss.backward()
            self.actorvae_optimizer.step()
            
            # ============ Actor训练 ============
            latent_actor_action = self.actor(obs_features_detached)
            actor_action = self.actor_vae_target.decode(obs_features_detached, z=latent_actor_action)
            q_pi = self.critic.q1(obs_features_detached, actor_action)
            
            actor_loss = -q_pi.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # ============ 软更新 ============
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor_vae.parameters(), self.actor_vae_target.parameters()):
                target_param.data.copy_(self.tau_vae * param.data + (1 - self.tau_vae) * target_param.data)
            # 更新目标编码器
            for param, target_param in zip(self.obs_encoder.parameters(), self.obs_encoder_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
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
        
        torch.save(self.obs_encoder.state_dict(), '%s/%s_obs_encoder.pth' % (directory, filename))
        torch.save(self.obs_encoder_target.state_dict(), '%s/%s_obs_encoder_target.pth' % (directory, filename))
    
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
        
        # 加载编码器
        self.obs_encoder.load_state_dict(torch.load('%s/%s_obs_encoder.pth' % (directory, filename)))
        
        # 尝试加载目标编码器，如果旧模型没有，则复制主编码器
        try:
            self.obs_encoder_target.load_state_dict(torch.load('%s/%s_obs_encoder_target.pth' % (directory, filename)))
        except FileNotFoundError:
            print("Warning: No target encoder found, copying from main encoder.")
            self.obs_encoder_target = copy.deepcopy(self.obs_encoder)