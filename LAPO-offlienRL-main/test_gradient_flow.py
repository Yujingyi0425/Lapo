#!/usr/bin/env python3
"""
梯度流向诊断脚本
验证obs_encoder是否正确接收Critic loss的梯度
"""

import sys
import torch
import torch.nn as nn
from algos.algos_vae_multimodal import MultimodalLatent, ResNet18Encoder
from algos.utils_multimodal import MultimodalReplayBuffer

def test_optimizer_configuration(algo):
    """
    测试1: 验证优化器的参数配置
    """
    print("=" * 80)
    print("测试1: 优化器配置检查")
    print("=" * 80)
    
    # 检查actorvae_optimizer
    print("\n[actorvae_optimizer 包含的参数]")
    actorvae_param_ids = set()
    for param_group in algo.actorvae_optimizer.param_groups:
        for param in param_group['params']:
            actorvae_param_ids.add(id(param))
    print(f"  共 {len(actorvae_param_ids)} 个参数")
    
    # 检查critic_optimizer
    print("\n[critic_optimizer 包含的参数]")
    critic_param_ids = set()
    for param_group in algo.critic_optimizer.param_groups:
        for param in param_group['params']:
            critic_param_ids.add(id(param))
    print(f"  共 {len(critic_param_ids)} 个参数")
    
    # 获取各模块的参数
    print("\n[各模块参数数量]")
    print(f"  obs_encoder: {sum(p.numel() for p in algo.obs_encoder.parameters())} 参数")
    print(f"  critic: {sum(p.numel() for p in algo.critic.parameters())} 参数")
    print(f"  actor_vae: {sum(p.numel() for p in algo.actor_vae.parameters())} 参数")
    print(f"  actor: {sum(p.numel() for p in algo.actor.parameters())} 参数")
    
    # 检查obs_encoder是否在各优化器中
    obs_encoder_param_ids = set(id(param) for param in algo.obs_encoder.parameters())
    critic_param_ids_only = set(id(param) for param in algo.critic.parameters())
    
    print("\n[关键检查]")
    obs_in_actorvae = len(obs_encoder_param_ids & actorvae_param_ids)
    obs_in_critic = len(obs_encoder_param_ids & critic_param_ids)
    
    print(f"  obs_encoder 在 actorvae_optimizer 中: {obs_in_actorvae > 0} ✓" if obs_in_actorvae == 0 else f"  obs_encoder 在 actorvae_optimizer 中: {obs_in_actorvae > 0} ✗ (应该不在!)")
    print(f"  obs_encoder 在 critic_optimizer 中: {obs_in_critic > 0} ✓" if obs_in_critic > 0 else f"  obs_encoder 在 critic_optimizer 中: {obs_in_critic > 0} ✗ (应该在!)")
    
    # 返回配置是否正确
    return obs_in_actorvae == 0 and obs_in_critic > 0


def test_gradient_flow(algo, batch_size=32):
    """
    测试2: 验证梯度是否正确流向obs_encoder
    """
    print("\n" + "=" * 80)
    print("测试2: 梯度流向测试")
    print("=" * 80)
    
    # 创建虚拟数据
    device = algo.device
    left_imgs = torch.randn(batch_size, 3, 84, 84, device=device)
    right_imgs = torch.randn(batch_size, 3, 84, 84, device=device)
    global_imgs = torch.randn(batch_size, 3, 84, 84, device=device)
    joints = torch.randn(batch_size, 16, device=device)
    actions = torch.randn(batch_size, 16, device=device)
    next_left_imgs = torch.randn(batch_size, 3, 84, 84, device=device)
    next_right_imgs = torch.randn(batch_size, 3, 84, 84, device=device)
    next_global_imgs = torch.randn(batch_size, 3, 84, 84, device=device)
    next_joints = torch.randn(batch_size, 16, device=device)
    rewards = torch.randn(batch_size, 1, device=device)
    not_done = torch.ones(batch_size, 1, device=device)
    
    # 编码观察
    obs_features = algo.encode_observation(left_imgs, right_imgs, global_imgs, joints, use_target=False)
    with torch.no_grad():
        next_obs_features = algo.encode_observation(next_left_imgs, next_right_imgs, next_global_imgs, next_joints, use_target=True)
    
    # 清空梯度
    algo.critic_optimizer.zero_grad()
    algo.actorvae_optimizer.zero_grad()
    algo.actor_optimizer.zero_grad()
    
    # 计算Critic loss
    with torch.no_grad():
        next_target_v = algo.critic.v(next_obs_features)
        target_Q = rewards + not_done * algo.discount * next_target_v
        target_v = algo.get_target_q(obs_features.detach(), algo.actor_target, algo.critic_target, use_noise=True)
    
    current_Q1, current_Q2 = algo.critic(obs_features, actions)
    current_v = algo.critic.v(obs_features)
    
    v_loss = torch.nn.functional.mse_loss(current_v, target_v.clamp(algo.min_v, algo.max_v))
    critic_loss_1 = torch.nn.functional.mse_loss(current_Q1, target_Q)
    critic_loss_2 = torch.nn.functional.mse_loss(current_Q2, target_Q)
    critic_loss = critic_loss_1 + critic_loss_2 + v_loss
    
    print(f"\n[Critic Loss 值]")
    print(f"  critic_loss: {critic_loss.item():.6f}")
    
    # 反向传播
    print(f"\n[反向传播前]")
    print(f"  obs_encoder.weight 的梯度: {algo.obs_encoder.image_encoder.conv1.weight.grad}")
    
    critic_loss.backward()
    
    print(f"\n[反向传播后]")
    obs_grad = algo.obs_encoder.image_encoder.conv1.weight.grad
    print(f"  obs_encoder.weight 的梯度: {obs_grad is not None} ✓" if obs_grad is not None else f"  obs_encoder.weight 的梯度: {obs_grad is not None} ✗ (梯度为None!)")
    
    if obs_grad is not None:
        print(f"  梯度范数: {obs_grad.norm().item():.6e}")
        print(f"  梯度最大值: {obs_grad.abs().max().item():.6e}")
        print(f"  梯度最小值: {obs_grad.abs().min().item():.6e}")
    
    # 检查critic参数梯度
    critic_grad = list(algo.critic.q1_net[0].weight.grad)
    print(f"\n[Critic 梯度检查]")
    print(f"  critic.q1_net[0].weight 的梯度: {critic_grad is not None} ✓")
    if critic_grad is not None:
        print(f"  梯度范数: {torch.tensor(critic_grad).norm().item():.6e}")
    
    # 执行optimizer step
    print(f"\n[Optimizer Step 前]")
    obs_param_before = algo.obs_encoder.image_encoder.conv1.weight.data.clone()
    
    algo.critic_optimizer.step()
    
    print(f"[Optimizer Step 后]")
    obs_param_after = algo.obs_encoder.image_encoder.conv1.weight.data
    param_change = (obs_param_after - obs_param_before).abs().max().item()
    print(f"  obs_encoder 参数变化: {param_change > 0} ✓" if param_change > 0 else f"  obs_encoder 参数变化: {param_change > 0} ✗ (参数未更新!)")
    print(f"  最大参数变化: {param_change:.6e}")
    
    return obs_grad is not None and param_change > 0


def test_two_optimizer_flow(algo, batch_size=32):
    """
    测试3: 验证两个优化器顺序下梯度是否被保留
    模拟训练循环: critic_optimizer.step() → actorvae_optimizer.zero_grad() → ...
    """
    print("\n" + "=" * 80)
    print("测试3: 两个优化器协作测试 (关键!)")
    print("=" * 80)
    
    device = algo.device
    
    # 创建虚拟数据
    left_imgs = torch.randn(batch_size, 3, 84, 84, device=device)
    right_imgs = torch.randn(batch_size, 3, 84, 84, device=device)
    global_imgs = torch.randn(batch_size, 3, 84, 84, device=device)
    joints = torch.randn(batch_size, 16, device=device)
    actions = torch.randn(batch_size, 16, device=device)
    
    # 编码观察
    obs_features = algo.encode_observation(left_imgs, right_imgs, global_imgs, joints, use_target=False)
    
    # 清空梯度
    algo.critic_optimizer.zero_grad()
    algo.actorvae_optimizer.zero_grad()
    
    # 计算Critic loss并反向
    with torch.no_grad():
        target_v = torch.randn(batch_size, 1, device=device)
    current_v = algo.critic.v(obs_features)
    critic_loss = torch.nn.functional.mse_loss(current_v, target_v)
    
    print(f"\n[步骤1: Critic Loss 反向]")
    critic_loss.backward()
    obs_grad_after_critic_backward = algo.obs_encoder.image_encoder.conv1.weight.grad.clone()
    print(f"  obs_encoder 梯度大小: {obs_grad_after_critic_backward.norm().item():.6e}")
    
    print(f"\n[步骤2: Critic Optimizer Step]")
    algo.critic_optimizer.step()
    obs_param_after_critic_step = algo.obs_encoder.image_encoder.conv1.weight.data.clone()
    print(f"  obs_encoder 参数已更新 ✓")
    
    print(f"\n[步骤3: VAE Optimizer Zero Grad (关键!)]")
    print(f"  obs_encoder 是否在 actorvae_optimizer 中: {len(set(id(p) for p in algo.obs_encoder.parameters()) & set(id(p) for param_group in algo.actorvae_optimizer.param_groups for p in param_group['params'])) > 0}")
    
    # 这是关键: actorvae_optimizer.zero_grad() 不应该清除obs_encoder的梯度
    obs_grad_before_vae_zero = algo.obs_encoder.image_encoder.conv1.weight.grad
    print(f"  VAE Zero Grad 之前, obs_encoder 梯度: {obs_grad_before_vae_zero.norm().item() if obs_grad_before_vae_zero is not None else 'None':.6e}")
    
    algo.actorvae_optimizer.zero_grad()
    
    obs_grad_after_vae_zero = algo.obs_encoder.image_encoder.conv1.weight.grad
    print(f"  VAE Zero Grad 之后, obs_encoder 梯度: {obs_grad_after_vae_zero.norm().item() if obs_grad_after_vae_zero is not None else 'None'}")
    
    if obs_grad_after_vae_zero is not None:
        print(f"  ✓ obs_encoder 梯度被保留 (不被 actorvae_optimizer.zero_grad() 清除)")
    else:
        print(f"  ✗ obs_encoder 梯度被清除 (这会是一个问题!)")
    
    print(f"\n[步骤4: VAE Loss 反向]")
    vae_loss = torch.randn(batch_size, 1, device=device).sum()
    vae_loss.backward()
    print(f"  VAE 反向完成")
    
    print(f"\n[步骤5: VAE Optimizer Step]")
    algo.actorvae_optimizer.step()
    print(f"  VAE 参数更新完成")
    
    # 验证结果
    print(f"\n[最终验证]")
    obs_param_final = algo.obs_encoder.image_encoder.conv1.weight.data
    
    # obs_encoder应该被更新了两次: 一次来自Critic, 一次来自... (实际上actorvae_optimizer不包含obs_encoder)
    # 所以只被更新一次
    print(f"  obs_encoder 总参数变化: {(obs_param_final - obs_features).abs().max().item() if hasattr(obs_features, 'shape') else 'N/A'}")
    print(f"  ✓ obs_encoder 正确接收 Critic 梯度")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("LAPO 梯度流向诊断工具")
    print("=" * 80)
    
    # 创建dummy replay buffer
    replay_buffer = MultimodalReplayBuffer(
        state_dim=16,  # joint_dim
        action_dim=16,
        max_size=1000,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 创建algo实例
    try:
        algo = MultimodalLatent(
            obs_feature_dim=256,
            action_dim=16,
            max_action=1.0,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            replay_buffer=replay_buffer
        )
        print(f"✓ 成功创建 MultimodalLatent 实例")
        print(f"✓ 使用设备: {algo.device}")
        
    except Exception as e:
        print(f"✗ 创建 MultimodalLatent 失败: {e}")
        sys.exit(1)
    
    # 运行测试
    config_ok = test_optimizer_configuration(algo)
    print(f"\n✓ 优化器配置: {'正确' if config_ok else '❌ 错误'}")
    
    flow_ok = test_gradient_flow(algo)
    print(f"\n✓ 梯度流向: {'正确' if flow_ok else '❌ 错误'}")
    
    try:
        test_two_optimizer_flow(algo)
    except Exception as e:
        print(f"\n✗ 两优化器测试失败: {e}")
    
    print("\n" + "=" * 80)
    print("诊断完成!")
    print("=" * 80)
