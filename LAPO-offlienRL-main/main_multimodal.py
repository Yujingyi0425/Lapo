#!/usr/bin/env python3
"""
多模态LAPO算法训练脚本 - 修正版
功能：
1. 从HDF5加载数据并进行 训练集/测试集 切分
2. 使用 MSE Loss 在测试集上评估策略性能 (Behavior Cloning指标)
3. 训练流程控制
"""
import argparse
import os
import torch
import numpy as np
from algos import utils_multimodal
from algos import algos_vae_multimodal as algos
from logger import logger, setup_logger

def validate_policy(policy, test_buffer, eval_samples=100):
    """
    [修改] 验证策略性能 (Behavior Cloning Validation)
    由于是离线训练且没有仿真环境，我们计算策略在测试集上的预测动作与真实动作的 MSE 误差。
    """
    print(f"Validating policy on test dataset ({eval_samples} samples)...")
    policy._set_eval_mode() # 强制进入评估模式 (Fix BatchNorm)
    
    # 采样测试数据
    # 注意：Buffer返回的图像是 0-255 的 uint8
    left_imgs, right_imgs, global_imgs, joints, actions, _, _, _, _, _, _ = test_buffer.sample(eval_samples)
    
    # 手动归一化图像 (0-255 -> 0-1)
    left_imgs = left_imgs / 255.0
    right_imgs = right_imgs / 255.0
    global_imgs = global_imgs / 255.0
    
    with torch.no_grad():
        # 1. 编码观察
        obs_features = policy.encode_observation(left_imgs, right_imgs, global_imgs, joints)
        
        # 2. 让Actor生成潜在变量
        latent_action = policy.actor(obs_features)
        
        # 3. VAE解码得到动作 (确定的解码，即 z=latent_action)
        predicted_action = policy.actor_vae_target.decode(obs_features, z=latent_action)
        
        # 4. 计算 MSE Loss (预测动作 vs 专家动作)
        # 注意：actions 在 buffer sample 时已经归一化，predicted_action 也是归一化的输出
        mse = ((predicted_action - actions) ** 2).mean(dim=1) # (Batch,)
        avg_mse = mse.mean().item()

    print("---------------------------------------")
    print(f"Validation Result:")
    print(f"Action MSE Loss: {avg_mse:.6f}")
    print("---------------------------------------")
    
    policy._set_train_mode() # 恢复训练模式
    return {'Val_MSE': avg_mse}

def split_buffer_inplace(full_buffer, ratio=0.9):
    """
    将 buffer 原地切分为训练集和测试集
    修复：增加了 indices_buf 和 image_metadata 的复制
    """
    total_size = full_buffer.size
    train_size = int(total_size * ratio)
    test_size = total_size - train_size
    
    print(f"Splitting data: Total {total_size} -> Train {train_size} | Test {test_size}")
    
    # 创建测试集 Buffer
    test_buffer = utils_multimodal.MultimodalReplayBuffer(
        action_dim=full_buffer.action_dim,
        joint_dim=full_buffer.joint_dim,
        device=str(full_buffer.device),
        max_size=test_size
    )
    
    # 1. 复制统计信息
    test_buffer.action_mean = full_buffer.action_mean
    test_buffer.action_std = full_buffer.action_std
    test_buffer.joint_mean = full_buffer.joint_mean
    test_buffer.joint_std = full_buffer.joint_std
    
    # 2. [关键修复] 复制图像元数据 (HDF5路径等)
    test_buffer.image_metadata = full_buffer.image_metadata.copy()
    
    # 3. 复制标量数据
    for key in full_buffer.storage:
        if full_buffer.storage[key] is not None:
            test_buffer.storage[key][:test_size] = full_buffer.storage[key][train_size : total_size]
    
    # 4. [关键修复] 复制索引映射数据
    # 如果不复制这个，测试集将不知道去 HDF5 的哪里读取对应的图像
    test_buffer.indices_buf[:test_size] = full_buffer.indices_buf[train_size : total_size]
    
    test_buffer.size = test_size
    test_buffer.ptr = test_size % test_size
    
    # 截断训练集
    full_buffer.size = train_size
    full_buffer.ptr = train_size % full_buffer.max_size
    
    return full_buffer, test_buffer
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 数据相关
    parser.add_argument("--hdf5_path", required=True, type=str, help="HDF5数据集路径")
    parser.add_argument("--train_test_split", default=0.9, type=float, help="训练集比例")
    
    # 实验设置
    parser.add_argument("--ExpID", default="0001", type=str)
    parser.add_argument('--log_dir', default='./results/', type=str)
    parser.add_argument("--save_model", default=True, type=bool)
    parser.add_argument("--save_freq", default=50000, type=int) # 每5w步保存
    parser.add_argument("--load_model", default=0, type=int)
    parser.add_argument("--seed", default=123, type=int)
    
    # 训练超参
    parser.add_argument("--max_timesteps", default=1000000, type=int)
    parser.add_argument("--eval_freq", default=5000, type=int)
    parser.add_argument('--batch_size', default=64, type=int) # 显存敏感
    
    # 算法参数 (Multimodal LAPO)
    parser.add_argument('--vae_lr', default=1e-4, type=float)
    parser.add_argument('--actor_lr', default=1e-4, type=float)
    parser.add_argument('--critic_lr', default=3e-4, type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--expectile', default=0.7, type=float)
    parser.add_argument('--kl_beta', default=0.5, type=float)
    parser.add_argument('--max_latent_action', default=2.0, type=float)
    parser.add_argument('--doubleq_min', default=0.75, type=float)
    parser.add_argument('--no_noise', action='store_true')
    
    # 网络结构
    parser.add_argument('--obs_feature_dim', default=256, type=int)
    parser.add_argument('--action_dim', default=16, type=int)
    parser.add_argument('--joint_dim', default=16, type=int)
    
    # 设备与调试
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--num_traj', default=None, type=int, help="调试用：只加载前N条轨迹")

    args = parser.parse_args()

    # 1. 设置日志
    file_name = f"Exp{args.ExpID}/multimodal_robot"
    folder_name = os.path.join(args.log_dir, file_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # 如果不是接着训练，检查是否覆盖
    if args.load_model == 0 and os.path.exists(os.path.join(folder_name, 'progress.csv')):
        print('实验记录已存在，建议更改 ExpID 防止覆盖。')
        # raise AssertionError 
    
    setup_logger(os.path.basename(folder_name), variant=vars(args), log_dir=folder_name)
    
    # 2. 设置种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 3. 加载完整数据集
    print(f"\n[1/4] Loading dataset from {args.hdf5_path}...")
    full_buffer = utils_multimodal.MultimodalReplayBuffer(
        action_dim=args.action_dim,
        joint_dim=args.joint_dim,
        device=args.device,
        max_size=int(1e6) # 请确保这里足够大以容纳你的数据
    )
    full_buffer.load_from_hdf5(args.hdf5_path, num_traj=args.num_traj)
    
    # 4. 切分数据集
    print(f"\n[2/4] Splitting dataset (Ratio: {args.train_test_split})...")
    train_buffer, test_buffer = split_buffer_inplace(full_buffer, ratio=args.train_test_split)
    
    # 5. 计算价值函数范围 (基于训练集)
    rewards = train_buffer.storage['reward'][:train_buffer.size]
    min_v = float(rewards.min() / (1 - args.discount))
    max_v = float(rewards.max() / (1 - args.discount))
    print(f"Value Range: [{min_v:.2f}, {max_v:.2f}]")

    # 6. 初始化策略
    print(f"\n[3/4] Initializing Policy...")
    latent_dim = args.action_dim * 2
    
    policy = algos.MultimodalLatent(
        action_dim=args.action_dim,
        latent_dim=latent_dim,
        max_action=1.0, # 动作已经在Buffer内部归一化了，所以网络输出范围是相对的
        min_v=min_v,
        max_v=max_v,
        replay_buffer=train_buffer, # 传入训练集Buffer
        device=args.device,
        joint_dim=args.joint_dim,
        obs_feature_dim=args.obs_feature_dim,
        discount=args.discount,
        tau=args.tau,
        vae_lr=args.vae_lr,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        max_latent_action=args.max_latent_action,
        expectile=args.expectile,
        kl_beta=args.kl_beta,
        no_noise=args.no_noise,
        doubleq_min=args.doubleq_min
    )

    # 加载模型 (如果需要)
    training_iters = 0
    if args.load_model != 0:
        print(f"Loading model_{args.load_model}...")
        policy.load(f'model_{args.load_model}', folder_name)
        training_iters = args.load_model

    # 7. 开始训练
    print(f"\n[4/4] Starting training for {args.max_timesteps} steps...")
    
    best_mse = float('inf')
    
    while training_iters < args.max_timesteps:
        # 训练一步 (包含 batch_size 个样本的更新)
        policy.train(iterations=int(args.eval_freq), batch_size=args.batch_size)
        training_iters += args.eval_freq
        
        # 记录训练步数
        logger.record_tabular('Training Steps', int(training_iters))
        
        # 定期保存
        if training_iters % args.save_freq == 0 and args.save_model:
            print(f"Saving model at step {training_iters}")
            policy.save(f'model_{training_iters}', folder_name)
        
        # 评估 (使用测试集验证 MSE)
        info = validate_policy(policy, test_buffer, eval_samples=100)
        
        # 记录评估指标
        for k, v in info.items():
            logger.record_tabular(k, v)
        
        # 保存最佳模型
        if info['Val_MSE'] < best_mse:
            best_mse = info['Val_MSE']
            print(f"New best model! MSE: {best_mse:.6f}")
            policy.save('model_best', folder_name)
            
        logger.dump_tabular()

    # 最终保存
    policy.save('model_final', folder_name)
    print("Training finished.")