## 多模态LAPO算法 - 机械臂数据集指南

本项目为LAPO（Latent Action Policy Optimization）离线强化学习算法的多模态扩展，支持处理真实机械臂采集的图像和关节数据。

### 项目结构

```
LAPO-offlienRL-main/
├── algos/
│   ├── algos_vae.py                  # 原始的LAPO算法实现
│   ├── algos_vae_multimodal.py        # 新增：多模态LAPO实现
│   ├── utils.py                       # 原始的数据缓冲区
│   └── utils_multimodal.py            # 新增：多模态数据缓冲区
├── main_d4rl.py                       # 原始的训练脚本（D4RL数据集）
├── main_multimodal.py                 # 新增：多模态训练脚本
├── logger.py                          # 日志记录
└── README.md
```

### 关键特性

#### 1. **多模态输入处理**
- **三路图像输入**：
  - 左手腕摄像头图像
  - 右手腕摄像头图像
  - 全局摄像头图像
- **关节数据**：16维度的双臂关节数据

#### 2. **ResNet18骨干网络**
- 使用预训练的ResNet18提取图像特征
- 三个图像编码器分别处理每路图像
- 融合层整合多模态信息

#### 3. **网络架构**

```
[三路图像] → ResNet18编码器 → 特征提取
                                    ↓
[关节数据] → FC网络 → 特征提取 → 融合层 → 融合特征
                                    ↓
                        [VAE、Actor、Critic]
```

#### 4. **HDF5数据加载**
支持从HDF5文件格式直接加载数据，期望格式：

```python
{
    'observations': {
        'left_image': np.array(shape=[N, H, W, 3]),      # 或 [N, 3, H, W]
        'right_image': np.array(shape=[N, H, W, 3]),     # 或 [N, 3, H, W]
        'global_image': np.array(shape=[N, H, W, 3]),    # 或 [N, 3, H, W]
        'joint': np.array(shape=[N, 16])                 # 16维关节数据
    },
    'actions': np.array(shape=[N, 16]),                   # 16维动作（关节目标）
    'rewards': np.array(shape=[N,]),                      # 标量奖励
    'terminals': np.array(shape=[N,])                     # 终止标志
}
```

### 使用方法

#### 前置要求

```bash
pip install torch torchvision h5py numpy tqdm
```

#### 基本使用

```bash
python main_multimodal.py \
    --hdf5_path /path/to/your/dataset.hdf5 \
    --device cuda \
    --ExpID 0001 \
    --batch_size 64 \
    --max_timesteps 100000
```

#### 常用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--hdf5_path` | 必需 | HDF5数据集文件路径 |
| `--batch_size` | 64 | 训练批次大小 |
| `--max_timesteps` | 1e6 | 最大训练步数 |
| `--eval_freq` | 5e3 | 评估频率（步数） |
| `--save_freq` | 5e4 | 模型保存频率（步数） |
| `--device` | cuda | 计算设备（cuda或cpu） |
| `--discount` | 0.99 | 折扣因子 γ |
| `--tau` | 0.005 | 软更新系数 |
| `--expectile` | 0.9 | 期望分位数（加权VAE） |
| `--kl_beta` | 1.0 | KL散度权重 |
| `--obs_feature_dim` | 256 | 观察特征维度 |
| `--action_dim` | 16 | 动作维度 |
| `--joint_dim` | 16 | 关节维度 |
| `--train_test_split` | 0.8 | 训练集占比 |

### 核心模块说明

#### 1. **ResNet18Encoder** (`algos_vae_multimodal.py`)
- 将单张图像编码为固定维度的特征向量
- 使用预训练的ResNet18骨干网络
- 输出: 256维特征向量

#### 2. **ImageJointEncoder** (`algos_vae_multimodal.py`)
```python
# 融合三路图像和关节数据
encoder = ImageJointEncoder(joint_dim=16, image_feature_dim=128, fusion_dim=256)
obs_feature = encoder(left_img, right_img, global_img, joint_data)
```

#### 3. **ActorVAE** (`algos_vae_multimodal.py`)
条件变分自编码器，学习观察到动作的分布：
```
Forward: (obs_feature, action) → (reconstructed_action, z_sample, mean, log_var)
Decode: (obs_feature, z) → action
```

#### 4. **MultimodalReplayBuffer** (`utils_multimodal.py`)
```python
# 创建缓冲区
buffer = MultimodalReplayBuffer(action_dim=16, joint_dim=16, device='cuda')

# 从HDF5加载数据
buffer.load_from_hdf5('/path/to/data.hdf5')

# 采样批次
batch = buffer.sample(batch_size=64)
# 返回: left_imgs, right_imgs, global_imgs, joints, actions, 
#       next_left_imgs, next_right_imgs, next_global_imgs, next_joints, 
#       rewards, not_done
```

### 算法流程

**每个训练迭代（iteration）：**

1. **Critic训练**
   - 目标Q值: $Q_{target} = r + \gamma V(s')$
   - 优化Q函数和V函数

2. **ActorVAE训练**（加权CVAE）
   - 计算优势: $A = Q(s') - V(s)$
   - 加权: $w = \begin{cases} \text{expectile} & \text{if } A > 0 \\ 1-\text{expectile} & \text{otherwise} \end{cases}$
   - 损失: $L = (MSE_{recon} + \beta KL) \cdot w$

3. **Actor训练**
   - 在隐空间最大化Q值
   - 损失: $L = -Q(s, VAE.decode(s, Actor(s)))$

4. **软更新**
   - $\theta_{target} \leftarrow \tau \theta + (1-\tau) \theta_{target}$

### 数据准备指南

#### HDF5文件创建示例

```python
import h5py
import numpy as np

# 创建HDF5文件
with h5py.File('robot_data.hdf5', 'w') as f:
    n_samples = 10000
    h, w = 84, 84
    
    # 创建观察组
    obs_group = f.create_group('observations')
    obs_group.create_dataset('left_image', data=np.zeros((n_samples, h, w, 3), dtype=np.uint8))
    obs_group.create_dataset('right_image', data=np.zeros((n_samples, h, w, 3), dtype=np.uint8))
    obs_group.create_dataset('global_image', data=np.zeros((n_samples, h, w, 3), dtype=np.uint8))
    obs_group.create_dataset('joint', data=np.zeros((n_samples, 16), dtype=np.float32))
    
    # 创建动作、奖励、终止标志
    f.create_dataset('actions', data=np.zeros((n_samples, 16), dtype=np.float32))
    f.create_dataset('rewards', data=np.zeros((n_samples,), dtype=np.float32))
    f.create_dataset('terminals', data=np.zeros((n_samples,), dtype=np.bool_))
```

### 输出和日志

训练期间会生成以下输出：

```
results/
└── Exp0001/
    └── multimodal_robot/
        ├── progress.csv                    # 训练进度
        ├── model_best_critic.pth           # 最佳模型
        ├── model_best_actor.pth
        ├── model_best_actor_vae.pth
        ├── model_best_obs_encoder.pth
        └── ...
```

### 常见问题

**Q: 如何调整图像编码器的复杂度？**
A: 修改 `ImageJointEncoder` 中的 `image_feature_dim` 参数（默认128）。增大会提高表达能力但增加计算量。

**Q: 如何处理不同尺寸的图像？**
A: ResNet18可以自动处理不同大小的输入。在加载数据时确保所有图像维度一致，或在加载前进行resize。

**Q: 如何加载预训练的模型继续训练？**
A: 使用 `--load_model` 参数：
```bash
python main_multimodal.py --hdf5_path ... --load_model 100000 --ExpID 0002
```

**Q: 模型过拟合怎么办？**
A: 
- 增加 `--expectile` 值（更保守的加权）
- 减小 `--kl_beta`（降低VAE正则化）
- 增加批次大小 `--batch_size`
- 使用更多训练数据

### 与原始LAPO的区别

| 特性 | 原始LAPO | 多模态LAPO |
|------|---------|----------|
| 观察空间 | 低维向量 | 多模态（图像+关节） |
| 图像处理 | 无 | ResNet18 |
| 数据来源 | D4RL环境 | 真实机械臂 |
| 数据格式 | NumPy数组 | HDF5文件 |
| 输入编码 | 直接使用 | 融合层处理 |

### 参考文献

- LAPO: https://github.com/sfujim/BCQ
- ResNet: He, K., et al. "Deep Residual Learning for Image Recognition." CVPR, 2016.
- 条件VAE: Sohn, K., et al. "Learning Structured Output Representation using Deep Conditional Generative Models." NIPS, 2015.

### 许可证

与原始LAPO项目保持一致。
