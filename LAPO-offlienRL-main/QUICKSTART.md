# 多模态LAPO快速入门指南

## 📋 概览

这是一个改进的LAPO（Latent Action Policy Optimization）离线强化学习实现，专门为处理真实机械臂采集的多模态数据而设计。

**核心改进：**
- ✅ 支持三路图像输入（左手腕、右手腕、全局）
- ✅ 支持16维关节数据处理  
- ✅ 使用ResNet18作为图像特征提取骨干
- ✅ 从HDF5格式文件加载训练数据
- ✅ 多模态特征融合层设计

## 🚀 快速开始

### 步骤1: 安装依赖

```bash
pip install torch torchvision h5py numpy tqdm
```

### 步骤2: 准备数据

#### 选项A：创建示例数据集（用于测试）

```bash
python create_dataset.py --create_sample --output sample_data.hdf5 --n_samples 5000
```

#### 选项B：使用自己的数据

确保数据是HDF5格式，文件结构如下：

```
your_data.hdf5
├── observations/
│   ├── left_image      [N, H, W, 3]   # 左手腕图像
│   ├── right_image     [N, H, W, 3]   # 右手腕图像
│   ├── global_image    [N, H, W, 3]   # 全局图像
│   └── joint           [N, 16]        # 关节数据
├── actions             [N, 16]        # 动作（关节目标）
├── rewards             [N]            # 奖励
└── terminals           [N]            # 终止标志
```

数据准备的Python代码示例：

```python
import h5py
import numpy as np

# 假设你已经采集了数据
left_imgs = ...  # shape: [N, H, W, 3], uint8
right_imgs = ...  # shape: [N, H, W, 3], uint8
global_imgs = ... # shape: [N, H, W, 3], uint8
joints = ...      # shape: [N, 16], float32
actions = ...     # shape: [N, 16], float32
rewards = ...     # shape: [N], float32
terminals = ...   # shape: [N], bool

with h5py.File('my_robot_data.hdf5', 'w') as f:
    obs = f.create_group('observations')
    obs.create_dataset('left_image', data=left_imgs)
    obs.create_dataset('right_image', data=right_imgs)
    obs.create_dataset('global_image', data=global_imgs)
    obs.create_dataset('joint', data=joints)
    f.create_dataset('actions', data=actions)
    f.create_dataset('rewards', data=rewards)
    f.create_dataset('terminals', data=terminals)
```

### 步骤3: 验证数据格式

```bash
python create_dataset.py --validate --hdf5_path your_data.hdf5
```

输出示例：
```
验证数据集: your_data.hdf5
============================================================

文件结构:
  ├─ observations/
  │  ├─ left_image: uint8, shape=(5000, 84, 84, 3)
  │  ├─ right_image: uint8, shape=(5000, 84, 84, 3)
  │  ├─ global_image: uint8, shape=(5000, 84, 84, 3)
  │  └─ joint: float32, shape=(5000, 16)
  ├─ actions: float32, shape=(5000, 16)
  ├─ rewards: float32, shape=(5000,)
  └─ terminals: bool, shape=(5000,)

✓ 数据格式验证成功!
```

### 步骤4: 开始训练

#### 基本命令

```bash
python main_multimodal.py \
    --hdf5_path sample_data.hdf5 \
    --device cuda \
    --ExpID 0001 \
    --batch_size 64 \
    --max_timesteps 50000
```

#### 参数说明

```bash
python main_multimodal.py \
    --hdf5_path sample_data.hdf5 \           # 必需：数据文件路径
    --device cuda \                          # 计算设备 (cuda/cpu)
    --ExpID exp_001 \                        # 实验ID
    --batch_size 64 \                        # 批大小（根据GPU显存调整）
    --max_timesteps 50000 \                  # 最大训练步数
    --eval_freq 1000 \                       # 评估间隔
    --save_freq 5000 \                       # 模型保存间隔
    --discount 0.99 \                        # 折扣因子
    --expectile 0.9 \                        # 期望值（加权采样）
    --kl_beta 1.0 \                          # KL散度权重
    --obs_feature_dim 256 \                  # 观察特征维度
    --train_test_split 0.8                   # 训练集比例
```

### 步骤5: 监查训练结果

训练结果保存在 `results/Exp{ExpID}/multimodal_robot/` 目录下：

```
results/
└── Exp0001/
    └── multimodal_robot/
        ├── progress.csv           # 训练进度记录
        ├── model_best_*.pth       # 最佳模型
        ├── model_final_*.pth      # 最终模型
        └── params.json            # 实验参数
```

查看训练进度：
```bash
cat results/Exp0001/multimodal_robot/progress.csv
```

## 📊 网络架构

### 多模态编码器

```
┌─────────────────────────────────────────┐
│          多模态输入                      │
│  三路图像 + 16维关节数据                 │
└──────────────────┬──────────────────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼          ▼
   ┌────────┐ ┌────────┐ ┌────────┐ ┌──────┐
   │ResNet18│ │ResNet18│ │ResNet18│ │  FC  │
   │        │ │        │ │        │ │ 网络 │
   └────┬───┘ └────┬───┘ └────┬───┘ └──┬───┘
        │          │          │         │
        └─────┬────┴────┬─────┘─────────┘
              │         │
         ┌────▼─────────▼────┐
         │   融合层(Fusion)   │
         │  [256维特征向量]   │
         └────┬───────────────┘
              │
        ┌─────▼──────┐
        │   VAE      │
        │   Actor    │
        │   Critic   │
        └────────────┘
```

### VAE架构

```
编码器: (obs_feature + action) → [256, 256, 256] → (mean, log_var)
                                        ↓
                                重参数化: z ~ N(μ, σ²)
                                        ↓
解码器: (obs_feature + z) → [256, 256, 256] → action
```

## 🔧 参数调整建议

### 对于小数据集（<10k样本）

```bash
--batch_size 32 \
--expectile 0.95 \          # 更保守的采样
--kl_beta 0.5 \             # 降低正则化
--tau 0.01                  # 更快的软更新
```

### 对于中等数据集（10k-100k样本）

```bash
--batch_size 64 \           # 推荐配置
--expectile 0.9 \
--kl_beta 1.0 \
--tau 0.005
```

### 对于大数据集（>100k样本）

```bash
--batch_size 128 \
--expectile 0.85 \          # 更激进的采样
--kl_beta 2.0 \             # 增强正则化
--tau 0.001                 # 更慢的软更新
```

### GPU显存不足时

```bash
--batch_size 32 \           # 减小批大小
--obs_feature_dim 128 \     # 减小特征维度
--device cpu                # 使用CPU (慢但无限制)
```

## 📈 监控训练

### 实时查看日志

```bash
tail -f results/Exp0001/multimodal_robot/progress.csv
```

### 关键指标

| 指标 | 说明 | 目标 |
|------|------|------|
| AverageReturn | 平均累积奖励 | 越高越好 |
| StdReturn | 标准差 | 越低越稳定 |
| Training Steps | 训练步数 | 达到max_timesteps |

## 💾 模型保存和加载

### 保存

模型会自动保存到：
```
results/Exp{ExpID}/multimodal_robot/model_*.pth
```

### 加载并继续训练

```bash
python main_multimodal.py \
    --hdf5_path sample_data.hdf5 \
    --ExpID exp_002 \
    --load_model 25000 \          # 加载第25000步的模型
    --max_timesteps 100000
```

### 加载模型进行推理

```python
from algos import algos_vae_multimodal as algos
from algos import utils_multimodal

# 创建策略
policy = algos.MultimodalLatent(...)

# 加载模型
policy.load('model_best', 'path/to/results')

# 推理
action = policy.select_action(left_img, right_img, global_img, joint)
```

## 🐛 常见问题

### Q: 数据加载失败，提示"缺少xxx键"

**A:** 检查HDF5文件结构：
```python
import h5py
with h5py.File('your_data.hdf5', 'r') as f:
    print(list(f.keys()))
    print(list(f['observations'].keys()))
```

### Q: 显存溢出 (CUDA out of memory)

**A:** 降低批大小或特征维度：
```bash
--batch_size 32 --obs_feature_dim 128
```

### Q: 训练速度很慢

**A:** 
1. 确保使用GPU：`--device cuda`
2. 增加批大小（显存允许的情况下）
3. 减小特征维度

### Q: 模型过拟合

**A:**
```bash
--expectile 0.95 \      # 更保守
--kl_beta 2.0 \         # 增加正则化
--batch_size 128        # 增加批大小
```

### Q: 模型欠拟合

**A:**
```bash
--expectile 0.8 \       # 更激进
--kl_beta 0.5 \         # 减少正则化
--obs_feature_dim 512   # 增加特征维度
```

## 📚 文件说明

| 文件 | 说明 |
|------|------|
| `algos_vae_multimodal.py` | 核心算法实现 |
| `utils_multimodal.py` | 多模态数据缓冲区 |
| `main_multimodal.py` | 训练脚本 |
| `create_dataset.py` | 数据集工具 |
| `README_MULTIMODAL.md` | 详细文档 |

## 📖 进阶用法

### 自定义网络架构

编辑 `algos_vae_multimodal.py` 中的网络参数：

```python
# 修改融合特征维度
self.fusion_fc = nn.Sequential(
    nn.Linear(total_feature_dim, 1024),  # 增加隐层
    nn.ReLU(),
    nn.Dropout(0.2),  # 增加dropout
    ...
)

# 修改ResNet18输出维度
image_feature_dim = 256  # 默认128，可调整
```

### 自定义损失函数

在 `MultimodalLatent.train()` 中修改损失计算：

```python
# 添加额外的正则化项
extra_loss = some_custom_loss()
actor_vae_loss = actor_vae_loss + 0.1 * extra_loss
```

## 🎓 理论背景

### LAPO算法

LAPO通过以下步骤进行离线强化学习：

1. **VAE训练**：学习动作分布，只关注高Q值的动作
2. **Actor训练**：在隐空间优化动作选择
3. **Critic训练**：估计状态-动作价值

### 多模态扩展

- 使用ResNet18提取视觉特征
- 融合层整合多模态信息
- 端到端学习可视化控制策略

## 📞 支持

如有问题，请检查：
1. 数据格式是否正确
2. GPU显存是否足够
3. 依赖包版本是否兼容

## 📝 许可证

保持与原始LAPO项目一致。
