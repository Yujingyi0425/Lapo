import numpy as np
import torch
import h5py
import cv2
from tqdm import tqdm

class MultimodalReplayBuffer(object):
    """
    多模态经验回放缓冲区 - 最终优化版
    
    特点：
    1. 懒加载 (Lazy Loading): 图像不占内存，采样时实时读取
    2. 向量化加载: 启动速度快
    3. 自动 Resize: 图像强制转换为 224x224 (CHW)
    4. 无归一化: 假设输入数据已归一化，直接输出原值
    """
    def __init__(self, action_dim, joint_dim=16, device='cpu', max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = torch.device(device)
        
        self.action_dim = action_dim
        self.joint_dim = joint_dim
        
        # ✅ 标量数据存储 (RAM) - 仅几百MB
        self.storage = {
            'joint': np.zeros((max_size, joint_dim), dtype=np.float32),
            'action': np.zeros((max_size, action_dim), dtype=np.float32),
            'next_joint': np.zeros((max_size, joint_dim), dtype=np.float32),
            'reward': np.zeros((max_size, 1), dtype=np.float32),
            'terminal': np.zeros((max_size, 1), dtype=np.float32),
        }
        
        # ✅ 索引映射存储 (RAM) - 记录每条数据在 HDF5 中的位置
        # shape: (max_size, 2) -> [curr_idx, next_idx]
        self.indices_buf = np.zeros((max_size, 2), dtype=np.int32)
        
        # 💾 图像元数据
        self.image_metadata = {
            'hdf5_path': None,
            'need_transpose': False, # 标记原始数据是否为 HWC
        }
        
        # 统计信息 (仅用于日志查看，不参与归一化)
        self.action_mean = None
        self.action_std = None
        self.joint_mean = None
        self.joint_std = None
        
        # 文件句柄缓存
        self._hdf5_cache = {}

    def _get_hdf5_file(self, hdf5_path):
        """获取文件句柄，避免重复打开"""
        if hdf5_path not in self._hdf5_cache:
            # swmr=True 尝试允许并发读取模式
            self._hdf5_cache[hdf5_path] = h5py.File(hdf5_path, 'r', swmr=True, libver='latest')
        return self._hdf5_cache[hdf5_path]

    def _load_and_resize_image(self, hdf5_file, dataset_key, index):
        """
        核心读取函数：读取 -> Resize -> Transpose
        目标输出: (3, 224, 224)
        """
        # 读取原始数据
        img = hdf5_file[dataset_key][index]
        
        # 目标: HWC (用于cv2) -> CHW (用于PyTorch)
        if self.image_metadata['need_transpose']:
            # 输入原本是 HWC (例如 480x640x3)，直接 Resize
            img_resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
            # HWC -> CHW
            return np.transpose(img_resized, (2, 0, 1))
        else:
            # 输入原本是 CHW (例如 3x480x640)，先转 HWC 才能 Resize
            img_hwc = np.transpose(img, (1, 2, 0))
            img_resized = cv2.resize(img_hwc, (224, 224), interpolation=cv2.INTER_LINEAR)
            # HWC -> CHW
            return np.transpose(img_resized, (2, 0, 1))

    def sample(self, batch_size):
        """
        采样 Batch
        注意：这里不再调用 normalize_joint/action
        """
        ind = np.random.randint(0, self.size, size=batch_size)
        
        # 1. 准备图像读取
        hdf5_path = self.image_metadata['hdf5_path']
        hdf5_file = self._get_hdf5_file(hdf5_path)
        
        # 确定 HDF5 中的键名
        if 'observations' in hdf5_file:
            obs = hdf5_file['observations']
            left_key = 'left_image' if 'left_image' in obs else 'left_wrist_image'
            right_key = 'right_image' if 'right_image' in obs else 'right_wrist_image'
            global_key = 'global_image'
        else:
            left_key = 'left_image'
            right_key = 'right_image'
            global_key = 'global_image'
        
        # 获取该 batch 对应的 HDF5 索引
        curr_indices = self.indices_buf[ind, 0]
        next_indices = self.indices_buf[ind, 1]
        
        # 预分配内存 (Batch, 3, 224, 224)
        left_imgs = np.empty((batch_size, 3, 224, 224), dtype=np.uint8)
        right_imgs = np.empty((batch_size, 3, 224, 224), dtype=np.uint8)
        global_imgs = np.empty((batch_size, 3, 224, 224), dtype=np.uint8)
        
        next_left_imgs = np.empty((batch_size, 3, 224, 224), dtype=np.uint8)
        next_right_imgs = np.empty((batch_size, 3, 224, 224), dtype=np.uint8)
        next_global_imgs = np.empty((batch_size, 3, 224, 224), dtype=np.uint8)
        
        # 循环读取并 Resize (IO密集型)
        for i in range(batch_size):
            c_idx = curr_indices[i]
            n_idx = next_indices[i]
            
            left_imgs[i] = self._load_and_resize_image(hdf5_file, left_key, c_idx)
            right_imgs[i] = self._load_and_resize_image(hdf5_file, right_key, c_idx)
            global_imgs[i] = self._load_and_resize_image(hdf5_file, global_key, c_idx)
            
            next_left_imgs[i] = self._load_and_resize_image(hdf5_file, left_key, n_idx)
            next_right_imgs[i] = self._load_and_resize_image(hdf5_file, right_key, n_idx)
            next_global_imgs[i] = self._load_and_resize_image(hdf5_file, global_key, n_idx)

        # 转换为 Tensor 并移动到 GPU
        batch_device = self.device
        
        return (
            torch.FloatTensor(left_imgs).to(batch_device),
            torch.FloatTensor(right_imgs).to(batch_device),
            torch.FloatTensor(global_imgs).to(batch_device),
            
            # 标量数据：直接返回，不做归一化
            torch.FloatTensor(self.storage['joint'][ind]).to(batch_device),
            torch.FloatTensor(self.storage['action'][ind]).to(batch_device),
            
            torch.FloatTensor(next_left_imgs).to(batch_device),
            torch.FloatTensor(next_right_imgs).to(batch_device),
            torch.FloatTensor(next_global_imgs).to(batch_device),
            
            # 标量数据：直接返回
            torch.FloatTensor(self.storage['next_joint'][ind]).to(batch_device),
            
            torch.FloatTensor(self.storage['reward'][ind]).to(batch_device),
            1.0 - torch.FloatTensor(self.storage['terminal'][ind]).to(batch_device)
        )

    # 🚫 禁用归一化函数，直接返回原值
    def normalize_joint(self, joint): return joint
    def unnormalize_joint(self, joint): return joint
    def normalize_action(self, action): return action
    def unnormalize_action(self, action): return action

    def load_from_hdf5(self, hdf5_path, num_traj=None):
        """
        向量化加载元数据 (秒级完成)
        """
        print(f"🚀 [Lazy Load] Loading metadata from {hdf5_path} (No Normalization)...")
        
        with h5py.File(hdf5_path, 'r') as f:
            if 'observations' in f:
                obs = f['observations']
                # 读取第一张图来检测格式
                if 'left_image' in obs: left_sample = obs['left_image'][0]
                else: left_sample = obs['left_wrist_image'][0]
                joints_dset = obs['joint']
            else:
                left_sample = f['left_image'][0]
                joints_dset = f['joint']
            
            actions = f['actions']
            rewards = f['rewards']
            terminals = f['terminals']
            
            total_len = len(actions)
            if num_traj is not None:
                total_len = min(total_len, num_traj)
            
            # 检测是否需要转置: HWC (shape[-1]==3) -> 需要转置
            need_transpose = (left_sample.shape[-1] == 3)
            
            # 计算实际加载量
            load_len = min(total_len, self.max_size - self.size)
            if load_len <= 0:
                print("⚠️ Buffer full, skipping load.")
                return

            print(f"  📥 Loading {load_len} transitions into RAM (Scalars only)...")
            
            start_ptr = self.ptr
            end_ptr = start_ptr + load_len
            
            # 处理 Buffer 回环 (简化版：如果溢出则截断)
            if end_ptr > self.max_size:
                load_len = self.max_size - start_ptr
                end_ptr = self.max_size
                print(f"  ⚠️ Truncating load to {load_len} items due to buffer limit.")

            # ✅ 向量化赋值：瞬间完成
            self.storage['joint'][start_ptr:end_ptr] = joints_dset[:load_len]
            # 确保 joints 足够长
            self.storage['next_joint'][start_ptr:end_ptr] = joints_dset[1:load_len+1]
            
            self.storage['action'][start_ptr:end_ptr] = actions[:load_len]
            self.storage['reward'][start_ptr:end_ptr] = rewards[:load_len]
            self.storage['terminal'][start_ptr:end_ptr] = terminals[:load_len]
            
            # 生成索引映射: (0, 1), (1, 2), ...
            indices = np.arange(load_len, dtype=np.int32)
            self.indices_buf[start_ptr:end_ptr, 0] = indices
            self.indices_buf[start_ptr:end_ptr, 1] = indices + 1
            
            # 更新指针
            self.ptr = (self.ptr + load_len) % self.max_size
            self.size = min(self.size + load_len, self.max_size)
            
            # 保存元数据
            self.image_metadata['hdf5_path'] = hdf5_path
            self.image_metadata['need_transpose'] = need_transpose

        self.compute_statistics()
        print(f"✅ Load Complete. Buffer size: {self.size}")
        print(f"   Image loading: On-the-fly (Resize to 224x224)")
        print(f"   Normalization: DISABLED (Assuming data is pre-normalized)")

    def compute_statistics(self):
        print("Computing statistics (For logging only)...")
        # 仅计算用于显示，不用于归一化
        self.action_mean = np.mean(self.storage['action'][:self.size], axis=0)
        self.action_std = np.std(self.storage['action'][:self.size], axis=0)
        self.joint_mean = np.mean(self.storage['joint'][:self.size], axis=0)
        self.joint_std = np.std(self.storage['joint'][:self.size], axis=0)