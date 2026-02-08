#!/usr/bin/env python3
"""
HDF5数据集创建和验证脚本
帮助用户将实际采集的机械臂数据转换为HDF5格式
"""
import h5py
import numpy as np
import os
from pathlib import Path


def create_sample_dataset(output_path='sample_robot_data.hdf5', n_samples=1000):
    """
    创建示例HDF5数据集
    用于测试多模态LAPO算法
    
    Args:
        output_path: 输出文件路径
        n_samples: 样本数量
    """
    print(f"创建示例数据集: {output_path}")
    print(f"样本数: {n_samples}")
    
    h, w = 84, 84  # 图像分辨率
    n_joints = 16  # 关节数
    
    with h5py.File(output_path, 'w') as f:
        # 创建观察组
        obs_group = f.create_group('observations')
        
        print("创建图像数据...")
        # 左手腕图像 (NHWC格式)
        obs_group.create_dataset(
            'left_image',
            shape=(n_samples, h, w, 3),
            dtype=np.uint8,
            data=np.random.randint(0, 256, (n_samples, h, w, 3), dtype=np.uint8)
        )
        
        # 右手腕图像
        obs_group.create_dataset(
            'right_image',
            shape=(n_samples, h, w, 3),
            dtype=np.uint8,
            data=np.random.randint(0, 256, (n_samples, h, w, 3), dtype=np.uint8)
        )
        
        # 全局图像
        obs_group.create_dataset(
            'global_image',
            shape=(n_samples, h, w, 3),
            dtype=np.uint8,
            data=np.random.randint(0, 256, (n_samples, h, w, 3), dtype=np.uint8)
        )
        
        print("创建关节数据...")
        # 关节数据 (16维)
        obs_group.create_dataset(
            'joint',
            shape=(n_samples, n_joints),
            dtype=np.float32,
            data=np.random.randn(n_samples, n_joints).astype(np.float32)
        )
        
        print("创建动作数据...")
        # 动作数据 (16维)
        f.create_dataset(
            'actions',
            shape=(n_samples, n_joints),
            dtype=np.float32,
            data=np.random.randn(n_samples, n_joints).astype(np.float32)
        )
        
        print("创建奖励数据...")
        # 奖励数据
        f.create_dataset(
            'rewards',
            shape=(n_samples,),
            dtype=np.float32,
            data=np.random.randn(n_samples).astype(np.float32)
        )
        
        print("创建终止标志...")
        # 终止标志
        terminals = np.zeros((n_samples,), dtype=bool)
        # 每500步一个episode结束
        for i in range(500, n_samples, 500):
            terminals[i] = True
        f.create_dataset(
            'terminals',
            data=terminals
        )
    
    print(f"✓ 数据集创建成功!")
    return output_path


def validate_hdf5_format(hdf5_path):
    """
    验证HDF5文件格式是否正确
    
    Args:
        hdf5_path: HDF5文件路径
    """
    print(f"\n验证数据集: {hdf5_path}")
    print("=" * 60)
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            print("\n文件结构:")
            
            def print_structure(name, obj):
                indent = "  " * (name.count('/'))
                if isinstance(obj, h5py.Dataset):
                    print(f"{indent}├─ {name.split('/')[-1]}: {obj.dtype}, shape={obj.shape}")
                else:
                    print(f"{indent}├─ {name.split('/')[-1]}/")
            
            f.visititems(print_structure)
            
            # 检查必需的键
            print("\n数据验证:")
            required_obs_keys = ['left_image', 'right_image', 'global_image', 'joint']
            required_keys = ['actions', 'rewards', 'terminals']
            
            # 检查observations
            if 'observations' in f:
                obs = f['observations']
                for key in required_obs_keys:
                    if key in obs:
                        print(f"✓ {key}: shape={obs[key].shape}, dtype={obs[key].dtype}")
                    else:
                        print(f"✗ 缺少 {key}")
            
            # 检查其他必需数据
            for key in required_keys:
                if key in f:
                    print(f"✓ {key}: shape={f[key].shape}, dtype={f[key].dtype}")
                else:
                    print(f"✗ 缺少 {key}")
            
            # 数据统计
            print("\n数据统计:")
            obs = f['observations']
            actions = f['actions'][:]
            rewards = f['rewards'][:]
            
            n_samples = len(actions)
            print(f"样本总数: {n_samples}")
            print(f"动作维度: {actions.shape[1]}")
            print(f"关节维度: {obs['joint'].shape[1]}")
            print(f"图像大小: {obs['left_image'].shape[1]}x{obs['left_image'].shape[2]}")
            print(f"奖励范围: [{rewards.min():.4f}, {rewards.max():.4f}]")
            print(f"奖励均值: {rewards.mean():.4f} ± {rewards.std():.4f}")
            
            print("\n✓ 数据格式验证成功!")
            return True
    
    except Exception as e:
        print(f"\n✗ 验证失败: {e}")
        return False


def convert_from_numpy(numpy_dir, output_path='converted_data.hdf5'):
    """
    从NumPy数组文件转换为HDF5格式
    
    Args:
        numpy_dir: 包含numpy文件的目录
        output_path: 输出HDF5文件路径
        
    期望的NumPy文件名:
        - observations_left_img.npy
        - observations_right_img.npy
        - observations_global_img.npy
        - observations_joint.npy
        - actions.npy
        - rewards.npy
        - terminals.npy
    """
    print(f"从NumPy文件转换: {numpy_dir}")
    
    file_mapping = {
        'left_image': os.path.join(numpy_dir, 'observations_left_img.npy'),
        'right_image': os.path.join(numpy_dir, 'observations_right_img.npy'),
        'global_image': os.path.join(numpy_dir, 'observations_global_img.npy'),
        'joint': os.path.join(numpy_dir, 'observations_joint.npy'),
        'actions': os.path.join(numpy_dir, 'actions.npy'),
        'rewards': os.path.join(numpy_dir, 'rewards.npy'),
        'terminals': os.path.join(numpy_dir, 'terminals.npy'),
    }
    
    # 检查文件是否存在
    for name, path in file_mapping.items():
        if not os.path.exists(path):
            print(f"✗ 缺少文件: {path}")
            return False
    
    print("加载NumPy数据...")
    with h5py.File(output_path, 'w') as f:
        obs_group = f.create_group('observations')
        
        # 加载图像数据
        left_img = np.load(file_mapping['left_image'])
        right_img = np.load(file_mapping['right_image'])
        global_img = np.load(file_mapping['global_image'])
        joint = np.load(file_mapping['joint'])
        
        # 处理图像格式 (如果需要转换NCHW -> NHWC)
        if left_img.ndim == 4 and left_img.shape[1] == 3:
            print("转换图像格式: NCHW -> NHWC")
            left_img = np.transpose(left_img, (0, 2, 3, 1))
            right_img = np.transpose(right_img, (0, 2, 3, 1))
            global_img = np.transpose(global_img, (0, 2, 3, 1))
        
        # 保存为uint8
        if left_img.dtype != np.uint8:
            left_img = (left_img * 255).astype(np.uint8) if left_img.max() <= 1 else left_img.astype(np.uint8)
            right_img = (right_img * 255).astype(np.uint8) if right_img.max() <= 1 else right_img.astype(np.uint8)
            global_img = (global_img * 255).astype(np.uint8) if global_img.max() <= 1 else global_img.astype(np.uint8)
        
        obs_group.create_dataset('left_image', data=left_img)
        obs_group.create_dataset('right_image', data=right_img)
        obs_group.create_dataset('global_image', data=global_img)
        obs_group.create_dataset('joint', data=joint.astype(np.float32))
        
        # 加载其他数据
        f.create_dataset('actions', data=np.load(file_mapping['actions']).astype(np.float32))
        f.create_dataset('rewards', data=np.load(file_mapping['rewards']).astype(np.float32))
        f.create_dataset('terminals', data=np.load(file_mapping['terminals']).astype(bool))
    
    print(f"✓ 转换成功! 输出: {output_path}")
    validate_hdf5_format(output_path)
    return True


def print_usage():
    """打印使用说明"""
    print("""
多模态LAPO数据集工具

用法:
    1. 创建示例数据集进行测试:
       python create_dataset.py --create_sample --output sample_data.hdf5
    
    2. 验证HDF5数据集格式:
       python create_dataset.py --validate --hdf5_path your_data.hdf5
    
    3. 从NumPy文件转换:
       python create_dataset.py --convert --numpy_dir /path/to/numpy --output converted_data.hdf5

选项:
    --create_sample     创建示例数据集
    --validate          验证HDF5文件格式
    --convert           从NumPy转换
    --output            输出文件路径 (默认: sample_robot_data.hdf5)
    --hdf5_path         HDF5文件路径
    --numpy_dir         NumPy文件目录
    --n_samples         样本数量 (默认: 1000)
    """)


if __name__ == '__main__':
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='HDF5数据集工具')
    parser.add_argument('--create_sample', action='store_true', help='创建示例数据集')
    parser.add_argument('--validate', action='store_true', help='验证HDF5文件')
    parser.add_argument('--convert', action='store_true', help='从NumPy转换')
    parser.add_argument('--output', type=str, default='sample_robot_data.hdf5', help='输出路径')
    parser.add_argument('--hdf5_path', type=str, help='HDF5文件路径')
    parser.add_argument('--numpy_dir', type=str, help='NumPy文件目录')
    parser.add_argument('--n_samples', type=int, default=1000, help='样本数')
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_dataset(args.output, args.n_samples)
    elif args.validate:
        if not args.hdf5_path:
            print("错误: --validate 需要 --hdf5_path 参数")
            sys.exit(1)
        validate_hdf5_format(args.hdf5_path)
    elif args.convert:
        if not args.numpy_dir:
            print("错误: --convert 需要 --numpy_dir 参数")
            sys.exit(1)
        convert_from_numpy(args.numpy_dir, args.output)
    else:
        print_usage()
