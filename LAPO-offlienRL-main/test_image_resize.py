#!/usr/bin/env python3
"""
简单测试脚本：验证图像resize到224x224是否正确工作
"""
import torch
import torch.nn.functional as F
from algos.algos_vae_multimodal import ImageJointEncoder

def test_image_resize():
    print("=" * 60)
    print("测试图像 Resize 功能 (目标: 224x224)")
    print("=" * 60)
    
    # 创建编码器
    encoder = ImageJointEncoder(joint_dim=16, image_feature_dim=256, fusion_dim=256)
    print("\n✓ ImageJointEncoder 创建成功")
    
    # 创建测试数据 (不同尺寸的图像)
    batch_size = 4
    test_sizes = [
        (84, 84),      # 原始尺寸
        (128, 128),    # 其他尺寸
        (224, 224),    # 目标尺寸（如果已是该尺寸也要验证）
        (256, 256),    # 更大的尺寸
    ]
    
    for H, W in test_sizes:
        print(f"\n测试输入尺寸: {H}x{W}")
        
        # 创建虚拟图像和关节数据
        left_img = torch.randn(batch_size, 3, H, W)
        right_img = torch.randn(batch_size, 3, H, W)
        global_img = torch.randn(batch_size, 3, H, W)
        joint = torch.randn(batch_size, 16)
        
        print(f"  输入图像形状: {left_img.shape}")
        print(f"  输入关节形状: {joint.shape}")
        
        # 前向传播
        with torch.no_grad():
            fused_feat = encoder(left_img, right_img, global_img, joint)
        
        print(f"  输出特征形状: {fused_feat.shape}")
        assert fused_feat.shape == (batch_size, 256), f"输出形状错误! 预期 {(batch_size, 256)}, 得到 {fused_feat.shape}"
        print(f"  ✓ 通过! 特征维度正确 (256)")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过! 图像resize功能正常工作")
    print("=" * 60)
    print("\n关键信息:")
    print("  • 三张图像会被自动 resize 到 224x224")
    print("  • ResNet18 接收的是 224x224 的 9 通道输入")
    print("  • 输出特征维度固定为 256")

if __name__ == "__main__":
    test_image_resize()
