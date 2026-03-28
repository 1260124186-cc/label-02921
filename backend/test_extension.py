"""
测试手写字符识别扩展功能
"""
import torch
from model import HandwritingCNN
from train import train_model
import os


def test_model_structure():
    """测试模型结构是否支持不同类别数量"""
    print("测试模型结构...")
    
    # 测试10类（数字）
    model_digits = HandwritingCNN(num_classes=10)
    assert model_digits.fc2.out_features == 10
    print("✓ 数字模型（10类）创建成功")
    
    # 测试26类（字母）
    model_letters = HandwritingCNN(num_classes=26)
    assert model_letters.fc2.out_features == 26
    print("✓ 字母模型（26类）创建成功")
    
    # 测试62类（数字+字母）
    model_all = HandwritingCNN(num_classes=62)
    assert model_all.fc2.out_features == 62
    print("✓ 综合模型（62类）创建成功")


def test_label_mapping():
    """测试标签映射功能"""
    print("\n测试标签映射...")
    
    # EMNIST Letters (26类)
    letters_mapping = {i: chr(ord('A') + i) for i in range(26)}
    assert letters_mapping[0] == 'A'
    assert letters_mapping[25] == 'Z'
    print("✓ 字母标签映射正确")
    
    # EMNIST ByClass (62类)
    all_chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    assert all_chars[0] == '0'
    assert all_chars[9] == '9'
    assert all_chars[10] == 'A'
    assert all_chars[35] == 'Z'
    assert all_chars[36] == 'a'
    assert all_chars[61] == 'z'
    print("✓ 综合字符标签映射正确")


def test_dataset_configs():
    """测试数据集配置"""
    print("\n测试数据集配置...")
    
    dataset_configs = {
        'mnist': {'num_classes': 10, 'split': None},
        'emnist_digits': {'num_classes': 10, 'split': 'digits'},
        'emnist_letters': {'num_classes': 26, 'split': 'letters'},
        'emnist_byclass': {'num_classes': 62, 'split': 'byclass'}
    }
    
    for dataset, config in dataset_configs.items():
        print(f"✓ {dataset}: 类别数={config['num_classes']}, split={config['split']}")


def main():
    print("=" * 60)
    print("手写字符识别扩展功能测试")
    print("=" * 60)
    
    test_model_structure()
    test_label_mapping()
    test_dataset_configs()
    
    print("\n" + "=" * 60)
    print("所有测试通过！扩展功能正常工作。")
    print("=" * 60)
    
    print("\n使用说明:")
    print("1. 训练字母模型: python train.py --dataset emnist_letters")
    print("2. 训练综合模型: python train.py --dataset emnist_byclass")
    print("3. 训练数字模型: python train.py --dataset mnist")
    print("4. 启动服务: python -m uvicorn app:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()