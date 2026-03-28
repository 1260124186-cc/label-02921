"""
训练手写数字/字母识别模型
"""
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import HandwritingCNN
import os


def train_model(epochs=5, batch_size=64, learning_rate=0.001, dataset_type='emnist_letters'):
    """训练模型
    dataset_type: 'mnist' | 'emnist_letters' | 'emnist_byclass' | 'emnist_digits'
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 根据数据集类型设置参数
    dataset_configs = {
        'mnist': {'num_classes': 10, 'dataset_class': datasets.MNIST, 'split': None},
        'emnist_digits': {'num_classes': 10, 'dataset_class': datasets.EMNIST, 'split': 'digits'},
        'emnist_letters': {'num_classes': 26, 'dataset_class': datasets.EMNIST, 'split': 'letters'},
        'emnist_byclass': {'num_classes': 62, 'dataset_class': datasets.EMNIST, 'split': 'byclass'}
    }

    if dataset_type not in dataset_configs:
        raise ValueError(f"不支持的数据集类型: {dataset_type}")

    config = dataset_configs[dataset_type]
    num_classes = config['num_classes']
    print(f"使用数据集: {dataset_type}, 类别数量: {num_classes}")

    # 数据预处理 - EMNIST需要特殊处理（旋转图像）
    if dataset_type.startswith('emnist'):
        transform = transforms.Compose([
            lambda img: img.rotate(-90, expand=True),
            lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
            transforms.ToTensor(),
            transforms.Normalize((0.1751,), (0.3231,))  # EMNIST的均值和标准差
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差
        ])

    # 加载数据集
    if dataset_type.startswith('emnist'):
        train_dataset = config['dataset_class'](
            root='./data', split=config['split'], train=True, download=True, transform=transform
        )
        test_dataset = config['dataset_class'](
            root='./data', split=config['split'], train=False, download=True, transform=transform
        )
    else:
        train_dataset = config['dataset_class'](
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = config['dataset_class'](
            root='./data', train=False, download=True, transform=transform
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = HandwritingCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f} - Acc: {100.*correct/total:.2f}%")

        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
        print(f"         Test Acc: {100.*test_correct/test_total:.2f}%")

    os.makedirs('saved_models', exist_ok=True)
    model_path = f'saved_models/handwriting_model_{dataset_type}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存: {model_path}")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='训练手写数字/字母识别模型')
    parser.add_argument('--epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=64, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--dataset', type=str, default='emnist_letters',
                        choices=['mnist', 'emnist_digits', 'emnist_letters', 'emnist_byclass'],
                        help='数据集类型')
    args = parser.parse_args()

    train_model(epochs=args.epochs, batch_size=args.batch_size,
                learning_rate=args.lr, dataset_type=args.dataset)
