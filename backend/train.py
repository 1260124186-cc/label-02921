"""
训练手写体识别模型
支持数字（MNIST）、字母（EMNIST）、中文等多种数据集
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from model import HandwritingCNN, CharacterSet
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class ChineseStrokeDataset(Dataset):
    """简单的中文笔画数据集生成器"""
    
    def __init__(self, train=True, transform=None, samples_per_class=100):
        self.transform = transform
        self.samples_per_class = samples_per_class
        self.characters = ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十']
        self.data = []
        self.targets = []
        self.train = train
        self._generate_data()
    
    def _generate_data(self):
        """生成中文笔画数据"""
        font_sizes = [36, 40, 44, 48]
        x_offsets = [-2, -1, 0, 1, 2]
        y_offsets = [-2, -1, 0, 1, 2]
        
        for char_idx, char in enumerate(self.characters):
            for i in range(self.samples_per_class):
                img = Image.new('L', (28, 28), color=0)
                draw = ImageDraw.Draw(img)
                
                try:
                    font_size = np.random.choice(font_sizes)
                    font = ImageFont.truetype('/System/Library/Fonts/PingFang.ttc', font_size)
                except:
                    try:
                        font = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', font_size)
                    except:
                        font = ImageFont.load_default()
                
                x_offset = np.random.choice(x_offsets)
                y_offset = np.random.choice(y_offsets)
                
                bbox = draw.textbbox((0, 0), char, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                x = (28 - text_width) // 2 + x_offset
                y = (28 - text_height) // 2 + y_offset
                
                draw.text((x, y), char, fill=255, font=font)
                
                if self.train and np.random.random() > 0.5:
                    angle = np.random.randint(-10, 10)
                    img = img.rotate(angle, fillcolor=0)
                
                img_array = np.array(img)
                self.data.append(img_array)
                self.targets.append(char_idx)
        
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx], mode='L')
        target = self.targets[idx]
        
        if self.transform:
            img = self.transform(img)
        
        return img, target


def train_model(dataset_type='emnist_balanced', epochs=10, batch_size=64, learning_rate=0.001):
    """训练模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"数据集类型: {dataset_type}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if dataset_type == 'digits':
        train_dataset = datasets.MNIST(
            root='./data', train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root='./data', train=False, download=True, transform=transform
        )
    elif dataset_type == 'emnist_balanced':
        train_dataset = datasets.EMNIST(
            root='./data', split='balanced', train=True, download=True, 
            transform=transform
        )
        test_dataset = datasets.EMNIST(
            root='./data', split='balanced', train=False, download=True, 
            transform=transform
        )
    elif dataset_type == 'chinese':
        train_dataset = ChineseStrokeDataset(train=True, transform=transform, samples_per_class=200)
        test_dataset = ChineseStrokeDataset(train=False, transform=transform, samples_per_class=50)
    else:
        raise ValueError(f"未知的数据集类型: {dataset_type}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_classes = CharacterSet.get_num_classes(dataset_type)
    model = HandwritingCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"类别数量: {num_classes}")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

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
    print(f"模型已保存至: {model_path}")
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='训练手写体识别模型')
    parser.add_argument('--dataset', type=str, default='emnist_balanced', 
                       choices=['digits', 'emnist_balanced', 'chinese'],
                       help='数据集类型: digits(数字), emnist_balanced(数字+字母), chinese(中文)')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=64, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    args = parser.parse_args()

    train_model(dataset_type=args.dataset, epochs=args.epochs, 
                batch_size=args.batch_size, learning_rate=args.lr)
