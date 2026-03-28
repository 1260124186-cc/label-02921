"""
手写体识别模型 - 使用 PyTorch 实现 CNN
支持数字、字母、中文等多种手写体识别
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HandwritingCNN(nn.Module):
    """卷积神经网络用于手写体识别"""

    def __init__(self, num_classes=47):
        super(HandwritingCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout1(x)
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class CharacterSet:
    """字符集定义"""

    @staticmethod
    def get_mapping(dataset_type='emnist_balanced'):
        """获取字符映射"""
        if dataset_type == 'digits':
            return {i: str(i) for i in range(10)}
        elif dataset_type == 'emnist_balanced':
            mapping = {}
            for i in range(10):
                mapping[i] = str(i)
            for i in range(10, 36):
                mapping[i] = chr(ord('A') + i - 10)
            for i in range(36, 47):
                mapping[i] = chr(ord('a') + i - 36)
            return mapping
        elif dataset_type == 'chinese':
            return {
                0: '一', 1: '二', 2: '三', 3: '四', 4: '五',
                5: '六', 6: '七', 7: '八', 8: '九', 9: '十'
            }
        else:
            raise ValueError(f"未知的数据集类型: {dataset_type}")

    @staticmethod
    def get_num_classes(dataset_type='emnist_balanced'):
        """获取类别数量"""
        if dataset_type == 'digits':
            return 10
        elif dataset_type == 'emnist_balanced':
            return 47
        elif dataset_type == 'chinese':
            return 10
        else:
            raise ValueError(f"未知的数据集类型: {dataset_type}")
