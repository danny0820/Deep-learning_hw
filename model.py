import torch
import torch.nn as nn
from torchvision.models import resnet50

class YourCNNModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # 使用 ResNet50 作為骨幹（從頭訓練，不使用預訓練權重）
        # ResNet50 比 ResNet18 更深，效果更好，且比 EfficientNet 更容易訓練
        resnet = resnet50(weights=None)  # weights=None 表示從頭訓練

        # 移除最後的全連接層
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # 添加自定義的分類器
        # ResNet50 的特徵維度是 2048
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        assert isinstance(x, torch.Tensor), "Input should be a torch Tensor"
        assert x.dim() == 4, "Input should be NHWC format"
        x = self.features(x)  # 特徵提取
        out = self.classifier(x)  # 分類
        return out
