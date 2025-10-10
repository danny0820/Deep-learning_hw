import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import os.path as osp
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
from torchvision.transforms import v2 as transforms

# ============================================
# 定義常數和類別
# ============================================
CLASS_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
data_folder = 'Lab3_data_flower_2025'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = len(CLASS_NAMES)

# ============================================
# 定義數據集類別
# ============================================
class FlowerData(Dataset):
    def __init__(self, root, split='train', mode='train', transform=None, use_unlabel=False):
        self.root = Path(root)
        self.split = split
        self.mode = mode
        self.transform = transform
        self.use_unlabel = use_unlabel

        self.paths = []
        self.labels = []
        self.rel_paths = []

        # Load data from unified CSV files
        if split == 'train' and use_unlabel:
            csv_file = self.root / 'unlabeled_train.csv'
        elif split == 'train':
            csv_file = self.root / 'train.csv'
        elif split == 'val':
            csv_file = self.root / 'val.csv'
        else:  # test
            csv_file = self.root / 'test.csv'

        # Read CSV file using pandas for better handling
        df = pd.read_csv(csv_file)

        for _, row in df.iterrows():
            file_path = self.root / row['file_name']
            self.paths.append(file_path)
            self.rel_paths.append(row['file_name'])

            # Handle labels
            if split == 'test' or (split == 'train' and use_unlabel):
                pass
            else:
                if pd.isna(row['groundtruth']) or row['groundtruth'] == '':
                    self.labels.append(-1)
                else:
                    self.labels.append(CLASS_TO_IDX[row['groundtruth']])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img_path = self.paths[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.mode == 'test' or (self.split == 'train' and self.use_unlabel):
            return img
        label = int(self.labels[index])
        return img, torch.tensor(label, dtype=torch.long)

# ============================================
# 定義模型結構
# ============================================
class BasicBlock(nn.Module):
    """
    ResNet BasicBlock
    """
    expansion = 1
    def __init__(self, in_c, out_c, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


def make_layer(in_c, out_c, blocks, stride):
    """
    創建 ResNet layer
    """
    down = None
    if stride != 1 or in_c != out_c:
        down = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride, bias=False),
            nn.BatchNorm2d(out_c)
        )
    layers = [BasicBlock(in_c, out_c, stride, down)]
    for _ in range(1, blocks):
        layers.append(BasicBlock(out_c, out_c))
    return nn.Sequential(*layers)

class ResNet34(nn.Module):
    """
    ResNet34 架構
    Layer 結構：[3, 4, 6, 3] blocks
    相比 ResNet18 的 [2, 2, 2, 2]，ResNet34 有更深的網路
    """
    def __init__(self, num_classes=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # ResNet34: [3, 4, 6, 3] blocks (vs ResNet18: [2, 2, 2, 2])
        self.layer1 = make_layer(64,   64,  blocks=3, stride=1)
        self.layer2 = make_layer(64,   128, blocks=4, stride=2)
        self.layer3 = make_layer(128,  256, blocks=6, stride=2)
        self.layer4 = make_layer(256,  512, blocks=3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

        # Kaiming 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        return self.head(x)

class YourCNNModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.model = ResNet34(num_classes=num_classes)

    def forward(self, x):
        assert isinstance(x, torch.Tensor), "Input should be a torch Tensor"
        assert x.dim() == 4, "Input should be NHWC format"
        out = self.model(x)
        return out

# ============================================
# 評估函數
# ============================================
def evaluate_model(model, data_loader, criterion, device, dataset_name="Test"):
    """
    評估模型在指定數據集上的表現
    """
    model.eval()

    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    all_predictions = []
    all_labels = []

    print(f"\n{'='*20} Evaluating on {dataset_name} Set {'='*20}")

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc=f"Evaluating {dataset_name}", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)

            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            total_loss += loss.item()

            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"Overall {dataset_name} Accuracy: {accuracy:.4f} ({total_correct}/{total_samples})")
    print(f"Overall {dataset_name} Loss: {avg_loss:.4f}")
    print(f"{'='*60}")

    print(f"\n{'Class':<15} {'Accuracy':<12} {'Correct/Total':<15}")
    print("-" * 45)
    for i, class_name in enumerate(CLASS_NAMES):
        if class_total[i] > 0:
            class_acc = class_correct[i] / class_total[i]
            print(f"{class_name:<15} {class_acc:.4f}       {class_correct[i]}/{class_total[i]}")
        else:
            print(f"{class_name:<15} N/A          0/0")

    return accuracy, avg_loss, class_correct, class_total, all_predictions, all_labels


# ============================================
# 主程式
# ============================================
if __name__ == "__main__":
    print("="*60)
    print("🌸 Flower Classification Model Evaluation")
    print("="*60)
    print(f"Device: {device}")

    # 定義資料轉換
    transforms_test = transforms.Compose([
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 載入數據集
    print("\nLoading datasets...")
    train_set = FlowerData(data_folder, split='train', mode='train', transform=transforms_test, use_unlabel=False)
    valid_set = FlowerData(data_folder, split='val', mode='train', transform=transforms_test, use_unlabel=False)

    batch_size = 64
    num_workers = 0

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=torch.cuda.is_available())

    print(f"✅ Train set: {len(train_set)} images")
    print(f"✅ Validation set: {len(valid_set)} images")

    # 載入模型
    print("\n" + "="*60)
    print("Loading trained model from: supurvised.pt")
    print("="*60)

    model = YourCNNModel(num_classes=num_classes)
    model = model.to(device)

    try:
        checkpoint = torch.load('supurvised.pt', map_location=device)
        model.load_state_dict(checkpoint)
        print("✅ Model loaded successfully!")
    except FileNotFoundError:
        print("❌ Error: 'supurvised.pt' not found!")
        print("Please check if the file exists in the current directory.")
        exit(1)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        exit(1)

    # 定義損失函數
    criterion = nn.CrossEntropyLoss()

    # 在驗證集上測試
    print("\n" + "="*60)
    print("Testing on Validation Set")
    print("="*60)

    val_acc, val_loss, val_class_correct, val_class_total, val_preds, val_labels = evaluate_model(
        model, val_loader, criterion, device, dataset_name="Validation"
    )

    # 在訓練集上測試
    print("\n" + "="*60)
    print("Testing on Training Set (Check Overfitting)")
    print("="*60)

    train_acc, train_loss, train_class_correct, train_class_total, train_preds, train_labels = evaluate_model(
        model, train_loader, criterion, device, dataset_name="Training"
    )

    # 總結報告
    print("\n" + "="*60)
    print("📊 FINAL SUMMARY")
    print("="*60)
    print(f"Training Accuracy:   {train_acc:.4f} | Loss: {train_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f} | Loss: {val_loss:.4f}")
    print(f"\nOverfitting Check:")
    gap = train_acc - val_acc
    if gap > 0.1:
        print(f"⚠️  High overfitting detected! (Gap: {gap:.4f})")
    elif gap > 0.05:
        print(f"⚠️  Moderate overfitting (Gap: {gap:.4f})")
    else:
        print(f"✅ Good generalization (Gap: {gap:.4f})")
    print("="*60)

    # 生成混淆矩陣
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, classification_report
        import seaborn as sns

        print("\n" + "="*60)
        print("📈 Generating Confusion Matrix")
        print("="*60)

        cm = confusion_matrix(val_labels, val_preds)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_NAMES,
                    yticklabels=CLASS_NAMES)
        plt.title('Confusion Matrix - Validation Set', fontsize=16)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        print("✅ Confusion matrix saved as 'confusion_matrix.png'")

        print("\n" + "="*60)
        print("📋 Classification Report")
        print("="*60)
        print(classification_report(val_labels, val_preds,
                                    target_names=CLASS_NAMES,
                                    digits=4))

    except ImportError:
        print("\n⚠️  sklearn or seaborn not available, skipping confusion matrix")
    except Exception as e:
        print(f"\n⚠️  Error generating confusion matrix: {e}")

    print("\n✅ Model evaluation completed!")
