import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import YourCNNModel
from dataset import FlowerData, get_transforms, CLASS_NAMES

# 設定
DATA_FOLDER = 'Lab3_data_flower_2025'
BATCH_SIZE = 128
NUM_WORKERS = 2
MAX_EPOCHS = 30
LOG_INTERVAL = 10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
CHECKPOINT_PATH = 'supervised.pt'

# 設定裝置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 載入資料
print("Loading data...")
transforms_train, transforms_test = get_transforms()

train_set = FlowerData(DATA_FOLDER, split='train', mode='train',
                       transform=transforms_train, use_unlabel=False)
valid_set = FlowerData(DATA_FOLDER, split='val', mode='train',
                       transform=transforms_test, use_unlabel=False)

print(f"Train set size: {len(train_set)}")
print(f"Valid set size: {len(valid_set)}")

# 建立 DataLoader
loader_kwargs = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                     pin_memory=torch.cuda.is_available())
if NUM_WORKERS > 0:
    loader_kwargs["persistent_workers"] = True

train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
val_loader = DataLoader(valid_set, shuffle=False, **loader_kwargs)

# 建立模型
print("Creating model...")
num_classes = len(CLASS_NAMES)
model = YourCNNModel(num_classes=num_classes)
model = model.to(device)

# 損失函數和優化器
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


def train_epoch(input_data, model, criterion, optimizer, epoch=None, total_epochs=None):
    """訓練一個 epoch"""
    model.train()
    loss_list = []
    total_count = 0
    acc_count = 0

    desc = f"Train | epoch {epoch}/{total_epochs}" if epoch is not None else "Train"
    pbar = tqdm(input_data, desc=desc, leave=False)

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # Forward, backward and optimize
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 計算準確率
        _, predicted = torch.max(outputs.data, 1)
        total_count += labels.size(0)
        acc_count += (predicted == labels).sum().item()
        loss_list.append(loss.item())

        running_acc = acc_count / total_count if total_count else 0.0
        lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{running_acc:.4f}", lr=f"{lr:.6f}")

    acc = acc_count / total_count if total_count else 0.0
    loss = sum(loss_list) / len(loss_list) if loss_list else 0.0
    return acc, loss


def validate(input_data, model, criterion, epoch=None, total_epochs=None):
    """驗證模型"""
    model.eval()
    loss_list = []
    total_count = 0
    acc_count = 0

    desc = f"Val   | epoch {epoch}/{total_epochs}" if epoch is not None else "Val"
    pbar = tqdm(input_data, desc=desc, leave=False)

    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)

            total_count += labels.size(0)
            acc_count += (predicted == labels).sum().item()
            loss_list.append(loss.item())

            running_acc = acc_count / total_count if total_count else 0.0
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{running_acc:.4f}")

    acc = acc_count / total_count if total_count else 0.0
    loss = sum(loss_list) / len(loss_list) if loss_list else 0.0
    return acc, loss


# 訓練主迴圈
print("Starting training...")
train_acc_list = []
train_loss_list = []
val_acc_list = []
val_loss_list = []

best_val_acc = 0.0

for epoch in range(1, MAX_EPOCHS + 1):
    train_acc, train_loss = train_epoch(train_loader, model, criterion, optimizer,
                                        epoch=epoch, total_epochs=MAX_EPOCHS)
    val_acc, val_loss = validate(val_loader, model, criterion,
                                  epoch=epoch, total_epochs=MAX_EPOCHS)

    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss)
    val_acc_list.append(val_acc)
    val_loss_list.append(val_loss)

    if epoch % LOG_INTERVAL == 0:
        lr = optimizer.param_groups[0]['lr']
        print('=' * 20, f'Epoch {epoch}/{MAX_EPOCHS}', '=' * 20)
        print('Train Acc: {:.6f} | Train Loss: {:.6f}'.format(train_acc, train_loss))
        print('  Val Acc: {:.6f} |   Val Loss: {:.6f} | LR: {:.6f}'.format(val_acc, val_loss, lr))

    # 儲存最佳模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f'[{epoch:03d}/{MAX_EPOCHS}] Saved best model with val acc {best_val_acc:.6f}')

print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.6f}")

# 視覺化訓練過程
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(len(train_loss_list)), train_loss_list, label='train')
plt.plot(range(len(val_loss_list)), val_loss_list, label='val')
plt.legend()
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(range(len(train_acc_list)), train_acc_list, label='train')
plt.plot(range(len(val_acc_list)), val_acc_list, label='val')
plt.legend()
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.savefig('training_history.png')
print("Saved training history plot to training_history.png")
plt.show()
