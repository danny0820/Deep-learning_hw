# Lab3 半監督式花朵分類 - 實作說明文件

## 目錄
1. [資料擴增 (Data Augmentation)](#1-資料擴增-data-augmentation)
2. [CNN 模型架構](#2-cnn-模型架構)
3. [損失函數與優化器 - 第一階段](#3-損失函數與優化器---第一階段監督式學習)
4. [訓練函數](#4-訓練函數)
5. [驗證函數](#5-驗證函數)
6. [偽標籤生成](#6-偽標籤生成)
7. [損失函數與優化器 - 第二階段](#7-損失函數與優化器---第二階段自我訓練)
8. [自我訓練迴圈](#8-自我訓練迴圈)

---

## 1. 資料擴增 (Data Augmentation)

### 1.1 為何需要資料擴增？
資料擴增是深度學習中非常重要的技術，主要目的包括：
- **增加訓練資料的多樣性**：透過各種變換產生更多訓練樣本
- **防止過擬合**：讓模型學習到更泛化的特徵，而非記憶特定圖片
- **提升模型穩健性**：使模型對圖片的旋轉、縮放、光線變化等更具容錯能力

### 1.2 訓練集資料擴增 (transforms_train)

```python
transforms_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

#### 各項轉換說明：

| 轉換 | 功能 | 為何使用 |
|------|------|----------|
| **Resize(256, 256)** | 將圖片統一調整為 256×256 | 確保所有圖片有相同的基礎尺寸，便於後續處理 |
| **RandomResizedCrop(224)** | 隨機裁切並縮放到 224×224<br>scale=(0.8, 1.0) | 1. 模擬不同距離拍攝的花朵<br>2. 強制模型學習局部特徵<br>3. 224×224 是 CNN 常用的輸入尺寸 |
| **RandomHorizontalFlip(p=0.5)** | 50% 機率水平翻轉 | 花朵左右對稱性高，翻轉不影響類別，可增加資料多樣性 |
| **RandomRotation(15)** | 隨機旋轉 ±15 度 | 模擬不同角度拍攝，花朵可能以任意角度出現在照片中 |
| **ColorJitter** | 調整亮度、對比、飽和度、色調 | 模擬不同光照條件和相機設定，提升對環境變化的容錯能力 |
| **ToImage()** | 轉換為 PyTorch Tensor | torchvision.transforms.v2 的新 API，取代舊的 ToTensor() |
| **ToDtype(float32, scale=True)** | 轉換資料型態並縮放到 [0,1] | scale=True 會將 [0,255] 縮放到 [0,1]，這是標準化前的必要步驟 |
| **Normalize** | 標準化<br>mean=[0.485, 0.456, 0.406]<br>std=[0.229, 0.224, 0.225] | 使用 ImageNet 統計值標準化，加速收斂並穩定訓練 |

### 1.3 驗證/測試集資料轉換 (transforms_test)

```python
transforms_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

#### 與訓練集的差異：

| 訓練集 | 測試集 | 原因 |
|--------|--------|------|
| RandomResizedCrop | **CenterCrop** | 測試時不需要隨機性，使用中心裁切確保一致性 |
| RandomHorizontalFlip | **無翻轉** | 測試時不需要資料擴增 |
| RandomRotation | **無旋轉** | 測試時保持原始方向 |
| ColorJitter | **無顏色調整** | 測試時使用原始顏色 |

**關鍵原則**：測試集只做必要的尺寸調整和標準化，不做隨機擴增。

---

## 2. CNN 模型架構

### 2.1 整體架構設計

```python
class YourCNNModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # 卷積層
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # 全連接層
        self.fc1 = nn.Linear(512 * 14 * 14, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
```

### 2.2 架構設計理念

#### 卷積層設計（4 個區塊）

```
輸入 (3, 224, 224)
    ↓ Conv1 (3→64) + BN + ReLU + MaxPool
(64, 112, 112)
    ↓ Conv2 (64→128) + BN + ReLU + MaxPool
(128, 56, 56)
    ↓ Conv3 (128→256) + BN + ReLU + MaxPool
(256, 28, 28)
    ↓ Conv4 (256→512) + BN + ReLU + MaxPool
(512, 14, 14)
    ↓ Flatten
98304 (512×14×14)
    ↓ FC1 + ReLU + Dropout
1024
    ↓ FC2 + ReLU + Dropout
512
    ↓ FC3 (輸出層)
5 (類別數)
```

#### 各層組件說明：

| 組件 | 功能 | 為何使用 |
|------|------|----------|
| **Conv2d** | 卷積層，提取特徵 | 自動學習圖片的空間特徵（邊緣、紋理、形狀） |
| **kernel_size=3, padding=1** | 3×3 卷積核，邊緣填充 1 | 保持特徵圖尺寸不變，3×3 是標準選擇（效能與效果平衡） |
| **BatchNorm2d** | 批次正規化 | 1. 加速訓練收斂<br>2. 減少內部協變量偏移<br>3. 允許使用較高學習率<br>4. 具有輕微正則化效果 |
| **ReLU** | 激活函數 | 引入非線性，使網路能學習複雜模式 |
| **MaxPool2d(2, 2)** | 最大池化，2×2 視窗 | 1. 降低特徵圖尺寸（減少參數量）<br>2. 增加感受野<br>3. 提供平移不變性 |
| **Dropout(0.5)** | 隨機丟棄 50% 神經元 | 防止過擬合，迫使網路學習更穩健的特徵 |

#### 通道數漸進式增加的原因：

- **3 → 64 → 128 → 256 → 512**
  - 早期層：低層特徵（邊緣、顏色），需要較少通道
  - 深層：高層特徵（複雜形狀、語義），需要更多通道來表達
  - 符合 CNN 的層次化特徵提取原理

### 2.3 前向傳播流程

```python
def forward(self, x):
    # 卷積區塊 1
    x = self.pool(F.relu(self.bn1(self.conv1(x))))
    # 卷積區塊 2
    x = self.pool(F.relu(self.bn2(self.conv2(x))))
    # 卷積區塊 3
    x = self.pool(F.relu(self.bn3(self.conv3(x))))
    # 卷積區塊 4
    x = self.pool(F.relu(self.bn4(self.conv4(x))))
    
    # 展平
    x = x.view(x.size(0), -1)
    
    # 全連接層
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = F.relu(self.fc2(x))
    x = self.dropout(x)
    out = self.fc3(x)
    return out
```

#### 操作順序的重要性：

1. **Conv → BN → ReLU → Pool**：這是標準順序
   - BatchNorm 在激活前正規化
   - ReLU 提供非線性
   - Pool 降維

2. **為何不在輸出層使用 Softmax？**
   - CrossEntropyLoss 內部已包含 LogSoftmax
   - 直接輸出 logits 更高效且數值更穩定

---

## 3. 損失函數與優化器 - 第一階段（監督式學習）

### 3.1 損失函數：CrossEntropyLoss

```python
criterion = nn.CrossEntropyLoss()
```

#### 為何選擇 CrossEntropyLoss？

| 特性 | 說明 |
|------|------|
| **適用場景** | 多分類問題（5 種花卉類別） |
| **數學原理** | 結合 LogSoftmax 和 Negative Log-Likelihood Loss |
| **優勢** | 1. 數值穩定（避免 log(0) 問題）<br>2. 自動處理 softmax<br>3. 對錯誤分類有較大懲罰 |
| **計算公式** | Loss = -log(exp(x[class]) / Σexp(x[i])) |

### 3.2 優化器：AdamW

```python
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
```

#### 參數說明：

| 參數 | 值 | 為何這樣設定 |
|------|----|----|
| **lr (學習率)** | 0.001 | 1. 適中的學習率，既不會太慢也不會不穩定<br>2. Adam 系列優化器的標準起始值<br>3. 可根據訓練過程調整（Learning Rate Scheduling） |
| **weight_decay** | 1e-4 (0.0001) | 1. L2 正則化係數<br>2. 防止權重過大，避免過擬合<br>3. 1e-4 是常用的經驗值 |

#### 為何選擇 AdamW 而非 Adam？

| 比較項目 | Adam | AdamW |
|---------|------|-------|
| **權重衰減實現** | 加在梯度上 | 直接作用於權重更新 |
| **正則化效果** | 與學習率相關聯 | 獨立於學習率 |
| **泛化能力** | 較弱 | **較強** ✓ |
| **理論基礎** | 原始實現 | 修正後的實現（更符合 L2 正則化原理） |

#### AdamW 的優勢：
1. **解耦權重衰減**：權重衰減不受學習率影響
2. **更好的泛化**：在許多任務上表現優於 Adam
3. **數值穩定**：減少訓練過程中的數值問題

---

## 4. 訓練函數

### 4.1 完整實作

```python
def train(input_data, model, criterion, optimizer, epoch=None, total_epochs=None):
    model.train()  # 設定為訓練模式
    loss_list = []
    total_count = 0
    acc_count = 0

    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        # 1. 清空梯度
        optimizer.zero_grad()
        # 2. 前向傳播
        outputs = model(images)
        # 3. 計算損失
        loss = criterion(outputs, labels)
        # 4. 反向傳播
        loss.backward()
        # 5. 更新權重
        optimizer.step()
        
        # 統計準確率
        _, predicted = torch.max(outputs.data, 1)
        total_count += labels.size(0)
        acc_count += (predicted == labels).sum().item()
        loss_list.append(loss.item())
```

### 4.2 五大步驟詳解

#### 步驟 1：`optimizer.zero_grad()` - 清空梯度

**為何需要？**
- PyTorch 預設會**累積梯度**
- 如果不清空，梯度會疊加到上一批次的梯度上
- 會導致錯誤的權重更新

**視覺化說明：**
```
批次 1: ∇W₁ = [0.5, 0.3, 0.2]
不清空 → 批次 2: ∇W₂ = [0.5, 0.3, 0.2] + [0.4, 0.2, 0.1] = [0.9, 0.5, 0.3] ✗ 錯誤！
清空   → 批次 2: ∇W₂ = [0.4, 0.2, 0.1] ✓ 正確！
```

#### 步驟 2：`outputs = model(images)` - 前向傳播

**發生了什麼？**
1. 圖片通過 Conv1 → BN1 → ReLU → Pool
2. 依序通過所有卷積層
3. 展平後通過全連接層
4. 得到 [batch_size, 5] 的 logits

**輸出格式：**
```python
outputs.shape = [128, 5]  # 假設 batch_size=128
# 範例：outputs[0] = [2.3, -1.2, 0.5, 1.8, -0.3]
# 表示第一張圖片對 5 個類別的預測分數
```

#### 步驟 3：`loss = criterion(outputs, labels)` - 計算損失

**CrossEntropyLoss 做了什麼？**
1. 對 outputs 套用 LogSoftmax
2. 使用真實標籤計算 Negative Log-Likelihood
3. 平均所有樣本的損失

**數學過程：**
```
假設：outputs[0] = [2.3, -1.2, 0.5, 1.8, -0.3], 真實標籤 = 0

1. Softmax: 
   p = exp([2.3, -1.2, 0.5, 1.8, -0.3]) / Σexp(...)
   = [0.65, 0.02, 0.11, 0.40, 0.05] (近似值)

2. Log: log(p[0]) = log(0.65) = -0.43

3. Negative: -(-0.43) = 0.43
```

#### 步驟 4：`loss.backward()` - 反向傳播

**自動微分魔法：**
- PyTorch 自動計算所有參數的梯度
- 使用鏈式法則從輸出層反向傳播到輸入層
- 梯度儲存在 `parameter.grad` 中

**計算的梯度：**
```python
# 每個參數都會有梯度
model.conv1.weight.grad  # Conv1 權重的梯度
model.fc1.bias.grad      # FC1 偏置的梯度
# ... 所有參數
```

#### 步驟 5：`optimizer.step()` - 更新權重

**AdamW 更新公式（簡化版）：**
```
for each parameter θ:
    1. 計算一階動量：m = β₁ * m + (1-β₁) * ∇θ
    2. 計算二階動量：v = β₂ * v + (1-β₂) * (∇θ)²
    3. 偏差修正：m̂ = m/(1-β₁ᵗ), v̂ = v/(1-β₂ᵗ)
    4. 權重更新：θ = θ - lr * m̂/√(v̂ + ε) - lr * λ * θ
                                                  ↑ weight decay
```

### 4.3 準確率統計

```python
_, predicted = torch.max(outputs.data, 1)
total_count += labels.size(0)
acc_count += (predicted == labels).sum().item()
```

**`torch.max(outputs.data, 1)` 詳解：**
```python
outputs = [[2.3, -1.2, 0.5, 1.8, -0.3],   # 樣本 0
           [0.2,  1.5, -0.8, 0.3,  2.1]]  # 樣本 1

values, indices = torch.max(outputs, 1)
# values  = [2.3, 2.1]    # 每個樣本的最大值
# indices = [0, 4]        # 最大值的索引（預測類別）
```

**準確率計算：**
```python
predicted = [0, 4]  # 預測類別
labels    = [0, 3]  # 真實類別
predicted == labels  # [True, False]
(predicted == labels).sum()  # 1（正確分類的數量）
```

---

## 5. 驗證函數

### 5.1 與訓練函數的差異

```python
def val(input_data, model, criterion, epoch=None, total_epochs=None):
    model.eval()  # ← 設定為評估模式
    
    with torch.no_grad():  # ← 不計算梯度
        for images, labels in pbar:
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            # 統計準確率和損失
```

### 5.2 關鍵差異說明

| 項目 | 訓練模式 | 評估模式 |
|------|---------|---------|
| **模式設定** | `model.train()` | `model.eval()` ✓ |
| **梯度計算** | 需要 | `torch.no_grad()` 不需要 ✓ |
| **Dropout** | 隨機丟棄 50% | **不丟棄**（使用所有神經元） |
| **BatchNorm** | 使用當前批次統計 | **使用訓練時的移動平均** |
| **反向傳播** | 有 | **無** |
| **權重更新** | 有 | **無** |

### 5.3 為何需要 `model.eval()` 和 `torch.no_grad()`？

#### `model.eval()` 的作用：

1. **Dropout 層行為改變：**
   ```python
   # 訓練時（train mode）
   x = Dropout(0.5)(x)  # 隨機丟棄 50%
   
   # 驗證時（eval mode）
   x = Dropout(0.5)(x)  # 不丟棄，但乘以 0.5（期望值校正）
   ```

2. **BatchNorm 層行為改變：**
   ```python
   # 訓練時
   mean, var = batch.mean(), batch.var()  # 使用當前批次統計
   
   # 驗證時
   mean, var = running_mean, running_var  # 使用訓練時累積的統計值
   ```

#### `torch.no_grad()` 的作用：

1. **節省記憶體**：不需要儲存中間計算結果
2. **加速計算**：跳過梯度計算
3. **防止錯誤**：避免在驗證時意外更新權重

**記憶體節省示例：**
```python
# 有梯度（訓練）：需要儲存所有中間結果
x1 = conv1(x)      # 儲存
x2 = relu(x1)      # 儲存
x3 = conv2(x2)     # 儲存
...

# 無梯度（驗證）：只需要最終結果
x1 = conv1(x)      # 不儲存
x2 = relu(x1)      # 不儲存
x3 = conv2(x2)     # 不儲存（可覆蓋）
```

---

## 6. 偽標籤生成

### 6.1 什麼是偽標籤（Pseudo-Labeling）？

**概念：**
- 使用訓練好的模型為**未標記資料**生成預測標籤
- 只保留**高信心度**的預測（confidence > threshold）
- 將這些帶有偽標籤的資料加入訓練集

**視覺化流程：**
```
未標記圖片 → 模型預測 → Softmax → 機率分布
                                    ↓
                           max_prob > 0.9?
                                    ↓
                           Yes: 加入訓練集（偽標籤）
                           No:  保留在未標記池
```

### 6.2 實作細節

```python
def get_pseudo_labels(model, threshold=0.9):
    model.eval()
    imgs_keep = []
    labels_keep = []
    remove_index = []
    soft_max = nn.Softmax(dim=1)
    
    with torch.no_grad():
        for idx, img in enumerate(unlabeled_set_list):
            img = img.to(device).unsqueeze(0)  # 加入 batch 維度
            
            # 1. 前向傳播
            outputs = model(img)
            
            # 2. 計算機率分布
            probs = soft_max(outputs)
            
            # 3. 獲取最大機率和預測類別
            max_prob, pred_label = torch.max(probs, 1)
            
            # 4. 篩選高信心預測
            if max_prob.item() >= threshold:
                imgs_keep.append(img.squeeze(0).cpu())
                labels_keep.append(pred_label.item())
                remove_index.append(idx)
```

### 6.3 關鍵參數：threshold = 0.9

#### 為何選擇 0.9？

| Threshold 值 | 效果 | 優缺點 |
|-------------|------|--------|
| **0.7-0.8** | 保留較多偽標籤 | ✓ 更多訓練資料<br>✗ 偽標籤可能不準確 |
| **0.9** ✓ | 平衡 | ✓ 較高準確率<br>✓ 適量資料<br>✓ 降低雜訊影響 |
| **0.95-0.99** | 只保留極高信心 | ✓ 偽標籤非常準確<br>✗ 可用資料太少 |

#### 機率分布範例：

```python
# 範例 1：高信心預測（max_prob = 0.95）
probs = [0.95, 0.02, 0.01, 0.01, 0.01]  # 預測類別 0
→ 通過 threshold，加入訓練集 ✓

# 範例 2：低信心預測（max_prob = 0.45）
probs = [0.45, 0.30, 0.15, 0.08, 0.02]  # 預測類別 0
→ 不通過 threshold，保留在未標記池 ✗

# 範例 3：中等信心（max_prob = 0.85）
probs = [0.85, 0.08, 0.04, 0.02, 0.01]  # 預測類別 0
→ 不通過 threshold (0.9)，保留在未標記池 ✗
```

### 6.4 Softmax 的作用

```python
soft_max = nn.Softmax(dim=1)
probs = soft_max(outputs)
```

**Softmax 將 logits 轉換為機率：**
```python
# 輸入 logits（原始預測分數）
outputs = [[2.3, -1.2, 0.5, 1.8, -0.3]]

# Softmax 計算
exp_outputs = exp([2.3, -1.2, 0.5, 1.8, -0.3])
            = [9.97, 0.30, 1.65, 6.05, 0.74]

sum_exp = 9.97 + 0.30 + 1.65 + 6.05 + 0.74 = 18.71

probs = exp_outputs / sum_exp
      = [0.533, 0.016, 0.088, 0.323, 0.040]
      
# 特性：所有機率和為 1
sum(probs) = 1.0 ✓
```

**為何需要 Softmax？**
- Logits 可以是任意實數，難以解釋
- Softmax 將其轉換為 [0, 1] 範圍的機率
- 便於設定 threshold 進行篩選

---

## 7. 損失函數與優化器 - 第二階段（自我訓練）

### 7.1 為何需要調整優化器？

在自我訓練階段，我們面臨不同的挑戰：
1. 模型已經過第一階段訓練，具有一定能力
2. 新加入的偽標籤可能包含雜訊
3. 需要更謹慎地微調模型

### 7.2 優化器配置

```python
criterion = nn.CrossEntropyLoss()  # 損失函數不變
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
```

#### 學習率降低：0.001 → 0.0005

| 階段 | 學習率 | 原因 |
|------|--------|------|
| **第一階段**<br>（監督式學習） | 0.001 | 1. 模型從零開始<br>2. 需要較大步伐探索<br>3. 資料標籤準確 |
| **第二階段**<br>（自我訓練） | 0.0005<br>（減半） | 1. 模型已預訓練<br>2. 偽標籤可能有誤<br>3. **較小步伐更穩定**<br>4. 避免遺忘已學習的知識 |

#### 學習率的影響：

**視覺化比較：**
```
高學習率 (0.001):
Loss ─┐    ┌─┐
      │  ┌─┘ │  大幅震盪
      └──┘   └─┐
               └─

低學習率 (0.0005):
Loss ─┐
      │  ┌─┐
      └──┘ └─┐  平穩下降
             └─
```

### 7.3 為何偽標籤需要更小的學習率？

#### 雜訊敏感度：

**範例場景：**
```
真實標籤：Daisy (雛菊)
偽標籤：Tulip (鬱金香) ✗ 錯誤！

使用大學習率 (0.001):
→ 模型會大幅調整權重以擬合這個錯誤標籤
→ 可能遺忘正確的知識

使用小學習率 (0.0005):
→ 權重調整較小
→ 即使偽標籤錯誤，影響也較有限
→ 保留更多原有知識
```

#### 數學角度：

```
權重更新：θ_new = θ_old - lr * ∇L

假設梯度 ∇L = 0.5（由錯誤偽標籤產生）

大學習率：
θ_new = θ - 0.001 * 0.5 = θ - 0.0005  # 較大變化

小學習率：
θ_new = θ - 0.0005 * 0.5 = θ - 0.00025  # 較小變化
```

---

## 8. 自我訓練迴圈

### 8.1 整體策略

```python
for epoch in range(n_epochs):
    # 每 N (10) 個 epochs 生成偽標籤
    if (epoch + 1) % N == 0 and len(unlabeled_set_list) > 0:
        pseudo_dataset, taken = get_pseudo_labels(model, threshold=0.9)
        if taken > 0:
            all_pseudo_datasets.append(pseudo_dataset)
            current_train_dataset = ConcatDataset([train_set] + all_pseudo_datasets)
            train_loader_ssl = DataLoader(current_train_dataset, shuffle=True, **loader_kwargs)
    
    # 訓練和驗證
    train_acc, train_loss = train(train_loader_ssl, model, criterion, optimizer)
    valid_acc, valid_loss = val(val_loader, model, criterion)
```

### 8.2 設計原理

#### 8.2.1 為何每 10 epochs 生成偽標籤？

| 頻率 | 優點 | 缺點 | 結論 |
|------|------|------|------|
| **每 1 epoch** | 最新的模型預測 | 1. 計算開銷大<br>2. 模型可能還不穩定<br>3. 偽標籤品質參差不齊 | ✗ 不推薦 |
| **每 10 epochs** ✓ | 1. 模型有足夠時間學習<br>2. 預測更穩定<br>3. 計算效率高 | 可能錯過一些改進 | ✓ **平衡選擇** |
| **每 20+ epochs** | 偽標籤品質高 | 1. 更新太慢<br>2. 可能錯過學習機會 | ✗ 太保守 |

#### 8.2.2 累積式策略 vs 替換式策略

**我們的實作（累積式）：**
```python
all_pseudo_datasets.append(pseudo_dataset)  # 累積
current_train_dataset = ConcatDataset([train_set] + all_pseudo_datasets)
```

**兩種策略比較：**

| 策略 | 實作 | 優點 | 缺點 |
|------|------|------|------|
| **累積式**<br>（我們使用） ✓ | 保留所有歷史偽標籤 | 1. 訓練資料持續增加<br>2. 利用早期高信心預測<br>3. 資料多樣性更高 | 1. 記憶體需求增加<br>2. 可能包含過時預測 |
| **替換式** | 每次只用最新偽標籤 | 1. 記憶體需求固定<br>2. 使用最新模型預測 | 1. 丟棄早期正確預測<br>2. 訓練資料較少 |

### 8.3 資料集演變過程

**視覺化時間軸：**
```
Epoch 1-9:
訓練集 = [原始已標記資料 (1200 張)]

Epoch 10:
生成偽標籤 → 新增 150 張高信心預測
訓練集 = [原始 (1200) + 偽標籤₁ (150)] = 1350 張

Epoch 11-19:
訓練集 = [原始 (1200) + 偽標籤₁ (150)] = 1350 張

Epoch 20:
生成偽標籤 → 新增 120 張高信心預測
訓練集 = [原始 (1200) + 偽標籤₁ (150) + 偽標籤₂ (120)] = 1470 張

Epoch 30:
訓練集 = [原始 (1200) + 偽標籤₁ (150) + 偽標籤₂ (120) + 偽標籤₃ (100)] = 1570 張
```

### 8.4 未標記資料池的動態變化

```python
# 從未標記池中移除已使用的資料
for i in reversed(remove_index):
    del unlabeled_set_list[i]
```

**為何要移除？**
1. **避免重複使用**：已標記的資料不應再被選取
2. **提高效率**：減少每次迭代的計算量
3. **確保多樣性**：每次選取不同的樣本

**資料池大小變化：**
```
初始：unlabeled_set_list = 1202 張

Epoch 10: 選取 150 張 → 剩餘 1052 張
Epoch 20: 選取 120 張 → 剩餘 932 張
Epoch 30: 選取 100 張 → 剩餘 832 張
...
```

### 8.5 ConcatDataset 的作用

```python
current_train_dataset = ConcatDataset([train_set] + all_pseudo_datasets)
```

**ConcatDataset 做了什麼？**
- 將多個資料集合併成一個
- 不複製資料，只是創建索引映射
- 支援 DataLoader 的所有操作（shuffle, batch, 等）

**索引映射範例：**
```python
train_set: 0-1199 (1200 張)
pseudo_dataset₁: 1200-1349 (150 張)
pseudo_dataset₂: 1350-1469 (120 張)

current_train_dataset[0] → train_set[0]
current_train_dataset[1200] → pseudo_dataset₁[0]
current_train_dataset[1350] → pseudo_dataset₂[0]
```

---

## 總結

### 完整流程圖

```
┌─────────────────────────────────────────────────────────┐
│ 第一階段：監督式學習                                    │
├─────────────────────────────────────────────────────────┤
│ 1. 資料擴增 (transforms_train)                          │
│    ↓                                                     │
│ 2. CNN 模型訓練 (4 Conv blocks + 3 FC layers)          │
│    ↓                                                     │
│ 3. 使用 CrossEntropyLoss + AdamW (lr=0.001)            │
│    ↓                                                     │
│ 4. 訓練 30 epochs → 儲存最佳模型                        │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│ 第二階段：自我訓練（Semi-Supervised Learning）          │
├─────────────────────────────────────────────────────────┤
│ 每 10 epochs:                                           │
│   ├─ 生成偽標籤 (threshold=0.9)                        │
│   ├─ 累積到訓練集                                       │
│   └─ 重新訓練                                           │
│                                                          │
│ 使用：                                                  │
│   - CrossEntropyLoss                                    │
│   - AdamW (lr=0.0005, 降低學習率)                      │
│   - 累積式資料策略                                      │
└─────────────────────────────────────────────────────────┘
                           ↓
                    最終模型 + 預測
```

### 關鍵設計決策總覽

| 組件 | 選擇 | 原因 |
|------|------|------|
| **資料擴增** | 訓練：多種隨機轉換<br>測試：只有基本轉換 | 訓練時增加多樣性，測試時保持一致性 |
| **模型架構** | 4 Conv + 3 FC<br>BatchNorm + Dropout | 適中的複雜度，平衡效能與過擬合 |
| **損失函數** | CrossEntropyLoss | 多分類標準選擇，數值穩定 |
| **優化器** | AdamW | 比 Adam 有更好的泛化能力 |
| **學習率** | 0.001 → 0.0005 | 第二階段降低，適應偽標籤雜訊 |
| **偽標籤閾值** | 0.9 | 平衡準確率與資料量 |
| **更新頻率** | 每 10 epochs | 給模型足夠時間穩定 |
| **資料策略** | 累積式 | 最大化訓練資料利用 |

這份文件詳細說明了 Lab3 實作的每個環節，包括技術選擇的理由、數學原理和實際效果。
