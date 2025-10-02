# Lab3 半監督式花朵分類 - 實作說明文件

## 目錄
1. [資料擴增 (Data Augmentation)](#1-資料擴增-data-augmentation)
2. [CNN 模型架構](#2-cnn-模型架構)
3. [損失函數與優化器 - 第一階段](#3-損失函數與優化器---第一階段監督式學習)
4. [學習率調度器](#4-學習率調度器)
5. [訓練函數](#5-訓練函數)
6. [驗證函數](#6-驗證函數)
7. [偽標籤生成](#7-偽標籤生成)
8. [損失函數與優化器 - 第二階段](#8-損失函數與優化器---第二階段自我訓練)
9. [自我訓練迴圈](#9-自我訓練迴圈)

---

## 1. 資料擴增 (Data Augmentation)

### 1.1 為何需要資料擴增？
資料擴增是深度學習中非常重要的技術，主要目的包括：
- **增加訓練資料的多樣性**：透過各種變換產生更多訓練樣本
- **防止過擬合**：讓模型學習到更泛化的特徵，而非記憶特定圖片
- **提升模型穩健性**：使模型對圖片的旋轉、縮放、光線變化等更具容錯能力

### 1.2 訓練集資料擴增 (transforms_train)

#### 完整代碼與逐行解說

```python
# 從 torchvision.transforms.v2 導入 transforms 模組
# v2 是新版 API，提供更強大和一致的轉換功能
from torchvision.transforms import v2 as transforms

# 使用 Compose 將多個轉換操作串聯起來
# Compose 會按照列表順序依次執行每個轉換
transforms_train = transforms.Compose([
    # 1️⃣ Resize：將圖片調整為 256×256
    # - 輸入：任意尺寸的圖片
    # - 輸出：256×256 的圖片
    # - 目的：統一所有圖片的基礎尺寸
    transforms.Resize((256, 256)),

    # 2️⃣ RandomResizedCrop：隨機裁切並縮放到 224×224
    # - scale=(0.8, 1.0)：隨機選取原圖 80%-100% 的區域
    # - 輸出：224×224 的圖片
    # - 目的：
    #   a) 模擬不同拍攝距離（遠近）
    #   b) 強制模型學習局部特徵，不依賴完整圖片
    #   c) 224×224 是許多 CNN 架構的標準輸入尺寸
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),

    # 3️⃣ RandomHorizontalFlip：以 50% 機率水平翻轉圖片
    # - p=0.5：每張圖片有 50% 機率被翻轉
    # - 目的：
    #   a) 花朵通常左右對稱，翻轉不改變類別
    #   b) 有效增加一倍的訓練資料量
    #   c) 提升模型對方向的不變性
    transforms.RandomHorizontalFlip(p=0.5),

    # 4️⃣ RandomRotation：隨機旋轉 ±15 度
    # - 15：旋轉角度範圍為 [-15°, +15°]
    # - 目的：
    #   a) 模擬不同拍攝角度
    #   b) 現實中花朵可能以任意角度出現
    #   c) 提升模型對旋轉的容錯能力
    # - 注意：角度不能太大，否則可能失真或超出圖片邊界
    transforms.RandomRotation(15),

    # 5️⃣ ColorJitter：隨機調整顏色屬性
    # - brightness=0.2：亮度在 [0.8, 1.2] 範圍內隨機調整
    # - contrast=0.2：對比度在 [0.8, 1.2] 範圍內隨機調整
    # - saturation=0.2：飽和度在 [0.8, 1.2] 範圍內隨機調整
    # - hue=0.1：色調在 [-0.1, +0.1] 範圍內隨機調整
    # - 目的：
    #   a) 模擬不同光照條件（陰天、晴天、室內、室外）
    #   b) 模擬不同相機設定
    #   c) 減少模型對特定顏色的依賴
    #   d) 提升模型在各種環境下的穩健性
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),

    # 6️⃣ ToImage：將 PIL Image 或 numpy array 轉換為 PyTorch Tensor
    # - torchvision.transforms.v2 的新 API
    # - 取代舊的 ToTensor()
    # - 輸出：Tensor 格式的圖片，shape 為 [C, H, W]
    transforms.ToImage(),

    # 7️⃣ ToDtype：轉換資料型態並縮放數值
    # - torch.float32：將資料轉為 32 位元浮點數
    # - scale=True：將 [0, 255] 的整數值縮放到 [0.0, 1.0] 的浮點數
    # - 目的：
    #   a) 神經網路通常使用浮點數運算
    #   b) [0, 1] 範圍的數值更適合訓練（數值穩定）
    transforms.ToDtype(torch.float32, scale=True),

    # 8️⃣ Normalize：標準化（Z-score normalization）
    # - mean=[0.485, 0.456, 0.406]：RGB 三個通道的平均值
    # - std=[0.229, 0.224, 0.225]：RGB 三個通道的標準差
    # - 這些值來自 ImageNet 資料集的統計
    # - 計算公式：output = (input - mean) / std
    # - 目的：
    #   a) 將數值分布標準化為均值 0、標準差 1
    #   b) 加速訓練收斂
    #   c) 減少內部協變量偏移（Internal Covariate Shift）
    #   d) 使用 ImageNet 統計值是遷移學習的常見做法
    # - 範例計算：
    #   輸入 R 通道值 = 0.8
    #   輸出 = (0.8 - 0.485) / 0.229 = 1.376
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

#### 轉換流程視覺化

```
原始圖片 (任意尺寸，如 800×600)
   ↓ Resize
256×256 圖片
   ↓ RandomResizedCrop (隨機裁取 80%-100% 區域)
224×224 圖片
   ↓ RandomHorizontalFlip (50% 機率翻轉)
224×224 圖片（可能已翻轉）
   ↓ RandomRotation (±15°)
224×224 圖片（可能已旋轉）
   ↓ ColorJitter (調整亮度、對比、飽和度、色調)
224×224 圖片（顏色已調整）
   ↓ ToImage
Tensor [3, 224, 224]，值範圍 [0, 255]
   ↓ ToDtype (scale=True)
Tensor [3, 224, 224]，值範圍 [0.0, 1.0]，dtype=float32
   ↓ Normalize (使用 ImageNet 統計值)
Tensor [3, 224, 224]，值範圍約 [-2, 2]，均值≈0，標準差≈1
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

#### 完整代碼與逐行解說

```python
# 驗證/測試集使用較簡單的轉換
# 目標：保持一致性，不引入隨機性
transforms_test = transforms.Compose([
    # 1️⃣ Resize：將圖片調整為 256×256
    # - 與訓練集相同的第一步
    # - 確保所有圖片有統一的基礎尺寸
    transforms.Resize((256, 256)),

    # 2️⃣ CenterCrop：從中心裁切 224×224
    # - ⚠️ 與訓練集的 RandomResizedCrop 不同
    # - 總是從圖片中心裁切，確保每次結果一致
    # - 不使用隨機裁切，因為測試時需要可重複的結果
    # - 目的：
    #   a) 獲得圖片的中心區域（通常包含主要內容）
    #   b) 確保同一張圖片每次處理結果相同
    #   c) 與訓練集保持相同的輸出尺寸 224×224
    transforms.CenterCrop(224),

    # 3️⃣ ToImage：轉換為 PyTorch Tensor
    # - 與訓練集相同
    # - 將 PIL Image 轉為 Tensor，shape 為 [C, H, W]
    transforms.ToImage(),

    # 4️⃣ ToDtype：轉換資料型態並縮放
    # - 與訓練集相同
    # - 將 [0, 255] 縮放到 [0.0, 1.0]
    # - dtype=torch.float32
    transforms.ToDtype(torch.float32, scale=True),

    # 5️⃣ Normalize：標準化
    # - 與訓練集使用完全相同的參數
    # - ⚠️ 非常重要：必須使用與訓練時相同的統計值
    # - 如果使用不同的 mean/std，模型性能會嚴重下降
    # - mean=[0.485, 0.456, 0.406]：ImageNet 的 RGB 平均值
    # - std=[0.229, 0.224, 0.225]：ImageNet 的 RGB 標準差
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

#### 轉換流程視覺化

```
原始測試圖片 (任意尺寸，如 640×480)
   ↓ Resize
256×256 圖片
   ↓ CenterCrop（從中心裁切，無隨機性）
224×224 圖片
   ↓ ToImage
Tensor [3, 224, 224]，值範圍 [0, 255]
   ↓ ToDtype (scale=True)
Tensor [3, 224, 224]，值範圍 [0.0, 1.0]，dtype=float32
   ↓ Normalize（與訓練集使用相同參數）
Tensor [3, 224, 224]，值範圍約 [-2, 2]，均值≈0，標準差≈1
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

我們使用 **ResNet50** 作為骨幹網路，並添加自定義分類器。ResNet50 是一個非常強大的卷積神經網路架構，具有 50 層深度。

#### 完整代碼與逐行解說

```python
# 導入必要的模組
from torchvision.models import resnet50
import torch.nn as nn

class YourCNNModel(nn.Module):
    """
    花朵分類模型
    - 骨幹網路：ResNet50（從頭訓練，不使用預訓練權重）
    - 分類器：自定義全連接層
    """
    def __init__(self, num_classes=5):
        # 調用父類的初始化方法
        # nn.Module 是所有神經網路模組的基類
        super().__init__()

        # ═══════════════════════════════════════════════════════
        # 第一部分：特徵提取器（Feature Extractor）
        # ═══════════════════════════════════════════════════════

        # 1️⃣ 創建 ResNet50 模型（不使用預訓練權重）
        # - weights=None：從頭開始訓練，不載入 ImageNet 預訓練權重
        # - 為什麼不用預訓練權重？
        #   a) 作業要求不能使用預訓練模型
        #   b) 從頭訓練可以完全適應我們的花朵資料集
        # - ResNet50 架構：
        #   輸入 [3, 224, 224]
        #   → Conv1 + MaxPool
        #   → Layer1 (3個 Bottleneck blocks)
        #   → Layer2 (4個 Bottleneck blocks)
        #   → Layer3 (6個 Bottleneck blocks)
        #   → Layer4 (3個 Bottleneck blocks)
        #   → AvgPool
        #   → Fully Connected (原始的分類層)
        resnet = resnet50(weights=None)

        # 2️⃣ 移除 ResNet50 的最後一層（全連接分類層）
        # - resnet.children()：獲取 ResNet50 的所有子模組
        # - [:-1]：取除了最後一個模組之外的所有模組
        # - 最後一個模組是 Linear(2048, 1000)，用於 ImageNet 的 1000 類分類
        # - 我們只需要 5 類分類，所以要替換這一層
        # - *list(...)：解包列表，將所有模組作為參數傳給 Sequential
        # - nn.Sequential：將多個模組串聯成一個模組
        # - 結果：self.features 包含 ResNet50 的所有層，除了最後的分類層
        #   輸出維度：[batch_size, 2048, 1, 1]
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # ═══════════════════════════════════════════════════════
        # 第二部分：分類器（Classifier）
        # ═══════════════════════════════════════════════════════

        # 3️⃣ 自定義分類器
        # 輸入：ResNet50 特徵提取器的輸出 [batch_size, 2048, 1, 1]
        # 輸出：5 個類別的預測分數 [batch_size, 5]
        self.classifier = nn.Sequential(
            # 第 1 層：Flatten - 展平多維 Tensor
            # - 輸入：[batch_size, 2048, 1, 1]
            # - 輸出：[batch_size, 2048]
            # - 目的：將 4D tensor 轉為 2D，以便輸入全連接層
            nn.Flatten(),

            # 第 2 層：Dropout - 防止過擬合
            # - p=0.5：訓練時隨機丟棄 50% 的神經元
            # - 測試時不丟棄（自動切換）
            # - 目的：
            #   a) 防止模型過度依賴某些特定神經元
            #   b) 強制網路學習更穩健的特徵表示
            #   c) 提升泛化能力
            nn.Dropout(0.5),

            # 第 3 層：Linear - 第一個全連接層
            # - 輸入：2048 維特徵向量
            # - 輸出：512 維特徵向量
            # - 參數量：2048 × 512 + 512 = 1,049,088 個參數
            # - 目的：降維並學習高層特徵組合
            nn.Linear(2048, 512),

            # 第 4 層：ReLU - 激活函數
            # - ReLU(x) = max(0, x)
            # - 引入非線性，使網路能學習複雜模式
            # - 相比 Sigmoid/Tanh：
            #   a) 計算更快（簡單的 max 運算）
            #   b) 緩解梯度消失問題
            #   c) 使網路更容易訓練
            nn.ReLU(),

            # 第 5 層：Dropout - 第二次防止過擬合
            # - p=0.3：較低的丟棄率（30%）
            # - 為何使用較低比率？
            #   a) 512 維特徵已經比 2048 少很多
            #   b) 避免過度正則化，保留足夠信息
            nn.Dropout(0.3),

            # 第 6 層：Linear - 輸出層
            # - 輸入：512 維特徵向量
            # - 輸出：5 維向量（對應 5 個花朵類別）
            # - 參數量：512 × 5 + 5 = 2,565 個參數
            # - ⚠️ 不需要 Softmax！
            #   CrossEntropyLoss 內部會自動處理
            #   輸出的是 logits（原始預測分數）
            nn.Linear(512, num_classes)
        )
```

### 2.2 前向傳播（Forward Pass）

#### 完整代碼與逐行解說

```python
def forward(self, x):
    """
    前向傳播函數

    參數:
        x: 輸入圖片 Tensor
           - shape: [batch_size, 3, 224, 224]
           - dtype: torch.float32
           - 值範圍: 標準化後約 [-2, 2]

    返回:
        out: 預測的 logits（原始分數）
           - shape: [batch_size, 5]
           - dtype: torch.float32
           - 值範圍: 任意實數（未經 Softmax）
    """

    # ════════════════════════════════════════════════
    # 輸入驗證（僅在開發階段使用，確保輸入正確）
    # ════════════════════════════════════════════════

    # 檢查 1：確保輸入是 PyTorch Tensor
    # - 如果傳入 numpy array 或其他類型，會拋出錯誤
    # - 目的：盡早發現錯誤，避免在深層網路中才出錯
    assert isinstance(x, torch.Tensor), "Input should be a torch Tensor"

    # 檢查 2：確保輸入是 4 維 Tensor
    # - 正確格式：[batch_size, channels, height, width]
    # - 錯誤格式示例：
    #   * [3, 224, 224]（缺少 batch 維度）
    #   * [batch_size, 224, 224, 3]（通道在最後，NHWC 格式）
    # - 目的：確保輸入符合 PyTorch 的 NCHW 格式
    assert x.dim() == 4, "Input should be NCHW format"

    # ════════════════════════════════════════════════
    # 實際的前向傳播
    # ════════════════════════════════════════════════

    # 步驟 1：特徵提取
    # - 輸入：x，shape = [batch_size, 3, 224, 224]
    # - 經過 ResNet50 的所有卷積層和池化層
    # - 輸出：x，shape = [batch_size, 2048, 1, 1]
    # - 這個 2048 維向量包含了圖片的高層語義特徵
    # - 1×1 的空間維度是由 Global Average Pooling 產生的
    x = self.features(x)

    # 步驟 2：分類
    # - 輸入：x，shape = [batch_size, 2048, 1, 1]
    # - 經過自定義分類器：
    #   1. Flatten: [batch_size, 2048, 1, 1] → [batch_size, 2048]
    #   2. Dropout(0.5): 訓練時隨機丟棄 50% 神經元
    #   3. Linear(2048→512): [batch_size, 2048] → [batch_size, 512]
    #   4. ReLU: 激活函數
    #   5. Dropout(0.3): 訓練時隨機丟棄 30% 神經元
    #   6. Linear(512→5): [batch_size, 512] → [batch_size, 5]
    # - 輸出：out，shape = [batch_size, 5]
    # - 5 個數值分別對應 5 個類別的預測分數（logits）
    out = self.classifier(x)

    # 返回預測結果
    # ⚠️ 注意：這裡返回的是 logits，不是機率
    # - Logits 可以是任意實數（可正可負）
    # - CrossEntropyLoss 會自動處理 Softmax 轉換
    # - 範例輸出：[2.3, -1.2, 0.5, 1.8, -0.3]
    return out
```

### 2.3 模型初始化與設備配置

#### 完整代碼與逐行解說

```python
# 1️⃣ 設定運算設備
# - 檢查是否有可用的 CUDA GPU
# - 如果有 GPU，使用 GPU 進行運算（速度快 10-100 倍）
# - 如果沒有 GPU，使用 CPU 進行運算
device = torch.device('cuda')
# 如果想強制使用 CPU，可以改為：
# device = torch.device('cpu')

# 2️⃣ 創建模型實例
# - num_classes=num_classes：花朵類別數（5）
# - CLASS_NAMES = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
# - 此時模型的所有參數都是隨機初始化的
# - 參數初始化通常使用 Kaiming 初始化（針對 ReLU 優化）
model = YourCNNModel(num_classes=num_classes)

# 3️⃣ 將模型移動到指定設備（GPU 或 CPU）
# - .to(device)：將模型的所有參數和緩衝區移動到指定設備
# - 如果 device 是 'cuda'，所有參數會被複製到 GPU 記憶體
# - 這一步必須在訓練前完成
# - ⚠️ 之後所有輸入數據也必須移動到相同設備
model = model.to(device)
```

### 2.4 ResNet50 架構詳解

#### ResNet50 完整結構

```
輸入圖片 [batch_size, 3, 224, 224]
   ↓
┌─────────────────────────────────────┐
│ Conv1: 7×7 卷積, stride=2, 64 通道  │
│ BatchNorm + ReLU                    │
│ MaxPool 3×3, stride=2               │
│ 輸出: [batch_size, 64, 56, 56]     │
└─────────────────────────────────────┘
   ↓
┌─────────────────────────────────────┐
│ Layer1: 3 個 Bottleneck blocks      │
│ 通道: 64 → 256                      │
│ 輸出: [batch_size, 256, 56, 56]    │
└─────────────────────────────────────┘
   ↓
┌─────────────────────────────────────┐
│ Layer2: 4 個 Bottleneck blocks      │
│ 通道: 256 → 512, stride=2          │
│ 輸出: [batch_size, 512, 28, 28]    │
└─────────────────────────────────────┘
   ↓
┌─────────────────────────────────────┐
│ Layer3: 6 個 Bottleneck blocks      │
│ 通道: 512 → 1024, stride=2         │
│ 輸出: [batch_size, 1024, 14, 14]   │
└─────────────────────────────────────┘
   ↓
┌─────────────────────────────────────┐
│ Layer4: 3 個 Bottleneck blocks      │
│ 通道: 1024 → 2048, stride=2        │
│ 輸出: [batch_size, 2048, 7, 7]     │
└─────────────────────────────────────┘
   ↓
┌─────────────────────────────────────┐
│ AdaptiveAvgPool2d(1, 1)             │
│ 全局平均池化                         │
│ 輸出: [batch_size, 2048, 1, 1]     │
└─────────────────────────────────────┘
   ↓
┌─────────────────────────────────────┐
│ 自定義分類器                         │
│ Flatten → Dropout → Linear → ReLU  │
│ → Dropout → Linear                  │
│ 輸出: [batch_size, 5]               │
└─────────────────────────────────────┘
```

#### Bottleneck Block 結構

每個 Bottleneck block 包含：
1. **1×1 卷積**：降維（減少通道數）
2. **3×3 卷積**：特徵提取
3. **1×1 卷積**：升維（恢復通道數）
4. **Shortcut 連接**：殘差連接，解決梯度消失問題

### 2.5 模型參數統計

| 組件 | 參數量 | 佔比 |
|------|--------|------|
| **ResNet50 骨幹** | 約 23.5M | 95.6% |
| **自定義分類器** | 約 1.05M | 4.4% |
| **總計** | **約 24.55M** | **100%** |

**訓練時記憶體需求**（batch_size=128）：
- 模型參數：約 95 MB
- 梯度：約 95 MB
- 中間激活值：約 500-800 MB
- **總計**：約 1.2-1.5 GB GPU 記憶體


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

## 4. 學習率調度器

### 4.1 什麼是學習率調度器（Learning Rate Scheduler）？

學習率調度器是一種在訓練過程中**動態調整學習率**的技術，幫助模型更好地收斂。

**為何需要調整學習率？**
- **訓練初期**：使用較大學習率快速逼近最優解
- **訓練後期**：使用較小學習率精細調整，避免震盪

### 4.2 CosineAnnealingLR 實作

#### 第一階段（監督式訓練）

```python
# 在定義 max_epochs 之後，訓練迴圈之前
max_epochs = 30
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=0)

# 在訓練迴圈中，每個 epoch 結束後
for epoch in range(1, max_epochs + 1):
    train_acc, train_loss = train(...)
    val_acc, val_loss = val(...)

    # 更新學習率
    scheduler.step()
```

#### 第二階段（自我訓練）

```python
# 在定義 n_epochs 之後
n_epochs = 30
scheduler_ssl = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=5e-5)

# 在訓練迴圈中
for epoch in range(n_epochs):
    train_acc, train_loss = train(...)
    valid_acc, valid_loss = val(...)

    # 更新學習率
    scheduler_ssl.step()
```

### 4.3 CosineAnnealingLR 參數說明

| 參數 | 說明 | 第一階段設定 | 第二階段設定 |
|------|------|-------------|-------------|
| **T_max** | 學習率週期長度（epochs） | `max_epochs` (30) | `n_epochs` (30) |
| **eta_min** | 學習率的最小值 | 0 | 5e-5 (0.00005) |

### 4.4 學習率變化曲線

#### 餘弦退火（Cosine Annealing）原理

```python
lr(t) = eta_min + (lr_initial - eta_min) * (1 + cos(π * t / T_max)) / 2
```

**視覺化：**
```
第一階段 (lr: 0.001 → 0):
LR
0.001 ┤╮
      │ ╲
0.0005│  ╲___
      │      ╲___
0.0000└──────────╲___
      0   15   30 epoch

第二階段 (lr: 0.0005 → 0.00005):
LR
0.0005┤╮
      │ ╲
0.0003│  ╲___
      │      ╲___
0.0001└──────────╲___
      0   15   30 epoch
```

### 4.5 為何選擇 CosineAnnealingLR？

| Scheduler 類型 | 學習率變化 | 優點 | 缺點 |
|---------------|-----------|------|------|
| **StepLR** | 階梯式下降 | 簡單、可控 | 變化不夠平滑 |
| **ExponentialLR** | 指數衰減 | 平滑下降 | 後期學習率可能過小 |
| **CosineAnnealingLR** ✓ | 餘弦曲線 | 1. **平滑且自然的下降**<br>2. 訓練後期仍有適度學習率<br>3. 常在影像分類任務表現優秀 | 需要預先知道總 epoch 數 |
| **ReduceLROnPlateau** | 根據驗證損失 | 自適應 | 可能過早降低學習率 |

### 4.6 學習率對訓練的影響

#### 不同學習率策略比較

**1. 固定學習率（無 Scheduler）**
```
Loss ─┐    訓練初期下降快
      │╲   ┌─┐
      │ ╲ ╱  │  後期震盪，難以收斂
      │  ╳   └─┐
      └───────╲└─
```

**2. 使用 CosineAnnealingLR**
```
Loss ─┐    訓練初期下降快
      │╲
      │ ╲___   中期穩定下降
      │     ╲___  後期平滑收斂
      └─────────╲___
```

### 4.7 第一階段 vs 第二階段的差異

| 項目 | 第一階段 | 第二階段 | 原因 |
|------|---------|---------|------|
| **初始學習率** | 0.001 | 0.0005 | 第二階段模型已預訓練 |
| **最小學習率** | 0 | 0.00005 | 第二階段保持微小學習以適應偽標籤 |
| **Scheduler 名稱** | `scheduler` | `scheduler_ssl` | 區分兩個階段的調度器 |
| **變化幅度** | 0.001 → 0<br>(100% 下降) | 0.0005 → 0.00005<br>(90% 下降) | 第二階段更保守 |

### 4.8 完整訓練週期中的學習率演變

```
第一階段 (30 epochs):
Epoch  1: lr = 0.001000
Epoch  5: lr = 0.000905
Epoch 10: lr = 0.000655
Epoch 15: lr = 0.000345
Epoch 20: lr = 0.000095
Epoch 25: lr = 0.000015
Epoch 30: lr = 0.000000

第二階段 (30 epochs):
Epoch  1: lr = 0.000500
Epoch  5: lr = 0.000452
Epoch 10: lr = 0.000327
Epoch 15: lr = 0.000173
Epoch 20: lr = 0.000048
Epoch 25: lr = 0.000008
Epoch 30: lr = 0.000050
```

### 4.9 何時呼叫 scheduler.step()？

**正確位置**：在每個 epoch 的訓練和驗證**之後**

```python
for epoch in range(max_epochs):
    train(...)      # 訓練
    validate(...)   # 驗證
    scheduler.step()  # ← 在這裡更新學習率（epoch 結束後）
```

**錯誤位置示例：**
```python
# ✗ 錯誤：在訓練迴圈內部
for images, labels in train_loader:
    ...
    optimizer.step()
    scheduler.step()  # ✗ 會導致學習率更新過快

# ✗ 錯誤：在訓練之前
for epoch in range(max_epochs):
    scheduler.step()  # ✗ 第一個 epoch 會用錯誤的學習率
    train(...)
```

### 4.10 監控學習率

在訓練過程中可以查看當前學習率：

```python
current_lr = optimizer.param_groups[0]['lr']
print(f'Current learning rate: {current_lr:.6f}')
```

這在我們的程式碼中已經實作（line 512, 709）：
```python
lr = optimizer.param_groups[0]['lr']
print('  Val Acc: {:.6f} |   Val Loss: {:.6f} | LR: {:.6f}'.format(val_acc, val_loss, lr))
```

---

## 5. 訓練函數

### 5.1 完整實作

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

### 5.2 五大步驟詳解

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

### 5.3 準確率統計

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

## 6. 驗證函數

### 6.1 與訓練函數的差異

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

### 6.2 關鍵差異說明

| 項目 | 訓練模式 | 評估模式 |
|------|---------|---------|
| **模式設定** | `model.train()` | `model.eval()` ✓ |
| **梯度計算** | 需要 | `torch.no_grad()` 不需要 ✓ |
| **Dropout** | 隨機丟棄 50% | **不丟棄**（使用所有神經元） |
| **BatchNorm** | 使用當前批次統計 | **使用訓練時的移動平均** |
| **反向傳播** | 有 | **無** |
| **權重更新** | 有 | **無** |

### 6.3 為何需要 `model.eval()` 和 `torch.no_grad()`？

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

## 7. 偽標籤生成

### 7.1 什麼是偽標籤（Pseudo-Labeling）？

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

### 7.2 實作細節

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

### 7.3 關鍵參數：threshold = 0.9

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

### 7.4 Softmax 的作用

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

## 8. 損失函數與優化器 - 第二階段（自我訓練）

### 8.1 為何需要調整優化器？

在自我訓練階段，我們面臨不同的挑戰：
1. 模型已經過第一階段訓練，具有一定能力
2. 新加入的偽標籤可能包含雜訊
3. 需要更謹慎地微調模型

### 8.2 優化器配置

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

### 8.3 為何偽標籤需要更小的學習率？

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

## 9. 自我訓練迴圈

### 9.1 整體策略

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

### 9.2 設計原理

#### 9.2.1 為何每 10 epochs 生成偽標籤？

| 頻率 | 優點 | 缺點 | 結論 |
|------|------|------|------|
| **每 1 epoch** | 最新的模型預測 | 1. 計算開銷大<br>2. 模型可能還不穩定<br>3. 偽標籤品質參差不齊 | ✗ 不推薦 |
| **每 10 epochs** ✓ | 1. 模型有足夠時間學習<br>2. 預測更穩定<br>3. 計算效率高 | 可能錯過一些改進 | ✓ **平衡選擇** |
| **每 20+ epochs** | 偽標籤品質高 | 1. 更新太慢<br>2. 可能錯過學習機會 | ✗ 太保守 |

#### 9.2.2 累積式策略 vs 替換式策略

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

### 9.3 資料集演變過程

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

### 9.4 未標記資料池的動態變化

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

### 9.5 ConcatDataset 的作用

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
│ 4. 學習率調度器 CosineAnnealingLR (0.001 → 0)          │
│    ↓                                                     │
│ 5. 訓練 30 epochs → 儲存最佳模型                        │
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
│   - 學習率調度器 CosineAnnealingLR (0.0005 → 0.00005) │
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
| **學習率調度器** | CosineAnnealingLR | 平滑降低學習率，幫助模型收斂 |
| **偽標籤閾值** | 0.9 | 平衡準確率與資料量 |
| **更新頻率** | 每 10 epochs | 給模型足夠時間穩定 |
| **資料策略** | 累積式 | 最大化訓練資料利用 |

這份文件詳細說明了 Lab3 實作的每個環節，包括技術選擇的理由、數學原理和實際效果。
