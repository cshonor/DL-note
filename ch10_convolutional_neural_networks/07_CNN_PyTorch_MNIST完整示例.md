# CNN 完整示例（PyTorch，MNIST 新手友好版）

> 目标：用最小可运行 CNN 结构跑通 `28×28` 灰度图分类。  
> 结构完全对应：卷积 -> ReLU -> 池化 -> 卷积 -> ReLU -> 池化 -> 展平 -> 全连接。

---

## 环境安装（如未安装）

```bash
pip install torch torchvision
```

---

## 完整代码（可直接运行）

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ======================
# 1. 数据准备（MNIST 28x28 手写数字）
# ======================
transform = transforms.Compose([
    transforms.ToTensor(),                 # 转为张量: (1, 28, 28)
    transforms.Normalize((0.1307,), (0.3081,))  # 标准化
])

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# ======================
# 2. 定义 CNN 模型
# ======================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # 第一组: 卷积 + ReLU + 池化
        # 输入: (1, 28, 28)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1  # 保持 28x28
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28 -> 14

        # 第二组: 卷积 + ReLU + 池化
        # 输入: (8, 14, 14)
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1  # 保持 14x14
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14 -> 7

        # 全连接分类头
        # (16, 7, 7) -> 展平后 16*7*7 = 784
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)  # 10 类数字: 0~9

    def forward(self, x):
        # 第1组
        x = self.conv1(x)   # (1, 28, 28) -> (8, 28, 28)
        x = self.relu1(x)
        x = self.pool1(x)   # -> (8, 14, 14)

        # 第2组
        x = self.conv2(x)   # (8, 14, 14) -> (16, 14, 14)
        x = self.relu2(x)
        x = self.pool2(x)   # -> (16, 7, 7)

        # 展平
        x = torch.flatten(x, 1)  # -> (batch, 784)

        # 全连接分类
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)          # 输出 logits（不手动 softmax）
        return x


# ======================
# 3. 初始化模型、损失、优化器
# ======================
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ======================
# 4. 训练 1 个 epoch（先跑通）
# ======================
def train(model, loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch: {epoch} | Batch: {batch_idx} | Loss: {loss.item():.4f}")


# ======================
# 5. 测试准确率
# ======================
def test(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
    acc = 100.0 * correct / len(loader.dataset)
    print(f"测试准确率: {acc:.2f}%")


if __name__ == "__main__":
    print("开始训练 CNN...")
    train(model, train_loader, criterion, optimizer, epoch=1)
    test(model, test_loader)
```

---

## 代码对应知识点速查

1. 输入：`28×28` 灰度图，通道数 `1`  
2. 第一层卷积：`Conv2d(1, 8, 3, padding=1)` -> 输出 8 张特征图  
3. 第一层池化：`MaxPool2d(2)` -> `28 -> 14`  
4. 第二层卷积：`Conv2d(8, 16, 3, padding=1)` -> 输出 16 张特征图  
5. 第二层池化：`14 -> 7`  
6. 展平：`16*7*7 = 784`  
7. 全连接分类：`784 -> 64 -> 10`

---

## 新手提醒

- 这里最后输出的是 `logits`，因为 `CrossEntropyLoss` 内部已经包含 `log_softmax`。  
- 训练时不要在模型末尾再手动加 `softmax`，否则会影响数值稳定性。
