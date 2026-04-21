# 五种激活/门控结构 PyTorch 可运行示例（第5个为 SwiGLU）

> 目标：给出统一结构、可直接运行的 5 组独立网络：  
> `Sigmoid / Tanh / ReLU / LeakyReLU / SwiGLU`，输出层统一 `Softmax`。

---

## 统一规则
- 前 4 个用传统激活函数（Sigmoid、Tanh、ReLU、LeakyReLU）
- 第 5 个用门控单元：SwiGLU（大模型常见 FFN 结构）
- 输出层统一：Softmax
- 示例均可直接复制运行

---

## 1) Sigmoid + Softmax

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net_Sigmoid(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=32, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=-1)
        return x

x = torch.randn(2, 16)
model = Net_Sigmoid()
print("Sigmoid out:", model(x).shape)
```

---

## 2) Tanh + Softmax

```python
class Net_Tanh(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=32, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=-1)
        return x

model = Net_Tanh()
print("Tanh out:", model(x).shape)
```

---

## 3) ReLU + Softmax

```python
class Net_ReLU(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=32, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=-1)
        return x

model = Net_ReLU()
print("ReLU out:", model(x).shape)
```

---

## 4) LeakyReLU + Softmax

```python
class Net_LeakyReLU(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=32, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.01)
        x = F.leaky_relu(self.fc2(x), 0.01)
        x = self.fc3(x)
        x = F.softmax(x, dim=-1)
        return x

model = Net_LeakyReLU()
print("LeakyReLU out:", model(x).shape)
```

---

## 5) SwiGLU + Softmax（大模型风格）

```python
class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class Net_SwiGLU(nn.Module):
    def __init__(self, in_dim=16, hidden_dim=64, out_dim=10):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)
        self.swiglu = SwiGLU(dim=hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.swiglu(x)
        x = self.fc_out(x)
        x = F.softmax(x, dim=-1)
        return x

model = Net_SwiGLU()
print("SwiGLU out:", model(x).shape)
```

---

## 重点说明

1. `SiLU == Swish`，而 `SwiGLU` 是更完整的门控结构  
   - `SiLU(x) = x * sigmoid(x)` 是单路激活  
   - `SwiGLU(x) = (x @ W1 经 SiLU) * (x @ W2)` 再映射回输出维度

2. 大模型常见 `GELU` 或 `SwiGLU` 结构  
   - ReLU 负半轴硬截断，存在坏死风险  
   - SiLU/GELU 更平滑，SwiGLU 通过门控提升表达能力

3. 本页示例是“教学简化结构”  
   - 用全连接替代 Transformer 块  
   - 但“隐藏层非线性/门控 + 输出 Softmax”的逻辑一致

---

## 一句收束
输入 -> Linear -> 激活 -> Linear -> 激活 -> Linear -> Softmax，  
只需替换隐藏层激活函数，就能直观看到不同非线性策略在同一结构中的用法。
