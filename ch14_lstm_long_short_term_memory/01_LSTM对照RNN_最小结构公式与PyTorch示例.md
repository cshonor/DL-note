# LSTM 对照 RNN：最小结构、公式与 PyTorch 示例

这篇只讲结构和关键变化，不讲玄学。  
目标是和你已经掌握的普通 RNN 无缝对照。

---

## 1) 一句话抓住 LSTM 本质
普通 RNN 只有一个记忆通道 `h`，长序列里早期信息容易丢。  
LSTM 在 RNN 基础上加了门控机制，控制：
- 忘掉什么
- 写入什么
- 输出什么

所以可理解为：**LSTM = 带阀门控制记忆的 RNN**。

---

## 2) 和普通 RNN 的核心对比

### 普通 RNN（vanilla RNN）
- 主要状态：`h_t`
- 递推：`h_t = tanh(W_xh @ x_t + W_hh @ h_(t-1) + b)`
- 痛点：长序列下，早期信息和梯度都更容易衰减

### LSTM
- 主要状态：`C_t`（长期记忆） + `h_t`（对外输出）
- 三个门：遗忘门 `f_t`、输入门 `i_t`、输出门 `o_t`
- 核心收益：长期记忆通过 `C_t` 这条通道更稳定传递

---

## 3) LSTM 单步结构（极简拆解）

输入：
- `x_t`：当前输入
- `h_(t-1)`：上一时刻隐藏状态
- `C_(t-1)`：上一时刻细胞状态

先拼接输入：`z_t = [h_(t-1), x_t]`

### 3.1 遗忘门（忘记旧记忆比例）
`f_t = sigmoid(W_f @ z_t + b_f)`

### 3.2 输入门 + 候选记忆（写入新信息比例）
`i_t = sigmoid(W_i @ z_t + b_i)`  
`C_t_candidate = tanh(W_c @ z_t + b_c)`

### 3.3 更新细胞状态（长期记忆主通道）
`C_t = f_t * C_(t-1) + i_t * C_t_candidate`

### 3.4 输出门 + 当前隐藏状态
`o_t = sigmoid(W_o @ z_t + b_o)`  
`h_t = o_t * tanh(C_t)`

---

## 4) 最直白的分工记忆
- `C_t`：长期记忆传送带，重点负责“记住多久”
- `h_t`：当前时刻对外输出，重点负责“当前拿出去用什么”
- 三个门都用 `sigmoid`，输出 0 到 1，表示通过比例

记忆口诀：
- 一门忘（`f`）
- 一门写（`i`）
- 一门出（`o`）

---

## 5) PyTorch 极简 LSTM 示例（NLP many-to-one）

```python
import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # x: [batch, seq_len]
        emb = self.embedding(x)              # [B, T, E]
        out, (h_n, c_n) = self.lstm(emb)     # out: [B, T, H]
        last_h = out[:, -1, :]               # many-to-one 常用最后一步
        logits = self.fc(last_h)             # [B, vocab_size]
        return logits
```

关键输出含义：
- `out`：每个时间步的隐藏状态序列
- `h_n`：最后一层最后时刻的隐藏状态
- `c_n`：最后一层最后时刻的细胞状态

---

## 6) LSTM 比普通 RNN 强在哪
1. 长序列信息保留更稳定  
2. 梯度在时间反传时更不容易快速衰减  
3. 文本、语音、长时序任务里通常显著优于 vanilla RNN

---

## 一句话收束
**LSTM 没改“按时间递推”的框架，只是在 RNN 上加了可学习门控，让长期记忆更可控、更稳定。**
