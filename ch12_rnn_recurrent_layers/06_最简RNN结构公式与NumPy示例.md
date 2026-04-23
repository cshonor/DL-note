# 最简 RNN 结构、公式与 NumPy 示例（不绕弯版）

这篇只讲一件事：**RNN 相比普通全连接，到底多了什么。**

---

## 1) 最简结论：RNN 只多了两样

对比普通全连接（MLP）：
- 普通全连接：只看当前输入 `x`
- RNN：看当前输入 `x_t`，还看上一步记忆 `h_(t-1)`

RNN 真正多出来的是：
1. 一个隐藏状态（记忆向量）`h`
2. 一组循环权重 `W_hh`（处理 `h_(t-1) -> h_t`）

其余都和常规网络一致：输入权重、偏置、激活、输出层。

---

## 2) 最简 RNN 单步公式（直接背）

`h_t = tanh(W_xh @ x_t + W_hh @ h_(t-1) + b_h)`

`y_t = W_hy @ h_t + b_y`

含义对照：
- `x_t`：当前时刻输入
- `h_(t-1)`：上一时刻记忆
- `W_xh`：输入到隐藏
- `W_hh`：隐藏到隐藏（RNN 独有）
- `W_hy`：隐藏到输出

一句话：**RNN 就是把上一步隐藏状态再送回当前步参与计算。**

---

## 3) 极简文字结构图

```text
输入序列：x0 -> x1 -> x2 -> x3 -> ...

每一步结构相同：

        x_t --(W_xh)--\
                        +--> tanh --> h_t --(W_hy)--> y_t
 h_(t-1) --(W_hh)-----/

其中 W_hh 是 RNN 区别于普通全连接的关键。
```

---

## 4) NumPy 极简代码（可直接抄）

```python
import numpy as np

input_size = 10
hidden_size = 20
output_size = 5

# 参数
W_xh = np.random.randn(hidden_size, input_size)    # x -> h
W_hh = np.random.randn(hidden_size, hidden_size)   # h_prev -> h (RNN独有)
W_hy = np.random.randn(output_size, hidden_size)   # h -> y
b_h = np.zeros((hidden_size, 1))
b_y = np.zeros((output_size, 1))

def rnn_step(x_t, h_prev):
    h_t = np.tanh(W_xh @ x_t + W_hh @ h_prev + b_h)
    y_t = W_hy @ h_t + b_y
    return h_t, y_t

# 跑一个长度为5的序列
h_prev = np.zeros((hidden_size, 1))
sequence = [np.random.randn(input_size, 1) for _ in range(5)]

for x_t in sequence:
    h_t, y_t = rnn_step(x_t, h_prev)
    h_prev = h_t
```

---

## 5) 最后对齐一句话

- RNN 多了记忆向量 `h`
- 多了循环权重 `W_hh`
- 每一步结构完全一样，参数共享

所以可以自然处理变长序列。
