# 交叉熵损失 + Sigmoid 激活：`delta` 与权重梯度对照

> 对应主线：`02_多输入神经网络_图解详解_SGD与BGD.md`。本文专注二分类里最常见的 Sigmoid + 交叉熵推导与对比。

---

## 一、模型设定（二分类）

- 线性层：`z = w1*x1 + w2*x2 + ... + wn*xn`  
- 概率输出：`y_hat = 1 / (1 + exp(-z))`  
- 标签：`y in {0, 1}`  
- 单样本交叉熵：`L = -( y*ln(y_hat) + (1-y)*ln(1-y_hat) )`

---

## 二、Sigmoid 导数

- `d(sigmoid(z))/dz = sigmoid(z) * (1 - sigmoid(z)) = y_hat * (1 - y_hat)`

---

## 三、节点 `delta`（`dL/dz`）如何化简

定义节点误差信号：`delta = dL/dz`。链式法则：

- `delta = (dL/dy_hat) * (dy_hat/dz)`

其中：

- `dL/dy_hat = -( y / y_hat - (1-y)/(1-y_hat) )`  
- `dy_hat/dz = y_hat * (1-y_hat)`

代入后可化简为：

- `delta = y_hat - y`

这就是经典结论：**Sigmoid + 交叉熵顶层，节点 delta 直接是 `预测概率 - 标签`。**

---

## 四、权重梯度（`weight_delta`）

对任意权重 `w_i`：

- `dL/dw_i = (dL/dz) * (dz/dw_i)`  
- `dz/dw_i = x_i`

所以：

- `dL/dw_i = delta * x_i = (y_hat - y) * x_i`

仍然是同一骨架：**`weight_delta_i = 节点delta * 对应输入特征`**。

---

## 五、与“线性输出 + 平方误差”并排对照

- 线性 + 平方误差（无 `0.5`）：`delta_linear = 2*(pred - y)`，`dL/dw_i = delta_linear * x_i`  
- Sigmoid + 交叉熵：`delta_ce = y_hat - y`，`dL/dw_i = delta_ce * x_i`

相同点：

- 权重梯度都遵循“节点误差信号 × 输入特征”

不同点：

- 节点误差信号 `delta` 的来源不同（损失函数 + 输出层激活决定）

---

## 六、更新步骤（统一形式）

1. 前向：`z -> y_hat`  
2. 节点误差信号：`delta = y_hat - y`  
3. 权重梯度：`weight_delta_i = delta * x_i`  
4. 参数更新：`w_i = w_i - alpha * weight_delta_i`

---

## 七、多分类一句话（Softmax + 交叉熵）

多分类顶层同样得到 `delta_i = y_hat_i - y_i`，再乘输入得到连接权重梯度。  
骨架不变：先算顶层 delta，再按连接输入分摊到各权重。

---

## 八、可运行对比脚本

- `mult_input_logistic_ce_vs_mse.py`：并排打印 MSE 版本与 Sigmoid+交叉熵版本的 `pred / delta / weight_delta / loss`，用于观察两者在同一数据上的训练行为差异。
