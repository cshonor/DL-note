"""第一个深度神经网络（纯 NumPy）：前向 + 反向 + 权重更新。

支持两种训练方式：
- mode="batch"：每轮用全量样本计算一次梯度（BGD）
- mode="sgd"：每轮随机抽 1 条样本更新一次（SGD）
"""

from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_deriv(activated: np.ndarray) -> np.ndarray:
    # 这里传入的是 sigmoid 输出值
    return activated * (1.0 - activated)


def run(mode: str = "batch", epochs: int = 10000, lr: float = 0.1, seed: int = 1) -> None:
    # 4 条样本，4 个特征，二分类标签
    x = np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    y = np.array([[0.0], [0.0], [1.0], [1.0]])

    # 输入(4) -> 隐藏1(4) -> 隐藏2(3) -> 输出(1)
    rng = np.random.default_rng(seed)
    w1 = 2.0 * rng.random((4, 4)) - 1.0
    w2 = 2.0 * rng.random((4, 3)) - 1.0
    w3 = 2.0 * rng.random((3, 1)) - 1.0

    for i in range(epochs):
        if mode == "sgd":
            idx = rng.integers(0, x.shape[0])
            l0 = x[idx : idx + 1]
            target = y[idx : idx + 1]
        else:
            l0 = x
            target = y

        # 前向传播
        l1 = sigmoid(l0 @ w1)
        l2 = sigmoid(l1 @ w2)
        l3 = sigmoid(l2 @ w3)

        # 反向传播
        l3_error = target - l3
        l3_delta = l3_error * sigmoid_deriv(l3)

        l2_error = l3_delta @ w3.T
        l2_delta = l2_error * sigmoid_deriv(l2)

        l1_error = l2_delta @ w2.T
        l1_delta = l1_error * sigmoid_deriv(l1)

        # 梯度下降更新
        w3 += lr * (l2.T @ l3_delta)
        w2 += lr * (l1.T @ l2_delta)
        w1 += lr * (l0.T @ l1_delta)

        if i % 1000 == 0:
            # 为了可比性，日志统一看全量样本误差
            pred_all = sigmoid(sigmoid(sigmoid(x @ w1) @ w2) @ w3)
            loss = np.mean(np.abs(y - pred_all))
            print(f"iter={i:5d} mode={mode:5s} mean_abs_error={loss:.6f}")

    print("\n===== final prediction =====")
    pred = sigmoid(sigmoid(sigmoid(x @ w1) @ w2) @ w3)
    print(pred)
    print("\n===== target =====")
    print(y)


if __name__ == "__main__":
    # 可改成 run(mode="sgd") 观察单样本更新
    run(mode="batch", epochs=10000, lr=0.1, seed=1)
