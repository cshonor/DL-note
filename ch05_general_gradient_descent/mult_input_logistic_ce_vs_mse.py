"""对比单神经元二分类：线性+MSE vs Sigmoid+CrossEntropy。

打印每轮的预测、delta、weight_delta、loss，便于对照“同骨架不同delta”。
"""

from __future__ import annotations

import math


def w_sum(a: list[float], b: list[float]) -> float:
    assert len(a) == len(b)
    return sum(a[i] * b[i] for i in range(len(a)))


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def ele_mul(scalar: float, vector: list[float]) -> list[float]:
    return [scalar * vector[i] for i in range(len(vector))]


def step_linear_mse(weights: list[float], x: list[float], y: float, alpha: float) -> tuple[float, float, float, list[float]]:
    pred = w_sum(x, weights)
    loss = (pred - y) ** 2
    delta = 2.0 * (pred - y)  # 严格对应无 0.5 的平方误差
    wds = ele_mul(delta, x)
    for i in range(len(weights)):
        weights[i] -= alpha * wds[i]
    return pred, loss, delta, wds


def step_sigmoid_ce(weights: list[float], x: list[float], y: float, alpha: float) -> tuple[float, float, float, list[float]]:
    z = w_sum(x, weights)
    y_hat = sigmoid(z)
    eps = 1e-12
    y_hat_clamped = min(max(y_hat, eps), 1.0 - eps)
    loss = -(y * math.log(y_hat_clamped) + (1.0 - y) * math.log(1.0 - y_hat_clamped))
    delta = y_hat - y  # 交叉熵+sigmoid 顶层简化
    wds = ele_mul(delta, x)
    for i in range(len(weights)):
        weights[i] -= alpha * wds[i]
    return y_hat, loss, delta, wds


def main() -> None:
    x = [8.5, 0.65, 1.2]
    y = 1.0
    alpha = 0.01
    epochs = 8

    w_mse = [0.1, 0.2, -0.1]
    w_ce = [0.1, 0.2, -0.1]

    print("x=", x, "y=", y, "alpha=", alpha)
    print("== linear + MSE ==")
    for ep in range(1, epochs + 1):
        pred, loss, delta, wds = step_linear_mse(w_mse, x, y, alpha)
        print(f"ep={ep:2d} pred={pred:.6f} loss={loss:.6f} delta={delta:+.6f} wds={[round(v, 4) for v in wds]} w={[round(v, 4) for v in w_mse]}")

    print("\n== sigmoid + cross entropy ==")
    for ep in range(1, epochs + 1):
        y_hat, loss, delta, wds = step_sigmoid_ce(w_ce, x, y, alpha)
        print(f"ep={ep:2d} y_hat={y_hat:.6f} loss={loss:.6f} delta={delta:+.6f} wds={[round(v, 4) for v in wds]} w={[round(v, 4) for v in w_ce]}")


if __name__ == "__main__":
    main()
