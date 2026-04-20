"""冻结权重 a 的最小可复现实验（带逐行解释日志）。

实验 A：只训练 b、c，始终冻结 a。
实验 B：先冻结 a 训练到几乎零误差，再解冻 a，观察其梯度是否消失。
"""

from __future__ import annotations


def dot(x: list[float], w: list[float]) -> float:
    return sum(xi * wi for xi, wi in zip(x, w))


def one_step(
    w: list[float],
    x: list[float],
    y: float,
    lr: float,
    freeze_idx: int = -1,
) -> tuple[float, float, float, list[float]]:
    """单样本一步训练：线性前向 + MSE + 手工梯度。"""
    pred = dot(x, w)
    error = (pred - y) ** 2
    delta = pred - y

    # dE/dw_i = (pred - y) * x_i
    grads = [delta * xi for xi in x]

    # 冻结某个权重：梯度直接置零，该权重完全不动
    if 0 <= freeze_idx < len(w):
        grads[freeze_idx] = 0.0

    for i in range(len(w)):
        w[i] -= lr * grads[i]

    return pred, error, delta, grads


def exp_a_freeze_forever() -> None:
    print("\n=== 实验 A：a 全程冻结，只训练 b/c ===")
    x = [8.5, 0.65, 1.2]
    y = 1.0
    w = [0.1, 0.2, -0.1]  # a, b, c
    lr = 0.01

    for ep in range(1, 121):
        pred, err, delta, grads = one_step(w, x, y, lr, freeze_idx=0)
        if ep == 1 or ep % 20 == 0 or ep == 120:
            print(
                f"ep={ep:3d} pred={pred:.6f} err={err:.8f} delta={delta:+.6f} "
                f"grads={[round(g, 6) for g in grads]} w={[round(v, 6) for v in w]}"
            )


def exp_b_freeze_then_unfreeze() -> None:
    print("\n=== 实验 B：先冻结 a，再解冻 a ===")
    x = [8.5, 0.65, 1.2]
    y = 1.0
    w = [0.1, 0.2, -0.1]  # a, b, c
    lr = 0.01
    unfreeze_epoch = 120

    for ep in range(1, 201):
        freeze_idx = 0 if ep < unfreeze_epoch else -1
        pred, err, delta, grads = one_step(w, x, y, lr, freeze_idx=freeze_idx)
        if ep in (1, 25, 50, 75, 100, 120, 125, 150, 175, 200):
            state = "freeze-a" if freeze_idx == 0 else "unfrozen"
            print(
                f"ep={ep:3d} [{state:8s}] pred={pred:.6f} err={err:.10f} "
                f"delta={delta:+.8f} grad_a={grads[0]:+.8f} w_a={w[0]:.6f}"
            )

    print(
        "\n观察点：如果误差在解冻前已接近 0，那么 delta≈0 => grad_a≈0，"
        "即使解冻后 a 也几乎不再更新。"
    )


if __name__ == "__main__":
    exp_a_freeze_forever()
    exp_b_freeze_then_unfreeze()
