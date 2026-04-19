"""多输入单输出：多轮 SGD（单样本或按场循环）或全批量 BGD；打印误差与权重轨迹。

与 01_多输入神经网络_四步训练与一次迭代.md、mult_input_one_step.py 同一前向与更新式。
"""

from __future__ import annotations

import argparse


def w_sum(a: list[float], b: list[float]) -> float:
    assert len(a) == len(b)
    return sum(a[i] * b[i] for i in range(len(a)))


def neural_network(input_vec: list[float], w: list[float]) -> float:
    return w_sum(input_vec, w)


def ele_mul(scalar: float, vector: list[float]) -> list[float]:
    return [scalar * vector[i] for i in range(len(vector))]


def sgd_step(weights: list[float], input_vec: list[float], true: float, alpha: float) -> tuple[float, float, float, list[float]]:
    pred = neural_network(input_vec, weights)
    error = (pred - true) ** 2
    delta = pred - true
    wds = ele_mul(delta, input_vec)
    for i in range(len(weights)):
        weights[i] -= alpha * wds[i]
    return pred, error, delta, wds


def bgd_step(
    weights: list[float],
    inputs: list[list[float]],
    targets: list[float],
    alpha: float,
) -> tuple[float, list[float]]:
    """对多条样本的 weight_delta 取平均后更新一次。"""
    n = len(inputs)
    acc = [0.0] * len(weights)
    total_err = 0.0
    for input_vec, true in zip(inputs, targets, strict=True):
        pred = neural_network(input_vec, weights)
        total_err += (pred - true) ** 2
        delta = pred - true
        wds = ele_mul(delta, input_vec)
        for i in range(len(weights)):
            acc[i] += wds[i]
    mean_wd = [acc[i] / n for i in range(len(weights))]
    for i in range(len(weights)):
        weights[i] -= alpha * mean_wd[i]
    mse = total_err / n
    return mse, mean_wd


def main() -> None:
    p = argparse.ArgumentParser(description="多输入单神经元训练日志")
    p.add_argument("--mode", choices=("sgd0", "sgd_cycle", "bgd"), default="sgd0", help="sgd0=只用第一场；sgd_cycle=四场轮流；bgd=每步用四场平均梯度")
    p.add_argument("--epochs", type=int, default=80, help="外层迭代次数")
    p.add_argument("--every", type=int, default=10, help="每隔多少轮打印一行")
    p.add_argument("--alpha", type=float, default=0.01)
    args = p.parse_args()

    toes = [8.5, 9.5, 9.9, 9.0]
    wlrec = [0.65, 0.8, 0.8, 0.9]
    nfans = [1.2, 1.3, 0.5, 1.0]
    win_or_lose_binary = [1, 1, 0, 1]
    inputs = [[toes[i], wlrec[i], nfans[i]] for i in range(4)]
    targets = [float(win_or_lose_binary[i]) for i in range(4)]

    weights = [0.1, 0.2, -0.1]
    alpha = args.alpha

    print("mode:", args.mode, "alpha:", alpha)
    print("init weights:", weights)

    if args.mode == "bgd":
        for ep in range(1, args.epochs + 1):
            mse, mean_wd = bgd_step(weights, inputs, targets, alpha)
            if ep == 1 or ep % args.every == 0 or ep == args.epochs:
                print(f"ep={ep:4d}  mse={mse:.6f}  mean_wd={[round(x, 4) for x in mean_wd]}  w={[round(x, 4) for x in weights]}")
        return

    # SGD
    idx = 0
    for ep in range(1, args.epochs + 1):
        if args.mode == "sgd0":
            x, y = inputs[0], targets[0]
        else:
            x, y = inputs[idx], targets[idx]
            idx = (idx + 1) % len(inputs)
        pred, err, delta, wds = sgd_step(weights, x, y, alpha)
        if ep == 1 or ep % args.every == 0 or ep == args.epochs:
            print(
                f"ep={ep:4d}  pred={pred:.4f}  err={err:.6f}  delta={delta:+.4f}  "
                f"wds={[round(v, 4) for v in wds]}  w={[round(v, 4) for v in weights]}"
            )


if __name__ == "__main__":
    main()
