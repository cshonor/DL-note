"""多输入单输出：一次前向 + 平方误差 + delta + 逐权重带 alpha 更新（与 01_多输入…md 数值一致）。"""


def w_sum(a, b):
    assert len(a) == len(b)
    return sum(a[i] * b[i] for i in range(len(a)))


def neural_network(input_vec, w):
    return w_sum(input_vec, w)


def ele_mul(scalar, vector):
    return [scalar * vector[i] for i in range(len(vector))]


def main():
    weights = [0.1, 0.2, -0.1]
    input_vec = [8.5, 0.65, 1.2]
    true = 1.0
    alpha = 0.01

    pred = neural_network(input_vec, weights)
    error = (pred - true) ** 2
    delta = pred - true
    weight_deltas = ele_mul(delta, input_vec)

    print("pred:", pred)
    print("error:", error)
    print("delta:", delta)
    print("weight_deltas:", weight_deltas)
    print("weights (before):", weights)

    for i in range(len(weights)):
        weights[i] -= alpha * weight_deltas[i]

    print("weights (after): ", weights)


if __name__ == "__main__":
    main()
