"""改进版全连接网络：整批向量化 + softmax/交叉熵 + He/Xavier 初始化 + L2 + 动量。

相对 network_MIST.py（基线）的改动，每条都能单独解释清楚：

1. 向量化：mini-batch 一次矩阵乘完成，不再逐样本 for 循环 —— 快一个量级。
2. softmax + 交叉熵代价：替代 sigmoid + 平方误差。平方误差的梯度带一个
   sigmoid'(z) 因子，输出层一旦饱和（预测很自信但错了）梯度趋近于 0，学得极慢；
   交叉熵下 delta = a - y，错得越离谱学得越快。
3. 初始化按 1/sqrt(n_in)（sigmoid/tanh 用 Xavier，ReLU 用 He）缩放。原版
   randn(y, x) 让 z 的标准差约等于 sqrt(784) ≈ 28，一开始几乎所有神经元就饱和了，
   这正是基线前三个 epoch 卡在 80% 左右的原因。
4. ReLU 隐藏层（可切回 sigmoid），配合 He 初始化。
5. L2 正则 + 动量 + 学习率衰减。
6. 支持保存/加载权重、固定随机种子复现。
"""
from __future__ import annotations

import time

import numpy as np


# ---------- 激活函数 ----------
def sigmoid(z):
    # 等价于 1/(1+exp(-z))，但对大负数不会溢出
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.abs(z))),
                    np.exp(-np.abs(z)) / (1.0 + np.exp(-np.abs(z))))


def sigmoid_prime(z):
    s = sigmoid(z)
    return s * (1.0 - s)


def relu(z):
    return np.maximum(z, 0.0)


def relu_prime(z):
    return (z > 0).astype(z.dtype)


ACTIVATIONS = {"sigmoid": (sigmoid, sigmoid_prime), "relu": (relu, relu_prime)}


def softmax(z):
    z = z - z.max(axis=0, keepdims=True)  # 减最大值防溢出
    e = np.exp(z)
    return e / e.sum(axis=0, keepdims=True)


class Network:
    def __init__(self, sizes, activation: str = "relu", seed: int | None = None):
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation 必须是 {list(ACTIVATIONS)} 之一")
        self.sizes = list(sizes)
        self.num_layers = len(sizes)
        self.activation_name = activation
        self.act, self.act_prime = ACTIVATIONS[activation]

        rng = np.random.default_rng(seed)
        # ReLU 用 He（gain=2），sigmoid 用 Xavier（gain=1）
        gain = 2.0 if activation == "relu" else 1.0
        self.weights = [rng.standard_normal((y, x)) * np.sqrt(gain / x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.zeros((y, 1)) for y in sizes[1:]]
        self._reset_momentum()

    def _reset_momentum(self):
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.v_b = [np.zeros_like(b) for b in self.biases]

    # ---------- 前向 ----------
    def feedforward(self, X):
        """X: (784, n) -> (10, n) 概率。最后一层是 softmax，其余是 act。"""
        A = np.atleast_2d(X)
        last = self.num_layers - 2
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            Z = w @ A + b
            A = softmax(Z) if i == last else self.act(Z)
        return A

    def predict(self, X):
        return np.argmax(self.feedforward(X), axis=0)

    # ---------- 反向 ----------
    def backprop(self, X, Y):
        """X: (784, m)，Y: (10, m) one-hot。返回按 batch 平均后的梯度。"""
        m = X.shape[1]
        A = X
        activations = [X]
        zs = []
        last = self.num_layers - 2
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            Z = w @ A + b
            zs.append(Z)
            A = softmax(Z) if i == last else self.act(Z)
            activations.append(A)

        # softmax + 交叉熵：delta 直接就是 a - y，没有激活导数因子
        delta = (activations[-1] - Y) / m
        nabla_b = [None] * len(self.biases)
        nabla_w = [None] * len(self.weights)
        nabla_b[-1] = delta.sum(axis=1, keepdims=True)
        nabla_w[-1] = delta @ activations[-2].T

        for li in range(2, self.num_layers):
            delta = (self.weights[-li + 1].T @ delta) * self.act_prime(zs[-li])
            nabla_b[-li] = delta.sum(axis=1, keepdims=True)
            nabla_w[-li] = delta @ activations[-li - 1].T
        return nabla_b, nabla_w

    def update_mini_batch(self, X, Y, eta, lmbda=0.0, n=1, mu=0.0):
        nabla_b, nabla_w = self.backprop(X, Y)
        for i in range(len(self.weights)):
            grad_w = nabla_w[i] + (lmbda / n) * self.weights[i]  # L2 权重衰减
            self.v_w[i] = mu * self.v_w[i] - eta * grad_w
            self.v_b[i] = mu * self.v_b[i] - eta * nabla_b[i]
            self.weights[i] += self.v_w[i]
            self.biases[i] += self.v_b[i]

    # ---------- 评估 ----------
    def accuracy(self, data):
        X, y = data
        return float(np.mean(self.predict(X) == y))

    def loss(self, data, lmbda=0.0):
        X, y = data
        P = self.feedforward(X)
        ce = -np.mean(np.log(P[y, np.arange(len(y))] + 1e-12))
        l2 = 0.5 * lmbda / X.shape[1] * sum(np.sum(w ** 2) for w in self.weights)
        return ce + l2

    # ---------- 训练 ----------
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda=0.0, mu=0.0, eval_data=None, lr_decay=1.0,
            early_stopping=0, seed=0, verbose=True):
        """带 L2 / 动量 / 学习率衰减 / 早停（保留最佳权重）的小批量 SGD。"""
        from mnist_loader import one_hot

        X, y = training_data
        Y = one_hot(y, self.sizes[-1])
        n = X.shape[1]
        rng = np.random.default_rng(seed)

        history = []
        best_acc, best_state, stale = -1.0, None, 0
        for j in range(epochs):
            t0 = time.time()
            perm = rng.permutation(n)
            Xs, Ys = X[:, perm], Y[:, perm]
            for k in range(0, n, mini_batch_size):
                self.update_mini_batch(Xs[:, k:k + mini_batch_size],
                                       Ys[:, k:k + mini_batch_size],
                                       eta, lmbda, n, mu)
            eta *= lr_decay

            rec = {"epoch": j + 1, "eta": eta, "secs": time.time() - t0}
            history.append(rec)
            if eval_data is None:
                if verbose:
                    print(f"周期 {j + 1:>3} 完成 ({rec['secs']:.1f}s)")
                continue
            improved = self._record_eval(rec, eval_data, best_acc, verbose)
            if improved:
                best_acc, stale, best_state = rec["eval_acc"], 0, self.get_state()
            else:
                stale += 1
                if early_stopping and stale >= early_stopping:
                    if verbose:
                        print(f"早停：验证准确率已连续 {stale} 轮没有提升")
                    break

        if best_state is not None:
            self.set_state(best_state)  # 回到验证集上最好的那组权重
            if verbose:
                print(f"恢复最佳权重（验证准确率 {best_acc * 100:.2f}%）")
        return history

    def _record_eval(self, rec, eval_data, best_acc, verbose):
        """在验证集上评估一轮，写进 rec，返回是否刷新了最佳成绩。"""
        rec["eval_acc"] = self.accuracy(eval_data)
        rec["eval_loss"] = self.loss(eval_data)
        if verbose:
            print(f"周期 {rec['epoch']:>3}: 验证 {rec['eval_acc'] * 100:.2f}% "
                  f"loss {rec['eval_loss']:.4f}  ({rec['secs']:.1f}s)")
        return rec["eval_acc"] > best_acc

    # ---------- 存取 ----------
    def get_state(self):
        return ([w.copy() for w in self.weights], [b.copy() for b in self.biases])

    def set_state(self, state):
        self.weights = [w.copy() for w in state[0]]
        self.biases = [b.copy() for b in state[1]]
        self._reset_momentum()

    def save(self, path):
        np.savez(path, sizes=np.array(self.sizes), activation=self.activation_name,
                 **{f"w{i}": w for i, w in enumerate(self.weights)},
                 **{f"b{i}": b for i, b in enumerate(self.biases)})

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        net = cls([int(s) for s in d["sizes"]], activation=str(d["activation"]))
        net.weights = [d[f"w{i}"] for i in range(len(net.weights))]
        net.biases = [d[f"b{i}"] for i in range(len(net.biases))]
        net._reset_momentum()
        return net
