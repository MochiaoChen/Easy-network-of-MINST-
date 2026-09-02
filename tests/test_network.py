"""数值梯度检查 + 基本性质测试。CI 里 pytest 之前是空跑（exit code 5）。"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mnist_loader import one_hot  # noqa: E402
from network import Network, softmax  # noqa: E402


def cross_entropy(net, X, Y):
    P = net.feedforward(X)
    return -np.sum(Y * np.log(P + 1e-12)) / X.shape[1]


@pytest.mark.parametrize("activation", ["sigmoid", "relu"])
def test_backprop_matches_numerical_gradient(activation):
    """反向传播的解析梯度必须和有限差分一致——这条测试能挡住绝大多数手推 BP 的错。"""
    rng = np.random.default_rng(0)
    net = Network([6, 5, 4, 3], activation=activation, seed=1)
    X = rng.random((6, 7))
    y = rng.integers(0, 3, size=7)
    Y = one_hot(y, 3)

    nabla_b, nabla_w = net.backprop(X, Y)
    eps = 1e-6
    for li in range(len(net.weights)):
        for _ in range(15):  # 每层随机抽查若干个权重
            i = rng.integers(net.weights[li].shape[0])
            j = rng.integers(net.weights[li].shape[1])
            orig = net.weights[li][i, j]
            net.weights[li][i, j] = orig + eps
            plus = cross_entropy(net, X, Y)
            net.weights[li][i, j] = orig - eps
            minus = cross_entropy(net, X, Y)
            net.weights[li][i, j] = orig
            numeric = (plus - minus) / (2 * eps)
            assert numeric == pytest.approx(nabla_w[li][i, j], abs=1e-6)

    for li in range(len(net.biases)):
        for i in range(net.biases[li].shape[0]):
            orig = net.biases[li][i, 0]
            net.biases[li][i, 0] = orig + eps
            plus = cross_entropy(net, X, Y)
            net.biases[li][i, 0] = orig - eps
            minus = cross_entropy(net, X, Y)
            net.biases[li][i, 0] = orig
            assert (plus - minus) / (2 * eps) == pytest.approx(nabla_b[li][i, 0], abs=1e-6)


def test_softmax_is_normalized_and_overflow_safe():
    z = np.array([[1000.0, -1000.0], [1000.0, 0.0], [-1000.0, 1000.0]])
    p = softmax(z)
    assert np.allclose(p.sum(axis=0), 1.0)
    assert np.isfinite(p).all()


def test_shapes_and_prediction_range():
    net = Network([784, 12, 10], seed=0)
    X = np.random.default_rng(0).random((784, 9))
    assert net.feedforward(X).shape == (10, 9)
    pred = net.predict(X)
    assert pred.shape == (9,) and pred.min() >= 0 and pred.max() <= 9


def test_init_scale_keeps_preactivations_sane():
    """原版 randn(y, x) 会让 z 的标准差 ~sqrt(784)，一开始就饱和。"""
    net = Network([784, 100, 10], activation="relu", seed=0)
    X = np.random.default_rng(0).random((784, 64))
    z = net.weights[0] @ X + net.biases[0]
    assert z.std() < 20


def test_learns_on_tiny_synthetic_problem():
    """在小的线性可分问题上，几百步内必须把训练集拟合到接近 100%。"""
    rng = np.random.default_rng(0)
    X = rng.random((20, 200))
    y = (X[:10].sum(axis=0) > X[10:].sum(axis=0)).astype(int)
    net = Network([20, 16, 2], seed=0)
    net.SGD((X, y), epochs=200, mini_batch_size=20, eta=0.1, mu=0.9,
            seed=0, verbose=False)
    assert net.accuracy((X, y)) > 0.95


def test_save_load_roundtrip(tmp_path):
    net = Network([8, 5, 3], activation="sigmoid", seed=3)
    X = np.random.default_rng(1).random((8, 4))
    before = net.feedforward(X)
    path = tmp_path / "m.npz"
    net.save(str(path))
    loaded = Network.load(str(path))
    assert loaded.sizes == net.sizes and loaded.activation_name == "sigmoid"
    assert np.allclose(loaded.feedforward(X), before)


def test_l2_shrinks_weights_without_data_gradient():
    """只有 L2 项时，权重应当被等比例收缩。"""
    net = Network([4, 3, 2], seed=0)
    net.backprop = lambda X, Y: ([np.zeros_like(b) for b in net.biases],
                                 [np.zeros_like(w) for w in net.weights])
    w0 = net.weights[0].copy()
    net.update_mini_batch(None, None, eta=0.1, lmbda=1.0, n=10, mu=0.0)
    assert np.allclose(net.weights[0], w0 * (1 - 0.1 * 1.0 / 10))
