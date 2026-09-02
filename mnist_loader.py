"""MNIST 数据加载：不依赖 TensorFlow，只用标准库 + numpy。

原版为了拿 11 MB 的数据集，整个装了一遍 TensorFlow（500 MB+）。
这里直接下载官方 npz 镜像并缓存到本地。
"""
import os
import urllib.request

import numpy as np

URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
# 优先复用已有缓存（例如以前装过 keras 留下的），否则放到仓库的 data/ 下
CACHE_CANDIDATES = [
    os.path.expanduser("~/.keras/datasets/mnist.npz"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mnist.npz"),
]


def _ensure_data() -> str:
    for path in CACHE_CANDIDATES:
        if os.path.exists(path):
            return path
    path = CACHE_CANDIDATES[-1]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"正在下载 MNIST（约 11 MB）到 {path} ...")
    urllib.request.urlretrieve(URL, path)
    return path


def load_raw():
    """返回原始的 (x_train, y_train), (x_test, y_test)，uint8 图像。"""
    with np.load(_ensure_data()) as d:
        return (d["x_train"], d["y_train"]), (d["x_test"], d["y_test"])


def load_data(validation_size: int = 10000, seed: int = 0):
    """返回 (train, validation, test)，每个都是 (X, y) 元组。

    X: (784, n) float64，已归一化到 [0, 1]；y: (n,) int64 标签。

    原版把测试集同时当验证集用（`return training_data, test_data, test_data`），
    等于拿考卷调超参，报出来的准确率是虚高的。这里从训练集里切出真正的验证集。
    """
    (x_train, y_train), (x_test, y_test) = load_raw()

    def prep(x, y):
        return x.reshape(len(x), -1).T.astype(np.float64) / 255.0, y.astype(np.int64)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x_train))
    x_train, y_train = x_train[perm], y_train[perm]

    n_val = validation_size
    val = prep(x_train[:n_val], y_train[:n_val])
    train = prep(x_train[n_val:], y_train[n_val:])
    test = prep(x_test, y_test)
    return train, val, test


def one_hot(y, num_classes: int = 10):
    """(n,) 标签 -> (num_classes, n) one-hot 矩阵。"""
    out = np.zeros((num_classes, len(y)))
    out[y, np.arange(len(y))] = 1.0
    return out
