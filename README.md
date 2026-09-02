# Easy-network-of-MNIST

一个从零手写（只用 numpy）的 MNIST 全连接网络，用来把反向传播讲明白。

仓库里有两份实现，故意都留着，方便对照：

| 文件 | 说明 | 测试集准确率 | 耗时 |
| --- | --- | --- | --- |
| `network_MIST.py` | 原始基线，算法一行没改 | **94.02%** | 17.7s（10 轮） |
| `network.py` + `train.py` | 改进版，默认 784-100-10 | **97.91%** | 5.8s（30 轮） |
| `train.py --sizes 784 300 100 10 --epochs 40` | 更深一点 | **98.38%** | 13.5s |

> 环境：macOS / Python 3.9 / numpy 2.0，CPU 单线程。

## 跑起来

```bash
pip install -r requirements.txt   # 只要 numpy
python3 train.py                  # 首次运行会自动下载 11 MB 的 MNIST 并缓存
```

常用参数：

```bash
python3 train.py --sizes 784 300 100 10 --epochs 40   # 更深的网络
python3 train.py --activation sigmoid --eta 0.5       # 换回 sigmoid
python3 train.py --save model.npz                     # 保存权重
python3 train.py --load model.npz --eval-only         # 只评估
python3 -m pytest                                     # 跑测试（含梯度校验）
```

原始基线仍可直接运行：`python3 network_MIST.py`。

## 改了什么，为什么

按对结果的影响从大到小：

**1. 交叉熵代价替代平方误差。** 平方误差的输出层梯度带一个 `sigmoid'(z)` 因子，
网络"很自信地答错"时 `sigmoid'(z) ≈ 0`，梯度几乎消失，错得越离谱学得越慢。
换成 softmax + 交叉熵后 `delta = a - y`，那个因子被消掉了。

**2. 初始化按 `1/sqrt(n_in)` 缩放。** 原版 `np.random.randn(y, x)` 让第一层
加权输入 `z` 的标准差约为 `sqrt(784) ≈ 28`，开局几乎所有神经元就已经饱和了——
这正是基线前三个周期卡在 80% 左右、第四个周期才突然跳到 92% 的原因。
sigmoid 用 Xavier，ReLU 用 He。

**3. 验证集不再等于测试集。** 原版 `load_mnist_data` 里 `return training_data,
test_data, test_data`，验证集和测试集是同一份数据，等于拿考卷调超参，报出来的
准确率是虚高的。现在从训练集切 10000 张做验证，测试集只在最后碰一次。

**4. 整批向量化。** mini-batch 的前向/反向用矩阵乘一次算完，不再逐样本 for 循环。
每轮从 1.7s 降到 0.2s，快了约 8 倍——这也是后面能随手试超参的前提。

**5. ReLU 隐藏层**（`--activation sigmoid` 可切回）、**L2 正则**、**动量**、
**学习率衰减**、**早停并回滚到验证集最优权重**。

**6. 去掉 TensorFlow 依赖。** 原版为了拿 11 MB 数据集要装 500 MB 的
TensorFlow。`mnist_loader.py` 直接下载官方 npz 并缓存，只依赖标准库和 numpy。

**7. 补测试。** CI 里写了 `pytest` 但一个测试都没有（pytest 收不到用例会以退出码 5
失败）。现在 `tests/test_network.py` 有 8 个用例，核心是**数值梯度校验**：
用有限差分逐个核对反向传播算出的偏导，sigmoid 和 ReLU 两条路径都过。手推 BP
的错基本都能被这一条挡住。

其他：固定随机种子可复现、`softmax`/`sigmoid` 做了溢出保护、支持权重存取、
命令行参数化。

## 各项改动的贡献

同样的 784-30-10 sigmoid 结构、同样 10 轮，只换代价函数和初始化（再加向量化）：

| 配置 | 测试集 | 耗时 |
| --- | --- | --- |
| 基线 784-30-10 sigmoid | 94.02% | 17.7s |
| ＋交叉熵、Xavier 初始化、向量化 | 96.44% | 4.5s |
| ＋L2、动量、学习率衰减、早停（30 轮） | 96.68% | 4.1s |
| ＋ReLU，隐藏层加到 100（30 轮） | 97.91% | 6.7s |
| 784-300-100-10（40 轮） | 98.38% | 13.5s |

再往上就该上卷积了——全连接网络在 MNIST 上大约到 98.5% 就见顶。

## 文件

- `network_MIST.py` — 原始基线，保留对照用
- `network.py` — 改进版网络
- `train.py` — 命令行训练入口
- `mnist_loader.py` — 数据加载（无需 TensorFlow）
- `tests/test_network.py` — 测试，含数值梯度校验

---

> Thanks to ZHE2018 who inspired me, and thanks to 3Blue1Brown who made videos that explained this network. And finally, this repo is pushed by Jinfei Liu who asked me to teach him these things.
