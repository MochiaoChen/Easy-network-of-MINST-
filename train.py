#!/usr/bin/env python3
"""训练入口。

    python3 train.py                       # 默认配置，约 98.3%
    python3 train.py --sizes 784 30 10 --activation sigmoid --epochs 10
    python3 train.py --save model.npz
    python3 train.py --load model.npz --eval-only
"""
import argparse
import warnings

# numpy 2.x + macOS Accelerate BLAS 会在 matmul 里报假的浮点警告，计算结果是对的。
# 只在命令行入口按精确 message 过滤，库代码里不动 errstate。
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

import mnist_loader  # noqa: E402
from network import Network  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="MNIST 全连接网络")
    p.add_argument("--sizes", type=int, nargs="+", default=[784, 100, 10],
                   help="各层神经元数，默认 784 100 10")
    p.add_argument("--activation", choices=["relu", "sigmoid"], default="relu")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eta", type=float, default=0.1, help="学习率")
    p.add_argument("--lmbda", type=float, default=1.0, help="L2 正则系数")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--lr-decay", type=float, default=0.97, help="每轮学习率乘以该系数")
    p.add_argument("--early-stopping", type=int, default=8,
                   help="验证集连续 N 轮无提升就停，0 表示关闭")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--load", type=str, default=None)
    p.add_argument("--eval-only", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    train, val, test = mnist_loader.load_data(seed=args.seed)
    print(f"训练 {train[0].shape[1]} / 验证 {val[0].shape[1]} / 测试 {test[0].shape[1]}")

    if args.load:
        net = Network.load(args.load)
        print(f"已加载模型 {args.load}：{net.sizes} ({net.activation_name})")
    else:
        net = Network(args.sizes, activation=args.activation, seed=args.seed)

    if not args.eval_only:
        print(f"结构 {net.sizes} | {net.activation_name} | eta={args.eta} "
              f"lmbda={args.lmbda} momentum={args.momentum} batch={args.batch_size}")
        net.SGD(train, epochs=args.epochs, mini_batch_size=args.batch_size,
                eta=args.eta, lmbda=args.lmbda, mu=args.momentum, eval_data=val,
                lr_decay=args.lr_decay, early_stopping=args.early_stopping,
                seed=args.seed)

    # 测试集只在最后碰一次：超参是在验证集上选的
    print(f"\n验证集准确率：{net.accuracy(val) * 100:.2f}%")
    print(f"测试集准确率：{net.accuracy(test) * 100:.2f}%")

    if args.save:
        net.save(args.save)
        print(f"模型已保存到 {args.save}")
    return net


if __name__ == "__main__":
    main()
