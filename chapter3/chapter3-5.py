import numpy as np
import torch
from torch.utils import data
from torch import nn
from d2l import torch as d2l
import torch
import torchvision
from torch.utils import data
from torchvision import transforms

"""
chapter 3-5 图像分类数据集
"""


def three_five_one():
    """
    chapter 3.5.1 读取 Fashion‐MNIST 数据集
    Fashion‐MNIST 由 10 个类别的图像组成,
    每个类别由训练数据集(train dataset)中的6000张图像和测试数据集(test dataset)中的1000张图像组成。
    因此,训练集和测试集分别包含60000和10000张图像
    """
    print("\n======== 3.5.1 ==========")
    # 通过 ToTensor 实例将图像数据从 PIL 类型变换成32位浮点数格式
    # 并除以255使得所有像素的数值均在 0~1 之间
    trans = transforms.ToTensor()
    # 训练数据集,download=True 自动下载数据集
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=False)
    # 测试数据集
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=False)
    print(f"mnist_train.len={len(mnist_train)}")
    print(f"mnist_test.len={len(mnist_test)}")
    # 可视化部分样本
    d2l.use_svg_display()
    X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
    print(f"X.len={len(X)}")
    titles = d2l.get_fashion_mnist_labels(y)
    d2l.show_images(X.reshape(18, 28, 28), 2, 9, titles)
    d2l.plt.show()


def three_five_two():
    """
    chapter 3.5.2 读取小批量样本
    """
    print("\n======== 3.5.2 ==========")
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=False)
    # 使用4个线程,按照 batch_size=256 读取全部数据集
    batch_size = 256
    number_workers = d2l.get_dataloader_workers()
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=number_workers)
    # 共需读取 235 次(60000/256≈235)
    print(f"train_iter.len={len(train_iter)}")
    timer = d2l.Timer()
    for X, y in train_iter:
        continue
    print(f'{timer.stop():.2f} sec')


def three_five_three():
    """
    chapter 3.5.3 整合所有组件
    """
    print("\n======== 3.5.3 ==========")
    train_iter, test_iter = d2l.load_data_fashion_mnist(32, resize=64)
    for X, y in train_iter:
        print(X.shape, X.dtype, y.shape, y.dtype)
        break


if __name__ == '__main__':
    # three_five_one()
    # three_five_two()
    three_five_three()
