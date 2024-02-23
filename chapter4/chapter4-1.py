"""
chapter 4.1 多层感知机(Multi-Layer Perceptron, 简称 MLP)
"""
import torch
from d2l import torch as d2l

"""
chapter 4.1.1 隐藏层(hidden layer)
隐藏层帮我我们处理非线性问题
"""

"""
chapter 4.1.2 激活函数(activation function)
1. 修正线性单元(Rectified linear unit,ReLU)
2. sigmoid 函数
3. tanh 函数
"""


def relu():
    """
    ReLU(x) = max(x, 0)
    使用ReLU的原因是,它求导表现得特别好:要么让参数消失,要么让参数通过。这使得优化表现得更好,并且 ReLU 减轻了困扰以往神经网络的梯度消失问题。
    """
    print("\n======== 修正线性单元 ReLU ==========")
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    # relu 函数图像
    y = torch.relu(x)
    d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
    # relu 导数
    y.backward(torch.ones_like(x), retain_graph=True)
    d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))


def sigmoid():
    """
    在 chapter 3.4 中已介绍过此函数
    """
    pass


def tanh():
    """
    函数: tanh(x) = (1−exp(−2x)) / (1+exp(−2x))
    梯度: 1- tanh²(x)
    1. tanh 函数将输出压缩值区间 (‐1, 1) 上
    2. 当输入在0附近时,tanh函数接近线性变换
    3. 函数的形状类似于sigmoid函数,不同的是tanh函数关于坐标系原点中心对称
    """
    print("\n======== 双曲正切函数 tanh ==========")
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = torch.tanh(x)
    d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
    # 求导数
    y.backward(torch.ones_like(x), retain_graph=True)
    d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))


if __name__ == '__main__':
    relu()
    tanh()
