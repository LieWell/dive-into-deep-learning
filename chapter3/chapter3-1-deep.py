import math
import numpy as np
import torch
from d2l import torch as d2l

"""
1、似然(likehood)
2、损失函数(loss function)
3、随机梯度下降（stochastic gradient descent，SGD）
"""


def like_hood():
    """

    :return:
    """
    pass


def loss_func():
    """
    损失函数用来度量模型的 预测值 f(x) 与真实值 Y 的差异程度的运算函数。它是一个非负实值函数，通常使用L(Y, f(x))来表示，损失函数越小，模型越精确。
    损失函数使用主要是在模型的训练阶段，每个批次的训练数据送入模型后，通过前向传播输出预测值，然后损失函数会计算出预测值和真实值之间的差异值，也就是损失值。
    得到损失值之后，模型通过反向传播去更新各个参数，来降低真实值与预测值之间的损失，使得模型生成的预测值往真实值方向靠拢，从而达到学习的目的。
    常用的损失函数有：
    1、L1 范数损失: L1 Loss
    2、均方误差: MSE Loss
    3、交叉熵损失 CrossEntropyLoss
    """
    pass


def sgd():
    """
    随机梯度下降
    http://sofasofa.io/tutorials/python_gradient_descent/
    """
    pass


if __name__ == '__main__':
    pass
