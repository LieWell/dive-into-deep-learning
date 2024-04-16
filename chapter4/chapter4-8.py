"""
chapter 4.8 数值稳定性和模型初始化
TODO  这一节太难了,作为了解,后面补充
"""

import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l


def four_eight_one():
    """
    chapter 4.8.1
    梯度消失(gradient vanishing) 参数更新过小,在每次更新时几乎不会移动,导致模型无法学习
    梯度爆炸(gradient exploding) 参数更新过大,破坏了模型的稳定收敛
    """

    # 梯度消失
    # 如结果图,当 sigmoid 函数的输入很大或是很小时,它的梯度都会消失
    # 因此更稳定的 ReLU 系列函数已经成为从业者的默认选择
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = torch.sigmoid(x)
    y.backward(torch.ones_like(x))
    d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
             legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))

    # 梯度爆炸
    M = torch.normal(0, 1, size=(4, 4))
    print('一个矩阵 \n', M)
    for i in range(100):
        M = torch.mm(M, torch.normal(0, 1, size=(4, 4)))
    print('乘以100个矩阵后\n', M)


if __name__ == "__main__":
    four_eight_one()
