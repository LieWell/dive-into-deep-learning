import math
import numpy as np
import torch
from d2l import torch as d2l

"""
chapter 3-1 线性回归
"""


def normal(x, mu, sigma):
    """
    正太分布函数
    :param x:
    :param mu:
    :param sigma:
    :return:
    """
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp(-0.5 / sigma ** 2 * (x - mu) ** 2)


def three_one_two():
    """
    chapter 3.1.2 矢量化加速
    对计算进行矢量化,从而利用线性代数库,而不是在Python中编写开销高昂的 for 循环
    """
    n = 10000
    a = torch.ones([n])
    b = torch.ones([n])

    # 创建一个计时器
    timer = d2l.Timer()

    # 调用 for 循环,做 10000 次加法
    c = torch.zeros(n)
    for i in range(n):
        c[i] = a[i] + b[i]
    print(f'\n{timer.stop():.5f} sec')

    # 直接使用向量加法
    timer.start()
    d = a + b
    print(d.sum())  # 检查向量 d 是否正确
    print(f'{timer.stop():.5f} sec')


def three_one_three():
    """
    chapter 3.1.3 正态分布与平方损失
    """
    x = np.arange(-7, 7, 0.01)  # x 轴范围 [-7,7],步长0.01
    params = [(0, 1), (0, 2), (3, 1)]
    d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
             ylabel='p(x)', figsize=(4.5, 2.5),
             legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])


if __name__ == '__main__':
    # three_one_two()
    three_one_three()
