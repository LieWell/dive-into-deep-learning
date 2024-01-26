import math
import numpy as np
import torch


def normal(x, mu, sigma):
    """
    正态分布概率密度函数
    :param x: 数据
    :param mu: 均值
    :param sigma: 方差
    """
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp(-0.5 / sigma ** 2 * (x - mu) ** 2)


def add_one(x):
    return x + 1


def for_test():
    x = np.arange(-5, 5, 1)
    print('\n', type(x))
    normal_result = normal(x, 0, 1)
    add_one_result = add_one(x)
    print(normal_result)
    print(add_one_result)

    # 入参是向量时,会对每个元素进行计算
    y = torch.ones(3)
    res = add_one(y)
    print(res)
    print(type(res))


if __name__ == '__main__':
    for_test()
