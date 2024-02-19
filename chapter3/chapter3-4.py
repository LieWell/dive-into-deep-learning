import numpy as np
import torch
from torch.utils import data
from torch import nn
from d2l import torch as d2l

"""
chapter 3-4 softmax 回归(softmax regression)
"""

"""
chapter 3.4.1分类问题
独热编码(one-hot encoding) 是一个向量，它的分量和类别一样多。
类别对应的分量设置为1，其他所有分量设置为0。
比如有三种类别, 猫,鸡和狗,那么可以使用如下三个向量来表示:猫:[1,0,0]、鸡:[0,1,0]、狗:[0,0,1]
"""


def logistic_regression():
    """
    logistic 回归(logistic regression)虽然被称为回归,但其实是一个分类模型,常用于二分类问题。
    其本质是：假设数据服从这个分布，然后使用极大似然估计做参数的估计。
    """


if __name__ == '__main__':
    pass
