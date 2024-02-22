import numpy as np
import torch
from torch.utils import data
from torch import nn
from d2l import torch as d2l

"""
chapter 3-4 softmax 回归(softmax regression)
本章节主要涉及概念性的问题
⭐️分类问题
⭐️独热编码(one-hot encoding) 
⭐️似然(likelihood)
⭐️损失函数(loss function)
⭐️随机梯度下降（stochastic gradient descent,SGD）
"""


def sigmoid_function():
    """
    S型函数（sigmoid function，或称乙状函数）是一种函数，因其函数图像形状像字母S得名。
    其形状曲线至少有2个焦点，也叫“二焦点曲线函数”。
    S型函数单调、有界、处处可微的实函数，它将模型的线性输出映射到 (0, 1) 的范围内，使其可以解释为概率。
    """
    pass


def softmax_function():
    """
    softmax本质上就是将结果映射到[0,1]区间且总和为1,特别适合表示概率,也可以叫做归一化
    softmax 函数定义 yˆi = e^i/∑ e^j, 其中 j=1,2,…,n
    举例我们有个输出为 [3,1,2],如何计算 softmax(i)
    S(1) = e^3/(e^3 + e^1 + e^2) ≈ 0.665
    S(2) = e^1/(e^3 + e^1 + e^2) ≈ 0.090
    S(3) = e^2/(e^3 + e^1 + e^2) ≈ 0.245
    由结果可知 i=1
    """
    c = torch.tensor([3., 1., 2.])
    r = nn.functional.softmax(c)
    # softmax[3,1,2]=tensor([0.6652, 0.0900, 0.2447])
    print(f"softmax[3,1,2]={r}")


def classification():
    """
    二分类: 逻辑回归
    多分类: 朴素贝叶斯,softmax
    多标签: 一篇新闻可能即属于机器学习、也属于推荐系统，即一篇新闻可能会有多个标签
    """
    pass


def logistic_regression():
    """
    逻辑回归的本质可以理解为通过引入激活函数（如Sigmoid函数）将线性回归模型转化为一个概率模型，从而能够进行分类任务。
    通俗的将解决的是 是/否 问题,比如: 用户是否会点击广告、是否会罹患某种疾病、是否是特定的人
    """
    pass


def one_hot_encoding():
    """
    独热编码(one-hot encoding) 是一个向量,它的分量和类别一样多。类别对应的分量设置为1,其他所有分量设置为0。
    比如有三种类别: 猫,鸡和狗,那么可以使用如下三个向量来表示:猫:[1,0,0]、鸡:[0,1,0]、狗:[0,0,1]
    """
    pass


def likelihood():
    """
    概率: 在已知参数后,预测结果
    似然: 在已知观测结果后,预估参数
    参考文件 functions/probability_likelihood.py
    """
    pass


def loss_func():
    """
    损失函数用来度量模型的 预测值 f(x) 与真实值 Y 的差异程度的运算函数。它是一个非负实值函数,通常使用L(Y, f(x))来表示。
    通俗来讲,只要一个函数能够度量变化,那么它就可以做为损失函数,不过为了计算方便,我们通常会使用一下几类函数做为损失函数:
    1. 0-1 损失函数
    2. 绝对值损失函数
    3. log对数损失函数
    4. 平方差损失函数
    5. 指数损失函数
    6. 交叉熵损失函数 --- 参考文件 functions/cross_entropy_loss.py
    等等等等
    """
    pass


def sgd():
    """
    随机梯度下降 --- 参考文件 functions/stochastic_gradient_descent.py
    """
    pass


if __name__ == '__main__':
    softmax_function()
