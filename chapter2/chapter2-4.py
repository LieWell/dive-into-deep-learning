import numpy as np
from d2l import torch as d2l  # 引入 沐神 封装好的一些工具

"""
2-4 微积分
"""


def f(x):
    """
    f(x)=3x^2-4x
    """
    return 3 * x ** 2 - 4 * x


def two_four_one():
    """
    2-4-1 导数和微分
    """
    print("\n======== 2.4.1 ==========")
    x = np.arange(0, 3, 0.1)
    # 源码有详细解释
    d2l.plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])


def two_four_two():
    """
    2-4-2 偏导数(partial derivative)
    例如函数 f(x,y)=x^2+xy−y^2 包含两个变量,
    函数 f 对 x 的偏到数就是将 y 看成常数后求导,即 ∂f/∂x = 2x+y;
    函数 f 对 y 的偏到数就是将 x 看成常数后求导,即 ∂f/∂y = x-2y
    """
    pass


def two_four_three():
    """
    2-4-3 梯度(gradient)
    连结一个多元函数对其所有变量的偏导数，以得到该函数的梯度(gradient)向量
    例如函数 f(x,y)=x^2+xy−y^2
    函数 f 的梯度 ∇f = [∂f/∂x,∂f/∂y],即[2x+y,x-2y],在(1,1)点的梯度为向量[3,-1]
    """
    pass


def two_four_four():
    """
    2-4-4 链式法则
    """
    pass


if __name__ == '__main__':
    two_four_one()
    two_four_two()
    two_four_three()
    two_four_four()
