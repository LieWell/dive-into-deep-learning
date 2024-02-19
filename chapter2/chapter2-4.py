import numpy as np
from d2l import torch as d2l  # 引入 沐神 封装好的一些工具

"""
2-4 微积分
此节内容重在理解
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
    """
    pass


def two_four_three():
    """
    2-4-3 梯度(gradient)
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
