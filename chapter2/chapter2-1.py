import torch
import numpy

"""
chapter 2.1 数据操作
"""


def two_one_one():
    """
    入门
    """
    print("\n======== 2.1.1 ==========")
    x = torch.arange(6, dtype=torch.float).reshape(3, 2)
    print("x=", x)
    print("x.shape=", x.shape)
    print("x.numel()=", x.numel())  # 元素总数


def two_one_two():
    """
    运算符
    """
    print("\n======== 2.1.2 ==========")
    x = torch.arange(12, dtype=torch.float32).reshape((3, 4))
    y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
    print("x+y=", x + y)
    print("x-y=", x - y)
    print("x*y=", x * y)
    print("x/y=", x / y)
    print("x**y=", x ** y)  # 求幂
    print("cat_x_y_dim_0=", torch.cat((x, y), dim=0))  # 沿 0 轴连结
    print("cat_x_y_dim_1=", torch.cat((x, y), dim=1))  # 沿 1 轴连结
    print("(x==y)=", x == y)


def two_one_three():
    """
    广播机制(broadcasting mechanism)
    1.通过复制适当的元素扩展两个数组,使之具有相同的形状
    2.按元素操作
    """
    print("\n======== 2.1.3 ==========")
    x = torch.arange(3).reshape((3, 1))
    y = torch.arange(2).reshape((1, 2))
    print("x=", x)
    print("y=", y)
    print("x+y=", x + y)
    print("x*y=", x * y)
    print(f"x/y={x / y}")


def two_one_four():
    """
    索引与切片
    """
    print("\n======== 2.1.4 ==========")
    # 创建一个2行,3列,深度4的张量
    # 在 0 轴上有两个元素,其坐标范围为 [0,1]
    # 在 1 轴上有两个元素,其坐标范围为 [0,2]
    # 在 2 轴上有两个元素,其坐标范围为 [0,3]
    x = torch.tensor([
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
        [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
    ])
    print(x[1])
    print(x[1][1])
    print(x[1][1][1])
    print(x[1][1][1].item())  # 转换成标量
    # 遍历打印
    y = x.numpy()
    loop_print(y)


def loop_print(ndarray):
    for v in ndarray:
        if isinstance(v, numpy.ndarray):
            loop_print(v)
        else:
            print(v, end=',')


def two_one_five():
    """
    节省内存
    """
    print("\n======== 2.1.5 ==========")
    x = torch.tensor(1)
    y = torch.tensor(1)
    print("x before=", id(x))
    x = x + y  # 操作后 x 内存地址改变(分配了新对象)
    print("x after=", id(x))
    # 原地操作方法1
    x += y  # 神奇,使用次操作地址不改变
    print("x after after=", id(x))
    # 原地操作方法2 --- 使用数组
    z = torch.arange(4).reshape(1, 4)
    x1 = torch.arange(4).reshape(1, 4)
    y1 = torch.arange(4).reshape(1, 4)
    print("z before=", z)
    print('z-id before=', id(z))
    z[:] = x1 + y1
    print("z after=", z)
    print('z-id after =', id(z))


def two_one_six():
    """
    numpy 与 tensor 类型转换
    """
    print("\n======== 2.1.6 ==========")
    x = torch.tensor(1)
    a = x.numpy()
    b = torch.tensor(a)
    print("x=", x)
    print("x.item=", x.item())
    print("x.item.type=", type(x.item()))
    print("x.type=", type(x))
    print("a.type=", type(a))
    print("b.type=", type(b))


if __name__ == '__main__':
    # two_one_one()
    # two_one_two()
    two_one_three()
    # two_one_four()
    # two_one_five()
    # two_one_six()
