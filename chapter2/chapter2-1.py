import torch
import numpy


def tensor_basic():
    x = torch.ones(2, 3, 4)
    print(x)
    print(x.shape)
    print(x.size())
    print(x.numel())


def tensor_element_operation():
    print("\n张量按元素运算(shape相同)")
    x = torch.ones(2, 2)
    y = torch.tensor([[2, 2], [2, 2]])
    print("==== 基础运算")
    print(x + y)
    print(x - y)
    print(x * y)
    print(x / y)
    print(x ** y)  # 求幂
    print("==== 张量连结(concatenate)")
    print(torch.cat((x, y), dim=0))  # 沿 x 轴
    print(torch.cat((x, y), dim=1))  # 沿 y 轴
    print("==== 张量是否相同")
    print(x == y)


def tensor_slice_operation():
    print("\n张量索引与切片")
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
    # 遍历
    y = x.numpy()
    loop_print(y)


def tensor_numpy():
    d = torch.ones((2, 2))
    x = d.numpy()
    y = torch.tensor(x)
    print('\n', type(x))
    print(type(y))


def loop_print(ndarray):
    for v in ndarray:
        if isinstance(v, numpy.ndarray):
            loop_print(v)
        else:
            print(v, end=',')


if __name__ == '__main__':
    # tensor_basic()
    # tensor_element_operation()
    # tensor_slice_operation()
    tensor_numpy()
