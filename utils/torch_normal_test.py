import torch

"""
torch 随机数相关的几个函数测试
"""


def rand():
    """
    返回一个张量，包含了从区间 [0, 1) 之间均匀分布中的一组随机数，数量由其 shape 决定
    """
    print('\n========== torch.fand ==========')
    # 返回2个随机数向量
    a = torch.rand(2)
    print(a)
    # 返回一个2行3列的矩阵
    b = torch.rand(2, 3)
    print(b)
    # 返回一组shape为(2, 3, 4)的随机数
    c = torch.rand(1, 2, 3)
    print(c)


def randn():
    """
    返回一个张量，包含了从标准正态分布（均值为0，方差为1，即高斯白噪声）中抽取的一组随机数。张量的形状由参数sizes定义。
    """
    print('\n========== torch.randn ==========')
    a_out = None
    a = torch.randn([2, 3], out=a_out)
    print('a:', a)
    print('a_out:', a_out)


def normal():
    """
    返回一个张量，包含从给定参数means,std的离散正态分布中抽取随机数。
    均值means是一个张量，包含每个输出元素相关的正态分布的均值。
    std是一个张量，包含每个输出元素相关的正态分布的标准差。
    均值和标准差的形状不须匹配，但每个张量的元素个数须相同。
    """
    print('\n========== torch.normal ==========')

    # 均值为 0 ,方差为1,形状为 [3] ===> 这个其实就是标准正态分布
    a = torch.normal(mean=0, std=1, size=[3])
    print('a:', a)

    # 均值为1,方差为2,形状为 (2,2)和[3,3]
    b = torch.normal(1, 2, (2, 2))
    b1 = torch.normal(1, 2, [3, 3])
    print('b:', b)
    print('b1:', b1)

    # 均值和方差是张量
    # 均值和方差都是向量,且长度相同才可以
    m = torch.tensor([1.0, 2.0, 3.0])
    s = torch.tensor([0.1, 0.2, 0.3])
    c = torch.normal(m, s)
    print('c:', c)

    # 形状为 [1,2]
    m1 = torch.tensor([
        [1.0, 2.0]
    ])
    # 形状为 [3,1]
    s1 = torch.tensor([[0.1], [0.2], [0.3]])
    # 使用广播机制将形状都扩展为 [2,3] 后再操作
    # 加法操作
    ss = m1 + s1
    print('sum:', ss)
    # 正态分布操作
    c1 = torch.normal(m1, s1)
    print('c1:', c1)


if __name__ == '__main__':
    # rand()
    # randn()
    normal()
