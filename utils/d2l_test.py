from d2l import torch as d2l
import torch


def d2l_test_synthetic_data():
    print('\n========d2l_synthetic_data_test======')
    w = torch.tensor([1.0, 2.0])
    print('w.shape=', w.shape)
    b = 3.0
    number = 3
    # 生成标准正态分布,其形状为 (3,2)
    x = d2l.normal(0, 1, (number, len(w)))
    print('\nX:', x)
    # matmul 是矩阵乘法
    y = d2l.matmul(x, w) + b
    print('\ny:', y)


if __name__ == '__main__':
    d2l_test_synthetic_data()
