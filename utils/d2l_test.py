from d2l import torch as d2l
import torch


def d2l_test_load_data_fashion_mnist():
    """
    Fashion‐MNIST 数据集由 10 个类别的图像组成,
    每个类别由训练数据集(train dataset)中的 6000 张图像和测试数据集(test dataset)中的 1000 张图像组成。
    因此,训练集和测试集总共分别包含60000和10000张图像
    """
    batch_size = 100
    # 返回的是一个 DataLoader
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    # X 张量 [batch_size, 1, 18, 18] 使用 nn.Flatten 展平后变为 batch 为 100 的 一维数组
    # y 是 对应的 100 个标签
    for X, y in train_iter:
        print(f"X.shape: {X.shape}, y: {y}")
        break


def d2l_test_synthetic_data():
    print('\n======== d2l_test_synthetic_data ======')
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
    # d2l_test_synthetic_data()
    d2l_test_load_data_fashion_mnist()
