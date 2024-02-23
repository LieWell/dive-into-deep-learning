"""
chapter 4.2 多层感知机从零开始实现
"""

import torch
from torch import nn
from d2l import torch as d2l

# 变量
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1, b1, W2, b2 = None, None, None, None


def set_W_b():
    weight_1 = nn.Parameter(torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
    bias_1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    weight_2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
    bias_2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
    return weight_1, bias_1, weight_2, bias_2


def relu(X):
    """ ReLU激活函数 """
    a = torch.zeros_like(X)
    return torch.max(X, a)


def net(X):
    """ 模型引入了 隐藏层 """
    global W1, b1, W2, b2
    X = X.reshape((-1, num_inputs))
    # 计算隐藏层
    H = relu(X @ W1 + b1)  # 这里“@”代表矩阵乘法
    # 返回输出层
    return H @ W2 + b2


def mlp_from_zero():
    print("\n======== 4.2 MLP 从零开始实现 ==========")
    # 加载数据
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    # 初始化参数
    global W1, b1, W2, b2
    W1, b1, W2, b2 = set_W_b()
    params = [W1, b1, W2, b2]
    # 交叉熵损失函数
    loss = nn.CrossEntropyLoss()
    # 训练
    num_epochs, lr = 10, 0.1
    updater = torch.optim.SGD(params, lr=lr)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
    d2l.plt.show()
    # 评估
    d2l.predict_ch3(net, test_iter)
    d2l.plt.show()


if __name__ == '__main__':
    mlp_from_zero()
