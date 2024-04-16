"""
chapter 4.6 暂退法(Dropout)
"""

import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

# 定义丢弃概率
dropout1 = 0.2
dropout2 = 0.5


def dropout_layer(X, dropout):
    """
    Inverted Dropout 实现
    """
    assert 0.0 <= dropout <= 1.0
    # 丢弃张量 X 的全部元素(将全部元素置为 0)
    if dropout == 1.0:
        return torch.zeros_like(X)
    # 保留张量 X 的全部元素(保留全部元素)
    if dropout == 0.0:
        return X
    # torch.rand(X.shape): 随机生成范围为 [0,1) 的张量
    #  > dropout: 将张量的每个值跟 dropout 进行比较, 转为为 True/False 张量
    # .float(): 将 True/False 张量 转换为 0/1 张量
    mask = (torch.rand(X.shape) > dropout).float()
    # 这里除以 (1.0 - dropout) 是 Inverted Dropout
    # 具体请参考文件 functions/dropout.py
    return mask * X / (1.0 - dropout)


def dropout_layer_test():
    X = torch.arange(16, dtype=torch.float32).reshape((2, 8))
    print(f"\n{X}")
    print(dropout_layer(X, 0.))
    print(dropout_layer(X, 0.5))
    print(dropout_layer(X, 1.))


class Net(nn.Module):

    def __init__(self,
                 num_inputs,
                 num_outputs,
                 num_hiddens1,
                 num_hiddens2,
                 is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        # 第 1 层, 784 --> 256
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        # 第 2 层, 256 --> 256
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        # 第 3 层, 256 --> 10
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, X):
        global dropout1, dropout2
        # X 是 28x28 的矩阵, num_inputs=784,因此这里将变为 1 行 784 列的矩阵
        temp_input = X.reshape((-1, self.num_inputs))
        # 送入第一层神经网络,拿到 256 个输出并使用激活函数激活
        # 注意,激活函数不会改变参与计算的神经元的数量,暂退法会改变参与计算的神经元数量
        H1 = self.relu(self.lin1(temp_input))
        # 只有在训练模型时才使用dropout
        if self.training:
            # 在第一个全连接层之后添加一个dropout层
            H1 = dropout_layer(H1, dropout1)
        # 第二层神经网络
        H2 = self.relu(self.lin2(H1))
        if self.training:
            # 在第二个全连接层之后添加一个dropout层
            H2 = dropout_layer(H2, dropout2)
        # 输出层
        out = self.lin3(H2)
        return out


def dropout_from_zero():
    """
    chapter 4.6.4 从零实现
    """
    print(f"\n======== chapter 4.6.4 ==========")
    # 基于 Fashion‐MNIST 数据集, 具有两个隐藏层
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
    num_epochs, lr, batch_size = 10, 0.5, 256
    loss = nn.CrossEntropyLoss()
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    d2l.plt.show()


def dropout_simple():
    """
    chapter 4.6.5 简洁实现
    """
    print(f"\n======== chapter 4.6.5 ==========")
    net = nn.Sequential(nn.Flatten(),  # 展平层
                        nn.Linear(784, 256),  # 第 1 层网络
                        nn.ReLU(),  # 激活函数
                        nn.Dropout(dropout1),  # 第 1 个 dropout 层
                        nn.Linear(256, 256),  # 第 2 层网络
                        nn.ReLU(),  # 激活函数
                        nn.Dropout(dropout2),  # 第 2 个 dropout 层
                        nn.Linear(256, 10))  # 输出层

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights)
    num_epochs, lr, batch_size = 10, 0.5, 256
    loss = nn.CrossEntropyLoss()
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    d2l.plt.show()


if __name__ == "__main__":
    # dropout_layer_test()
    # dropout_from_zero()
    dropout_simple()
