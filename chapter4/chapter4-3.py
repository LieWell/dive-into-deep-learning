"""
chapter 4.3 多层感知机简洁实现
"""

import torch
from torch import nn
from d2l import torch as d2l


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


def mlp_simple():
    print("\n======== 4.3 MLP 简洁实现 ==========")
    # 加载数据集
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    net = nn.Sequential(nn.Flatten(),
                        nn.Linear(784, 256),
                        nn.ReLU(),
                        nn.Linear(256, 10))
    net.apply(init_weights)
    lr, num_epochs = 0.1, 10
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, optimizer)
    d2l.plt.show()
    # 评估
    d2l.predict_ch3(net, test_iter, n=10)
    d2l.plt.show()


if __name__ == '__main__':
    mlp_simple()
