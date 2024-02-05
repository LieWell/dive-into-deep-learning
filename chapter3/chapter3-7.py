"""
chapter 3-6 softmax 回归简洁实现
此节代码有问题
"""

import torch
from torch import nn
from d2l import torch as d2l


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


def three_seven():
    # 加载数据集
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    # 3.7.1 初始化模型参数
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    net.apply(init_weights)
    # 3.7.2 重新审视 Softmax 实现
    loss = nn.CrossEntropyLoss(reduction='none')
    # 3.7.3 优化算法
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)
    # 3.7.4 训练
    num_epochs = 10
    # 直接运行会报错,修改了 d2l.train_epoch_ch3 函数
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    d2l.plt.show()


if __name__ == '__main__':
    three_seven()
