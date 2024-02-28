"""
chapter 4.5 权重衰减
"""

import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l


def prepare_data():
    """
    f(x) = 0.05 + ∑0.01x + ε, ε∼N(0,0.01²)
    函数中使用了求和,因此函数的项实际上是不确定的:
    f(0) = 0.05
    f(1) = 0.05 + (0.01 * 1) + ε
    f(2) = 0.05 + (0.01 * 1) + (0.01 * 2) + ε
    标签同时被均值为0,标准差为0.01高斯噪声破坏。为了使过拟合的效果更加明显,我们可以将问题的维数增加到d = 200,并使用一个只包含20个样本的小训练集
    """
    n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
    # 初始化参数
    # 权重为 torch.Size([200, 1]) 的张量,偏置为 0.5 的标量
    true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
    print(f"true_w.len={len(true_w)};true_w.shape={true_w.shape};true_b={true_b}")
    # 根据 w 与 b 生成标准正态分布的数据,该数据增加了噪声
    train_data = d2l.synthetic_data(true_w, true_b, n_train)
    train_iter = d2l.load_array(train_data, batch_size)
    test_data = d2l.synthetic_data(true_w, true_b, n_test)
    test_iter = d2l.load_array(test_data, batch_size, is_train=False)
    return train_iter, test_iter, num_inputs, batch_size


def init_params(num_inputs):
    """ 使用正态分布初始化参数 """
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


def l2_penalty(w):
    """
    定义 L2 范数惩罚 --- 实现这一惩罚最方便的方法是对所有项求平方后并将它们求和
    """
    return torch.sum(w.pow(2)) / 2


def weight_decay_from_zero(lambd):
    print(f"\n======== weight_decay_from_zero ==========")
    train_iter, test_iter, num_inputs, batch_size = prepare_data()
    w, b = init_params(num_inputs)
    # lambda 关键字用来定义匿名函数
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了 L2 范数惩罚项, lambd 参数为 0 时惩罚项会消失
            # 广播机制使 l2_penalty(w) 成为一个长度为 batch_size 的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
            if (epoch + 1) % 5 == 0:
                animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                         d2l.evaluate_loss(net, test_iter, loss)))
    d2l.plt.show()
    print('w的L2范数是:', torch.norm(w).item())


def weight_decay_simple(wd):
    print(f"\n======== weight_decay_simple ==========")
    train_iter, test_iter, num_inputs, batch_size = prepare_data()
    w, b = init_params(num_inputs)
    # 定义线性模型
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    print(f"net.parameters={net.parameters}")
    print(f"net[0]={net[0]}")
    for param in net.parameters():
        param.data.normal_()  # 使用标准正态分布填充 data
    # 损失函数为均方误差
    loss = nn.MSELoss()
    num_epochs, lr = 100, 0.003
    # 偏置不参与衰减
    # 权重衰减由 wd 参数决定
    trainer = torch.optim.SGD([
        {"params": net[0].weight, 'weight_decay': wd},
        {"params": net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()  # TODO 这里使用 mean() 的意义是什么
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss), d2l.evaluate_loss(net, test_iter, loss)))
    d2l.plt.show()
    print('w的L2范数:', net[0].weight.norm().item())


def four_five():
    """
    chapter 4.5.1 高纬线性回归
    """
    print(f"\n======== chapter 4.5.1 ==========")
    # 用 lambd = 0禁用权重衰减后运行这个代码。
    # 注意,这里训练误差有了减少,但测试误差没有减少,这意味着出现了严重的过拟合
    # weight_decay_from_zero(lambd=0)
    # 使用权重衰减后运行这个代码
    # 注意,在这里训练误差增大,但测试误差减小。这正是我们期望从正则化中得到的效果。
    # weight_decay_from_zero(lambd=3)
    # 简约实现
    # weight_decay_simple(0)
    weight_decay_simple(3)


if __name__ == "__main__":
    four_five()
