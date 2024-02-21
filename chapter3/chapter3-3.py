import torch
from torch import nn
from d2l import torch as d2l

"""
chapter 3-3 线性回归的简洁实现
本节利用 torch 预定义的资源重新实现线性回归模型,本质上是就是函数库的调用,因此不再细分章节实现
"""


def three_three():
    print("\n======== 3.3 ==========")
    # 生成数据集
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = d2l.synthetic_data(true_w, true_b, 1000)
    # 读取数据集
    batch_size = 10
    data_iter = d2l.load_array((features, labels), batch_size)
    # 定义模型
    net = nn.Sequential(nn.Linear(2, 1))
    # 初始化参数
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)
    # 定义损失函数
    loss = nn.MSELoss()
    # 定义优化算法
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)
    # 训练
    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            # 每次重置梯度
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')
    # 观察训练参数与预先设置的真实参数之间的误差
    w = net[0].weight.data
    print('w的估计误差:', true_w - w.reshape(true_w.shape))
    b = net[0].bias.data
    print('b的估计误差:', true_b - b)


if __name__ == '__main__':
    three_three()
