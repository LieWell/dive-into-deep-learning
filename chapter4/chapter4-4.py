"""
chapter 4.4 模型选择、欠拟合和过拟合
"""

import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l


def prepare_train_data(n_train=100, n_test=100):
    """
    根据以下多项式构造数据集
    f(x) = 5 + 1.2x - 3.4(x²/2!) + 5.6(x³/3!) + ε, ε∼N(0,0.1²)
    等价于 ===>
    f(x) = 5(x⁰/0!) + 1.2(x¹/1!) - 3.4(x²/2!) + 5.6(x³/3!) + ε, ε∼N(0,0.1²)

    :param n_train: 训练数据集数据量
    :param n_test: 测试数据集数据量
    """
    print(f"\n======== prepare_train_data ==========")
    # 多项式的项数
    # 这里是 20 项,实际有效的是 4 项,多余的项会对训练造成影响
    max_degree = 20
    # 多项式的系数,这里给了4项,再多的可以认为都是0
    true_w = np.zeros(max_degree)
    true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])
    # 随机生成200个特征并随机打乱
    features = np.random.normal(size=(n_train + n_test, 1))
    np.random.shuffle(features)
    print(f"features.shape={features.shape}")  # (200, 1)
    # 生成指数序列并计算每个特征的指数
    poly = np.arange(max_degree).reshape(1, -1)  # (1, 20)
    print(f"poly.shape={poly.shape}")
    poly_features = np.power(features, poly)
    print(f"poly_features.shape={poly_features.shape}")  # (200, 20)
    # 将每个特征除以 (n-1)!
    for i in range(max_degree):
        # poly_features[:, i] 按列取
        # features 是 200 行,1 列,广播机制使其扩展到了 20 列
        # 因此需要针对每一列除以对应的阶乘
        poly_features[:, i] /= math.gamma(i + 1)
    # labels的维度:(n_train+n_test,)
    # 因为只有前四项赋值了系数,因此后面的项都是0,点积之后是个标量(没有 keep dim)
    labels = np.dot(poly_features, true_w)
    # 预测标签增加噪声 ε
    labels += np.random.normal(scale=0.1, size=labels.shape)
    print(f"labels.shape={labels.shape}")  # (200,)
    # NumPy ndarray 转换为 tensor
    true_w, features, poly_features, labels = \
        [torch.tensor(x, dtype=torch.float32) for x in [true_w, features, poly_features, labels]]
    # 打印前两项验证下生成的数据是否正确
    print(f"features[:2]={features[:2]}")
    print(f"poly_features[:2, :]={poly_features[:2, :]}")
    print(f"labels[:2]={labels[:2]}")
    return true_w, features, poly_features, labels


def start_to_train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    """
    :param train_features: 训练数据集
    :param test_features:  测试数据集
    :param train_labels:  训练标签
    :param test_labels: 测试标签
    :param num_epochs:  训练轮数
    """
    # 损失函数使用均方误差
    loss = nn.MSELoss()
    # 获取 shape 的最后一项,可以认为是列数量
    input_shape = train_features.shape[-1]
    print(f"input_shape.last_shape={input_shape}")
    # 使用内置的线性模型
    # 不设置偏置,因为我们已经在多项式中实现了它(就是常量 5)
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    # 设置批次大小并加载数据
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1, 1)), batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1, 1)), batch_size, is_train=False)
    # 使用小批量随机梯度下降算法更新参数
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    # 定义 animator 会凸起
    animator = d2l.Animator(xlabel='epoch',
                            ylabel='loss',
                            yscale='log',
                            xlim=[1, num_epochs],
                            ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    # 分批次训练
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        # 绘制特定的曲线便于观察
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss), d2l.evaluate_loss(net, test_iter, loss)))
    # 观测结果
    d2l.plt.show()
    print('weight:', net[0].weight.data.numpy())


def four_four_four():
    """
    chapter 4.4.4 多项式回归
    """
    print(f"\n======== chapter 4.4.4 ==========")
    # 构造数据
    # 训练数据集数量,测试数据集数量
    n_train, n_test = (100, 100)
    true_w, features, poly_features, labels = prepare_train_data(n_train, n_test)

    # 正常拟合
    # 学习到的参数 weight: [[4.9957824  1.1921009 -3.4157283  5.6153574]] 与预设值差距不大
    # start_to_train(poly_features[:n_train, :4], poly_features[n_train:, :4], labels[:n_train], labels[n_train:])

    # 欠拟合
    # 仅使用两个特征进行预测
    # 学习到的参数  weight: [[2.8127165 4.673477 ]] 与预设值有较大差距
    # start_to_train(poly_features[:n_train, :2], poly_features[n_train:, :2], labels[:n_train], labels[n_train:])

    # 过拟合
    # 从多项式特征中选取所有维度,训练批次也改成了1500
    # 虽然训练损失可以有效地降低,但测试损失仍然较高
    start_to_train(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:],
                   num_epochs=1500)


if __name__ == "__main__":
    four_four_four()
