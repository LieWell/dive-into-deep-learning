import torch
import random
from d2l import torch as d2l

"""
chapter 3-2 线性回归的从零开始实现
"""


def three_two_one():
    """
    chapter 3.2.1 生成数据集
    """
    print("\n======== 3.2.1 ==========")
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    # synthetic_data 函数已被封装,后续不再说明
    features, labels = d2l.synthetic_data(true_w, true_b, 1000)
    # features 是标准正态分布矩阵, shape=(1000,1), 可以理解为权重(面积/年龄)
    # labels 是一个列向量,shape=(1000,1),可以理解成每行特征对应的房价
    # --- 当然了,labels 值是根据 feature + bias + noise 计算得来的,这样画图更清晰
    print('features:', features, '\nlabels:', labels)
    print('features.shape:', features.shape, '\nlabels.shape:', labels.shape)
    d2l.set_figsize()
    d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
    # pycharm 需手动调用 show 函数,后续不再说明
    d2l.plt.show()


def data_iter(batch_size, features, labels):
    """
    根据输入的特征矩阵和标签向量随机抽取 batch_size 大小的样本
    :param batch_size: 批量大小
    :param features: 特征矩阵
    :param labels: 标签向量
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    # shuffle 函数将列表元素打乱
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


def three_two_two():
    """
    chapter 3.2.2 读取数据集
    每次抽取一小批样本来计算并更新模型
    """
    print("\n======== 3.2.2 ==========")
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    # synthetic_data 函数已被封装,后续不再说明
    features, labels = d2l.synthetic_data(true_w, true_b, 1000)
    # 如下迭代对教学来说很好,但它的执行效率很低,可能会在实际问题上陷入麻烦。
    # 例如,它要求我们将所有数据加载到内存中,并执行大量的随机内存访问。
    # 在深度学习框架中实现的内置迭代器效率要高得多,它可以处理存储在文件中的数据和数据流提供的数据。
    batch_size = 10
    for X, y in data_iter(batch_size, features, labels):
        print(X, '\n', y)
        break


def three_two_three():
    """
    chapter 3.2.3 初始化模型参数
    """
    print("\n======== 3.2.3 ==========")
    # 生成数据时已经知道真正的参数: 特征 [2, -3.4],权重: 4.2
    # 那如何获取初始参数呢,很简单 --- 随机一个
    # 从正态分布里面随便挑一组作为特征初始值
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    print(f"w={w}")
    # 偏置就直接设为 0
    b = torch.zeros(1, requires_grad=True)
    print(f"b={b}")


def three_two_four():
    """
    chapter 3.2.4 定义模型
    使用 d2l 包定义好的 linreg 函数
    核心就是 X * w + b
    """
    print("\n======== 3.2.4 ==========")
    pass


def three_two_five():
    """
    chapter 3.2.5 定义损失函数
    使用 d2l 包封装好的均方损失函数 squared_loss
    TODO 如何选择损失函数
    """
    print("\n======== 3.2.5 ==========")
    pass


def three_two_six():
    """
    chapter 3.2.6 定义优化算法
    使用小批量随机梯度下降,在每一步中,使用从数据集中随机抽取的一个小批量,然后根据参数计算损失(函数)的梯度。
    接下来,朝着减少损失的方向更新我们的参数
    """
    print("\n======== 3.2.6 ==========")
    pass


def three_two_seven():
    """
    chapter 3.2.7 训练
    """
    print("\n======== 3.2.7 ==========")
    # 模拟的数据
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = d2l.synthetic_data(true_w, true_b, 1000)

    lr = 0.03  # 学习率 --- 超参数,其设置后续会讨论
    num_epochs = 3  # 训练次数 --- 超参数,其设置后续会讨论

    net = d2l.linreg  # 网络 --- 线性模型
    loss = d2l.squared_loss  # 损失函数 --- 均方损失函数

    # weight --- 参数,由模型动态调整
    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    # bias --- 参数,由模型动态调整
    b = torch.zeros(1, requires_grad=True)

    # 随机抽样大小
    batch_size = 10
    # 一共训练三轮
    for epoch in range(num_epochs):
        # 随机从模拟的数据中取出一部分
        for X, y in data_iter(batch_size, features, labels):
            # 通过神经网络获取预测值
            y_hat = net(X, w, b)
            # 计算这次小批量中预测值与真实值的损失
            l_value = loss(y_hat, y)
            # 因为 l_value 形状是(batch_size,1)，而不是一个标量。
            # 因此将 l_value 中的所有元素被加到一起，并计算关于[w,b]的梯度
            l_value.sum().backward()
            # 使用参数的梯度更新参数
            d2l.sgd([w, b], lr, batch_size)
        # 完成一轮训练后,观察一下在整个样本下的效果(损失有没有减小)
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
    # 完成全部训练后,观察训练结果和真实值之间的误差
    print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
    print(f'b的估计误差: {true_b - b}')


if __name__ == '__main__':
    # three_two_one()
    # three_two_two()
    # three_two_three()
    three_two_seven()
