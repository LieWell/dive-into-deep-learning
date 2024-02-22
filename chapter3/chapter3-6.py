import torch
from d2l import torch as d2l

"""
chapter 3-6 softmax 回归从零开始实现
"""

# 定义全局变量
W, b, lr, epoch = None, None, None, None


def set_W_b_lr_epoch():
    """
    chapter 3.6.1 初始化参数模型
    初始化权重与偏置
    :return:
    """
    num_inputs = 784  # 28 * 28 图像转换为向量
    num_outputs = 10  # 数据集中共有 10 类数据,因此输出也是 10 类
    # 权重是 (784,10) 矩阵,使用正态分布初始化权重,需计算梯度
    weight = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    # 偏置是 (,10) 向量,初始化为 0,需计算梯度
    bias = torch.zeros(num_outputs, requires_grad=True)
    # 学习率
    step = 0.1
    # 训练迭代次数
    train_number = 10
    return weight, bias, step, train_number


def softmax(X, is_print=False):
    """
    chapter 3.6.2 定义 softmax 操作
    1. 对每个项求幂,使用 exp 函数
    2. 对每一行求和(每行都是一个样本), 得到每个样本的规范化常数
    3. 将每一行除以其规范化常数,确保结果的和=1
    """
    if is_print:
        print("\n======== 3.6.2 定义 softmax 操作  ==========")
        # X是满足正态分布的样本,大小为 4x3
        X = torch.normal(0, 1, (4, 3))
        print(f"X={X};X.shape={X.shape}")
    # 对每一项求幂
    x_exp = torch.exp(X)
    if is_print:
        print(f"x_exp={x_exp};X_exp.shape={x_exp.shape}")
    # 保持维度对每一行求和
    partition = x_exp.sum(1, keepdim=True)
    if is_print:
        print(f"partition={partition};partition.shape={partition.shape}")
    # 归一化
    # 这里用到了广播机制, partition 由 (w,1) 扩展为 (w,h)
    # 确保了每一行的每个元素都除以相同的规范化常数
    max_p = x_exp / partition
    if is_print:
        print(f"softmax={max_p}")
    return max_p


def net(X, is_print=False):
    """
    chapter 3.6.3 定义模型
    定义softmax网络模型,网络模型定义了输入如何通过网络映射到输出。
    """
    global W, b
    if is_print:
        print("\n======== 3.6.3 定义模型  ==========")
        X = torch.normal(0, 1, (4, 3))
        print(f"X={X};X.shape={X.shape}")
        W = torch.normal(0, 0.01, (4, 3))
        b = torch.zeros(3)
        print(f"X={X};X.shape={X.shape}")
        print(f"W={W};W.shape={W.shape}")
        print(f"b={b};b.shape={b.shape}")
    # 将数据传递到模型之前，使用 reshape 函数将每张原始图像展平为向量
    # -1 表示自动计算行数,列数由权重行决定
    features = X.reshape((-1, W.shape[0]))
    if is_print:
        print(f"features:{features}")
    # 计算当前层输出
    # matmul 函数解释参考 utils/torch_matmul_test.py
    labels = torch.matmul(features, W) + b
    if is_print:
        print(f"labels:{labels}")
    # 返回 softmax 结果
    ret = softmax(labels)
    if is_print:
        print(f"softmax:{ret}")
    return ret


def cross_entropy(y_hat, y):
    """
    交叉熵损失函数
    TODO 这里不明白是为什么
    """
    return - torch.log(y_hat[range(len(y_hat)), y])


def updater(batch_size):
    global W, b, lr
    return d2l.sgd([W, b], lr, batch_size)


def three_six():
    print("\n======== 3.6 softmax 回归从零开始实现 ==========")
    # 读取数据集
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # 初始化参数
    global W, b, lr, epoch
    W, b, lr, epoch = set_W_b_lr_epoch()

    # 训练模型并绘制图像
    d2l.train_ch3(net, train_iter, test_iter, cross_entropy, epoch, updater)
    d2l.plt.show()

    # 打印训练后的参数
    print(f"W={W};W.shape={W.shape}")
    print(f"b={b};b.shape={b.shape}")

    # 使用测试集进行测试
    d2l.predict_ch3(net, test_iter)
    d2l.plt.show()


if __name__ == '__main__':
    # softmax(None, is_print=True)
    # net(None, is_print=True)
    three_six()
