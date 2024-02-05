import torch
from d2l import torch as d2l

"""
chapter 3-6 softmax 回归从零开始
!!!本章节内容无法正常运行!!!
"""


def softmax(X):
    """
    定义 softmax 操作
    1.对每个项求幂
    2. 对每一行求和(沿 1 轴 sum), 得到每个样本的规范化常数
    3. 将每一行除以其规范化常数,确保结果的和=1
    """
    # X_exp.shape=(w,h)
    X_exp = torch.exp(X)
    print(f"X_exp.shape={X_exp.shape}")
    # 沿 1 轴 sum <==> 每行求和
    # 保持维度结果是 (w,1)
    partition = X_exp.sum(1, keepdim=True)
    print(f"partition.shape={partition.shape}")
    # X_exp.shape=(w,h)
    # partition.shape(w,1)
    # 利用广播机制相除, (w,1) -> (w,h),由于列是扩展出来的,因此矩阵的每一行都相同
    return X_exp / partition


def net(X):
    """
    定义softmax网络模型,网络模型定义了输入如何通过网络映射到输出。
    """
    # 将数据传递到模型之前，我们使用 reshape 函数将每张原始图像展平为向量
    # 每个元素都可以认为是一种特征
    features = X.reshape((-1, W.shape[0]))
    # 计算当前层输出
    labels = torch.matmul(features, W) + b
    # 返回 softmax 结果 --- 归一化？？？
    return softmax(labels)


def cross_entropy(y_hat, y):
    """
    交叉熵损失函数
    """
    return - torch.log(y_hat[range(len(y_hat)), y])


def updater(W, b, lr, batch_size):
    return d2l.sgd([W, b], lr, batch_size)


print("\n======== 3.6 ==========")
# 读取数据集
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# chapter 3.6.1 初始化参数模型
num_inputs = 784  # 28 * 28 图像转换为向量
num_outputs = 10  # 数据集中共有 10 类数据,因此输出也是 10 类
# 权重是 (784,10) 矩阵
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
# 偏置是 (,10) 向量
b = torch.zeros(num_outputs, requires_grad=True)
lr = 0.1
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

if __name__ == '__main__':
    pass
