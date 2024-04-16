import torchvision
from torch.utils import data
from torchvision import transforms

from d2l import torch as d2l

"""
chapter 3-5 图像分类数据集
"""


def three_five_one():
    """
    chapter 3.5.1 读取 Fashion‐MNIST 数据集
    测试情况可参考文件 utils/d2l_test.py/d2l_test_load_data_fashion_mnist
    """
    print("\n======== 3.5.1 ==========")
    # 通过 ToTensor 实例将图像数据从 PIL 类型变换成32位浮点数格式
    # 并除以255使得所有像素的数值均在 0~1 之间
    trans = transforms.ToTensor()
    # 训练数据集,train=True 返回训练数据集, download=True 自动下载数据集
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=False)
    # 测试数据集,train=False 返回测试数据集
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=False)
    print(f"mnist_train.len={len(mnist_train)}")
    print(f"mnist_test.len={len(mnist_test)}")
    # 可视化部分样本
    d2l.use_svg_display()
    # 按照 batch_size=18 返回数据,训练集公有 60000 数据,因此 datas 长度为 60000/18≈3334
    datas = data.DataLoader(mnist_train, batch_size=18)
    print(f"datas={datas};datas.len={len(datas)}")
    # 每次迭代按照 batch_size 返回数据
    # 分别是特征与对应的标签
    X, y = next(iter(datas))
    print(f"X.len={len(X)};X.shape={X.shape};y.len={len(y)};y.shape={y.shape}")
    titles = d2l.get_fashion_mnist_labels(y)
    print(f"y.label={titles}")
    # TODO X 原始 shape=[18, 1, 28, 28], 这里相当于除去了 通道维度 ????
    d2l.show_images(X.reshape(18, 28, 28), 2, 9, titles)
    d2l.plt.show()


def three_five_two():
    """
    chapter 3.5.2 读取小批量样本
    通过内置迭代器并随机访问增加性能
    """
    print("\n======== 3.5.2 ==========")
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=False)
    # 使用4个线程,按照 batch_size=256 读取全部数据集
    batch_size = 256
    number_workers = d2l.get_dataloader_workers()
    # shuffle=Ture表示在每一次 epoch 中都打乱所有数据的顺序，然后以 batch 为单位从头到尾按顺序取用数据。
    # 这样的结果就是不同 epoch 中的数据都是乱序的。
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=number_workers)
    # 共需读取 235 次(60000/256≈235)
    print(f"train_iter.len={len(train_iter)}")
    timer = d2l.Timer()
    for X, y in train_iter:
        continue
    print(f'{timer.stop():.2f} sec')


def three_five_three():
    """
    chapter 3.5.3 整合所有组件
    """
    print("\n======== 3.5.3 ==========")
    train_iter, test_iter = d2l.load_data_fashion_mnist(32, resize=64)
    for X, y in train_iter:
        print(X.shape, X.dtype, y.shape, y.dtype)
        # 打印 1 次后就 break 掉
        break


if __name__ == '__main__':
    # three_five_one()
    # three_five_two()
    three_five_three()
