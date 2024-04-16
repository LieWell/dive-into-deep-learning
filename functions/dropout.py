"""
暂退法(dropout)
"""

"""
Vanilla Dropout: 原论文提出
    1.随机概率p随机dropout部分神经元,并前向传播
    2.计算前向传播的损失,应用反向传播和梯度更新(对剩余的未被dropout的神经元)
    3.恢复所有神经元的,并重复过程1
Vanilla Dropout 训练过程理解起来很简单,但是有个问题,就是测试时会比较麻烦。
为了保持 模型的分布相同,测试时也需要保持模型分布和训练时一样,需要以一个概率p来丢失部分神经元(即乘以1-p的概率来保留),
这样会不太方便,而且同一个输入可能每次预测的结果不一样,不稳定。

Inverted Dropout:
在训练阶段,同样应用p的概率来随机失活,不过额外提前除以1-p,这样相当于将网络的分布提前“拉伸”了,
好处就是在预测阶段,网络无需再乘以1-p(来压缩分布),这样预测时网络无需改动,输出也更加稳定。
"""

"""
暂退法不改变输入的期望
TODO 待补充
"""

import torch


def dropout_sample():
    # size=(3,3) 随机张量
    before_dropout = torch.arange(12).reshape(3, 4)
    print(f"\nbefore dropout={before_dropout}")

    # 丢弃率设为一半
    dropout = 0.5

    # torch.rand(X.shape): 随机生成范围为 [0,1) 的张量
    #  > dropout: 将张量的每个值跟 dropout 进行比较, 转为为 True/False 张量
    # .float(): 将 True/False 张量 转换为 0/1 张量
    mask = (torch.rand(before_dropout.shape) > dropout).float()
    # 提前除以 (1.0 - dropout) 方便预测
    after_dropout = mask * before_dropout / (1.0 - dropout)
    print(f"\nafter dropout={after_dropout}")


if __name__ == "__main__":
    dropout_sample()
