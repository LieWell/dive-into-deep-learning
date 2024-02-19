import torch
from torch.distributions import multinomial
from d2l import torch as d2l

"""
2-6 概率
"""


def two_six_one():
    """
    2-6-1 基本概率论
    """
    print("\n======== 2.6.1 ==========")
    # 生成一个概率向量,其中概率都为 1/6,模拟扔骰子的行为
    fair_probs = torch.ones([6]) / 6
    print(f'fair_probs:{fair_probs}')
    # 采样 500 次(进行 500 组实验)
    times = torch.Size([500])
    # 每组实验扔骰子 10 次
    total_count = 10
    # 得到一个 500*6 的矩阵,行表示进行的500组实验,列表示在每组实验中投掷10次的结果
    counts = multinomial.Multinomial(total_count, fair_probs).sample(times, )
    print(f'counts:{counts}\ncounts_shape:{counts.shape}')
    # 按照行(dim=0)进行累加,每一行表示随着实验次数的增加,每个数字出现的次数
    cum_counts = counts.cumsum(dim=0)
    print(f'cum_counts:{cum_counts}\ncum_counts_shape:{cum_counts.shape}')
    # 将累加后的矩阵按照列(dim=1)求和,并且保持维度
    # 因为每组实验都是投掷10次,因此最终得到 500*1 的矩阵,每行表示随着实验次数的增加,总的实验次数
    time_counts = cum_counts.sum(dim=1, keepdims=True)
    print(f'time_counts:{time_counts}\ntime_counts_shape:{time_counts.shape}')
    # 使用每个数字出现次数除以总次数得到概率
    estimates = cum_counts / time_counts
    print(f'estimates:{estimates}')
    # 绘图
    d2l.set_figsize((6, 4.5))
    # estimates 表示 6 个数字的概率,均放到图中
    for i in range(6):
        d2l.plt.plot(estimates[:, i].numpy(),
                     label=("P(die=" + str(i + 1) + ")"))
    d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')  # 理论概率 1/6,使用黑色虚线标识
    d2l.plt.gca().set_xlabel('Groups of experiments')  # 横轴表示实验次数
    d2l.plt.gca().set_ylabel('Estimated probability')  # 纵轴表示出现的概率
    d2l.plt.legend()
    d2l.plt.show()


if __name__ == '__main__':
    two_six_one()
