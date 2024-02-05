import numpy
import torch
from torchvision import transforms

"""
chapter 2.3 线性代数
"""


def two_three_one():
    """
    标量
    """
    print("\n======== 2.3.1 ==========")
    x = torch.tensor(3.0)
    y = torch.tensor(2.0)
    print(x + y)
    print(x - y)
    print(x * y)
    print(x / y)
    print(x ** y)  # x 的 y 次幂


def two_three_two():
    """
    向量
    """
    print("\n======== 2.3.2 ==========")
    x = torch.arange(4)
    print("x=", x)
    print("len(x)=", len(x))  # 使用 len 函数访问向量的长度
    print("x.shape=", x.shape)  # 向量维度是其包含的元素个数(长度)
    y = torch.ones(1, 2, 3)
    print("y=", y)
    print("y.shape=", y.shape)  # 张量的维度是其具有的轴数


def two_three_three():
    """
    矩阵
    """
    print("\n======== 2.3.3 ==========")
    A = torch.arange(20).reshape(5, 4)  # 5x4 矩阵
    print("A=", A)
    print("A.T=", A.T)  # 矩阵的转置 4x5 矩阵
    # 方阵的特殊类型 --- 对称矩阵(symmetric matrix)
    B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
    print("B=", B)
    print("(B == B.T)?", B == B.T)


def two_three_four():
    """
    张量
    """
    print("\n======== 2.3.4 ==========")
    # 图像以 n 维数组形式出现,其中 3 个轴对应于高度、宽度,颜色通道(channel)
    # 高度/宽度是标量, 颜色通道是长度为三的向量(RGB值)
    X = torch.tensor([[[0, 0, 0]]])  # 手动定义了一个点,颜色值为 (0,0,0)
    print("X=", X)
    # 随机生成一张图片
    random_pic()


def random_pic():
    """
    随机生成一张图片
    """
    w = 10  # 图片宽度
    h = 10  # 图片高度
    channels = 3  # 每个通道均使用 RGB 三种颜色表示
    # randn 函数返回符合标准正态分布(均值为0,方差为1,即高斯白噪声)的随机数。
    img = torch.randn(channels, w, h)
    # toPILImage 默认将张量的数值范围从[0, 1]转换为[0, 255]
    # toPILImage 处理 tensor 时,channel 在前,可以理解为图像被分成了 RGB 三层
    to_pil = transforms.ToPILImage()
    pic = to_pil(img)
    pic.save("../data/random2-3-4.jpg")


def two_three_five():
    """
    chapter 2.3.5 张量算法的基本性质
    """
    print("\n======== 2.3.5 ==========")
    a = 2
    x = torch.arange(9).reshape(3, 3)
    print("a=", a)
    print("x=", x)
    print("a+x=", a + x)
    print("a*x=", a * x)
    print("x+x=", x + x)
    print("x*x=", x * x)  # Hadamard 积,符号 A ⊙ B


def two_three_six():
    """
    chapter 2.3.6 降维
    """
    print("\n======== 2.3.6 ==========")
    print("======== 向量运算 ==========")
    x = torch.arange(6, dtype=torch.float).reshape(3, 2)  # 定义一个 3x2 的矩阵
    print("x=", x)
    print("x.sum=", x.sum())
    print("x.cumsum=", x.cumsum(axis=0))  # 保持维度并求累计和
    # 沿轴 0 求和(将矩阵从下到上堆叠,列维度会消失,变成一个轴)
    x_sum_0 = x.sum(axis=0)
    print('x_sum_0={}, shape={}'.format(x_sum_0, x_sum_0.shape))
    # 沿轴 1 求和(将矩阵从右到左堆叠,行维度会消失,变成一个轴)
    x_sum_1 = x.sum(axis=1)
    print('x_sum_1={}, shape={}'.format(x_sum_1, x_sum_1.shape))
    # 沿轴 0  求平均值
    x_mean_0 = x.mean(axis=0)
    print("x_mean_0=", x_mean_0)
    # 沿轴 1  求平均值
    x_mean_1 = x.mean(axis=1)
    print("x_mean_1=", x_mean_1)
    # 保持轴的维度
    # 以sum函数沿轴0操作为例
    # 不保持轴维度结果为 tensor([6., 9.]) // 维度为 1
    # 保持轴维度结果为 tensor([[6., 9.]]) // 维度为 2
    x_sum_dim_0 = x.sum(axis=0, keepdims=True)
    print('x_sum_dim_0={}, shape={}'.format(x_sum_dim_0, x_sum_dim_0.shape))
    x_sum_dim_1 = x.sum(axis=1, keepdims=True)
    print('x_sum_dim_1={},shape={}'.format(x_sum_dim_1, x_sum_dim_1.shape))
    # 由于保持了轴维度，因此支持通过广播方法继续计算
    print("x + x_sum_dim_0=", x + x_sum_dim_0)
    print("x * x_sum_dim_1=", x * x_sum_dim_1)
    print("======== 矩阵运算 ==========")
    # 矩阵的降维运算
    # size=(2,3,3) 可以看做 2 层,每层是 (3,3)矩阵
    y = torch.arange(18, dtype=torch.float).reshape(2, 3, 3)
    print("y=", y)
    print("y.shape=", y.shape)
    print("y.length=", len(y))
    y_sum = y.sum()
    print('y_sum={},shape={}'.format(y_sum, y_sum.shape))
    # 沿 0 轴求和
    # 0轴代表是层,求和就是上下 2 层叠加,不保持维度的话 size=(3,3)
    y_sum_0 = y.sum(axis=0)
    print('y_sum_0={},shape={}'.format(y_sum_0, y_sum_0.shape))
    # 沿 1 轴求和
    # 1 轴代表行, 层不变, (3,3)矩阵求和后变为(,3)的向量, 不保持维度的话, 最终shape=(2,3)
    y_sum_1 = y.sum(axis=1)
    print('y_sum_1={},shape={}'.format(y_sum_1, y_sum_1.shape))
    # 保持1轴的维度,那么 shape 将是 (2,1,3)
    y_sum_dim_1 = y.sum(axis=1, keepdims=True)
    print('y_sum_dim_1={},shape={}'.format(y_sum_dim_1, y_sum_dim_1.shape))


def two_three_seven():
    """
    点积(dot product)
    两个具有相同维度的向量才可以计算点积
    """
    print("\n======== 2.3.7 ==========")
    x = torch.ones(4, dtype=torch.float32)
    print("x=", x)
    y = torch.ones(4, dtype=torch.float32)
    print("y=", y)
    print("<x,y>=", torch.dot(x, y))


def two_three_eight():
    """
    矩阵-向量积(matrix‐vector product)
    矩阵和向量做乘积时,矩阵的列维度(沿轴1的长度)必须等于向量的维度,这样矩阵的每一行(向量)才可以与向量做点积运算
    举例如下计算过程,原始对象:
    A =[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19]]
    x = [0, 1, 2, 3]
    计算步骤:
    C1 = 0*0 + 1*1 + 2*2 + 3*3 = 14      // 第1行计算点积
    C2 = 0*4 + 1*5 + 2*6 + 3*7 = 38      // 第2行计算点积
    C3 = 0*8 + 1*9 + 2*10 + 3*11 = 62    // 第3行计算点积
    C4 = 0*12 + 1*13 + 2*14 + 3*15 = 86  // 第4行计算点积
    C5 = 0*16 + 1*17 + 2*18 + 3*19 = 110 // 第5行计算点积
    结果是一个长度为 5 的向量
    """
    print("\n======== 2.3.8 ==========")
    # 定义矩阵
    A = torch.arange(20).reshape(5, 4)
    print('A={},shape={}'.format(A, A.shape))
    # 定义向量
    x = torch.arange(4)
    print('x={},shape={}'.format(x, x.shape))
    # 计算矩阵-向量积
    # mv 表示 matrix 与 vector 相乘
    r = torch.mv(A, x)
    print("A.mv.x=", r)
    print("(A.mv.x).size=", r.shape)


def two_three_nine():
    """
    矩阵‐矩阵乘法(matrix‐matrix multiplication)
    A是一个 m*n 的矩阵, B是一个 x*y 的矩阵,则 AB 是 m*y 的矩阵
    可以简单的理解为矩阵 A 与 (B 的每个列向量)做乘积,然后简单的拼在一起
    举例计算:
    A=[[1, 2, 3],
       [4, 5, 6]]
    B=[[7, 8],
       [9, 10],
       [11,12]]
    A.shape=2x3, B.shape=3x2 --> (AB).shape=2x2
    A作为矩阵不变,B可以拆分成两个列向量[7,9,11] 和 [8,10,12]
    分别做两次矩阵-向量积运算,得到两个长度为 2 的列向量,然后沿 0 轴拼接成为 2x2 向量
    先计算 A 和 [7,9,11] 的点积：
    C1 = 1*7 + 2*9 + 3*11 = 58
    C2 = 4*7 + 5*9 + 6*11 = 139
    结果为[58,139](注意结果是列向量,写成行向量是为了方便查看)
    再计算 A 和 [8,10,12] 的点积:
    D1 = 1*8 + 2*10 + 3*12 = 64
    D2 = 4*8 + 5*10 + 6*12 = 154
    结果为[64,154](注意结果是列向量,写成行向量是为了方便查看)
    然后将 C 和 D 沿 0 轴拼接,结果为:[[C1,D1],[C2,D2]]
    """
    print("\n======== 2.3.9 ==========")
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    y = torch.tensor([[7, 8], [9, 10], [11, 12]])
    print("x=", x)
    print("y=", y)
    # 计算 矩阵 与 矩阵 乘积
    # mm 表示 matrix 与 matrix 相乘
    r = torch.mm(x, y)
    print("xy=", r)
    print("xy.size=", r.shape)


def two_three_ten():
    """
    范数(norm)
    向量的 Lp 范数: 向量每个元素绝对值的 x 次方和 的 x 次方根,常用的有 L1、L2 范数
    即: ||x||p = (sum(xi)ᵖ)¹/ₚ
    TODO 暂时不理解范数的作用
    """
    print("\n======== 2.3.10 ==========")
    x = torch.tensor([3.0, -4.0])
    print("x=", x)
    # 向量 L2 范数: 向量每个元素平方和的平方根
    # 即: √(3*3 + -4*-4) = 5
    print("x.norm=", torch.norm(x))
    # 向量 L1 范数: 向量每个元素的绝对值之和 <==> 向量每个元素绝对值1次方和的1次方根
    # 即: |3|+|-4|=7
    print("x.abs=", torch.abs(x).sum())
    # 矩阵 Frobenius 范数: 矩阵每个元素的平方和的平方根
    y = torch.ones(2, 2)
    print("y=", y)
    print("y.norm=", torch.norm(y))
    z = torch.arange(6, dtype=torch.float).reshape(3, 2, 1)
    print("z=", z)
    print("z.norm=", torch.norm(z))


if __name__ == '__main__':
    # two_three_one()
    # two_three_two()
    # two_three_three()
    # two_three_four()
    # two_three_five()
    two_three_six()
# two_three_seven()
# two_three_eight()
# two_three_nine()
# two_three_ten()
