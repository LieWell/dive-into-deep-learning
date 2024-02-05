import torch

"""
torch.matmul() 函数测试
torch.matmul 函数根据传入参数的张量维度有很多重载函数。
传入参数是向量或者矩阵,其结果与 dot, mv, mm 等价;传入参数维度较高时,有独特的计算逻辑
"""


def print_info(A, B):
    """打印函数信息便于观察结果"""
    print(f"A: {A}\nB: {B}")
    print(f"A 的维度: {A.dim()},\t B 的维度: {B.dim()}")
    print(f"A 的元素总数: {A.numel()},\t B 的元素总数: {B.numel()}")
    print(f"torch.matmul(A, B): {torch.matmul(A, B)}")
    print(f"torch.matmul(A, B).size(): {torch.matmul(A, B).size()}")


def one_one_dim():
    """输入为(向量,向量),等价与 torch.dot"""
    print('\n========== torch.matmul(vector,vector) ==========')
    A = torch.randint(0, 5, size=(2,))
    B = torch.randint(0, 5, size=(2,))
    print(f"A.dot.B={torch.dot(A, B)}")
    print_info(A, B)


def two_two_dim():
    """输入为(矩阵,矩阵),等价与 torch.mm"""
    print('\n========== torch.matmul(matrix,matrix) ==========')
    A = torch.randint(0, 5, size=(2, 3))
    B = torch.randint(0, 5, size=(3, 4))
    print(f"A.mm.B={torch.mm(A, B)}")
    print_info(A, B)


def one_two_dim():
    """
    输入为(向量,矩阵),按照广播处理。
    即从 size 的尾部开始一一比对,如果维度不够,则扩展一维,令初始值为 1 再进行计算,计算完之后移除扩展的维度。
    用下面的例子来说就是扩展成 (1, 2) 后,(1, 2) * (2, 2) => (1, 2) => (2, )
    """
    print('\n========== torch.matmul(vector,matrix) ==========')
    A = torch.randint(0, 5, size=(2,))
    B = torch.randint(0, 5, size=(2, 2))
    print_info(A, B)


def two_one_dim():
    """
    输入为(矩阵,向量),等价与 torch.mv
    """
    print('\n========== torch.matmul(matrix,vector) ==========')
    A = torch.randint(0, 5, size=(2, 2))
    B = torch.randint(0, 5, size=(2,))
    print(f"A.mv.B={torch.mv(A, B)}")
    print_info(A, B)


def mul_two_dim():
    """
    输入为(多维,矩阵),将 size 的最后两项看成矩阵,前面的看成 batch,做矩阵的 mm 操作后与 batch 拼接
    """
    print('\n========== torch.matmul(mul,matrix) ==========')
    # 首先将 A 拆分成 2 个 3x4 矩阵
    # 将每个矩阵跟 B 做 mm 操作,得到结果 size=(3x6)
    # 拼接上 batch=2,最终结果 size=(2,3,6)
    A = torch.randint(0, 5, size=(2, 3, 4))
    B = torch.randint(0, 5, size=(4, 6))
    print_info(A, B)


def mul_one_dim():
    """
    输入为(多维,向量),将 size 最后两项看成矩阵,前面的看成 batch，做 矩阵与向量的 mv 操作后与 batch 拼接
    """
    print('\n========== torch.matmul(mul,matrix) ==========')
    # 首先将 A 拆分成 2 个 3x4 矩阵
    # 将每个矩阵跟 B 做 mv 操作,得到结果 size= (3,)
    # 拼接上 batch=2,最终结果 size=(2,3)
    A = torch.randint(0, 5, size=(2, 3, 4))
    B = torch.randint(0, 5, size=(4,))
    print_info(A, B)


def mul_mul_dim():
    """
    输入为(多维,多维),将 size 最后两项看成矩阵，前面的看成 batch，做 矩阵与矩阵的 mm 操作后与 batch 拼接
    显然 batch 部分可能不同,此时按照广播机制处理的
    举例:
    A.size() = (j, 1, m, n)
    B.size() =(k, n, p)
    A 和 B 的最后两项做 mm 操作,其结果 size=(m,p)
    A.batch=(j,1) 与 B.batch=(k,) 维度不同,广播扩展结果 size=(j,k)
    最终 torch.matmul(A, B).size() = (j, k, m, p)。
    """
    print('\n========== torch.matmul(mul,matrix) ==========')
    # 结果 size=(2,4,3,6)
    A = torch.randint(0, 5, size=(2, 4, 3, 4))
    B = torch.randint(0, 5, size=(4, 4, 6))
    print_info(A, B)


if __name__ == '__main__':
    # 固定 torch 的随机数种子,以便重现结果
    torch.manual_seed(0)
    # one_one_dim()
    # two_two_dim()
    # one_two_dim()
    # two_one_dim()
    # mul_two_dim()
    # mul_one_dim()
    mul_mul_dim()
