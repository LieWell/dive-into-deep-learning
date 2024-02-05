import torch


def list_range():
    print("\n========list and range test ========")
    x = torch.arange(0, 12).reshape(6, 2)
    print(f"x={x}")
    range_value = range(len(x))
    print(f"range_value.type:{type(range_value)}")
    print(f"range_value:{range_value}")
    list_value = list(range_value)
    print(f"list_value.type:{type(list_value)}")
    print(f"list_value:{list_value}")


def torch_sum():
    print("\n========torch.sum test ========")
    X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(f"X.shape={X.shape}")
    # 沿 0 轴求和
    sum_0_dim = X.sum(0, keepdim=True)
    print(f"x.sum_0_dim={sum_0_dim}")
    print(f"x.sum_0_dim.shape={sum_0_dim.shape}")
    sum_0_no_dim = X.sum(0)
    print(f"x.sum_0_no_dim={sum_0_no_dim}")
    print(f"x.sum_0_no_dim.shape={sum_0_no_dim.shape}")
    # 沿 1 轴求和
    sum_1_dim = X.sum(1, keepdim=True)
    print(f"x.sum_1_dim={sum_1_dim}")
    print(f"x.sum_1_dim.shape={sum_1_dim.shape}")
    sum_1_no_dim = X.sum(1)
    print(f"x.sum_1_no_dim={sum_1_no_dim}")
    print(f"x.sum_1_no_dim.shape={sum_1_no_dim.shape}")


def softmax(X):
    # X_exp.shape=(w,h)
    X_exp = torch.exp(X)
    print(f"X_exp={X_exp}")
    print(f"X_exp.shape={X_exp.shape}")
    # 沿 1 轴 sum <==> 每行求和
    # 保持维度结果是 (w,1)
    partition = X_exp.sum(1, keepdim=True)
    print(f"partition={partition}")
    print(f"partition.shape={partition.shape}")
    # X_exp.shape=(w,h)
    # partition.shape(w,1)
    # 利用广播机制做除法
    res = X_exp / partition
    print(f"res={res}")
    print(f"res.shape={res.shape}")
    return res


def softmax_test():
    print("\n========softmax test ========")
    # 定义 3x4 的矩阵
    X = torch.rand(12).reshape(3, 4)
    print(f"X={X}")
    p = softmax(X)
    print(f"softmax(X)={p}")
    print(f"softmax(X).sum={p.sum(1)}")


if __name__ == '__main__':
    # list_range()
    # torch_sum()
    softmax_test()
