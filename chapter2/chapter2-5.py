import torch

"""
2-5 自动微分
"""


def two_five_one():
    """
    2-5-1 一个简单的例子
    """
    print("\n======== 2.5.1 ==========")
    x = torch.arange(4.0)
    print(f"x={x};x.shape={x.shape}")
    x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)
    print(f"x.grad={x.grad}")  # 默认值是None
    # 等价于函数 f(x)= 2*(x^2),只不过变量 x 是向量
    y = 2 * torch.dot(x, x)
    print(f"y={y}")
    # 通过反向传播自动计算 x 的梯度
    y.backward()
    print(f"x.grad={x.grad}")
    # 验证梯度是否正确
    # 对 f(x) 求导可得 f(x)'=4x
    print(f"x.grad==4x?{x.grad == 4 * x}")
    # 默认情况下,PyTorch会累积梯度,因此需要清除之前的值
    x.grad.zero_()
    print(f"!!!clear gran done!!!")
    # 等价于 f(x)=x
    x_sum = x.sum()
    print(f"x_sum={x_sum};x_sum.shape={x_sum.shape}")
    # 对 f(x) 求导可得 f(x)'= 1
    x_sum.backward()
    print(f"after x.sum and x.grad={x.grad}")


def two_five_two():
    """
    2-5-2 非标量变量的反向传播
    """
    print("\n======== 2.5.2 ==========")
    x = torch.arange(4.0, requires_grad=True)
    print(f"x={x};x.shape={x.shape}")
    # 本意是想定义函数 f(x)=x,执行后错误: RuntimeError: grad can be implicitly created only for scalar outputs
    # 本意是: 一般而言只有对标量输出才能计算梯度,因为计算一个点的导数才有意义对吧
    # 因此我们需要将 y 变成一个标量,这里可以使用 sum() 函数
    y = x.sum()
    y.backward()
    print(f"x.grad={x.grad}")
    x.grad.zero_()  # 清除梯度
    # 但是为啥是 sum() 函数呢？
    # sum() 等价于 x 点乘 一个相同维度的全为 1 的向量,因为每个元素做的是 "乘以1" 的操作,完成后还是其自身,所以不影响求导
    # 换一种写法
    z = x
    z.backward(torch.ones_like(z))  # grad_tensors 需要与输入 tensor 大小一致
    print(f"x.grad={x.grad}")


def two_five_three():
    """
    2-5-3 分离计算
    """
    print("\n======== 2.5.3 ==========")
    x = torch.arange(4.0, requires_grad=True)
    print(f"x={x};x.shape={x.shape}")
    # 定义函数 f(x)=x^2
    y = x * x
    print(f"y=(x^2)={y};y.shape={y.shape}")
    # 问题: 定义一个新的函数 z = yx, 但是只计算 z 关于 x 的梯度(偏导数),并将 y 当成是一个常数
    # 解决方案:
    # 分离 y 来返回一个新变量 u,该变量与 y 具有相同的值,但丢弃计算图中如何计算y的任何信息,换句话说,梯度不会向后流经 u 到x
    u = y.detach()
    print(f"u={u};u.shape={u.shape}")
    # 现在对 z 求导,则 z' = u(u被看成常数了嘛)
    z = u * x
    z.sum().backward()
    print(f"after y.detach x.grad={x.grad}")
    print(f"x.grad==u?{x.grad == u}")
    # 由于记录了 y 的计算结果,我们可以随后在 y 上调用反向传播, 得到 y=x^2 关于的 x 的导数,即 2x
    x.grad.zero_()
    y.sum().backward()
    print(f"x.grad==2x?{x.grad == 2 * x}")


def two_five_four():
    """
    2-5-4 Python控制流的梯度计算
    """
    print("\n======== 2.5.4 ==========")
    a = torch.randn(size=(), requires_grad=True)
    print(f"a={a};a.shape={a.shape}")
    # func(a) 函数对于任何输入a,存在某个常量标量k,使得f(a)=k*a,其中 k 的值取决于输入 a
    # 定义函数 d = func(a) 等价于 d = ka(k是常数), 则有 k = d /a
    # 对 a 求导数 d' = k, 其结果恰好是 k,因此可以通过 d/a 来验证求导是否正确
    d = func(a)
    print(f"d={d};d.shape={d.shape}")
    d.backward()
    print(f"a.grad == d/a? {a.grad == d / a}")


def func(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


if __name__ == '__main__':
    # two_five_one()
    # two_five_two()
    # two_five_three()
    two_five_four()
