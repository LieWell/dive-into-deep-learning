import torch

"""
torch 自动求解梯度与逃避追踪测试
"""


def requires_grad_test_one():
    """
    一元函数的梯度计算
    """
    print("\n======== requires_grad test one ========")
    x = torch.tensor(2, dtype=torch.float64, requires_grad=True)
    # y = x^2 + 3x + 1
    y = x * x + 3 * x + 1
    # 计算针对特定 x 值的导数
    # y' = 2x+3 ==> y(2)' = 2*2+3 = 7
    y.backward()
    print(f"x.gard={x.grad}")


def requires_grad_test_two():
    """
    二元函数的梯度计算
    :return:
    """
    print("\n======== requires_grad test two ========")
    a = torch.tensor(1.0)
    b = torch.tensor(2.0, requires_grad=True)
    # y = 3a^2 + 9b
    y = 3 * a * a + 9 * b
    # 计算针对特定 b 值的偏导数
    # 当多个参数时,将其他参数看成常数
    # y' = 0(常数求导为0) + 9
    y.backward()
    print(f"x.gard={b.grad}")


def requires_grad_test_three():
    """
    TODO 不是很懂,待补充
    """
    print("\n======== requires_grad test three ========")
    pass


def no_gard_test_one():
    """
    使用 no_grad 避免 autograd(自动计算梯度)
    """
    print("\n======== no_grad test one ========")
    a = torch.ones(2, requires_grad=True)
    b = a * 2
    print(a, a.grad, a.requires_grad)  # tensor([1., 1.], requires_grad=True) None True
    b.sum().backward(retain_graph=True)
    print(a, a.grad, a.requires_grad)  # tensor([1., 1.], requires_grad=True) tensor([2., 2.]) True
    with torch.no_grad():
        a = a + a.grad  # tensor([3., 3.]) None False // 新生成的变量 a 由于 no_grad 导致无法计算梯度
        print(a, a.grad, a.requires_grad)
    b.sum().backward(retain_graph=True)
    print(a, a.grad, a.requires_grad)  # tensor([3., 3.]) None False


def no_gard_test_two():
    """
    使用 no_grad 避免 autograd(自动计算梯度)
    """
    a = torch.ones(2, requires_grad=True)
    b = a * 2
    print(a, a.grad, a.requires_grad)
    b.sum().backward(retain_graph=True)
    print(a, a.grad, a.requires_grad)
    with torch.no_grad():
        a += a.grad  # += 运算符特殊
        print(a, a.grad, a.requires_grad)
        a.grad.zero_()  # 手动梯度清零
    b.sum().backward(retain_graph=True)
    print(a, a.grad, a.requires_grad)


if __name__ == '__main__':
    # requires_grad_test_one()
    # requires_grad_test_two()
    # requires_grad_test_three()
    no_gard_test_one()
    no_gard_test_two()
