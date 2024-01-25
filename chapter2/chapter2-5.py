import torch

"""
2-5 自动微分
TODO 本节完全不懂～～～
"""


def two_five_one():
    """
    2-5-1 一个简单的例子
    """
    print("\n======== 2.5.1 ==========")
    x = torch.arange(4.0)
    print("x=", x)
    x.requires_grad_(True)  # TODO 不理解其作用
    print("x.grad=", x.grad)


if __name__ == '__main__':
    two_five_one()
