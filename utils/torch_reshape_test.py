import torch


def reshape_test_one():
    print('\n========== torch.reshape one ==========')
    # 生成 3x4 矩阵
    x = torch.arange(0, 12).reshape(3, 4)
    print(f"x={x}")
    y = torch.reshape(x, (-1, 1))
    print(f"y={y}")


def reshape_test_two():
    print('\n========== torch.reshape two ==========')


if __name__ == '__main__':
    reshape_test_one()
