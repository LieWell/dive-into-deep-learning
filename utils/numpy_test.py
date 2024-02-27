import numpy as np
import torch


def numpy_pow_test():
    print(f"\n========== numpy_pow_test ==========")
    # shape=4x1
    x = np.arange(4).reshape(-1, 1)
    np.random.shuffle(x)
    print(f"x={x};x.shape={x.shape}")
    # shape=1x2
    power = np.arange(2).reshape(1, -1)
    print(f"power={power};power.shape={power.shape}")
    # 广播机制 shape=4x2
    power_x = np.power(x, power)
    print(f"power_x={power_x};power_x.shape={power_x.shape}")
    for i in range(2):
        print(f"power_x[:, {i}]={power_x[:, i]}")
    print(f"power_x.shape={power_x.shape}")
    print(f"power_x.shape[-1]={power_x.shape[-1]}")


if __name__ == '__main__':
    numpy_pow_test()
