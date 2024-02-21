"""
python 基础函数测试
"""


def iter_next_test():
    """
    迭代器是一个可以记住遍历的位置的对象。
    迭代器对象从集合的第一个元素开始访问，直到所有的元素被访问完结束。迭代器只能往前不会后退。
    """
    print("\n======== iter_next_test ========")
    a = [1, 2, 3]
    it = iter(a)  # 创建迭代器对象
    print(f'1:{next(it)}')  # 输出迭代器下一项
    print(f'2:{next(it)}')
    print(f'3:{next(it)}')
    # print(f'4:{next(it)}')  # 到达末尾不允许再次访问


def x_for_y_in_z_test():
    """
    X for Y in Z 结构
    """
    print("\n======== x_for_y_in_z_test ========")
    print(f'x:{[x for x in range(1, 11)]}')  # 原样输出,并封装为数组
    print(f'x*2:{[x * 2 for x in range(1, 11)]}')  # 对每个 x 执行 x*2 的操作,并封装为数组


if __name__ == '__main__':
    iter_next_test()
    x_for_y_in_z_test()
