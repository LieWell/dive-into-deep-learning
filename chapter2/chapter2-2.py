import os
import pandas as pd
import torch


def prepare_dataset_file():
    os.makedirs(os.path.join('..', 'data'), exist_ok=True)
    return os.path.join('..', 'data', 'house_tiny.csv')


def read_write_dataset():
    # 写数据
    data_file = prepare_dataset_file()
    with open(data_file, 'w') as f:
        f.write('NumRooms,Alley,Price\n')
        f.write('NA,Pave,127500\n')
        f.write('2,NA,106000\n')
        f.write('4,NA,178100\n')
        f.write('NA,NA,140000\n')
    # 读数据
    data = pd.read_csv(data_file)
    print('\n')
    print(data)


def deal_dataset():
    data_file = prepare_dataset_file()
    data = pd.read_csv(data_file)
    # 通过位置索引 iloc 将 data 分成 inputs 和 outputs 两部分
    # inputs 为前两列,outputs为最后一列
    inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
    # 对于 inputs 中缺少的数值,使用同一列的均值替换 NaN 项
    # 注意这里增加了条件限制: 仅在改列值是数值的情况下才参与运算,这里是 (2+4)/2=3
    inputs = inputs.fillna(inputs.mean(numeric_only=True))
    print(inputs)
    # get_dummies 将分类变量转换为虚拟/指标变量(谁明白这是啥意思)
    # 大概就是将 Allay 属性值通过列的形式展示,这里 Allay 有 pave 和 NaN 两种值,所以就是两列
    #       NumRooms  Alley_Pave  Alley_nan
    # 0       3.0        True      False
    # 1       2.0       False       True
    # 2       4.0       False       True
    # 3       3.0       False       True
    # 注意我们这里展示的值是 True/False,等同于 1/0
    inputs = pd.get_dummies(inputs, dummy_na=True)
    print(inputs)

    # 转换为张量
    x = torch.tensor(inputs.to_numpy(dtype=float))
    y = torch.tensor(outputs.to_numpy(dtype=float))  # outputs 不需要处理
    print(x)
    print(y)


if __name__ == '__main__':
    deal_dataset()
