"""
随机梯度下降(stochastic gradient descent,简称 SGD)
求解步骤:
梯度为0的向量就是优化问题的解,沿着梯度的反方向进行线性搜索,每次搜索迭代的步长为特定的数值α,直到梯度与 0 向量非常接近为止。
(1) 设置初始点,设置步长(也叫做学习率)α,迭代终止的误差忍耐度 tol
(2) 计算目标函数 f 在 xi 上的梯度 ∇f(x^i)
(3) 计算 x^(i+1),公式如下
    x^(i+1) = x^i - α∇f(x^i)
(4) 计算梯度 ∇f(x^(i+1))。
    如果梯度的二范数∥∇f(x^(i+1))∥^2≤tol,则迭代停止,最优解的取值为x^(i+1);
    如果梯度的二范数大于tol,那么i=i+1,并返回第（3）步。
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

"""
全批量梯度下降（full-batch gradient descent）
数据集来自 sofasofa.io, 目标时通过自变量 id 预测 questions 和 answers 数量。
以预测问题数(questions)为例:
1. 建立回归模型
y = β0+β1x,其中 y 表示问题数;x 表示变量 id;β=(β0,β1)为回归系数。
2. 定义损失函数
使用均方误差损失函数 L(β) = (1/n)*∑(Y−Ŷ)^2
3. 按照求解步骤求解
"""


def full_batch_gradient_descent():
    """
    全批量梯度下降
    """
    # 载入数据集
    train = pd.read_csv('../data/sofa/train.csv')

    # 设置初始值
    beta = [1, 1]
    alpha = 0.2  # 学习率
    tol_L = 0.1  # 更切合实际的作法是设置对于损失函数的变动的阈值tol

    # 对 x 进行归一化
    max_x = max(train['id'])
    x = train['id'] / max_x  # 现在 sum(xi)=1
    y = train['questions']  # 训练集的 questions 字段

    # 进行第一次计算
    grad = compute_grad(beta, x, y)
    loss = rmse(beta, x, y)
    beta = update_beta(beta, alpha, grad)
    loss_new = rmse(beta, x, y)

    # 开始迭代
    i = 1
    while np.abs(loss_new - loss) > tol_L:
        beta = update_beta(beta, alpha, grad)
        grad = compute_grad(beta, x, y)
        loss = loss_new
        loss_new = rmse(beta, x, y)
        i += 1
        print('Round %s Diff RMSE %s' % (i, abs(loss_new - loss)))
    print('Coef: %s \nIntercept %s' % (beta[1], beta[0]))
    print('Our Coef: %s \nOur Intercept %s' % (beta[1] / max_x, beta[0]))

    # 训练误差
    res = rmse(beta, x, y)
    print('Our RMSE: %s' % res)

    # 使用标准模块 sklearn.linear_model.LinearRegression 进行检验
    sklearn_check(train, y)


def sklearn_check(train, y):
    lr = LinearRegression()
    lr.fit(train[['id']], train[['questions']])
    print('Sklearn Coef: %s' % lr.coef_[0][0])
    print('Sklearn Coef: %s' % lr.intercept_[0])
    res = rmse([936.051219649, 2.19487084], train['id'], y)
    print('Sklearn RMSE: %s' % res)


def compute_grad(beta, x, y):
    """
    计算均方误差损失函数L(β)在点(β0,β1)上的梯度∇L(βi)
    """
    grad = [0, 0]
    grad[0] = 2. * np.mean(beta[0] + beta[1] * x - y)
    grad[1] = 2. * np.mean(x * (beta[0] + beta[1] * x - y))
    return np.array(grad)


def update_beta(beta, alpha, grad):
    """
    计算 β^(i+1)
    """
    new_beta = np.array(beta) - alpha * grad
    return new_beta


def rmse(beta, x, y):
    """
    计算 RMSE 的函数
    """
    squared_err = (beta[0] + beta[1] * x - y) ** 2
    res = np.sqrt(np.mean(squared_err))
    return res


"""
随机梯度下降法（Stochastic Gradient Decent, SGD）
SGD是对全批量梯度下降法计算效率的改进算法。
本质上来说，我们预期随机梯度下降法得到的结果和全批量梯度下降法相接近；SGD的优势是更快地计算梯度。
但是SGD尽管加快了每次迭代的计算速度，但是也带了收敛不稳定的缺陷
"""


def stochastic_gradient_decent():
    # 导入数据
    train = pd.read_csv('../data/sofa/train.csv')

    # 初始设置
    beta = [1, 1]
    alpha = 0.2
    tol_L = 0.1

    # 对x进行归一化
    max_x = max(train['id'])
    x = train['id'] / max_x
    y = train['questions']

    # 进行第一次计算
    np.random.seed(10)
    grad = compute_grad_SGD(beta, x, y)
    loss = rmse(beta, x, y)
    beta = update_beta(beta, alpha, grad)
    loss_new = rmse(beta, x, y)

    # 开始迭代
    i = 1
    while np.abs(loss_new - loss) > tol_L:
        beta = update_beta(beta, alpha, grad)
        grad = compute_grad_SGD(beta, x, y)
        if i % 100 == 0:
            loss = loss_new
            loss_new = rmse(beta, x, y)
            print('Round %s Diff RMSE %s' % (i, abs(loss_new - loss)))
        i += 1

    print('Coef: %s \nIntercept %s' % (beta[1], beta[0]))
    print('Our Coef: %s \nOur Intercept %s' % (beta[1] / max_x, beta[0]))
    res = rmse(beta, x, y)
    print('Our RMSE: %s' % res)


def compute_grad_SGD(beta, x, y):
    grad = [0, 0]
    # 随机选择一个样本
    r = np.random.randint(0, len(x))
    grad[0] = 2. * np.mean(beta[0] + beta[1] * x[r] - y[r])
    grad[1] = 2. * np.mean(x[r] * (beta[0] + beta[1] * x[r] - y[r]))
    return np.array(grad)


"""
小批量随机梯度下降法（Mini-batch Stochastic Gradient Decent）
小批量随机梯度下降法是对速度和稳定性进行妥协后的产物。
全量梯度下降使用全部样本,随机梯度下降使用单个样本,小批量随机梯度下降使用 batch_size 数量的样本
"""


def mini_batch_stochastic_gradient_decent():
    # 导入数据
    train = pd.read_csv('../data/sofa/train.csv')
    # 初始设置
    beta = [1, 1]
    alpha = 0.2
    tol_L = 0.1
    batch_size = 16

    # 对x进行归一化
    max_x = max(train['id'])
    x = train['id'] / max_x
    y = train['questions']

    # 进行第一次计算
    np.random.seed(10)
    grad = compute_grad_batch(beta, batch_size, x, y)
    loss = rmse(beta, x, y)
    beta = update_beta(beta, alpha, grad)
    loss_new = rmse(beta, x, y)

    # 开始迭代
    i = 1
    while np.abs(loss_new - loss) > tol_L:
        beta = update_beta(beta, alpha, grad)
        grad = compute_grad_batch(beta, batch_size, x, y)
        if i % 100 == 0:
            loss = loss_new
            loss_new = rmse(beta, x, y)
            print('Round %s Diff RMSE %s' % (i, abs(loss_new - loss)))
        i += 1
    print('Coef: %s \nIntercept %s' % (beta[1], beta[0]))
    print('Our Coef: %s \nOur Intercept %s' % (beta[1] / max_x, beta[0]))
    res = rmse(beta, x, y)
    print('Our RMSE: %s' % res)


def compute_grad_batch(beta, batch_size, x, y):
    grad = [0, 0]
    # 每次选择 batch_size 数量的样本
    r = np.random.choice(range(len(x)), batch_size, replace=False)
    grad[0] = 2. * np.mean(beta[0] + beta[1] * x[r] - y[r])
    grad[1] = 2. * np.mean(x[r] * (beta[0] + beta[1] * x[r] - y[r]))
    return np.array(grad)


if __name__ == '__main__':
    # full_batch_gradient_descent()
    # stochastic_gradient_decent()
    mini_batch_stochastic_gradient_decent()
