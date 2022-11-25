import numpy as np

def _sigmoid(x):
    if x >= 0:  # 避免数据溢出
        return 1. / (1. + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))


def _sigmoid_grad(x):
    return _sigmoid(x) * (1 - _sigmoid(x))


def sigmoid(Z):
    # 解决溢出问题
    # 把大于0和小于0的元素分别处理
    # 原来的sigmoid函数是 1/(1+np.exp(-Z))
    # 当Z是比较小的负数时会出现上溢，此时可以通过计算exp(Z) / (1+exp(Z)) 来解决

    mask = (Z > 0)
    positive_out = np.zeros_like(Z, dtype='float64')
    negative_out = np.zeros_like(Z, dtype='float64')

    # 大于0的情况
    positive_out = 1 / (1 + np.exp(-Z, positive_out, where=mask))
    # 清除对小于等于0元素的影响
    positive_out[~mask] = 0

    # 小于等于0的情况
    expZ = np.exp(Z, negative_out, where=~mask)
    negative_out = expZ / (1 + expZ)
    # 清除对大于0元素的影响
    negative_out[mask] = 0

    return positive_out + negative_out


def sigmoid_grad(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))