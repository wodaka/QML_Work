import numpy as np

class Adam:  # 学习率取为10 #3e-3 速度0.2衰减 #1e-2 速度0.5衰减
    def __init__(self, lr=0.03, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):  # params = {0:a1,1:a2...} grads=[g1,g2,...]
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():  # 生成空的m和v数组
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1

        for key in params.keys():
            self.m[key] = self.beta1 * (self.m[key]) + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            params[key] -= self.lr * (self.m[key] / (1.0 - self.beta1 ** self.iter)) / (
                        np.sqrt(self.v[key] / (1.0 - self.beta2 ** self.iter)) + 1e-8)