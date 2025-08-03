import numpy as np

class RunningMeanStd:
    def __init__(self, num_features, epsilon=1e-4):
        self.mean = np.zeros(num_features)
        self.var = np.ones(num_features)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        new_var = (self.var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / total_count) / total_count

        # 防止方差变为零
        new_var[new_var < 1e-8] = 1e-8

        self.mean, self.var, self.count = new_mean, new_var, total_count

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var)