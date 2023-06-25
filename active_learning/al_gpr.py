import numpy as np
from math import inf
from modAL.models import BayesianOptimizer
from modAL.acquisition import max_EI
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF

from active_learning import ActiveLearner

def max_std(optimizer, X, n_instances=1):
    std_values = optimizer.predict(X, return_std=True)[1]
    # query_idx = np.argmax(std_values)

    max_idx = np.argpartition(-std_values, n_instances - 1, axis=0)[:n_instances]
    return max_idx

    # return query_idx, X[query_idx]


def max_std_repr(optimizer, X, n_instances=1):
    alpha = 0.5
    n_samples = X.shape[0]
    # Get distance pairs
    d_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples-1):
        d_matrix[i, i+1:] = np.linalg.norm(X[i] - X[i+1:], axis=1)

    cl = np.zeros(n_samples)
    # Get distance for each sample
    for i in range(n_samples):
        s1 = np.sum(d_matrix[:i, i])
        s2 = np.sum(d_matrix[i, i:])
        cl[i] = 1 / (s1 + s2)

    #Normalize distance
    cl_max = np.max(cl)
    cl_min = np.min(cl)
    cl = (cl - cl_min) / (cl_max - cl_min)

    means, std_values = optimizer.predict(X, return_std=True)

    #normalize std
    std_max = np.max(std_values)
    std_min = np.min(std_values)
    std_values = (std_values - std_min) / (std_max - std_min)

    values = std_values + alpha * cl

    max_idx = np.argpartition(-values, n_instances - 1, axis=0)[:n_instances]
    return max_idx



class AL_GPR(ActiveLearner):

    def __init__(self, x_train, y_train, x_test, y_test):
        # kernel = RBF(length_scale=1e-2)
        kernel = Matern(length_scale=1e-2)
        regressor = GaussianProcessRegressor(kernel=kernel)
        super().__init__(estimator=regressor, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
        self.optimizer = BayesianOptimizer(estimator=regressor, query_strategy=max_std_repr)

    def teach(self, train_x, train_y):
        self.optimizer.teach(train_x, train_y)

    def query(self, data_x):
        query_idx, _ = self.optimizer.query(data_x)
        return query_idx

    def initialize_model(self, known_idx):
        self.optimizer.teach(self.x_train[known_idx], self.y_train[known_idx])

