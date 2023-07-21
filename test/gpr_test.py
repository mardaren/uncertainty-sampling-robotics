import numpy as np
import timeit
from data_loader import DataLoader

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic, DotProduct
from sklearn.metrics import mean_absolute_error
from data_loader import DataLoader


def gpr_test():
    data_loader = DataLoader(task_num=1, test_size=0.33)
    time_start = timeit.default_timer()
    kernel = Matern(length_scale=1e-3)
    regressor = GaussianProcessRegressor(kernel=kernel)
    regressor.fit(X=data_loader.x_train, y=data_loader.y_train)

    y_pred = regressor.predict(X=data_loader.x_test)  #, return_std=True)

    test_mae = mean_absolute_error(y_true=data_loader.y_test, y_pred=y_pred)

    time_elapsed = timeit.default_timer() - time_start

    print(f"mae score: {test_mae}")
    print(f"time elapsed: {time_elapsed}")