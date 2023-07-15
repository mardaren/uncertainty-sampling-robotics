import numpy as np
import timeit
from data_loader import DataLoader
from active_learning import ActiveLearner, AL_Incremental, AL_Robotic

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RationalQuadratic, DotProduct
from sklearn.metrics import mean_absolute_error

from pb.task import Task

if __name__ == '__main__':
    data_loader = DataLoader(task_num=1)
    task = Task()

    # time_start = timeit.default_timer()
    # kernel = Matern(length_scale=1e-3)
    # regressor = GaussianProcessRegressor(kernel=kernel)
    # regressor.fit(X=data_loader.x_train, y=data_loader.y_train)
    #
    # y_pred = regressor.predict(X=data_loader.x_test)  #, return_std=True)
    #
    # test_mae = mean_absolute_error(y_true=data_loader.y_test, y_pred=y_pred)
    #
    # time_elapsed = timeit.default_timer() - time_start
    #
    # print(f"mae score: {test_mae}")
    # print(f"time elapsed: {time_elapsed}")
    # exit(1)

    learner = ActiveLearner(data_loader=data_loader)
    # learner = AL_Incremental(data_loader=data_loader)
    # learner = AL_Robotic(task=task)
    learner.run()


