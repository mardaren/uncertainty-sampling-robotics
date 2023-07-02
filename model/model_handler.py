import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF


class BaseModelHandler:

    def __init__(self, estimator):
        self.estimator = estimator

    def teach(self, sample_x, sample_y):
        pass

    def get_predictions(self, data_x):
        pass


class GPRHandler(BaseModelHandler):

    def __init__(self):
        kernel = Matern(length_scale=1e-2)
        regressor = GaussianProcessRegressor(kernel=kernel)
        super().__init__(estimator=regressor)
        self.data_x = None
        self.data_y = None

    def teach(self, sample_x, sample_y):  # maybe not needed, after test it ******************
        if self.data_y is None:
            self.data_x = sample_x
            self.data_y = sample_y
        else:
            self.data_x = np.append(self.data_x, sample_x, axis=0)
            self.data_y = np.append(self.data_y, sample_y, axis=0)

        self.estimator.fit(self.data_x, self.data_y)
        print()

    def get_predictions(self, data_x):
        return self.estimator.predict(data_x, return_std=True)
