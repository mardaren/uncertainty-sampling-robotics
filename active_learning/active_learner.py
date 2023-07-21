import timeit
import numpy as np
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt

from data_loader import DataLoader
from active_learning.data_holder import DataHolder
from query import VarianceClosenessBased
from model import GPRHandler


class ActiveLearner:

    def __init__(self, data_loader: DataLoader):
        self.data_holder = DataHolder(data_loader=data_loader, data_selector=VarianceClosenessBased())
        self.model_handler = GPRHandler()

        self.time_start = 0

    def init_run(self):
        self.time_start = timeit.default_timer()
        self.data_holder.initial_step()
        # init model with it
        self.model_handler.teach(sample_x=self.data_holder.known_x(), sample_y=self.data_holder.known_y())

    def update(self, **kwargs):
        query_idx = self.data_holder.step(**kwargs)
        self.model_handler.teach(sample_x=self.data_holder.x_train[query_idx], sample_y=self.data_holder.y_train[query_idx])


    def run(self):
        self.init_run()
        while True:
            # if aim is reached, break. Else continue
            y_pred_unknown = self.ask_model(self.data_holder.unknown_x())
            unknown_mae = self.measure(self.data_holder.unknown_y(), y_pred_unknown[0])
            time_elapsed = timeit.default_timer() - self.time_start
            print("************************")
            print(f"Total Samples: {self.data_holder.known_idx.shape[0]}")
            print(f"Unknown data MAE: {unknown_mae}")
            print(f"Time Elapsed: {time_elapsed}")
            if unknown_mae < 0.3:
                y_pred_known = self.ask_model(self.data_holder.known_x())[0]
                y_pred_test = self.ask_model(self.data_holder.x_test)[0]
                known_mae = self.measure(self.data_holder.known_y(), y_pred_known)
                test_mae = self.measure(self.data_holder.y_test, y_pred_test)
                print(f"Known data MAE: {known_mae}")
                print(f"Test data MAE: {test_mae}")
                self.plot_result_x_y(y_pred_test=y_pred_test, y_pred_unknown=y_pred_unknown[0])
                break
            self.update(std_values=y_pred_unknown[1])

    def ask_model(self, data_x):
        return self.model_handler.get_predictions(data_x=data_x)

    def measure(self, y_true, y_pred):
        return mean_absolute_error(y_true=y_true, y_pred=y_pred)

    def plot_result(self, y_pred_test, y_pred_unknown):
        plt.title('Test Data')
        plt.plot(self.data_holder.x_test[:, 0], self.data_holder.y_test[:, 0], '.')
        plt.plot(self.data_holder.x_test[:, 0], y_pred_test[:, 0], '.')
        plt.show()
        plt.title('Unknown Data')
        plt.plot(self.data_holder.unknown_x()[:, 0], self.data_holder.unknown_y()[:, 0], '.')
        plt.plot(self.data_holder.unknown_x()[:, 0], y_pred_unknown[:, 0], '.')
        plt.show()

    def plot_result2(self, y_pred_test, y_pred_unknown):
        plt.title('Test Data')
        plt.plot(self.data_holder.y_test[:, 0], self.data_holder.y_test[:, 1], '.')
        plt.plot(y_pred_test[:, 0], y_pred_test[:, 1], '.')
        plt.show()
        plt.title('Unknown Data')
        plt.plot(self.data_holder.unknown_y()[:, 0], self.data_holder.unknown_y()[:, 1], '.')
        plt.plot(y_pred_unknown[:, 0], y_pred_unknown[:, 1], '.')
        plt.show()

    def plot_result_x_y(self, y_pred_test, y_pred_unknown):
        plt.title('Test Data')
        y_test_x, y_test_y = self.change_to_x_y(self.data_holder.y_test)
        plt.plot(y_test_x, y_test_y, '.')
        y_pred_test_x, y_pred_test_y = self.change_to_x_y(y_pred_test)
        plt.plot(y_pred_test_x, y_pred_test_y, '.')
        plt.show()
        plt.title('Unknown Data')
        unknown_y_x, unknown_y_y = self.change_to_x_y(self.data_holder.unknown_y())
        plt.plot(unknown_y_x, unknown_y_y, '.')
        y_pred_unknown_x, y_pred_unknown_y = self.change_to_x_y(y_pred_unknown)
        plt.plot(y_pred_unknown_x, y_pred_unknown_y, '.')
        plt.show()

    def change_to_x_y(self, output):
        thetas = output[:, 0]
        distances = output[:, 1]
        x = np.sin(thetas) * distances
        y = np.cos(thetas) * distances
        return x, y
