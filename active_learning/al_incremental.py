import timeit
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt

from data_loader import DataLoader
from active_learning.data_holder import DataHolder
from query import VarianceClosenessBased
from model import GPRHandler
from active_learning import ActiveLearner


class AL_Incremental(ActiveLearner):

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
            if unknown_mae < 0.25:
                print("First Phase successful")
                self.data_holder.update_unknown_idx(num_samples=50)
                y_pred_unknown = self.ask_model(self.data_holder.unknown_x())
                unknown_mae = self.measure(self.data_holder.unknown_y(), y_pred_unknown[0])
                if unknown_mae < 0.25:
                    y_pred_known = self.ask_model(self.data_holder.known_x())[0]
                    y_pred_test = self.ask_model(self.data_holder.x_test)[0]
                    known_mae = self.measure(self.data_holder.known_y(), y_pred_known)
                    test_mae = self.measure(self.data_holder.y_test, y_pred_test)
                    print(f"Known data MAE: {known_mae}")
                    print(f"Test data MAE: {test_mae}")
                    self.plot_result_x_y(y_pred_test=y_pred_test, y_pred_unknown=y_pred_unknown[0])
                    break
                print("Second Phase failed")
            self.update(std_values=y_pred_unknown[1])

