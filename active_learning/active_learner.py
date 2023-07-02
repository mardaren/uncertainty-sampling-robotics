import timeit
from sklearn.metrics import mean_absolute_error

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

    def update(self):
        sample = self.data_holder.step()

    def run(self):
        self.init_run()
        while True:
            # if aim is reached, break. Else continue
            y_pred_unknown = self.ask_model(self.data_holder.unknown_x())
            unknown_mae = self.measure(self.data_holder.unknown_y(), y_pred_unknown)
            time_elapsed = timeit.default_timer() - self.time_start
            print("************************")
            print(f"Total Samples: {self.data_holder.known_idx.shape[0]}")
            print(f"Unknown data MAE: {unknown_mae}")
            print(f"Time Elapsed: {time_elapsed}")
            break
            if unknown_mae < 0.4:
                break
            self.update()
            break

    def ask_model(self, data_x):
        return self.model_handler.get_predictions(data_x=data_x)

    def measure(self, y_true, y_pred):
        return mean_absolute_error(y_true=y_true, y_pred=y_pred)