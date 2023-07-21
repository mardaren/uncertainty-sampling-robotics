import timeit
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt

from data_loader import DataLoader
from active_learning.data_holder_robotic import DataHolderRobotic
from query import VarianceClosenessBased
from model import GPRHandler
from pb.task import Task

from active_learning import ActiveLearner


class AL_Robotic(ActiveLearner):

    def __init__(self, task: Task):
        self.data_holder = DataHolderRobotic(task=task, data_selector=VarianceClosenessBased())
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
            if len(self.data_holder.unknown_idx) == 0:
                print("Number of unknown samples are depleted")
                self.data_holder.update_unknown_idx(num_samples=10)
            y_pred_unknown = self.ask_model(self.data_holder.unknown_x())
            unknown_mae = self.measure(self.data_holder.unknown_y(), y_pred_unknown[0])
            time_elapsed = timeit.default_timer() - self.time_start
            print("************************")
            print(f"Total Samples: {self.data_holder.known_idx.shape[0]}")
            print(f"Unknown length: {self.data_holder.unknown_idx.shape[0]}")
            print(f"Unknown data MAE: {unknown_mae}")
            print(f"Time Elapsed: {time_elapsed}")
            if unknown_mae < 0.3:
                print("First Phase successful")
                self.data_holder.update_unknown_idx(num_samples=10)
                y_pred_unknown = self.ask_model(self.data_holder.unknown_x())
                unknown_mae = self.measure(self.data_holder.unknown_y(), y_pred_unknown[0])
                if unknown_mae < 0.3:
                    print("Second phase successful")
                    self.data_holder.get_test()
                    y_pred_known = self.ask_model(self.data_holder.known_x())[0]
                    y_pred_test = self.ask_model(self.data_holder.x_test)[0]
                    known_mae = self.measure(self.data_holder.known_y(), y_pred_known)
                    test_mae = self.measure(self.data_holder.y_test, y_pred_test)
                    print(f"Known data MAE: {known_mae}")
                    print(f"Test data MAE: {test_mae}")
                    self.plot_result2(y_pred_test=y_pred_test, y_pred_unknown=y_pred_unknown[0])
                    break
                print("Second Phase failed")
            self.update(std_values=y_pred_unknown[1])
