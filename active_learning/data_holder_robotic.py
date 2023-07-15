import numpy as np
from numpy.random import default_rng

from data_loader import DataLoader
from query import DataSelector

from pb.task import Task


class DataHolderRobotic:

    def __init__(self, task: Task, data_selector: DataSelector):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        self.known_idx = None
        self.unknown_idx = None

        self.task = task
        self.data_selector = data_selector

    def initial_step(self):
        self.x_train, self.y_train = self.task.run(N=45)
        self.known_idx, self.unknown_idx = self.data_selector.initial_query(self.x_train)

    def step(self, **kwargs):
        rel_query_idx = self.data_selector.query(**kwargs)
        query_idx = self.unknown_idx[rel_query_idx]

        self.known_idx = np.append(self.known_idx, query_idx, axis=0)
        self.unknown_idx = np.delete(self.unknown_idx, rel_query_idx, axis=0)

        return query_idx

    def update_unknown_idx(self, num_samples):
        self.unknown_idx = np.append(self.unknown_idx, self.x_train.shape[0]+np.array(range(num_samples)))

        rng = default_rng()

        x_new, y_new = self.task.run(N=num_samples)
        self.x_train = np.append(self.x_train, x_new, axis=0)
        self.y_train = np.append(self.y_train, y_new, axis=0)

        # recalculate cl values
        self.data_selector.create_table(self.x_train[self.unknown_idx])

    def get_test(self):
        self.x_test, self.y_test = self.task.run(N=300)

    def known_x(self):
        return self.x_train[self.known_idx]

    def known_y(self):
        return self.y_train[self.known_idx]

    def unknown_x(self):
        return self.x_train[self.unknown_idx]

    def unknown_y(self):
        return self.y_train[self.unknown_idx]
