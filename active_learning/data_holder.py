import numpy as np

from data_loader import DataLoader
from query import DataSelector


class DataHolder:

    def __init__(self, data_loader: DataLoader, data_selector: DataSelector):
        self.x_train = data_loader.x_train
        self.x_test = data_loader.x_test
        self.y_train = data_loader.y_train
        self.y_test = data_loader.y_test

        self.known_idx = None
        self.unknown_idx = None

        self.data_selector = data_selector

    def initial_step(self):
        self.known_idx, self.unknown_idx = self.data_selector.initial_query(self.x_train)

    def step(self, **kwargs):
        rel_query_idx = self.data_selector.query(**kwargs)
        query_idx = self.unknown_idx[rel_query_idx]

        self.known_idx = np.append(self.known_idx, query_idx, axis=0)
        self.unknown_idx = np.delete(self.unknown_idx, rel_query_idx, axis=0)

        return query_idx

    def known_x(self):
        return self.x_train[self.known_idx]

    def known_y(self):
        return self.y_train[self.known_idx]

    def unknown_x(self):
        return self.x_train[self.unknown_idx]

    def unknown_y(self):
        return self.y_train[self.unknown_idx]
