import timeit

import numpy as np
from numpy.random import default_rng

from sklearn.metrics import mean_absolute_error
from sklearn.cluster import DBSCAN, KMeans

from matplotlib import pyplot as plt


class ActiveLearner:

    def __init__(self, estimator, x_train, y_train, x_test, y_test):
        self.estimator = estimator
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def teach(self, train_x, train_y):
        pass

    def query(self, data_x):
        return None, None

    def get_init_data(self):
        size = 50
        rng = default_rng()

        known_idx = np.array(rng.choice(len(self.x_train), size=size, replace=False))
        unknown_idx = np.array(list(set(range(self.y_train.shape[0])) - set(known_idx)))

        return known_idx, unknown_idx

    def get_clusters(self, n_clusters=10, n_instances=1):

        clustering = KMeans(n_clusters=n_clusters).fit(self.x_train)
        labels = clustering.labels_
        cluster_centers = clustering.cluster_centers_
        known_idx = []
        for c_idx in range(n_clusters):
            c_sample_idx = np.where(labels == c_idx)[0]
            c_samples = self.x_train[c_sample_idx]
            diff = np.linalg.norm(c_samples - cluster_centers[c_idx], axis=1)
            min_rel_idx = np.argmin(diff)
            min_idx = c_sample_idx[min_rel_idx]
            known_idx.append(min_idx)

        known_idx = np.array(known_idx)
        unknown_idx = np.array(list(set(range(self.y_train.shape[0])) - set(known_idx)))

        return known_idx, unknown_idx


    def initialize_model(self, known_idx):
        pass

    def get_accuracy(self, known_idx, unknown_idx, divide: bool = True):  # divide option later
        known_y_pred = self.estimator.predict(self.x_train[known_idx])
        unknown_y_pred = self.estimator.predict(self.x_train[unknown_idx])
        # test_y_pred = self.estimator.predict(self.y_train)

        known_mae = mean_absolute_error(y_true=self.y_train[known_idx], y_pred=known_y_pred)
        unknown_mae = mean_absolute_error(y_true=self.y_train[unknown_idx], y_pred=unknown_y_pred)
        # test_mae = mean_absolute_error(y_true=self.y_test, y_pred=test_y_pred)

        return known_mae, unknown_mae, unknown_y_pred

    def run(self, index_update: bool = True, is_print: bool = True):
        # Define parameters
        time_start = timeit.default_timer()
        print_idx = 0
        print_interval = 10

        # Initialization step
        # known_idx, unknown_idx = self.get_init_data()
        known_idx, unknown_idx = self.get_clusters()

        # you must teach the active_learning known samples
        self.initialize_model(known_idx=known_idx)

        # Loop step
        while True:
            # Query Step
            rel_query_idx = self.query(data_x=self.x_train[unknown_idx])
            query_idx = unknown_idx[rel_query_idx]

            if index_update:
                known_idx =  np.append(known_idx, query_idx)
                unknown_idx = np.delete(unknown_idx, rel_query_idx, axis=0)

            # Teach Step
            self.teach(train_x=self.x_train[query_idx], train_y=self.y_train[query_idx])

            # Testing at some intervals
            if is_print: # and print_idx // print_interval == 0:
                known_mae, unknown_mae, unknown_y_pred = self.get_accuracy(known_idx=known_idx, unknown_idx=unknown_idx)
                time_elapsed = timeit.default_timer() - time_start
                print("************************")
                print(f"Iteration {print_idx + 1}")
                print(f"Total Samples: {known_idx.shape[0]}")
                print(f"Known data MAE: {known_mae}")
                print(f"Unknown data MAE: {unknown_mae}")
                print(f"Time Elapsed: {time_elapsed}")
                if unknown_mae < 0.033:
                    test_y_pred = self.estimator.predict(self.x_test)
                    test_mae = mean_absolute_error(y_true=self.y_test, y_pred=test_y_pred)
                    print(f"Test data MAE: {test_mae}")
                    plt.title('Test Data')
                    plt.plot(self.x_test[:, 0], self.y_test, '.')
                    plt.plot(self.x_test[:, 0], test_y_pred, '.')
                    plt.show()
                    plt.title('Unknown Data')
                    plt.plot(self.x_train[unknown_idx, 0], self.y_train[unknown_idx], '.')
                    plt.plot(self.x_train[unknown_idx, 0], unknown_y_pred, '.')
                    plt.show()
                    break

            print_idx += 1
