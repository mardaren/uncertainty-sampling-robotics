import numpy as np

from sklearn.cluster import KMeans

class DataSelector:

    def initial_query(self, data_x, **kwargs) -> (np.array, np.array):
        pass

    def query(self):
        pass


class VarianceClosenessBased(DataSelector):

    def __init__(self):
        self.dist_table: np.array = None

    def initial_query(self, data_x, n_clusters=10):
        clustering = KMeans(n_clusters=n_clusters).fit(data_x)
        labels = clustering.labels_
        cluster_centers = clustering.cluster_centers_
        known_idx = []
        for c_idx in range(n_clusters):
            c_sample_idx = np.where(labels == c_idx)[0]
            c_samples = data_x[c_sample_idx]
            diff = np.linalg.norm(c_samples - cluster_centers[c_idx], axis=1)
            min_rel_idx = np.argmin(diff)
            min_idx = c_sample_idx[min_rel_idx]
            known_idx.append(min_idx)

        known_idx = np.array(known_idx)
        unknown_idx = np.array(list(set(range(data_x.shape[0])) - set(known_idx)))

        return known_idx, unknown_idx

    def create_table(self, data_x):
        n_samples = data_x.shape[0]
        d_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples - 1):
            d_matrix[i, i + 1:] = np.linalg.norm(data_x[i] - data_x[i + 1:], axis=1)
        self.dist_table = d_matrix

    def update_table(self, idx):
        d_matrix = np.delete(self.dist_table, idx, axis=0)
        d_matrix = np.delete(d_matrix, idx, axis=1)
        self.dist_table = d_matrix


# def max_std_repr(optimizer, X, n_instances=1):
#     alpha = 0.5
#     n_samples = X.shape[0]
#     # Get distance pairs
#     d_matrix = np.zeros((n_samples, n_samples))
#     for i in range(n_samples-1):
#         d_matrix[i, i+1:] = np.linalg.norm(X[i] - X[i+1:], axis=1)
#
#     cl = np.zeros(n_samples)
#     # Get distance for each sample
#     for i in range(n_samples):
#         s1 = np.sum(d_matrix[:i, i])
#         s2 = np.sum(d_matrix[i, i:])
#         cl[i] = 1 / (s1 + s2)
#
#     #Normalize distance
#     cl_max = np.max(cl)
#     cl_min = np.min(cl)
#     cl = (cl - cl_min) / (cl_max - cl_min)
#
#     means, std_values = optimizer.predict(X, return_std=True)
#
#     #normalize std
#     std_max = np.max(std_values)
#     std_min = np.min(std_values)
#     std_values = (std_values - std_min) / (std_max - std_min)
#
#     values = std_values + alpha * cl
#
#     max_idx = np.argpartition(-values, n_instances - 1, axis=0)[:n_instances]
#     return max_idx

# def get_init_data(self):
#     size = 50
#     rng = default_rng()
#
#     known_idx = np.array(rng.choice(len(self.x_train), size=size, replace=False))
#     unknown_idx = np.array(list(set(range(self.y_train.shape[0])) - set(known_idx)))
#
#     return known_idx, unknown_idx
