import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataLoader:

    def __init__(self, task_num: int = 0):
        self.data_path = 'arda-data.npy'
        self.test_percentage = 0.2
        self.random_state = 42
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.scaler = StandardScaler()

        data = self.__get_data()
        data_x, data_y = self.__get_task_data(data=data, task_num=task_num)

        data_y = self.change_y(data_y)

        self.get_train_test(data_x=data_x, data_y=data_y)

    def __get_data(self):
        return np.load(self.data_path, allow_pickle=True)

    def __get_task_data(self, data, task_num: int):
        task_data = data[task_num]
        actions = task_data[:, 1]
        effects = task_data[:, 0]
        return actions, effects

    def get_train_test(self, data_x, data_y):  # random olmasını istiyorsak, degistirebiliriz de
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(data_x, data_y, test_size=0.33,
                                                                                random_state=self.random_state)

    # def scale_data(self, data_x):
    #     for column in data_x:
    #         print()


    def change_y(self, data_y):
        x_diff = data_y[:, 0] - 0.6
        y_diff = data_y[:, 1]
        theta = np.arctan2(x_diff, y_diff)
        dist = x_diff / np.sin(theta)  # dist = np.sqrt(x_diff **2 + y_diff **2)

        data_y = np.append(theta.reshape(-1, 1), dist.reshape(-1, 1), axis=1)
        return data_y
