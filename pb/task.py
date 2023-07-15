import numpy as np

from pb.envs.objects import Objects
from pb.envs.objects_obstacle import ObjectsObstacle
import math


class Task:
    def __init__(self):
        # self.env = Objects()  # Task 1    duvar yok
        # self.env = ObjectsObstacle(obstacles=1)    # Task 2 hareketli duvar
        self.env = ObjectsObstacle(obstacles=2)    # Task 3    L duvar

        self.available_actions = np.arange(0, 180, 0.2)


    def run_episode(self):
        choice_idx = np.random.randint(low=0, high=self.available_actions.shape[0], size=1)
        action_choice = self.available_actions[choice_idx][0]  # Choose random degree as an action
        self.available_actions = np.delete(self.available_actions, choice_idx)

        self.env.reset()

        object_pos_ori_last = self.env.episode(action_choice)  # positıon at 0. ındex
        effect = object_pos_ori_last[0][:2]

        action = [math.sin(math.radians(action_choice)), math.cos(math.radians(action_choice))]
        return action, effect

    def run(self, N):
        actions = []
        effects = []
        print("Running sample ", end=' ')
        for i in range(N):
            print(f"{i}", end=', ')
            x, y = self.run_episode()
            actions.append(x)
            effects.append(y)
        print()

        effects = np.array(effects)
        x_diff = effects[:, 0] - 0.6
        y_diff = effects[:, 1]
        theta = np.arctan2(x_diff, y_diff)
        dist = x_diff / np.sin(theta)
        effects = np.append(theta.reshape(-1, 1), dist.reshape(-1, 1), axis=1)

        return np.array(actions), effects
