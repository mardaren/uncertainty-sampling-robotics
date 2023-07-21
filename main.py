from data_loader import DataLoader
from pb.task import Task

from active_learning import ActiveLearner, AL_Incremental, AL_Robotic
from test import gpr_test


def run_al(run_type):
    types = ["static", "incremental", "robot"]

    active_learner = None
    if run_type == "robot":
        task = Task()
        active_learner = AL_Robotic(task=task)
    elif run_type == "static":
        data_loader = DataLoader(task_num=1, test_size=0.33)
        active_learner = ActiveLearner(data_loader=data_loader)
    elif run_type == "incremental":
        data_loader = DataLoader(task_num=1, test_size=0.95)
        active_learner = AL_Incremental(data_loader=data_loader)
    active_learner.run()


if __name__ == '__main__':

    # gpr_test()

    run_al("robot")
