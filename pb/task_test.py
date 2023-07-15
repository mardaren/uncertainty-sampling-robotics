from pb.task import Task

task = Task()
actions, effects = task.run(N=5)
print(actions)
print(effects)
