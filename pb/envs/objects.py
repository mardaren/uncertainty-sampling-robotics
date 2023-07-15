import pybullet_data

import numpy as np

import os
import random
import time
import math
from pb import utils


class Objects:
    def __init__(self):

        self.tableUid = None
        self.robot = None
        self.duck = None
        self.p = utils.connect(gui=0)
        self.hz = 240.0
        self.p.setTimeStep(1.0 / self.hz)
        self.p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40,
                                          cameraTargetPosition=[0.55, -0.35, 0.2])

        self.link = 11  # Total 12 links in panda. Last link -> gripper

        self.unit_step_length = 0.01  # 1cm

        self.tableUid = self.p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane_transparent.urdf"),
                                        basePosition=[0, 0, 0])
        self.robot = self.p.loadURDF(os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf"),
                                     useFixedBase=True)
        self.init_robot()
        self.p.setGravity(0, 0, -10)

        self.duck_init_pos = [0.6, 0, 0.05]
        self.duck_init_ori = self.p.getQuaternionFromEuler([0, 0, 0])
        self.random_variable = 2.5
        self.duck = self.spawn_object()

    def initialize(self):
        self.init_robot()

    def init_robot(self):
        self.p.resetJointState(self.robot, 0, 0)
        self.p.resetJointState(self.robot, 1, 0.307)
        self.p.resetJointState(self.robot, 2, 0)
        self.p.resetJointState(self.robot, 3, -2.7)
        self.p.resetJointState(self.robot, 4, 0)
        self.p.resetJointState(self.robot, 5, 3)
        self.p.resetJointState(self.robot, 6, -2.356)

        self.p.resetJointState(self.robot, 9, 0.01)
        self.p.resetJointState(self.robot, 10, 0.01)

    def spawn_object(self):
        duck = None
        if self.random_variable < 1:

            duck = self.p.loadURDF(os.path.join(pybullet_data.getDataPath(), "cube.urdf"),
                                   basePosition=self.duck_init_pos,
                                   globalScaling=0.07)

        elif self.random_variable < 2:

            duck = self.p.loadURDF(os.path.join(pybullet_data.getDataPath(), "soccerball.urdf"),
                                   basePosition=self.duck_init_pos, globalScaling=0.07)
            self.p.changeDynamics(duck, -1, linearDamping=0, angularDamping=0,
                                  rollingFriction=0.0001, spinningFriction=0.0001, restitution=0.9)

        elif self.random_variable < 3:

            duck = self.p.loadURDF(os.path.join(pybullet_data.getDataPath(), "soccerball.urdf"),
                                   basePosition=self.duck_init_pos, globalScaling=0.1)
            self.p.changeDynamics(duck, -1, linearDamping=0, angularDamping=0,
                                  rollingFriction=0.0001, spinningFriction=0.0001, restitution=0.9)

        elif self.random_variable < 4:

            duck = self.p.loadURDF(os.path.join(pybullet_data.getDataPath(), "cube.urdf"),
                                   basePosition=self.duck_init_pos,
                                   globalScaling=0.1)

        return duck

    def get_robot_pos_ori(self, robot_id):
        gripper_information = self.p.getLinkState(self.robot, robot_id)
        return list(gripper_information[0]), list(self.p.getEulerFromQuaternion(gripper_information[1]))

    def get_base_pos_ori(self, obj_id):
        info = self.p.getBasePositionAndOrientation(obj_id)
        return list(info[0]), list(self.p.getEulerFromQuaternion(info[1]))

    def get_base_velocity_linear(self, obj_id):
        info = self.p.getBaseVelocity(obj_id)
        return list(info[0])

    def run(self, bodyUniqueId, endEffectorLinkIndex, pos):
        ik_joints = [0, 1, 2, 3, 4, 5, 6]
        conf = self.p.calculateInverseKinematics(bodyUniqueId=bodyUniqueId, endEffectorLinkIndex=endEffectorLinkIndex,
                                                 targetPosition=pos)

        if type(conf) is None:
            print('Failure!')
            # path = None
            return

        self.p.setJointMotorControlArray(self.robot, ik_joints, self.p.POSITION_CONTROL, conf[:-2])

        iterations = 0
        while self.wait(pos) and iterations < 100:
            iterations += 1
            self.p.stepSimulation()
            time.sleep(1.0 / self.hz)

    def wait(self, goal_pos):
        pos, ori = self.get_robot_pos_ori(self.link)

        distance = np.linalg.norm(np.array(pos[:2]) - np.array(goal_pos[:2]))
        if distance < 0.01:
            return False

        return True

    def push_degree(self, degree, distance=20):
        pos, ori = self.get_robot_pos_ori(self.link)
        o_pos, o_ori = self.get_base_pos_ori(self.duck)
        # Move to the target point 1
        pos[0] = o_pos[0] - distance * math.sin(math.radians(degree)) * self.unit_step_length
        pos[1] = o_pos[1] + distance * math.cos(math.radians(degree)) * self.unit_step_length
        self.run(self.robot, self.link, pos)

        if degree <= 90:
            rotation = 90 - degree
        else:
            rotation = 270 - degree

        self.p.resetJointState(self.robot, 6, math.radians(rotation) - 2.356)      # Rotate hand

        # # Move to the target point 2
        pos[0] = 1.25 * o_pos[0]
        pos[1] = 1.25 * o_pos[1]
        self.run(self.robot, self.link, pos)

        k = 0
        while k < 50:
            k += 1
            self.p.stepSimulation()
            time.sleep(1.0 / self.hz)

    def reset(self):
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING, 0)
        self.init_robot()
        self.p.resetBasePositionAndOrientation(self.duck, self.duck_init_pos, self.duck_init_ori)
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING, 1)

    def episode(self, action):
        self.push_degree(action)

        return self.get_base_pos_ori(self.duck)

    def close(self):
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING, 0)
        self.init_robot()
        self.p.resetBasePositionAndOrientation(self.duck, self.duck_init_pos, self.duck_init_ori)
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING, 1)

    def render(self, mode='human'):
        view_matrix = self.p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.6, 0, 0.05],
                                                               distance=.4,
                                                               yaw=90,
                                                               pitch=-90,
                                                               roll=0,
                                                               upAxisIndex=2)

        proj_matrix = self.p.computeProjectionMatrixFOV(fov=60,
                                                        aspect=float(960) / 720,
                                                        nearVal=0.1,
                                                        farVal=100.0)

        # Returns (width, height, rgbPixels, depthPixels, segmentationMaskBuffer)
        # rgb pixels -> list of [char RED,char GREEN,char BLUE, char ALPHA] [0..width*height]
        # depthPixels > list of float [0..width*height]
        # segmentationMaskBuffer -> list of int [0..width*height]
        (_, _, px, _, _) = self.p.getCameraImage(width=64,
                                                 height=64,
                                                 viewMatrix=view_matrix,
                                                 projectionMatrix=proj_matrix,
                                                 renderer=self.p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (64, 64, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array
