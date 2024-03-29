from .robot_env import crawler_cam_env

from gym import spaces
import numpy as np
import rospy
import rospkg
import rosparam
import os
import math

from tf.transformations import euler_from_quaternion
from gazebo_msgs.srv import GetLinkState

def LoadYamlFileParamsTest(rospackage_name, rel_path_from_package_to_file, yaml_file_name):

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path(rospackage_name)
    config_dir = os.path.join(pkg_path, rel_path_from_package_to_file) 
    path_config_file = os.path.join(config_dir, yaml_file_name)
    
    paramlist=rosparam.load_file(path_config_file)
    
    for params, ns in paramlist:
        rosparam.upload_params(ns,params)


class WalkXCamTaskEnv(crawler_cam_env.CrawlerRobotEnv):
    def __init__(self, **kwargs):
        # Load all the params first
        LoadYamlFileParamsTest("crawler", "config", "walk_forward_param.yaml")
        # Variables that we retrieve through the param server, loded when launch training launch.
        self.reward_x_vel = rospy.get_param('/crawler/reward_x_vel')
        self.reward_y_vel = rospy.get_param('/crawler/reward_y_vel')
        self.reward_height_thd = rospy.get_param('/crawler/reward_height_thd')
        self.reward_height_k = rospy.get_param('/crawler/reward_height_k')
        self.reward_ori_k = rospy.get_param('/crawler/reward_ori_k')

        # self.punish_knee_thd = rospy.get_param('/crawler/punish_knee_thd')
        self.punish_knee= rospy.get_param('/crawler/punish_knee')

        self.effort_penalty = rospy.get_param('/crawler/effort_penalty')
        self.effort_max = rospy.get_param('/crawler/effort_max')
        self.epoch_steps = rospy.get_param('/crawler/epoch_steps')
        self.running_step = rospy.get_param('/crawler/running_step')
        rospy.wait_for_service('gazebo/get_link_state')
        self.get_link_state = rospy.ServiceProxy('gazebo/get_link_state', GetLinkState)
        # Construct the RobotEnv so we know the dimension of cmd
        super(WalkXCamTaskEnv, self).__init__(**kwargs)
        # Only variable needed to be set here
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(16 * self.n, 1), dtype=np.float32)
        self._init_env_variables()
        
        self.obs_space = self.single_obs_space * self.n
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_space * self.n, 1), dtype=np.float32)
        
        self.cmd = np.zeros(self.n * 16)

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        self.steps = 0
        self.cmd = np.zeros(self.n * 16)
        self.last_knee_land_cnt = np.zeros(self.n)

    def _set_action(self, action):
        """
        Move the robot based on the action variable given
        """
        self.cmd = self.effort_max * action
        self.move_joints(self.cmd)
        rospy.sleep(self.running_step)
        self.steps += 1

    def _get_obs(self):
        """
        Here we define what sensor data of our robots observations
        To know which Variables we have acces to, we need to read the
        MyRobotEnv API DOCS
        :return: observations
        """
        states = self.obs_states()
        obs = []
        for i in range(self.n):
            r = self.robots[i]
            joints, global_pos, global_vel, camera = states[i]
            orientation_q = global_pos.orientation
            orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            (r.roll, r.pitch, r.yaw) = euler_from_quaternion (orientation_list)
            single_obs = []
            single_obs.append(np.concatenate([
                joints.position,
                joints.velocity,
                joints.effort,
                (
                    global_pos.position.x,
                    global_pos.position.y,
                    global_pos.position.z,
                    r.roll,
                    r.pitch,
                    r.yaw,
                    global_vel.linear.x,
                    global_vel.linear.y,
                    global_vel.linear.z,
                    global_vel.angular.x,
                    global_vel.angular.y,
                    global_vel.angular.z
                ),
            ]))
            img = np.zeros([3,64,64], dtype=np.uint8)
            f = np.frombuffer(camera[0::3], dtype=np.uint8)
            img[0] = f.reshape(64,64)
            f = np.frombuffer(camera[1::3], dtype=np.uint8)
            img[1] = f.reshape(64,64)
            f = np.frombuffer(camera[2::3], dtype=np.uint8)
            img[2] = f.reshape(64,64)
            single_obs.append(img)
            obs.append(single_obs)
        return obs

    def _is_done(self, observations):
        """
        Decide if episode is done based on the observations
        """
        done = self.steps >= self.epoch_steps
        return done

    def _compute_reward(self, observations, done):
        """
        Return the reward based on the observations given
        """
        rewards = np.zeros(self.n, dtype=float)
        for i in range(self.n):
            r = self.robots[i]
            reward = r.global_vel.linear.x * self.reward_x_vel + abs(r.global_vel.linear.y) * self.reward_y_vel
            reward -= self.reward_ori_k * ( 1 - math.cos(r.roll) * math.cos(r.pitch) )
            if r.global_pos.position.z < self.reward_height_thd: # punishment
                reward += self.reward_height_k * (r.global_pos.position.z - self.reward_height_thd)
            reward -= self.effort_penalty * sum(map(abs, r.joints.effort)) / self.effort_max
            knee_land_cnt = r.knee_land_cnt - self.last_knee_land_cnt[i]
            reward -= knee_land_cnt * self.punish_knee
            rewards[i] = reward
            self.last_knee_land_cnt[i] = r.knee_land_cnt
        return rewards
        
    # Internal TaskEnv Methods


class WalkXTaskEnv_v1(WalkXCamTaskEnv):
    def __init__(self, **kwargs):
        super(WalkXTaskEnv_v1, self).__init__(**kwargs)
        self.lastPos = np.zeros([2, self.n])

    def _init_env_variables(self):
        self.lastPos = np.zeros([2, self.n])
        WalkXCamTaskEnv._init_env_variables(self)

    def _compute_reward(self, observations, done):
        rewards = np.zeros(self.n, dtype=float)
        for i in range(self.n):
            r = self.robots[i]
            x = r.global_pos.position.x
            y = r.global_pos.position.y
            lastx = self.lastPos[0,i]
            lasty = self.lastPos[1,i]
            reward = ( (x-lastx) * self.reward_x_vel + abs(y-lasty) * self.reward_y_vel ) / self.running_step
            reward -= self.reward_ori_k * ( 1 - math.cos(r.roll) * math.cos(r.pitch) )
            if r.global_pos.position.z < self.reward_height_thd: # punishment
                reward += self.reward_height_k * (r.global_pos.position.z - self.reward_height_thd)
            reward -= self.effort_penalty * sum(map(abs, r.joints.effort)) / self.effort_max
            knee_land_cnt = r.knee_land_cnt - self.last_knee_land_cnt[i]
            reward -= knee_land_cnt * self.punish_knee
            rewards[i] = reward
            self.last_knee_land_cnt[i] = r.knee_land_cnt
            self.lastPos[0,i] = x
            self.lastPos[1,i] = y
        return rewards

class WalkXTaskEnv_v2(WalkXTaskEnv_v1):
    def __init__(self, **kwargs):
        super(WalkXTaskEnv_v2, self).__init__(**kwargs)
        self.reward_step = 0
        self.xdis = np.zeros([self.n])

    def _init_env_variables(self):
        self.reward_step = 0
        self.xdis = np.zeros([self.n])
        WalkXTaskEnv_v1._init_env_variables(self)

    def _compute_reward(self, observations, done):
        self.reward_step += 1
        rewards = WalkXTaskEnv_v1._compute_reward(self, observations, done)
        if self.reward_step % 128 == 0:        
            for i in range(self.n):
                currentDis = self.robots[i].global_pos.position.x
                rewards[i] += 5*(currentDis - self.xdis[i])
                self.xdis[i] = currentDis
        return rewards
    
class WalkXTaskEnv_v3(WalkXCamTaskEnv):
    def __init__(self, **kwargs):
        super(WalkXTaskEnv_v3, self).__init__(**kwargs)
        self.reward_step = 0
        self.xdis = np.zeros([self.n])

    def _init_env_variables(self):
        self.reward_step = 0
        self.xdis = np.zeros([self.n])
        WalkXCamTaskEnv._init_env_variables(self)

    def _compute_reward(self, observations, done):
        self.reward_step += 1
        rewards = WalkXCamTaskEnv._compute_reward(self, observations, done)
        if self.reward_step % 128 == 0:        
            for i in range(self.n):
                #currentDis = self.robots[i].global_pos.position.x
                #rewards[i] += 10*(currentDis - self.xdis[i])
                #self.xdis[i] = currentDis
                self.xdis[i] = self.robots[i].global_pos.position.x - self.xdis[i]
                rewards[i] += 20 * self.xdis[i]
        return rewards