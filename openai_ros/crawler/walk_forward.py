from .robot_env import crawler_env

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


class WalkXTaskEnv(crawler_env.CrawlerRobotEnv):
    def __init__(self, **kwargs):
        # Load all the params first
        LoadYamlFileParamsTest("crawler", "config", "walk_forwad_param.yaml")
        # Variables that we retrieve through the param server, loded when launch training launch.
        self.reward_x_vel = rospy.get_param('/crawler/reward_x_vel')
        self.reward_y_vel = rospy.get_param('/crawler/reward_y_vel')
        self.reward_height_thd = rospy.get_param('/crawler/reward_height_thd')
        self.reward_height_k = rospy.get_param('/crawler/reward_height_k')
        self.reward_ori_k = rospy.get_param('/crawler/reward_ori_k')

        self.punish_knee_thd = rospy.get_param('/crawler/punish_knee_thd')
        self.punish_knee= rospy.get_param('/crawler/punish_knee')

        self.effort_penalty = rospy.get_param('/crawler/effort_penalty')
        self.effort_max = rospy.get_param('/crawler/effort_max')
        self.epoch_steps = rospy.get_param('/crawler/epoch_steps')
        self.running_step = rospy.get_param('/crawler/running_step')
        rospy.wait_for_service('gazebo/get_link_state')
        self.get_link_state = rospy.ServiceProxy('gazebo/get_link_state', GetLinkState)
        # Construct the RobotEnv so we know the dimension of cmd
        super(WalkXTaskEnv, self).__init__(**kwargs)
        # Only variable needed to be set here
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(16 * self.n, 1), dtype=np.float32)
        self._init_env_variables()
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(60 * self.n, 1), dtype=np.float32)
        
    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        self.steps = 0
        self.cmd = np.zeros(self.n * 16)

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
        obs = np.zeros(60 * self.n, dtype=float)
        for i in range(self.n):
            r = self.robots[i]
            joints, global_pos, global_vel = states[i]
            orientation_q = global_pos.orientation
            orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
            (r.roll, r.pitch, r.yaw) = euler_from_quaternion (orientation_list)
            obs[60 * i: 60 * i + 60] = np.concatenate([
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
                )
            ])
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
            reward = r.global_vel.linear.x * self.reward_x_vel + r.global_vel.linear.y * self.reward_y_vel
            reward -= self.reward_ori_k * ( 1 - math.cos(r.roll) * math.cos(r.pitch) )
            if r.global_pos.position.z < self.reward_height_thd: # punishment
                reward += self.reward_height_k * (r.global_pos.position.z - self.reward_height_thd)
            reward -= self.effort_penalty * sum(map(abs, r.joints.effort)) / self.effort_max
            knee_land_cnt = 0
            knee_land_cnt += self.get_link_state(r.ns[1:]+'::leg4_B', None).link_state.pose.position.z < self.punish_knee_thd
            knee_land_cnt += self.get_link_state(r.ns[1:]+'::leg4_F', None).link_state.pose.position.z < self.punish_knee_thd
            knee_land_cnt += self.get_link_state(r.ns[1:]+'::leg4_L', None).link_state.pose.position.z < self.punish_knee_thd
            knee_land_cnt += self.get_link_state(r.ns[1:]+'::leg4_R', None).link_state.pose.position.z < self.punish_knee_thd
            reward -= knee_land_cnt * self.punish_knee
            rewards[i] = reward
        return rewards
        
    # Internal TaskEnv Methods


class WalkXTaskEnv_v1(WalkXTaskEnv):
    def __init__(self, **kwargs):
        super(WalkXTaskEnv_v1, self).__init__(**kwargs)
        self.lastPos = np.zeros([2, self.n])

    def _init_env_variables(self):
        self.lastPos = np.zeros([2, self.n])
        WalkXTaskEnv._init_env_variables(self)

    def _compute_reward(self, observations, done):
        rewards = np.zeros(self.n, dtype=float)
        for i in range(self.n):
            r = self.robots[i]
            x = r.global_pos.position.x
            y = r.global_pos.position.y
            lastx = self.lastPos[0,i]
            lasty = self.lastPos[1,i]
            reward = ( (x-lastx) * self.reward_x_vel + (y-lasty) * self.reward_y_vel ) / self.running_step
            reward -= self.reward_ori_k * ( 1 - math.cos(r.roll) * math.cos(r.pitch) )
            if r.global_pos.position.z < self.reward_height_thd: # punishment
                reward += self.reward_height_k * (r.global_pos.position.z - self.reward_height_thd)
            reward -= self.effort_penalty * sum(map(abs, r.joints.effort)) / self.effort_max
            knee_land_cnt = 0
            knee_land_cnt += self.get_link_state(r.ns[1:]+'::leg4_B', None).link_state.pose.position.z < self.punish_knee_thd
            knee_land_cnt += self.get_link_state(r.ns[1:]+'::leg4_F', None).link_state.pose.position.z < self.punish_knee_thd
            knee_land_cnt += self.get_link_state(r.ns[1:]+'::leg4_L', None).link_state.pose.position.z < self.punish_knee_thd
            knee_land_cnt += self.get_link_state(r.ns[1:]+'::leg4_R', None).link_state.pose.position.z < self.punish_knee_thd
            reward -= knee_land_cnt * self.punish_knee
            rewards[i] = reward
            self.lastPos[0,i] = x
            self.lastPos[1,i] = y
        return rewards