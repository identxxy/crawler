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
    def __init__(self):
        # Load all the params first
        LoadYamlFileParamsTest("crawler", "config", "walk_forwad_param.yaml")
        # Variables that we retrieve through the param server, loded when launch training launch.
        self.reward_x_vel = rospy.get_param('/crawler/reward_x_vel')
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
        super(WalkXTaskEnv, self).__init__()
        # Only variable needed to be set here
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(16, 1), dtype=np.float32)
        self._init_env_variables()
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(60, 1), dtype=np.float32)
        
        rospy.logdebug("END init TestTaskEnv")

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        pass

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        self.roll = 0
        self.pitch = 0
        self.yaw = 0
        self.cmd = np.zeros(len(self.publisher_list))
        self.steps = 0


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
        joints, global_pos, global_vel = self.obs_joints()
        orientation_q = global_pos.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (self.roll, self.pitch, self.yaw) = euler_from_quaternion (orientation_list)
        return np.concatenate([
            joints.position,
            joints.velocity,
            joints.effort,
            (
                global_pos.position.x,
                global_pos.position.y,
                global_pos.position.z,
                self.roll,
                self.pitch,
                self.yaw,
                global_vel.linear.x,
                global_vel.linear.y,
                global_vel.linear.z,
                global_vel.angular.x,
                global_vel.angular.y,
                global_vel.angular.z
            )
        ])

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
        reward = self.global_vel.linear.x * self.reward_x_vel
        reward -= self.reward_ori_k * ( 1 - math.cos(self.roll) * math.cos(self.pitch) )
        if self.global_pos.position.z < self.reward_height_thd: # punishment
            reward += self.reward_height_k * (self.global_pos.position.z - self.reward_height_thd)
        reward -= self.effort_penalty * sum(map(abs, self.joints.effort)) / self.effort_max
        knee_land_cnt = 0
        knee_land_cnt += self.get_link_state(self.ns[1:]+'::leg4_B', None).link_state.pose.position.z < self.punish_knee_thd
        knee_land_cnt += self.get_link_state(self.ns[1:]+'::leg4_F', None).link_state.pose.position.z < self.punish_knee_thd
        knee_land_cnt += self.get_link_state(self.ns[1:]+'::leg4_L', None).link_state.pose.position.z < self.punish_knee_thd
        knee_land_cnt += self.get_link_state(self.ns[1:]+'::leg4_R', None).link_state.pose.position.z < self.punish_knee_thd
        reward -= knee_land_cnt * self.punish_knee
        return reward
        
    # Internal TaskEnv Methods

