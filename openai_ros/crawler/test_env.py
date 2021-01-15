from .robot_env import crawler_env

from gym import spaces
import numpy as np
import rospy
import rospkg
import rosparam
import os

def LoadYamlFileParamsTest(rospackage_name, rel_path_from_package_to_file, yaml_file_name):

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path(rospackage_name)
    config_dir = os.path.join(pkg_path, rel_path_from_package_to_file) 
    path_config_file = os.path.join(config_dir, yaml_file_name)
    
    paramlist=rosparam.load_file(path_config_file)
    
    for params, ns in paramlist:
        rosparam.upload_params(ns,params)


class TestTaskEnv(crawler_env.CrawlerRobotEnv):
    def __init__(self):
        # Load all the params first
        LoadYamlFileParamsTest("crawler", "config", "training_params.yaml")
        # Construct the RobotEnv so we know the dimension of cmd
        super(TestTaskEnv, self).__init__()
        # Only variable needed to be set here
        number_actions = rospy.get_param('/crawler/n_actions')
        self.action_space = spaces.Discrete(number_actions)
        self._init_env_variables()
        
        # This is the most common case of Box observation type
        high = np.array([ np.pi * 5/6, np.pi *5/6])
        low = np.array([0.0, 0.0])
            
        self.observation_space = spaces.Box(low, high)
        
        # Variables that we retrieve through the param server, loded when launch training launch.
        self.stop_z_pos = rospy.get_param('/crawler/stop_z_pos')
        self.stop_z_cnt = rospy.get_param('/crawler/stop_z_cnt')
        self.running_step = rospy.get_param('/crawler/running_step')
        self.effort_step = rospy.get_param('/crawler/effort_step')

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
        self.cmd = np.zeros(len(self.publisher_list))
        self.done_cnt = 0


    def _set_action(self, action):
        """
        Move the robot based on the action variable given
        """
        if action == 0:
            self.cmd[4:] = 0
        elif action == 1:
            self.cmd[4:] = self.effort_step
        elif action == 2:
            self.cmd[4:] = - self.effort_step
        self.move_joints(self.cmd)
        rospy.sleep(self.running_step)

    def _get_obs(self):
        """
        Here we define what sensor data of our robots observations
        To know which Variables we have acces to, we need to read the
        MyRobotEnv API DOCS
        :return: observations
        """
        data = self.obs_joints()
        obs = data.position[4:]
        return np.array(obs)

    def _is_done(self, observations):
        """
        Decide if episode is done based on the observations
        """
        done = self.global_pos.position.z > self.stop_z_pos
        if done:
            self.done_cnt = self.done_cnt + 1
        else:
            self.done_cnt = 0
        return self.done_cnt > self.stop_z_cnt

    def _compute_reward(self, observations, done):
        """
        Return the reward based on the observations given
        """
        reward = self.global_pos.position.z > self.stop_z_pos
        return reward
        
    # Internal TaskEnv Methods

