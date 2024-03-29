from .robot_env import crawler_env

from gym import spaces
import numpy as np
import rospy
import rospkg
import rosparam
import os

from tf.transformations import euler_from_quaternion

def LoadYamlFileParamsTest(rospackage_name, rel_path_from_package_to_file, yaml_file_name):

    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path(rospackage_name)
    config_dir = os.path.join(pkg_path, rel_path_from_package_to_file) 
    path_config_file = os.path.join(config_dir, yaml_file_name)
    
    paramlist=rosparam.load_file(path_config_file)
    
    for params, ns in paramlist:
        rosparam.upload_params(ns,params)


class StandupTaskEnv(crawler_env.CrawlerRobotEnv):
    def __init__(self, **kwargs):
        # Load all the params first
        LoadYamlFileParamsTest("crawler", "config", "standup_param.yaml")
        # Variables that we retrieve through the param server, loded when launch training launch.
        self.reward_height_b = rospy.get_param('/crawler/reward_height_b')
        self.reward_height_k = rospy.get_param('/crawler/reward_height_k')
        self.effort_penalty = rospy.get_param('/crawler/effort_penalty')
        self.effort_max = rospy.get_param('/crawler/effort_max')
        self.epoch_steps = rospy.get_param('/crawler/epoch_steps')
        self.running_step = rospy.get_param('/crawler/running_step')
        # Construct the RobotEnv so we know the dimension of cmd
        super(StandupTaskEnv, self).__init__(**kwargs)
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
        self.steps = 0
        self.cmd = np.zeros(self.n * 16)
        for r in self.robots:
            r.roll = 0
            r.pitch = 0
            r.yaw = 0


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
            reward = (r.global_pos.position.z - self.reward_height_b) * self.reward_height_k
            reward -= self.effort_penalty * sum(map(abs, r.joints.effort)) / self.effort_max
            rewards[i] = reward
        return rewards
        
    # Internal TaskEnv Methods

