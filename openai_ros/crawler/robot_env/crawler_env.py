from . import robot_gazebo_env
import rospy
import numpy as np
from gym.utils import seeding

from std_msgs.msg import Float64
from rosgraph_msgs.msg import Clock
from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import JointState

class CrawlerRobotEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all Robot environments.
    """

    def __init__(self, **kwargs):
        """Initializes a new Robot environment.
        """
        # Variables that we give through the constructor.
        # namespace
        self.id = kwargs['robot_id']
        self.ns = "/crawler_" + str(self.id)
        self.controllers_list = [
            'joint_state_controller',
            'joint1_B_controller',
            'joint1_F_controller',
            'joint1_L_controller',
            'joint1_R_controller',
            'joint2_B_controller',
            'joint2_F_controller',
            'joint2_L_controller',
            'joint2_R_controller',
            'joint3_B_controller',
            'joint3_F_controller',
            'joint3_L_controller',
            'joint3_R_controller',
            'joint4_B_controller',
            'joint4_F_controller',
            'joint4_L_controller',
            'joint4_R_controller'
        ]

        # Internal Vars
        self.publisher_list = []
        # The joint_state_controller control no joint but pub the state of all joints
        for n in self.controllers_list[1:]:
            self.publisher_list.append(
                rospy.Publisher(self.ns + '/' + n + '/command', Float64, queue_size=1))
        self.joint_subscriber = rospy.Subscriber(self.ns + '/joint_states', JointState, self.joints_callback)
        self.global_subscriber = rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_callback)
        self.joints = None
        self.global_pos = None
        self.global_vel = None
        self.model_index = None

        reset_controls_bool = True
        
        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        
        super(CrawlerRobotEnv, self).__init__(controllers_list=self.controllers_list,
                                                robot_name_space=self.ns,
                                                reset_controls=reset_controls_bool)
        rospy.logdebug("END init CrawlerRobotEnv")

    def model_callback(self, data):
        if self.model_index:
            self.global_pos = data.pose[self.model_index]
            self.global_vel = data.twist[self.model_index]

    def joints_callback(self, data):
        self.joints = data
    
    # Methods needed by the RobotGazeboEnv
    # ----------------------------

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        self.joints = None
        while self.joints is None and not rospy.is_shutdown():
            try:
                self.joints = rospy.wait_for_message(
                    self.ns + '/joint_states', JointState, timeout=1.0)
            except:
                rospy.logerr(
                    "Current /joint_states not ready yet.\n Do you spawn the robot and launch ros_control?")
            try:
                self.model_index = rospy.wait_for_message('/gazebo/model_states', ModelStates, 3).name.index(self.ns[1:])
            except rospy.exceptions.ROSException:
                rospy.logerr("Robot model does not exist.")

        # rospy.logdebug("ALL SYSTEMS READY")
        return True

    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()
    
    
    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()
        
    # Methods that the TrainingEnvironment will need.
    # ----------------------------
    def move_joints(self, cmd):
        for i in range(len(self.publisher_list)):
            joint_value = Float64()
            joint_value.data = cmd[i]
            self.publisher_list[i].publish(joint_value)

    def obs_joints(self):
        return self.joints, self.global_pos, self.global_vel