from . import robot_gazebo_env
import rospy
import numpy as np
from gym.utils import seeding

from std_msgs.msg import Float64
from rosgraph_msgs.msg import Clock
from gazebo_msgs.msg import ModelStates, ContactsState
from sensor_msgs.msg import JointState, Image

CAMERA_SIZE = 64 * 64

class Robot():
    def __init__(self, i, displacement_xyz):
        self.i = i
        self.displacement_xyz = displacement_xyz
        self.ns = "/crawler_" + str(self.i)
        self.publisher_list = []
        # The joint_state_controller control no joint but pub the state of all joints
        self.joint_subscriber = rospy.Subscriber(self.ns + '/joint_states', JointState, self.joints_callback)
        self.global_subscriber = rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_callback)
        self.camera_subscriber = rospy.Subscriber(self.ns + '/camera/image_raw', Image, self.camera_callback)
        self.contact_subscriber = rospy.Subscriber(self.ns + '/contact_msgs', ContactsState, self.contact_callback)
        self.joints = None
        self.global_pos = None
        self.global_vel = None
        self.model_index = None
        self.knee_land_cnt = 0
        self.roll = 0
        self.pitch = 0
        self.yaw = 0

    def model_callback(self, data):
        if self.model_index:
            self.global_pos = data.pose[self.model_index]
            self.global_pos.position.x -= self.displacement_xyz[0] * self.i
            self.global_pos.position.y -= self.displacement_xyz[1] * self.i
            self.global_pos.position.z -= self.displacement_xyz[2] * self.i
            self.global_vel = data.twist[self.model_index]

    def camera_callback(self, data):
        if self.model_index:
            self.camera = data

    def contact_callback(self, data):
        if self.model_index:
            self.knee_land_cnt += len(data.states)


    def joints_callback(self, data):
        self.joints = data

class CrawlerRobotEnv(robot_gazebo_env.RobotGazeboEnv):
    """Superclass for all Robot environments.
    """

    def __init__(self, **kwargs):
        """Initializes a new Robot environment.
        """
        # Variables that we give through the constructor.
        # namespace
        self.n = kwargs['n']
        self.robots = [Robot(i, kwargs['displacement_xyz']) for i in range(self.n)]
        self.controllers_list = [
            'joint_state_controller',
            'joint1_LB_controller',
            'joint1_LF_controller',
            'joint1_RB_controller',
            'joint1_RF_controller',
            'joint2_LB_controller',
            'joint2_LF_controller',
            'joint2_RB_controller',
            'joint2_RF_controller',
            'joint3_LB_controller',
            'joint3_LF_controller',
            'joint3_RB_controller',
            'joint3_RF_controller',
            'joint4_LB_controller',
            'joint4_LF_controller',
            'joint4_RB_controller',
            'joint4_RF_controller'
        ]
        for r in self.robots:
            for n in self.controllers_list[1:]:
                r.publisher_list.append(
                    rospy.Publisher(r.ns + '/' + n + '/command', Float64, queue_size=1))

        self.all_controllers_list = []
        for r in self.robots:
            for c in self.controllers_list:
                self.all_controllers_list.append(r.ns + '/' + c)
        reset_controls_bool = True
        super(CrawlerRobotEnv, self).__init__( n=self.n, robot_name_spaces=['crawler_'+str(i) for i in range(self.n)],
                                                controllers_list=self.controllers_list,
                                                reset_controls=reset_controls_bool)
        self.single_obs_space = (60 + CAMERA_SIZE) 
        rospy.logdebug("END init CrawlerRobotEnv")

    
    # Methods needed by the RobotGazeboEnv
    # ----------------------------

    def _check_all_systems_ready(self):
        """
        Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        for r in self.robots:
            r.joints = None
            r.camera = None
            while r.joints is None and not rospy.is_shutdown():
                try:
                    r.model_index = rospy.wait_for_message('/gazebo/model_states', ModelStates, 3).name.index(r.ns[1:])
                except rospy.exceptions.ROSException:
                    rospy.logerr("Robot model does not exist.")
                try:
                    r.joints = rospy.wait_for_message(
                        r.ns + '/joint_states', JointState, timeout=3.0)
                except:
                    rospy.logerr("Current /joint_states not ready yet.\n\
                     Do you spawn the robot and launch ros_control?")
                try:
                    r.camera = rospy.wait_for_message(
                        r.ns + '/camera/image_raw', Image, timeout=3.0)
                except rospy.exceptions.ROSException:
                    rospy.logerr("image not ready.")
                try:
                    r.knee_land_cnt = 0
                    rospy.wait_for_message(
                        r.ns + '/contact_msgs', ContactsState, timeout=3.0)
                except rospy.exceptions.ROSException:
                    rospy.logerr("contact not ready.")

        # rospy.logdebug("ALL SYSTEMS READY")
        return True

    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        cmd = np.zeros(16 * self.n)
        self.move_joints(cmd)
    
    
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
        for i in range(len(self.robots)):
            r = self.robots[i]
            for j in range(len(r.publisher_list)):
                joint_value = Float64()
                joint_value.data = cmd[16 * i + j]
                r.publisher_list[j].publish(joint_value)

    def obs_states(self):
        return [(r.joints, r.global_pos, r.global_vel, r.camera.data) for r in self.robots]
