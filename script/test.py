import rospy 
from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import JointState

rospy.init_node('test')
try:
    msg = rospy.wait_for_message( '/crawler/joint_states', JointState, timeout=10.0)
except rospy.ROSException:
    print('f')
