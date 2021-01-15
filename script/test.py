import rospy
from gazebo_msgs.msg import ModelStates

data = None

rospy.init_node('test')

def callback(d):
    global data
    data = d

sub = rospy.Subscriber('/gazebo/model_states', ModelStates, callback)
print('end')