import rospy 
import gym
import crawler.register_all_env
import rospy

rospy.init_node('test')
env1 = gym.make('CrawlerWalkXEnv-v0', robot_id=0)
env2 = gym.make('CrawlerWalkXEnv-v0', robot_id=1)
env.reset()

