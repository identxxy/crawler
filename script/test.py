import rospy 
import gym
import crawler.register_all_env
import rospy

rospy.init_node('test')
env = gym.make('CrawlerStandupEnv-v0', n=2)
obs, reward, done, info = env.step([0.0] * 32)
obs, reward, done, info = env.step([0.0] * 32)
obs, reward, done, info = env.step([0.0] * 32)
obs, reward, done, info = env.step([0.0] * 32)
env.reset()

