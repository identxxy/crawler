#!/usr/bin/env python

import gym
import time
import numpy
import random
import os
import qlearn
from gym import wrappers
from functools import reduce

# ROS packages required
import rospy
import rospkg
import crawler.register_all_env

if __name__ == '__main__':
    rospy.init_node('crawler_gym_test', anonymous=True, log_level=rospy.INFO)

    # Create the Gym environment
    # env = gym.make('CrawlerTrainingEnv-v0')
    env = gym.make('CrawlerTestEnv-v0')
    rospy.loginfo ( "Gym environment done")
        
    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('crawler')
    outdir = pkg_path + '/training_results'
    env = wrappers.Monitor(env, outdir, force=True) 
    rospy.loginfo ( "Monitor Wrapper started")

    last_time_steps = numpy.ndarray(0)


    Alpha = rospy.get_param("/crawler/alpha")
    Epsilon = rospy.get_param("/crawler/epsilon")
    Gamma = rospy.get_param("/crawler/gamma")
    epsilon_discount = rospy.get_param("/crawler/epsilon_discount")
    nepisodes = rospy.get_param("/crawler/nepisodes")
    nsteps = rospy.get_param("/crawler/nsteps")

    qlearn = qlearn.QLearn(actions=range(env.action_space.n),
                    alpha=Alpha, gamma=Gamma, epsilon=Epsilon)
    initial_epsilon = qlearn.epsilon

    start_time = time.time()
    highest_reward = 0

 
    for x in range(nepisodes):

        
        cumulated_reward = 0  
        done = False
        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount
        

        
        observation = env.reset()
        state = ''.join(map(str, observation))
        
 
        for i in range(nsteps):
            

            action = qlearn.chooseAction(state)

            observation, reward, done, info = env.step(action)

            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))


            qlearn.learn(state, action, reward, nextState)

            if not(done):
                state = nextState
            else:
   
                last_time_steps = numpy.append(last_time_steps, [int(i + 1)])
                break

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        rospy.loginfo ( ("EP: "+str(x+1)+" - [alpha: "+str(round(qlearn.alpha,2))+" - gamma: "+str(round(qlearn.gamma,2))+" - epsilon: "+str(round(qlearn.epsilon,2))+"] - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s)))

    rospy.loginfo ( ("\n|"+str(nepisodes)+"|"+str(qlearn.alpha)+"|"+str(qlearn.gamma)+"|"+str(initial_epsilon)+"*"+str(epsilon_discount)+"|"+str(highest_reward)+"| PICTURE |"))

    l = last_time_steps.tolist()
    l.sort()

    #print("Parameters: a="+str)
    rospy.loginfo("Overall score: {:0.2f}".format(last_time_steps.mean()))
    rospy.loginfo("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()
