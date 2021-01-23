import os
import sys
import gym
from gym import wrappers
import random
import numpy as np
import math

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from ACNet import save_checkpoint, load_checkpoint

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.load = 0
        self.batchsize = 0
        self.memory = []

    def push(self, events):
        #Events = list(zip(*events))
        self.memory.append(map(lambda x: torch.cat(x,0), events))
        self.load += len(events[1])
        self.batchsize += 1
        #if len(self.memory)>self.capacity:
            #del self.memory[0]

    def clear(self):
        self.memory = []
        self.load = 0
        self.batchsize = 0

    def pull(self):
        Memory = list(zip(*self.memory))
        data = map(lambda x: torch.stack(x,0).permute(1,0,2), Memory)
        return data

def train(env, model, optimizer, shared_obs_stats, params):
    memory = ReplayMemory(params.num_steps)
    state = env.reset()
    state = Variable(torch.Tensor(state).unsqueeze(0))
    done = True
    episode = params.initial_model
    model.train()

    # horizon loop
    for t in range(params.time_horizon):
        #model.eval()
        episode_length = 0
        while(memory.load < params.num_steps):
            states = []
            h = []
            c = []
            actions = []
            rewards = []
            values = []
            returns = []
            advantages = []
            logprobs = []
            av_reward = 0
            cum_reward = 0
            cum_done = 0
            hx = torch.zeros((1,params.lstmhiddensize))
            cx = torch.zeros((1,params.lstmhiddensize))
            

            # n steps loops
            for step in range(params.num_steps):
                episode_length += 1
                shared_obs_stats.observes(state)
                state = shared_obs_stats.normalize(state)
                states.append(state)
                mu, sigma, v, hx, cx = model.single_forward(state, hx, cx)
                #h.append(hx)
                #c.append(cx)
                action = (mu + torch.exp(sigma)*Variable(torch.randn(mu.size())))
                actions.append(action)
                log_prob = -0.5 * ((action - mu) / torch.exp(sigma)).pow(2) - 0.5 * math.log(2 * math.pi) - sigma
                log_prob = log_prob.sum(-1, keepdim=True)
                logprobs.append(log_prob)
                values.append(v)
                env_action = torch.tanh(action).data.squeeze().numpy()
                state, reward, done, _ = env.step(env_action)
                done = (done or episode_length >= params.max_episode_length)
                cum_reward += reward
                # reward = max(min(reward, 1), -1)
                rewards.append(reward)

                if done:
                    episode += 1
                    cum_done += 1
                    av_reward += cum_reward
                    cum_reward = 0
                    episode_length = 0
                    state = env.reset()
                state = Variable(torch.Tensor(state).unsqueeze(0))
                if done:
                    break

            # one last step
            R = torch.zeros(1, 1)
            if not done:
                _,_,v = model(state)
                R = v.data

            # compute returns and GAE(lambda) advantages:
            R = Variable(R)
            values.append(R)
            A = Variable(torch.zeros(1, 1))
            for i in reversed(range(len(rewards))):
                td = rewards[i] + params.gamma*values[i+1].data[0,0] - values[i].data[0,0]
                A = float(td) + params.gamma*params.gae_param*A
                advantages.insert(0, A)
                R = A + values[i]
                returns.insert(0, R)

            # store usefull info:
            memory.push([states, actions, returns, advantages, logprobs])

         
        #model.train()
        # epochs
        batch_states, batch_actions, batch_returns, batch_advantages, batch_logprobs = memory.pull()
        batch_actions = Variable(batch_actions.data, requires_grad=False)
        batch_states = Variable(batch_states.data, requires_grad=False)
        batch_returns = Variable(batch_returns.data, requires_grad=False)
        batch_advantages = Variable(batch_advantages.data, requires_grad=False)
        batch_logprobs = Variable(batch_logprobs.data, requires_grad=False)

        for k in range(params.num_epoch):

            # new probas
            hx = torch.zeros((memory.batchsize,params.lstmhiddensize))
            cx = torch.zeros((memory.batchsize,params.lstmhiddensize))
            Mu, Sigma, V_pred= model(batch_states, hx, cx)     #size: length * batch * sigma_size

            log_probs = -0.5 * ((batch_actions - Mu)/ torch.exp(Sigma)).pow(2) - 0.5 * math.log(2 * math.pi) - Sigma
            log_probs = log_probs.sum(-1, keepdim=True)
            dist_entropy = 0.5 + 0.5 * math.log(2 * math.pi) + Sigma
            dist_entropy = dist_entropy.mean().sum(-1, keepdim=True)

            # ratio
            ratio = torch.exp(log_probs - batch_logprobs)

            # clip loss
            surr1 = ratio * batch_advantages.expand_as(ratio) # surrogate from conservative policy iteration
            surr2 = ratio.clamp(1-params.clip, 1+params.clip) * batch_advantages.expand_as(ratio)
            loss_clip = - torch.mean(torch.min(surr1, surr2))

            # value loss
            loss_value = (V_pred - batch_returns).pow(2).mean()

            # entropy
            loss_ent = - params.ent_coeff * dist_entropy

            # gradient descent step
            total_loss = (loss_clip + loss_value + loss_ent)

            loss = torch.square(log_probs - torch.full_like(batch_logprobs,0)).mean() + torch.square(V_pred - torch.full_like(batch_returns,1.0)).mean()

            print(loss)
            optimizer.zero_grad()
            loss.backward()
            #loss.backward()
            #nn.utils.clip_grad_norm(model.parameters(), params.max_grad_norm)
            optimizer.step()

        # finish, print:
        if episode % params.save_interval ==0:
            save_checkpoint(params.save_path, episode, model, optimizer)
        print('episode',episode,'av_reward',av_reward/float(cum_done), 'total loss', loss)
        memory.clear()

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


