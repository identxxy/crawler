import sys, time, os, argparse, socket
import torch
import torch.optim as optim
from torch.autograd import Variable
import gym

from Training import *
from ACNet import acNet, load_checkpoint, Shared_obs_stats

import crawler.register_all_env
import rospy

# import pybullet_envs

#config param
parser = argparse.ArgumentParser(description = "crawlerD");

## env
parser.add_argument('--num_steps',     type=int,   default=2048,    help='Input length to the network for training');
parser.add_argument('--batch_size',     type=int,   default=64,    help='Batch size, number of speakers per batch');
parser.add_argument('--fps', type=int,  default=1000,    help='fps');
parser.add_argument('--nDataLoaderThread', type=int, default=5,     help='Number of loader threads');
parser.add_argument('--env_name',      type=str,   default="CrawlerStandupEnv-v0", help='env_name');

## Training details
parser.add_argument('--save_interval',  type=int,   default=20,     help='Test and save every [test_interval] epochs');
parser.add_argument('--time_horizon',      type=int,   default=500,    help='Maximum number of epochs');
parser.add_argument('--max_episode_length',      type=int,   default=1000,    help='Maximum number of episodes');

## Optimizer
parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam');
parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler');
parser.add_argument('--lr',             type=float, default=1e-4,  help='Learning rate');
parser.add_argument('--weight_decay',   type=float, default=0,      help='Weight decay in the optimizer');
parser.add_argument('--num_epoch',      type=int,   default=2,    help='number of epochs of optimize');

## Loss functions
parser.add_argument('--gamma',             type=float, default=0.99,  help='gamma');
parser.add_argument('--gae_param',             type=float, default=0.95,  help='gae_param');
parser.add_argument('--clip',             type=float, default=0.2,  help='clip');
parser.add_argument('--ent_coeff',             type=float, default=0.01,  help='ent_coeff');
parser.add_argument('--max_grad_norm',             type=float, default=0.5,  help='max_grad_norm');
parser.add_argument('--seed',             type=float, default=1,  help='seed');

## Load and save
parser.add_argument('--cont_train',  type=bool,   default=False,     help='continues training');
parser.add_argument('--initial_model',  type=int,   default=0,     help='old model num');
parser.add_argument('--save_path',      type=str,   default="exp", help='Path for model and logs');

## Model definition
parser.add_argument('--inputsize',         type=int,   default=61,     help='inputsize');
parser.add_argument('--hiddensize', nargs='+', type=int,   default = [100, 100, 100],  help='hiddensize')
parser.add_argument('--lstmhiddensize',   type=int,   default=100,  help='Embedding size in the LSTM layer');
parser.add_argument('--outputsize',           type=int,   default=12,    help='outputsize');

## For test only
parser.add_argument('--mode',           type=str, help='train test demo')

params = parser.parse_args();

def main():
    rospy.init_node('crawler_gyb_ppo', anonymous=True, log_level=rospy.INFO)
    env = gym.make(params.env_name)

    torch.manual_seed(params.seed)
    num_inputs = env.observation_space.shape[0]
    num_outputs = env.action_space.shape[0]
    model = acNet(num_inputs, num_outputs, params.hiddensize, params.lstmhiddensize)
    shared_obs_stats = Shared_obs_stats(num_inputs)
    optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay = params.weight_decay)


    #train mode
    if params.mode == 'train':

        if not (os.path.exists(params.save_path)):
            os.makedirs(params.save_path)

        if params.cont_train:
            load_checkpoint(params.save_path, params.initial_model, model, optimizer)



        print('Python Version:', sys.version)
        print('PyTorch Version:', torch.__version__)
        print('Number of GPUs:', torch.cuda.device_count())
        print('Save path:', params.save_path)

        train(env, model, optimizer, shared_obs_stats, params)


    #test mode
    elif params.mode == 'test':
        state = env.reset()
        state = Variable(torch.Tensor(state).unsqueeze(0))
        hx = torch.zeros((1, params.lstmhiddensize))
        cx = torch.zeros((1, params.lstmhiddensize))
        load_checkpoint(params.save_path, params.initial_model, model, optimizer)
        model.eval()
        while (True):
            cum_reward = 0
            for step in range(params.num_steps):
                shared_obs_stats.observes(state)
                state = shared_obs_stats.normalize(state)
                mu, sigma, v, hx, cx = model(state, hx, cx)
                action = (mu + torch.exp(sigma) * Variable(torch.randn(mu.size())))
                log_prob = -0.5 * ((action - mu) / torch.exp(sigma)).pow(2) - 0.5 * math.log(2 * math.pi) - sigma
                log_prob = log_prob.sum(-1, keepdim=True)
                env_action = action.data.squeeze().numpy()
                state, reward, done, _ = env.step(env_action)
                cum_reward += reward
                # reward = max(min(reward, 1), -1)
                state = Variable(torch.Tensor(state).unsqueeze(0))

    #demo mode
    elif params.mode == 'demo':
        return

    else:
        quit()


if __name__ == '__main__':
    main()
