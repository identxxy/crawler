import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class acNet(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, lstm_hidden_size):
        super(acNet, self).__init__()

        self.linear = nn.Linear(num_inputs, hidden_size[0])

        self.hidden = []
        for i in range(len(hidden_size)-1):
            self.hidden.append(nn.Linear(hidden_size[i], hidden_size[i+1]))

        self.lstm = nn.LSTMCell(hidden_size[-1], lstm_hidden_size)
        self.critic_linear = nn.Sequential(nn.Linear(lstm_hidden_size, 100), nn.ReLU(), nn.Linear(100, 1))
        self.actor_linear = nn.Linear(lstm_hidden_size, 60)
        self.mu_linear = nn.Linear(60, num_actions)
        self.sigma_linear = nn.Linear(60, num_actions)

    def forward(self, x, hx, cx):

        x = torch.tanh(self.linear(x.view(x.size(0), -1)))

        for layer in self.hidden:
            x = torch.tanh(layer(x))

        hx, cx = self.lstm(x, (hx, cx))
        actor = torch.tanh(self.actor_linear(hx))
        mu = self.mu_linear(actor)
        sigma = self.sigma_linear(actor)
        return mu, sigma, self.critic_linear(hx), hx, cx


def save_checkpoint(save_path, episode, model, optimizer):
    if save_path == None:
        return
    save_path = '%s/%d.pt'%(save_path,episode)
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}

    torch.save(state_dict, save_path)

    print(f'Model saved to ==> {save_path}')


def load_checkpoint(save_path, episode, model, optimizer):
    save_path = '%s/%d.pt' % (save_path, episode)
    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    print(f'Model loaded from <== {save_path}')


class Shared_grad_buffers():
    def __init__(self, model):
        self.grads = {}
        for name, p in model.named_parameters():
            self.grads[name+'_grad'] = torch.ones(p.size()).share_memory_()

    def add_gradient(self, model):
        for name, p in model.named_parameters():
            self.grads[name+'_grad'] += p.grad.data

    def reset(self):
        for name,grad in self.grads.items():
            self.grads[name].fill_(0)

class Shared_obs_stats():
    def __init__(self, num_inputs):
        self.n = torch.zeros(num_inputs).share_memory_()
        self.mean = torch.zeros(num_inputs).share_memory_()
        self.mean_diff = torch.zeros(num_inputs).share_memory_()
        self.var = torch.zeros(num_inputs).share_memory_()

    def observes(self, obs):
        # observation mean var updates
        x = obs.data.squeeze()
        self.n += 1.
        last_mean = self.mean.clone()
        self.mean += (x-self.mean)/self.n
        self.mean_diff += (x-last_mean)*(x-self.mean)
        self.var = torch.clamp(self.mean_diff/self.n, min=1e-2)

    def normalize(self, inputs):
        obs_mean = Variable(self.mean.unsqueeze(0).expand_as(inputs))
        obs_std = Variable(torch.sqrt(self.var).unsqueeze(0).expand_as(inputs))
        return torch.clamp((inputs-obs_mean)/obs_std, -5., 5.)