import torch.nn as nn
import torch.nn.functional as F


class acNet(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(acNet, self).__init__()

        self.linear = nn.Linear(61, 40)
        self.lstm = nn.LSTMCell(40, 80)
        self.critic_linear = nn.Linear(80, 1)
        self.actor_linear = nn.Linear(80, 2)

    def forward(self, x, hx, cx):

        x = self.linear(x.view(x.size(0), -1))
        hx, cx = self.lstm(x, (hx, cx))
        return self.actor_linear(hx), self.critic_linear(hx), hx, cx


def save_checkpoint(save_path, model, optimizer):
    if save_path == None:
        return
    save_path = save_path
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}

    torch.save(state_dict, save_path)

    print(f'Model saved to ==> {save_path}')


def load_checkpoint(model, optimizer):
    save_path = f'net.pt'
    state_dict = torch.load(save_path)
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    print(f'Model loaded from <== {save_path}')

    return val_loss, train_loss_his, val_loss_his