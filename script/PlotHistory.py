import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch import load

parser = argparse.ArgumentParser(description = "Plot")
parser.add_argument('--filename',           type=str, help='pt file path')
params = parser.parse_args()

def main():
    filename = params.filename
    state_dict = load(filename)
    plot_dict = state_dict['plot_dict']

    reward = plot_dict['reward']
    loss = plot_dict['loss']
    fig = plt.figure()
    ax = fig.add_subplot(211)
    plt.plot(np.arange(1, len(reward)+1), reward)
    plt.ylabel('Reward')
    plt.xlabel('Episode #')
    
    ax = fig.add_subplot(212)
    plt.plot(np.arange(1, len(loss)+1), loss)
    plt.ylabel('Loss')
    plt.xlabel('Episode #')
    plt.show()

if __name__ == '__main__':
    main()