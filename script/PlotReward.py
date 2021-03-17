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
    rewards_history = state_dict['rewards_history']


    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(rewards_history)+1), rewards_history)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    

if __name__ == '__main__':
    main()