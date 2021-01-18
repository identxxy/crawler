import sys, time, os, argparse, socket
import torch
import glob
from Training import *
from ACNet import acNet, save_checkpoint, load_checkpoint



#config param
parser = argparse.ArgumentParser(description = "crawlerD");

## Data loader
parser.add_argument('--max_frames',     type=int,   default=200,    help='Input length to the network for training');
parser.add_argument('--batch_size',     type=int,   default=200,    help='Batch size, number of speakers per batch');
parser.add_argument('--fps', type=int,  default=500,    help='fps');
parser.add_argument('--nDataLoaderThread', type=int, default=5,     help='Number of loader threads');
parser.add_argument('--augment',        type=bool,  default=False,  help='Augment input')

## Training details
parser.add_argument('--test_interval',  type=int,   default=10,     help='Test and save every [test_interval] epochs');
parser.add_argument('--max_epoch',      type=int,   default=500,    help='Maximum number of epochs');
parser.add_argument('--trainfunc',      type=str,   default="",     help='Loss function');

## Optimizer
parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam');
parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler');
parser.add_argument('--lr',             type=float, default=0.001,  help='Learning rate');
parser.add_argument("--lr_decay",       type=float, default=0.95,   help='Learning rate decay every [test_interval] epochs');
parser.add_argument('--weight_decay',   type=float, default=0,      help='Weight decay in the optimizer');

## Loss functions

## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights');
parser.add_argument('--save_path',      type=str,   default="exps/exp1", help='Path for model and logs');

## Training and test data
parser.add_argument('--train_list',     type=str,   default="data/train_list.txt",  help='Train list');
parser.add_argument('--test_list',      type=str,   default="data/test_list.txt",   help='Evaluation list');
parser.add_argument('--train_path',     type=str,   default="data/voxceleb2", help='Absolute path to the train set');
parser.add_argument('--test_path',      type=str,   default="data/voxceleb1", help='Absolute path to the test set');
parser.add_argument('--musan_path',     type=str,   default="data/musan_split", help='Absolute path to the test set');
parser.add_argument('--rir_path',       type=str,   default="data/RIRS_NOISES/simulated_rirs", help='Absolute path to the test set');

## Model definition
parser.add_argument('--model',          type=str,   default="",     help='Name of model definition');
parser.add_argument('--inputsize',         type=int,   default=61,     help='inputsize');
parser.add_argument('--hiddensize1',      type=int,  default=48,  help='Embedding size in the first FC layer')
parser.add_argument('--hiddensize2',   type=int,   default=80,  help='Embedding size in the LSTM layer');
parser.add_argument('--outputsize',           type=int,   default=12,    help='outputsize');

## For test only
parser.add_argument('--mode',           dest=str, help='train test demo')

args = parser.parse_args();

def main():
    Net = acNet()

    #train mode
    if args.mode == 'train':
        args.model_save_path = args.save_path + "/model"
        args.result_save_path = args.save_path + "/result"
        args.feat_save_path = ""

        if not (os.path.exists(args.model_save_path)):
            os.makedirs(args.model_save_path)

        if not (os.path.exists(args.result_save_path)):
            os.makedirs(args.result_save_path)

        n_gpus = torch.cuda.device_count()

        print('Python Version:', sys.version)
        print('PyTorch Version:', torch.__version__)
        print('Number of GPUs:', torch.cuda.device_count())
        print('Save path:', args.save_path)

        train(Net,**vars(args))


    #test mode
    elif args.mode == 'test':
        return

    #demo mode
    elif args.mode == 'demo':
        return

    else:
        quit()


if __name__ == '__main__':
    main()
