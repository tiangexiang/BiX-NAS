import argparse
import numpy as np
from pytorch_version import search

def argparsing():
    parser = argparse.ArgumentParser(description='BiX-NAS Phase2 searching')
    parser.add_argument('--epochs', default=40, type=int, help='trining epochs')
    parser.add_argument('--batch_size', default=2, type=int, help='batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=3e-5, type=float, help='learning rate decay')
    parser.add_argument('--num_class', default=1, type=int, help='model output channel number')
    parser.add_argument('--multiplier', default=1.0, type=float, help='parameter multiplier')
    parser.add_argument('--iter', default=3, type=int, help='recurrent iteration')
    parser.add_argument('--train_data', default='./data/train', type=str, help='data path')
    parser.add_argument('--valid_data', default='./data/valid', type=str, help='data path')
    parser.add_argument('--exp', default='1', type=str, help='experiment number')
    parser.add_argument('--gene', default=None, type=str, help='searched gene file')
    parser.add_argument('--backend', default='pytorch', choices=['pytorch'], type=str, help='support pytorch only')
    parser.add_argument('--random_num', default=15, type=int, help='num random')
    parser.add_argument('--save_gene', default=None, type=str, help='path to save genes')
    parser.add_argument('--valid_dataset', default='monuseg', choices=['monuseg', 'tnbc'], type=str, help='which dataset to validate?')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = argparsing()
    #codes = decode(args.gene, args.iter, 4)
    search(args)
