import argparse
import numpy as np

parser = argparse.ArgumentParser(description='DRN')

parser.add_argument('--n_threads', type=int, default=6,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--data_train', type=str, default='NYUv2',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='NYUv2',
                    help='test dataset name')
parser.add_argument('--scale', type=int, default=16,
                    help='super resolution scale')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=1,
                    help='number of color channels to use')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')
parser.add_argument('--n_blocks', type=int, default=16,
                    help='number of residual blocks, 16|30|40|80')
parser.add_argument('--n_feats', type=int, default=16,
                    help='number of feature maps')
parser.add_argument('--negval', type=float, default=0.2, 
                    help='Negative value parameter for Leaky ReLU')



args = parser.parse_args()


# scale = [2,4] for 4x SR to load data
# scale = [2,4,8] for 8x SR to load data
args.scale = [pow(2, s+1) for s in range(int(np.log2(args.scale)))]


for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

