import os
from os.path import join
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Add scene names to training list')
parser.add_argument('--num_scenes', type=int, default=80, help='number of the scenes to add')

args = parser.parse_args()

train_scenes_path = '/home/erik/TUM/Pointnet2.ScanNet/data/scannetv2_100_train.txt'
all_train_scenes_path = '/home/erik/TUM/Pointnet2.ScanNet/data/scannetv2_train.txt'
with open(train_scenes_path) as f:
    train_scenes = f.readlines()

with open(all_train_scenes_path) as f:
    all_train_scenes = f.readlines()

train_scenes = [scene.strip() for scene in train_scenes]
all_train_scenes = [scene.strip() for scene in all_train_scenes]
remaining_scenes = list(set(all_train_scenes).difference(set(train_scenes)))
new_scenes = np.random.choice(remaining_scenes, size=args.num_scenes, replace=False)

with open(train_scenes_path, 'a') as f:
    for scene in new_scenes:
        f.write(scene + '\n')
