import os
import sys
import json
import argparse
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from pathlib import Path
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from shutil import copytree
from scipy.stats import mode

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from lib.solver import Solver
from lib.dataset import ScannetDataset, ScannetDatasetActiveLearning, ScannetDatasetWholeScene, collate_random, collate_wholescene
from lib.loss import WeightedCrossEntropyLoss
from lib.config import CONF
from lib.inference_utils import mc_forward, prep_visualization, filter_points
from eval import filter_points, eval_wholescene


def get_dataloader(args, scene_list, phase):
    if args.use_wholescene:
        dataset = ScannetDatasetWholeScene(scene_list, is_weighting=not args.no_weighting, use_color=args.use_color, use_normal=args.use_normal, use_multiview=args.use_multiview)
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_wholescene, num_workers=args.num_workers, pin_memory=True)
    else:
        dataset = ScannetDatasetActiveLearning(phase, scene_list, is_weighting=not args.no_weighting, use_color=args.use_color, use_normal=args.use_normal, use_multiview=args.use_multiview)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_random, num_workers=args.num_workers, pin_memory=True)

    return dataset, dataloader

def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params

def get_solver(args, dataset, dataloader, stamp, weight):
    model = get_model(args, stamp)
    num_params = get_num_params(model)
    criterion = WeightedCrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    solver = Solver(model, dataset, dataloader, criterion, optimizer, args.batch_size, stamp, args.use_wholescene, args.ds, args.df)

    return solver, num_params

def get_model(args, stamp, mc_drop=False):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../pointnet2/'))
    Pointnet = importlib.import_module("pointnet2_semseg")
    input_channels = int(args.use_color) * 3 + int(args.use_normal) * 3 + int(args.use_multiview) * 128
    model = Pointnet.get_model(num_classes=CONF.NUM_CLASSES, is_msg=args.use_msg, input_channels=input_channels, use_xyz=not args.no_xyz, bn=not args.no_bn, mc_drop=mc_drop).cuda()

    return model

def get_scene_list(path):
    scene_list = []
    with open(path) as f:
        for scene_id in f.readlines():
            scene_list.append(scene_id.strip())

    scene_list = sorted(scene_list, key=lambda x: int(x.split("_")[0][5:]))

    return scene_list

def save_info(args, root, train_examples, val_examples, num_params):
    info = {}
    for key, value in vars(args).items():
        info[key] = value
    
    info["num_train"] = train_examples
    info["num_val"] = val_examples
    info["num_params"] = num_params

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)


def choose_uncertain_scenes(args, scene_list, stamp, num_scenes=100, use_gt=False):
    model_path = os.path.join(args.output, stamp, "model.pth")
    Pointnet = importlib.import_module("pointnet2_semseg")
    input_channels = int(args.use_color) * 3 + int(args.use_normal) * 3 + int(args.use_multiview) * 128
    model = Pointnet.get_model(num_classes=CONF.NUM_CLASSES, is_msg=args.use_msg, input_channels=input_channels, use_xyz=not args.no_xyz, bn=not args.no_bn, mc_drop=not use_gt).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    scene_uncertainties = {}
    for scene in scene_list:
        dataset = ScannetDatasetWholeScene([scene], use_color=args.use_color, use_normal=args.use_normal, use_multiview=args.use_multiview)
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_wholescene)
        if use_gt:
            _, _, _, _, _, pointmiou_per_class_array, _, masks = eval_wholescene(args, model, dataloader)
            avg_pointmiou_per_class = np.sum(pointmiou_per_class_array * masks, axis=0)/np.sum(masks, axis=0)
            scene_entropy = np.sum(avg_pointmiou_per_class[masks.squeeze().astype(bool)]) / np.sum(masks.squeeze())
        else:
            data = next(iter(dataloader))
            coords, feats, targets, weights, _ = data
            coords, feats, targets, weights = coords.cuda(), feats.cuda(), targets.cuda(), weights.cuda()
            preds = mc_forward(args.batch_size, model, coords, feats)
            _, _, _, MC = preds.shape
            coords = coords.squeeze(0).view(-1, 3).cpu().numpy()
            preds = preds.view(-1, MC).cpu().numpy()
            targets = targets.squeeze(0).view(-1).cpu().numpy()
            weights = weights.squeeze(0).view(-1).cpu().numpy()
            coords, preds, targets, weights = filter_points(coords, preds, targets, weights)
            scene_entropy = np.zeros(preds.shape[0])
            for c in range(CONF.NUM_CLASSES):
                p = np.sum(preds == c, axis=1, dtype=np.float32) / MC
                scene_entropy = scene_entropy - (p * np.log2(p + 1e-12))
            mc_preds = mode(preds, axis=1)[0].squeeze()
            class_entropies = np.zeros(CONF.NUM_CLASSES)
            for c in range(CONF.NUM_CLASSES):
                class_entropies[c] = np.mean(scene_entropy[targets == c])
            scene_entropy = np.sum(class_entropies[~np.isnan(class_entropies)]) / np.sum(~np.isnan(class_entropies))
        scene_uncertainties[scene] = scene_entropy
    
    print(list(OrderedDict(sorted(scene_uncertainties.items(), key=lambda x: x[1], reverse=not use_gt)).items())[:num_scenes])

    return list(OrderedDict(sorted(scene_uncertainties.items(), key=lambda x: x[1], reverse=not use_gt)).keys())[:num_scenes]

def train(args, train_dataset, val_dataset, stamp):
    # init training dataset
    print("preparing data...")
    
    # dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_random, num_workers=args.num_workers, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_random, num_workers=args.num_workers, pin_memory=True)

    dataset = {
        "train": train_dataset,
        "val": val_dataset
    }
    dataloader = {
        "train": train_dataloader,
        "val": val_dataloader
    }
    weight = train_dataset.labelweights
    train_examples = len(train_dataset)
    val_examples = len(val_dataset)

    print("initializing...")
    #stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    #if args.tag: stamp += "_"+args.tag.upper()
    root = os.path.join(args.output, stamp)
    os.makedirs(root, exist_ok=True)
    solver, num_params = get_solver(args, dataset, dataloader, stamp, weight)
    
    print("\n[info]")
    print("Train examples: {}".format(train_examples))
    print("Evaluation examples: {}".format(val_examples))
    print("Start training...\n")
    save_info(args, root, train_examples, val_examples, num_params)
    solver(args.epoch, args.verbose)

def evaluate_pixel_miou(args, stamp):
    # prepare data
    print("preparing data...")
    scene_list = get_scene_list("data/scannetv2_100_test.txt")
    dataset = ScannetDatasetWholeScene(scene_list, use_color=args.use_color, use_normal=args.use_normal, use_multiview=args.use_multiview)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_wholescene)

    # load model
    print("loading model...")
    model_path = os.path.join(args.output, stamp, "model.pth")
    Pointnet = importlib.import_module("pointnet2_semseg")
    input_channels = int(args.use_color) * 3 + int(args.use_normal) * 3 + int(args.use_multiview) * 128
    model = Pointnet.get_model(num_classes=CONF.NUM_CLASSES, is_msg=args.use_msg, input_channels=input_channels, use_xyz=not args.no_xyz, bn=not args.no_bn).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # eval
    print("evaluating...")
    _, _, _, _, _, pointmiou_per_class_array, _, masks = eval_wholescene(args, model, dataloader)
    
    avg_pointmiou_per_class = np.sum(pointmiou_per_class_array * masks, axis=0)/np.sum(masks, axis=0)
    avg_pointmiou = np.mean(avg_pointmiou_per_class)

    return avg_pointmiou

def run_experiments(args):
    rand_gen = np.random.default_rng(args.seed)
    all_scenes_pool = get_scene_list(CONF.SCANNETV2_TRAIN)
    val_scene_list = rand_gen.choice(all_scenes_pool, size=20, replace=False)
    train_scenes_pool = list(set(all_scenes_pool).difference(set(val_scene_list)))
    print("Train Pool is of size {}".format(len(train_scenes_pool)))
    initial_scenes_list = list(rand_gen.choice(train_scenes_pool, size=args.num_scenes, replace=False))
    random_dataset = ScannetDatasetActiveLearning("train", initial_scenes_list, npoints=args.n_points_train, is_weighting=not args.no_weighting, use_color=args.use_color, use_normal=args.use_normal, use_multiview=args.use_multiview, points_increment=args.points_increment)
    mc_dataset = deepcopy(random_dataset)
    gt_dataset = deepcopy(random_dataset)
    val_dataset = ScannetDataset("val", initial_scenes_list, is_weighting=not args.no_weighting, use_color=args.use_color, use_normal=args.use_normal, use_multiview=args.use_multiview)
    out_dir = Path(args.output)
    experiments = list(out_dir.glob('experiment*'))
    if len(experiments) == 0:
        experiment_prefix = "experiment_0"
    else:
        max_id = max(experiments, key=lambda x: int(x.name.split("_")[1]))
        experiment_prefix = "experiment_" + str(int(max_id.name.split("_")[1]) + 1)
    experiment = experiment_prefix + "_rand_0"
    train(args, random_dataset, val_dataset, experiment)
    rand_0_pixel_miou = evaluate_pixel_miou(args, experiment)
    copytree(os.path.join(args.output, experiment), os.path.join(args.output, experiment_prefix + "_mc_0"))
    copytree(os.path.join(args.output, experiment), os.path.join(args.output, experiment_prefix + "_gt_0"))
    tb_path = os.path.join(args.output, experiment_prefix, "tensorboard")
    os.makedirs(tb_path, exist_ok=True)
    logger = SummaryWriter(tb_path)
    logger.add_scalars(
        "eval/{}".format("point_miou_active"),
        {
            "random": rand_0_pixel_miou,
            "mc_dropout": rand_0_pixel_miou,
            "gt_choose": rand_0_pixel_miou
        },
        args.points_increment
    )
    for i in range(1, args.n_active_iters):
        #pointcloud visualization
        data = prep_visualization(mc_dataset, gt_dataset, random_dataset)
        for scene_id in random_dataset.scene_list:
            mc_scene_data = mc_dataset.scene_data[scene_id][mc_dataset.selected_mask[scene_id]]
            gt_scene_data = gt_dataset.scene_data[scene_id][gt_dataset.selected_mask[scene_id]]
            random_scene_data = random_dataset.scene_data[scene_id][random_dataset.selected_mask[scene_id]]
            mc_filtered, _, _, _ = filter_points(mc_scene_data, mc_scene_data, mc_scene_data, mc_scene_data)
            random_filtered, _, _, _ = filter_points(random_scene_data, random_scene_data, random_scene_data, random_scene_data)
            gt_filtered, _, _, _ = filter_points(gt_scene_data, gt_scene_data, gt_scene_data, gt_scene_data)
            print("GT: ", gt_scene_data.shape, gt_filtered.shape)
            print("MC: ", mc_scene_data.shape, mc_filtered.shape)
            print("RANDOM: ", random_scene_data.shape, random_filtered.shape)
            
        for scene_id, (vertex, colors) in data.items():
            vertex = torch.as_tensor(vertex, dtype=torch.float).unsqueeze(0)
            colors = torch.as_tensor(colors, dtype=torch.uint8).unsqueeze(0)
            logger.add_mesh(
                '{}/{}'.format(experiment_prefix, scene_id), 
                vertex, 
                colors, 
                config_dict={
                    "camera": {"cls": "PerspectiveCamera", "fov": 75}, 
                    "material": {"cls": "MeshBasicMaterial", "reflectivity": 1}
                }, 
                global_step=i * args.points_increment
            )
        #Perform MC active learning
        prev_mc_experiment = experiment_prefix + "_mc_" + str(i - 1)
        model = get_model(args, prev_mc_experiment, mc_drop=True)
        model.eval()
        mc_dataset.choose_new_points(heuristic="mc", model=model)
        experiment = experiment_prefix + "_mc_" + str(i)
        train(args, mc_dataset, val_dataset, experiment)
        mc_pixel_miou = evaluate_pixel_miou(args, experiment)

        #Perform gt active learning
        prev_gt_experiment = experiment_prefix + "_gt_" + str(i - 1)
        model = get_model(args, prev_gt_experiment, mc_drop=False)
        model.eval()
        gt_dataset.choose_new_points(heuristic="gt", model=model)
        experiment = experiment_prefix + "_gt_" + str(i)
        train(args, gt_dataset, val_dataset, experiment)
        gt_pixel_miou = evaluate_pixel_miou(args, experiment)

        #perform random choice
        prev_rand_experiment = experiment_prefix + "_rand_" + str(i - 1)
        model = get_model(args, prev_rand_experiment, mc_drop=False)
        model.eval()
        random_dataset.choose_new_points(heuristic="random")
        experiment = experiment_prefix + "_rand_" + str(i)
        train(args, random_dataset, val_dataset, experiment)
        random_pixel_miou = evaluate_pixel_miou(args, experiment)
        logger.add_scalars(
            "eval/{}".format("point_miou_active"),
            {
                "random": random_pixel_miou,
                "mc_dropout": mc_pixel_miou,
                "gt_choose": gt_pixel_miou
            },
            (i + 1) * args.points_increment
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, help='Output directory', default='active_outputs_pointwise')
    parser.add_argument('--gpu', type=str, help='gpu', default='0')
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--points_increment', type=int, help='points to add in each active cycle', default=100)
    parser.add_argument('--n_active_iters', type=int, help='number of active learning cycles', default=10)
    parser.add_argument('--n_points_train', type=int, help='number of active learning cycles', default=8192)
    parser.add_argument('--seed', type=int, help='Random Seed', default=1311)
    parser.add_argument('--epoch', type=int, help='number of epochs', default=500)
    parser.add_argument('--verbose', type=int, help='iterations of showing verbose', default=10)
    parser.add_argument('--num_workers', type=int, help='number of workers in dataloader', default=0)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-3)
    parser.add_argument('--wd', type=float, help='weight decay', default=0)
    parser.add_argument('--ds', type=int, help='decay step', default=100)
    parser.add_argument('--df', type=float, help='decay factor', default=0.7)
    parser.add_argument('--num_scenes', type=int, help='number of scenes in each step', default=30)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_weighting", action="store_true", help="weight the classes")
    parser.add_argument('--no_bn', action="store_true", help="do not apply batch normalization in pointnet++")
    parser.add_argument('--no_xyz', action="store_true", help="do not apply coordinates as features in pointnet++")
    parser.add_argument("--use_wholescene", action="store_true", help="on the whole scene or on a random chunk")
    parser.add_argument("--use_msg", action="store_true", help="apply multiscale grouping or not")
    parser.add_argument("--use_color", action="store_true", help="use color values or not")
    parser.add_argument("--use_normal", action="store_true", help="use normals or not")
    parser.add_argument("--use_multiview", action="store_true", help="use multiview image features or not")
    args = parser.parse_args()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    run_experiments(args)
