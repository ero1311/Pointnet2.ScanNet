import os
import sys
import json
import argparse
import importlib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from scipy.stats import mode
from tensorboardX import SummaryWriter
import plotly.express as px
sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from lib.solver import Solver
from lib.dataset import ScannetDataset, ScannetDatasetWholeScene, collate_random, collate_wholescene
from lib.loss import WeightedCrossEntropyLoss
from lib.config import CONF
from eval import filter_points, eval_wholescene


def get_dataloader(args, scene_list, phase):
    if args.use_wholescene:
        dataset = ScannetDatasetWholeScene(scene_list, is_weighting=not args.no_weighting, use_color=args.use_color, use_normal=args.use_normal, use_multiview=args.use_multiview)
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_wholescene, num_workers=args.num_workers, pin_memory=True)
    else:
        dataset = ScannetDataset(phase, scene_list, is_weighting=not args.no_weighting, use_color=args.use_color, use_normal=args.use_normal, use_multiview=args.use_multiview)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_random, num_workers=args.num_workers, pin_memory=True)

    return dataset, dataloader

def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params

def get_solver(args, dataset, dataloader, stamp, weight):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../pointnet2/'))
    Pointnet = importlib.import_module("pointnet2_semseg")
    input_channels = int(args.use_color) * 3 + int(args.use_normal) * 3 + int(args.use_multiview) * 128
    model = Pointnet.get_model(num_classes=CONF.NUM_CLASSES, is_msg=args.use_msg, input_channels=input_channels, use_xyz=not args.no_xyz, bn=not args.no_bn).cuda()

    num_params = get_num_params(model)
    criterion = WeightedCrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    solver = Solver(model, dataset, dataloader, criterion, optimizer, args.batch_size, stamp, args.use_wholescene, args.ds, args.df)

    return solver, num_params

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

def mc_forward(args, model, coords, feats, mc_iters=20):
    pred = []
    coord_chunk, feat_chunk = torch.split(coords.squeeze(0), args.batch_size, 0), torch.split(feats.squeeze(0), args.batch_size, 0)
    assert len(coord_chunk) == len(feat_chunk)
    for coord, feat in zip(coord_chunk, feat_chunk):
        b_size, n_points, _ = coord.shape
        outputs = torch.cuda.FloatTensor(b_size, n_points, mc_iters)
        with torch.no_grad():
            for i in range(mc_iters):
                outputs[:, :, i] = model(torch.cat([coord, feat], dim=2)).max(-1)[1]
        pred.append(outputs.detach().cpu())

    pred = torch.cat(pred, dim=0).unsqueeze(0) # (1, CK, N, C)

    return pred


def compute_uncertainties(args, stamp, num_scenes=100):
    scene_list = get_scene_list("data/scannetv2_100_test.txt")
    model_path = os.path.join(args.output, stamp, "model.pth")
    Pointnet = importlib.import_module("pointnet2_semseg")
    input_channels = int(args.use_color) * 3 + int(args.use_normal) * 3 + int(args.use_multiview) * 128
    model = Pointnet.get_model(num_classes=CONF.NUM_CLASSES, is_msg=args.use_msg, input_channels=input_channels, use_xyz=not args.no_xyz, bn=not args.no_bn, mc_drop=True).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    tb_path = os.path.join(args.output, stamp, "tensorboard")
    #logger = SummaryWriter(tb_path)

    scene_uncertainties = []
    for i, scene in enumerate(scene_list):
        dataset = ScannetDatasetWholeScene([scene], use_color=args.use_color, use_normal=args.use_normal, use_multiview=args.use_multiview)
        dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_wholescene)
        data = next(iter(dataloader))
        coords, feats, targets, weights, _ = data
        coords, feats, targets, weights = coords.cuda(), feats.cuda(), targets.cuda(), weights.cuda()
        preds = mc_forward(args, model, coords, feats)
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
        #scene_entropy = np.var(preds, axis=1)
        
        #logger.add_histogram('scene_uncertainties_drop_2_025', scene_entropy, i)
        mc_preds = mode(preds, axis=1)[0].squeeze()
        class_entropies = np.zeros(CONF.NUM_CLASSES)
        for c in range(CONF.NUM_CLASSES):
            class_entropies[c] = np.mean(scene_entropy[targets == c])
        scene_entropy = np.sum(class_entropies[~np.isnan(class_entropies)]) / np.sum(~np.isnan(class_entropies))
        scene_entropy = np.mean(scene_entropy)
        scene_uncertainties.append(scene_entropy)
    
    #logger.close()
    
    return np.array(scene_uncertainties)


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
    avg_pointmiou_per_scene = np.sum(pointmiou_per_class_array * masks, axis=1)/np.sum(masks, axis=1)
    avg_pointmiou = np.mean(avg_pointmiou_per_class)

    return avg_pointmiou_per_scene, avg_pointmiou

def build_histograms(args):
    experiment_prefix = args.experiment
    experiment_prefix += "_mc_"
    uncertainties = []
    mc_per_scene_pixel_miou = []
    models_used = []

    for i in range(6):
        experiment = experiment_prefix + str(i)
        curr_uncertainties = compute_uncertainties(args, experiment)
        curr_mc_per_scene_pixel_miou, mc_pixel_miou = evaluate_pixel_miou(args, experiment)
        curr_model = 'model_mIoU_{}'.format(round(mc_pixel_miou, 3))
        min_unc, max_unc = min(curr_uncertainties), max(curr_uncertainties)
        for j in range(args.nbins):
            start = min_unc + j * (max_unc - min_unc) / args.nbins
            end = start + (max_unc - min_unc) / args.nbins
            uncertainties.append(start)
            mc_per_scene_pixel_miou.append(np.mean(curr_mc_per_scene_pixel_miou[np.where((curr_uncertainties > start) & (curr_uncertainties < end))]))
        models_used.extend(args.nbins * [curr_model])


    data = {'uncertainty': uncertainties, 'pixel_miou': mc_per_scene_pixel_miou, 'model': models_used}
    fig = px.bar(data_frame=data, x='uncertainty', y='pixel_miou', facet_col='model')
    fig.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, help='Output directory', default='active_outputs')
    parser.add_argument('--gpu', type=str, help='gpu', default='0')
    parser.add_argument('--experiment', type=str, help='Experiment to use', default='experiment_0')
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--nbins', type=int, help='batch size', default=10)
    parser.add_argument('--verbose', type=int, help='iterations of showing verbose', default=10)
    parser.add_argument('--num_workers', type=int, help='number of workers in dataloader', default=0)
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

    build_histograms(args)
