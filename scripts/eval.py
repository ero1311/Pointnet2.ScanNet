import os
import argparse
import importlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# for PointNet2.PyTorch module
import sys
sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../pointnet2/'))
from lib.config import CONF
from lib.dataset import ScannetDatasetWholeScene, collate_wholescene
from lib.inference_utils import forward, filter_points, eval_one_batch

def get_scene_list(path):
    scene_list = []
    with open(path) as f:
        for scene_id in f.readlines():
            scene_list.append(scene_id.strip())

    scene_list = sorted(scene_list, key=lambda x: int(x.split("_")[0][5:]))

    return scene_list


def eval_wholescene(args, model, dataloader):
    # init
    pointacc_list = []
    pointacc_per_class_array = np.zeros((len(dataloader), CONF.NUM_CLASSES))
    voxacc_list = []
    voxacc_per_class_array = np.zeros((len(dataloader), CONF.NUM_CLASSES))
    voxcaliacc_list = []
    pointmiou_per_class_array = np.zeros((len(dataloader), CONF.NUM_CLASSES))
    voxmiou_per_class_array = np.zeros((len(dataloader), CONF.NUM_CLASSES))
    masks = np.zeros((len(dataloader), CONF.NUM_CLASSES))

    # iter
    for load_idx, data in enumerate(tqdm(dataloader)):
        # feed
        pointacc, pointacc_per_class, voxacc, voxacc_per_class, voxcaliacc, pointmiou, voxmiou, mask = eval_one_batch(args.batch_size, model, data)

        # dump
        pointacc_list.append(pointacc)
        pointacc_per_class_array[load_idx] = pointacc_per_class
        voxacc_list.append(voxacc)
        voxacc_per_class_array[load_idx] = voxacc_per_class
        voxcaliacc_list.append(voxcaliacc)
        pointmiou_per_class_array[load_idx] = pointmiou
        voxmiou_per_class_array[load_idx] = voxmiou
        masks[load_idx] = mask

    return pointacc_list, pointacc_per_class_array, voxacc_list, voxacc_per_class_array, voxcaliacc_list, pointmiou_per_class_array, voxmiou_per_class_array, masks

def evaluate(args):
    # prepare data
    print("preparing data...")
    scene_list = get_scene_list("data/scannetv2_100_test.txt")
    dataset = ScannetDatasetWholeScene(scene_list, use_color=args.use_color, use_normal=args.use_normal, use_multiview=args.use_multiview)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_wholescene)

    # load model
    print("loading model...")
    model_path = os.path.join(CONF.OUTPUT_ROOT, args.folder, "model.pth")
    Pointnet = importlib.import_module("pointnet2_semseg")
    input_channels = int(args.use_color) * 3 + int(args.use_normal) * 3 + int(args.use_multiview) * 128
    model = Pointnet.get_model(num_classes=CONF.NUM_CLASSES, is_msg=args.use_msg, input_channels=input_channels, use_xyz=not args.no_xyz, bn=not args.no_bn).cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # eval
    print("evaluating...")
    pointacc_list, pointacc_per_class_array, voxacc_list, voxacc_per_class_array, voxcaliacc_list, pointmiou_per_class_array, voxmiou_per_class_array, masks = eval_wholescene(args, model, dataloader)
    
    avg_pointacc = np.mean(pointacc_list)
    avg_pointacc_per_class = np.sum(pointacc_per_class_array * masks, axis=0)/np.sum(masks, axis=0)

    avg_voxacc = np.mean(voxacc_list)
    avg_voxacc_per_class = np.sum(voxacc_per_class_array * masks, axis=0)/np.sum(masks, axis=0)

    avg_voxcaliacc = np.mean(voxcaliacc_list)
    
    avg_pointmiou_per_class = np.sum(pointmiou_per_class_array * masks, axis=0)/np.sum(masks, axis=0)
    avg_pointmiou = np.mean(avg_pointmiou_per_class)

    avg_voxmiou_per_class = np.sum(voxmiou_per_class_array * masks, axis=0)/np.sum(masks, axis=0)
    avg_voxmiou = np.mean(avg_voxmiou_per_class)

    # report
    print()
    print("Point accuracy: {}".format(avg_pointacc))
    print("Point accuracy per class: {}".format(np.mean(avg_pointacc_per_class)))
    print("Voxel accuracy: {}".format(avg_voxacc))
    print("Voxel accuracy per class: {}".format(np.mean(avg_voxacc_per_class)))
    print("Calibrated voxel accuracy: {}".format(avg_voxcaliacc))
    print("Point miou: {}".format(avg_pointmiou))
    print("Voxel miou: {}".format(avg_voxmiou))
    print()

    print("Point acc/voxel acc/point miou/voxel miou per class:")
    for l in range(CONF.NUM_CLASSES):
        print("Class {}: {}/{}/{}/{}".format(CONF.NYUCLASSES[l], avg_pointacc_per_class[l], avg_voxacc_per_class[l], avg_pointmiou_per_class[l], avg_voxmiou_per_class[l]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, help='output folder containing the best model from training', required=True)
    parser.add_argument('--batch_size', type=int, help='size of the batch/chunk', default=32)
    parser.add_argument('--gpu', type=str, help='gpu', default='0')
    parser.add_argument('--no_bn', action="store_true", help="do not apply batch normalization in pointnet++")
    parser.add_argument('--no_xyz', action="store_true", help="do not apply coordinates as features in pointnet++")
    parser.add_argument("--use_msg", action="store_true", help="apply multiscale grouping or not")
    parser.add_argument("--use_color", action="store_true", help="use color values or not")
    parser.add_argument("--use_normal", action="store_true", help="use normals or not")
    parser.add_argument("--use_multiview", action="store_true", help="use multiview image features or not")
    args = parser.parse_args()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    evaluate(args)
