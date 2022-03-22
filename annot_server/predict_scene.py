import os
import sys
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

from lib.pc_util import read_ply_xyzrgbnormal
from lib.inference_utils import forward

def get_scene_labels(file_path, idx_to_label, model, point_classes):
    scene_data = read_ply_xyzrgbnormal(file_path)
    coordmax = scene_data[:, :3].max(axis=0)
    coordmin = scene_data[:, :3].min(axis=0)
    xlength = 1.5
    ylength = 1.5
    nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/xlength).astype(np.int32)
    nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/ylength).astype(np.int32)

    point_set_ini = scene_data[:, :3]
    color = scene_data[:, 3:6] / 255.
    normal = scene_data[:, 6:9]
    point_set_ini = np.concatenate([point_set_ini, color], axis=1)
    point_set_ini = np.concatenate([point_set_ini, normal], axis=1)
    npoints = 8192
    point_labels = np.zeros(scene_data.shape[0])
    print("tpuma")
    for i in range(nsubvolume_x):
        for j in range(nsubvolume_y):
            curmin = coordmin+[i*xlength, j*ylength, 0]
            curmax = coordmin+[(i+1)*xlength, (j+1)*ylength, coordmax[2]-coordmin[2]]
            mask = np.sum((point_set_ini[:, :3]>=(curmin-0.01))*(point_set_ini[:, :3]<=(curmax+0.01)), axis=1)==3
            cur_point_set = point_set_ini[mask,:]
            block_labels = np.zeros(cur_point_set.shape[0])
            if len(cur_point_set) == 0:
                continue

            remainder = cur_point_set.shape[0] % npoints
            choice = np.random.choice(cur_point_set.shape[0], size = npoints - remainder, replace=True)
            cur_point_set = np.concatenate([cur_point_set, cur_point_set[choice].copy()], axis=0)
            cur_point_sets = np.split(cur_point_set, cur_point_set.shape[0] // npoints, axis=0)
            for k in range(len(cur_point_sets)):
                point_set = cur_point_sets[k] # Nx3
                mask_start, mask_end = k * npoints, (k + 1) * npoints
                with torch.no_grad():
                    point_set = np.expand_dims(point_set, 0) # 1xNx3
                    point_set_t = torch.FloatTensor(point_set)
                    coords = point_set_t[:, :, :3]
                    feats = point_set_t[:, :, 3:]
                    coords, feats = coords.unsqueeze(0), feats.unsqueeze(0)
                    coords, feats = coords.cuda(), feats.cuda()
                    preds = forward(1, model, coords, feats)
                    preds = preds.squeeze(0).view(-1).cpu().numpy()
                    preds = preds[:len(mask[mask==True][mask_start:mask_end])]
                    block_labels[mask_start:mask_end] = preds.copy()
            point_labels[mask] = block_labels.copy()

    pre_annots = {}
    for seg_id in idx_to_label:
        class_name = point_classes[int(point_labels[int(seg_id)])]
        if class_name not in pre_annots:
            pre_annots[class_name] = []
        pre_annots[class_name].append(int(seg_id))
    
    return pre_annots

