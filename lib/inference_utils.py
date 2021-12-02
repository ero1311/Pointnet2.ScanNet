import numpy as np
import sys
import torch
from lib.pc_util import point_cloud_label_to_surface_voxel_label_fast
sys.path.append(".")
from lib.config import CONF

def forward(batch_size, model, coords, feats):
    pred = []
    coord_chunk, feat_chunk = torch.split(coords.squeeze(0), batch_size, 0), torch.split(feats.squeeze(0), batch_size, 0)
    assert len(coord_chunk) == len(feat_chunk)
    for coord, feat in zip(coord_chunk, feat_chunk):
        output = model(torch.cat([coord, feat], dim=2))
        pred.append(output)

    pred = torch.cat(pred, dim=0).unsqueeze(0) # (1, CK, N, C)
    outputs = pred.max(3)[1]

    return outputs

def mc_forward(batch_size, model, coords, feats, mc_iters=20):
    pred = []
    coord_chunk, feat_chunk = torch.split(coords.squeeze(0), batch_size, 0), torch.split(feats.squeeze(0), batch_size, 0)
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

def filter_points(coords, preds, targets, weights):
    assert coords.shape[0] == preds.shape[0] == targets.shape[0] == weights.shape[0]
    coord_hash = [hash(str(coords[point_idx][0]) + str(coords[point_idx][1]) + str(coords[point_idx][2])) for point_idx in range(coords.shape[0])]
    _, coord_ids = np.unique(np.array(coord_hash), return_index=True)
    coord_filtered, pred_filtered, target_filtered, weight_filtered = coords[coord_ids], preds[coord_ids], targets[coord_ids], weights[coord_ids]

    return coord_filtered, pred_filtered, target_filtered, weight_filtered


def compute_acc(coords, preds, targets, weights):
    coords, preds, targets, weights = filter_points(coords, preds, targets, weights)
    seen_classes = np.unique(targets)
    mask = np.zeros(CONF.NUM_CLASSES)
    mask[seen_classes] = 1

    total_correct = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(CONF.NUM_CLASSES)]
    total_correct_class = [0 for _ in range(CONF.NUM_CLASSES)]

    total_correct_vox = 0
    total_seen_vox = 0
    total_seen_class_vox = [0 for _ in range(CONF.NUM_CLASSES)]
    total_correct_class_vox = [0 for _ in range(CONF.NUM_CLASSES)]

    labelweights = np.zeros(CONF.NUM_CLASSES)
    labelweights_vox = np.zeros(CONF.NUM_CLASSES)

    correct = np.sum(preds == targets) # evaluate only on 20 categories but not unknown
    total_correct += correct
    total_seen += targets.shape[0]
    tmp,_ = np.histogram(targets,range(CONF.NUM_CLASSES+1))
    labelweights += tmp
    for l in seen_classes:
        total_seen_class[l] += np.sum(targets==l)
        total_correct_class[l] += np.sum((preds==l) & (targets==l))

    _, uvlabel, _ = point_cloud_label_to_surface_voxel_label_fast(coords, np.concatenate((np.expand_dims(targets,1),np.expand_dims(preds,1)),axis=1), res=0.02)
    total_correct_vox += np.sum(uvlabel[:,0]==uvlabel[:,1])
    total_seen_vox += uvlabel[:,0].shape[0]
    tmp,_ = np.histogram(uvlabel[:,0],range(CONF.NUM_CLASSES+1))
    labelweights_vox += tmp
    for l in seen_classes:
        total_seen_class_vox[l] += np.sum(uvlabel[:,0]==l)
        total_correct_class_vox[l] += np.sum((uvlabel[:,0]==l) & (uvlabel[:,1]==l))

    pointacc = total_correct / float(total_seen)
    voxacc = total_correct_vox / float(total_seen_vox)

    labelweights = labelweights.astype(np.float32)/np.sum(labelweights.astype(np.float32))
    labelweights_vox = labelweights_vox.astype(np.float32)/np.sum(labelweights_vox.astype(np.float32))
    caliweights = labelweights_vox
    voxcaliacc = np.average(np.array(total_correct_class_vox)/(np.array(total_seen_class_vox,dtype=np.float)+1e-8),weights=caliweights)

    pointacc_per_class = np.zeros(CONF.NUM_CLASSES)
    voxacc_per_class = np.zeros(CONF.NUM_CLASSES)
    for l in seen_classes:
        pointacc_per_class[l] = total_correct_class[l]/(total_seen_class[l] + 1e-8)
        voxacc_per_class[l] = total_correct_class_vox[l]/(total_seen_class_vox[l] + 1e-8)

    return pointacc, pointacc_per_class, voxacc, voxacc_per_class, voxcaliacc, mask

def compute_miou(coords, preds, targets, weights):
    coords, preds, targets, weights = filter_points(coords, preds, targets, weights)
    seen_classes = np.unique(targets)
    mask = np.zeros(CONF.NUM_CLASSES)
    mask[seen_classes] = 1

    pointmiou = np.zeros(CONF.NUM_CLASSES)
    voxmiou = np.zeros(CONF.NUM_CLASSES)

    uvidx, uvlabel, _ = point_cloud_label_to_surface_voxel_label_fast(coords, np.concatenate((np.expand_dims(targets,1),np.expand_dims(preds,1)),axis=1), res=0.02)
    for l in seen_classes:
        target_label = np.arange(targets.shape[0])[targets==l]
        pred_label = np.arange(preds.shape[0])[preds==l]
        num_intersection_label = np.intersect1d(pred_label, target_label).shape[0]
        num_union_label = np.union1d(pred_label, target_label).shape[0]
        pointmiou[l] = num_intersection_label / (num_union_label + 1e-8)

        target_label_vox = uvidx[(uvlabel[:, 0] == l)]
        pred_label_vox = uvidx[(uvlabel[:, 1] == l)]
        num_intersection_label_vox = np.intersect1d(pred_label_vox, target_label_vox).shape[0]
        num_union_label_vox = np.union1d(pred_label_vox, target_label_vox).shape[0]
        voxmiou[l] = num_intersection_label_vox / (num_union_label_vox + 1e-8)

    return pointmiou, voxmiou, mask

def eval_one_batch(batch_size, model, data):
    # unpack
    coords, feats, targets, weights, _ = data
    coords, feats, targets, weights = coords.cuda(), feats.cuda(), targets.cuda(), weights.cuda()

    # feed
    preds = forward(batch_size, model, coords, feats)

    # eval
    coords = coords.squeeze(0).view(-1, 3).cpu().numpy()     # (CK*N, C)
    preds = preds.squeeze(0).view(-1).cpu().numpy()          # (CK*N, C)
    targets = targets.squeeze(0).view(-1).cpu().numpy()      # (CK*N, C)
    weights = weights.squeeze(0).view(-1).cpu().numpy()      # (CK*N, C)
    pointacc, pointacc_per_class, voxacc, voxacc_per_class, voxcaliacc, acc_mask = compute_acc(coords, preds, targets, weights)
    pointmiou, voxmiou, miou_mask = compute_miou(coords, preds, targets, weights)
    assert acc_mask.all() == miou_mask.all()

    return pointacc, pointacc_per_class, voxacc, voxacc_per_class, voxcaliacc, pointmiou, voxmiou, acc_mask

def prep_visualization(mc_dataset, gt_dataset, random_dataset):
    def get_selected_and_not_selected(dataset, scene_id):
        current_data = dataset.scene_data[scene_id][dataset.selected_mask[scene_id]]
        scene_data = dataset.scene_data[scene_id][np.logical_not(dataset.selected_mask[scene_id])]
        return current_data, scene_data
    
    scene_ids = random_dataset.scene_list[:5]
    data = {}
    for scene_id in scene_ids:
        mc_chosen, mc_rest = get_selected_and_not_selected(mc_dataset, scene_id)
        gt_chosen, gt_rest = get_selected_and_not_selected(gt_dataset, scene_id)
        random_chosen, random_rest = get_selected_and_not_selected(random_dataset, scene_id)
        vertex = []
        colors = []
        max_x_coord = -np.inf
        for i in range(mc_chosen.shape[0]):
            if mc_chosen[i][0] > max_x_coord:
                max_x_coord = mc_chosen[i][0]
            vertex.append(
                [
                    mc_chosen[i][0],
                    mc_chosen[i][1],
                    mc_chosen[i][2]
                ]
            )
            colors.append(
                [
                    CONF.PALETTE[int(mc_chosen[i][-1])][0],
                    CONF.PALETTE[int(mc_chosen[i][-1])][1],
                    CONF.PALETTE[int(mc_chosen[i][-1])][2]
                ]
            )
        for i in range(mc_rest.shape[0]):
            if mc_rest[i][0] > max_x_coord:
                max_x_coord = mc_rest[i][0]

        max_x_coord += 1
        for i in range(gt_chosen.shape[0]):
            vertex.append(
                [
                    gt_chosen[i][0] + max_x_coord,
                    gt_chosen[i][1],
                    gt_chosen[i][2]
                ]
            )
            colors.append(
                [
                    CONF.PALETTE[int(gt_chosen[i][-1])][0],
                    CONF.PALETTE[int(gt_chosen[i][-1])][1],
                    CONF.PALETTE[int(gt_chosen[i][-1])][2]
                ]
            )

        for i in range(random_chosen.shape[0]):
            vertex.append(
                [
                    random_chosen[i][0] + (2 * max_x_coord),
                    random_chosen[i][1],
                    random_chosen[i][2]
                ]
            )
            colors.append(
                [
                    CONF.PALETTE[int(random_chosen[i][-1])][0],
                    CONF.PALETTE[int(random_chosen[i][-1])][1],
                    CONF.PALETTE[int(random_chosen[i][-1])][2]
                ]
            )

        data[scene_id] = (vertex, colors)
    
    return data
