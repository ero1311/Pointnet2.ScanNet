from re import A
from flask import Flask, request
import json
import os
import sys
from flask_cors import cross_origin
import importlib
import torch
from predict_scene import get_scene_labels
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../pointnet2/'))

from lib.config import CONF
from preprocessing.scannet_util import g_raw2scannet

RAW2SCANNET = g_raw2scannet

Pointnet = importlib.import_module("pointnet2_semseg")
model = Pointnet.get_model(num_classes=CONF.NUM_CLASSES, is_msg=False, input_channels=6, use_xyz=True, bn=True, mc_drop=False).cuda()
model.load_state_dict(torch.load(CONF.ANNOT_SERVER_MODEL_PATH))
model.eval()

app = Flask(__name__)
@app.route('/loadstate')
@cross_origin()
def load_state():
    info_data = {}
    try:
        address = os.path.join(CONF.ANNOT_SERVER_ROOT, "public/data/info.json")
        with open(address) as f:
            info_data = json.load(f)
    except Exception as e:
        return {
            'message': 'failed to load user state',
            'status': 400
        }
    
    return {
            'data': info_data,
            'message': 'OK',
            'status': 200
        }

@app.route('/update', methods=['POST'])
@cross_origin()
def update_state():
    data = json.loads(request.data)
    address = os.path.join(CONF.ANNOT_SERVER_ROOT, "public/data/info.json")
    try:
        with open(address, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        return {
            'message': "Failed to update state",
            'status': 400
        }
    return {
        'message': 'OK',
        'status': 200
    }

@app.route('/save', methods=['POST'])
@cross_origin()
def save_annots():
    data = json.loads(request.data)
    address = CONF.ANNOT_SERVER_LABEL.format(data['file_name'])
    try:
        with open(address, 'w') as f:
            json.dump(data['annotations'], f)
    except Exception as e:
        return {
            'message': "Failed to save annotations",
            'status': 400
        }
    return {
        'message': 'OK',
        'status': 200
    }

@app.route('/delete', methods=['POST'])
@cross_origin()
def delete_annots():
    data = json.loads(request.data)
    try:
        os.remove(CONF.ANNOT_SERVER_LABEL.format(data['file_name']))
        os.remove(CONF.ANNOT_SERVER_PREDS.format(data['file_name']))
    except Exception as e:
        return {
            'message': "Failed to delete annotations",
            'status': 400
        }
    return {
        'message': 'OK',
        'status': 200
    }

@app.route('/load/')
@app.route('/load/<filename>')
@cross_origin()
def load_annots(filename=None):
    annots = {}
    preds = {}
    try:
        annots_address = CONF.ANNOT_SERVER_LABEL.format(filename)
        preds_address = CONF.ANNOT_SERVER_PREDS.format(filename)
        with open(annots_address) as f:
            annots = json.load(f)
        if os.path.exists(preds_address):
            with open(preds_address) as f:
                preds = json.load(f)
    except Exception as e:
        return {
            'message': 'failed to load {}'.format(filename),
            'status': 400
        }
    
    return {
            'data': annots,
            'data_preds': preds,
            'message': 'OK',
            'status': 200
        }

@app.route('/preannot', methods=['POST'])
@cross_origin()
def get_preannotations():
    data = json.loads(request.data)
    file_path = os.path.join(CONF.SCANNET_DIR, data["file_name"], '{}_vh_clean_2.ply'.format(data["file_name"]))
    annots_save = CONF.ANNOT_SERVER_LABEL.format(data['file_name'])
    preds_save = CONF.ANNOT_SERVER_PREDS.format(data['file_name'])
    annots = {}
    try:
        annots = get_scene_labels(file_path, data["seg_idx"], model, CONF.NYUCLASSES)
        with open(annots_save, 'w') as f:
            json.dump(annots, f)
        with open(preds_save, 'w') as f:
            json.dump(annots, f)
    except Exception as e:
        print(str(e))
        return {
            'message': "Failed to get pre-annotations",
            'status': 400
        }
    return {
        'data': annots,
        'preds': annots,
        'message': 'OK',
        'status': 200
    }

@app.route('/checkpreannot/')
@app.route('/checkpreannot/<filename>')
@cross_origin()
def check_preannots(filename=None):
    wrong_segs = []
    gts = {}
    try:
        gt_address = os.path.join(CONF.SCANNET_DIR, filename, '{}.aggregation.json'.format(filename))
        preds_address = CONF.ANNOT_SERVER_PREDS.format(filename)
        with open(gt_address) as f:
            gt_aggs = json.load(f)
        with open(preds_address) as f:
            preds = json.load(f)
        for ann_inst in gt_aggs["segGroups"]:
            if RAW2SCANNET[ann_inst["label"]] not in gts:
                gts[RAW2SCANNET[ann_inst["label"]]] = []
            gts[RAW2SCANNET[ann_inst["label"]]].extend(ann_inst["segments"])
        for class_name, seg_ids in gts.items():
            if class_name in preds['classes']:
                wrong_segs.extend(list(set(seg_ids).difference(set(preds['classes'][class_name]))))
            else:
                wrong_segs.extend(seg_ids)
    except Exception as e:
        return {
            'message': 'failed to load {}'.format(filename),
            'status': 400
        }
    
    return {
            'data': wrong_segs,
            'message': 'OK',
            'status': 200
        }

@app.route('/closestimage', methods=['POST'])
@cross_origin()
def get_closest_image():
    path = ""
    data = json.loads(request.data)
    pose_roots = os.path.join(CONF.SCANNET_DIR, data["file_name"], "pose")
    try:
        pose_files = os.listdir(pose_roots)
        poses_npy_dict = {}
        curr_pose = np.array(data["curr_pose"]).reshape(4, 4)
        for pose_name in pose_files:
            with open(os.path.join(pose_roots, pose_name)) as f:
                pose_data = f.readlines()
            data_npy = np.array([[float(i) for i in line.strip().split(' ')] for line in pose_data])
            poses_npy_dict[pose_name] = data_npy
        min_score = np.inf
        min_pose = None
        for key, value in poses_npy_dict.items():
            norm = np.linalg.norm((curr_pose - value), ord='nuc')
            if norm < min_score:
                min_score = norm
                min_pose = key
        path = os.path.join(data["file_name"], "color", min_pose.split(".")[0] + ".jpg")
    except Exception as e:
        return {
            'message': 'failed to get the closest image',
            'data': path,
            'status': 400
        }
    return {
        'message': 'OK',
        'data': path,
        'status': 200
    }