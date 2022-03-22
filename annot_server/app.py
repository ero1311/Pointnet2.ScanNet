from flask import Flask, request
import json
import os
import sys
from flask_cors import cross_origin
import importlib
import torch
from predict_scene import get_scene_labels

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../pointnet2/'))

from lib.config import CONF

Pointnet = importlib.import_module("pointnet2_semseg")
model = Pointnet.get_model(num_classes=CONF.NUM_CLASSES, is_msg=False, input_channels=6, use_xyz=True, bn=True, mc_drop=False).cuda()
model.load_state_dict(torch.load(CONF.ANNOT_SERVER_MODEL_PATH))
model.eval()

app = Flask(__name__)

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
    try:
        address = CONF.ANNOT_SERVER_LABEL.format(filename)
        with open(address) as f:
            annots = json.load(f)
    except Exception as e:
        return {
            'message': 'failed to load {}'.format(filename),
            'status': 400
        }
    
    return {
            'data': annots,
            'message': 'OK',
            'status': 200
        }

@app.route('/preannot', methods=['POST'])
@cross_origin()
def get_preannotations():
    data = json.loads(request.data)
    file_path = os.path.join(CONF.SCANNET_DIR, data["file_name"], '{}_vh_clean_2.ply'.format(data["file_name"]))
    annots_save = CONF.ANNOT_SERVER_LABEL.format(data['file_name'])
    annots = {}
    try:
        annots = get_scene_labels(file_path, data["seg_idx"], model, CONF.NYUCLASSES)
        with open(annots_save, 'w') as f:
            json.dump(annots, f)
    except Exception as e:
        print(str(e))
        return {
            'message': "Failed to get pre-annotations",
            'status': 400
        }
    return {
        'data': annots,
        'message': 'OK',
        'status': 200
    }