import json
import os
import argparse
import numpy as np

import sys
sys.path.append(".")
from lib.config import CONF
from lib.pc_util import read_ply_xyzrgbnormal
from plyfile import PlyElement, PlyData
from generate_n_color import generate_n_colors
import torch

def visualize(args):
    print("visualizing...")
    overseg_files = os.listdir(args.oversegment_base)
    scene_base = read_ply_xyzrgbnormal(args.scene_path)
    with open(args.scene_path, 'rb') as f:
        plydata = PlyData.read(f)
    output_f = plydata["face"]
    scene_base = np.concatenate([scene_base, np.zeros((scene_base.shape[0], 1))], axis=1)
    visual_path = os.path.join('/home/erik/TUM/Pointnet2.ScanNet', 'visualizations', args.scene_name)
    os.makedirs(visual_path, exist_ok=True)
    for overseg_file in overseg_files:
        with open(os.path.join(args.oversegment_base, overseg_file)) as f:
            seg_data = json.load(f)
        labels = np.array(seg_data['segIndices']).reshape(-1, 1)
        scene = np.concatenate([scene_base, labels], axis=1)
        PALETTE = generate_n_colors(np.unique(labels).shape[0])
        segid_to_colorid = dict(zip(np.unique(labels), np.arange(np.unique(labels).shape[0])))
        vertex = []
        for i in range(scene.shape[0]):
            vertex.append(
                (
                    scene[i][0],
                    scene[i][1],
                    scene[i][2],
                    PALETTE[segid_to_colorid[int(scene[i][-1])]][0],
                    PALETTE[segid_to_colorid[int(scene[i][-1])]][1],
                    PALETTE[segid_to_colorid[int(scene[i][-1])]][2],
                    scene[i][6],
                    scene[i][7],
                    scene[i][8]
                )
            )
        vertex = np.array(
            vertex,
            dtype=[
                ("x", np.dtype("float32")), 
                ("y", np.dtype("float32")), 
                ("z", np.dtype("float32")),
                ("red", np.dtype("uint8")),
                ("green", np.dtype("uint8")),
                ("blue", np.dtype("uint8")),
                ("nx", np.dtype("float32")), 
                ("ny", np.dtype("float32")), 
                ("nz", np.dtype("float32"))
            ]
        )
        output_pc = PlyElement.describe(vertex, "vertex")
        output_pc = PlyData([output_pc])#, output_f])
        output_pc.write(os.path.join(visual_path, overseg_file[:-5] + '.ply'))
    vertex = []
    for i in range(scene.shape[0]):
        vertex.append(
            (
                scene[i][0],
                scene[i][1],
                scene[i][2],
                scene[i][3],
                scene[i][4],
                scene[i][5]
            )
        )
    vertex = np.array(
        vertex,
        dtype=[
            ("x", np.dtype("float32")), 
            ("y", np.dtype("float32")), 
            ("z", np.dtype("float32")),
            ("red", np.dtype("uint8")),
            ("green", np.dtype("uint8")),
            ("blue", np.dtype("uint8")),
        ]
    )
    output_pc = PlyElement.describe(vertex, "vertex")
    output_pc = PlyData([output_pc, output_f])
    output_pc.write(os.path.join(visual_path, args.scene_name + '.ply'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_path", type=str, required=True)
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--oversegment_base", type=str, required=True)
    args = parser.parse_args()

    visualize(args)
    print("done!")