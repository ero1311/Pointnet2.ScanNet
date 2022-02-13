import os
import sys
import json
import argparse
import time

sys.path.append(".")
from lib.felzenswalb import segment_graph, write_json, read_ply_vertices_edges
from lib.config import CONF
from lib.utils import get_eta


def overseg_scene(scene_name, out_filename, k_thresh, seg_min_verts):
    data_folder = os.path.join(CONF.SCANNET_DIR, scene_name)
    ply_filename = os.path.join(data_folder, '%s_vh_clean_2.ply' % (scene_name))
    vertices, edges = read_ply_vertices_edges(ply_filename)
    seg_idx = segment_graph(vertices, edges, kthresh=k_thresh, segMinVerts=seg_min_verts)
    write_json(out_filename, scene_name, k_thresh, seg_min_verts, seg_idx)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_thresh", type=float, default=0.01)
    parser.add_argument("--seg_min_verts", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(CONF.PREP_OVERSEGS, exist_ok=True)

    for i, scene_name in enumerate(CONF.SCENE_NAMES):
        try:
            start = time.time()
            out_filename = scene_name + '.json'
            overseg_scene(scene_name, os.path.join(CONF.PREP_OVERSEGS, out_filename), args.k_thresh, args.seg_min_verts)
            
            # report
            num_left = len(CONF.SCENE_NAMES) - i - 1
            eta = get_eta(start, time.time(), 0, num_left)
            print("preprocessed {}, {} left, ETA: {}h {}m {}s".format(
                scene_name,
                num_left,
                eta["h"],
                eta["m"],
                eta["s"]
            ))

        except Exception as e:
            print(scene_name+'ERROR!!')

    print("done!")

