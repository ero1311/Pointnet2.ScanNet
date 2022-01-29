from cv2 import norm
from pc_util import compute_normal
from plyfile import PlyData
from os.path import join, basename, isfile
import numpy as np
import json
import argparse

class Uni_elt(object):

    def __init__(self, rank, p, size):
        self.rank = rank
        self.p = p
        self.size = size

class Universe(object):

    def __init__(self, num_elements) -> None:
        self.num = num_elements
        self.elts = [Uni_elt(0, i, 1) for i in range(self.num)]
    
    def find(self, x):
        y = x
        while y != self.elts[y].p:
            y = self.elts[y].p
        self.elts[x].p = y
        return y
    
    def join(self, x, y):
        if self.elts[x].rank > self.elts[y].rank:
            self.elts[y].p = x
            self.elts[x].size += self.elts[y].size
        else:
            self.elts[x].p = y
            self.elts[y].size += self.elts[x].size
            if self.elts[x].rank == self.elts[y].rank:
                self.elts[y].rank += 1
        self.num -= 1
    
    def size(self, x):
        return self.elts[x].size
    
    def num_sets(self):
        return self.num

class Edge(object):

    def __init__(self, node1, node2, w_n=0):
        self.node1 = node1
        self.node2 = node2
        self.w_n = w_n

    def __eq__(self, other):
        return (self.node1 == other.node1 and self.node2 == other.node2) or (self.node1 == other.node2 and self.node2 == other.node1)
    
    def __hash__(self):
        return hash(int(str(self.node1) + str(self.node2)) + int(str(self.node2) + str(self.node1)))
    
    def __str__(self):
        return "Node 1: {}, Node 2: {}, Weight Normal: {}".format(self.node1, self.node2, self.w_n)

def read_ply_xyzrgbnormal(filename):
    """ read XYZ RGB normals point cloud from filename PLY file """
    assert(isfile(filename))
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 9], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        vertices[:,3] = plydata['vertex'].data['red']
        vertices[:,4] = plydata['vertex'].data['green']
        vertices[:,5] = plydata['vertex'].data['blue']

        # compute normals
        edges = set()
        faces = []
        xyz = np.array([[x, y, z] for x, y, z, _, _, _, _ in plydata["vertex"].data])
        for f in plydata["face"].data:
            faces.append(f[0])
            edges.add(Edge(f[0][0], f[0][1]))
            edges.add(Edge(f[0][1], f[0][2]))
            edges.add(Edge(f[0][0], f[0][2]))
        face = np.array([f[0] for f in plydata["face"].data])
        nxnynz = compute_normal(xyz, face)
        vertices[:,6:] = nxnynz
    return vertices, list(edges)

def segment_graph(vertices, edges, kthresh=0.01, segMinVerts=20, R=0.5):
    for edge in edges:
        norm1 = np.linalg.norm(vertices[edge.node1, 6:].reshape(1, 3))
        norm2 = np.linalg.norm(vertices[edge.node2, 6:].reshape(3, 1))
        edge.w_n = 1 - np.abs(vertices[edge.node1, 6:].reshape(1, 3) @ vertices[edge.node2, 6:].reshape(3, 1)).squeeze() / (norm1 * norm2)
    
    edges = sorted(edges, key=lambda x: x.w_n)
    u = Universe(vertices.shape[0])
    threshold = vertices.shape[0] * [kthresh]
    cnt = 0
    for edge in edges:
        a = u.find(edge.node1)
        b = u.find(edge.node2)
        if a != b:
            if edge.w_n <= threshold[a] and edge.w_n <= threshold[b]:
                u.join(a, b)
                a = u.find(a)
                threshold[a] = edge.w_n + kthresh / u.size(a)
                cnt += 1
    
    print("Join times: {}".format(cnt))

    for edge in edges:
        a = u.find(edge.node1)
        b = u.find(edge.node2)
        if a != b and (u.size(a) < segMinVerts or u.size(b) < segMinVerts):
            u.join(a, b)
    
    out_comps = []
    for i in range(vertices.shape[0]):
        out_comps.append(int(u.find(i)))
    
    return out_comps
    
def write_json(file_name, scene_name, kthresh, seg_min_verts, seg_idx):
    out_data = {
        'params': {
            'kThresh': kthresh,
            'segMinVerts': seg_min_verts
        },
        'sceneId': scene_name,
        'segIndices': seg_idx
    }
    with open(file_name, 'w') as f:
        json.dump(out_data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_path", type=str, required=True)
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--k_thresh", type=float, default=0.01)
    parser.add_argument("--seg_min_verts", type=int, default=20)
    parser.add_argument("--output_base", type=str, required=True)
    args = parser.parse_args()
    vertices, edges = read_ply_xyzrgbnormal(args.scene_path)
    seg_idx = segment_graph(vertices, edges, args.k_thresh, args.seg_min_verts)
    out_json = basename(args.scene_path).split('.')[0] + "_" + str(args.k_thresh) + "_" + str(args.seg_min_verts) + '.segs.json'
    out_json = join(args.output_base, out_json)
    write_json(out_json, args.scene_name, args.k_thresh, args.seg_min_verts, seg_idx)
    print("Number of segments: {}".format(np.unique(np.array(seg_idx)).shape[0]))