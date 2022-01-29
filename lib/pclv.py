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

    def __init__(self, node1, node2, w_n=0, w_c=0, w_d=0):
        self.node1 = node1
        self.node2 = node2
        self.w_n = w_n
        self.w_c = w_c
        self.w_d = w_d

    def __eq__(self, other):
        return (self.node1 == other.node1 and self.node2 == other.node2) or (self.node1 == other.node2 and self.node2 == other.node1)
    
    def __hash__(self):
        return hash(int(str(self.node1) + str(self.node2)) + int(str(self.node2) + str(self.node1)))
    
    def __str__(self):
        return "Node 1: {}, Node 2: {}, Weight Normal: {}, Color Normal: {}, Distance Normal: {}".format(self.node1, self.node2, self.w_n, self.w_c, self.w_d)

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
        d_max, d_min = 0, np.inf
        for f in plydata["face"].data:
            faces.append(f[0])
            edges.add(Edge(f[0][0], f[0][1]))
            dd = np.sqrt(np.square(xyz[f[0][0]] - xyz[f[0][1]]).sum())
            if dd > d_max:
                d_max = dd
            if dd < d_min:
                d_min = dd
            edges.add(Edge(f[0][1], f[0][2]))
            dd = np.sqrt(np.square(xyz[f[0][1]] - xyz[f[0][2]]).sum())
            if dd > d_max:
                d_max = dd
            if dd < d_min:
                d_min = dd
            edges.add(Edge(f[0][0], f[0][2]))
            dd = np.sqrt(np.square(xyz[f[0][0]] - xyz[f[0][2]]).sum())
            if dd > d_max:
                d_max = dd
            if dd < d_min:
                d_min = dd
        face = np.array([f[0] for f in plydata["face"].data])
        nxnynz = compute_normal(xyz, face)
        vertices[:,6:] = nxnynz
    return vertices, list(edges), d_max, d_min

def segment_graph(vertices, edges, d_max, d_min, kthresh=0.01, segMinVerts=20):
    for edge in edges:
        norm1 = np.linalg.norm(vertices[edge.node1, 6:].reshape(1, 3))
        norm2 = np.linalg.norm(vertices[edge.node2, 6:].reshape(1, 3))
        edge.w_n = 1 - np.abs(vertices[edge.node1, 6:].reshape(1, 3) @ vertices[edge.node2, 6:].reshape(3, 1)).squeeze() / (norm1 * norm2)
        edge.w_c = np.sqrt(np.square((vertices[edge.node1, 3:6] - vertices[edge.node2, 3:6]) / 255).sum() / 3)
        edge.w_d = (np.sqrt(np.square((vertices[edge.node1, :3] - vertices[edge.node2, :3])).sum()) - d_min) / (d_max - d_min)
    
    edges = sorted(edges, key=lambda x: x.w_c)
    u = Universe(vertices.shape[0])
    threshold_c = vertices.shape[0] * [kthresh]
    threshold_n = vertices.shape[0] * [kthresh]
    threshold_d = vertices.shape[0] * [kthresh]
    cnt = 0
    for edge in edges:
        a = u.find(edge.node1)
        b = u.find(edge.node2)
        if a != b:
            if edge.w_n <= threshold_n[a] and edge.w_n <= threshold_n[b] and edge.w_c <= threshold_c[a] and edge.w_c <= threshold_c[b] and edge.w_d <= threshold_d[a] and edge.w_d <= threshold_d[b]:
                u.join(a, b)
                a = u.find(a)
                threshold_n[a] = edge.w_n + kthresh / u.size(a)
                threshold_c[a] = edge.w_c + kthresh / u.size(a)
                threshold_d[a] = edge.w_d + kthresh / u.size(a)
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
    vertices, edges, d_max, d_min = read_ply_xyzrgbnormal(args.scene_path)
    seg_idx = segment_graph(vertices, edges, d_max, d_min, args.k_thresh, args.seg_min_verts)
    out_json = basename(args.scene_path).split('.')[0] + "_" + str(args.k_thresh) + "_" + str(args.seg_min_verts) + '.segs.json'
    out_json = join(args.output_base, out_json)
    write_json(out_json, args.scene_name, args.k_thresh, args.seg_min_verts, seg_idx)
    print("Number of segments: {}".format(np.unique(np.array(seg_idx)).shape[0]))