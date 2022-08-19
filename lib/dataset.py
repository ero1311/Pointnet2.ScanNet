from operator import le
import os
import sys
import time
import h5py
import random
from numpy.core.numeric import True_
from numpy.core.shape_base import block
from numpy.lib.index_tricks import _ix__dispatcher
import torch
import numpy as np
import multiprocessing as mp
from torch._C import _tracer_warn_use_python
from tqdm import tqdm
from prefetch_generator import background
from collections import OrderedDict
import importlib

sys.path.append(".")
from lib.config import CONF
from lib.inference_utils import mc_forward, forward, filter_points, eval_one_batch
#from pointnet2.pointnet2_utils import furthest_point_sample
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../pointnet2/'))
sample_utils = importlib.import_module("pointnet2_utils")

class ScannetDataset():
    def __init__(self, phase, scene_list, num_classes=21, npoints=2048, is_weighting=True, use_multiview=False, use_color=False, use_normal=False):
        self.phase = phase
        assert phase in ["train", "val", "test"]
        self.scene_list = scene_list
        self.num_classes = num_classes
        self.npoints = npoints
        self.is_weighting = is_weighting
        self.use_multiview = use_multiview
        self.use_color = use_color
        self.use_normal = use_normal
        self.chunk_data = {} # init in generate_chunks()

        self._prepare_weights()

    def _prepare_weights(self):
        semantic_labels_list = []
        for scene_id in tqdm(self.scene_list):
            scene_data = np.load(CONF.SCANNETV2_FILE.format(scene_id))
            semantic_labels_list.append(scene_data[:, 11])

        if self.is_weighting:
            labelweights = np.zeros(self.num_classes)
            for seg in semantic_labels_list:
                tmp,_ = np.histogram(seg,range(self.num_classes + 1))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        else:
            self.labelweights = np.ones(self.num_classes)

    @background()
    def __getitem__(self, index):
        start = time.time()

        # load chunks
        scene_id = self.scene_list[index]
        scene_data = self.generate_chunk(scene_id)
        # unpack
        xyz = scene_data[:, :3] # include xyz by default
        rgb = scene_data[:, 3:6] / 255. # normalize the rgb values to [0, 1]
        xyz_min = np.amin(xyz, axis=0)
        xyz -= xyz_min
        label = scene_data[:, 11].astype(np.int32)
        
        if self.phase == "train":
            point_set = self._augment(point_set)
        
        xyz_min = np.amin(xyz, axis=0)
        XYZ = xyz - xyz_min
        xyz_max = np.amax(XYZ, axis=0)
        XYZ = XYZ/xyz_max
        # prepare mask
        ptcloud = []
        ptcloud.append(xyz)
        ptcloud.append(rgb)
        ptcloud.append(XYZ)
        point_set = np.concatenate(ptcloud, axis=1)
        # prepare mask
        curmin = np.min(point_set, axis=0)[:3]
        curmax = np.max(point_set, axis=0)[:3]
        mask = np.sum((point_set[:, :3] >= (curmin - 0.01)) * (point_set[:, :3] <= (curmax + 0.01)), axis=1) == 3
        sample_weight = self.labelweights[label]
        sample_weight *= mask

        fetch_time = time.time() - start

        return point_set, label, sample_weight, fetch_time

    def __len__(self):
        return len(self.scene_list)

    def _augment(self, point_set):
        # translate the chunk center to the origin
        center = np.mean(point_set[:, :3], axis=0)
        coords = point_set[:, :3] - center

        p = np.random.choice(np.arange(0.01, 1.01, 0.01), size=1)[0]
        if p < 1 / 8:
            # random translation
            coords = self._translate(coords)
        elif p >= 1 / 8 and p < 2 / 8:
            # random rotation
            coords = self._rotate(coords)
        elif p >= 2 / 8 and p < 3 / 8:
            # random scaling
            coords = self._scale(coords)
        elif p >= 3 / 8 and p < 4 / 8:
            # random translation
            coords = self._translate(coords)
            # random rotation
            coords = self._rotate(coords)
        elif p >= 4 / 8 and p < 5 / 8:
            # random translation
            coords = self._translate(coords)
            # random scaling
            coords = self._scale(coords)
        elif p >= 5 / 8 and p < 6 / 8:
            # random rotation
            coords = self._rotate(coords)
            # random scaling
            coords = self._scale(coords)
        elif p >= 6 / 8 and p < 7 / 8:
            # random translation
            coords = self._translate(coords)
            # random rotation
            coords = self._rotate(coords)
            # random scaling
            coords = self._scale(coords)
        else:
            # no augmentation
            pass

        # translate the chunk center back to the original center
        coords += center
        point_set[:, :3] = coords

        return point_set

    def _translate(self, point_set):
        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]

        coords = point_set[:, :3]
        coords += [x_factor, y_factor, z_factor]
        point_set[:, :3] = coords

        return point_set

    def _rotate(self, point_set):
        coords = point_set[:, :3]

        # x rotation matrix
        theta = np.random.choice(np.arange(-5, 5.001, 0.001), size=1)[0] * 3.14 / 180 # in radians
        Rx = np.array(
            [[1, 0, 0],
             [0, np.cos(theta), -np.sin(theta)],
             [0, np.sin(theta), np.cos(theta)]]
        )

        # y rotation matrix
        theta = np.random.choice(np.arange(-5, 5.001, 0.001), size=1)[0] * 3.14 / 180 # in radians
        Ry = np.array(
            [[np.cos(theta), 0, np.sin(theta)],
             [0, 1, 0],
             [-np.sin(theta), 0, np.cos(theta)]]
        )

        # z rotation matrix
        theta = np.random.choice(np.arange(-5, 5.001, 0.001), size=1)[0] * 3.14 / 180 # in radians
        Rz = np.array(
            [[np.cos(theta), -np.sin(theta), 0],
             [np.sin(theta), np.cos(theta), 0],
             [0, 0, 1]]
        )

        # rotate
        R = np.matmul(np.matmul(Rz, Ry), Rx)
        coords = np.matmul(R, coords.T).T

        # dump
        point_set[:, :3] = coords

        return point_set

    def _scale(self, point_set):
        # scaling factors
        factor = np.random.choice(np.arange(0.95, 1.051, 0.001), size=1)[0]

        coords = point_set[:, :3]
        coords *= [factor, factor, factor]
        point_set[:, :3] = coords

        return point_set

    def generate_chunk(self, scene_id):
        """
            note: must be called before training
        """

        scene = np.load(CONF.SCANNETV2_FILE.format(scene_id))
        semantic = scene[:, 11].astype(np.int32)

        coordmax = np.max(scene, axis=0)[:3]
        coordmin = np.min(scene, axis=0)[:3]
        
        for _ in range(5):
            curcenter = scene[np.random.choice(len(semantic), 1)[0],:3]
            curmin = curcenter-[0.75,0.75,1.5]
            curmax = curcenter+[0.75,0.75,1.5]
            curmin[2] = coordmin[2]
            curmax[2] = coordmax[2]
            curchoice = np.sum((scene[:, :3]>=(curmin-0.2))*(scene[:, :3]<=(curmax+0.2)),axis=1)==3
            cur_point_set = scene[curchoice]
            cur_semantic_seg = semantic[curchoice]

            if len(cur_semantic_seg)==0:
                continue

            mask = np.sum((cur_point_set[:, :3]>=(curmin-0.01))*(cur_point_set[:, :3]<=(curmax+0.01)),axis=1)==3
            vidx = np.ceil((cur_point_set[mask,:3]-curmin)/(curmax-curmin)*[31.0,31.0,62.0])
            vidx = np.unique(vidx[:,0]*31.0*62.0+vidx[:,1]*62.0+vidx[:,2])
            isvalid = np.sum(cur_semantic_seg>0)/len(cur_semantic_seg)>=0.7 and len(vidx)/31.0/31.0/62.0>=0.02

            if isvalid:
                break
        
        # store chunk
        chunk = cur_point_set

        choices = np.random.choice(chunk.shape[0], self.npoints, replace=True)
        chunk = chunk[choices]
        return chunk
            
class ScannetDatasetWholeScene():
    def __init__(self, scene_list, npoints=2048, is_weighting=True, use_color=False, use_normal=False, use_multiview=False):
        self.scene_list = scene_list
        self.npoints = npoints
        self.is_weighting = is_weighting
        self.use_color = use_color
        self.use_normal = use_normal
        self.use_multiview = use_multiview

        self._load_scene_file()

    def _load_scene_file(self):
        self.scene_points_list = []
        self.semantic_labels_list = []

        for scene_id in tqdm(self.scene_list):
            scene_data = np.load(CONF.SCANNETV2_FILE.format(scene_id))
            self.semantic_labels_list.append(scene_data[:, 11].astype(np.int32))

        if self.is_weighting:
            labelweights = np.zeros(CONF.NUM_CLASSES)
            for seg in self.semantic_labels_list:
                tmp,_ = np.histogram(seg,range(CONF.NUM_CLASSES + 1))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        else:
            self.labelweights = np.ones(CONF.NUM_CLASSES)

    @background()
    def __getitem__(self, index):
        start = time.time()
        scene_id = self.scene_list[index]
        scene_data = np.load(CONF.SCANNETV2_FILE.format(scene_id))

        # unpack
        point_set_ini = scene_data[:, :3] # include xyz by default
        color = scene_data[:, 3:6] / 255. # normalize the rgb values to [0, 1]

        if self.use_color:
            point_set_ini = np.concatenate([point_set_ini, color], axis=1)

        semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
        coordmax = point_set_ini[:, :3].max(axis=0)
        coordmin = point_set_ini[:, :3].min(axis=0)
        xlength = 1.5
        ylength = 1.5
        nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/xlength).astype(np.int32)
        nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/ylength).astype(np.int32)
        point_sets = list()
        semantic_segs = list()
        sample_weights = list()

        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                curmin = coordmin+[i*xlength, j*ylength, 0]
                curmax = coordmin+[(i+1)*xlength, (j+1)*ylength, coordmax[2]-coordmin[2]]
                mask = np.sum((point_set_ini[:, :3]>=(curmin-0.01))*(point_set_ini[:, :3]<=(curmax+0.01)), axis=1)==3
                cur_point_set = point_set_ini[mask,:]
                cur_semantic_seg = semantic_seg_ini[mask]
                if len(cur_semantic_seg) == 0:
                    continue
                
                remainder = len(cur_semantic_seg) % self.npoints
                n_splits = len(cur_semantic_seg) // self.npoints
                choice = np.random.choice(len(cur_semantic_seg), size = self.npoints - remainder, replace=True)
                cur_point_set = np.concatenate([cur_point_set, cur_point_set[choice].copy()], axis=0)
                cur_semantic_seg = np.concatenate([cur_semantic_seg, cur_semantic_seg[choice].copy()], axis=0)
                mask = np.concatenate([mask, mask[choice].copy()], axis=0)
                cur_point_sets = np.split(cur_point_set, len(cur_semantic_seg) // self.npoints, axis=0)
                cur_semantic_segs = np.split(cur_semantic_seg, len(cur_semantic_seg) // self.npoints, axis=0)
                for k in range(len(cur_point_sets)):
                    point_set = cur_point_sets[k] # Nx3
                    semantic_seg = cur_semantic_segs[k] # N
                    # if sum(mask)/float(len(mask))<0.01:
                    #     continue
                    xyz = point_set[:, :3] # include xyz by default
                    rgb = point_set[:, 3:6]
                    xyz_min = np.amin(xyz, axis=0)
                    xyz -= xyz_min

                    xyz_min = np.amin(xyz, axis=0)
                    XYZ = xyz - xyz_min
                    xyz_max = np.amax(XYZ, axis=0)
                    XYZ = XYZ/xyz_max
                    # prepare mask
                    ptcloud = []
                    ptcloud.append(xyz)
                    ptcloud.append(rgb)
                    ptcloud.append(XYZ)
                    point_set = np.concatenate(ptcloud, axis=1)
                    sample_weight = self.labelweights[semantic_seg]
                    point_sets.append(np.expand_dims(point_set,0)) # 1xNx3
                    semantic_segs.append(np.expand_dims(semantic_seg,0)) # 1xN
                    sample_weights.append(np.expand_dims(sample_weight,0)) # 1xN

        point_sets = np.concatenate(tuple(point_sets),axis=0)
        semantic_segs = np.concatenate(tuple(semantic_segs),axis=0)
        sample_weights = np.concatenate(tuple(sample_weights),axis=0)

        fetch_time = time.time() - start

        return point_sets, semantic_segs, sample_weights, fetch_time

    def __len__(self):
        return len(self.scene_list)

class ScannetDatasetActiveLearning():
    def __init__(self, phase, scene_list, num_classes=21, npoints=8192, is_weighting=True, use_multiview=False, use_color=False, use_normal=False, heuristic="random"):
        self.phase = phase
        assert phase in ["train", "val", "test"]
        self.scene_list = scene_list
        self.num_classes = num_classes
        self.npoints = npoints
        self.is_weighting = is_weighting
        self.use_multiview = use_multiview
        self.use_color = use_color
        self.use_normal = use_normal
        self.heuristic = heuristic
        self.chunk_data = {} # init in generate_chunks()

        self._init_data()
        self._prepare_weights()

    def _init_data(self):
        self.selected_mask = {}
        for scene_id in tqdm(self.scene_list):
            scene_data = np.load(CONF.SCANNETV2_FILE.format(scene_id))
            selected_segment = np.random.choice(np.unique(scene_data[:, 9]), size=1)
            curchoice = (scene_data[:, 9] == selected_segment)
            self.selected_mask[scene_id] = curchoice.copy()

    def _prepare_weights(self):
        if self.is_weighting:
            labelweights = np.zeros(self.num_classes)
            semantic_labels_list = []
            for scene_id in self.scene_list:
                scene_data = np.load(CONF.SCANNETV2_FILE.format(scene_id))
                semantic_labels_list.append(scene_data[self.selected_mask[scene_id]][:, 11])
            for seg in semantic_labels_list:
                tmp,_ = np.histogram(seg,range(self.num_classes + 1))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        else:
            self.labelweights = np.ones(self.num_classes)
        print("PREPARE WEIGHTS: ", self.labelweights.shape)

    def _copy_points(self, other_dataset):
        for scene_id in self.scene_list:
            self.selected_mask[scene_id] = other_dataset.selected_mask[scene_id].copy()
        self._prepare_weights()

    @background()
    def __getitem__(self, index):
        start = time.time()

        # load chunks
        scene_id = self.scene_list[index]
        scene_data = self.generate_chunk(scene_id)
        # unpack
        xyz = scene_data[:, :3] # include xyz by default
        rgb = scene_data[:, 3:6] / 255. # normalize the rgb values to [0, 1]
        xyz_min = np.amin(xyz, axis=0)
        xyz -= xyz_min
        label = scene_data[:, 11].astype(np.int32)

        if self.phase == "train":
            xyz = self._augment(xyz)

        xyz_min = np.amin(xyz, axis=0)
        XYZ = xyz - xyz_min
        xyz_max = np.amax(XYZ, axis=0)
        XYZ = XYZ/xyz_max
        # prepare mask
        ptcloud = []
        ptcloud.append(xyz)
        ptcloud.append(rgb)
        ptcloud.append(XYZ)
        ptcloud = np.concatenate(ptcloud, axis=1)

        curmin = np.min(ptcloud, axis=0)[:3]
        curmax = np.max(ptcloud, axis=0)[:3]
        mask = np.sum((ptcloud[:, :3] >= (curmin - 0.01)) * (ptcloud[:, :3] <= (curmax + 0.01)), axis=1) == 3
        sample_weight = self.labelweights[label]
        sample_weight *= mask

        fetch_time = time.time() - start

        return ptcloud, label, sample_weight, fetch_time

    def __len__(self):
        return len(self.scene_list)

    def _augment(self, point_set):
        # translate the chunk center to the origin
        center = np.mean(point_set[:, :3], axis=0)
        coords = point_set[:, :3] - center

        p = np.random.choice(np.arange(0.01, 1.01, 0.01), size=1)[0]
        if p < 1 / 8:
            # random translation
            coords = self._translate(coords)
        elif p >= 1 / 8 and p < 2 / 8:
            # random rotation
            coords = self._rotate(coords)
        elif p >= 2 / 8 and p < 3 / 8:
            # random scaling
            coords = self._scale(coords)
        elif p >= 3 / 8 and p < 4 / 8:
            # random translation
            coords = self._translate(coords)
            # random rotation
            coords = self._rotate(coords)
        elif p >= 4 / 8 and p < 5 / 8:
            # random translation
            coords = self._translate(coords)
            # random scaling
            coords = self._scale(coords)
        elif p >= 5 / 8 and p < 6 / 8:
            # random rotation
            coords = self._rotate(coords)
            # random scaling
            coords = self._scale(coords)
        elif p >= 6 / 8 and p < 7 / 8:
            # random translation
            coords = self._translate(coords)
            # random rotation
            coords = self._rotate(coords)
            # random scaling
            coords = self._scale(coords)
        else:
            # no augmentation
            pass

        # translate the chunk center back to the original center
        coords += center
        point_set[:, :3] = coords

        return point_set

    def _translate(self, point_set):
        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]

        coords = point_set[:, :3]
        coords += [x_factor, y_factor, z_factor]
        point_set[:, :3] = coords

        return point_set

    def _rotate(self, point_set):
        coords = point_set[:, :3]

        # x rotation matrix
        theta = np.random.choice(np.arange(-5, 5.001, 0.001), size=1)[0] * 3.14 / 180 # in radians
        Rx = np.array(
            [[1, 0, 0],
             [0, np.cos(theta), -np.sin(theta)],
             [0, np.sin(theta), np.cos(theta)]]
        )

        # y rotation matrix
        theta = np.random.choice(np.arange(-5, 5.001, 0.001), size=1)[0] * 3.14 / 180 # in radians
        Ry = np.array(
            [[np.cos(theta), 0, np.sin(theta)],
             [0, 1, 0],
             [-np.sin(theta), 0, np.cos(theta)]]
        )

        # z rotation matrix
        theta = np.random.choice(np.arange(-5, 5.001, 0.001), size=1)[0] * 3.14 / 180 # in radians
        Rz = np.array(
            [[np.cos(theta), -np.sin(theta), 0],
             [np.sin(theta), np.cos(theta), 0],
             [0, 0, 1]]
        )

        # rotate
        R = np.matmul(np.matmul(Rz, Ry), Rx)
        coords = np.matmul(R, coords.T).T

        # dump
        point_set[:, :3] = coords

        return point_set

    def _scale(self, point_set):
        # scaling factors
        factor = np.random.choice(np.arange(0.95, 1.051, 0.001), size=1)[0]

        coords = point_set[:, :3]
        coords *= [factor, factor, factor]
        point_set[:, :3] = coords

        return point_set

    def generate_chunk(self, scene_id):
        """
            note: must be called before training
        """

        scene = np.load(CONF.SCANNETV2_FILE.format(scene_id))[self.selected_mask[scene_id]]
        semantic = scene[:, 11].astype(np.int32)

        coordmax = np.max(scene, axis=0)[:3]
        coordmin = np.min(scene, axis=0)[:3]
        num_points_selected = 0
        curcenter = scene[np.random.choice(len(semantic), 1)[0],:3]
        while True:
            curmin = curcenter-[0.75,0.75,1.5]
            curmax = curcenter+[0.75,0.75,1.5]
            curmin[2] = coordmin[2]
            curmax[2] = coordmax[2]
            curchoice = np.sum((scene[:, :3]>=(curmin-0.2))*(scene[:, :3]<=(curmax+0.2)),axis=1)==3
            cur_point_set = scene[curchoice]
            cur_semantic_seg = semantic[curchoice]
            if self.use_multiview:
                cur_feature = feature[curchoice]

            if len(cur_semantic_seg)==0:
                continue
            
            if curchoice.sum() > num_points_selected:
                curcenter = cur_point_set[:, :3].mean(axis=0)
                num_points_selected = curchoice.sum()
            else:
                break
        
        # store chunk
        chunk = cur_point_set

        choices = np.random.choice(chunk.shape[0], self.npoints, replace=True)
        chunk = chunk[choices]
        return chunk


    def choose_new_points(self, model=None, mc_iters=20, n_segs=1):
        '''
        Choose new points to be annotated and added to the dataset

        :param model: model to use in case of mc and gt. Default: None
        :type model: torch.nn.Module
        :param mc_iters: number of Monte Carlo dropout iterations to perform. Default: 20
        :type mc_iters: int
        '''

        xlength = 1.5
        ylength = 1.5
        for scene_id in tqdm(self.scene_list):
            scene_data = np.load(CONF.SCANNETV2_FILE.format(scene_id))
            scene = scene_data[np.logical_not(self.selected_mask[scene_id])]
            if scene.shape[0] == 0:
                continue
            coordmax = scene[:, :3].max(axis=0)
            coordmin = scene[:, :3].min(axis=0)
            nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/xlength).astype(np.int32)
            nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/ylength).astype(np.int32)

            point_set_ini = scene_data[:, :3]
            color = scene_data[:, 3:6] / 255.

            if self.use_color:
                point_set_ini = np.concatenate([point_set_ini, color], axis=1)

            if self.heuristic == 'random':
                current_choice = np.zeros_like(scene_data[:, 9], dtype=np.bool)
                if len(np.unique(scene[:, 9])) < n_segs:
                    rand_seg_id = np.unique(scene[:, 9])
                else:
                    rand_seg_id = np.random.choice(np.unique(scene[:, 9]), size=n_segs, replace=False)
                for curr_seg_id in rand_seg_id:
                    current_choice = np.logical_or(current_choice, (scene_data[:, 9] == curr_seg_id))     
            else:
                point_scores = np.zeros(scene_data.shape[0])
                point_weights = np.zeros_like(point_scores)
                point_mask = np.zeros_like(self.selected_mask[scene_id], dtype=np.bool)
                segment_scores = []
                npoints = 2048

                for i in range(nsubvolume_x):
                    for j in range(nsubvolume_y):
                        curmin = coordmin+[i*xlength, j*ylength, 0]
                        curmax = coordmin+[(i+1)*xlength, (j+1)*ylength, coordmax[2]-coordmin[2]]
                        mask = np.sum((point_set_ini[:, :3]>=(curmin-0.01))*(point_set_ini[:, :3]<=(curmax+0.01)), axis=1)==3
                        mask = np.logical_and(mask, np.logical_not(point_mask))
                        cur_point_set = point_set_ini[mask,:]
                        cur_semantic_seg = scene_data[mask, 11].astype(np.int32)
                        scores = np.zeros(cur_semantic_seg.shape)
                        block_weights = np.zeros(cur_semantic_seg.shape)
                        block_mask = mask.copy()
                        if len(cur_semantic_seg) == 0:
                            continue

                        remainder = len(cur_semantic_seg) % npoints
                        choice = np.random.choice(len(cur_semantic_seg), size = npoints - remainder, replace=True)
                        cur_point_set = np.concatenate([cur_point_set, cur_point_set[choice].copy()], axis=0)
                        cur_semantic_seg = np.concatenate([cur_semantic_seg, cur_semantic_seg[choice].copy()], axis=0)
                        cur_point_sets = np.split(cur_point_set, len(cur_semantic_seg) // npoints, axis=0)
                        cur_semantic_segs = np.split(cur_semantic_seg, len(cur_semantic_seg) // npoints, axis=0)
                        for k in range(len(cur_point_sets)):
                            point_set = cur_point_sets[k] # Nx3
                            mask_start, mask_end = k * npoints, (k + 1) * npoints
                            semantic_seg = cur_semantic_segs[k] # N
                            sample_weight = self.labelweights[semantic_seg]
                            xyz = point_set[:, :3] # include xyz by default
                            rgb = point_set[:, 3:6] # normalize the rgb values to [0, 1]
                            xyz_min = np.amin(xyz, axis=0)
                            xyz -= xyz_min
                            xyz_min = np.amin(xyz, axis=0)
                            XYZ = xyz - xyz_min
                            xyz_max = np.amax(XYZ, axis=0)
                            XYZ = XYZ/xyz_max
                            # prepare mask
                            ptcloud = []
                            ptcloud.append(xyz)
                            ptcloud.append(rgb)
                            ptcloud.append(XYZ)
                            point_set = np.concatenate(ptcloud, axis=1)
                            with torch.no_grad():
                                point_set = np.expand_dims(point_set, 0) # 1xNx3
                                semantic_seg = np.expand_dims(semantic_seg, 0) # 1x1xN
                                sample_weight = np.expand_dims(sample_weight, 0) # 1x1xN
                                point_set_t = torch.FloatTensor(point_set)
                                targets = torch.LongTensor(semantic_seg)
                                weights = torch.FloatTensor(sample_weight)
                                coords = point_set_t[:, :, :3]
                                feats = point_set_t[:, :, 3:]
                                coords, feats, targets, weights = coords.unsqueeze(0), feats.unsqueeze(0), targets.unsqueeze(0), weights.unsqueeze(0)
                                coords, feats, targets, weights = coords.cuda(), feats.cuda(), targets.cuda(), weights.cuda()
                                if self.heuristic == "gt":
                                    preds = forward(1, model, coords, feats)
                                    coords = coords.squeeze(0).view(-1, 3).cpu().numpy()
                                    preds = preds.squeeze(0).view(-1).cpu().numpy()
                                    targets = targets.squeeze(0).view(-1).cpu().numpy()
                                    weights = weights.squeeze(0).view(-1).cpu().numpy()
                                    preds = preds[:len(mask[mask==True][mask_start:mask_end])]
                                    targets = targets[:len(mask[mask==True][mask_start:mask_end])]
                                    weights = weights[:len(mask[mask==True][mask_start:mask_end])]
                                    block_mask[mask==True][mask_start:mask_end] = preds != targets
                                    scores[mask_start:mask_end] = (preds != targets).astype(float)
                                    block_weights[mask_start:mask_end] = weights.copy()
                                elif self.heuristic == "mc":
                                    preds = mc_forward(1, model, coords, feats, mc_iters)
                                    coords = coords.squeeze(0).view(-1, 3).cpu().numpy()
                                    preds = preds.view(-1, CONF.NUM_CLASSES).cpu().numpy()
                                    targets = targets.squeeze(0).view(-1).cpu().numpy()
                                    weights = weights.squeeze(0).view(-1).cpu().numpy()
                                    preds = preds[:len(mask[mask==True][mask_start:mask_end])]
                                    scene_entropy = np.zeros(preds.shape[0])
                                    pred_weights = self.labelweights[preds.argmax(axis=1)]
                                    for c in range(CONF.NUM_CLASSES):
                                        scene_entropy = scene_entropy - (preds[:, c] * np.log2(preds[:, c] + 1e-12))
                                    scores[mask_start:mask_end] = pred_weights * scene_entropy
                        point_scores[mask] = scores.copy()
                        point_weights[mask] = block_weights.copy()
                        point_mask = np.logical_or(point_mask, mask)
                        
                point_weights = point_weights * point_scores
                for i in np.unique(scene[:, 9]):
                    mask = (scene_data[:, 9] == i)
                    if self.heuristic == "mc":
                        segment_scores.append((float(np.sum(point_scores[mask])), i))
                    elif self.heuristic == "gt":
                        segment_scores.append((float(np.sum(point_weights[mask])), i))
                segment_scores = sorted(segment_scores, key=lambda x: x[0], reverse=True)
                current_choice = np.zeros_like(scene_data[:, 9], dtype=np.bool)
                for curr_id in range(min(n_segs, len(segment_scores))):
                    current_choice = np.logical_or(current_choice, (scene_data[:, 9] == segment_scores[curr_id][1]))
            self.selected_mask[scene_id] = np.logical_or(self.selected_mask[scene_id], current_choice)          
            self._prepare_weights()
        print("HEURISTIC: ", self.heuristic, "WEIGHTS: ", self.labelweights)

def collate_random(data):
    '''
    for ScannetDataset: collate_fn=collate_random

    return: 
        coords               # torch.FloatTensor(B, N, 3)
        feats                # torch.FloatTensor(B, N, 3)
        semantic_segs        # torch.FloatTensor(B, N)
        sample_weights       # torch.FloatTensor(B, N)
        fetch_time           # float
    '''

    # load data
    (
        point_set, 
        semantic_seg, 
        sample_weight,
        fetch_time 
    ) = zip(*data)

    # convert to tensor
    point_set = torch.FloatTensor(point_set)
    semantic_seg = torch.LongTensor(semantic_seg)
    sample_weight = torch.FloatTensor(sample_weight)

    # split points to coords and feats
    coords = point_set[:, :, :3]
    feats = point_set[:, :, 3:]

    # pack
    batch = (
        coords,             # (B, N, 3)
        feats,              # (B, N, 3)
        semantic_seg,      # (B, N)
        sample_weight,     # (B, N)
        sum(fetch_time)          # float
    )

    return batch

def collate_wholescene(data):
    '''
    for ScannetDataset: collate_fn=collate_random

    return: 
        coords               # torch.FloatTensor(B, C, N, 3)
        feats                # torch.FloatTensor(B, C, N, 3)
        semantic_segs        # torch.FloatTensor(B, C, N)
        sample_weights       # torch.FloatTensor(B, C, N)
        fetch_time           # float
    '''

    # load data
    (
        point_sets, 
        semantic_segs, 
        sample_weights,
        fetch_time 
    ) = zip(*data)

    # convert to tensor
    point_sets = torch.FloatTensor(point_sets)
    semantic_segs = torch.LongTensor(semantic_segs)
    sample_weights = torch.FloatTensor(sample_weights)

    # split points to coords and feats
    coords = point_sets[:, :, :, :3]
    feats = point_sets[:, :, :, 3:]

    # pack
    batch = (
        coords,             # (B, N, 3)
        feats,              # (B, N, 3)
        semantic_segs,      # (B, N)
        sample_weights,     # (B, N)
        sum(fetch_time)          # float
    )

    return batch
